import asyncio
import json
import multiprocessing as mp
import os
import signal
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from datetime import timedelta

import httpx
import pytest
import websockets

import granian
from granian import stats
from granian._granian import BUILD_GIL, ASGIWorker, WSGIWorker
from granian.errors import ConfigurationError
from granian.server.embed import Server as EmbeddedGranian
from granian.stats import Load, StatsReporter, stats_reporter


PGO_RUN = bool(os.getenv('PGO_RUN'))
WINDOWS = [60.0, 300.0, 900.0]


def _handle(interface='asgi'):
    """The `LoadHandle` of a real (not serving) worker."""
    worker_cls = {'asgi': ASGIWorker, 'wsgi': WSGIWorker}[interface]
    return worker_cls(1, (None, None), None).load_handle()


def _raw(busy_ns=0):
    return (2, 4, 1, 3, 5), [0.5, 0.25, 0.125], [1e6, 2e6, 3e6], (10**9, busy_ns, 4 * 10**6, 7)


# -- API ----------------------------------------------------------------------------------------


def test_exports():
    for name in ('stats_reporter', 'StatsReporter', 'Load', 'LoadCounters', 'LoadCallback'):
        assert name in granian.__all__
        assert getattr(granian, name) is getattr(stats, name)


def test_subscription_forms():
    reporter = StatsReporter()

    def direct(load):
        pass

    assert reporter.on_load(direct, every=timedelta(seconds=2)) is direct

    @reporter.on_load
    def bare(load):
        pass

    @reporter.on_load(every=3)
    def decorated(load):
        pass

    with reporter as sr:
        assert sr is reporter
        sr.on_load(direct)

    subs = [(sub.callback, sub.every) for sub in reporter._subs]
    assert subs == [
        (direct, timedelta(seconds=2)),
        (bare, timedelta(seconds=15)),
        (decorated, timedelta(seconds=3)),
        (direct, timedelta(seconds=15)),
    ]


def test_every_is_raised_to_the_sample_interval():
    reporter = StatsReporter()
    reporter.on_load(lambda load: None, every=timedelta(milliseconds=10))

    assert reporter._subs[0].every == timedelta(seconds=1)


def test_active_only_inside_a_worker():
    reporter = StatsReporter()
    assert reporter.active is False

    reporter._attach(1, _handle(), True)
    assert reporter.active is True


def test_wsgi_rejects_coroutine_callbacks(caplog):
    reporter = StatsReporter()

    async def early(load):
        pass

    async def late(load):
        pass

    def sync(load):
        pass

    reporter.on_load(early)
    reporter.on_load(sync)
    reporter._attach(1, _handle('wsgi'), False)
    reporter.on_load(late)

    assert [sub.callback for sub in reporter._subs] == [sync]
    assert 'coroutine function' in caplog.text


def test_async_interfaces_require_py312(monkeypatch):
    monkeypatch.setattr(stats, '_PYV', 311)

    reporter = StatsReporter()
    reporter.on_load(lambda load: None)
    with pytest.raises(ConfigurationError):
        reporter._attach(1, _handle(), True)

    reporter = StatsReporter()
    reporter._attach(1, _handle(), True)
    with pytest.raises(ConfigurationError):
        reporter.on_load(lambda load: None)

    reporter = StatsReporter()
    reporter.on_load(lambda load: None)
    reporter._attach(1, _handle('wsgi'), False)


def test_load_snapshot():
    load = stats._make_load(*_raw(busy_ns=1_500_000_000))

    assert (load.workers, load.lanes, load.busy_lanes, load.in_flight, load.backlog) == (2, 4, 1, 3, 5)
    assert list(load.utilization) == [timedelta(minutes=1), timedelta(minutes=5), timedelta(minutes=15)]
    assert load.utilization[timedelta(minutes=5)] == 0.25
    assert load.queue_wait[timedelta(minutes=15)] == timedelta(milliseconds=3)
    assert load.counters.uptime == timedelta(seconds=1)
    assert load.counters.busy == timedelta(seconds=1.5)
    assert load.counters.wait == timedelta(milliseconds=4)
    assert load.counters.events == 7

    with pytest.raises(TypeError):
        load.utilization[timedelta(minutes=1)] = 1.0
    with pytest.raises(TypeError):
        load.queue_wait[timedelta(minutes=1)] = timedelta()
    with pytest.raises(FrozenInstanceError):
        load.workers = 3


def test_deliver_runs_on_the_lowest_worker_loop():
    reporter = StatsReporter()
    loop_low, loop_high = asyncio.new_event_loop(), asyncio.new_event_loop()
    received = []

    async def coro(load):
        await asyncio.sleep(0)
        received.append(('coro', load.workers))

    reporter.on_load(lambda load: received.append(('sync', load.workers)), every=1)
    reporter.on_load(coro, every=1)
    reporter._attach(2, _handle(), True, loop_high)
    reporter._attach(1, _handle(), True, loop_low)
    for sub in reporter._subs:
        sub.last -= 10

    reporter._deliver(*_raw())
    # not due yet: nothing more gets scheduled
    reporter._deliver(*_raw())

    loop_high.run_until_complete(asyncio.sleep(0.01))
    assert received == []
    loop_low.run_until_complete(asyncio.sleep(0.01))
    assert sorted(received) == [('coro', 2), ('sync', 2)]

    # the lowest worker is gone: the next one takes over
    reporter._detach(1)
    for sub in reporter._subs:
        sub.last -= 10
    reporter._deliver(*_raw())
    loop_high.run_until_complete(asyncio.sleep(0.01))
    assert len(received) == 4

    loop_low.close()
    loop_high.close()


# -- servers ------------------------------------------------------------------------------------


def _serve(interface, kwargs):
    granian.Granian(f'tests.apps.load:{interface}_app', interface=interface, **kwargs).serve()


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def _load_server(interface, tmp_path, monkeypatch, subscribe=None, **kwargs):
    out = tmp_path / 'load.jsonl'
    monkeypatch.setenv('LOAD_STATS_OUT', str(out))
    if subscribe:
        monkeypatch.setenv('LOAD_STATS_SUBSCRIBE', subscribe)
    else:
        monkeypatch.delenv('LOAD_STATS_SUBSCRIBE', raising=False)
    port = _free_port()
    kwargs = {'port': port, 'loop': 'asyncio', 'blocking_threads': 1, 'runtime_mode': 'st', **kwargs}
    proc = mp.get_context('spawn').Process(target=_serve, args=(interface, kwargs))
    proc.start()
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(('127.0.0.1', port), timeout=0.05).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Cannot bind server')
                await asyncio.sleep(0.05)
        yield port, out
    finally:
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()


def _rows(out):
    if not out.exists():
        return []
    return [json.loads(line) for line in out.read_text().splitlines() if line]


async def _wait_rows(out, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _rows(out)
        if any(predicate(row) for row in rows):
            return rows
        await asyncio.sleep(0.1)
    raise AssertionError(f'no matching snapshot in {_rows(out)}')


async def _hit(port, count, path='/'):
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(client.get(f'http://localhost:{port}{path}') for _ in range(count)))


def _check_snapshot(row):
    assert row['windows'] == WINDOWS
    assert all(0.0 <= value <= 1.0 for value in row['utilization'])
    assert all(value >= 0.0 for value in row['queue_wait'])


@pytest.mark.asyncio
@pytest.mark.skipif(PGO_RUN, reason='PGO build')
@pytest.mark.skipif(not BUILD_GIL, reason='free-threaded Python')
@pytest.mark.parametrize(('interface', 'subscribe'), [('asgi', 'async'), ('rsgi', 'sync'), ('wsgi', 'sync')])
@pytest.mark.parametrize('scheduler', ['fifo', 'fair'])
async def test_server_wide_snapshot(tmp_path, monkeypatch, interface, subscribe, scheduler):
    blocking_threads = 2 if interface == 'wsgi' else 1
    async with _load_server(
        interface, tmp_path, monkeypatch, subscribe, workers=2, blocking_threads=blocking_threads, scheduler=scheduler
    ) as (port, out):
        await _wait_rows(out, lambda row: row['workers'] == 2)
        await _hit(port, 20)
        rows = await _wait_rows(out, lambda row: row['events'] >= 20 and row['busy'] > 0)

    for row in rows:
        _check_snapshot(row)
        assert row['kind'] == subscribe
    last = rows[-1]
    assert last['lanes'] == 2 * blocking_threads
    # callbacks run in a single worker for the whole server
    assert len({row['pid'] for row in rows}) == 1
    for prev, row in zip(rows, rows[1:], strict=False):
        for counter in ('uptime', 'busy', 'wait', 'events'):
            assert row[counter] >= prev[counter]
    if interface == 'wsgi':
        assert all(row['thread'] != 'MainThread' for row in rows)


@pytest.mark.asyncio
@pytest.mark.skipif(PGO_RUN, reason='PGO build')
@pytest.mark.skipif(not BUILD_GIL, reason='free-threaded Python')
async def test_callbacks_fire_once_per_every(tmp_path, monkeypatch):
    async with _load_server('asgi', tmp_path, monkeypatch, 'sync', workers=3) as (_, out):
        await _wait_rows(out, lambda row: row['workers'] == 3)
        start = len(_rows(out))
        await asyncio.sleep(4)
        rows = _rows(out)[start:]

    # `every` is 200ms, raised to the 1s sample interval, across 3 workers
    assert 3 <= len(rows) <= 5


@pytest.mark.asyncio
@pytest.mark.skipif(PGO_RUN, reason='PGO build')
@pytest.mark.skipif(not BUILD_GIL, reason='free-threaded Python')
@pytest.mark.parametrize('interface', ['asgi', 'wsgi'])
async def test_subscribing_mid_run_enables_measurement(tmp_path, monkeypatch, interface):
    async with _load_server(interface, tmp_path, monkeypatch) as (port, out):
        await _hit(port, 5)
        await asyncio.sleep(1.5)
        assert _rows(out) == []

        await _hit(port, 1, '/subscribe')
        await _wait_rows(out, lambda row: True)
        await _hit(port, 10)
        rows = await _wait_rows(out, lambda row: row['events'] >= 10 and row['busy'] > 0)

    # only the requests made after the subscription are measured
    assert rows[-1]['events'] < 15


@pytest.mark.asyncio
@pytest.mark.skipif(PGO_RUN, reason='PGO build')
@pytest.mark.skipif(not BUILD_GIL, reason='free-threaded Python')
async def test_websocket_counts_as_in_flight(tmp_path, monkeypatch):
    async with _load_server('asgi', tmp_path, monkeypatch, 'sync') as (port, out):
        await _wait_rows(out, lambda row: True)
        async with websockets.connect(f'ws://localhost:{port}/ws') as ws:
            await ws.send('ping')
            assert await ws.recv() == 'ping'
            rows = await _wait_rows(out, lambda row: row['in_flight'] >= 1)
            connected = rows[-1]['time']
        # once closed, it no longer counts
        await _wait_rows(out, lambda row: row['time'] > connected and row['in_flight'] == 0, timeout=5)


@pytest.mark.asyncio
@pytest.mark.skipif(PGO_RUN, reason='PGO build')
@pytest.mark.skipif(not BUILD_GIL, reason='free-threaded Python')
async def test_counters_survive_a_respawn_and_the_emitter_moves(tmp_path, monkeypatch):
    async with _load_server('asgi', tmp_path, monkeypatch, 'sync', workers=2, respawn_failed_workers=True) as (
        port,
        out,
    ):
        await _wait_rows(out, lambda row: row['workers'] == 2)
        await _hit(port, 20)
        rows = await _wait_rows(out, lambda row: row['events'] >= 20)
        emitter = rows[-1]['pid']
        os.kill(emitter, signal.SIGKILL)
        rows = await _wait_rows(out, lambda row: row['pid'] != emitter and row['workers'] == 2, timeout=15)

    before = [row for row in rows if row['pid'] == emitter]
    after = [row for row in rows if row['pid'] != emitter]
    assert after
    assert after[0]['events'] >= before[-1]['events']
    assert after[0]['busy'] >= before[-1]['busy']
    for prev, row in zip(rows, rows[1:], strict=False):
        assert row['busy'] >= prev['busy']
        assert row['events'] >= prev['events']


@pytest.mark.skipif(not BUILD_GIL, reason='free-threaded Python')
def test_embedded_server(server_port):
    received = []

    async def app(scope, protocol):
        protocol.response_str(200, [], 'hello')

    def report(load):
        received.append(load)

    server = EmbeddedGranian(app, port=server_port)
    loop = asyncio.new_event_loop()

    async def client():
        await asyncio.sleep(1)
        async with httpx.AsyncClient() as h:
            for _ in range(5):
                await h.get(f'http://localhost:{server_port}')
        deadline = time.monotonic() + 10
        while not any(load.counters.events >= 5 for load in received) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        server.stop()

    stats_reporter.on_load(report, every=1)
    try:
        server_task = loop.create_task(server.serve())
        loop.run_until_complete(client())
        loop.run_until_complete(server_task)
    finally:
        stats_reporter._subs = [sub for sub in stats_reporter._subs if sub.callback is not report]
        loop.close()

    assert received
    assert isinstance(received[-1], Load)
    assert received[-1].workers == 1
    assert received[-1].lanes == 1
    assert received[-1].counters.events >= 5


def test_monitoring_tool_id_option():
    assert granian.Granian('tests.apps.asgi:app', monitoring_tool_id=2).monitoring_tool_id == 2
    assert granian.Granian('tests.apps.asgi:app', fq_monitoring_tool_id=3).monitoring_tool_id == 3
    assert granian.Granian('tests.apps.asgi:app').monitoring_tool_id == 4
