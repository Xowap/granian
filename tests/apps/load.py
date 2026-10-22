"""Apps for the load stats tests.

Every snapshot a callback receives is appended as a JSON line to `$LOAD_STATS_OUT`. The
callback is subscribed at import time when `$LOAD_STATS_SUBSCRIBE` is set, or by a request
to `/subscribe` otherwise.
"""

import asyncio
import json
import os
import threading
import time
from datetime import timedelta

from granian import stats_reporter


OUT = os.environ.get('LOAD_STATS_OUT')


def _record(load, kind):
    row = {
        'pid': os.getpid(),
        'kind': kind,
        'thread': threading.current_thread().name,
        'time': time.monotonic(),
        'workers': load.workers,
        'lanes': load.lanes,
        'busy_lanes': load.busy_lanes,
        'in_flight': load.in_flight,
        'backlog': load.backlog,
        'windows': [window.total_seconds() for window in load.utilization],
        'utilization': list(load.utilization.values()),
        'queue_wait': [value.total_seconds() for value in load.queue_wait.values()],
        'uptime': load.counters.uptime.total_seconds(),
        'busy': load.counters.busy.total_seconds(),
        'wait': load.counters.wait.total_seconds(),
        'events': load.counters.events,
    }
    with open(OUT, 'a') as f:
        f.write(json.dumps(row) + '\n')


def report(load):
    _record(load, 'sync')


async def report_async(load):
    await asyncio.sleep(0)
    _record(load, 'async')


def _subscribe(is_async):
    stats_reporter.on_load(report_async if is_async else report, every=timedelta(milliseconds=200))


if os.environ.get('LOAD_STATS_SUBSCRIBE'):
    _subscribe(os.environ.get('LOAD_STATS_SUBSCRIBE') == 'async')


def _spin(seconds):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


async def asgi_app(scope, receive, send):
    if scope['type'] == 'lifespan':
        while True:
            msg = await receive()
            await send({'type': msg['type'] + '.complete'})
            if msg['type'] == 'lifespan.shutdown':
                return
    if scope['type'] == 'websocket':
        await receive()
        await send({'type': 'websocket.accept'})
        while True:
            msg = await receive()
            if msg['type'] == 'websocket.disconnect':
                return
            await send({'type': 'websocket.send', 'text': msg.get('text')})
    if scope['path'] == '/subscribe':
        _subscribe(True)
    else:
        _spin(0.02)
        await asyncio.sleep(0.01)
    await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    await send({'type': 'http.response.body', 'body': b'ok'})


async def rsgi_app(scope, protocol):
    _spin(0.02)
    await asyncio.sleep(0.01)
    protocol.response_bytes(200, [], b'ok')


def wsgi_app(environ, start_response):
    if environ['PATH_INFO'] == '/subscribe':
        _subscribe(False)
    else:
        time.sleep(0.05)
    start_response('200 OK', [])
    return [b'ok']
