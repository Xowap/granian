"""Server load statistics.

Subscribe to `stats_reporter` to be told, periodically, how busy the whole server is::

    from granian import stats_reporter

    @stats_reporter.on_load(every=timedelta(seconds=30))
    def report(load):
        print(load.utilization[timedelta(minutes=1)])

Load is only measured once something subscribes: until then Granian doesn't pay for it.
Subscriptions are usually made while the application is imported, but may be made at any
time; requests already running when the first subscription is made are not counted.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from types import MappingProxyType
from typing import Any

from ._compat import _PY_312, _PYV
from .errors import ConfigurationError
from .log import logger


#: How often load is sampled and aggregated; also the shortest `every` of a subscription.
_SAMPLE_INTERVAL = timedelta(seconds=1)
#: Averaging windows of `Load.utilization` and `Load.queue_wait` (mirrors `src/load.rs`).
_WINDOWS = (timedelta(minutes=1), timedelta(minutes=5), timedelta(minutes=15))


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadCounters:
    """Totals since the server started, including the workers that were respawned since.

    They never decrease.
    """

    uptime: timedelta
    #: Time Python lanes spent running requests.
    busy: timedelta
    #: Time events spent queued before a Python thread picked them up.
    wait: timedelta
    #: Events taken from the queues.
    events: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Load:
    """The load of the whole server, aggregated across its workers.

    A *lane* is where Python code runs: the event loop of an ASGI/RSGI worker, or one of
    the blocking threads of a WSGI worker.
    """

    #: Workers currently reporting.
    workers: int
    #: Lanes of those workers.
    lanes: int
    #: Lanes running a request when sampled.
    busy_lanes: int
    #: Requests started and not finished yet (connected WebSockets included).
    in_flight: int
    #: Events waiting in the queues for a Python thread.
    backlog: int
    #: Per averaging window, the share of lane time spent running requests, in [0, 1].
    utilization: Mapping[timedelta, float]
    #: Per averaging window, how long events waited for a Python thread.
    queue_wait: Mapping[timedelta, timedelta]
    counters: LoadCounters


LoadCallback = Callable[[Load], None] | Callable[[Load], Awaitable[None]]


def _ns(value: float) -> timedelta:
    return timedelta(microseconds=value / 1_000)


def _make_load(gauges: tuple, util: list[float], qwait: list[float], counters: tuple) -> Load:
    """Build a `Load` from the raw snapshot sent by the Rust side."""
    workers, lanes, busy_lanes, in_flight, backlog = gauges
    uptime, busy, wait, events = counters
    return Load(
        workers=workers,
        lanes=lanes,
        busy_lanes=busy_lanes,
        in_flight=in_flight,
        backlog=backlog,
        utilization=MappingProxyType(dict(zip(_WINDOWS, util, strict=True))),
        queue_wait=MappingProxyType({window: _ns(value) for window, value in zip(_WINDOWS, qwait, strict=True)}),
        counters=LoadCounters(uptime=_ns(uptime), busy=_ns(busy), wait=_ns(wait), events=events),
    )


def _as_timedelta(value: timedelta | float) -> timedelta:
    if isinstance(value, timedelta):
        return value
    return timedelta(seconds=value)


class _Subscription:
    __slots__ = ['callback', 'every', 'last']

    def __init__(self, callback: LoadCallback, every: timedelta):
        self.callback = callback
        self.every = every
        self.last = time.monotonic()


class _Worker:
    __slots__ = ['handle', 'is_async', 'loop']

    def __init__(self, handle: Any, is_async: bool, loop: Any):
        #: The worker's `LoadHandle`: enables its measurements, queues WSGI callbacks.
        self.handle = handle
        self.is_async = is_async
        self.loop = loop


class StatsReporter:
    """Process-wide registry of load subscriptions.

    Callbacks run once for the whole server, in a single worker: the lowest-numbered one
    still running. When that role moves to another worker (the previous one died, or a
    lower one came back), a callback may fire up to one `every` early or late.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._subs: list[_Subscription] = []
        self._workers: dict[int, _Worker] = {}
        self._attached = False
        self._tasks: set[asyncio.Future] = set()

    @property
    def active(self) -> bool:
        """True once a Granian worker runs in this process: outside of Granian, callbacks never run."""
        return self._attached

    def __enter__(self) -> StatsReporter:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def on_load(
        self, callback: LoadCallback | None = None, /, *, every: timedelta | float = timedelta(seconds=15)
    ) -> Any:
        """Call `callback` with the server `Load` every `every` (at least once a second).

        On ASGI and RSGI the callback runs on the worker's event loop, and coroutine functions
        run as a task; on WSGI it runs on one of the worker's threads, and coroutine functions
        are rejected. Without `callback`, return a decorator. Returns the callback.
        """
        if callback is None:
            return partial(self._on_load_decorator, every=every)

        sub = _Subscription(callback, max(_as_timedelta(every), _SAMPLE_INTERVAL))
        with self._lock:
            workers = list(self._workers.values())
            for worker in workers:
                self._check_interface(worker.is_async)
                if not self._accepts(worker.is_async, callback):
                    return callback
            self._subs.append(sub)
        for worker in workers:
            worker.handle.enable()
        return callback

    def _on_load_decorator(self, callback: LoadCallback, *, every: timedelta | float) -> LoadCallback:
        return self.on_load(callback, every=every)

    @staticmethod
    def _check_interface(is_async: bool) -> None:
        if is_async and _PYV < _PY_312:
            logger.error('Load stats require Python >= 3.12 for ASGI and RSGI interfaces')
            raise ConfigurationError('stats_reporter')

    @staticmethod
    def _accepts(is_async: bool, callback: LoadCallback) -> bool:
        if not is_async and inspect.iscoroutinefunction(callback):
            logger.error(f'Load stats callback {callback!r} is a coroutine function, which is not supported on WSGI')
            return False
        return True

    def _validate(self, is_async: bool) -> None:
        """Reject the subscriptions already made if this process can't serve them."""
        with self._lock:
            if self._subs:
                self._check_interface(is_async)

    def _attach(self, worker_id: int, handle: Any, is_async: bool, loop: Any = None) -> None:
        """Called by every worker spawner with the worker's `LoadHandle`, right before the
        worker starts serving. The worker only measures load once something subscribed."""
        with self._lock:
            self._validate(is_async)
            self._subs = [sub for sub in self._subs if self._accepts(is_async, sub.callback)]
            self._workers[worker_id] = _Worker(handle, is_async, loop)
            self._attached = True
            subscribed = bool(self._subs)
        handle.set_deliver(self._deliver)
        if subscribed:
            handle.enable()

    def _detach(self, worker_id: int) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def _deliver(self, gauges: tuple, util: list[float], qwait: list[float], counters: tuple) -> None:
        """Called with every server snapshot, in the process chosen to run the callbacks."""
        now = time.monotonic()
        slack = _SAMPLE_INTERVAL.total_seconds() / 2
        with self._lock:
            workers = [
                (worker_id, worker)
                for worker_id, worker in sorted(self._workers.items())
                if worker.loop is None or not worker.loop.is_closed()
            ]
            if not workers:
                return
            due = [sub for sub in self._subs if now - sub.last >= sub.every.total_seconds() - slack]
            for sub in due:
                sub.last = now
        if not due:
            return

        load = _make_load(gauges, util, qwait, counters)
        _, worker = workers[0]
        for sub in due:
            try:
                if worker.is_async:
                    worker.loop.call_soon_threadsafe(self._run_async, sub.callback, load)
                else:
                    worker.handle.push_system(partial(self._run_sync, sub.callback, load))
            except RuntimeError:
                # the loop got closed in the meantime
                pass

    @staticmethod
    def _run_sync(callback: LoadCallback, load: Load) -> None:
        try:
            callback(load)
        except Exception:
            logger.exception(f'Load stats callback {callback!r} failed')

    def _run_async(self, callback: LoadCallback, load: Load) -> None:
        try:
            result = callback(load)
        except Exception:
            logger.exception(f'Load stats callback {callback!r} failed')
            return
        if inspect.isawaitable(result):
            task = asyncio.ensure_future(result)
            self._tasks.add(task)
            task.add_done_callback(partial(self._task_done, callback))

    def _task_done(self, callback: LoadCallback, task: asyncio.Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(f'Load stats callback {callback!r} failed', exc_info=task.exception())


stats_reporter = StatsReporter()
