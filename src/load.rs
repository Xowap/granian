//! Server load statistics, reported to `granian.stats.stats_reporter` subscribers.
//!
//! Each worker owns a [`WorkerLoad`], created with its configuration and handed down to its
//! runtimes, blocking pools and requests, the same way as its `WorkerMetrics`. Unlike
//! metrics, it is switched on at runtime: nothing is measured until the Python side
//! subscribes and enables it through the worker's [`LoadHandle`].
//!
//! Once enabled, a sampler thread takes a [`Sample`] of the worker every `SAMPLE_INTERVAL`
//! and hands it to the server's [`LoadAggregator`]: directly when they share the process
//! (free-threaded, embedded, Windows), over a datagram channel in multi-process mode. The
//! aggregator merges the samples of every worker into a [`Snapshot`], delivered to the
//! worker that runs the callbacks (the "emitter").

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    hash::{BuildHasher, Hasher},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::fd::{FromRawFd, OwnedFd, RawFd};

use super::{
    blocking::BlockingTask,
    scheduler::{Scheduler, SchedulerImpl},
};

/// Sampling and aggregation period.
pub(crate) const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
/// Averaging windows of `utilization` and `queue_wait` (mirrored in `granian.stats`).
pub(crate) const WINDOWS: [Duration; 3] = [
    Duration::from_secs(60),
    Duration::from_secs(300),
    Duration::from_mins(15),
];
/// A worker that hasn't reported for this long is no longer counted as live.
const STALE: Duration = Duration::from_secs(3);
/// A worker that hasn't reported for this long is gone: only its counters are kept.
const RETIRE: Duration = Duration::from_secs(600);

/// A Python object that may be dropped without the interpreter attached (e.g. by a thread
/// ending at shutdown): it is released if possible, leaked otherwise.
struct Detached(std::mem::ManuallyDrop<Py<PyAny>>);

impl Detached {
    fn new(obj: Py<PyAny>) -> Self {
        Self(std::mem::ManuallyDrop::new(obj))
    }

    fn get(&self) -> &Py<PyAny> {
        &self.0
    }
}

impl Drop for Detached {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and the value is not used afterwards.
        let mut obj = Some(unsafe { std::mem::ManuallyDrop::take(&mut self.0) });
        if Python::try_attach(|_| drop(obj.take())).is_none() {
            std::mem::forget(obj);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Per-worker measurements
// ---------------------------------------------------------------------------------------------

pub(crate) type ArcWorkerLoad = Arc<WorkerLoad>;

/// The load of one worker: what its requests and pool tasks consumed, measured only once
/// `enable` has been called.
pub(crate) struct WorkerLoad {
    enabled: AtomicBool,
    /// Wakes the sampler up once enabled.
    started: tokio::sync::Notify,
    /// Random, so the main process can tell a respawned worker from the one it replaces
    /// while both report on the same slot.
    instance: u64,
    lanes: usize,
    /// Whether pool tasks are request work (WSGI), rather than event plumbing (ASGI/RSGI).
    pool_measure: bool,
    busy_ns: AtomicU64,
    /// Time spent by events in scheduler queues. Turns waited on the fair scheduler's loop
    /// lane are not events, so they are not counted here.
    wait_ns: AtomicU64,
    events: AtomicU64,
    in_flight: AtomicUsize,
    busy_lanes: AtomicUsize,
    scheds: Mutex<Vec<Weak<SchedulerImpl>>>,
    /// The code object the `sys.monitoring` hooks instrument, with its tool ID (async only).
    monitor: Mutex<Option<(Detached, u8)>>,
    /// `stats_reporter._deliver`, set when the worker attaches to the stats reporter.
    deliver: Mutex<Option<Detached>>,
}

impl WorkerLoad {
    /// `lanes`: 1 on ASGI/RSGI, the pool size on WSGI (`pool_measure`).
    pub(crate) fn new(lanes: usize, pool_measure: bool) -> Self {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u32(std::process::id());
        if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            hasher.write_u128(now.as_nanos());
        }
        Self {
            enabled: AtomicBool::new(false),
            started: tokio::sync::Notify::new(),
            instance: hasher.finish(),
            lanes,
            pool_measure,
            busy_ns: AtomicU64::new(0),
            wait_ns: AtomicU64::new(0),
            events: AtomicU64::new(0),
            in_flight: AtomicUsize::new(0),
            busy_lanes: AtomicUsize::new(0),
            scheds: Mutex::new(Vec::new()),
            monitor: Mutex::new(None),
            deliver: Mutex::new(None),
        }
    }

    #[inline]
    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Count a request as in flight for as long as the guard lives. Requests started before
    /// load stats got enabled are never counted (nor discounted).
    pub(crate) fn in_flight(self: &Arc<Self>) -> Option<InFlight> {
        self.enabled().then(|| {
            self.in_flight.fetch_add(1, Ordering::Relaxed);
            InFlight(self.clone())
        })
    }

    #[inline]
    pub(crate) fn add_busy(&self, ns: u64) {
        self.busy_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// The event loop started running a measured request step.
    #[inline]
    pub(crate) fn lane_busy(&self) {
        self.busy_lanes.fetch_add(1, Ordering::Relaxed);
    }

    /// The measured step ended.
    #[inline]
    pub(crate) fn lane_idle(&self) {
        self.busy_lanes.fetch_sub(1, Ordering::Relaxed);
    }

    /// Run a pool task that was queued while measuring: account its queue wait and, on
    /// WSGI, the time it holds its pool thread.
    pub(crate) fn run_task<F: FnOnce(Python)>(&self, py: Python, pushed_at: Instant, task: F) {
        self.wait_ns
            .fetch_add(pushed_at.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.events.fetch_add(1, Ordering::Relaxed);
        if self.pool_measure {
            self.lane_busy();
            let t0 = Instant::now();
            task(py);
            self.add_busy(t0.elapsed().as_nanos() as u64);
            self.lane_idle();
        } else {
            task(py);
        }
    }

    /// Track one of the worker's schedulers, for `backlog` and to queue WSGI callbacks.
    pub(crate) fn add_scheduler(&self, sched: &Arc<SchedulerImpl>) {
        let mut scheds = self.scheds.lock().unwrap();
        if !scheds.iter().any(|s| std::ptr::eq(s.as_ptr(), Arc::as_ptr(sched))) {
            scheds.push(Arc::downgrade(sched));
        }
    }

    fn schedulers(&self) -> Vec<Arc<SchedulerImpl>> {
        self.scheds.lock().unwrap().iter().filter_map(Weak::upgrade).collect()
    }

    fn sample(&self) -> Sample {
        Sample {
            instance: self.instance,
            workers: 1,
            lanes: self.lanes as u64,
            busy_lanes: self.busy_lanes.load(Ordering::Relaxed) as u64,
            in_flight: self.in_flight.load(Ordering::Relaxed) as u64,
            backlog: self.schedulers().iter().map(|s| s.len() as u64).sum(),
            busy_ns: self.busy_ns.load(Ordering::Relaxed),
            wait_ns: self.wait_ns.load(Ordering::Relaxed),
            events: self.events.load(Ordering::Relaxed),
        }
    }

    /// Record the code object the occupancy hooks instrument (async interfaces), installing
    /// them right away when something needs them already (`now`: the fair scheduler).
    pub(crate) fn set_monitor_target(
        &self,
        py: Python,
        scheduler: &crate::callbacks::CallbackScheduler,
        tool_id: u8,
        now: bool,
    ) {
        match scheduler.cb.getattr(py, "__code__") {
            Ok(code) => *self.monitor.lock().unwrap() = Some((Detached::new(code), tool_id)),
            Err(err) => log::error!("Cannot find the code object to monitor: {err:?}"),
        }
        if now || self.enabled() {
            self.install_monitoring(py);
        }
    }

    fn install_monitoring(&self, py: Python) {
        let target = self
            .monitor
            .lock()
            .unwrap()
            .as_ref()
            .map(|(code, tool_id)| (code.get().clone_ref(py), *tool_id));
        if let Some((code, tool_id)) = target
            && let Err(err) = crate::accounting::install_monitoring_once(py, code.bind(py), tool_id)
        {
            log::error!("Cannot install sys.monitoring hooks for lane occupancy: {err:?}");
        }
    }

    /// Start measuring (idempotent).
    fn enable(&self, py: Python) {
        if self.enabled.swap(true, Ordering::AcqRel) {
            return;
        }
        self.install_monitoring(py);
        self.started.notify_one();
    }

    /// Hand a snapshot to the stats reporter, if this worker is the emitter.
    fn deliver(&self, snapshot: &Snapshot) {
        if snapshot.emit != self.instance {
            return;
        }
        _ = Python::try_attach(|py| {
            let deliver = self.deliver.lock().unwrap().as_ref().map(|d| d.get().clone_ref(py));
            if let Some(deliver) = deliver {
                deliver_py(py, &deliver, snapshot);
            }
        });
    }
}

/// Keeps a request counted in its worker's `in_flight`.
pub(crate) struct InFlight(ArcWorkerLoad);

impl InFlight {
    pub(crate) fn worker(&self) -> &WorkerLoad {
        &self.0
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

fn deliver_py(py: Python, deliver: &Py<PyAny>, s: &Snapshot) {
    let res = deliver.call1(
        py,
        (
            (s.workers, s.lanes, s.busy_lanes, s.in_flight, s.backlog),
            s.util.to_vec(),
            s.qwait_ns.to_vec(),
            (s.uptime_ns, s.busy_ns, s.wait_ns, s.events),
        ),
    );
    if let Err(err) = res {
        log::warn!("Cannot deliver load stats: {err:?}");
    }
}

/// The worker's side of its `LoadAggregator`, given to the worker by the server.
pub(crate) enum LoadSink {
    /// Multi-process mode: a datagram socket to the main process.
    #[cfg(unix)]
    Ipc(OwnedFd),
    /// In-process aggregation, on the worker's slot.
    Local(Py<LoadAggregator>, usize),
}

impl LoadSink {
    /// What the worker spawners pass: the aggregator itself, or the channel's descriptor.
    pub(crate) fn from_py(obj: &Bound<PyAny>, slot: usize) -> PyResult<Self> {
        if let Ok(aggregator) = obj.cast::<LoadAggregator>() {
            return Ok(Self::Local(aggregator.clone().unbind(), slot));
        }
        #[cfg(unix)]
        {
            let fd: RawFd = obj.extract()?;
            // SAFETY: Python hands over a duplicated descriptor that nothing else owns.
            Ok(Self::Ipc(unsafe { OwnedFd::from_raw_fd(fd) }))
        }
        #[cfg(not(unix))]
        Err(pyo3::exceptions::PyTypeError::new_err("expected a LoadAggregator"))
    }
}

/// Start the worker's sampler thread. It sleeps until load stats get enabled, then sends a
/// sample every `SAMPLE_INTERVAL` until `stop` changes.
pub(crate) fn spawn_sampler(
    load: ArcWorkerLoad,
    sink: Option<LoadSink>,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> Option<std::thread::JoinHandle<()>> {
    let sink = sink?;
    Some(std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
            log::error!("Cannot build the load sampler runtime");
            return;
        };
        rt.block_on(async {
            tokio::select! {
                () = load.started.notified() => {},
                _ = stop.changed() => return,
            }
            match &sink {
                #[cfg(unix)]
                LoadSink::Ipc(fd) => sample_ipc(&load, fd, &mut stop).await,
                LoadSink::Local(aggregator, slot) => {
                    let mut ival = interval();
                    loop {
                        tokio::select! {
                            _ = ival.tick() => aggregator.get().collect(*slot, load.sample()),
                            _ = stop.changed() => break,
                        }
                    }
                }
            }
        });
        Python::attach(|_| drop(sink));
    }))
}

/// Multi-process mode: samples go to the main process, snapshots come back.
#[cfg(unix)]
async fn sample_ipc(load: &WorkerLoad, fd: &OwnedFd, stop: &mut tokio::sync::watch::Receiver<bool>) {
    let Some(sock) = datagram(fd) else {
        return;
    };
    let mut buf = vec![0u8; 4096];
    let mut ival = interval();
    loop {
        tokio::select! {
            _ = ival.tick() => {
                _ = sock.try_send(&LoadMessage::Sample(load.sample()).encode());
            },
            res = sock.recv(&mut buf) => match res {
                Ok(n) => {
                    if let Some(LoadMessage::Snapshot(snapshot)) = LoadMessage::decode(&buf[..n]) {
                        load.deliver(&snapshot);
                    }
                }
                Err(err) => {
                    log::debug!("Load stats channel error: {err:?}");
                    tokio::time::sleep(SAMPLE_INTERVAL).await;
                }
            },
            _ = stop.changed() => break,
        }
    }
}

#[cfg(unix)]
fn datagram(fd: &OwnedFd) -> Option<tokio::net::UnixDatagram> {
    let sock = fd
        .try_clone()
        .map(std::os::unix::net::UnixDatagram::from)
        .and_then(|sock| sock.set_nonblocking(true).map(|()| sock))
        .and_then(tokio::net::UnixDatagram::from_std);
    match sock {
        Ok(sock) => Some(sock),
        Err(err) => {
            log::error!("Cannot set up the load stats channel: {err:?}");
            None
        }
    }
}

fn interval() -> tokio::time::Interval {
    let mut ival = tokio::time::interval_at(tokio::time::Instant::now() + SAMPLE_INTERVAL, SAMPLE_INTERVAL);
    ival.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ival
}

// ---------------------------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------------------------

/// What a worker reports every `SAMPLE_INTERVAL`: gauges at sampling time, and counters
/// accumulated since the worker started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Sample {
    pub instance: u64,
    pub workers: u64,
    pub lanes: u64,
    pub busy_lanes: u64,
    pub in_flight: u64,
    pub backlog: u64,
    pub busy_ns: u64,
    pub wait_ns: u64,
    pub events: u64,
}

/// The server-wide load, as handed to the Python side. `emit` is the instance that must run
/// the callbacks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub workers: u64,
    pub lanes: u64,
    pub busy_lanes: u64,
    pub in_flight: u64,
    pub backlog: u64,
    pub util: [f64; 3],
    pub qwait_ns: [f64; 3],
    pub uptime_ns: u64,
    pub busy_ns: u64,
    pub wait_ns: u64,
    pub events: u64,
    pub emit: u64,
}

#[derive(Debug, Serialize, Deserialize)]
enum LoadMessage {
    Sample(Sample),
    Snapshot(Snapshot),
}

impl LoadMessage {
    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("load message serialization");
        buf
    }

    fn decode(buf: &[u8]) -> Option<Self> {
        ciborium::from_reader(buf).ok()
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Totals {
    busy: u64,
    wait: u64,
    events: u64,
}

impl Totals {
    fn add(&mut self, s: &Sample) {
        self.busy += s.busy_ns;
        self.wait += s.wait_ns;
        self.events += s.events;
    }
}

struct Instance {
    last: Sample,
    first_seen: Instant,
    seen: Instant,
}

#[derive(Default)]
struct Slot {
    instances: Vec<Instance>,
    retired: Totals,
}

/// Merges the samples of every worker slot into a [`Snapshot`].
///
/// A slot can hold several instances at once (a respawned worker overlaps the one it
/// replaces), so samples are kept per instance: counters add up across instances and, when
/// an instance is gone, they stay in the slot's retired totals, so they never decrease.
pub(crate) struct Aggregator {
    slots: Vec<Slot>,
    birth: Instant,
    prev: Option<(Instant, Totals)>,
    util: [f64; 3],
    qwait: [f64; 3],
}

impl Aggregator {
    pub(crate) fn new(slots: usize, birth: Instant) -> Self {
        Self {
            slots: (0..slots.max(1)).map(|_| Slot::default()).collect(),
            birth,
            prev: None,
            util: [0.0; 3],
            qwait: [0.0; 3],
        }
    }

    pub(crate) fn collect(&mut self, slot: usize, sample: Sample, now: Instant) {
        let Some(slot) = self.slots.get_mut(slot) else {
            return;
        };
        match slot.instances.iter_mut().find(|i| i.last.instance == sample.instance) {
            Some(inst) => {
                // counters are cumulative: never let a late sample move them back
                let prev = inst.last;
                inst.last = Sample {
                    busy_ns: sample.busy_ns.max(prev.busy_ns),
                    wait_ns: sample.wait_ns.max(prev.wait_ns),
                    events: sample.events.max(prev.events),
                    ..sample
                };
                inst.seen = now;
            }
            None => slot.instances.push(Instance {
                last: sample,
                first_seen: now,
                seen: now,
            }),
        }
    }

    /// Compute the snapshot at `now`, with the slot of the emitter and how many instances
    /// share it. `None` while no worker is reporting.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn tick(&mut self, now: Instant) -> Option<(Snapshot, usize, usize)> {
        let mut totals = Totals::default();
        let mut live = Sample::default();
        let mut emitter: Option<(usize, u64, usize)> = None;

        for (idx, slot) in self.slots.iter_mut().enumerate() {
            let retired = &mut slot.retired;
            slot.instances.retain(|i| {
                let gone = now.saturating_duration_since(i.seen) > RETIRE;
                if gone {
                    retired.add(&i.last);
                }
                !gone
            });
            totals.busy += slot.retired.busy;
            totals.wait += slot.retired.wait;
            totals.events += slot.retired.events;

            let mut newest: Option<&Instance> = None;
            let mut count = 0;
            for inst in &slot.instances {
                totals.add(&inst.last);
                if now.saturating_duration_since(inst.seen) > STALE {
                    continue;
                }
                count += 1;
                live.workers += inst.last.workers;
                live.lanes += inst.last.lanes;
                live.busy_lanes += inst.last.busy_lanes;
                live.in_flight += inst.last.in_flight;
                live.backlog += inst.last.backlog;
                if newest.is_none_or(|n| inst.first_seen >= n.first_seen) {
                    newest = Some(inst);
                }
            }
            if emitter.is_none()
                && let Some(newest) = newest
            {
                emitter = Some((idx, newest.last.instance, count));
            }
        }

        let (slot, emit, copies) = emitter?;

        if let Some((t, prev)) = self.prev {
            totals.busy = totals.busy.max(prev.busy);
            totals.wait = totals.wait.max(prev.wait);
            totals.events = totals.events.max(prev.events);
            let dt = now.saturating_duration_since(t).as_secs_f64();
            if dt > 0.0 {
                let busy = (totals.busy - prev.busy) as f64;
                let wait = (totals.wait - prev.wait) as f64;
                let events = totals.events - prev.events;
                let u = if live.lanes > 0 {
                    (busy / (live.lanes as f64 * dt * 1e9)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let w = if events > 0 { wait / events as f64 } else { 0.0 };
                for (i, window) in WINDOWS.iter().enumerate() {
                    let a = (-dt / window.as_secs_f64()).exp();
                    self.util[i] = self.util[i] * a + u * (1.0 - a);
                    self.qwait[i] = self.qwait[i] * a + w * (1.0 - a);
                }
            }
        }
        self.prev = Some((now, totals));

        let snapshot = Snapshot {
            workers: live.workers,
            lanes: live.lanes,
            busy_lanes: live.busy_lanes.min(live.lanes),
            in_flight: live.in_flight,
            backlog: live.backlog,
            util: self.util,
            qwait_ns: self.qwait,
            uptime_ns: now.saturating_duration_since(self.birth).as_nanos() as u64,
            busy_ns: totals.busy,
            wait_ns: totals.wait,
            events: totals.events,
            emit,
        };
        Some((snapshot, slot, copies))
    }
}

// ---------------------------------------------------------------------------------------------
// Python API
// ---------------------------------------------------------------------------------------------

/// A worker's load, as seen by `granian.stats.stats_reporter` (see `Worker.load_handle`).
#[pyclass(frozen, module = "granian._granian")]
pub(crate) struct LoadHandle {
    load: ArcWorkerLoad,
}

impl LoadHandle {
    pub(crate) fn new(load: ArcWorkerLoad) -> Self {
        Self { load }
    }
}

#[pymethods]
impl LoadHandle {
    /// Start measuring the worker (idempotent).
    fn enable(&self, py: Python) {
        self.load.enable(py);
    }

    /// Where the snapshots must go when this worker is the emitter.
    fn set_deliver(&self, deliver: Py<PyAny>) {
        *self.load.deliver.lock().unwrap() = Some(Detached::new(deliver));
    }

    /// Run `cb()` on one of the worker's pool threads, ahead of queued requests (WSGI).
    /// Returns `False` if the worker is gone.
    fn push_system(&self, cb: Py<PyAny>) -> bool {
        let Some(sched) = self.load.schedulers().into_iter().next() else {
            return false;
        };
        let cb = Detached::new(cb);
        sched
            .push_system(BlockingTask::new(move |py| {
                if let Err(err) = cb.get().call0(py) {
                    err.print(py);
                }
            }))
            .is_ok()
    }
}

/// Merges the samples of every worker of the server into its snapshots.
///
/// In-process (`run_local`), the workers hand it their samples directly, and it delivers the
/// snapshots to `deliver` (the stats reporter of the process). In multi-process mode (`run`,
/// in the main process), samples and snapshots go through one datagram channel per worker
/// slot, and each snapshot only goes to the emitter. Either way it stays idle until a worker
/// sends its first sample.
#[pyclass(frozen, module = "granian._granian")]
pub(crate) struct LoadAggregator {
    inner: Arc<Mutex<Aggregator>>,
    started: Arc<tokio::sync::Notify>,
    deliver: Option<Detached>,
}

impl LoadAggregator {
    fn collect(&self, slot: usize, sample: Sample) {
        self.inner.lock().unwrap().collect(slot, sample, Instant::now());
        self.started.notify_one();
    }
}

#[pymethods]
impl LoadAggregator {
    #[new]
    #[pyo3(signature = (slots, deliver=None))]
    fn new(slots: usize, deliver: Option<Py<PyAny>>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Aggregator::new(slots, Instant::now()))),
            started: Arc::new(tokio::sync::Notify::new()),
            deliver: deliver.map(Detached::new),
        }
    }

    /// Aggregate in-process, until `sig` is set.
    fn run_local(pyself: Py<Self>, sig: Py<crate::workers::WorkerSignal>) {
        let mut stop = sig.get().arx.lock().unwrap().clone().unwrap();
        Python::attach(|_| drop(sig));

        std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
                log::error!("Cannot build the load aggregator runtime");
                return;
            };
            let this = pyself.get();
            rt.block_on(async {
                tokio::select! {
                    () = this.started.notified() => {},
                    _ = stop.changed() => return,
                }
                let mut ival = interval();
                loop {
                    tokio::select! {
                        _ = ival.tick() => {
                            let tick = this.inner.lock().unwrap().tick(Instant::now());
                            if let (Some((snapshot, _, _)), Some(deliver)) = (tick, this.deliver.as_ref()) {
                                _ = Python::try_attach(|py| deliver_py(py, deliver.get(), &snapshot));
                            }
                        },
                        _ = stop.changed() => break,
                    }
                }
            });
            Python::attach(|_| drop(pyself));
        });
    }

    /// Serve the per-slot channels (`fds`, owned from now on) until `sig` is set.
    #[cfg(unix)]
    fn run(pyself: Py<Self>, sig: Py<crate::workers::WorkerSignal>, fds: Vec<RawFd>) {
        // SAFETY: Python hands over duplicated descriptors that nothing else owns.
        let fds: Vec<OwnedFd> = fds.into_iter().map(|fd| unsafe { OwnedFd::from_raw_fd(fd) }).collect();
        let mut stop = sig.get().arx.lock().unwrap().clone().unwrap();
        Python::attach(|_| drop(sig));

        std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
                log::error!("Cannot build the load aggregator runtime");
                return;
            };
            let this = pyself.get();
            rt.block_on(async {
                let Some(socks) = fds
                    .iter()
                    .map(|fd| datagram(fd).map(Arc::new))
                    .collect::<Option<Vec<_>>>()
                else {
                    return;
                };

                for (idx, sock) in socks.iter().enumerate() {
                    let sock = sock.clone();
                    let inner = this.inner.clone();
                    let started = this.started.clone();
                    tokio::spawn(async move {
                        let mut buf = vec![0u8; 4096];
                        while let Ok(n) = sock.recv(&mut buf).await {
                            if let Some(LoadMessage::Sample(sample)) = LoadMessage::decode(&buf[..n]) {
                                inner.lock().unwrap().collect(idx, sample, Instant::now());
                                started.notify_one();
                            }
                        }
                    });
                }

                tokio::select! {
                    () = this.started.notified() => {},
                    _ = stop.changed() => return,
                }
                let mut ival = interval();
                loop {
                    tokio::select! {
                        _ = ival.tick() => {
                            let tick = this.inner.lock().unwrap().tick(Instant::now());
                            if let Some((snapshot, slot, copies)) = tick {
                                // instances sharing a slot read from the same socket: send one
                                // copy each, so the emitter gets one whichever reads first
                                let msg = LoadMessage::Snapshot(snapshot).encode();
                                for _ in 0..copies {
                                    _ = socks[slot].try_send(&msg);
                                }
                            }
                        },
                        _ = stop.changed() => break,
                    }
                }
            });
            Python::attach(|_| drop(pyself));
        });
    }
}

pub(crate) fn init_pymodule(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<LoadAggregator>()?;
    module.add_class::<LoadHandle>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(instance: u64, lanes: u64, busy_ns: u64) -> Sample {
        Sample {
            instance,
            workers: 1,
            lanes,
            busy_ns,
            ..Sample::default()
        }
    }

    const SEC: u64 = 1_000_000_000;

    #[test]
    fn utilization_follows_the_decay_formula() {
        let t0 = Instant::now();
        let mut agg = Aggregator::new(1, t0);
        agg.collect(0, sample(1, 2, 0), t0);
        agg.tick(t0).unwrap();

        let t1 = t0 + Duration::from_secs(1);
        // 1 s busy over 2 lanes in 1 s: half utilized
        agg.collect(0, sample(1, 2, SEC), t1);
        let (snap, _, _) = agg.tick(t1).unwrap();
        for (i, window) in WINDOWS.iter().enumerate() {
            let a = (-1.0 / window.as_secs_f64()).exp();
            assert!((snap.util[i] - 0.5 * (1.0 - a)).abs() < 1e-9);
        }
        assert_eq!(snap.lanes, 2);
        assert_eq!(snap.busy_ns, SEC);
    }

    #[test]
    fn utilization_is_clamped() {
        let t0 = Instant::now();
        let mut agg = Aggregator::new(1, t0);
        agg.collect(0, sample(1, 1, 0), t0);
        agg.tick(t0);
        let t1 = t0 + Duration::from_secs(1);
        agg.collect(0, sample(1, 1, 10 * SEC), t1);
        let (snap, _, _) = agg.tick(t1).unwrap();
        assert!(snap.util.iter().all(|u| (0.0..=1.0).contains(u)));
    }

    #[test]
    fn queue_wait_is_per_event() {
        let t0 = Instant::now();
        let mut agg = Aggregator::new(1, t0);
        agg.collect(0, sample(1, 1, 0), t0);
        agg.tick(t0);
        let t1 = t0 + Duration::from_secs(1);
        let mut s = sample(1, 1, 0);
        s.wait_ns = 40_000_000;
        s.events = 4;
        agg.collect(0, s, t1);
        let (snap, _, _) = agg.tick(t1).unwrap();
        let a = (-1.0 / 60.0f64).exp();
        assert!((snap.qwait_ns[0] - 10_000_000.0 * (1.0 - a)).abs() < 1e-3);
    }

    #[test]
    fn counters_never_decrease_across_respawns() {
        let t0 = Instant::now();
        let mut agg = Aggregator::new(1, t0);
        agg.collect(0, sample(1, 1, 5 * SEC), t0);
        let (first, _, _) = agg.tick(t0).unwrap();

        // the respawned worker starts from zero, overlapping the old one for a while
        let t1 = t0 + Duration::from_secs(1);
        agg.collect(0, sample(1, 1, 6 * SEC), t1);
        agg.collect(0, sample(2, 1, 0), t1);
        let (overlap, _, _) = agg.tick(t1).unwrap();
        assert_eq!(overlap.workers, 2);
        assert!(overlap.busy_ns >= first.busy_ns);

        // the old one stops reporting: excluded from the gauges, kept in the counters
        let mut busy = overlap.busy_ns;
        for secs in 2..=700 {
            let t = t0 + Duration::from_secs(secs);
            agg.collect(0, sample(2, 1, secs), t);
            let (snap, _, _) = agg.tick(t).unwrap();
            assert!(snap.busy_ns >= busy);
            busy = snap.busy_ns;
            if secs > 5 {
                assert_eq!(snap.workers, 1);
            }
        }
        assert!(busy >= 6 * SEC);
    }

    #[test]
    fn stale_workers_are_excluded() {
        let t0 = Instant::now();
        let mut agg = Aggregator::new(2, t0);
        agg.collect(0, sample(1, 1, 0), t0);
        agg.collect(1, sample(2, 1, 0), t0);
        assert_eq!(agg.tick(t0).unwrap().0.workers, 2);

        let t1 = t0 + Duration::from_secs(4);
        agg.collect(1, sample(2, 1, 0), t1);
        let (snap, slot, _) = agg.tick(t1).unwrap();
        assert_eq!(snap.workers, 1);
        assert_eq!(slot, 1);

        assert!(agg.tick(t1 + Duration::from_secs(4)).is_none());
    }

    #[test]
    fn emitter_is_the_lowest_live_slot() {
        let t0 = Instant::now();
        let mut agg = Aggregator::new(3, t0);
        agg.collect(1, sample(11, 1, 0), t0);
        agg.collect(2, sample(12, 1, 0), t0);
        let (snap, slot, copies) = agg.tick(t0).unwrap();
        assert_eq!((snap.emit, slot, copies), (11, 1, 1));

        // slot 0 comes (back) up: it takes the role over
        let t1 = t0 + Duration::from_secs(1);
        for (idx, inst) in [(0, 10), (1, 11), (2, 12)] {
            agg.collect(idx, sample(inst, 1, 0), t1);
        }
        assert_eq!(agg.tick(t1).unwrap().0.emit, 10);

        // slot 0 dies: the next lowest takes over
        let t2 = t1 + Duration::from_secs(4);
        for (idx, inst) in [(1, 11), (2, 12)] {
            agg.collect(idx, sample(inst, 1, 0), t2);
        }
        assert_eq!(agg.tick(t2).unwrap().0.emit, 11);

        // a respawn of slot 1 overlapping its predecessor: the newest instance emits
        let t3 = t2 + Duration::from_secs(1);
        for (idx, inst) in [(1, 11), (1, 21), (2, 12)] {
            agg.collect(idx, sample(inst, 1, 0), t3);
        }
        let (snap, slot, copies) = agg.tick(t3).unwrap();
        assert_eq!((snap.emit, slot, copies), (21, 1, 2));
    }

    #[test]
    fn messages_roundtrip() {
        let msg = LoadMessage::Sample(sample(7, 3, 42)).encode();
        match LoadMessage::decode(&msg) {
            Some(LoadMessage::Sample(s)) => assert_eq!(s, sample(7, 3, 42)),
            other => panic!("unexpected {other:?}"),
        }
    }
}
