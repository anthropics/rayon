//! Code that decides when workers should go to sleep. See README.md
//! for an overview.

use crate::SpinPolicy;
use crate::latch::CoreLatch;
use crate::sync::{Condvar, Mutex};
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

mod counters;
pub(crate) use self::counters::THREADS_MAX;
use self::counters::{AtomicCounters, JobsEventCounter};

/// The `Sleep` struct is embedded into each registry. It governs the waking and sleeping
/// of workers. It has callbacks that are invoked periodically at significant events,
/// such as when workers are looping and looking for work, when latches are set, or when
/// jobs are published, and it either blocks threads or wakes them in response to these
/// events. See the [`README.md`] in this module for more details.
///
/// [`README.md`] README.md
pub(super) struct Sleep {
    /// One "sleep state" per worker. Used to track if a worker is sleeping and to have
    /// them block.
    worker_sleep_states: Vec<CachePadded<WorkerSleepState>>,

    counters: AtomicCounters,

    /// How idle workers spin searching for work (see [`SpinPolicy`]).
    policy: SpinPolicy,

    /// Adaptive policy: number of workers idle at the top of their main
    /// loop, i.e. with no job in flight. When this is the whole pool no
    /// parallel region is in progress, so nothing but a newly injected
    /// job can produce work. (The inactive count in `counters` also
    /// includes workers waiting in a `join` for a stolen job, which a
    /// still-running region will resume.)
    idle_workers: CachePadded<AtomicUsize>,

    /// Number of idle workers currently in the yield/steal spin rounds.
    /// Maintained only under `SpinPolicy::ProducerBounded`: at most one
    /// searcher per currently-executing worker (each producer owns one
    /// deque a searcher could steal from), minimum one. Workers beyond
    /// the budget skip straight to the sleepy protocol (announce via the
    /// JEC, one final search, then block), so a draining pool tapers its
    /// searchers with its producers instead of paying pool-width sweeps.
    searchers: AtomicUsize,
}

/// An instance of this struct is created when a thread becomes idle.
/// It is consumed when the thread finds work, and passed by `&mut`
/// reference for operations that preserve the idle state. (In other
/// words, producing one of these structs is evidence the thread is
/// idle.) It tracks state such as how long the thread has been idle.
pub(super) struct IdleState {
    /// What is worker index of the idle thread?
    worker_index: usize,

    /// How many rounds have we been circling without sleeping?
    rounds: u32,

    /// Once we become sleepy, what was the sleepy counter value?
    /// Set to `INVALID_SLEEPY_COUNTER` otherwise.
    jobs_counter: JobsEventCounter,

    /// Whether this idle episode reached the point of blocking on the
    /// condvar.
    slept: bool,

    /// Whether this idle thread holds one of the bounded searcher slots
    /// (only meaningful under `SpinPolicy::Bounded`).
    is_searcher: bool,

    /// Adaptive policy: this episode is the worker's main loop being idle
    /// (counted in `Sleep::idle_workers`), not a `join` waiting for a
    /// stolen job.
    top_level: bool,

    /// Adaptive policy: whether this worker keeps spinning once the pool
    /// is fully idle (no region in progress), as remembered from earlier
    /// episodes. Copied in from the worker at the start of the episode.
    spin_through_idle: bool,

    /// Adaptive policy: whether this worker spins while a region is in
    /// progress even though it expects long idle gaps, as remembered from
    /// the previous episode. Copied in from the worker at the start of
    /// the episode.
    spin_while_active: bool,

    /// Adaptive policy: a spin round observed the pool fully idle.
    saw_idle: bool,

    /// Adaptive policy: the current run of rounds was cut short by the
    /// policy rather than spun to its end.
    stopped_early: bool,

    /// Adaptive policy: this episode spun a full window after observing
    /// the pool fully idle (rather than stopping early at that point).
    spun_through_idle: bool,

    /// Adaptive policy: the pool was fully idle when this episode went
    /// to sleep.
    slept_idle: bool,

    /// Adaptive policy: this episode slept, but was woken sooner than a
    /// spin window would have lasted -- spinning would have found the
    /// work.
    short_sleep: bool,

    /// Adaptive policy: the current run of rounds began with a wake from
    /// sleep (so `saw_idle` describes the run after the last sleep).
    woke: bool,
}

/// The "sleep state" for an individual worker.
#[derive(Default)]
struct WorkerSleepState {
    /// Set to true when the worker goes to sleep; set to false when
    /// the worker is notified or when it wakes.
    is_blocked: Mutex<bool>,

    condvar: Condvar,
}

const ROUNDS_UNTIL_SLEEPY: u32 = 32;
const ROUNDS_UNTIL_SLEEPING: u32 = ROUNDS_UNTIL_SLEEPY + 1;

/// Adaptive policy: a sleep shorter than this, begun with the pool idle,
/// counts as a gap spinning would have bridged. On the order of what
/// `ROUNDS_UNTIL_SLEEPY` rounds of yield plus steal sweep take on an
/// unloaded machine; deliberately not measured, since on a loaded one a
/// window stretches (yields deschedule, sweeps contend) and a measured
/// yardstick would then call ever longer gaps short.
const SHORT_SLEEP: Duration = Duration::from_micros(200);

/// `Instant::now()` where the platform has a clock; `None` on targets
/// (wasm32-unknown-unknown) where it would panic. Without a clock the
/// adaptive policy never judges a sleep short, and so only ever turns
/// spinning through idle off.
#[inline]
fn now() -> Option<Instant> {
    if cfg!(target_arch = "wasm32") {
        None
    } else {
        Some(Instant::now())
    }
}

impl Sleep {
    pub(super) fn new(n_threads: usize, policy: SpinPolicy) -> Sleep {
        assert!(n_threads <= THREADS_MAX);
        Sleep {
            worker_sleep_states: (0..n_threads).map(|_| Default::default()).collect(),
            counters: AtomicCounters::new(),
            policy,
            idle_workers: CachePadded::new(AtomicUsize::new(0)),
            searchers: AtomicUsize::new(0),
        }
    }

    /// Under `ProducerBounded`, try to take a searcher slot; the budget
    /// is one searcher per currently-active (executing) worker, minimum
    /// one. Trivially true for the other policies (no shared state
    /// touched).
    #[inline]
    fn try_acquire_searcher(&self) -> bool {
        if self.policy != SpinPolicy::ProducerBounded {
            return true;
        }
        let inactive = self.counters.load(Ordering::Relaxed).inactive_threads();
        let max = Ord::max(self.worker_sleep_states.len() - inactive, 1);
        self.searchers
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < max).then_some(n + 1)
            })
            .is_ok()
    }

    #[inline]
    fn release_searcher(&self, idle_state: &mut IdleState) {
        if idle_state.is_searcher {
            self.searchers.fetch_sub(1, Ordering::Relaxed);
        }
        idle_state.is_searcher = false;
    }

    /// Adaptive policy: how many workers are idle at the top of their
    /// loop (maintained under the adaptive policy only).
    #[cfg(test)]
    pub(super) fn idle_workers(&self) -> usize {
        self.idle_workers.load(Ordering::Relaxed)
    }

    /// Adaptive policy: whether every worker is idle at the top of its
    /// loop, so no parallel region is in progress. Stealable work is only
    /// produced by a running region (injected jobs wake a thread
    /// explicitly), so once this is true spinning can only pay off if a
    /// new job is injected within the spin window.
    #[inline]
    fn pool_idle(&self) -> bool {
        self.idle_workers.load(Ordering::Relaxed) == self.worker_sleep_states.len()
    }

    #[inline]
    pub(super) fn start_looking(
        &self,
        worker_index: usize,
        top_level: bool,
        spin_through_idle: bool,
        spin_while_active: bool,
    ) -> IdleState {
        self.counters.add_inactive_thread();
        let top_level = top_level && self.policy == SpinPolicy::Adaptive;
        if top_level {
            self.idle_workers.fetch_add(1, Ordering::Relaxed);
        }

        // A worker denied a searcher slot under the bound skips the
        // yield/steal spin rounds and enters the sleepy protocol directly:
        // announce via the JEC, one final search, then block. This is the
        // stock protocol from round ROUNDS_UNTIL_SLEEPY on, so the
        // no-lost-wakeup reasoning is unchanged.
        let (spin, is_searcher) = match self.policy {
            SpinPolicy::Unbounded | SpinPolicy::Adaptive => (true, false),
            SpinPolicy::ProducerBounded => {
                let got = self.try_acquire_searcher();
                (got, got)
            }
        };
        IdleState {
            worker_index,
            rounds: if spin { 0 } else { ROUNDS_UNTIL_SLEEPY },
            jobs_counter: JobsEventCounter::DUMMY,
            slept: false,
            is_searcher,
            top_level,
            spin_through_idle,
            spin_while_active,
            saw_idle: false,
            stopped_early: false,
            spun_through_idle: false,
            slept_idle: false,
            short_sleep: false,
            woke: false,
        }
    }

    #[inline]
    pub(super) fn work_found(&self, idle_state: &mut IdleState) {
        self.release_searcher(idle_state);
        if idle_state.top_level {
            self.idle_workers.fetch_sub(1, Ordering::Relaxed);
        }
        // If we were the last idle thread and other threads are still sleeping,
        // then we should wake up another thread.
        let threads_to_wake = self.counters.sub_inactive_thread();
        self.wake_any_threads(threads_to_wake as u32);
    }

    #[inline]
    pub(super) fn no_work_found(
        &self,
        idle_state: &mut IdleState,
        latch: &CoreLatch,
        has_injected_jobs: impl FnOnce() -> bool,
    ) {
        if idle_state.rounds < ROUNDS_UNTIL_SLEEPY {
            // Adaptive policy: decide each round whether to spin on or go
            // sleepy now (announce via the JEC, one final search, then
            // block -- the stock protocol from ROUNDS_UNTIL_SLEEPY on, so
            // the no-lost-wakeup reasoning is unchanged).
            if self.policy == SpinPolicy::Adaptive {
                let spin = if self.pool_idle() {
                    // No region in progress: only a newly injected job
                    // can end the wait. Spin on only if this worker
                    // remembers such gaps being shorter than a window.
                    idle_state.saw_idle = true;
                    idle_state.spin_through_idle
                } else {
                    // A region is in progress and may produce stealable
                    // work. Spin on if this worker expects more work
                    // within a window either way, or if its last search
                    // succeeded without a sleep -- work is flowing to it
                    // here. A worker that had to sleep for its last job
                    // does not: on a pool much wider than a region's
                    // parallelism most workers only ever sweep empty
                    // deques, slowing the ones doing the work.
                    idle_state.spin_through_idle || idle_state.spin_while_active
                };
                if !spin {
                    idle_state.stopped_early = true;
                    idle_state.rounds = ROUNDS_UNTIL_SLEEPY;
                    return;
                }
            }
            thread::yield_now();
            idle_state.rounds += 1;
        } else if idle_state.rounds == ROUNDS_UNTIL_SLEEPY {
            // Record whether this run of rounds spun a whole window
            // through an idle pool; an early stop lands here too.
            idle_state.spun_through_idle = idle_state.saw_idle && !idle_state.stopped_early;
            idle_state.stopped_early = false;
            idle_state.jobs_counter = self.announce_sleepy();
            idle_state.rounds += 1;
            thread::yield_now();
        } else if idle_state.rounds < ROUNDS_UNTIL_SLEEPING {
            idle_state.rounds += 1;
            thread::yield_now();
        } else {
            debug_assert_eq!(idle_state.rounds, ROUNDS_UNTIL_SLEEPING);
            self.sleep(idle_state, latch, has_injected_jobs);
        }
    }

    #[cold]
    fn announce_sleepy(&self) -> JobsEventCounter {
        self.counters
            .increment_jobs_event_counter_if(JobsEventCounter::is_active)
            .jobs_counter()
    }

    #[cold]
    fn sleep(
        &self,
        idle_state: &mut IdleState,
        latch: &CoreLatch,
        has_injected_jobs: impl FnOnce() -> bool,
    ) {
        let worker_index = idle_state.worker_index;

        if !latch.get_sleepy() {
            return;
        }

        let sleep_state = &self.worker_sleep_states[worker_index];
        let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
        debug_assert!(!*is_blocked);

        // Our latch was signalled. We should wake back up fully as we
        // will have some stuff to do.
        if !latch.fall_asleep() {
            idle_state.wake_fully();
            return;
        }

        loop {
            let counters = self.counters.load(Ordering::SeqCst);

            // Check if the JEC has changed since we got sleepy.
            debug_assert!(idle_state.jobs_counter.is_sleepy());
            if counters.jobs_counter() != idle_state.jobs_counter {
                // JEC has changed, so a new job was posted, but for some reason
                // we didn't see it. We should return to just before the SLEEPY
                // state so we can do another search and (if we fail to find
                // work) go back to sleep.
                idle_state.wake_partly();
                latch.wake_up();
                return;
            }

            // Otherwise, let's move from IDLE to SLEEPING.
            if self.counters.try_add_sleeping_thread(counters) {
                break;
            }
        }

        // Successfully registered as asleep. A bounded searcher gives up
        // its slot so another idle worker may spin in its place.
        self.release_searcher(idle_state);
        idle_state.slept = true;
        // Judge each sleep on its own: an episode may sleep more than once.
        idle_state.slept_idle = idle_state.saw_idle;
        idle_state.short_sleep = false;
        let blocked_at = if self.policy == SpinPolicy::Adaptive {
            now()
        } else {
            None
        };

        // We have one last check for injected jobs to do. This protects against
        // deadlock in the very unlikely event that
        //
        // - an external job is being injected while we are sleepy
        // - that job triggers the rollover over the JEC such that we don't see it
        // - we are the last active worker thread
        std::sync::atomic::fence(Ordering::SeqCst);
        if has_injected_jobs() {
            // If we see an externally injected job, then we have to 'wake
            // ourselves up'. (Ordinarily, `sub_sleeping_thread` is invoked by
            // the one that wakes us.)
            self.counters.sub_sleeping_thread();
            // Work arrived before we even blocked: as short a sleep as
            // there is.
            idle_state.short_sleep = true;
        } else {
            // If we don't see an injected job (the normal case), then flag
            // ourselves as asleep and wait till we are notified.
            //
            // (Note that `is_blocked` is held under a mutex and the mutex was
            // acquired *before* we incremented the "sleepy counter". This means
            // that whomever is coming to wake us will have to wait until we
            // release the mutex in the call to `wait`, so they will see this
            // boolean as true.)
            *is_blocked = true;
            while *is_blocked {
                is_blocked = sleep_state.condvar.wait(is_blocked).unwrap();
            }
            // A sleep shorter than a spin window means the work we were
            // woken for arrived while spinning would still have been
            // searching: the pool was not idle after all.
            if let Some(at) = blocked_at {
                idle_state.short_sleep = at.elapsed() < SHORT_SLEEP;
            }
        }

        // Update other state:
        idle_state.wake_fully();
        // `wake_fully` grants a fresh spin budget; under a bound the budget
        // requires a slot. The adaptive policy starts a fresh window: what
        // it saw of the pool before sleeping no longer holds.
        match self.policy {
            SpinPolicy::Unbounded => {}
            SpinPolicy::Adaptive => {
                // A fresh run of rounds; what the sleep just ended showed
                // (`spun_through_idle`, `slept_idle`, `short_sleep`) is
                // kept for the verdict once work is found.
                idle_state.saw_idle = false;
                idle_state.woke = true;
            }
            SpinPolicy::ProducerBounded => {
                idle_state.is_searcher = self.try_acquire_searcher();
                if !idle_state.is_searcher {
                    idle_state.rounds = ROUNDS_UNTIL_SLEEPY;
                }
            }
        }
        latch.wake_up();
    }

    /// Notify the given thread that it should wake up (if it is
    /// sleeping).  When this method is invoked, we typically know the
    /// thread is asleep, though in rare cases it could have been
    /// awoken by (e.g.) new work having been posted.
    pub(super) fn notify_worker_latch_is_set(&self, target_worker_index: usize) {
        self.wake_specific_thread(target_worker_index);
    }

    /// Signals that `num_jobs` new jobs were injected into the thread
    /// pool from outside. This function will ensure that there are
    /// threads available to process them, waking threads from sleep
    /// if necessary.
    ///
    /// # Parameters
    ///
    /// - `num_jobs` -- lower bound on number of jobs available for stealing.
    ///   We'll try to get at least one thread per job.
    #[inline]
    pub(super) fn new_injected_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // This fence is needed to guarantee that threads
        // as they are about to fall asleep, observe any
        // new jobs that may have been injected.
        std::sync::atomic::fence(Ordering::SeqCst);

        self.new_jobs(num_jobs, queue_was_empty)
    }

    /// Signals that `num_jobs` new jobs were pushed onto a thread's
    /// local deque. This function will try to ensure that there are
    /// threads available to process them, waking threads from sleep
    /// if necessary. However, this is not guaranteed: under certain
    /// race conditions, the function may fail to wake any new
    /// threads; in that case the existing thread should eventually
    /// pop the job.
    ///
    /// # Parameters
    ///
    /// - `num_jobs` -- lower bound on number of jobs available for stealing.
    ///   We'll try to get at least one thread per job.
    #[inline]
    pub(super) fn new_internal_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        self.new_jobs(num_jobs, queue_was_empty)
    }

    /// Common helper for `new_injected_jobs` and `new_internal_jobs`.
    #[inline]
    fn new_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // Read the counters and -- if sleepy workers have announced themselves
        // -- announce that there is now work available. The final value of `counters`
        // with which we exit the loop thus corresponds to a state when
        let counters = self
            .counters
            .increment_jobs_event_counter_if(JobsEventCounter::is_sleepy);
        let num_awake_but_idle = counters.awake_but_idle_threads();
        let num_sleepers = counters.sleeping_threads();

        if num_sleepers == 0 {
            // nobody to wake
            return;
        }

        // Promote from u16 to u32 so we can interoperate with
        // num_jobs more easily.
        let num_awake_but_idle = num_awake_but_idle as u32;
        let num_sleepers = num_sleepers as u32;

        // If the queue is non-empty, then we always wake up a worker
        // -- clearly the existing idle jobs aren't enough. Otherwise,
        // check to see if we have enough idle workers.
        if !queue_was_empty {
            let num_to_wake = Ord::min(num_jobs, num_sleepers);
            self.wake_any_threads(num_to_wake);
        } else if num_awake_but_idle < num_jobs {
            let num_to_wake = Ord::min(num_jobs - num_awake_but_idle, num_sleepers);
            self.wake_any_threads(num_to_wake);
        }
    }

    #[cold]
    fn wake_any_threads(&self, mut num_to_wake: u32) {
        if num_to_wake > 0 {
            for i in 0..self.worker_sleep_states.len() {
                if self.wake_specific_thread(i) {
                    num_to_wake -= 1;
                    if num_to_wake == 0 {
                        return;
                    }
                }
            }
        }
    }

    fn wake_specific_thread(&self, index: usize) -> bool {
        let sleep_state = &self.worker_sleep_states[index];

        let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
        if *is_blocked {
            *is_blocked = false;
            sleep_state.condvar.notify_one();

            // When the thread went to sleep, it will have incremented
            // this value. When we wake it, its our job to decrement
            // it. We could have the thread do it, but that would
            // introduce a delay between when the thread was
            // *notified* and when this counter was decremented. That
            // might mislead people with new work into thinking that
            // there are sleeping threads that they should try to
            // wake, when in fact there is nothing left for them to
            // do.
            self.counters.sub_sleeping_thread();

            true
        } else {
            false
        }
    }
}

impl IdleState {
    fn wake_fully(&mut self) {
        self.rounds = 0;
        self.jobs_counter = JobsEventCounter::DUMMY;
    }

    fn wake_partly(&mut self) {
        self.rounds = ROUNDS_UNTIL_SLEEPY;
        self.jobs_counter = JobsEventCounter::DUMMY;
    }

    /// Adaptive policy: whether this episode found its work without
    /// having to sleep, i.e. work was flowing to this worker.
    pub(super) fn found_awake(&self) -> bool {
        !self.slept
    }

    /// Adaptive policy: what this episode showed about spinning through
    /// an idle pool, for the worker to remember.
    pub(super) fn idle_gap(&self) -> Verdict {
        if !self.slept || self.woke {
            // The last run of rounds ended by finding work.
            if self.saw_idle {
                // Work arrived while spinning through an idle pool.
                return Verdict::Hit;
            } else if !self.slept {
                // Stole from a still-active pool: says nothing about gaps.
                return Verdict::Unknown;
            }
            // Woke and found work without spinning through idle: the
            // sleep just ended is the evidence.
        }
        if self.short_sleep && self.slept_idle {
            // Slept with the pool idle, but work arrived within a window:
            // spinning would have found it. (A short sleep while a region
            // was in progress is churn within it and says nothing about
            // the gaps between regions.)
            Verdict::Hit
        } else if self.spun_through_idle {
            // Spun a whole window through an idle pool for nothing.
            Verdict::Miss
        } else {
            // Slept without spinning through idle (stopped early, or a
            // region was in progress when we went sleepy): no verdict.
            Verdict::Unknown
        }
    }
}

/// Adaptive policy: what an idle episode showed about whether spinning
/// through an idle pool would find work within a spin window.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// Spinning found work, or would have.
    Hit,
    /// A full spin window found nothing.
    Miss,
    /// The episode did not test it.
    Unknown,
}
