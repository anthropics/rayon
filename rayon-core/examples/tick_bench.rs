//! Tick-shaped workload benchmark for the idle-spin policies.
//!
//! Models a control loop (e.g. a reconcile tick) that runs several short
//! parallel regions back to back, separated by short sequential stretches,
//! with an idle gap between ticks. This is the shape where the per-worker
//! adaptive policy can lose: every region ends with the pool idle, so
//! workers sleep after each region and have to be woken for the next one.
//!
//! Configuration is via environment variables (all optional):
//!
//! | var                | default   | meaning                                        |
//! |--------------------|-----------|------------------------------------------------|
//! | `BENCH_POLICY`     | adaptive  | adaptive \| bounded \| unbounded               |
//! | `BENCH_THREADS`    | ncpu/2    | pool width                                     |
//! | `BENCH_TICKS`      | 2000      | number of ticks measured                       |
//! | `BENCH_REGIONS`    | 8         | parallel regions per tick                      |
//! | `BENCH_LEAVES`     | 64        | leaf tasks per region (split by nested join)   |
//! | `BENCH_LEAF_US`    | 10        | busy work per leaf task, microseconds          |
//! | `BENCH_GAP_US`     | 20        | sequential work between regions, microseconds  |
//! | `BENCH_TICK_MS`    | 10        | tick period (idle gap = period - tick work)    |
//!
//! Output: one line of tab-separated fields (policy, threads, tick p50/p99,
//! region p50/p99, process CPU in cores over the measured window).
//!
//! ```text
//! cargo run --release -p rayon-core --example tick_bench
//! BENCH_POLICY=unbounded cargo run --release -p rayon-core --example tick_bench
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Burn roughly `us` microseconds of CPU without touching memory.
fn busy(us: u64) {
    let end = Instant::now() + Duration::from_micros(us);
    let mut x = 0u64;
    while Instant::now() < end {
        for _ in 0..64 {
            x = black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
        }
    }
}

/// A parallel region: `leaves` leaf tasks reached by recursive `join`,
/// the way `par_iter` and nested `par_join!` split their work.
fn region(leaves: usize, leaf_us: u64) {
    if leaves <= 1 {
        busy(leaf_us);
    } else {
        let half = leaves / 2;
        rayon_core::join(|| region(half, leaf_us), || region(leaves - half, leaf_us));
    }
}

/// utime + stime of this process, in seconds (Linux).
fn proc_cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the parenthesised comm: index 13/14 (0-based from field 0)
    // are utime/stime in clock ticks.
    let rest = stat.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks: u64 = f.get(11).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0)
        + f.get(12).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    ticks as f64 / 100.0
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn main() {
    let policy: String = env_or("BENCH_POLICY", "adaptive".to_string());
    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let threads: usize = env_or("BENCH_THREADS", (ncpu / 2).max(1));
    let ticks: usize = env_or("BENCH_TICKS", 2000);
    let regions: usize = env_or("BENCH_REGIONS", 8);
    let leaves: usize = env_or("BENCH_LEAVES", 64);
    let leaf_us: u64 = env_or("BENCH_LEAF_US", 10);
    let gap_us: u64 = env_or("BENCH_GAP_US", 20);
    let tick_ms: u64 = env_or("BENCH_TICK_MS", 10);

    let builder = rayon_core::ThreadPoolBuilder::new().num_threads(threads);
    let builder = match policy.as_str() {
        "adaptive" => builder,
        "bounded" => builder.bounded_searchers(),
        "unbounded" => builder.unbounded_searchers(),
        other => panic!("BENCH_POLICY={other}: expected adaptive|bounded|unbounded"),
    };
    let pool = builder.build().unwrap();

    // Warm up: get every worker created and scheduled once.
    for _ in 0..50 {
        pool.install(|| region(leaves, leaf_us));
        std::thread::sleep(Duration::from_millis(tick_ms));
    }

    let mut tick_lat = Vec::with_capacity(ticks);
    let mut region_lat = Vec::with_capacity(ticks * regions);
    let mut by_index: Vec<Vec<f64>> = vec![Vec::with_capacity(ticks); regions];
    let period = Duration::from_millis(tick_ms);
    let cpu0 = proc_cpu_seconds();
    let wall0 = Instant::now();
    let mut next = wall0;
    for _ in 0..ticks {
        next += period;
        let t0 = Instant::now();
        for (r, lat) in by_index.iter_mut().enumerate() {
            if r > 0 {
                busy(gap_us);
            }
            let r0 = Instant::now();
            pool.install(|| region(leaves, leaf_us));
            let ms = r0.elapsed().as_secs_f64() * 1e3;
            region_lat.push(ms);
            lat.push(ms);
        }
        tick_lat.push(t0.elapsed().as_secs_f64() * 1e3);
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now;
        }
    }
    let wall = wall0.elapsed().as_secs_f64();
    let cpu = proc_cpu_seconds() - cpu0;
    if std::env::var_os("BENCH_VERBOSE").is_some() {
        for (i, v) in by_index.iter_mut().enumerate() {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "  region[{i}] p50={:.3}ms p90={:.3}ms p99={:.3}ms",
                percentile(v, 0.5),
                percentile(v, 0.9),
                percentile(v, 0.99)
            );
        }
    }
    tick_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    region_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Ideal region time with perfect parallelism, for reference.
    let ideal_region_ms = (leaves as f64 * leaf_us as f64 / threads as f64).ceil() / 1e3;
    println!(
        "policy={policy}\tthreads={threads}\tregions={regions}\tleaves={leaves}\tleaf_us={leaf_us}\tgap_us={gap_us}\ttick_ms={tick_ms}\t\
         tick_p50_ms={:.3}\ttick_p99_ms={:.3}\tregion_p50_ms={:.3}\tregion_p99_ms={:.3}\tregion_ideal_ms={:.3}\tcpu_cores={:.2}",
        percentile(&tick_lat, 0.5),
        percentile(&tick_lat, 0.99),
        percentile(&region_lat, 0.5),
        percentile(&region_lat, 0.99),
        ideal_region_ms,
        cpu / wall,
    );
}
