//! Per-stage timing/perf instrumentation — whisper.h's `whisper_timings`/
//! `whisper_get_timings`/`whisper_print_timings`/`whisper_reset_timings`,
//! plus `whisper_print_system_info`.
//!
//! Global, process-wide accumulators (matching whisper.cpp's own
//! per-`whisper_context` running totals) rather than per-`Model` state, so
//! the parallel `--processors` transcription path (separate threads, each
//! with its own `Model` reference) can record into the same counters
//! without needing `&mut Model`.
//!
//! These numbers cover the same major named stages whisper.cpp's own
//! timings do (mel, encode, decode, sample) — they are not a full,
//! gapless partition of wall-clock time; bookkeeping in between (logit
//! suppression rules, segment building, beam-search candidate ranking)
//! isn't separately charged to any stage, same as upstream.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static MEL_US: AtomicU64 = AtomicU64::new(0);
static ENCODE_US: AtomicU64 = AtomicU64::new(0);
static DECODE_US: AtomicU64 = AtomicU64::new(0);
static SAMPLE_US: AtomicU64 = AtomicU64::new(0);

/// Per-stage timing breakdown, in milliseconds — matches whisper.cpp's
/// `whisper_timings` fields, read via [`get_timings`].
///
/// `sample_ms` is measured only along the greedy decode path
/// (`decode_window_once`'s explicit token-sampling step); beam search's
/// per-candidate top-k ranking is interleaved with logit-suppression and
/// bookkeeping across every live beam in a way that has no single clean
/// seam to instrument without added risk to the validated beam-search
/// implementation, so beam search's forward passes are charged to
/// `decode_ms` and its candidate selection isn't separately measured.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub sample_ms: f64,
}

impl Timings {
    pub fn total_ms(&self) -> f64 {
        self.mel_ms + self.encode_ms + self.decode_ms + self.sample_ms
    }
}

/// Current accumulated timings — `whisper_get_timings`.
pub fn get_timings() -> Timings {
    Timings {
        mel_ms: MEL_US.load(Ordering::Relaxed) as f64 / 1000.0,
        encode_ms: ENCODE_US.load(Ordering::Relaxed) as f64 / 1000.0,
        decode_ms: DECODE_US.load(Ordering::Relaxed) as f64 / 1000.0,
        sample_ms: SAMPLE_US.load(Ordering::Relaxed) as f64 / 1000.0,
    }
}

/// Zero every accumulator — `whisper_reset_timings`.
pub fn reset_timings() {
    MEL_US.store(0, Ordering::Relaxed);
    ENCODE_US.store(0, Ordering::Relaxed);
    DECODE_US.store(0, Ordering::Relaxed);
    SAMPLE_US.store(0, Ordering::Relaxed);
}

/// Log the current breakdown through [`crate::log`] — `whisper_print_timings`.
pub fn print_timings() {
    let t = get_timings();
    crate::log::log(format!(
        "timings: mel = {:.2} ms, encode = {:.2} ms, decode = {:.2} ms, sample = {:.2} ms, total = {:.2} ms",
        t.mel_ms,
        t.encode_ms,
        t.decode_ms,
        t.sample_ms,
        t.total_ms()
    ));
}

pub(crate) fn record_mel(d: Duration) {
    MEL_US.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}
pub(crate) fn record_encode(d: Duration) {
    ENCODE_US.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}
pub(crate) fn record_decode(d: Duration) {
    DECODE_US.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}
pub(crate) fn record_sample(d: Duration) {
    SAMPLE_US.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}

/// Time `f`, recording the elapsed duration via `record`, and returning
/// `f`'s own result unchanged.
pub(crate) fn timed<T>(record: impl FnOnce(Duration), f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    record(start.elapsed());
    out
}

/// A system/build info summary — `whisper_print_system_info`. Reports only
/// what this crate's own accelerated paths actually use (runtime-detected
/// AVX2 dequantization, AVX-512 VNNI int8 dot products) plus the detected
/// thread count, rather than listing hardware features (NEON, CUDA,
/// Metal, ...) this pure-CPU, pure-Rust crate has no code path for.
pub fn print_system_info() -> String {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::arch::is_x86_feature_detected!("avx2") as i32;
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = 0;
    #[cfg(target_arch = "x86_64")]
    let avx512_vnni = (std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512vl")) as i32;
    #[cfg(not(target_arch = "x86_64"))]
    let avx512_vnni = 0;
    format!("system_info: n_threads = {threads} | AVX2 = {avx2} | AVX512_VNNI = {avx512_vnni}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_records_and_reset_zeroes_every_stage() {
        // Both halves share the same process-global atomics, so they're
        // kept in one test rather than split across two (which could race
        // against each other under parallel test execution — nothing else
        // in this crate's test suite drives the real transcribe pipeline,
        // so these atomics are otherwise untouched by concurrent tests).
        reset_timings();
        let result = timed(record_mel, || {
            std::thread::sleep(Duration::from_millis(2));
            42
        });
        assert_eq!(result, 42);
        assert!(get_timings().mel_ms >= 1.0);

        timed(record_encode, || {
            std::thread::sleep(Duration::from_millis(1))
        });
        timed(record_decode, || {
            std::thread::sleep(Duration::from_millis(1))
        });
        timed(record_sample, || {
            std::thread::sleep(Duration::from_millis(1))
        });
        assert!(get_timings().total_ms() > 0.0);

        reset_timings();
        assert_eq!(get_timings(), Timings::default());
    }

    #[test]
    fn total_ms_sums_all_four_stages() {
        let t = Timings {
            mel_ms: 1.0,
            encode_ms: 2.0,
            decode_ms: 3.0,
            sample_ms: 4.0,
        };
        assert_eq!(t.total_ms(), 10.0);
    }

    #[test]
    fn print_system_info_reports_thread_count_and_known_flags() {
        let info = print_system_info();
        assert!(info.contains("n_threads ="));
        assert!(info.contains("AVX2 ="));
        assert!(info.contains("AVX512_VNNI ="));
    }
}
