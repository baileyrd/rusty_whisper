//! Native microphone capture, via the `cpal` crate — mirrors whisper.cpp's
//! `audio_async` (`examples/common-sdl.cpp`), used by `whisper-stream` and
//! `whisper-command` (`examples/stream/stream.cpp`,
//! `examples/command/command.cpp` in whisper.cpp v1.9.1).
//!
//! Opt-in via the `mic` feature: `cpal` is rusty-whisper's only crates.io
//! dependency, so it stays out of the default build.
//!
//! Unlike SDL (which whisper.cpp asks for 16kHz mono f32 directly, letting
//! SDL itself resample/convert under the hood), `cpal` hands back whatever
//! rate/channel-count/sample-format the device's default input config
//! reports. `to_mono`/`resample_linear` below convert each callback's
//! samples to 16kHz mono f32 before they reach the ring buffer, so
//! everything downstream sees exactly what whisper.cpp's `audio_async`
//! would have produced.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// whisper.cpp's `WHISPER_SAMPLE_RATE`.
pub const SAMPLE_RATE: u32 = 16_000;

/// Fixed-capacity ring buffer matching `audio_async`'s wraparound math:
/// `pos` always points at the next write slot, `len` is the number of
/// currently-valid (not-yet-overwritten) samples.
struct RingBuffer {
    audio: Vec<f32>,
    pos: usize,
    len: usize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            audio: vec![0.0; capacity.max(1)],
            pos: 0,
            len: 0,
        }
    }

    /// Appends new samples, overwriting the oldest ones once the ring is
    /// full. A single push bigger than the whole capacity keeps only its
    /// newest samples — mirrors `audio_async::callback`'s truncation when
    /// an SDL/cpal chunk exceeds the buffer size.
    fn push(&mut self, samples: &[f32]) {
        let cap = self.audio.len();
        let samples = if samples.len() > cap {
            &samples[samples.len() - cap..]
        } else {
            samples
        };
        let n = samples.len();
        if n == 0 {
            return;
        }
        if self.pos + n > cap {
            let n0 = cap - self.pos;
            self.audio[self.pos..].copy_from_slice(&samples[..n0]);
            self.audio[..n - n0].copy_from_slice(&samples[n0..]);
        } else {
            self.audio[self.pos..self.pos + n].copy_from_slice(samples);
        }
        self.pos = (self.pos + n) % cap;
        self.len = (self.len + n).min(cap);
    }

    /// Returns the last `n_samples` pushed, oldest-first, ending at the
    /// most recently written sample — mirrors `audio_async::get`.
    fn get(&self, n_samples: usize) -> Vec<f32> {
        let cap = self.audio.len();
        let n = n_samples.min(self.len);
        if n == 0 {
            return Vec::new();
        }
        let s0 = (self.pos + cap - n) % cap;
        let mut out = Vec::with_capacity(n);
        if s0 + n > cap {
            let n0 = cap - s0;
            out.extend_from_slice(&self.audio[s0..]);
            out.extend_from_slice(&self.audio[..n - n0]);
        } else {
            out.extend_from_slice(&self.audio[s0..s0 + n]);
        }
        out
    }

    /// Resets the ring to empty without touching its contents — mirrors
    /// `audio_async::clear()`.
    fn clear(&mut self) {
        self.pos = 0;
        self.len = 0;
    }
}

/// Downmixes interleaved multi-channel samples to mono by averaging each
/// frame — cpal gives us the device's native channel count; SDL did this
/// kind of conversion for whisper.cpp internally.
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let channels = channels as usize;
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Linear resampling from the device's native rate to `to_rate`. cpal
/// (unlike SDL's `audio_async::init`) doesn't resample for us, so this
/// runs on every callback before samples reach the ring buffer.
fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).round() as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let i0 = src.floor() as usize;
            let frac = (src - i0 as f64) as f32;
            let s0 = samples[i0.min(samples.len() - 1)];
            let s1 = samples[(i0 + 1).min(samples.len() - 1)];
            s0 + (s1 - s0) * frac
        })
        .collect()
}

fn log_stream_error(err: cpal::StreamError) {
    crate::log::log(format!("mic capture stream error: {err}"));
}

/// Lists input device names for the default host — purely informational
/// (e.g. for a `--capture-device` picker), mirrors `audio_async::init`'s
/// enumeration-for-logging via `SDL_GetNumAudioDevices`/`GetAudioDeviceName`.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices.filter_map(|d| d.name().ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// A running (or paused) capture session, backed by a `len_ms`-capacity
/// ring buffer of 16kHz mono f32 samples.
pub struct MicCapture {
    stream: Stream,
    ring: Arc<Mutex<RingBuffer>>,
    running: Arc<AtomicBool>,
    len_ms: usize,
}

/// Opens a capture device (`device_name`, or the OS default input device
/// when `None`) and builds a `len_ms`-capacity ring buffer for it — the
/// cpal equivalent of `audio_async::init`. Capture doesn't start until
/// [`MicCapture::resume`] is called, matching `audio_async`'s
/// open-then-`resume()` lifecycle.
pub fn init(len_ms: usize, device_name: Option<&str>) -> Result<MicCapture, String> {
    let host = cpal::default_host();
    let device = match device_name {
        Some(name) => host
            .input_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| format!("no input device named {name:?}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| "no default input device found".to_string())?,
    };

    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let in_rate = config.sample_rate().0;
    let in_channels = config.channels();
    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();

    let capacity = (SAMPLE_RATE as usize * len_ms) / 1000;
    let ring = Arc::new(Mutex::new(RingBuffer::new(capacity)));
    let running = Arc::new(AtomicBool::new(false));

    let ring_cb = ring.clone();
    let running_cb = running.clone();
    let push = move |native: &[f32]| {
        if !running_cb.load(Ordering::Acquire) {
            return;
        }
        let mono = to_mono(native, in_channels);
        let resampled = resample_linear(&mono, in_rate, SAMPLE_RATE);
        ring_cb.lock().unwrap().push(&resampled);
    };

    let stream = match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &_| push(data),
                log_stream_error,
                None,
            )
            .map_err(|e| e.to_string())?,
        SampleFormat::I16 => device
            .build_input_stream(
                &stream_config,
                move |data: &[i16], _: &_| {
                    let f: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    push(&f);
                },
                log_stream_error,
                None,
            )
            .map_err(|e| e.to_string())?,
        SampleFormat::U16 => device
            .build_input_stream(
                &stream_config,
                move |data: &[u16], _: &_| {
                    let f: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    push(&f);
                },
                log_stream_error,
                None,
            )
            .map_err(|e| e.to_string())?,
        other => return Err(format!("unsupported input sample format: {other:?}")),
    };

    Ok(MicCapture {
        stream,
        ring,
        running,
        len_ms,
    })
}

impl MicCapture {
    /// Starts (or resumes) capture — mirrors `audio_async::resume()`.
    pub fn resume(&self) -> Result<(), String> {
        self.stream.play().map_err(|e| e.to_string())?;
        self.running.store(true, Ordering::Release);
        Ok(())
    }

    /// Stops capture without closing the device — mirrors
    /// `audio_async::pause()`.
    pub fn pause(&self) -> Result<(), String> {
        self.running.store(false, Ordering::Release);
        self.stream.pause().map_err(|e| e.to_string())
    }

    /// Resets the ring buffer to empty — mirrors `audio_async::clear()`.
    pub fn clear(&self) {
        self.ring.lock().unwrap().clear();
    }

    /// Returns the last `ms` milliseconds of captured audio, oldest-first.
    /// `ms <= 0` means "the whole buffer" — mirrors `audio_async::get()`.
    pub fn get(&self, ms: i64) -> Vec<f32> {
        let ms = if ms <= 0 { self.len_ms as i64 } else { ms };
        let n_samples = ((SAMPLE_RATE as i64 * ms) / 1000).max(0) as usize;
        self.ring.lock().unwrap().get(n_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_basic_push_get() {
        let mut rb = RingBuffer::new(10);
        rb.push(&[1.0, 2.0, 3.0]);
        assert_eq!(rb.get(3), vec![1.0, 2.0, 3.0]);
        assert_eq!(rb.get(2), vec![2.0, 3.0]);
        assert_eq!(rb.get(10), vec![1.0, 2.0, 3.0]); // clamped to len
    }

    #[test]
    fn ring_buffer_wraps_across_multiple_pushes() {
        let mut rb = RingBuffer::new(5);
        rb.push(&[1.0, 2.0, 3.0]);
        rb.push(&[4.0, 5.0, 6.0]); // wraps: buffer now holds [6,2,3,4,5] with pos=1
        assert_eq!(rb.get(5), vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(rb.get(2), vec![5.0, 6.0]);
    }

    #[test]
    fn ring_buffer_push_bigger_than_capacity_keeps_newest() {
        let mut rb = RingBuffer::new(3);
        rb.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(rb.get(3), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn ring_buffer_clear_resets_to_empty() {
        let mut rb = RingBuffer::new(5);
        rb.push(&[1.0, 2.0, 3.0]);
        rb.clear();
        assert_eq!(rb.get(5), Vec::<f32>::new());
        rb.push(&[9.0]);
        assert_eq!(rb.get(5), vec![9.0]);
    }

    #[test]
    fn ring_buffer_get_zero_or_empty() {
        let rb = RingBuffer::new(5);
        assert_eq!(rb.get(0), Vec::<f32>::new());
        assert_eq!(rb.get(5), Vec::<f32>::new());
    }

    #[test]
    fn to_mono_passthrough_for_mono_input() {
        assert_eq!(to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_mono_averages_stereo_frames() {
        assert_eq!(to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
    }

    #[test]
    fn resample_linear_same_rate_is_passthrough() {
        let s = vec![1.0, 2.0, 3.0];
        assert_eq!(resample_linear(&s, 16000, 16000), s);
    }

    #[test]
    fn resample_linear_empty_input() {
        assert_eq!(resample_linear(&[], 44100, 16000), Vec::<f32>::new());
    }

    #[test]
    fn resample_linear_downsamples_to_expected_length() {
        let input: Vec<f32> = (0..441).map(|i| i as f32).collect();
        let out = resample_linear(&input, 44100, 16000);
        // 441 samples at 44100Hz is 10ms; at 16000Hz that's 160 samples.
        assert_eq!(out.len(), 160);
    }

    #[test]
    fn resample_linear_upsamples_to_expected_length() {
        let input: Vec<f32> = (0..160).map(|i| i as f32).collect();
        let out = resample_linear(&input, 16000, 44100);
        assert_eq!(out.len(), 441);
    }

    #[test]
    fn list_input_devices_does_not_panic() {
        // No real audio hardware in CI/sandboxes — this just has to not
        // panic, an empty list is a valid (and expected) answer here.
        let _ = list_input_devices();
    }

    #[test]
    fn init_returns_a_result_without_panicking() {
        // Headless environments typically have no capture device at all;
        // this is a genuine integration test of that "no device" path,
        // not a mock — same shape as the ffmpeg-not-found test in
        // `server.rs`. If a device *is* present (e.g. a real desktop),
        // clean up instead of leaving the stream running.
        match init(1000, None) {
            Ok(cap) => {
                let _ = cap.pause();
            }
            Err(e) => assert!(!e.is_empty()),
        }
    }
}
