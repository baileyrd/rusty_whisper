//! `whisper-stream`: real-time microphone transcription — mirrors
//! whisper.cpp's `examples/stream/stream.cpp` (v1.9.1). Two modes:
//! a fixed sliding window (default) redraws its last line in place as the
//! window grows, trimming back to `--keep` every `--length`/`--step - 1`
//! iterations; `--step 0` instead waits for a simple energy/high-pass VAD
//! ("has it gone quiet after talking?") to trigger a full-window decode
//! with real timestamps.
//!
//! Requires the `mic` feature (native capture via `cpal`) — this binary
//! is not built without it (see `Cargo.toml`'s `required-features`).

use std::io::Write as _;
use std::time::{Duration, Instant};

use rusty_whisper::{mic, model, transcribe, wav};

struct Params {
    model_path: Option<String>,
    step_ms: i64,
    length_ms: i64,
    keep_ms: i64,
    capture_id: i64,
    audio_ctx: Option<usize>,
    beam_size: i64,
    vad_thold: f32,
    freq_thold: f32,
    translate: bool,
    no_fallback: bool,
    keep_context: bool,
    tinydiarize: bool,
    save_audio: bool,
    language: String,
    list_devices: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            model_path: None,
            step_ms: 3000,
            length_ms: 10000,
            keep_ms: 200,
            capture_id: -1,
            audio_ctx: None,
            beam_size: -1,
            vad_thold: 0.6,
            freq_thold: 100.0,
            translate: false,
            no_fallback: false,
            keep_context: false,
            tinydiarize: false,
            save_audio: false,
            language: "en".to_string(),
            list_devices: false,
        }
    }
}

fn print_help() {
    eprintln!("whisper-stream: real-time microphone transcription (requires --features mic)");
    eprintln!("  --model, -m PATH        model to load (required)");
    eprintln!(
        "  --step MS               sliding-window step (default 3000; 0 = VAD-triggered mode)"
    );
    eprintln!("  --length MS             window length (default 10000)");
    eprintln!("  --keep MS               audio kept from the previous window (default 200)");
    eprintln!("  -c, --capture N         capture device index (default: OS default device)");
    eprintln!("  --list-devices          list capture device names (with their index) and exit");
    eprintln!("  -mt, --max-tokens N     accepted for CLI parity; whisper.cpp itself always");
    eprintln!(
        "                          overrides this to unlimited in stream.cpp, so this is too"
    );
    eprintln!("  -ac, --audio-ctx N      limit encoder audio context (accepted, currently unused; 0 = full)");
    eprintln!("  -bs, --beam-size N      beam search width (>1 enables it; default -1 = greedy)");
    eprintln!("  -vth, --vad-thold F     energy VAD threshold (VAD mode only, default 0.6)");
    eprintln!("  -fth, --freq-thold F    high-pass filter cutoff Hz before VAD (default 100.0)");
    eprintln!("  -tr, --translate        translate to English");
    eprintln!("  -nf, --no-fallback      disable the temperature fallback ladder");
    eprintln!("  -kc, --keep-context     carry decoded text forward as a prompt (sliding-window mode only)");
    eprintln!(
        "  -tdrz, --tinydiarize    accepted for CLI parity; speaker-turn marking not yet applied"
    );
    eprintln!("  -sa, --save-audio       save all captured audio to a timestamped .wav file");
    eprintln!("  -ng, --no-gpu           accepted for CLI parity (no-op; this crate is CPU-only)");
    eprintln!("  -fa, --flash-attn, -nfa, --no-flash-attn");
    eprintln!("                          accepted for CLI parity (no-op; this crate is CPU-only)");
    eprintln!("  -l, --language CODE     source language, or \"auto\" to detect (default \"en\")");
}

fn parse_args() -> Result<Params, ()> {
    let mut p = Params::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        macro_rules! next_parse {
            ($flag:expr) => {
                match args.next().and_then(|v| v.parse().ok()) {
                    Some(v) => v,
                    None => {
                        eprintln!("{} requires a value", $flag);
                        return Err(());
                    }
                }
            };
        }
        match arg.as_str() {
            "--model" | "-m" => p.model_path = args.next(),
            "--step" => p.step_ms = next_parse!("--step"),
            "--length" => p.length_ms = next_parse!("--length"),
            "--keep" => p.keep_ms = next_parse!("--keep"),
            "--capture" | "-c" => p.capture_id = next_parse!("--capture"),
            "--list-devices" => p.list_devices = true,
            "--max-tokens" | "-mt" => {
                let _: i64 = next_parse!("--max-tokens");
            }
            "--audio-ctx" | "-ac" => {
                p.audio_ctx = match args.next().and_then(|v| v.parse().ok()) {
                    Some(0) | None => None,
                    Some(n) => Some(n),
                };
            }
            "--beam-size" | "-bs" => p.beam_size = next_parse!("--beam-size"),
            "--vad-thold" | "-vth" => p.vad_thold = next_parse!("--vad-thold"),
            "--freq-thold" | "-fth" => p.freq_thold = next_parse!("--freq-thold"),
            "--translate" | "-tr" => p.translate = true,
            "--no-fallback" | "-nf" => p.no_fallback = true,
            "--keep-context" | "-kc" => p.keep_context = true,
            "--tinydiarize" | "-tdrz" => p.tinydiarize = true,
            "--save-audio" | "-sa" => p.save_audio = true,
            "--no-gpu" | "-ng" => {}
            "--flash-attn" | "-fa" | "--no-flash-attn" | "-nfa" => {}
            "--language" | "-l" => {
                p.language = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--language requires a code");
                        return Err(());
                    }
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other} (see --help)");
                return Err(());
            }
        }
    }
    Ok(p)
}

/// One-pole high-pass IIR filter, in place — mirrors whisper.cpp's
/// `high_pass_filter` (`common.cpp`).
fn high_pass_filter(data: &mut [f32], cutoff: f32, sample_rate: u32) {
    if data.is_empty() {
        return;
    }
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let dt = 1.0 / sample_rate as f32;
    let alpha = dt / (rc + dt);
    let mut y = data[0];
    for i in 1..data.len() {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

/// Simple energy VAD — mirrors whisper.cpp's `vad_simple` (`common.cpp`):
/// `true` means the trailing `last_ms` has gone quiet relative to the
/// whole buffer's average energy (i.e. it fires on a pause *after*
/// speech, not on speech onset). Mutates `pcmf32` in place via the
/// high-pass pre-filter, same as the reference.
fn vad_simple(
    pcmf32: &mut [f32],
    sample_rate: u32,
    last_ms: usize,
    vad_thold: f32,
    freq_thold: f32,
) -> bool {
    let n_samples = pcmf32.len();
    let n_samples_last = (sample_rate as usize * last_ms) / 1000;
    if n_samples_last >= n_samples {
        return false;
    }
    if freq_thold > 0.0 {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }
    let mean_abs = |s: &[f32]| -> f32 { s.iter().map(|v| v.abs()).sum::<f32>() / s.len() as f32 };
    let energy_all = mean_abs(pcmf32);
    let energy_last = mean_abs(&pcmf32[n_samples - n_samples_last..]);
    energy_last <= vad_thold * energy_all
}

fn timestamp_filename() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // whisper.cpp names this file from `strftime("%Y%m%d%H%M%S", localtime(...))`;
    // this crate has no chrono-equivalent dependency (the `mic` feature's
    // only crate is `cpal`), so this uses the epoch second count instead —
    // still a unique, monotonically increasing per-run name.
    format!("{secs}.wav")
}

fn main() -> std::process::ExitCode {
    let p = match parse_args() {
        Ok(p) => p,
        Err(()) => return std::process::ExitCode::FAILURE,
    };

    if p.list_devices {
        for (i, name) in mic::list_input_devices().iter().enumerate() {
            println!("{i}: {name}");
        }
        return std::process::ExitCode::SUCCESS;
    }

    let Some(model_path) = &p.model_path else {
        eprintln!("--model is required (see --help)");
        return std::process::ExitCode::FAILURE;
    };
    let model = match std::fs::File::open(model_path)
        .and_then(|f| model::load_model(&mut std::io::BufReader::new(f)))
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load model {model_path}: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let device_name: Option<String> = if p.capture_id >= 0 {
        let devices = mic::list_input_devices();
        match devices.get(p.capture_id as usize) {
            Some(name) => Some(name.clone()),
            None => {
                eprintln!(
                    "no capture device at index {} (found {} device(s); see --list-devices)",
                    p.capture_id,
                    devices.len()
                );
                return std::process::ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    // whisper.cpp: keep_ms = min(keep_ms, step_ms); length_ms = max(length_ms, step_ms).
    let keep_ms = p.keep_ms.min(p.step_ms);
    let length_ms = p.length_ms.max(p.step_ms);
    let use_vad = p.step_ms <= 0;
    let n_samples_step = ((mic::SAMPLE_RATE as i64 * p.step_ms) / 1000).max(0) as usize;
    let n_samples_len = ((mic::SAMPLE_RATE as i64 * length_ms) / 1000).max(0) as usize;
    let n_samples_keep = ((mic::SAMPLE_RATE as i64 * keep_ms) / 1000).max(0) as usize;
    let n_new_line = if !use_vad {
        ((length_ms / p.step_ms.max(1)) - 1).max(1) as u64
    } else {
        1
    };

    let capture = match mic::init(length_ms as usize, device_name.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to open capture device: {e} (see --list-devices)");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = capture.resume() {
        eprintln!("failed to start capture: {e}");
        return std::process::ExitCode::FAILURE;
    }

    let mut wav_writer = if p.save_audio {
        match wav::WavWriter::create(
            std::path::Path::new(&timestamp_filename()),
            mic::SAMPLE_RATE,
        ) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("--save-audio: failed to create output file: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    let mut base_opts = transcribe::Options {
        language: if p.language == "auto" {
            None
        } else {
            Some(p.language.clone())
        },
        translate: p.translate,
        beam_size: if p.beam_size > 1 {
            p.beam_size as usize
        } else {
            1
        },
        no_fallback: p.no_fallback,
        audio_ctx: p.audio_ctx,
        tinydiarize: p.tinydiarize,
        ..transcribe::Options::default()
    };

    eprintln!(
        "whisper-stream: {} (step {} ms, length {} ms, keep {} ms)",
        if use_vad {
            "VAD-triggered mode"
        } else {
            "sliding-window mode"
        },
        p.step_ms,
        length_ms,
        keep_ms
    );

    if use_vad {
        run_vad_mode(
            &model,
            &capture,
            &base_opts,
            length_ms,
            &p,
            wav_writer.as_mut(),
        );
    } else {
        run_sliding_window_mode(
            &model,
            &capture,
            &mut base_opts,
            p.step_ms,
            n_samples_step,
            n_samples_len,
            n_samples_keep,
            n_new_line,
            p.keep_context,
            wav_writer.as_mut(),
        );
    }

    std::process::ExitCode::SUCCESS
}

#[allow(clippy::too_many_arguments)]
fn run_sliding_window_mode(
    model: &model::Model,
    capture: &mic::MicCapture,
    base_opts: &mut transcribe::Options,
    step_ms: i64,
    n_samples_step: usize,
    n_samples_len: usize,
    n_samples_keep: usize,
    n_new_line: u64,
    keep_context: bool,
    mut wav_writer: Option<&mut wav::WavWriter>,
) {
    let mut pcmf32_old: Vec<f32> = Vec::new();
    let mut n_iter: u64 = 0;
    let mut prompt: Option<String> = None;

    loop {
        let pcmf32_new = loop {
            let chunk = capture.get(step_ms);
            if chunk.len() > 2 * n_samples_step {
                eprintln!("WARNING: cannot process audio fast enough, dropping audio");
                capture.clear();
                continue;
            }
            if chunk.len() >= n_samples_step {
                capture.clear();
                break chunk;
            }
            std::thread::sleep(Duration::from_millis(1));
        };

        if let Some(w) = wav_writer.as_deref_mut() {
            let _ = w.write(&pcmf32_new);
        }

        let n_samples_new = pcmf32_new.len();
        let n_samples_take = pcmf32_old
            .len()
            .min((n_samples_keep + n_samples_len).saturating_sub(n_samples_new));
        let mut window: Vec<f32> = pcmf32_old[pcmf32_old.len() - n_samples_take..].to_vec();
        window.extend_from_slice(&pcmf32_new);
        pcmf32_old = window.clone();

        base_opts.initial_prompt = prompt.clone();
        let transcript = transcribe::transcribe(model, &window, base_opts);
        let text = transcript
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");

        print!("\x1b[2K\r{text}");
        let _ = std::io::stdout().flush();

        n_iter += 1;
        if n_iter.is_multiple_of(n_new_line) {
            println!();
            let keep_from = pcmf32_old.len().saturating_sub(n_samples_keep);
            pcmf32_old = pcmf32_old[keep_from..].to_vec();
            if keep_context {
                prompt = Some(
                    transcript
                        .segments
                        .iter()
                        .map(|s| s.text.as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                );
            }
        }
    }
}

fn run_vad_mode(
    model: &model::Model,
    capture: &mic::MicCapture,
    opts: &transcribe::Options,
    length_ms: i64,
    p: &Params,
    mut wav_writer: Option<&mut wav::WavWriter>,
) {
    let mut t_last = Instant::now()
        .checked_sub(Duration::from_millis(2000))
        .unwrap_or_else(Instant::now);
    let mut n_transcription: u64 = 0;

    loop {
        if t_last.elapsed() < Duration::from_millis(2000) {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let mut pcmf32_new = capture.get(2000);
        if let Some(w) = wav_writer.as_deref_mut() {
            let _ = w.write(&pcmf32_new);
        }
        if !vad_simple(
            &mut pcmf32_new,
            mic::SAMPLE_RATE,
            1000,
            p.vad_thold,
            p.freq_thold,
        ) {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        let window = capture.get(length_ms);
        t_last = Instant::now();

        n_transcription += 1;
        println!("### Transcription {n_transcription} START");
        let transcript = transcribe::transcribe(model, &window, opts);
        for seg in &transcript.segments {
            println!("[{:.3} --> {:.3}]  {}", seg.t0, seg.t1, seg.text.trim());
        }
        println!("### Transcription {n_transcription} END");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_pass_filter_removes_dc_offset() {
        let mut data = vec![1.0f32; 1000]; // pure DC
        high_pass_filter(&mut data, 100.0, 16000);
        // A one-pole high-pass on a constant signal should decay toward 0.
        assert!(data[999].abs() < data[1].abs());
        assert!(data[999].abs() < 0.1);
    }

    #[test]
    fn high_pass_filter_empty_is_a_no_op() {
        let mut data: Vec<f32> = Vec::new();
        high_pass_filter(&mut data, 100.0, 16000);
        assert!(data.is_empty());
    }

    #[test]
    fn vad_simple_true_when_tail_is_quiet() {
        let mut samples = vec![0.5f32; 8000];
        samples.extend(vec![0.0f32; 8000]); // second half silent
        assert!(vad_simple(&mut samples, 16000, 500, 0.6, 0.0));
    }

    #[test]
    fn vad_simple_false_when_still_loud() {
        let mut samples = vec![0.5f32; 16000];
        assert!(!vad_simple(&mut samples, 16000, 500, 0.6, 0.0));
    }

    #[test]
    fn vad_simple_false_when_not_enough_samples() {
        let mut samples = vec![0.5f32; 100];
        assert!(!vad_simple(&mut samples, 16000, 1000, 0.6, 0.0));
    }

    #[test]
    fn timestamp_filename_ends_in_wav() {
        assert!(timestamp_filename().ends_with(".wav"));
    }
}
