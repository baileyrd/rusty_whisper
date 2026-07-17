//! CLI for rusty-whisper.
//!
//! Current capabilities (pipeline through the mel front-end + model
//! inspection; full transcription lands with PLAN.md phases 4-5):
//!
//! ```text
//! rusty-whisper --model ggml-tiny.en.bin --audio speech.wav
//! rusty-whisper --audio speech.wav          # mel stats with generated filters
//! rusty-whisper --model ggml-tiny.en.bin    # model info dump
//! ```

use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;

use rusty_whisper::{audio, model, tokenizer::Tokenizer, wav};

fn main() -> ExitCode {
    let mut model_path = None;
    let mut audio_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => model_path = args.next(),
            "--audio" | "-f" => audio_path = args.next(),
            "--help" | "-h" => {
                eprintln!("usage: rusty-whisper [--model GGML_BIN] [--audio WAV_16KHZ_MONO]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other} (see --help)");
                return ExitCode::FAILURE;
            }
        }
    }
    if model_path.is_none() && audio_path.is_none() {
        eprintln!("nothing to do: pass --model and/or --audio (see --help)");
        return ExitCode::FAILURE;
    }

    let loaded = match &model_path {
        Some(p) => match File::open(p).and_then(|f| model::load_model(&mut BufReader::new(f))) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("failed to load model {p}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    if let Some(m) = &loaded {
        let hp = &m.hparams;
        println!("model: {} ({})", hp.model_type(), if hp.is_multilingual() { "multilingual" } else { "English-only" });
        println!("  n_vocab={} n_mels={} ftype={}", hp.n_vocab, hp.n_mels, hp.ftype);
        println!("  encoder: ctx={} state={} heads={} layers={}", hp.n_audio_ctx, hp.n_audio_state, hp.n_audio_head, hp.n_audio_layer);
        println!("  decoder: ctx={} state={} heads={} layers={}", hp.n_text_ctx, hp.n_text_state, hp.n_text_head, hp.n_text_layer);
        println!("  tensors: {}", m.tensors.len());
        let tok = Tokenizer::new(m.vocab.clone(), hp);
        println!("  special tokens: eot={} sot={} timestamps from {}", tok.eot, tok.sot, tok.timestamp_begin);
    }

    if let Some(p) = &audio_path {
        let wav = match File::open(p).and_then(|f| wav::read_wav(&mut BufReader::new(f))) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to read {p}: {e}");
                return ExitCode::FAILURE;
            }
        };
        if wav.sample_rate as usize != audio::SAMPLE_RATE {
            eprintln!(
                "{p} is {} Hz; resample to 16 kHz first (ffmpeg -i in -ar 16000 -ac 1 out.wav)",
                wav.sample_rate
            );
            return ExitCode::FAILURE;
        }
        let n_mels = loaded.as_ref().map(|m| m.hparams.n_mels as usize).unwrap_or(audio::N_MEL);
        // Prefer the filterbank embedded in the model file; fall back to ours.
        let filters = match &loaded {
            Some(m) if !m.mel_filters.is_empty() => m.mel_filters.clone(),
            _ => audio::mel_filterbank(n_mels, audio::N_FFT, audio::SAMPLE_RATE),
        };
        let (mel, n_frames) = audio::log_mel_spectrogram(&wav.samples, &filters, n_mels);
        let secs = wav.samples.len() as f32 / audio::SAMPLE_RATE as f32;
        let mean = mel.iter().sum::<f32>() / mel.len() as f32;
        println!("audio: {secs:.2} s -> log-mel {n_mels} x {n_frames} (mean {mean:.4})");
        println!("(transcription pending encoder/decoder — PLAN.md phases 4-5)");
    }

    ExitCode::SUCCESS
}
