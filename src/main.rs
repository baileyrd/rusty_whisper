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
use std::io::{BufReader, BufWriter};
use std::process::ExitCode;

use rusty_whisper::{audio, model, output, tokenizer::Tokenizer, transcribe, wav};

/// Which `-o*` output files to write, and under what base path.
struct OutputFormats {
    txt: bool,
    vtt: bool,
    srt: bool,
    csv: bool,
    json: bool,
    json_full: bool,
    words: bool,
    file: Option<String>,
    /// `--font-path`/`-fp`, burned into the `-owts` script's `drawtext`
    /// filter. Defaults to whisper.cpp's own default, which is macOS-only —
    /// override it on other platforms.
    font_path: String,
}

impl Default for OutputFormats {
    fn default() -> Self {
        OutputFormats {
            txt: false,
            vtt: false,
            srt: false,
            csv: false,
            json: false,
            json_full: false,
            words: false,
            file: None,
            font_path: "/System/Library/Fonts/Supplemental/Courier New Bold.ttf".to_string(),
        }
    }
}

impl OutputFormats {
    fn any(&self) -> bool {
        self.txt || self.vtt || self.srt || self.csv || self.json || self.json_full || self.words
    }

    /// Writes every requested format to `<base>.<ext>`, where `<base>` is
    /// `--output-file` if given, else the audio path with its extension
    /// stripped.
    fn write_all(
        &self,
        audio_path: &str,
        transcript: &transcribe::Transcript,
        offset_n: usize,
    ) -> std::io::Result<()> {
        let base = self
            .file
            .clone()
            .unwrap_or_else(|| match audio_path.rsplit_once('.') {
                Some((stem, _ext)) => stem.to_string(),
                None => audio_path.to_string(),
            });
        let segments = &transcript.segments;
        if self.txt {
            let mut w = BufWriter::new(File::create(format!("{base}.txt"))?);
            output::write_txt(segments, &mut w)?;
        }
        if self.vtt {
            let mut w = BufWriter::new(File::create(format!("{base}.vtt"))?);
            output::write_vtt(segments, &mut w)?;
        }
        if self.srt {
            let mut w = BufWriter::new(File::create(format!("{base}.srt"))?);
            output::write_srt(segments, offset_n, &mut w)?;
        }
        if self.csv {
            let mut w = BufWriter::new(File::create(format!("{base}.csv"))?);
            output::write_csv(segments, &mut w)?;
        }
        if self.json_full {
            // -ojf includes and supersedes plain -oj: same file, fuller content.
            let mut w = BufWriter::new(File::create(format!("{base}.json"))?);
            output::write_json_full(&transcript.language, segments, &mut w)?;
        } else if self.json {
            let mut w = BufWriter::new(File::create(format!("{base}.json"))?);
            output::write_json(&transcript.language, segments, &mut w)?;
        }
        if self.words {
            let mut w = BufWriter::new(File::create(format!("{base}.wts"))?);
            output::write_wts(segments, audio_path, &self.font_path, &mut w)?;
        }
        Ok(())
    }
}

/// Same as `transcribe::transcribe`, but drives `Stream` window-by-window so
/// `--print-progress`/`-pp` can report percent-complete to stderr.
fn transcribe_with_progress(
    m: &model::Model,
    samples: &[f32],
    opts: &transcribe::Options,
) -> transcribe::Transcript {
    let mut stream = transcribe::Stream::new(m, opts.clone());
    let mut segments = Vec::new();
    let total = samples.len().max(1);
    let mut fed = 0usize;
    for chunk in samples.chunks(audio::N_SAMPLES_30S) {
        segments.extend(stream.feed(chunk));
        fed += chunk.len();
        eprint!("\rprogress: {:3}%", (fed * 100 / total).min(100));
    }
    segments.extend(stream.finish());
    eprintln!("\rprogress: 100%");
    let language = stream.language().unwrap_or("en").to_string();
    transcribe::Transcript { segments, language }
}

/// Renders a segment's text token-by-token when `colors` and/or
/// `confidence` are requested (using each token's own decoded piece and
/// probability), otherwise the segment's plain trimmed text.
///
/// Colors are a 3-tier ANSI approximation of whisper.cpp's confidence
/// gradient (green >= 0.8, yellow >= 0.5, red below), not a verified match
/// to its exact color ramp.
fn format_segment_text(seg: &transcribe::Segment, colors: bool, confidence: bool) -> String {
    if (!colors && !confidence) || seg.tokens.is_empty() {
        return seg.text.clone();
    }
    let mut out = String::new();
    for tk in &seg.tokens {
        let text = tk.text.clone();
        let text = if confidence {
            format!("{}({:.0}%)", text.trim(), tk.prob * 100.0)
        } else {
            text
        };
        if colors {
            let color = if tk.prob >= 0.8 {
                "\x1b[32m" // green
            } else if tk.prob >= 0.5 {
                "\x1b[33m" // yellow
            } else {
                "\x1b[31m" // red
            };
            out.push_str(color);
            out.push_str(&text);
            out.push_str("\x1b[0m");
        } else {
            out.push_str(&text);
        }
    }
    out.trim().to_string()
}

fn main() -> ExitCode {
    let mut model_path = None;
    let mut audio_paths: Vec<String> = Vec::new();
    let mut beam_size = 5usize;
    let mut language: Option<String> = None;
    let mut translate = false;
    let mut dense = false;
    let mut convert_gguf: Option<String> = None;
    let mut outputs = OutputFormats::default();
    let mut max_len = 0usize;
    let mut split_on_word = false;
    let mut word_thold = 0.01f32;
    let mut temperature = 0.0f32;
    let mut temperature_inc = 0.2f32;
    let mut best_of = 1usize;
    let mut entropy_threshold = 2.4f32;
    let mut no_speech_threshold = 0.6f32;
    let mut no_fallback = false;
    let mut max_context: Option<usize> = None;
    let mut audio_ctx: Option<usize> = None;
    let mut offset_t_ms = 0u64;
    let mut offset_n = 0usize;
    let mut duration_ms = 0u64;
    let mut debug_mode = false;
    let mut no_prints = false;
    let mut print_special = false;
    let mut print_colors = false;
    let mut print_confidence = false;
    let mut print_progress = false;
    let mut log_score = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => model_path = args.next(),
            "--audio" | "-f" => match args.next() {
                Some(p) => audio_paths.push(p),
                None => {
                    eprintln!("--audio requires a path");
                    return ExitCode::FAILURE;
                }
            },
            "--language" | "-l" => {
                language = args.next().filter(|l| l != "auto");
                if let Some(l) = &language {
                    if rusty_whisper::tokenizer::lang_id_from_code(l).is_none() {
                        eprintln!("unknown language code: {l}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--translate" => translate = true,
            "--dense" => dense = true,
            "--convert-gguf" => convert_gguf = args.next(),
            "--output-txt" | "-otxt" => outputs.txt = true,
            "--output-vtt" | "-ovtt" => outputs.vtt = true,
            "--output-srt" | "-osrt" => outputs.srt = true,
            "--output-csv" | "-ocsv" => outputs.csv = true,
            "--output-json" | "-oj" => outputs.json = true,
            "--output-json-full" | "-ojf" => outputs.json_full = true,
            "--output-words" | "-owts" => outputs.words = true,
            "--font-path" | "-fp" => {
                outputs.font_path = match args.next() {
                    Some(p) => p,
                    None => {
                        eprintln!("--font-path requires a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--output-file" | "-of" => outputs.file = args.next(),
            "--max-len" | "-ml" => {
                max_len = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--max-len requires an integer");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--split-on-word" | "-sow" => split_on_word = true,
            "--word-thold" | "-wt" => {
                word_thold = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--word-thold requires a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--temperature" | "-tp" => {
                temperature = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--temperature requires a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--temperature-inc" | "-tpi" => {
                temperature_inc = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--temperature-inc requires a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--best-of" | "-bo" => {
                best_of = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) if n >= 1 => n,
                    _ => {
                        eprintln!("--best-of requires a positive integer");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--entropy-thold" | "-et" => {
                entropy_threshold = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--entropy-thold requires a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--no-speech-thold" | "-nth" => {
                no_speech_threshold = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--no-speech-thold requires a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--no-fallback" | "-nf" => no_fallback = true,
            "--max-context" | "-mc" => {
                max_context = match args.next().and_then(|v| v.parse::<i64>().ok()) {
                    Some(n) if n < 0 => None,
                    Some(n) => Some(n as usize),
                    None => {
                        eprintln!("--max-context requires an integer");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--audio-ctx" | "-ac" => {
                audio_ctx = match args.next().and_then(|v| v.parse().ok()) {
                    Some(0) | None => None,
                    Some(n) => Some(n),
                }
            }
            "--offset-t" | "-ot" => {
                offset_t_ms = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--offset-t requires an integer (milliseconds)");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--offset-n" | "-on" => {
                offset_n = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--offset-n requires an integer");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--duration" | "-d" => {
                duration_ms = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--duration requires an integer (milliseconds)");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--beam" | "-b" => {
                beam_size = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) if n >= 1 => n,
                    _ => {
                        eprintln!("--beam requires a positive integer");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--version" => {
                println!("rusty-whisper {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--debug-mode" | "-debug" => debug_mode = true,
            "--no-prints" | "-np" => no_prints = true,
            "--print-special" | "-ps" => print_special = true,
            "--print-colors" | "-pc" => print_colors = true,
            "--print-confidence" => print_confidence = true,
            "--print-progress" | "-pp" => print_progress = true,
            "--log-score" | "-ls" => log_score = true,
            "--help" | "-h" => {
                eprintln!("usage: rusty-whisper [--model GGML_BIN] [--audio WAV_16KHZ_MONO|-] [--beam N] [--language CODE|auto] [--translate] [--dense]");
                eprintln!("  --audio -   stream WAV from stdin, printing segments as windows fill");
                eprintln!(
                    "  --dense     dequantize weights at load: faster decoding, 2-3x the memory"
                );
                eprintln!(
                    "  --convert-gguf OUT  write the loaded model as GGUF (needs --features gguf)"
                );
                eprintln!(
                    "  --output-txt/-vtt/-srt/-csv/-json  write a transcript file alongside stdout"
                );
                eprintln!(
                    "  --output-json-full, -ojf  like -json but with per-token id/prob/logprob/timestamps"
                );
                eprintln!(
                    "  --output-words, -owts  write a .wts karaoke script (ffmpeg drawtext captions)"
                );
                eprintln!(
                    "  --font-path, -fp PATH  font burned into -owts captions (default: macOS system font)"
                );
                eprintln!(
                    "  --output-file, -of PATH  base path for -o* files (default: audio path minus its extension)"
                );
                eprintln!(
                    "  --max-len, -ml N     cap segment length in characters by splitting long segments (0 = off)"
                );
                eprintln!(
                    "  --split-on-word, -sow  when splitting on --max-len, break at word boundaries"
                );
                eprintln!(
                    "  --word-thold, -wt N  word-timestamp probability threshold (accepted, currently unused)"
                );
                eprintln!(
                    "  --temperature, -tp N / --temperature-inc, -tpi N  fallback temperature ladder start/step (default 0.0/0.2)"
                );
                eprintln!(
                    "  --best-of, -bo N     independent samples per temperature > 0, keep the best (default 1)"
                );
                eprintln!(
                    "  --entropy-thold, -et N  reject low-entropy (collapsed) decodes below this, retry at higher temperature"
                );
                eprintln!(
                    "  --no-speech-thold, -nth N  treat a window as silence above this no-speech probability"
                );
                eprintln!("  --no-fallback, -nf   disable the temperature fallback ladder");
                eprintln!(
                    "  --max-context, -mc N  cap prior-window context tokens carried forward (-1 = model default)"
                );
                eprintln!(
                    "  --audio-ctx, -ac N   limit encoder audio context (accepted, currently unused; 0 = full)"
                );
                eprintln!("  --offset-t, -ot MS   skip this many ms of audio before transcribing");
                eprintln!(
                    "  --offset-n, -on N    starting segment index for numbered output (e.g. .srt)"
                );
                eprintln!(
                    "  --duration, -d MS    only transcribe this many ms of audio (0 = to the end)"
                );
                eprintln!("  --version            print the version and exit");
                eprintln!("  --debug-mode, -debug  print extra diagnostics to stderr");
                eprintln!("  --no-prints, -np     suppress diagnostic output, print only results");
                eprintln!("  --print-special, -ps  include special/control tokens in segment text");
                eprintln!(
                    "  --print-colors, -pc  ANSI-color console segment text by token confidence"
                );
                eprintln!("  --print-confidence   append a (NN%) confidence suffix per token");
                eprintln!(
                    "  --print-progress, -pp  print percent-complete to stderr while transcribing"
                );
                eprintln!(
                    "  --log-score, -ls     print each window's chosen decode score to stderr"
                );
                return ExitCode::SUCCESS;
            }
            // Bare positional arguments (including "-" for stdin) are audio
            // paths too, matching whisper.cpp's repeatable -f / positional
            // file list.
            other if !other.starts_with('-') || other == "-" => {
                audio_paths.push(other.to_string());
            }
            other => {
                eprintln!("unknown argument: {other} (see --help)");
                return ExitCode::FAILURE;
            }
        }
    }
    if model_path.is_none() && audio_paths.is_empty() {
        eprintln!("nothing to do: pass --model and/or --audio (see --help)");
        return ExitCode::FAILURE;
    }

    let loaded = match &model_path {
        Some(p) => match File::open(p).and_then(|f| model::load_model(&mut BufReader::new(f))) {
            Ok(mut m) => {
                if dense {
                    m.densify();
                }
                Some(m)
            }
            Err(e) => {
                eprintln!("failed to load model {p}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    if let Some(out_path) = &convert_gguf {
        #[cfg(feature = "gguf")]
        {
            let Some(m) = &loaded else {
                eprintln!("--convert-gguf needs --model");
                return ExitCode::FAILURE;
            };
            let write = File::create(out_path).and_then(|f| {
                let mut w = std::io::BufWriter::new(f);
                rusty_whisper::gguf::write(m, &mut w)
            });
            match write {
                Ok(()) => println!("wrote {out_path}"),
                Err(e) => {
                    eprintln!("failed to write {out_path}: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        #[cfg(not(feature = "gguf"))]
        {
            let _ = out_path;
            eprintln!("--convert-gguf requires building with `--features gguf`");
            return ExitCode::FAILURE;
        }
    }

    if let Some(m) = &loaded {
        if no_prints {
            // --no-prints: skip the model-info dump, print only results.
        } else {
            let hp = &m.hparams;
            println!(
                "model: {} ({})",
                hp.model_type(),
                if hp.is_multilingual() {
                    "multilingual"
                } else {
                    "English-only"
                }
            );
            println!(
                "  n_vocab={} n_mels={} ftype={}",
                hp.n_vocab, hp.n_mels, hp.ftype
            );
            println!(
                "  encoder: ctx={} state={} heads={} layers={}",
                hp.n_audio_ctx, hp.n_audio_state, hp.n_audio_head, hp.n_audio_layer
            );
            println!(
                "  decoder: ctx={} state={} heads={} layers={}",
                hp.n_text_ctx, hp.n_text_state, hp.n_text_head, hp.n_text_layer
            );
            println!("  tensors: {}", m.tensors.len());
            let tok = Tokenizer::new(m.vocab.clone(), hp);
            println!(
                "  special tokens: eot={} sot={} timestamps from {}",
                tok.eot, tok.sot, tok.timestamp_begin
            );
        }
    }

    // Streaming mode: read WAV from stdin, emit segments as windows fill.
    if audio_paths.len() == 1 && audio_paths[0] == "-" {
        let Some(m) = &loaded else {
            eprintln!("streaming needs --model");
            return ExitCode::FAILURE;
        };
        let mut ws = match wav::WavStream::new(std::io::stdin().lock()) {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("failed to read wav from stdin: {e}");
                return ExitCode::FAILURE;
            }
        };
        if ws.sample_rate as usize != audio::SAMPLE_RATE {
            eprintln!(
                "stdin audio is {} Hz; resample to 16 kHz first",
                ws.sample_rate
            );
            return ExitCode::FAILURE;
        }
        let opts = transcribe::Options {
            beam_size,
            language,
            translate,
            max_len,
            split_on_word,
            word_thold,
            temperatures: transcribe::temperature_ladder(temperature, temperature_inc),
            best_of,
            entropy_threshold,
            no_speech_threshold,
            no_fallback,
            max_context,
            audio_ctx,
            print_special,
            ..Default::default()
        };
        let mut stream = transcribe::Stream::new(m, opts);
        let print_seg = |s: &transcribe::Segment| {
            println!(
                "[{} --> {}]  {}",
                transcribe::format_timestamp(s.t0),
                transcribe::format_timestamp(s.t1),
                s.text
            );
        };
        loop {
            let chunk = match ws.read_frames(audio::SAMPLE_RATE) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("read error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if chunk.is_empty() {
                break;
            }
            stream.feed(&chunk).iter().for_each(&print_seg);
        }
        stream.finish().iter().for_each(&print_seg);
        eprintln!("language: {}", stream.language().unwrap_or("?"));
        return ExitCode::SUCCESS;
    }

    let mut had_error = false;
    for p in &audio_paths {
        if audio_paths.len() > 1 && !no_prints {
            println!("=== {p} ===");
        }
        let wav = match File::open(p).and_then(|f| wav::read_wav(&mut BufReader::new(f))) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to read {p}: {e}");
                had_error = true;
                continue;
            }
        };
        if wav.sample_rate as usize != audio::SAMPLE_RATE {
            eprintln!(
                "{p} is {} Hz; resample to 16 kHz first (ffmpeg -i in -ar 16000 -ac 1 out.wav)",
                wav.sample_rate
            );
            had_error = true;
            continue;
        }
        let secs = wav.samples.len() as f32 / audio::SAMPLE_RATE as f32;
        println!("audio: {secs:.2} s");

        if let Some(m) = &loaded {
            let t0 = std::time::Instant::now();
            let opts = transcribe::Options {
                beam_size,
                language: language.clone(),
                translate,
                max_len,
                split_on_word,
                word_thold,
                temperatures: transcribe::temperature_ladder(temperature, temperature_inc),
                best_of,
                entropy_threshold,
                no_speech_threshold,
                no_fallback,
                max_context,
                audio_ctx,
                print_special,
                ..Default::default()
            };
            let offset_secs = offset_t_ms as f32 / 1000.0;
            let offset_samples =
                ((offset_t_ms as usize * audio::SAMPLE_RATE) / 1000).min(wav.samples.len());
            let window = if duration_ms > 0 {
                let end = offset_samples + (duration_ms as usize * audio::SAMPLE_RATE) / 1000;
                &wav.samples[offset_samples..end.min(wav.samples.len())]
            } else {
                &wav.samples[offset_samples..]
            };
            if debug_mode {
                eprintln!(
                    "debug: transcribing {} samples ({:.2}s), offset={offset_secs:.2}s",
                    window.len(),
                    window.len() as f32 / audio::SAMPLE_RATE as f32
                );
            }
            let mut result = if print_progress {
                transcribe_with_progress(m, window, &opts)
            } else {
                transcribe::transcribe(m, window, &opts)
            };
            for s in &mut result.segments {
                s.t0 += offset_secs;
                s.t1 += offset_secs;
                for tk in &mut s.tokens {
                    tk.t0 += offset_secs;
                    tk.t1 += offset_secs;
                }
            }
            let elapsed = t0.elapsed().as_secs_f32();
            println!(
                "transcribed in {elapsed:.2} s ({:.2}x realtime), language: {}",
                secs / elapsed,
                result.language
            );
            println!("---");
            for s in &result.segments {
                println!(
                    "[{} --> {}]  {}",
                    transcribe::format_timestamp(s.t0),
                    transcribe::format_timestamp(s.t1),
                    format_segment_text(s, print_colors, print_confidence)
                );
                if log_score && !s.tokens.is_empty() {
                    let avg_lp: f32 =
                        s.tokens.iter().map(|t| t.logprob).sum::<f32>() / s.tokens.len() as f32;
                    eprintln!("  score: avg_logprob={avg_lp:.3}");
                }
            }
            if outputs.any() {
                if let Err(e) = outputs.write_all(p, &result, offset_n) {
                    eprintln!("failed to write output file: {e}");
                    had_error = true;
                }
            }
        } else {
            println!("(pass --model to transcribe)");
        }
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg_with_tokens(tokens: Vec<(&str, f32)>) -> transcribe::Segment {
        transcribe::Segment {
            t0: 0.0,
            t1: 1.0,
            text: tokens
                .iter()
                .map(|(t, _)| *t)
                .collect::<Vec<_>>()
                .join("")
                .trim()
                .to_string(),
            tokens: tokens
                .into_iter()
                .map(|(text, prob)| transcribe::TokenInfo {
                    id: 0,
                    text: text.to_string(),
                    prob,
                    logprob: prob.ln(),
                    t0: 0.0,
                    t1: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn format_segment_text_plain_when_no_flags() {
        let seg = seg_with_tokens(vec![("Hello", 0.9), (" world", 0.9)]);
        assert_eq!(format_segment_text(&seg, false, false), "Hello world");
    }

    #[test]
    fn format_segment_text_colors_wrap_each_token_in_ansi() {
        let seg = seg_with_tokens(vec![("Hi", 0.9), (" there", 0.3)]);
        let s = format_segment_text(&seg, true, false);
        assert!(s.contains("\x1b[32m")); // high confidence -> green
        assert!(s.contains("\x1b[31m")); // low confidence -> red
        assert!(s.contains("\x1b[0m"));
    }

    #[test]
    fn format_segment_text_confidence_appends_percentage() {
        let seg = seg_with_tokens(vec![("word", 0.5)]);
        let s = format_segment_text(&seg, false, true);
        assert_eq!(s, "word(50%)");
    }

    #[test]
    fn format_segment_text_falls_back_when_no_token_data() {
        let seg = transcribe::Segment {
            t0: 0.0,
            t1: 1.0,
            text: "plain text".to_string(),
            tokens: Vec::new(),
        };
        assert_eq!(format_segment_text(&seg, true, true), "plain text");
    }
}
