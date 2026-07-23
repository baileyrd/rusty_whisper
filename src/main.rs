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
            output::write_srt(segments, &mut w)?;
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

fn main() -> ExitCode {
    let mut model_path = None;
    let mut audio_path = None;
    let mut beam_size = 5usize;
    let mut language: Option<String> = None;
    let mut translate = false;
    let mut dense = false;
    let mut convert_gguf: Option<String> = None;
    let mut outputs = OutputFormats::default();
    let mut max_len = 0usize;
    let mut split_on_word = false;
    let mut word_thold = 0.01f32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => model_path = args.next(),
            "--audio" | "-f" => audio_path = args.next(),
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
            "--beam" | "-b" => {
                beam_size = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) if n >= 1 => n,
                    _ => {
                        eprintln!("--beam requires a positive integer");
                        return ExitCode::FAILURE;
                    }
                }
            }
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

    // Streaming mode: read WAV from stdin, emit segments as windows fill.
    if audio_path.as_deref() == Some("-") {
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
        let secs = wav.samples.len() as f32 / audio::SAMPLE_RATE as f32;
        println!("audio: {secs:.2} s");

        if let Some(m) = &loaded {
            let t0 = std::time::Instant::now();
            let opts = transcribe::Options {
                beam_size,
                language,
                translate,
                ..Default::default()
            };
            let result = transcribe::transcribe(m, &wav.samples, &opts);
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
                    s.text
                );
            }
            if outputs.any() {
                if let Err(e) = outputs.write_all(p, &result) {
                    eprintln!("failed to write output file: {e}");
                    return ExitCode::FAILURE;
                }
            }
        } else {
            println!("(pass --model to transcribe)");
        }
    }

    ExitCode::SUCCESS
}
