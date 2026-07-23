//! `whisper-command`: voice-command / assistant mode — mirrors whisper.cpp's
//! `examples/command/command.cpp` (v1.9.1). Three modes, chosen the same way
//! upstream does (first match wins):
//!
//! 1. **Guided** (`-cmd`/`--commands FILE` given): a single forced greedy
//!    decode step scores each candidate phrase in the file by the softmax
//!    probability mass its single-token prefixes get, and always reports
//!    the highest-scoring one (no threshold).
//! 2. **Always-prompt** (`-p`/`--prompt` given, no grammar): every utterance
//!    is transcribed unconstrained, split by word count into an assumed
//!    prompt part and command part, and accepted if the prompt part is
//!    similar enough to the activation phrase.
//! 3. **General-purpose** (the default): wake-phrase detection (fuzzy
//!    match against the activation phrase, optionally grammar-constrained
//!    via the `"prompt"` rule), then grammar-constrained (`"root"` rule)
//!    decoding of the following utterance.
//!
//! Requires the `mic` feature (native capture via `cpal`).

use std::io::Write as _;
use std::time::Duration;

use rusty_whisper::{audio, decoder, encoder, grammar, mic, model, tensor, tokenizer, transcribe};

const DEFAULT_PROMPT: &str = "Ok Whisper, start listening for commands.";
const PROMPT_MS: i64 = 5000;
const COMMAND_MS: i64 = 8000;

struct Params {
    model_path: Option<String>,
    commands_path: Option<String>,
    context: String,
    prompt: String,
    grammar_spec: Option<String>,
    grammar_penalty: f32,
    capture_id: i64,
    vad_thold: f32,
    freq_thold: f32,
    language: String,
    output_path: Option<String>,
    list_devices: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            model_path: None,
            commands_path: None,
            context: String::new(),
            prompt: String::new(),
            grammar_spec: None,
            grammar_penalty: 100.0,
            capture_id: -1,
            vad_thold: 0.6,
            freq_thold: 100.0,
            language: "en".to_string(),
            output_path: None,
            list_devices: false,
        }
    }
}

fn print_help() {
    eprintln!("whisper-command: voice-command / assistant mode (requires --features mic)");
    eprintln!("  --model, -m PATH        model to load (required)");
    eprintln!("  -cmd, --commands FILE   guided mode: one candidate phrase per line");
    eprintln!(
        "  -ctx, --context STRING  text primed as the model's initial prompt in every decode"
    );
    eprintln!("  -p, --prompt STRING     activation phrase (always-prompt/general-purpose modes;");
    eprintln!("                          default \"{DEFAULT_PROMPT}\" in general-purpose mode)");
    eprintln!("  --grammar FILE_or_TEXT  GBNF-lite grammar (general-purpose mode's \"prompt\"/\"root\" rules)");
    eprintln!(
        "  --grammar-penalty F     logit penalty for grammar-inconsistent tokens (default 100.0)"
    );
    eprintln!("  -c, --capture N         capture device index (default: OS default device)");
    eprintln!("  --list-devices          list capture device names (with their index) and exit");
    eprintln!("  -vth, --vad-thold F     energy VAD threshold (default 0.6)");
    eprintln!("  -fth, --freq-thold F    high-pass filter cutoff Hz before VAD (default 100.0)");
    eprintln!("  -l, --language CODE     source language (default \"en\")");
    eprintln!("  -f, --file PATH         append recognized commands/text to this file");
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
        macro_rules! next_string {
            ($flag:expr) => {
                match args.next() {
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
            "--commands" | "-cmd" => p.commands_path = args.next(),
            "--context" | "-ctx" => p.context = next_string!("--context"),
            "--prompt" | "-p" => p.prompt = next_string!("--prompt"),
            "--grammar" => p.grammar_spec = args.next(),
            "--grammar-penalty" => p.grammar_penalty = next_parse!("--grammar-penalty"),
            "--capture" | "-c" => p.capture_id = next_parse!("--capture"),
            "--list-devices" => p.list_devices = true,
            "--vad-thold" | "-vth" => p.vad_thold = next_parse!("--vad-thold"),
            "--freq-thold" | "-fth" => p.freq_thold = next_parse!("--freq-thold"),
            "--language" | "-l" => p.language = next_string!("--language"),
            "--file" | "-f" => p.output_path = args.next(),
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

/// Normalized Levenshtein similarity in `[0, 1]` (1 = identical) — mirrors
/// whisper.cpp's `similarity()` (`common.cpp`).
fn similarity(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 && m == 0 {
        return 1.0;
    }
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    1.0 - prev[m] as f32 / n.max(m) as f32
}

fn read_commands_file(path: &str) -> std::io::Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    Ok(content
        .lines()
        .map(|l| l.trim().to_lowercase())
        .filter(|l| !l.is_empty())
        .collect())
}

/// `--grammar`'s value is a file path if one exists at that path, otherwise
/// treated as inline GBNF-lite source directly — mirrors whisper.cpp's
/// `is_file_exist` check on `--grammar`.
fn load_grammar_source(spec: &str) -> String {
    std::fs::read_to_string(spec).unwrap_or_else(|_| spec.to_string())
}

fn parse_grammar_rule(source: &str, rule: &str) -> Option<grammar::Grammar> {
    match grammar::Grammar::parse(source, rule) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("--grammar: rule {rule:?} unavailable ({e}); continuing unconstrained");
            None
        }
    }
}

/// Guided/command-list mode's scoring: one forced greedy decode step after
/// a "select one from the available words: ..." prompt, then each
/// candidate is scored by the mean softmax probability of whichever of its
/// prefixes happen to encode as a single vocabulary token — mirrors
/// whisper.cpp's command-list scoring in `command.cpp` exactly (including
/// always returning a best match, with no reject threshold).
///
/// Note: whisper.cpp's own command-list mode encodes the same
/// (VAD-high-pass-filtered) 2000ms buffer it just ran `vad_simple` on,
/// rather than re-pulling a fresh copy — this mirrors that.
fn guided_mode_score(
    model: &model::Model,
    tok: &tokenizer::Tokenizer,
    filters: &[f32],
    window: &[f32],
    commands: &[String],
) -> Option<(usize, f32)> {
    let n_mels = model.hparams.n_mels as usize;
    let padded = audio::pad_or_trim(window, audio::N_SAMPLES_30S);
    let (mel, n_frames) = audio::log_mel_spectrogram(&padded, filters, n_mels);
    let mel = tensor::Tensor::from_vec(&[n_mels, n_frames], mel);
    let enc_out = encoder::encode(model, &mel);

    let mut prompt_tokens = decoder::sot_sequence(tok, model.hparams.is_multilingual(), 0);
    let k_prompt = format!(
        "select one from the available words: {}. selected word: ",
        commands.join(", ")
    );
    prompt_tokens.extend(tok.encode(&k_prompt));

    let mut dec = decoder::Decoder::new(model, &enc_out);
    let logits = dec.forward(&prompt_tokens);
    let n_vocab = logits.shape[1];
    let last_row_start = (logits.shape[0] - 1) * n_vocab;
    let mut last_row =
        tensor::Tensor::from_vec(&[1, n_vocab], logits.data[last_row_start..].to_vec());
    tensor::softmax(&mut last_row);

    let mut scores = vec![0.0f32; commands.len()];
    for (i, cmd) in commands.iter().enumerate() {
        let chars: Vec<char> = cmd.chars().collect();
        let mut single_token_probs = Vec::new();
        for l in 0..chars.len() {
            let prefix: String = chars[..=l].iter().collect();
            let piece = format!(" {prefix}");
            let ids = tok.encode(&piece);
            if ids.len() == 1 {
                single_token_probs.push(last_row.data[ids[0] as usize]);
            }
        }
        if !single_token_probs.is_empty() {
            scores[i] = single_token_probs.iter().sum::<f32>() / single_token_probs.len() as f32;
        }
    }
    let total: f32 = scores.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let (best_i, best_score) = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s / total))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    Some((best_i, best_score))
}

/// Slides a window of lengths `0.8..=1.2 * prompt.chars().count()` over
/// `text`'s prefix, returning the char-length that best matches `prompt`
/// by [`similarity`] — mirrors general-purpose mode's prompt/command split
/// in `command.cpp`. Everything after the returned length is the command.
fn best_prompt_cut(text: &str, prompt: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let plen = prompt.chars().count() as f32;
    let lo = (plen * 0.8).floor().max(0.0) as usize;
    let hi = ((plen * 1.2).ceil() as usize).min(chars.len());
    if lo > chars.len() {
        return chars.len();
    }
    let prompt_lower = prompt.to_lowercase();
    let mut best_len = lo.min(chars.len());
    let mut best_sim = -1.0f32;
    for len in lo..=hi.max(lo) {
        if len > chars.len() {
            break;
        }
        let candidate: String = chars[..len].iter().collect();
        let sim = similarity(&candidate.to_lowercase(), &prompt_lower);
        if sim > best_sim {
            best_sim = sim;
            best_len = len;
        }
    }
    best_len
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

    let commands = match &p.commands_path {
        Some(path) => match read_commands_file(path) {
            Ok(c) if !c.is_empty() => Some(c),
            Ok(_) => {
                eprintln!("--commands: {path} has no usable phrases");
                return std::process::ExitCode::FAILURE;
            }
            Err(e) => {
                eprintln!("--commands: failed to read {path}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let grammar_source = p.grammar_spec.as_deref().map(load_grammar_source);

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
    let capture = match mic::init(COMMAND_MS as usize, device_name.as_deref()) {
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

    let mut out_file = match &p.output_path {
        Some(path) => match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("--file: failed to open {path}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let language = if p.language == "auto" {
        None
    } else {
        Some(p.language.clone())
    };
    let base_opts = transcribe::Options {
        language: language.clone(),
        initial_prompt: if p.context.is_empty() {
            None
        } else {
            Some(p.context.clone())
        },
        grammar_penalty: p.grammar_penalty,
        ..transcribe::Options::default()
    };

    if let Some(commands) = commands {
        eprintln!(
            "whisper-command: guided mode ({} candidate phrase(s))",
            commands.len()
        );
        let tok = tokenizer::Tokenizer::new(model.vocab.clone(), &model.hparams);
        let n_mels = model.hparams.n_mels as usize;
        let filters = if model.mel_filters.is_empty() {
            audio::mel_filterbank(n_mels, audio::N_FFT, audio::SAMPLE_RATE)
        } else {
            model.mel_filters.clone()
        };
        run_guided_mode(
            &model,
            &tok,
            &filters,
            &capture,
            &commands,
            &p,
            out_file.as_mut(),
        );
    } else if !p.prompt.is_empty() && grammar_source.is_none() {
        eprintln!(
            "whisper-command: always-prompt mode (activation phrase {:?})",
            p.prompt
        );
        run_always_prompt_mode(&model, &capture, &base_opts, &p, out_file.as_mut());
    } else {
        let prompt = if p.prompt.is_empty() {
            DEFAULT_PROMPT.to_string()
        } else {
            p.prompt.clone()
        };
        eprintln!("whisper-command: general-purpose mode (say {prompt:?} to begin)");
        if grammar_source.is_none() {
            eprintln!("--grammar not given: \"prompt\"/\"root\" rules unavailable, decoding unconstrained");
        }
        run_general_purpose_mode(
            &model,
            &capture,
            &base_opts,
            &prompt,
            grammar_source.as_deref(),
            &p,
            out_file.as_mut(),
        );
    }

    std::process::ExitCode::SUCCESS
}

fn run_guided_mode(
    model: &model::Model,
    tok: &tokenizer::Tokenizer,
    filters: &[f32],
    capture: &mic::MicCapture,
    commands: &[String],
    p: &Params,
    mut out_file: Option<&mut std::fs::File>,
) {
    loop {
        std::thread::sleep(Duration::from_millis(100));
        let mut window = capture.get(2000);
        if !whisper_command_vad_simple(
            &mut window,
            mic::SAMPLE_RATE,
            1000,
            p.vad_thold,
            p.freq_thold,
        ) {
            continue;
        }
        if let Some((idx, score)) = guided_mode_score(model, tok, filters, &window, commands) {
            println!("detected command: {} (p = {:.3})", commands[idx], score);
            if let Some(f) = out_file.as_deref_mut() {
                let _ = writeln!(f, "{}", commands[idx]);
            }
        }
        capture.clear();
    }
}

fn run_always_prompt_mode(
    model: &model::Model,
    capture: &mic::MicCapture,
    base_opts: &transcribe::Options,
    p: &Params,
    mut out_file: Option<&mut std::fs::File>,
) {
    loop {
        std::thread::sleep(Duration::from_millis(100));
        let mut probe = capture.get(2000);
        if !whisper_command_vad_simple(
            &mut probe,
            mic::SAMPLE_RATE,
            1000,
            p.vad_thold,
            p.freq_thold,
        ) {
            continue;
        }
        let window = capture.get(COMMAND_MS);
        capture.clear();

        let transcript = transcribe::transcribe(model, &window, base_opts);
        let text = transcript
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        let words: Vec<&str> = text.split_whitespace().collect();
        let k = p.prompt.split_whitespace().count().min(words.len());
        let (prompt_words, command_words) = words.split_at(k);
        let prompt_text = prompt_words.join(" ");
        let command_text = command_words.join(" ");

        if !command_text.is_empty()
            && similarity(&prompt_text.to_lowercase(), &p.prompt.to_lowercase()) > 0.7
        {
            println!("Command: {command_text}");
            if let Some(f) = out_file.as_deref_mut() {
                let _ = writeln!(f, "{command_text}");
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_general_purpose_mode(
    model: &model::Model,
    capture: &mic::MicCapture,
    base_opts: &transcribe::Options,
    prompt: &str,
    grammar_source: Option<&str>,
    p: &Params,
    mut out_file: Option<&mut std::fs::File>,
) {
    let prompt_grammar = grammar_source.and_then(|s| parse_grammar_rule(s, "prompt"));
    let root_grammar = grammar_source.and_then(|s| parse_grammar_rule(s, "root"));
    let mut have_prompt = false;
    let mut wake_audio: Vec<f32> = Vec::new();

    loop {
        std::thread::sleep(Duration::from_millis(100));
        let mut probe = capture.get(2000);
        if !whisper_command_vad_simple(
            &mut probe,
            mic::SAMPLE_RATE,
            1000,
            p.vad_thold,
            p.freq_thold,
        ) {
            continue;
        }

        if !have_prompt {
            let window = capture.get(PROMPT_MS);
            capture.clear();
            let mut opts = base_opts.clone();
            opts.grammar = prompt_grammar.clone();
            let transcript = transcribe::transcribe(model, &window, &opts);
            let text = transcript
                .segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let len_ratio = text.chars().count() as f32 / prompt.chars().count().max(1) as f32;
            if similarity(&text.to_lowercase(), &prompt.to_lowercase()) >= 0.8
                && (0.8..=1.2).contains(&len_ratio)
            {
                have_prompt = true;
                wake_audio = window;
                println!("activated: listening for a command...");
            } else {
                eprintln!("(say {prompt:?} to begin)");
            }
        } else {
            let cmd_window = capture.get(COMMAND_MS);
            capture.clear();
            let mut buf = wake_audio.clone();
            buf.extend(std::iter::repeat_n(0.0f32, 3 * mic::SAMPLE_RATE as usize));
            buf.extend_from_slice(&cmd_window);

            let mut opts = base_opts.clone();
            opts.grammar = root_grammar.clone();
            let transcript = transcribe::transcribe(model, &buf, &opts);
            let text = transcript
                .segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let cut = best_prompt_cut(&text, prompt);
            let command_text: String = text
                .chars()
                .skip(cut)
                .collect::<String>()
                .trim()
                .to_string();
            println!("Command: {command_text}");
            if let Some(f) = out_file.as_deref_mut() {
                let _ = writeln!(f, "{command_text}");
            }
        }
    }
}

/// One-pole high-pass IIR filter, in place — mirrors whisper.cpp's
/// `high_pass_filter` (`common.cpp`), same as `whisper-stream`'s copy.
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

/// Simple energy VAD — mirrors whisper.cpp's `vad_simple` (`common.cpp`),
/// same as `whisper-stream`'s copy.
fn whisper_command_vad_simple(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_identical_strings_is_one() {
        assert_eq!(similarity("hello", "hello"), 1.0);
    }

    #[test]
    fn similarity_completely_different_is_low() {
        assert!(similarity("abc", "xyz") < 0.5);
    }

    #[test]
    fn similarity_empty_strings() {
        assert_eq!(similarity("", ""), 1.0);
        assert_eq!(similarity("abc", ""), 0.0);
        assert_eq!(similarity("", "abc"), 0.0);
    }

    #[test]
    fn similarity_one_edit_apart() {
        // "cat" -> "bat" is a single substitution: 1 - 1/3.
        let s = similarity("cat", "bat");
        assert!((s - (1.0 - 1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn best_prompt_cut_finds_exact_prefix() {
        let text = "ok whisper start listening for commands turn on the lights";
        let prompt = "ok whisper start listening for commands";
        let cut = best_prompt_cut(text, prompt);
        let command: String = text.chars().skip(cut).collect();
        assert_eq!(command.trim(), "turn on the lights");
    }

    #[test]
    fn read_commands_file_trims_and_lowercases() {
        let dir =
            std::env::temp_dir().join(format!("rusty-whisper-command-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("commands.txt");
        std::fs::write(&path, "  Lights On  \n\nLIGHTS OFF\n").unwrap();
        let cmds = read_commands_file(path.to_str().unwrap()).unwrap();
        assert_eq!(
            cmds,
            vec!["lights on".to_string(), "lights off".to_string()]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_commands_file_rejects_missing_file() {
        assert!(read_commands_file("/nonexistent/commands.txt").is_err());
    }

    #[test]
    fn load_grammar_source_falls_back_to_inline_text() {
        let src = load_grammar_source("root ::= \"on\" | \"off\"");
        assert_eq!(src, "root ::= \"on\" | \"off\"");
    }

    #[test]
    fn vad_simple_true_when_tail_is_quiet() {
        let mut samples = vec![0.5f32; 8000];
        samples.extend(vec![0.0f32; 8000]);
        assert!(whisper_command_vad_simple(
            &mut samples,
            16000,
            500,
            0.6,
            0.0
        ));
    }
}
