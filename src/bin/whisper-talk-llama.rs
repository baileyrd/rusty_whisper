//! `whisper-talk-llama`: voice chatbot (STT -> LLM -> TTS) — mirrors
//! whisper.cpp's `examples/talk-llama/talk-llama.cpp` (v1.9.1), with one
//! deliberate architectural swap: instead of linking llama.cpp (or porting
//! it), the LLM side talks HTTP to a separately-run `rusty_llama --serve`
//! process (or anything else speaking the same OpenAI-compatible
//! `/v1/chat/completions` wire format) via [`rusty_whisper::llm_client`].
//! Upstream's raw-continuation-text + antiprompt-string-match generation
//! loop becomes a `messages[]` array instead; stopping is handled by the
//! server's own chat-template/EOS logic rather than an antiprompt match.
//!
//! Requires the `mic` feature (native capture via `cpal`).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusty_whisper::llm_client::{ChatClient, ChatOptions, Message};
use rusty_whisper::{json, mic, model, transcribe};

/// whisper.cpp's default persona template (`examples/talk-llama/talk-llama.cpp`,
/// the `{0}`/`{1}`/`{2}`/`{3}`/`{4}` placeholders below), used verbatim as
/// this crate's default `system` message content when `--prompt-file` isn't
/// given. `{0}`=person, `{1}`=bot_name, `{2}`=current UTC HH:MM (upstream
/// uses local time; this crate has no timezone-aware date/time dependency,
/// see `civil_from_days` below), `{3}`=current UTC year, `{4}`=the ":"
/// speaker/utterance separator used throughout the few-shot dialogue.
const DEFAULT_PERSONA_TEMPLATE: &str = " Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}'s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock.
{0}{4} What year is it?
{1}{4} We are in {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue";

struct Params {
    model_path: Option<String>,
    llama_host: String,
    llama_port: u16,
    llama_model: String,
    person: String,
    bot_name: String,
    prompt_file: Option<String>,
    session_path: Option<String>,
    wake_command: Option<String>,
    heard_ok: Option<String>,
    speak_cmd: Option<String>,
    speak_file: PathBuf,
    speak_voice_id: String,
    language: String,
    vad_thold: f32,
    freq_thold: f32,
    voice_ms: i64,
    capture_id: i64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: Option<u64>,
    max_tokens: Option<usize>,
    output_path: Option<String>,
    list_devices: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            model_path: None,
            llama_host: "127.0.0.1".to_string(),
            llama_port: 8080,
            llama_model: String::new(),
            // whisper.cpp defaults these to "Georgi"/"LLaMA" (its author's
            // name and the model family it happened to demo with); this
            // crate isn't tied to either, so a generic default reads better.
            person: "User".to_string(),
            bot_name: "Assistant".to_string(),
            prompt_file: None,
            session_path: None,
            wake_command: None,
            heard_ok: None,
            speak_cmd: None,
            speak_file: PathBuf::from("./to_speak.txt"),
            speak_voice_id: "2".to_string(),
            language: "en".to_string(),
            vad_thold: 0.6,
            freq_thold: 100.0,
            voice_ms: 10000,
            capture_id: -1,
            temperature: 0.30,
            top_k: 5,
            top_p: 0.80,
            min_p: 0.01,
            seed: None,
            max_tokens: None,
            output_path: None,
            list_devices: false,
        }
    }
}

fn print_help() {
    eprintln!(
        "whisper-talk-llama: voice chatbot, STT -> LLM (via HTTP) -> TTS (requires --features mic)"
    );
    eprintln!("  --model, -m PATH        whisper model to load (required)");
    eprintln!("  --llama-host HOST       rusty_llama --serve host (default 127.0.0.1)");
    eprintln!("  --llama-port PORT       rusty_llama --serve port (default 8080)");
    eprintln!("  --llama-model NAME      \"model\" field sent in each request (default: empty)");
    eprintln!(
        "  --person STRING         your name in the persona/prompt template (default \"User\")"
    );
    eprintln!("  --bot-name STRING       the assistant's name (default \"Assistant\")");
    eprintln!("  --prompt-file PATH      override the built-in persona template (same {{0}}../{{4}} placeholders)");
    eprintln!("  --session PATH          persist/resume conversation history (JSON) across runs");
    eprintln!("  --wake-command STRING   require this phrase (fuzzy-matched) at the start of every utterance");
    eprintln!("  --heard-ok STRING       spoken (if --speak set) once the wake check passes, before the LLM call");
    eprintln!("  --speak COMMAND         TTS command to run per reply (default: none, text-only)");
    eprintln!("  --speak-file PATH       file the reply text is written to before --speak runs (default ./to_speak.txt)");
    eprintln!("  --speak-voice-id ID     extra arg passed to --speak (default \"2\", matching whisper.cpp's default)");
    eprintln!("  -l, --language CODE     source language (default \"en\")");
    eprintln!("  -vth, --vad-thold F     energy VAD threshold (default 0.6)");
    eprintln!("  -fth, --freq-thold F    high-pass filter cutoff Hz before VAD (default 100.0)");
    eprintln!("  -vms, --voice-ms N      capture window pulled once VAD triggers (default 10000)");
    eprintln!("  -c, --capture N         capture device index (default: OS default device)");
    eprintln!("  --list-devices          list capture device names (with their index) and exit");
    eprintln!("  -t, --temp F            LLM sampling temperature (default 0.30)");
    eprintln!("  --top-k N               (default 5)");
    eprintln!("  --top-p F               (default 0.80)");
    eprintln!("  --min-p F               (default 0.01)");
    eprintln!(
        "  --seed N                LLM sampling seed (default: unset, i.e. non-deterministic)"
    );
    eprintln!("  --max-tokens N          caps the LLM reply length (default: unset). whisper.cpp's -mt caps");
    eprintln!("                          transcription instead; this crate has no analogous knob there, so");
    eprintln!("                          the flag is repurposed for the LLM side rather than left a no-op");
    eprintln!("  -f, --file PATH         append the running transcript to this file");
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
            "--llama-host" => p.llama_host = next_string!("--llama-host"),
            "--llama-port" => p.llama_port = next_parse!("--llama-port"),
            "--llama-model" => p.llama_model = next_string!("--llama-model"),
            "--person" => p.person = next_string!("--person"),
            "--bot-name" => p.bot_name = next_string!("--bot-name"),
            "--prompt-file" => p.prompt_file = args.next(),
            "--session" => p.session_path = args.next(),
            "--wake-command" => p.wake_command = args.next(),
            "--heard-ok" => p.heard_ok = args.next(),
            "--speak" => p.speak_cmd = args.next(),
            "--speak-file" => p.speak_file = PathBuf::from(next_string!("--speak-file")),
            "--speak-voice-id" => p.speak_voice_id = next_string!("--speak-voice-id"),
            "--language" | "-l" => p.language = next_string!("--language"),
            "--vad-thold" | "-vth" => p.vad_thold = next_parse!("--vad-thold"),
            "--freq-thold" | "-fth" => p.freq_thold = next_parse!("--freq-thold"),
            "--voice-ms" | "-vms" => p.voice_ms = next_parse!("--voice-ms"),
            "--capture" | "-c" => p.capture_id = next_parse!("--capture"),
            "--list-devices" => p.list_devices = true,
            "--temp" | "-t" => p.temperature = next_parse!("--temp"),
            "--top-k" => p.top_k = next_parse!("--top-k"),
            "--top-p" => p.top_p = next_parse!("--top-p"),
            "--min-p" => p.min_p = next_parse!("--min-p"),
            "--seed" => p.seed = Some(next_parse!("--seed")),
            "--max-tokens" => p.max_tokens = Some(next_parse!("--max-tokens")),
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

/// Normalized Levenshtein similarity in `[0, 1]` — mirrors whisper.cpp's
/// `similarity()` (`common.cpp`), same as `whisper-command`'s copy.
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
/// same as `whisper-stream`/`whisper-command`'s copies.
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

/// Converts a day count since the Unix epoch (1970-01-01) into a proleptic
/// Gregorian `(year, month, day)` — Howard Hinnant's `civil_from_days`
/// algorithm (public domain). Used only for the persona template's
/// current-year placeholder; this crate has no chrono-equivalent
/// dependency, so "current time" here means UTC, not local time like
/// whisper.cpp's `strftime`/`localtime`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// `(year, hour, minute)` for the persona template's `{2}`/`{3}` placeholders.
fn utc_year_and_time() -> (i64, u32, u32) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let secs_of_day = secs.rem_euclid(86400);
    let (year, _, _) = civil_from_days(days);
    (
        year,
        (secs_of_day / 3600) as u32,
        ((secs_of_day % 3600) / 60) as u32,
    )
}

fn render_persona(template: &str, person: &str, bot_name: &str) -> String {
    let (year, hour, minute) = utc_year_and_time();
    template
        .replace("{0}", person)
        .replace("{1}", bot_name)
        .replace("{2}", &format!("{hour:02}:{minute:02}"))
        .replace("{3}", &year.to_string())
        .replace("{4}", ":")
}

fn load_session(path: &Path) -> Vec<Message> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = json::parse(&content) else {
        eprintln!(
            "--session: {} is not valid JSON, starting fresh",
            path.display()
        );
        return Vec::new();
    };
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let role = item.get("role")?.as_str()?.to_string();
            let content = item.get("content")?.as_str()?.to_string();
            Some(Message { role, content })
        })
        .collect()
}

fn save_session(path: &Path, history: &[Message]) {
    let items: Vec<String> = history
        .iter()
        .map(|m| {
            format!(
                "{{\"role\":{},\"content\":{}}}",
                json::escape(&m.role),
                json::escape(&m.content)
            )
        })
        .collect();
    if let Err(e) = std::fs::write(path, format!("[{}]", items.join(","))) {
        eprintln!("--session: failed to save {}: {e}", path.display());
    }
}

/// Writes `text` to `speak_file` then runs `speak_cmd [its own args...]
/// speak_voice_id speak_file`, blocking until it exits — mirrors whisper.cpp's
/// `speak_with_file` (`common-whisper.cpp`), except via `Command` (argv,
/// no shell) rather than upstream's `system()` call, avoiding a shell-
/// injection footgun for no behavioral loss.
fn speak(speak_cmd: &str, speak_file: &Path, voice_id: &str, text: &str) {
    if let Err(e) = std::fs::write(speak_file, text) {
        eprintln!("--speak: failed to write {}: {e}", speak_file.display());
        return;
    }
    let mut parts = speak_cmd.split_whitespace();
    let Some(program) = parts.next() else {
        return;
    };
    let status = std::process::Command::new(program)
        .args(parts)
        .arg(voice_id)
        .arg(speak_file)
        .status();
    if let Err(e) = status {
        eprintln!("--speak: failed to run {speak_cmd:?}: {e}");
    }
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
    let whisper_model = match std::fs::File::open(model_path)
        .and_then(|f| model::load_model(&mut std::io::BufReader::new(f)))
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load whisper model {model_path}: {e}");
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
    let capture = match mic::init(p.voice_ms as usize, device_name.as_deref()) {
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

    let persona_template = match &p.prompt_file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(s) => s.trim_end_matches('\n').to_string(),
            Err(e) => {
                eprintln!("--prompt-file: failed to read {path}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        },
        None => DEFAULT_PERSONA_TEMPLATE.to_string(),
    };
    let system_message = Message::system(render_persona(&persona_template, &p.person, &p.bot_name));

    let session_path = p.session_path.as_ref().map(PathBuf::from);
    let mut history: Vec<Message> = session_path
        .as_deref()
        .map(load_session)
        .unwrap_or_default();

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
    let whisper_opts = transcribe::Options {
        language,
        initial_prompt: Some(format!(
            "A conversation with a person called {}.",
            p.bot_name
        )),
        ..transcribe::Options::default()
    };

    let llm = ChatClient::new(p.llama_host.clone(), p.llama_port, p.llama_model.clone());
    let chat_opts = ChatOptions {
        temperature: Some(p.temperature),
        top_p: Some(p.top_p),
        top_k: Some(p.top_k),
        min_p: Some(p.min_p),
        max_tokens: p.max_tokens,
        seed: p.seed,
    };

    eprintln!(
        "whisper-talk-llama: listening (person={:?}, bot_name={:?}, llama={}:{}{})",
        p.person,
        p.bot_name,
        p.llama_host,
        p.llama_port,
        if p.wake_command.is_some() {
            ", wake-word gated"
        } else {
            ""
        }
    );

    let mut t_last = Instant::now()
        .checked_sub(Duration::from_millis(2000))
        .unwrap_or_else(Instant::now);

    loop {
        if t_last.elapsed() < Duration::from_millis(2000) {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let mut probe = capture.get(2000);
        if !vad_simple(
            &mut probe,
            mic::SAMPLE_RATE,
            1250,
            p.vad_thold,
            p.freq_thold,
        ) {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let window = capture.get(p.voice_ms);
        capture.clear();
        t_last = Instant::now();

        let transcript = transcribe::transcribe(&whisper_model, &window, &whisper_opts);
        let heard: String = transcript
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        if heard.is_empty() {
            continue;
        }

        if let Some(wake) = &p.wake_command {
            let words: Vec<&str> = heard.split_whitespace().collect();
            let k = wake.split_whitespace().count().min(words.len());
            let prefix = words[..k].join(" ");
            if similarity(&prefix.to_lowercase(), &wake.to_lowercase()) < 0.7 {
                continue; // discard: doesn't start with the wake phrase
            }
            if let Some(ack) = &p.heard_ok {
                if let Some(cmd) = &p.speak_cmd {
                    speak(cmd, &p.speak_file, &p.speak_voice_id, ack);
                }
            }
        }

        println!("{}: {heard}", p.person);
        if let Some(f) = out_file.as_mut() {
            let _ = writeln!(f, "{}: {heard}", p.person);
        }

        history.push(Message::user(heard));
        let mut request_messages = vec![system_message.clone()];
        request_messages.extend(history.iter().cloned());

        match llm.chat(&request_messages, &chat_opts) {
            Ok(reply) => {
                let reply = reply.trim().to_string();
                println!("{}: {reply}", p.bot_name);
                history.push(Message::assistant(reply.clone()));
                if let Some(f) = out_file.as_mut() {
                    let _ = writeln!(f, "{}: {reply}", p.bot_name);
                }
                if let Some(path) = &session_path {
                    save_session(path, &history);
                }
                if let Some(cmd) = &p.speak_cmd {
                    speak(cmd, &p.speak_file, &p.speak_voice_id, &reply);
                }
            }
            Err(e) => {
                eprintln!("LLM request failed: {e}");
                history.pop(); // don't leave a dangling unanswered user turn
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_identical_strings_is_one() {
        assert_eq!(similarity("hello", "hello"), 1.0);
    }

    #[test]
    fn vad_simple_true_when_tail_is_quiet() {
        let mut samples = vec![0.5f32; 8000];
        samples.extend(vec![0.0f32; 8000]);
        assert!(vad_simple(&mut samples, 16000, 500, 0.6, 0.0));
    }

    #[test]
    fn civil_from_days_epoch_is_1970_01_01() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn civil_from_days_known_date() {
        // 2000-01-01 is 10957 days after the epoch.
        assert_eq!(civil_from_days(10957), (2000, 1, 1));
    }

    #[test]
    fn render_persona_substitutes_all_placeholders() {
        let out = render_persona("{0} meets {1} at {2} in {3}, sep{4}", "Alice", "Bob");
        assert!(out.starts_with("Alice meets Bob at "));
        assert!(out.contains(", sep:"));
        assert!(!out.contains('{'));
    }

    #[test]
    fn load_session_missing_file_is_empty() {
        assert!(load_session(Path::new("/nonexistent/session.json")).is_empty());
    }

    #[test]
    fn session_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "rusty-whisper-talk-llama-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.json");
        let history = vec![Message::user("hi"), Message::assistant("hello!")];
        save_session(&path, &history);
        let loaded = load_session(&path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].role, "user");
        assert_eq!(loaded[0].content, "hi");
        assert_eq!(loaded[1].role, "assistant");
        assert_eq!(loaded[1].content, "hello!");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_persona_template_has_no_stray_placeholders_after_render() {
        let rendered = render_persona(DEFAULT_PERSONA_TEMPLATE, "User", "Assistant");
        assert!(!rendered.contains("{0}"));
        assert!(!rendered.contains("{1}"));
        assert!(!rendered.contains("{2}"));
        assert!(!rendered.contains("{3}"));
        assert!(!rendered.contains("{4}"));
    }
}
