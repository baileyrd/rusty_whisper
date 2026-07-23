//! Request/response logic for `whisper-server`'s `POST /inference` and
//! `POST /load` — multipart upload parsing, per-request `Options`
//! overrides, `response_format` negotiation, and the `/load` model-path
//! extraction. Kept separate from the `whisper-server` binary (which owns
//! only TCP/HTTP plumbing — see `crate::http`) so all of this is
//! unit-testable without a socket.

use std::collections::HashMap;
use std::io;

use crate::transcribe::{Options, Transcript};

#[derive(Debug, PartialEq)]
pub struct MultipartField {
    pub name: String,
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > haystack.len() || needle.is_empty() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// Parse a `multipart/form-data` body (RFC 2046) into its fields. `boundary`
/// is the value from the request's `Content-Type: multipart/form-data;
/// boundary=...` header, without the leading `--`. Malformed parts (no
/// blank line ending the header block, no `name=` in `Content-Disposition`)
/// are skipped rather than erroring the whole request.
pub fn parse_multipart(body: &[u8], boundary: &str) -> Vec<MultipartField> {
    let marker = format!("--{boundary}").into_bytes();
    let mut fields = Vec::new();
    let mut positions = Vec::new();
    let mut pos = 0;
    while let Some(i) = find(body, &marker, pos) {
        positions.push(i);
        pos = i + marker.len();
    }
    for w in positions.windows(2) {
        let (start, end) = (w[0] + marker.len(), w[1]);
        if start > end {
            continue;
        }
        let mut part = &body[start..end];
        // Each part starts with \r\n right after the boundary marker.
        part = part.strip_prefix(b"\r\n").unwrap_or(part);
        // ...and the boundary marker is preceded by \r\n that closes the body.
        let part = part.strip_suffix(b"\r\n").unwrap_or(part);

        let Some(header_end) = find(part, b"\r\n\r\n", 0) else {
            continue;
        };
        let header_block = String::from_utf8_lossy(&part[..header_end]);
        let data = part[header_end + 4..].to_vec();

        let mut name = None;
        let mut filename = None;
        for line in header_block.split("\r\n") {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            if !key.trim().eq_ignore_ascii_case("content-disposition") {
                continue;
            }
            for piece in value.split(';') {
                let piece = piece.trim();
                if let Some(v) = piece.strip_prefix("name=") {
                    name = Some(v.trim_matches('"').to_string());
                } else if let Some(v) = piece.strip_prefix("filename=") {
                    filename = Some(v.trim_matches('"').to_string());
                }
            }
        }
        if let Some(name) = name {
            fields.push(MultipartField {
                name,
                filename,
                data,
            });
        }
    }
    fields
}

/// Extracts the boundary token from a `Content-Type` header value, e.g.
/// `multipart/form-data; boundary=----WebKitFormBoundaryABC` ->
/// `Some("----WebKitFormBoundaryABC")`. Returns `None` for anything that
/// isn't `multipart/form-data` or has no boundary.
pub fn multipart_boundary(content_type: &str) -> Option<String> {
    if !content_type
        .split(';')
        .next()?
        .trim()
        .eq_ignore_ascii_case("multipart/form-data")
    {
        return None;
    }
    content_type.split(';').skip(1).find_map(|p| {
        let p = p.trim();
        p.strip_prefix("boundary=")
            .map(|b| b.trim_matches('"').to_string())
    })
}

/// Parsed `POST /inference` request: the transcribed-audio `Options`
/// overrides plus the request-level settings that live outside `Options`
/// (crop window, response format, post-hoc diarization, VAD gate).
/// Fields whisper.cpp's server documents but this crate doesn't have a
/// clean way to apply per-request (`audio_ctx`, `word_thold`,
/// `no_timestamps`, `no_language_probabilities`, `detect_language`,
/// `tinydiarize`) are still parsed into the matching `Options` field where
/// one exists (so they round-trip and don't silently vanish) but carry the
/// same "accepted, not fully applied" caveat those fields already have on
/// the CLI — see their doc comments on `Options` itself.
pub struct InferenceRequest {
    pub opts: Options,
    pub response_format: String,
    pub diarize: bool,
    pub offset_ms: u64,
    pub duration_ms: u64,
}

/// Builds an `InferenceRequest` from a `POST /inference` form's non-file
/// fields (name -> value). Unknown fields are ignored; fields present but
/// unparseable as their expected type are also ignored (left at the
/// `Options` default) rather than rejecting the whole request over one bad
/// field.
pub fn parse_inference_fields(fields: &HashMap<String, String>) -> InferenceRequest {
    fn parse<T: std::str::FromStr>(fields: &HashMap<String, String>, key: &str) -> Option<T> {
        fields.get(key).and_then(|v| v.parse::<T>().ok())
    }
    let get = |k: &str| fields.get(k).map(|s| s.as_str());
    let flag = |k: &str| matches!(get(k), Some("true") | Some("1"));

    let mut opts = Options::default();
    if let Some(v) = get("language") {
        if v != "auto" {
            opts.language = Some(v.to_string());
        }
    }
    opts.translate = flag("translate");
    opts.tinydiarize = flag("tinydiarize");
    opts.suppress_non_speech = flag("suppress_nst");
    opts.carry_initial_prompt = flag("carry_initial_prompt");
    if let Some(v) = get("prompt") {
        if !v.is_empty() {
            opts.initial_prompt = Some(v.to_string());
        }
    }
    if let Some(v) = parse::<usize>(fields, "max_context") {
        opts.max_context = Some(v);
    }
    if let Some(v) = parse::<usize>(fields, "max_len") {
        opts.max_len = v;
    }
    if let Some(v) = parse::<usize>(fields, "best_of") {
        opts.best_of = v;
    }
    if let Some(v) = parse::<usize>(fields, "beam_size") {
        opts.beam_size = v;
    }
    if let Some(v) = parse::<usize>(fields, "audio_ctx") {
        opts.audio_ctx = Some(v);
    }
    if let Some(v) = parse::<f32>(fields, "word_thold") {
        opts.word_thold = v;
    }
    if let Some(v) = parse::<f32>(fields, "entropy_thold") {
        opts.entropy_threshold = v;
    }
    if let Some(v) = parse::<f32>(fields, "logprob_thold") {
        opts.logprob_threshold = v;
    }
    if let Some(v) = parse::<f32>(fields, "no_speech_thold") {
        opts.no_speech_threshold = v;
    }
    let temperature = parse::<f32>(fields, "temperature").unwrap_or(0.0);
    let temperature_inc = parse::<f32>(fields, "temperature_inc").unwrap_or(0.2);
    opts.temperatures = crate::transcribe::temperature_ladder(temperature, temperature_inc);

    InferenceRequest {
        opts,
        response_format: get("response_format").unwrap_or("json").to_string(),
        diarize: flag("diarize"),
        offset_ms: parse::<u64>(fields, "offset").unwrap_or(0),
        duration_ms: parse::<u64>(fields, "duration").unwrap_or(0),
    }
}

/// Renders a `Transcript` per `response_format` ("json" | "text" | "srt" |
/// "vtt" | "verbose_json", matching whisper.cpp's server; unrecognized
/// values fall back to "json" rather than erroring, since the request
/// already succeeded in transcribing). Returns `(content_type, body)`.
pub fn format_transcript(
    format: &str,
    transcript: &Transcript,
) -> io::Result<(&'static str, Vec<u8>)> {
    let mut buf = Vec::new();
    let content_type = match format {
        "text" => {
            crate::output::write_txt(&transcript.segments, &mut buf)?;
            "text/plain; charset=utf-8"
        }
        "srt" => {
            crate::output::write_srt(&transcript.segments, 0, &mut buf)?;
            "application/x-subrip"
        }
        "vtt" => {
            crate::output::write_vtt(&transcript.segments, &mut buf)?;
            "text/vtt"
        }
        "verbose_json" => {
            crate::output::write_json_full(&transcript.language, &transcript.segments, &mut buf)?;
            "application/json"
        }
        _ => {
            crate::output::write_json(&transcript.language, &transcript.segments, &mut buf)?;
            "application/json"
        }
    };
    Ok((content_type, buf))
}

/// Extracts the model path from a `POST /load` request body: a JSON body
/// (`Content-Type: application/json`, `{"model": "path"}`) has its
/// `"model"` string field pulled out via [`json_string_field`]; any other
/// body is treated as the literal path, trimmed. Returns `None` for an
/// empty/missing path either way.
pub fn extract_load_model_path(content_type: Option<&str>, body: &[u8]) -> Option<String> {
    let is_json = content_type
        .and_then(|ct| ct.split(';').next())
        .is_some_and(|ct| ct.trim().eq_ignore_ascii_case("application/json"));
    let path = if is_json {
        json_string_field(body, "model")?
    } else {
        String::from_utf8_lossy(body).trim().to_string()
    };
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// Pulls one string field's value out of a JSON object by a literal text
/// scan — deliberately *not* a general JSON parser (this project is
/// zero-dependency and `/load`'s request body only ever needs this one
/// field). Doesn't handle escaped quotes within the value; a model path
/// containing a literal `"` isn't a realistic input.
fn json_string_field(body: &[u8], field: &str) -> Option<String> {
    let text = std::str::from_utf8(body).ok()?;
    let key_pat = format!("\"{field}\"");
    let after_key = &text[text.find(&key_pat)? + key_pat.len()..];
    let after_colon = after_key[after_key.find(':')? + 1..].trim_start();
    let rest = after_colon.strip_prefix('"')?;
    Some(rest[..rest.find('"')?].to_string())
}

static TMP_FILE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Transcode arbitrary audio bytes into a 16kHz mono WAV by shelling out to
/// `ffmpeg` — whisper.cpp's server `--convert` flag, for uploads that
/// aren't already the WAV format the pipeline expects. This is a runtime
/// external-process invocation (`std::process::Command`), not a Rust crate
/// dependency, so it doesn't trip this project's zero-dependency stance
/// the way vendoring an audio-decoding crate would.
///
/// Writes `bytes` and ffmpeg's output to temp files under `tmp_dir`
/// (matching whisper.cpp's own `--tmp-dir`), removing both again before
/// returning — on the error paths too, not just on success. Returns a
/// plain [`io::Error`] (not a panic) if `ffmpeg` isn't installed, exits
/// non-zero, or produces something rusty_whisper's own reader still can't
/// parse — this is a request-time failure a caller should see as a 4xx/5xx
/// response, not a reason to crash the server.
pub fn convert_with_ffmpeg(bytes: &[u8], tmp_dir: &std::path::Path) -> io::Result<Vec<u8>> {
    std::fs::create_dir_all(tmp_dir)?;
    let n = TMP_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = format!("rusty-whisper-upload-{}-{n}", std::process::id());
    let input_path = tmp_dir.join(format!("{stem}.input"));
    let output_path = tmp_dir.join(format!("{stem}.wav"));
    let cleanup = || {
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    };

    std::fs::write(&input_path, bytes)?;
    let run = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(&input_path)
        .args(["-ar", "16000", "-ac", "1", "-f", "wav"])
        .arg(&output_path)
        .output();
    let output = match run {
        Ok(o) => o,
        Err(e) => {
            cleanup();
            return Err(e);
        }
    };
    if !output.status.success() {
        cleanup();
        return Err(io::Error::other(format!(
            "ffmpeg exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let wav_bytes = std::fs::read(&output_path);
    cleanup();
    wav_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_multipart(boundary: &str, fields: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        // fields: (name, filename, data)
        let mut out = Vec::new();
        for (name, filename, data) in fields {
            out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            match filename {
                Some(fname) => out.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes(),
                ),
                None => out.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                ),
            }
            out.extend_from_slice(data);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        out
    }

    #[test]
    fn multipart_boundary_extracts_from_content_type() {
        assert_eq!(
            multipart_boundary("multipart/form-data; boundary=abc123"),
            Some("abc123".to_string())
        );
        assert_eq!(
            multipart_boundary("multipart/form-data; boundary=\"abc 123\""),
            Some("abc 123".to_string())
        );
        assert_eq!(multipart_boundary("application/json"), None);
        assert_eq!(multipart_boundary("multipart/form-data"), None);
    }

    #[test]
    fn parse_multipart_extracts_file_and_text_fields() {
        let body = build_multipart(
            "B",
            &[
                ("file", Some("audio.wav"), b"RIFF....fake wav bytes"),
                ("language", None, b"en"),
                ("translate", None, b"true"),
            ],
        );
        let fields = parse_multipart(&body, "B");
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name, "file");
        assert_eq!(fields[0].filename.as_deref(), Some("audio.wav"));
        assert_eq!(fields[0].data, b"RIFF....fake wav bytes");
        assert_eq!(fields[1].name, "language");
        assert_eq!(fields[1].data, b"en");
        assert_eq!(fields[2].name, "translate");
        assert_eq!(fields[2].data, b"true");
    }

    #[test]
    fn parse_multipart_empty_body_yields_no_fields() {
        assert!(parse_multipart(b"", "B").is_empty());
    }

    #[test]
    fn parse_multipart_handles_binary_data_containing_crlf() {
        let binary = b"\r\n\x00\x01\xff--not-a-boundary\r\n".to_vec();
        let body = build_multipart("BOUND", &[("file", Some("a.wav"), &binary)]);
        let fields = parse_multipart(&body, "BOUND");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].data, binary);
    }

    #[test]
    fn parse_inference_fields_defaults_to_json_and_no_overrides() {
        let req = parse_inference_fields(&HashMap::new());
        assert_eq!(req.response_format, "json");
        assert!(!req.diarize);
        assert_eq!(req.offset_ms, 0);
        assert_eq!(req.duration_ms, 0);
        assert_eq!(req.opts.language, None);
        assert!(!req.opts.translate);
    }

    #[test]
    fn parse_inference_fields_applies_known_overrides() {
        let mut fields = HashMap::new();
        fields.insert("language".to_string(), "de".to_string());
        fields.insert("translate".to_string(), "true".to_string());
        fields.insert("response_format".to_string(), "srt".to_string());
        fields.insert("best_of".to_string(), "3".to_string());
        fields.insert("beam_size".to_string(), "1".to_string());
        fields.insert("max_len".to_string(), "42".to_string());
        fields.insert("diarize".to_string(), "true".to_string());
        fields.insert("offset".to_string(), "1000".to_string());
        fields.insert("duration".to_string(), "5000".to_string());
        fields.insert("prompt".to_string(), "hello".to_string());
        fields.insert("carry_initial_prompt".to_string(), "1".to_string());

        let req = parse_inference_fields(&fields);
        assert_eq!(req.opts.language, Some("de".to_string()));
        assert!(req.opts.translate);
        assert_eq!(req.response_format, "srt");
        assert_eq!(req.opts.best_of, 3);
        assert_eq!(req.opts.beam_size, 1);
        assert_eq!(req.opts.max_len, 42);
        assert!(req.diarize);
        assert_eq!(req.offset_ms, 1000);
        assert_eq!(req.duration_ms, 5000);
        assert_eq!(req.opts.initial_prompt, Some("hello".to_string()));
        assert!(req.opts.carry_initial_prompt);
    }

    #[test]
    fn parse_inference_fields_language_auto_means_detect() {
        let mut fields = HashMap::new();
        fields.insert("language".to_string(), "auto".to_string());
        let req = parse_inference_fields(&fields);
        assert_eq!(req.opts.language, None);
    }

    #[test]
    fn parse_inference_fields_bad_numeric_value_is_ignored() {
        let mut fields = HashMap::new();
        fields.insert("best_of".to_string(), "not-a-number".to_string());
        let req = parse_inference_fields(&fields);
        assert_eq!(req.opts.best_of, Options::default().best_of);
    }

    fn sample_transcript() -> Transcript {
        Transcript {
            segments: vec![crate::transcribe::Segment {
                t0: 0.0,
                t1: 1.0,
                text: "hi".to_string(),
                tokens: Vec::new(),
            }],
            language: "en".to_string(),
        }
    }

    #[test]
    fn format_transcript_covers_every_known_format() {
        let t = sample_transcript();
        for (fmt, expect_ct) in [
            ("json", "application/json"),
            ("verbose_json", "application/json"),
            ("text", "text/plain; charset=utf-8"),
            ("srt", "application/x-subrip"),
            ("vtt", "text/vtt"),
            ("unknown-format", "application/json"),
        ] {
            let (ct, body) = format_transcript(fmt, &t).unwrap();
            assert_eq!(ct, expect_ct, "format {fmt}");
            assert!(!body.is_empty(), "format {fmt}");
        }
    }

    #[test]
    fn extract_load_model_path_from_json_body() {
        let body = br#"{"model": "/models/ggml-base.bin"}"#;
        assert_eq!(
            extract_load_model_path(Some("application/json"), body),
            Some("/models/ggml-base.bin".to_string())
        );
    }

    #[test]
    fn extract_load_model_path_from_json_body_with_other_fields() {
        let body = br#"{"foo": 1, "model": "/m.bin", "bar": true}"#;
        assert_eq!(
            extract_load_model_path(Some("application/json; charset=utf-8"), body),
            Some("/m.bin".to_string())
        );
    }

    #[test]
    fn extract_load_model_path_treats_non_json_body_as_a_literal_path() {
        assert_eq!(
            extract_load_model_path(None, b"/models/ggml-base.bin\n"),
            Some("/models/ggml-base.bin".to_string())
        );
        assert_eq!(
            extract_load_model_path(Some("text/plain"), b"  /m.bin  "),
            Some("/m.bin".to_string())
        );
    }

    #[test]
    fn extract_load_model_path_empty_is_none() {
        assert_eq!(extract_load_model_path(None, b""), None);
        assert_eq!(extract_load_model_path(None, b"   "), None);
        assert_eq!(
            extract_load_model_path(Some("application/json"), br#"{"model": ""}"#),
            None
        );
    }

    #[test]
    fn extract_load_model_path_json_missing_field_is_none() {
        assert_eq!(
            extract_load_model_path(Some("application/json"), br#"{"other": "x"}"#),
            None
        );
    }

    #[test]
    fn convert_with_ffmpeg_cleans_up_temp_files_even_when_ffmpeg_is_missing() {
        // This environment (like many CI/sandbox setups) has no ffmpeg
        // installed, so this exercises the real "binary not found" error
        // path rather than mocking it -- and confirms it's a plain Err,
        // not a panic, plus that no temp files are left behind.
        let dir = std::env::temp_dir().join(format!(
            "rw_ffmpeg_test_{}_{}",
            std::process::id(),
            TMP_FILE_COUNTER.load(std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let result = convert_with_ffmpeg(b"not really audio", &dir);
        assert!(result.is_err());

        let leftover: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert!(
            leftover.is_empty(),
            "temp files should be cleaned up on failure, found: {leftover:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
