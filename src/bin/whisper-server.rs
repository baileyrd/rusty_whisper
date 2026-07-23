//! `whisper-server`: HTTP core + `/inference` + `/load` for rusty-whisper —
//! mirrors whisper.cpp's `examples/server/server.cpp` (routing, `GET
//! /health`, `GET /` static serving, CORS, `POST /inference`, `POST
//! /load`).

use std::collections::HashMap;
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rusty_whisper::http::{self, Request, Response};
use rusty_whisper::{audio, model, server, transcribe, wav};

struct ServerState {
    ready: AtomicBool,
    /// Locked for the whole `/inference` call — whisper.cpp's own server
    /// serializes inference access on a single mutex too, rather than
    /// letting concurrent requests race on the model's decode state.
    model: Mutex<Option<model::Model>>,
    /// `--convert`: shell out to `ffmpeg` for uploads that aren't already
    /// a 16kHz WAV, instead of rejecting them.
    convert: bool,
    /// `--tmp-dir`: where `--convert`'s ffmpeg input/output temp files go.
    tmp_dir: PathBuf,
}

fn main() -> ExitCode {
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut public_dir: Option<PathBuf> = None;
    let mut model_path: Option<String> = None;
    let mut convert = false;
    let mut tmp_dir: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => match args.next() {
                Some(v) => host = v,
                None => {
                    eprintln!("--host requires a value");
                    return ExitCode::FAILURE;
                }
            },
            "--port" => match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => port = v,
                None => {
                    eprintln!("--port requires a number");
                    return ExitCode::FAILURE;
                }
            },
            "--public" => match args.next() {
                Some(v) => public_dir = Some(PathBuf::from(v)),
                None => {
                    eprintln!("--public requires a directory");
                    return ExitCode::FAILURE;
                }
            },
            "--model" | "-m" => model_path = args.next(),
            "--convert" => convert = true,
            "--tmp-dir" => match args.next() {
                Some(v) => tmp_dir = Some(PathBuf::from(v)),
                None => {
                    eprintln!("--tmp-dir requires a directory");
                    return ExitCode::FAILURE;
                }
            },
            "--help" | "-h" => {
                eprintln!("whisper-server: HTTP core + /inference + /load (GET /health, GET /, CORS, POST /inference, POST /load)");
                eprintln!("  --host HOST     bind address (default 127.0.0.1)");
                eprintln!("  --port PORT     bind port (default 8080)");
                eprintln!(
                    "  --public DIR    serve static files from DIR instead of the built-in page"
                );
                eprintln!("  --model, -m PATH  load a model at startup (required for /inference; GET /health reports 503 until loaded)");
                eprintln!(
                    "  --convert       shell out to ffmpeg to transcode non-16kHz-WAV /inference uploads"
                );
                eprintln!(
                    "  --tmp-dir DIR   where --convert writes its ffmpeg input/output temp files (default: system temp dir)"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other} (see --help)");
                return ExitCode::FAILURE;
            }
        }
    }

    let state = Arc::new(ServerState {
        ready: AtomicBool::new(model_path.is_none()),
        model: Mutex::new(None),
        convert,
        tmp_dir: tmp_dir.unwrap_or_else(std::env::temp_dir),
    });
    if let Some(path) = model_path {
        spawn_model_load(state.clone(), path);
    }

    let listener = match TcpListener::bind((host.as_str(), port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {host}:{port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("whisper-server listening on http://{host}:{port}");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        let state = state.clone();
        let public_dir = public_dir.clone();
        std::thread::spawn(move || handle_connection(stream, public_dir, &state));
    }
    ExitCode::SUCCESS
}

/// Loads a model on a background thread and swaps it into `state` once
/// done, marking the server ready again — used both for the initial
/// `--model` load and `POST /load`. Swapping through `state.model`'s mutex
/// (rather than loading while holding it) means an in-flight `/inference`
/// request finishes against the old model instead of racing the swap, and
/// the new model can't be read until the swap is visible.
fn spawn_model_load(state: Arc<ServerState>, path: String) {
    std::thread::spawn(move || {
        match std::fs::File::open(&path).and_then(|f| model::load_model(&mut BufReader::new(f))) {
            Ok(m) => {
                *state.model.lock().unwrap() = Some(m);
                state.ready.store(true, Ordering::Release);
            }
            Err(e) => eprintln!("failed to load model {path}: {e}"),
        }
    });
}

fn handle_connection(mut stream: TcpStream, public_dir: Option<PathBuf>, state: &Arc<ServerState>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let req = match http::parse_request(&mut reader) {
        Ok(r) => r,
        Err(_) => {
            let _ = Response::json(400, r#"{"error":"bad request"}"#).write_to(&mut stream);
            return;
        }
    };
    let resp = match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/inference") => handle_inference(&req, state).with_cors(),
        ("POST", "/load") => handle_load(&req, state).with_cors(),
        _ => http::route(&req, public_dir.as_deref(), &state.ready),
    };
    let _ = resp.write_to(&mut stream);
}

/// `POST /load`: hot-swaps the loaded model without restarting the
/// server. Body is either `{"model": "path"}` (`Content-Type:
/// application/json`) or the raw path as plain text — see
/// `server::extract_load_model_path`. Loading happens in the background
/// (same as the initial `--model` load); the response is `202` immediately,
/// with `GET /health` the way to poll for completion.
fn handle_load(req: &Request, state: &Arc<ServerState>) -> Response {
    let path = server::extract_load_model_path(req.header("content-type"), &req.body);
    let Some(path) = path else {
        return Response::json(400, r#"{"error":"missing 'model' path"}"#);
    };
    state.ready.store(false, Ordering::Release);
    spawn_model_load(state.clone(), path);
    Response::json(202, r#"{"status":"loading"}"#)
}

fn handle_inference(req: &Request, state: &ServerState) -> Response {
    let Some(content_type) = req.header("content-type") else {
        return Response::json(400, r#"{"error":"missing Content-Type header"}"#);
    };
    let Some(boundary) = server::multipart_boundary(content_type) else {
        return Response::json(
            400,
            r#"{"error":"expected multipart/form-data with a boundary"}"#,
        );
    };
    let fields = server::parse_multipart(&req.body, &boundary);
    let Some(file_field) = fields.iter().find(|f| f.name == "file") else {
        return Response::json(400, r#"{"error":"missing required 'file' field"}"#);
    };

    let wav_data = match wav::read_wav(&mut std::io::Cursor::new(&file_field.data)) {
        Ok(w) if w.sample_rate as usize == audio::SAMPLE_RATE => w,
        parsed => {
            // Either not a WAV at all, or a WAV at the wrong sample rate --
            // both need ffmpeg to fix. Without --convert, tell the caller
            // exactly what was wrong rather than just "invalid".
            if !state.convert {
                return match parsed {
                    Ok(w) => Response::json(
                        400,
                        format!(
                            r#"{{"error":"audio is {} Hz; must be 16 kHz (or start the server with --convert)"}}"#,
                            w.sample_rate
                        ),
                    ),
                    Err(e) => Response::json(
                        400,
                        format!(
                            r#"{{"error":"invalid wav file: {e} (or start the server with --convert for non-WAV uploads)"}}"#
                        ),
                    ),
                };
            }
            match server::convert_with_ffmpeg(&file_field.data, &state.tmp_dir) {
                Ok(converted) => match wav::read_wav(&mut std::io::Cursor::new(&converted)) {
                    Ok(w) => w,
                    Err(e) => {
                        return Response::json(
                            500,
                            format!(r#"{{"error":"ffmpeg produced an unreadable wav: {e}"}}"#),
                        );
                    }
                },
                Err(e) => {
                    return Response::json(
                        400,
                        format!(r#"{{"error":"ffmpeg conversion failed: {e}"}}"#),
                    );
                }
            }
        }
    };

    let text_fields: HashMap<String, String> = fields
        .iter()
        .filter(|f| f.name != "file")
        .map(|f| {
            (
                f.name.clone(),
                String::from_utf8_lossy(&f.data).into_owned(),
            )
        })
        .collect();
    let inference_req = server::parse_inference_fields(&text_fields);

    let offset_samples = ((inference_req.offset_ms as usize * audio::SAMPLE_RATE) / 1000)
        .min(wav_data.samples.len());
    let window: &[f32] = if inference_req.duration_ms > 0 {
        let end = offset_samples + (inference_req.duration_ms as usize * audio::SAMPLE_RATE) / 1000;
        &wav_data.samples[offset_samples..end.min(wav_data.samples.len())]
    } else {
        &wav_data.samples[offset_samples..]
    };

    let model_guard = state.model.lock().unwrap();
    let Some(model) = model_guard.as_ref() else {
        return Response::json(503, r#"{"error":"no model loaded"}"#);
    };
    let transcript = transcribe::transcribe(model, window, &inference_req.opts);
    drop(model_guard);

    match server::format_transcript(&inference_req.response_format, &transcript) {
        Ok((content_type, body)) => Response::new(200).with_body(content_type, body),
        Err(e) => Response::json(500, format!(r#"{{"error":"{e}"}}"#)),
    }
}
