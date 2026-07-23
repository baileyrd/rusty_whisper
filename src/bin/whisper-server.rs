//! `whisper-server`: HTTP core for rusty-whisper — mirrors whisper.cpp's
//! `examples/server/server.cpp` (routing, `GET /health`, `GET /` static
//! serving, CORS). `POST /inference` (issue #52) and `POST /load` hot
//! model-swap (issue #53) aren't implemented yet — this binary is the
//! HTTP core they'll build on.

use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rusty_whisper::http::{self, Response};
use rusty_whisper::model;

fn main() -> ExitCode {
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut public_dir: Option<PathBuf> = None;
    let mut model_path: Option<String> = None;

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
            "--help" | "-h" => {
                eprintln!("whisper-server: HTTP core (GET /health, GET /, CORS)");
                eprintln!("  --host HOST     bind address (default 127.0.0.1)");
                eprintln!("  --port PORT     bind port (default 8080)");
                eprintln!(
                    "  --public DIR    serve static files from DIR instead of the built-in page"
                );
                eprintln!("  --model, -m PATH  load a model at startup (GET /health reports 503 until loaded)");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other} (see --help)");
                return ExitCode::FAILURE;
            }
        }
    }

    let ready = Arc::new(AtomicBool::new(model_path.is_none()));
    if let Some(path) = model_path {
        let ready = ready.clone();
        std::thread::spawn(move || {
            match std::fs::File::open(&path)
                .and_then(|f| model::load_model(&mut std::io::BufReader::new(f)))
            {
                Ok(_) => ready.store(true, Ordering::Release),
                Err(e) => eprintln!("failed to load model {path}: {e}"),
            }
        });
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
        let ready = ready.clone();
        let public_dir = public_dir.clone();
        std::thread::spawn(move || handle_connection(stream, public_dir, &ready));
    }
    ExitCode::SUCCESS
}

fn handle_connection(mut stream: TcpStream, public_dir: Option<PathBuf>, ready: &AtomicBool) {
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
    let resp = http::route(&req, public_dir.as_deref(), ready);
    let _ = resp.write_to(&mut stream);
}
