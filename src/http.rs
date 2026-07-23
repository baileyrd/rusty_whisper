//! Minimal hand-rolled HTTP/1.1 core for `whisper-server` — whisper.cpp's
//! `examples/server/server.cpp` equivalent: basic request routing, `GET
//! /health`, `GET /` (static file serving), and unconditional CORS headers
//! + `OPTIONS` preflight handling.
//!
//! No HTTP crate dependency (this project is zero-dependency): request
//! parsing and response serialization are hand-rolled over
//! `std::io::Read`/`Write`, deliberately supporting only what a local
//! transcription server actually needs — one request per connection, no
//! keep-alive, no chunked transfer-encoding, `Content-Length` bodies only.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};

#[derive(Debug, PartialEq)]
pub struct Request {
    pub method: String,
    /// Path only — any `?query` is split off into [`Request::query`].
    pub path: String,
    pub query: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    /// Case-insensitive header lookup (HTTP header names aren't
    /// case-sensitive; callers shouldn't have to remember this crate
    /// stores them lowercased).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|s| s.as_str())
    }
}

/// Read and parse one request from `r`: the request line, headers up to
/// the blank line, then a `Content-Length`-sized body if present (chunked
/// transfer-encoding isn't supported — this server only ever needs to
/// read requests small clients send in one shot).
pub fn parse_request(r: &mut impl Read) -> io::Result<Request> {
    let mut reader = BufReader::new(r);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let line = line.trim_end();
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty request line"))?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request target"))?;
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.to_string(), String::new()),
    };

    let mut headers = HashMap::new();
    loop {
        let mut hline = String::new();
        reader.read_line(&mut hline)?;
        let hline = hline.trim_end_matches(['\r', '\n']);
        if hline.is_empty() {
            break;
        }
        if let Some((k, v)) = hline.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let body = match headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(len) if len > 0 => {
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            buf
        }
        _ => Vec::new(),
    };

    Ok(Request {
        method,
        path,
        query,
        headers,
        body,
    })
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16) -> Self {
        Response {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.to_string(), value.into()));
        self
    }

    pub fn with_body(mut self, content_type: &str, body: Vec<u8>) -> Self {
        self.body = body;
        self.with_header("Content-Type", content_type)
    }

    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Response::new(status).with_body("application/json", body.into().into_bytes())
    }

    /// Every response gets these (matches whisper.cpp's server, which
    /// sends permissive CORS headers unconditionally rather than
    /// reflecting a specific origin). Public so callers building their own
    /// routes outside [`route`] (e.g. `POST /inference`) apply the same
    /// headers rather than reimplementing them.
    pub fn with_cors(self) -> Self {
        self.with_header("Access-Control-Allow-Origin", "*")
            .with_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            .with_header("Access-Control-Allow-Headers", "*")
    }

    fn reason(status: u16) -> &'static str {
        match status {
            200 => "OK",
            202 => "Accepted",
            204 => "No Content",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            503 => "Service Unavailable",
            _ => "Internal Server Error",
        }
    }

    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write!(
            w,
            "HTTP/1.1 {} {}\r\n",
            self.status,
            Self::reason(self.status)
        )?;
        write!(w, "Content-Length: {}\r\n", self.body.len())?;
        for (k, v) in &self.headers {
            write!(w, "{k}: {v}\r\n")?;
        }
        write!(w, "\r\n")?;
        w.write_all(&self.body)?;
        w.flush()
    }
}

/// Whether the server has finished starting up — `GET /health` reports
/// `503` while this is false, matching whisper.cpp's readiness probe
/// (which flips true once initial model load, or a `POST /load` swap,
/// completes).
pub trait Readiness {
    fn is_ready(&self) -> bool;
}

impl Readiness for std::sync::atomic::AtomicBool {
    fn is_ready(&self) -> bool {
        self.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Serves static files under `root`: `path` is resolved relative to it
/// (`/` maps to `index.html`), rejecting any path that would escape
/// `root` via `..` — the request path is attacker-controlled input.
/// Returns `None` (caller should 404) rather than an empty/error response
/// when nothing matches, so a caller can fall back to other routes.
pub fn serve_static(root: &std::path::Path, path: &str) -> Option<Response> {
    let rel = path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    if rel.split('/').any(|seg| seg == "..") {
        return None;
    }
    let full = root.join(rel);
    let bytes = std::fs::read(&full).ok()?;
    let content_type = match full.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css",
        Some("js") => "text/javascript",
        Some("json") => "application/json",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    };
    Some(Response::new(200).with_body(content_type, bytes))
}

/// A minimal built-in demo page, used when `--public` isn't given — just
/// enough to confirm the server is up and describe `/health` (`/inference`
/// lands with issue #52).
pub const BUILTIN_INDEX_HTML: &str = "<!doctype html>\n<html><head><title>rusty-whisper server</title></head>\n<body><h1>rusty-whisper server</h1><p>See <a href=\"/health\">/health</a>.</p></body></html>\n";

/// Route one request. `public_dir`: `--public` override, else the
/// built-in demo page serves `/`. `ready`: backs `GET /health`.
pub fn route(
    req: &Request,
    public_dir: Option<&std::path::Path>,
    ready: &impl Readiness,
) -> Response {
    if req.method == "OPTIONS" {
        return Response::new(204).with_cors();
    }
    if req.method != "GET" {
        return Response::json(405, r#"{"error":"method not allowed"}"#).with_cors();
    }
    if req.path == "/health" {
        return if ready.is_ready() {
            Response::json(200, r#"{"status":"ok"}"#)
        } else {
            Response::json(503, r#"{"status":"loading"}"#)
        }
        .with_cors();
    }
    if let Some(dir) = public_dir {
        return match serve_static(dir, &req.path) {
            Some(resp) => resp.with_cors(),
            None => Response::json(404, r#"{"error":"not found"}"#).with_cors(),
        };
    }
    if req.path == "/" {
        return Response::new(200)
            .with_body(
                "text/html; charset=utf-8",
                BUILTIN_INDEX_HTML.as_bytes().to_vec(),
            )
            .with_cors();
    }
    Response::json(404, r#"{"error":"not found"}"#).with_cors()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct AlwaysReady;
    impl Readiness for AlwaysReady {
        fn is_ready(&self) -> bool {
            true
        }
    }

    #[test]
    fn parse_request_reads_method_path_query_and_headers() {
        let raw = "GET /foo?bar=1 HTTP/1.1\r\nHost: localhost\r\nX-Test: yes\r\n\r\n";
        let req = parse_request(&mut Cursor::new(raw)).unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/foo");
        assert_eq!(req.query, "bar=1");
        assert_eq!(req.header("host"), Some("localhost"));
        assert_eq!(req.header("Host"), Some("localhost")); // case-insensitive
        assert_eq!(req.header("x-test"), Some("yes"));
        assert!(req.body.is_empty());
    }

    #[test]
    fn parse_request_reads_a_content_length_body() {
        let raw = "POST /inference HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
        let req = parse_request(&mut Cursor::new(raw)).unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.body, b"hello");
    }

    #[test]
    fn parse_request_no_query_string() {
        let raw = "GET / HTTP/1.1\r\n\r\n";
        let req = parse_request(&mut Cursor::new(raw)).unwrap();
        assert_eq!(req.path, "/");
        assert_eq!(req.query, "");
    }

    #[test]
    fn response_write_to_produces_a_well_formed_status_line_and_headers() {
        let resp = Response::json(200, r#"{"a":1}"#).with_header("X-Extra", "1");
        let mut out = Vec::new();
        resp.write_to(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Length: 7\r\n"));
        assert!(text.contains("Content-Type: application/json\r\n"));
        assert!(text.contains("X-Extra: 1\r\n"));
        assert!(text.ends_with(r#"{"a":1}"#));
    }

    #[test]
    fn response_reason_phrases_match_their_status_codes() {
        for (status, phrase) in [
            (200, "OK"),
            (202, "Accepted"),
            (204, "No Content"),
            (400, "Bad Request"),
            (404, "Not Found"),
            (405, "Method Not Allowed"),
            (503, "Service Unavailable"),
        ] {
            let mut out = Vec::new();
            Response::new(status).write_to(&mut out).unwrap();
            let text = String::from_utf8(out).unwrap();
            assert!(
                text.starts_with(&format!("HTTP/1.1 {status} {phrase}\r\n")),
                "status {status}: {text}"
            );
        }
    }

    #[test]
    fn route_health_reports_ok_when_ready_and_loading_when_not() {
        let req = Request {
            method: "GET".into(),
            path: "/health".into(),
            query: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
        };
        let ready_resp = route(&req, None, &AlwaysReady);
        assert_eq!(ready_resp.status, 200);

        let not_ready = AtomicBool::new(false);
        let loading_resp = route(&req, None, &not_ready);
        assert_eq!(loading_resp.status, 503);

        not_ready.store(true, Ordering::Release);
        let now_ready_resp = route(&req, None, &not_ready);
        assert_eq!(now_ready_resp.status, 200);
    }

    #[test]
    fn route_options_is_a_cors_preflight_no_body() {
        let req = Request {
            method: "OPTIONS".into(),
            path: "/anything".into(),
            query: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
        };
        let resp = route(&req, None, &AlwaysReady);
        assert_eq!(resp.status, 204);
        assert!(resp.body.is_empty());
        assert!(resp
            .headers
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "*"));
    }

    #[test]
    fn route_root_serves_the_builtin_page_without_a_public_dir() {
        let req = Request {
            method: "GET".into(),
            path: "/".into(),
            query: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
        };
        let resp = route(&req, None, &AlwaysReady);
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, BUILTIN_INDEX_HTML.as_bytes());
    }

    #[test]
    fn route_unknown_path_is_404() {
        let req = Request {
            method: "GET".into(),
            path: "/nope".into(),
            query: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
        };
        let resp = route(&req, None, &AlwaysReady);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn route_non_get_non_options_is_405() {
        let req = Request {
            method: "DELETE".into(),
            path: "/health".into(),
            query: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
        };
        let resp = route(&req, None, &AlwaysReady);
        assert_eq!(resp.status, 405);
    }

    #[test]
    fn serve_static_serves_index_and_rejects_path_traversal() {
        let dir = std::env::temp_dir().join(format!("rw_http_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), "<p>hi</p>").unwrap();
        std::fs::write(dir.join("style.css"), "body{}").unwrap();

        let idx = serve_static(&dir, "/").unwrap();
        assert_eq!(idx.body, b"<p>hi</p>");
        assert_eq!(
            idx.headers
                .iter()
                .find(|(k, _)| k == "Content-Type")
                .map(|(_, v)| v.as_str()),
            Some("text/html; charset=utf-8")
        );

        let css = serve_static(&dir, "/style.css").unwrap();
        assert_eq!(css.body, b"body{}");

        assert!(serve_static(&dir, "/../../etc/passwd").is_none());
        assert!(serve_static(&dir, "/missing.txt").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }
}
