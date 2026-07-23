//! Minimal HTTP/1.1 client for OpenAI-compatible chat-completions
//! endpoints (`POST /v1/chat/completions`) — `whisper-talk-llama`'s LLM
//! backend, talking to a separately-run `rusty_llama --serve` process (or
//! any other server speaking the same wire format). Hand-rolled over
//! `std::net::TcpStream`, matching this crate's zero-dependency stance and
//! `src/http.rs`'s server-side hand-rolled parsing: one request per
//! connection, no keep-alive, `Content-Length` framing only.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: "system".to_string(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: "user".to_string(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Sampling knobs forwarded to the server's OpenAI-compatible
/// `SamplingParams` (see `rusty_llama::server`) — all optional, included in
/// the request body only when set.
#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
}

pub struct ChatClient {
    host: String,
    port: u16,
    model: String,
    timeout: Duration,
}

impl ChatClient {
    pub fn new(host: impl Into<String>, port: u16, model: impl Into<String>) -> Self {
        ChatClient {
            host: host.into(),
            port,
            model: model.into(),
            timeout: Duration::from_secs(120),
        }
    }

    /// Sends `messages` to `POST /v1/chat/completions` (non-streaming) and
    /// returns the assistant's reply text (`choices[0].message.content`).
    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String, String> {
        let addr = format!("{}:{}", self.host, self.port);
        let body = build_request_body(&self.model, messages, opts);

        let mut stream =
            TcpStream::connect(&addr).map_err(|e| format!("connect to {addr}: {e}"))?;
        stream.set_read_timeout(Some(self.timeout)).ok();
        stream.set_write_timeout(Some(self.timeout)).ok();

        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: {addr}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {body}",
            body.len(),
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| format!("write to {addr}: {e}"))?;

        let (status, resp_body) =
            read_response(&mut stream).map_err(|e| format!("read from {addr}: {e}"))?;
        if !(200..300).contains(&status) {
            return Err(format!(
                "{addr} returned HTTP {status}: {}",
                extract_error(&resp_body)
            ));
        }
        extract_reply(&resp_body)
    }
}

fn build_request_body(model: &str, messages: &[Message], opts: &ChatOptions) -> String {
    let msgs: Vec<String> = messages
        .iter()
        .map(|m| {
            format!(
                "{{\"role\":{},\"content\":{}}}",
                crate::json::escape(&m.role),
                crate::json::escape(&m.content)
            )
        })
        .collect();
    let mut fields = vec![
        format!("\"model\":{}", crate::json::escape(model)),
        format!("\"messages\":[{}]", msgs.join(",")),
        "\"stream\":false".to_string(),
    ];
    if let Some(v) = opts.temperature {
        fields.push(format!("\"temperature\":{v}"));
    }
    if let Some(v) = opts.top_p {
        fields.push(format!("\"top_p\":{v}"));
    }
    if let Some(v) = opts.top_k {
        fields.push(format!("\"top_k\":{v}"));
    }
    if let Some(v) = opts.min_p {
        fields.push(format!("\"min_p\":{v}"));
    }
    if let Some(v) = opts.max_tokens {
        fields.push(format!("\"max_tokens\":{v}"));
    }
    if let Some(v) = opts.seed {
        fields.push(format!("\"seed\":{v}"));
    }
    format!("{{{}}}", fields.join(","))
}

/// Reads an HTTP/1.1 response's status line, headers (for
/// `Content-Length`), then exactly that many body bytes.
fn read_response(stream: &mut TcpStream) -> std::io::Result<(u16, String)> {
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed status line")
        })?;

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok((status, String::from_utf8_lossy(&body).into_owned()))
}

fn extract_reply(body: &str) -> Result<String, String> {
    let value = crate::json::parse(body).map_err(|e| format!("invalid JSON response: {e}"))?;
    value
        .get("choices")
        .and_then(|c| c.index(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(crate::json::Value::as_str)
        .map(|s| s.to_string())
        .ok_or_else(|| format!("unexpected response shape: {body}"))
}

fn extract_error(body: &str) -> String {
    crate::json::parse(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.get("message")).cloned())
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| body.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_body_shape() {
        let body = build_request_body(
            "tinyllama",
            &[Message::system("be terse"), Message::user("hi")],
            &ChatOptions::default(),
        );
        let v = crate::json::parse(&body).unwrap();
        assert_eq!(
            v.get("model").and_then(crate::json::Value::as_str),
            Some("tinyllama")
        );
        assert_eq!(v.get("stream"), Some(&crate::json::Value::Bool(false)));
        let msgs = v.get("messages").unwrap();
        assert_eq!(
            msgs.index(0)
                .unwrap()
                .get("role")
                .and_then(crate::json::Value::as_str),
            Some("system")
        );
        assert_eq!(
            msgs.index(1)
                .unwrap()
                .get("content")
                .and_then(crate::json::Value::as_str),
            Some("hi")
        );
    }

    #[test]
    fn build_request_body_escapes_content() {
        let body = build_request_body(
            "m",
            &[Message::user("say \"hi\"\nplease")],
            &ChatOptions::default(),
        );
        let v = crate::json::parse(&body).unwrap();
        assert_eq!(
            v.get("messages")
                .unwrap()
                .index(0)
                .unwrap()
                .get("content")
                .and_then(crate::json::Value::as_str),
            Some("say \"hi\"\nplease")
        );
    }

    #[test]
    fn build_request_body_includes_set_sampling_params() {
        let opts = ChatOptions {
            temperature: Some(0.3),
            top_k: Some(5),
            max_tokens: Some(128),
            ..Default::default()
        };
        let body = build_request_body("m", &[Message::user("hi")], &opts);
        let v = crate::json::parse(&body).unwrap();
        assert_eq!(v.get("temperature"), Some(&crate::json::Value::Number(0.3)));
        assert_eq!(v.get("top_k"), Some(&crate::json::Value::Number(5.0)));
        assert_eq!(
            v.get("max_tokens"),
            Some(&crate::json::Value::Number(128.0))
        );
        assert_eq!(v.get("top_p"), None);
    }

    #[test]
    fn extract_reply_reads_nested_content() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"hello there"}}]}"#;
        assert_eq!(extract_reply(body), Ok("hello there".to_string()));
    }

    #[test]
    fn extract_reply_errors_on_unexpected_shape() {
        assert!(extract_reply(r#"{"foo": "bar"}"#).is_err());
        assert!(extract_reply("not json").is_err());
    }

    #[test]
    fn extract_error_reads_openai_style_error() {
        let body = r#"{"error":{"message":"model not loaded","type":"invalid_request_error"}}"#;
        assert_eq!(extract_error(body), "model not loaded");
    }

    #[test]
    fn extract_error_falls_back_to_raw_body() {
        assert_eq!(
            extract_error("internal server error"),
            "internal server error"
        );
    }

    #[test]
    fn chat_errors_on_connection_refused() {
        // Nothing is listening on this port (a genuine integration test of
        // the connect-failure path, not a mock).
        let client = ChatClient::new("127.0.0.1", 1, "m");
        let err = client
            .chat(&[Message::user("hi")], &ChatOptions::default())
            .unwrap_err();
        assert!(err.contains("connect to"));
    }
}
