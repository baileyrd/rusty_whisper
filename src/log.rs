//! Custom log sink — Rust-idiomatic equivalent of whisper.cpp's
//! `whisper_log_set`. A settable closure hook, not a dependency on the
//! `log` crate (this project is zero-dependency): default behavior mirrors
//! whisper.cpp's own default sink, writing messages to stderr.

use std::sync::{Mutex, OnceLock};

type LogFn = Box<dyn Fn(&str) + Send + Sync>;

fn default_sink(msg: &str) {
    eprintln!("{msg}");
}

static SINK: OnceLock<Mutex<LogFn>> = OnceLock::new();

fn sink() -> &'static Mutex<LogFn> {
    SINK.get_or_init(|| Mutex::new(Box::new(default_sink)))
}

/// Install a custom log sink, replacing the default (stderr) one —
/// `whisper_log_set`. Applies to every subsequent log call for the
/// lifetime of the process (or until [`reset_log_sink`] is called).
pub fn set_log_sink(f: impl Fn(&str) + Send + Sync + 'static) {
    *sink().lock().unwrap() = Box::new(f);
}

/// Restore the default stderr sink.
pub fn reset_log_sink() {
    *sink().lock().unwrap() = Box::new(default_sink);
}

/// Route a message through the currently installed sink. Library code
/// calls this instead of `eprintln!` directly so callers can intercept,
/// silence, or redirect it via [`set_log_sink`].
pub(crate) fn log(msg: impl AsRef<str>) {
    (sink().lock().unwrap())(msg.as_ref());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    #[test]
    fn set_log_sink_intercepts_and_reset_restores_default() {
        // One test covering set -> log -> assert -> reset: the sink is a
        // process-global static, so keeping the whole sequence in a single
        // test avoids racing against other tests that might touch it.
        let captured = Arc::new(StdMutex::new(Vec::<String>::new()));
        let captured_clone = captured.clone();
        set_log_sink(move |msg| captured_clone.lock().unwrap().push(msg.to_string()));

        log("hello");
        log(format!("world {}", 42));

        // Other tests running concurrently may log through this
        // process-global sink too (e.g. model loading) while it's
        // installed here, so check our own messages appear in order
        // rather than asserting an exact, possibly-interleaved vector.
        let got = captured.lock().unwrap();
        let hello_pos = got.iter().position(|m| m == "hello");
        let world_pos = got.iter().position(|m| m == "world 42");
        assert!(hello_pos.is_some() && world_pos.is_some());
        assert!(hello_pos.unwrap() < world_pos.unwrap());
        drop(got);

        reset_log_sink();
        // No assertion on stderr output itself (nothing to capture without
        // process-level redirection); this just verifies reset doesn't panic
        // and a subsequent log() call runs through the default sink path.
        log("back to stderr");
    }
}
