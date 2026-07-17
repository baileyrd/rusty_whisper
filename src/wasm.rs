//! Browser FFI: hand-rolled C-ABI exports for wasm32-unknown-unknown —
//! no wasm-bindgen, keeping the crate dependency-free. See `demo/` for the
//! JS side.
//!
//! Protocol: JS calls [`wasm_alloc`] and copies bytes into linear memory,
//! then [`wasm_load_model`] / [`wasm_transcribe`]. Results come back as a
//! JSON string read from [`wasm_result_ptr`] / [`wasm_result_len`].
//!
//! This module is the only place in the crate with `unsafe`: it is pure
//! pointer glue at the FFI boundary (reconstructing slices/Vecs that JS
//! allocated through us); the inference path stays safe Rust. Wasm is
//! single-threaded, so state lives in a thread_local.

use std::cell::RefCell;

use crate::model::{load_model, Model};
use crate::transcribe::{transcribe, Options};

thread_local! {
    static MODEL: RefCell<Option<Model>> = const { RefCell::new(None) };
    static RESULT: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Allocate `len` bytes in wasm memory for JS to fill.
#[no_mangle]
pub extern "C" fn wasm_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free a buffer from [`wasm_alloc`] (same `len`).
///
/// # Safety
/// `ptr`/`len` must come from a prior `wasm_alloc(len)` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn wasm_free(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

/// Parse a ggml model from `len` bytes at `ptr`. Returns 0 on success.
///
/// # Safety
/// `ptr`/`len` must describe a valid, filled `wasm_alloc` buffer.
#[no_mangle]
pub unsafe extern "C" fn wasm_load_model(ptr: *const u8, len: usize) -> i32 {
    let bytes = std::slice::from_raw_parts(ptr, len);
    match load_model(&mut std::io::Cursor::new(bytes)) {
        Ok(m) => {
            MODEL.with(|s| *s.borrow_mut() = Some(m));
            0
        }
        Err(e) => {
            RESULT.with(|s| *s.borrow_mut() = format!("{{\"error\":{}}}", json_string(&e.to_string())));
            -1
        }
    }
}

/// Transcribe `n` f32 samples (16 kHz mono, [-1, 1]) at `ptr`.
/// Returns 0 on success; fetch JSON via result_ptr/result_len.
///
/// # Safety
/// `ptr` must point at `n` valid f32s in wasm memory; a model must be
/// loaded.
#[no_mangle]
pub unsafe extern "C" fn wasm_transcribe(ptr: *const f32, n: usize, beam_size: usize) -> i32 {
    let samples = std::slice::from_raw_parts(ptr, n);
    MODEL.with(|s| {
        let guard = s.borrow();
        let Some(model) = guard.as_ref() else {
            RESULT.with(|r| *r.borrow_mut() = "{\"error\":\"no model loaded\"}".to_string());
            return -1;
        };
        let opts = Options { beam_size: beam_size.max(1), ..Default::default() };
        let t = transcribe(model, samples, &opts);
        let mut json = String::from("{\"language\":");
        json.push_str(&json_string(&t.language));
        json.push_str(",\"segments\":[");
        for (i, seg) in t.segments.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"t0\":{:.2},\"t1\":{:.2},\"text\":{}}}",
                seg.t0,
                seg.t1,
                json_string(&seg.text)
            ));
        }
        json.push_str("]}");
        RESULT.with(|r| *r.borrow_mut() = json);
        0
    })
}

#[no_mangle]
pub extern "C" fn wasm_result_ptr() -> *const u8 {
    RESULT.with(|s| s.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_result_len() -> usize {
    RESULT.with(|s| s.borrow().len())
}

/// Minimal JSON string encoder (quotes, escapes).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::json_string;

    #[test]
    fn json_escaping() {
        assert_eq!(json_string("hi"), "\"hi\"");
        assert_eq!(json_string("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
        assert_eq!(json_string("\u{1}"), "\"\\u0001\"");
    }
}
