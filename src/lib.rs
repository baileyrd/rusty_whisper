//! rusty-whisper: a pure-Rust port of whisper.cpp.
//!
//! Status: audio front-end, model loading, tensor core, and vocab are
//! implemented; encoder/decoder forward passes are in progress — see PLAN.md.

pub mod audio;
pub mod decoder;
pub mod dtw;
pub mod encoder;
#[cfg(feature = "gguf")]
pub mod gguf;
pub mod grammar;
pub mod log;
pub mod model;
pub mod output;
pub mod quant;
pub mod tensor;
pub mod tokenizer;
pub mod transcribe;
pub mod vad;
pub mod wasm;
pub mod wav;
