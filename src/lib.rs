//! rusty-whisper: a pure-Rust port of whisper.cpp.
//!
//! Status: audio front-end, model loading, tensor core, and vocab are
//! implemented; encoder/decoder forward passes are in progress — see PLAN.md.

pub mod audio;
pub mod decoder;
pub mod encoder;
pub mod model;
pub mod quant;
pub mod tensor;
pub mod tokenizer;
pub mod transcribe;
pub mod wav;
