# Browser demo

Whisper running entirely client-side: the pure-Rust port compiled to
WebAssembly (~190 KB), no wasm-bindgen, no JS dependencies. Audio is
decoded and resampled by the browser (`decodeAudioData` +
`OfflineAudioContext`), so any format the browser plays works as input.

## Build & run

```sh
rustup target add wasm32-unknown-unknown
# Build the cdylib on demand (see the note in ../Cargo.toml).
cargo rustc --lib --release --target wasm32-unknown-unknown --crate-type cdylib
cp ../target/wasm32-unknown-unknown/release/rusty_whisper.wasm .
python3 -m http.server -d .   # any static server works
```

Open http://localhost:8000, pick a ggml model
(e.g. [`ggml-tiny.en-q5_1.bin`](https://huggingface.co/ggerganov/whisper.cpp/tree/main),
32 MB) and an audio file.

Wasm runs single-threaded without FMA, so expect roughly 2x audio
duration for tiny on a laptop. Quantized models are the right choice
here — weights stay quantized in memory.

## FFI protocol

`src/wasm.rs` exports a tiny C ABI (the only `unsafe` in the crate,
confined to pointer glue): `wasm_alloc`/`wasm_free`,
`wasm_load_model(ptr, len)`, `wasm_transcribe(ptr, n_samples, beam)`,
and `wasm_result_ptr`/`wasm_result_len` returning JSON
`{language, segments: [{t0, t1, text}]}`.
