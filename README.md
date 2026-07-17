# rusty-whisper

A pure-Rust, zero-dependency port of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI Whisper
speech recognition without a C/C++ toolchain.

**Status: early.** The audio front-end (log-mel spectrogram), ggml `.bin`
model loader (F32/F16), tensor core, WAV reader, and vocabulary handling are
implemented and tested. The encoder/decoder forward passes and sampling loop
are in progress — see [PLAN.md](PLAN.md) for the roadmap.

## Try it

```sh
# Inspect a whisper.cpp model file
cargo run --release -- --model ggml-tiny.en.bin

# Run audio through the mel front-end (16 kHz mono 16-bit PCM WAV)
cargo run --release -- --model ggml-tiny.en.bin --audio speech.wav

cargo test
```

Model files are the standard whisper.cpp ones, e.g. from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).

## Layout

- `src/audio.rs` — Hann window, mixed-radix FFT, Slaney mel filterbank,
  log-mel normalization (mirror of `log_mel_spectrogram`)
- `src/model.rs` — ggml `.bin` parser: hparams, embedded mel filters, vocab,
  tensors (F32/F16)
- `src/tensor.rs` — naive matmul / conv1d / layernorm / gelu / softmax
- `src/tokenizer.rs` — token decoding + Whisper special-token layout
- `src/wav.rs` — minimal 16-bit PCM WAV reader
