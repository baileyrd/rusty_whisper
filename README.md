# rusty-whisper

A pure-Rust, zero-dependency port of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI Whisper
speech recognition without a C/C++ toolchain.

**Status: it transcribes.** The full pipeline — log-mel front-end, ggml
`.bin` loader (F32/F16), encoder, KV-cached decoder, greedy sampling —
works end to end: whisper.cpp's jfk.wav sample with `ggml-tiny.en.bin`
produces the canonical transcript, identical to whisper.cpp's output.
Still early: single 30 s window, greedy only, no timestamps, naive kernels
(~9 s per window for tiny on CPU). See [PLAN.md](PLAN.md) for the roadmap.

## Try it

```sh
# Transcribe (16 kHz mono 16-bit PCM WAV, first 30 s)
cargo run --release -- --model ggml-tiny.en.bin --audio speech.wav

# Inspect a whisper.cpp model file
cargo run --release -- --model ggml-tiny.en.bin

cargo test
```

Model files are the standard whisper.cpp ones, e.g. from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).

## Layout

- `src/audio.rs` — Hann window, mixed-radix FFT, Slaney mel filterbank,
  log-mel normalization (mirror of `log_mel_spectrogram`)
- `src/model.rs` — ggml `.bin` parser: hparams, embedded mel filters, vocab,
  tensors (F32/F16)
- `src/encoder.rs` — conv stem + transformer stack; multi-head attention
  (causal + incremental variants)
- `src/decoder.rs` — KV-cached decoder, cross-attention, greedy sampling
- `src/tensor.rs` — naive matmul / conv1d / layernorm / gelu / softmax
- `src/tokenizer.rs` — token decoding + Whisper special-token layout
- `src/wav.rs` — minimal 16-bit PCM WAV reader
