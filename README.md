# rusty-whisper

A pure-Rust, zero-dependency port of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI Whisper
speech recognition without a C/C++ toolchain.

**Status: it transcribes**, with timestamps, arbitrary-length audio
(30 s windows with seek-to-last-timestamp), conditioning on past text, and
temperature fallback. whisper.cpp's jfk.wav sample with `ggml-tiny.en.bin`
produces the canonical transcript and segment times, identical to
whisper.cpp's output. Still to come: beam search, quantized models, and
fast kernels (currently ~1-2x realtime for tiny on CPU). See
[PLAN.md](PLAN.md) for the roadmap.

## Try it

```sh
# Transcribe (16 kHz mono 16-bit PCM WAV, any length)
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
- `src/transcribe.rs` — chunking, timestamp rules, temperature fallback
  (mirror of `whisper_full`)
- `src/tensor.rs` — naive matmul / conv1d / layernorm / gelu / softmax
- `src/tokenizer.rs` — token decoding + Whisper special-token layout
- `src/wav.rs` — minimal 16-bit PCM WAV reader
