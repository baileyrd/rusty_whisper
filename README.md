# rusty-whisper

A pure-Rust, zero-dependency port of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI Whisper
speech recognition without a C/C++ toolchain.

**Status: it transcribes**, with timestamps, arbitrary-length audio
(30 s windows with seek-to-last-timestamp), conditioning on past text, and
temperature fallback. whisper.cpp's jfk.wav sample with `ggml-tiny.en.bin`
produces the canonical transcript and segment times, identical to
whisper.cpp's output. Beam search (default beam 5, `--beam 1` for greedy), quantized models
(Q4/Q5/Q8 ggml formats), and multilingual models with language
auto-detection (`--language CODE` to force, `--translate` for
X -> English) are supported. Runs ~4-7x realtime for tiny on a
4-core CPU (the build uses `target-cpu=native`; see `.cargo/config.toml`).
See [PLAN.md](PLAN.md) for the roadmap.

## Try it

```sh
# Transcribe (16 kHz mono 16-bit PCM WAV, any length)
cargo run --release -- --model ggml-tiny.en-q5_1.bin --audio speech.wav

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
