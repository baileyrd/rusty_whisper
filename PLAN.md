# Porting whisper.cpp to Rust — plan

Goal: a dependency-free, pure-Rust reimplementation of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) that loads the same
ggml `.bin` model files and produces the same transcriptions.

## Why a port (and not bindings)

- `whisper-rs` already wraps whisper.cpp over FFI. If you just need Whisper
  from Rust today, use that (or Candle's whisper example).
- A pure port gives: no C/C++ toolchain, trivial cross-compilation (including
  wasm), memory safety through the whole inference path, and a codebase we
  fully control.
- Trade-off: whisper.cpp's hand-tuned SIMD/GPU backends took years; we start
  with correct-but-naive kernels and optimize incrementally.

## Architecture (mirrors whisper.cpp)

```
audio (PCM 16 kHz) ──► log-mel spectrogram ──► encoder (conv + transformer)
                                                      │ cross-attention
model.bin ──► loader (hparams, mel filters,           ▼
              vocab, tensors)              decoder (transformer, autoregressive)
                                                      │
                                           sampling (greedy / beam) ──► tokens ──► text
```

Module map:

| whisper.cpp                        | rusty-whisper       |
|------------------------------------|---------------------|
| `log_mel_spectrogram` + custom FFT | `src/audio.rs`      |
| ggml tensors + ops                 | `src/tensor.rs`     |
| model loading (`whisper_model_load`) | `src/model.rs`    |
| vocab / BPE                        | `src/tokenizer.rs`  |
| `whisper_encode` / `whisper_decode`| `src/encoder.rs`, `src/decoder.rs` |
| `whisper_full` (chunking, sampling)| `src/transcribe.rs` |
| `main.cpp` CLI                     | `src/main.rs`       |

## Phases

- [x] **0. Scaffold + plan** (this commit)
- [x] **1. Audio front-end** — Hann window, mixed-radix FFT (n=400), Slaney
  mel filterbank, log-mel normalization identical to whisper.cpp. Testable
  without model weights.
- [x] **2. Model loader** — parse ggml `.bin` (magic `ggml`, 11 i32 hparams,
  embedded mel filters, vocab, tensor records). F32/F16 first; quantized
  formats (Q5_0/Q8_0…) later.
- [x] **3. Tensor core (naive)** — matmul, conv1d, layernorm, gelu, softmax,
  embedding lookup. Correctness first; parallelism/SIMD in phase 7.
- [x] **4. Encoder forward** — 2× conv1d + GELU, sinusoidal positions,
  N transformer blocks (pre-LN, self-attn, MLP), final LN. Validated to run
  against real ggml-tiny.en.bin weights with sane output statistics.
- [x] **5. Decoder + greedy sampling** — token/position embeddings, masked
  self-attn with KV cache, cross-attn to encoder output, greedy loop
  (timestamps disabled via `<|notimestamps|>` for now). End-to-end
  validated: jfk.wav + ggml-tiny.en.bin produces the canonical transcript,
  identical to whisper.cpp.
- [x] **6. Full pipeline** — 30 s chunking with seek-to-last-timestamp,
  OpenAI timestamp rules (initial-timestamp forcing, pairing, monotonicity,
  timestamp-vs-text probability), conditioning on past text with an
  unconditioned retry for collapsed windows, temperature fallback gated on
  compression ratio + avg log-prob. Validated on multi-window repetitive
  audio. Beam search (default beam 5 at temperature 0, like whisper.cpp):
  beams share the window's cross-attention K/V via Arc and fork their
  self-attention caches; EOT hypotheses are finalized only when they rank
  in the global top beam_size, and selection is by average log-prob over
  complete hypotheses.
- [x] **7. Performance (first pass)** — `target-cpu=native`, vectorizable
  matmul kernels (per-lane accumulator arrays — a scalar `s += a*b` chain
  can't be auto-vectorized without float reassociation), 4-row register
  blocking, threading over rows (or B-columns for the skinny logits
  matmul), parallel conv1d/softmax/gelu/FFT, mel via one matmul,
  cross-attention K/V pre-split per head. 9.25 s -> 2.3 s for jfk.wav on a
  4-core AVX-512 box (4.7x realtime; 6.7x on longer audio). Remaining
  ideas: L1 blocking of B, f16 compute, per-head parallel attention,
  quantized matmul (with phase 8).
- [x] **8a. Quantized models** — Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 block formats,
  dequantized to f32 at load (quantized compute kernels remain a possible
  optimization). Validated with ggml-tiny.en-q5_1.bin (32 MB vs 78 MB).
- [x] **8b. Multilingual** — language auto-detection (one decoder step on
  `[sot]`, softmax over the language tokens), `--language CODE`, and the
  translate task. Validated with multilingual ggml-tiny.bin (auto-detects
  `en` on jfk.wav and reproduces whisper.cpp's canonical multilingual
  output; forcing `--language de` hallucinates German exactly like
  upstream). Foreign-language audio validation still outstanding — this
  environment has no way to produce a 16 kHz foreign-speech WAV.
- [x] **8c. Wasm + model-size validation** — builds for wasm32-wasip1
  (thread pool falls back to serial; `target-cpu=native` scoped to host
  triples) and runs end to end under wasmtime: tiny.en-q5_1 transcribes
  jfk.wav correctly in the sandbox at 0.35x realtime (single-threaded, no
  FMA — segment boundaries drift a few frames vs native). base.en and
  small.en validated natively with canonical transcripts (1.8x / 0.64x
  realtime).
- [x] **8d. Quantized-in-memory weights** — 2-D matrices stay in their
  ggml blocks; matmuls dequantize one B row at a time into an f32 scratch
  and reuse the fast dot kernels. Encoder hits f32 parity (unpack
  amortizes over 1500 rows); the decoder pays ~1.5x (the 51k-row token
  embedding re-unpacks every token for logits — safe-Rust unpack can't
  match ggml's SIMD shuffles). `--dense` restores the dequantize-at-load
  behavior. small.en-q5_1: 372 MB @ 27 s vs 1094 MB @ 17 s, user's
  choice. Future: intrinsics-based unpack, batched-beam logits
  projection.
- [x] **8e. large-v3-turbo + streaming input** — large-v3-turbo-q5_0
  validated with zero code changes (128 mels, 51866-token vocab, 587
  tensors; 712 MB RSS quantized vs ~2.4 GB dense; 0.12x realtime on 4
  cores). Streaming: `transcribe::Stream` (feed/finish, bounded buffer)
  with `--audio -` reading WAV from stdin and emitting segments as each
  30 s window fills — validated by drip-feeding 1 s chunks.
- [ ] **8f. Nice-to-haves** — GGUF format, browser demo
  (wasm32-unknown-unknown + JS glue), intrinsics-based unpack, batched
  beam logits.

## Validation strategy

Every phase is validated against reference output:
1. Unit tests with analytically known results (pure tones → mel bins, FFT vs
   naive DFT, layernorm statistics).
2. Once phases 4–5 land: run whisper.cpp on the same audio + model with
   `--print-tensors`-style dumps and diff intermediate activations
   (mel → encoder out → first decoder logits) within tolerance.
3. End-to-end: jfk.wav with tiny.en must produce the canonical transcript.
