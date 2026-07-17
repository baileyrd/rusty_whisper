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
- [ ] **8b. Nice-to-haves** — GGUF format, streaming input, wasm build,
  large-v3 (128 mel) support, translate task, language auto-detection.

## Validation strategy

Every phase is validated against reference output:
1. Unit tests with analytically known results (pure tones → mel bins, FFT vs
   naive DFT, layernorm statistics).
2. Once phases 4–5 land: run whisper.cpp on the same audio + model with
   `--print-tensors`-style dumps and diff intermediate activations
   (mel → encoder out → first decoder logits) within tolerance.
3. End-to-end: jfk.wav with tiny.en must produce the canonical transcript.
