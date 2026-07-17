# Release notes

Newest first. Versions are milestone markers over the porting history
(see [PLAN.md](PLAN.md)); dates are when the work merged to `main`.

---

## 0.7.0 — AVX2 dequantization

**2026-07-17**

The memory/speed trade-off from 0.4.0 mostly dissolves: quantized-in-
memory now decodes as fast as `--dense` on AVX2 machines, keeping the
2-3x memory win with no downside.

### 🚀 Performance

- Runtime-detected AVX2 kernels dequantize all five ggml block formats
  (SIMD nibble split, shuffle-based Q5 high-bit expansion). Decoder on
  tiny-q5_1: **27.9 → 16.3 ms/token**; full jfk.wav 5.48 → 4.49 s —
  level with `--dense` (4.62 s) at half the RAM

### 🔧 Under the hood

- The kernels deliberately use mul+add instead of FMA so their output is
  **bit-identical** to the safe scalar path — asserted by a test over
  random blocks of every dtype
- `unsafe` remains confined to two audited leaf modules (wasm FFI glue,
  SIMD kernels); the inference pipeline stays safe Rust
- README overhauled; this release-notes file added

---

## 0.6.0 — Whisper in a browser tab

**2026-07-17** · [PR #6](https://github.com/baileyrd/rusty_whisper/pull/6)

Speech recognition fully client-side: a ~190 KB wasm module, a single
HTML page, and nothing uploaded anywhere.

### ⭐ Features

- `demo/`: browser demo — pick any ggml model and any audio file your
  browser can play; audio is decoded and resampled to 16 kHz client-side
  via `decodeAudioData` + `OfflineAudioContext`
- Hand-rolled C-ABI FFI (`src/wasm.rs`) instead of wasm-bindgen, keeping
  the crate dependency-free; results returned as JSON

### 🔧 Under the hood

- The FFI module is the only `unsafe` in the crate, confined to pointer
  glue at the boundary — the inference path stays safe Rust
- Validated end to end in headless Chromium: tiny.en-q5_1 loads through
  the file picker and transcribes jfk.wav correctly in 22.5 s

---

## 0.5.1 — Batched beam logits

**2026-07-17** · [PR #5](https://github.com/baileyrd/rusty_whisper/pull/5)

### 🚀 Performance

- Beam search now projects logits for all surviving beams in one matmul:
  the tied token embedding — the decoder's largest matrix — is read (and
  for quantized weights, unpacked) once per step instead of once per
  beam. ~10% end-to-end at beam 5 on tiny; the gain grows with beam
  width and model size
- `Decoder::forward` split into `forward_hidden` + stateless
  `project_logits`, pinned equal to the combined path by test

---

## 0.5.0 — Streaming input and large-v3-turbo

**2026-07-17** · [PR #4](https://github.com/baileyrd/rusty_whisper/pull/4)

### ⭐ Features

- `transcribe::Stream`: incremental `feed()`/`finish()` API — windows
  decode as 30 s of audio accumulates, segments return as they finalize,
  and consumed samples are dropped (bounded memory)
- CLI `--audio -`: transcribe WAV from stdin, printing segments live
  (`ffmpeg ... -f wav - | rusty-whisper --audio -`)

### 🔧 Under the hood

- large-v3-turbo-q5_0 validated with **zero code changes**: 128 mel
  bands, 51866-token vocab, 587 tensors, 712 MB peak RSS (dense f32
  would need ~2.4 GB)
- File mode is a thin wrapper over the streaming path — outputs are
  byte-identical

---

## 0.4.0 — Quantized weights stay quantized

**2026-07-17** · [PR #3](https://github.com/baileyrd/rusty_whisper/pull/3)

Quantized model files no longer balloon to f32 at load.

### ⭐ Features

- 2-D weight matrices stay in their ggml blocks in memory; matmuls
  dequantize one weight row at a time into a scratch that feeds the
  vectorized f32 kernels. small.en-q5_1: **1087 MB → 372 MB** peak RSS
- `--dense` restores dequantize-at-load for maximum decode speed —
  the trade-off is explicit and user-selectable

### 🔧 Under the hood

- An int8 activation-quantization + integer-dot path was built and
  benchmarked first; in safe Rust it lost to dequant-on-the-fly f32
- Encoder reaches f32 parity (unpack amortizes over 1500 rows); the
  decoder pays ~1.5x for the per-token embedding unpack

---

## 0.3.0 — Wasm, and validation beyond tiny

**2026-07-17** · [PR #2](https://github.com/baileyrd/rusty_whisper/pull/2)

### ⭐ Features

- Builds for `wasm32-wasip1` and runs a full transcription inside a
  wasmtime sandbox (0.35x realtime, single-threaded)

### 🐛 Fixes

- The thread-pool helper spawned threads unconditionally — a panic on
  wasm; it now falls back to serial on single-core targets
- `target-cpu=native` rustflags scoped to host triples so
  cross-compiles work
- Stop the window loop when under 1 s of audio remains — decoding a
  0.4 s sliver produced a trailing `[BLANK_AUDIO]` segment

### 🔧 Under the hood

- base.en (1.8x realtime) and small.en-q5_1 (0.64x) validated with
  canonical transcripts

---

## 0.2.0 — Multilingual

**2026-07-17** · [PR #1](https://github.com/baileyrd/rusty_whisper/pull/1)

### ⭐ Features

- Language auto-detection (one decoder step on `[sot]`, softmax over the
  language tokens), run on the first window when no language is given
- `--language CODE` (full 100-language table) and `--translate`
  (X → English)

### 🐛 Fixes

- Timestamp rules: with fewer than two sampled tokens the penultimate
  must count as a timestamp (OpenAI's `len(tokens) < 2 or ...`), making
  the initial timestamp a segment opener. Getting it backwards forced a
  spurious second timestamp — invisible on English-only models (which
  re-emitted `0.00`) but shifting every multilingual segment by seconds

---

## 0.1.0 — The port itself

**2026-07-17** · initial history

From an empty repository to a working, fast transcriber, mirroring
whisper.cpp's architecture in dependency-free Rust.

### ⭐ Features

- Audio front-end: Hann window, mixed-radix FFT (validated against a
  naive DFT), Slaney mel filterbank, whisper.cpp's exact log-mel
  normalization
- ggml `.bin` model loader: hparams, embedded mel filters, vocab,
  F32/F16 and Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 tensors
- Encoder (conv stem + transformer stack) and KV-cached decoder with
  cross-attention; incremental decoding pinned equal to full forward
  passes by test
- Full `whisper_full` pipeline: OpenAI timestamp rules, 30 s chunking
  with seek-to-last-timestamp, conditioning on past text, temperature
  fallback gated on avg log-prob and compression ratio
- Beam search (beams share cross-attention K/V via `Arc`, fork their
  self-attention caches; EOT hypotheses finalized only from the global
  top candidates)

### 🚀 Performance

- 4x end to end over the naive baseline: `target-cpu=native`,
  SIMD-friendly dot kernels with per-lane accumulators (a scalar
  `s += a*b` chain can't auto-vectorize), 4-row register blocking,
  threading over rows or columns by shape, mel via one matmul,
  cross-attention K/V pre-split per head

### 🐛 Fixes

- Compression-ratio gate calibrated with zlib-like match/stream costs —
  without them, legitimately repetitive audio tripped the gate and
  escalated into sampling noise
- A conditioned window that decodes to nothing (prompt already contains
  the phrase) retries unconditioned instead of silently dropping speech

### 🔧 Under the hood

- End-to-end validation: jfk.wav + ggml-tiny.en.bin reproduces
  whisper.cpp's transcript and segmentation exactly
