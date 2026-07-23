# Release notes

Newest first. Versions are milestone markers over the porting history
(see [PLAN.md](PLAN.md)); dates are when the work merged to `main`.

---

## Unreleased

### ⭐ Features

- `--output-txt`/`-otxt`, `--output-vtt`/`-ovtt`, `--output-srt`/`-osrt`,
  `--output-csv`/`-ocsv`, `--output-json`/`-oj` write transcript files
  alongside the stdout output, matching whisper.cpp's output formats;
  `--output-file`/`-of` sets the base path (default: the audio path minus
  its extension)
- `--max-len`/`-ml` caps segment length in characters, splitting long
  segments into several with `t0`/`t1` interpolated proportionally to text
  length; `--split-on-word`/`-sow` breaks at word boundaries instead of an
  arbitrary character offset. `--word-thold`/`-wt` is accepted for CLI
  parity but not yet applied — it needs per-word alignment probabilities
  that land with token-level/DTW timestamps
- `--output-json-full`/`-ojf` extends `-oj`'s JSON with a per-segment
  `"tokens"` array (`id`, decoded `text`, `p`, `plog`, interpolated
  `t0`/`t1`) — the decoder now tracks each sampled token's clamped
  log-probability (`WindowDecode::token_logprobs`, both the greedy and beam
  paths) instead of only the window's running sum
- `--output-words`/`-owts` writes a `.wts` karaoke script: a bash script
  driving `ffmpeg` to burn in each segment's text as a synced caption
  (`drawtext`, `enable='between(t,t0,t1)'`) over a plain color background;
  `--font-path`/`-fp` sets the caption font (default matches whisper.cpp's
  own default, which is macOS-only — override it elsewhere)
- Sampling controls now match whisper.cpp's parameter set: `--temperature`/
  `-tp` and `--temperature-inc`/`-tpi` build the fallback temperature ladder
  (`transcribe::temperature_ladder`) instead of a hardcoded one;
  `--best-of`/`-bo` draws N independent greedy samples per temperature and
  keeps the best by average log-probability; `--entropy-thold`/`-et` adds a
  low-entropy (collapsed-decode) quality gate alongside the existing
  compression-ratio/log-prob ones; `--no-speech-thold`/`-nth` suppresses
  segments for windows the model considers silent; `--no-fallback`/`-nf`
  disables the fallback ladder entirely. `--entropy-thold`'s exact
  whisper.cpp semantics weren't independently verified this pass — treat it
  as a best-effort match
- `--max-context`/`-mc` caps prior-window context tokens carried forward
  (`-1` keeps the model's own default); `--offset-t`/`-ot` and
  `--duration`/`-d` skip ahead into and limit how much of the input audio is
  transcribed (reported timestamps stay relative to the original file);
  `--offset-n`/`-on` shifts the starting index used by numbered output
  (`.srt`). `--audio-ctx`/`-ac` is accepted for CLI parity but not yet
  applied — safely truncating the encoder's context touches
  positional-embedding and cross-attention shape assumptions validated
  against real model weights, not worth the correctness risk this pass

### 🐛 Fixes

- The whole-file transcription path (the default, non-streaming CLI
  invocation) wasn't actually passing `max_len`/`split_on_word`/
  `word_thold`/the temperature ladder/`best_of`/`entropy_threshold`/
  `no_speech_threshold`/`no_fallback` into `Options` — only `--audio -`
  streaming mode was. All of those flags were silently no-ops on the
  common path since the PRs that introduced them (differing indentation
  between the two `Options` construction sites meant a find-and-replace
  landed on only one of them). Both sites are now kept in sync.

### ⭐ Features

- Console/debug output flags: `--version` prints the crate version and
  exits; `--no-prints`/`-np` suppresses the model-info dump, printing only
  results; `--print-progress`/`-pp` reports percent-complete to stderr
  (`transcribe_with_progress` drives `Stream` window-by-window instead of
  the one-shot `transcribe()` helper); `--print-special`/`-ps` includes
  special/control tokens' own vocab text inline via the new
  `Tokenizer::decode_with_specials` (limited practical effect in the
  current pipeline — see its doc comment); `--print-colors`/`-pc`
  ANSI-colors console segment text by per-token confidence (a 3-tier
  approximation of whisper.cpp's gradient, not a verified color match);
  `--print-confidence` appends a `(NN%)` suffix per token; `--log-score`/
  `-ls` prints each segment's average token log-probability to stderr;
  `--debug-mode`/`-debug` prints window-size/offset diagnostics to stderr
- `--audio`/`-f` is now repeatable, and bare positional arguments (plus
  `-` for stdin) are treated as audio paths too, batch-transcribing each
  one in turn — matching whisper.cpp. A per-file error (unreadable file,
  wrong sample rate, output-write failure) is reported and that file is
  skipped rather than aborting the whole batch; the process exits nonzero
  if any file failed. A `=== path ===` header separates each file's
  output when more than one is given
- `--detect-language`/`-dl` prints the detected language (and confidence)
  for each `--audio` file and exits without transcribing, via new
  `transcribe::detect_language_only`. English-only models report `("en",
  1.0)` without running the model, since detection isn't meaningful for
  them
- `--diarize`/`-di` tags each printed segment with whichever stereo
  channel had the higher RMS energy in that span (`(speaker 0)`/
  `(speaker 1)`), matching whisper.cpp's crude stereo diarization. `WavData`
  now carries `channel_samples` (per-channel, alongside the existing
  downmixed `samples` the pipeline itself uses) via a new
  `wav::diarize_speaker` helper; a file that isn't stereo gets a warning
  and no tags rather than an error. `--tinydiarize`/`-tdrz` is accepted
  for CLI/option-surface parity but **not applied** — detecting the
  `[_TT_]` speaker-turn token needs a vocab lookup plus an exemption from
  the shared, golden-transcript-validated suppression logic that runs on
  every decode step, not worth the risk without a real tinydiarize model
  to validate against
- `--suppress-nst`/`-sns` suppresses non-speech text tokens during
  decoding (punctuation/symbol-only tokens like `"..."` or `"♪"`, no
  alphanumeric characters) via a new `Tokenizer::is_non_speech_token` set
  computed once from the vocab. `--suppress-regex` is accepted for CLI/
  option-surface parity but **not applied** — rusty_whisper is
  zero-dependency and has no regex engine; adding one (or hand-rolling
  one) is its own scope, not a corner to cut inside this issue
- `--processors`/`-p` splits the input into N contiguous chunks and
  transcribes them in parallel, one `std::thread::scope`d thread per
  chunk, via new `transcribe::transcribe_parallel` — distinct from the
  existing intra-inference (matmul/conv/softmax) threading. Each chunk
  runs the full windowed pipeline independently (no cross-chunk context),
  same trade-off as whisper.cpp's `whisper_full_parallel`; results are
  concatenated with segment/token timestamps shifted back to the full
  input's timeline
- `--grammar`/`--grammar-rule`/`--grammar-penalty` constrain decoding to a
  GBNF-lite grammar via a new `grammar` module: rules built from string
  literals, rule references, alternation, concatenation, and parenthesized
  grouping — covers whisper.cpp's own primary use case (short,
  command-style grammars). Character classes (`[a-z]`) and repetition
  operators (`*`, `+`, `?`) are **not supported** and are a parse error
  rather than a silent mismatch; full GBNF (matching llama.cpp's
  character-level grammar engine) is a substantially larger undertaking
  than this pass's scope. Since the supported subset has no
  repetition/recursion, a grammar's language is finite: it's expanded to
  every complete candidate string up front (rejecting cycles, capped at
  4096 candidates) and matched via a prefix trie during decoding, applying
  `--grammar-penalty` (default 100.0) as a soft logit penalty — not a hard
  mask — to tokens that would violate it. `--grammar` accepts a file path
  or inline grammar text

### 🔧 Under the hood

- GitHub Actions CI: `rustfmt --check`, `clippy -D warnings` (default and
  `gguf` features), tests on both feature sets plus doc tests, and
  release builds for `wasm32-wasip1` and `wasm32-unknown-unknown`. The
  x86_64 runners exercise the runtime-detected AVX2 dequant path and its
  bit-identity test.
- One-time `cargo fmt` normalization of the whole tree; the codebase is
  now rustfmt-clean and CI enforces it

---

## 0.8.0 — GGUF, opt-in

**2026-07-17**

The last roadmap item. Off by default behind `--features gguf`, because
whisper.cpp defines no official whisper-GGUF schema — the metadata
mapping here is ours, documented in `src/gguf.rs`, ready to adapt if
upstream standardizes one.

### ⭐ Features

- Load GGUF (v2/v3) whisper models — `load_model` sniffs the magic, so
  `.gguf` files work everywhere `.bin` files do
- `--convert-gguf OUT` converts a loaded model to GGUF; quantized
  weights are copied verbatim (lossless), f16 tensors are stored as f32

### 🔧 Under the hood

- Round-trip tested synthetically (dense + quantized tensors, filters,
  byte-level vocab) and end-to-end: tiny.en-q5_1 `.bin` → `.gguf` →
  identical jfk.wav transcript
- Default builds get clear errors: loading a `.gguf` or passing
  `--convert-gguf` without the feature says exactly what to rebuild with

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
