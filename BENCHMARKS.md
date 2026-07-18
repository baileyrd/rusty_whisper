# Benchmarks

A direct, matched comparison against [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
on CPU. rusty_whisper is a naive-but-correct pure-Rust port; whisper.cpp
is years of hand-tuned C++/SIMD. The goal here is an honest measurement,
not a favorable one.

## Setup

| | |
|---|---|
| Machine | 4-core Intel Xeon @ 2.10 GHz, AVX-512 (VNNI, AMX_INT8) |
| whisper.cpp | v1.9.1 (ggml 080bbbe), `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON` |
| rusty_whisper | main (int8 GEMM path), `--release` with `target-cpu=native` (default `.cargo/config.toml`) |
| Threads | 4 on both |
| Decoding | greedy on both (whisper.cpp `-bs 1`, rusty `--beam 1`) |
| Audio | `samples/jfk.wav` — 11.0 s, 16 kHz mono |
| Models | the **same** ggml `.bin` files fed to both |

rusty_whisper uses a **true int8 quantized matmul** (AVX2 `maddubs`,
AVX-512 VNNI `dpbusd`) — it does not dequantize weights to f32 before
multiplying — with both the weight (N) and activation-row (M) dimensions
cache-blocked (8 and 64 respectively; see "Why whisper.cpp is still
faster" below). K is not blocked — see that section for why.

Both binaries report the same CPU features. The comparison is strictly
CPU-to-CPU: whisper.cpp is given no BLAS and no GPU backend, either of
which would widen the gap in its favor.

Commands:

```sh
# whisper.cpp
whisper-cli -m ggml-tiny.en-q5_1.bin -f samples/jfk.wav -t 4 -bs 1

# rusty_whisper
rusty-whisper --model ggml-tiny.en-q5_1.bin --audio samples/jfk.wav --beam 1
```

Times exclude model load and WAV decode (whisper.cpp "total time";
rusty_whisper "transcribed in"). Best of several runs, warm cache.

## Wall-clock

| Model | whisper.cpp | rusty_whisper (int8) | was (N-only) | was (unblocked) | Ratio |
|---|---|---|---|---|---|
| tiny.en-q5_1 | **0.60 s** (18.3× RT) | 1.60 s (6.9× RT) | 1.66 s | 1.70 s | **2.7× slower** |
| large-v3-turbo-q5_0 | **21.1 s** | 33.3 s | 33.9 s | 42.5 s | **1.6× slower** |

RT = realtime multiple (11 s of audio ÷ wall-clock), best of several runs
each. Transcripts are identical in text, and segment boundaries match
whisper.cpp's (e.g. `00:00:07.740`) rather than snapping to whole seconds.

N-blocking (see Setup) was the big win for large-v3-turbo — **20% faster
wall-clock**, pulling the ratio from 2.1× to 1.6× — but did nothing for
tiny, which stayed flat within run-to-run noise: tiny's quantized
activation matrix (largest layer ~576 KiB) already fits comfortably in
L2/L3, so re-sweeping it once per weight row was already cheap;
large-v3-turbo's (~2 MiB) didn't, so cutting the number of full sweeps 8x
(one per `N_BLOCK`-row weight group instead of one per row) cut real
memory traffic. Adding M-blocking on top (a `[M_BLOCK, cols]` tile of the
output/activations stays resident across the N sweep instead of the
whole `[m, cols]` output being swept once per N-group) gave a further,
smaller ~3-4% on large-v3-turbo — measured as a controlled A/B, 10 runs
each, with clean separation between the two samples (M-blocked: 32.8-33.7
s; not: 33.9-37.4 s) — and no measurable change on tiny.

## Where the gap is (tiny.en-q5_1)

Per-stage breakdown from the int8 GEMM path's original measurement
session (kept for illustration — the encoder/decoder code is unchanged
since, so the shape of the gap still applies even though this file's
wall-clock numbers above were refreshed on different hardware):

| Stage | whisper.cpp | rusty_whisper (int8) | Ratio |
|---|---|---|---|
| mel | 24 ms | 142 ms | ~6× |
| **encode** | **481 ms** | **1409 ms** | **2.9×** |
| decode | 2.9 ms/token | 10.9 ms/token | ~3.8× |

The encoder's matmuls dominate and account for essentially the whole gap.

## Why whisper.cpp is still faster

Both now do int8 GEMM, so the earlier "f32 vs int8" gap is closed. What
remains:

1. **AMX + weight repacking.** whisper.cpp's banner reports
   `AVX512_VNNI = 1 | AMX_INT8 = 1 | REPACK = 1`: it repacks weights into
   a tile-friendly layout and multiplies on the AMX int8 matrix units.
   rusty_whisper uses VNNI `dpbusd` (a fused 32-MAC, one tier below AMX)
   and unpacks weights on the fly. This is the largest remaining factor,
   but an AMX (`TDPBUSD` tile) tier was prototyped and dropped: on this
   KVM-virtualized machine it silently corrupted encoder output under any
   genuinely multi-core execution, and the corruption survived every
   mitigation tried (per-thread tile-data permission, a global mutex
   serializing all tile instructions, eliminating heap allocation between
   `ldtilecfg` and the tile ops, and pinning each thread to a fixed core)
   while single-core runs were 100% reliable across 40+ repeats — strong
   evidence of a hypervisor/kernel AMX tile-state save-restore bug rather
   than anything fixable in this crate. Not worth shipping given the
   correctness risk; may be revisited on non-virtualized hardware.
2. **GEMM maturity** — packing and optional BLAS. N and M are now
   cache-blocked (see Setup); K isn't, since at these problem sizes (a
   few thousand elements at most, a few KiB of int8) a whole row already
   fits L1, so tiling it further would add loop overhead for no cache
   benefit — M/N blocking earn their keep because the *matrices*, not
   individual rows, don't fit cache. Weight rows are still unpacked
   fresh on every call rather than packed once per matmul, and
   rusty_whisper uses 4-row-blocked kernels over the autovectorizer
   rather than hand-written micro-kernels.
3. **Fused attention** vs rusty_whisper materializing the score matrix.
4. **GPU backends** (Metal/CUDA/Vulkan) that rusty_whisper has none of.

## Memory

Keeping quantized weights quantized in memory (the default) roughly
halves resident memory. With the int8 path this is now also the *faster*
option, so `--dense` (dequantize at load) only trades memory for nothing
on AVX2+ CPUs — it remains for CPUs without AVX2:

| Model | rusty_whisper (quantized) | rusty_whisper `--dense` |
|---|---|---|
| tiny.en-q5_1 | ~89 MB | ~192 MB |
| small.en-q5_1 | ~372 MB | ~1094 MB |

## Takeaway

With a true int8 GEMM and M/N cache blocking, the pure-Rust port lands
within **~1.6–2.7× of whisper.cpp on CPU** — closer on larger, more
encoder-bound models, where cache blocking helps most — with
byte-identical transcripts, zero dependencies, and a browser/wasm target
whisper.cpp can't match. For real-time tiny/base transcription it is
comfortably fast enough; for large models or GPU throughput, whisper.cpp
still wins. An **AMX int8** path with weight repacking, matching
whisper.cpp's fastest kernels, remains the biggest theoretical
optimization left — but see above on why it isn't shipped on this
hardware.

*Numbers were gathered on one ephemeral cloud VM and will vary with
hardware; treat the ratios as more durable than the absolute times.*
