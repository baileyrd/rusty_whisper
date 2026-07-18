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
multiplying.

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

| Model | whisper.cpp | rusty_whisper (int8) | Ratio |
|---|---|---|---|
| tiny.en-q5_1 | **0.62 s** (17.7× RT) | 1.70 s (6.5× RT) | **2.7× slower** |
| large-v3-turbo-q5_0 | **20.1 s** | 42.5 s | **2.1× slower** |

RT = realtime multiple (11 s of audio ÷ wall-clock), best of several runs
each. Transcripts are identical in text, and segment boundaries match
whisper.cpp's (e.g. `00:00:07.740`) rather than snapping to whole seconds.

The int8 GEMM path (VNNI/AVX2, no AMX — see below) is unchanged from the
prior revision of this file; these ratios are a fresh remeasurement on a
different ephemeral VM instance (2.10 GHz here vs 2.80 GHz previously),
not a result of any code change. The absolute times moved with the
hardware; the ratios are the more durable number and stayed in the same
~2-3× band.

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
2. **GEMM maturity** — cache blocking, packing, optional BLAS.
   rusty_whisper uses 4-row-blocked kernels over the autovectorizer.
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

With a true int8 GEMM, the pure-Rust port lands within **~2–3× of
whisper.cpp on CPU** — closer on larger, more encoder-bound models — with
byte-identical transcripts, zero dependencies, and a browser/wasm target
whisper.cpp can't match. For real-time tiny/base transcription it is
comfortably fast enough; for large models or GPU throughput, whisper.cpp
still wins. An **AMX int8** path with weight repacking, matching
whisper.cpp's fastest kernels, remains the biggest theoretical
optimization left — but see above on why it isn't shipped on this
hardware.

*Numbers were gathered on one ephemeral cloud VM and will vary with
hardware; treat the ratios as more durable than the absolute times.*
