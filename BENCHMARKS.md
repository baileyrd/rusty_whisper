# Benchmarks

A direct, matched comparison against [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
on CPU. rusty_whisper is a naive-but-correct pure-Rust port; whisper.cpp
is years of hand-tuned C++/SIMD. The goal here is an honest measurement,
not a favorable one.

## Setup

| | |
|---|---|
| Machine | 4-core Intel Xeon @ 2.80 GHz, AVX-512 (VNNI, AMX_INT8) |
| whisper.cpp | v1.9.1 (ggml 080bbbe), `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON` |
| rusty_whisper | v0.8.0, `--release` with `target-cpu=native` (default `.cargo/config.toml`) |
| Threads | 4 on both |
| Decoding | greedy on both (whisper.cpp `-bs 1`, rusty `--beam 1`) |
| Audio | `samples/jfk.wav` — 11.0 s, 16 kHz mono |
| Models | the **same** ggml `.bin` files fed to both |

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

| Model | whisper.cpp | rusty_whisper | rusty_whisper `--dense` | Ratio |
|---|---|---|---|---|
| tiny.en-q5_1 | **0.65 s** (17× RT) | 1.86 s (5.9× RT) | 1.62 s | **2.9× slower** |
| large-v3-turbo-q5_0 | **20.5 s** | 53.8 s | — | **2.6× slower** |

RT = realtime multiple (11 s of audio ÷ wall-clock). Transcripts are
identical in text; whisper.cpp places segment boundaries a few frames
differently (e.g. 7.74 s vs 8.00 s).

## Where the gap is (tiny.en-q5_1)

| Stage | whisper.cpp | rusty_whisper | Ratio |
|---|---|---|---|
| mel | 15 ms | 119 ms | ~8× |
| **encode** | **443 ms** | **1482 ms** | **3.4×** |
| decode | 2.8 ms/token | 8.5 ms/token | 3.0× |

The encoder's matmuls dominate and account for essentially the whole gap.

## Why whisper.cpp is faster

1. **Native int8 GEMM.** whisper.cpp repacks quantized weights and
   multiplies in int8 using AVX-512 VNNI / AMX instructions (its banner
   reports `AVX512_VNNI = 1 | AMX_INT8 = 1 | REPACK = 1`). rusty_whisper
   dequantizes each weight row to f32 and runs f32 GEMM — so `--dense`
   (dequantize once at load) is even slightly *faster* for it, the
   opposite of whisper.cpp. This is the single largest factor and the
   clearest avenue to close the gap.
2. **GEMM maturity** — cache blocking, weight packing, optional BLAS.
   rusty_whisper uses autovectorized tiled loops.
3. **Fused attention** vs rusty_whisper materializing the score matrix.
4. **GPU backends** (Metal/CUDA/Vulkan) that rusty_whisper has none of.

## Memory

Keeping quantized weights quantized in memory (the default; `--dense`
opts out) roughly halves resident memory, at dense-comparable speed on
AVX2+ CPUs:

| Model | rusty_whisper (quantized) | rusty_whisper `--dense` |
|---|---|---|
| tiny.en-q5_1 | ~89 MB | ~192 MB |
| small.en-q5_1 | ~372 MB | ~1094 MB |

## Takeaway

A naive pure-Rust f32 port lands within **~2.6–3× of whisper.cpp's
hand-tuned int8/AMX kernels on CPU**, with byte-identical transcripts,
zero dependencies, and a browser/wasm target whisper.cpp can't match.
For real-time tiny/base transcription it is comfortably fast enough; for
large models or GPU throughput, whisper.cpp wins decisively. The biggest
single optimization left for rusty_whisper is a true int8 quantized
matmul path.

*Numbers were gathered on one ephemeral cloud VM and will vary with
hardware; treat the ratios as more durable than the absolute times.*
