# rusty-whisper

A pure-Rust, **zero-dependency** port of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI Whisper
speech recognition with no C/C++ toolchain, no build scripts, and no
crates.io dependencies. `unsafe` is confined to two audited leaf
modules — pointer glue at the wasm FFI boundary, and runtime-detected
AVX2 dequantization kernels that are equivalence-tested bit-for-bit
against the safe scalar path; everything else, the whole inference
pipeline included, is safe Rust.

It loads standard whisper.cpp ggml `.bin` model files and reproduces
whisper.cpp's canonical transcripts, validated against real models from
tiny through large-v3-turbo.

## Features

- **Full pipeline**: log-mel front-end, transformer encoder/decoder with
  KV caching, timestamps, 30 s chunking with seek-to-last-timestamp,
  conditioning on past text, temperature fallback with quality gates
- **Beam search** (default beam 5; `--beam 1` for greedy)
- **Quantized models** (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0), kept quantized in
  memory for 2-3x less RAM at dense-comparable speed on AVX2 CPUs
  (runtime-detected SIMD unpack; `--dense` dequantizes at load instead)
- **Multilingual**: language auto-detection, `--language CODE` to force,
  `--translate` for X → English
- **Streaming**: `--audio -` transcribes WAV from stdin, emitting
  segments as each 30 s window fills
- **Wasm**: runs under WASI runtimes and fully client-side in the
  browser — see [demo/](demo/)
- **GGUF** (opt-in, `--features gguf`): load GGUF models and convert
  `.bin` → `.gguf` with `--convert-gguf OUT`. whisper.cpp has no official
  whisper-GGUF schema yet, so the metadata mapping is ours (documented in
  `src/gguf.rs`); quantized weights convert losslessly
- CPU performance: multi-threaded, SIMD-friendly kernels built with
  `target-cpu=native` (see `.cargo/config.toml`); roughly 4-7x realtime
  for tiny on a 4-core AVX-512 machine

## Quickstart

```sh
# Get a model (any whisper.cpp ggml .bin works; q5_1 is a good default)
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin

# Transcribe a 16 kHz mono 16-bit PCM WAV of any length
cargo run --release -- --model ggml-tiny.en-q5_1.bin --audio speech.wav

# Anything-to-text via ffmpeg, streamed, printing segments live
ffmpeg -i talk.mp3 -ar 16000 -ac 1 -f wav - | \
  cargo run --release -- --model ggml-tiny.en-q5_1.bin --audio -
```

Output:

```
[00:00:00.000 --> 00:00:08.000]  And so my fellow Americans ask not what your country can do for you
[00:00:08.000 --> 00:00:11.000]  ask what you can do for your country.
```

## CLI

| Flag | Meaning |
|---|---|
| `--model`, `-m` | ggml `.bin` model file (f16, f32, or quantized) |
| `--audio`, `-f` | 16 kHz mono 16-bit PCM WAV, or `-` for stdin streaming |
| `--beam`, `-b` | beam size for the temperature-0 decode (default 5) |
| `--language`, `-l` | ISO code (`de`, `fr`, ...) or `auto` (default) |
| `--translate` | translate to English instead of transcribing |
| `--dense` | dequantize weights at load: faster decoding, 2-3x the memory |
| `--convert-gguf OUT` | write the loaded model as GGUF (`--features gguf` builds only) |

Running with only `--model` prints model info (hparams, tensor count,
special-token layout).

## Library

```rust
use rusty_whisper::{model, transcribe, wav};

let m = model::load_model(&mut std::io::BufReader::new(std::fs::File::open("ggml-tiny.en-q5_1.bin")?))?;
let audio = wav::read_wav(&mut std::fs::File::open("speech.wav")?)?;
let result = transcribe::transcribe(&m, &audio.samples, &transcribe::Options::default());
for seg in &result.segments {
    println!("[{:.2} --> {:.2}] {}", seg.t0, seg.t1, seg.text);
}
```

For incremental input use `transcribe::Stream` (`feed()` samples as they
arrive, `finish()` at end of input) — the CLI's stdin mode is built on it.

## Layout

| Module | Contents |
|---|---|
| `src/audio.rs` | Hann window, mixed-radix FFT, Slaney mel filterbank, log-mel (mirror of `log_mel_spectrogram`) |
| `src/model.rs` | ggml `.bin` parser: hparams, mel filters, vocab, tensors |
| `src/quant.rs` | ggml block formats, quantized-in-memory weights, quantized matmul |
| `src/tensor.rs` | f32 kernels: matmul, conv1d, layernorm, gelu, softmax; threading |
| `src/encoder.rs` | conv stem + transformer stack; multi-head attention |
| `src/decoder.rs` | KV-cached decoder, cross-attention, beam forking |
| `src/transcribe.rs` | chunking, timestamp rules, beam search, temperature fallback (mirror of `whisper_full`) |
| `src/tokenizer.rs` | vocab, special-token layout, language table |
| `src/wav.rs` | WAV reading, whole-file and streaming |
| `src/wasm.rs` | C-ABI exports for the browser demo |
| `src/gguf.rs` | GGUF read/write (feature `gguf`) |

## Validation

Every stage is unit-tested (58 tests), and the pipeline is validated
end to end against whisper.cpp's canonical outputs: jfk.wav with
tiny/base/small (f16 and quantized), multilingual models with language
auto-detection, large-v3-turbo (128 mel bands), multi-window repetitive
audio, streamed vs whole-file equivalence, and the browser demo driven
headlessly in Chromium. See [PLAN.md](PLAN.md) for the porting history
and [RELEASE_NOTES.md](RELEASE_NOTES.md) for what landed when.
