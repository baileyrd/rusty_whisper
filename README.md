# rusty-whisper

[![CI](https://github.com/baileyrd/rusty_whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/baileyrd/rusty_whisper/actions/workflows/ci.yml)

A pure-Rust, **zero-dependency by default** port of
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI Whisper
speech recognition with no C/C++ toolchain, no build scripts, and no
crates.io dependencies in the default build. The only exception is the
opt-in `mic` feature (native microphone capture via `cpal`, for
`whisper-stream`/`whisper-command`); file/stdin transcription and the
HTTP server need none of it. `unsafe` is confined to two audited leaf
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
- **Native microphone capture** (opt-in, `--features mic`): the `mic`
  module wraps `cpal` behind whisper.cpp's `audio_async` ring-buffer
  semantics (`src/mic.rs`) — a building block for live-capture tools
  (`whisper-stream`, `whisper-command`, `whisper-talk-llama`) built on top
  of it. On Linux, `cpal`'s ALSA backend needs `libasound2-dev` (or
  equivalent) installed to build.
- **`whisper-stream`** (opt-in, `--features mic`): real-time microphone
  transcription, mirroring whisper.cpp's `examples/stream` — a sliding
  window that redraws in place, or `--step 0` for VAD-triggered full-window
  decodes with timestamps (see `--help`)
- **`whisper-command`** (opt-in, `--features mic`): voice-command mode,
  mirroring whisper.cpp's `examples/command` — guided (`--commands FILE`),
  always-prompt, and general-purpose (wake phrase + grammar-constrained
  decode) modes (see `--help`)
- **`whisper-talk-llama`** (opt-in, `--features mic`): voice chatbot (STT ->
  LLM -> TTS), mirroring whisper.cpp's `examples/talk-llama` — talks HTTP to
  a separately-run [`rusty_llama --serve`](https://github.com/baileyrd/rusty_llama)
  (or anything else speaking the same OpenAI-compatible chat-completions
  wire format) instead of linking an LLM engine in-process; TTS is an
  external shell-out, same as upstream (see `--help`)
- CPU performance: multi-threaded, SIMD-friendly kernels (including a
  true int8 matmul via AVX2/AVX-512 VNNI) built with `target-cpu=native`
  (see `.cargo/config.toml`); roughly 6x realtime for tiny on a 4-core
  AVX-512 machine — within ~1.6-2.4x of whisper.cpp on CPU (closer on
  larger models), with identical transcripts (see
  [BENCHMARKS.md](BENCHMARKS.md))

## Quickstart

```sh
# Get a model (any whisper.cpp ggml .bin works; q5_1 is a good default)
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin

# Transcribe the bundled sample (see samples/)
cargo run --release -- --model ggml-tiny.en-q5_1.bin --audio samples/jfk.wav

# ...or your own 16 kHz mono 16-bit PCM WAV of any length
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

## whisper-stream

Real-time microphone transcription, mirroring whisper.cpp's
`examples/stream/stream.cpp`. Needs `--features mic` (native capture via
`cpal`; see the Features list above for the Linux build prerequisite):

```sh
cargo run --release --features mic --bin whisper-stream -- --model ggml-tiny.en-q5_1.bin
```

| Flag | Meaning |
|---|---|
| `--step`/`--length`/`--keep` | sliding-window sizing in ms (default 3000/10000/200); `--step 0` switches to VAD-triggered mode |
| `-vth`/`-fth` | VAD-mode energy threshold / high-pass cutoff Hz (default 0.6 / 100.0) |
| `-c`, `--capture N` | capture device index — see `--list-devices` |
| `-kc`, `--keep-context` | carry decoded text forward as a prompt (sliding-window mode) |
| `-sa`, `--save-audio` | save all captured audio to a timestamped `.wav` |
| `-bs`, `-tr`, `-nf`, `-l` | beam size, translate, no-fallback, language — same meaning as the main CLI |

`--tinydiarize`, `--audio-ctx`, `--no-gpu`/`--flash-attn` are accepted for
CLI parity but currently no-ops (see `--help`) — same scope cuts as the
main CLI's `--tinydiarize`/`--audio-ctx` and this crate's CPU-only design.

## whisper-command

Voice-command / assistant mode, mirroring whisper.cpp's
`examples/command/command.cpp`. Needs `--features mic`:

```sh
# Guided mode: pick the best match from a fixed phrase list
cargo run --release --features mic --bin whisper-command -- \
  --model ggml-tiny.en-q5_1.bin --commands commands.txt

# General-purpose mode: say the activation phrase, then a --grammar-constrained command
cargo run --release --features mic --bin whisper-command -- \
  --model ggml-tiny.en-q5_1.bin --grammar commands.gbnf
```

| Flag | Meaning |
|---|---|
| `-cmd`, `--commands FILE` | guided mode: one candidate phrase per line, scored by a single forced decode step |
| `-p`, `--prompt STRING` | activation phrase (always-prompt mode if set with no `--grammar`; general-purpose mode's wake phrase otherwise, default `"Ok Whisper, start listening for commands."`) |
| `--grammar FILE_or_TEXT` | GBNF-lite grammar (general-purpose mode's `"prompt"`/`"root"` rules — see `src/grammar.rs`) |
| `-ctx`, `--context STRING` | text primed as the model's initial prompt every decode |
| `-c`, `--capture N` | capture device index — see `--list-devices` |
| `-f`, `--file PATH` | append recognized commands/text to a file |

Mode is chosen the same way as upstream: `--commands` set → guided;
else `--prompt` set with no `--grammar` → always-prompt; else
general-purpose (the default, using whatever `--prompt`/`--grammar` was
given).

## whisper-talk-llama

Voice chatbot (STT -> LLM -> TTS), mirroring whisper.cpp's
`examples/talk-llama/talk-llama.cpp`. The LLM side is a separately-run
[`rusty_llama --serve`](https://github.com/baileyrd/rusty_llama) process
(or any other server speaking the same OpenAI-compatible
`/v1/chat/completions` format), rather than an in-process LLM engine —
start it yourself first, then:

```sh
cargo run --release --features mic --bin whisper-talk-llama -- \
  --model ggml-tiny.en-q5_1.bin --llama-port 8080
```

| Flag | Meaning |
|---|---|
| `--llama-host`/`--llama-port` | where `rusty_llama --serve` (or equivalent) is listening (default `127.0.0.1:8080`) |
| `--person`/`--bot-name` | names filled into the persona template (defaults `"User"`/`"Assistant"` — whisper.cpp defaults to its author's name/`"LLaMA"`, not meaningful here) |
| `--prompt-file PATH` | override the built-in persona template (same `{0}`../`{4}` placeholders) |
| `--session PATH` | persist/resume conversation history (JSON) across runs |
| `--wake-command STRING` | require this phrase (fuzzy-matched) at the start of every utterance |
| `--speak COMMAND` | TTS command to run per reply (default: none, text-only — no TTS script ships with this crate, unlike whisper.cpp's own repo) |
| `-t`/`--top-k`/`--top-p`/`--min-p`/`--seed` | LLM sampling params, forwarded to the server's chat-completions request |
| `--max-tokens N` | caps the LLM reply length — repurposed from whisper.cpp's `-mt` (which caps transcription; this crate has no analogous knob there) |

Conversation history is sent as a `messages[]` array each turn (system
persona + accumulated user/assistant turns) rather than upstream's raw
growing-text-buffer-plus-antiprompt-match scheme; the server's own
chat-template/EOS handling replaces the antiprompt match for knowing when
a reply is done.

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
| `src/mic.rs` | native microphone capture via `cpal` (feature `mic`) |
| `src/json.rs` | minimal JSON value parser + string escaper (zero-dependency) |
| `src/llm_client.rs` | HTTP client for OpenAI-compatible chat-completions endpoints (`whisper-talk-llama`) |

## Validation

Every stage is unit-tested (58 tests), and the pipeline is validated
end to end against whisper.cpp's canonical outputs: jfk.wav with
tiny/base/small (f16 and quantized), multilingual models with language
auto-detection, large-v3-turbo (128 mel bands), multi-window repetitive
audio, streamed vs whole-file equivalence, and the browser demo driven
headlessly in Chromium. See [PLAN.md](PLAN.md) for the porting history
and [RELEASE_NOTES.md](RELEASE_NOTES.md) for what landed when.
