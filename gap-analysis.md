# Gap analysis: rusty_whisper vs. whisper.cpp v1.9.1

Reference pinned at tag `v1.9.1` (latest release, 2026-06-19). Scope per
2026-07-23 scoping round: **all** whisper.cpp example tools (cli, server,
stream, command, talk-llama), not just the core transcription CLI. GPU
backends are tracked but labeled `needs-human` (not auto-implemented) per
rusty_whisper's zero-dependency, pure-Rust, CPU-only design.

Current rusty_whisper surface (baseline, for reference): CLI flags
`--model/-m`, `--audio/-f`, `--language/-l`, `--translate`, `--dense`,
`--convert-gguf`, `--beam/-b`, `--help/-h`; library `model::load_model`,
`wav::read_wav`, `transcribe::{transcribe, Options, Stream}`. `Options`
already has `language`, `translate`, `beam_size`, `condition_on_past`,
`temperatures` (fallback ladder), `compression_ratio_threshold`,
`logprob_threshold`, `max_initial_ts` — several sampling knobs exist
internally but aren't exposed via CLI or fully aligned with whisper.cpp's
parameter set.

| Symbol | Category | Platforms | Reference | Breaking? | Est. size | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Output file formats (txt/vtt/srt/csv/json) | fn (new) | both | `cli.cpp` `-otxt/-ovtt/-osrt/-ocsv/-oj/-of` | no | M | Currently stdout-only. Needs a writer module per format plus `--output-file` base-path handling. |
| `--output-json-full` (token-level JSON) | fn (new) | both | `cli.cpp` `-ojf` | no | M | Depends on token-level data (id/prob/logprob/timestamps) being surfaced from the decoder. |
| Karaoke `.wts` + `--font-path` | fn (new) | both | `cli.cpp` `-owts/-fp` | no | S | Niche; shells out to ffmpeg itself only via the generated script, not this process. Low priority. |
| Segment/word formatting: `--max-len`, `--split-on-word`, `--word-thold` | fn (new) | both | `cli.cpp` `-ml/-sow/-wt` | no | M | Controls segment splitting granularity in output. |
| Sampling controls: `--temperature`, `--temperature-inc`, `--best-of`, `--entropy-thold`, `--no-speech-thold`, `--no-fallback` | fn (existing) | both | `cli.cpp` `-tp/-tpi/-bo/-et/-nth/-nf` | no | M | `Options` has `temperatures`/`compression_ratio_threshold`/`logprob_threshold` already; add the missing fields and CLI flags to match whisper.cpp's exact knob set. |
| Context/window controls: `--max-context`, `--audio-ctx`, `--offset-t`, `--offset-n`, `--duration` | fn (new) | both | `cli.cpp` `-mc/-ac/-ot/-on/-d` | no | M | Trimming/windowing controls, no existing equivalent. |
| Initial prompt: `--prompt`, `--carry-initial-prompt` | fn (new) | both | `cli.cpp` `--prompt/--carry-initial-prompt` | no | S | |
| Console/debug output flags: `--debug-mode`, `--no-prints`, `--print-special`, `--print-colors`, `--print-confidence`, `--print-progress`, `--version`, `--log-score` | fn (new) | both | `cli.cpp` various | no | M | Bundle as one CLI-polish issue. |
| Multiple input files (repeatable `-f`, positional args) | fn (existing) | both | `cli.cpp` `-f` (repeatable) | no | S | rusty_whisper's `--audio` currently takes one path only. |
| `--detect-language` (detect-only, exit) | fn (new) | both | `cli.cpp` `-dl` | no | S | Language auto-detect logic already exists internally (used for `--language auto`); needs a CLI mode that runs it and exits. |
| Diarization: `--diarize` (stereo), `--tinydiarize` + `speaker_turn_next` | fn (new) | both | `cli.cpp` `-di/-tdrz`, `whisper_full_get_segment_speaker_turn_next` | no | M | Stereo diarize needs 2-channel WAV support (check `wav.rs`, currently mono-oriented); tinydiarize needs the `[_TT_]` special token recognized from `-tdrz` fine-tuned models. |
| Token suppression: `--suppress-regex`, `--suppress-nst` | fn (new) | both | `cli.cpp` `--suppress-regex/-sns` | no | S | |
| Parallel processors: `--processors` (chunk-parallel `whisper_full_parallel`) | fn (new) | both | `cli.cpp` `-p`, `whisper_full_parallel` | no | M | Splits audio across N independent decode passes; distinct from existing beam/thread parallelism. |
| GBNF grammar-constrained decoding | fn (new) | both | `cli.cpp` `--grammar/--grammar-rule/--grammar-penalty`, `whisper_grammar_element` API | no | L | Needs a GBNF parser + logit-penalty hook in the decoder sampler. Prerequisite for `whisper-command`'s general-purpose mode. |
| DTW token-level timestamp alignment | fn (new) | both | `cli.cpp` `-dtw`, alignment head presets, `t_dtw` in `whisper_token_data` | no | L | Dynamic time warping over cross-attention weights; needs per-model alignment-head presets ported from whisper.cpp's tables. |
| VAD preprocessing (Silero-style) | fn (new) | both | `cli.cpp` `--vad` + `-vm/-vt/-vspd/-vsd/-vmsd/-vp/-vo`, standalone `whisper_vad_*` API | no | L | Needs a VAD model loader (likely another ggml block format, same shape as the existing model loader) plus the VAD network's forward pass. No new external dependency expected. |
| Library callback hooks (new_segment/progress/encoder_begin/abort/logits_filter) | fn (new) | both | `whisper.h` callback fields on `whisper_full_params` | no | M | Rust-idiomatic equivalent: closures/trait objects on `Options`. |
| Timing/perf instrumentation (`get_timings`/`print_timings`/`reset_timings`, `print_system_info`) | fn (new) | both | `whisper.h` | no | S | |
| Custom log sink (`whisper_log_set` equivalent) | fn (new) | both | `whisper.h` | no | S | |
| Public language-table API (`lang_str`, `lang_str_full`, `lang_max_id`, `is_multilingual`) | fn (existing, check) | both | `whisper.h` | no | S | May already exist internally in `tokenizer.rs` without being `pub` — verify before filing as new work vs. just exporting. |
| Public model-introspection API (`model_n_vocab/audio_ctx/...`, `model_type_readable`) | fn (existing, check) | both | `whisper.h` | no | S | CLI already prints model info per README; verify whether the underlying fields are `pub` on `model::Model` already. |
| CUDA backend | fn (new) | linux/windows | whisper.cpp CUDA backend | no | L | **needs-human** — vendor SDK, contradicts zero-dependency pure-Rust design. |
| Metal backend | fn (new) | macos | whisper.cpp Metal backend | no | L | **needs-human** — same reason. |
| Vulkan backend | fn (new) | both | whisper.cpp Vulkan backend | no | L | **needs-human** — same reason. |
| HIP/ROCm backend | fn (new) | linux | whisper.cpp HIP backend | no | L | **needs-human** — same reason. |
| OpenVINO encoder acceleration | fn (new) | both | `cli.cpp` `-oved`, `whisper_ctx_init_openvino_encoder` | no | L | **needs-human** — external SDK dependency. |
| CANN (Ascend NPU) backend | fn (new) | linux | whisper.cpp CANN backend | no | L | **needs-human** — vendor SDK. |
| MUSA (Moore Threads) backend | fn (new) | linux | whisper.cpp MUSA backend | no | L | **needs-human** — vendor SDK. |
| OpenBLAS / Accelerate CPU BLAS path | fn (new) | both | whisper.cpp build options | no | M | **needs-human** — external C library dependency, contradicts zero-dependency design even though it's CPU-only. |
| GPU control flags (`--no-gpu`, `--device`, `--flash-attn`/`--no-flash-attn`) | fn (new) | both | `cli.cpp` `-ng/-dev/-fa/-nfa` | no | S | **needs-human** — meaningless without an actual GPU backend; bundle with whichever backend issue lands first. |
| HTTP server core: routing, `GET /health`, `GET /` static, CORS/OPTIONS | fn (new) | both | `examples/server/server.cpp` | no | L | New binary. Zero-dependency stance makes a hand-rolled HTTP/1.1 server the default assumption; if that turns out impractical, adding a minimal HTTP dependency is a stop-and-ask per the loop's rules — flagging now so it's not a surprise later. |
| `POST /inference` endpoint (multipart upload, full param surface, format negotiation) | fn (new) | both | `examples/server/server.cpp` | no | L | Depends on the output-formats gap above and the HTTP server core. |
| `POST /load` hot model-swap | fn (new) | both | `examples/server/server.cpp` | no | S | Depends on HTTP server core. |
| `--convert` (ffmpeg shell-out for non-WAV uploads) | fn (new) | both | `examples/server/server.cpp` | no | S | Shells out to an external `ffmpeg` binary at runtime (not a Rust dependency), lower priority. |
| Native microphone capture | fn (new) | both | `examples/stream/stream.cpp` (SDL2) | no | L | **Likely needs a new dependency** (e.g. `cpal`) or raw OS FFI (ALSA/CoreAudio/WASAPI) as a new audited `unsafe` leaf module, same shape as the existing wasm FFI/AVX2 modules. Prerequisite for `whisper-stream` and `whisper-command`. Flagging the dependency question now since it blocks two downstream tools. |
| `whisper-stream` sliding-window loop + energy VAD trigger + CLI flags | fn (new) | both | `examples/stream/stream.cpp` | no | L | Depends on native mic capture. |
| `whisper-command` (command-list / prompt-similarity / grammar-guided modes) | fn (new) | both | `examples/command/command.cpp` | no | L | Depends on native mic capture and GBNF grammar support above. |
| `whisper-talk-llama` (voice chatbot: STT → LLM → TTS) | fn (new) | both | `examples/talk-llama/talk-llama.cpp` | no | XL | **needs-human**, filed as a single tracking issue (not auto-implemented). Depends on native mic capture above. Rather than vendoring a C++ llama.cpp port, integration should target [`rusty_llama`](https://github.com/baileyrd/rusty_llama) (a sibling pure-Rust llama.cpp port) as the LLM engine once it has sufficient parity of its own; TTS still needs an external shell-out (no in-tree TTS in either project). |

## Summary

- **38 candidate rows** after judgment (raw whisper.cpp surface is much
  larger; noise from platform-`cfg`, duplicate/overlapping flags, and
  internal-only struct fields with no CLI exposure was filtered out).
- **9 rows** are GPU/accelerator-backend related and will be filed
  `needs-human` per the 2026-07-23 scoping decision — visible as issues,
  never auto-implemented.
- **1 row** (`whisper-talk-llama`) is flagged for exclusion/deferral rather
  than filing — see notes above.
- **2 rows** (HTTP server core, native mic capture) carry an explicit
  "likely needs a new dependency" flag — filing proceeds, but implementation
  will hit the loop's standing stop-and-ask on the first of these it reaches.
- No breaking-change rows: every gap is a pure addition to rusty_whisper's
  current public surface.
