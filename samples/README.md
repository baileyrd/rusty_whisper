# Sample audio

Small, ready-to-use clips for trying rusty-whisper. All are 16 kHz mono
16-bit PCM WAV — the format the CLI expects.

| File | Length | Source |
|------|--------|--------|
| `jfk.wav` | 11 s | John F. Kennedy, 1961 inaugural address (excerpt) |

`jfk.wav` is a U.S. federal government work and is in the public domain.
It is the same reference clip [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
ships, so output can be compared directly.

## Use

```sh
cargo run --release -- --model ggml-tiny.en-q5_1.bin --audio samples/jfk.wav
```

Expected transcript:

```
And so my fellow Americans ask not what your country can do for you,
ask what you can do for your country.
```

## Bring your own

Any audio your system can decode works once converted to 16 kHz mono:

```sh
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le my-clip.wav
```
