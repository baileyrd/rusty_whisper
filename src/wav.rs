//! Minimal RIFF/WAV reader: 16-bit PCM -> mono f32 in [-1, 1].
//!
//! Deliberately small — whisper.cpp shells out to a full decoder library;
//! we support the one format everyone converts to anyway
//! (`ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`). Sample-rate conversion is out
//! of scope: the caller gets the file's rate back and must feed 16 kHz.

use std::io::{self, Read};

pub struct WavData {
    pub sample_rate: u32,
    /// Mono samples in [-1, 1] (channels averaged).
    pub samples: Vec<f32>,
    /// Per-channel samples in [-1, 1], in file order (1 entry for mono
    /// files, 2 for stereo, etc). Used for `--diarize`'s crude
    /// per-channel-energy speaker tagging; the transcription pipeline
    /// itself always uses the downmixed `samples`.
    pub channel_samples: Vec<Vec<f32>>,
}

/// Incremental WAV reader: parses the header eagerly, then yields mono f32
/// frames in chunks — for transcribing from a pipe/stdin as audio arrives.
pub struct WavStream<R: Read> {
    r: R,
    pub sample_rate: u32,
    channels: usize,
    /// Bytes left in the data chunk (u32::MAX-size streams read to EOF).
    remaining: usize,
}

impl<R: Read> WavStream<R> {
    pub fn new(mut r: R) -> io::Result<Self> {
        let mut header = [0u8; 12];
        r.read_exact(&mut header)?;
        if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a RIFF/WAVE file",
            ));
        }
        let (mut sample_rate, mut channels, mut bits, mut format) = (0u32, 0u16, 0u16, 0u16);
        loop {
            let mut chunk_hdr = [0u8; 8];
            r.read_exact(&mut chunk_hdr)?;
            let size = u32::from_le_bytes(chunk_hdr[4..8].try_into().unwrap()) as usize;
            match &chunk_hdr[0..4] {
                b"fmt " => {
                    let mut fmt = vec![0u8; size];
                    r.read_exact(&mut fmt)?;
                    format = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
                    channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
                    sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                    bits = u16::from_le_bytes(fmt[14..16].try_into().unwrap());
                }
                b"data" => {
                    if format != 1 || bits != 16 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("only 16-bit PCM supported (got format {format}, {bits}-bit)"),
                        ));
                    }
                    if channels == 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "zero channels"));
                    }
                    return Ok(WavStream {
                        r,
                        sample_rate,
                        channels: channels as usize,
                        remaining: size,
                    });
                }
                _ => {
                    let mut skip = vec![0u8; size + (size & 1)];
                    r.read_exact(&mut skip)?;
                }
            }
        }
    }

    /// Read up to `max_frames` mono frames; empty Vec = end of stream.
    pub fn read_frames(&mut self, max_frames: usize) -> io::Result<Vec<f32>> {
        let bytes_per_frame = 2 * self.channels;
        let want = (max_frames * bytes_per_frame).min(self.remaining);
        let mut raw = vec![0u8; want];
        let mut filled = 0;
        while filled < want {
            match self.r.read(&mut raw[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        self.remaining -= filled;
        let frames = filled / bytes_per_frame;
        let mut out = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut acc = 0.0f32;
            for c in 0..self.channels {
                let off = f * bytes_per_frame + c * 2;
                acc += i16::from_le_bytes(raw[off..off + 2].try_into().unwrap()) as f32 / 32768.0;
            }
            out.push(acc / self.channels as f32);
        }
        Ok(out)
    }
}

pub fn read_wav(r: &mut impl Read) -> io::Result<WavData> {
    let mut header = [0u8; 12];
    r.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a RIFF/WAVE file",
        ));
    }

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut format = 0u16;
    let mut data: Option<Vec<u8>> = None;

    // Walk chunks until we've seen fmt and data.
    loop {
        let mut chunk_hdr = [0u8; 8];
        if r.read_exact(&mut chunk_hdr).is_err() {
            break;
        }
        let id = &chunk_hdr[0..4];
        let size = u32::from_le_bytes(chunk_hdr[4..8].try_into().unwrap()) as usize;
        match id {
            b"fmt " => {
                let mut fmt = vec![0u8; size];
                r.read_exact(&mut fmt)?;
                format = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
                channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                bits = u16::from_le_bytes(fmt[14..16].try_into().unwrap());
            }
            b"data" => {
                let mut d = vec![0u8; size];
                r.read_exact(&mut d)?;
                data = Some(d);
            }
            _ => {
                // Skip unknown chunk (chunks are word-aligned).
                let mut skip = vec![0u8; size + (size & 1)];
                r.read_exact(&mut skip)?;
            }
        }
        if data.is_some() && sample_rate != 0 {
            break;
        }
    }

    let data = data.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no data chunk"))?;
    if format != 1 || bits != 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("only 16-bit PCM supported (got format {format}, {bits}-bit)"),
        ));
    }
    if channels == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "zero channels"));
    }

    let ch = channels as usize;
    let n_frames = data.len() / (2 * ch);
    let mut samples = Vec::with_capacity(n_frames);
    let mut channel_samples: Vec<Vec<f32>> = vec![Vec::with_capacity(n_frames); ch];
    for f in 0..n_frames {
        let mut acc = 0.0f32;
        for (c, ch_samples) in channel_samples.iter_mut().enumerate() {
            let off = (f * ch + c) * 2;
            let s = i16::from_le_bytes(data[off..off + 2].try_into().unwrap());
            let v = s as f32 / 32768.0;
            acc += v;
            ch_samples.push(v);
        }
        samples.push(acc / ch as f32);
    }
    Ok(WavData {
        sample_rate,
        samples,
        channel_samples,
    })
}

/// Crude stereo speaker tagging, matching whisper.cpp's `--diarize`: over a
/// stereo file, tags a `[t0, t1)` span with whichever channel had the
/// higher RMS energy in that span (0 = left/"speaker 0", 1 =
/// right/"speaker 1"). Returns `None` for anything but exactly 2 channels,
/// or an empty span.
pub fn diarize_speaker(
    channel_samples: &[Vec<f32>],
    sample_rate: u32,
    t0: f32,
    t1: f32,
) -> Option<usize> {
    if channel_samples.len() != 2 || t1 <= t0 {
        return None;
    }
    let start = ((t0 * sample_rate as f32).max(0.0)) as usize;
    let end = ((t1 * sample_rate as f32).max(0.0)) as usize;
    let rms = |ch: &[f32]| -> f32 {
        let end = end.min(ch.len());
        if start >= end {
            return 0.0;
        }
        let slice = &ch[start..end];
        (slice.iter().map(|v| v * v).sum::<f32>() / slice.len() as f32).sqrt()
    };
    let (l, r) = (rms(&channel_samples[0]), rms(&channel_samples[1]));
    Some(if r > l { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_len = samples.len() * 2;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // PCM
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        b.extend_from_slice(&(channels * 2).to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }

    #[test]
    fn reads_mono_16bit() {
        let wav = make_wav(16000, 1, &[0, 16384, -16384, 32767]);
        let out = read_wav(&mut Cursor::new(wav)).unwrap();
        assert_eq!(out.sample_rate, 16000);
        assert_eq!(out.samples.len(), 4);
        assert!((out.samples[1] - 0.5).abs() < 1e-4);
        assert!((out.samples[2] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn downmixes_stereo() {
        // L=1.0-ish, R=0 -> mono 0.5-ish
        let wav = make_wav(16000, 2, &[32767, 0]);
        let out = read_wav(&mut Cursor::new(wav)).unwrap();
        assert_eq!(out.samples.len(), 1);
        assert!((out.samples[0] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn wav_stream_matches_whole_file_read() {
        let samples: Vec<i16> = (0..40000)
            .map(|i| ((i * 37) % 20000) as i16 - 10000)
            .collect();
        let bytes = make_wav(16000, 1, &samples);
        let whole = read_wav(&mut Cursor::new(bytes.clone())).unwrap();
        let mut st = WavStream::new(Cursor::new(bytes)).unwrap();
        assert_eq!(st.sample_rate, 16000);
        let mut streamed = Vec::new();
        loop {
            let chunk = st.read_frames(1234).unwrap();
            if chunk.is_empty() {
                break;
            }
            streamed.extend(chunk);
        }
        assert_eq!(streamed.len(), whole.samples.len());
        assert_eq!(streamed, whole.samples);
    }

    #[test]
    fn rejects_non_wav() {
        let out = read_wav(&mut Cursor::new(b"OggS but not really a wav".to_vec()));
        assert!(out.is_err());
    }

    #[test]
    fn diarize_picks_the_louder_channel() {
        let quiet = vec![0.01f32; 1000];
        let loud = vec![0.5f32; 1000];
        assert_eq!(
            diarize_speaker(&[loud.clone(), quiet.clone()], 1000, 0.0, 1.0),
            Some(0)
        );
        assert_eq!(diarize_speaker(&[quiet, loud], 1000, 0.0, 1.0), Some(1));
    }

    #[test]
    fn diarize_none_unless_exactly_stereo() {
        let ch = vec![0.1f32; 100];
        assert_eq!(
            diarize_speaker(std::slice::from_ref(&ch), 100, 0.0, 1.0),
            None
        );
        assert_eq!(
            diarize_speaker(&[ch.clone(), ch.clone(), ch], 100, 0.0, 1.0),
            None
        );
    }

    #[test]
    fn diarize_none_for_empty_span() {
        let ch = vec![0.1f32; 100];
        assert_eq!(diarize_speaker(&[ch.clone(), ch], 100, 1.0, 1.0), None);
    }
}
