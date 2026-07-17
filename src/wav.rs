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
}

pub fn read_wav(r: &mut impl Read) -> io::Result<WavData> {
    let mut header = [0u8; 12];
    r.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a RIFF/WAVE file"));
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
    for f in 0..n_frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            let off = (f * ch + c) * 2;
            let s = i16::from_le_bytes(data[off..off + 2].try_into().unwrap());
            acc += s as f32 / 32768.0;
        }
        samples.push(acc / ch as f32);
    }
    Ok(WavData { sample_rate, samples })
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
    fn rejects_non_wav() {
        let out = read_wav(&mut Cursor::new(b"OggS but not really a wav".to_vec()));
        assert!(out.is_err());
    }
}
