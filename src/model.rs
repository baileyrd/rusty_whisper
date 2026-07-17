//! ggml `.bin` model file loader — the same format `whisper_model_load`
//! reads (produced by whisper.cpp's `convert-pt-to-ggml.py`).
//!
//! Layout (all integers little-endian i32 unless noted):
//! ```text
//! u32 magic = 0x67676d6c ("ggml")
//! hparams: n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
//!          n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype
//! mel filters: n_mel, n_fft, then n_mel*n_fft f32
//! vocab: n_tokens, then per token { len, bytes }
//! tensors until EOF: { n_dims, name_len, dtype, dims[n_dims], name, data }
//! ```
//! F32 and F16 tensors are supported; quantized dtypes are rejected for now
//! (PLAN.md phase 7).

use std::collections::HashMap;
use std::io::{self, Read};

use crate::quant::{block_bytes, QTensor, Weight, QK};
use crate::tensor::Tensor;

pub const GGML_MAGIC: u32 = 0x6767_6d6c;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HParams {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
}

impl HParams {
    /// Model size name by encoder layer count, as whisper.cpp reports it.
    pub fn model_type(&self) -> &'static str {
        match self.n_audio_layer {
            4 => "tiny",
            6 => "base",
            12 => "small",
            24 => "medium",
            32 => "large",
            _ => "unknown",
        }
    }

    pub fn is_multilingual(&self) -> bool {
        self.n_vocab >= 51865
    }
}

pub struct Model {
    pub hparams: HParams,
    /// n_mels x (n_fft/2+1) row-major, embedded in the file.
    pub mel_filters: Vec<f32>,
    /// token id -> raw bytes (BPE tokens are byte-level, not always UTF-8).
    pub vocab: Vec<Vec<u8>>,
    /// Quantized 2-D matrices stay in their block format; everything else
    /// (biases, layernorms, convs, f16/f32 matrices) is dense f32.
    pub tensors: HashMap<String, Weight>,
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            // Subnormal: value = f * 2^-24. Shift the msb into the implicit
            // bit; msb at position p gives exponent p - 24 (bias 127).
            let p = 31 - f.leading_zeros();
            let mantissa = (f << (10 - p)) & 0x3ff;
            (sign << 31) | ((p + 103) << 23) | (mantissa << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// f32 -> f16 bits, round-to-nearest. (Only tests produce f16 today; the
/// loader just reads them.)
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> zero
        }
        let frac = frac | 0x0080_0000; // implicit bit
        let shift = (14 - e) as u32;
        let half = (frac >> shift) as u16;
        let round = ((frac >> (shift - 1)) & 1) as u16;
        return sign | (half + round);
    }
    let half = (((e as u32) << 10) | (frac >> 13)) as u16;
    let round = ((frac >> 12) & 1) as u16;
    sign | (half + round)
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32_vec(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

impl Model {
    /// Dequantize all weights to dense f32. Trades memory (~2-3x the RSS)
    /// for decode speed: quantized weights are otherwise unpacked on every
    /// use, which the decoder's per-token logits projection feels most.
    pub fn densify(&mut self) {
        for w in self.tensors.values_mut() {
            if let Weight::Quant(q) = w {
                *w = Weight::Dense(q.to_dense());
            }
        }
    }
}

pub fn load_model(r: &mut impl Read) -> io::Result<Model> {
    let magic = read_i32(r)? as u32;
    #[cfg(feature = "gguf")]
    if magic == crate::gguf::GGUF_MAGIC {
        return crate::gguf::load(r);
    }
    if magic != GGML_MAGIC {
        #[cfg(not(feature = "gguf"))]
        if magic == 0x4655_4747 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "this is a GGUF file; rebuild with `--features gguf` to load it",
            ));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad magic {magic:#x}, expected 'ggml' ({GGML_MAGIC:#x})"),
        ));
    }

    let hp = HParams {
        n_vocab: read_i32(r)?,
        n_audio_ctx: read_i32(r)?,
        n_audio_state: read_i32(r)?,
        n_audio_head: read_i32(r)?,
        n_audio_layer: read_i32(r)?,
        n_text_ctx: read_i32(r)?,
        n_text_state: read_i32(r)?,
        n_text_head: read_i32(r)?,
        n_text_layer: read_i32(r)?,
        n_mels: read_i32(r)?,
        ftype: read_i32(r)?,
    };

    // Embedded mel filterbank.
    let n_mel = read_i32(r)? as usize;
    let n_fft_bins = read_i32(r)? as usize;
    let mel_filters = read_f32_vec(r, n_mel * n_fft_bins)?;

    // Vocab. The file may hold fewer tokens than hparams.n_vocab; whisper.cpp
    // synthesizes placeholder names for the rest — ids are what matter.
    let n_tokens = read_i32(r)? as usize;
    let mut vocab = Vec::with_capacity(hp.n_vocab as usize);
    for _ in 0..n_tokens {
        let len = read_i32(r)? as usize;
        let mut word = vec![0u8; len];
        r.read_exact(&mut word)?;
        vocab.push(word);
    }
    for i in n_tokens..hp.n_vocab as usize {
        vocab.push(format!("[_extra_token_{i}]").into_bytes());
    }

    // Tensors until EOF.
    let mut tensors = HashMap::new();
    loop {
        let n_dims = match read_i32(r) {
            Ok(v) => v as usize,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };
        let name_len = read_i32(r)? as usize;
        let dtype = read_i32(r)?;
        let mut dims = [1usize; 3];
        for d in dims.iter_mut().take(n_dims) {
            *d = read_i32(r)? as usize;
        }
        let mut name = vec![0u8; name_len];
        r.read_exact(&mut name)?;
        let name = String::from_utf8_lossy(&name).into_owned();

        let n_elems: usize = dims.iter().product();
        // ggml stores dims innermost-first (ne[0] = fastest-varying); flip to
        // our row-major convention where the last dim is fastest.
        let shape: Vec<usize> = dims[..n_dims.max(1)].iter().rev().cloned().collect();
        let weight = match dtype {
            0 => Weight::Dense(Tensor::from_vec(&shape, read_f32_vec(r, n_elems)?)),
            1 => {
                let mut bytes = vec![0u8; n_elems * 2];
                r.read_exact(&mut bytes)?;
                let data = bytes
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect();
                Weight::Dense(Tensor::from_vec(&shape, data))
            }
            t => {
                match block_bytes(t) {
                    Some(bb) => {
                        if !n_elems.is_multiple_of(QK) {
                            return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("tensor '{name}': {n_elems} elements not divisible by block size"),
                        ));
                        }
                        let mut raw = vec![0u8; n_elems / QK * bb];
                        r.read_exact(&mut raw)?;
                        let qt = QTensor {
                            shape: shape.clone(),
                            dtype: t,
                            raw,
                        };
                        if n_dims == 2 {
                            // Matmul weights: keep quantized (see quant.rs).
                            Weight::Quant(qt)
                        } else {
                            Weight::Dense(qt.to_dense())
                        }
                    }
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("tensor '{name}': unsupported dtype {t}"),
                        ))
                    }
                }
            }
        };
        tensors.insert(name, weight);
    }

    Ok(Model {
        hparams: hp,
        mel_filters,
        vocab,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn f16_conversion_reference_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    /// Build a tiny synthetic model file and round-trip it.
    #[test]
    fn loads_synthetic_model() {
        let mut buf: Vec<u8> = Vec::new();
        let w32 = |b: &mut Vec<u8>, v: i32| b.extend_from_slice(&v.to_le_bytes());

        w32(&mut buf, GGML_MAGIC as i32);
        for v in [3, 1500, 8, 2, 4, 448, 8, 2, 4, 2, 1] {
            w32(&mut buf, v); // hparams: n_vocab=3, n_mels=2, ...
        }
        // mel filters: 2 x 3
        w32(&mut buf, 2);
        w32(&mut buf, 3);
        for v in [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // vocab: 2 tokens in file (hparams says 3 -> one synthesized)
        w32(&mut buf, 2);
        for word in [b"hi".as_slice(), b"yo".as_slice()] {
            w32(&mut buf, word.len() as i32);
            buf.extend_from_slice(word);
        }
        // one f32 tensor [ne0=2, ne1=3] named "w"
        w32(&mut buf, 2); // n_dims
        w32(&mut buf, 1); // name_len
        w32(&mut buf, 0); // f32
        w32(&mut buf, 2); // ne[0]
        w32(&mut buf, 3); // ne[1]
        buf.push(b'w');
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // one f16 tensor [ne0=2] named "b"
        w32(&mut buf, 1);
        w32(&mut buf, 1);
        w32(&mut buf, 1); // f16
        w32(&mut buf, 2);
        buf.push(b'b');
        buf.extend_from_slice(&0x3c00u16.to_le_bytes()); // 1.0
        buf.extend_from_slice(&0xc000u16.to_le_bytes()); // -2.0

        let m = load_model(&mut Cursor::new(buf)).unwrap();
        assert_eq!(m.hparams.n_vocab, 3);
        assert_eq!(m.hparams.model_type(), "tiny");
        assert!(!m.hparams.is_multilingual());
        assert_eq!(m.mel_filters, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(m.vocab.len(), 3);
        assert_eq!(m.vocab[1], b"yo");
        // dims flipped to row-major: [3, 2]
        let w = m.tensors["w"].dense();
        assert_eq!(w.shape, vec![3, 2]);
        assert_eq!(w.data, vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(m.tensors["b"].dense().data, vec![1.0, -2.0]);
    }

    #[test]
    fn dequant_known_blocks_via_qtensor() {
        // Q8_0: d = 0.5, quants 0, 1, -2, then 3s.
        let mut raw = vec![0u8; 34];
        raw[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        raw[2] = 0;
        raw[3] = 1;
        raw[4] = (-2i8) as u8;
        for b in raw[5..34].iter_mut() {
            *b = 3;
        }
        let q8 = QTensor {
            shape: vec![1, 32],
            dtype: 8,
            raw,
        };
        let d = q8.to_dense();
        assert_eq!(&d.data[..3], &[0.0, 0.5, -1.0]);
        assert_eq!(d.data[31], 1.5);

        // Q4_0: d = 1.0; nibble pair (low=8 -> 0.0, high=15 -> 7.0).
        let mut raw = vec![0u8; 18];
        raw[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        raw[2] = 0xF8;
        let d = QTensor {
            shape: vec![1, 32],
            dtype: 2,
            raw,
        }
        .to_dense();
        assert_eq!(d.data[0], 0.0);
        assert_eq!(d.data[16], 7.0);
        assert_eq!(d.data[1], -8.0);

        // Q5_0: element 0 has its high bit set (q=16 -> 0.0), element 16
        // does not (q=0 -> -16.0).
        let mut raw = vec![0u8; 22];
        raw[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        raw[2..6].copy_from_slice(&1u32.to_le_bytes());
        let d = QTensor {
            shape: vec![1, 32],
            dtype: 6,
            raw,
        }
        .to_dense();
        assert_eq!(d.data[0], 0.0);
        assert_eq!(d.data[16], -16.0);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // exact f16-representable values by design
    fn f16_round_trip() {
        for v in [0.0f32, 1.0, -2.0, 0.333251953125, 65504.0, 6.1035156e-5] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "round-trip {v}");
        }
        assert!(f16_to_f32(f32_to_f16(1e20)).is_infinite());
    }

    #[test]
    fn loads_quantized_tensor_from_synthetic_file() {
        let mut buf: Vec<u8> = Vec::new();
        let w32 = |b: &mut Vec<u8>, v: i32| b.extend_from_slice(&v.to_le_bytes());
        w32(&mut buf, GGML_MAGIC as i32);
        for v in [3, 1500, 8, 2, 4, 448, 8, 2, 4, 2, 8] {
            w32(&mut buf, v);
        }
        w32(&mut buf, 0); // no mel filters
        w32(&mut buf, 0);
        w32(&mut buf, 0); // no vocab tokens in file (3 synthesized)
                          // one 2-D q8_0 tensor [ne0=32, ne1=1], d = 1.0, quants all 2
        w32(&mut buf, 2);
        w32(&mut buf, 1);
        w32(&mut buf, 8); // dtype q8_0
        w32(&mut buf, 32);
        w32(&mut buf, 1);
        buf.push(b'q');
        buf.extend_from_slice(&0x3c00u16.to_le_bytes());
        buf.extend_from_slice(&[2u8; 32]);
        let m = load_model(&mut Cursor::new(buf)).unwrap();
        // 2-D quantized stays quantized; dequantizes to the expected values.
        match &m.tensors["q"] {
            Weight::Quant(qt) => {
                assert_eq!(qt.shape, vec![1, 32]);
                assert_eq!(qt.to_dense().data, vec![2.0; 32]);
            }
            Weight::Dense(_) => panic!("2-D quantized tensor should stay quantized"),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let buf = 0xdeadbeefu32.to_le_bytes().to_vec();
        assert!(load_model(&mut Cursor::new(buf)).is_err());
    }
}
