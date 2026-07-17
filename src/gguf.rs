//! GGUF container support (feature `gguf`, off by default).
//!
//! GGUF is llama.cpp's successor to the legacy ggml container: a
//! versioned header, typed key/value metadata, tensor descriptors with
//! explicit offsets, and an aligned data section. whisper.cpp itself
//! still distributes legacy `.bin` files and defines no official
//! whisper-GGUF schema, so the metadata mapping here is ours:
//!
//! | key | type | meaning |
//! |---|---|---|
//! | `general.architecture` | string | `"whisper"` |
//! | `general.alignment` | u32 | data alignment (32) |
//! | `whisper.n_vocab` ... `whisper.n_mels`, `whisper.ftype` | i32 | the 11 hparams, same names as the struct fields |
//! | `tokenizer.ggml.tokens` | array of string | vocab, byte-level |
//!
//! Mel filters travel as a tensor named `whisper.mel_filters`
//! (`[n_mels, n_fft_bins]` f32); model tensors keep their whisper.cpp
//! names and ggml type ids (quantized blocks are copied verbatim, so a
//! `.bin -> .gguf` conversion is lossless for quantized weights; f16
//! tensors are stored as f32, since the loader converts them anyway).
//!
//! `rusty-whisper --model x.bin --convert-gguf x.gguf` converts;
//! `load_model` sniffs the magic, so `.gguf` files load transparently.

use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::model::{HParams, Model};
use crate::quant::{block_bytes, QTensor, Weight, QK};
use crate::tensor::Tensor;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

// GGUF metadata value type ids.
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;

const HPARAM_KEYS: [&str; 11] = [
    "whisper.n_vocab",
    "whisper.n_audio_ctx",
    "whisper.n_audio_state",
    "whisper.n_audio_head",
    "whisper.n_audio_layer",
    "whisper.n_text_ctx",
    "whisper.n_text_state",
    "whisper.n_text_head",
    "whisper.n_text_layer",
    "whisper.n_mels",
    "whisper.ftype",
];

fn hparam_values(hp: &HParams) -> [i32; 11] {
    [
        hp.n_vocab,
        hp.n_audio_ctx,
        hp.n_audio_state,
        hp.n_audio_head,
        hp.n_audio_layer,
        hp.n_text_ctx,
        hp.n_text_state,
        hp.n_text_head,
        hp.n_text_layer,
        hp.n_mels,
        hp.ftype,
    ]
}

fn err(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

// ---------------------------------------------------------------- reading

/// Metadata values we understand; everything else is skipped on read.
enum Value {
    I64(i64),
    Strings(Vec<Vec<u8>>),
    Other,
}

struct Parser<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(err("gguf: truncated file".into()));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> io::Result<Vec<u8>> {
        let len = self.u64()? as usize;
        Ok(self.take(len)?.to_vec())
    }

    /// Parse (or skip) one metadata value of type `t`.
    fn value(&mut self, t: u32) -> io::Result<Value> {
        Ok(match t {
            0 | 1 => Value::I64(self.take(1)?[0] as i64),
            2 | 3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            T_U32 => Value::I64(self.u32()? as i64),
            T_I32 => Value::I64(self.u32()? as i32 as i64),
            T_F32 => {
                self.take(4)?;
                Value::Other
            }
            7 => Value::I64(self.take(1)?[0] as i64),
            T_STRING => Value::Strings(vec![self.string()?]),
            T_ARRAY => {
                let elem_t = self.u32()?;
                let n = self.u64()? as usize;
                if elem_t == T_STRING {
                    let mut v = Vec::with_capacity(n);
                    for _ in 0..n {
                        v.push(self.string()?);
                    }
                    Value::Strings(v)
                } else {
                    for _ in 0..n {
                        self.value(elem_t)?;
                    }
                    Value::Other
                }
            }
            T_U64 | 11 => Value::I64(self.u64()? as i64),
            12 => {
                self.take(8)?;
                Value::Other
            }
            t => return Err(err(format!("gguf: unknown metadata type {t}"))),
        })
    }
}

/// Load a GGUF whisper model. `r` is positioned just after the 4-byte
/// magic (consumed by `load_model`'s sniffing).
pub fn load(r: &mut impl Read) -> io::Result<Model> {
    // Sequential Read only, so pull the rest of the file into memory and
    // parse in place (tensor offsets require random access).
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    let mut p = Parser { buf: &buf, pos: 0 };

    let version = p.u32()?;
    if !(2..=3).contains(&version) {
        return Err(err(format!("gguf: unsupported version {version}")));
    }
    let n_tensors = p.u64()? as usize;
    let n_kv = p.u64()? as usize;

    let mut ints: HashMap<String, i64> = HashMap::new();
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    let mut alignment = ALIGNMENT;
    for _ in 0..n_kv {
        let key = String::from_utf8_lossy(&p.string()?).into_owned();
        let t = p.u32()?;
        match p.value(t)? {
            Value::I64(v) => {
                if key == "general.alignment" {
                    alignment = v as usize;
                }
                ints.insert(key, v);
            }
            Value::Strings(v) => {
                if key == "tokenizer.ggml.tokens" {
                    tokens = v;
                } else if key == "general.architecture"
                    && v.first().map(|s| s.as_slice()) != Some(b"whisper")
                {
                    return Err(err("gguf: general.architecture is not \"whisper\"".into()));
                }
            }
            Value::Other => {}
        }
    }

    let mut hp = HParams::default();
    let fields: [&mut i32; 11] = [
        &mut hp.n_vocab,
        &mut hp.n_audio_ctx,
        &mut hp.n_audio_state,
        &mut hp.n_audio_head,
        &mut hp.n_audio_layer,
        &mut hp.n_text_ctx,
        &mut hp.n_text_state,
        &mut hp.n_text_head,
        &mut hp.n_text_layer,
        &mut hp.n_mels,
        &mut hp.ftype,
    ];
    for (key, field) in HPARAM_KEYS.iter().zip(fields) {
        *field = *ints
            .get(*key)
            .ok_or_else(|| err(format!("gguf: missing metadata key {key}")))?
            as i32;
    }

    // Tensor descriptors, then the aligned data section.
    struct Info {
        name: String,
        shape: Vec<usize>,
        dtype: i32,
        offset: usize,
    }
    let mut infos = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = String::from_utf8_lossy(&p.string()?).into_owned();
        let n_dims = p.u32()? as usize;
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            ne.push(p.u64()? as usize);
        }
        let dtype = p.u32()? as i32;
        let offset = p.u64()? as usize;
        // GGUF dims are fastest-varying first; flip to our row-major shape.
        ne.reverse();
        infos.push(Info {
            name,
            shape: ne,
            dtype,
            offset,
        });
    }
    // Alignment is relative to the file start, but our buffer begins after
    // the 4-byte magic that load_model already consumed.
    let abs = p.pos + 4;
    let data_start = abs.div_ceil(alignment) * alignment - 4;

    let mut vocab = tokens;
    for i in vocab.len()..hp.n_vocab as usize {
        vocab.push(format!("[_extra_token_{i}]").into_bytes());
    }

    let mut mel_filters = Vec::new();
    let mut tensors = HashMap::new();
    for info in infos {
        let n_elems: usize = info.shape.iter().product();
        let n_bytes = match info.dtype {
            0 => n_elems * 4,
            1 => n_elems * 2,
            t => {
                let bb = block_bytes(t).ok_or_else(|| {
                    err(format!(
                        "gguf: tensor '{}': unsupported type {t}",
                        info.name
                    ))
                })?;
                n_elems / QK * bb
            }
        };
        let start = data_start + info.offset;
        if start + n_bytes > buf.len() {
            return Err(err(format!("gguf: tensor '{}' out of bounds", info.name)));
        }
        let raw = &buf[start..start + n_bytes];
        let weight = match info.dtype {
            0 => Weight::Dense(Tensor::from_vec(
                &info.shape,
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )),
            1 => Weight::Dense(Tensor::from_vec(
                &info.shape,
                raw.chunks_exact(2)
                    .map(|c| crate::model::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
            )),
            t => {
                let qt = QTensor {
                    shape: info.shape.clone(),
                    dtype: t,
                    raw: raw.to_vec(),
                };
                if info.shape.len() == 2 {
                    Weight::Quant(qt)
                } else {
                    Weight::Dense(qt.to_dense())
                }
            }
        };
        if info.name == "whisper.mel_filters" {
            mel_filters = weight.dense().data.clone();
        } else {
            tensors.insert(info.name, weight);
        }
    }

    Ok(Model {
        hparams: hp,
        mel_filters,
        vocab,
        tensors,
    })
}

// ---------------------------------------------------------------- writing

struct Writer<W: Write> {
    w: W,
}

impl<W: Write> Writer<W> {
    fn u32(&mut self, v: u32) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn u64(&mut self, v: u64) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn string(&mut self, s: &[u8]) -> io::Result<()> {
        self.u64(s.len() as u64)?;
        self.w.write_all(s)
    }

    fn kv_i32(&mut self, key: &str, v: i32) -> io::Result<()> {
        self.string(key.as_bytes())?;
        self.u32(T_I32)?;
        self.u32(v as u32)
    }
}

/// Serialize a model as GGUF. Quantized weights are written verbatim;
/// dense weights as f32.
pub fn write(model: &Model, w: &mut impl Write) -> io::Result<()> {
    let mut out = Writer { w };
    // Deterministic tensor order.
    let mut names: Vec<&String> = model.tensors.keys().collect();
    names.sort();

    let has_filters = !model.mel_filters.is_empty();
    let n_tensors = names.len() + has_filters as usize;
    out.u32(GGUF_MAGIC)?;
    out.u32(VERSION)?;
    out.u64(n_tensors as u64)?;
    out.u64((3 + HPARAM_KEYS.len()) as u64)?; // arch, alignment, tokens + hparams

    out.string(b"general.architecture")?;
    out.u32(T_STRING)?;
    out.string(b"whisper")?;
    out.string(b"general.alignment")?;
    out.u32(T_U32)?;
    out.u32(ALIGNMENT as u32)?;
    for (key, v) in HPARAM_KEYS.iter().zip(hparam_values(&model.hparams)) {
        out.kv_i32(key, v)?;
    }
    out.string(b"tokenizer.ggml.tokens")?;
    out.u32(T_ARRAY)?;
    out.u32(T_STRING)?;
    out.u64(model.vocab.len() as u64)?;
    for tok in &model.vocab {
        out.string(tok)?;
    }

    // Tensor descriptors: mel filters first, then sorted model tensors.
    let filter_shape = [model.hparams.n_mels as usize, crate::audio::N_FREQS];
    let mut described: Vec<(&str, Vec<usize>, i32, usize)> = Vec::new(); // (name, shape, dtype, n_bytes)
    if has_filters {
        described.push((
            "whisper.mel_filters",
            filter_shape.to_vec(),
            0,
            model.mel_filters.len() * 4,
        ));
    }
    for name in &names {
        let (shape, dtype, n_bytes) = match &model.tensors[*name] {
            Weight::Dense(t) => (t.shape.clone(), 0, t.data.len() * 4),
            Weight::Quant(q) => (q.shape.clone(), q.dtype, q.raw.len()),
        };
        described.push((name.as_str(), shape, dtype, n_bytes));
    }

    let mut header = Vec::new();
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(described.len());
    {
        let mut h = Writer { w: &mut header };
        for (name, shape, dtype, n_bytes) in &described {
            h.string(name.as_bytes())?;
            h.u32(shape.len() as u32)?;
            for d in shape.iter().rev() {
                h.u64(*d as u64)?;
            }
            h.u32(*dtype as u32)?;
            h.u64(offset as u64)?;
            offsets.push(offset);
            offset = (offset + n_bytes).div_ceil(ALIGNMENT) * ALIGNMENT;
        }
    }
    out.w.write_all(&header)?;

    // Pad to the aligned data-section start (byte position counted
    // explicitly — `Write` gives us no cursor), then emit tensor data at
    // the recorded offsets.
    let prefix = gguf_prefix_len(model, &header);
    write_padding(out.w, prefix.div_ceil(ALIGNMENT) * ALIGNMENT - prefix)?;

    let mut cursor = 0usize;
    let mut idx = 0usize;
    if has_filters {
        write_padding(out.w, offsets[idx] - cursor)?;
        for v in &model.mel_filters {
            out.w.write_all(&v.to_le_bytes())?;
        }
        cursor = offsets[idx] + model.mel_filters.len() * 4;
        idx += 1;
    }
    for name in &names {
        write_padding(out.w, offsets[idx] - cursor)?;
        let n_bytes = match &model.tensors[*name] {
            Weight::Dense(t) => {
                for v in &t.data {
                    out.w.write_all(&v.to_le_bytes())?;
                }
                t.data.len() * 4
            }
            Weight::Quant(q) => {
                out.w.write_all(&q.raw)?;
                q.raw.len()
            }
        };
        cursor = offsets[idx] + n_bytes;
        idx += 1;
    }
    Ok(())
}

fn write_padding(w: &mut impl Write, n: usize) -> io::Result<()> {
    if n > 0 {
        w.write_all(&vec![0u8; n])?;
    }
    Ok(())
}

/// Byte length of everything before the data section: header + metadata +
/// tensor descriptors (`descriptors` already serialized).
fn gguf_prefix_len(model: &Model, descriptors: &[u8]) -> usize {
    let s = |b: usize| 8 + b; // string: u64 len + bytes
    let mut n = 4 + 4 + 8 + 8; // magic, version, counts
    n += s("general.architecture".len()) + 4 + s("whisper".len());
    n += s("general.alignment".len()) + 4 + 4;
    for key in HPARAM_KEYS {
        n += s(key.len()) + 4 + 4;
    }
    n += s("tokenizer.ggml.tokens".len()) + 4 + 4 + 8;
    for tok in &model.vocab {
        n += s(tok.len());
    }
    n + descriptors.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::load_model;
    use std::io::Cursor;

    /// A small model with dense + quantized tensors, filters, and vocab.
    fn sample_model() -> Model {
        let hp = HParams {
            n_vocab: 4,
            n_audio_ctx: 10,
            n_audio_state: 8,
            n_audio_head: 2,
            n_audio_layer: 1,
            n_text_ctx: 6,
            n_text_state: 8,
            n_text_head: 2,
            n_text_layer: 1,
            n_mels: 2,
            ftype: 1,
        };
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.conv1.bias".to_string(),
            Weight::Dense(Tensor::from_vec(&[3], vec![1.0, -2.0, 0.5])),
        );
        // One q8_0 tensor [2, 32]: d = 1.0, quants = row index + 1.
        let mut raw = Vec::new();
        for row in 0..2u8 {
            raw.extend_from_slice(&0x3c00u16.to_le_bytes());
            raw.extend_from_slice(&[row + 1; 32]);
        }
        tensors.insert(
            "decoder.token_embedding.weight".to_string(),
            Weight::Quant(QTensor {
                shape: vec![2, 32],
                dtype: 8,
                raw,
            }),
        );
        Model {
            hparams: hp,
            mel_filters: (0..2 * crate::audio::N_FREQS)
                .map(|i| i as f32 * 0.5)
                .collect(),
            vocab: vec![
                b"a".to_vec(),
                b"bc".to_vec(),
                Vec::new(),
                b"\xff\xfe".to_vec(),
            ],
            tensors,
        }
    }

    #[test]
    fn round_trip_via_load_model() {
        let m = sample_model();
        let mut bytes = Vec::new();
        write(&m, &mut bytes).unwrap();
        assert_eq!(&bytes[0..4], b"GGUF");
        // Through the magic-sniffing front door.
        let loaded = load_model(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(loaded.hparams, m.hparams);
        assert_eq!(loaded.vocab, m.vocab);
        assert_eq!(loaded.mel_filters, m.mel_filters);
        assert_eq!(loaded.tensors.len(), 2);
        assert_eq!(
            loaded.tensors["encoder.conv1.bias"].dense().data,
            vec![1.0, -2.0, 0.5]
        );
        match &loaded.tensors["decoder.token_embedding.weight"] {
            Weight::Quant(q) => {
                assert_eq!(q.dtype, 8);
                assert_eq!(q.shape, vec![2, 32]);
                let d = q.to_dense();
                assert_eq!(&d.data[..32], &[1.0f32; 32]);
                assert_eq!(&d.data[32..], &[2.0f32; 32]);
            }
            Weight::Dense(_) => panic!("quantized tensor should stay quantized"),
        }
    }

    #[test]
    fn rejects_wrong_architecture() {
        let m = sample_model();
        let mut bytes = Vec::new();
        write(&m, &mut bytes).unwrap();
        // Patch the architecture value "whisper" -> "whooper".
        let pos = bytes.windows(7).position(|w| w == b"whisper").unwrap();
        bytes[pos..pos + 7].copy_from_slice(b"whooper");
        assert!(load_model(&mut Cursor::new(bytes)).is_err());
    }
}
