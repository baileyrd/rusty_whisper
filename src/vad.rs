//! Voice activity detection (VAD) preprocessing — port of whisper.cpp's
//! `--vad` flag, backed by a Silero VAD v4/v5 model converted to the same
//! legacy ggml binary layout the main Whisper model files use (see
//! upstream's `models/convert-silero-vad-to-ggml.py`; prebuilt files are
//! published at `huggingface.co/ggml-org/whisper-vad`).
//!
//! Three independent stages, mirroring `whisper_vad()` in whisper.cpp
//! v1.9.1:
//! 1. [`load_vad_model`] reads the file (same magic + generic
//!    `{n_dims, name, dtype, data}` tensor records [`crate::model`] uses,
//!    a different fixed header ahead of them).
//! 2. [`detect_speech_probs`] runs the network window-by-window (512
//!    samples/32ms @16kHz: reflect-padded STFT-via-conv frontend, 4 conv+
//!    ReLU encoder layers, a single LSTMCell whose hidden/cell state
//!    persists across windows, a linear+sigmoid head) to get a per-window
//!    speech probability.
//! 3. [`segments_from_probs`] turns those probabilities into speech
//!    segments via a hysteresis threshold plus duration/padding rules — a
//!    port of Silero's own `get_speech_timestamps` (`utils_vad.py`), which
//!    whisper.cpp's `whisper_vad_segments_from_probs` itself ports to C++.
//! 4. [`crop_and_map`]/[`map_processed_to_original_time`] concatenate the
//!    detected speech spans (whisper.cpp crops the *audio itself* before
//!    running the main encoder, not just marking timestamps) and remap the
//!    resulting transcript's timestamps back to the original timeline.
//!
//! Caveat: implemented from whisper.cpp's source layout and the published
//! Silero algorithm; not verified end-to-end against a real Silero ggml
//! file's reference output (none was available in this environment). The
//! network math and segmentation state machine are each unit-tested in
//! isolation against hand-derived values instead.

use std::collections::HashMap;
use std::io::{self, Read};

use crate::tensor::{conv1d, linear, Tensor};

pub const VAD_MAGIC: u32 = crate::model::GGML_MAGIC;

/// Fixed architectural constant (not stored in the file): Silero's 4-layer
/// conv encoder strides. Every published Silero v4/v5 ggml file has
/// exactly 4 encoder layers with these strides.
const ENCODER_STRIDES: [usize; 4] = [1, 2, 2, 1];

#[derive(Clone, Debug)]
pub struct VadHParams {
    pub window_size: usize,
    pub context_size: usize,
    /// Per encoder layer: `(in_ch, out_ch, kernel)`.
    pub encoder_layers: Vec<(usize, usize, usize)>,
    /// STFT-conv hop length and the LSTM cell's input width (same value).
    pub lstm_input_size: usize,
    pub lstm_hidden_size: usize,
}

pub struct VadModel {
    pub hparams: VadHParams,
    pub tensors: HashMap<String, Tensor>,
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_i32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_f32_vec(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Read a Silero VAD ggml model: fixed header (model-type string, version
/// triple, window/context size, per-layer encoder channel/kernel dims,
/// LSTM input/hidden size, final-conv in/out) followed by the generic
/// `{n_dims, name_len, dtype, dims[], name, data}` tensor records
/// `crate::model::load_model` also uses (same magic, same tensor-record
/// shape — whisper.cpp never migrated either model type to GGUF).
pub fn load_vad_model(r: &mut impl Read) -> io::Result<VadModel> {
    let magic = read_i32(r)? as u32;
    if magic != VAD_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad magic {magic:#x}, expected 'ggml' ({VAD_MAGIC:#x})"),
        ));
    }
    let _model_type = read_string(r)?;
    let _version = (read_i32(r)?, read_i32(r)?, read_i32(r)?);
    let window_size = read_i32(r)? as usize;
    let context_size = read_i32(r)? as usize;
    let n_encoder_layers = read_i32(r)? as usize;
    let mut encoder_layers = Vec::with_capacity(n_encoder_layers);
    for _ in 0..n_encoder_layers {
        let in_ch = read_i32(r)? as usize;
        let out_ch = read_i32(r)? as usize;
        let kernel = read_i32(r)? as usize;
        encoder_layers.push((in_ch, out_ch, kernel));
    }
    let lstm_input_size = read_i32(r)? as usize;
    let lstm_hidden_size = read_i32(r)? as usize;
    let _final_conv_in = read_i32(r)?;
    let _final_conv_out = read_i32(r)?;
    if encoder_layers.len() != ENCODER_STRIDES.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported VAD encoder layer count {} (expected {})",
                encoder_layers.len(),
                ENCODER_STRIDES.len()
            ),
        ));
    }

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

        let n_elems: usize = dims[..n_dims.max(1)].iter().product();
        let shape: Vec<usize> = dims[..n_dims.max(1)].iter().rev().cloned().collect();
        let data = match dtype {
            0 => read_f32_vec(r, n_elems)?,
            1 => {
                let mut bytes = vec![0u8; n_elems * 2];
                r.read_exact(&mut bytes)?;
                bytes
                    .chunks_exact(2)
                    .map(|c| crate::model::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect()
            }
            t => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("VAD tensor '{name}': unsupported dtype {t} (expected f32/f16)"),
                ))
            }
        };
        tensors.insert(name, Tensor::from_vec(&shape, data));
    }

    Ok(VadModel {
        hparams: VadHParams {
            window_size,
            context_size,
            encoder_layers,
            lstm_input_size,
            lstm_hidden_size,
        },
        tensors,
    })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reflect-pad a 1-D signal by `pad` samples on each side — numpy/PyTorch
/// "reflect" mode: mirrors without repeating the edge sample (`[3,2,1 | 1,
/// 2, 3, 4 | 3, 2]` for `pad=2`, not `[2,1 | ... | 4,3]`).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(x[pad - i]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[n - 2 - i]);
    }
    out
}

/// STFT-as-convolution frontend: correlate the (already reflect-padded)
/// window against the precomputed STFT basis, hop = the LSTM's input
/// width. `basis` is `[n_filters, 1, kernel]` (same `[out_ch, in_ch,
/// kernel]` layout `conv1d` expects) but this conv has no bias and no
/// implicit `kernel/2` padding, so it's a small dedicated loop rather than
/// a call to the shared `conv1d`.
fn stft_conv(padded: &[f32], basis: &Tensor, hop: usize) -> Tensor {
    let (n_filters, kernel) = (basis.shape[0], basis.shape[2]);
    let t = padded.len();
    let t_out = (t - kernel) / hop + 1;
    let mut out = Tensor::zeros(&[n_filters, t_out]);
    for f in 0..n_filters {
        let w = &basis.data[f * kernel..(f + 1) * kernel];
        for ot in 0..t_out {
            let start = ot * hop;
            let mut sum = 0.0f32;
            for (k, &wk) in w.iter().enumerate() {
                sum += wk * padded[start + k];
            }
            out.data[f * t_out + ot] = sum;
        }
    }
    out
}

struct LstmState {
    h: Vec<f32>,
    c: Vec<f32>,
}

impl LstmState {
    fn new(hidden: usize) -> Self {
        LstmState {
            h: vec![0.0; hidden],
            c: vec![0.0; hidden],
        }
    }
}

/// Standard PyTorch `LSTMCell` gate order `[i, f, g, o]`: `i,f,o` sigmoid,
/// `g` tanh, `c' = f*c + i*g`, `h' = o*tanh(c')`. `w_ih`/`w_hh` are
/// `[4*hidden, in]`/`[4*hidden, hidden]` (whisper's usual `[out, in]`
/// linear-weight layout), so each gate projection is just [`linear`].
fn lstm_cell(
    x: &[f32],
    state: &mut LstmState,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &[f32],
    b_hh: &[f32],
) {
    let hidden = state.h.len();
    let x_t = Tensor::from_vec(&[1, x.len()], x.to_vec());
    let h_t = Tensor::from_vec(&[1, hidden], state.h.clone());
    let gi = linear(&x_t, w_ih, Some(b_ih));
    let gh = linear(&h_t, w_hh, Some(b_hh));

    let i_gate: Vec<f32> = (0..hidden)
        .map(|k| sigmoid(gi.data[k] + gh.data[k]))
        .collect();
    let f_gate: Vec<f32> = (0..hidden)
        .map(|k| sigmoid(gi.data[hidden + k] + gh.data[hidden + k]))
        .collect();
    let g_gate: Vec<f32> = (0..hidden)
        .map(|k| (gi.data[2 * hidden + k] + gh.data[2 * hidden + k]).tanh())
        .collect();
    let o_gate: Vec<f32> = (0..hidden)
        .map(|k| sigmoid(gi.data[3 * hidden + k] + gh.data[3 * hidden + k]))
        .collect();

    for k in 0..hidden {
        let c_new = f_gate[k] * state.c[k] + i_gate[k] * g_gate[k];
        state.c[k] = c_new;
        state.h[k] = o_gate[k] * c_new.tanh();
    }
}

/// One window's forward pass: STFT-conv frontend -> magnitude -> 4x
/// (conv1d + ReLU) encoder -> first timestep -> LSTMCell (state persists
/// across calls) -> ReLU -> linear -> sigmoid speech probability.
fn forward_window(model: &VadModel, window: &[f32], state: &mut LstmState) -> f32 {
    let hp = &model.hparams;
    let padded = reflect_pad(window, hp.context_size);
    let basis = &model.tensors["_model.stft.forward_basis_buffer"];
    let stft = stft_conv(&padded, basis, hp.lstm_input_size);

    let n_freq = stft.shape[0] / 2;
    let t_frames = stft.shape[1];
    let mut mag = Tensor::zeros(&[n_freq, t_frames]);
    for f in 0..n_freq {
        for t in 0..t_frames {
            let re = stft.data[f * t_frames + t];
            let im = stft.data[(f + n_freq) * t_frames + t];
            mag.data[f * t_frames + t] = (re * re + im * im).sqrt();
        }
    }

    let mut cur = mag;
    for (i, stride) in ENCODER_STRIDES.iter().enumerate() {
        let w = &model.tensors[&format!("_model.encoder.{i}.reparam_conv.weight")];
        let b = &model.tensors[&format!("_model.encoder.{i}.reparam_conv.bias")];
        cur = conv1d(&cur, w, &b.data, *stride);
        for v in cur.data.iter_mut() {
            *v = v.max(0.0);
        }
    }

    // Only the first timestep feeds the LSTM (matches whisper.cpp's
    // `ggml_view_2d` slice — with the reference window/context sizes the
    // encoder output is already length 1, this is just explicit about it).
    let hidden_in = cur.shape[0];
    let t0_len = cur.shape[1];
    let x: Vec<f32> = (0..hidden_in).map(|c| cur.data[c * t0_len]).collect();

    let w_ih = &model.tensors["_model.decoder.rnn.weight_ih"];
    let w_hh = &model.tensors["_model.decoder.rnn.weight_hh"];
    let b_ih = &model.tensors["_model.decoder.rnn.bias_ih"].data;
    let b_hh = &model.tensors["_model.decoder.rnn.bias_hh"].data;
    lstm_cell(&x, state, w_ih, w_hh, b_ih, b_hh);

    let h_relu: Vec<f32> = state.h.iter().map(|v| v.max(0.0)).collect();
    let h_t = Tensor::from_vec(&[1, h_relu.len()], h_relu);
    let dec_w = &model.tensors["_model.decoder.decoder.2.weight"];
    let dec_b = &model.tensors["_model.decoder.decoder.2.bias"].data;
    let logit = linear(&h_t, dec_w, Some(dec_b));
    sigmoid(logit.data[0])
}

/// Run the VAD network over `samples`, one non-overlapping `window_size`
/// window at a time (the last window is zero-padded), LSTM state carried
/// across windows. Returns one speech probability per window.
pub fn detect_speech_probs(model: &VadModel, samples: &[f32]) -> Vec<f32> {
    let ws = model.hparams.window_size;
    if ws == 0 {
        return Vec::new();
    }
    let mut state = LstmState::new(model.hparams.lstm_hidden_size);
    let mut probs = Vec::with_capacity(samples.len().div_ceil(ws));
    let mut i = 0;
    while i < samples.len() {
        let end = (i + ws).min(samples.len());
        let mut window = vec![0.0f32; ws];
        window[..end - i].copy_from_slice(&samples[i..end]);
        probs.push(forward_window(model, &window, &mut state));
        i += ws;
    }
    probs
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VadSegment {
    pub start_sec: f32,
    pub end_sec: f32,
}

/// Mirrors whisper.cpp's `-vt/-vspd/-vsd/-vmsd/-vp` flags.
#[derive(Clone, Copy, Debug)]
pub struct VadOptions {
    pub threshold: f32,
    pub min_speech_duration_ms: u64,
    pub min_silence_duration_ms: u64,
    pub max_speech_duration_s: f32,
    pub speech_pad_ms: u64,
}

impl Default for VadOptions {
    fn default() -> Self {
        VadOptions {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            max_speech_duration_s: f32::MAX,
            speech_pad_ms: 30,
        }
    }
}

/// Turn per-window speech probabilities into speech segments: a
/// hysteresis threshold (`threshold` to start, `threshold - 0.15` clamped
/// to `>= 0.01` to end) plus minimum-speech/-silence duration filters, an
/// (approximate) max-speech-duration splitter, and start/end padding that
/// merges segments left closer together than twice the pad. Direct port of
/// Silero's own `get_speech_timestamps` (`utils_vad.py`), which
/// `whisper_vad_segments_from_probs` itself ports to C++ — including its
/// `0`-as-"unset" sentinel idiom for `temp_end`/`prev_end`/`next_start`,
/// replicated verbatim rather than turned into `Option` for clarity, so
/// the control flow stays a direct transliteration.
pub fn segments_from_probs(
    probs: &[f32],
    window_size: usize,
    sample_rate: usize,
    total_samples: usize,
    opts: &VadOptions,
) -> Vec<VadSegment> {
    let threshold = opts.threshold;
    let neg_threshold = (threshold - 0.15).max(0.01);
    let min_speech_samples = (sample_rate as u64 * opts.min_speech_duration_ms / 1000) as usize;
    let min_silence_samples = (sample_rate as u64 * opts.min_silence_duration_ms / 1000) as usize;
    let speech_pad_samples = (sample_rate as u64 * opts.speech_pad_ms / 1000) as usize;
    let max_speech_samples = (opts.max_speech_duration_s * sample_rate as f32) as usize;
    // Extra silence allowance while hunting for a split point inside an
    // over-long speech run (Silero hard-codes 98ms here).
    let min_silence_samples_at_max_speech = sample_rate * 98 / 1000;

    let mut triggered = false;
    let mut speeches: Vec<(usize, usize)> = Vec::new();
    let mut cur_start = 0usize;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        let t = i * window_size;

        if p >= threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = t;
            }
        }

        if p >= threshold && !triggered {
            triggered = true;
            cur_start = t;
            continue;
        }

        if triggered && t.saturating_sub(cur_start) > max_speech_samples {
            if prev_end != 0 {
                speeches.push((cur_start, prev_end));
                if next_start < prev_end {
                    triggered = false;
                } else {
                    cur_start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push((cur_start, t));
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                continue;
            }
        }

        if p < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = t;
            }
            if t.saturating_sub(temp_end) > min_silence_samples_at_max_speech {
                prev_end = temp_end;
            }
            if t.saturating_sub(temp_end) < min_silence_samples {
                continue;
            }
            if temp_end.saturating_sub(cur_start) > min_speech_samples {
                speeches.push((cur_start, temp_end));
            }
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    }

    if triggered {
        let end = (probs.len() * window_size).min(total_samples);
        if end.saturating_sub(cur_start) > min_speech_samples {
            speeches.push((cur_start, end));
        }
    }

    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            speeches[i].0 = speeches[i].0.saturating_sub(speech_pad_samples);
        }
        if i + 1 != n {
            let gap = speeches[i + 1].0.saturating_sub(speeches[i].1);
            if gap < 2 * speech_pad_samples {
                speeches[i].1 += gap / 2;
                speeches[i + 1].0 = speeches[i + 1].0.saturating_sub(gap / 2);
            } else {
                speeches[i].1 = (speeches[i].1 + speech_pad_samples).min(total_samples);
                speeches[i + 1].0 = speeches[i + 1].0.saturating_sub(speech_pad_samples);
            }
        } else {
            speeches[i].1 = (speeches[i].1 + speech_pad_samples).min(total_samples);
        }
    }

    speeches
        .into_iter()
        .map(|(s, e)| VadSegment {
            start_sec: s as f32 / sample_rate as f32,
            end_sec: e as f32 / sample_rate as f32,
        })
        .collect()
}

/// Concatenate speech segments (each extended by `overlap_s` past its
/// detected end, matching whisper.cpp's `--vad-samples-overlap`)
/// separated by 100ms of silence, and record a breakpoint table mapping
/// cropped-timeline seconds back to original-audio seconds — whisper.cpp
/// crops the sample buffer itself before running the encoder, it doesn't
/// just filter timestamps after the fact.
pub fn crop_and_map(
    samples: &[f32],
    segments: &[VadSegment],
    sample_rate: usize,
    overlap_s: f32,
) -> (Vec<f32>, Vec<(f32, f32)>) {
    const SILENCE_GAP_S: f32 = 0.1;
    let overlap_samples = (overlap_s * sample_rate as f32) as usize;
    let silence_samples = (SILENCE_GAP_S * sample_rate as f32) as usize;
    let mut out = Vec::new();
    let mut mapping = Vec::with_capacity(segments.len());
    for seg in segments {
        let start = (seg.start_sec * sample_rate as f32) as usize;
        let end =
            (((seg.end_sec * sample_rate as f32) as usize) + overlap_samples).min(samples.len());
        if start >= end {
            continue;
        }
        let processed_start = out.len() as f32 / sample_rate as f32;
        mapping.push((processed_start, seg.start_sec));
        out.extend_from_slice(&samples[start..end]);
        out.extend(std::iter::repeat_n(0.0f32, silence_samples));
    }
    (out, mapping)
}

/// Reverse [`crop_and_map`]'s breakpoint table: shift a timestamp in the
/// cropped/processed timeline back to the original audio's timeline. Falls
/// back to identity if `mapping` is empty (VAD found no speech at all).
pub fn map_processed_to_original_time(t: f32, mapping: &[(f32, f32)]) -> f32 {
    let mut anchor = match mapping.first() {
        Some(&a) => a,
        None => return t,
    };
    for &(proc_t, orig_t) in mapping {
        if proc_t <= t {
            anchor = (proc_t, orig_t);
        } else {
            break;
        }
    }
    anchor.1 + (t - anchor.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> VadOptions {
        VadOptions::default()
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let padded = reflect_pad(&x, 2);
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn lstm_cell_matches_hand_computed_gates() {
        // hidden=1, input=1: pin every weight/bias so the gate math is
        // checkable by hand instead of trusting the matmul plumbing.
        let w_ih = Tensor::from_vec(&[4, 1], vec![1.0, 1.0, 1.0, 1.0]);
        let w_hh = Tensor::from_vec(&[4, 1], vec![0.0, 0.0, 0.0, 0.0]);
        let b_ih = [0.0, 0.0, 0.0, 0.0];
        let b_hh = [0.0, 0.0, 0.0, 0.0];
        let mut state = LstmState::new(1);
        lstm_cell(&[0.0], &mut state, &w_ih, &w_hh, &b_ih, &b_hh);
        // x=0 -> all gate preactivations 0 -> i=f=o=sigmoid(0)=0.5, g=tanh(0)=0.
        // c' = f*0 + i*g = 0.5*0 + 0.5*0 = 0; h' = o*tanh(c') = 0.5*0 = 0.
        assert!((state.c[0] - 0.0).abs() < 1e-6);
        assert!((state.h[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stft_conv_shape_matches_hop_and_kernel() {
        // 1 filter, kernel=4, hop=2, input length 8 -> t_out = (8-4)/2+1 = 3.
        let basis = Tensor::from_vec(&[1, 1, 4], vec![1.0, 0.0, 0.0, 0.0]);
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = stft_conv(&input, &basis, 2);
        assert_eq!(out.shape, vec![1, 3]);
        // Filter is a delta at k=0, so each output is just input[start].
        assert_eq!(out.data, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn segments_from_probs_finds_a_single_speech_span() {
        // window_size=1 sample for arithmetic simplicity: 10 silent windows,
        // 5 speech, 10 silent. sample_rate chosen so min_speech/min_silence
        // filters (in samples) are tiny relative to the span.
        let mut probs = vec![0.0f32; 10];
        probs.extend(vec![0.9f32; 5]);
        probs.extend(vec![0.0f32; 10]);
        let o = VadOptions {
            min_speech_duration_ms: 0,
            min_silence_duration_ms: 0,
            speech_pad_ms: 0,
            ..opts()
        };
        let segs = segments_from_probs(&probs, 1, 1000, probs.len(), &o);
        assert_eq!(segs.len(), 1);
        assert!((segs[0].start_sec - 0.010).abs() < 1e-6);
        assert!((segs[0].end_sec - 0.015).abs() < 1e-6);
    }

    #[test]
    fn segments_from_probs_drops_speech_shorter_than_min_duration() {
        let mut probs = vec![0.0f32; 5];
        probs.extend(vec![0.9f32; 2]); // a 2-window blip
        probs.extend(vec![0.0f32; 5]);
        let o = VadOptions {
            min_speech_duration_ms: 5000, // way longer than the blip
            min_silence_duration_ms: 0,
            speech_pad_ms: 0,
            ..opts()
        };
        let segs = segments_from_probs(&probs, 1, 1000, probs.len(), &o);
        assert!(segs.is_empty());
    }

    #[test]
    fn segments_from_probs_pads_start_and_end() {
        let mut probs = vec![0.0f32; 10];
        probs.extend(vec![0.9f32; 5]);
        probs.extend(vec![0.0f32; 10]);
        let o = VadOptions {
            min_speech_duration_ms: 0,
            min_silence_duration_ms: 0,
            speech_pad_ms: 2, // 2 samples at sample_rate=1000 -> 0.002s
            ..opts()
        };
        let segs = segments_from_probs(&probs, 1, 1000, probs.len(), &o);
        assert_eq!(segs.len(), 1);
        assert!((segs[0].start_sec - 0.008).abs() < 1e-6);
        assert!((segs[0].end_sec - 0.017).abs() < 1e-6);
    }

    #[test]
    fn segments_from_probs_touches_but_does_not_overlap_when_gap_is_small() {
        // Two speech spans separated by a silence gap smaller than
        // 2*speech_pad_samples: padding must not let them overlap. Silero's
        // algorithm doesn't fuse the two entries into one, it just pulls
        // their boundaries in to meet exactly (this mirrors that, rather
        // than "fixing" it into a single merged segment).
        let mut probs = vec![0.0f32; 5];
        probs.extend(vec![0.9f32; 5]);
        probs.extend(vec![0.0f32; 2]); // short gap
        probs.extend(vec![0.9f32; 5]);
        probs.extend(vec![0.0f32; 5]);
        let o = VadOptions {
            min_speech_duration_ms: 0,
            min_silence_duration_ms: 0, // don't let the short gap alone close a segment early
            speech_pad_ms: 5,
            ..opts()
        };
        let segs = segments_from_probs(&probs, 1, 1000, probs.len(), &o);
        assert_eq!(segs.len(), 2);
        assert_eq!(
            segs[0].end_sec, segs[1].start_sec,
            "boundaries meet, don't overlap"
        );
        assert!((segs[0].start_sec - 0.0).abs() < 1e-6);
        assert!((segs[0].end_sec - 0.011).abs() < 1e-6);
        assert!((segs[1].end_sec - 0.022).abs() < 1e-6);
    }

    #[test]
    fn segments_from_probs_empty_input_yields_no_segments() {
        let segs = segments_from_probs(&[], 1, 16000, 0, &opts());
        assert!(segs.is_empty());
    }

    #[test]
    fn crop_and_map_concatenates_segments_with_silence_gaps() {
        let samples: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let segs = [
            VadSegment {
                start_sec: 0.0,
                end_sec: 0.005,
            }, // [0,5)
            VadSegment {
                start_sec: 0.010,
                end_sec: 0.015,
            }, // [10,15)
        ];
        let (cropped, mapping) = crop_and_map(&samples, &segs, 1000, 0.0);
        // 5 samples + 100 silence + 5 samples = 110, then trailing silence.
        assert_eq!(cropped.len(), 5 + 100 + 5 + 100);
        assert_eq!(&cropped[..5], &samples[0..5]);
        assert_eq!(&cropped[105..110], &samples[10..15]);
        assert_eq!(mapping, vec![(0.0, 0.0), (0.105, 0.010)]);
    }

    #[test]
    fn map_processed_to_original_time_shifts_by_the_right_segment_anchor() {
        let mapping = vec![(0.0, 1.0), (0.105, 5.0)];
        // Inside the first segment: offset from its anchor.
        assert!((map_processed_to_original_time(0.002, &mapping) - 1.002).abs() < 1e-6);
        // Inside the second segment: offset from *its* anchor, not the first.
        assert!((map_processed_to_original_time(0.106, &mapping) - 5.001).abs() < 1e-6);
    }

    #[test]
    fn map_processed_to_original_time_identity_when_no_segments() {
        assert_eq!(map_processed_to_original_time(3.0, &[]), 3.0);
    }

    /// Build a tiny synthetic VAD ggml file (4 encoder layers, tiny
    /// channel counts) and round-trip it through the loader, matching
    /// `model.rs`'s `loads_synthetic_model` pattern.
    fn synthetic_vad_model_bytes() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        let w32 = |b: &mut Vec<u8>, v: i32| b.extend_from_slice(&v.to_le_bytes());
        let wstr = |b: &mut Vec<u8>, s: &str| {
            w32(b, s.len() as i32);
            b.extend_from_slice(s.as_bytes());
        };
        let wf32 = |b: &mut Vec<u8>, v: f32| b.extend_from_slice(&v.to_le_bytes());

        w32(&mut buf, VAD_MAGIC as i32);
        wstr(&mut buf, "silero-16k-test");
        for v in [1, 0, 0] {
            w32(&mut buf, v); // version
        }
        w32(&mut buf, 6); // window_size
        w32(&mut buf, 1); // context_size
        w32(&mut buf, 4); // n_encoder_layers
        for _ in 0..4 {
            w32(&mut buf, 2); // in_ch
            w32(&mut buf, 2); // out_ch
            w32(&mut buf, 3); // kernel
        }
        w32(&mut buf, 2); // lstm_input_size (also stft hop here)
        w32(&mut buf, 2); // lstm_hidden_size
        w32(&mut buf, 2); // final_conv_in (unused by the loader beyond parsing)
        w32(&mut buf, 1); // final_conv_out

        let write_tensor =
            |buf: &mut Vec<u8>, name: &str, dims: &[i32], dtype: i32, data: &[f32]| {
                w32(buf, dims.len() as i32);
                w32(buf, name.len() as i32);
                w32(buf, dtype);
                for &d in dims {
                    w32(buf, d);
                }
                buf.extend_from_slice(name.as_bytes());
                for &v in data {
                    wf32(buf, v);
                }
            };

        // STFT basis: kernel=4, in_ch=1, 4 filters -> ne=[kernel,in_ch,n_filters]=[4,1,4]
        // (fastest-first; kernel and n_filters happen to match here, but the
        // loader's dim-reversal still lands filters on shape[0] as expected).
        write_tensor(
            &mut buf,
            "_model.stft.forward_basis_buffer",
            &[4, 1, 4],
            0,
            &[
                0.1, 0.2, -0.1, 0.05, // filter 0
                0.0, 0.3, 0.1, -0.2, // filter 1
                0.2, -0.1, 0.05, 0.1, // filter 2 (imag half starts here, n_filters/2=2)
                -0.1, 0.1, 0.2, 0.0, // filter 3
            ],
        );
        for i in 0..4usize {
            write_tensor(
                &mut buf,
                &format!("_model.encoder.{i}.reparam_conv.weight"),
                &[3, 2, 2], // ne = [kernel, in_ch, out_ch]
                0,
                &(0..12)
                    .map(|k| 0.05 * (k as f32 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect::<Vec<_>>(),
            );
            write_tensor(
                &mut buf,
                &format!("_model.encoder.{i}.reparam_conv.bias"),
                &[2],
                0,
                &[0.01, -0.01],
            );
        }
        write_tensor(
            &mut buf,
            "_model.decoder.rnn.weight_ih",
            &[2, 8],
            0,
            &(0..16).map(|k| 0.02 * k as f32).collect::<Vec<_>>(),
        );
        write_tensor(
            &mut buf,
            "_model.decoder.rnn.weight_hh",
            &[2, 8],
            0,
            &(0..16).map(|k| -0.02 * k as f32).collect::<Vec<_>>(),
        );
        write_tensor(&mut buf, "_model.decoder.rnn.bias_ih", &[8], 0, &[0.0; 8]);
        write_tensor(&mut buf, "_model.decoder.rnn.bias_hh", &[8], 0, &[0.0; 8]);
        write_tensor(
            &mut buf,
            "_model.decoder.decoder.2.weight",
            &[2, 1],
            0,
            &[0.3, -0.4],
        );
        write_tensor(&mut buf, "_model.decoder.decoder.2.bias", &[1], 0, &[0.1]);

        buf
    }

    #[test]
    fn loads_synthetic_vad_model() {
        let buf = synthetic_vad_model_bytes();
        let m = load_vad_model(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(m.hparams.window_size, 6);
        assert_eq!(m.hparams.context_size, 1);
        assert_eq!(m.hparams.encoder_layers, vec![(2, 2, 3); 4]);
        assert_eq!(m.hparams.lstm_hidden_size, 2);
        assert_eq!(
            m.tensors["_model.stft.forward_basis_buffer"].shape,
            vec![4, 1, 4]
        );
        assert_eq!(
            m.tensors["_model.decoder.decoder.2.weight"].shape,
            vec![1, 2]
        );
    }

    #[test]
    fn rejects_bad_vad_magic() {
        let mut buf = synthetic_vad_model_bytes();
        buf[0] = 0xff; // corrupt the magic
        let err = load_vad_model(&mut std::io::Cursor::new(buf))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn detect_speech_probs_end_to_end_synthetic_model() {
        let buf = synthetic_vad_model_bytes();
        let m = load_vad_model(&mut std::io::Cursor::new(buf)).unwrap();
        // 3 full windows (window_size=6) plus one partial (zero-padded) window.
        let samples: Vec<f32> = (0..20).map(|i| (i as f32 * 0.1).sin()).collect();
        let probs = detect_speech_probs(&m, &samples);
        assert_eq!(probs.len(), 4);
        for &p in &probs {
            assert!(
                (0.0..=1.0).contains(&p) && p.is_finite(),
                "prob out of range: {p}"
            );
        }
        // Recurrent state means the two silent (all-zero) windows below
        // shouldn't produce identical probabilities: an LSTM cell whose
        // state never advances would collapse to a constant output.
        let silent = vec![0.0f32; 12];
        let probs_silent = detect_speech_probs(&m, &silent);
        assert_eq!(probs_silent.len(), 2);
        assert_ne!(
            probs_silent[0], probs_silent[1],
            "identical output on identical input+state suggests the LSTM state isn't advancing"
        );
    }
}
