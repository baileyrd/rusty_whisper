//! Whisper audio encoder — port of whisper.cpp's `whisper_encode`.
//!
//! mel [n_mels, 2*n_ctx] -> conv1 (s1) + GELU -> conv2 (s2) + GELU
//!   -> transpose to [n_ctx, n_state] -> + positional embedding
//!   -> n_audio_layer x { pre-LN self-attention, pre-LN MLP } -> ln_post
//! -> encoder output [n_ctx, n_state], consumed by the decoder's
//! cross-attention.

use crate::model::Model;
use crate::tensor::{conv1d, gelu, layernorm, linear, matmul, matmul_t, softmax, transpose, Tensor};

/// Look up a tensor by its whisper.cpp name, with a clear error.
fn t<'m>(model: &'m Model, name: &str) -> &'m Tensor {
    model
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("model file is missing tensor '{name}'"))
}

fn bias<'m>(model: &'m Model, name: &str) -> &'m [f32] {
    &t(model, name).data
}

/// Multi-head attention over already-projected q/k/v, each `[t, n_state]`.
/// `causal` masks position i from attending to j > i (decoder self-attn;
/// the encoder never masks). q may have fewer rows than k/v (incremental
/// decoding); rows of q are aligned to the *end* of the k/v sequence.
pub fn multi_head_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_head: usize, causal: bool) -> Tensor {
    let (t_q, n_state) = (q.shape[0], q.shape[1]);
    let t_kv = k.shape[0];
    assert_eq!(n_state % n_head, 0);
    assert_eq!(k.shape[1], n_state);
    assert_eq!(v.shape, k.shape);
    let dh = n_state / n_head;
    let scale = 1.0 / (dh as f32).sqrt();

    let slice_head = |x: &Tensor, rows: usize, h: usize| -> Tensor {
        let mut out = Tensor::zeros(&[rows, dh]);
        for r in 0..rows {
            out.data[r * dh..(r + 1) * dh]
                .copy_from_slice(&x.data[r * n_state + h * dh..r * n_state + (h + 1) * dh]);
        }
        out
    };

    let mut out = Tensor::zeros(&[t_q, n_state]);
    for h in 0..n_head {
        let qh = slice_head(q, t_q, h);
        let kh = slice_head(k, t_kv, h);
        let vh = slice_head(v, t_kv, h);
        // scores[i,j] = qh_i . kh_j * scale
        let mut scores = matmul_t(&qh, &kh);
        for s in scores.data.iter_mut() {
            *s *= scale;
        }
        if causal {
            // Query row i sits at absolute position (t_kv - t_q + i).
            let offset = t_kv - t_q;
            for i in 0..t_q {
                for j in (offset + i + 1)..t_kv {
                    scores.data[i * t_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        softmax(&mut scores);
        let oh = matmul(&scores, &vh);
        for r in 0..t_q {
            out.data[r * n_state + h * dh..r * n_state + (h + 1) * dh]
                .copy_from_slice(&oh.data[r * dh..(r + 1) * dh]);
        }
    }
    out
}

/// One pre-LN transformer self-attention sub-block (shared shape between
/// encoder and decoder; `prefix` like "encoder.blocks.3").
fn self_attention_block(model: &Model, x: &mut Tensor, prefix: &str, n_head: usize, causal: bool) {
    let mut cur = x.clone();
    layernorm(&mut cur, bias(model, &format!("{prefix}.attn_ln.weight")), bias(model, &format!("{prefix}.attn_ln.bias")));
    let q = linear(&cur, t(model, &format!("{prefix}.attn.query.weight")), Some(bias(model, &format!("{prefix}.attn.query.bias"))));
    // Whisper's key projection has no bias.
    let k = linear(&cur, t(model, &format!("{prefix}.attn.key.weight")), None);
    let v = linear(&cur, t(model, &format!("{prefix}.attn.value.weight")), Some(bias(model, &format!("{prefix}.attn.value.bias"))));
    let attn = multi_head_attention(&q, &k, &v, n_head, causal);
    let proj = linear(&attn, t(model, &format!("{prefix}.attn.out.weight")), Some(bias(model, &format!("{prefix}.attn.out.bias"))));
    for (xv, pv) in x.data.iter_mut().zip(&proj.data) {
        *xv += pv;
    }
}

/// One pre-LN MLP sub-block: ln -> fc(4x) -> gelu -> fc -> residual.
fn mlp_block(model: &Model, x: &mut Tensor, prefix: &str) {
    let mut cur = x.clone();
    layernorm(&mut cur, bias(model, &format!("{prefix}.mlp_ln.weight")), bias(model, &format!("{prefix}.mlp_ln.bias")));
    let mut h = linear(&cur, t(model, &format!("{prefix}.mlp.0.weight")), Some(bias(model, &format!("{prefix}.mlp.0.bias"))));
    gelu(&mut h);
    let out = linear(&h, t(model, &format!("{prefix}.mlp.2.weight")), Some(bias(model, &format!("{prefix}.mlp.2.bias"))));
    for (xv, ov) in x.data.iter_mut().zip(&out.data) {
        *xv += ov;
    }
}

/// Run the encoder. `mel` is `[n_mels, n_frames]` with
/// `n_frames == 2 * n_audio_ctx` (3000 for the standard 30 s window).
/// Returns `[n_audio_ctx, n_audio_state]`.
pub fn encode(model: &Model, mel: &Tensor) -> Tensor {
    let hp = &model.hparams;
    let (n_ctx, n_state, n_head) = (hp.n_audio_ctx as usize, hp.n_audio_state as usize, hp.n_audio_head as usize);
    assert_eq!(mel.shape[0], hp.n_mels as usize, "mel bands != model n_mels");
    assert_eq!(mel.shape[1], 2 * n_ctx, "mel frames must be 2*n_audio_ctx (pad_or_trim the audio)");

    // Conv stem.
    let mut cur = conv1d(mel, t(model, "encoder.conv1.weight"), bias(model, "encoder.conv1.bias"), 1);
    gelu(&mut cur);
    let mut cur = conv1d(&cur, t(model, "encoder.conv2.weight"), bias(model, "encoder.conv2.bias"), 2);
    gelu(&mut cur);

    // [n_state, n_ctx] -> [n_ctx, n_state], plus sinusoidal positions
    // (precomputed in the model file).
    let mut x = transpose(&cur);
    let pos = t(model, "encoder.positional_embedding");
    assert_eq!(pos.shape, vec![n_ctx, n_state]);
    for (xv, pv) in x.data.iter_mut().zip(&pos.data) {
        *xv += pv;
    }

    for l in 0..hp.n_audio_layer as usize {
        let prefix = format!("encoder.blocks.{l}");
        self_attention_block(model, &mut x, &prefix, n_head, false);
        mlp_block(model, &mut x, &prefix);
    }

    layernorm(&mut x, bias(model, "encoder.ln_post.weight"), bias(model, "encoder.ln_post.bias"));
    x
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::HParams;
    use std::collections::HashMap;

    /// Deterministic pseudo-random weights in [-0.1, 0.1].
    pub(crate) fn fill(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        let data = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 0.2 - 0.1
            })
            .collect();
        Tensor::from_vec(shape, data)
    }

    /// A tiny synthetic encoder-only model: n_mels=2, n_state=4, 2 heads,
    /// 1 layer, n_audio_ctx=3 (6 mel frames).
    pub(crate) fn toy_model() -> Model {
        let hp = HParams {
            n_vocab: 51864,
            n_audio_ctx: 3,
            n_audio_state: 4,
            n_audio_head: 2,
            n_audio_layer: 1,
            n_text_ctx: 4,
            n_text_state: 4,
            n_text_head: 2,
            n_text_layer: 1,
            n_mels: 2,
            ftype: 0,
        };
        let mut tensors = HashMap::new();
        let mut add = |name: &str, shape: &[usize], seed: u32| {
            tensors.insert(name.to_string(), fill(shape, seed));
        };
        add("encoder.conv1.weight", &[4, 2, 3], 1);
        add("encoder.conv1.bias", &[4], 2);
        add("encoder.conv2.weight", &[4, 4, 3], 3);
        add("encoder.conv2.bias", &[4], 4);
        add("encoder.positional_embedding", &[3, 4], 5);
        let p = "encoder.blocks.0";
        add(&format!("{p}.attn_ln.weight"), &[4], 6);
        add(&format!("{p}.attn_ln.bias"), &[4], 7);
        add(&format!("{p}.attn.query.weight"), &[4, 4], 8);
        add(&format!("{p}.attn.query.bias"), &[4], 9);
        add(&format!("{p}.attn.key.weight"), &[4, 4], 10);
        add(&format!("{p}.attn.value.weight"), &[4, 4], 11);
        add(&format!("{p}.attn.value.bias"), &[4], 12);
        add(&format!("{p}.attn.out.weight"), &[4, 4], 13);
        add(&format!("{p}.attn.out.bias"), &[4], 14);
        add(&format!("{p}.mlp_ln.weight"), &[4], 15);
        add(&format!("{p}.mlp_ln.bias"), &[4], 16);
        add(&format!("{p}.mlp.0.weight"), &[16, 4], 17);
        add(&format!("{p}.mlp.0.bias"), &[16], 18);
        add(&format!("{p}.mlp.2.weight"), &[4, 16], 19);
        add(&format!("{p}.mlp.2.bias"), &[4], 20);
        add("encoder.ln_post.weight", &[4], 21);
        add("encoder.ln_post.bias", &[4], 22);
        Model { hparams: hp, mel_filters: vec![], vocab: vec![], tensors }
    }

    #[test]
    fn encoder_output_shape_and_finiteness() {
        let m = toy_model();
        let mel = fill(&[2, 6], 99);
        let out = encode(&m, &mel);
        assert_eq!(out.shape, vec![3, 4]);
        assert!(out.data.iter().all(|v| v.is_finite()));
        // ln_post makes each row zero-mean/unit-var before weight/bias; with
        // our weights the output can't be all zeros.
        assert!(out.data.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn encoder_is_deterministic() {
        let m = toy_model();
        let mel = fill(&[2, 6], 99);
        assert_eq!(encode(&m, &mel).data, encode(&m, &mel).data);
    }

    #[test]
    fn mha_single_head_hand_computed() {
        // T=2, d=1, one head. q=[0,0] -> uniform attention; huge q -> peaked.
        let k = Tensor::from_vec(&[2, 1], vec![1.0, -1.0]);
        let v = Tensor::from_vec(&[2, 1], vec![10.0, 20.0]);
        let q0 = Tensor::from_vec(&[1, 1], vec![0.0]);
        let out = multi_head_attention(&q0, &k, &v, 1, false);
        assert!((out.data[0] - 15.0).abs() < 1e-4, "uniform attention averages v");
        let q_big = Tensor::from_vec(&[1, 1], vec![50.0]);
        let out = multi_head_attention(&q_big, &k, &v, 1, false);
        assert!((out.data[0] - 10.0).abs() < 1e-3, "peaked attention picks v[0]");
    }

    #[test]
    fn mha_causal_mask_blocks_future() {
        // v distinct per position; with causal mask, position 0 can only see
        // itself regardless of scores.
        let k = Tensor::from_vec(&[3, 1], vec![0.0, 100.0, 100.0]);
        let v = Tensor::from_vec(&[3, 1], vec![7.0, 8.0, 9.0]);
        let q = Tensor::from_vec(&[3, 1], vec![1.0, 1.0, 1.0]);
        let out = multi_head_attention(&q, &k, &v, 1, true);
        assert!((out.data[0] - 7.0).abs() < 1e-5);
        // Position 2 sees everything; scores favor k=100 positions.
        assert!(out.data[2] > 7.5);
    }

    #[test]
    fn mha_incremental_query_aligns_to_end() {
        // 1 query row against 3 kv rows must equal the last row of the full
        // 3-query causal pass — the invariant KV-cached decoding relies on.
        let k = fill(&[3, 4], 31);
        let v = fill(&[3, 4], 32);
        let q_full = fill(&[3, 4], 33);
        let full = multi_head_attention(&q_full, &k, &v, 2, true);
        let q_last = Tensor::from_vec(&[1, 4], q_full.data[8..12].to_vec());
        let inc = multi_head_attention(&q_last, &k, &v, 2, true);
        for (a, b) in inc.data.iter().zip(&full.data[8..12]) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
