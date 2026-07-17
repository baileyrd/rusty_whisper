//! Whisper text decoder — port of whisper.cpp's `whisper_decode` plus a
//! greedy sampling loop.
//!
//! Per step: token + learned positional embeddings -> n_text_layer x
//! { causal self-attention (KV-cached), cross-attention to the encoder
//! output, MLP } -> final LN -> logits via the tied token-embedding matrix.
//!
//! Cross-attention K/V depend only on the encoder output, so they are
//! projected once per audio window. Self-attention K/V grow one row per
//! decoded token in the cache.

use crate::encoder::{bias, mha_split_kv, mlp_block, multi_head_attention, split_heads, t};
use crate::model::Model;
use crate::quant::linear_w;
use crate::tensor::{layernorm, Tensor};
use crate::tokenizer::Tokenizer;

/// Per-layer, per-head cross-attention K/V — fixed for a whole audio
/// window and shared between beams via `Arc` (it is by far the largest
/// piece of decoder state).
struct CrossKv {
    k: Vec<Vec<Tensor>>,
    v: Vec<Vec<Tensor>>,
}

pub struct Decoder<'m> {
    model: &'m Model,
    cross: std::sync::Arc<CrossKv>,
    /// Per-layer self-attention K/V cache, `n_past` rows of `n_state`.
    self_k: Vec<Vec<f32>>,
    self_v: Vec<Vec<f32>>,
    n_past: usize,
}

impl<'m> Decoder<'m> {
    pub fn new(model: &'m Model, enc_out: &Tensor) -> Self {
        let n_layer = model.hparams.n_text_layer as usize;
        let mut cross_k = Vec::with_capacity(n_layer);
        let mut cross_v = Vec::with_capacity(n_layer);
        let n_head = model.hparams.n_text_head as usize;
        for l in 0..n_layer {
            let p = format!("decoder.blocks.{l}");
            // Like self-attention: key has no bias, value does.
            let k = linear_w(enc_out, t(model, &format!("{p}.cross_attn.key.weight")), None);
            let v = linear_w(
                enc_out,
                t(model, &format!("{p}.cross_attn.value.weight")),
                Some(bias(model, &format!("{p}.cross_attn.value.bias"))),
            );
            cross_k.push(split_heads(&k, n_head));
            cross_v.push(split_heads(&v, n_head));
        }
        Decoder {
            model,
            cross: std::sync::Arc::new(CrossKv { k: cross_k, v: cross_v }),
            self_k: vec![Vec::new(); n_layer],
            self_v: vec![Vec::new(); n_layer],
            n_past: 0,
        }
    }

    /// An independent decoding branch: shares the window's cross K/V,
    /// copies only the (small) self-attention cache. Used by beam search.
    pub fn fork(&self) -> Self {
        Decoder {
            model: self.model,
            cross: self.cross.clone(),
            self_k: self.self_k.clone(),
            self_v: self.self_v.clone(),
            n_past: self.n_past,
        }
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Reset the self-attention cache (cross K/V stay — same audio window).
    pub fn reset(&mut self) {
        for k in self.self_k.iter_mut() {
            k.clear();
        }
        for v in self.self_v.iter_mut() {
            v.clear();
        }
        self.n_past = 0;
    }

    /// Run `tokens` through the decoder as positions
    /// `n_past .. n_past + tokens.len()`, extending the KV cache.
    /// Returns logits `[tokens.len(), n_vocab]`.
    pub fn forward(&mut self, tokens: &[u32]) -> Tensor {
        let hidden = self.forward_hidden(tokens);
        self.project_logits(&hidden)
    }

    /// The final-layernormed hidden states `[tokens.len(), n_state]`,
    /// without the logits projection. Beam search batches the projection
    /// across beams (see [`Decoder::project_logits`]) — the tied token
    /// embedding is by far the largest matrix in the decoder, and
    /// projecting each beam separately re-reads (and for quantized
    /// weights, re-unpacks) all of it per beam per step.
    pub fn forward_hidden(&mut self, tokens: &[u32]) -> Tensor {
        let hp = &self.model.hparams;
        let (n_state, n_head) = (hp.n_text_state as usize, hp.n_text_head as usize);
        let n_tok = tokens.len();
        assert!(n_tok > 0);
        assert!(
            self.n_past + n_tok <= hp.n_text_ctx as usize,
            "decoder context overflow: {} + {n_tok} > {}",
            self.n_past,
            hp.n_text_ctx
        );

        // Token + positional embeddings (the token matrix may be quantized;
        // gather dequantizes just the needed rows).
        let emb = t(self.model, "decoder.token_embedding.weight");
        let pos = t(self.model, "decoder.positional_embedding").dense();
        let mut x = Tensor::zeros(&[n_tok, n_state]);
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            assert!(tok < hp.n_vocab as usize, "token id {tok} out of range");
            let row = &mut x.data[i * n_state..(i + 1) * n_state];
            emb.row_f32(tok, row);
            let p = &pos.data[(self.n_past + i) * n_state..(self.n_past + i + 1) * n_state];
            for (o, pv) in row.iter_mut().zip(p) {
                *o += pv;
            }
        }

        for l in 0..hp.n_text_layer as usize {
            let p = format!("decoder.blocks.{l}");

            // Causal self-attention over cache + new tokens.
            let mut cur = x.clone();
            layernorm(&mut cur, bias(self.model, &format!("{p}.attn_ln.weight")), bias(self.model, &format!("{p}.attn_ln.bias")));
            let q = linear_w(&cur, t(self.model, &format!("{p}.attn.query.weight")), Some(bias(self.model, &format!("{p}.attn.query.bias"))));
            let k_new = linear_w(&cur, t(self.model, &format!("{p}.attn.key.weight")), None);
            let v_new = linear_w(&cur, t(self.model, &format!("{p}.attn.value.weight")), Some(bias(self.model, &format!("{p}.attn.value.bias"))));
            self.self_k[l].extend_from_slice(&k_new.data);
            self.self_v[l].extend_from_slice(&v_new.data);
            let t_kv = self.n_past + n_tok;
            let k_all = Tensor::from_vec(&[t_kv, n_state], self.self_k[l].clone());
            let v_all = Tensor::from_vec(&[t_kv, n_state], self.self_v[l].clone());
            let attn = multi_head_attention(&q, &k_all, &v_all, n_head, true);
            let proj = linear_w(&attn, t(self.model, &format!("{p}.attn.out.weight")), Some(bias(self.model, &format!("{p}.attn.out.bias"))));
            for (xv, pv) in x.data.iter_mut().zip(&proj.data) {
                *xv += pv;
            }

            // Cross-attention to the (precomputed) encoder K/V.
            let mut cur = x.clone();
            layernorm(&mut cur, bias(self.model, &format!("{p}.cross_attn_ln.weight")), bias(self.model, &format!("{p}.cross_attn_ln.bias")));
            let q = linear_w(&cur, t(self.model, &format!("{p}.cross_attn.query.weight")), Some(bias(self.model, &format!("{p}.cross_attn.query.bias"))));
            let attn = mha_split_kv(&q, &self.cross.k[l], &self.cross.v[l], false);
            let proj = linear_w(&attn, t(self.model, &format!("{p}.cross_attn.out.weight")), Some(bias(self.model, &format!("{p}.cross_attn.out.bias"))));
            for (xv, pv) in x.data.iter_mut().zip(&proj.data) {
                *xv += pv;
            }

            mlp_block(self.model, &mut x, &p);
        }
        self.n_past += n_tok;

        layernorm(&mut x, bias(self.model, "decoder.ln.weight"), bias(self.model, "decoder.ln.bias"));
        x
    }

    /// Tied output head: logits = hidden . token_embedding^T. `hidden` may
    /// stack rows from multiple beams — the projection is stateless.
    pub fn project_logits(&self, hidden: &Tensor) -> Tensor {
        linear_w(hidden, t(self.model, "decoder.token_embedding.weight"), None)
    }
}

/// The initial token sequence for transcription without timestamps
/// (`sot [lang transcribe] no_timestamps`).
pub fn sot_sequence(tok: &Tokenizer, multilingual: bool, lang_id: u32) -> Vec<u32> {
    let mut seq = vec![tok.sot];
    if multilingual {
        seq.push(tok.lang_begin + lang_id);
        seq.push(tok.transcribe);
    }
    seq.push(tok.no_timestamps);
    seq
}

/// Greedy decoding of one 30 s window: feed the SOT sequence, then argmax
/// one token at a time until end-of-text or the context limit.
/// Returns only the sampled text tokens (no specials).
pub fn greedy_decode(model: &Model, enc_out: &Tensor, tok: &Tokenizer) -> Vec<u32> {
    let mut dec = Decoder::new(model, enc_out);
    let prompt = sot_sequence(tok, model.hparams.is_multilingual(), 0);
    // Whisper reserves half the text context for a window's output.
    let max_new = model.hparams.n_text_ctx as usize / 2 - prompt.len();

    let mut logits = dec.forward(&prompt);
    let mut out = Vec::new();
    for _ in 0..max_new {
        let row = &logits.data[(logits.shape[0] - 1) * logits.shape[1]..];
        // Timestamps are disabled and specials must not be sampled: restrict
        // to text tokens plus end-of-text. (The full suppression list and
        // temperature fallback arrive with PLAN.md phase 6.)
        let mut best = tok.eot;
        let mut best_v = row[tok.eot as usize];
        for (id, &v) in row.iter().enumerate().take(tok.eot as usize) {
            if v > best_v {
                best = id as u32;
                best_v = v;
            }
        }
        if best == tok.eot {
            break;
        }
        out.push(best);
        logits = dec.forward(&[best]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::tests::{fill, toy_model};
    use crate::model::{HParams, Model};
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    /// Extend the encoder toy model with decoder tensors.
    /// n_state=4, 2 heads, 1 layer, n_text_ctx=4, tiny vocab of 8.
    fn toy_model_full() -> Model {
        let mut m = toy_model();
        m.hparams = HParams { n_vocab: 8, ..m.hparams };
        let mut add = |tensors: &mut HashMap<String, crate::quant::Weight>, name: &str, shape: &[usize], seed: u32| {
            tensors.insert(name.to_string(), crate::quant::Weight::Dense(fill(shape, seed)));
        };
        let ts = &mut m.tensors;
        add(ts, "decoder.token_embedding.weight", &[8, 4], 40);
        add(ts, "decoder.positional_embedding", &[4, 4], 41);
        let p = "decoder.blocks.0";
        add(ts, &format!("{p}.attn_ln.weight"), &[4], 42);
        add(ts, &format!("{p}.attn_ln.bias"), &[4], 43);
        add(ts, &format!("{p}.attn.query.weight"), &[4, 4], 44);
        add(ts, &format!("{p}.attn.query.bias"), &[4], 45);
        add(ts, &format!("{p}.attn.key.weight"), &[4, 4], 46);
        add(ts, &format!("{p}.attn.value.weight"), &[4, 4], 47);
        add(ts, &format!("{p}.attn.value.bias"), &[4], 48);
        add(ts, &format!("{p}.attn.out.weight"), &[4, 4], 49);
        add(ts, &format!("{p}.attn.out.bias"), &[4], 50);
        add(ts, &format!("{p}.cross_attn_ln.weight"), &[4], 51);
        add(ts, &format!("{p}.cross_attn_ln.bias"), &[4], 52);
        add(ts, &format!("{p}.cross_attn.query.weight"), &[4, 4], 53);
        add(ts, &format!("{p}.cross_attn.query.bias"), &[4], 54);
        add(ts, &format!("{p}.cross_attn.key.weight"), &[4, 4], 55);
        add(ts, &format!("{p}.cross_attn.value.weight"), &[4, 4], 56);
        add(ts, &format!("{p}.cross_attn.value.bias"), &[4], 57);
        add(ts, &format!("{p}.cross_attn.out.weight"), &[4, 4], 58);
        add(ts, &format!("{p}.cross_attn.out.bias"), &[4], 59);
        add(ts, &format!("{p}.mlp_ln.weight"), &[4], 60);
        add(ts, &format!("{p}.mlp_ln.bias"), &[4], 61);
        add(ts, &format!("{p}.mlp.0.weight"), &[16, 4], 62);
        add(ts, &format!("{p}.mlp.0.bias"), &[16], 63);
        add(ts, &format!("{p}.mlp.2.weight"), &[4, 16], 64);
        add(ts, &format!("{p}.mlp.2.bias"), &[4], 65);
        add(ts, "decoder.ln.weight", &[4], 66);
        add(ts, "decoder.ln.bias", &[4], 67);
        m
    }

    fn toy_enc_out() -> Tensor {
        fill(&[3, 4], 99)
    }

    #[test]
    fn forward_logits_shape() {
        let m = toy_model_full();
        let mut dec = Decoder::new(&m, &toy_enc_out());
        let logits = dec.forward(&[0, 3]);
        assert_eq!(logits.shape, vec![2, 8]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
        assert_eq!(dec.n_past(), 2);
    }

    #[test]
    fn kv_cached_incremental_matches_full_forward() {
        // The invariant the whole decoding loop rests on: feeding tokens one
        // at a time through the cache gives the same final logits as feeding
        // them all at once.
        let m = toy_model_full();
        let tokens = [1u32, 5, 2, 7];

        let mut full = Decoder::new(&m, &toy_enc_out());
        let all = full.forward(&tokens);
        let last_full = &all.data[3 * 8..];

        let mut inc = Decoder::new(&m, &toy_enc_out());
        let mut last = Tensor::zeros(&[1, 8]);
        for &tk in &tokens {
            last = inc.forward(&[tk]);
        }
        for (a, b) in last.data.iter().zip(last_full) {
            assert!((a - b).abs() < 1e-4, "incremental {a} vs full {b}");
        }
    }

    #[test]
    fn reset_clears_self_cache_but_keeps_cross() {
        let m = toy_model_full();
        let mut dec = Decoder::new(&m, &toy_enc_out());
        let first = dec.forward(&[2]);
        dec.forward(&[4]).data.iter().for_each(|v| assert!(v.is_finite()));
        dec.reset();
        assert_eq!(dec.n_past(), 0);
        let again = dec.forward(&[2]);
        for (a, b) in first.data.iter().zip(&again.data) {
            assert!((a - b).abs() < 1e-6, "reset must reproduce the first step");
        }
    }

    #[test]
    fn split_forward_equals_combined() {
        // forward() must equal forward_hidden() + project_logits(), and the
        // projection must be batchable: stacking two hidden rows projects
        // to the same logits as projecting them separately.
        let m = toy_model_full();
        let mut a = Decoder::new(&m, &toy_enc_out());
        let combined = a.forward(&[1, 5]);
        let mut b = Decoder::new(&m, &toy_enc_out());
        let hidden = b.forward_hidden(&[1, 5]);
        let split = b.project_logits(&hidden);
        assert_eq!(combined.shape, split.shape);
        for (x, y) in combined.data.iter().zip(&split.data) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn forked_decoder_branches_independently() {
        let m = toy_model_full();
        // Branch after a shared prefix; each branch must equal a fresh
        // decoder fed its full sequence.
        let mut a = Decoder::new(&m, &toy_enc_out());
        a.forward(&[1, 5]);
        let mut b = a.fork();
        let la = a.forward(&[2]);
        let lb = b.forward(&[7]);

        let mut fresh_a = Decoder::new(&m, &toy_enc_out());
        fresh_a.forward(&[1, 5]);
        let want_a = fresh_a.forward(&[2]);
        let mut fresh_b = Decoder::new(&m, &toy_enc_out());
        fresh_b.forward(&[1, 5]);
        let want_b = fresh_b.forward(&[7]);

        for (x, y) in la.data.iter().zip(&want_a.data) {
            assert!((x - y).abs() < 1e-6);
        }
        for (x, y) in lb.data.iter().zip(&want_b.data) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn context_overflow_panics() {
        let m = toy_model_full();
        let mut dec = Decoder::new(&m, &toy_enc_out());
        dec.forward(&[0, 1, 2, 3]);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.forward(&[4])));
        assert!(r.is_err());
    }

    #[test]
    fn sot_sequence_layouts() {
        let tok_en = Tokenizer::new(vec![], &HParams { n_vocab: 51864, ..Default::default() });
        assert_eq!(sot_sequence(&tok_en, false, 0), vec![50257, 50362]);
        let tok_ml = Tokenizer::new(vec![], &HParams { n_vocab: 51865, ..Default::default() });
        assert_eq!(sot_sequence(&tok_ml, true, 0), vec![50258, 50259, 50359, 50363]);
    }
}
