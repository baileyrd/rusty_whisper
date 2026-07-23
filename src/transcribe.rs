//! Full transcription pipeline — port of whisper.cpp's `whisper_full`.
//!
//! Long audio is processed in 30 s windows. Each window is decoded with
//! timestamp tokens enabled, following OpenAI's timestamp rules; the window
//! then advances to the last decoded timestamp. Decoding is conditioned on
//! the previous window's text (via `<|startofprev|>`), and each window is
//! retried at increasing temperatures when the output fails quality checks
//! (repetition via compression ratio, or low average log-probability).
//! Beam search is a possible future refinement; whisper.cpp's default
//! strategy is exactly this greedy + fallback scheme.

use crate::audio;
use crate::decoder::Decoder;
use crate::encoder;
use crate::model::Model;
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

/// Per-token decode data for `--output-json-full`/`-ojf`-style consumers.
/// `t0`/`t1` are interpolated by the token's character position within its
/// segment (same approximation `Options::max_len` splitting uses) — precise
/// per-token alignment needs DTW/cross-attention data, not yet implemented.
#[derive(Clone, Debug)]
pub struct TokenInfo {
    pub id: u32,
    /// The token's own decoded piece (may include a leading space, as
    /// whisper.cpp's BPE-style vocab does).
    pub text: String,
    /// Softmax probability of this token at the step it was sampled.
    pub prob: f32,
    /// Log of `prob` (clamped the same way decode-quality scoring is).
    pub logprob: f32,
    pub t0: f32,
    pub t1: f32,
}

#[derive(Clone, Debug)]
pub struct Segment {
    /// Start/end in seconds, absolute over the whole input.
    pub t0: f32,
    pub t1: f32,
    pub text: String,
    /// Per-token breakdown of `text` (empty if the pipeline that produced
    /// this segment didn't populate it — currently always populated by
    /// `transcribe`/`Stream`).
    pub tokens: Vec<TokenInfo>,
}

#[derive(Clone)]
pub struct Options {
    /// ISO code for multilingual models ("de", "fr", ...); None = detect
    /// from the first window. Ignored by English-only models.
    pub language: Option<String>,
    /// Translate to English instead of transcribing (multilingual models).
    pub translate: bool,
    /// Beams for the temperature-0 decode (1 = greedy). The fallback
    /// ladder always samples greedily, as in whisper.cpp.
    pub beam_size: usize,
    /// Condition on the previous window's text.
    pub condition_on_past: bool,
    /// Temperature ladder for the fallback scheme.
    pub temperatures: Vec<f32>,
    /// Reject a decode whose text compresses better than this (repetition).
    pub compression_ratio_threshold: f32,
    /// Reject a decode whose mean token log-prob is below this.
    pub logprob_threshold: f32,
    /// First timestamp must be within this many seconds of the window start.
    pub max_initial_ts: f32,
    /// Cap segment length in characters by splitting long segments into
    /// several (0 = disabled, i.e. one segment per decoded timestamp span).
    /// Mirrors whisper.cpp's `--max-len`/`-ml`.
    pub max_len: usize,
    /// When splitting on `max_len`, break at word boundaries instead of at
    /// an arbitrary character offset. Mirrors `--split-on-word`/`-sow`.
    pub split_on_word: bool,
    /// Word-timestamp probability threshold. Currently unused: rusty_whisper
    /// doesn't yet compute per-word alignment probabilities (that lands with
    /// token-level/DTW timestamps); reserved so the option surface matches
    /// whisper.cpp's `--word-thold`/`-wt` ahead of that landing.
    pub word_thold: f32,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            language: None,
            translate: false,
            beam_size: 5,
            condition_on_past: true,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            max_initial_ts: 1.0,
            max_len: 0,
            split_on_word: false,
            word_thold: 0.01,
        }
    }
}

/// Crude LZ77-style compressibility estimate of `bytes`: original length
/// divided by the number of emitted literals/matches. Highly repetitive
/// text scores high — the same signal OpenAI gets from zlib.
pub fn compression_ratio(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 1.0;
    }
    // Cost model calibrated against zlib (what OpenAI divides by): literals
    // cost 1 byte, a match of length >= 4 costs ~4, plus ~12 bytes of
    // stream overhead. Without match/stream costs, one long repeat looks
    // infinitely compressible and legitimate repetition in the audio
    // (choruses, repeated phrases) falsely trips the quality gate.
    let mut encoded = 12usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let window_start = i.saturating_sub(1024);
        let mut best = 0usize;
        for s in window_start..i {
            let mut l = 0usize;
            while i + l < bytes.len() && bytes[s + l % (i - s)] == bytes[i + l] && l < 255 {
                l += 1;
            }
            best = best.max(l);
        }
        if best >= 4 {
            encoded += 4;
            i += best;
        } else {
            encoded += 1;
            i += 1;
        }
    }
    bytes.len() as f32 / encoded as f32
}

/// Deterministic LCG for temperature sampling.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

struct WindowDecode {
    tokens: Vec<u32>,
    /// Clamped log-probability of each sampled token, aligned 1:1 with `tokens`.
    token_logprobs: Vec<f32>,
    avg_logprob: f32,
}

/// Apply suppression + timestamp rules to one logits row, in place.
/// `n_sampled` counts tokens sampled so far this window.
#[allow(clippy::too_many_arguments)]
fn apply_rules(
    row: &mut [f32],
    tok: &Tokenizer,
    last: Option<u32>,
    second_last: Option<u32>,
    max_ts_seen: Option<u32>,
    n_sampled: usize,
    max_initial_ts_id: u32,
    blank_id: Option<u32>,
) {
    let ts_begin = tok.timestamp_begin as usize;

    let n_vocab = row.len();
    // Never sample non-timestamp specials (sot, language, task, notimestamps...).
    for v in row[tok.eot as usize + 1..ts_begin.min(n_vocab)].iter_mut() {
        *v = f32::NEG_INFINITY;
    }

    if n_sampled == 0 {
        // First token must be a timestamp near the window start; also
        // suppress blank/EOT openers.
        for (id, v) in row.iter_mut().enumerate() {
            let id = id as u32;
            let ok = tok.is_timestamp(id) && id <= max_initial_ts_id;
            if !ok {
                *v = f32::NEG_INFINITY;
            }
        }
        return;
    }
    if let Some(b) = blank_id {
        if n_sampled == 1 {
            row[b as usize] = f32::NEG_INFINITY;
        }
    }

    // Timestamp pairing: after a segment-closing timestamp the next token
    // must be a timestamp or EOT; after a segment-opening one, text. With
    // fewer than two sampled tokens the penultimate counts as a timestamp
    // (OpenAI's `len(tokens) < 2 or ...`) so the initial timestamp is
    // treated as an opener — getting this backwards forces a spurious
    // second timestamp that silently shifts every segment.
    let last_is_ts = last.map(|t| tok.is_timestamp(t)).unwrap_or(false);
    let second_is_ts = second_last.map(|t| tok.is_timestamp(t)).unwrap_or(true);
    if last_is_ts {
        if second_is_ts {
            for v in row[ts_begin..].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        } else {
            for v in row[..tok.eot as usize].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Timestamps never decrease.
    if let Some(mts) = max_ts_seen {
        let cut = if last_is_ts && !second_is_ts {
            mts
        } else {
            mts + 1
        };
        for v in row[ts_begin..(cut as usize).min(n_vocab)].iter_mut() {
            *v = f32::NEG_INFINITY;
        }
    }

    // If the total timestamp probability beats every text token, commit to a
    // timestamp (log-sum-exp over the timestamp range vs max text logit).
    let max_row = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_row.is_finite() {
        let ts_lse: f32 = max_row
            + row[ts_begin..]
                .iter()
                .map(|v| (v - max_row).exp())
                .sum::<f32>()
                .ln();
        let max_text = row[..ts_begin]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if ts_lse > max_text {
            for v in row[..ts_begin].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    row.iter().map(|v| v - lse).collect()
}

fn sample(row: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
    }
    let scaled: Vec<f32> = row.iter().map(|v| v / temperature).collect();
    let probs: Vec<f32> = log_softmax(&scaled).iter().map(|lp| lp.exp()).collect();
    let mut r = rng.next_f32();
    for (i, p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// The resolved decoding task for a run: language + transcribe/translate.
#[derive(Clone, Copy)]
struct Task {
    lang_id: u32,
    translate: bool,
}

/// Prompt: [sot_prev, past text...] + sot sequence (timestamps enabled,
/// so no <|notimestamps|>).
fn build_prompt(tok: &Tokenizer, model: &Model, prompt_past: &[u32], task: Task) -> Vec<u32> {
    let n_ctx_half = model.hparams.n_text_ctx as usize / 2;
    let mut prompt = Vec::new();
    if !prompt_past.is_empty() {
        prompt.push(tok.sot_prev);
        let keep = prompt_past.len().min(n_ctx_half - 1);
        prompt.extend_from_slice(&prompt_past[prompt_past.len() - keep..]);
    }
    prompt.push(tok.sot);
    if model.hparams.is_multilingual() {
        prompt.push(tok.lang_begin + task.lang_id);
        prompt.push(if task.translate {
            tok.translate
        } else {
            tok.transcribe
        });
    }
    prompt
}

/// Detect the spoken language from an encoded window: one decoder step on
/// `[sot]`, softmax restricted to the language tokens. Returns (lang_id,
/// probability).
pub fn detect_language(dec: &mut Decoder, tok: &Tokenizer) -> (u32, f32) {
    dec.reset();
    let logits = dec.forward(&[tok.sot]);
    dec.reset();
    let row = &logits.data[..logits.shape[1]];
    let lo = tok.lang_begin as usize;
    let hi = (lo + tok.n_langs as usize).min(row.len());
    let langs = &row[lo..hi];
    let max = langs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = langs.iter().map(|v| (v - max).exp()).sum();
    let (best, best_v) = langs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    (best as u32, (best_v - max).exp() / sum)
}

/// Keep the `k` largest (logprob, id) pairs from a row.
fn top_k(lps: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (id, &lp) in lps.iter().enumerate() {
        if lp == f32::NEG_INFINITY {
            continue;
        }
        if best.len() < k || lp > best.last().unwrap().0 {
            let pos = best.partition_point(|&(b, _)| b > lp);
            best.insert(pos, (lp, id as u32));
            best.truncate(k);
        }
    }
    best
}

/// Decode one window at a given temperature.
#[allow(clippy::too_many_arguments)]
fn decode_window(
    dec: &mut Decoder,
    tok: &Tokenizer,
    model: &Model,
    prompt_past: &[u32],
    opts: &Options,
    task: Task,
    temperature: f32,
    blank_id: Option<u32>,
) -> WindowDecode {
    let hp = &model.hparams;
    let n_ctx_half = hp.n_text_ctx as usize / 2;
    dec.reset();
    let prompt = build_prompt(tok, model, prompt_past, task);

    let max_initial_ts_id = tok.timestamp_begin + (opts.max_initial_ts / 0.02) as u32;
    let mut rng = Rng(42);
    let mut logits = dec.forward(&prompt);
    let mut tokens: Vec<u32> = Vec::new();
    let mut token_logprobs: Vec<f32> = Vec::new();
    let mut sum_logprob = 0.0f32;
    let mut max_ts_seen: Option<u32> = None;

    for _ in 0..n_ctx_half {
        let n_vocab = logits.shape[1];
        let row = &mut logits.data[(logits.shape[0] - 1) * n_vocab..];
        let last = tokens.last().copied();
        let second_last = tokens.len().checked_sub(2).map(|i| tokens[i]);
        apply_rules(
            row,
            tok,
            last,
            second_last,
            max_ts_seen,
            tokens.len(),
            max_initial_ts_id,
            blank_id,
        );

        let logprobs = log_softmax(row);
        let id = sample(row, temperature, &mut rng);
        let clamped_lp = logprobs[id as usize].max(-30.0);
        sum_logprob += clamped_lp;
        if id == tok.eot {
            break;
        }
        if tok.is_timestamp(id) {
            max_ts_seen = Some(max_ts_seen.map_or(id, |m| m.max(id)));
        }
        tokens.push(id);
        token_logprobs.push(clamped_lp);
        if dec.n_past() + 1 > hp.n_text_ctx as usize {
            break;
        }
        logits = dec.forward(&[id]);
    }

    let avg_logprob = sum_logprob / (tokens.len() + 1) as f32;
    WindowDecode {
        tokens,
        token_logprobs,
        avg_logprob,
    }
}

/// Beam-search decode of one window (temperature 0). Beams share the
/// window's cross-attention K/V and fork their self-attention caches;
/// scoring is cumulative log-probability, final selection by average.
fn decode_window_beam(
    dec: &mut Decoder,
    tok: &Tokenizer,
    model: &Model,
    prompt_past: &[u32],
    opts: &Options,
    task: Task,
    blank_id: Option<u32>,
) -> WindowDecode {
    struct Beam<'m> {
        dec: Decoder<'m>,
        tokens: Vec<u32>,
        token_logprobs: Vec<f32>,
        sum_lp: f32,
        row: Vec<f32>,
        max_ts: Option<u32>,
    }

    let hp = &model.hparams;
    let n_ctx_half = hp.n_text_ctx as usize / 2;
    let beam_size = opts.beam_size;
    dec.reset();
    let prompt = build_prompt(tok, model, prompt_past, task);
    let max_initial_ts_id = tok.timestamp_begin + (opts.max_initial_ts / 0.02) as u32;

    let n_state = hp.n_text_state as usize;
    let hidden = dec.forward_hidden(&prompt);
    let last_hidden = Tensor::from_vec(
        &[1, n_state],
        hidden.data[(hidden.shape[0] - 1) * n_state..].to_vec(),
    );
    let logits = dec.project_logits(&last_hidden);
    let n_vocab = logits.shape[1];
    let mut beams = vec![Beam {
        dec: dec.fork(),
        tokens: Vec::new(),
        token_logprobs: Vec::new(),
        sum_lp: 0.0,
        row: logits.data,
        max_ts: None,
    }];
    // (tokens, token_logprobs, sum_lp) of hypotheses that reached EOT (or the context cap).
    let mut finished: Vec<(Vec<u32>, Vec<f32>, f32)> = Vec::new();

    for _ in 0..n_ctx_half {
        // Candidates from every live beam, EOT continuations included; only
        // those ranking in the global top beam_size survive. Finalizing every
        // beam's EOT option unconditionally would flood `finished` with
        // confident-but-short hypotheses and end the search prematurely.
        let mut cands: Vec<(usize, u32, f32, f32)> = Vec::new(); // (parent, id, new_sum, step_lp)
        for (bi, b) in beams.iter_mut().enumerate() {
            let last = b.tokens.last().copied();
            let second_last = b.tokens.len().checked_sub(2).map(|i| b.tokens[i]);
            apply_rules(
                &mut b.row,
                tok,
                last,
                second_last,
                b.max_ts,
                b.tokens.len(),
                max_initial_ts_id,
                blank_id,
            );
            let lps = log_softmax(&b.row);
            for (lp, id) in top_k(&lps, beam_size) {
                let step_lp = lp.max(-30.0);
                cands.push((bi, id, b.sum_lp + step_lp, step_lp));
            }
        }
        cands.sort_by(|a, b| b.2.total_cmp(&a.2));
        cands.truncate(beam_size);

        // Advance surviving beams to their hidden states, then project all
        // beams' logits in ONE matmul — the tied embedding matrix is read
        // (and, if quantized, unpacked) once per step instead of per beam.
        let mut next: Vec<Beam> = Vec::with_capacity(cands.len());
        let mut hiddens: Vec<f32> = Vec::with_capacity(cands.len() * n_state);
        for (parent, id, new_sum, step_lp) in cands {
            if id == tok.eot {
                finished.push((
                    beams[parent].tokens.clone(),
                    beams[parent].token_logprobs.clone(),
                    new_sum,
                ));
                continue;
            }
            let p = &beams[parent];
            if p.dec.n_past() + 1 > hp.n_text_ctx as usize {
                finished.push((p.tokens.clone(), p.token_logprobs.clone(), p.sum_lp));
                continue;
            }
            let mut dec = p.dec.fork();
            let mut tokens = p.tokens.clone();
            let mut token_logprobs = p.token_logprobs.clone();
            let h = dec.forward_hidden(&[id]);
            hiddens.extend_from_slice(&h.data[..n_state]);
            let max_ts = if tok.is_timestamp(id) {
                Some(p.max_ts.map_or(id, |m| m.max(id)))
            } else {
                p.max_ts
            };
            tokens.push(id);
            token_logprobs.push(step_lp);
            next.push(Beam {
                dec,
                tokens,
                token_logprobs,
                sum_lp: new_sum,
                row: Vec::new(),
                max_ts,
            });
        }
        if !next.is_empty() {
            let stacked = Tensor::from_vec(&[next.len(), n_state], hiddens);
            let logits = next[0].dec.project_logits(&stacked);
            for (r, b) in next.iter_mut().enumerate() {
                b.row = logits.data[r * n_vocab..(r + 1) * n_vocab].to_vec();
            }
        }
        if finished.len() >= beam_size {
            // Enough complete hypotheses — don't let incomplete beams
            // compete in the final ranking.
            beams = Vec::new();
            break;
        }
        if next.is_empty() {
            beams = next;
            break;
        }
        beams = next;
    }

    for b in beams {
        finished.push((b.tokens, b.token_logprobs, b.sum_lp));
    }
    let (tokens, token_logprobs, sum_lp) = finished
        .into_iter()
        .max_by(|a, b| {
            let avg_a = a.2 / (a.0.len() + 1) as f32;
            let avg_b = b.2 / (b.0.len() + 1) as f32;
            avg_a.total_cmp(&avg_b)
        })
        .unwrap();
    let avg_logprob = sum_lp / (tokens.len() + 1) as f32;
    WindowDecode {
        tokens,
        token_logprobs,
        avg_logprob,
    }
}

/// (start, end, text-tokens-with-logprob) — `end = None` for an
/// unterminated final segment.
type ParsedSegment = (f32, Option<f32>, Vec<(u32, f32)>);

/// Split a window's token stream into (start, end, text-tokens) segments,
/// carrying each text token's log-probability alongside it (`token_logprobs`
/// must be the same length as `tokens`, 1:1). An unterminated final segment
/// gets `end = None`.
fn parse_segments(tokens: &[u32], token_logprobs: &[f32], tok: &Tokenizer) -> Vec<ParsedSegment> {
    let mut segments = Vec::new();
    let mut open: Option<f32> = None;
    let mut text: Vec<(u32, f32)> = Vec::new();
    for (&tk, &lp) in tokens.iter().zip(token_logprobs) {
        if tok.is_timestamp(tk) {
            let ts = tok.timestamp_seconds(tk);
            if open.is_some() && !text.is_empty() {
                segments.push((open.unwrap(), Some(ts), std::mem::take(&mut text)));
                open = None;
            } else {
                open = Some(ts);
                text.clear();
            }
        } else if open.is_some() {
            text.push((tk, lp));
        }
    }
    if let (Some(t0), false) = (open, text.is_empty()) {
        segments.push((t0, None, text));
    }
    segments
}

/// Per-token breakdown of a segment's decoded text, with `t0`/`t1`
/// interpolated by each token's own decoded-character span (see
/// `TokenInfo`'s docs for the approximation this relies on).
fn token_infos(
    toks_with_lp: &[(u32, f32)],
    tok: &Tokenizer,
    seg_t0: f32,
    seg_t1: f32,
) -> Vec<TokenInfo> {
    let pieces: Vec<String> = toks_with_lp
        .iter()
        .map(|&(id, _)| tok.decode(&[id]))
        .collect();
    let total_chars: usize = pieces.iter().map(|p| p.chars().count()).sum();
    let duration = seg_t1 - seg_t0;
    let mut char_offset = 0usize;
    let mut out = Vec::with_capacity(toks_with_lp.len());
    for (&(id, lp), piece) in toks_with_lp.iter().zip(&pieces) {
        let piece_chars = piece.chars().count();
        let start = char_offset;
        let end = char_offset + piece_chars;
        char_offset = end;
        let (t0, t1) = if total_chars == 0 {
            (seg_t0, seg_t1)
        } else {
            (
                seg_t0 + duration * (start as f32 / total_chars as f32),
                seg_t0 + duration * (end as f32 / total_chars as f32),
            )
        };
        out.push(TokenInfo {
            id,
            text: piece.clone(),
            prob: lp.exp(),
            logprob: lp,
            t0,
            t1,
        });
    }
    out
}

/// Split segments longer than `opts.max_len` characters into several,
/// linearly interpolating each sub-segment's `t0`/`t1` by its share of the
/// original segment's character count. A no-op when `max_len == 0`.
///
/// This is an approximation: without per-word alignment (cross-attention or
/// DTW, not yet implemented), there's no ground-truth timing for where in
/// the segment's audio span each word actually falls, so duration is
/// distributed proportionally to text length rather than measured.
fn split_long_segments(segments: Vec<Segment>, opts: &Options) -> Vec<Segment> {
    if opts.max_len == 0 {
        return segments;
    }
    let mut out = Vec::with_capacity(segments.len());
    for seg in segments {
        out.extend(split_one_segment(seg, opts.max_len, opts.split_on_word));
    }
    out
}

fn split_one_segment(seg: Segment, max_len: usize, split_on_word: bool) -> Vec<Segment> {
    let total_chars = seg.text.chars().count();
    if total_chars <= max_len {
        return vec![seg];
    }
    let chunks: Vec<&str> = if split_on_word {
        wrap_by_word(&seg.text, max_len)
    } else {
        wrap_by_char(&seg.text, max_len)
    };
    let duration = seg.t1 - seg.t0;
    let text_start = seg.text.as_ptr() as usize;
    let mut spans = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        // Byte offsets of `chunk` within the original `seg.text` (every
        // chunk is a genuine subslice), converted to char counts — this
        // stays exact even when wrapping drops separator whitespace between
        // chunks, unlike accumulating each chunk's own char count.
        let byte_start = chunk.as_ptr() as usize - text_start;
        let byte_end = byte_start + chunk.len();
        let start = seg.text[..byte_start].chars().count();
        let end = start + seg.text[byte_start..byte_end].chars().count();
        let (t0, t1) = if total_chars == 0 {
            (seg.t0, seg.t1)
        } else {
            (
                seg.t0 + duration * (start as f32 / total_chars as f32),
                seg.t0 + duration * (end as f32 / total_chars as f32),
            )
        };
        spans.push((t0, t1, chunk.trim().to_string()));
    }
    // Partition the parent's per-token data across chunks by each token's
    // own (independently interpolated) t0 — an approximation on top of an
    // approximation, since chunk and token boundaries aren't derived from
    // exactly the same character basis (trimmed segment text vs. raw
    // per-token pieces). Good enough absent real per-token alignment data.
    let mut token_cursor = 0usize;
    let mut out = Vec::with_capacity(spans.len());
    for (i, (t0, t1, text)) in spans.iter().enumerate() {
        let is_last = i + 1 == spans.len();
        let mut toks = Vec::new();
        while token_cursor < seg.tokens.len() && (is_last || seg.tokens[token_cursor].t0 < *t1) {
            toks.push(seg.tokens[token_cursor].clone());
            token_cursor += 1;
        }
        out.push(Segment {
            t0: *t0,
            t1: *t1,
            text: text.clone(),
            tokens: toks,
        });
    }
    out.retain(|s| !s.text.is_empty());
    out
}

/// Greedily pack words (whitespace-separated) into lines of at most
/// `max_len` chars. A single word longer than `max_len` becomes its own
/// (over-length) line rather than being split mid-word.
fn wrap_by_word(text: &str, max_len: usize) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut line_start = 0usize;
    // Byte offset just past the last word added to the current line — the
    // cut point on wrap, distinct from the current (possibly overflowing)
    // word's own end.
    let mut line_end = 0usize;
    let mut line_chars = 0usize;
    let mut cursor = 0usize;
    while cursor < text.len() {
        let word_start = cursor;
        while cursor < text.len() && !text[cursor..].starts_with(char::is_whitespace) {
            cursor += text[cursor..].chars().next().unwrap().len_utf8();
        }
        let word_end = cursor;
        let word = &text[word_start..word_end];
        let word_chars = word.chars().count();
        while cursor < text.len() && text[cursor..].starts_with(char::is_whitespace) {
            cursor += text[cursor..].chars().next().unwrap().len_utf8();
        }
        if line_chars > 0 && line_chars + 1 + word_chars > max_len {
            lines.push(&text[line_start..line_end]);
            line_start = word_start;
            line_chars = word_chars;
        } else {
            line_chars += if line_chars > 0 { 1 } else { 0 } + word_chars;
        }
        line_end = word_end;
    }
    if line_start < text.len() {
        lines.push(&text[line_start..line_end.max(line_start)]);
    }
    lines
}

/// Split into fixed-width chunks of at most `max_len` chars, ignoring word
/// boundaries (UTF-8 char boundaries are still respected).
fn wrap_by_char(text: &str, max_len: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut chars_in_chunk = 0usize;
    for (i, c) in text.char_indices() {
        if chars_in_chunk == max_len {
            chunks.push(&text[start..i]);
            start = i;
            chars_in_chunk = 0;
        }
        chars_in_chunk += 1;
        let _ = c;
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

pub struct Transcript {
    pub segments: Vec<Segment>,
    /// ISO code of the language transcribed (specified or detected).
    pub language: String,
}

/// Incremental transcription: `feed()` audio as it arrives and get back
/// segments as 30 s windows fill; `finish()` drains the remainder. Memory
/// is bounded — consumed samples are dropped from the buffer.
pub struct Stream<'m> {
    model: &'m Model,
    opts: Options,
    tok: Tokenizer,
    filters: Vec<f32>,
    blank_id: Option<u32>,
    /// Samples not yet consumed; `buf[0]` is absolute sample `offset`.
    buf: Vec<f32>,
    offset: usize,
    prompt_past: Vec<u32>,
    task: Option<Task>,
}

impl<'m> Stream<'m> {
    pub fn new(model: &'m Model, opts: Options) -> Self {
        use crate::tokenizer::lang_id_from_code;
        let tok = Tokenizer::new(model.vocab.clone(), &model.hparams);
        let n_mels = model.hparams.n_mels as usize;
        let filters = if model.mel_filters.is_empty() {
            audio::mel_filterbank(n_mels, audio::N_FFT, audio::SAMPLE_RATE)
        } else {
            model.mel_filters.clone()
        };
        let blank_id = model.vocab.iter().position(|w| w == b" ").map(|i| i as u32);
        // Resolve language + task; None = auto-detect on the first window.
        let task = if model.hparams.is_multilingual() {
            opts.language.as_deref().map(|code| Task {
                lang_id: lang_id_from_code(code).unwrap_or(0),
                translate: opts.translate,
            })
        } else {
            Some(Task {
                lang_id: 0,
                translate: false,
            })
        };
        Stream {
            model,
            opts,
            tok,
            filters,
            blank_id,
            buf: Vec::new(),
            offset: 0,
            prompt_past: Vec::new(),
            task,
        }
    }

    /// ISO code once known (immediately if specified or English-only;
    /// after the first processed window when auto-detecting).
    pub fn language(&self) -> Option<&'static str> {
        self.task
            .map(|t| crate::tokenizer::LANGUAGES[t.lang_id as usize])
    }

    /// Feed samples; returns segments finalized by newly-complete windows.
    pub fn feed(&mut self, samples: &[f32]) -> Vec<Segment> {
        self.buf.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.buf.len() >= audio::N_SAMPLES_30S {
            out.extend(self.process_window());
        }
        out
    }

    /// Process everything still buffered (call at end of input). Windows
    /// under 1 s are dropped — a sliver of trailing audio decodes as noise
    /// ("[BLANK_AUDIO]" and friends), as in whisper.cpp.
    pub fn finish(&mut self) -> Vec<Segment> {
        let mut out = Vec::new();
        while self.buf.len() > audio::SAMPLE_RATE {
            out.extend(self.process_window());
        }
        out
    }

    /// Decode one window at the buffer start and consume up to its last
    /// timestamp.
    fn process_window(&mut self) -> Vec<Segment> {
        let model = self.model;
        let opts = &self.opts;
        let tok = &self.tok;
        let n_mels = model.hparams.n_mels as usize;
        let window_secs = (self.buf.len() as f32 / audio::SAMPLE_RATE as f32).min(30.0);
        let window = audio::pad_or_trim(&self.buf, audio::N_SAMPLES_30S);
        let (mel, n_frames) = audio::log_mel_spectrogram(&window, &self.filters, n_mels);
        let mel = Tensor::from_vec(&[n_mels, n_frames], mel);
        let enc_out = encoder::encode(model, &mel);
        let mut dec = Decoder::new(model, &enc_out);
        let task = *self.task.get_or_insert_with(|| {
            let (lang_id, _prob) = detect_language(&mut dec, tok);
            Task {
                lang_id,
                translate: opts.translate,
            }
        });

        // Temperature ladder until the decode passes the quality gates.
        let blank_id = self.blank_id;
        let run_ladder = |dec: &mut Decoder, past_all: &[u32]| -> WindowDecode {
            let mut best: Option<WindowDecode> = None;
            for &temp in &opts.temperatures {
                // High temperatures decode without past conditioning
                // (whisper.cpp drops it at t > 0.5 to break repetition loops).
                let past: &[u32] = if temp <= 0.5 { past_all } else { &[] };
                let wd = if temp <= 0.0 && opts.beam_size > 1 {
                    decode_window_beam(dec, tok, model, past, opts, task, blank_id)
                } else {
                    decode_window(dec, tok, model, past, opts, task, temp, blank_id)
                };
                let text = tok.decode(&wd.tokens);
                let ok_compression =
                    compression_ratio(text.as_bytes()) < opts.compression_ratio_threshold;
                let ok_logprob = wd.avg_logprob > opts.logprob_threshold;
                best = Some(wd);
                if ok_compression && ok_logprob {
                    break;
                }
            }
            best.unwrap()
        };

        let past: &[u32] = if opts.condition_on_past {
            &self.prompt_past
        } else {
            &[]
        };
        let mut wd = run_ladder(&mut dec, past);
        // A conditioned decode of audible audio can collapse to nothing when
        // the prompt already contains the same phrase (the model treats the
        // window as "already transcribed"). Retry unconditioned.
        if !past.is_empty()
            && parse_segments(&wd.tokens, &wd.token_logprobs, tok)
                .iter()
                .all(|(_, _, t)| t.is_empty())
        {
            wd = run_ladder(&mut dec, &[]);
        }

        let offset_secs = self.offset as f32 / audio::SAMPLE_RATE as f32;
        let parsed = parse_segments(&wd.tokens, &wd.token_logprobs, tok);
        let mut segments = Vec::new();
        let mut last_ts = 0.0f32;
        for (t0, t1, toks) in &parsed {
            let end = t1.unwrap_or(window_secs);
            last_ts = last_ts.max(end);
            let ids: Vec<u32> = toks.iter().map(|&(id, _)| id).collect();
            let text = tok.decode(&ids).trim().to_string();
            if !text.is_empty() {
                let seg_t0 = offset_secs + t0;
                let seg_t1 = offset_secs + end;
                segments.push(Segment {
                    t0: seg_t0,
                    t1: seg_t1,
                    text,
                    tokens: token_infos(toks, tok, seg_t0, seg_t1),
                });
            }
        }

        // Condition the next window on this one's text tokens.
        for (_, _, toks) in &parsed {
            self.prompt_past.extend(
                toks.iter()
                    .map(|&(id, _)| id)
                    .filter(|id| !tok.is_special(*id)),
            );
        }
        let keep = model.hparams.n_text_ctx as usize / 2 - 1;
        if self.prompt_past.len() > keep {
            self.prompt_past.drain(..self.prompt_past.len() - keep);
        }

        // Advance to the last timestamp; guard against stalling.
        let advance_secs = if last_ts >= 1.0 { last_ts } else { 30.0 };
        let advance = ((advance_secs * audio::SAMPLE_RATE as f32) as usize).min(self.buf.len());
        self.buf.drain(..advance);
        self.offset += advance;
        split_long_segments(segments, opts)
    }
}

/// Transcribe arbitrary-length 16 kHz mono audio into timed segments.
pub fn transcribe(model: &Model, samples: &[f32], opts: &Options) -> Transcript {
    let mut stream = Stream::new(model, opts.clone());
    let mut segments = stream.feed(samples);
    segments.extend(stream.finish());
    let language = stream.language().unwrap_or("en").to_string();
    Transcript { segments, language }
}

/// `[hh:mm:ss.mmm]` formatting for CLI output.
pub fn format_timestamp(secs: f32) -> String {
    let ms = (secs * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HParams;

    fn tok_en() -> Tokenizer {
        let mut vocab = vec![Vec::new(); 400];
        vocab[100] = b"Hello".to_vec();
        vocab[101] = b" world".to_vec();
        Tokenizer::new(
            vocab,
            &HParams {
                n_vocab: 51864,
                ..Default::default()
            },
        )
    }

    #[test]
    fn compression_ratio_flags_repetition() {
        let normal = b"And so my fellow Americans, ask not what your country can do for you.";
        let repetitive = b"la la la la la la la la la la la la la la la la la la la la la la";
        assert!(
            compression_ratio(normal) < 2.4,
            "{}",
            compression_ratio(normal)
        );
        assert!(
            compression_ratio(repetitive) > 2.4,
            "{}",
            compression_ratio(repetitive)
        );
    }

    #[test]
    fn parse_segments_pairs_and_trailing() {
        let t = tok_en();
        let b = t.timestamp_begin;
        // <0.00> Hello world <1.00><1.50> Hello <2.00>
        let tokens = vec![b, 100, 101, b + 50, b + 75, 100, b + 100];
        let lps = vec![0.0f32; tokens.len()];
        let segs = parse_segments(&tokens, &lps, &t);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].0, 0.0);
        assert_eq!(segs[0].1, Some(1.0));
        assert_eq!(
            segs[0].2.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            vec![100, 101]
        );
        assert_eq!(segs[1].0, 1.5);
        assert_eq!(segs[1].1, Some(2.0));
    }

    #[test]
    fn parse_segments_unterminated() {
        let t = tok_en();
        let b = t.timestamp_begin;
        let segs = parse_segments(&[b + 10, 100], &[0.0, 0.0], &t);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].1, None);
    }

    #[test]
    fn rules_force_initial_timestamp() {
        let t = tok_en();
        let mut row = vec![0.0f32; 51864];
        row[100] = 10.0; // text token would win without rules
        apply_rules(
            &mut row,
            &t,
            None,
            None,
            None,
            0,
            t.timestamp_begin + 50,
            Some(220),
        );
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert!(t.is_timestamp(best));
        assert!(best <= t.timestamp_begin + 50, "respects max_initial_ts");
    }

    #[test]
    fn rules_pair_timestamps() {
        let t = tok_en();
        let b = t.timestamp_begin;
        // After a lone timestamp, text must be suppressed.
        let mut row = vec![0.0f32; 51864];
        row[100] = 10.0;
        apply_rules(&mut row, &t, Some(b + 50), Some(100), None, 3, b + 50, None);
        assert_eq!(row[100], f32::NEG_INFINITY);
        // After a timestamp pair, timestamps must be suppressed.
        let mut row = vec![0.0f32; 51864];
        row[(b + 60) as usize] = 10.0;
        apply_rules(
            &mut row,
            &t,
            Some(b + 50),
            Some(b + 40),
            Some(b + 50),
            4,
            b + 50,
            None,
        );
        assert!(row[(b + 60) as usize..]
            .iter()
            .all(|&v| v == f32::NEG_INFINITY));
    }

    #[test]
    fn rules_initial_timestamp_is_an_opener() {
        // After the very first (initial) timestamp, text must follow — not
        // another timestamp.
        let t = tok_en();
        let b = t.timestamp_begin;
        let mut row = vec![0.0f32; 51864];
        row[(b + 100) as usize] = 10.0; // a later timestamp would win unruled
        apply_rules(&mut row, &t, Some(b), None, Some(b), 1, b + 50, None);
        assert!(row[b as usize..].iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(row[100].is_finite());
    }

    #[test]
    fn rules_monotonic_timestamps() {
        let t = tok_en();
        let b = t.timestamp_begin;
        let mut row = vec![0.0f32; 51864];
        // Last was text, a timestamp was seen at +50: earlier ts must be dead.
        apply_rules(
            &mut row,
            &t,
            Some(100),
            Some(b + 50),
            Some(b + 50),
            5,
            b + 50,
            None,
        );
        assert!(row[b as usize..(b + 51) as usize]
            .iter()
            .all(|&v| v == f32::NEG_INFINITY));
    }

    #[test]
    fn top_k_orders_and_skips_masked() {
        let lps = vec![-5.0, -1.0, f32::NEG_INFINITY, -0.5, -3.0];
        let got = top_k(&lps, 3);
        assert_eq!(got, vec![(-0.5, 3), (-1.0, 1), (-3.0, 4)]);
        // k larger than candidates: masked entries never appear.
        assert_eq!(top_k(&lps, 10).len(), 4);
    }

    #[test]
    fn sampling_temperature_zero_is_argmax() {
        let mut rng = Rng(7);
        let row = vec![0.1, 5.0, -2.0];
        assert_eq!(sample(&row, 0.0, &mut rng), 1);
    }

    #[test]
    fn timestamp_formatting() {
        assert_eq!(format_timestamp(0.0), "00:00:00.000");
        assert_eq!(format_timestamp(11.0), "00:00:11.000");
        assert_eq!(format_timestamp(3725.5), "01:02:05.500");
    }

    fn seg(t0: f32, t1: f32, text: &str) -> Segment {
        Segment {
            t0,
            t1,
            text: text.to_string(),
            tokens: Vec::new(),
        }
    }

    #[test]
    fn max_len_zero_disables_splitting() {
        let segments = vec![seg(
            0.0,
            10.0,
            "a very long segment that would otherwise split",
        )];
        let opts = Options::default();
        let out = split_long_segments(segments.clone(), &opts);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, segments[0].text);
    }

    #[test]
    fn short_segment_is_untouched() {
        let segments = vec![seg(0.0, 1.0, "short")];
        let opts = Options {
            max_len: 20,
            ..Default::default()
        };
        let out = split_long_segments(segments, &opts);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "short");
    }

    #[test]
    fn split_on_word_keeps_words_whole_and_covers_the_span() {
        let segments = vec![seg(0.0, 10.0, "one two three four five six seven eight")];
        let opts = Options {
            max_len: 12,
            split_on_word: true,
            ..Default::default()
        };
        let out = split_long_segments(segments, &opts);
        assert!(out.len() > 1);
        for s in &out {
            assert!(s.text.chars().count() <= 12, "line too long: {:?}", s.text);
            assert!(!s.text.contains('\n'));
        }
        // Reassembling the words in order reproduces the original text.
        let rejoined = out
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(rejoined, "one two three four five six seven eight");
        // Timestamps are monotonic and stay within the original span.
        assert_eq!(out.first().unwrap().t0, 0.0);
        assert_eq!(out.last().unwrap().t1, 10.0);
        for w in out.windows(2) {
            assert!(w[0].t1 <= w[1].t0 + 1e-6);
        }
    }

    #[test]
    fn split_by_char_ignores_word_boundaries() {
        let segments = vec![seg(0.0, 4.0, "abcdefgh")];
        let opts = Options {
            max_len: 3,
            split_on_word: false,
            ..Default::default()
        };
        let out = split_long_segments(segments, &opts);
        assert_eq!(
            out.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
            vec!["abc", "def", "gh"]
        );
    }

    #[test]
    fn overlong_single_word_becomes_its_own_line() {
        let lines = wrap_by_word("supercalifragilisticexpialidocious", 10);
        assert_eq!(lines, vec!["supercalifragilisticexpialidocious"]);
    }

    #[test]
    fn token_infos_span_the_segment_and_preserve_order() {
        let t = tok_en();
        let toks = vec![(100u32, -0.1f32), (101u32, -0.2f32)]; // "Hello", " world"
        let infos = token_infos(&toks, &t, 0.0, 2.0);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].id, 100);
        assert_eq!(infos[0].text, "Hello");
        assert_eq!(infos[1].text, " world");
        assert_eq!(infos[0].logprob, -0.1);
        assert!((infos[0].prob - (-0.1f32).exp()).abs() < 1e-6);
        assert_eq!(infos[0].t0, 0.0);
        assert_eq!(infos.last().unwrap().t1, 2.0);
        assert!(infos[0].t1 <= infos[1].t0 + 1e-6);
    }

    #[test]
    fn split_long_segments_partitions_tokens_across_chunks() {
        let t = tok_en();
        // "Hello world" repeated: 100 = "Hello", 101 = " world".
        let toks: Vec<(u32, f32)> = vec![(100, -0.1), (101, -0.1), (100, -0.1), (101, -0.1)];
        let ids: Vec<u32> = toks.iter().map(|&(id, _)| id).collect();
        let text = t.decode(&ids);
        let mut s = seg(0.0, 10.0, &text);
        let n_tokens = toks.len();
        s.tokens = token_infos(&toks, &t, 0.0, 10.0);
        let opts = Options {
            max_len: (text.chars().count() / 2).max(1),
            split_on_word: true,
            ..Default::default()
        };
        let out = split_long_segments(vec![s], &opts);
        assert!(out.len() > 1, "expected the segment to split");
        let total_tokens: usize = out.iter().map(|c| c.tokens.len()).sum();
        assert_eq!(total_tokens, n_tokens);
    }
}
