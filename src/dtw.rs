//! Token-level timestamp alignment via dynamic time warping (DTW) over
//! decoder cross-attention weights — port of whisper.cpp's `-dtw`
//! (`whisper_exp_compute_token_level_timestamps_dtw` in `src/whisper.cpp`).
//!
//! Three independent, pure pieces, composed by [`token_time_indices`]:
//! per-head normalization, a width-7 median filter over the time axis, and
//! the DTW dynamic program + backtrace itself. None of this touches the
//! decoder — callers capture cross-attention weights for the alignment
//! heads listed in [`alignment_heads`] and hand them in as plain slices.

/// `(decoder_layer, head)` pairs used for alignment, 0-indexed — verbatim
/// from whisper.cpp v1.9.1's `g_aheads_*` tables (`src/whisper.cpp`).
/// Accepts the same preset names as whisper.cpp's `-dtw` flag (dotted, not
/// hyphenated: `"large.v3.turbo"`, not `"large-v3-turbo"`).
pub fn alignment_heads(preset: &str) -> Option<&'static [(usize, usize)]> {
    Some(match preset {
        "tiny.en" => &TINY_EN,
        "tiny" => &TINY,
        "base.en" => &BASE_EN,
        "base" => &BASE,
        "small.en" => &SMALL_EN,
        "small" => &SMALL,
        "medium.en" => &MEDIUM_EN,
        "medium" => &MEDIUM,
        "large.v1" => &LARGE_V1,
        "large.v2" => &LARGE_V2,
        "large.v3" => &LARGE_V3,
        "large.v3.turbo" => &LARGE_V3_TURBO,
        _ => return None,
    })
}

static TINY_EN: [(usize, usize); 8] = [
    (1, 0),
    (2, 0),
    (2, 5),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
];
static TINY: [(usize, usize); 6] = [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)];
static BASE_EN: [(usize, usize); 5] = [(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)];
static BASE: [(usize, usize); 8] = [
    (3, 1),
    (4, 2),
    (4, 3),
    (4, 7),
    (5, 1),
    (5, 2),
    (5, 4),
    (5, 6),
];
static SMALL_EN: [(usize, usize); 19] = [
    (6, 6),
    (7, 0),
    (7, 3),
    (7, 8),
    (8, 2),
    (8, 5),
    (8, 7),
    (9, 0),
    (9, 4),
    (9, 8),
    (9, 10),
    (10, 0),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 6),
    (10, 11),
    (11, 2),
    (11, 4),
];
static SMALL: [(usize, usize); 10] = [
    (5, 3),
    (5, 9),
    (8, 0),
    (8, 4),
    (8, 7),
    (8, 8),
    (9, 0),
    (9, 7),
    (9, 9),
    (10, 5),
];
static MEDIUM_EN: [(usize, usize); 18] = [
    (11, 4),
    (14, 1),
    (14, 12),
    (14, 14),
    (15, 4),
    (16, 0),
    (16, 4),
    (16, 9),
    (17, 12),
    (17, 14),
    (18, 7),
    (18, 10),
    (18, 15),
    (20, 0),
    (20, 3),
    (20, 9),
    (20, 14),
    (21, 12),
];
static MEDIUM: [(usize, usize); 6] = [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)];
static LARGE_V1: [(usize, usize); 9] = [
    (9, 19),
    (11, 2),
    (11, 4),
    (11, 17),
    (22, 7),
    (22, 11),
    (22, 17),
    (23, 2),
    (23, 15),
];
static LARGE_V2: [(usize, usize); 23] = [
    (10, 12),
    (13, 17),
    (16, 11),
    (16, 12),
    (16, 13),
    (17, 15),
    (17, 16),
    (18, 4),
    (18, 11),
    (18, 19),
    (19, 11),
    (21, 2),
    (21, 3),
    (22, 3),
    (22, 9),
    (22, 12),
    (23, 5),
    (23, 7),
    (23, 13),
    (25, 5),
    (26, 1),
    (26, 12),
    (27, 15),
];
static LARGE_V3: [(usize, usize); 10] = [
    (7, 0),
    (10, 17),
    (12, 18),
    (13, 12),
    (16, 1),
    (17, 14),
    (19, 11),
    (21, 4),
    (24, 1),
    (25, 6),
];
static LARGE_V3_TURBO: [(usize, usize); 6] = [(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)];

/// Per-`(head, time)` z-score normalization across the token axis — matches
/// `ggml_norm` applied to whisper.cpp's `[n_tokens, n_audio_tokens,
/// n_heads]` tensor along its token dimension (eps `1e-9`).
///
/// `weights` is `n_heads` matrices, each row-major `[n_tokens, n_time]`
/// (row = token, column = time step).
fn normalize_heads(weights: &mut [Vec<f32>], n_tokens: usize, n_time: usize) {
    const EPS: f32 = 1e-9;
    for head in weights.iter_mut() {
        for time in 0..n_time {
            let mut mean = 0.0f32;
            for tok in 0..n_tokens {
                mean += head[tok * n_time + time];
            }
            mean /= n_tokens as f32;
            let mut var = 0.0f32;
            for tok in 0..n_tokens {
                let d = head[tok * n_time + time] - mean;
                var += d * d;
            }
            var /= n_tokens as f32;
            let inv_std = 1.0 / (var + EPS).sqrt();
            for tok in 0..n_tokens {
                let v = &mut head[tok * n_time + time];
                *v = (*v - mean) * inv_std;
            }
        }
    }
}

/// Median filter of `width` (must be odd) along the time axis, independent
/// per `(token, ...)` row, with reflect padding at the boundaries — matches
/// whisper.cpp's `median_filter` custom op exactly (including its reflect
/// index formula).
fn median_filter_row(row: &[f32], width: usize) -> Vec<f32> {
    assert!(width % 2 == 1, "median filter width must be odd");
    let half = (width / 2) as isize;
    let n = row.len() as isize;
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f32; row.len()];
    let mut window = vec![0.0f32; width];
    for k in 0..n {
        for (slot, off) in window.iter_mut().zip(-half..=half) {
            let mut idx = k + off;
            if idx < 0 {
                idx = -idx;
            }
            if idx >= n {
                idx = 2 * (n - 1) - idx;
            }
            // The reference only reflects once and asserts `width < n` to
            // guarantee that's enough; guard the same case here instead of
            // asserting, since callers may hand us a short trailing window.
            *slot = row[idx.clamp(0, n - 1) as usize];
        }
        window.sort_by(|a, b| a.total_cmp(b));
        out[k as usize] = window[width / 2];
    }
    out
}

/// DTW cost-matrix cell move: which neighbor the optimal path came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Move {
    Diag,
    Up,
    Left,
}

/// Dynamic time warping over a `[n_tokens, n_time]` cost matrix (row-major,
/// lower = better), returning the optimal monotonic alignment path as
/// `(token_index, time_index)` pairs in forward order. Exact port of
/// whisper.cpp's `dtw_and_backtrace` (itself a port of OpenAI Whisper's
/// `timing.py::dtw_cpu`) — including its tie-break quirk: a cost tied
/// exactly between the diagonal and "up" moves resolves to "left", not
/// "diagonal", because the comparison chain is `if c0<c1&&c0<c2 {diag} else
/// if c1<c0&&c1<c2 {up} else {left}`. Replicated verbatim, not "fixed", so
/// output matches the reference bit-for-bit on tied inputs.
fn dtw_and_backtrace(cost: &[f32], n_tokens: usize, n_time: usize) -> Vec<(usize, usize)> {
    let (n, m) = (n_tokens, n_time);
    let w = m + 1;
    let mut dp = vec![f32::INFINITY; (n + 1) * w];
    let mut trace = vec![None::<Move>; (n + 1) * w];
    dp[0] = 0.0;

    for j in 1..=m {
        for i in 1..=n {
            let c0 = dp[(i - 1) * w + (j - 1)]; // diagonal
            let c1 = dp[(i - 1) * w + j]; // up
            let c2 = dp[i * w + (j - 1)]; // left
            let (c, mv) = if c0 < c1 && c0 < c2 {
                (c0, Move::Diag)
            } else if c1 < c0 && c1 < c2 {
                (c1, Move::Up)
            } else {
                (c2, Move::Left)
            };
            dp[i * w + j] = cost[(i - 1) * m + (j - 1)] + c;
            trace[i * w + j] = Some(mv);
        }
    }
    // Boundary rows/columns the DP loop above never wrote (i==0 or j==0),
    // forced after the fact — matches the reference exactly.
    for slot in trace.iter_mut().take(m + 1) {
        *slot = Some(Move::Left); // row i=0
    }
    for i in 0..=n {
        trace[i * w] = Some(Move::Up); // column j=0
    }

    let mut path = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        path.push((i - 1, j.saturating_sub(1)));
        match trace[i * w + j].unwrap() {
            Move::Diag => {
                i -= 1;
                j -= 1;
            }
            Move::Up => i -= 1,
            Move::Left => j -= 1,
        }
    }
    path.reverse();
    path
}

/// Full pipeline: per-head normalize, median-filter (width 7), mean across
/// heads (negated, since DTW here minimizes but attention is a similarity
/// score), then DTW + backtrace. Returns, for each text token index in
/// order, the time-step index (20 ms units) at which the DTW path first
/// reaches that token — i.e. each token's onset, not an interval. A token
/// with no path step (shouldn't happen for a well-formed path, but a
/// pathological all-empty input) is left absent from the result.
///
/// `weights` is one `[n_tokens, n_time]` row-major matrix per captured
/// alignment head (already restricted to just the text-token rows —
/// exclude any SOT-sequence prefix and the trailing EOT column before
/// calling this, as whisper.cpp does).
pub fn token_time_indices(
    mut weights: Vec<Vec<f32>>,
    n_tokens: usize,
    n_time: usize,
) -> Vec<usize> {
    if n_tokens == 0 || n_time == 0 || weights.is_empty() {
        return Vec::new();
    }
    normalize_heads(&mut weights, n_tokens, n_time);

    // whisper.cpp hardcodes width 7 and requires it to be < n_time; clamp
    // down to the largest odd width that fits for short windows instead.
    let width = if n_time > 7 {
        7
    } else {
        ((n_time.saturating_sub(1)) | 1).max(1).min(n_time)
    };
    for head in weights.iter_mut() {
        for tok in 0..n_tokens {
            let row = &head[tok * n_time..(tok + 1) * n_time];
            let filtered = median_filter_row(row, width);
            head[tok * n_time..(tok + 1) * n_time].copy_from_slice(&filtered);
        }
    }

    let n_heads = weights.len();
    let mut cost = vec![0.0f32; n_tokens * n_time];
    for head in &weights {
        for (c, &v) in cost.iter_mut().zip(head.iter()) {
            *c += v;
        }
    }
    for c in cost.iter_mut() {
        *c = -(*c / n_heads as f32); // mean, then negate (similarity -> cost)
    }

    let path = dtw_and_backtrace(&cost, n_tokens, n_time);

    let mut timestamps = vec![None::<usize>; n_tokens];
    let mut last_tok: Option<usize> = None;
    for (tok, time) in path {
        if last_tok != Some(tok) {
            timestamps[tok] = Some(time);
            last_tok = Some(tok);
        }
    }
    timestamps.into_iter().map(|t| t.unwrap_or(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_heads_known_presets() {
        assert_eq!(alignment_heads("tiny"), Some(&TINY[..]));
        assert_eq!(alignment_heads("large.v3.turbo"), Some(&LARGE_V3_TURBO[..]));
        assert_eq!(alignment_heads("tiny-en"), None); // hyphenated, not accepted
        assert_eq!(alignment_heads("bogus"), None);
    }

    #[test]
    fn median_filter_removes_a_single_spike() {
        let row = vec![1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0];
        let out = median_filter_row(&row, 3);
        assert_eq!(out[3], 1.0, "the spike is filtered out");
    }

    #[test]
    fn median_filter_reflects_at_boundaries() {
        // width 3 at index 0 reflects index -1 -> index 1.
        let row = vec![5.0, 1.0, 1.0];
        let out = median_filter_row(&row, 3);
        // window at k=0: [reflect(-1)=row[1]=1.0, row[0]=5.0, row[1]=1.0] -> median 1.0
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn dtw_perfect_diagonal_alignment() {
        // 3 tokens, 3 time steps, cost is 0 on the diagonal and high off it
        // -> the optimal path should visit (0,0),(1,1),(2,2).
        let cost = vec![
            0.0, 9.0, 9.0, //
            9.0, 0.0, 9.0, //
            9.0, 9.0, 0.0,
        ];
        let path = dtw_and_backtrace(&cost, 3, 3);
        let token_at_last_time_step: Vec<usize> = path
            .iter()
            .filter(|&&(_, t)| t == 2)
            .map(|&(tok, _)| tok)
            .collect();
        assert!(token_at_last_time_step.contains(&2));
        // The path must be monotonic in both token and time.
        for w in path.windows(2) {
            assert!(w[1].0 >= w[0].0);
            assert!(w[1].1 >= w[0].1);
        }
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 2)));
    }

    #[test]
    fn dtw_tie_break_prefers_left_over_diagonal() {
        // A fully flat (all-zero) cost matrix: every move ties at every
        // step, so the reference's tie-break quirk (falls through to
        // "left") always wins. Backtrace runs backward from (n, m), so a
        // run of tied Left moves keeps the *last* token index fixed while
        // sweeping backward through time — in forward order that means the
        // final token ends up claiming the trailing run of time steps, and
        // earlier tokens get squeezed toward the start. For 2 tokens, 3
        // time steps this produces exactly (0,0), (1,0), (1,1), (1,2).
        let cost = vec![0.0f32; 2 * 3];
        let path = dtw_and_backtrace(&cost, 2, 3);
        assert_eq!(path, vec![(0, 0), (1, 0), (1, 1), (1, 2)]);
    }

    #[test]
    fn token_time_indices_end_to_end_single_head() {
        // 2 tokens, 4 time steps; token 0 strongly aligned to steps 0-1,
        // token 1 to steps 2-3.
        #[rustfmt::skip]
        let head = vec![
            5.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 5.0,
        ];
        let times = token_time_indices(vec![head], 2, 4);
        assert_eq!(times.len(), 2);
        assert_eq!(times[0], 0, "token 0 onsets at the first time step");
        assert!(
            times[1] >= times[0],
            "token 1 onsets no earlier than token 0"
        );
    }

    #[test]
    fn token_time_indices_empty_input() {
        assert_eq!(token_time_indices(vec![], 0, 0), Vec::<usize>::new());
    }
}
