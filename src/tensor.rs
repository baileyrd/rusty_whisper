//! Minimal tensor core — the subset of ggml ops Whisper inference needs.
//!
//! Correctness-first, naive implementations. Layout is row-major; a matrix of
//! shape `[rows, cols]` stores element `(r, c)` at `r * cols + c`. SIMD and
//! threading come in a later phase (see PLAN.md phase 7).

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: vec![0.0; n],
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape/data mismatch"
        );
        Tensor {
            shape: shape.to_vec(),
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[self.shape.len() - 1]
    }
}

/// Work (in multiply-adds) below which a matmul stays single-threaded.
pub(crate) const PAR_THRESHOLD: usize = 1 << 20;

pub(crate) fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Run `f(chunk_index, rows_chunk)` over `out` split into row chunks on all
/// cores. `rows` is the total row count, `cols` the row width in `out`.
pub(crate) fn par_row_chunks(
    out: &mut [f32],
    rows: usize,
    cols: usize,
    f: impl Fn(usize, &mut [f32]) + Sync,
) {
    let threads = n_threads();
    if threads <= 1 {
        // Single-core boxes and wasm (no thread spawning) run serially.
        f(0, out);
        return;
    }
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (c, orows) in out.chunks_mut(chunk * cols).enumerate() {
            let f = &f;
            s.spawn(move || f(c * chunk, orows));
        }
    });
}

/// C = A[m,k] x B[k,n], parallelized over output rows.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul inner dims: {k} vs {k2}");
    let mut out = Tensor::zeros(&[m, n]);

    let rows = |i0: usize, orows: &mut [f32]| {
        for (r, orow) in orows.chunks_mut(n).enumerate() {
            let arow = &a.data[(i0 + r) * k..(i0 + r + 1) * k];
            for (p, &av) in arow.iter().enumerate() {
                if av == 0.0 {
                    continue;
                }
                let brow = &b.data[p * n..(p + 1) * n];
                for (o, bv) in orow.iter_mut().zip(brow) {
                    *o += av * bv;
                }
            }
        }
    };
    if n_threads() > 1 && m * n * k > PAR_THRESHOLD && m > 1 {
        par_row_chunks(&mut out.data, m, n, rows);
    } else {
        rows(0, &mut out.data);
    }
    out
}

const LANES: usize = 8;

/// SIMD-friendly dot product: per-lane accumulator arrays give LLVM
/// independent chains it can vectorize without reassociating float adds
/// (a plain `s += a*b` loop compiles to a scalar latency-bound FMA chain).
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (xa, xb) in (&mut ca).zip(&mut cb) {
        let xa: &[f32; LANES] = xa.try_into().unwrap();
        let xb: &[f32; LANES] = xb.try_into().unwrap();
        for l in 0..LANES {
            acc[l] += xa[l] * xb[l];
        }
    }
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    acc.iter().sum::<f32>() + tail
}

/// Like [`dot`] but four A rows against one B row at a time — B is loaded
/// once per 4 outputs, and the 4 accumulator sets hide FMA latency.
#[inline]
pub(crate) fn dot4(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    let mut acc = [[0.0f32; LANES]; 4];
    let k = b.len();
    let k8 = k / LANES * LANES;
    let mut p = 0;
    while p < k8 {
        let xb: &[f32; LANES] = b[p..p + LANES].try_into().unwrap();
        let x0: &[f32; LANES] = a0[p..p + LANES].try_into().unwrap();
        let x1: &[f32; LANES] = a1[p..p + LANES].try_into().unwrap();
        let x2: &[f32; LANES] = a2[p..p + LANES].try_into().unwrap();
        let x3: &[f32; LANES] = a3[p..p + LANES].try_into().unwrap();
        for l in 0..LANES {
            acc[0][l] += x0[l] * xb[l];
            acc[1][l] += x1[l] * xb[l];
            acc[2][l] += x2[l] * xb[l];
            acc[3][l] += x3[l] * xb[l];
        }
        p += LANES;
    }
    let mut out = [0.0f32; 4];
    for (r, (acc_r, a)) in acc.iter().zip([a0, a1, a2, a3]).enumerate() {
        let tail: f32 = a[k8..].iter().zip(&b[k8..]).map(|(x, y)| x * y).sum();
        out[r] = acc_r.iter().sum::<f32>() + tail;
    }
    out
}

/// The matmul_t micro-kernel: rows `i0..` of A (as `arows`, `nrows * k`)
/// against all `n` rows of `b`, writing `nrows * n` outputs. Processes A in
/// blocks of 4 rows so each B row is loaded once per block instead of once
/// per row.
fn mmt_kernel(arows: &[f32], b: &[f32], out: &mut [f32], k: usize, n: usize) {
    let nrows = arows.len() / k;
    let mut i = 0;
    while i + 4 <= nrows {
        let a0 = &arows[i * k..(i + 1) * k];
        let a1 = &arows[(i + 1) * k..(i + 2) * k];
        let a2 = &arows[(i + 2) * k..(i + 3) * k];
        let a3 = &arows[(i + 3) * k..(i + 4) * k];
        for j in 0..n {
            let s = dot4(a0, a1, a2, a3, &b[j * k..(j + 1) * k]);
            out[i * n + j] = s[0];
            out[(i + 1) * n + j] = s[1];
            out[(i + 2) * n + j] = s[2];
            out[(i + 3) * n + j] = s[3];
        }
        i += 4;
    }
    while i < nrows {
        let arow = &arows[i * k..(i + 1) * k];
        for j in 0..n {
            out[i * n + j] = dot(arow, &b[j * k..(j + 1) * k]);
        }
        i += 1;
    }
}

/// C = A[m,k] x B^T where B is [n,k] — the natural layout for weight
/// matrices stored as `[out_features, in_features]` (ggml convention).
/// Parallelized over A rows when A is tall, or over B rows when A is skinny
/// (a single decoder token against the 51k-row vocabulary matrix).
pub fn matmul_t(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (n, k2) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul_t inner dims: {k} vs {k2}");
    let mut out = Tensor::zeros(&[m, n]);
    let threads = n_threads();

    if threads > 1 && m * n * k > PAR_THRESHOLD {
        if m >= threads {
            par_row_chunks(&mut out.data, m, n, |i0, orows| {
                let nrows = orows.len() / n;
                mmt_kernel(&a.data[i0 * k..(i0 + nrows) * k], &b.data, orows, k, n);
            });
        } else {
            // Column split: each thread takes a slice of B's rows and fills
            // a private [m, cols] block, copied back afterwards.
            let chunk = n.div_ceil(threads);
            let blocks: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..n.div_ceil(chunk))
                    .map(|c| {
                        let (a, b) = (&a.data, &b.data);
                        s.spawn(move || {
                            let j0 = c * chunk;
                            let cols = chunk.min(n - j0);
                            let mut block = vec![0.0f32; m * cols];
                            mmt_kernel(a, &b[j0 * k..(j0 + cols) * k], &mut block, k, cols);
                            (j0, block)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (j0, block) in blocks {
                let cols = block.len() / m;
                for i in 0..m {
                    out.data[i * n + j0..i * n + j0 + cols]
                        .copy_from_slice(&block[i * cols..(i + 1) * cols]);
                }
            }
        }
    } else {
        mmt_kernel(&a.data, &b.data, &mut out.data, k, n);
    }
    out
}

/// Transpose a 2-D tensor.
pub fn transpose(x: &Tensor) -> Tensor {
    let (r, c) = (x.shape[0], x.shape[1]);
    let mut out = Tensor::zeros(&[c, r]);
    for i in 0..r {
        for j in 0..c {
            out.data[j * r + i] = x.data[i * c + j];
        }
    }
    out
}

/// y = x W^T + b with W stored `[out, in]` (the layout every Whisper linear
/// uses after the loader's dim flip).
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&[f32]>) -> Tensor {
    let mut y = matmul_t(x, w);
    if let Some(b) = b {
        add_bias(&mut y, b);
    }
    y
}

/// x = x + bias, broadcasting bias over rows.
pub fn add_bias(x: &mut Tensor, bias: &[f32]) {
    let cols = x.cols();
    assert_eq!(cols, bias.len());
    for row in x.data.chunks_mut(cols) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// LayerNorm over the last dimension with weight/bias (eps as ggml: 1e-5).
pub fn layernorm(x: &mut Tensor, weight: &[f32], bias: &[f32]) {
    const EPS: f32 = 1e-5;
    let cols = x.cols();
    assert_eq!(cols, weight.len());
    assert_eq!(cols, bias.len());
    for row in x.data.chunks_mut(cols) {
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for (v, (w, b)) in row.iter_mut().zip(weight.iter().zip(bias)) {
            *v = (*v - mean) * inv * w + b;
        }
    }
}

/// GELU, tanh approximation — the variant ggml uses. Parallel for large
/// tensors (the encoder applies it to multi-megabyte activations).
pub fn gelu(x: &mut Tensor) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let one = |chunk: &mut [f32]| {
        for v in chunk.iter_mut() {
            let t = *v;
            *v = 0.5 * t * (1.0 + (SQRT_2_OVER_PI * (t + 0.044715 * t * t * t)).tanh());
        }
    };
    let n = x.data.len();
    if n_threads() > 1 && n > PAR_THRESHOLD {
        par_row_chunks(&mut x.data, n, 1, |_, chunk| one(chunk));
    } else {
        one(&mut x.data);
    }
}

/// Softmax over the last dimension, in place. Parallel over rows for large
/// tensors (attention scores are [1500, 1500] per head in the encoder).
pub fn softmax(x: &mut Tensor) {
    let cols = x.cols();
    let rows_fn = |_i0: usize, rows: &mut [f32]| {
        for row in rows.chunks_mut(cols) {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    };
    let rows = x.data.len() / cols;
    if n_threads() > 1 && x.data.len() > PAR_THRESHOLD && rows > 1 {
        par_row_chunks(&mut x.data, rows, cols, rows_fn);
    } else {
        rows_fn(0, &mut x.data);
    }
}

/// 1-D convolution: input [in_ch, t], weight [out_ch, in_ch, kernel],
/// stride `stride`, padding `kernel/2` (whisper's conv1 uses stride 1,
/// conv2 stride 2, both kernel 3, pad 1). Output [out_ch, t_out].
pub fn conv1d(input: &Tensor, weight: &Tensor, bias: &[f32], stride: usize) -> Tensor {
    let (in_ch, t) = (input.shape[0], input.shape[1]);
    let (out_ch, w_in_ch, kernel) = (weight.shape[0], weight.shape[1], weight.shape[2]);
    assert_eq!(in_ch, w_in_ch);
    assert_eq!(out_ch, bias.len());
    let pad = kernel / 2;
    let t_out = (t + 2 * pad - kernel) / stride + 1;
    let mut out = Tensor::zeros(&[out_ch, t_out]);

    let channels = |oc0: usize, orows: &mut [f32]| {
        for (r, orow) in orows.chunks_mut(t_out).enumerate() {
            let oc = oc0 + r;
            for (ot, o) in orow.iter_mut().enumerate() {
                let mut sum = bias[oc];
                for ic in 0..in_ch {
                    let wrow =
                        &weight.data[(oc * in_ch + ic) * kernel..(oc * in_ch + ic + 1) * kernel];
                    let irow = &input.data[ic * t..(ic + 1) * t];
                    for (k, &w) in wrow.iter().enumerate() {
                        let it = (ot * stride + k) as isize - pad as isize;
                        if it >= 0 && (it as usize) < t {
                            sum += w * irow[it as usize];
                        }
                    }
                }
                *o = sum;
            }
        }
    };
    if n_threads() > 1 && out_ch * t_out * in_ch * kernel > PAR_THRESHOLD {
        par_row_chunks(&mut out.data, out_ch, t_out, channels);
    } else {
        channels(0, &mut out.data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_known_values() {
        let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::from_vec(&[3, 2], vec![7., 8., 9., 10., 11., 12.]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_t_agrees_with_matmul() {
        let a = Tensor::from_vec(&[2, 3], vec![1., -2., 3., 0.5, 5., -6.]);
        let b = Tensor::from_vec(&[3, 4], (0..12).map(|i| i as f32 * 0.3 - 1.0).collect());
        // Transpose b into [4,3] and check matmul_t matches matmul.
        let mut bt = Tensor::zeros(&[4, 3]);
        for i in 0..3 {
            for j in 0..4 {
                bt.data[j * 3 + i] = b.data[i * 4 + j];
            }
        }
        let c1 = matmul(&a, &b);
        let c2 = matmul_t(&a, &bt);
        for (x, y) in c1.data.iter().zip(&c2.data) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_t_parallel_path_matches_serial() {
        // Big enough to cross the threading threshold.
        let (m, k, n) = (64, 200, 96);
        let a = Tensor::from_vec(
            &[m, k],
            (0..m * k)
                .map(|i| ((i * 31 + 7) % 17) as f32 - 8.0)
                .collect(),
        );
        let b = Tensor::from_vec(
            &[n, k],
            (0..n * k)
                .map(|i| ((i * 13 + 3) % 11) as f32 - 5.0)
                .collect(),
        );
        let big = matmul_t(&a, &b);
        // Reference: row-by-row serial computation.
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| a.data[i * k + p] * b.data[j * k + p]).sum();
                assert_eq!(big.data[i * n + j], want, "mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn matmul_t_skinny_column_split_matches_serial() {
        // m=1 with a huge n exercises the column-split parallel path (the
        // decoder's logits projection shape).
        let (m, k, n) = (1, 128, 12000);
        let a = Tensor::from_vec(&[m, k], (0..k).map(|i| (i % 7) as f32 - 3.0).collect());
        let b = Tensor::from_vec(
            &[n, k],
            (0..n * k)
                .map(|i| ((i * 19 + 5) % 23) as f32 - 11.0)
                .collect(),
        );
        let fast = matmul_t(&a, &b);
        for j in 0..n {
            let want: f32 = (0..k).map(|p| a.data[p] * b.data[j * k + p]).sum();
            assert_eq!(fast.data[j], want, "mismatch at col {j}");
        }
    }

    #[test]
    fn matmul_t_odd_rows_hit_kernel_remainder() {
        // 5 rows: one 4-block plus a remainder row.
        let (m, k, n) = (5, 33, 9);
        let a = Tensor::from_vec(
            &[m, k],
            (0..m * k)
                .map(|i| ((i * 3 + 1) % 13) as f32 - 6.0)
                .collect(),
        );
        let b = Tensor::from_vec(
            &[n, k],
            (0..n * k)
                .map(|i| ((i * 7 + 2) % 11) as f32 - 5.0)
                .collect(),
        );
        let got = matmul_t(&a, &b);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| a.data[i * k + p] * b.data[j * k + p]).sum();
                assert_eq!(got.data[i * n + j], want, "mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn transpose_round_trip() {
        let x = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let t = transpose(&x);
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1., 4., 2., 5., 3., 6.]);
        assert_eq!(transpose(&t), x);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut x = Tensor::from_vec(&[2, 4], vec![1., 2., 3., 4., -1., 0., 1., 100.]);
        softmax(&mut x);
        for row in x.data.chunks(4) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        // Large logit dominates without overflow.
        assert!(x.data[7] > 0.999);
    }

    #[test]
    fn layernorm_statistics() {
        let mut x = Tensor::from_vec(&[1, 4], vec![1., 2., 3., 4.]);
        layernorm(&mut x, &[1.0; 4], &[0.0; 4]);
        let mean: f32 = x.data.iter().sum::<f32>() / 4.0;
        let var: f32 = x.data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_reference_points() {
        let mut x = Tensor::from_vec(&[1, 3], vec![0.0, 1.0, -1.0]);
        gelu(&mut x);
        assert!(x.data[0].abs() < 1e-7);
        assert!((x.data[1] - 0.841_19).abs() < 1e-3);
        assert!((x.data[2] + 0.158_81).abs() < 1e-3);
    }

    #[test]
    fn conv1d_identity_kernel() {
        // kernel=1 with identity weights is a pass-through.
        let input = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let weight = Tensor::from_vec(&[2, 2, 1], vec![1., 0., 0., 1.]);
        let out = conv1d(&input, &weight, &[0.0, 0.0], 1);
        assert_eq!(out.data, input.data);
    }

    #[test]
    fn conv1d_stride_two_halves_time() {
        let input = Tensor::from_vec(&[1, 6], vec![1., 1., 1., 1., 1., 1.]);
        let weight = Tensor::from_vec(&[1, 1, 3], vec![1., 1., 1.]);
        let out = conv1d(&input, &weight, &[0.0], 2);
        assert_eq!(out.shape, vec![1, 3]);
        // Interior windows sum three ones.
        assert_eq!(out.data[1], 3.0);
    }
}
