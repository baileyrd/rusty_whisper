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
        Tensor { shape: shape.to_vec(), data: vec![0.0; n] }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/data mismatch");
        Tensor { shape: shape.to_vec(), data }
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[self.shape.len() - 1]
    }
}

/// C = A[m,k] x B[k,n]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul inner dims: {k} vs {k2}");
    let mut out = Tensor::zeros(&[m, n]);
    for i in 0..m {
        let arow = &a.data[i * k..(i + 1) * k];
        for p in 0..k {
            let av = arow[p];
            if av == 0.0 {
                continue;
            }
            let brow = &b.data[p * n..(p + 1) * n];
            let orow = &mut out.data[i * n..(i + 1) * n];
            for j in 0..n {
                orow[j] += av * brow[j];
            }
        }
    }
    out
}

/// C = A[m,k] x B^T where B is [n,k] — the natural layout for weight
/// matrices stored as `[out_features, in_features]` (ggml convention).
pub fn matmul_t(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (n, k2) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul_t inner dims: {k} vs {k2}");
    let mut out = Tensor::zeros(&[m, n]);
    for i in 0..m {
        let arow = &a.data[i * k..(i + 1) * k];
        for j in 0..n {
            let brow = &b.data[j * k..(j + 1) * k];
            out.data[i * n + j] = arow.iter().zip(brow).map(|(x, y)| x * y).sum();
        }
    }
    out
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

/// GELU, tanh approximation — the variant ggml uses.
pub fn gelu(x: &mut Tensor) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in x.data.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (SQRT_2_OVER_PI * (t + 0.044715 * t * t * t)).tanh());
    }
}

/// Softmax over the last dimension, in place.
pub fn softmax(x: &mut Tensor) {
    let cols = x.cols();
    for row in x.data.chunks_mut(cols) {
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
    for oc in 0..out_ch {
        for ot in 0..t_out {
            let mut sum = bias[oc];
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let it = (ot * stride + k) as isize - pad as isize;
                    if it >= 0 && (it as usize) < t {
                        sum += weight.data[(oc * in_ch + ic) * kernel + k]
                            * input.data[ic * t + it as usize];
                    }
                }
            }
            out.data[oc * t_out + ot] = sum;
        }
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
