//! Quantized weights: storage and compute.
//!
//! ggml block formats keep 32 elements per block with an f16 scale (and for
//! the `_1` variants an f16 min). Instead of dequantizing whole tensors to
//! f32 at load (4-5x the RAM and memory traffic), 2-D weight matrices stay
//! in their quantized blocks and matmuls run against them directly:
//! activations are quantized per 32-block to int8 (`d_a`, `q_a[32]`), and
//! each block contributes
//!
//!   dot += d_a * d_b * sum(q_a * q_b)  +  m_b * d_a * sum(q_a)
//!
//! where `(d_b, m_b, q_b)` come from the weight block (`m_b = 0` for the
//! offset-centered `_0` formats). The inner sum is an integer dot product.

use crate::tensor::{add_bias, matmul_t, n_threads, Tensor, PAR_THRESHOLD};

pub const QK: usize = 32;

fn half(bytes: &[u8]) -> f32 {
    crate::model::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Bytes per block for a ggml quantized dtype.
pub fn block_bytes(dtype: i32) -> Option<usize> {
    match dtype {
        2 => Some(18),  // Q4_0
        3 => Some(20),  // Q4_1
        6 => Some(22),  // Q5_0
        7 => Some(24),  // Q5_1
        8 => Some(34),  // Q8_0
        _ => None,
    }
}

/// Split 16 packed bytes into low/high nibbles with `offset` subtracted —
/// flat byte ops over fixed arrays, which the vectorizer handles well.
#[inline]
fn unpack_nibbles(packed: &[u8], q: &mut [i8], offset: i8) {
    let packed: &[u8; QK / 2] = packed[..QK / 2].try_into().unwrap();
    for j in 0..QK / 2 {
        q[j] = (packed[j] & 0x0F) as i8 - offset;
        q[j + QK / 2] = (packed[j] >> 4) as i8 - offset;
    }
}

/// byte -> its 8 bits expanded to `bit << 4`, so high-bit application is
/// table adds instead of per-element shifts (this runs for every element
/// of every Q5 weight on every use — the decoder's token-embedding matrix
/// alone is 20M+ elements per decoded token).
const fn build_hi_table() -> [[i8; 8]; 256] {
    let mut t = [[0i8; 8]; 256];
    let mut b = 0;
    while b < 256 {
        let mut j = 0;
        while j < 8 {
            t[b][j] = (((b >> j) & 1) as i8) << 4;
            j += 1;
        }
        b += 1;
    }
    t
}
static HI_BITS: [[i8; 8]; 256] = build_hi_table();

/// Add the 5th bit from the `qh` word (bit j -> element j).
#[inline]
fn apply_high_bits(qh: u32, q: &mut [i8]) {
    for (i, &byte) in qh.to_le_bytes().iter().enumerate() {
        let hi = &HI_BITS[byte as usize];
        let dst: &mut [i8; 8] = (&mut q[i * 8..i * 8 + 8]).try_into().unwrap();
        for l in 0..8 {
            dst[l] += hi[l];
        }
    }
}

/// Unpack one block to centered int8 quants + (scale, min).
/// Invariant: x[i] = d * q[i] + m.
fn unpack_block(dtype: i32, block: &[u8], q: &mut [i8]) -> (f32, f32) {
    match dtype {
        2 => {
            unpack_nibbles(&block[2..], q, 8);
            (half(block), 0.0)
        }
        3 => {
            unpack_nibbles(&block[4..], q, 0);
            (half(block), half(&block[2..]))
        }
        6 => {
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            unpack_nibbles(&block[6..], q, 0);
            apply_high_bits(qh, q);
            for v in q.iter_mut().take(QK) {
                *v -= 16;
            }
            (half(block), 0.0)
        }
        7 => {
            let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            unpack_nibbles(&block[8..], q, 0);
            apply_high_bits(qh, q);
            (half(block), half(&block[2..]))
        }
        8 => {
            for j in 0..QK {
                q[j] = block[2 + j] as i8;
            }
            (half(block), 0.0)
        }
        t => panic!("unpack of unsupported dtype {t}"),
    }
}

/// A tensor kept in ggml quantized blocks, row-major: each of `shape[0]`
/// rows is `shape[1] / 32` consecutive blocks.
pub struct QTensor {
    pub shape: Vec<usize>,
    pub dtype: i32,
    pub raw: Vec<u8>,
}

impl QTensor {
    /// Dequantize row `i` into `out` (len `shape[1]`).
    pub fn row_f32(&self, i: usize, out: &mut [f32]) {
        let k = self.shape[1];
        let nb = k / QK;
        let bb = block_bytes(self.dtype).unwrap();
        let mut q = [0i8; QK];
        for bl in 0..nb {
            let block = &self.raw[(i * nb + bl) * bb..(i * nb + bl + 1) * bb];
            let (d, m) = unpack_block(self.dtype, block, &mut q);
            for l in 0..QK {
                out[bl * QK + l] = d * q[l] as f32 + m;
            }
        }
    }

    pub fn to_dense(&self) -> Tensor {
        let mut t = Tensor::zeros(&self.shape);
        let bb = block_bytes(self.dtype).unwrap();
        let mut q = [0i8; QK];
        for (block, out) in self.raw.chunks_exact(bb).zip(t.data.chunks_exact_mut(QK)) {
            let (d, m) = unpack_block(self.dtype, block, &mut q);
            for l in 0..QK {
                out[l] = d * q[l] as f32 + m;
            }
        }
        t
    }
}

/// A model weight: dense f32 or quantized blocks.
pub enum Weight {
    Dense(Tensor),
    Quant(QTensor),
}

impl Weight {
    pub fn shape(&self) -> &[usize] {
        match self {
            Weight::Dense(t) => &t.shape,
            Weight::Quant(q) => &q.shape,
        }
    }

    /// The dense tensor, for weights that are never quantized (biases,
    /// layernorms, positional embeddings, conv kernels).
    pub fn dense(&self) -> &Tensor {
        match self {
            Weight::Dense(t) => t,
            Weight::Quant(_) => panic!("expected dense weight, found quantized"),
        }
    }

    /// Copy row `i` as f32 (embedding gather).
    pub fn row_f32(&self, i: usize, out: &mut [f32]) {
        match self {
            Weight::Dense(t) => {
                let k = t.shape[t.shape.len() - 1];
                out.copy_from_slice(&t.data[i * k..(i + 1) * k]);
            }
            Weight::Quant(q) => q.row_f32(i, out),
        }
    }
}

/// C = A[m,k] x B^T with B quantized `[n,k]`. Each B row is dequantized
/// once into an f32 scratch and fed to the vectorized f32 dot kernels —
/// the unpack cost amortizes over all A rows, and only one row of B is
/// ever dense in memory per worker. Parallel over B-row chunks.
pub fn matmul_t_q(a: &Tensor, b: &QTensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (n, k2) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul_t_q inner dims: {k} vs {k2}");
    assert_eq!(k % QK, 0, "k must be a multiple of {QK}");
    let nb = k / QK;
    let bb = block_bytes(b.dtype).unwrap();
    let mut out = Tensor::zeros(&[m, n]);

    // Each worker fills columns [j0, j0+cols) of every output row into a
    // private [m, cols] buffer, copied back afterwards.
    let cols_block = |j0: usize, cols: usize, block_out: &mut [f32]| {
        let mut fb = vec![0.0f32; k];
        let mut q = [0i8; QK];
        for j in j0..j0 + cols {
            for bl in 0..nb {
                let block = &b.raw[(j * nb + bl) * bb..(j * nb + bl + 1) * bb];
                let (d, mn) = unpack_block(b.dtype, block, &mut q);
                for l in 0..QK {
                    fb[bl * QK + l] = d * q[l] as f32 + mn;
                }
            }
            let mut i = 0;
            while i + 4 <= m {
                let s = crate::tensor::dot4(
                    &a.data[i * k..(i + 1) * k],
                    &a.data[(i + 1) * k..(i + 2) * k],
                    &a.data[(i + 2) * k..(i + 3) * k],
                    &a.data[(i + 3) * k..(i + 4) * k],
                    &fb,
                );
                for r in 0..4 {
                    block_out[(i + r) * cols + (j - j0)] = s[r];
                }
                i += 4;
            }
            while i < m {
                block_out[i * cols + (j - j0)] = crate::tensor::dot(&a.data[i * k..(i + 1) * k], &fb);
                i += 1;
            }
        }
    };

    let threads = n_threads();
    if threads > 1 && m * n * k > PAR_THRESHOLD {
        let chunk = n.div_ceil(threads);
        let blocks: Vec<(usize, usize, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..n.div_ceil(chunk))
                .map(|c| {
                    let cols_block = &cols_block;
                    s.spawn(move || {
                        let j0 = c * chunk;
                        let cols = chunk.min(n - j0);
                        let mut buf = vec![0.0f32; m * cols];
                        cols_block(j0, cols, &mut buf);
                        (j0, cols, buf)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (j0, cols, buf) in blocks {
            for i in 0..m {
                out.data[i * n + j0..i * n + j0 + cols].copy_from_slice(&buf[i * cols..(i + 1) * cols]);
            }
        }
    } else {
        let mut buf = vec![0.0f32; m * n];
        cols_block(0, n, &mut buf);
        out.data.copy_from_slice(&buf);
    }
    out
}

/// y = x W^T + b, dispatching on the weight representation.
pub fn linear_w(x: &Tensor, w: &Weight, b: Option<&[f32]>) -> Tensor {
    let mut y = match w {
        Weight::Dense(t) => matmul_t(x, t),
        Weight::Quant(q) => matmul_t_q(x, q),
    };
    if let Some(b) = b {
        add_bias(&mut y, b);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantize an f32 slice to Q8_0 blocks (test helper — the loader only
    /// ever reads quantized files, it doesn't produce them).
    fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
        let mut raw = Vec::new();
        for block in x.chunks(QK) {
            let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            raw.extend_from_slice(&crate::model::f32_to_f16(d).to_le_bytes());
            for &v in block {
                raw.push(((v * id).round().clamp(-127.0, 127.0) as i8) as u8);
            }
        }
        raw
    }

    fn test_matrix(n: usize, k: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n * k)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn unpack_matches_f32_dequant_reference() {
        // Q5_1 block: d=0.5, m=2.0, bit 0 set, first nibble 3 -> q=19.
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x4000u16.to_le_bytes());
        block[4..8].copy_from_slice(&1u32.to_le_bytes());
        block[8] = 0x03;
        let mut q = [0i8; QK];
        let (d, m) = unpack_block(7, &block, &mut q);
        assert_eq!(d * q[0] as f32 + m, 11.5);
        assert_eq!(d * q[1] as f32 + m, 2.0);
    }

    #[test]
    fn qtensor_row_roundtrip() {
        let x = test_matrix(3, 64, 7);
        let qt = QTensor { shape: vec![3, 64], dtype: 8, raw: quantize_q8_0(&x) };
        let mut row = vec![0.0f32; 64];
        qt.row_f32(1, &mut row);
        for (a, b) in row.iter().zip(&x[64..128]) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn matmul_t_q_matches_dense_within_quant_error() {
        let (m, k, n) = (7, 96, 33);
        let a = Tensor::from_vec(&[m, k], test_matrix(1, m * k, 11));
        let bx = test_matrix(n, k, 13);
        let qt = QTensor { shape: vec![n, k], dtype: 8, raw: quantize_q8_0(&bx) };
        let fast = matmul_t_q(&a, &qt);
        let reference = matmul_t(&a, &qt.to_dense());
        let rms_ref = (reference.data.iter().map(|v| v * v).sum::<f32>() / reference.data.len() as f32).sqrt();
        for (f, r) in fast.data.iter().zip(&reference.data) {
            assert!(
                (f - r).abs() < 0.02 * rms_ref.max(1.0),
                "quantized {f} vs dense-on-dequant {r} (rms {rms_ref})"
            );
        }
    }

    #[test]
    fn matmul_t_q_parallel_matches_serial() {
        // Large enough to cross PAR_THRESHOLD.
        let (m, k, n) = (8, 128, 1200);
        let a = Tensor::from_vec(&[m, k], test_matrix(1, m * k, 21));
        let qt = QTensor { shape: vec![n, k], dtype: 8, raw: quantize_q8_0(&test_matrix(n, k, 23)) };
        let par = matmul_t_q(&a, &qt);
        // Serial reference: same math, one chunk.
        let dense = matmul_t_q(&Tensor::from_vec(&[1, k], a.data[..k].to_vec()), &qt);
        for j in 0..n {
            assert_eq!(par.data[j], dense.data[j], "col {j}");
        }
    }
}
