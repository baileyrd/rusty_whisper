//! Audio front-end: 16 kHz PCM -> log-mel spectrogram.
//!
//! Mirrors whisper.cpp's `log_mel_spectrogram` (which itself mirrors OpenAI
//! Whisper's `audio.py`): periodic Hann window, 400-point FFT with hop 160,
//! Slaney-normalized mel filterbank, log10 + dynamic-range clamp + affine
//! normalization.

use crate::tensor::{matmul_t, par_row_chunks, Tensor};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MEL: usize = 80;
/// Number of frequency bins kept from the FFT (onesided spectrum).
pub const N_FREQS: usize = N_FFT / 2 + 1;

/// Complex FFT, interleaved-free: separate re/im slices, in -> out.
/// Cooley-Tukey for even n, naive DFT for odd n (n = 400 = 2^4 * 25, so the
/// recursion bottoms out in a 25-point DFT — same approach as whisper.cpp).
fn fft(re: &[f32], im: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    if n == 1 {
        return (re.to_vec(), im.to_vec());
    }
    if n % 2 == 1 {
        return dft(re, im);
    }
    let half = n / 2;
    let (mut er, mut ei) = (Vec::with_capacity(half), Vec::with_capacity(half));
    let (mut or, mut oi) = (Vec::with_capacity(half), Vec::with_capacity(half));
    for i in 0..half {
        er.push(re[2 * i]);
        ei.push(im[2 * i]);
        or.push(re[2 * i + 1]);
        oi.push(im[2 * i + 1]);
    }
    let (er, ei) = fft(&er, &ei);
    let (or_, oi_) = fft(&or, &oi);

    let mut out_re = vec![0.0; n];
    let mut out_im = vec![0.0; n];
    for k in 0..half {
        let theta = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        let (s, c) = theta.sin_cos();
        let tr = c * or_[k] - s * oi_[k];
        let ti = c * oi_[k] + s * or_[k];
        out_re[k] = er[k] + tr;
        out_im[k] = ei[k] + ti;
        out_re[k + half] = er[k] - tr;
        out_im[k + half] = ei[k] - ti;
    }
    (out_re, out_im)
}

fn dft(re: &[f32], im: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let mut out_re = vec![0.0; n];
    let mut out_im = vec![0.0; n];
    for k in 0..n {
        let (mut sr, mut si) = (0.0f32, 0.0f32);
        for t in 0..n {
            let theta = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
            let (s, c) = theta.sin_cos();
            sr += re[t] * c - im[t] * s;
            si += re[t] * s + im[t] * c;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
    (out_re, out_im)
}

/// Periodic Hann window of length `n`.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

/// Slaney-scale hz->mel (librosa default, what Whisper's filters use).
fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / (6.4f32.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * (6.4f32.ln() / 27.0)).exp()
    }
}

/// Slaney-normalized triangular mel filterbank, `n_mels` x `N_FREQS`,
/// row-major. Matches `librosa.filters.mel(sr=16000, n_fft=400, n_mels=..)`,
/// which is what ggml model files embed. The loader prefers the embedded
/// filters; this generator exists for tests and for files without them.
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(fmax);
    // n_mels + 2 band edges, uniform in mel space.
    let edges: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_max * i as f32 / (n_mels + 1) as f32))
        .collect();
    let mut filters = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let (f_lo, f_center, f_hi) = (edges[m], edges[m + 1], edges[m + 2]);
        // Slaney normalization: constant filter energy per band.
        let norm = 2.0 / (f_hi - f_lo);
        for k in 0..n_freqs {
            let f = k as f32 * sample_rate as f32 / n_fft as f32;
            let w = ((f - f_lo) / (f_center - f_lo)).min((f_hi - f) / (f_hi - f_center));
            if w > 0.0 {
                filters[m * n_freqs + k] = w * norm;
            }
        }
    }
    filters
}

/// Samples in a full 30-second Whisper window.
pub const N_SAMPLES_30S: usize = 30 * SAMPLE_RATE;

/// Zero-pad or trim to exactly `n` samples (Whisper models consume fixed
/// 30 s windows; longer audio is chunked at the pipeline level).
pub fn pad_or_trim(samples: &[f32], n: usize) -> Vec<f32> {
    let mut out = samples[..samples.len().min(n)].to_vec();
    out.resize(n, 0.0);
    out
}

/// Log-mel spectrogram of `samples` (16 kHz mono f32 in [-1, 1]).
///
/// `filters` is an `n_mels` x `N_FREQS` row-major filterbank (embedded in the
/// model file, or from [`mel_filterbank`]). Returns `(data, n_frames)` where
/// `data` is `n_mels` x `n_frames` row-major — the layout whisper.cpp uses.
pub fn log_mel_spectrogram(samples: &[f32], filters: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
    assert_eq!(filters.len(), n_mels * N_FREQS, "filterbank shape mismatch");
    let window = hann_window(N_FFT);

    // Reflection-pad N_FFT/2 on both sides (torch.stft center=True).
    let pad = N_FFT / 2;
    let n = samples.len();
    let padded: Vec<f32> = (0..n + 2 * pad)
        .map(|i| {
            let j = i as isize - pad as isize;
            let j = if j < 0 {
                (-j) as usize
            } else if j as usize >= n {
                2 * (n - 1) - j as usize
            } else {
                j as usize
            };
            samples[j.min(n - 1)]
        })
        .collect();

    // Same frame count as OpenAI: exactly n / hop frames. FFTs run in
    // parallel into a [n_frames, N_FREQS] power matrix; the filterbank is
    // then applied as one matmul (mel[m,t] = filters[m,:] . power[t,:]).
    let n_frames = n / HOP_LENGTH;
    let mut power = Tensor::zeros(&[n_frames, N_FREQS]);
    par_row_chunks(&mut power.data, n_frames, N_FREQS, |t0, rows| {
        for (r, prow) in rows.chunks_mut(N_FREQS).enumerate() {
            let t = t0 + r;
            let frame = &padded[t * HOP_LENGTH..t * HOP_LENGTH + N_FFT];
            let re: Vec<f32> = frame.iter().zip(&window).map(|(x, w)| x * w).collect();
            let im = vec![0.0f32; N_FFT];
            let (fr, fi) = fft(&re, &im);
            for k in 0..N_FREQS {
                prow[k] = fr[k] * fr[k] + fi[k] * fi[k];
            }
        }
    });
    let filt = Tensor::from_vec(&[n_mels, N_FREQS], filters.to_vec());
    let mut mel = matmul_t(&filt, &power).data;
    for v in mel.iter_mut() {
        *v = v.max(1e-10).log10();
    }

    // Dynamic-range compression: clamp to (max - 8), then (x + 4) / 4.
    let mmax = mel.iter().cloned().fold(f32::MIN, f32::max);
    for v in mel.iter_mut() {
        *v = (v.max(mmax - 8.0) + 4.0) / 4.0;
    }
    (mel, n_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        // 400-point mixed-radix FFT must equal the naive DFT.
        let re: Vec<f32> = (0..400).map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5).collect();
        let im = vec![0.0f32; 400];
        let (fr, fi) = fft(&re, &im);
        let (dr, di) = dft(&re, &im);
        for k in 0..400 {
            assert!((fr[k] - dr[k]).abs() < 1e-2, "re bin {k}: {} vs {}", fr[k], dr[k]);
            assert!((fi[k] - di[k]).abs() < 1e-2, "im bin {k}: {} vs {}", fi[k], di[k]);
        }
    }

    #[test]
    fn hann_endpoints_and_symmetry() {
        let w = hann_window(N_FFT);
        assert!(w[0].abs() < 1e-7);
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-6);
        for i in 1..N_FFT {
            assert!((w[i] - w[N_FFT - i]).abs() < 1e-5);
        }
    }

    #[test]
    fn filterbank_shape_and_coverage() {
        let fb = mel_filterbank(N_MEL, N_FFT, SAMPLE_RATE);
        assert_eq!(fb.len(), N_MEL * N_FREQS);
        // Every mel band has some support; every band is non-negative.
        for m in 0..N_MEL {
            let row = &fb[m * N_FREQS..(m + 1) * N_FREQS];
            assert!(row.iter().any(|&v| v > 0.0), "empty mel band {m}");
            assert!(row.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn pure_tone_lands_in_matching_mel_band() {
        // 1 kHz tone: the hottest mel band's filter must peak near 1 kHz.
        let fb = mel_filterbank(N_MEL, N_FFT, SAMPLE_RATE);
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let (mel, n_frames) = log_mel_spectrogram(&samples, &fb, N_MEL);
        assert_eq!(n_frames, SAMPLE_RATE / HOP_LENGTH);

        // Middle frame, hottest band.
        let t = n_frames / 2;
        let hot = (0..N_MEL)
            .max_by(|&a, &b| mel[a * n_frames + t].total_cmp(&mel[b * n_frames + t]))
            .unwrap();
        let row = &fb[hot * N_FREQS..(hot + 1) * N_FREQS];
        let peak_bin = (0..N_FREQS).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        let peak_hz = peak_bin as f32 * SAMPLE_RATE as f32 / N_FFT as f32;
        assert!((peak_hz - 1000.0).abs() < 100.0, "hot band peaks at {peak_hz} Hz");
    }

    #[test]
    fn output_is_normalized() {
        let fb = mel_filterbank(N_MEL, N_FFT, SAMPLE_RATE);
        let samples = vec![0.01f32; SAMPLE_RATE];
        let (mel, _) = log_mel_spectrogram(&samples, &fb, N_MEL);
        // After clamp to max-8 and (x+4)/4, spread is at most 2.0.
        let max = mel.iter().cloned().fold(f32::MIN, f32::max);
        let min = mel.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max - min <= 2.0 + 1e-5);
    }
}
