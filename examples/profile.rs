use rusty_whisper::{audio, decoder::Decoder, encoder, model, tensor::Tensor, tokenizer::Tokenizer};
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m = model::load_model(&mut BufReader::new(File::open(&args[1]).unwrap())).unwrap();
    let samples = vec![0.01f32; audio::N_SAMPLES_30S];

    let t = Instant::now();
    let (mel, n_frames) = audio::log_mel_spectrogram(&samples, &m.mel_filters, m.hparams.n_mels as usize);
    println!("mel:     {:.3} s", t.elapsed().as_secs_f32());

    let mel = Tensor::from_vec(&[m.hparams.n_mels as usize, n_frames], mel);
    let t = Instant::now();
    let enc = encoder::encode(&m, &mel);
    println!("encoder: {:.3} s", t.elapsed().as_secs_f32());

    let tok = Tokenizer::new(m.vocab.clone(), &m.hparams);
    let t = Instant::now();
    let mut dec = Decoder::new(&m, &enc);
    println!("cross-kv: {:.3} s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let mut logits = dec.forward(&[tok.sot, tok.no_timestamps]);
    let n = 30;
    for _ in 0..n {
        let row = &logits.data[(logits.shape[0] - 1) * logits.shape[1]..];
        let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        logits = dec.forward(&[best.min(tok.eot - 1)]);
    }
    println!("decoder: {:.1} ms/token", t.elapsed().as_secs_f32() * 1000.0 / (n + 1) as f32);
}
