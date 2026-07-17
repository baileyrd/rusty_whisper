//! Whisper vocabulary: token-id -> text decoding and the special-token ids.
//!
//! Whisper's tokens are byte-level BPE, so an individual token need not be
//! valid UTF-8 — we accumulate bytes and convert lossily at the end, exactly
//! like whisper.cpp. Encoding (text -> ids) is only needed for prompt
//! injection and comes later.

use crate::model::HParams;

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    pub eot: u32,
    pub sot: u32,
    pub translate: u32,
    pub transcribe: u32,
    pub sot_prev: u32,
    pub no_speech: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    /// First language token (<|en|>); language i is `sot + 1 + i`.
    pub lang_begin: u32,
    pub n_langs: u32,
}

impl Tokenizer {
    pub fn new(vocab: Vec<Vec<u8>>, hp: &HParams) -> Self {
        // Special-token layout differs between English-only and multilingual
        // vocabularies (same scheme whisper.cpp hardcodes).
        let multilingual = hp.is_multilingual();
        let eot = if multilingual { 50257 } else { 50256 };
        let n_langs: u32 = if hp.n_vocab >= 51866 { 100 } else { 99 }; // large-v3 adds yue
        let sot = eot + 1;
        let translate = sot + 1 + n_langs;
        Tokenizer {
            vocab,
            eot,
            sot,
            lang_begin: sot + 1,
            n_langs,
            translate,
            transcribe: translate + 1,
            // Between transcribe and sot_prev sits <|startoflm|>; then
            // <|nospeech|>, <|notimestamps|>, and the timestamp range.
            sot_prev: translate + 3,
            no_speech: translate + 4,
            no_timestamps: translate + 5,
            timestamp_begin: translate + 6,
        }
    }

    pub fn is_special(&self, id: u32) -> bool {
        id >= self.eot
    }

    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.timestamp_begin
    }

    /// Timestamp token -> seconds (each step is 20 ms).
    pub fn timestamp_seconds(&self, id: u32) -> f32 {
        (id - self.timestamp_begin) as f32 * 0.02
    }

    /// Decode a token sequence to text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.is_special(id) {
                continue;
            }
            if let Some(tok) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(tok);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp_en() -> HParams {
        HParams { n_vocab: 51864, ..Default::default() }
    }

    fn hp_multi() -> HParams {
        HParams { n_vocab: 51865, ..Default::default() }
    }

    #[test]
    fn special_ids_english_only() {
        let t = Tokenizer::new(vec![], &hp_en());
        assert_eq!(t.eot, 50256);
        assert_eq!(t.sot, 50257);
        assert_eq!(t.timestamp_begin, 50363);
    }

    #[test]
    fn special_ids_multilingual() {
        let t = Tokenizer::new(vec![], &hp_multi());
        assert_eq!(t.eot, 50257);
        assert_eq!(t.sot, 50258);
        assert_eq!(t.transcribe, 50359);
        assert_eq!(t.timestamp_begin, 50364);
    }

    #[test]
    fn decode_skips_specials_and_joins_bytes() {
        let mut vocab = vec![Vec::new(); 10];
        vocab[1] = b"He".to_vec();
        vocab[2] = b"llo".to_vec();
        let mut t = Tokenizer::new(vocab, &hp_en());
        t.eot = 5; // pretend 5+ are special for this toy vocab
        assert_eq!(t.decode(&[1, 2, 7]), "Hello");
    }

    #[test]
    fn timestamp_math() {
        let t = Tokenizer::new(vec![], &hp_multi());
        assert!(t.is_timestamp(t.timestamp_begin));
        assert_eq!(t.timestamp_seconds(t.timestamp_begin + 50), 1.0);
    }
}
