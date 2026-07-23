//! Whisper vocabulary: token-id -> text decoding and the special-token ids.
//!
//! Whisper's tokens are byte-level BPE, so an individual token need not be
//! valid UTF-8 — we accumulate bytes and convert lossily at the end, exactly
//! like whisper.cpp. Encoding (text -> ids) is only needed for prompt
//! injection and comes later.

use crate::model::HParams;

/// Language codes in Whisper token order: language i's token is
/// `sot + 1 + i`. (large-v3 appends "yue" as id 99.)
pub const LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Language id (token offset from `lang_begin`) for an ISO code.
pub fn lang_id_from_code(code: &str) -> Option<u32> {
    LANGUAGES.iter().position(|&c| c == code).map(|i| i as u32)
}

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
    /// Non-text-token ids whose decoded text has no alphanumeric
    /// characters (e.g. "...", "♪", "[MUSIC]"-style bracket fillers) —
    /// whisper.cpp's `--suppress-nst`/`-sns` suppression set.
    non_speech_ids: std::collections::HashSet<u32>,
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
        let non_speech_ids = vocab
            .iter()
            .enumerate()
            .filter(|(id, bytes)| {
                (*id as u32) < eot && !bytes.is_empty() && {
                    let text = String::from_utf8_lossy(bytes);
                    let trimmed = text.trim();
                    !trimmed.is_empty() && !trimmed.chars().any(|c| c.is_alphanumeric())
                }
            })
            .map(|(id, _)| id as u32)
            .collect();
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
            non_speech_ids,
        }
    }

    /// Whether `id` is a "non-speech" text token — punctuation/symbol-only,
    /// no alphanumeric characters (see `--suppress-nst`/`-sns`).
    pub fn is_non_speech_token(&self, id: u32) -> bool {
        self.non_speech_ids.contains(&id)
    }

    /// All non-speech token ids (see `is_non_speech_token`).
    pub fn non_speech_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.non_speech_ids.iter().copied()
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
        self.decode_impl(ids, false)
    }

    /// Decode a token sequence to text, including special/control tokens'
    /// own vocab text inline (e.g. `<|startoftranscript|>`) — whisper.cpp's
    /// `--print-special`/`-ps`.
    pub fn decode_with_specials(&self, ids: &[u32]) -> String {
        self.decode_impl(ids, true)
    }

    fn decode_impl(&self, ids: &[u32], include_specials: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.is_special(id) && !include_specials {
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
        HParams {
            n_vocab: 51864,
            ..Default::default()
        }
    }

    fn hp_multi() -> HParams {
        HParams {
            n_vocab: 51865,
            ..Default::default()
        }
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
    fn decode_with_specials_includes_special_token_text() {
        let mut vocab = vec![Vec::new(); 10];
        vocab[1] = b"He".to_vec();
        vocab[2] = b"llo".to_vec();
        vocab[7] = b"<|endoftext|>".to_vec();
        let mut t = Tokenizer::new(vocab, &hp_en());
        t.eot = 5;
        assert_eq!(t.decode(&[1, 2, 7]), "Hello");
        assert_eq!(t.decode_with_specials(&[1, 2, 7]), "Hello<|endoftext|>");
    }

    #[test]
    fn non_speech_tokens_are_symbol_only() {
        let mut vocab = vec![Vec::new(); 10];
        vocab[1] = b"Hello".to_vec();
        vocab[2] = b"...".to_vec();
        vocab[3] = " ♪".as_bytes().to_vec();
        vocab[4] = b"a1".to_vec(); // alphanumeric mixed with nothing else
        let t = Tokenizer::new(vocab, &hp_en());
        assert!(!t.is_non_speech_token(1));
        assert!(t.is_non_speech_token(2));
        assert!(t.is_non_speech_token(3));
        assert!(!t.is_non_speech_token(4));
        assert!(!t.is_non_speech_token(0)); // empty vocab entry
    }

    #[test]
    fn language_table() {
        assert_eq!(lang_id_from_code("en"), Some(0));
        assert_eq!(lang_id_from_code("de"), Some(2));
        assert_eq!(lang_id_from_code("yue"), Some(99));
        assert_eq!(lang_id_from_code("xx"), None);
        // Token for language i is sot + 1 + i.
        let t = Tokenizer::new(vec![], &hp_multi());
        assert_eq!(t.lang_begin + lang_id_from_code("de").unwrap(), 50261);
    }

    #[test]
    fn timestamp_math() {
        let t = Tokenizer::new(vec![], &hp_multi());
        assert!(t.is_timestamp(t.timestamp_begin));
        assert_eq!(t.timestamp_seconds(t.timestamp_begin + 50), 1.0);
    }
}
