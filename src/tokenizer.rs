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

/// Full English language names, index-aligned with [`LANGUAGES`] — matches
/// whisper.cpp's `g_lang` table (`whisper_lang_str_full`).
pub const LANGUAGE_NAMES: [&str; 100] = [
    "english",
    "chinese",
    "german",
    "spanish",
    "russian",
    "korean",
    "french",
    "japanese",
    "portuguese",
    "turkish",
    "polish",
    "catalan",
    "dutch",
    "arabic",
    "swedish",
    "italian",
    "indonesian",
    "hindi",
    "finnish",
    "vietnamese",
    "hebrew",
    "ukrainian",
    "greek",
    "malay",
    "czech",
    "romanian",
    "danish",
    "hungarian",
    "tamil",
    "norwegian",
    "thai",
    "urdu",
    "croatian",
    "bulgarian",
    "lithuanian",
    "latin",
    "maori",
    "malayalam",
    "welsh",
    "slovak",
    "telugu",
    "persian",
    "latvian",
    "bengali",
    "serbian",
    "azerbaijani",
    "slovenian",
    "kannada",
    "estonian",
    "macedonian",
    "breton",
    "basque",
    "icelandic",
    "armenian",
    "nepali",
    "mongolian",
    "bosnian",
    "kazakh",
    "albanian",
    "swahili",
    "galician",
    "marathi",
    "punjabi",
    "sinhala",
    "khmer",
    "shona",
    "yoruba",
    "somali",
    "afrikaans",
    "occitan",
    "georgian",
    "belarusian",
    "tajik",
    "sindhi",
    "gujarati",
    "amharic",
    "yiddish",
    "lao",
    "uzbek",
    "faroese",
    "haitian creole",
    "pashto",
    "turkmen",
    "nynorsk",
    "maltese",
    "sanskrit",
    "luxembourgish",
    "myanmar",
    "tibetan",
    "tagalog",
    "malagasy",
    "assamese",
    "tatar",
    "hawaiian",
    "lingala",
    "hausa",
    "bashkir",
    "javanese",
    "sundanese",
    "cantonese",
];

/// ISO code for a language id (reverse of [`lang_id_from_code`]) —
/// `whisper_lang_str`.
pub fn lang_str(id: u32) -> Option<&'static str> {
    LANGUAGES.get(id as usize).copied()
}

/// Full English name for a language id — `whisper_lang_str_full`.
pub fn lang_str_full(id: u32) -> Option<&'static str> {
    LANGUAGE_NAMES.get(id as usize).copied()
}

/// Highest valid language id — `whisper_lang_max_id`.
pub fn lang_max_id() -> u32 {
    LANGUAGES.len() as u32 - 1
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
    /// Reverse of `vocab`: token bytes -> id, built once for [`encode`](Self::encode)'s
    /// greedy longest-match. Later entries win ties (matches `Vec`
    /// iteration order into a `HashMap` insert); the model files this
    /// loads from never define the same byte sequence twice, so this
    /// doesn't matter in practice.
    encode_map: std::collections::HashMap<Vec<u8>, u32>,
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
        let encode_map = vocab
            .iter()
            .enumerate()
            .map(|(id, bytes)| (bytes.clone(), id as u32))
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
            encode_map,
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

    /// Encode arbitrary text into this model's vocabulary — needed to turn
    /// an initial `--prompt` into decode context. Mirrors whisper.cpp's own
    /// `whisper_tokenize`/`tokenize()` exactly: a GPT-2-style pretokenizer
    /// (splitting into contraction suffixes, letter runs, digit runs,
    /// "other" runs, and whitespace, each with the leading-space-grouping
    /// quirk the original regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|
    /// ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` has), followed by GREEDY
    /// LONGEST-MATCH against the vocabulary — not rank-ordered BPE merges.
    /// This is not a simplification: whisper.cpp's own encoder works this
    /// way (its `whisper_vocab` struct never stores merge ranks, only
    /// token <-> id maps), which is exactly why no separately-vendored
    /// merge-rank table is needed here — `vocab` (already loaded from the
    /// model file) is the only data this requires.
    ///
    /// whisper.cpp's pretokenizer regex uses POSIX ASCII character classes
    /// (`std::regex` has no Unicode property escapes), so this operates on
    /// raw bytes rather than decoded chars: non-ASCII UTF-8 bytes (multi-
    /// byte characters) fall into the catch-all "other" class byte-by-byte
    /// instead of being grouped as letters, same as upstream.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pretokenize(text.as_bytes()) {
            self.encode_piece(piece, &mut out);
        }
        out
    }

    /// Greedy longest-match of `piece` against `encode_map`: at each
    /// position, take the longest byte-slice starting there that exists in
    /// the vocabulary; if none does (down to a single byte), log it and
    /// skip that one byte — mirrors whisper.cpp's own fallback for the
    /// same case.
    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut i = 0;
        while i < piece.len() {
            let mut matched = None;
            for len in (1..=piece.len() - i).rev() {
                if let Some(&id) = self.encode_map.get(&piece[i..i + len]) {
                    matched = Some((id, len));
                    break;
                }
            }
            match matched {
                Some((id, len)) => {
                    out.push(id);
                    i += len;
                }
                None => {
                    crate::log::log(format!(
                        "encode: unknown byte {:#04x} at offset {i}, skipping",
                        piece[i]
                    ));
                    i += 1;
                }
            }
        }
    }
}

fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// GPT-2's pretokenizer regex, hand-rolled over raw bytes (see
/// [`Tokenizer::encode`]'s doc comment for why bytes, not chars).
fn pretokenize(bytes: &[u8]) -> Vec<&[u8]> {
    const CONTRACTIONS: [&[u8]; 7] = [b"'s", b"'t", b"'re", b"'ve", b"'m", b"'ll", b"'d"];
    let n = bytes.len();
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < n {
        if let Some(c) = CONTRACTIONS.iter().find(|c| bytes[i..].starts_with(c)) {
            pieces.push(&bytes[i..i + c.len()]);
            i += c.len();
            continue;
        }

        let has_lead_space = bytes[i] == b' ';
        let class_pos = if has_lead_space { i + 1 } else { i };
        if let Some(&cb) = bytes.get(class_pos) {
            if cb.is_ascii_alphabetic() || cb.is_ascii_digit() || !is_ascii_ws(cb) {
                let is_class = |b: u8| -> bool {
                    if cb.is_ascii_alphabetic() {
                        b.is_ascii_alphabetic()
                    } else if cb.is_ascii_digit() {
                        b.is_ascii_digit()
                    } else {
                        !is_ascii_ws(b) && !b.is_ascii_alphabetic() && !b.is_ascii_digit()
                    }
                };
                let start = i;
                let mut j = class_pos;
                while j < n && is_class(bytes[j]) {
                    j += 1;
                }
                pieces.push(&bytes[start..j]);
                i = j;
                continue;
            }
        }

        // Whitespace run: matches `\s+(?!\S)` backtracked by one byte
        // (leaving it for the next piece's leading-space check) unless the
        // run reaches the end of the input, where the lookahead is moot.
        let start = i;
        let mut j = i;
        while j < n && is_ascii_ws(bytes[j]) {
            j += 1;
        }
        let run_len = j - start;
        let consume = if j == n {
            run_len
        } else {
            (run_len - 1).max(1)
        };
        pieces.push(&bytes[start..start + consume]);
        i = start + consume;
    }
    pieces
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
    fn lang_str_round_trips_with_lang_id_from_code() {
        for code in LANGUAGES {
            let id = lang_id_from_code(code).unwrap();
            assert_eq!(lang_str(id), Some(code));
        }
    }

    #[test]
    fn lang_str_full_known_values() {
        assert_eq!(lang_str_full(0), Some("english"));
        assert_eq!(lang_str_full(2), Some("german"));
        assert_eq!(lang_str_full(99), Some("cantonese"));
    }

    #[test]
    fn lang_str_out_of_range_is_none() {
        assert_eq!(lang_str(100), None);
        assert_eq!(lang_str_full(100), None);
    }

    #[test]
    fn lang_max_id_matches_table_length() {
        assert_eq!(lang_max_id(), 99);
        assert_eq!(LANGUAGES.len(), LANGUAGE_NAMES.len());
    }

    #[test]
    fn timestamp_math() {
        let t = Tokenizer::new(vec![], &hp_multi());
        assert!(t.is_timestamp(t.timestamp_begin));
        assert_eq!(t.timestamp_seconds(t.timestamp_begin + 50), 1.0);
    }

    /// 256 single-byte tokens (ids 0-255, guaranteeing a fallback for any
    /// byte, same as a real byte-level BPE vocab) plus a handful of
    /// "merged" multi-byte tokens to exercise greedy longest-match.
    fn synthetic_encode_vocab() -> Vec<Vec<u8>> {
        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        vocab.push(b"he".to_vec()); // 256
        vocab.push(b"llo".to_vec()); // 257
        vocab.push(b"hello".to_vec()); // 258 - longer, should win over he+llo
        vocab.push(b" the".to_vec()); // 259
        vocab.push(b"the".to_vec()); // 260
        vocab
    }

    #[test]
    fn encode_prefers_the_longest_vocab_match() {
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        assert_eq!(t.encode("hello"), vec![258]);
    }

    #[test]
    fn encode_groups_a_leading_space_with_the_following_word() {
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        assert_eq!(t.encode(" the"), vec![259]);
        assert_eq!(t.encode("the"), vec![260]);
    }

    #[test]
    fn encode_falls_back_to_individual_bytes_when_unmatched() {
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        // No multi-byte entry for "xyz" or any sub-run of it.
        assert_eq!(t.encode("xyz"), vec![b'x' as u32, b'y' as u32, b'z' as u32]);
    }

    #[test]
    fn encode_splits_contractions_from_the_preceding_word() {
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        // "don't" pretokenizes as ["don", "'t"] (the apostrophe isn't
        // alphabetic, so it can't extend the preceding letter run) --
        // encoding it should match encoding the two pieces separately.
        let mut expected = t.encode("don");
        expected.extend(t.encode("'t"));
        assert_eq!(t.encode("don't"), expected);
    }

    #[test]
    fn encode_decode_round_trips_representative_strings() {
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        for s in ["hello", " the", "the hello", "a  b", "hi!", "don't stop"] {
            assert_eq!(t.decode(&t.encode(s)), s, "round-trip failed for {s:?}");
        }
    }

    #[test]
    fn encode_empty_string_is_empty() {
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        assert!(t.encode("").is_empty());
    }

    #[test]
    fn encode_trailing_whitespace_run_is_one_piece() {
        // A whitespace run at the very end of input has nothing to leave a
        // byte behind for, so it should be consumed as a single run rather
        // than split byte-by-byte the way an interior run is.
        let t = Tokenizer::new(synthetic_encode_vocab(), &hp_en());
        let ids = t.encode("hi   ");
        assert_eq!(t.decode(&ids), "hi   ");
    }
}
