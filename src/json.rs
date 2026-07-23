//! A minimal JSON value parser + string escaper — used by `whisper-server`'s
//! `/inference` response construction (indirectly, via hand-rolled
//! formatting elsewhere) and directly by `whisper-talk-llama`'s HTTP client
//! to read nested fields (`choices[0].message.content`) out of an
//! OpenAI-compatible chat-completions response, which a single-field scan
//! (see `server::json_string_field`) can't do. Zero-dependency by design —
//! this crate has no `serde`.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    /// Insertion order isn't preserved (a `BTreeMap` sorts by key) — fine
    /// for this crate's only use case, reading fields back out by name.
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(map) => map.get(key),
            _ => None,
        }
    }

    pub fn index(&self, i: usize) -> Option<&Value> {
        match self {
            Value::Array(items) => items.get(i),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// Escapes `s` as a JSON string literal, including the surrounding quotes.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn parse(input: &str) -> Result<Value, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;
    let value = parse_value(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos != chars.len() {
        return Err(format!("trailing data at offset {pos}"));
    }
    Ok(value)
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_value(chars: &[char], pos: &mut usize) -> Result<Value, String> {
    skip_ws(chars, pos);
    match chars.get(*pos) {
        Some('{') => parse_object(chars, pos),
        Some('[') => parse_array(chars, pos),
        Some('"') => Ok(Value::String(parse_string(chars, pos)?)),
        Some('t') => parse_literal(chars, pos, "true", Value::Bool(true)),
        Some('f') => parse_literal(chars, pos, "false", Value::Bool(false)),
        Some('n') => parse_literal(chars, pos, "null", Value::Null),
        Some(c) if c.is_ascii_digit() || *c == '-' => parse_number(chars, pos),
        Some(c) => Err(format!("unexpected character {c:?} at offset {pos}")),
        None => Err("unexpected end of input".to_string()),
    }
}

fn parse_literal(
    chars: &[char],
    pos: &mut usize,
    lit: &str,
    value: Value,
) -> Result<Value, String> {
    let lit_chars: Vec<char> = lit.chars().collect();
    if chars.len() < *pos + lit_chars.len() || chars[*pos..*pos + lit_chars.len()] != lit_chars[..]
    {
        return Err(format!("expected {lit:?} at offset {pos}"));
    }
    *pos += lit_chars.len();
    Ok(value)
}

fn parse_object(chars: &[char], pos: &mut usize) -> Result<Value, String> {
    *pos += 1; // '{'
    let mut map = BTreeMap::new();
    skip_ws(chars, pos);
    if chars.get(*pos) == Some(&'}') {
        *pos += 1;
        return Ok(Value::Object(map));
    }
    loop {
        skip_ws(chars, pos);
        if chars.get(*pos) != Some(&'"') {
            return Err(format!("expected object key at offset {pos}"));
        }
        let key = parse_string(chars, pos)?;
        skip_ws(chars, pos);
        if chars.get(*pos) != Some(&':') {
            return Err(format!("expected ':' at offset {pos}"));
        }
        *pos += 1;
        let value = parse_value(chars, pos)?;
        map.insert(key, value);
        skip_ws(chars, pos);
        match chars.get(*pos) {
            Some(',') => {
                *pos += 1;
            }
            Some('}') => {
                *pos += 1;
                break;
            }
            _ => return Err(format!("expected ',' or '}}' at offset {pos}")),
        }
    }
    Ok(Value::Object(map))
}

fn parse_array(chars: &[char], pos: &mut usize) -> Result<Value, String> {
    *pos += 1; // '['
    let mut items = Vec::new();
    skip_ws(chars, pos);
    if chars.get(*pos) == Some(&']') {
        *pos += 1;
        return Ok(Value::Array(items));
    }
    loop {
        items.push(parse_value(chars, pos)?);
        skip_ws(chars, pos);
        match chars.get(*pos) {
            Some(',') => {
                *pos += 1;
            }
            Some(']') => {
                *pos += 1;
                break;
            }
            _ => return Err(format!("expected ',' or ']' at offset {pos}")),
        }
    }
    Ok(Value::Array(items))
}

fn parse_string(chars: &[char], pos: &mut usize) -> Result<String, String> {
    *pos += 1; // opening quote
    let mut out = String::new();
    loop {
        match chars.get(*pos) {
            None => return Err("unterminated string".to_string()),
            Some('"') => {
                *pos += 1;
                break;
            }
            Some('\\') => {
                *pos += 1;
                match chars.get(*pos) {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        let cp = parse_hex4(chars, *pos + 1)?;
                        *pos += 4;
                        if (0xD800..=0xDBFF).contains(&cp) {
                            // High surrogate: expect a following \uXXXX low surrogate.
                            if chars.get(*pos + 1) == Some(&'\\')
                                && chars.get(*pos + 2) == Some(&'u')
                            {
                                let low = parse_hex4(chars, *pos + 3)?;
                                *pos += 6;
                                if (0xDC00..=0xDFFF).contains(&low) {
                                    let c = 0x10000 + (cp - 0xD800) * 0x400 + (low - 0xDC00);
                                    out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                } else {
                                    out.push('\u{FFFD}');
                                }
                            } else {
                                out.push('\u{FFFD}');
                            }
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                    }
                    other => return Err(format!("invalid escape {other:?}")),
                }
                *pos += 1;
            }
            Some(&c) => {
                out.push(c);
                *pos += 1;
            }
        }
    }
    Ok(out)
}

fn parse_hex4(chars: &[char], start: usize) -> Result<u32, String> {
    let s: String = chars
        .get(start..start + 4)
        .ok_or("truncated \\u escape")?
        .iter()
        .collect();
    u32::from_str_radix(&s, 16).map_err(|e| e.to_string())
}

fn parse_number(chars: &[char], pos: &mut usize) -> Result<Value, String> {
    let start = *pos;
    if chars.get(*pos) == Some(&'-') {
        *pos += 1;
    }
    while chars.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
        *pos += 1;
    }
    if chars.get(*pos) == Some(&'.') {
        *pos += 1;
        while chars.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
            *pos += 1;
        }
    }
    if matches!(chars.get(*pos), Some('e') | Some('E')) {
        *pos += 1;
        if matches!(chars.get(*pos), Some('+') | Some('-')) {
            *pos += 1;
        }
        while chars.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
            *pos += 1;
        }
    }
    let s: String = chars[start..*pos].iter().collect();
    s.parse::<f64>()
        .map(Value::Number)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_object() {
        let v = parse(r#"{"a": 1, "b": "two", "c": true, "d": null}"#).unwrap();
        assert_eq!(v.get("a"), Some(&Value::Number(1.0)));
        assert_eq!(v.get("b").and_then(Value::as_str), Some("two"));
        assert_eq!(v.get("c"), Some(&Value::Bool(true)));
        assert_eq!(v.get("d"), Some(&Value::Null));
    }

    #[test]
    fn parses_nested_array_and_object() {
        let v = parse(r#"{"choices":[{"message":{"role":"assistant","content":"hi there"}}]}"#)
            .unwrap();
        let content = v
            .get("choices")
            .and_then(|c| c.index(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str);
        assert_eq!(content, Some("hi there"));
    }

    #[test]
    fn parses_escapes() {
        let v = parse(r#""a\"b\\c\nd""#).unwrap();
        assert_eq!(v.as_str(), Some("a\"b\\c\nd"));
    }

    #[test]
    fn parses_unicode_escape() {
        let v = parse(r#""\u0041\u00e9""#).unwrap();
        assert_eq!(v.as_str(), Some("A\u{e9}"));
    }

    #[test]
    fn parses_surrogate_pair() {
        // U+1F600 (grinning face) as a UTF-16 surrogate pair.
        let v = parse(r#""\ud83d\ude00""#).unwrap();
        assert_eq!(v.as_str(), Some("\u{1F600}"));
    }

    #[test]
    fn parses_numbers() {
        assert_eq!(parse("42").unwrap(), Value::Number(42.0));
        assert_eq!(parse("-3.5").unwrap(), Value::Number(-3.5));
        assert_eq!(parse("1e3").unwrap(), Value::Number(1000.0));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(parse("{}garbage").is_err());
    }

    #[test]
    fn rejects_truncated_input() {
        assert!(parse(r#"{"a":"#).is_err());
        assert!(parse(r#""unterminated"#).is_err());
    }

    #[test]
    fn escape_round_trips_through_parse() {
        let s = "hello \"world\"\n\t\\ done";
        let escaped = escape(s);
        let v = parse(&escaped).unwrap();
        assert_eq!(v.as_str(), Some(s));
    }

    #[test]
    fn escape_control_char() {
        assert_eq!(escape("\u{1}"), "\"\\u0001\"");
    }
}
