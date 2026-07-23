//! Output file writers for transcription results — mirrors whisper.cpp's
//! `-otxt`/`-ovtt`/`-osrt`/`-ocsv`/`-oj`/`-ojf` (`examples/cli/cli.cpp`).

use std::io::{self, Write};

use crate::transcribe::Segment;

/// Plain text: one line per segment, no timestamps.
pub fn write_txt<W: Write>(segments: &[Segment], w: &mut W) -> io::Result<()> {
    for seg in segments {
        writeln!(w, "{}", seg.text.trim())?;
    }
    Ok(())
}

/// WebVTT: `HH:MM:SS.mmm --> HH:MM:SS.mmm` blocks separated by a blank line.
pub fn write_vtt<W: Write>(segments: &[Segment], w: &mut W) -> io::Result<()> {
    writeln!(w, "WEBVTT")?;
    writeln!(w)?;
    for seg in segments {
        writeln!(
            w,
            "{} --> {}",
            crate::transcribe::format_timestamp(seg.t0),
            crate::transcribe::format_timestamp(seg.t1)
        )?;
        writeln!(w, "{}", seg.text.trim())?;
        writeln!(w)?;
    }
    Ok(())
}

/// SubRip: 1-based index, `HH:MM:SS,mmm --> HH:MM:SS,mmm` (comma decimal
/// separator, unlike VTT's period), text, blank line.
pub fn write_srt<W: Write>(segments: &[Segment], w: &mut W) -> io::Result<()> {
    for (i, seg) in segments.iter().enumerate() {
        writeln!(w, "{}", i + 1)?;
        writeln!(w, "{} --> {}", srt_timestamp(seg.t0), srt_timestamp(seg.t1))?;
        writeln!(w, "{}", seg.text.trim())?;
        writeln!(w)?;
    }
    Ok(())
}

fn srt_timestamp(secs: f32) -> String {
    crate::transcribe::format_timestamp(secs).replace('.', ",")
}

/// CSV: header row `start,end,text` (start/end in milliseconds), text
/// double-quoted with embedded quotes escaped by doubling.
pub fn write_csv<W: Write>(segments: &[Segment], w: &mut W) -> io::Result<()> {
    writeln!(w, "start,end,text")?;
    for seg in segments {
        let t0_ms = (seg.t0 * 1000.0).round() as i64;
        let t1_ms = (seg.t1 * 1000.0).round() as i64;
        let text = seg.text.trim().replace('"', "\"\"");
        writeln!(w, "{t0_ms},{t1_ms},\"{text}\"")?;
    }
    Ok(())
}

/// JSON: `{"language": ..., "transcription": [{"offsets": {"from", "to"},
/// "timestamps": {"from", "to"}, "text"}, ...]}` — the plain `-oj` shape.
/// For token-level data see [`write_json_full`] (`-ojf`).
pub fn write_json<W: Write>(language: &str, segments: &[Segment], w: &mut W) -> io::Result<()> {
    write_json_impl(language, segments, w, false)
}

/// Extended JSON (`-ojf`): same shape as [`write_json`], plus a `"tokens"`
/// array per segment with each token's `id`, decoded `text`, `p`
/// (probability), `plog` (log-probability), and `t0`/`t1` (interpolated —
/// see [`crate::transcribe::TokenInfo`]'s docs for the caveat).
pub fn write_json_full<W: Write>(
    language: &str,
    segments: &[Segment],
    w: &mut W,
) -> io::Result<()> {
    write_json_impl(language, segments, w, true)
}

fn write_json_impl<W: Write>(
    language: &str,
    segments: &[Segment],
    w: &mut W,
    full: bool,
) -> io::Result<()> {
    writeln!(w, "{{")?;
    writeln!(w, "  \"language\": \"{}\",", json_escape(language))?;
    writeln!(w, "  \"transcription\": [")?;
    for (i, seg) in segments.iter().enumerate() {
        let t0_ms = (seg.t0 * 1000.0).round() as i64;
        let t1_ms = (seg.t1 * 1000.0).round() as i64;
        writeln!(w, "    {{")?;
        writeln!(
            w,
            "      \"timestamps\": {{ \"from\": \"{}\", \"to\": \"{}\" }},",
            srt_timestamp(seg.t0),
            srt_timestamp(seg.t1)
        )?;
        writeln!(
            w,
            "      \"offsets\": {{ \"from\": {t0_ms}, \"to\": {t1_ms} }},",
        )?;
        if full {
            writeln!(w, "      \"text\": \"{}\",", json_escape(seg.text.trim()))?;
            writeln!(w, "      \"tokens\": [")?;
            for (j, tk) in seg.tokens.iter().enumerate() {
                writeln!(w, "        {{")?;
                writeln!(w, "          \"id\": {},", tk.id)?;
                writeln!(w, "          \"text\": \"{}\",", json_escape(&tk.text))?;
                writeln!(w, "          \"p\": {:.6},", tk.prob)?;
                writeln!(w, "          \"plog\": {:.6},", tk.logprob)?;
                writeln!(
                    w,
                    "          \"timestamps\": {{ \"from\": \"{}\", \"to\": \"{}\" }}",
                    srt_timestamp(tk.t0),
                    srt_timestamp(tk.t1)
                )?;
                write!(w, "        }}")?;
                writeln!(w, "{}", if j + 1 < seg.tokens.len() { "," } else { "" })?;
            }
            writeln!(w, "      ]")?;
        } else {
            writeln!(w, "      \"text\": \"{}\"", json_escape(seg.text.trim()))?;
        }
        write!(w, "    }}")?;
        if i + 1 < segments.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs() -> Vec<Segment> {
        vec![
            Segment {
                t0: 0.0,
                t1: 2.5,
                text: " Hello world".to_string(),
                tokens: Vec::new(),
            },
            Segment {
                t0: 2.5,
                t1: 5.0,
                text: " Second segment with \"quotes\"".to_string(),
                tokens: Vec::new(),
            },
        ]
    }

    #[test]
    fn txt_one_line_per_segment_trimmed() {
        let mut out = Vec::new();
        write_txt(&segs(), &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "Hello world\nSecond segment with \"quotes\"\n"
        );
    }

    #[test]
    fn vtt_has_header_and_period_timestamps() {
        let mut out = Vec::new();
        write_vtt(&segs(), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("WEBVTT\n\n"));
        assert!(s.contains("00:00:00.000 --> 00:00:02.500"));
        assert!(s.contains("Hello world"));
    }

    #[test]
    fn srt_has_index_and_comma_timestamps() {
        let mut out = Vec::new();
        write_srt(&segs(), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("1\n00:00:00,000 --> 00:00:02,500\n"));
        assert!(s.contains("\n2\n00:00:02,500 --> 00:00:05,000\n"));
    }

    #[test]
    fn csv_header_and_quote_escaping() {
        let mut out = Vec::new();
        write_csv(&segs(), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        let mut lines = s.lines();
        assert_eq!(lines.next().unwrap(), "start,end,text");
        assert_eq!(lines.next().unwrap(), "0,2500,\"Hello world\"");
        assert_eq!(
            lines.next().unwrap(),
            "2500,5000,\"Second segment with \"\"quotes\"\"\""
        );
    }

    #[test]
    fn json_is_well_formed_and_escapes() {
        let mut out = Vec::new();
        write_json("en", &segs(), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\"language\": \"en\""));
        assert!(s.contains("\"text\": \"Second segment with \\\"quotes\\\"\""));
        // Brace balance as a cheap well-formedness check (no JSON dep).
        let opens = s.matches('{').count();
        let closes = s.matches('}').count();
        assert_eq!(opens, closes);
    }

    #[test]
    fn empty_segments_still_produce_valid_shells() {
        let mut out = Vec::new();
        write_json("en", &[], &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\"transcription\": [\n  ]\n") || s.contains("[\n  ]"));

        let mut out = Vec::new();
        write_vtt(&[], &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "WEBVTT\n\n");
    }

    #[test]
    fn json_full_includes_token_array_and_stays_balanced() {
        use crate::transcribe::TokenInfo;

        let mut segments = segs();
        segments[0].tokens = vec![
            TokenInfo {
                id: 100,
                text: "Hello".to_string(),
                prob: 0.9,
                logprob: -0.105,
                t0: 0.0,
                t1: 1.0,
            },
            TokenInfo {
                id: 101,
                text: " world".to_string(),
                prob: 0.8,
                logprob: -0.223,
                t0: 1.0,
                t1: 2.5,
            },
        ];

        let mut out = Vec::new();
        write_json_full("en", &segments, &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\"tokens\": ["));
        assert!(s.contains("\"id\": 100"));
        assert!(s.contains("\"text\": \"Hello\""));
        assert!(s.contains("\"p\": 0.900000"));
        // The second segment has no tokens: an empty array, not an error.
        assert!(s.contains("\"tokens\": [\n      ]") || s.contains("\"tokens\": []"));
        let opens = s.matches('{').count();
        let closes = s.matches('}').count();
        assert_eq!(opens, closes);
        let brackets_open = s.matches('[').count();
        let brackets_close = s.matches(']').count();
        assert_eq!(brackets_open, brackets_close);
    }
}
