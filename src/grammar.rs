//! GBNF-lite grammar-constrained decoding — whisper.cpp's `--grammar`/
//! `--grammar-rule`/`--grammar-penalty`.
//!
//! This supports the subset of GBNF that covers whisper.cpp's own primary
//! use case (command-style grammars, e.g. `whisper-command`'s
//! general-purpose mode): rules built from string literals, rule
//! references, and alternation/concatenation/parenthesized grouping.
//! **Not supported**: character classes (`[a-z]`), repetition operators
//! (`*`, `+`, `?`), and negation — a rule using any of these is a parse
//! error rather than a silent mismatch. Full GBNF (matching llama.cpp's
//! grammar engine, which resolves these operators against a live
//! character-level parse stack) is a substantially larger undertaking than
//! this pass's scope; this subset is a deliberate, documented cut, not a
//! silent approximation.
//!
//! Because the supported subset has no repetition/recursion operators, the
//! start rule's language is finite: [`Grammar::parse`] enumerates every
//! complete string it can produce up front (rejecting a grammar whose rule
//! graph cycles, and capping the enumeration size), then decoding is
//! constrained to only ever produce a prefix of one of those strings.

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum Term {
    Literal(String),
    Ref(String),
}

type Alternative = Vec<Term>;

/// A parsed grammar: every complete string its start rule can produce.
#[derive(Clone, Debug)]
pub struct Grammar {
    candidates: Vec<String>,
    /// Prefix trie over `candidates`, so consistency checks are
    /// O(text length) instead of O(candidate count × text length) — this
    /// matters because the decode loop checks every vocabulary token
    /// (tens of thousands) against the grammar at every step.
    trie: TrieNode,
}

/// Cap on enumerated candidate strings — grammars are meant for small,
/// command-style phrase lists; a cap keeps a mistakenly large/ambiguous
/// grammar from hanging the parser instead of failing fast.
const MAX_CANDIDATES: usize = 4096;

#[derive(Clone, Debug, Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
}

impl TrieNode {
    fn insert(&mut self, s: &str) {
        let mut node = self;
        for c in s.chars() {
            node = node.children.entry(c).or_default();
        }
        node.is_end = true;
    }

    /// Walks `text` from the root; `None` if some prefix of `text` isn't on
    /// any candidate path, otherwise the node reached (`is_end` tells the
    /// caller whether `text` exactly completes a candidate).
    fn walk(&self, text: &str) -> Option<&TrieNode> {
        let mut node = self;
        for c in text.chars() {
            node = node.children.get(&c)?;
        }
        Some(node)
    }
}

impl Grammar {
    /// Parses GBNF-lite `source` and expands `start_rule` into its full set
    /// of candidate strings.
    pub fn parse(source: &str, start_rule: &str) -> Result<Grammar, String> {
        let rules = parse_rules(source)?;
        if !rules.contains_key(start_rule) {
            return Err(format!("unknown start rule {start_rule:?}"));
        }
        let mut candidates = Vec::new();
        let mut stack = vec![start_rule.to_string()];
        expand(
            &rules,
            start_rule,
            String::new(),
            &mut candidates,
            &mut stack,
        )?;
        if candidates.is_empty() {
            return Err(format!("rule {start_rule:?} produces no strings"));
        }
        candidates.sort();
        candidates.dedup();
        let mut trie = TrieNode::default();
        for c in &candidates {
            trie.insert(c);
        }
        Ok(Grammar { candidates, trie })
    }

    /// All complete strings this grammar's start rule can produce.
    pub fn candidates(&self) -> &[String] {
        &self.candidates
    }

    /// Whether `text` is a prefix of (or equal to) at least one candidate —
    /// i.e. decoding could still legally continue from here.
    pub fn is_consistent_prefix(&self, text: &str) -> bool {
        self.trie.walk(text).is_some()
    }

    /// Whether `text` exactly completes one of the candidates.
    pub fn is_complete(&self, text: &str) -> bool {
        self.trie.walk(text).is_some_and(|n| n.is_end)
    }
}

fn parse_rules(source: &str) -> Result<HashMap<String, Vec<Alternative>>, String> {
    let mut rules = HashMap::new();
    for (lineno, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, expr) = line
            .split_once("::=")
            .ok_or_else(|| format!("line {}: expected \"name ::= ...\"", lineno + 1))?;
        let name = name.trim().to_string();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!("line {}: invalid rule name {name:?}", lineno + 1));
        }
        let alternatives =
            parse_alternatives(expr.trim()).map_err(|e| format!("line {}: {e}", lineno + 1))?;
        rules.insert(name, alternatives);
    }
    Ok(rules)
}

fn parse_alternatives(expr: &str) -> Result<Vec<Alternative>, String> {
    // Split on top-level `|` (not inside quotes or parens).
    let mut alts = Vec::new();
    let mut depth = 0i32;
    let mut in_quote = false;
    let mut start = 0usize;
    let bytes = expr.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_quote => in_quote = true,
            b'"' if in_quote => in_quote = false,
            b'(' if !in_quote => depth += 1,
            b')' if !in_quote => depth -= 1,
            b'|' if !in_quote && depth == 0 => {
                alts.push(parse_sequence(expr[start..i].trim())?);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    alts.push(parse_sequence(expr[start..].trim())?);
    Ok(alts)
}

fn parse_sequence(expr: &str) -> Result<Alternative, String> {
    let mut terms = Vec::new();
    let bytes = expr.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' => i += 1,
            b'"' => {
                let end = expr[i + 1..]
                    .find('"')
                    .ok_or_else(|| "unterminated string literal".to_string())?
                    + i
                    + 1;
                terms.push(Term::Literal(expr[i + 1..end].to_string()));
                i = end + 1;
            }
            b'(' => {
                let mut depth = 1i32;
                let mut j = i + 1;
                while j < bytes.len() && depth > 0 {
                    match bytes[j] {
                        b'(' => depth += 1,
                        b')' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    return Err("unbalanced parentheses".to_string());
                }
                let inner = &expr[i + 1..j - 1];
                if j < bytes.len() && matches!(bytes[j], b'*' | b'+' | b'?') {
                    return Err(
                        "repetition operators (*, +, ?) are not supported by this grammar subset"
                            .to_string(),
                    );
                }
                // A parenthesized group with top-level alternation isn't
                // representable as a single flattened Term in this subset;
                // reject rather than silently mis-expand it.
                if inner.contains('|') {
                    return Err(
                        "alternation inside parentheses is not supported — write it as a named rule instead"
                            .to_string(),
                    );
                }
                terms.extend(parse_sequence(inner.trim())?);
                i = j;
            }
            b'[' => {
                return Err(
                    "character classes ([...]) are not supported by this grammar subset"
                        .to_string(),
                );
            }
            _ => {
                let end = expr[i..]
                    .find(|c: char| c.is_whitespace() || c == '(' || c == '"')
                    .map(|p| p + i)
                    .unwrap_or(expr.len());
                let word = &expr[i..end];
                if word.is_empty() {
                    return Err(format!("unexpected character {:?}", bytes[i] as char));
                }
                if let Some(stripped) = word.strip_suffix(['*', '+', '?']) {
                    let _ = stripped;
                    return Err(
                        "repetition operators (*, +, ?) are not supported by this grammar subset"
                            .to_string(),
                    );
                }
                terms.push(Term::Ref(word.to_string()));
                i = end;
            }
        }
    }
    Ok(terms)
}

fn expand(
    rules: &HashMap<String, Vec<Alternative>>,
    rule: &str,
    prefix: String,
    out: &mut Vec<String>,
    stack: &mut Vec<String>,
) -> Result<(), String> {
    let alternatives = rules
        .get(rule)
        .ok_or_else(|| format!("undefined rule {rule:?}"))?;
    for alt in alternatives {
        expand_sequence(rules, alt, 0, prefix.clone(), out, stack)?;
        if out.len() > MAX_CANDIDATES {
            return Err(format!(
                "grammar produces more than {MAX_CANDIDATES} candidate strings"
            ));
        }
    }
    Ok(())
}

fn expand_sequence(
    rules: &HashMap<String, Vec<Alternative>>,
    seq: &[Term],
    pos: usize,
    prefix: String,
    out: &mut Vec<String>,
    stack: &mut Vec<String>,
) -> Result<(), String> {
    if pos == seq.len() {
        out.push(prefix);
        return Ok(());
    }
    match &seq[pos] {
        Term::Literal(s) => expand_sequence(rules, seq, pos + 1, prefix + s, out, stack),
        Term::Ref(name) => {
            if stack.contains(name) {
                return Err(format!("grammar rule cycle through {name:?}"));
            }
            stack.push(name.clone());
            let alternatives = rules
                .get(name)
                .ok_or_else(|| format!("undefined rule {name:?}"))?;
            for alt in alternatives {
                // Expand this reference's alternative fully, then continue
                // the outer sequence from each resulting completion.
                let mut heads = Vec::new();
                expand_sequence(rules, alt, 0, prefix.clone(), &mut heads, stack)?;
                for head in heads {
                    expand_sequence(rules, seq, pos + 1, head, out, stack)?;
                    if out.len() > MAX_CANDIDATES {
                        stack.pop();
                        return Err(format!(
                            "grammar produces more than {MAX_CANDIDATES} candidate strings"
                        ));
                    }
                }
            }
            stack.pop();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_alternation() {
        let g = Grammar::parse(r#"root ::= "yes" | "no""#, "root").unwrap();
        let mut c = g.candidates().to_vec();
        c.sort();
        assert_eq!(c, vec!["no".to_string(), "yes".to_string()]);
    }

    #[test]
    fn concatenation_and_references() {
        let src = r#"
            root ::= greeting " " name
            greeting ::= "hi" | "hello"
            name ::= "sam" | "kai"
        "#;
        let g = Grammar::parse(src, "root").unwrap();
        let mut c = g.candidates().to_vec();
        c.sort();
        assert_eq!(
            c,
            vec![
                "hello kai".to_string(),
                "hello sam".to_string(),
                "hi kai".to_string(),
                "hi sam".to_string(),
            ]
        );
    }

    #[test]
    fn parenthesized_grouping() {
        let g = Grammar::parse(r#"root ::= "turn " ("on" ) " lights""#, "root").unwrap();
        assert_eq!(g.candidates(), &["turn on lights".to_string()]);
    }

    #[test]
    fn rejects_char_classes() {
        assert!(Grammar::parse(r#"root ::= [a-z]"#, "root").is_err());
    }

    #[test]
    fn rejects_repetition_operators() {
        assert!(Grammar::parse(r#"root ::= "a"*"#, "root").is_err());
        assert!(Grammar::parse(
            r#"root ::= word+
word ::= "a""#,
            "root"
        )
        .is_err());
    }

    #[test]
    fn rejects_cycles() {
        let src = "root ::= a\na ::= root";
        assert!(Grammar::parse(src, "root").is_err());
    }

    #[test]
    fn rejects_unknown_start_rule() {
        assert!(Grammar::parse(r#"root ::= "x""#, "missing").is_err());
    }

    #[test]
    fn consistency_checks() {
        let g = Grammar::parse(r#"root ::= "turn on" | "turn off""#, "root").unwrap();
        assert!(g.is_consistent_prefix(""));
        assert!(g.is_consistent_prefix("turn"));
        assert!(g.is_consistent_prefix("turn on"));
        assert!(!g.is_consistent_prefix("turn sideways"));
        assert!(g.is_complete("turn on"));
        assert!(!g.is_complete("turn"));
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let src = "# a comment\n\nroot ::= \"ok\"\n";
        let g = Grammar::parse(src, "root").unwrap();
        assert_eq!(g.candidates(), &["ok".to_string()]);
    }
}
