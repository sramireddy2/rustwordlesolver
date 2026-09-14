//! Loading and parsing the dictionaries.
//!
//! Two lists ship in `data/`: the answers (what the puzzle can be) and the
//! extra allowed guesses (what you may type but which will never be the
//! answer). They are kept separate on purpose: the best splitting guess is
//! frequently not a possible answer.

use crate::types::{WORD_LEN, Word};

pub const BUNDLED_ANSWERS: &str = include_str!("../data/answers.txt");
pub const BUNDLED_GUESSES: &str = include_str!("../data/guesses.txt");

pub fn parse_word(s: &str) -> Result<Word, String> {
    let s = s.trim();
    if s.len() != WORD_LEN {
        return Err(format!("{s:?} is not {WORD_LEN} letters"));
    }
    let mut w = [0u8; WORD_LEN];
    for (dst, b) in w.iter_mut().zip(s.bytes()) {
        let b = b.to_ascii_lowercase();
        if !b.is_ascii_lowercase() {
            return Err(format!("{s:?} contains a non-letter"));
        }
        *dst = b;
    }
    Ok(w)
}

pub fn word_str(w: &Word) -> &str {
    // Words are only ever built by `parse_word`, which guarantees ASCII.
    std::str::from_utf8(w).unwrap()
}

/// Parse a newline-separated list. Blank lines and `#` comments are skipped;
/// anything else that isn't a valid word is an error, since a silently
/// dropped answer would make the solver wrong rather than merely slower.
pub fn parse_list(text: &str) -> Result<Vec<Word>, String> {
    text.lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'))
        .map(|(n, l)| parse_word(l).map_err(|e| format!("line {n}: {e}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_word_normalises_and_validates() {
        assert_eq!(parse_word(" CrAnE\n").unwrap(), *b"crane");
        assert!(parse_word("cran").is_err());
        assert!(parse_word("cr4ne").is_err());
        assert!(parse_word("cranes").is_err());
    }

    #[test]
    fn parse_list_skips_blanks_and_comments_but_not_junk() {
        let ok = parse_list("crane\n\n# comment\n  shale \n").unwrap();
        assert_eq!(ok, vec![*b"crane", *b"shale"]);
        let err = parse_list("crane\ntoolong\n").unwrap_err();
        assert!(err.starts_with("line 2:"), "{err}");
    }

    #[test]
    fn bundled_lists_are_sane() {
        let answers = parse_list(BUNDLED_ANSWERS).unwrap();
        let guesses = parse_list(BUNDLED_GUESSES).unwrap();
        assert_eq!(answers.len(), 2315);
        assert_eq!(guesses.len(), 10657);
        assert!(answers.contains(b"crane"));
        assert!(guesses.contains(b"soare"));
    }
}
