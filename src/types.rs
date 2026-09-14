//! Zero-cost newtypes. A word index and a feedback pattern are both small
//! integers; wrapping them means the compiler stops you passing one as the
//! other.

use std::fmt;

pub const WORD_LEN: usize = 5;
/// 3 colours per tile, 5 tiles.
pub const NUM_PATTERNS: usize = 243;

/// Five ASCII lowercase letters.
pub type Word = [u8; WORD_LEN];

/// Index into [`Context::allowed_guesses`](crate::table::Context::allowed_guesses).
/// Answers occupy the low indices, so `id.0 < num_answers` means "could be
/// the answer".
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct WordId(pub u16);

impl WordId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

pub const BLACK: u8 = 0;
pub const YELLOW: u8 = 1;
pub const GREEN: u8 = 2;

/// The row of tiles for one guess, packed base-3: tile `i` contributes
/// `colour * 3^i`, with black = 0, yellow = 1, green = 2.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Pattern(pub u8);

impl Pattern {
    /// All green: 2 + 2·3 + 2·9 + 2·27 + 2·81.
    pub const WIN: Pattern = Pattern(242);

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_colors(colors: [u8; WORD_LEN]) -> Pattern {
        let mut packed = 0u8;
        let mut place = 1u8;
        for c in colors {
            debug_assert!(c <= GREEN);
            packed += c * place;
            place = place.wrapping_mul(3); // overflows only after the last tile
        }
        Pattern(packed)
    }

    pub fn to_colors(self) -> [u8; WORD_LEN] {
        let mut colors = [BLACK; WORD_LEN];
        let mut p = self.0;
        for c in &mut colors {
            *c = p % 3;
            p /= 3;
        }
        colors
    }

    /// The tiles Wordle would show for `guess` if the answer were `answer`.
    ///
    /// Two passes, because letters are consumed. Greens are claimed first;
    /// then each remaining guess letter goes yellow only while unclaimed
    /// copies of it are left in the answer. This is what makes `llama` vs
    /// `lucky` score a single yellow, not two.
    pub fn score(guess: &Word, answer: &Word) -> Pattern {
        let mut colors = [BLACK; WORD_LEN];
        let mut unclaimed = [0u8; 26];

        for i in 0..WORD_LEN {
            if guess[i] == answer[i] {
                colors[i] = GREEN;
            } else {
                unclaimed[(answer[i] - b'a') as usize] += 1;
            }
        }
        for i in 0..WORD_LEN {
            if colors[i] == GREEN {
                continue;
            }
            let slot = &mut unclaimed[(guess[i] - b'a') as usize];
            if *slot > 0 {
                *slot -= 1;
                colors[i] = YELLOW;
            }
        }
        Pattern::from_colors(colors)
    }

    /// Parse what a human types: `g` green, `y` yellow, `b`/`x`/`.`/`-` black.
    pub fn parse(s: &str) -> Result<Pattern, String> {
        let s = s.trim();
        if s.len() != WORD_LEN {
            return Err(format!("pattern must be {WORD_LEN} characters, got {:?}", s));
        }
        let mut colors = [BLACK; WORD_LEN];
        for (c, ch) in colors.iter_mut().zip(s.bytes()) {
            *c = match ch.to_ascii_lowercase() {
                b'g' => GREEN,
                b'y' => YELLOW,
                b'b' | b'x' | b'.' | b'-' => BLACK,
                other => return Err(format!("unexpected {:?} in pattern", other as char)),
            };
        }
        Ok(Pattern::from_colors(colors))
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in self.to_colors() {
            f.write_str(match c {
                GREEN => "g",
                YELLOW => "y",
                _ => "b",
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;

    fn score(g: &str, a: &str) -> String {
        Pattern::score(&parse_word(g).unwrap(), &parse_word(a).unwrap()).to_string()
    }

    #[test]
    fn win_constant_is_all_green() {
        assert_eq!(Pattern::from_colors([GREEN; 5]), Pattern::WIN);
        assert_eq!(Pattern::WIN.to_string(), "ggggg");
    }

    #[test]
    fn every_pattern_round_trips() {
        for p in 0..NUM_PATTERNS as u8 {
            let pat = Pattern(p);
            assert_eq!(Pattern::from_colors(pat.to_colors()), pat);
            assert_eq!(Pattern::parse(&pat.to_string()).unwrap(), pat);
        }
    }

    #[test]
    fn parse_accepts_aliases_and_rejects_junk() {
        assert_eq!(Pattern::parse("GYBX.").unwrap().to_string(), "gybbb");
        assert!(Pattern::parse("gybb").is_err());
        assert!(Pattern::parse("gybbz").is_err());
    }

    #[test]
    fn basic_scoring() {
        assert_eq!(score("crane", "crane"), "ggggg");
        assert_eq!(score("crane", "moldy"), "bbbbb");
        assert_eq!(score("crane", "shale"), "bbgbg");
        assert_eq!(score("crane", "adobe"), "bbybg");
    }

    #[test]
    fn duplicate_letters_are_consumed() {
        // One 'l' in the answer: the first guess 'l' takes it, the second is black.
        assert_eq!(score("llama", "lucky"), "gbbbb");
        // Green claims before yellow: the b at index 2 is green, so only one
        // of the other two b's can be yellow.
        assert_eq!(score("bobby", "abbey"), "ybgbg");
        // Two e's in the guess, one in the answer: only the first goes yellow.
        assert_eq!(score("speed", "abide"), "bbyby");
        // Two e's in the guess, two in the answer, neither in place: both yellow.
        assert_eq!(score("speed", "erase"), "ybyyb");
    }
}
