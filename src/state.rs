//! Which answers are still possible, and the running state of one game.

use crate::table::Context;
use crate::types::{Pattern, WordId};

/// `u64` words in the bitset. 37 × 64 = 2,368 bits covers the 2,315 official
/// answers with a little slack. Raise it if you load a bigger answer list;
/// `Context::new` checks and refuses rather than silently truncating.
pub const BITSET_WORDS: usize = 37;
pub const MAX_ANSWERS: usize = BITSET_WORDS * 64;

/// A fixed bitset over the answer list: bit `i` set means `WordId(i)` is
/// still consistent with everything observed. 296 bytes, `Copy`, no heap —
/// cloning one for a lookahead branch is a memcpy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CandidateSet {
    bits: [u64; BITSET_WORDS],
    count: usize,
}

impl CandidateSet {
    /// Every answer `0..n` is a candidate.
    pub fn all(n: usize) -> CandidateSet {
        assert!(
            n <= MAX_ANSWERS,
            "{n} answers exceed MAX_ANSWERS = {MAX_ANSWERS}"
        );
        let mut bits = [0u64; BITSET_WORDS];
        let (full, rem) = (n / 64, n % 64);
        for b in &mut bits[..full] {
            *b = u64::MAX;
        }
        if rem > 0 {
            bits[full] = (1u64 << rem) - 1;
        }
        CandidateSet { bits, count: n }
    }

    pub fn empty() -> CandidateSet {
        CandidateSet {
            bits: [0; BITSET_WORDS],
            count: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Strategies ask this about *every* allowed guess, most of which are
    /// not answers and so lie past the end of the bitset. Those are simply
    /// not candidates; they must not index out of bounds.
    #[inline]
    pub fn contains(&self, id: WordId) -> bool {
        let i = id.index();
        i < MAX_ANSWERS && self.bits[i / 64] >> (i % 64) & 1 == 1
    }

    pub fn insert(&mut self, id: WordId) {
        let i = id.index();
        let mask = 1u64 << (i % 64);
        if self.bits[i / 64] & mask == 0 {
            self.bits[i / 64] |= mask;
            self.count += 1;
        }
    }

    pub fn remove(&mut self, id: WordId) {
        let i = id.index();
        let mask = 1u64 << (i % 64);
        if self.bits[i / 64] & mask != 0 {
            self.bits[i / 64] &= !mask;
            self.count -= 1;
        }
    }

    /// Keep only the candidates that would have produced `observed` for
    /// `guess`. That single comparison *is* the whole constraint system —
    /// there is no separate "letter must be at position 2" logic to get wrong.
    pub fn filter(&mut self, guess: WordId, observed: Pattern, ctx: &Context) {
        let mut count = 0;
        for (w, chunk) in self.bits.iter_mut().enumerate() {
            let mut remaining = *chunk;
            let mut keep = 0u64;
            while remaining != 0 {
                let tz = remaining.trailing_zeros();
                remaining &= remaining - 1;
                let id = WordId((w * 64 + tz as usize) as u16);
                if ctx.get_pattern(guess, id) == observed {
                    keep |= 1u64 << tz;
                }
            }
            count += keep.count_ones() as usize;
            *chunk = keep;
        }
        self.count = count;
    }

    /// Set bits in ascending order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            bits: &self.bits,
            word: 0,
            current: self.bits[0],
        }
    }

    pub fn first(&self) -> Option<WordId> {
        self.iter().next()
    }
}

pub struct Iter<'a> {
    bits: &'a [u64; BITSET_WORDS],
    word: usize,
    current: u64,
}

impl Iterator for Iter<'_> {
    type Item = WordId;

    #[inline]
    fn next(&mut self) -> Option<WordId> {
        while self.current == 0 {
            self.word += 1;
            if self.word >= BITSET_WORDS {
                return None;
            }
            self.current = self.bits[self.word];
        }
        let tz = self.current.trailing_zeros();
        self.current &= self.current - 1;
        Some(WordId((self.word * 64 + tz as usize) as u16))
    }
}

impl<'a> IntoIterator for &'a CandidateSet {
    type Item = WordId;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// One game in progress: the surviving candidates plus what was played.
pub struct Game<'a> {
    ctx: &'a Context,
    candidates: CandidateSet,
    history: Vec<(WordId, Pattern)>,
}

impl<'a> Game<'a> {
    pub fn new(ctx: &'a Context) -> Game<'a> {
        Game {
            ctx,
            candidates: CandidateSet::all(ctx.num_answers()),
            history: Vec::new(),
        }
    }

    pub fn candidates(&self) -> &CandidateSet {
        &self.candidates
    }

    pub fn history(&self) -> &[(WordId, Pattern)] {
        &self.history
    }

    /// 1-based number of the guess about to be played.
    pub fn turn(&self) -> usize {
        self.history.len() + 1
    }

    pub fn is_solved(&self) -> bool {
        matches!(self.history.last(), Some((_, p)) if *p == Pattern::WIN)
    }

    pub fn observe(&mut self, guess: WordId, observed: Pattern) {
        self.candidates.filter(guess, observed, self.ctx);
        self.history.push((guess, observed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;

    fn ctx() -> Context {
        let w = |l: &[&str]| l.iter().map(|s| parse_word(s).unwrap()).collect();
        Context::new(
            w(&["crane", "shale", "stone", "adobe", "abbey"]),
            w(&["soare", "puppy"]),
        )
        .unwrap()
    }

    #[test]
    fn all_sets_exactly_n_bits_across_word_boundaries() {
        for n in [0, 1, 63, 64, 65, 130, MAX_ANSWERS] {
            let s = CandidateSet::all(n);
            assert_eq!(s.len(), n);
            assert_eq!(s.iter().count(), n);
            assert_eq!(s.iter().map(|id| id.index()).last(), n.checked_sub(1));
            if n < MAX_ANSWERS {
                assert!(!s.contains(WordId(n as u16)));
            }
        }
    }

    #[test]
    fn contains_is_false_past_the_answer_range() {
        let s = CandidateSet::all(MAX_ANSWERS);
        assert!(s.contains(WordId(MAX_ANSWERS as u16 - 1)));
        assert!(!s.contains(WordId(MAX_ANSWERS as u16)));
        assert!(!s.contains(WordId(u16::MAX)));
    }

    #[test]
    fn insert_remove_contains() {
        let mut s = CandidateSet::empty();
        s.insert(WordId(70));
        s.insert(WordId(70));
        s.insert(WordId(3));
        assert_eq!(s.len(), 2);
        assert!(s.contains(WordId(70)));
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![WordId(3), WordId(70)]);
        s.remove(WordId(3));
        s.remove(WordId(3));
        assert_eq!(s.len(), 1);
        assert_eq!(s.first(), Some(WordId(70)));
    }

    #[test]
    fn filter_keeps_only_consistent_answers() {
        let ctx = ctx();
        let mut s = CandidateSet::all(ctx.num_answers());
        let crane = ctx.find(b"crane").unwrap();
        // Pretend the answer is "shale": crane -> bbgbg.
        s.filter(crane, Pattern::parse("bbgbg").unwrap(), &ctx);
        let left: Vec<_> = s.iter().map(|id| ctx.word_str(id)).collect();
        assert_eq!(left, vec!["shale"]);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn the_answer_always_survives() {
        let ctx = ctx();
        for &answer in &ctx.answers {
            let mut game = Game::new(&ctx);
            for g in 0..ctx.num_guesses() as u16 {
                let g = WordId(g);
                game.observe(g, ctx.get_pattern(g, answer));
                assert!(
                    game.candidates().contains(answer),
                    "{} eliminated",
                    ctx.word_str(answer)
                );
            }
            assert_eq!(game.candidates().len(), 1);
            assert!(
                game.history()
                    .iter()
                    .any(|&(g, p)| g == answer && p == Pattern::WIN)
            );
            assert_eq!(game.turn(), ctx.num_guesses() + 1);
        }
    }

    #[test]
    fn solved_means_the_last_guess_was_all_green() {
        let ctx = ctx();
        let shale = ctx.find(b"shale").unwrap();
        let crane = ctx.find(b"crane").unwrap();
        let mut game = Game::new(&ctx);
        assert!(!game.is_solved());
        game.observe(crane, ctx.get_pattern(crane, shale));
        assert!(!game.is_solved());
        assert_eq!(game.turn(), 2);
        game.observe(shale, Pattern::WIN);
        assert!(game.is_solved());
    }
}
