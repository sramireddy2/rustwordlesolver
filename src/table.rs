//! The precomputed pattern table, and the [`Context`] that owns it.

use rayon::prelude::*;

use crate::state::MAX_ANSWERS;
use crate::types::{Pattern, Word, WordId};
use crate::words::{self, parse_list, word_str};

/// Everything immutable the solver needs: the dictionaries, the pattern
/// matrix, and the maths lookup tables. Built once, shared by reference.
pub struct Context {
    /// Every word you may guess. **Answers come first**, so
    /// `allowed_guesses[..answers.len()]` is the answer list and
    /// `id.0 < answers.len()` means the word could be the answer.
    pub allowed_guesses: Vec<Word>,
    /// `answers[i] == WordId(i)`; kept as a list for convenient iteration.
    pub answers: Vec<WordId>,
    /// Flattened `guesses × answers`, row-major:
    /// `table[guess * answers.len() + answer]`. One byte per cell, ~30 MB for
    /// the official lists, and the reason scoring is a load and not a call.
    pub table: Vec<Pattern>,
    pub math: MathTables,
}

impl Context {
    /// `answers` become ids `0..answers.len()`; any `extra_guesses` that are
    /// not already answers follow. Duplicates are dropped.
    pub fn new(answers: Vec<Word>, extra_guesses: Vec<Word>) -> Result<Context, String> {
        let mut allowed_guesses: Vec<Word> =
            Vec::with_capacity(answers.len() + extra_guesses.len());
        let mut seen = std::collections::HashSet::with_capacity(allowed_guesses.capacity());
        for w in answers {
            if seen.insert(w) {
                allowed_guesses.push(w);
            }
        }
        let num_answers = allowed_guesses.len();
        for w in extra_guesses {
            if seen.insert(w) {
                allowed_guesses.push(w);
            }
        }

        if num_answers == 0 {
            return Err("no answers".into());
        }
        if num_answers > MAX_ANSWERS {
            return Err(format!(
                "{num_answers} answers, but CandidateSet holds at most {MAX_ANSWERS}; \
                 raise state::BITSET_WORDS"
            ));
        }
        if allowed_guesses.len() > u16::MAX as usize {
            return Err(format!(
                "{} guesses do not fit a u16 WordId",
                allowed_guesses.len()
            ));
        }

        let table = build_table(&allowed_guesses, num_answers);
        let math = MathTables::new(num_answers);
        let answers = (0..num_answers as u16).map(WordId).collect();

        Ok(Context {
            allowed_guesses,
            answers,
            table,
            math,
        })
    }

    pub fn from_lists(answers: &str, extra_guesses: &str) -> Result<Context, String> {
        Context::new(parse_list(answers)?, parse_list(extra_guesses)?)
    }

    /// The official lists compiled into the binary.
    pub fn bundled() -> Context {
        Context::from_lists(words::BUNDLED_ANSWERS, words::BUNDLED_GUESSES)
            .expect("bundled word lists are valid")
    }

    #[inline]
    pub fn get_pattern(&self, guess: WordId, answer: WordId) -> Pattern {
        debug_assert!(answer.index() < self.answers.len());
        self.table[guess.index() * self.answers.len() + answer.index()]
    }

    #[inline]
    pub fn num_answers(&self) -> usize {
        self.answers.len()
    }

    #[inline]
    pub fn num_guesses(&self) -> usize {
        self.allowed_guesses.len()
    }

    #[inline]
    pub fn is_answer(&self, id: WordId) -> bool {
        id.index() < self.answers.len()
    }

    pub fn word(&self, id: WordId) -> &Word {
        &self.allowed_guesses[id.index()]
    }

    pub fn word_str(&self, id: WordId) -> &str {
        word_str(self.word(id))
    }

    /// Linear scan; fine for a REPL, don't put it in a hot loop.
    pub fn find(&self, word: &Word) -> Option<WordId> {
        self.allowed_guesses
            .iter()
            .position(|w| w == word)
            .map(|i| WordId(i as u16))
    }
}

/// One row per guess, computed in parallel. ~30M `Pattern::score` calls for
/// the official lists; tens of milliseconds in release.
fn build_table(guesses: &[Word], num_answers: usize) -> Vec<Pattern> {
    let answers = &guesses[..num_answers];
    let mut table = vec![Pattern(0); guesses.len() * num_answers];
    table
        .par_chunks_mut(num_answers)
        .zip(guesses.par_iter())
        .for_each(|(row, guess)| {
            for (cell, answer) in row.iter_mut().zip(answers) {
                *cell = Pattern::score(guess, answer);
            }
        });
    table
}

/// Lookup tables so the entropy inner loop never calls `log2`.
pub struct MathTables {
    /// `x_log2_x[x] = x · log2(x)`, with `0 · log2(0) = 0`. Indexed by bucket
    /// size, so it needs `num_answers + 1` entries.
    x_log2_x: Vec<f64>,
}

impl MathTables {
    pub fn new(max_count: usize) -> MathTables {
        let x_log2_x = (0..=max_count)
            .map(|x| {
                if x == 0 {
                    0.0
                } else {
                    (x as f64) * (x as f64).log2()
                }
            })
            .collect();
        MathTables { x_log2_x }
    }

    #[inline]
    pub fn x_log2_x(&self, x: usize) -> f64 {
        self.x_log2_x[x]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;

    fn words(list: &[&str]) -> Vec<Word> {
        list.iter().map(|w| parse_word(w).unwrap()).collect()
    }

    #[test]
    fn answers_come_first_and_duplicates_collapse() {
        let ctx = Context::new(
            words(&["crane", "shale"]),
            words(&["shale", "soare", "crane", "soare"]),
        )
        .unwrap();
        assert_eq!(ctx.num_answers(), 2);
        assert_eq!(ctx.num_guesses(), 3);
        assert_eq!(ctx.word_str(WordId(2)), "soare");
        assert!(ctx.is_answer(WordId(1)));
        assert!(!ctx.is_answer(WordId(2)));
        assert_eq!(ctx.find(b"soare"), Some(WordId(2)));
        assert_eq!(ctx.find(b"zzzzz"), None);
    }

    #[test]
    fn table_agrees_with_direct_scoring() {
        let ctx = Context::new(
            words(&["crane", "shale", "abbey", "lucky"]),
            words(&["bobby", "llama", "speed"]),
        )
        .unwrap();
        for g in 0..ctx.num_guesses() as u16 {
            for &a in &ctx.answers {
                let g = WordId(g);
                assert_eq!(
                    ctx.get_pattern(g, a),
                    Pattern::score(ctx.word(g), ctx.word(a))
                );
            }
        }
        for &a in &ctx.answers {
            assert_eq!(ctx.get_pattern(a, a), Pattern::WIN);
        }
    }

    #[test]
    fn rejects_empty_and_oversized() {
        assert!(Context::new(vec![], vec![]).is_err());
        // MAX_ANSWERS + 1 distinct words: base-26 encode the index.
        let too_many: Vec<Word> = (0..=MAX_ANSWERS)
            .map(|i| {
                let mut w = [b'a'; 5];
                let mut n = i;
                for c in w.iter_mut() {
                    *c = b'a' + (n % 26) as u8;
                    n /= 26;
                }
                w
            })
            .collect();
        assert!(Context::new(too_many, vec![]).is_err());
    }

    #[test]
    fn math_tables() {
        let m = MathTables::new(8);
        assert_eq!(m.x_log2_x(0), 0.0);
        assert_eq!(m.x_log2_x(1), 0.0);
        assert_eq!(m.x_log2_x(2), 2.0);
        assert_eq!(m.x_log2_x(8), 24.0);
    }
}
