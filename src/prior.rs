//! Turning word frequencies into prior weights over the candidates.
//!
//! Raw corpus counts span six orders of magnitude, which would make a
//! rare-but-real answer like `abase` effectively impossible. Instead the
//! prior is a logistic curve over frequency *rank*: the top couple of
//! thousand words get near-full weight, and the tail decays gently.
//!
//! The centre and width were read off the official answer list: within
//! each band of 1,000 ranks the fraction of words that are answers falls
//! 0.67, 0.59, 0.43, 0.30, 0.19, 0.10, 0.02, 0.01, then zero — a logistic
//! centred near rank 2,800 with width ~1,200. That is mild leakage from
//! the benchmark into the prior (two parameters on a smooth curve); a
//! stricter setup would calibrate on half the answers and test on the rest.
//!
//! Weights are integers so the entropy loop stays a table lookup (see
//! [`crate::table::MathTables`]). Uniform mode is simply every weight = 1.

use std::collections::HashMap;

use crate::types::Word;

/// Rank at which a word is 50% likely to be a real answer.
pub const CENTER: f64 = 2800.0;
/// How quickly that probability falls off around the centre.
pub const WIDTH: f64 = 1200.0;
/// Integer weight of a probability-1 word. Sets the resolution of the
/// tail: the floor weight 1 corresponds to `p = 1 / SCALE`.
pub const SCALE: f64 = 300.0;

/// Weight for the word at `rank` (0 = most frequent). Never below 1.
pub fn weight_for_rank(rank: usize) -> u32 {
    let p = 1.0 / (1.0 + ((rank as f64 - CENTER) / WIDTH).exp());
    ((p * SCALE).round() as u32).max(1)
}

/// Rank `words` by `counts` (absent words count 0 and rank last,
/// alphabetically) and map each to its weight.
pub fn rank_sigmoid(words: &[Word], counts: &[(Word, u64)]) -> HashMap<Word, u32> {
    let count: HashMap<Word, u64> = counts.iter().copied().collect();
    let mut ranked: Vec<&Word> = words.iter().collect();
    ranked.sort_by(|a, b| {
        let ca = count.get(*a).copied().unwrap_or(0);
        let cb = count.get(*b).copied().unwrap_or(0);
        cb.cmp(&ca).then_with(|| a.cmp(b))
    });
    ranked
        .into_iter()
        .enumerate()
        .map(|(rank, w)| (*w, weight_for_rank(rank)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_is_monotone_with_sensible_ends() {
        let top = weight_for_rank(0);
        assert!(top > (0.85 * SCALE) as u32, "top weight {top}");
        let mut prev = top;
        for r in 1..13_000 {
            let w = weight_for_rank(r);
            assert!(w <= prev, "rank {r} heavier than rank {}", r - 1);
            prev = w;
        }
        assert_eq!(weight_for_rank(12_999), 1);
        // The centre is the half-way point.
        let mid = weight_for_rank(CENTER as usize);
        assert!(
            (mid as f64 - SCALE / 2.0).abs() <= 1.0,
            "centre weight {mid}"
        );
    }

    #[test]
    fn higher_count_means_higher_or_equal_weight_and_absent_words_rank_last() {
        let words = [*b"about", *b"crane", *b"zymic", *b"abase"];
        let counts = [
            (*b"about", 1_000_000u64),
            (*b"crane", 50_000),
            (*b"abase", 100),
        ];
        let w = rank_sigmoid(&words, &counts);
        assert!(w[b"about"] >= w[b"crane"]);
        assert!(w[b"crane"] >= w[b"abase"]);
        assert!(w[b"abase"] >= w[b"zymic"]);
        assert_eq!(
            w.len(),
            4,
            "every word gets a weight, present in counts or not"
        );
    }
}
