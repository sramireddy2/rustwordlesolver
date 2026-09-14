//! Entropy first, minimax to break ties, and a preference for guesses that
//! could themselves be the answer when everything else is equal.
//!
//! That last slot matters more than it looks: two guesses with identical
//! histograms teach you the same amount, but only the one that is a
//! candidate can end the game this turn.

use std::cmp::Reverse;

use super::{Strategy, candidate_list, entropy_and_worst, histogram, pick_best};
use crate::state::CandidateSet;
use crate::table::Context;
use crate::types::WordId;

pub struct Hybrid;

impl Strategy for Hybrid {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet) -> WordId {
        let cands = candidate_list(candidates);
        let n = cands.len();
        pick_best(ctx, |g| {
            let (bits, worst) = entropy_and_worst(&histogram(ctx, g, &cands), n, &ctx.math);
            (bits, Reverse(worst), candidates.contains(g))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;

    #[test]
    fn candidate_wins_an_exact_tie() {
        let w = |l: &[&str]| l.iter().map(|s| parse_word(s).unwrap()).collect();
        // Put the non-answer "soare" at a LOWER id than the answer "crane" by
        // listing it as an answer too, then remove it from the candidates.
        // Both split the rest perfectly; only "crane" can win outright.
        let ctx = Context::new(
            w(&["soare", "crane", "shale", "stone", "adobe"]),
            w(&["puppy"]),
        )
        .unwrap();
        let mut cands = CandidateSet::all(5);
        cands.remove(ctx.find(b"soare").unwrap());
        let best = Hybrid.best_guess(&ctx, &cands);
        assert_eq!(ctx.word_str(best), "crane");
        // Entropy alone would have taken the lower id.
        let best = super::super::entropy::Entropy.best_guess(&ctx, &cands);
        assert_eq!(ctx.word_str(best), "soare");
    }
}
