//! Minimax: play the guess whose *worst* tile pattern leaves the fewest
//! candidates.
//!
//! Deliberately pure. Hundreds of guesses usually share the same worst-case
//! bucket, so this breaks ties only toward a word that could be the answer
//! and otherwise takes the lowest id. That makes it a weak solver and an
//! honest baseline: the gap between it and
//! [`Hybrid`](super::hybrid::Hybrid) is what entropy buys you.

use std::cmp::Reverse;

use super::{Strategy, candidate_list, histogram, pick_best, worst_case};
use crate::state::CandidateSet;
use crate::table::Context;
use crate::types::WordId;

pub struct Minimax;

impl Strategy for Minimax {
    fn name(&self) -> &'static str {
        "minimax"
    }

    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet, _turns_left: usize) -> WordId {
        let cands = candidate_list(candidates);
        // Tuples compare left to right: smaller worst case first (Reverse
        // makes "smaller" win a max), then `true` beats `false` so a possible
        // answer wins the tie.
        pick_best(ctx, |g| {
            (
                Reverse(worst_case(&histogram(ctx, g, &cands))),
                candidates.contains(g),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;

    #[test]
    fn minimises_the_largest_bucket() {
        let w = |l: &[&str]| l.iter().map(|s| parse_word(s).unwrap()).collect();
        // Answers share a lot of letters; "clamp" separates every one of
        // them while each answer scores its twin identically to itself.
        let ctx = Context::new(w(&["match", "patch", "latch", "hatch"]), w(&["clamp"])).unwrap();
        let best = Minimax.best_guess(&ctx, &CandidateSet::all(4), 6);
        assert_eq!(ctx.word_str(best), "clamp");
    }
}
