//! Shannon entropy: play the guess whose tiles, on average, tell you the most.
//!
//! Greedy and one step deep. It does not care whether the guess could itself
//! be the answer, which is exactly what [`Hybrid`](super::hybrid::Hybrid)
//! adds on top.

use super::{Strategy, candidate_list, entropy, histogram, pick_best};
use crate::state::CandidateSet;
use crate::table::Context;
use crate::types::WordId;

pub struct Entropy;

impl Strategy for Entropy {
    fn name(&self) -> &'static str {
        "entropy"
    }

    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet, _turns_left: usize) -> WordId {
        let cands = candidate_list(candidates);
        let mass = ctx.mass(&cands);
        pick_best(ctx, |g| {
            entropy(&histogram(ctx, g, &cands), mass, &ctx.math)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;

    #[test]
    fn prefers_the_perfect_splitter() {
        let w = |l: &[&str]| l.iter().map(|s| parse_word(s).unwrap()).collect();
        // "crane" and "soare" both split these four perfectly; "puppy" learns
        // nothing. Exact tie goes to the lower id, which is the answer "crane".
        let ctx = Context::new(
            w(&["crane", "shale", "stone", "adobe"]),
            w(&["puppy", "soare"]),
        )
        .unwrap();
        let best = Entropy.best_guess(&ctx, &CandidateSet::all(4), 6);
        assert_eq!(ctx.word_str(best), "crane");
    }
}
