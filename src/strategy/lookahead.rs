//! Exact endgame search: once few candidates remain, stop estimating and
//! compute the guess that minimises the expected number of guesses to finish.
//!
//! For a candidate set `C` of size `n` and a guess `g`:
//!
//! ```text
//! E(g, C) = 1 + Σ_buckets (|B| / n) · E*(B)      over non-winning buckets B
//! E*(C)   = min_g E(g, C)
//! ```
//!
//! Every greedy strategy optimises a proxy for this (bits, worst bucket).
//! Here we optimise the thing itself, which is only affordable because
//! the search is bounded, memoised, and never walks empty buckets.
//! Above [`Lookahead::threshold`] candidates it defers to [`Hybrid`].

use std::collections::HashMap;

use super::hybrid::Hybrid;
use super::{Strategy, candidate_list};
use crate::state::CandidateSet;
use crate::table::Context;
use crate::types::{NUM_PATTERNS, Pattern, WordId};

pub const DEFAULT_THRESHOLD: usize = 16;

pub struct Lookahead {
    /// Search exactly at or below this many candidates; use hybrid above.
    pub threshold: usize,
}

impl Default for Lookahead {
    fn default() -> Lookahead {
        Lookahead {
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl Strategy for Lookahead {
    fn name(&self) -> &'static str {
        "lookahead"
    }

    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet) -> WordId {
        if candidates.len() > self.threshold {
            return Hybrid.best_guess(ctx, candidates);
        }
        Search::new(ctx).best(candidates).0
    }
}

/// Optimistic cost of finishing from `m` candidates: guess one of them and
/// have it separate all the rest. `1` for one candidate, `1.5` for two.
#[inline]
pub fn lower_bound(m: usize) -> f64 {
    2.0 - 1.0 / m as f64
}

/// One exact search. Holds the memo so equal subsets reached through
/// different guesses are solved once.
pub struct Search<'a> {
    ctx: &'a Context,
    memo: HashMap<CandidateSet, (WordId, f64)>,
    /// Subsets solved exactly. Exposed so tests and benchmarks can see how
    /// hard the bound is working.
    pub nodes: usize,
}

impl<'a> Search<'a> {
    pub fn new(ctx: &'a Context) -> Search<'a> {
        Search {
            ctx,
            memo: HashMap::new(),
            nodes: 0,
        }
    }

    /// The optimal guess from `cands`, and the expected total guesses to
    /// finish if you play it and keep playing optimally.
    pub fn best(&mut self, cands: &CandidateSet) -> (WordId, f64) {
        let n = cands.len();
        debug_assert!(n > 0);
        if n <= 2 {
            // Nothing to learn: guess one. Solved now, or next turn.
            return (cands.first().unwrap(), lower_bound(n));
        }
        if let Some(&hit) = self.memo.get(cands) {
            return hit;
        }
        self.nodes += 1;

        let list = candidate_list(cands);
        let n_f = n as f64;

        // Rank every guess by its optimistic cost. Guesses that leave all n
        // candidates in one bucket make no progress and are dropped; that
        // can never be all of them, since guessing a candidate always splits
        // off the winning bucket.
        let n_answers = self.ctx.num_answers();
        let mut ranked: Vec<(f64, WordId)> = Vec::with_capacity(self.ctx.num_guesses());
        let mut hist = [0u16; NUM_PATTERNS];
        let mut touched: Vec<usize> = Vec::with_capacity(n);
        for g in 0..self.ctx.num_guesses() as u16 {
            let row = &self.ctx.table[g as usize * n_answers..][..n_answers];
            touched.clear();
            for &a in &list {
                let p = row[a.index()].index();
                if hist[p] == 0 {
                    touched.push(p);
                }
                hist[p] += 1;
            }
            let mut bound = 1.0;
            let mut progress = true;
            for &p in &touched {
                let c = hist[p] as usize;
                hist[p] = 0;
                if c == n {
                    progress = false;
                } else if p != Pattern::WIN.index() {
                    bound += (c as f64 / n_f) * lower_bound(c);
                }
            }
            if progress {
                ranked.push((bound, WordId(g)));
            }
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut best = (ranked[0].1, f64::INFINITY);
        for &(bound, g) in &ranked {
            if bound >= best.1 {
                break;
            }
            let exact = self.expected(g, &list, best.1);
            if exact < best.1 {
                best = (g, exact);
            }
        }

        self.memo.insert(*cands, best);
        best
    }

    /// `E(guess, list)`, or `+inf` as soon as it provably cannot beat
    /// `cutoff`. Never memoised, because a cutoff result is not exact.
    fn expected(&mut self, guess: WordId, list: &[WordId], cutoff: f64) -> f64 {
        let n = list.len() as f64;
        let n_answers = self.ctx.num_answers();
        let row = &self.ctx.table[guess.index() * n_answers..][..n_answers];

        // Group the candidates by the pattern this guess gives them. At most
        // `list.len()` groups, so a linear scan beats a map.
        let mut groups: Vec<(Pattern, CandidateSet)> = Vec::new();
        for &a in list {
            let p = row[a.index()];
            if p == Pattern::WIN {
                continue; // the guess itself: 0 further guesses
            }
            match groups.iter_mut().find(|(q, _)| *q == p) {
                Some((_, set)) => set.insert(a),
                None => {
                    let mut set = CandidateSet::empty();
                    set.insert(a);
                    groups.push((p, set));
                }
            }
        }

        // Start from the optimistic total, then replace each group's bound
        // with its exact cost; bail the moment the total can't beat cutoff.
        let mut total = 1.0
            + groups
                .iter()
                .map(|(_, s)| (s.len() as f64 / n) * lower_bound(s.len()))
                .sum::<f64>();
        for (_, set) in &groups {
            let weight = set.len() as f64 / n;
            let (_, exact) = self.best(set);
            total += weight * (exact - lower_bound(set.len()));
            if total >= cutoff {
                return f64::INFINITY;
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::Solver;
    use crate::words::parse_word;

    fn ctx(answers: &[&str], extra: &[&str]) -> Context {
        let w = |l: &[&str]| l.iter().map(|s| parse_word(s).unwrap()).collect();
        Context::new(w(answers), w(extra)).unwrap()
    }

    #[test]
    fn lower_bound_base_cases() {
        assert_eq!(lower_bound(1), 1.0);
        assert_eq!(lower_bound(2), 1.5);
        assert!(lower_bound(3) > 1.5 && lower_bound(3) < 2.0);
    }

    #[test]
    fn separating_guess_is_worth_exactly_two() {
        // "clamp" splits all four; playing it costs 1 + 1. Playing "match"
        // costs 1 + (3/4)·2 = 2.5, since the other three collapse together.
        let ctx = ctx(&["match", "patch", "latch", "hatch"], &["clamp"]);
        let mut search = Search::new(&ctx);
        let (g, e) = search.best(&CandidateSet::all(4));
        assert_eq!(ctx.word_str(g), "clamp");
        assert_eq!(e, 2.0);
    }

    #[test]
    fn expected_value_matches_realised_play() {
        // If the model is right, the average length of the games it plays
        // against every answer must equal the value it predicted.
        let ctx = ctx(
            &[
                "crane", "shale", "stone", "adobe", "abbey", "lucky", "match", "patch", "latch",
            ],
            &["clamp", "soare", "puppy", "llama"],
        );
        let strategy = Lookahead { threshold: 100 };
        let solver = Solver::new(&ctx, &strategy);
        let realised: f64 = ctx
            .answers
            .iter()
            .map(|&a| solver.solve(a).len() as f64)
            .sum::<f64>()
            / ctx.num_answers() as f64;

        let (_, predicted) = Search::new(&ctx).best(&CandidateSet::all(ctx.num_answers()));
        assert!(
            (realised - predicted).abs() < 1e-9,
            "predicted {predicted}, realised {realised}"
        );
    }

    #[test]
    fn never_worse_than_hybrid_by_its_own_measure() {
        let ctx = ctx(
            &[
                "crane", "shale", "stone", "adobe", "abbey", "lucky", "match", "patch", "latch",
            ],
            &["clamp", "soare", "puppy", "llama"],
        );
        let all = CandidateSet::all(ctx.num_answers());
        let list = candidate_list(&all);
        let hybrid_pick = Hybrid.best_guess(&ctx, &all);
        let mut search = Search::new(&ctx);
        let (_, optimum) = search.best(&all);
        let hybrid_cost = search.expected(hybrid_pick, &list, f64::INFINITY);
        assert!(optimum <= hybrid_cost, "{optimum} > {hybrid_cost}");
    }

    #[test]
    fn defers_to_hybrid_above_threshold() {
        let ctx = ctx(&["crane", "shale", "stone", "adobe"], &["puppy", "soare"]);
        let all = CandidateSet::all(4);
        let small = Lookahead { threshold: 2 };
        assert_eq!(small.best_guess(&ctx, &all), Hybrid.best_guess(&ctx, &all));
    }
}
