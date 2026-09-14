//! Exact endgame search: once few candidates remain, stop estimating and
//! compute the guess that minimises the expected number of guesses to finish.
//!
//! For a candidate set `C` with total prior weight `W` and a guess `g`:
//!
//! ```text
//! E(g, C) = 1 + Σ_buckets (W_B / W) · E*(B)      over non-winning buckets B
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

/// Optimistic cost of finishing from a set of total weight `mass` whose
/// heaviest member weighs `heaviest`: guess that member and have it separate
/// all the rest. Uniform: `2 − 1/m`. One candidate: exactly 1.
#[inline]
pub fn lower_bound(mass: u64, heaviest: u32) -> f64 {
    2.0 - heaviest as f64 / mass as f64
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
            // Nothing to learn: guess the likelier one. Done now with
            // probability w/W, otherwise next turn.
            let pick = self.ctx.most_likely(cands).unwrap();
            let mass = self.ctx.mass(&candidate_list(cands));
            return (pick, lower_bound(mass, self.ctx.weight(pick)));
        }
        if let Some(&hit) = self.memo.get(cands) {
            return hit;
        }
        self.nodes += 1;

        let list = candidate_list(cands);
        let mass = self.ctx.mass(&list);
        let mass_f = mass as f64;

        // Rank every guess by its optimistic cost. Guesses that leave all
        // candidates in one bucket make no progress and are dropped; that
        // can never be all of them, since guessing a candidate always splits
        // off the winning bucket.
        let n_answers = self.ctx.num_answers();
        let mut ranked: Vec<(f64, WordId)> = Vec::with_capacity(self.ctx.num_guesses());
        let mut bucket_mass = [0u64; NUM_PATTERNS];
        let mut bucket_max = [0u32; NUM_PATTERNS];
        let mut touched: Vec<usize> = Vec::with_capacity(n);
        for g in 0..self.ctx.num_guesses() as u16 {
            let row = &self.ctx.table[g as usize * n_answers..][..n_answers];
            touched.clear();
            for &a in &list {
                let p = row[a.index()].index();
                let w = self.ctx.weight(a);
                if bucket_mass[p] == 0 {
                    touched.push(p);
                }
                bucket_mass[p] += w as u64;
                bucket_max[p] = bucket_max[p].max(w);
            }
            let mut bound = 1.0;
            let mut progress = true;
            for &p in &touched {
                let (m, h) = (bucket_mass[p], bucket_max[p]);
                bucket_mass[p] = 0;
                bucket_max[p] = 0;
                if m == mass {
                    progress = false;
                } else if p != Pattern::WIN.index() {
                    bound += (m as f64 / mass_f) * lower_bound(m, h);
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
            let exact = self.expected(g, &list, mass, best.1);
            if exact < best.1 {
                best = (g, exact);
            }
        }

        self.memo.insert(*cands, best);
        best
    }

    /// `E(guess, list)`, or `+inf` as soon as it provably cannot beat
    /// `cutoff`. Never memoised, because a cutoff result is not exact.
    fn expected(&mut self, guess: WordId, list: &[WordId], mass: u64, cutoff: f64) -> f64 {
        let mass_f = mass as f64;
        let n_answers = self.ctx.num_answers();
        let row = &self.ctx.table[guess.index() * n_answers..][..n_answers];

        // Group the candidates by the pattern this guess gives them. At most
        // `list.len()` groups, so a linear scan beats a map.
        struct Group {
            pattern: Pattern,
            set: CandidateSet,
            mass: u64,
            heaviest: u32,
        }
        let mut groups: Vec<Group> = Vec::new();
        for &a in list {
            let p = row[a.index()];
            if p == Pattern::WIN {
                continue; // the guess itself: 0 further guesses
            }
            let w = self.ctx.weight(a);
            match groups.iter_mut().find(|g| g.pattern == p) {
                Some(g) => {
                    g.set.insert(a);
                    g.mass += w as u64;
                    g.heaviest = g.heaviest.max(w);
                }
                None => {
                    let mut set = CandidateSet::empty();
                    set.insert(a);
                    groups.push(Group {
                        pattern: p,
                        set,
                        mass: w as u64,
                        heaviest: w,
                    });
                }
            }
        }

        // Start from the optimistic total, then replace each group's bound
        // with its exact cost; bail the moment the total can't beat cutoff.
        let mut total = 1.0
            + groups
                .iter()
                .map(|g| (g.mass as f64 / mass_f) * lower_bound(g.mass, g.heaviest))
                .sum::<f64>();
        for g in &groups {
            let weight = g.mass as f64 / mass_f;
            let (_, exact) = self.best(&g.set);
            total += weight * (exact - lower_bound(g.mass, g.heaviest));
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

    fn words(l: &[&str]) -> Vec<crate::types::Word> {
        l.iter().map(|s| parse_word(s).unwrap()).collect()
    }

    fn uniform(answers: &[&str], extra: &[&str]) -> Context {
        Context::new(words(answers), words(extra)).unwrap()
    }

    /// Nine answers with distinct weights, so a weighted bug can't hide.
    fn weighted() -> Context {
        let answers = [
            "crane", "shale", "stone", "adobe", "abbey", "lucky", "match", "patch", "latch",
        ];
        Context::with_prior(
            words(&answers),
            words(&["clamp", "soare", "puppy", "llama"]),
            |w| answers.iter().position(|a| a.as_bytes() == w).unwrap() as u32 * 3 + 1,
        )
        .unwrap()
    }

    #[test]
    fn lower_bound_base_cases() {
        assert_eq!(lower_bound(1, 1), 1.0);
        assert_eq!(lower_bound(2, 1), 1.5);
        assert_eq!(
            lower_bound(4, 3),
            1.25,
            "heavy favourite: usually done in one"
        );
        assert!(lower_bound(3, 1) > 1.5 && lower_bound(3, 1) < 2.0);
    }

    #[test]
    fn separating_guess_is_worth_exactly_two() {
        // "clamp" splits all four; playing it costs 1 + 1. Playing "match"
        // costs 1 + (3/4)·2 = 2.5, since the other three collapse together.
        let ctx = uniform(&["match", "patch", "latch", "hatch"], &["clamp"]);
        let mut search = Search::new(&ctx);
        let (g, e) = search.best(&CandidateSet::all(4));
        assert_eq!(ctx.word_str(g), "clamp");
        assert_eq!(e, 2.0);
    }

    #[test]
    fn expected_value_matches_realised_play() {
        // If the model is right, the prior-weighted average length of the
        // games it plays against every answer equals the value it predicted.
        for ctx in [
            uniform(
                &[
                    "crane", "shale", "stone", "adobe", "abbey", "lucky", "match", "patch", "latch",
                ],
                &["clamp", "soare", "puppy", "llama"],
            ),
            weighted(),
        ] {
            let strategy = Lookahead { threshold: 100 };
            let solver = Solver::new(&ctx, &strategy);
            let realised: f64 = ctx
                .answers
                .iter()
                .map(|&a| ctx.weight(a) as f64 * solver.solve(a).len() as f64)
                .sum::<f64>()
                / ctx.total_weight as f64;

            let (_, predicted) = Search::new(&ctx).best(&CandidateSet::all(ctx.num_answers()));
            assert!(
                (realised - predicted).abs() < 1e-9,
                "predicted {predicted}, realised {realised}"
            );
        }
    }

    #[test]
    fn never_worse_than_hybrid_by_its_own_measure() {
        let ctx = weighted();
        let all = CandidateSet::all(ctx.num_answers());
        let list = candidate_list(&all);
        let hybrid_pick = Hybrid.best_guess(&ctx, &all);
        let mut search = Search::new(&ctx);
        let (_, optimum) = search.best(&all);
        let hybrid_cost = search.expected(hybrid_pick, &list, ctx.mass(&list), f64::INFINITY);
        assert!(optimum <= hybrid_cost, "{optimum} > {hybrid_cost}");
    }

    #[test]
    fn two_left_guesses_the_likelier_one() {
        let ctx = weighted();
        let mut two = CandidateSet::empty();
        two.insert(ctx.find(b"crane").unwrap()); // weight 1
        two.insert(ctx.find(b"latch").unwrap()); // weight 25
        let (pick, e) = Search::new(&ctx).best(&two);
        assert_eq!(ctx.word_str(pick), "latch");
        assert!((e - (1.0 + 1.0 / 26.0)).abs() < 1e-12);
    }

    #[test]
    fn defers_to_hybrid_above_threshold() {
        let ctx = uniform(&["crane", "shale", "stone", "adobe"], &["puppy", "soare"]);
        let all = CandidateSet::all(4);
        let small = Lookahead { threshold: 2 };
        assert_eq!(small.best_guess(&ctx, &all), Hybrid.best_guess(&ctx, &all));
    }
}
