//! Exact endgame search: once few candidates remain, stop estimating and
//! compute the guess that minimises the expected number of guesses to finish
//! — without ever running past Wordle's six-guess cap.
//!
//! For a candidate set `C` with total prior weight `W`, `t` turns left, and
//! a guess `g`:
//!
//! ```text
//! cost(g, C, t) = ( Σ_B (W_B/W) · fail(B, t−1),  1 + Σ_B (W_B/W) · guesses(B, t−1) )
//! cost*(C, t)   = min_g cost(g, C, t)          over non-winning buckets B
//! ```
//!
//! A [`Cost`] is compared lexicographically: probability of failing first,
//! expected guesses second. Any reduction in failure beats any saving in
//! guesses, with no penalty constant to tune. With enough turns the first
//! component is 0 everywhere and this is plain expected-guess minimisation.
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

    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet, turns_left: usize) -> WordId {
        if candidates.len() > self.threshold {
            return Hybrid.best_guess(ctx, candidates, turns_left);
        }
        Search::new(ctx).best(candidates, turns_left).0
    }
}

/// The value of a position, compared field by field in this order: a lower
/// chance of failing always wins; expected guesses only break ties.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Cost {
    /// Probability mass (0..=1) of answers that will not be found in time.
    pub fail: f64,
    /// Expected number of guesses made, counting those in lost games.
    pub guesses: f64,
}

impl Cost {
    pub const INFINITE: Cost = Cost {
        fail: f64::INFINITY,
        guesses: f64::INFINITY,
    };

    fn scaled(self, p: f64) -> Cost {
        Cost {
            fail: self.fail * p,
            guesses: self.guesses * p,
        }
    }
}

impl std::ops::Add for Cost {
    type Output = Cost;
    fn add(self, o: Cost) -> Cost {
        Cost {
            fail: self.fail + o.fail,
            guesses: self.guesses + o.guesses,
        }
    }
}

impl std::ops::Sub for Cost {
    type Output = Cost;
    fn sub(self, o: Cost) -> Cost {
        Cost {
            fail: self.fail - o.fail,
            guesses: self.guesses - o.guesses,
        }
    }
}

/// Optimistic cost of finishing a set of total weight `mass` whose heaviest
/// member weighs `heaviest`, with `turns` guesses still allowed.
///
/// One turn: you must play the likeliest word and hope — exact, not a
/// bound. Two or more: assume that word also separates all the rest, so
/// nothing fails and it takes `2 − w_max/W` guesses (`1` for a singleton).
#[inline]
pub fn lower_bound(mass: u64, heaviest: u32, turns: usize) -> Cost {
    let p_hit = heaviest as f64 / mass as f64;
    match turns {
        0 => Cost {
            fail: 1.0,
            guesses: 0.0,
        },
        1 => Cost {
            fail: 1.0 - p_hit,
            guesses: 1.0,
        },
        _ => Cost {
            fail: 0.0,
            guesses: 2.0 - p_hit,
        },
    }
}

/// One exact search. Holds the memo so equal subsets reached through
/// different guesses are solved once.
pub struct Search<'a> {
    ctx: &'a Context,
    memo: HashMap<(CandidateSet, u8), (WordId, Cost)>,
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

    /// The optimal guess from `cands` with `turns_left` guesses allowed, and
    /// the cost of playing it and continuing optimally.
    pub fn best(&mut self, cands: &CandidateSet, turns_left: usize) -> (WordId, Cost) {
        let n = cands.len();
        debug_assert!(n > 0);
        let list = candidate_list(cands);
        let mass = self.ctx.mass(&list);

        // With two candidates, or on the last turn, there is nothing to
        // learn: play the likeliest word. `lower_bound` is exact here.
        if n <= 2 || turns_left <= 1 {
            let pick = self.ctx.most_likely(cands).unwrap();
            return (pick, lower_bound(mass, self.ctx.weight(pick), turns_left));
        }
        let key = (*cands, turns_left as u8);
        if let Some(&hit) = self.memo.get(&key) {
            return hit;
        }
        self.nodes += 1;

        let mass_f = mass as f64;
        let after = turns_left - 1;

        // Rank every guess by its optimistic cost. Guesses that leave all
        // candidates in one bucket make no progress and are dropped; that
        // can never be all of them, since guessing a candidate always splits
        // off the winning bucket.
        let n_answers = self.ctx.num_answers();
        let mut ranked: Vec<(Cost, WordId)> = Vec::with_capacity(self.ctx.num_guesses());
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
            let mut bound = Cost {
                fail: 0.0,
                guesses: 1.0,
            };
            let mut progress = true;
            for &p in &touched {
                let (m, h) = (bucket_mass[p], bucket_max[p]);
                bucket_mass[p] = 0;
                bucket_max[p] = 0;
                if m == mass {
                    progress = false;
                } else if p != Pattern::WIN.index() {
                    bound = bound + lower_bound(m, h, after).scaled(m as f64 / mass_f);
                }
            }
            if progress {
                ranked.push((bound, WordId(g)));
            }
        }
        ranked.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("costs are never NaN")
                .then(a.1.cmp(&b.1))
        });

        let mut best = (ranked[0].1, Cost::INFINITE);
        for &(bound, g) in &ranked {
            if bound >= best.1 {
                break;
            }
            let exact = self.expected(g, &list, mass, turns_left, best.1);
            if exact < best.1 {
                best = (g, exact);
            }
        }

        self.memo.insert(key, best);
        best
    }

    /// `cost(guess, list, turns_left)`, or [`Cost::INFINITE`] as soon as it
    /// provably cannot beat `cutoff`. Never memoised, because a cutoff
    /// result is not exact.
    fn expected(
        &mut self,
        guess: WordId,
        list: &[WordId],
        mass: u64,
        turns_left: usize,
        cutoff: Cost,
    ) -> Cost {
        let mass_f = mass as f64;
        let after = turns_left - 1;
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
        let mut total = Cost {
            fail: 0.0,
            guesses: 1.0,
        };
        for g in &groups {
            total = total + lower_bound(g.mass, g.heaviest, after).scaled(g.mass as f64 / mass_f);
        }
        for g in &groups {
            let p = g.mass as f64 / mass_f;
            let (_, exact) = self.best(&g.set, after);
            total = total + (exact - lower_bound(g.mass, g.heaviest, after)).scaled(p);
            if total >= cutoff {
                return Cost::INFINITE;
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{Solver, WORDLE_TURNS};
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

    fn cost(fail: f64, guesses: f64) -> Cost {
        Cost { fail, guesses }
    }

    #[test]
    fn cost_orders_failure_before_speed() {
        assert!(cost(0.0, 5.0) < cost(0.001, 1.0));
        assert!(cost(0.0, 1.5) < cost(0.0, 2.0));
        assert!(cost(0.0, 2.0) < Cost::INFINITE);
    }

    #[test]
    fn lower_bound_base_cases() {
        assert_eq!(lower_bound(1, 1, 6), cost(0.0, 1.0));
        assert_eq!(lower_bound(2, 1, 6), cost(0.0, 1.5));
        assert_eq!(lower_bound(4, 3, 6), cost(0.0, 1.25), "heavy favourite");
        assert_eq!(
            lower_bound(4, 3, 1),
            cost(0.25, 1.0),
            "last turn: 3-in-4 hit"
        );
        assert_eq!(
            lower_bound(4, 3, 0),
            cost(1.0, 0.0),
            "no turns: certain loss"
        );
    }

    #[test]
    fn separating_guess_is_worth_exactly_two() {
        // "clamp" splits all four; playing it costs 1 + 1. Playing "match"
        // costs 1 + (3/4)·2 = 2.5, since the other three collapse together.
        let ctx = uniform(&["match", "patch", "latch", "hatch"], &["clamp"]);
        let mut search = Search::new(&ctx);
        let (g, c) = search.best(&CandidateSet::all(4), WORDLE_TURNS);
        assert_eq!(ctx.word_str(g), "clamp");
        assert_eq!(c, cost(0.0, 2.0));
    }

    #[test]
    fn the_cap_changes_the_decision() {
        // "match" is a 1000:1 favourite. Unconstrained, guess it: usually
        // done in one, and the other three can still be separated later.
        // With only two turns left that "later" doesn't exist -- if match is
        // wrong, one turn can't tell patch/latch/hatch apart -- so the
        // search must switch to "clamp", which never fails.
        let ctx = Context::with_prior(
            words(&["match", "patch", "latch", "hatch"]),
            words(&["clamp"]),
            |w| if w == b"match" { 1000 } else { 1 },
        )
        .unwrap();
        let all = CandidateSet::all(4);
        let mut search = Search::new(&ctx);

        let (g, c) = search.best(&all, 3);
        assert_eq!(ctx.word_str(g), "match");
        assert_eq!(c.fail, 0.0);
        assert!((c.guesses - (1.0 + 3.0 / 1003.0 * 2.0)).abs() < 1e-12);

        let (g, c) = search.best(&all, 2);
        assert_eq!(ctx.word_str(g), "clamp");
        // 1/1003 fractions are not exact in binary; the bound-then-refine
        // accumulation leaves ~1e-16 of dust.
        assert_eq!(c.fail, 0.0);
        assert!((c.guesses - 2.0).abs() < 1e-12, "{c:?}");

        // Cornered: one turn, four candidates. Best is the favourite, and
        // the cost is honest about the 3/1003 that gets lost.
        let (g, c) = search.best(&all, 1);
        assert_eq!(ctx.word_str(g), "match");
        assert!((c.fail - 3.0 / 1003.0).abs() < 1e-12);
        assert_eq!(c.guesses, 1.0);
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

            let (_, predicted) =
                Search::new(&ctx).best(&CandidateSet::all(ctx.num_answers()), WORDLE_TURNS);
            assert_eq!(predicted.fail, 0.0);
            assert!(
                (realised - predicted.guesses).abs() < 1e-9,
                "predicted {predicted:?}, realised {realised}"
            );
        }
    }

    #[test]
    fn never_worse_than_hybrid_by_its_own_measure() {
        let ctx = weighted();
        let all = CandidateSet::all(ctx.num_answers());
        let list = candidate_list(&all);
        let hybrid_pick = Hybrid.best_guess(&ctx, &all, WORDLE_TURNS);
        let mut search = Search::new(&ctx);
        let (_, optimum) = search.best(&all, WORDLE_TURNS);
        let hybrid_cost = search.expected(
            hybrid_pick,
            &list,
            ctx.mass(&list),
            WORDLE_TURNS,
            Cost::INFINITE,
        );
        assert!(optimum <= hybrid_cost, "{optimum:?} > {hybrid_cost:?}");
    }

    #[test]
    fn two_left_guesses_the_likelier_one() {
        let ctx = weighted();
        let mut two = CandidateSet::empty();
        two.insert(ctx.find(b"crane").unwrap()); // weight 1
        two.insert(ctx.find(b"latch").unwrap()); // weight 25
        let (pick, c) = Search::new(&ctx).best(&two, 6);
        assert_eq!(ctx.word_str(pick), "latch");
        assert!((c.guesses - (1.0 + 1.0 / 26.0)).abs() < 1e-12);
        let (pick, c) = Search::new(&ctx).best(&two, 1);
        assert_eq!(ctx.word_str(pick), "latch");
        assert!((c.fail - 1.0 / 26.0).abs() < 1e-12);
    }

    #[test]
    fn defers_to_hybrid_above_threshold() {
        let ctx = uniform(&["crane", "shale", "stone", "adobe"], &["puppy", "soare"]);
        let all = CandidateSet::all(4);
        let small = Lookahead { threshold: 2 };
        assert_eq!(
            small.best_guess(&ctx, &all, WORDLE_TURNS),
            Hybrid.best_guess(&ctx, &all, WORDLE_TURNS)
        );
    }
}
