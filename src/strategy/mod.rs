//! How to choose the next guess.
//!
//! Every strategy sees the same two things: the immutable [`Context`] and the
//! set of answers still possible. It returns a [`WordId`] from the *full*
//! guess pool — the best splitter is often a word that cannot be the answer.
//!
//! The maths every strategy shares lives here. For one candidate guess `g`,
//! [`histogram`] sorts the remaining candidates into 243 buckets by the tiles
//! `g` would produce against each. Everything a strategy wants to know is a
//! function of those bucket sizes: [`entropy`] measures how evenly `g` splits
//! them on average, [`worst_case`] is the size of the biggest one.

pub mod entropy;
pub mod hybrid;
pub mod lookahead;
pub mod minimax;

use std::cmp::Ordering;
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::state::CandidateSet;
use crate::table::{Context, MathTables};
use crate::types::{NUM_PATTERNS, Pattern, WordId};

pub trait Strategy: Sync {
    /// Short identifier, for CLI flags and report tables.
    fn name(&self) -> &'static str;

    /// The next word to play. `candidates` is never empty when this is called.
    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet) -> WordId;
}

/// Every built-in strategy, in the order the reports print them.
pub fn all() -> Vec<Box<dyn Strategy>> {
    vec![
        Box::new(entropy::Entropy),
        Box::new(minimax::Minimax),
        Box::new(hybrid::Hybrid),
        Box::new(lookahead::Lookahead::default()),
    ]
}

/// Look a strategy up by its [`Strategy::name`], for CLI flags.
/// `lookahead:N` sets the exact-search threshold, for sweeping it.
pub fn by_name(name: &str) -> Option<Box<dyn Strategy>> {
    if let Some(n) = name.strip_prefix("lookahead:") {
        let threshold = n.parse().ok()?;
        return Some(Box::new(lookahead::Lookahead { threshold }));
    }
    all().into_iter().find(|s| s.name() == name)
}

/// The candidate set as a dense list. Strategies build this once per
/// decision so the 13k histogram passes loop a contiguous slice instead of
/// walking bitset words.
pub fn candidate_list(candidates: &CandidateSet) -> Vec<WordId> {
    candidates.iter().collect()
}

/// Bucket counts indexed by [`Pattern`]. 486 bytes, lives on the stack.
pub type Histogram = [u16; NUM_PATTERNS];

/// Sort `candidates` into buckets by the pattern `guess` would produce
/// against each. This is the innermost loop of the whole solver: one byte
/// load per candidate, so it takes the table row and a dense candidate list
/// rather than re-deriving either per call.
#[inline]
pub fn histogram(ctx: &Context, guess: WordId, candidates: &[WordId]) -> Histogram {
    let n = ctx.num_answers();
    let row = &ctx.table[guess.index() * n..][..n];
    let mut hist = [0u16; NUM_PATTERNS];
    for &a in candidates {
        hist[row[a.index()].index()] += 1;
    }
    hist
}

/// Expected information from a guess, in bits, given `n` candidates.
///
/// `H = log2(n) - (1/n) · Σ_b count_b · log2(count_b)` — the usual
/// `-Σ p log p` rearranged so the loop is table lookups, not logarithms.
/// 0 means the guess teaches nothing; `log2(n)` means it identifies the
/// answer outright.
#[inline]
pub fn entropy(hist: &Histogram, n: usize, math: &MathTables) -> f64 {
    let sum: f64 = hist.iter().map(|&c| math.x_log2_x(c as usize)).sum();
    (n as f64).log2() - sum / n as f64
}

/// [`entropy`] and [`worst_case`] in one pass over the histogram.
///
/// Once few candidates remain, the 243-bucket pass costs more than the
/// histogram itself, so a strategy that wants both numbers should not walk
/// the buckets twice. Bit-for-bit identical to calling the two separately.
#[inline]
pub fn entropy_and_worst(hist: &Histogram, n: usize, math: &MathTables) -> (f64, u16) {
    let mut sum = 0.0;
    let mut worst = 0u16;
    for &c in hist {
        sum += math.x_log2_x(c as usize);
        worst = worst.max(c);
    }
    ((n as f64).log2() - sum / n as f64, worst)
}

/// Size of the largest bucket: how many candidates could remain if the tiles
/// come back as unhelpfully as possible.
#[inline]
pub fn worst_case(hist: &Histogram) -> u16 {
    *hist.iter().max().expect("histogram is non-empty")
}

/// Score every allowed guess in parallel and return the best.
///
/// Ties are broken toward the lower [`WordId`], so the result is identical
/// run to run regardless of how rayon schedules the work. Because answers
/// sit at the low ids, that also means "prefer a word that could be the
/// answer" among exact ties.
pub fn pick_best<S, F>(ctx: &Context, score: F) -> WordId
where
    S: PartialOrd + Send,
    F: Fn(WordId) -> S + Sync,
{
    (0..ctx.num_guesses() as u16)
        .into_par_iter()
        .map(|i| {
            let id = WordId(i);
            (score(id), id)
        })
        .reduce_with(|best, other| match other.0.partial_cmp(&best.0) {
            Some(Ordering::Greater) => other,
            Some(Ordering::Equal) if other.1 < best.1 => other,
            _ => best,
        })
        .expect("guess pool is non-empty")
        .1
}

/// Hard cap on guesses in [`Solver::solve`]. Wordle allows 6; anything past
/// that is already a failure, and this only exists so a broken strategy
/// cannot loop forever.
pub const MAX_TURNS: usize = 10;

/// Drives a [`Strategy`] through a game, adding the two things that are rules
/// of the game rather than strategy: play a candidate outright when at most
/// two remain (there is nothing left to learn), and remember the opening
/// guess, which is identical for every game and dominates batch runs.
pub struct Solver<'a> {
    ctx: &'a Context,
    strategy: &'a dyn Strategy,
    opener: OnceLock<WordId>,
}

impl<'a> Solver<'a> {
    pub fn new(ctx: &'a Context, strategy: &'a dyn Strategy) -> Solver<'a> {
        Solver {
            ctx,
            strategy,
            opener: OnceLock::new(),
        }
    }

    pub fn strategy(&self) -> &dyn Strategy {
        self.strategy
    }

    pub fn next_guess(&self, candidates: &CandidateSet) -> WordId {
        assert!(
            !candidates.is_empty(),
            "no candidates left; the feedback was contradictory"
        );
        if candidates.len() <= 2 {
            return candidates.first().unwrap();
        }
        if candidates.len() == self.ctx.num_answers() {
            return *self
                .opener
                .get_or_init(|| self.strategy.best_guess(self.ctx, candidates));
        }
        self.strategy.best_guess(self.ctx, candidates)
    }

    /// Play a full game against a known answer, using the table as the
    /// oracle. Returns the guesses played, ending with `answer` if solved
    /// within [`MAX_TURNS`].
    pub fn solve(&self, answer: WordId) -> Vec<WordId> {
        let mut candidates = CandidateSet::all(self.ctx.num_answers());
        let mut played = Vec::with_capacity(6);
        while played.len() < MAX_TURNS {
            let guess = self.next_guess(&candidates);
            played.push(guess);
            let pattern = self.ctx.get_pattern(guess, answer);
            if pattern == Pattern::WIN {
                break;
            }
            candidates.filter(guess, pattern, self.ctx);
        }
        played
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::parse_word;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn ctx() -> Context {
        let w = |l: &[&str]| l.iter().map(|s| parse_word(s).unwrap()).collect();
        Context::new(
            w(&["crane", "shale", "stone", "adobe"]),
            w(&["puppy", "soare"]),
        )
        .unwrap()
    }

    fn all(ctx: &Context) -> Vec<WordId> {
        ctx.answers.clone()
    }

    #[test]
    fn histogram_counts_every_candidate_once() {
        let ctx = ctx();
        let crane = ctx.find(b"crane").unwrap();
        let hist = histogram(&ctx, crane, &all(&ctx));
        assert_eq!(hist.iter().map(|&c| c as usize).sum::<usize>(), 4);
        assert_eq!(hist[Pattern::WIN.index()], 1);
        // "crane" gives every candidate a different pattern.
        assert_eq!(hist.iter().filter(|&&c| c > 0).count(), 4);
    }

    #[test]
    fn entropy_spans_zero_to_log2_n() {
        let ctx = ctx();
        let cands = all(&ctx);
        // No shared letters with any candidate: one bucket, nothing learned.
        let puppy = ctx.find(b"puppy").unwrap();
        let h = histogram(&ctx, puppy, &cands);
        assert_eq!(entropy(&h, 4, &ctx.math), 0.0);
        assert_eq!(worst_case(&h), 4);
        // Perfect split: log2(4) = 2 bits, worst bucket 1.
        let crane = ctx.find(b"crane").unwrap();
        let h = histogram(&ctx, crane, &cands);
        assert_eq!(entropy(&h, 4, &ctx.math), 2.0);
        assert_eq!(worst_case(&h), 1);
    }

    #[test]
    fn fused_pass_matches_the_separate_reductions() {
        let ctx = ctx();
        let cands = all(&ctx);
        for g in 0..ctx.num_guesses() as u16 {
            let h = histogram(&ctx, WordId(g), &cands);
            let (e, w) = entropy_and_worst(&h, cands.len(), &ctx.math);
            assert_eq!(e, entropy(&h, cands.len(), &ctx.math));
            assert_eq!(w, worst_case(&h));
        }
    }

    #[test]
    fn pick_best_is_deterministic_on_ties() {
        let ctx = ctx();
        // Every guess scores equally; the lowest id must win, every time.
        for _ in 0..20 {
            assert_eq!(pick_best(&ctx, |_| 1.0f64), WordId(0));
        }
        // Tuples order lexicographically: the second field breaks the tie.
        let best = pick_best(&ctx, |id| (1.0f64, id.0 == 3));
        assert_eq!(best, WordId(3));
    }

    /// Plays the lowest-id candidate and counts how often it was asked.
    struct Counting(AtomicUsize);
    impl Strategy for Counting {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn best_guess(&self, _ctx: &Context, c: &CandidateSet) -> WordId {
            self.0.fetch_add(1, AtomicOrdering::SeqCst);
            c.first().unwrap()
        }
    }

    #[test]
    fn solver_caches_the_opener_and_skips_the_endgame() {
        let ctx = ctx();
        let strategy = Counting(AtomicUsize::new(0));
        let solver = Solver::new(&ctx, &strategy);
        let everything = CandidateSet::all(ctx.num_answers());

        assert_eq!(solver.next_guess(&everything), WordId(0));
        assert_eq!(solver.next_guess(&everything), WordId(0));
        assert_eq!(
            strategy.0.load(AtomicOrdering::SeqCst),
            1,
            "opener computed once"
        );

        let mut two = CandidateSet::empty();
        two.insert(WordId(2));
        two.insert(WordId(3));
        assert_eq!(solver.next_guess(&two), WordId(2));
        assert_eq!(
            strategy.0.load(AtomicOrdering::SeqCst),
            1,
            "endgame bypasses the strategy"
        );
    }

    #[test]
    fn solve_ends_on_the_answer() {
        let ctx = ctx();
        let strategy = Counting(AtomicUsize::new(0));
        let solver = Solver::new(&ctx, &strategy);
        for &answer in &ctx.answers {
            let played = solver.solve(answer);
            assert_eq!(played.last(), Some(&answer), "{}", ctx.word_str(answer));
            assert!(played.len() <= ctx.num_answers());
        }
    }
}
