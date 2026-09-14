//! Every strategy against the real word lists. A sample, not the full 2,315:
//! the full run is `cargo run --release --bin evaluate`.

use std::sync::OnceLock;

use wordlesolver::strategy::{self, entropy, histogram};
use wordlesolver::{CandidateSet, Context, Solver};

fn ctx() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::bundled)
}

#[test]
fn every_strategy_solves_a_sample_within_six() {
    let ctx = ctx();
    for strategy in strategy::all() {
        let solver = Solver::new(ctx, strategy.as_ref());
        let mut total = 0;
        let mut games = 0;
        for &answer in ctx.answers.iter().step_by(50) {
            let played = solver.solve(answer);
            let word = ctx.word_str(answer);
            assert_eq!(
                played.last(),
                Some(&answer),
                "{}: {word} not reached",
                strategy.name()
            );
            assert!(
                played.len() <= 6,
                "{}: {word} took {} guesses: {:?}",
                strategy.name(),
                played.len(),
                played.iter().map(|&g| ctx.word_str(g)).collect::<Vec<_>>()
            );
            total += played.len();
            games += 1;
        }
        eprintln!(
            "{:>8}: {games} games, {:.3} guesses on average",
            strategy.name(),
            total as f64 / games as f64
        );
    }
}

#[test]
fn openers_are_strong_and_stable() {
    let ctx = ctx();
    let all = CandidateSet::all(ctx.num_answers());
    let cands = strategy::candidate_list(&all);
    for strategy in strategy::all() {
        let opener = strategy.best_guess(ctx, &all);
        let again = strategy.best_guess(ctx, &all);
        assert_eq!(
            opener,
            again,
            "{} opener is not deterministic",
            strategy.name()
        );

        let bits = entropy(&histogram(ctx, opener, &cands), cands.len(), &ctx.math);
        eprintln!(
            "{:>8} opens with {} ({bits:.3} bits)",
            strategy.name(),
            ctx.word_str(opener)
        );
        // log2(2315) ≈ 11.2 bits is the ceiling; any sane opener clears 5.
        assert!(
            bits > 5.0,
            "{} opener {} only worth {bits} bits",
            strategy.name(),
            ctx.word_str(opener)
        );
    }
}

#[test]
fn by_name_round_trips() {
    for s in strategy::all() {
        assert_eq!(strategy::by_name(s.name()).unwrap().name(), s.name());
    }
    assert!(strategy::by_name("nope").is_none());
}
