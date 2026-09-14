//! Every strategy against the real word lists. A sample, not the full 2,315:
//! the full run is `cargo run --release --bin evaluate`.

use std::sync::OnceLock;

use wordlesolver::strategy::{self, WORDLE_TURNS, entropy, histogram};
use wordlesolver::words::{BUNDLED_ANSWERS, parse_list};
use wordlesolver::{CandidateSet, Context, Solver};

fn ctx() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::curated)
}

fn open_ctx() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::open)
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
        let opener = strategy.best_guess(ctx, &all, WORDLE_TURNS);
        let again = strategy.best_guess(ctx, &all, WORDLE_TURNS);
        assert_eq!(
            opener,
            again,
            "{} opener is not deterministic",
            strategy.name()
        );

        let bits = entropy(&histogram(ctx, opener, &cands), ctx.mass(&cands), &ctx.math);
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
fn open_mode_puts_most_of_the_prior_on_real_answers() {
    let ctx = open_ctx();
    assert_eq!(
        ctx.num_answers(),
        12_972,
        "every allowed word is a candidate"
    );
    assert_eq!(ctx.num_guesses(), 12_972);
    // Answer ids are shared with curated mode: the first 2,315 words.
    let answers = parse_list(BUNDLED_ANSWERS).unwrap();
    for (i, w) in answers.iter().enumerate() {
        assert_eq!(ctx.word(wordlesolver::WordId(i as u16)), w);
    }
    let on_answers: u64 = (0..answers.len() as u16)
        .map(|i| ctx.weight(wordlesolver::WordId(i)) as u64)
        .sum();
    let share = on_answers as f64 / ctx.total_weight as f64;
    eprintln!(
        "open mode: {:.1}% of prior mass on the 2,315 real answers (total weight {})",
        share * 100.0,
        ctx.total_weight
    );
    // Measured 48.9% with the bundled prior: the 2,315 real answers weigh
    // about as much as the other 10,657 words together, a 4.6x per-word
    // lift. A regression guard, not a target -- tuning the prior to push
    // this up would be fitting it to the benchmark.
    assert!(share > 0.45, "prior share on answers only {share}");
}

#[test]
fn open_mode_solves_a_sample_of_real_answers() {
    let ctx = open_ctx();
    let strategy = strategy::by_name("hybrid").unwrap();
    let solver = Solver::new(ctx, strategy.as_ref());
    let all = CandidateSet::all(ctx.num_answers());
    let opener = solver.next_guess(&all, WORDLE_TURNS);
    eprintln!("open mode hybrid opens with {}", ctx.word_str(opener));

    let mut total = 0;
    let mut games = 0;
    for i in (0..2315u16).step_by(100) {
        let answer = wordlesolver::WordId(i);
        let played = solver.solve(answer);
        assert_eq!(
            played.last(),
            Some(&answer),
            "{} not reached",
            ctx.word_str(answer)
        );
        total += played.len();
        games += 1;
    }
    eprintln!(
        "open mode hybrid: {games} games, {:.3} guesses on average",
        total as f64 / games as f64
    );
}

#[test]
fn by_name_round_trips() {
    for s in strategy::all() {
        assert_eq!(strategy::by_name(s.name()).unwrap().name(), s.name());
    }
    assert!(strategy::by_name("nope").is_none());
}
