//! Play every answer against each strategy and report how they did.
//!
//!     cargo run --release --bin evaluate                 # all strategies, curated list
//!     cargo run --release --bin evaluate -- --open       # solver doesn't know the answer list
//!     cargo run --release --bin evaluate -- --open --uniform   # ...and no frequency prior
//!     cargo run --release --bin evaluate -- --strategy hybrid --limit 200
//!
//! The benchmark always scores against the 2,315 real answers. What changes
//! with `--open` is what the *solver* is allowed to know: in curated mode
//! its candidates are exactly those 2,315 words; in open mode every one of
//! the 12,972 allowed words is a candidate, weighted by the frequency prior
//! (or not, with `--uniform`).
//!
//! This is the benchmark. Judge every change to a strategy by this table,
//! not by how a few hand-played games felt.

use std::process::ExitCode;
use std::time::Instant;

use rayon::prelude::*;
use wordlesolver::strategy::{self, MAX_TURNS, Strategy};
use wordlesolver::words::{BUNDLED_ANSWERS, parse_list};
use wordlesolver::{Context, Solver, WordId};

struct Args {
    strategies: Vec<Box<dyn Strategy>>,
    limit: Option<usize>,
    open: bool,
    uniform: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut strategies = None;
    let mut limit = None;
    let mut open = false;
    let mut uniform = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--strategy" | "-s" => {
                let name = args.next().ok_or("--strategy needs a name")?;
                strategies = Some(if name == "all" {
                    strategy::all()
                } else {
                    vec![strategy::by_name(&name).ok_or_else(|| {
                        format!("unknown strategy {name:?}; try: {}", known_names())
                    })?]
                });
            }
            "--limit" | "-n" => {
                let n = args.next().ok_or("--limit needs a number")?;
                limit = Some(n.parse().map_err(|_| format!("bad --limit {n:?}"))?);
            }
            "--open" => open = true,
            "--uniform" => uniform = true,
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unexpected argument {other:?}\n{USAGE}")),
        }
    }
    if uniform && !open {
        return Err(
            "--uniform only makes sense with --open (curated mode is already uniform)".into(),
        );
    }
    Ok(Args {
        strategies: strategies.unwrap_or_else(strategy::all),
        limit,
        open,
        uniform,
    })
}

const USAGE: &str = "usage: evaluate [--strategy <name|all>] [--limit <n>] [--open [--uniform]]";

fn known_names() -> String {
    strategy::all()
        .iter()
        .map(|s| s.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Guess counts for every answer, bucketed 1..=6 plus "failed".
struct Report {
    name: &'static str,
    games: usize,
    total_guesses: usize,
    by_turns: [usize; MAX_TURNS + 1],
    failures: Vec<(WordId, usize)>,
    seconds: f64,
}

impl Report {
    fn average(&self) -> f64 {
        self.total_guesses as f64 / self.games as f64
    }
}

fn evaluate(ctx: &Context, strategy: &dyn Strategy, answers: &[WordId]) -> Report {
    let solver = Solver::new(ctx, strategy);
    let start = Instant::now();

    // Outer loop over answers in parallel; each `solve` runs its own
    // parallel guess scan inside. Rayon nests these on one pool.
    let turns: Vec<usize> = answers
        .par_iter()
        .map(|&answer| solver.solve(answer).len())
        .collect();

    let mut report = Report {
        name: strategy.name(),
        games: answers.len(),
        total_guesses: 0,
        by_turns: [0; MAX_TURNS + 1],
        failures: Vec::new(),
        seconds: start.elapsed().as_secs_f64(),
    };
    for (&answer, &t) in answers.iter().zip(&turns) {
        report.total_guesses += t;
        report.by_turns[t] += 1;
        if t > 6 {
            report.failures.push((answer, t));
        }
    }
    report
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let t = Instant::now();
    let ctx = match (args.open, args.uniform) {
        (false, _) => Context::curated(),
        (true, false) => Context::open(),
        (true, true) => Context::open_uniform(),
    };
    let mode = match (args.open, args.uniform) {
        (false, _) => "curated: solver knows the 2,315 answers",
        (true, false) => "open: every word is a candidate, frequency prior",
        (true, true) => "open: every word is a candidate, uniform prior",
    };
    eprintln!(
        "{mode}\n{} candidates, {} allowed guesses, table built in {:.0} ms",
        ctx.num_answers(),
        ctx.num_guesses(),
        t.elapsed().as_secs_f64() * 1000.0
    );

    // The benchmark answers are always the real list. In curated mode they
    // are the candidates; in open mode they are the first 2,315 ids.
    let real_answers: Vec<WordId> = parse_list(BUNDLED_ANSWERS)
        .expect("bundled answers")
        .iter()
        .map(|w| ctx.find(w).expect("every answer is an allowed guess"))
        .collect();
    if args.open {
        let on_answers: u64 = real_answers.iter().map(|&a| ctx.weight(a) as u64).sum();
        eprintln!(
            "{:.1}% of the prior mass sits on the real answers",
            on_answers as f64 / ctx.total_weight as f64 * 100.0
        );
    }
    eprintln!();

    let answers = match args.limit {
        Some(n) => &real_answers[..n.min(real_answers.len())],
        None => &real_answers[..],
    };

    println!(
        "{:<9} {:>6}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {:>7}",
        "strategy", "avg", "1", "2", "3", "4", "5", "6", "fail", "time"
    );
    let mut reports = Vec::new();
    for strategy in &args.strategies {
        let r = evaluate(&ctx, strategy.as_ref(), answers);
        let failed: usize = r.by_turns[7..].iter().sum();
        println!(
            "{:<9} {:>6.3}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {:>6.2}s",
            r.name,
            r.average(),
            r.by_turns[1],
            r.by_turns[2],
            r.by_turns[3],
            r.by_turns[4],
            r.by_turns[5],
            r.by_turns[6],
            failed,
            r.seconds
        );
        reports.push(r);
    }

    for r in &reports {
        if !r.failures.is_empty() {
            let words: Vec<String> = r
                .failures
                .iter()
                .map(|&(w, t)| format!("{}({t})", ctx.word_str(w)))
                .collect();
            println!("\n{} failed on: {}", r.name, words.join(" "));
        }
    }
    ExitCode::SUCCESS
}
