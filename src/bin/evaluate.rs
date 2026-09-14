//! Play every answer against each strategy and report how they did.
//!
//!     cargo run --release --bin evaluate                 # all strategies, all answers
//!     cargo run --release --bin evaluate -- --strategy hybrid --limit 200
//!
//! This is the benchmark. Judge every change to a strategy by this table,
//! not by how a few hand-played games felt.

use std::process::ExitCode;
use std::time::Instant;

use rayon::prelude::*;
use wordlesolver::strategy::{self, MAX_TURNS, Strategy};
use wordlesolver::{Context, Solver, WordId};

struct Args {
    strategies: Vec<Box<dyn Strategy>>,
    limit: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut strategies = None;
    let mut limit = None;
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
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unexpected argument {other:?}\n{USAGE}")),
        }
    }
    Ok(Args {
        strategies: strategies.unwrap_or_else(strategy::all),
        limit,
    })
}

const USAGE: &str = "usage: evaluate [--strategy <name|all>] [--limit <n>]";

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
    let ctx = Context::bundled();
    eprintln!(
        "{} answers, {} allowed guesses, table built in {:.0} ms\n",
        ctx.num_answers(),
        ctx.num_guesses(),
        t.elapsed().as_secs_f64() * 1000.0
    );

    let answers = match args.limit {
        Some(n) => &ctx.answers[..n.min(ctx.answers.len())],
        None => &ctx.answers[..],
    };

    println!(
        "{:<8} {:>6}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {:>7}",
        "strategy", "avg", "1", "2", "3", "4", "5", "6", "fail", "time"
    );
    let mut reports = Vec::new();
    for strategy in &args.strategies {
        let r = evaluate(&ctx, strategy.as_ref(), answers);
        let failed: usize = r.by_turns[7..].iter().sum();
        println!(
            "{:<8} {:>6.3}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {:>6.2}s",
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
