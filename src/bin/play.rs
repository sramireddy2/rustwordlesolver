//! Interactive solver: it suggests a word, you type back the tiles.
//!
//!     cargo run --release --bin play                  # hybrid strategy
//!     cargo run --release --bin play -- --strategy entropy
//!
//! At the prompt, type the tiles you got (`g` green, `y` yellow, `b` gray),
//! or `<word> <tiles>` if you played something other than the suggestion.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::time::Instant;

use wordlesolver::strategy::{self, Strategy};
use wordlesolver::words::parse_word;
use wordlesolver::{Context, Game, Pattern, Solver, WordId};

const TURNS: usize = 6;
/// Show the remaining candidates once the list is short enough to read.
const SHOW_CANDIDATES_AT: usize = 10;

fn parse_args() -> Result<Box<dyn Strategy>, String> {
    let mut strategy: Option<Box<dyn Strategy>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--strategy" | "-s" => {
                let name = args.next().ok_or("--strategy needs a name")?;
                strategy = Some(strategy::by_name(&name).ok_or_else(|| {
                    let names: Vec<_> = strategy::all().iter().map(|s| s.name()).collect();
                    format!("unknown strategy {name:?}; try: {}", names.join(", "))
                })?);
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unexpected argument {other:?}\n{USAGE}")),
        }
    }
    Ok(strategy.unwrap_or_else(|| Box::new(strategy::hybrid::Hybrid)))
}

const USAGE: &str = "usage: play [--strategy <name>]";

enum Input {
    Quit,
    /// The word actually played (defaults to the suggestion) and its tiles.
    Turn(WordId, Pattern),
}

/// `bygbb`, or `crane bygbb`, or `q`.
fn parse_input(line: &str, suggested: WordId, ctx: &Context) -> Result<Input, String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    match parts.as_slice() {
        ["q" | "quit" | "exit"] => Ok(Input::Quit),
        [tiles] => Ok(Input::Turn(suggested, Pattern::parse(tiles)?)),
        [word, tiles] => {
            let w = parse_word(word)?;
            let id = ctx
                .find(&w)
                .ok_or_else(|| format!("{word:?} is not in the allowed-guess list"))?;
            Ok(Input::Turn(id, Pattern::parse(tiles)?))
        }
        [] => Err("type the tiles you got, e.g. bygbb".into()),
        _ => Err("expected `<tiles>` or `<word> <tiles>`".into()),
    }
}

fn main() -> ExitCode {
    let strategy = match parse_args() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let t = Instant::now();
    let ctx = Context::bundled();
    println!(
        "wordlesolver — {} answers, {} allowed guesses, ready in {:.0} ms",
        ctx.num_answers(),
        ctx.num_guesses(),
        t.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "Strategy: {}. Tiles: g = green, y = yellow, b = gray. q to quit.\n",
        strategy.name()
    );

    let solver = Solver::new(&ctx, strategy.as_ref());
    let mut game = Game::new(&ctx);
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while game.turn() <= TURNS {
        let cands = game.candidates();
        if cands.is_empty() {
            println!("No answers are consistent with those tiles.");
            println!("Either a pattern was mistyped, or the answer isn't in the word list.");
            return ExitCode::FAILURE;
        }

        let suggested = solver.next_guess(cands);
        println!(
            "Turn {}: play {}   ({} candidate{})",
            game.turn(),
            ctx.word_str(suggested).to_uppercase(),
            cands.len(),
            if cands.len() == 1 { "" } else { "s" }
        );
        if cands.len() <= SHOW_CANDIDATES_AT && cands.len() > 1 {
            let list: Vec<&str> = cands.iter().map(|id| ctx.word_str(id)).collect();
            println!("        could be: {}", list.join(" "));
        }

        let (played, tiles) = loop {
            print!("> ");
            io::stdout().flush().ok();
            let Some(Ok(line)) = lines.next() else {
                println!();
                return ExitCode::SUCCESS;
            };
            match parse_input(&line, suggested, &ctx) {
                Ok(Input::Quit) => return ExitCode::SUCCESS,
                Ok(Input::Turn(w, p)) => break (w, p),
                Err(e) => println!("  {e}"),
            }
        };

        if tiles == Pattern::WIN {
            println!("\nSolved in {}.", game.turn());
            return ExitCode::SUCCESS;
        }
        game.observe(played, tiles);
    }

    println!("\nOut of turns.");
    let left: Vec<&str> = game
        .candidates()
        .iter()
        .map(|id| ctx.word_str(id))
        .collect();
    if !left.is_empty() {
        println!("It was one of: {}", left.join(" "));
    }
    ExitCode::SUCCESS
}
