use std::time::Instant;
use wordlesolver::Context;

fn main() {
    let t = Instant::now();
    let ctx = Context::curated();
    let built = t.elapsed();
    println!(
        "{} answers x {} guesses = {} cells ({:.1} MB) built in {:?}",
        ctx.num_answers(),
        ctx.num_guesses(),
        ctx.table.len(),
        ctx.table.len() as f64 / 1e6,
        built
    );
}
