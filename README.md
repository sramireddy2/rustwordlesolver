# wordlesolver

An entropy-driven Wordle solver in Rust. Solves every one of the 2,315
official answers within six guesses, averaging 3.46.

## Run

```
cargo run --release --bin play                # interactive: it suggests, you type the tiles back
cargo run --release --bin evaluate            # play every answer, report average guesses per strategy
cargo run --release --example table_timing    # how long the 30 MB pattern table takes to build
```

`play` prompts with a word; answer with the tiles you got (`g` green, `y`
yellow, `b` gray), or `<word> <tiles>` if you played something else.

```
Turn 1: play SOARE   (2315 candidates)
> bbgyg
Turn 2: play TRACK   (14 candidates)
```

## Results

All 2,315 answers, `cargo run --release --bin evaluate`:

| strategy | avg guesses | 1 | 2 | 3 | 4 | 5 | 6 | fail | opener |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entropy | 3.526 | 0 | 31 | 1105 | 1111 | 67 | 1 | 0 | soare |
| minimax | 3.567 | 1 | 53 | 1001 | 1154 | 105 | 1 | 0 | arise |
| hybrid  | 3.463 | 0 | 44 | 1219 |  988 | 63 | 1 | 0 | soare |

`soare` is worth 5.886 bits as an opener on this list — the same figure
3Blue1Brown derived. A full run takes 2–3 s per strategy on a laptop.

Time the strategies in *separate* runs when comparing speed: three in a row
in one process heats the CPU and the later ones throttle.

## Test

```
cargo test
```

Unit tests use tiny hand-built word lists; `tests/strategies.rs` runs every
strategy against the real lists on a sample of answers.

## How it works

Everything reduces to small integers:

- a **word** is a `WordId(u16)` into one list of all 12,972 allowed guesses,
  ordered so the 2,315 possible answers come first;
- the tiles Wordle shows are a **`Pattern(u8)`**, base-3 packed (3⁵ = 243);
- a 30 MB **table** holds the pattern for every (guess, answer) pair, built once
  at startup (~75 ms in release), so scoring is a byte load;
- the answers still possible are a **`CandidateSet`**: a 37-word `u64` bitset,
  `Copy`, no heap;
- a **`Strategy`** turns a candidate set into the next guess. For each of the
  12,972 guesses it sorts the candidates into 243 buckets by the tiles that
  guess would produce, then scores the bucket shape. Entropy maximises
  expected information (`log2 n − Σ c·log2 c / n`, via a lookup table so the
  loop has no logarithms); minimax minimises the largest bucket; hybrid does
  entropy with minimax as the tie-breaker and prefers a guess that could be
  the answer when everything else is equal. Scoring runs across all cores
  with `rayon`; ties break to the lower `WordId` so results are reproducible.

## Layout

```
src/
├── lib.rs           exports
├── types.rs         WordId, Pattern, and Pattern::score (the tile rules)
├── words.rs         parsing the word lists
├── table.rs         Context: dictionaries + pattern table + log tables
├── state.rs         CandidateSet bitset, Game history
├── strategy/
│   ├── mod.rs       Strategy trait, histogram/entropy/minimax maths, Solver
│   ├── entropy.rs
│   ├── minimax.rs
│   └── hybrid.rs
└── bin/
    ├── play.rs      interactive REPL
    └── evaluate.rs  batch simulation over every answer
data/
├── answers.txt      2,315 possible answers
└── guesses.txt      10,657 extra allowed guesses
```

`PLAN.md` is the original step-by-step design; the module layout above
supersedes its file names.

## Toolchain note

This machine has Visual Studio 2022 without the C++ build tools, so Rust is
installed with the **GNU** host toolchain (`x86_64-pc-windows-gnu`), which
brings its own linker. If you ever need MSVC, install the "Desktop development
with C++" workload in the Visual Studio Installer first, then
`rustup default stable-x86_64-pc-windows-msvc`.
