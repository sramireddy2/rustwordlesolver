# wordlesolver

An entropy-driven Wordle solver in Rust.

## Run

```
cargo run --release --bin play                # interactive: it suggests, you type the tiles back
cargo run --release --bin evaluate            # play every answer, report average guesses per strategy
cargo run --release --example table_timing    # how long the 30 MB pattern table takes to build
```

## Test

```
cargo test
```

## How it works

Everything reduces to small integers:

- a **word** is a `WordId(u16)` into one list of all 12,972 allowed guesses,
  ordered so the 2,315 possible answers come first;
- the tiles Wordle shows are a **`Pattern(u8)`**, base-3 packed (3⁵ = 243);
- a 30 MB **table** holds the pattern for every (guess, answer) pair, built once
  at startup (~100 ms in release), so scoring is a byte load;
- the answers still possible are a **`CandidateSet`**: a 37-word `u64` bitset,
  `Copy`, no heap;
- a **`Strategy`** turns a candidate set into the next guess. Entropy maximises
  expected information, minimax bounds the worst case, hybrid does entropy with
  minimax as the tie-breaker.

## Layout

```
src/
├── lib.rs           exports
├── types.rs         WordId, Pattern, and Pattern::score (the tile rules)
├── words.rs         parsing the word lists
├── table.rs         Context: dictionaries + pattern table + log tables
├── state.rs         CandidateSet bitset, Game history
├── strategy/
│   ├── mod.rs       Strategy trait, shared histogram/entropy/minimax maths, Solver
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

See `PLAN.md` for the step-by-step design.

## Toolchain note

This machine has Visual Studio 2022 without the C++ build tools, so Rust is
installed with the **GNU** host toolchain (`x86_64-pc-windows-gnu`), which
brings its own linker. If you ever need MSVC, install the "Desktop development
with C++" workload in the Visual Studio Installer first, then
`rustup default stable-x86_64-pc-windows-msvc`.
