# wordlesolver

An entropy-driven Wordle solver in Rust. Given the official answer list it
solves every one of the 2,315 answers within six guesses, averaging 3.46;
without that list, using only word frequency, it averages 3.64.

## Run

```
cargo run --release --bin play                # interactive: it suggests, you type the tiles back
cargo run --release --bin play -- --open      # ...without assuming the curated answer list
cargo run --release --bin evaluate            # play every answer, report average guesses per strategy
cargo run --release --bin evaluate -- --open  # ...with the solver not knowing the answer list
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
| lookahead | 3.461 | 0 | 49 | 1218 | 979 | 69 | 0 | 0 | soare |

`soare` is worth 5.886 bits as an opener on this list — the same figure
3Blue1Brown derived. A full run takes 2–3 s per strategy on a laptop.

`lookahead` is hybrid until 16 candidates remain, then an exact
branch-and-bound search for the guess minimising expected guesses to
finish. It is provably optimal from there, never needs a sixth guess, and is
*faster* than hybrid in the endgame (its bound is O(n) per guess, hybrid's
reduction is O(243)). Raising the threshold is not worth it: at 24 the
search takes 200× longer for 0.001 guesses, because the `2 − 1/m` bound is
only tight when a perfectly separating guess exists, which stops being true
in the teens. `--strategy lookahead:N` sweeps it if you want to see for
yourself. The remaining gap to the ~3.42 optimum lives in turns 1–2.

Time the strategies in *separate* runs when comparing speed: three in a row
in one process heats the CPU and the later ones throttle.

### Open mode: the solver doesn't get the answer list

The curated list is insider information. `--open` makes every one of the
12,972 allowed words a candidate, weighted by a prior on how common the
word is (Google n-gram counts, `data/frequencies.txt`, shaped by a logistic
curve over frequency rank — see `src/prior.rs`). The benchmark still scores
against the real 2,315 answers.

| mode | prior | strategy | avg | 6 | fail |
| --- | --- | --- | --- | --- | --- |
| curated | — | hybrid | 3.463 | 1 | 0 |
| open | uniform | hybrid | 3.852 | 17 | 0 |
| open | frequency | hybrid | 3.789 | 4 | 0 |
| open | frequency | lookahead | **3.640** | 21 | 0 |

Knowing the list is worth ~0.4 guesses; the prior recovers a sixth of that
for hybrid and, combined with lookahead, half of it. The prior puts 48.9%
of its mass on the real answers — they weigh about as much as the other
10,657 words together.

Lookahead's objective is a pair, `(probability of failing, expected
guesses)`, compared in that order, so it never trades a loss for speed and
there is no penalty constant to tune. Without the first component it
scored 3.637 but lost `vaunt` and `woozy` in seven — under the prior those
weigh 1 in 875,532, and sacrificing them to save fractions on common words
is exactly what plain expected-guess minimisation should do. The cap costs
0.003 guesses. `Solver` also plays the likeliest candidate on the last
turn whatever the strategy says, since anything else is a certain loss.

One caveat remains: the prior's centre and width were read off the answer
list the benchmark uses (two parameters on a smooth curve, so mild, but
real leakage).

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
- each candidate has an integer **prior weight** (all 1 in curated mode), so
  histograms, entropy and the lookahead recurrence all run over probability
  mass while the inner loop stays integer;
- the answers still possible are a **`CandidateSet`**: a 203-word `u64`
  bitset (1.6 KB), `Copy`, no heap;
- a **`Strategy`** turns a candidate set into the next guess. For each of the
  12,972 guesses it sorts the candidates into 243 buckets by the tiles that
  guess would produce, then scores the bucket shape. Entropy maximises
  expected information (`log2 n − Σ c·log2 c / n`, via a lookup table so the
  loop has no logarithms); minimax minimises the largest bucket; hybrid does
  entropy with minimax as the tie-breaker and prefers a guess that could be
  the answer when everything else is equal; lookahead switches to an exact
  search once few candidates remain, minimising failure probability first
  and expected guesses second within the six-guess cap. Scoring runs across
  all cores with `rayon`; ties break to the lower `WordId` so results are
  reproducible.

## Layout

```
src/
├── lib.rs           exports
├── types.rs         WordId, Pattern, and Pattern::score (the tile rules)
├── words.rs         parsing the word and frequency lists
├── prior.rs         frequency rank -> integer prior weight
├── table.rs         Context: dictionaries + prior + pattern table + log tables
├── state.rs         CandidateSet bitset, Game history
├── strategy/
│   ├── mod.rs       Strategy trait, histogram/entropy/minimax maths, Solver
│   ├── entropy.rs
│   ├── minimax.rs
│   ├── hybrid.rs
│   └── lookahead.rs
└── bin/
    ├── play.rs      interactive REPL
    └── evaluate.rs  batch simulation over every answer
data/
├── answers.txt      2,315 possible answers
├── guesses.txt      10,657 extra allowed guesses
└── frequencies.txt  n-gram counts for the 8,092 of those words in the corpus
```

`PLAN.md` is the original step-by-step design; the module layout above
supersedes its file names.

## Toolchain note

This machine has Visual Studio 2022 without the C++ build tools, so Rust is
installed with the **GNU** host toolchain (`x86_64-pc-windows-gnu`), which
brings its own linker. If you ever need MSVC, install the "Desktop development
with C++" workload in the Visual Studio Installer first, then
`rustup default stable-x86_64-pc-windows-msvc`.
