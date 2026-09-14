# Wordle solver — implementation plan

Goal: an entropy-driven solver over the official word lists that plays all
~2,300 answers in well under a minute and averages ~3.6 guesses (the greedy
entropy ceiling; true optimum is ~3.42).

Each step ends with something you can test. Do them in order — every step
leans on the representation chosen in the one before.

---

## 0. Decide the representations first

These decisions drive everything else. Get them down before writing code.

| Thing | Representation | Why |
| --- | --- | --- |
| Word | `[u8; 5]` (`Copy`) | 5 bytes, no heap, cheap to compare/hash |
| Feedback pattern | `u8`, base-3 (gray=0, yellow=1, green=2), `pattern = Σ color[i] * 3^i` | 3⁵ = 243 patterns fit in a byte, and a `[T; 243]` array replaces a `HashMap` as the histogram |
| Word identity | `u16` index into the word list | 2 bytes per entry, direct indexing into precomputed tables |
| Candidate set | `Vec<u16>` of answer indices | small, contiguous, `retain` is one pass |
| Pattern table | `Vec<u8>` of `guesses.len() × answers.len()`, `table[g * n_answers + a]` | scoring becomes one byte lookup; ~30 MB (12,972 × 2,309) |

Two lists, kept separate:
- **answers** (~2,309): what the puzzle can be. This is the candidate pool.
- **guesses** (~12,972, superset of answers): what you may type. This is the
  guess pool. The best splitter is frequently not a possible answer.

---

## 1. Word lists

- [ ] Get the official lists (answers + allowed guesses) into `data/`. Two files.
- [ ] `include_str!` both, parse to `Vec<[u8; 5]>`. Reject anything not exactly
      five ASCII lowercase letters.
- [ ] Order the guess list so answers come first (indices `0..n_answers`), then
      the rest. Now "is this guess a possible answer?" is `idx < n_answers`.
- [ ] Tests: counts are what you expect, no duplicates, every answer appears in
      the guess list at the same index.

---

## 2. Feedback scoring → `u8`

`fn score(guess: &[u8;5], answer: &[u8;5]) -> u8`

Two passes, because letters are consumed:
1. Mark greens. For every non-green answer position, bump a `[u8; 26]` count of
   unclaimed letters.
2. For every non-green guess position, if its letter still has count > 0,
   decrement and mark yellow; else gray.

Encode to base-3 as you go. Also write `decode(u8) -> [Color; 5]` for display
and `parse("gybbb") -> u8` for user input.

- [ ] Tests — the duplicate-letter cases are where solvers go wrong:
  - `llama` vs `lucky` → only the first `l` scores
  - `bobby` vs `abbey` → the green `b` is claimed before any yellow `b`
  - `speed` vs `abide` → one `e` yellow, one gray
  - all-gray, all-green, `parse(decode(p)) == p` for all 243 `p`

---

## 3. Pattern table

`struct PatternTable { n_answers: usize, data: Vec<u8> }`
`fn get(&self, guess: u16, answer: u16) -> u8`

- [ ] Build it at startup: 30M calls to `score`. Time it in `--release`; expect
      well under a second. If you want it instant, cache to a file keyed by a
      hash of the word lists, or generate it in `build.rs`.
- [ ] Tests: `get(g, a) == score(words[g], words[a])` for a sample; `get(a, a)`
      is all-green (242) for every answer.

---

## 4. Candidate filtering

`fn filter(cands: &mut Vec<u16>, guess: u16, observed: u8, table: &PatternTable)`
→ `cands.retain(|&a| table.get(guess, a) == observed)`

This is the whole constraint system. You never write "letter x must be at
position 2" logic — a candidate survives iff it would have produced the same
tiles. Equivalent, and impossible to get subtly wrong.

- [ ] Test (property-style): for many random answers and random guess
      sequences, the true answer is never removed.

---

## 5. Shannon entropy

For a guess `g` against candidates `C` (|C| = N):

```
counts[243] = 0
for a in C: counts[table.get(g, a)] += 1
H(g) = log2(N) − (1/N) · Σ_b  counts[b] · log2(counts[b])
```

The rearranged form matters: precompute `NLOG2N[n] = n·log2(n)` for
`n in 0..=n_answers` once, and the inner loop is pure integer histogramming
plus 243 table lookups. No `f64::log2` calls per candidate.

`fn entropy(g: u16, cands: &[u16], table: &PatternTable, nlog2n: &[f32]) -> f32`
`fn best_guess(cands, table, ...) -> u16` — scan the **full guess pool**, take
the max.

Cost per turn: `|guesses| × |cands|` byte lookups. Turn 1 is ~30M (~tens of
ms in release); later turns are tiny.

- [ ] Tests:
  - a guess sharing no letters with any candidate → H = 0
  - a guess giving every candidate a distinct pattern → H = log2(N)
  - H is never negative and never exceeds log2(N)
  - the opener (turn 1) is stable and lands near known-good words
    (`soare`, `tares`, `crane`, `slate`… — exact ranking depends on your list)
- [ ] The opener never changes for a given list. Compute once and cache it
      (a `const`, or a lazily-initialised static) so `cargo run` starts instantly.

---

## 6. Minimax / worst-case

Same histogram, different reduction:

```
worst(g) = max_b counts[b]
```

Entropy maximises *average* information; minimax bounds the *worst* branch.
Neither dominates, so make the choice explicit:

```rust
enum Strategy { Entropy, Minimax, Hybrid }
```

Hybrid, which is what you'll probably ship:
1. Maximise entropy.
2. Break ties by minimum worst-case bucket.
3. Break remaining ties by preferring a guess that is itself a candidate — it
   has a `1/N` chance of ending the game now, which pure entropy is blind to.

Endgame rule: if `N <= 2`, just guess a candidate. There is nothing to learn.

- [ ] Tests: `worst(g) == N` for a useless guess, `== 1` for a perfect split;
      hybrid prefers the candidate word among entropy-tied guesses.

---

## 7. Game loop, CLI, and a simulator

Two modes:

- **`play`** — interactive. Suggest, read `<word> <pattern>`, filter, repeat.
  Handle contradictory input (empty candidates) with a clear message.
- **`simulate [--strategy ...]`** — play every answer in the list against the
  solver using `score` as the oracle. Report average guesses, the 1–6
  distribution, and any failures (>6). **This is your benchmark**; every
  strategy change gets judged by it, not by feel.

- [ ] Test: `simulate` solves 100% of answers within 6 with the hybrid strategy.

---

## 8. Performance pass (only after 7 works)

Measure with `simulate` in `--release`. Then, in order of payoff:

1. **Cached opener** (step 5) — removes the single largest cost.
2. **Skip hopeless guesses**: any guess whose histogram already has a bucket
   > current best worst-case can be abandoned early when running Minimax.
3. **Parallel guess scan** with `rayon` (`par_iter` over the guess pool).
   Drop-in; the histogram is per-guess so nothing is shared.
4. **Candidate bitset** (`[u64; 37]` for 2,309 answers) instead of `Vec<u16>`
   if you add lookahead — set intersection becomes a few ANDs.

Stretch goals, in rough order of value:
- **Two-ply lookahead** when `N` is small (≤ ~30): pick the guess minimising
  expected *total* guesses rather than maximising one-step entropy. Closes most
  of the gap from ~3.6 to ~3.42.
- **Hard mode**: restrict the guess pool to words consistent with feedback.
- **Answer priors**: weight candidates by word frequency instead of uniform.

---

## Memory budget (sanity check)

| Structure | Size |
| --- | --- |
| Guess list, `[u8;5]` × 12,972 | ~65 KB |
| Pattern table, `u8` × 12,972 × 2,309 | ~30 MB |
| `NLOG2N`, `f32` × 2,310 | ~9 KB |
| Candidates, `u16` × ≤2,309 | ≤ 5 KB |
| Histogram, `[u16; 243]` | 486 B, on the stack |

The table is the only thing that isn't negligible, and it's what makes every
scoring step a byte load instead of a function call.
