//! An entropy-driven Wordle solver.
//!
//! Everything in the engine reduces to small integers: a word is a [`WordId`]
//! into [`Context::allowed_guesses`], and the tiles Wordle shows for a guess
//! are a [`Pattern`] (one byte, base-3). The [`Context`] precomputes the
//! pattern for every (guess, answer) pair once, so scoring anywhere else is a
//! byte load. A [`CandidateSet`] is a fixed bitset over the answer list, and a
//! [`Strategy`] turns a candidate set into the next guess.
//!
//! [`WordId`]: types::WordId
//! [`Pattern`]: types::Pattern
//! [`Context`]: table::Context
//! [`Context::allowed_guesses`]: table::Context::allowed_guesses
//! [`CandidateSet`]: state::CandidateSet
//! [`Strategy`]: strategy::Strategy

pub mod state;
pub mod strategy;
pub mod table;
pub mod types;
pub mod words;

pub use state::{CandidateSet, Game};
pub use strategy::Strategy;
pub use table::Context;
pub use types::{Pattern, Word, WordId};
