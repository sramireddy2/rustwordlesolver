//! How to choose the next guess.
//!
//! Every strategy sees the same two things: the immutable [`Context`] and the
//! set of answers still possible. It returns a [`WordId`] from the *full*
//! guess pool — the best splitter is often a word that cannot be the answer.

use crate::state::CandidateSet;
use crate::table::Context;
use crate::types::WordId;

pub trait Strategy: Sync {
    /// Short identifier, for CLI flags and report tables.
    fn name(&self) -> &'static str;

    /// The next word to play. `candidates` is never empty when this is called.
    fn best_guess(&self, ctx: &Context, candidates: &CandidateSet) -> WordId;
}
