use crate::core::{idm_scalar_ref, erd_round_scalar_ref};

// ==========================================
// PUBLIC API
// ==========================================

// PRNG STREAM STEP (1x IDM + 1x ERD, counter-driven)
// Call sequentially inside the PRNG loop.
#[inline]
pub(crate) fn stream_step(
    s0: u128,
    s1: u128,
    step: u64,
) -> (u128, u128) {
    let (s0, s1) = idm_scalar_ref(s0, s1);
    erd_round_scalar_ref(s0, s1, step)
}

// STANDALONE HASH PIPELINE (1x IDM + 1x ERD, step=0)
// Stateless — no counter evolution.
#[inline]
pub(crate) fn evaluate_scalar(
    s0_in: u128,
    s1_in: u128,
) -> (u128, u128) {
    let (s0, s1) = idm_scalar_ref(s0_in, s1_in);
    erd_round_scalar_ref(s0, s1, 0)
}