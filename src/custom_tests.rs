// ==============================================
// Lustro Premiere Custom Tests Selection
// ==============================================

use rayon::prelude::*;
use rayon::slice::ParallelSlice;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use numpy::PyArray2;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray, PyUntypedArrayMethods};

use crate::dispatch::{self, LustroError};

// ==============================================
// HELPERS
// ==============================================

// Count set bits using trailing-zero iteration.
fn count_bits_128(acc: &mut [u64], lo: u64, hi: u64) {
    let mut x = lo;
    while x != 0 {
        let b = x.trailing_zeros() as usize;
        acc[b] += 1;
        x &= x - 1;
    }
    let mut x = hi;
    while x != 0 {
        let b = x.trailing_zeros() as usize;
        acc[64 + b] += 1;
        x &= x - 1;
    }
}

// Build a one-hot mask in the corresponding state word.
#[inline(always)]
fn b16_bit_mask(bit: usize) -> (u64, u64, u64, u64) {
    match bit {
        0..=63    => (1u64 << bit,           0, 0, 0),
        64..=127  => (0, 1u64 << (bit - 64),  0, 0),
        128..=191 => (0, 0, 1u64 << (bit - 128), 0),
        _         => (0, 0, 0, 1u64 << (bit - 192)),
    }
}

// Accumulate set differential bits into the input-bit row.
fn b16_accumulate_jacobian(
    jacobian:  &mut [u64],
    input_bit: usize,
    d_w0: u64,
    d_w1: u64,
    d_w2: u64,
    d_w3: u64,
) {
    let row = input_bit * 256;

    let mut x = d_w0;
    while x != 0 {
        let b = x.trailing_zeros() as usize;
        jacobian[row + b] += 1;
        x &= x - 1;
    }
    let mut x = d_w1;
    while x != 0 {
        let b = x.trailing_zeros() as usize;
        jacobian[row + 64 + b] += 1;
        x &= x - 1;
    }
    let mut x = d_w2;
    while x != 0 {
        let b = x.trailing_zeros() as usize;
        jacobian[row + 128 + b] += 1;
        x &= x - 1;
    }
    let mut x = d_w3;
    while x != 0 {
        let b = x.trailing_zeros() as usize;
        jacobian[row + 192 + b] += 1;
        x &= x - 1;
    }
}

// Rotate a 128-bit value represented as (lo, hi).
fn rotl128(lo: u64, hi: u64, rot: u32) -> (u64, u64) {
    let rot = rot % 128;
    if rot == 0 {
        return (lo, hi);
    }
    if rot < 64 {
        let new_lo = (lo << rot) | (hi >> (64 - rot));
        let new_hi = (hi << rot) | (lo >> (64 - rot));
        (new_lo, new_hi)
    } else {
        let r = rot - 64;
        if r == 0 {
            // Half-width rotation reduces to a word swap.
            return (hi, lo);
        }
        let new_lo = (hi << r) | (lo >> (64 - r));
        let new_hi = (lo << r) | (hi >> (64 - r));
        (new_lo, new_hi)
    }
}

// Rotate a 256-bit value using word and intra-word shifts.
fn rotl256(w0: u64, w1: u64, w2: u64, w3: u64, rot: u32) -> (u64, u64, u64, u64) {
    let rot = (rot % 256) as usize;
    if rot == 0 {
        return (w0, w1, w2, w3);
    }
    let words = [w0, w1, w2, w3];
    let word_shift = rot / 64;
    let bit_shift  = rot % 64;

    let mut out = [0u64; 4];
    if bit_shift == 0 {
        for i in 0..4 {
            out[i] = words[(i + 4 - word_shift) % 4];
        }
    } else {
        for i in 0..4 {
            let src      = words[(i + 4 - word_shift) % 4];
            let src_prev = words[(i + 4 - word_shift + 3) % 4];
            out[i] = (src << bit_shift) | (src_prev >> (64 - bit_shift));
        }
    }
    (out[0], out[1], out[2], out[3])
}

// B2 AVALANCHE / SAC (FULL BATCH)
// Reuses pre-evaluated outputs to avoid redundant evaluation.
#[pyfunction]
pub fn b2_evaluate_flipped<'py>(
    py: Python<'py>,
    orig_states: PyReadonlyArray2<'py, u64>,
    eval_states: PyReadonlyArray2<'py, u64>,
    flip_s0_lo: u64,
    flip_s0_hi: u64,
    flip_s1_lo: u64,
    flip_s1_hi: u64,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    if orig_states.shape()[1] != 4 || eval_states.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both inputs must have shape (n, 4)",
        ));
    }
    if orig_states.shape()[0] != eval_states.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "orig_states and eval_states must have the same row count",
        ));
    }
    let orig_arr   = orig_states.as_array();
    let orig_slice = orig_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("orig_states must be C-contiguous")
    })?;
    let eval_arr   = eval_states.as_array();
    let eval_slice = eval_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("eval_states must be C-contiguous")
    })?;

    let (sac_flat, av_hist) = py.allow_threads(|| {
        b2_evaluate_flipped_internal(
            orig_slice, eval_slice,
            flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi,
        )
    });

    Ok((
        sac_flat.to_pyarray_bound(py),
        av_hist.to_pyarray_bound(py),
    ))
}

fn b2_evaluate_flipped_internal(
    orig_slice: &[u64],
    eval_slice: &[u64],
    flip_s0_lo: u64,
    flip_s0_hi: u64,
    flip_s1_lo: u64,
    flip_s1_hi: u64,
) -> (Vec<u64>, Vec<u64>) {
    debug_assert_eq!(orig_slice.len() % 4, 0);
    debug_assert_eq!(orig_slice.len(), eval_slice.len());

    let n = orig_slice.len() / 4;

    let mut flipped_buf = Vec::<u64>::with_capacity(n * 4);

    for i in 0..n {
        flipped_buf.push(orig_slice[i * 4]     ^ flip_s0_lo);
        flipped_buf.push(orig_slice[i * 4 + 1] ^ flip_s0_hi);
        flipped_buf.push(orig_slice[i * 4 + 2] ^ flip_s1_lo);
        flipped_buf.push(orig_slice[i * 4 + 3] ^ flip_s1_hi);
    }

    match crate::dispatch::evaluate(&mut flipped_buf) {
        crate::dispatch::LustroError::Success => {},
        e => panic!("dispatch::evaluate (flip) failed: {:?}", e),
    }

    // Accumulate output-bit flip frequencies and differential Hamming weights.
    let mut sac_flat = vec![0u64; 256];
    let mut av_hist  = vec![0u64; 257];

    for i in 0..n {
        let d_w0 = eval_slice[i * 4]     ^ flipped_buf[i * 4];
        let d_w1 = eval_slice[i * 4 + 1] ^ flipped_buf[i * 4 + 1];
        let d_w2 = eval_slice[i * 4 + 2] ^ flipped_buf[i * 4 + 2];
        let d_w3 = eval_slice[i * 4 + 3] ^ flipped_buf[i * 4 + 3];

        let hw = (d_w0.count_ones()
                + d_w1.count_ones()
                + d_w2.count_ones()
                + d_w3.count_ones()) as usize;
        debug_assert!(hw <= 256);
        av_hist[hw] += 1;

        count_bits_128(&mut sac_flat[0..128],   d_w0, d_w1);
        count_bits_128(&mut sac_flat[128..256], d_w2, d_w3);
    }

    (sac_flat, av_hist)
}


// B2 LANE-SEPARATED SAC
//
// Measure SAC separately for each input/output lane pair.
#[pyfunction]
pub fn b2_evaluate_flipped_lanes<'py>(
    py: Python<'py>,
    orig_states: PyReadonlyArray2<'py, u64>,
    eval_states: PyReadonlyArray2<'py, u64>,
    flip_s0_lo: u64,
    flip_s0_hi: u64,
    flip_s1_lo: u64,
    flip_s1_hi: u64,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    if orig_states.shape()[1] != 4 || eval_states.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both inputs must have shape (n, 4)",
        ));
    }
    if orig_states.shape()[0] != eval_states.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "orig_states and eval_states must have the same row count",
        ));
    }
    let orig_arr   = orig_states.as_array();
    let orig_slice = orig_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("orig_states must be C-contiguous")
    })?;
    let eval_arr   = eval_states.as_array();
    let eval_slice = eval_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("eval_states must be C-contiguous")
    })?;

    let (sac_s0o0, sac_s0o1, sac_s1o0, sac_s1o1, sac_full, av_hist) = py.allow_threads(|| {
        b2_evaluate_flipped_lanes_internal(
            orig_slice, eval_slice,
            flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi,
        )
    });

    Ok((
        sac_s0o0.to_pyarray_bound(py),
        sac_s0o1.to_pyarray_bound(py),
        sac_s1o0.to_pyarray_bound(py),
        sac_s1o1.to_pyarray_bound(py),
        sac_full.to_pyarray_bound(py),
        av_hist.to_pyarray_bound(py),
    ))
}

fn b2_evaluate_flipped_lanes_internal(
    orig_slice: &[u64],
    eval_slice: &[u64],
    flip_s0_lo: u64,
    flip_s0_hi: u64,
    flip_s1_lo: u64,
    flip_s1_hi: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    debug_assert_eq!(orig_slice.len() % 4, 0);
    debug_assert_eq!(orig_slice.len(), eval_slice.len());

    // Single-lane constraint: flip must not span both s0 and s1 simultaneously.
    let flip_in_s0 = (flip_s0_lo | flip_s0_hi) != 0;
    let flip_in_s1 = (flip_s1_lo | flip_s1_hi) != 0;
    debug_assert!(
        !(flip_in_s0 && flip_in_s1),
        "b2_evaluate_flipped_lanes: mask must not span both s0 and s1"
    );

    let n = orig_slice.len() / 4;

    let mut flipped_buf = Vec::<u64>::with_capacity(n * 4);

    for i in 0..n {
        flipped_buf.push(orig_slice[i * 4]     ^ flip_s0_lo);
        flipped_buf.push(orig_slice[i * 4 + 1] ^ flip_s0_hi);
        flipped_buf.push(orig_slice[i * 4 + 2] ^ flip_s1_lo);
        flipped_buf.push(orig_slice[i * 4 + 3] ^ flip_s1_hi);
    }

    match crate::dispatch::evaluate(&mut flipped_buf) {
        crate::dispatch::LustroError::Success => {},
        e => panic!("dispatch::evaluate (flip) failed: {:?}", e),
    }

    let mut sac_s0o0 = vec![0u64; 128];
    let mut sac_s0o1 = vec![0u64; 128];
    let mut sac_s1o0 = vec![0u64; 128];
    let mut sac_s1o1 = vec![0u64; 128];
    let mut sac_full = vec![0u64; 256];
    let mut av_hist  = vec![0u64; 257];

    for i in 0..n {
        let d_w0 = eval_slice[i * 4]     ^ flipped_buf[i * 4];
        let d_w1 = eval_slice[i * 4 + 1] ^ flipped_buf[i * 4 + 1];
        let d_w2 = eval_slice[i * 4 + 2] ^ flipped_buf[i * 4 + 2];
        let d_w3 = eval_slice[i * 4 + 3] ^ flipped_buf[i * 4 + 3];

        let hw = (d_w0.count_ones()
                + d_w1.count_ones()
                + d_w2.count_ones()
                + d_w3.count_ones()) as usize;
        debug_assert!(hw <= 256);
        av_hist[hw] += 1;

        count_bits_128(&mut sac_full[0..128],   d_w0, d_w1);
        count_bits_128(&mut sac_full[128..256], d_w2, d_w3);

        if flip_in_s0 {
            count_bits_128(&mut sac_s0o0, d_w0, d_w1);
            count_bits_128(&mut sac_s0o1, d_w2, d_w3);
        } else {
            count_bits_128(&mut sac_s1o0, d_w0, d_w1);
            count_bits_128(&mut sac_s1o1, d_w2, d_w3);
        }
    }

    (sac_s0o0, sac_s0o1, sac_s1o0, sac_s1o1, sac_full, av_hist)
}


// B2 CONDITIONAL SAC
// Split differential statistics by the original value of flip_bit.
#[pyfunction]
#[pyo3(signature = (orig_states, eval_states, flip_bit, flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi))]
pub fn b2_conditional_sac<'py>(
    py: Python<'py>,
    orig_states: PyReadonlyArray2<'py, u64>,
    eval_states: PyReadonlyArray2<'py, u64>,
    flip_bit: usize,
    flip_s0_lo: u64,
    flip_s0_hi: u64,
    flip_s1_lo: u64,
    flip_s1_hi: u64,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    u64,
    u64,
)> {
    if orig_states.shape()[1] != 4 || eval_states.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both inputs must have shape (n, 4)",
        ));
    }
    if orig_states.shape()[0] != eval_states.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "orig_states and eval_states must have the same row count",
        ));
    }
    if flip_bit >= 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("flip_bit must be < 256"));
    }

    let orig_arr   = orig_states.as_array();
    let orig_slice = orig_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("orig_states must be C-contiguous")
    })?;
    let eval_arr   = eval_states.as_array();
    let eval_slice = eval_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("eval_states must be C-contiguous")
    })?;

    let (cond0_sac, cond1_sac, cond0_cnt, cond1_cnt) = py.allow_threads(|| {
        b2_conditional_sac_internal(
            orig_slice, eval_slice, flip_bit,
            flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi,
        )
    });

    Ok((
        cond0_sac.to_pyarray_bound(py),
        cond1_sac.to_pyarray_bound(py),
        cond0_cnt,
        cond1_cnt,
    ))
}

fn b2_conditional_sac_internal(
    orig_slice: &[u64],
    eval_slice: &[u64],
    flip_bit: usize,
    flip_s0_lo: u64,
    flip_s0_hi: u64,
    flip_s1_lo: u64,
    flip_s1_hi: u64,
) -> (Vec<u64>, Vec<u64>, u64, u64) {
    debug_assert_eq!(orig_slice.len() % 4, 0);
    debug_assert_eq!(orig_slice.len(), eval_slice.len());

    let n = orig_slice.len() / 4;

    let mut flipped_buf = Vec::<u64>::with_capacity(n * 4);

    for i in 0..n {
        flipped_buf.push(orig_slice[i * 4]     ^ flip_s0_lo);
        flipped_buf.push(orig_slice[i * 4 + 1] ^ flip_s0_hi);
        flipped_buf.push(orig_slice[i * 4 + 2] ^ flip_s1_lo);
        flipped_buf.push(orig_slice[i * 4 + 3] ^ flip_s1_hi);
    }

    match crate::dispatch::evaluate(&mut flipped_buf) {
        crate::dispatch::LustroError::Success => {},
        e => panic!("dispatch::evaluate (flip) failed: {:?}", e),
    }

    let mut cond0_sac = vec![0u64; 256];
    let mut cond1_sac = vec![0u64; 256];
    let mut cond0_cnt = 0u64;
    let mut cond1_cnt = 0u64;

    // Resolve the state word and intra-word bit position.
    let (word_idx, bit_in_word) = match flip_bit {
        0..=63    => (0usize, flip_bit),
        64..=127  => (1usize, flip_bit - 64),
        128..=191 => (2usize, flip_bit - 128),
        _         => (3usize, flip_bit - 192),
    };

    for i in 0..n {
        let input_word      = orig_slice[i * 4 + word_idx];
        let input_bit_value = (input_word >> bit_in_word) & 1;

        let d_w0 = eval_slice[i * 4]     ^ flipped_buf[i * 4];
        let d_w1 = eval_slice[i * 4 + 1] ^ flipped_buf[i * 4 + 1];
        let d_w2 = eval_slice[i * 4 + 2] ^ flipped_buf[i * 4 + 2];
        let d_w3 = eval_slice[i * 4 + 3] ^ flipped_buf[i * 4 + 3];

        if input_bit_value == 0 {
            count_bits_128(&mut cond0_sac[0..128],   d_w0, d_w1);
            count_bits_128(&mut cond0_sac[128..256], d_w2, d_w3);
            cond0_cnt += 1;
        } else {
            count_bits_128(&mut cond1_sac[0..128],   d_w0, d_w1);
            count_bits_128(&mut cond1_sac[128..256], d_w2, d_w3);
            cond1_cnt += 1;
        }
    }

    (cond0_sac, cond1_sac, cond0_cnt, cond1_cnt)
}

// B13 STRUCTURED & TRUNCATED HIGHER-ORDER DIFFERENTIAL AUDIT
//
// Δ² = f(s) ⊕ f(s⊕b1) ⊕ f(s⊕b2) ⊕ f(s⊕b1⊕b2)
//
// Tests architecture-motivated bit pairs using Hamming-weight,
// per-bit and truncated low-bit differential statistics.
//
// Pre-evaluated outputs and reusable buffers avoid redundant evaluation.

// STRUCTURAL PAIRS CATALOGUE
pub const STRUCTURAL_PAIRS: &[(u8, u8)] = &[
    (63,  64),
    (127, 128),
    (191, 192),
    (0,   1),
    (62,  63),
    (64,  65),
    (126, 127),
    (0,   32),
    (0,   64),
    (63,  127),
    (0,   128),
    (127, 191),
    (0,   255),
];

pub const NUM_PAIRS: usize = 13;

// PHASE B HISTOGRAM LAYOUT
const SOURCES: usize = 5;

const LOW8_BUCKETS:  usize = 256;
const LOW12_BUCKETS: usize = 4096;
const LOW16_BUCKETS: usize = 65536;

const LOW8_MASK:  u64 = 0xFF;
const LOW12_MASK: u64 = 0xFFF;
const LOW16_MASK: u64 = 0xFFFF;

// Histogram block size for one source.
const PER_SOURCE_LEN: usize = LOW8_BUCKETS + LOW12_BUCKETS + LOW16_BUCKETS;

const PHASE_B_LEN: usize = SOURCES * PER_SOURCE_LEN;

// HELPERS
//
// Build a one-hot mask in the corresponding state word.
#[inline(always)]
fn bit_mask(bit: usize) -> (u64, u64, u64, u64) {
    debug_assert!(bit < 256);
    match bit {
        0..=63    => (1u64 << bit,            0, 0, 0),
        64..=127  => (0, 1u64 << (bit - 64),  0, 0),
        128..=191 => (0, 0, 1u64 << (bit - 128), 0),
        _         => (0, 0, 0, 1u64 << (bit - 192)),
    }
}

// Evaluate one structural pair using shared pre-allocated buffers.
fn b13_st_pair_internal(
    orig_slice: &[u64],
    eval_slice: &[u64],
    b1: usize,
    b2: usize,
    buf_b1:     &mut Vec<u64>,
    buf_b2:     &mut Vec<u64>,
    buf_b1b2:   &mut Vec<u64>,
    bit_counts: &mut [u64],
    phase_b:    &mut [u64],
) -> Result<(u64, u64, u64), String> {
    debug_assert!(orig_slice.len() % 4 == 0);
    debug_assert_eq!(orig_slice.len(), eval_slice.len());
    let n = orig_slice.len() / 4;

    let (f1w0, f1w1, f1w2, f1w3) = bit_mask(b1);
    let (f2w0, f2w1, f2w2, f2w3) = bit_mask(b2);

    let f12w0 = f1w0 ^ f2w0;
    let f12w1 = f1w1 ^ f2w1;
    let f12w2 = f1w2 ^ f2w2;
    let f12w3 = f1w3 ^ f2w3;

    buf_b1.clear();
    buf_b2.clear();
    buf_b1b2.clear();

    for i in 0..n {
        let w0 = orig_slice[i * 4];
        let w1 = orig_slice[i * 4 + 1];
        let w2 = orig_slice[i * 4 + 2];
        let w3 = orig_slice[i * 4 + 3];

        buf_b1.push(w0 ^ f1w0);    buf_b1.push(w1 ^ f1w1);
        buf_b1.push(w2 ^ f1w2);    buf_b1.push(w3 ^ f1w3);

        buf_b2.push(w0 ^ f2w0);    buf_b2.push(w1 ^ f2w1);
        buf_b2.push(w2 ^ f2w2);    buf_b2.push(w3 ^ f2w3);

        buf_b1b2.push(w0 ^ f12w0); buf_b1b2.push(w1 ^ f12w1);
        buf_b1b2.push(w2 ^ f12w2); buf_b1b2.push(w3 ^ f12w3);
    }

    match crate::dispatch::evaluate(buf_b1) {
        crate::dispatch::LustroError::Success => {},
        e => return Err(format!("b13_st evaluate(b1) failed: {:?}", e)),
    }
    match crate::dispatch::evaluate(buf_b2) {
        crate::dispatch::LustroError::Success => {},
        e => return Err(format!("b13_st evaluate(b2) failed: {:?}", e)),
    }
    match crate::dispatch::evaluate(buf_b1b2) {
        crate::dispatch::LustroError::Success => {},
        e => return Err(format!("b13_st evaluate(b1b2) failed: {:?}", e)),
    }

    let mut hw_sum     = 0u64;
    let mut hw_sq_sum  = 0u64;
    let mut zero_cnt   = 0u64;

    for i in 0..n {
        // Compute the second-order derivative from four evaluations.
        let d_w0 = eval_slice[i*4]   ^ buf_b1[i*4]   ^ buf_b2[i*4]   ^ buf_b1b2[i*4];
        let d_w1 = eval_slice[i*4+1] ^ buf_b1[i*4+1] ^ buf_b2[i*4+1] ^ buf_b1b2[i*4+1];
        let d_w2 = eval_slice[i*4+2] ^ buf_b1[i*4+2] ^ buf_b2[i*4+2] ^ buf_b1b2[i*4+2];
        let d_w3 = eval_slice[i*4+3] ^ buf_b1[i*4+3] ^ buf_b2[i*4+3] ^ buf_b1b2[i*4+3];

        let fold = d_w0 ^ d_w1 ^ d_w2 ^ d_w3;

        let hw = (d_w0.count_ones() + d_w1.count_ones()
                + d_w2.count_ones() + d_w3.count_ones()) as u64;
        hw_sum    += hw;
        hw_sq_sum += hw * hw;
        if hw == 0 { zero_cnt += 1; }

        count_bits_128(&mut bit_counts[0..128],   d_w0, d_w1);
        count_bits_128(&mut bit_counts[128..256], d_w2, d_w3);

        // Accumulate truncated low-bit histograms for each source.
        let sources = [d_w0, d_w1, d_w2, d_w3, fold];
        for (s, &src) in sources.iter().enumerate() {
            let base = s * PER_SOURCE_LEN;
            phase_b[base + (src & LOW8_MASK)  as usize] += 1;
            phase_b[base + LOW8_BUCKETS + (src & LOW12_MASK) as usize] += 1;
            phase_b[base + LOW8_BUCKETS + LOW12_BUCKETS + (src & LOW16_MASK) as usize] += 1;
        }
    }

    Ok((hw_sum, hw_sq_sum, zero_cnt))
}

// B13 entry point.
//
// Reuses pre-evaluated outputs and three buffers across all structural pairs.

#[pyfunction]
pub fn b13_structured_truncated<'py>(
    py: Python<'py>,
    orig_states: PyReadonlyArray2<'py, u64>,
    eval_states: PyReadonlyArray2<'py, u64>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    if orig_states.shape()[1] != 4 || eval_states.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both inputs must have shape (n, 4)",
        ));
    }
    if orig_states.shape()[0] != eval_states.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "orig_states and eval_states must have the same row count",
        ));
    }
    if orig_states.shape()[0] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "states must not be empty",
        ));
    }

    let orig_arr = orig_states.as_array();
    let orig_slice = orig_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("orig_states must be C-contiguous")
    })?;
    let eval_arr = eval_states.as_array();
    let eval_slice = eval_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("eval_states must be C-contiguous")
    })?;

    let result = py.allow_threads(|| {
        b13_structured_truncated_internal(orig_slice, eval_slice)
    });

    let (hw_sum, hw_sq, zero, bc, pb) = result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok((
        hw_sum.to_pyarray_bound(py),
        hw_sq.to_pyarray_bound(py),
        zero.to_pyarray_bound(py),
        bc.to_pyarray_bound(py),
        pb.to_pyarray_bound(py),
    ))
}

fn b13_structured_truncated_internal(
    orig_slice: &[u64],
    eval_slice: &[u64],
) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
    let n = orig_slice.len() / 4;

    let mut hw_sum = vec![0u64; NUM_PAIRS];
    let mut hw_sq  = vec![0u64; NUM_PAIRS];
    let mut zero   = vec![0u64; NUM_PAIRS];
    let mut bc     = vec![0u64; NUM_PAIRS * 256];
    let mut pb     = vec![0u64; NUM_PAIRS * PHASE_B_LEN];

    let mut buf_b1   = Vec::<u64>::with_capacity(n * 4);
    let mut buf_b2   = Vec::<u64>::with_capacity(n * 4);
    let mut buf_b1b2 = Vec::<u64>::with_capacity(n * 4);

    for (p, &(b1, b2)) in STRUCTURAL_PAIRS.iter().enumerate() {
        let bc_slice = &mut bc[p * 256..(p + 1) * 256];
        let pb_slice = &mut pb[p * PHASE_B_LEN..(p + 1) * PHASE_B_LEN];

        let (s, sq, z) = b13_st_pair_internal(
            orig_slice, eval_slice,
            b1 as usize, b2 as usize,
            &mut buf_b1, &mut buf_b2, &mut buf_b1b2,
            bc_slice, pb_slice,
        )?;

        hw_sum[p] = s;
        hw_sq[p]  = sq;
        zero[p]   = z;
    }

    Ok((hw_sum, hw_sq, zero, bc, pb))
}

// METADATA HELPERS
#[pyfunction]
pub fn b13_st_pairs_meta<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let flat: Vec<u8> = STRUCTURAL_PAIRS.iter()
        .flat_map(|&(b1, b2)| [b1, b2])
        .collect();
    Ok(flat.to_pyarray_bound(py))
}

// Expose histogram layout constants to Python.
#[pyfunction]
pub fn b13_st_layout() -> (usize, usize, usize, usize, usize, usize) {
    (NUM_PAIRS, SOURCES, LOW8_BUCKETS, LOW12_BUCKETS, LOW16_BUCKETS, PHASE_B_LEN)
}

// B16 — JACOBIAN / INFLUENCE MATRIX
//
// Chunk size chosen to bound staging-buffer memory.
const B16_CHUNK_STATES: usize = 12 * 1024;

#[pyfunction]
pub fn b16_jacobian_diff<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>, // jacobian flat 256x256 — flip count per (input_bit, output_bit)
    Bound<'py, PyArray1<u64>>, // row_hw_sum    — HW sum per input bit
    Bound<'py, PyArray1<u64>>, // row_hw_sq_sum — HW^2 sum per input bit (variance)
    Bound<'py, PyArray1<u64>>, // row_min_hw    — minimum observed HW per input bit
    Bound<'py, PyArray1<u64>>, // row_count     — sample count per input bit
)> {
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err("Input must have shape (n, 4)"));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let (jacobian, row_hw_sum, row_hw_sq_sum, row_min_hw, row_count) =
        py.allow_threads(|| b16_internal(slice));

    Ok((
        jacobian.to_pyarray_bound(py),
        row_hw_sum.to_pyarray_bound(py),
        row_hw_sq_sum.to_pyarray_bound(py),
        row_min_hw.to_pyarray_bound(py),
        row_count.to_pyarray_bound(py),
    ))
}

fn b16_internal(
    slice: &[u64],
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    debug_assert_eq!(slice.len() % 4, 0);

    let mut jacobian      = vec![0u64; 256 * 256];
    let mut row_hw_sum    = vec![0u64; 256];
    let mut row_hw_sq_sum = vec![0u64; 256];
    let mut row_min_hw    = vec![256u64; 256];
    let mut row_count     = vec![0u64; 256];

    let mut buf       = vec![0u64; B16_CHUNK_STATES * 8];
    let mut flip_bits = vec![0usize; B16_CHUNK_STATES];

    let mut offset = 0usize;

    while offset < slice.len() {
        let states_in_chunk = ((slice.len() - offset) / 4).min(B16_CHUNK_STATES);

        for i in 0..states_in_chunk {
            let src = offset + i * 4;
            let dst = i * 8;

            let w0 = slice[src];
            let w1 = slice[src + 1];
            let w2 = slice[src + 2];
            let w3 = slice[src + 3];

            buf[dst]     = w0;
            buf[dst + 1] = w1;
            buf[dst + 2] = w2;
            buf[dst + 3] = w3;

            let bit = (offset / 4 + i) % 256;
            flip_bits[i] = bit;

            let (m0, m1, m2, m3) = b16_bit_mask(bit);

            buf[dst + 4] = w0 ^ m0;
            buf[dst + 5] = w1 ^ m1;
            buf[dst + 6] = w2 ^ m2;
            buf[dst + 7] = w3 ^ m3;
        }

        let mut orig_buf = vec![0u64; states_in_chunk * 4];
        let mut flip_buf = vec![0u64; states_in_chunk * 4];

        for i in 0..states_in_chunk {
            let src = i * 8;
            let dst = i * 4;
            orig_buf[dst]     = buf[src];
            orig_buf[dst + 1] = buf[src + 1];
            orig_buf[dst + 2] = buf[src + 2];
            orig_buf[dst + 3] = buf[src + 3];
            flip_buf[dst]     = buf[src + 4];
            flip_buf[dst + 1] = buf[src + 5];
            flip_buf[dst + 2] = buf[src + 6];
            flip_buf[dst + 3] = buf[src + 7];
        }

        match dispatch::evaluate(&mut orig_buf) {
            LustroError::Success => {},
            e => panic!("dispatch::evaluate (orig) failed: {:?}", e),
        }
        match dispatch::evaluate(&mut flip_buf) {
            LustroError::Success => {},
            e => panic!("dispatch::evaluate (flip) failed: {:?}", e),
        }

        for i in 0..states_in_chunk {
            let bit = flip_bits[i];
            let dst = i * 4;

            let d_w0 = orig_buf[dst]     ^ flip_buf[dst];
            let d_w1 = orig_buf[dst + 1] ^ flip_buf[dst + 1];
            let d_w2 = orig_buf[dst + 2] ^ flip_buf[dst + 2];
            let d_w3 = orig_buf[dst + 3] ^ flip_buf[dst + 3];

            let hw = (d_w0.count_ones()
                    + d_w1.count_ones()
                    + d_w2.count_ones()
                    + d_w3.count_ones()) as u64;

            row_hw_sum[bit]    += hw;
            row_hw_sq_sum[bit] += hw * hw;
            row_count[bit]     += 1;
            if hw < row_min_hw[bit] {
                row_min_hw[bit] = hw;
            }

            b16_accumulate_jacobian(&mut jacobian, bit, d_w0, d_w1, d_w2, d_w3);
        }

        offset += states_in_chunk * 4;
    }

    (jacobian, row_hw_sum, row_hw_sq_sum, row_min_hw, row_count)
}

// B22 — ROTATIONAL BIT CORRELATION + CHAIN
//
// Measure rotational differential persistence across repeated evaluations.
#[pyfunction]
pub fn b22_rot_chain<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    rotation: u32,
    steps: u32,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    u64, u64, u64, u64, u64,
)> {
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input must have shape (n, 4)",
        ));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")
    })?;
    let (ones, lt112, lt104, lt96, lt88, lt80) =
        py.allow_threads(|| b22_rot_chain_internal(slice, rotation, steps));
    Ok((ones.to_pyarray_bound(py), lt112, lt104, lt96, lt88, lt80))
}

fn b22_rot_chain_internal(slice: &[u64], rotation: u32, steps: u32) -> (Vec<u64>, u64, u64, u64, u64, u64) {
    debug_assert_eq!(slice.len() % 4, 0);
    let n   = slice.len() / 4;
    let rot = rotation % 128;

    let mut x_buf  = Vec::<u64>::with_capacity(n * 4);
    let mut xr_buf = Vec::<u64>::with_capacity(n * 4);

    let mut ones        = vec![0u64; 256];
    let mut still_lt112 = vec![true; n];
    let mut still_lt104 = vec![true; n];
    let mut still_lt96  = vec![true; n];
    let mut still_lt88  = vec![true; n];
    let mut still_lt80  = vec![true; n];

    for i in 0..n {
        let w0 = slice[i * 4];
        let w1 = slice[i * 4 + 1];
        let w2 = slice[i * 4 + 2];
        let w3 = slice[i * 4 + 3];
        x_buf.push(w0);
        x_buf.push(w1);
        x_buf.push(w2);
        x_buf.push(w3);
        let (rw0, rw1) = rotl128(w0, w1, rot);
        let (rw2, rw3) = rotl128(w2, w3, rot);
        xr_buf.push(rw0);
        xr_buf.push(rw1);
        xr_buf.push(rw2);
        xr_buf.push(rw3);
    }

    let mut first_step = true;

    for _ in 0..steps {
        match crate::dispatch::evaluate(&mut x_buf) {
            crate::dispatch::LustroError::Success => {},
            e => panic!("dispatch::evaluate (x) failed: {:?}", e),
        }
        match crate::dispatch::evaluate(&mut xr_buf) {
            crate::dispatch::LustroError::Success => {},
            e => panic!("dispatch::evaluate (xr) failed: {:?}", e),
        }

        for i in 0..n {
            let x_w0  = x_buf[i * 4];
            let x_w1  = x_buf[i * 4 + 1];
            let x_w2  = x_buf[i * 4 + 2];
            let x_w3  = x_buf[i * 4 + 3];
            let xr_w0 = xr_buf[i * 4];
            let xr_w1 = xr_buf[i * 4 + 1];
            let xr_w2 = xr_buf[i * 4 + 2];
            let xr_w3 = xr_buf[i * 4 + 3];

            let (rx_w0, rx_w1) = rotl128(x_w0, x_w1, rot);
            let (rx_w2, rx_w3) = rotl128(x_w2, x_w3, rot);

            let d_w0 = xr_w0 ^ rx_w0;
            let d_w1 = xr_w1 ^ rx_w1;
            let d_w2 = xr_w2 ^ rx_w2;
            let d_w3 = xr_w3 ^ rx_w3;

            let hw = (d_w0.count_ones()
                    + d_w1.count_ones()
                    + d_w2.count_ones()
                    + d_w3.count_ones()) as usize;

            if hw >= 112 { still_lt112[i] = false; }
            if hw >= 104 { still_lt104[i] = false; }
            if hw >= 96  { still_lt96[i]  = false; }
            if hw >= 88  { still_lt88[i]  = false; }
            if hw >= 80  { still_lt80[i]  = false; }

            if first_step {
                count_bits_128(&mut ones[0..128],   d_w0, d_w1);
                count_bits_128(&mut ones[128..256], d_w2, d_w3);
            }
        }
        first_step = false;
    }

    let chain_lt112 = still_lt112.iter().filter(|&&v| v).count() as u64;
    let chain_lt104 = still_lt104.iter().filter(|&&v| v).count() as u64;
    let chain_lt96  = still_lt96.iter().filter(|&&v| v).count() as u64;
    let chain_lt88  = still_lt88.iter().filter(|&&v| v).count() as u64;
    let chain_lt80  = still_lt80.iter().filter(|&&v| v).count() as u64;

    (ones, chain_lt112, chain_lt104, chain_lt96, chain_lt88, chain_lt80)
}

// Measure rotational differential persistence on the full 256-bit state.
#[pyfunction]
pub fn b22_rot_chain_256<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    rotation: u32,
    steps: u32,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    u64, u64, u64, u64, u64,
)> {
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input must have shape (n, 4)",
        ));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")
    })?;
    let (ones, lt112, lt104, lt96, lt88, lt80) =
        py.allow_threads(|| b22_rot_chain_256_internal(slice, rotation, steps));
    Ok((ones.to_pyarray_bound(py), lt112, lt104, lt96, lt88, lt80))
}

fn b22_rot_chain_256_internal(slice: &[u64], rotation: u32, steps: u32) -> (Vec<u64>, u64, u64, u64, u64, u64) {
    debug_assert_eq!(slice.len() % 4, 0);
    let n   = slice.len() / 4;
    let rot = rotation % 256;

    let mut x_buf  = Vec::<u64>::with_capacity(n * 4);
    let mut xr_buf = Vec::<u64>::with_capacity(n * 4);

    let mut ones        = vec![0u64; 256];
    let mut still_lt112 = vec![true; n];
    let mut still_lt104 = vec![true; n];
    let mut still_lt96  = vec![true; n];
    let mut still_lt88  = vec![true; n];
    let mut still_lt80  = vec![true; n];

    for i in 0..n {
        let w0 = slice[i * 4];
        let w1 = slice[i * 4 + 1];
        let w2 = slice[i * 4 + 2];
        let w3 = slice[i * 4 + 3];
        x_buf.push(w0);
        x_buf.push(w1);
        x_buf.push(w2);
        x_buf.push(w3);
        let (rw0, rw1, rw2, rw3) = rotl256(w0, w1, w2, w3, rot);
        xr_buf.push(rw0);
        xr_buf.push(rw1);
        xr_buf.push(rw2);
        xr_buf.push(rw3);
    }

    let mut first_step = true;

    for _ in 0..steps {
        match crate::dispatch::evaluate(&mut x_buf) {
            crate::dispatch::LustroError::Success => {},
            e => panic!("dispatch::evaluate (x) failed: {:?}", e),
        }
        match crate::dispatch::evaluate(&mut xr_buf) {
            crate::dispatch::LustroError::Success => {},
            e => panic!("dispatch::evaluate (xr) failed: {:?}", e),
        }

        for i in 0..n {
            let x_w0  = x_buf[i * 4];
            let x_w1  = x_buf[i * 4 + 1];
            let x_w2  = x_buf[i * 4 + 2];
            let x_w3  = x_buf[i * 4 + 3];
            let xr_w0 = xr_buf[i * 4];
            let xr_w1 = xr_buf[i * 4 + 1];
            let xr_w2 = xr_buf[i * 4 + 2];
            let xr_w3 = xr_buf[i * 4 + 3];

            let (rx_w0, rx_w1, rx_w2, rx_w3) = rotl256(x_w0, x_w1, x_w2, x_w3, rot);

            let d_w0 = xr_w0 ^ rx_w0;
            let d_w1 = xr_w1 ^ rx_w1;
            let d_w2 = xr_w2 ^ rx_w2;
            let d_w3 = xr_w3 ^ rx_w3;

            let hw = (d_w0.count_ones()
                    + d_w1.count_ones()
                    + d_w2.count_ones()
                    + d_w3.count_ones()) as usize;

            if hw >= 112 { still_lt112[i] = false; }
            if hw >= 104 { still_lt104[i] = false; }
            if hw >= 96  { still_lt96[i]  = false; }
            if hw >= 88  { still_lt88[i]  = false; }
            if hw >= 80  { still_lt80[i]  = false; }

            if first_step {
                count_bits_128(&mut ones[0..128],   d_w0, d_w1);
                count_bits_128(&mut ones[128..256], d_w2, d_w3);
            }
        }
        first_step = false;
    }

    let chain_lt112 = still_lt112.iter().filter(|&&v| v).count() as u64;
    let chain_lt104 = still_lt104.iter().filter(|&&v| v).count() as u64;
    let chain_lt96  = still_lt96.iter().filter(|&&v| v).count() as u64;
    let chain_lt88  = still_lt88.iter().filter(|&&v| v).count() as u64;
    let chain_lt80  = still_lt80.iter().filter(|&&v| v).count() as u64;

    (ones, chain_lt112, chain_lt104, chain_lt96, chain_lt88, chain_lt80)
}


// B32 — GLOBAL CONVERGENCE
#[inline(always)]
fn b32_fp(state: &[u64; 4], fp_mask: u64) -> u64 {
    (state[0].wrapping_mul(0x9E3779B97F4A7C15)
   ^ state[1].wrapping_mul(0xD6E8FEB86659FD93)
   ^ state[2].wrapping_mul(0x94D049BB133111EB)
   ^ state[3].wrapping_mul(0xBF58476D1CE4E5B9))
   & fp_mask
}

// Distinguished points are selected by a zero prefix of s0_lo.
#[pyfunction]
pub fn b32_find_dp<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    max_steps: u64,
    dp_bits: u32,
    fp_bits: u32,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err(
            "Input must have shape (n, 4)",
        ));
    }
    if max_steps == 0 {
        return Err(PyValueError::new_err(
            "max_steps must be > 0",
        ));
    }
    if dp_bits == 0 || dp_bits > 64 {
        return Err(PyValueError::new_err(
            "dp_bits must be in 1..=64",
        ));
    }
    if fp_bits == 0 || fp_bits > 64 {
        return Err(PyValueError::new_err(
            "fp_bits must be in 1..=64",
        ));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let n = arr.shape()[0];
    let flat = py.allow_threads(|| {
        b32_find_dp_internal(slice, max_steps, dp_bits, fp_bits)
    });
    let arr2 = numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
        .expect("shape mismatch");
    Ok(arr2.to_pyarray_bound(py))
}

// Process independent macro-chunks in parallel.
const B32_MACRO_CHUNK_STATES: usize = 50_000;
const B32_MACRO_CHUNK_WORDS:  usize = B32_MACRO_CHUNK_STATES * 4;

fn b32_find_dp_internal(
    slice: &[u64],
    max_steps: u64,
    dp_bits: u32,
    fp_bits: u32,
) -> Vec<u64> {
    debug_assert_eq!(slice.len() % 4, 0);
    let n = slice.len() / 4;

    let dp_mask: u64 = if dp_bits == 64 {
        u64::MAX
    } else {
        !((1u64 << (64 - dp_bits)) - 1)
    };
    let fp_mask: u64 = if fp_bits == 64 {
        u64::MAX
    } else {
        (1u64 << fp_bits) - 1
    };

    let mut flat = vec![0u64; n * 3];

    flat.par_chunks_mut(B32_MACRO_CHUNK_STATES * 3)
        .zip(slice.par_chunks(B32_MACRO_CHUNK_WORDS))
        .for_each(|(out_chunk, in_chunk)| {
            let chunk_n = in_chunk.len() / 4;
            for i in 0..chunk_n {
                let mut state = [
                    in_chunk[i * 4],
                    in_chunk[i * 4 + 1],
                    in_chunk[i * 4 + 2],
                    in_chunk[i * 4 + 3],
                ];
                let mut found = false;
                let mut steps = 0u64;

                for step in 0..max_steps {
                    match crate::dispatch::evaluate_st(&mut state) {
                        crate::dispatch::LustroError::Success => {},
                        e => panic!("dispatch::evaluate_st failed: {:?}", e),
                    }
                    steps = step + 1;
                    if (state[0] & dp_mask) == 0 {
                        out_chunk[i * 3]     = b32_fp(&state, fp_mask);
                        out_chunk[i * 3 + 1] = steps;
                        out_chunk[i * 3 + 2] = 0; // reason: found
                        found = true;
                        break;
                    }
                }
                if !found {
                    out_chunk[i * 3]     = 0;
                    out_chunk[i * 3 + 1] = steps;
                    out_chunk[i * 3 + 2] = 1; // reason: timeout
                }
            }
        });

    flat
}

// Track occupancy of buckets formed from the upper bits of the XOR projection.
#[pyfunction]
pub fn b32_isotropy<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    steps: u32,
    bucket_bits: u32,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if bucket_bits == 0 || bucket_bits > 24 {
        return Err(PyValueError::new_err("bucket_bits must be in 1..=24"));
    }
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err("Input must have shape (n, 4)"));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let counts = py.allow_threads(|| b32_isotropy_internal(slice, steps, bucket_bits));

    Ok(counts.to_pyarray_bound(py))
}

fn b32_isotropy_internal(slice: &[u64], steps: u32, bucket_bits: u32) -> Vec<u64> {
    debug_assert_eq!(slice.len() % 4, 0);

    let n_buckets    = 1usize << bucket_bits;
    let bucket_shift = 64 - bucket_bits;

    slice
        .par_chunks(B32_MACRO_CHUNK_WORDS)

        .fold(

            || vec![0u64; n_buckets],

            |mut counts: Vec<u64>, macro_chunk: &[u64]| {

                for chunk in macro_chunk.chunks_exact(4) {
                    let mut state = [
                        chunk[0],
                        chunk[1],
                        chunk[2],
                        chunk[3],
                    ];

                    for _ in 0..steps {
                        match crate::dispatch::evaluate_st(&mut state) {
                            crate::dispatch::LustroError::Success => {},
                            e => panic!("dispatch::evaluate_st failed: {:?}", e),
                        }
                        let bucket_src = state[0] ^ state[1] ^ state[2] ^ state[3];
                        let bucket = (bucket_src >> bucket_shift) as usize;
                        counts[bucket] += 1;
                    }
                }

                counts
            }
        )

        .reduce(

            || vec![0u64; n_buckets],

            |mut a: Vec<u64>, b: Vec<u64>| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            }
        )
}

// B32C — ORBIT MIXING
//
// Accumulate per-step Hamming-weight, per-word and per-bit statistics.
#[pyfunction]
pub fn b32_orbit_mixing<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    max_steps: u32,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if max_steps == 0 || max_steps > 1024 {
        return Err(PyValueError::new_err("max_steps must be in 1..=1024"));
    }
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err("Input must have shape (n, 4)"));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let out = py.allow_threads(|| b32_orbit_mixing_internal(slice, max_steps));
    Ok(out.to_pyarray_bound(py))
}

fn b32_orbit_mixing_internal(slice: &[u64], max_steps: u32) -> Vec<u64> {
    debug_assert_eq!(slice.len() % 4, 0);

    let steps  = max_steps as usize;
    let stride = 5 + 256; // per step: 5 summary + 256 bit counts
    let acc_len = steps * stride;

    slice
        .par_chunks(B32_MACRO_CHUNK_WORDS)

        .fold(

            || vec![0u64; acc_len],

            |mut acc: Vec<u64>, macro_chunk: &[u64]| {

                for chunk in macro_chunk.chunks_exact(4) {
                    let mut state = [
                        chunk[0],
                        chunk[1],
                        chunk[2],
                        chunk[3],
                    ];

                    for step in 0..steps {
                        match crate::dispatch::evaluate_st(&mut state) {
                            crate::dispatch::LustroError::Success => {},
                            e => panic!("dispatch::evaluate_st failed: {:?}", e),
                        }

                        let pc0 = state[0].count_ones() as u64;
                        let pc1 = state[1].count_ones() as u64;
                        let pc2 = state[2].count_ones() as u64;
                        let pc3 = state[3].count_ones() as u64;
                        let hw  = pc0 + pc1 + pc2 + pc3;

                        let base      = step * stride;
                        acc[base]     += hw;
                        acc[base + 1] += pc0;
                        acc[base + 2] += pc1;
                        acc[base + 3] += pc2;
                        acc[base + 4] += pc3;

                        let mut x = state[0];
                        while x != 0 {
                            let b = x.trailing_zeros() as usize;
                            acc[base + 5 + b] += 1;
                            x &= x - 1;
                        }
                        let mut x = state[1];
                        while x != 0 {
                            let b = x.trailing_zeros() as usize;
                            acc[base + 5 + 64 + b] += 1;
                            x &= x - 1;
                        }
                        let mut x = state[2];
                        while x != 0 {
                            let b = x.trailing_zeros() as usize;
                            acc[base + 5 + 128 + b] += 1;
                            x &= x - 1;
                        }
                        let mut x = state[3];
                        while x != 0 {
                            let b = x.trailing_zeros() as usize;
                            acc[base + 5 + 192 + b] += 1;
                            x &= x - 1;
                        }
                    }
                }

                acc
            }
        )

        .reduce(

            || vec![0u64; acc_len],

            |mut a: Vec<u64>, b: Vec<u64>| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            }
        )
}

// B32D — H0 baseline using iid uniform fingerprints.
#[pyfunction]
pub fn b32_simulate_h0<'py>(
    py: Python<'py>,
    n_rep: u32,
    max_orbit: u32,
    fp_bits: u32,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if fp_bits == 0 || fp_bits > 56 {
        return Err(PyValueError::new_err("fp_bits must be in 1..=56"));
    }
    if max_orbit == 0 || max_orbit > 1_000_000 {
        return Err(PyValueError::new_err("max_orbit must be in 1..=16384"));
    }
    if n_rep == 0 {
        return Err(PyValueError::new_err("n_rep must be > 0"));
    }
    let out = py.allow_threads(|| b32_simulate_h0_internal(n_rep, max_orbit, fp_bits, seed));
    Ok(out.to_pyarray_bound(py))
}

fn b32_simulate_h0_internal(n_rep: u32, max_orbit: u32, fp_bits: u32, seed: u64) -> Vec<u64> {
    let fp_mask = if fp_bits == 64 { u64::MAX } else { (1u64 << fp_bits) - 1 };
    let n = n_rep as usize;

    let mut out = vec![0u64; n * 3];

    // SplitMix64 provides independent deterministic RNG streams.
    #[inline(always)]
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    // Derive deterministic per-chunk RNG streams to avoid cross-thread sequence sharing.
    out.par_chunks_mut(B32_MACRO_CHUNK_STATES * 3)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let chunk_n = out_chunk.len() / 3;

            let mut rng_state = seed
                .wrapping_add(0x6C62272E07BB0142)
                .wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));

            let mut seen: std::collections::HashMap<u64, u32> =
                std::collections::HashMap::with_capacity(max_orbit as usize);

            for i in 0..chunk_n {
                seen.clear();
                let mut found     = 0u64;
                let mut gap       = 0u64;
                let mut fp_at_rec = 0u64;

                for step in 0..max_orbit {
                    let fp = splitmix64(&mut rng_state) & fp_mask;
                    if let Some(&first_pos) = seen.get(&fp) {
                        found     = 1;
                        gap       = (step - first_pos) as u64;
                        fp_at_rec = fp;
                        break;
                    }
                    seen.insert(fp, step);
                }

                out_chunk[i * 3]     = found;
                out_chunk[i * 3 + 1] = gap;
                out_chunk[i * 3 + 2] = fp_at_rec;
            }
        });

    out
}

// B32D — FINGERPRINT RECURRENCE
//
// Detect the first repeated fingerprint within the orbit window.
#[pyfunction]
pub fn b32_cycle_signature<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    max_orbit: u32,
    fp_bits: u32,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if fp_bits == 0 || fp_bits > 56 {
        return Err(PyValueError::new_err("fp_bits must be in 1..=56"));
    }
    if max_orbit == 0 || max_orbit > 1_000_000 {
        return Err(PyValueError::new_err("max_orbit must be in 1..=16384"));
    }
    let arr = states.as_array();
    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err("Input must have shape (n, 4)"));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let out = py.allow_threads(|| b32_cycle_signature_internal(slice, max_orbit, fp_bits));

    Ok(out.to_pyarray_bound(py))
}

fn b32_cycle_signature_internal(slice: &[u64], max_orbit: u32, fp_bits: u32) -> Vec<u64> {
    debug_assert_eq!(slice.len() % 4, 0);

    let n       = slice.len() / 4;
    let fp_mask = if fp_bits == 64 { u64::MAX } else { (1u64 << fp_bits) - 1 };

    let mut out = vec![0u64; n * 2];

    out.par_chunks_mut(B32_MACRO_CHUNK_STATES * 2)
        .zip(slice.par_chunks(B32_MACRO_CHUNK_WORDS))
        .for_each(|(out_chunk, in_chunk)| {
            let chunk_n = in_chunk.len() / 4;

            let mut seen: std::collections::HashMap<u64, u32> =
                std::collections::HashMap::with_capacity(max_orbit as usize);

            for i in 0..chunk_n {
                seen.clear();

                let mut state = [
                    in_chunk[i * 4],
                    in_chunk[i * 4 + 1],
                    in_chunk[i * 4 + 2],
                    in_chunk[i * 4 + 3],
                ];

                let mut recurrence_step = 0u64;
                let mut recurrence_fp   = 0u64;

                for step in 0..max_orbit {
                    match crate::dispatch::evaluate_st(&mut state) {
                        crate::dispatch::LustroError::Success => {},
                        e => panic!("dispatch::evaluate_st failed: {:?}", e),
                    }

                    let fp = b32_fp(&state, fp_mask);

                    if let Some(&first_pos) = seen.get(&fp) {
                        // Record the recurrence gap to the first occurrence.
                        recurrence_step = (step - first_pos) as u64;
                        recurrence_fp   = fp;
                        break;
                    }

                    seen.insert(fp, step);
                }

                out_chunk[i * 2]     = recurrence_step;
                out_chunk[i * 2 + 1] = recurrence_fp;
            }
        });

    out
}

// Chunking bounds per-thread histogram allocation overhead.
const B32E_MACRO_CHUNK_STATES: usize = 200_000;
const B32E_MACRO_CHUNK_WORDS:  usize = B32E_MACRO_CHUNK_STATES * 4;

#[pyfunction]
pub fn b32e_state_space_profile<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    checkpoints: Vec<u32>,
    n_windows: usize,
) -> PyResult<Bound<'py, PyArray1<u64>>> {

    let arr = states.as_array();

    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err("Input must have shape (n, 4)"));
    }
    if checkpoints.is_empty() {
        return Err(PyValueError::new_err("checkpoints must not be empty"));
    }
    if n_windows == 0 || 256 % n_windows != 0 || 256 / n_windows > 16 {
        return Err(PyValueError::new_err(
            "n_windows must be a divisor of 256 with window width <= 16 bits (valid: 16,32,64,128,256)"
        ));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let out = py.allow_threads(|| {
        b32e_state_space_profile_internal(slice, &checkpoints, n_windows)
    });

    Ok(out.to_pyarray_bound(py))
}

fn b32e_state_space_profile_internal(
    slice: &[u64],
    checkpoints: &[u32],
    n_windows: usize,
) -> Vec<u64> {

    let n_cp = checkpoints.len();
    // Extract fixed-width windows, including word-straddling ranges.
    let bits_per_window = 256 / n_windows;
    debug_assert!(bits_per_window < 64);
    let n_buckets = 1usize << bits_per_window;
    let bucket_mask = (1u64 << bits_per_window) - 1;
    let hist_len = n_cp * n_windows * n_buckets;

    debug_assert_eq!(slice.len() % 4, 0);

    slice
        .par_chunks(B32E_MACRO_CHUNK_WORDS)
        .fold(
            || vec![0u64; hist_len],
            |mut hist: Vec<u64>, macro_chunk: &[u64]| {
                for chunk in macro_chunk.chunks_exact(4) {
                    let mut state = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let mut prev_cp = 0u32;

                    for (cp_idx, &checkpoint) in checkpoints.iter().enumerate() {
                        let delta = checkpoint - prev_cp;
                        for _ in 0..delta {
                            match crate::dispatch::evaluate_st(&mut state) {
                                crate::dispatch::LustroError::Success => {}
                                e => panic!("dispatch::evaluate_st failed: {:?}", e),
                            }
                        }
                        prev_cp = checkpoint;

                        let checkpoint_base = cp_idx * n_windows * n_buckets;

                        for window_idx in 0..n_windows {
                            let bit_start = window_idx * bits_per_window;
                            let word_idx  = bit_start / 64;
                            let bit_off   = bit_start % 64;

                            let bucket = if bit_off + bits_per_window <= 64 {
                                (state[word_idx] >> bit_off) & bucket_mask
                            } else {
                                let lo_bits = 64 - bit_off;
                                let lo = state[word_idx] >> bit_off;
                                let hi = state[word_idx + 1] << lo_bits;
                                (lo | hi) & bucket_mask
                            } as usize;

                            hist[checkpoint_base + window_idx * n_buckets + bucket] += 1;
                        }
                    }
                }
                hist
            }
        )
        .reduce(
            || vec![0u64; hist_len],
            |mut a: Vec<u64>, b: Vec<u64>| {
                for (x, y) in a.iter_mut().zip(b.iter()) { *x += *y; }
                a
            }
        )
}

// B32F — NEAR-NEIGHBOUR DISTANCE RETENTION
//
// Pair each state with a single-bit neighbour and track HD retention
// at selected checkpoints and the minimum HD over the trajectory.
const B32F_HD_BINS: usize = 257; // HD 0..=256

// Process neighbour pairs in independent macro-chunks.
const B32F_MACRO_CHUNK_PAIRS: usize = 50_000;

#[pyfunction]
pub fn b32f_distance_retention<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    max_steps: u32,
    checkpoints: Vec<u32>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {

    let arr = states.as_array();

    if arr.shape()[1] != 4 {
        return Err(PyValueError::new_err("Input must have shape (n, 4)"));
    }
    if arr.shape()[0] % 256 != 0 {
        return Err(PyValueError::new_err(
            "n must be a multiple of 256 (round-robin bit coverage)"
        ));
    }
    if checkpoints.is_empty() {
        return Err(PyValueError::new_err("checkpoints must not be empty"));
    }
    if max_steps == 0 || max_steps < *checkpoints.last().unwrap() {
        return Err(PyValueError::new_err(
            "max_steps must be >= last checkpoint"
        ));
    }

    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err("Array must be C-contiguous")
    })?;

    let out = py.allow_threads(|| {
        b32f_distance_retention_internal(slice, max_steps, &checkpoints)
    });

    Ok(out.to_pyarray_bound(py))
}

fn b32f_distance_retention_internal(
    slice: &[u64],
    max_steps: u32,
    checkpoints: &[u32],
) -> Vec<u64> {

    let n_cp = checkpoints.len();
    let hd_hist_len = n_cp * B32F_HD_BINS;
    let out_len = hd_hist_len + B32F_HD_BINS;

    debug_assert_eq!(slice.len() % 4, 0);

    const CHUNK_WORDS: usize = B32F_MACRO_CHUNK_PAIRS * 4;

    slice
        .par_chunks(CHUNK_WORDS)
        .enumerate()
        .fold(
            || vec![0u64; out_len],
            |mut acc: Vec<u64>, (chunk_idx, macro_chunk): (usize, &[u64])| {
                let chunk_state_offset = chunk_idx * B32F_MACRO_CHUNK_PAIRS;

                for (local_i, chunk) in macro_chunk.chunks_exact(4).enumerate() {
                    let global_i = chunk_state_offset + local_i;
                    let bit = global_i % 256;
                    let (m0, m1, m2, m3) = b16_bit_mask(bit);

                    let w0 = chunk[0]; let w1 = chunk[1];
                    let w2 = chunk[2]; let w3 = chunk[3];

                    let mut a = [w0,      w1,      w2,      w3     ];
                    let mut b = [w0^m0,   w1^m1,   w2^m2,   w3^m3  ];

                    let mut min_hd: u32 = 256;
                    let mut prev_cp: u32 = 0;

                    for (cp_idx, &checkpoint) in checkpoints.iter().enumerate() {
                        let delta = checkpoint - prev_cp;
                        let mut hd_last: u32 = 0;

                        for _ in 0..delta {
                            match crate::dispatch::evaluate_st(&mut a) {
                                crate::dispatch::LustroError::Success => {}
                                e => panic!("evaluate_st (A) failed: {:?}", e),
                            }
                            match crate::dispatch::evaluate_st(&mut b) {
                                crate::dispatch::LustroError::Success => {}
                                e => panic!("evaluate_st (B) failed: {:?}", e),
                            }
                            hd_last = (a[0]^b[0]).count_ones()
                                    + (a[1]^b[1]).count_ones()
                                    + (a[2]^b[2]).count_ones()
                                    + (a[3]^b[3]).count_ones();
                            if hd_last < min_hd { min_hd = hd_last; }
                        }

                        prev_cp = checkpoint;
                        acc[cp_idx * B32F_HD_BINS + hd_last as usize] += 1;
                    }

                    for _ in *checkpoints.last().unwrap()..max_steps {
                        match crate::dispatch::evaluate_st(&mut a) {
                            crate::dispatch::LustroError::Success => {}
                            e => panic!("evaluate_st (A) failed: {:?}", e),
                        }
                        match crate::dispatch::evaluate_st(&mut b) {
                            crate::dispatch::LustroError::Success => {}
                            e => panic!("evaluate_st (B) failed: {:?}", e),
                        }
                        let hd = ((a[0]^b[0]).count_ones()
                                + (a[1]^b[1]).count_ones()
                                + (a[2]^b[2]).count_ones()
                                + (a[3]^b[3]).count_ones()) as u32;
                        if hd < min_hd { min_hd = hd; }
                    }

                    acc[hd_hist_len + min_hd as usize] += 1;
                }
                acc
            }
        )
        .reduce(
            || vec![0u64; out_len],
            |mut a: Vec<u64>, b: Vec<u64>| {
                for (x, y) in a.iter_mut().zip(b.iter()) { *x += *y; }
                a
            }
        )
}

// B51 — ALGEBRAIC DEGREE TEST
//
// EXACT uses a full cube sum for one output bit.
// PROB tests the full output vector using randomized affine offsets.
#[pyfunction]
pub fn b51_exact_degree<'py>(
    py: Python<'py>,
    base_state: PyReadonlyArray1<'py, u64>,
    variable_bits: Vec<u32>,
    output_bit: u32,
    batch_states: usize,
) -> PyResult<bool> {
    let base = base_state.as_slice()?;
    if base.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "base_state must contain 4 u64 values",
        ));
    }
    if variable_bits.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variable_bits empty",
        ));
    }
    if variable_bits.len() > 63 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variable_bits > 63 unsupported",
        ));
    }
    if variable_bits.iter().any(|&b| b >= 256) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variable_bits: all values must be in 0..=255",
        ));
    }
    {
        let mut seen = std::collections::HashSet::new();
        if variable_bits.iter().any(|b| !seen.insert(b)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "variable_bits: duplicate bit indices are not allowed",
            ));
        }
    }
    if output_bit >= 256 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "output_bit must be < 256",
        ));
    }
    if batch_states == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "batch_states must be > 0",
        ));
    }

    let result = py.allow_threads(|| -> Result<bool, &'static str> {
        let degree       = variable_bits.len();
        let total_points = 1u64 << degree;

        // Precompute word/mask pairs for Gray-code bit updates.
        let masks: Vec<(usize, u64)> = variable_bits
            .iter()
            .map(|&b| {
                let b = b as usize;
                (b / 64, 1u64 << (b % 64))
            })
            .collect();

        let out_word = (output_bit / 64) as usize;
        let out_bit  = output_bit % 64;

        let mut batch = vec![0u64; batch_states * 4];

        let mut parity: u8 = 0;

        // Maintain the cube point incrementally using Gray-code transitions.
        let mut current = [base[0], base[1], base[2], base[3]];
        let mut prev_gray: u64 = 0;

        let mut processed: u64 = 0;

        while processed < total_points {
            let remaining     = (total_points - processed) as usize;
            let current_batch = remaining.min(batch_states);

            // Advance the cube point by the single Gray-code bit change.
            for i in 0..current_batch {
                let idx  = processed + i as u64;
                let gray = idx ^ (idx >> 1);
                let diff = gray ^ prev_gray;

                if diff != 0 {
                    let changed       = diff.trailing_zeros() as usize;
                    let (word, mask)  = masks[changed];
                    current[word]    ^= mask;
                }
                prev_gray = gray;

                let off = i * 4;
                batch[off]     = current[0];
                batch[off + 1] = current[1];
                batch[off + 2] = current[2];
                batch[off + 3] = current[3];
            }

            match crate::dispatch::evaluate(&mut batch[..current_batch * 4]) {
                crate::dispatch::LustroError::Success => {},
                _ => return Err("dispatch::evaluate failed in b51_exact_degree"),
            }

            // Cube sum for the selected output bit.
            for i in 0..current_batch {
                let off = i * 4;
                let bit = (batch[off + out_word] >> out_bit) & 1;
                parity ^= bit as u8;
            }

            processed += current_batch as u64;
        }

        Ok(parity != 0)
    });

    match result {
        Ok(v)  => Ok(v),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

// B51 PROBABILISTIC — FULL OUTPUT VECTOR
#[pyfunction]
pub fn b51_prob_degree<'py>(
    py: Python<'py>,
    base_state: PyReadonlyArray1<'py, u64>,
    variable_bits: Vec<u32>,
    cubes: u32,
    batch_states: usize,
    seed: u64,
) -> PyResult<u32> {
    let base = base_state.as_slice()?;
    if base.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "base_state must contain 4 u64 values",
        ));
    }
    if variable_bits.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variable_bits empty",
        ));
    }
    if variable_bits.len() > 63 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variable_bits > 63 unsupported",
        ));
    }
    if variable_bits.iter().any(|&b| b >= 256) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variable_bits: all values must be in 0..=255",
        ));
    }
    {
        let mut seen = std::collections::HashSet::new();
        if variable_bits.iter().any(|b| !seen.insert(b)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "variable_bits: duplicate bit indices are not allowed",
            ));
        }
    }
    if cubes == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cubes must be > 0",
        ));
    }
    if batch_states == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "batch_states must be > 0",
        ));
    }
    if seed == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seed must be != 0",
        ));
    }

    let result = py.allow_threads(|| -> Result<u32, &'static str> {
        let degree       = variable_bits.len();
        let total_points = 1u64 << degree;

        // Precompute word/mask pairs for Gray-code bit updates.
        let masks: Vec<(usize, u64)> = variable_bits
            .iter()
            .map(|&b| {
                let b = b as usize;
                (b / 64, 1u64 << (b % 64))
            })
            .collect();

        let mut batch = vec![0u64; batch_states * 4];

        // XorShift64 provides deterministic randomized offsets.
        let mut rng = seed;
        #[inline(always)]
        fn xorshift64(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }

        let mut nonzero_count: u32 = 0;

        for _cube in 0..cubes {
            // Random affine offsets decorrelate independent cube samples.
            let affine = [
                xorshift64(&mut rng),
                xorshift64(&mut rng),
                xorshift64(&mut rng),
                xorshift64(&mut rng),
            ];

            // Maintain each cube point incrementally from the affine base.
            let mut current = [
                base[0] ^ affine[0],
                base[1] ^ affine[1],
                base[2] ^ affine[2],
                base[3] ^ affine[3],
            ];
            let mut prev_gray: u64 = 0;

            let mut acc = [0u64; 4];

            let mut processed: u64 = 0;

            while processed < total_points {
                let remaining     = (total_points - processed) as usize;
                let current_batch = remaining.min(batch_states);

                // Advance cube points using Gray-code transitions.
                for i in 0..current_batch {
                    let idx  = processed + i as u64;
                    let gray = idx ^ (idx >> 1);
                    let diff = gray ^ prev_gray;

                    if diff != 0 {
                        let changed       = diff.trailing_zeros() as usize;
                        let (word, mask)  = masks[changed];
                        current[word]    ^= mask;
                    }
                    prev_gray = gray;

                    let off = i * 4;
                    batch[off]     = current[0];
                    batch[off + 1] = current[1];
                    batch[off + 2] = current[2];
                    batch[off + 3] = current[3];
                }

                match crate::dispatch::evaluate(&mut batch[..current_batch * 4]) {
                    crate::dispatch::LustroError::Success => {},
                    _ => return Err("dispatch::evaluate failed in b51_prob_degree"),
                }

                // Compute the full-vector cube sum.
                for i in 0..current_batch {
                    let off = i * 4;
                    acc[0] ^= batch[off];
                    acc[1] ^= batch[off + 1];
                    acc[2] ^= batch[off + 2];
                    acc[3] ^= batch[off + 3];
                }

                processed += current_batch as u64;
            }

            // A nonzero cube sum indicates a nonzero degree-th derivative.
            if (acc[0] | acc[1] | acc[2] | acc[3]) != 0 {
                nonzero_count += 1;
            }
        }

        Ok(nonzero_count)
    });

    match result {
        Ok(v)  => Ok(v),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

// B53 — LINEAR CORRELATION / WALSH TEST
//
// Compute Walsh coefficients for selected input/output mask pairs.
#[pyfunction]
pub fn b53_linear_correlation<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u64>,
    masks: PyReadonlyArray2<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {

    let states_arr = states.as_array();
    let masks_arr  = masks.as_array();

    if states_arr.shape().len() != 2 || states_arr.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "states must have shape (n, 4)",
        ));
    }
    if masks_arr.shape().len() != 2 || masks_arr.shape()[1] != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "masks must have shape (k, 8)",
        ));
    }

    let n = states_arr.shape()[0];
    let k = masks_arr.shape()[0];

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("states empty"));
    }
    if k == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("masks empty"));
    }

    let states_slice = states.as_slice()?;
    let masks_slice  = masks.as_slice()?;

    let sums = py.allow_threads(|| {

        let mut outputs = states_slice.to_vec();

        match crate::dispatch::evaluate(&mut outputs) {
            crate::dispatch::LustroError::Success => {},
            e => panic!("dispatch::evaluate failed: {:?}", e),
        }

        let mut sums = vec![0i64; k];

        // Keep masks inner to reduce repeated mask loads.
        for i in 0..n {
            let off = i * 4;

            let x0 = states_slice[off];
            let x1 = states_slice[off + 1];
            let x2 = states_slice[off + 2];
            let x3 = states_slice[off + 3];

            let y0 = outputs[off];
            let y1 = outputs[off + 1];
            let y2 = outputs[off + 2];
            let y3 = outputs[off + 3];

            for (mask_idx, mask) in masks_slice.chunks_exact(8).enumerate() {
                // Exploit parity linearity over XOR.
                let px = ((x0 & mask[0]) ^ (x1 & mask[1]) ^ (x2 & mask[2]) ^ (x3 & mask[3]))
                    .count_ones() & 1;
                let py_bit = ((y0 & mask[4]) ^ (y1 & mask[5]) ^ (y2 & mask[6]) ^ (y3 & mask[7]))
                    .count_ones() & 1;
                sums[mask_idx] += 1 - ((px ^ py_bit) as i64 * 2);
            }
        }

        sums
    });

    Ok(sums.to_pyarray_bound(py))
}

// MC baseline variant — accepts precomputed outputs, no evaluate call.
#[pyfunction]
pub fn b53_walsh_precomputed<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArray2<'py, u64>,
    outputs: PyReadonlyArray2<'py, u64>,
    masks: PyReadonlyArray2<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {

    if inputs.shape()[1] != 4 || outputs.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inputs and outputs must have shape (n, 4)",
        ));
    }
    if masks.shape()[1] != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "masks must have shape (k, 8)",
        ));
    }
    if inputs.shape()[0] != outputs.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inputs and outputs must have the same row count",
        ));
    }

    let n = inputs.shape()[0];
    let k = masks.shape()[0];

    if n == 0 { return Err(pyo3::exceptions::PyValueError::new_err("inputs empty")); }
    if k == 0 { return Err(pyo3::exceptions::PyValueError::new_err("masks empty")); }

    let in_slice   = inputs.as_slice()?;
    let out_slice  = outputs.as_slice()?;
    let mask_slice = masks.as_slice()?;

    let sums = py.allow_threads(|| {
        let mut sums = vec![0i64; k];

        for i in 0..n {
            let off = i * 4;

            let x0 = in_slice[off];
            let x1 = in_slice[off + 1];
            let x2 = in_slice[off + 2];
            let x3 = in_slice[off + 3];

            let y0 = out_slice[off];
            let y1 = out_slice[off + 1];
            let y2 = out_slice[off + 2];
            let y3 = out_slice[off + 3];

            for (mask_idx, mask) in mask_slice.chunks_exact(8).enumerate() {
                // Exploit parity linearity over XOR.
                let px = ((x0 & mask[0]) ^ (x1 & mask[1]) ^ (x2 & mask[2]) ^ (x3 & mask[3]))
                    .count_ones() & 1;
                let py_bit = ((y0 & mask[4]) ^ (y1 & mask[5]) ^ (y2 & mask[6]) ^ (y3 & mask[7]))
                    .count_ones() & 1;
                sums[mask_idx] += 1 - ((px ^ py_bit) as i64 * 2);
            }
        }

        sums
    });

    Ok(sums.to_pyarray_bound(py))
}