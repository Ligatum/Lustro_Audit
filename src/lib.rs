use pyo3::prelude::*;

pub mod api;
mod core;
pub mod custom_tests;
pub mod dispatch;

use custom_tests::{
    b13_st_layout, b13_st_pairs_meta, b13_structured_truncated, b16_jacobian_diff, b22_rot_chain,
    b22_rot_chain_256, b2_conditional_sac, b2_evaluate_flipped, b2_evaluate_flipped_lanes,
    b32_cycle_signature, b32_find_dp, b32_isotropy, b32_orbit_mixing, b32_simulate_h0,
    b32e_state_space_profile, b32f_distance_retention, b51_exact_degree, b51_prob_degree,
    b53_linear_correlation, b53_walsh_precomputed, evaluate_baseline_inplace,
};

#[pymodule]
fn lustro_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_baseline_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(b16_jacobian_diff, m)?)?;
    m.add_function(wrap_pyfunction!(b2_evaluate_flipped, m)?)?;
    m.add_function(wrap_pyfunction!(b2_evaluate_flipped_lanes, m)?)?;
    m.add_function(wrap_pyfunction!(b2_conditional_sac, m)?)?;
    m.add_function(wrap_pyfunction!(b13_structured_truncated, m)?)?;
    m.add_function(wrap_pyfunction!(b13_st_pairs_meta, m)?)?;
    m.add_function(wrap_pyfunction!(b13_st_layout, m)?)?;
    m.add_function(wrap_pyfunction!(b22_rot_chain, m)?)?;
    m.add_function(wrap_pyfunction!(b22_rot_chain_256, m)?)?;
    m.add_function(wrap_pyfunction!(b32_isotropy, m)?)?;
    m.add_function(wrap_pyfunction!(b32_cycle_signature, m)?)?;
    m.add_function(wrap_pyfunction!(b32_find_dp, m)?)?;
    m.add_function(wrap_pyfunction!(b32_orbit_mixing, m)?)?;
    m.add_function(wrap_pyfunction!(b32_simulate_h0, m)?)?;
    m.add_function(wrap_pyfunction!(b32e_state_space_profile, m)?)?;
    m.add_function(wrap_pyfunction!(b32f_distance_retention, m)?)?;
    m.add_function(wrap_pyfunction!(b51_exact_degree, m)?)?;
    m.add_function(wrap_pyfunction!(b51_prob_degree, m)?)?;
    m.add_function(wrap_pyfunction!(b53_linear_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(b53_walsh_precomputed, m)?)?;

    Ok(())
}

// ===========================================================================================
// These wrappers are not production implementations and are intended to stress the engine
// in its primal form. They exists solely to evaluate raw statistical behavior of Lustro Core.
// ===========================================================================================
// ====================================================================
// SMHASHER TEST ADAPTER
// ====================================================================

/// # Safety
/// `key` must be null or valid for reads of `len` bytes (`len` must be >= 0).
/// `out` must be null or valid for writes of 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn lustro_smhasher_core(
    key: *const u8,
    len: i32,
    seed: u32,
    out: *mut u8,
) {
    if key.is_null() || out.is_null() || len < 0 {
        return;
    }
    let mut s0 = (seed as u128) | ((seed as u128) << 32);
    s0 |= s0 << 64;
    s0 ^= 0x55555555555555555555555555555555;
    let mut s1 = s0 ^ 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA;
    let key_slice = unsafe { std::slice::from_raw_parts(key, len as usize) };
    let chunks = key_slice.chunks_exact(32);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let b0 = u128::from_le_bytes(chunk[0..16].try_into().unwrap());
        let b1 = u128::from_le_bytes(chunk[16..32].try_into().unwrap());
        s0 ^= b0;
        s1 ^= b1;
        let (next_s0, next_s1) = crate::api::evaluate_scalar(s0, s1);
        s0 = next_s0;
        s1 = next_s1;
    }
    let length_bits = (len as u64) * 8;
    let len_bytes = length_bits.to_le_bytes();
    // Single-block padding requires: remainder + 0x80 (1B) + length (8B) <= 32
    // Therefore: remainder.len() <= 23
    if remainder.len() <= 23 {
        let mut buffer = [0u8; 32];
        buffer[..remainder.len()].copy_from_slice(remainder);
        buffer[remainder.len()] = 0x80;
        buffer[24..32].copy_from_slice(&len_bytes);

        let b0 = u128::from_le_bytes(buffer[0..16].try_into().unwrap());
        let b1 = u128::from_le_bytes(buffer[16..32].try_into().unwrap());
        s0 ^= b0;
        s1 ^= b1;
        let (next_s0, next_s1) = crate::api::evaluate_scalar(s0, s1);
        s0 = next_s0;
        s1 = next_s1;
    } else {
        let mut buffer1 = [0u8; 32];
        buffer1[..remainder.len()].copy_from_slice(remainder);
        buffer1[remainder.len()] = 0x80;
        let b0 = u128::from_le_bytes(buffer1[0..16].try_into().unwrap());
        let b1 = u128::from_le_bytes(buffer1[16..32].try_into().unwrap());
        s0 ^= b0;
        s1 ^= b1;
        let (next_s0, next_s1) = crate::api::evaluate_scalar(s0, s1);
        s0 = next_s0;
        s1 = next_s1;
        let mut buffer2 = [0u8; 32];
        buffer2[24..32].copy_from_slice(&len_bytes);
        let b0 = u128::from_le_bytes(buffer2[0..16].try_into().unwrap());
        let b1 = u128::from_le_bytes(buffer2[16..32].try_into().unwrap());
        s0 ^= b0;
        s1 ^= b1;
        let (next_s0, next_s1) = crate::api::evaluate_scalar(s0, s1);
        s0 = next_s0;
        s1 = next_s1;
    }

    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, 32) };
    out_slice[0..16].copy_from_slice(&s0.to_le_bytes());
    out_slice[16..32].copy_from_slice(&s1.to_le_bytes());
}

// ====================================================================
// PRACTRAND, BIGCRUSH, NIST - PRNG STREAM TEST ADAPTER
// ====================================================================

#[repr(C)]
pub struct LustroPrngContext {
    s0: u128,
    s1: u128,
    step: u64,
    buffer: [u32; 8],
    buf_idx: usize,
}

#[no_mangle]
pub extern "C" fn lustro_prng_create(seed: u32) -> *mut LustroPrngContext {
    let mut s0 = (seed as u128) | ((seed as u128) << 32);
    s0 |= s0 << 64;
    s0 ^= 0x55555555555555555555555555555555;
    let s1 = s0 ^ 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA;

    let ctx = Box::new(LustroPrngContext {
        s0,
        s1,
        step: 0,
        buffer: [0; 8],
        buf_idx: 8,
    });

    Box::into_raw(ctx)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32(ctx_ptr: *mut LustroPrngContext) -> u32 {
    if ctx_ptr.is_null() {
        return 0;
    }

    let ctx = unsafe { &mut *ctx_ptr };

    if ctx.buf_idx >= 8 {
        let (next_s0, next_s1) = crate::api::stream_step(ctx.s0, ctx.s1, ctx.step);

        ctx.s0 = next_s0;
        ctx.s1 = next_s1;

        // Packets may also be interleaved, instead of serialized
        ctx.buffer[0] = ctx.s0 as u32;
        ctx.buffer[1] = (ctx.s0 >> 32) as u32;
        ctx.buffer[2] = (ctx.s0 >> 64) as u32;
        ctx.buffer[3] = (ctx.s0 >> 96) as u32;
        ctx.buffer[4] = ctx.s1 as u32;
        ctx.buffer[5] = (ctx.s1 >> 32) as u32;
        ctx.buffer[6] = (ctx.s1 >> 64) as u32;
        ctx.buffer[7] = (ctx.s1 >> 96) as u32;

        ctx.step = ctx.step.wrapping_add(1);
        ctx.buf_idx = 0;
    }

    let val = ctx.buffer[ctx.buf_idx];
    ctx.buf_idx += 1;
    val
}

/// # Safety
/// `ctx_ptr` must be null or a pointer previously returned by `lustro_prng_create`,
/// not already freed. Must not be used again after this call.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_free(ctx_ptr: *mut LustroPrngContext) {
    if !ctx_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx_ptr);
        }
    }
}

// TEMPORAL LANE ADAPTERS (lane0..lane7)

#[inline(always)]
fn extract_lane(s0: u128, s1: u128, lane: u8) -> u32 {
    match lane {
        0 => s0 as u32,
        1 => (s0 >> 32) as u32,
        2 => (s0 >> 64) as u32,
        3 => (s0 >> 96) as u32,
        4 => s1 as u32,
        5 => (s1 >> 32) as u32,
        6 => (s1 >> 64) as u32,
        7 => (s1 >> 96) as u32,
        _ => unreachable!(),
    }
}

#[inline(always)]
fn next_u32_lane_impl(ctx_ptr: *mut LustroPrngContext, lane: u8) -> u32 {
    if ctx_ptr.is_null() {
        return 0;
    }
    let ctx = unsafe { &mut *ctx_ptr };
    let (next_s0, next_s1) = crate::api::stream_step(ctx.s0, ctx.s1, ctx.step);
    ctx.s0 = next_s0;
    ctx.s1 = next_s1;
    ctx.step = ctx.step.wrapping_add(1);
    extract_lane(ctx.s0, ctx.s1, lane)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane0(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 0)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane1(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 1)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane2(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 2)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane3(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 3)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane4(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 4)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane5(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 5)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane6(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 6)
}

/// # Safety
/// `ctx_ptr` must be null or a valid pointer returned by `lustro_prng_create`
/// and not yet passed to `lustro_prng_free`.
#[no_mangle]
pub unsafe extern "C" fn lustro_prng_next_u32_lane7(ctx_ptr: *mut LustroPrngContext) -> u32 {
    next_u32_lane_impl(ctx_ptr, 7)
}
