#![allow(non_snake_case)]

use std::sync::OnceLock;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::api::{evaluate_scalar, stream_step};

// ==========================================
// CONFIG
// ==========================================

// One Lustro state = 4 x u64 = 32 bytes
const WORDS_PER_STATE: usize = 4;

// MT becomes worthwhile above this size
const MT_THRESHOLD_WORDS: usize = 24 * 1024;

// Rayon work chunk
const PARALLEL_CHUNK_WORDS: usize = 12 * 1024;

// ------------------------------------------
// THREAD TOPOLOGY FLAGS
// ------------------------------------------
// HT_BLOCK: true  = use physical cores only
//           false = use all logical threads
const HT_BLOCK: bool = false;

// PIN_THREADS: true  = pin each Rayon worker to a physical core
//              false = let the OS scheduler decide placement
const PIN_THREADS: bool = false;

// ==========================================
// ERROR
// ==========================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum LustroError {
    Success = 0,
    InvalidInput = 1,
    Panic = 2, // caught panic OR pool init failure — treat it as an internal execution failure
}

// ==========================================
// PANIC REPORTING
// ==========================================

// Default panic hook reports message/location; this adds the dispatch context.
fn report_panic(context: &str) {
    eprintln!("[lustro] panic in {context}");
}

// ==========================================
// CPU TOPOLOGY — NO EXTERNAL DEPENDENCIES
// ==========================================

// Detects HT ratio via CPUID leaf 0xB (Extended Topology).
static HT_RATIO: OnceLock<usize> = OnceLock::new();

fn ht_ratio() -> usize {
    *HT_RATIO.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            detect_ht_ratio_x86()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            1
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn detect_ht_ratio_x86() -> usize {
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    unsafe {
        let max_leaf = __cpuid(0).eax;
        if max_leaf < 0xB {
            return 1;
        }

        let mut level: u32 = 0;

        loop {
            let res = __cpuid_count(0xB, level);

            let level_type = (res.ecx >> 8) & 0xFF;
            let count = (res.ebx & 0xFFFF) as usize;

            if level_type == 1 && count > 0 {
                return count;
            }
            if count == 0 {
                break;
            }
            level += 1;
            if level > 8 {
                break;
            }
        }
        1
    }
}

// Uses raw OS API — no external crates required.
// On Windows: SetThreadAffinityMask.
#[cfg(all(target_os = "windows", feature = "thread-pinning"))]
fn pin_thread_to_core(core_id: usize) {
    extern "system" {
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn SetThreadAffinityMask(
            h_thread: *mut core::ffi::c_void,
            dw_thread_affinity_mask: usize,
        ) -> usize;
    }

    let mask = match 1usize.checked_shl(core_id as u32) {
        Some(m) if m != 0 => m,
        _ => return,
    };

    unsafe {
        let result = SetThreadAffinityMask(GetCurrentThread(), mask);
        debug_assert_ne!(result, 0, "SetThreadAffinityMask failed");
    }
}

#[cfg(not(all(target_os = "windows", feature = "thread-pinning")))]
#[inline(always)]
fn pin_thread_to_core(_core_id: usize) {}

// ==========================================
// RAYON POOL
// ==========================================

// NOTE: This topology model assumes uniform SMT ratio.
// Hybrid P/E-core CPUs (e.g. Intel Alder Lake) may not be represented perfectly.

// Pool is initialized once; init errors are retained.
static RAYON_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

fn build_pool() -> Result<rayon::ThreadPool, String> {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let ht_active = HT_BLOCK && std::env::var_os("LUSTRO_DISABLE_HT").is_none();

    let ratio = if ht_active { ht_ratio() } else { 1 };
    let physical = (logical / ratio).max(1);

    if physical <= 1 {
        return ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|e| e.to_string());
    }

    let pin_active = PIN_THREADS && std::env::var_os("LUSTRO_DISABLE_AFFINITY").is_none();

    ThreadPoolBuilder::new()
        .num_threads(physical)
        .start_handler(move |thread_id| {
            if !pin_active {
                return;
            }
            let core_id = thread_id.saturating_mul(ratio);
            if core_id < logical {
                pin_thread_to_core(core_id);
            }
        })
        .build()
        .map_err(|e| e.to_string())
}

pub fn lustro_init() {
    let _ = RAYON_POOL.get_or_init(build_pool);
}

// Shared pool accessor.
fn get_pool() -> Result<&'static rayon::ThreadPool, &'static String> {
    RAYON_POOL.get_or_init(build_pool).as_ref()
}

// ==========================================
// INTERNAL SCALAR PIPELINE
// ==========================================

#[inline(always)]
fn process_scalar(data: &mut [u64]) {
    debug_assert_eq!(
        data.len() % WORDS_PER_STATE,
        0,
        "process_scalar: data length must be a multiple of WORDS_PER_STATE"
    );
    for chunk in data.chunks_exact_mut(WORDS_PER_STATE) {
        let s0 = ((chunk[1] as u128) << 64) | (chunk[0] as u128);
        let s1 = ((chunk[3] as u128) << 64) | (chunk[2] as u128);

        let (o0, o1) = evaluate_scalar(s0, s1);

        chunk[0] = o0 as u64;
        chunk[1] = (o0 >> 64) as u64;
        chunk[2] = o1 as u64;
        chunk[3] = (o1 >> 64) as u64;
    }
}

#[inline(always)]
fn process_scalar_step(data: &mut [u64], step: u64) {
    debug_assert_eq!(
        data.len() % WORDS_PER_STATE,
        0,
        "process_scalar_step: data length must be a multiple of WORDS_PER_STATE"
    );
    for chunk in data.chunks_exact_mut(WORDS_PER_STATE) {
        let s0 = ((chunk[1] as u128) << 64) | (chunk[0] as u128);
        let s1 = ((chunk[3] as u128) << 64) | (chunk[2] as u128);

        let (o0, o1) = stream_step(s0, s1, step);

        chunk[0] = o0 as u64;
        chunk[1] = (o0 >> 64) as u64;
        chunk[2] = o1 as u64;
        chunk[3] = (o1 >> 64) as u64;
    }
}

// ==========================================
// INTERNAL PARALLEL PIPELINE
// ==========================================

#[inline(always)]
fn process_parallel(data: &mut [u64]) -> LustroError {
    let pool = match get_pool() {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("[lustro] pool init failed in process_parallel: {e}");
            return LustroError::Panic;
        }
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| {
            data.par_chunks_mut(PARALLEL_CHUNK_WORDS)
                .for_each(process_scalar);
        });
    }));
    if result.is_err() {
        report_panic("process_parallel");
        return LustroError::Panic;
    }
    LustroError::Success
}

#[inline(always)]
fn process_parallel_step(data: &mut [u64], step: u64) -> LustroError {
    let pool = match get_pool() {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("[lustro] pool init failed in process_parallel_step: {e}");
            return LustroError::Panic;
        }
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| {
            data.par_chunks_mut(PARALLEL_CHUNK_WORDS)
                .for_each(|chunk| process_scalar_step(chunk, step));
        });
    }));
    if result.is_err() {
        report_panic("process_parallel_step");
        return LustroError::Panic;
    }
    LustroError::Success
}

// ==========================================
// INTERNAL DISPATCH
// ==========================================

#[inline(always)]
fn dispatch(data: &mut [u64]) -> LustroError {
    if data.is_empty() {
        return LustroError::Success;
    }
    if !data.len().is_multiple_of(WORDS_PER_STATE) {
        return LustroError::InvalidInput;
    }
    if data.len() < MT_THRESHOLD_WORDS {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_scalar(data);
        }));
        if result.is_err() {
            report_panic("dispatch");
            return LustroError::Panic;
        }
        return LustroError::Success;
    }
    process_parallel(data)
}

#[inline(always)]
fn dispatch_step(data: &mut [u64], step: u64) -> LustroError {
    if data.is_empty() {
        return LustroError::Success;
    }
    if !data.len().is_multiple_of(WORDS_PER_STATE) {
        return LustroError::InvalidInput;
    }
    if data.len() < MT_THRESHOLD_WORDS {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_scalar_step(data, step);
        }));
        if result.is_err() {
            report_panic("dispatch_step");
            return LustroError::Panic;
        }
        return LustroError::Success;
    }
    process_parallel_step(data, step)
}

// ==========================================
// PUBLIC EVALUATE API
// ==========================================

// FULL EVALUATE
#[inline]
pub fn evaluate(data: &mut [u64]) -> LustroError {
    dispatch(data)
}

// SINGLE-THREADED EVALUATE
#[inline]
pub fn evaluate_st(data: &mut [u64]) -> LustroError {
    if data.is_empty() {
        return LustroError::Success;
    }
    if !data.len().is_multiple_of(WORDS_PER_STATE) {
        return LustroError::InvalidInput;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_scalar(data);
    }));
    if result.is_err() {
        report_panic("evaluate_st");
        return LustroError::Panic;
    }
    LustroError::Success
}

// FULL EVALUATE — STREAM (counter-driven, single step index for the whole call)
#[inline]
pub fn evaluate_step(data: &mut [u64], step: u64) -> LustroError {
    dispatch_step(data, step)
}

// SINGLE-THREADED EVALUATE — STREAM
#[inline]
pub fn evaluate_step_st(data: &mut [u64], step: u64) -> LustroError {
    if data.is_empty() {
        return LustroError::Success;
    }
    if !data.len().is_multiple_of(WORDS_PER_STATE) {
        return LustroError::InvalidInput;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_scalar_step(data, step);
    }));
    if result.is_err() {
        report_panic("evaluate_step_st");
        return LustroError::Panic;
    }
    LustroError::Success
}
