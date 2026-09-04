// PYTHON BINDINGS LAYER (PYO3)
// PURPOSE: SAFE / PYTHON-FRIENDLY ACCESS TO SCALAR DISPATCH API
// NOTE: ALL SEMANTICS DELEGATED TO dispatch.rs (NO LOGIC HERE)
#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};

use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, ToPyArray};

use crate::dispatch;
use crate::dispatch::LustroError;

// Base seeds for MT chain generation — each chain derives unique seeds from index
const SEED_S0_BASE: u128 = 0x1234567890ABCDEF1234567890ABCDEF_u128;
const SEED_S1_BASE: u128 = 0xFEDCBA0987654321FEDCBA0987654321_u128;

// ==========================================
// PYTHON CLASS
// ==========================================

#[pyclass]
pub struct LustroCoreV1Py;

// ==========================================
// PYTHON METHODS
// ==========================================

#[pymethods]
impl LustroCoreV1Py {

    #[new]
    pub fn new() -> Self {
        Self
    }

    // ==========================================
    // LIFECYCLE
    // ==========================================

    pub fn lustro_init(&self, py: Python<'_>) {
        py.allow_threads(|| {
            crate::dispatch::lustro_init();
        });
    }

    // ==========================================
    // BATCH — COPY
    // ==========================================

    // FULL PIPELINE (IDM + 1x ERD) — COPY VERSION
    pub fn evaluate<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyArray1<u64>>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {

        let slice = unsafe { data.as_slice()? };

        if slice.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "Input length must be a multiple of 4"
            ));
        }

        let mut buffer = slice.to_vec();

        let result = py.allow_threads(|| {
            dispatch::evaluate(&mut buffer)
        });
        match result {
            LustroError::Success => Ok(buffer.to_pyarray_bound(py)),
            e => Err(PyRuntimeError::new_err(format!("LustroError: {:?}", e))),
        }
    }

    // FULL PIPELINE (IDM + 1x ERD) — SINGLE-THREADED COPY VERSION
    pub fn evaluate_st<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyArray1<u64>>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {

        let slice = unsafe { data.as_slice()? };

        if slice.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "Input length must be a multiple of 4"
            ));
        }

        let mut buffer = slice.to_vec();

        let result = py.allow_threads(|| {
            dispatch::evaluate_st(&mut buffer)
        });
        match result {
            LustroError::Success => Ok(buffer.to_pyarray_bound(py)),
            e => Err(PyRuntimeError::new_err(format!("LustroError: {:?}", e))),
        }
    }

    // ==========================================
    // BATCH — INPLACE
    // ==========================================

    // FULL PIPELINE (IDM + 1x ERD) — INPLACE VERSION
    pub fn evaluate_inplace(
        &self,
        py: Python<'_>,
        data: &Bound<PyArray1<u64>>,
    ) -> PyResult<()> {

        let writeable: bool = data
            .getattr("flags")?
            .getattr("writeable")?
            .extract()?;

        if !writeable {
            return Err(PyValueError::new_err(
                "array must be writeable"
            ));
        }

        let mut rw = data.readwrite();
        let slice = rw.as_slice_mut()?;

        if slice.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "Input length must be a multiple of 4"
            ));
        }

        let result = py.allow_threads(|| {
            dispatch::evaluate(slice)
        });
        match result {
            LustroError::Success => Ok(()),
            e => Err(PyRuntimeError::new_err(format!("LustroError: {:?}", e))),
        }
    }

    // FULL PIPELINE (IDM + 1x ERD) — SINGLE-THREADED INPLACE VERSION
    pub fn evaluate_inplace_st(
        &self,
        py: Python<'_>,
        data: &Bound<PyArray1<u64>>,
    ) -> PyResult<()> {

        let writeable: bool = data
            .getattr("flags")?
            .getattr("writeable")?
            .extract()?;

        if !writeable {
            return Err(PyValueError::new_err(
                "array must be writeable"
            ));
        }

        let mut rw = data.readwrite();
        let slice = rw.as_slice_mut()?;

        if slice.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "Input length must be a multiple of 4"
            ));
        }

        let result = py.allow_threads(|| {
            dispatch::evaluate_st(slice)
        });
        match result {
            LustroError::Success => Ok(()),
            e => Err(PyRuntimeError::new_err(format!("LustroError: {:?}", e))),
        }
    }

    // ==========================================
    // PRNG CHAIN - INPLACE
    // ==========================================

    // Each step: IDM + ERD with counter-based perturbation.
    #[pyo3(text_signature = "(s0, s1, data)")]
    pub fn generate_chain_inplace(
        &self,
        py: Python<'_>,
        s0: u128,
        s1: u128,
        data: &Bound<PyArray2<u64>>,
    ) -> PyResult<()> {
        let writeable: bool = data
            .getattr("flags")?
            .getattr("writeable")?
            .extract()?;

        if !writeable {
            return Err(PyValueError::new_err("array must be writeable"));
        }

        let mut rw = data.readwrite();
        let slice = rw.as_slice_mut()?;

        if slice.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "array length must be a multiple of 4"
            ));
        }

        py.allow_threads(|| {
            let mut cur_s0 = s0;
            let mut cur_s1 = s1;
            let mut step: u64 = 0;

            for chunk in slice.chunks_exact_mut(4) {
                let (next_s0, next_s1) =
                    crate::api::stream_step(cur_s0, cur_s1, step);

                cur_s0 = next_s0;
                cur_s1 = next_s1;

                chunk[0] = cur_s0 as u64;
                chunk[1] = (cur_s0 >> 64) as u64;
                chunk[2] = cur_s1 as u64;
                chunk[3] = (cur_s1 >> 64) as u64;

                step = step.wrapping_add(1);
            }
        });

        Ok(())
    }

    // ==========================================
    // PRNG MULTI-CHAIN EVOLUTION — INPLACE
    // ==========================================

    // Runs N independent chains in parallel using Rayon global pool.
    // Each chain gets its own seed derived from chain index.
    #[pyo3(text_signature = "(chain_length, data)")]
    pub fn generate_chains_mc_inplace(
        &self,
        py: Python<'_>,
        chain_length: usize,
        data: &Bound<PyArray2<u64>>,
    ) -> PyResult<()> {
        if chain_length == 0 {
            return Err(PyValueError::new_err("chain_length must be > 0"));
        }

        let writeable: bool = data
            .getattr("flags")?
            .getattr("writeable")?
            .extract()?;

        if !writeable {
            return Err(PyValueError::new_err("array must be writeable"));
        }

        let mut rw = data.readwrite();
        let slice = rw.as_slice_mut()?;

        if slice.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "array length must be a multiple of 4"
            ));
        }

        let total_states = slice.len() / 4;

        if total_states % chain_length != 0 {
            return Err(PyValueError::new_err(
                "total states must be a multiple of chain_length"
            ));
        }

        let words_per_chain = chain_length * 4;

        py.allow_threads(|| {
            use rayon::prelude::ParallelSliceMut;
            use rayon::prelude::IndexedParallelIterator;
            use rayon::prelude::ParallelIterator;

            slice
                .par_chunks_exact_mut(words_per_chain)
                .enumerate()
                .for_each(|(chain_idx, chunk): (usize, &mut [u64])| {

                    let mix: u128 = chain_idx as u128;
                    let s0: u128 = SEED_S0_BASE
                        ^ mix.wrapping_mul(0x9E3779B97F4A7C15u128);
                    let s1: u128 = SEED_S1_BASE
                        ^ mix.wrapping_mul(0xD6E8FEB86659FD93u128);

                    let mut cur_s0 = s0;
                    let mut cur_s1 = s1;
                    let mut step: u64 = 0;

                    for c in chunk.chunks_exact_mut(4) {
                        let (next_s0, next_s1) =
                            crate::api::stream_step(cur_s0, cur_s1, step);

                        cur_s0 = next_s0;
                        cur_s1 = next_s1;

                        c[0] = cur_s0 as u64;
                        c[1] = (cur_s0 >> 64) as u64;
                        c[2] = cur_s1 as u64;
                        c[3] = (cur_s1 >> 64) as u64;

                        step = step.wrapping_add(1);
                    }
                });
        });

        Ok(())
    }
}