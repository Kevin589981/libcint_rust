//! High-level ERI driver: screening wrapper and parallel batch fill.
//!
//! This module provides two public functions on top of the bare
//! `int2e_cart_bare` primitive:
//!
//! * `int2e_cart` — single-quartet with optional Cauchy-Schwarz screening.
//! * `int2e_fill_cart` — fill the entire (nao × nao × nao × nao) ERI tensor
//!   in parallel using `rayon`.
//!
//! # Parallelism safety
//! The output tensor is a flat F-order array.  For shell quartet (I,J,K,L),
//! the written elements are exactly those AO indices
//! `a ∈ [lo_I,hi_I)`, `b ∈ [lo_J,hi_J)`, `c ∈ [lo_K,hi_K)`, `d ∈ [lo_L,hi_L)`.
//! Because AO ranges for distinct shells are non-overlapping, and the
//! F-order flat mapping is injective, different quartets write to disjoint
//! memory regions.  `rayon::par_iter` is therefore safe with raw-pointer
//! writes (wrapped via `SendPtr`).

use rayon::prelude::*;
use crate::types::{BasSlot, ncart};
use crate::optimizer::CINTOpt;
use super::eri::int2e_cart_bare;

// ─────────────────────────────────────────────────────────────────
// Single-quartet entry with optional CS screening
// ─────────────────────────────────────────────────────────────────

/// Compute `(ij|kl)` for one shell quartet, with optional CS screening.
///
/// If `opt` is `Some(o)` and the quartet fails the Schwarz bound, `out` is
/// zeroed and 0 is returned immediately without invoking the integral engine.
pub fn int2e_cart(
    out:  *mut f64,
    dims: *const i32,
    shls: *const i32,
    atm:  *const i32,
    natm: i32,
    bas:  *const i32,
    nbas: i32,
    env:  *const f64,
    opt:  Option<&CINTOpt>,
) -> i32 {
    // Cauchy-Schwarz shell-level screening
    if let Some(o) = opt {
        let s = unsafe { std::slice::from_raw_parts(shls, 4) };
        let (i, j, k, l) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
        if !o.passes(i, j, k, l) {
            // Zero output and return early
            let (nfi, nfj, nfk, nfl) = unsafe {
                let bi = BasSlot::from_raw(bas, i);
                let bj = BasSlot::from_raw(bas, j);
                let bk = BasSlot::from_raw(bas, k);
                let bl = BasSlot::from_raw(bas, l);
                (
                    ncart(bi.ang_of()) * bi.nctr_of(),
                    ncart(bj.ang_of()) * bj.nctr_of(),
                    ncart(bk.ang_of()) * bk.nctr_of(),
                    ncart(bl.ang_of()) * bl.nctr_of(),
                )
            };
            let n = nfi * nfj * nfk * nfl;
            unsafe { std::ptr::write_bytes(out, 0, n); }
            return 0;
        }
    }
    int2e_cart_bare(out, dims, shls, atm, natm, bas, nbas, env)
}

// ─────────────────────────────────────────────────────────────────
// Batch fill
// ─────────────────────────────────────────────────────────────────

/// Aggregates all raw pointer arguments for the parallel fill.
///
/// # Safety / thread-safety guarantee
/// * `atm`, `bas`, `env` are read-only → safe to read concurrently.
/// * `out` is written in non-overlapping regions per shell quartet.
/// * `ao_loc` is read-only after construction.
///
/// `unsafe impl Send + Sync` is sound under these invariants.
struct EriContext {
    atm:    *const i32,
    natm:   i32,
    bas:    *const i32,
    nbas:   i32,
    env:    *const f64,
    out:    *mut f64,
    ao_loc: Vec<usize>,
    nao:    usize,
    opt:    *const CINTOpt,
}
unsafe impl Send for EriContext {}
unsafe impl Sync for EriContext {}

/// Fill the full `(nao × nao × nao × nao)` ERI tensor in Fortran (column-major)
/// order using all available CPU cores.
///
/// `out` must point to an allocated buffer of at least `nao⁴` `f64` values,
/// zeroed by the caller.  `nao` is inferred from the basis set.
///
/// Shell quartets failing the Cauchy-Schwarz bound are skipped (their output
/// block remains zero).
///
/// # Safety
/// All pointer arguments must satisfy the same validity requirements as
/// `int2e_cart`.
pub unsafe fn int2e_fill_cart(
    out:  *mut f64,
    atm:  *const i32, natm: i32,
    bas:  *const i32, nbas: i32,
    env:  *const f64,
    opt:  *const CINTOpt,
) {
    let nb = nbas as usize;

    // Compute AO offsets (ao_loc[i] = sum of nctr*ncart for shells 0..i)
    let mut ao_loc = vec![0usize; nb + 1];
    for i in 0..nb {
        let bi = BasSlot::from_raw(bas, i);
        ao_loc[i + 1] = ao_loc[i] + ncart(bi.ang_of()) * bi.nctr_of();
    }
    let nao = ao_loc[nb];

    // Build flat list of all shell quartets
    let mut quartets: Vec<(usize, usize, usize, usize)> =
        Vec::with_capacity(nb * nb * nb * nb);
    for i in 0..nb {
        for j in 0..nb {
            for k in 0..nb {
                for l in 0..nb {
                    quartets.push((i, j, k, l));
                }
            }
        }
    }

    // All raw pointers are collected into one struct so that the closure
    // captures `ctx: &EriContext` (a reference), not individual raw fields.
    // Rust 2021's "disjoint capture" would decompose struct fields accessed
    // directly, but does NOT decompose through a shared reference.
    let ctx = EriContext { atm, natm, bas, nbas, env, out, ao_loc, nao, opt };
    let ctx_ref: &EriContext = &ctx;   // &EriContext is Send because EriContext: Sync

    quartets.par_iter().for_each(move |&(i, j, k, l)| {
        let EriContext { atm, natm, bas, nbas, env, out, ref ao_loc, nao, opt } = *ctx_ref;
        // Optional CS screening
        let opt_ref: Option<&CINTOpt> = if opt.is_null() { None } else { unsafe { Some(&*opt) } };
        if let Some(o) = opt_ref {
            if !o.passes(i, j, k, l) { return; }
        }

        let bi = unsafe { BasSlot::from_raw(bas, i) };
        let bj = unsafe { BasSlot::from_raw(bas, j) };
        let bk = unsafe { BasSlot::from_raw(bas, k) };
        let bl = unsafe { BasSlot::from_raw(bas, l) };

        let nfi = ncart(bi.ang_of()) * bi.nctr_of();
        let nfj = ncart(bj.ang_of()) * bj.nctr_of();
        let nfk = ncart(bk.ang_of()) * bk.nctr_of();
        let nfl = ncart(bl.ang_of()) * bl.nctr_of();

        // Compute integral into a local buffer
        let mut tmp = vec![0.0_f64; nfi * nfj * nfk * nfl];
        let shls = [i as i32, j as i32, k as i32, l as i32];
        unsafe {
            int2e_cart_bare(
                tmp.as_mut_ptr(),
                std::ptr::null(),
                shls.as_ptr(),
                atm, natm, bas, nbas, env,
            );
        }

        // Scatter tmp → global F-order tensor
        // tmp[a + nfi*(b + nfj*(c + nfk*d))] →
        //     out[lo_i+a + nao*(lo_j+b + nao*(lo_k+c + nao*(lo_l+d)))]
        let lo_i = ao_loc[i]; let lo_j = ao_loc[j];
        let lo_k = ao_loc[k]; let lo_l = ao_loc[l];

        for d in 0..nfl {
            for c in 0..nfk {
                for b in 0..nfj {
                    for a in 0..nfi {
                        let src_idx = a + nfi * (b + nfj * (c + nfk * d));
                        let dst_idx = (lo_i + a)
                            + nao * ((lo_j + b)
                            + nao * ((lo_k + c)
                            + nao * (lo_l + d)));
                        unsafe { *out.add(dst_idx) = tmp[src_idx]; }
                    }
                }
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal H-like 2-centre system and verify that `int2e_fill_cart`
    /// produces the same result as calling `int2e_cart_bare` for every quartet.
    #[test]
    fn fill_matches_bare_h2() {
        // Two s-shells at different centres (same as in the POC validation)
        let mut env = vec![0.0_f64; 40];
        env[20] = 0.0; env[21] = 0.0; env[22] = 0.0; // atom 0 at origin
        env[23] = 0.0; env[24] = 0.0; env[25] = 1.4; // atom 1 at 1.4 a0
        env[26] = 1.24;  // exponent (STO-3G-like)
        env[27] = 1.0;   // contraction coefficient

        let atm: Vec<i32> = vec![
            1, 20, 1, 0, 0, 0,
            1, 23, 1, 0, 0, 0,
        ];
        let bas: Vec<i32> = vec![
            0, 0, 1, 1, 0, 26, 27, 0,  // shell 0: s on atom 0
            1, 0, 1, 1, 0, 26, 27, 0,  // shell 1: s on atom 1
        ];
        let nbas: i32 = 2;
        let natm: i32 = 2;
        let nao: usize = 2;  // 1 s-function per shell

        // Fill via batch driver
        let mut full = vec![0.0_f64; nao * nao * nao * nao];
        unsafe {
            int2e_fill_cart(
                full.as_mut_ptr(),
                atm.as_ptr(), natm,
                bas.as_ptr(), nbas,
                env.as_ptr(),
                std::ptr::null(),
            );
        }

        // Compare with per-quartet bare calls
        for i in 0..nbas as usize {
            for j in 0..nbas as usize {
                for k in 0..nbas as usize {
                    for l in 0..nbas as usize {
                        let shls = [i as i32, j as i32, k as i32, l as i32];
                        let mut tmp = vec![0.0_f64; 1];
                        unsafe {
                            int2e_cart_bare(
                                tmp.as_mut_ptr(),
                                std::ptr::null(),
                                shls.as_ptr(),
                                atm.as_ptr(), natm,
                                bas.as_ptr(), nbas,
                                env.as_ptr(),
                            );
                        }
                        let idx = i + nao * (j + nao * (k + nao * l));
                        assert!(
                            (full[idx] - tmp[0]).abs() < 1e-14,
                            "mismatch at ({},{},{},{}): fill={:.6e} bare={:.6e}",
                            i, j, k, l, full[idx], tmp[0]
                        );
                    }
                }
            }
        }
    }
}
