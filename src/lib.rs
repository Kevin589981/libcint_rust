//! libcint-rs: A Rust reimplementation of libcint with C-ABI compatibility.
//!
//! This crate exposes C-compatible integral functions that can be used as a
//! drop-in replacement for libcint by PySCF (via ctypes) for s and p shells.
//!
//! # ABI
//! All functions follow the libcint signature:
//! ```c
//! CACHE_SIZE_T int{1,2}e_*_cart(double *out, int *dims, int *shls,
//!     int *atm, int natm, int *bas, int nbas, double *env,
//!     CINTOpt *opt, double *cache);
//! ```
//! `opt` and `cache` are ignored (pass NULL from Python).

#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

pub mod types;
pub mod rys;
pub mod recur;
pub mod transform;
pub mod int1e;
pub mod int2e;
pub mod optimizer;

// Re-export commonly used types
pub use types::{AtmSlot, BasSlot, Env, EnvVars};
pub use optimizer::CINTOpt;

// ─────────────────────────────────────────────────────────────────
// Helper: turn opaque opt pointer into Option<&CINTOpt>
// ─────────────────────────────────────────────────────────────────
#[inline]
unsafe fn opt_ref(opt: *const std::ffi::c_void) -> Option<&'static CINTOpt> {
    if opt.is_null() { None } else { Some(&*(opt as *const CINTOpt)) }
}

// ─────────────────────────────────────────────────────────────────
// C-ABI exports
// ─────────────────────────────────────────────────────────────────

/// Overlap integral <i|j> (Cartesian basis).
#[no_mangle]
pub unsafe extern "C" fn int1e_ovlp_cart(
    out:   *mut f64,
    dims:  *const i32,
    shls:  *const i32,
    atm:   *const i32,
    natm:  i32,
    bas:   *const i32,
    nbas:  i32,
    env:   *const f64,
    _opt:  *const std::ffi::c_void,
    _cache: *mut f64,
) -> i32 {
    int1e::int1e_ovlp_cart(out, dims, shls, atm, natm, bas, nbas, env)
}

/// Kinetic energy integral <i|−½∇²|j> (Cartesian basis).
#[no_mangle]
pub unsafe extern "C" fn int1e_kin_cart(
    out:   *mut f64,
    dims:  *const i32,
    shls:  *const i32,
    atm:   *const i32,
    natm:  i32,
    bas:   *const i32,
    nbas:  i32,
    env:   *const f64,
    _opt:  *const std::ffi::c_void,
    _cache: *mut f64,
) -> i32 {
    int1e::int1e_kin_cart(out, dims, shls, atm, natm, bas, nbas, env)
}

/// Nuclear attraction integral <i|V_nuc|j> (Cartesian basis).
#[no_mangle]
pub unsafe extern "C" fn int1e_nuc_cart(
    out:   *mut f64,
    dims:  *const i32,
    shls:  *const i32,
    atm:   *const i32,
    natm:  i32,
    bas:   *const i32,
    nbas:  i32,
    env:   *const f64,
    _opt:  *const std::ffi::c_void,
    _cache: *mut f64,
) -> i32 {
    int1e::int1e_nuc_cart(out, dims, shls, atm, natm, bas, nbas, env)
}

/// Two-electron repulsion integral (ij|kl) in Cartesian basis, with optional CS screening.
#[no_mangle]
pub unsafe extern "C" fn int2e_cart(
    out:   *mut f64,
    dims:  *const i32,
    shls:  *const i32,
    atm:   *const i32,
    natm:  i32,
    bas:   *const i32,
    nbas:  i32,
    env:   *const f64,
    opt:   *const std::ffi::c_void,
    _cache: *mut f64,
) -> i32 {
    int2e::int2e_cart(out, dims, shls, atm, natm, bas, nbas, env, opt_ref(opt))
}

/// Overlap optimizer stub (libcint compat — sets opt to NULL).
#[no_mangle]
pub unsafe extern "C" fn int1e_ovlp_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}

/// Nuclear optimizer stub.
#[no_mangle]
pub unsafe extern "C" fn int1e_nuc_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}

/// Build Cauchy-Schwarz screening table for the given molecular system.
///
/// Allocates a `CINTOpt` on the heap and stores its pointer at `*opt`.
#[no_mangle]
pub unsafe extern "C" fn int2e_optimizer(
    opt:  *mut *mut CINTOpt,
    atm:  *const i32, natm: i32,
    bas:  *const i32, nbas: i32,
    env:  *const f64,
) {
    if opt.is_null() { return; }
    let new_opt = CINTOpt::build(atm, natm, bas, nbas, env);
    *opt = Box::into_raw(new_opt);
}

/// Free a `CINTOpt` previously allocated by `int2e_optimizer`.
#[no_mangle]
pub unsafe extern "C" fn CINTdel_optimizer(opt: *mut *mut CINTOpt) {
    if opt.is_null() || (*opt).is_null() { return; }
    let _ = Box::from_raw(*opt);
    *opt = std::ptr::null_mut();
}

// ─────────────────────────────────────────────────────────────────
// Spherical-harmonic variants
//
// PySCF calls the _sph functions by default.  For shells with l ≤ 1 the
// Cartesian → spherical transformation is the identity, so we simply
// forward to the Cartesian implementation.  For l ≥ 2 a full cart2sph
// contraction would be needed here; that path is guarded by the transform
// module and will panic for l > 2 until implemented.
// ─────────────────────────────────────────────────────────────────

/// Overlap integral (spherical basis) — for l ≤ 1 identical to Cartesian.
#[no_mangle]
pub unsafe extern "C" fn int1e_ovlp_sph(
    out: *mut f64, dims: *const i32, shls: *const i32,
    atm: *const i32, natm: i32, bas: *const i32, nbas: i32,
    env: *const f64, _opt: *const std::ffi::c_void, _cache: *mut f64,
) -> i32 {
    int1e::int1e_ovlp_cart(out, dims, shls, atm, natm, bas, nbas, env)
}

/// Kinetic energy integral (spherical basis) — for l ≤ 1 identical to Cartesian.
#[no_mangle]
pub unsafe extern "C" fn int1e_kin_sph(
    out: *mut f64, dims: *const i32, shls: *const i32,
    atm: *const i32, natm: i32, bas: *const i32, nbas: i32,
    env: *const f64, _opt: *const std::ffi::c_void, _cache: *mut f64,
) -> i32 {
    int1e::int1e_kin_cart(out, dims, shls, atm, natm, bas, nbas, env)
}

/// Nuclear attraction integral (spherical basis) — for l ≤ 1 identical to Cartesian.
#[no_mangle]
pub unsafe extern "C" fn int1e_nuc_sph(
    out: *mut f64, dims: *const i32, shls: *const i32,
    atm: *const i32, natm: i32, bas: *const i32, nbas: i32,
    env: *const f64, _opt: *const std::ffi::c_void, _cache: *mut f64,
) -> i32 {
    int1e::int1e_nuc_cart(out, dims, shls, atm, natm, bas, nbas, env)
}

/// ERI (spherical basis) — for l ≤ 1 identical to Cartesian, with optional CS screening.
#[no_mangle]
pub unsafe extern "C" fn int2e_sph(
    out: *mut f64, dims: *const i32, shls: *const i32,
    atm: *const i32, natm: i32, bas: *const i32, nbas: i32,
    env: *const f64, opt: *const std::ffi::c_void, _cache: *mut f64,
) -> i32 {
    int2e::int2e_cart(out, dims, shls, atm, natm, bas, nbas, env, opt_ref(opt))
}

// ─── Optimizer stubs for sph/kin variants ───────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn int1e_kin_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}

#[no_mangle]
pub unsafe extern "C" fn int1e_ovlp_sph_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}

#[no_mangle]
pub unsafe extern "C" fn int1e_kin_sph_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}

#[no_mangle]
pub unsafe extern "C" fn int1e_nuc_sph_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}

#[no_mangle]
pub unsafe extern "C" fn int2e_sph_optimizer(
    opt:  *mut *mut CINTOpt,
    atm:  *const i32, natm: i32,
    bas:  *const i32, nbas: i32,
    env:  *const f64,
) {
    int2e_optimizer(opt, atm, natm, bas, nbas, env);
}

// ─────────────────────────────────────────────────────────────────
// Batch ERI fill (Phase 2 — rayon parallel)
// ─────────────────────────────────────────────────────────────────

/// Fill the full `nao⁴` ERI tensor in F-order using all CPU cores.
///
/// `out` must be pre-allocated (`nao*nao*nao*nao` f64s) and zeroed.
/// Pass `opt` from `int2e_optimizer` to enable CS screening, or NULL.
#[no_mangle]
pub unsafe extern "C" fn int2e_fill_cart(
    out:  *mut f64,
    atm:  *const i32, natm: i32,
    bas:  *const i32, nbas: i32,
    env:  *const f64,
    opt:  *const CINTOpt,
) {
    int2e::int2e_fill_cart(out, atm, natm, bas, nbas, env, opt)
}
