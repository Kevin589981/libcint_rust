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

// Re-export commonly used types
pub use types::{AtmSlot, BasSlot, Env, EnvVars};

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

/// Two-electron repulsion integral (ij|kl) in Cartesian basis.
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
    _opt:  *const std::ffi::c_void,
    _cache: *mut f64,
) -> i32 {
    int2e::int2e_cart(out, dims, shls, atm, natm, bas, nbas, env)
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

/// ERI optimizer stub.
#[no_mangle]
pub unsafe extern "C" fn int2e_optimizer(
    opt: *mut *mut std::ffi::c_void,
    _atm: *const i32, _natm: i32,
    _bas: *const i32, _nbas: i32,
    _env: *const f64,
) {
    if !opt.is_null() { *opt = std::ptr::null_mut(); }
}
