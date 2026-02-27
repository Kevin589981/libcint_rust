//! Cartesian to real spherical harmonic (RSH) transformation.
//!
//! Only l = 0 and l = 1 are needed for the POC (s/p shells).
//! For l ≤ 1 the number of Cartesian and spherical functions is the same,
//! so the "transform" is just a harmless multiplication by the matrix below.
//!
//! libcint Cartesian order for p: (x, y, z) — matches (px, py, pz).
//! RSH order for p: (p_{-1}, p_0, p_{+1}) in the complex basis,
//! but for the **real** spherical harmonics in libcint / PySCF (and in the
//! Cartesian integral routines with CINTc2s_bra/ket_sph) the ordering is
//! simply (x, y, z) ↦ (x, y, z) with unit coefficients, because the real
//! spherical harmonics for l=1 are exactly proportional to px/py/pz.
//!
//! The matrix for l=1 (from libcint's cart2sph.c autocode section) is:
//!   RSH_x  = cart_x   (coefficient 1)
//!   RSH_y  = cart_y   (coefficient 1)
//!   RSH_z  = cart_z   (coefficient 1)
//! Hence for both l=0 and l=1 the transformation is trivially the identity.

/// Apply the Cartesian → real spherical harmonic transformation in-place.
///
/// `buf`:  Cartesian buffer with shape `[n_cart × n_right]` (column-major, like libcint).
/// `l`:    angular momentum of the shell being transformed.
///
/// For l > 1 this function panics (not yet implemented).
pub fn cart2sph_inplace(buf: &mut [f64], l: usize, n_right: usize) {
    match l {
        0 | 1 => { /* identity — no work needed */ }
        _ => unimplemented!("cart2sph not implemented for l > 1 in POC"),
    }
    let _ = n_right;
}

/// Transform a Cartesian block `gcart` of shape `[n_bra × nfi]` (n_bra fast)
/// into a spherical block `gsph` of shape `[n_bra × nsi]` (n_bra fast).
///
/// For l = 0 and l = 1 the transform is the identity (same size), so this just copies.
pub fn cart2sph_block(gsph: &mut [f64], gcart: &[f64], l: usize, n_bra: usize) {
    match l {
        0 | 1 => {
            let nf = (l + 1) * (l + 2) / 2;   // = 2*l+1 for l≤1
            debug_assert_eq!(nf, 2 * l + 1, "l≤1: nfc == nfs");
            let n = n_bra * nf;
            debug_assert!(gsph.len() >= n && gcart.len() >= n);
            gsph[..n].copy_from_slice(&gcart[..n]);
        }
        _ => unimplemented!("cart2sph not implemented for l > 1 in POC"),
    }
}

// ─────────────────────────────────────────────────────────────────
// Transformation matrices for reference (not used in l≤1 identity path)
// ─────────────────────────────────────────────────────────────────

/// Return the column-major transformation matrix `cart → sph` for angular momentum `l`.
/// Shape: `[nfc × nfs]` where `nfc = (l+1)(l+2)/2` and `nfs = 2l+1`.
///
/// Only l = 0 and l = 1 are implemented (both are identities).
pub fn c2s_matrix(l: usize) -> Vec<f64> {
    match l {
        0 => vec![1.0],
        1 => vec![
            // 3×3 identity (column-major):
            // cart order: x, y, z;  sph order: x, y, z (same)
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        _ => unimplemented!("c2s_matrix: l > 1 not implemented"),
    }
}
