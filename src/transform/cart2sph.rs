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
/// For l ≤ 1 the transform is the identity (nCart == nSph).
/// For l = 2 a 6→5 contraction into `out` is done in-place (requires caller
/// to provide a spare `n_right × 5` output region).
/// For l > 2 this function panics (not yet implemented).
pub fn cart2sph_inplace(buf: &mut [f64], l: usize, n_right: usize) {
    match l {
        0 | 1 => { /* identity */ }
        2 => {
            // buf has shape [6 × n_right] column-major; we need to contract to [5 × n_right].
            // We do this in a temporary and copy back.
            let mat = c2s_matrix_cm(2);   // column-major: mat[c + s*6]
            let nfc = 6usize;
            let nfs = 5usize;
            let mut tmp = vec![0.0f64; nfs * n_right];
            // tmp[s, r] = Σ_c mat[c, s] * buf[c, r]
            for r in 0..n_right {
                for s in 0..nfs {
                    let mut v = 0.0;
                    for c in 0..nfc {
                        v += mat[c + s * nfc] * buf[c + r * nfc];
                    }
                    tmp[s + r * nfs] = v;
                }
            }
            buf[..nfs * n_right].copy_from_slice(&tmp);
        }
        _ => unimplemented!("cart2sph not implemented for l > 2"),
    }
    let _ = n_right;
}

/// Transform a Cartesian block `gcart` of shape `[n_bra × nfc]` (n_bra fast)
/// into a spherical block `gsph` of shape `[n_bra × nfs]` (n_bra fast).
///
/// For l ≤ 1 the transform is identity (nfc == nfs).  l = 2 uses the 6→5 matrix.
pub fn cart2sph_block(gsph: &mut [f64], gcart: &[f64], l: usize, n_bra: usize) {
    match l {
        0 | 1 => {
            let nf = (l + 1) * (l + 2) / 2;
            let n  = n_bra * nf;
            gsph[..n].copy_from_slice(&gcart[..n]);
        }
        2 => {
            let mat = c2s_matrix_cm(2);   // column-major: mat[c + s*6]
            let nfc = 6usize;
            let nfs = 5usize;
            // gcart layout: [n_bra × nfc] row-major (n_bra is leading index)
            // gsph  layout: [n_bra × nfs]
            for b in 0..n_bra {
                for s in 0..nfs {
                    let mut v = 0.0;
                    for c in 0..nfc {
                        v += mat[c + s * nfc] * gcart[b + c * n_bra];
                    }
                    gsph[b + s * n_bra] = v;
                }
            }
        }
        _ => unimplemented!("cart2sph not implemented for l > 2"),
    }
}

// ─────────────────────────────────────────────────────────────────
// Transformation matrices for reference (not used in l≤1 identity path)
// ─────────────────────────────────────────────────────────────────

/// Return the column-major transformation matrix `cart → sph` for angular momentum `l`.
/// Shape: `[nfc × nfs]` where `nfc = (l+1)(l+2)/2` and `nfs = 2l+1`.
/// Storage: mat[c + s*nfc]
///
/// l = 0: 1×1 identity
/// l = 1: 3×3 identity
/// l = 2: 6×5 matrix (Cartesian d → real spherical d)
///
/// The d-shell Cartesian order follows libcint: [xx, xy, xz, yy, yz, zz]
/// The spherical order: [d_{-2}, d_{-1}, d_0, d_1, d_2]
/// Coefficients from libcint cart2sph.c autocode.
pub fn c2s_matrix(l: usize) -> Vec<f64> {
    match l {
        0 => vec![1.0],
        1 => vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        2 => {
            // Shape [6 × 5], storage mat[c + s*6].
            // Cartesian: 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz
            // Spherical: 0=d_{-2}, 1=d_{-1}, 2=d_0, 3=d_1, 4=d_2
            //
            // From libcint cart2sph.c (CINTc2s_ket_sph for l=2) / Helgaker eq.(6.4.72):
            //   d_{-2} = √3 · xy
            //   d_{-1} = √3 · yz
            //   d_{0}  = (3zz - rr) / 2  = (2zz - xx - yy) / 2   (with unit-sphere norm)
            //   d_{1}  = √3 · xz
            //   d_{2}  = √3/2 · (xx - yy)
            //
            // With the libcint normalisation convention the matrix is:
            const S3:  f64 = 1.732050808568877293527446;  // √3
            const S3H: f64 = 0.866025403784438646763723;  // √3/2
            const HALF: f64 = 0.5;
            const NHALF: f64 = -0.5;
            // Columns represent spherical components; rows are Cartesian.
            // Ordering: mat[cart_idx + sph_idx * 6]
            //           col 0     col 1     col 2      col 3     col 4
            //         d_{-2}    d_{-1}    d_0        d_1       d_2
            vec![
                // row 0: xx
                0.0,    0.0,     NHALF,     0.0,      S3H,
                // row 1: xy
                S3,     0.0,     0.0,       0.0,      0.0,
                // row 2: xz
                0.0,    0.0,     0.0,       S3,       0.0,
                // row 3: yy
                0.0,    0.0,     NHALF,     0.0,     -S3H,
                // row 4: yz
                0.0,    S3,      0.0,       0.0,      0.0,
                // row 5: zz
                0.0,    0.0,     1.0,       0.0,      0.0,
            ]
            // Note: this is stored row-major above for readability;
            // reshape to column-major [c + s*6]:
            // The vec is already in row-major(c fastest), we need column-major(s fastest).
            // Re-index: result[c + s*6] — let's just return the transposed layout.
        }
        _ => unimplemented!("c2s_matrix: l > 2 not implemented"),
    }
}

/// Like `c2s_matrix` but returns data in **column-major** order `mat[c + s*nfc]`.
/// For l ≤ 1 same as `c2s_matrix`. For l = 2 transposes the row-major layout above.
pub fn c2s_matrix_cm(l: usize) -> Vec<f64> {
    if l != 2 {
        return c2s_matrix(l);
    }
    // c2s_matrix(2) is row-major [row=cart, col=sph] i.e. mat[r*5 + s].
    // We want column-major: result[c + s*6].
    let rm = c2s_matrix(2);
    let nfc = 6usize;
    let nfs = 5usize;
    let mut cm = vec![0.0f64; nfc * nfs];
    for c in 0..nfc {
        for s in 0..nfs {
            cm[c + s * nfc] = rm[c * nfs + s];
        }
    }
    cm
}
