//! CINTOpt: pre-computed screening data for Cauchy-Schwarz integral screening.
//!
//! The Cauchy-Schwarz inequality gives the rigorous upper bound:
//!   |(ij|kl)| ≤ sqrt(|(ij|ij)|) · sqrt(|(kl|kl)|)
//!
//! By pre-computing `sqrt(|(ij|ij)|)` for every shell pair (i,j) before the
//! main SCF loop, we can skip entire shell quartets whose bound falls below the
//! numerical threshold, avoiding most of the ERI computation for large molecules
//! where the two-electron integral matrix is sparse.
//!
//! # C-ABI
//! ```c
//! void int2e_optimizer(CINTOpt **opt, int *atm, int natm,
//!                      int *bas, int nbas, double *env);
//! void CINTdel_optimizer(CINTOpt **opt);
//! ```
//!
//! The `CINTOpt` pointer is heap-allocated by `int2e_optimizer` and freed by
//! `CINTdel_optimizer`.  Pass `opt = NULL` to skip screening entirely.

use crate::types::{BasSlot, ncart};

/// Screening threshold: skip quartet (ij|kl) if
/// `sqrt_schwarz[(i,j)] × sqrt_schwarz[(k,l)] < SCHWARZ_THRESH`.
///
/// 1e-12 ensures integral errors stay below ~1e-12 (compatible with the
/// default EXPCUTOFF-based screening in libcint).
pub const SCHWARZ_THRESH: f64 = 1e-12;

/// Pre-computed Cauchy-Schwarz screening data for one molecular system.
pub struct CINTOpt {
    /// Number of basis shells.
    pub nbas: usize,
    /// `sqrt_schwarz[i * nbas + j]` = `sqrt(max_{a∈i, b∈j} |(ia,jb|ia,jb)|)`.
    /// Filled for all (i,j) pairs with i ≤ j; symmetric.
    pub sqrt_schwarz: Vec<f64>,
    /// Screening threshold.
    pub threshold: f64,
}

impl CINTOpt {
    /// Compute the Schwarz screening table for the given molecular system.
    ///
    /// # Safety
    /// All pointers must be valid for the lifetime of this call and the
    /// returned `Box<CINTOpt>`.
    pub unsafe fn build(
        atm: *const i32, natm: i32,
        bas: *const i32, nbas: i32,
        env: *const f64,
    ) -> Box<Self> {
        let nb = nbas as usize;
        let mut sqrt_schwarz = vec![0.0_f64; nb * nb];

        for i in 0..nb {
            for j in 0..=i {
                // Compute (ij|ij): shell quartet (i,j,i,j)
                let bi = BasSlot::from_raw(bas, i);
                let bj = BasSlot::from_raw(bas, j);
                let li = bi.ang_of(); let nfi = ncart(li); let nci = bi.nctr_of();
                let lj = bj.ang_of(); let nfj = ncart(lj); let ncj = bj.nctr_of();
                let ni = nfi * nci;
                let nj = nfj * ncj;

                let mut diag_buf = vec![0.0_f64; ni * nj * ni * nj];
                let shls = [i as i32, j as i32, i as i32, j as i32];
                // Call bare (no-opt) ERI function to avoid circular dependency
                crate::int2e::int2e_cart_bare(
                    diag_buf.as_mut_ptr(),
                    std::ptr::null(),
                    shls.as_ptr(),
                    atm, natm, bas, nbas, env,
                );

                // Schwarz bound = max over diagonal elements (ia,jb|ia,jb)
                let mut max_val = 0.0_f64;
                for ia in 0..ni {
                    for jb in 0..nj {
                        // Column-major: out[row + ni*(col_j + nj*(col_k + nk*col_l))]
                        // For (ia,jb|ia,jb): row=ia, col_j=jb, col_k=ia, col_l=jb
                        let idx = ia + ni * (jb + nj * (ia + ni * jb));
                        let v = diag_buf[idx].abs();
                        if v > max_val { max_val = v; }
                    }
                }
                let sq = max_val.sqrt();
                sqrt_schwarz[i * nb + j] = sq;
                sqrt_schwarz[j * nb + i] = sq;
            }
        }

        Box::new(CINTOpt { nbas: nb, sqrt_schwarz, threshold: SCHWARZ_THRESH })
    }

    /// Return `true` if shell quartet (i,j|k,l) is above the screening
    /// threshold and must be computed; `false` if it can safely be skipped.
    #[inline]
    pub fn passes(&self, i: usize, j: usize, k: usize, l: usize) -> bool {
        let nb = self.nbas;
        self.sqrt_schwarz[i * nb + j] * self.sqrt_schwarz[k * nb + l]
            >= self.threshold
    }
}
