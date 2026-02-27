//! Kinetic energy integral <i| -½∇² |j>.
//!
//! Uses the Obara-Saika kinetic energy formula, which expresses the 3D
//! kinetic energy integral as a sum of products of 1D overlap integrals:
//!
//!   T_{ij} = T1d(a_x,b_x)·S1d(a_y,b_y)·S1d(a_z,b_z)
//!          + S1d(a_x,b_x)·T1d(a_y,b_y)·S1d(a_z,b_z)
//!          + S1d(a_x,b_x)·S1d(a_y,b_y)·T1d(a_z,b_z)
//!
//! where the 1D kinetic integral is:
//!   T1d(m,n) = aj·(2n+1)·S1d(m,n) − 2·aj²·S1d(m,n+2) − n·(n–1)/2·S1d(m,n–2)
//!
//! Reference: Obara & Saika, JCP 84, 3963 (1986), eq.(22).

use crate::types::{
    AtmSlot, BasSlot, Env,
    common_fac_sp, cart_comp, ncart,
    SQRTPI,
};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────
// Public C-ABI entry point
// ─────────────────────────────────────────────────────────────────

/// Compute kinetic energy integrals `<i|−½∇²|j>` for shell pair `(shls[0], shls[1])`.
///
/// # C signature (matching libcint)
/// ```c
/// CACHE_SIZE_T int1e_kin_cart(double *out, int *dims, int *shls,
///     int *atm, int natm, int *bas, int nbas, double *env,
///     CINTOpt *opt, double *cache);
/// ```
/// Returns 1 if the shell pair contributes, 0 otherwise.
pub fn int1e_kin_cart(
    out:  *mut f64,
    dims: *const i32,
    shls: *const i32,
    atm:  *const i32,
    natm: i32,
    bas:  *const i32,
    nbas: i32,
    env:  *const f64,
) -> i32 {
    let _ = natm; let _ = nbas;
    unsafe {
        let shls = std::slice::from_raw_parts(shls, 2);
        let ev   = Env::from_raw(env, 10000);
        let i_sh  = shls[0] as usize;
        let j_sh  = shls[1] as usize;
        let bi    = BasSlot::from_raw(bas, i_sh);
        let bj    = BasSlot::from_raw(bas, j_sh);

        let i_l    = bi.ang_of();
        let j_l    = bj.ang_of();
        let nfi    = ncart(i_l);
        let nfj    = ncart(j_l);
        let nprim_i = bi.nprim_of();
        let nprim_j = bj.nprim_of();
        let nctr_i  = bi.nctr_of();
        let nctr_j  = bj.nctr_of();

        let ai_s = AtmSlot::from_raw(atm, bi.atom_of());
        let aj_s = AtmSlot::from_raw(atm, bj.atom_of());
        let ri = ev.coords(ai_s.ptr_coord());
        let rj = ev.coords(aj_s.ptr_coord());

        let expi = ev.exps(bi.ptr_exp(), nprim_i);
        let expj = ev.exps(bj.ptr_exp(), nprim_j);
        let coei = ev.coeffs(bi.ptr_coeff(), nprim_i, nctr_i);
        let coej = ev.coeffs(bj.ptr_coeff(), nprim_j, nctr_j);

        let rirj = [ri[0]-rj[0], ri[1]-rj[1], ri[2]-rj[2]];
        let r2ij = rirj[0]*rirj[0] + rirj[1]*rirj[1] + rirj[2]*rirj[2];

        let out_ni: usize;
        let out_nj: usize;
        if dims.is_null() {
            out_ni = nfi * nctr_i;
            out_nj = nfj * nctr_j;
        } else {
            let d = std::slice::from_raw_parts(dims, 2);
            out_ni = d[0] as usize;
            out_nj = d[1] as usize;
        }
        let out_sl = std::slice::from_raw_parts_mut(out, out_ni * out_nj);
        for v in out_sl.iter_mut() { *v = 0.0; }

        let common_factor = common_fac_sp(i_l) * common_fac_sp(j_l);
        let mut has_value = false;

        for jp in 0..nprim_j {
            let aj = expj[jp];
            for ip in 0..nprim_i {
                let ai   = expi[ip];
                let aij  = ai + aj;
                let exp_ij = -(ai * aj / aij) * r2ij;
                if exp_ij < -50.0 { continue; }

                // Product centre P = (ai·Ri + aj·Rj) / aij
                let px = (ai*ri[0] + aj*rj[0]) / aij;
                let py = (ai*ri[1] + aj*rj[1]) / aij;
                let pz = (ai*ri[2] + aj*rj[2]) / aij;

                let expij = exp_ij.exp();
                // Overlap prefactor (π/aij)^(3/2) absorbed into gz0
                let s0  = SQRTPI / aij.sqrt();
                let gz0 = common_factor * expij * s0 * s0 * s0;

                let aij2 = 0.5 / aij;

                // P - A and P - B (per coordinate, re-used below)
                let pa = [px - ri[0], py - ri[1], pz - ri[2]];
                let pb = [px - rj[0], py - rj[1], pz - rj[2]];

                has_value = true;
                kin_accumulate(
                    out_sl, out_ni,
                    i_l, j_l, nfi, nfj,
                    aj, gz0, aij2, &pa, &pb,
                    coei, ip, nctr_i,
                    coej, jp, nctr_j,
                    nprim_i, nprim_j,
                );
            }
        }

        if has_value { 1 } else { 0 }
    }
}

// ─────────────────────────────────────────────────────────────────
// Inner accumulation
// ─────────────────────────────────────────────────────────────────

/// General 1D overlap integral S(m, n; pa, pb, aij2).
///
/// Built via OS upward recursion:
///   S(k+1, 0) = pa · S(k,0) + k·aij2·S(k-1, 0)
/// followed by j-transfer:
///   S(m, n) = S(m+1, n-1) + (Ax−Bx)·S(m, n-1)
///           = S(m+1, n-1) + (pb−pa)·S(m, n-1)
fn s_1d(m: usize, n: usize, pa: f64, pb: f64, aij2: f64) -> f64 {
    let nmax = m + n;
    // Upward recursion for S(k, 0)
    let mut g = vec![0.0f64; nmax + 2];
    g[0] = 1.0;
    if nmax >= 1 { g[1] = pa; }
    for k in 1..nmax {
        g[k + 1] = pa * g[k] + k as f64 * aij2 * g[k - 1];
    }
    if n == 0 { return g[m]; }
    // j-transfer into a (nmax+1) × (n+1) array
    let abx = pb - pa;   // Ax − Bx
    let cols = n + 1;
    let mut arr = vec![0.0f64; (nmax + 1) * cols];
    for row in 0..=nmax { arr[row * cols] = g[row]; }
    for col in 1..=n {
        for row in 0..=(nmax - col) {
            arr[row * cols + col] = arr[(row + 1) * cols + col - 1]
                + abx * arr[row * cols + col - 1];
        }
    }
    arr[m * cols + n]
}

/// 1D kinetic energy integral T(m, n) acting on the `b` index:
///   T(m,n) = aj·(2n+1)·S(m,n) − 2·aj²·S(m,n+2) − n·(n−1)/2·S(m,n−2)
#[inline]
fn t_1d(m: usize, n: usize, aj: f64, pa: f64, pb: f64, aij2: f64) -> f64 {
    let term1 = aj * (2 * n + 1) as f64 * s_1d(m, n, pa, pb, aij2);
    let term2 = -2.0 * aj * aj * s_1d(m, n + 2, pa, pb, aij2);
    let term3 = if n >= 2 {
        -(n as f64 * (n - 1) as f64) * 0.5 * s_1d(m, n - 2, pa, pb, aij2)
    } else {
        0.0
    };
    term1 + term2 + term3
}

/// Accumulate kinetic energy contributions for one primitive pair into `out`.
#[allow(clippy::too_many_arguments)]
fn kin_accumulate(
    out:    &mut [f64],
    out_ni: usize,
    i_l:    usize,
    j_l:    usize,
    nfi:    usize,
    nfj:    usize,
    aj:     f64,
    gz0:    f64,
    aij2:   f64,
    pa:     &[f64; 3],
    pb:     &[f64; 3],
    coei:   &[f64],
    ip:     usize,
    nctr_i: usize,
    coej:   &[f64],
    jp:     usize,
    nctr_j: usize,
    nprim_i: usize,
    nprim_j: usize,
) {
    let mut i_nx = Vec::new(); let mut i_ny = Vec::new(); let mut i_nz = Vec::new();
    let mut j_nx = Vec::new(); let mut j_ny = Vec::new(); let mut j_nz = Vec::new();
    cart_comp(&mut i_nx, &mut i_ny, &mut i_nz, i_l);
    cart_comp(&mut j_nx, &mut j_ny, &mut j_nz, j_l);

    // Convert to usize once
    let i_nx: Vec<usize> = i_nx.iter().map(|&v| v as usize).collect();
    let i_ny: Vec<usize> = i_ny.iter().map(|&v| v as usize).collect();
    let i_nz: Vec<usize> = i_nz.iter().map(|&v| v as usize).collect();
    let j_nx: Vec<usize> = j_nx.iter().map(|&v| v as usize).collect();
    let j_ny: Vec<usize> = j_ny.iter().map(|&v| v as usize).collect();
    let j_nz: Vec<usize> = j_nz.iter().map(|&v| v as usize).collect();

    // Pre-compute all 1D overlap and kinetic integrals needed.
    // Unique (m,n) pairs are just the product of Cartesian components.
    // Small nfi/nfj → inner loops are tiny.

    for ic in 0..nctr_i {
        let ci = coei[ip + ic * nprim_i];
        for jc in 0..nctr_j {
            let cj   = coej[jp + jc * nprim_j];
            let cij  = ci * cj * gz0;

            for ni in 0..nfi {
                let mx = i_nx[ni]; let my = i_ny[ni]; let mz = i_nz[ni];
                for nj in 0..nfj {
                    let nx = j_nx[nj]; let ny = j_ny[nj]; let nz = j_nz[nj];

                    // 1D overlap factors
                    let sx = s_1d(mx, nx, pa[0], pb[0], aij2);
                    let sy = s_1d(my, ny, pa[1], pb[1], aij2);
                    let sz = s_1d(mz, nz, pa[2], pb[2], aij2);

                    // 1D kinetic factors
                    let tx = t_1d(mx, nx, aj, pa[0], pb[0], aij2);
                    let ty = t_1d(my, ny, aj, pa[1], pb[1], aij2);
                    let tz = t_1d(mz, nz, aj, pa[2], pb[2], aij2);

                    // T_3d = Tx·Sy·Sz + Sx·Ty·Sz + Sx·Sy·Tz
                    let v = cij * (tx*sy*sz + sx*ty*sz + sx*sy*tz);

                    let row = nfi * ic + ni;
                    let col = nfj * jc + nj;
                    out[row + col * out_ni] += v;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that T = p²/(2m) → for a 1s Gaussian φ = exp(-ζ·r²),
    /// the kinetic energy is T = 3ζ/2 · S (diagonal s-s case).
    ///
    /// int1e_kin_cart over a single contracted s-shell with exponent ζ and
    /// coefficient c should yield T = 3·ζ · S/2 = 3ζ/2 · (π/2ζ)^(3/2) · c².
    /// (All constants are tested only up to relative tolerance.)
    #[test]
    fn s_s_kinetic_diagonal() {
        // Build minimal atm/bas/env for two identical s-shells at the origin.
        // env layout: [0..20](header), [20..23](coord A), [23..26](coord B), [26](exp), [27](coeff)
        let mut env = vec![0.0f64; 40];
        // coords at origin
        for i in 0..6 { env[20 + i] = 0.0; }
        let zeta = 1.0_f64;
        env[26] = zeta;    // exponent
        env[27] = 1.0;     // contraction coeff (unnormalized)

        // atm: [charge, ptr_coord, nuc_mod, ptr_zeta, ...]
        let atm: Vec<i32> = vec![
            1, 20, 1, 0, 0, 0,  // atom 0 at env[20]
            1, 23, 1, 0, 0, 0,  // atom 1 at env[23]
        ];
        // bas: [atom_of, ang_of, nprim, nctr, kappa, ptr_exp, ptr_coeff, reserved]
        let bas: Vec<i32> = vec![
            0, 0, 1, 1, 0, 26, 27, 0,  // shell 0: s on atom 0
            1, 0, 1, 1, 0, 26, 27, 0,  // shell 1: s on atom 1 (same exponent)
        ];

        let shls: Vec<i32> = vec![0, 0];  // (0,0) diagonal
        let mut out = vec![0.0f64; 1];

        let ret = int1e_kin_cart(
            out.as_mut_ptr(),
            std::ptr::null(),
            shls.as_ptr(),
            atm.as_ptr(),
            2,
            bas.as_ptr(),
            2,
            env.as_ptr(),
        );
        assert_eq!(ret, 1);

        // Expected: T_{ss} = 3ζ/2 · (π/2ζ)^(3/2) · (common_fac_sp(0))^2
        // With our normalization gz0 = common_fac^2 · (π/aij)^(3/2) · exp(0)
        // and the kinetic integral of the s-s pair = T_1d(0,0,ζ) · 1 · 1 + perms
        // T_1d(0,0,ζ) = ζ·(1)·1 - 2ζ²·S(0,2) = ζ - 2ζ²·aij2 = ζ - 2ζ²·(1/(2*2ζ)) = ζ - ζ/2 = ζ/2
        // T total = 3 · (ζ/2). Then multiply gz0 = (0.282...)^2 · (π/(2ζ))^(3/2)
        // We just check the ratio T/S = ζ (for diagonal)
        let s_expected = {
            let mut s_out = vec![0.0f64; 1];
            let s_shls = vec![0i32, 0];
            crate::int1e::int1e_ovlp_cart(
                s_out.as_mut_ptr(),
                std::ptr::null(),
                s_shls.as_ptr(),
                atm.as_ptr(), 2,
                bas.as_ptr(), 2,
                env.as_ptr(),
            );
            s_out[0]
        };

        // For diagonal ss pair at origin: T = (3/2)·ζ · S
        // (3 spatial dimensions each contribute ζ/2·S to the kinetic energy)
        let ratio = out[0] / s_expected;
        let expected_ratio = 1.5 * zeta;
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "T_{{ss}}/S_{{ss}} = {} expected {}", ratio, expected_ratio
        );
    }

    /// Sanity check: T_{pp} diagonal should give 5ζ/2 · S_{pp} for px–px.
    #[test]
    fn p_p_kinetic_diagonal() {
        let mut env = vec![0.0f64; 40];
        let zeta = 1.5_f64;
        env[26] = zeta;
        env[27] = 1.0;

        let atm: Vec<i32> = vec![1, 20, 1, 0, 0, 0];
        let bas: Vec<i32> = vec![0, 1, 1, 1, 0, 26, 27, 0];
        let shls = vec![0i32, 0];
        let mut out = vec![0.0f64; 9];  // 3×3

        let ret = int1e_kin_cart(
            out.as_mut_ptr(),
            std::ptr::null(),
            shls.as_ptr(),
            atm.as_ptr(), 1,
            bas.as_ptr(), 1,
            env.as_ptr(),
        );
        assert_eq!(ret, 1);

        // Get overlap for comparison (should be diagonal for p placed at origin)
        let mut s_out = vec![0.0f64; 9];
        crate::int1e::int1e_ovlp_cart(
            s_out.as_mut_ptr(),
            std::ptr::null(),
            shls.as_ptr(),
            atm.as_ptr(), 1,
            bas.as_ptr(), 1,
            env.as_ptr(),
        );

        // For px-px diagonal at origin: ratio should be 5*zeta/2
        // T_1d(1,1) = 3*ζ*S(1,1) - 2ζ²*S(1,3) - 0
        // The total should obey T_pp = (ζ + 2*ζ^?) ... we just check it is positive
        // and proportional to S
        let ratio = out[0] / s_out[0];
        // Expected: for l=1, px-px: T = (5/2)*ζ * S  (textbook result)
        let expected = 5.0 * zeta / 2.0;
        assert!(
            (ratio - expected).abs() < 1e-9,
            "T_pp/S_pp = {} expected {}", ratio, expected
        );
    }
}
