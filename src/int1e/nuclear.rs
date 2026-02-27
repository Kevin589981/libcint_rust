//! Nuclear attraction integral <i|V_nuc|j> = Σ_A Z_A <i|1/|r-R_A||j>.
//!
//! Uses a 1-centre Rys quadrature (Boys function) approach.
//! Follows libcint's `CINTg1e_nuc` logic.

use std::f64::consts::PI;
use crate::types::{
    AtmSlot, BasSlot, Env, SQRTPI, common_fac_sp, cart_comp_l, ncart,
    ATM_SLOTS,
};
use crate::rys::rys_roots;

const POINT_NUC: i32 = 1;

/// Compute nuclear attraction integrals `<i|V|j>` for shell pair `(shls[0], shls[1])`.
///
/// `out` has shape `[nfi*nctr_i × nfj*nctr_j]` in column-major order.
pub fn int1e_nuc_cart(
    out:  *mut f64,
    dims: *const i32,
    shls: *const i32,
    atm:  *const i32,
    natm: i32,
    bas:  *const i32,
    nbas: i32,
    env:  *const f64,
) -> i32 {
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

        let common_fac = common_fac_sp(i_l) * common_fac_sp(j_l);

        let mut has_value = false;
        let nroots = (i_l + j_l) / 2 + 1;

        for jp in 0..nprim_j {
            let aj = expj[jp];
            for ip in 0..nprim_i {
                let ai   = expi[ip];
                let aij  = ai + aj;
                let exp_ij = -(ai * aj / aij) * r2ij;
                if exp_ij < -50.0 { continue; }
                let expij = exp_ij.exp();

                let rij = [
                    (ai*ri[0] + aj*rj[0]) / aij,
                    (ai*ri[1] + aj*rj[1]) / aij,
                    (ai*ri[2] + aj*rj[2]) / aij,
                ];
                let fac_prim = common_fac * expij;

                // Sum over all nuclei
                for ia in 0..natm as usize {
                    let atm_a = AtmSlot::from_raw(atm, ia);
                    let za = atm_a.charge();
                    if za == 0 { continue; }
                    let charge = -(za as f64);
                    let rc = ev.coords(atm_a.ptr_coord());
                    let crij = [rc[0]-rij[0], rc[1]-rij[1], rc[2]-rij[2]];
                    let x = aij * (crij[0]*crij[0] + crij[1]*crij[1] + crij[2]*crij[2]);

                    let mut u = [0.0f64; 8];
                    let mut w = [0.0f64; 8];
                    rys_roots(nroots, x, &mut u[..nroots], &mut w[..nroots]);

                    let fac1 = 2.0 * PI / aij * charge * fac_prim;
                    has_value = true;

                    for ic in 0..nctr_i {
                        let ci = coei[ip + ic * nprim_i];
                        for jc in 0..nctr_j {
                            let cj = coej[jp + jc * nprim_j];
                            let cij = ci * cj * fac1;
                            nuc_primitive_sum(
                                out_sl, out_ni,
                                i_l, j_l, nfi, nfj,
                                nfi * ic, nfj * jc,
                                &ri, &rj, &rij, &rirj, &crij,
                                aij, &u[..nroots], &w[..nroots], nroots, cij,
                            );
                        }
                    }
                }
            }
        }

        if has_value { 1 } else { 0 }
    }
}

/// Sum Rys-quadrature contributions for a single primitive pair and single nucleus.
#[allow(clippy::too_many_arguments)]
fn nuc_primitive_sum(
    out:    &mut [f64],
    out_ni: usize,
    i_l: usize,
    j_l: usize,
    nfi: usize,
    nfj: usize,
    i_off: usize,
    j_off: usize,
    ri: &[f64; 3],
    rj: &[f64; 3],
    rij: &[f64; 3],
    rirj: &[f64; 3],
    crij: &[f64; 3],
    aij: f64,
    u: &[f64],
    w: &[f64],
    nroots: usize,
    fac: f64,
) {
    let (i_nx, i_ny, i_nz) = cart_comp_l(i_l);
    let (j_nx, j_ny, j_nz) = cart_comp_l(j_l);

    let aij2 = 0.5 / aij;

    // For each Rys root, compute the g-polynomial at this (u, w) point
    for irys in 0..nroots {
        let ui = u[irys];
        let wi = w[irys];
        // Modified t^2: t2 = u/(1+u)
        let t2 = ui / (1.0 + ui);
        let rt = aij2 * (1.0 - t2);  // aij2 - aij2*t2
        // Polynomial origins shifted by Rys parameter
        let r0x = rij[0] - ri[0] + t2 * crij[0];
        let r0y = rij[1] - ri[1] + t2 * crij[1];
        let r0z = rij[2] - ri[2] + t2 * crij[2];
        let rbx = rirj[0];
        let rby = rirj[1];
        let rbz = rirj[2];

        // Compute 1D g-polynomial for given (m, n): g1d(m, n, r0, rb, rt)
        let g1d = |m: usize, n: usize, r0: f64, rb: f64| -> f64 {
            // Based on CINTg1e_nuc single-root upward recursion:
            //   g[0] = 1
            //   g[1] = r0 * g[0]
            //   g[i+1] = i*rt * g[i-1] + r0 * g[i]
            // then j-transfer:
            //   g[n, j] = g[n+1, j-1] + rb * g[n, j-1]
            let nmax = m + n;
            if nmax == 0 { return 1.0; }
            let mut g = vec![0.0f64; (nmax + 2) * (nmax + 2)];
            let s = nmax + 2;
            g[0] = 1.0;
            for i in 0..nmax {
                let ii = i as f64;
                g[i+1] = r0 * g[i] + if i > 0 { ii * rt * g[i-1] } else { 0.0 };
            }
            // j-transfer
            for j in 1..=n {
                for i in 0..=(nmax-j) {
                    g[j * s + i] = g[(j-1)*s + i + 1] + rb * g[(j-1)*s + i];
                }
            }
            g[n * s + m]
        };

        for ni in 0..nfi {
            for nj in 0..nfj {
                let v = wi
                    * g1d(i_nx[ni] as usize, j_nx[nj] as usize, r0x, rbx)
                    * g1d(i_ny[ni] as usize, j_ny[nj] as usize, r0y, rby)
                    * g1d(i_nz[ni] as usize, j_nz[nj] as usize, r0z, rbz);
                let row = i_off + ni;
                let col = j_off + nj;
                out[row + col * out_ni] += fac * v;
            }
        }
    }
}
