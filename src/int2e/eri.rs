//! Two-electron repulsion integral (ERI) kernel.
//!
//! Implements `int2e_cart` — the C-ABI entry point compatible with libcint's
//! `int2e_cart` function.  For each shell quartet (i,j|k,l) it:
//!   1. Iterates over all primitive quadruplets (ip, jp, kp, lp).
//!   2. Calls `g0_2e` to fill the g-buffer via Rys quadrature.
//!   3. Uses `g2e_index_xyz` to extract the Cartesian integral elements.
//!   4. Contracts over primitives into the output array.
//!
//! The output buffer `out` has shape `[nfi*nci × nfj*ncj × nfk*nck × nfl*ncl]`
//! in Fortran column-major order (i.e. i is the fastest index).

use crate::types::{
    AtmSlot, BasSlot, Env, SQRTPI,
    EnvVars, common_fac_sp, ncart,
};
use crate::recur::{g0_2e, g2e_index_xyz};

// ─────────────────────────────────────────────────────────────────
// public C-ABI entry point
// ─────────────────────────────────────────────────────────────────

/// Compute two-electron repulsion integrals `(ij|kl)` for shell quartet
/// `(shls[0], shls[1], shls[2], shls[3])`.
///
/// This is the **bare** implementation with no Cauchy-Schwarz screening.
/// It is called by the C-ABI wrapper (which does the opt check) and by
/// `CINTOpt::build` when building screening data.
///
/// Returns 1 if the result is non-zero, 0 otherwise.
pub fn int2e_cart_bare(
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
        let shls = std::slice::from_raw_parts(shls, 4);

        let ev = Env::from_raw(env, 10000);
        let i_sh = shls[0] as usize;
        let j_sh = shls[1] as usize;
        let k_sh = shls[2] as usize;
        let l_sh = shls[3] as usize;

        let bi = BasSlot::from_raw(bas, i_sh);
        let bj = BasSlot::from_raw(bas, j_sh);
        let bk = BasSlot::from_raw(bas, k_sh);
        let bl = BasSlot::from_raw(bas, l_sh);

        let i_l = bi.ang_of();
        let j_l = bj.ang_of();
        let k_l = bk.ang_of();
        let l_l = bl.ang_of();

        let nfi = ncart(i_l);
        let nfj = ncart(j_l);
        let nfk = ncart(k_l);
        let nfl = ncart(l_l);
        let nf  = nfi * nfj * nfk * nfl;

        let nprim_i = bi.nprim_of();
        let nprim_j = bj.nprim_of();
        let nprim_k = bk.nprim_of();
        let nprim_l = bl.nprim_of();
        let nctr_i  = bi.nctr_of();
        let nctr_j  = bj.nctr_of();
        let nctr_k  = bk.nctr_of();
        let nctr_l  = bl.nctr_of();

        let ri = ev.coords(AtmSlot::from_raw(atm, bi.atom_of()).ptr_coord());
        let rj = ev.coords(AtmSlot::from_raw(atm, bj.atom_of()).ptr_coord());
        let rk = ev.coords(AtmSlot::from_raw(atm, bk.atom_of()).ptr_coord());
        let rl = ev.coords(AtmSlot::from_raw(atm, bl.atom_of()).ptr_coord());

        let expi = ev.exps(bi.ptr_exp(), nprim_i);
        let expj = ev.exps(bj.ptr_exp(), nprim_j);
        let expk = ev.exps(bk.ptr_exp(), nprim_k);
        let expl = ev.exps(bl.ptr_exp(), nprim_l);
        let coei = ev.coeffs(bi.ptr_coeff(), nprim_i, nctr_i);
        let coej = ev.coeffs(bj.ptr_coeff(), nprim_j, nctr_j);
        let coek = ev.coeffs(bk.ptr_coeff(), nprim_k, nctr_k);
        let coel = ev.coeffs(bl.ptr_coeff(), nprim_l, nctr_l);

        let rirj = [ri[0]-rj[0], ri[1]-rj[1], ri[2]-rj[2]];
        let rkrl = [rk[0]-rl[0], rk[1]-rl[1], rk[2]-rl[2]];
        let r2_ij = rirj[0]*rirj[0] + rirj[1]*rirj[1] + rirj[2]*rirj[2];
        let r2_kl = rkrl[0]*rkrl[0] + rkrl[1]*rkrl[1] + rkrl[2]*rkrl[2];

        // Build EnvVars for the quartet
        let envs = build_envs_2e(
            i_l, j_l, k_l, l_l,
            nfi, nfj, nfk, nfl,
            nctr_i, nctr_j, nctr_k, nctr_l,
            &ri, &rj, &rk, &rl,
            &rirj, &rkrl,
        );

        // Determine output dimensions
        let out_ni = nfi * nctr_i;
        let out_nj = nfj * nctr_j;
        let out_nk = nfk * nctr_k;
        let out_nl = nfl * nctr_l;
        let out_n  = out_ni * out_nj * out_nk * out_nl;
        let out_sl = std::slice::from_raw_parts_mut(out, out_n);
        for v in out_sl.iter_mut() { *v = 0.0; }

        // Build index table (uses g_stride etc. from envs)
        let idx = g2e_index_xyz(&envs);

        // Common factor: (π^(5/2)) * fac_sp(i) * fac_sp(j) * fac_sp(k) * fac_sp(l)
        // common_factor from g2e.c: (M_PI*M_PI*M_PI)*2/SQRTPI = 2*pi^(5/2)
        let common_fac = 2.0 * SQRTPI.powi(5)
            * common_fac_sp(i_l) * common_fac_sp(j_l)
            * common_fac_sp(k_l) * common_fac_sp(l_l);

        let g_buf_size = 3 * envs.g_size;
        let mut g = vec![0.0f64; g_buf_size];

        let mut has_value = false;

        for lp in 0..nprim_l {
            let al = expl[lp];
            for kp in 0..nprim_k {
                let ak    = expk[kp];
                let akl   = ak + al;
                let exp_kl = -(ak * al / akl) * r2_kl;
                if exp_kl < -50.0 { continue; }
                let expkl = exp_kl.exp();

                let rkl_c = [
                    (ak*rk[0] + al*rl[0]) / akl,
                    (ak*rk[1] + al*rl[1]) / akl,
                    (ak*rk[2] + al*rl[2]) / akl,
                ];

                for jp in 0..nprim_j {
                    let aj = expj[jp];
                    for ip in 0..nprim_i {
                        let ai   = expi[ip];
                        let aij  = ai + aj;
                        let exp_ij = -(ai * aj / aij) * r2_ij;
                        if exp_ij < -50.0 { continue; }
                        let expij = exp_ij.exp();

                        let rij_c = [
                            (ai*ri[0] + aj*rj[0]) / aij,
                            (ai*ri[1] + aj*rj[1]) / aij,
                            (ai*ri[2] + aj*rj[2]) / aij,
                        ];

                        let fac_prim = common_fac * expij * expkl;

                        // Build per-primitive EnvVars (all fields set)
                        let mut ev2 = envs.clone();
                        ev2.ai  = ai;  ev2.aj = aj;
                        ev2.ak  = ak;  ev2.al = al;
                        ev2.fac = fac_prim;
                        ev2.rij = rij_c;
                        ev2.rkl = rkl_c;
                        // rx_in_rijrx  is rj (ibase=false) or ri (ibase=true)
                        ev2.rx_in_rijrx = if ev2.ibase { ri } else { rj };
                        ev2.rx_in_rklrx = if ev2.kbase { rk } else { rl };

                        // Zero g buffer
                        for v in g.iter_mut() { *v = 0.0; }
                        let ok = g0_2e(&mut g, &ev2);
                        if !ok { continue; }

                        // Contract into output
                        for ic in 0..nctr_i {
                            let ci = coei[ip + ic * nprim_i];
                            for jc in 0..nctr_j {
                                let cj = coej[jp + jc * nprim_j];
                                for kc in 0..nctr_k {
                                    let ck = coek[kp + kc * nprim_k];
                                    for lc in 0..nctr_l {
                                        let cl = coel[lp + lc * nprim_l];
                                        let c = ci * cj * ck * cl;
                                        accumulate_gout(
                                            out_sl,
                                            &g,
                                            &idx,
                                            nf,
                                            &ev2,
                                            c,
                                            ic, jc, kc, lc,
                                            nfi, nfj, nfk, nfl,
                                            nctr_i, nctr_j, nctr_k, nctr_l,
                                        );
                                        has_value = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if has_value { 1 } else { 0 }
    }
}

/// Extract Cartesian integrals from the g-buffer and accumulate into `out`.
#[allow(clippy::too_many_arguments)]
fn accumulate_gout(
    out:    &mut [f64],
    g:      &[f64],
    idx:    &[usize],
    nf:     usize,
    envs:   &EnvVars,
    c:      f64,
    ic: usize, jc: usize, kc: usize, lc: usize,
    nfi: usize, nfj: usize, nfk: usize, nfl: usize,
    nctr_i: usize, nctr_j: usize, nctr_k: usize, nctr_l: usize,
) {
    let nroots = envs.nrys_roots;
    for n in 0..nf {
        let ix = idx[3 * n];
        let iy = idx[3 * n + 1];
        let iz = idx[3 * n + 2];
        let mut s = 0.0;
        for r in 0..nroots {
            s += g[ix + r] * g[iy + r] * g[iz + r];
        }
        // n indexes (j, l, k, i) with i fastest:
        // n = i + nfi*(k + nfk*(l + nfl*j))  (libcint order)
        let ni = n % nfi;
        let tmp = n / nfi;
        let nk = tmp % nfk;
        let tmp = tmp / nfk;
        let nl = tmp % nfl;
        let nj = tmp / nfl;

        let row = nfi * ic + ni;
        let col_j = nfj * jc + nj;
        let col_k = nfk * kc + nk;
        let col_l = nfl * lc + nl;

        // Column-major: out[i + ni*(j + nj*(k + nk*l))]
        let flat_col = col_j + (nfj * nctr_j) * (col_k + (nfk * nctr_k) * col_l);
        let idx_out  = row + (nfi * nctr_i) * flat_col;
        out[idx_out] += c * s;
    }
}

// ─────────────────────────────────────────────────────────────────
// EnvVars construction for a 2e integral
// ─────────────────────────────────────────────────────────────────

/// Build a skeleton `EnvVars` for the given shell angular momenta.
/// Per-primitive fields (ai/aj/ak/al/fac/rij/rkl) are set to zero and
/// must be updated in the inner loop.
#[allow(clippy::too_many_arguments)]
fn build_envs_2e(
    i_l: usize, j_l: usize, k_l: usize, l_l: usize,
    nfi: usize, nfj: usize, nfk: usize, nfl: usize,
    nctr_i: usize, nctr_j: usize, nctr_k: usize, nctr_l: usize,
    ri: &[f64; 3], rj: &[f64; 3], rk: &[f64; 3], rl: &[f64; 3],
    rirj: &[f64; 3], rkrl: &[f64; 3],
) -> EnvVars {
    // Angular-momentum ceilings (ng[] = 0 for standard ERI, so ceil = l)
    let li_ceil = i_l;
    let lj_ceil = j_l;
    let lk_ceil = k_l;
    let ll_ceil = l_l;

    let nrys_roots = (li_ceil + lj_ceil + lk_ceil + ll_ceil) / 2 + 1;
    let rys_order  = nrys_roots;

    // ibase = li_ceil > lj_ceil  (libcint convention)
    let ibase = li_ceil > lj_ceil;
    let kbase = lk_ceil > ll_ceil;

    // Strides (see CINTinit_int2e_EnvVars logic)
    let di: usize; let dj: usize; let dk: usize; let dl: usize;
    let dli: usize; let dlk: usize; let dlj: usize; let dll: usize;

    if ibase {
        dli = li_ceil + lj_ceil + 1;
        dlj = lj_ceil + 1;
    } else {
        dli = li_ceil + 1;
        dlj = li_ceil + lj_ceil + 1;
    }
    if kbase {
        dlk = lk_ceil + ll_ceil + 1;
        dll = ll_ceil + 1;
    } else {
        dlk = lk_ceil + 1;
        dll = lk_ceil + ll_ceil + 1;
    }

    let g_stride_i = nrys_roots;
    let g_stride_k = nrys_roots * dli;
    let g_stride_l = nrys_roots * dli * dlk;
    let g_stride_j = nrys_roots * dli * dlk * dll;
    let g_size     = nrys_roots * dli * dlk * dll * dlj;

    let g2d_ijmax = if ibase { g_stride_i } else { g_stride_j };
    let g2d_klmax = if kbase { g_stride_k } else { g_stride_l };

    // rirj and rkrl directions
    let rirj_ev = if ibase {
        [ri[0]-rj[0], ri[1]-rj[1], ri[2]-rj[2]]
    } else {
        [rj[0]-ri[0], rj[1]-ri[1], rj[2]-ri[2]]
    };
    let rkrl_ev = if kbase {
        [rk[0]-rl[0], rk[1]-rl[1], rk[2]-rl[2]]
    } else {
        [rl[0]-rk[0], rl[1]-rk[1], rl[2]-rk[2]]
    };

    EnvVars {
        i_l, j_l, k_l, l_l,
        nfi, nfj, nfk, nfl,
        nf: nfi * nfj * nfk * nfl,
        x_ctr: [nctr_i, nctr_j, nctr_k, nctr_l],
        li_ceil, lj_ceil, lk_ceil, ll_ceil,
        nrys_roots, rys_order,
        g_stride_i, g_stride_k, g_stride_l, g_stride_j,
        g_size,
        g2d_ijmax, g2d_klmax,
        ibase, kbase,
        common_factor: 1.0,  // absorbed into fac
        expcutoff: 60.0,
        ai: 0.0, aj: 0.0, ak: 0.0, al: 0.0,
        fac: 0.0,
        rirj: rirj_ev, rkrl: rkrl_ev,
        rij: [0.0; 3], rkl: [0.0; 3],
        ri: *ri, rj: *rj, rk: *rk, rl: *rl,
        rx_in_rijrx: if ibase { *ri } else { *rj },
        rx_in_rklrx: if kbase { *rk } else { *rl },
        ncomp_e1: 1, ncomp_e2: 1, ncomp_tensor: 1,
        gbits: 0,
    }
}
