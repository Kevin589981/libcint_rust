//! Core 2-electron primitive integral kernel.
//!
//! Implements the `g0_2e` function which:
//!   1. Computes Rys roots/weights via the Boys function
//!   2. Builds the Rys2eT intermediate quantities (b00, b10, b01, c00, c0p)
//!   3. Initialises the g buffer (gx=1, gy=1, gz=w)
//!   4. Dispatches to the appropriate 2D/4D recursion step
//!
//! The g buffer layout is gx[0..g_size] | gy[g_size..2*g_size] | gz[2*g_size..3*g_size].

use crate::types::{EnvVars, MXRYSROOTS};
use crate::rys::rys_roots;
use crate::recur::g0_2d4d::{
    g0_2e_2d, g0_2e_2d4d_unrolled,
    g0_lj2d_4d, g0_kj2d_4d, g0_il2d_4d, g0_ik2d_4d,
};

/// Intermediate Rys-quadrature quantities for each root.
/// Direct translation of `Rys2eT` in libcint's `g2e.h`.
#[derive(Clone, Default)]
pub struct Rys2eT {
    /// b00[irys] = u2 * tmp4   where u2 = a0*u[irys], tmp4 = 0.5/(u2*(aij+akl)+aij*akl)
    pub b00: [f64; MXRYSROOTS],
    /// b10[irys] = b00 + 0.5*tmp4_over_akl_factor  (used for ij-recursion)
    pub b10: [f64; MXRYSROOTS],
    /// b01[irys] = b00 + 0.5*tmp4_over_aij_factor  (used for kl-recursion)
    pub b01: [f64; MXRYSROOTS],
    /// c00[irys] = (rP - rA)[component],  rP = weighted Rys centroid for ij pair
    pub c00x: [f64; MXRYSROOTS],
    pub c00y: [f64; MXRYSROOTS],
    pub c00z: [f64; MXRYSROOTS],
    /// c0p[irys] = (rQ - rC)[component],  rQ = weighted Rys centroid for kl pair
    pub c0px: [f64; MXRYSROOTS],
    pub c0py: [f64; MXRYSROOTS],
    pub c0pz: [f64; MXRYSROOTS],
}

/// Fill the g buffer for a single primitive quartet (i,j,k,l) and return 1 on success.
///
/// On entry:
///   - `g` must have length `3 * envs.g_size` (zero-initialised by caller).
///   - `envs.rij` and `envs.rkl` are the Gaussian-product centres of the ij and kl pairs.
///   - `envs.rx_in_rijrx` and `envs.rx_in_rklrx` are the reference origins (= ri or rj, rk or rl).
///   - `envs.ai`, `envs.aj`, `envs.ak`, `envs.al` are the current exponents.
///   - `envs.fac` is the combined prefactor (geometry × contraction coefficients).
///
/// On return the gz section of `g` contains the scaled Rys weights (× fac1), and
/// gx/gy are filled with all polynomial factors up to the required angular momenta.
/// The caller then reads off integrals via `g2e_index_xyz`.
pub fn g0_2e(g: &mut [f64], envs: &EnvVars) -> bool {
    let nroots = envs.nrys_roots;
    let aij = envs.ai + envs.aj;
    let akl = envs.ak + envs.al;
    let a1  = aij * akl;
    let a0  = a1 / (aij + akl);
    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * envs.fac;

    let xij_kl = envs.rij[0] - envs.rkl[0];
    let yij_kl = envs.rij[1] - envs.rkl[1];
    let zij_kl = envs.rij[2] - envs.rkl[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;
    let x = a0 * rr;

    // Rys roots u[] and weights w[] ─ w written into gz section of g
    let g_size = envs.g_size;
    let mut u = [0_f64; MXRYSROOTS];
    let mut w = [0_f64; MXRYSROOTS];
    rys_roots(nroots, x, &mut u[..nroots], &mut w[..nroots]);

    // rP - rA  and  rQ - rC
    let rijrx = envs.rij[0] - envs.rx_in_rijrx[0];
    let rijry = envs.rij[1] - envs.rx_in_rijrx[1];
    let rijrz = envs.rij[2] - envs.rx_in_rijrx[2];
    let rklrx = envs.rkl[0] - envs.rx_in_rklrx[0];
    let rklry = envs.rkl[1] - envs.rx_in_rklrx[1];
    let rklrz = envs.rkl[2] - envs.rx_in_rklrx[2];

    let mut bc = Rys2eT::default();

    for irys in 0..nroots {
        let u2   = a0 * u[irys];
        let tmp4 = 0.5 / (u2 * (aij + akl) + a1);
        let tmp5 = u2 * tmp4;      // = b00
        let tmp2 = 2.0 * tmp5 * akl;
        let tmp3 = 2.0 * tmp5 * aij;

        bc.b00[irys] = tmp5;
        bc.b10[irys] = tmp5 + tmp4 * akl;
        bc.b01[irys] = tmp5 + tmp4 * aij;

        bc.c00x[irys] = rijrx - tmp2 * xij_kl;
        bc.c00y[irys] = rijry - tmp2 * yij_kl;
        bc.c00z[irys] = rijrz - tmp2 * zij_kl;

        bc.c0px[irys] = rklrx + tmp3 * xij_kl;
        bc.c0py[irys] = rklry + tmp3 * yij_kl;
        bc.c0pz[irys] = rklrz + tmp3 * zij_kl;

        // Write weighted gz value
        g[2 * g_size + irys] = w[irys] * fac1;
    }

    // Trivial case: ss|ss — gx=gy=1 already (rest of g is zeros, only gz needed)
    if g_size == 1 {
        g[0] = 1.0;
        g[1] = 1.0;
        return true;
    }

    // Dispatch to 2D+4D recursion based on ibase/kbase flags and rys_order
    dispatch_g0_2d4d(g, &bc, envs);
    true
}

/// Select and invoke the correct 2D+4D recursion variant.
pub(crate) fn dispatch_g0_2d4d(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let rys_order = (envs.li_ceil + envs.lj_ceil + envs.lk_ceil + envs.ll_ceil) / 2 + 1;

    if rys_order <= 2 {
        // Unrolled fast path
        g0_2e_2d4d_unrolled(g, bc, envs);
        return;
    }

    match (envs.kbase, envs.ibase) {
        (true,  true)  => { g0_2e_2d(g, bc, envs); g0_ik2d_4d(g, envs); }
        (true,  false) => { g0_2e_2d(g, bc, envs); g0_kj2d_4d(g, envs); }
        (false, true)  => { g0_2e_2d(g, bc, envs); g0_il2d_4d(g, envs); }
        (false, false) => { g0_2e_2d(g, bc, envs); g0_lj2d_4d(g, envs); }
    }
}
