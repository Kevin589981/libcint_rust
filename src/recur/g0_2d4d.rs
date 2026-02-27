//! 2D Obara-Saika upward recursion and 4D horizontal transfer functions.
//!
//! Implements the g-buffer filling pipeline:
//!   1. `g0_2e_2d`          — general upward OS recursion in 2D (m,n directions)
//!   2. `g0_*2d_4d`         — horizontal transfer from 2D to full 4D array
//!   3. `g0_2e_2d4d_unrolled` — fast path (rys_order ≤ 2) using pre-expanded formulas
//!   4. `g2e_index_xyz`     — build index table for integral extraction

use crate::types::{EnvVars, cart_comp_l};
use crate::recur::g0_2e::Rys2eT;

// ─────────────────────────────────────────────────────────────────
// General 2D upward recursion
// ─────────────────────────────────────────────────────────────────

/// Fill g[0..g_size] (gx), g[g_size..2*g_size] (gy), g[2*g_size..3*g_size] (gz)
/// using the Obara-Saika two-centre recurrence in the (m) and (n) directions.
///
/// On entry gz[irys] = w[irys]*fac1 already set; gx[irys]=gy[irys]=1.
/// On return all (n, m, nm) entries are filled up to nmax/mmax.
pub fn g0_2e_2d(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let nroots = envs.nrys_roots;
    let nmax = envs.li_ceil + envs.lj_ceil;   // max ij degree
    let mmax = envs.lk_ceil + envs.ll_ceil;   // max kl degree
    let dn = envs.g2d_ijmax;                  // stride for n-direction (ij)
    let dm = envs.g2d_klmax;                  // stride for m-direction (kl)
    let g_size = envs.g_size;

    // Split g into three components
    let (gxyz, _) = g.split_at_mut(3 * g_size);
    let (gx_gy, gz) = gxyz.split_at_mut(2 * g_size);
    let (gx, gy)    = gx_gy.split_at_mut(g_size);

    // Initialise base level gx=gy=1 for each root (gz already set by g0_2e)
    for i in 0..nroots {
        gx[i] = 1.0;
        gy[i] = 1.0;
    }

    for i in 0..nroots {
        let c00x = bc.c00x[i];
        let c00y = bc.c00y[i];
        let c00z = bc.c00z[i];
        let c0px = bc.c0px[i];
        let c0py = bc.c0py[i];
        let c0pz = bc.c0pz[i];
        let b10  = bc.b10[i];
        let b01  = bc.b01[i];
        let b00  = bc.b00[i];

        // ─── n-direction (ij recursion): g[m=0, n+1] = c00*g[n] + n*b10*g[n-1] ───
        if nmax > 0 {
            let mut s0x = gx[i];
            let mut s0y = gy[i];
            let mut s0z = gz[i];
            let mut s1x = c00x * s0x;
            let mut s1y = c00y * s0y;
            let mut s1z = c00z * s0z;
            gx[i + dn]   = s1x;
            gy[i + dn]   = s1y;
            gz[i + dn]   = s1z;
            for n in 1..nmax {
                let s2x = c00x * s1x + (n as f64) * b10 * s0x;
                let s2y = c00y * s1y + (n as f64) * b10 * s0y;
                let s2z = c00z * s1z + (n as f64) * b10 * s0z;
                gx[i + (n + 1) * dn] = s2x;
                gy[i + (n + 1) * dn] = s2y;
                gz[i + (n + 1) * dn] = s2z;
                s0x = s1x; s0y = s1y; s0z = s1z;
                s1x = s2x; s1y = s2y; s1z = s2z;
            }
        }

        // ─── m-direction (kl recursion): g[m+1, n=0] = c0p*g[m] + m*b01*g[m-1] ───
        if mmax > 0 {
            let mut s0x = gx[i];
            let mut s0y = gy[i];
            let mut s0z = gz[i];
            let mut s1x = c0px * s0x;
            let mut s1y = c0py * s0y;
            let mut s1z = c0pz * s0z;
            gx[i + dm]   = s1x;
            gy[i + dm]   = s1y;
            gz[i + dm]   = s1z;
            for m in 1..mmax {
                let s2x = c0px * s1x + (m as f64) * b01 * s0x;
                let s2y = c0py * s1y + (m as f64) * b01 * s0y;
                let s2z = c0pz * s1z + (m as f64) * b01 * s0z;
                gx[i + (m + 1) * dm] = s2x;
                gy[i + (m + 1) * dm] = s2y;
                gz[i + (m + 1) * dm] = s2z;
                s0x = s1x; s0y = s1y; s0z = s1z;
                s1x = s2x; s1y = s2y; s1z = s2z;
            }

            // ─── n=1, m recursion using b00 ─────────────────────────────────────────
            if nmax > 0 {
                let mut s0x = gx[i + dn];
                let mut s0y = gy[i + dn];
                let mut s0z = gz[i + dn];
                let mut s1x = c0px * s0x + b00 * gx[i];
                let mut s1y = c0py * s0y + b00 * gy[i];
                let mut s1z = c0pz * s0z + b00 * gz[i];
                gx[i + dn + dm] = s1x;
                gy[i + dn + dm] = s1y;
                gz[i + dn + dm] = s1z;
                for m in 1..mmax {
                    let s2x = c0px * s1x + (m as f64) * b01 * s0x + b00 * gx[i + m * dm];
                    let s2y = c0py * s1y + (m as f64) * b01 * s0y + b00 * gy[i + m * dm];
                    let s2z = c0pz * s1z + (m as f64) * b01 * s0z + b00 * gz[i + m * dm];
                    gx[i + dn + (m + 1) * dm] = s2x;
                    gy[i + dn + (m + 1) * dm] = s2y;
                    gz[i + dn + (m + 1) * dm] = s2z;
                    s0x = s1x; s0y = s1y; s0z = s1z;
                    s1x = s2x; s1y = s2y; s1z = s2z;
                }
            }
        }

        // ─── Mixed m≥1, n≥2: g[m, n+1] = c00*g[m,n] + n*b10*g[m,n-1] + m*b00*g[m-1,n] ───
        for m in 1..=mmax {
            let off = m * dm + i;
            let mut s0x = gx[off];
            let mut s0y = gy[off];
            let mut s0z = gz[off];
            let mut s1x = gx[off + dn];
            let mut s1y = gy[off + dn];
            let mut s1z = gz[off + dn];
            for n in 1..nmax {
                let s2x = c00x * s1x + (n as f64) * b10 * s0x + (m as f64) * b00 * gx[off + n * dn - dm];
                let s2y = c00y * s1y + (n as f64) * b10 * s0y + (m as f64) * b00 * gy[off + n * dn - dm];
                let s2z = c00z * s1z + (n as f64) * b10 * s0z + (m as f64) * b00 * gz[off + n * dn - dm];
                gx[off + (n + 1) * dn] = s2x;
                gy[off + (n + 1) * dn] = s2y;
                gz[off + (n + 1) * dn] = s2z;
                s0x = s1x; s0y = s1y; s0z = s1z;
                s1x = s2x; s1y = s2y; s1z = s2z;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// 4D horizontal transfer functions
// ─────────────────────────────────────────────────────────────────

/// 2d based on l,j — lj transfer (default: ibase=false, kbase=false).
/// g(i,k→,l,j) from the 2D (l+j direction) filled by g0_2e_2d.
pub fn g0_lj2d_4d(g: &mut [f64], envs: &EnvVars) {
    let li = envs.li_ceil;
    let lk = envs.lk_ceil;
    if li == 0 && lk == 0 { return; }

    let nmax = envs.li_ceil + envs.lj_ceil;
    let mmax = envs.lk_ceil + envs.ll_ceil;
    let nroots = envs.nrys_roots;
    let di = envs.g_stride_i;
    let dk = envs.g_stride_k;
    let dl = envs.g_stride_l;
    let dj = envs.g_stride_j;
    let g_size = envs.g_size;
    let [rix, riy, riz] = envs.rirj;
    let [rkx, rky, rkz] = envs.rkrl;

    // i-step: g(i,...,j) = rirj * g(i-1,...,j) + g(i-1,...,j+1)
    for i in 1..=li {
        for j in 0..=(nmax - i) {
            for l in 0..=mmax {
                for k in 0..=lk {
                    let ptr = j * dj + l * dl + k * dk + i * di;
                    for c in 0..3 {
                        let base = c * g_size;
                        let r = [rix, riy, riz][c];
                        for n in 0..nroots {
                            g[base + ptr + n] = r * g[base + ptr - di + n] + g[base + ptr - di + dj + n];
                        }
                    }
                }
            }
        }
    }

    // k-step: g(...,k,l,...) = rkrl * g(...,k-1,l,...) + g(...,k-1,l+1,...)
    for k in 1..=lk {
        for l in 0..=(mmax - k) {
            for j in 0..=(nmax - li).min(dj / nroots - 1) {
                let ptr = j * dj + l * dl + k * dk;
                for n in 0..dk {  // dk = nroots * dli
                    for c in 0..3 {
                        let base = c * g_size;
                        let r = [rkx, rky, rkz][c];
                        g[base + ptr + n] = r * g[base + ptr - dk + n] + g[base + ptr - dk + dl + n];
                    }
                }
            }
        }
    }
}

/// Simpler loop-based lj2d_4d matching the C reference exactly
pub fn g0_lj2d_4d_ref(g: &mut [f64], envs: &EnvVars) {
    let li = envs.li_ceil;
    let lk = envs.lk_ceil;
    if li == 0 && lk == 0 { return; }

    let nmax = envs.li_ceil + envs.lj_ceil;
    let mmax = envs.lk_ceil + envs.ll_ceil;
    let lj   = envs.lj_ceil;
    let nroots = envs.nrys_roots;
    let di = envs.g_stride_i;
    let dk = envs.g_stride_k;
    let dl = envs.g_stride_l;
    let dj = envs.g_stride_j;
    let g_size = envs.g_size;
    let [rix, riy, riz] = envs.rirj;
    let [rkx, rky, rkz] = envs.rkrl;

    // i-step
    for i in 1..=li {
        for j in 0..=(nmax - i) {
            for l in 0..=mmax {
                for k in 0..=lk {
                    let ptr = j * dj + l * dl + k * dk + i * di;
                    for n in 0..nroots {
                        g[         ptr + n] = rix * g[         ptr - di + n] + g[         ptr - di + dj + n];
                        g[g_size + ptr + n] = riy * g[g_size + ptr - di + n] + g[g_size + ptr - di + dj + n];
                        g[2*g_size+ptr + n] = riz * g[2*g_size+ptr - di + n] + g[2*g_size+ptr - di + dj + n];
                    }
                }
            }
        }
    }

    // k-step
    for k in 1..=lk {
        for l in 0..=(mmax - k) {
            for j in 0..=lj {
                let ptr = j * dj + l * dl + k * dk;
                for n in 0..dk {
                    g[         ptr + n] = rkx * g[         ptr - dk + n] + g[         ptr - dk + dl + n];
                    g[g_size + ptr + n] = rky * g[g_size + ptr - dk + n] + g[g_size + ptr - dk + dl + n];
                    g[2*g_size+ptr + n] = rkz * g[2*g_size+ptr - dk + n] + g[2*g_size+ptr - dk + dl + n];
                }
            }
        }
    }
}

/// 2d based on k,j — kj transfer (kbase=true, ibase=false).
pub fn g0_kj2d_4d(g: &mut [f64], envs: &EnvVars) {
    let li = envs.li_ceil;
    let ll = envs.ll_ceil;
    if li == 0 && ll == 0 { return; }

    let nmax = envs.li_ceil + envs.lj_ceil;
    let mmax = envs.lk_ceil + envs.ll_ceil;
    let lj   = envs.lj_ceil;
    let nroots = envs.nrys_roots;
    let di = envs.g_stride_i;
    let dk = envs.g_stride_k;
    let dl = envs.g_stride_l;
    let dj = envs.g_stride_j;
    let g_size = envs.g_size;
    let [rix, riy, riz] = envs.rirj;
    let [rkx, rky, rkz] = envs.rkrl;

    // i-step: g(i,...,j) = rirj * g(i-1,...,j) + g(i-1,...,j+1)
    for i in 1..=li {
        for j in 0..=(nmax - i) {
            for k in 0..=mmax {
                let ptr = j * dj + k * dk + i * di;
                for n in 0..nroots {
                    g[         ptr + n] = rix * g[         ptr - di + n] + g[         ptr - di + dj + n];
                    g[g_size + ptr + n] = riy * g[g_size + ptr - di + n] + g[g_size + ptr - di + dj + n];
                    g[2*g_size+ptr + n] = riz * g[2*g_size+ptr - di + n] + g[2*g_size+ptr - di + dj + n];
                }
            }
        }
    }

    // l-step: g(...,k,l,...) = rkrl * g(...,k,l-1,...) + g(...,k+1,l-1,...)
    for l in 1..=ll {
        for k in 0..=(mmax - l) {
            for j in 0..=lj {
                let ptr = j * dj + l * dl + k * dk;
                for n in 0..dk {
                    g[         ptr + n] = rkx * g[         ptr - dl + n] + g[         ptr - dl + dk + n];
                    g[g_size + ptr + n] = rky * g[g_size + ptr - dl + n] + g[g_size + ptr - dl + dk + n];
                    g[2*g_size+ptr + n] = rkz * g[2*g_size+ptr - dl + n] + g[2*g_size+ptr - dl + dk + n];
                }
            }
        }
    }
}

/// 2d based on i,l — il transfer (ibase=true, kbase=false).
pub fn g0_il2d_4d(g: &mut [f64], envs: &EnvVars) {
    let lk = envs.lk_ceil;
    let lj = envs.lj_ceil;
    if lj == 0 && lk == 0 { return; }

    let nmax = envs.li_ceil + envs.lj_ceil;
    let mmax = envs.lk_ceil + envs.ll_ceil;
    let ll   = envs.ll_ceil;
    let nroots = envs.nrys_roots;
    let di = envs.g_stride_i;
    let dk = envs.g_stride_k;
    let dl = envs.g_stride_l;
    let dj = envs.g_stride_j;
    let g_size = envs.g_size;
    let [rix, riy, riz] = envs.rirj;
    let [rkx, rky, rkz] = envs.rkrl;

    // k-step: g(...,k,l,...) = rkrl * g(...,k-1,l,...) + g(...,k-1,l+1,...)
    for k in 1..=lk {
        for l in 0..=(mmax - k) {
            for i in 0..=nmax {
                let ptr = l * dl + k * dk + i * di;
                for n in 0..nroots {
                    g[         ptr + n] = rkx * g[         ptr - dk + n] + g[         ptr - dk + dl + n];
                    g[g_size + ptr + n] = rky * g[g_size + ptr - dk + n] + g[g_size + ptr - dk + dl + n];
                    g[2*g_size+ptr + n] = rkz * g[2*g_size+ptr - dk + n] + g[2*g_size+ptr - dk + dl + n];
                }
            }
        }
    }

    // j-step: g(i,...,j) = rirj * g(i,...,j-1) + g(i+1,...,j-1)
    for j in 1..=lj {
        for l in 0..=ll {
            for k in 0..=lk {
                let ptr = j * dj + l * dl + k * dk;
                let n_end = dk - di * j;  // same as C: ptr+dk-di*j
                for n in 0..n_end {
                    g[         ptr + n] = rix * g[         ptr - dj + n] + g[         ptr - dj + di + n];
                    g[g_size + ptr + n] = riy * g[g_size + ptr - dj + n] + g[g_size + ptr - dj + di + n];
                    g[2*g_size+ptr + n] = riz * g[2*g_size+ptr - dj + n] + g[2*g_size+ptr - dj + di + n];
                }
            }
        }
    }
}

/// 2d based on i,k — ik transfer (ibase=true, kbase=true).
pub fn g0_ik2d_4d(g: &mut [f64], envs: &EnvVars) {
    let lj = envs.lj_ceil;
    let ll = envs.ll_ceil;
    if lj == 0 && ll == 0 { return; }

    let nmax = envs.li_ceil + envs.lj_ceil;
    let mmax = envs.lk_ceil + envs.ll_ceil;
    let lk   = envs.lk_ceil;
    let nroots = envs.nrys_roots;
    let di = envs.g_stride_i;
    let dk = envs.g_stride_k;
    let dl = envs.g_stride_l;
    let dj = envs.g_stride_j;
    let g_size = envs.g_size;
    let [rix, riy, riz] = envs.rirj;
    let [rkx, rky, rkz] = envs.rkrl;

    // l-step: g(...,k,l,...) = rkrl * g(...,k,l-1,...) + g(...,k+1,l-1,...)
    for l in 1..=ll {
        for k in 0..=(mmax - l) {
            for i in 0..=nmax {
                let ptr = l * dl + k * dk + i * di;
                for n in 0..nroots {
                    g[         ptr + n] = rkx * g[         ptr - dl + n] + g[         ptr - dl + dk + n];
                    g[g_size + ptr + n] = rky * g[g_size + ptr - dl + n] + g[g_size + ptr - dl + dk + n];
                    g[2*g_size+ptr + n] = rkz * g[2*g_size+ptr - dl + n] + g[2*g_size+ptr - dl + dk + n];
                }
            }
        }
    }

    // j-step: g(i,...,j) = rirj * g(i,...,j-1) + g(i+1,...,j-1)
    for j in 1..=lj {
        for l in 0..=ll {
            for k in 0..=lk {
                let ptr = j * dj + l * dl + k * dk;
                let n_end = dk - di * j;
                for n in 0..n_end {
                    g[         ptr + n] = rix * g[         ptr - dj + n] + g[         ptr - dj + di + n];
                    g[g_size + ptr + n] = riy * g[g_size + ptr - dj + n] + g[g_size + ptr - dj + di + n];
                    g[2*g_size+ptr + n] = riz * g[2*g_size+ptr - dj + n] + g[2*g_size+ptr - dj + di + n];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Unrolled fast path (rys_order ≤ 2)
// ─────────────────────────────────────────────────────────────────

/// Fast-path 2D+4D for small angular momentum cases (rys_order ≤ 2).
/// Dispatches on (li_ceil<<6)|(lj_ceil<<4)|(lk_ceil<<2)|ll_ceil.
pub fn g0_2e_2d4d_unrolled(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let type_ijkl = (envs.li_ceil << 6) | (envs.lj_ceil << 4)
                  | (envs.lk_ceil << 2) |  envs.ll_ceil;
    match type_ijkl {
        0b00000000 => _g0_2d4d_0000(g, bc, envs),
        0b00000001 => _g0_2d4d_0001(g, bc, envs),
        0b00000010 => _g0_2d4d_0002(g, bc, envs),
        0b00000011 => _g0_2d4d_0003(g, bc, envs),
        0b00000100 => _g0_2d4d_0010(g, bc, envs),
        0b00000101 => _g0_2d4d_0011(g, bc, envs),
        0b00000110 => _g0_2d4d_0012(g, bc, envs),
        0b00001000 => _g0_2d4d_0020(g, bc, envs),
        0b00001001 => _g0_2d4d_0021(g, bc, envs),
        0b00001100 => _g0_2d4d_0030(g, bc, envs),
        0b00010000 => _g0_2d4d_0100(g, bc, envs),
        0b00010001 => _g0_2d4d_0101(g, bc, envs),
        0b00010010 => _g0_2d4d_0102(g, bc, envs),
        0b00010100 => _g0_2d4d_0110(g, bc, envs),
        0b00010101 => _g0_2d4d_0111(g, bc, envs),
        0b00011000 => _g0_2d4d_0120(g, bc, envs),
        0b00100000 => _g0_2d4d_0200(g, bc, envs),
        0b00100001 => _g0_2d4d_0201(g, bc, envs),
        0b00100100 => _g0_2d4d_0210(g, bc, envs),
        0b00110000 => _g0_2d4d_0300(g, bc, envs),
        0b01000000 => _g0_2d4d_1000(g, bc, envs),
        0b01000001 => _g0_2d4d_1001(g, bc, envs),
        0b01000010 => _g0_2d4d_1002(g, bc, envs),
        0b01000100 => _g0_2d4d_1010(g, bc, envs),
        0b01000101 => _g0_2d4d_1011(g, bc, envs),
        0b01001000 => _g0_2d4d_1020(g, bc, envs),
        0b01010000 => _g0_2d4d_1100(g, bc, envs),
        0b01010001 => _g0_2d4d_1101(g, bc, envs),
        0b01010100 => _g0_2d4d_1110(g, bc, envs),
        0b01100000 => _g0_2d4d_1200(g, bc, envs),
        0b10000000 => _g0_2d4d_2000(g, bc, envs),
        0b10000001 => _g0_2d4d_2001(g, bc, envs),
        0b10000100 => _g0_2d4d_2010(g, bc, envs),
        0b10010000 => _g0_2d4d_2100(g, bc, envs),
        0b11000000 => _g0_2d4d_3000(g, bc, envs),
        _ => {
            // Fall through to general path
            g0_2e_2d(g, bc, envs);
            match (envs.kbase, envs.ibase) {
                (true,  true)  => g0_ik2d_4d(g, envs),
                (true,  false) => g0_kj2d_4d(g, envs),
                (false, true)  => g0_il2d_4d(g, envs),
                (false, false) => g0_lj2d_4d_ref(g, envs),
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Unrolled helper functions (direct translation from g2e.c)
// Naming: _g0_2d4d_IJKL where IJKL = (li,lj,lk,ll)
// ─────────────────────────────────────────────────────────────────

#[inline(always)]
fn _g0_2d4d_0000(g: &mut [f64], _bc: &Rys2eT, _envs: &EnvVars) {
    g[0] = 1.0;
    g[1] = 1.0;
    // g[2] = w[0]  (already set)
}

#[inline(always)]
fn _g0_2d4d_0001(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (cpx, cpy, cpz) = (&bc.c0px, &bc.c0py, &bc.c0pz);
    g[0] = 1.0;
    g[1] = cpx[0];
    g[2] = 1.0;
    g[3] = cpy[0];
    // g[4] = w[0]
    g[5] = cpz[0] * g[4];
}

#[inline(always)]
fn _g0_2d4d_0002(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (cpx, cpy, cpz, b01) = (&bc.c0px, &bc.c0py, &bc.c0pz, &bc.b01);
    g[0] = 1.0; g[1] = 1.0;
    g[2] = cpx[0]; g[3] = cpx[1];
    g[4] = cpx[0]*cpx[0] + b01[0]; g[5] = cpx[1]*cpx[1] + b01[1];
    g[6] = 1.0; g[7] = 1.0;
    g[8] = cpy[0]; g[9] = cpy[1];
    g[10] = cpy[0]*cpy[0] + b01[0]; g[11] = cpy[1]*cpy[1] + b01[1];
    // g[12]=w[0], g[13]=w[1]
    g[14] = cpz[0]*g[12]; g[15] = cpz[1]*g[13];
    g[16] = cpz[0]*g[14] + b01[0]*g[12]; g[17] = cpz[1]*g[15] + b01[1]*g[13];
}

#[inline(always)]
fn _g0_2d4d_0003(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (cpx, cpy, cpz, b01) = (&bc.c0px, &bc.c0py, &bc.c0pz, &bc.b01);
    g[0] = 1.0; g[1] = 1.0;
    g[2] = cpx[0]; g[3] = cpx[1];
    g[4] = cpx[0]*cpx[0] + b01[0]; g[5] = cpx[1]*cpx[1] + b01[1];
    g[6] = cpx[0]*(g[4] + 2.0*b01[0]); g[7] = cpx[1]*(g[5] + 2.0*b01[1]);
    g[8] = 1.0; g[9] = 1.0;
    g[10] = cpy[0]; g[11] = cpy[1];
    g[12] = cpy[0]*cpy[0] + b01[0]; g[13] = cpy[1]*cpy[1] + b01[1];
    g[14] = cpy[0]*(g[12] + 2.0*b01[0]); g[15] = cpy[1]*(g[13] + 2.0*b01[1]);
    // g[16]=w[0], g[17]=w[1]
    g[18] = cpz[0]*g[16]; g[19] = cpz[1]*g[17];
    g[20] = cpz[0]*g[18] + b01[0]*g[16]; g[21] = cpz[1]*g[19] + b01[1]*g[17];
    g[22] = cpz[0]*g[20] + 2.0*b01[0]*g[18]; g[23] = cpz[1]*g[21] + 2.0*b01[1]*g[19];
}

#[inline(always)]
fn _g0_2d4d_0010(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    // same as 0001
    _g0_2d4d_0001(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_0011(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (cpx, cpy, cpz, b01) = (&bc.c0px, &bc.c0py, &bc.c0pz, &bc.b01);
    let (xkxl, ykyl, zkzl) = (envs.rkrl[0], envs.rkrl[1], envs.rkrl[2]);
    g[0] = 1.0; g[1] = 1.0;
    g[4] = cpx[0]; g[5] = cpx[1];
    g[6] = cpx[0]*(xkxl+cpx[0]) + b01[0]; g[7] = cpx[1]*(xkxl+cpx[1]) + b01[1];
    g[2] = xkxl + cpx[0]; g[3] = xkxl + cpx[1];
    g[12] = 1.0; g[13] = 1.0;
    g[16] = cpy[0]; g[17] = cpy[1];
    g[18] = cpy[0]*(ykyl+cpy[0]) + b01[0]; g[19] = cpy[1]*(ykyl+cpy[1]) + b01[1];
    g[14] = ykyl + cpy[0]; g[15] = ykyl + cpy[1];
    // g[24]=w[0], g[25]=w[1]
    g[28] = cpz[0]*g[24]; g[29] = cpz[1]*g[25];
    g[30] = g[28]*(zkzl+cpz[0]) + b01[0]*g[24]; g[31] = g[29]*(zkzl+cpz[1]) + b01[1]*g[25];
    g[26] = g[24]*(zkzl+cpz[0]); g[27] = g[25]*(zkzl+cpz[1]);
}

#[inline(always)]
fn _g0_2d4d_0012(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (cpx, cpy, cpz, b01) = (&bc.c0px, &bc.c0py, &bc.c0pz, &bc.b01);
    let (xkxl, ykyl, zkzl) = (envs.rkrl[0], envs.rkrl[1], envs.rkrl[2]);
    g[0] = 1.0; g[1] = 1.0;
    g[4] = cpx[0]; g[5] = cpx[1];
    g[8] = cpx[0]*cpx[0]+b01[0]; g[9] = cpx[1]*cpx[1]+b01[1];
    g[10] = g[8]*(xkxl+cpx[0])+cpx[0]*2.0*b01[0]; g[11] = g[9]*(xkxl+cpx[1])+cpx[1]*2.0*b01[1];
    g[6] = cpx[0]*(xkxl+cpx[0])+b01[0]; g[7] = cpx[1]*(xkxl+cpx[1])+b01[1];
    g[2] = xkxl+cpx[0]; g[3] = xkxl+cpx[1];
    g[16] = 1.0; g[17] = 1.0;
    g[20] = cpy[0]; g[21] = cpy[1];
    g[24] = cpy[0]*cpy[0]+b01[0]; g[25] = cpy[1]*cpy[1]+b01[1];
    g[26] = g[24]*(ykyl+cpy[0])+cpy[0]*2.0*b01[0]; g[27] = g[25]*(ykyl+cpy[1])+cpy[1]*2.0*b01[1];
    g[22] = cpy[0]*(ykyl+cpy[0])+b01[0]; g[23] = cpy[1]*(ykyl+cpy[1])+b01[1];
    g[18] = ykyl+cpy[0]; g[19] = ykyl+cpy[1];
    // g[32]=w[0], g[33]=w[1]
    g[36] = cpz[0]*g[32]; g[37] = cpz[1]*g[33];
    g[40] = cpz[0]*g[36]+b01[0]*g[32]; g[41] = cpz[1]*g[37]+b01[1]*g[33];
    g[42] = g[40]*(zkzl+cpz[0])+2.0*b01[0]*g[36]; g[43] = g[41]*(zkzl+cpz[1])+2.0*b01[1]*g[37];
    g[38] = g[36]*(zkzl+cpz[0])+b01[0]*g[32]; g[39] = g[37]*(zkzl+cpz[1])+b01[1]*g[33];
    g[34] = g[32]*(zkzl+cpz[0]); g[35] = g[33]*(zkzl+cpz[1]);
}

#[inline(always)]
fn _g0_2d4d_0020(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0002(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_0021(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (cpx, cpy, cpz, b01) = (&bc.c0px, &bc.c0py, &bc.c0pz, &bc.b01);
    let (xkxl, ykyl, zkzl) = (envs.rkrl[0], envs.rkrl[1], envs.rkrl[2]);
    g[0] = 1.0; g[1] = 1.0;
    g[2] = cpx[0]; g[3] = cpx[1];
    g[4] = cpx[0]*cpx[0]+b01[0]; g[5] = cpx[1]*cpx[1]+b01[1];
    g[8] = xkxl+cpx[0]; g[9] = xkxl+cpx[1];
    g[10] = cpx[0]*(xkxl+cpx[0])+b01[0]; g[11] = cpx[1]*(xkxl+cpx[1])+b01[1];
    g[12] = g[4]*(xkxl+cpx[0])+cpx[0]*2.0*b01[0]; g[13] = g[5]*(xkxl+cpx[1])+cpx[1]*2.0*b01[1];
    g[16] = 1.0; g[17] = 1.0;
    g[18] = cpy[0]; g[19] = cpy[1];
    g[20] = cpy[0]*cpy[0]+b01[0]; g[21] = cpy[1]*cpy[1]+b01[1];
    g[24] = ykyl+cpy[0]; g[25] = ykyl+cpy[1];
    g[26] = cpy[0]*(ykyl+cpy[0])+b01[0]; g[27] = cpy[1]*(ykyl+cpy[1])+b01[1];
    g[28] = g[20]*(ykyl+cpy[0])+cpy[0]*2.0*b01[0]; g[29] = g[21]*(ykyl+cpy[1])+cpy[1]*2.0*b01[1];
    // g[32]=w[0], g[33]=w[1]
    g[34] = cpz[0]*g[32]; g[35] = cpz[1]*g[33];
    g[36] = cpz[0]*g[34]+b01[0]*g[32]; g[37] = cpz[1]*g[35]+b01[1]*g[33];
    g[40] = g[32]*(zkzl+cpz[0]); g[41] = g[33]*(zkzl+cpz[1]);
    g[42] = g[34]*(zkzl+cpz[0])+b01[0]*g[32]; g[43] = g[35]*(zkzl+cpz[1])+b01[1]*g[33];
    g[44] = g[36]*(zkzl+cpz[0])+2.0*b01[0]*g[34]; g[45] = g[37]*(zkzl+cpz[1])+2.0*b01[1]*g[35];
}

#[inline(always)]
fn _g0_2d4d_0030(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0003(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_0100(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x, c0y, c0z) = (&bc.c00x, &bc.c00y, &bc.c00z);
    g[0] = 1.0; g[1] = c0x[0];
    g[2] = 1.0; g[3] = c0y[0];
    // g[4] = w[0]
    g[5] = c0z[0] * g[4];
}

#[inline(always)]
fn _g0_2d4d_0101(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x, c0y, c0z, cpx, cpy, cpz, b00) =
        (&bc.c00x, &bc.c00y, &bc.c00z, &bc.c0px, &bc.c0py, &bc.c0pz, &bc.b00);
    g[0] = 1.0; g[1] = 1.0;
    g[2] = cpx[0]; g[3] = cpx[1];
    g[4] = c0x[0]; g[5] = c0x[1];
    g[6] = cpx[0]*c0x[0]+b00[0]; g[7] = cpx[1]*c0x[1]+b00[1];
    g[8] = 1.0; g[9] = 1.0;
    g[10] = cpy[0]; g[11] = cpy[1];
    g[12] = c0y[0]; g[13] = c0y[1];
    g[14] = cpy[0]*c0y[0]+b00[0]; g[15] = cpy[1]*c0y[1]+b00[1];
    // g[16]=w[0], g[17]=w[1]
    g[18] = cpz[0]*g[16]; g[19] = cpz[1]*g[17];
    g[20] = c0z[0]*g[16]; g[21] = c0z[1]*g[17];
    g[22] = cpz[0]*g[20]+b00[0]*g[16]; g[23] = cpz[1]*g[21]+b00[1]*g[17];
}

#[inline(always)]
fn _g0_2d4d_0102(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b01) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b01);
    g[0]=1.0; g[1]=1.0;
    g[2]=cpx[0]; g[3]=cpx[1];
    g[6]=c0x[0]; g[7]=c0x[1];
    g[4]=cpx[0]*cpx[0]+b01[0]; g[5]=cpx[1]*cpx[1]+b01[1];
    g[8]=cpx[0]*c0x[0]+b00[0]; g[9]=cpx[1]*c0x[1]+b00[1];
    g[10]=cpx[0]*(g[8]+b00[0])+b01[0]*c0x[0]; g[11]=cpx[1]*(g[9]+b00[1])+b01[1]*c0x[1];
    g[12]=1.0; g[13]=1.0;
    g[14]=cpy[0]; g[15]=cpy[1];
    g[18]=c0y[0]; g[19]=c0y[1];
    g[16]=cpy[0]*cpy[0]+b01[0]; g[17]=cpy[1]*cpy[1]+b01[1];
    g[20]=cpy[0]*c0y[0]+b00[0]; g[21]=cpy[1]*c0y[1]+b00[1];
    g[22]=cpy[0]*(g[20]+b00[0])+b01[0]*c0y[0]; g[23]=cpy[1]*(g[21]+b00[1])+b01[1]*c0y[1];
    // g[24]=w[0], g[25]=w[1]
    g[26]=cpz[0]*g[24]; g[27]=cpz[1]*g[25];
    g[30]=c0z[0]*g[24]; g[31]=c0z[1]*g[25];
    g[28]=cpz[0]*g[26]+b01[0]*g[24]; g[29]=cpz[1]*g[27]+b01[1]*g[25];
    g[32]=cpz[0]*g[30]+b00[0]*g[24]; g[33]=cpz[1]*g[31]+b00[1]*g[25];
    g[34]=cpz[0]*g[32]+b01[0]*g[30]+b00[0]*g[26]; g[35]=cpz[1]*g[33]+b01[1]*g[31]+b00[1]*g[27];
}

#[inline(always)]
fn _g0_2d4d_0110(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0101(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_0111(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b01) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b01);
    let (xkxl, ykyl, zkzl) = (envs.rkrl[0], envs.rkrl[1], envs.rkrl[2]);
    g[0]=1.0; g[1]=1.0;
    g[12]=c0x[0]; g[13]=c0x[1];
    g[4]=cpx[0]; g[5]=cpx[1];
    g[16]=cpx[0]*c0x[0]+b00[0]; g[17]=cpx[1]*c0x[1]+b00[1];
    g[6]=cpx[0]*(xkxl+cpx[0])+b01[0]; g[7]=cpx[1]*(xkxl+cpx[1])+b01[1];
    g[18]=g[16]*(xkxl+cpx[0])+cpx[0]*b00[0]+b01[0]*c0x[0];
    g[19]=g[17]*(xkxl+cpx[1])+cpx[1]*b00[1]+b01[1]*c0x[1];
    g[2]=xkxl+cpx[0]; g[3]=xkxl+cpx[1];
    g[14]=c0x[0]*(xkxl+cpx[0])+b00[0]; g[15]=c0x[1]*(xkxl+cpx[1])+b00[1];
    g[24]=1.0; g[25]=1.0;
    g[36]=c0y[0]; g[37]=c0y[1];
    g[28]=cpy[0]; g[29]=cpy[1];
    g[40]=cpy[0]*c0y[0]+b00[0]; g[41]=cpy[1]*c0y[1]+b00[1];
    g[30]=cpy[0]*(ykyl+cpy[0])+b01[0]; g[31]=cpy[1]*(ykyl+cpy[1])+b01[1];
    g[42]=g[40]*(ykyl+cpy[0])+cpy[0]*b00[0]+b01[0]*c0y[0];
    g[43]=g[41]*(ykyl+cpy[1])+cpy[1]*b00[1]+b01[1]*c0y[1];
    g[26]=ykyl+cpy[0]; g[27]=ykyl+cpy[1];
    g[38]=c0y[0]*(ykyl+cpy[0])+b00[0]; g[39]=c0y[1]*(ykyl+cpy[1])+b00[1];
    // g[48]=w[0], g[49]=w[1]
    g[60]=c0z[0]*g[48]; g[61]=c0z[1]*g[49];
    g[52]=cpz[0]*g[48]; g[53]=cpz[1]*g[49];
    g[64]=cpz[0]*g[60]+b00[0]*g[48]; g[65]=cpz[1]*g[61]+b00[1]*g[49];
    g[54]=g[52]*(zkzl+cpz[0])+b01[0]*g[48]; g[55]=g[53]*(zkzl+cpz[1])+b01[1]*g[49];
    g[66]=g[64]*(zkzl+cpz[0])+b01[0]*g[60]+b00[0]*g[52];
    g[67]=g[65]*(zkzl+cpz[1])+b01[1]*g[61]+b00[1]*g[53];
    g[50]=g[48]*(zkzl+cpz[0]); g[51]=g[49]*(zkzl+cpz[1]);
    g[62]=g[60]*(zkzl+cpz[0])+b00[0]*g[48]; g[63]=g[61]*(zkzl+cpz[1])+b00[1]*g[49];
}

#[inline(always)]
fn _g0_2d4d_0120(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0102(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_0200(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,b10) = (&bc.c00x,&bc.c00y,&bc.c00z,&bc.b10);
    g[0]=1.0; g[1]=1.0;
    g[2]=c0x[0]; g[3]=c0x[1];
    g[4]=c0x[0]*c0x[0]+b10[0]; g[5]=c0x[1]*c0x[1]+b10[1];
    g[6]=1.0; g[7]=1.0;
    g[8]=c0y[0]; g[9]=c0y[1];
    g[10]=c0y[0]*c0y[0]+b10[0]; g[11]=c0y[1]*c0y[1]+b10[1];
    // g[12]=w[0], g[13]=w[1]
    g[14]=c0z[0]*g[12]; g[15]=c0z[1]*g[13];
    g[16]=c0z[0]*g[14]+b10[0]*g[12]; g[17]=c0z[1]*g[15]+b10[1]*g[13];
}

#[inline(always)]
fn _g0_2d4d_0201(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b10) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b10);
    g[0]=1.0; g[1]=1.0;
    g[4]=c0x[0]; g[5]=c0x[1];
    g[8]=c0x[0]*c0x[0]+b10[0]; g[9]=c0x[1]*c0x[1]+b10[1];
    g[2]=cpx[0]; g[3]=cpx[1];
    g[6]=cpx[0]*c0x[0]+b00[0]; g[7]=cpx[1]*c0x[1]+b00[1];
    g[10]=c0x[0]*(g[6]+b00[0])+b10[0]*cpx[0]; g[11]=c0x[1]*(g[7]+b00[1])+b10[1]*cpx[1];
    g[12]=1.0; g[13]=1.0;
    g[16]=c0y[0]; g[17]=c0y[1];
    g[20]=c0y[0]*c0y[0]+b10[0]; g[21]=c0y[1]*c0y[1]+b10[1];
    g[14]=cpy[0]; g[15]=cpy[1];
    g[18]=cpy[0]*c0y[0]+b00[0]; g[19]=cpy[1]*c0y[1]+b00[1];
    g[22]=c0y[0]*(g[18]+b00[0])+b10[0]*cpy[0]; g[23]=c0y[1]*(g[19]+b00[1])+b10[1]*cpy[1];
    // g[24]=w[0], g[25]=w[1]
    g[28]=c0z[0]*g[24]; g[29]=c0z[1]*g[25];
    g[32]=c0z[0]*g[28]+b10[0]*g[24]; g[33]=c0z[1]*g[29]+b10[1]*g[25];
    g[26]=cpz[0]*g[24]; g[27]=cpz[1]*g[25];
    g[30]=cpz[0]*g[28]+b00[0]*g[24]; g[31]=cpz[1]*g[29]+b00[1]*g[25];
    g[34]=c0z[0]*g[30]+b10[0]*g[26]+b00[0]*g[28]; g[35]=c0z[1]*g[31]+b10[1]*g[27]+b00[1]*g[29];
}

#[inline(always)]
fn _g0_2d4d_0210(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b10) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b10);
    g[0]=1.0; g[1]=1.0;
    g[2]=cpx[0]; g[3]=cpx[1];
    g[4]=c0x[0]; g[5]=c0x[1];
    g[6]=cpx[0]*c0x[0]+b00[0]; g[7]=cpx[1]*c0x[1]+b00[1];
    g[8]=c0x[0]*c0x[0]+b10[0]; g[9]=c0x[1]*c0x[1]+b10[1];
    g[10]=c0x[0]*(g[6]+b00[0])+b10[0]*cpx[0]; g[11]=c0x[1]*(g[7]+b00[1])+b10[1]*cpx[1];
    g[12]=1.0; g[13]=1.0;
    g[14]=cpy[0]; g[15]=cpy[1];
    g[16]=c0y[0]; g[17]=c0y[1];
    g[18]=cpy[0]*c0y[0]+b00[0]; g[19]=cpy[1]*c0y[1]+b00[1];
    g[20]=c0y[0]*c0y[0]+b10[0]; g[21]=c0y[1]*c0y[1]+b10[1];
    g[22]=c0y[0]*(g[18]+b00[0])+b10[0]*cpy[0]; g[23]=c0y[1]*(g[19]+b00[1])+b10[1]*cpy[1];
    // g[24]=w[0], g[25]=w[1]
    g[26]=cpz[0]*g[24]; g[27]=cpz[1]*g[25];
    g[28]=c0z[0]*g[24]; g[29]=c0z[1]*g[25];
    g[30]=cpz[0]*g[28]+b00[0]*g[24]; g[31]=cpz[1]*g[29]+b00[1]*g[25];
    g[32]=c0z[0]*g[28]+b10[0]*g[24]; g[33]=c0z[1]*g[29]+b10[1]*g[25];
    g[34]=c0z[0]*g[30]+b10[0]*g[26]+b00[0]*g[28]; g[35]=c0z[1]*g[31]+b10[1]*g[27]+b00[1]*g[29];
}

#[inline(always)]
fn _g0_2d4d_0300(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0003(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_1000(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0100(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_1001(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00);
    g[0]=1.0; g[1]=1.0;
    g[2]=c0x[0]; g[3]=c0x[1];
    g[4]=cpx[0]; g[5]=cpx[1];
    g[6]=cpx[0]*c0x[0]+b00[0]; g[7]=cpx[1]*c0x[1]+b00[1];
    g[8]=1.0; g[9]=1.0;
    g[10]=c0y[0]; g[11]=c0y[1];
    g[12]=cpy[0]; g[13]=cpy[1];
    g[14]=cpy[0]*c0y[0]+b00[0]; g[15]=cpy[1]*c0y[1]+b00[1];
    // g[16]=w[0], g[17]=w[1]
    g[18]=c0z[0]*g[16]; g[19]=c0z[1]*g[17];
    g[20]=cpz[0]*g[16]; g[21]=cpz[1]*g[17];
    g[22]=cpz[0]*g[18]+b00[0]*g[16]; g[23]=cpz[1]*g[19]+b00[1]*g[17];
}

#[inline(always)]
fn _g0_2d4d_1002(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b01) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b01);
    g[0]=1.0; g[1]=1.0;
    g[2]=c0x[0]; g[3]=c0x[1];
    g[4]=cpx[0]; g[5]=cpx[1];
    g[6]=cpx[0]*c0x[0]+b00[0]; g[7]=cpx[1]*c0x[1]+b00[1];
    g[8]=cpx[0]*cpx[0]+b01[0]; g[9]=cpx[1]*cpx[1]+b01[1];
    g[10]=cpx[0]*(g[6]+b00[0])+b01[0]*c0x[0]; g[11]=cpx[1]*(g[7]+b00[1])+b01[1]*c0x[1];
    g[12]=1.0; g[13]=1.0;
    g[14]=c0y[0]; g[15]=c0y[1];
    g[16]=cpy[0]; g[17]=cpy[1];
    g[18]=cpy[0]*c0y[0]+b00[0]; g[19]=cpy[1]*c0y[1]+b00[1];
    g[20]=cpy[0]*cpy[0]+b01[0]; g[21]=cpy[1]*cpy[1]+b01[1];
    g[22]=cpy[0]*(g[18]+b00[0])+b01[0]*c0y[0]; g[23]=cpy[1]*(g[19]+b00[1])+b01[1]*c0y[1];
    // g[24]=w[0], g[25]=w[1]
    g[26]=c0z[0]*g[24]; g[27]=c0z[1]*g[25];
    g[28]=cpz[0]*g[24]; g[29]=cpz[1]*g[25];
    g[30]=cpz[0]*g[26]+b00[0]*g[24]; g[31]=cpz[1]*g[27]+b00[1]*g[25];
    g[32]=cpz[0]*g[28]+b01[0]*g[24]; g[33]=cpz[1]*g[29]+b01[1]*g[25];
    g[34]=cpz[0]*g[30]+b01[0]*g[26]+b00[0]*g[28]; g[35]=cpz[1]*g[31]+b01[1]*g[27]+b00[1]*g[29];
}

#[inline(always)]
fn _g0_2d4d_1010(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_1001(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_1011(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b01) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b01);
    let (xkxl, ykyl, zkzl) = (envs.rkrl[0], envs.rkrl[1], envs.rkrl[2]);
    g[0]=1.0; g[1]=1.0;
    g[2]=c0x[0]; g[3]=c0x[1];
    g[8]=cpx[0]; g[9]=cpx[1];
    g[10]=cpx[0]*c0x[0]+b00[0]; g[11]=cpx[1]*c0x[1]+b00[1];
    g[12]=cpx[0]*(xkxl+cpx[0])+b01[0]; g[13]=cpx[1]*(xkxl+cpx[1])+b01[1];
    g[14]=g[10]*(xkxl+cpx[0])+cpx[0]*b00[0]+b01[0]*c0x[0];
    g[15]=g[11]*(xkxl+cpx[1])+cpx[1]*b00[1]+b01[1]*c0x[1];
    g[4]=xkxl+cpx[0]; g[5]=xkxl+cpx[1];
    g[6]=c0x[0]*(xkxl+cpx[0])+b00[0]; g[7]=c0x[1]*(xkxl+cpx[1])+b00[1];
    g[24]=1.0; g[25]=1.0;
    g[26]=c0y[0]; g[27]=c0y[1];
    g[32]=cpy[0]; g[33]=cpy[1];
    g[34]=cpy[0]*c0y[0]+b00[0]; g[35]=cpy[1]*c0y[1]+b00[1];
    g[36]=cpy[0]*(ykyl+cpy[0])+b01[0]; g[37]=cpy[1]*(ykyl+cpy[1])+b01[1];
    g[38]=g[34]*(ykyl+cpy[0])+cpy[0]*b00[0]+b01[0]*c0y[0];
    g[39]=g[35]*(ykyl+cpy[1])+cpy[1]*b00[1]+b01[1]*c0y[1];
    g[28]=ykyl+cpy[0]; g[29]=ykyl+cpy[1];
    g[30]=c0y[0]*(ykyl+cpy[0])+b00[0]; g[31]=c0y[1]*(ykyl+cpy[1])+b00[1];
    // g[48]=w[0], g[49]=w[1]
    g[50]=c0z[0]*g[48]; g[51]=c0z[1]*g[49];
    g[56]=cpz[0]*g[48]; g[57]=cpz[1]*g[49];
    g[58]=cpz[0]*g[50]+b00[0]*g[48]; g[59]=cpz[1]*g[51]+b00[1]*g[49];
    g[60]=g[56]*(zkzl+cpz[0])+b01[0]*g[48]; g[61]=g[57]*(zkzl+cpz[1])+b01[1]*g[49];
    g[62]=g[58]*(zkzl+cpz[0])+b01[0]*g[50]+b00[0]*g[56];
    g[63]=g[59]*(zkzl+cpz[1])+b01[1]*g[51]+b00[1]*g[57];
    g[52]=g[48]*(zkzl+cpz[0]); g[53]=g[49]*(zkzl+cpz[1]);
    g[54]=g[50]*(zkzl+cpz[0])+b00[0]*g[48]; g[55]=g[51]*(zkzl+cpz[1])+b00[1]*g[49];
}

#[inline(always)]
fn _g0_2d4d_1020(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_1002(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_1100(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (c0x,c0y,c0z,b10) = (&bc.c00x,&bc.c00y,&bc.c00z,&bc.b10);
    let (xixj, yiyj, zizj) = (envs.rirj[0], envs.rirj[1], envs.rirj[2]);
    g[0]=1.0; g[1]=1.0;
    g[4]=c0x[0]; g[5]=c0x[1];
    g[6]=c0x[0]*(xixj+c0x[0])+b10[0]; g[7]=c0x[1]*(xixj+c0x[1])+b10[1];
    g[2]=xixj+c0x[0]; g[3]=xixj+c0x[1];
    g[12]=1.0; g[13]=1.0;
    g[16]=c0y[0]; g[17]=c0y[1];
    g[18]=c0y[0]*(yiyj+c0y[0])+b10[0]; g[19]=c0y[1]*(yiyj+c0y[1])+b10[1];
    g[14]=yiyj+c0y[0]; g[15]=yiyj+c0y[1];
    // g[24]=w[0], g[25]=w[1]
    g[28]=c0z[0]*g[24]; g[29]=c0z[1]*g[25];
    g[30]=g[28]*(zizj+c0z[0])+b10[0]*g[24]; g[31]=g[29]*(zizj+c0z[1])+b10[1]*g[25];
    g[26]=g[24]*(zizj+c0z[0]); g[27]=g[25]*(zizj+c0z[1]);
}

#[inline(always)]
fn _g0_2d4d_1101(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b10) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b10);
    let (xixj, yiyj, zizj) = (envs.rirj[0], envs.rirj[1], envs.rirj[2]);
    g[0]=1.0; g[1]=1.0;
    g[8]=c0x[0]; g[9]=c0x[1];
    g[4]=cpx[0]; g[5]=cpx[1];
    g[12]=cpx[0]*c0x[0]+b00[0]; g[13]=cpx[1]*c0x[1]+b00[1];
    g[10]=c0x[0]*(xixj+c0x[0])+b10[0]; g[11]=c0x[1]*(xixj+c0x[1])+b10[1];
    g[2]=xixj+c0x[0]; g[3]=xixj+c0x[1];
    g[14]=g[12]*(xixj+c0x[0])+c0x[0]*b00[0]+b10[0]*cpx[0];
    g[15]=g[13]*(xixj+c0x[1])+c0x[1]*b00[1]+b10[1]*cpx[1];
    g[6]=cpx[0]*(xixj+c0x[0])+b00[0]; g[7]=cpx[1]*(xixj+c0x[1])+b00[1];
    g[24]=1.0; g[25]=1.0;
    g[32]=c0y[0]; g[33]=c0y[1];
    g[28]=cpy[0]; g[29]=cpy[1];
    g[36]=cpy[0]*c0y[0]+b00[0]; g[37]=cpy[1]*c0y[1]+b00[1];
    g[34]=c0y[0]*(yiyj+c0y[0])+b10[0]; g[35]=c0y[1]*(yiyj+c0y[1])+b10[1];
    g[26]=yiyj+c0y[0]; g[27]=yiyj+c0y[1];
    g[38]=g[36]*(yiyj+c0y[0])+c0y[0]*b00[0]+b10[0]*cpy[0];
    g[39]=g[37]*(yiyj+c0y[1])+c0y[1]*b00[1]+b10[1]*cpy[1];
    g[30]=cpy[0]*(yiyj+c0y[0])+b00[0]; g[31]=cpy[1]*(yiyj+c0y[1])+b00[1];
    // g[48]=w[0], g[49]=w[1]
    g[56]=c0z[0]*g[48]; g[57]=c0z[1]*g[49];
    g[52]=cpz[0]*g[48]; g[53]=cpz[1]*g[49];
    g[60]=cpz[0]*g[56]+b00[0]*g[48]; g[61]=cpz[1]*g[57]+b00[1]*g[49];
    g[58]=g[56]*(zizj+c0z[0])+b10[0]*g[48]; g[59]=g[57]*(zizj+c0z[1])+b10[1]*g[49];
    g[50]=g[48]*(zizj+c0z[0]); g[51]=g[49]*(zizj+c0z[1]);
    g[62]=g[60]*(zizj+c0z[0])+b10[0]*g[52]+b00[0]*g[56];
    g[63]=g[61]*(zizj+c0z[1])+b10[1]*g[53]+b00[1]*g[57];
    g[54]=zizj*g[52]+cpz[0]*g[56]+b00[0]*g[48];
    g[55]=zizj*g[53]+cpz[1]*g[57]+b00[1]*g[49];
}

#[inline(always)]
fn _g0_2d4d_1110(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    _g0_2d4d_1101(g, bc, envs);
}

#[inline(always)]
fn _g0_2d4d_1200(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (c0x,c0y,c0z,b10) = (&bc.c00x,&bc.c00y,&bc.c00z,&bc.b10);
    let (xixj, yiyj, zizj) = (envs.rirj[0], envs.rirj[1], envs.rirj[2]);
    g[0]=1.0; g[1]=1.0;
    g[4]=c0x[0]; g[5]=c0x[1];
    g[8]=c0x[0]*c0x[0]+b10[0]; g[9]=c0x[1]*c0x[1]+b10[1];
    g[10]=g[8]*(xixj+c0x[0])+c0x[0]*2.0*b10[0]; g[11]=g[9]*(xixj+c0x[1])+c0x[1]*2.0*b10[1];
    g[6]=c0x[0]*(xixj+c0x[0])+b10[0]; g[7]=c0x[1]*(xixj+c0x[1])+b10[1];
    g[2]=xixj+c0x[0]; g[3]=xixj+c0x[1];
    g[16]=1.0; g[17]=1.0;
    g[20]=c0y[0]; g[21]=c0y[1];
    g[24]=c0y[0]*c0y[0]+b10[0]; g[25]=c0y[1]*c0y[1]+b10[1];
    g[26]=g[24]*(yiyj+c0y[0])+c0y[0]*2.0*b10[0]; g[27]=g[25]*(yiyj+c0y[1])+c0y[1]*2.0*b10[1];
    g[22]=c0y[0]*(yiyj+c0y[0])+b10[0]; g[23]=c0y[1]*(yiyj+c0y[1])+b10[1];
    g[18]=yiyj+c0y[0]; g[19]=yiyj+c0y[1];
    // g[32]=w[0], g[33]=w[1]
    g[36]=c0z[0]*g[32]; g[37]=c0z[1]*g[33];
    g[40]=c0z[0]*g[36]+b10[0]*g[32]; g[41]=c0z[1]*g[37]+b10[1]*g[33];
    g[42]=g[40]*(zizj+c0z[0])+2.0*b10[0]*g[36]; g[43]=g[41]*(zizj+c0z[1])+2.0*b10[1]*g[37];
    g[38]=g[36]*(zizj+c0z[0])+b10[0]*g[32]; g[39]=g[37]*(zizj+c0z[1])+b10[1]*g[33];
    g[34]=g[32]*(zizj+c0z[0]); g[35]=g[33]*(zizj+c0z[1]);
}

#[inline(always)]
fn _g0_2d4d_2000(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0200(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_2001(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    let (c0x,c0y,c0z,cpx,cpy,cpz,b00,b10) =
        (&bc.c00x,&bc.c00y,&bc.c00z,&bc.c0px,&bc.c0py,&bc.c0pz,&bc.b00,&bc.b10);
    g[0]=1.0; g[1]=1.0;
    g[2]=c0x[0]; g[3]=c0x[1];
    g[4]=c0x[0]*c0x[0]+b10[0]; g[5]=c0x[1]*c0x[1]+b10[1];
    g[6]=cpx[0]; g[7]=cpx[1];
    g[8]=cpx[0]*c0x[0]+b00[0]; g[9]=cpx[1]*c0x[1]+b00[1];
    g[10]=c0x[0]*(g[8]+b00[0])+b10[0]*cpx[0]; g[11]=c0x[1]*(g[9]+b00[1])+b10[1]*cpx[1];
    g[12]=1.0; g[13]=1.0;
    g[14]=c0y[0]; g[15]=c0y[1];
    g[16]=c0y[0]*c0y[0]+b10[0]; g[17]=c0y[1]*c0y[1]+b10[1];
    g[18]=cpy[0]; g[19]=cpy[1];
    g[20]=cpy[0]*c0y[0]+b00[0]; g[21]=cpy[1]*c0y[1]+b00[1];
    g[22]=c0y[0]*(g[20]+b00[0])+b10[0]*cpy[0]; g[23]=c0y[1]*(g[21]+b00[1])+b10[1]*cpy[1];
    // g[24]=w[0], g[25]=w[1]
    g[26]=c0z[0]*g[24]; g[27]=c0z[1]*g[25];
    g[28]=c0z[0]*g[26]+b10[0]*g[24]; g[29]=c0z[1]*g[27]+b10[1]*g[25];
    g[30]=cpz[0]*g[24]; g[31]=cpz[1]*g[25];
    g[32]=cpz[0]*g[26]+b00[0]*g[24]; g[33]=cpz[1]*g[27]+b00[1]*g[25];
    g[34]=c0z[0]*g[32]+b10[0]*g[30]+b00[0]*g[26]; g[35]=c0z[1]*g[33]+b10[1]*g[31]+b00[1]*g[27];
}

#[inline(always)]
fn _g0_2d4d_2010(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_2001(g, bc, _envs);
}

#[inline(always)]
fn _g0_2d4d_2100(g: &mut [f64], bc: &Rys2eT, envs: &EnvVars) {
    let (c0x,c0y,c0z,b10) = (&bc.c00x,&bc.c00y,&bc.c00z,&bc.b10);
    let (xixj, yiyj, zizj) = (envs.rirj[0], envs.rirj[1], envs.rirj[2]);
    g[0]=1.0; g[1]=1.0;
    g[2]=c0x[0]; g[3]=c0x[1];
    g[4]=c0x[0]*c0x[0]+b10[0]; g[5]=c0x[1]*c0x[1]+b10[1];
    g[12]=g[4]*(xixj+c0x[0])+c0x[0]*2.0*b10[0]; g[13]=g[5]*(xixj+c0x[1])+c0x[1]*2.0*b10[1];
    g[10]=c0x[0]*(xixj+c0x[0])+b10[0]; g[11]=c0x[1]*(xixj+c0x[1])+b10[1];
    g[8]=xixj+c0x[0]; g[9]=xixj+c0x[1];
    g[16]=1.0; g[17]=1.0;
    g[18]=c0y[0]; g[19]=c0y[1];
    g[20]=c0y[0]*c0y[0]+b10[0]; g[21]=c0y[1]*c0y[1]+b10[1];
    g[28]=g[20]*(yiyj+c0y[0])+c0y[0]*2.0*b10[0]; g[29]=g[21]*(yiyj+c0y[1])+c0y[1]*2.0*b10[1];
    g[26]=c0y[0]*(yiyj+c0y[0])+b10[0]; g[27]=c0y[1]*(yiyj+c0y[1])+b10[1];
    g[24]=yiyj+c0y[0]; g[25]=yiyj+c0y[1];
    // g[32]=w[0], g[33]=w[1]
    g[34]=c0z[0]*g[32]; g[35]=c0z[1]*g[33];
    g[36]=c0z[0]*g[34]+b10[0]*g[32]; g[37]=c0z[1]*g[35]+b10[1]*g[33];
    g[44]=g[36]*(zizj+c0z[0])+2.0*b10[0]*g[34]; g[45]=g[37]*(zizj+c0z[1])+2.0*b10[1]*g[35];
    g[42]=g[34]*(zizj+c0z[0])+b10[0]*g[32]; g[43]=g[35]*(zizj+c0z[1])+b10[1]*g[33];
    g[40]=g[32]*(zizj+c0z[0]); g[41]=g[33]*(zizj+c0z[1]);
}

#[inline(always)]
fn _g0_2d4d_3000(g: &mut [f64], bc: &Rys2eT, _envs: &EnvVars) {
    _g0_2d4d_0003(g, bc, _envs);
}

// ─────────────────────────────────────────────────────────────────
// Index-building for integral extraction
// ─────────────────────────────────────────────────────────────────

/// Build the `idx` lookup table for selecting the correct g-buffer elements
/// when contracting into the final integral array.
///
/// Returns a `Vec<usize>` of length `3 * nf` where every triplet `(idx[3n], idx[3n+1], idx[3n+2])`
/// gives the indices into g for the x, y, z polynomial factors of the n-th Cartesian integral component.
/// The ordering over Cartesian components is (j, l, k, i) with fast-running i.
pub fn g2e_index_xyz(envs: &EnvVars) -> Vec<usize> {
    let i_l = envs.i_l;
    let j_l = envs.j_l;
    let k_l = envs.k_l;
    let l_l = envs.l_l;
    let nfi  = envs.nfi;
    let nfj  = envs.nfj;
    let nfk  = envs.nfk;
    let nfl  = envs.nfl;
    let di   = envs.g_stride_i;
    let dk   = envs.g_stride_k;
    let dl   = envs.g_stride_l;
    let dj   = envs.g_stride_j;
    let g_size = envs.g_size;

    let (i_nx, i_ny, i_nz) = cart_comp_l(i_l);
    let (j_nx, j_ny, j_nz) = cart_comp_l(j_l);
    let (k_nx, k_ny, k_nz) = cart_comp_l(k_l);
    let (l_nx, l_ny, l_nz) = cart_comp_l(l_l);

    let ofx: usize = 0;
    let ofy: usize = g_size;
    let ofz: usize = g_size * 2;

    let mut idx = Vec::with_capacity(3 * nfi * nfj * nfk * nfl);

    for j in 0..nfj {
        for l in 0..nfl {
            let oflx = ofx + dj * j_nx[j] as usize + dl * l_nx[l] as usize;
            let ofly = ofy + dj * j_ny[j] as usize + dl * l_ny[l] as usize;
            let oflz = ofz + dj * j_nz[j] as usize + dl * l_nz[l] as usize;
            for k in 0..nfk {
                let ofkx = oflx + dk * k_nx[k] as usize;
                let ofky = ofly + dk * k_ny[k] as usize;
                let ofkz = oflz + dk * k_nz[k] as usize;
                for i in 0..nfi {
                    idx.push(ofkx + di * i_nx[i] as usize);
                    idx.push(ofky + di * i_ny[i] as usize);
                    idx.push(ofkz + di * i_nz[i] as usize);
                }
            }
        }
    }

    idx
}
