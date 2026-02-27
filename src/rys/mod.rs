/// Boys function and Rys quadrature roots/weights.
///
/// Boys function:  F_m(t) = ∫₀¹ s^{2m} e^{-t·s²} ds
/// Rys roots uᵢ:  ∫₀¹ f(t²) e^{-x·t²} dt ≈ Σᵢ wᵢ · f(tᵢ²),  tᵢ² = uᵢ/(uᵢ+1)

pub const PIE4: f64 = 0.7853981633974483;
pub const SQRTPIE4: f64 = 0.8862269254527580;

extern "C" { fn erf(x: f64) -> f64; }

/// Boys function: fills f[0..=mmax] with F_0(t)..F_mmax(t).
pub fn gamma_inc_like(f: &mut [f64], t: f64, mmax: usize) {
    const EPS: f64 = 1e-15;
    const TP: [f64; 40] = [
        0., 0., 0.866, 1.295, 1.705, 2.106, 2.501, 2.892, 3.280, 3.666,
        4.050, 4.433, 4.814, 5.194, 5.573, 5.951, 6.328, 6.704, 7.079, 7.454,
        7.827, 8.200, 8.572, 8.944, 9.315, 9.685,10.054,10.423,10.791,11.159,
       11.526,11.893,12.259,12.624,12.989,13.354,13.718,14.082,14.445,14.808,
    ];
    if t < 1e-15 {
        let mut b = 1.0_f64;
        for fm in f.iter_mut().take(mmax + 1) { *fm = 1.0 / b; b += 2.0; }
        return;
    }
    let tp = if mmax < TP.len() { TP[mmax] } else { 4.0 + mmax as f64 * 0.378 };
    if t <= tp {
        let e = 0.5 * (-t).exp();
        let mut b = mmax as f64 + 0.5;
        let mut x = e; let mut sum = e;
        loop { x *= t / b; b += 1.0; sum += x; if x <= EPS * e { break; } }
        f[mmax] = sum / (mmax as f64 + 0.5);
        let mut bi = mmax as f64 + 0.5;
        for i in (0..mmax).rev() { bi -= 1.0; f[i] = (e + t * f[i + 1]) / bi; }
    } else {
        let sr = t.sqrt();
        f[0] = SQRTPIE4 / sr * unsafe { erf(sr) };
        let e = (-t).exp();
        let b = 0.5 / t;
        for i in 1..=mmax { f[i] = b * ((2 * i - 1) as f64 * f[i - 1] - e); }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Wheeler / modified-Chebyshev: build Jacobi (tridiagonal) matrix from moments
// ──────────────────────────────────────────────────────────────────────────────
fn build_jacobi(n: usize, mu: &[f64], alpha: &mut [f64], beta: &mut [f64]) {
    let len = 2 * n;
    let mut s = vec![vec![0.0_f64; len + 1]; n + 2];
    for k in 0..len { s[1][k] = mu[k]; }
    alpha[0] = mu[1] / mu[0];
    beta[0] = 0.0;
    for j in 1..n {
        for k in j..(len - j) {
            s[j + 1][k] = s[j][k + 1]
                - alpha[j - 1] * s[j][k]
                - beta[j - 1] * s[j - 1][k];
        }
        if s[j + 1][j].abs() < 1e-300 { break; }
        alpha[j] = s[j + 1][j + 1] / s[j + 1][j]
            - s[j][j] / s[j][j - 1];
        beta[j] = s[j + 1][j] / s[j][j - 1];
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Symmetric tridiagonal eigensolver – implicit-shift QL (NR tqli).
// d = diagonal (in: values, out: eigenvalues).
// e = subdiagonal, e[0] unused (in-out: workspace).
// z = n×n (in: identity; out: eigenvectors as columns, z[row][col]).
// ──────────────────────────────────────────────────────────────────────────────
fn tqli(d: &mut Vec<f64>, e: &mut Vec<f64>, z: &mut Vec<Vec<f64>>) {
    let n = d.len();
    for i in 1..n { e[i - 1] = e[i]; }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iter = 0i32;
        loop {
            let m = (l..n - 1).find(|&m| {
                let dd = d[m].abs() + d[m + 1].abs();
                e[m].abs() <= f64::EPSILON * dd
            }).unwrap_or(n - 1);
            if m == l { break; }
            assert!(iter < 64, "tqli: convergence failure");
            iter += 1;
            let g0 = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r0 = (g0 * g0 + 1.0).sqrt();
            let mut g = d[m] - d[l] + e[l] / (g0 + r0.copysign(g0));
            let (mut s, mut c, mut p) = (1.0_f64, 1.0_f64, 0.0_f64);
            let mut done = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = (f * f + g * g).sqrt();
                e[i + 1] = r;
                if r.abs() < 1e-300 {
                    d[i + 1] -= p; e[m] = 0.0; done = true; break;
                }
                s = f / r; c = g / r;
                let ge = d[i + 1] - p;
                let r2 = (d[i] - ge) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = ge + p;
                g = c * r2 - b;
                for row in 0..n {
                    let fv = z[row][i + 1];
                    z[row][i + 1] = s * z[row][i] + c * fv;
                    z[row][i] = c * z[row][i] - s * fv;
                }
            }
            if !done { d[l] -= p; e[l] = g; e[m] = 0.0; }
        }
    }
    // Sort ascending by eigenvalue
    for i in 0..n {
        let (mut k, mut p) = (i, d[i]);
        for j in i + 1..n { if d[j] < p { k = j; p = d[j]; } }
        if k != i {
            d[k] = d[i]; d[i] = p;
            for row in 0..n { z[row].swap(i, k); }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public interface
// ──────────────────────────────────────────────────────────────────────────────

/// Compute Rys quadrature nodes u[0..n] and weights w[0..n] for parameter x.
pub fn rys_roots(nroots: usize, x: f64, u: &mut [f64], w: &mut [f64]) {
    assert!(nroots >= 1);
    assert!(u.len() >= nroots && w.len() >= nroots);
    let mut mu = vec![0.0_f64; 2 * nroots];
    if x < 3e-7 {
        for k in 0..2 * nroots {
            mu[k] = 1.0 / (2 * k + 1) as f64 - x / (2 * k + 3) as f64;
        }
    } else {
        gamma_inc_like(&mut mu, x, 2 * nroots - 1);
    }
    rys_from_moments(nroots, &mu, u, w);
}

/// Compute Rys nodes and weights from moments μ_k = F_k(x).
pub fn rys_from_moments(n: usize, mu: &[f64], u: &mut [f64], w: &mut [f64]) {
    let mu0 = mu[0];
    if n == 1 {
        let t2 = (mu[1] / mu[0]).max(0.0).min(1.0 - 1e-14);
        u[0] = t2 / (1.0 - t2).max(1e-300);
        w[0] = mu0;
        return;
    }
    let mut alpha = vec![0.0_f64; n];
    let mut beta  = vec![0.0_f64; n];
    build_jacobi(n, mu, &mut alpha, &mut beta);

    let mut d: Vec<f64> = alpha;
    let mut e: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { beta[i].max(0.0).sqrt() }).collect();
    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|i| { let mut v = vec![0.0_f64; n]; v[i] = 1.0; v })
        .collect();
    tqli(&mut d, &mut e, &mut z);

    for i in 0..n {
        let t2 = d[i].max(0.0).min(1.0 - 1e-14);
        u[i] = t2 / (1.0 - t2).max(1e-300);
        let v0 = z[0][i];
        w[i] = mu0 * v0 * v0;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boys_f0_zero() {
        let mut f = [0.0_f64; 1];
        gamma_inc_like(&mut f, 0.0, 0);
        assert!((f[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn boys_recurrence() {
        // F_{m-1}(x) = (2x·F_m(x) + exp(-x)) / (2m-1)
        let x = 2.5_f64;
        let mut f = [0.0_f64; 5];
        gamma_inc_like(&mut f, x, 4);
        let e = (-x).exp();
        for m in 1..=4_usize {
            let lhs = f[m - 1];
            let rhs = (2.0 * x * f[m] + e) / (2 * m - 1) as f64;
            assert!((lhs - rhs).abs() < 1e-12, "recurrence m={}: lhs={} rhs={}", m, lhs, rhs);
        }
    }

    #[test]
    fn rys1_exact() {
        let x = 1.5_f64;
        let mut u = [0.0_f64; 1]; let mut w = [0.0_f64; 1];
        rys_roots(1, x, &mut u, &mut w);
        let mut f = [0.0_f64; 2];
        gamma_inc_like(&mut f, x, 1);
        assert!((w[0] - f[0]).abs() < 1e-12, "w={} F0={}", w[0], f[0]);
        let eu = f[1] / (f[0] - f[1]);
        assert!((u[0] - eu).abs() < 1e-10, "u={} exp={}", u[0], eu);
    }

    #[test]
    fn rys2_moments() {
        let x = 2.0_f64;
        let mut u = [0.0_f64; 2]; let mut w = [0.0_f64; 2];
        rys_roots(2, x, &mut u, &mut w);
        let mut f = [0.0_f64; 4];
        gamma_inc_like(&mut f, x, 3);
        let sw: f64 = w.iter().sum();
        assert!((sw - f[0]).abs() < 1e-10, "sum_w={} F0={}", sw, f[0]);
        let sm1: f64 = (0..2).map(|i| w[i] * u[i] / (u[i] + 1.0)).sum();
        assert!((sm1 - f[1]).abs() < 1e-9, "sum_w*t2={} F1={}", sm1, f[1]);
    }

    #[test]
    fn rys3_moments() {
        let x = 3.0_f64;
        let mut u = [0.0_f64; 3]; let mut w = [0.0_f64; 3];
        rys_roots(3, x, &mut u, &mut w);
        let mut f = [0.0_f64; 6];
        gamma_inc_like(&mut f, x, 5);
        let sw: f64 = w.iter().sum();
        assert!((sw - f[0]).abs() < 1e-8, "sum_w={} F0={}", sw, f[0]);
        let sm1: f64 = (0..3).map(|i| w[i] * u[i] / (u[i] + 1.0)).sum();
        assert!((sm1 - f[1]).abs() < 1e-8, "sum_wt2={} F1={}", sm1, f[1]);
    }
}
