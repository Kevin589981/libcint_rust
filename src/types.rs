/// Data structure definitions corresponding to libcint's atm/bas/env layout.
///
/// libcint uses three flat integer/float arrays to describe the molecular system:
///   atm[natm][ATM_SLOTS]  — atom data
///   bas[nbas][BAS_SLOTS]  — basis function data
///   env[]                 — floating-point storage for coordinates/exponents/coefficients

// ─── env global offsets (cint.h) ────────────────────────────────────────────
pub const PTR_EXPCUTOFF:     usize = 0;
pub const PTR_COMMON_ORIG:   usize = 1;  // 3 floats
pub const PTR_RINV_ORIG:     usize = 4;  // 3 floats
pub const PTR_RINV_ZETA:     usize = 7;
pub const PTR_RANGE_OMEGA:   usize = 8;
pub const PTR_F12_ZETA:      usize = 9;
pub const PTR_GTG_ZETA:      usize = 10;
pub const NGRIDS:            usize = 11;
pub const PTR_GRIDS:         usize = 12;
pub const PTR_ENV_START:     usize = 20;

// ─── atm slot indices ────────────────────────────────────────────────────────
pub const CHARGE_OF:     usize = 0;
pub const PTR_COORD:     usize = 1;
pub const NUC_MOD_OF:    usize = 2;
pub const PTR_ZETA:      usize = 3;
pub const PTR_FRAC_CHARGE: usize = 4;
pub const RESERVE_ATMSLOT: usize = 5;
pub const ATM_SLOTS:     usize = 6;

// ─── bas slot indices ────────────────────────────────────────────────────────
pub const ATOM_OF:       usize = 0;
pub const ANG_OF:        usize = 1;
pub const NPRIM_OF:      usize = 2;
pub const NCTR_OF:       usize = 3;
pub const KAPPA_OF:      usize = 4;
pub const PTR_EXP:       usize = 5;
pub const PTR_COEFF:     usize = 6;
pub const RESERVE_BASLOT: usize = 7;
pub const BAS_SLOTS:     usize = 8;

// ─── ng[] indices (integral type parameters) ─────────────────────────────────
pub const IINC:   usize = 0;
pub const JINC:   usize = 1;
pub const KINC:   usize = 2;
pub const LINC:   usize = 3;
pub const GSHIFT: usize = 4;
pub const POS_E1: usize = 5;
pub const POS_E2: usize = 6;
pub const TENSOR: usize = 7;

// ─── Other constants ─────────────────────────────────────────────────────────
pub const EXPCUTOFF:    f64 = 60.0;
pub const MIN_EXPCUTOFF: f64 = 40.0;
pub const MXRYSROOTS:   usize = 32;
pub const ANG_MAX:      usize = 15;
pub const LMAX1:        usize = 16;
pub const SQRTPI: f64 = 1.7724538509055160272981674833411451;

// ─── Safe views into the raw C arrays ────────────────────────────────────────

/// Read-only view of one atom row in the `atm` array.
#[derive(Debug, Clone, Copy)]
pub struct AtmSlot<'a> {
    data: &'a [i32],  // slice of ATM_SLOTS elements
}

impl<'a> AtmSlot<'a> {
    #[inline]
    pub unsafe fn from_raw(atm: *const i32, iatom: usize) -> Self {
        let data = std::slice::from_raw_parts(atm.add(iatom * ATM_SLOTS), ATM_SLOTS);
        AtmSlot { data }
    }

    #[inline] pub fn charge(&self)    -> i32 { self.data[CHARGE_OF] }
    #[inline] pub fn ptr_coord(&self) -> usize { self.data[PTR_COORD] as usize }
    #[inline] pub fn nuc_mod(&self)   -> i32 { self.data[NUC_MOD_OF] }
    #[inline] pub fn ptr_zeta(&self)  -> usize { self.data[PTR_ZETA] as usize }
}

/// Read-only view of one basis-shell row in the `bas` array.
#[derive(Debug, Clone, Copy)]
pub struct BasSlot<'a> {
    data: &'a [i32],
}

impl<'a> BasSlot<'a> {
    #[inline]
    pub unsafe fn from_raw(bas: *const i32, ibas: usize) -> Self {
        let data = std::slice::from_raw_parts(bas.add(ibas * BAS_SLOTS), BAS_SLOTS);
        BasSlot { data }
    }

    #[inline] pub fn atom_of(&self)  -> usize { self.data[ATOM_OF]  as usize }
    #[inline] pub fn ang_of(&self)   -> usize { self.data[ANG_OF]   as usize }
    #[inline] pub fn nprim_of(&self) -> usize { self.data[NPRIM_OF] as usize }
    #[inline] pub fn nctr_of(&self)  -> usize { self.data[NCTR_OF]  as usize }
    #[inline] pub fn kappa_of(&self) -> i32   { self.data[KAPPA_OF] }
    #[inline] pub fn ptr_exp(&self)  -> usize { self.data[PTR_EXP]  as usize }
    #[inline] pub fn ptr_coeff(&self)-> usize { self.data[PTR_COEFF]as usize }
}

/// Safe wrapper around the flat `env` f64 array.
#[derive(Debug, Clone, Copy)]
pub struct Env<'a> {
    data: &'a [f64],
}

impl<'a> Env<'a> {
    /// # Safety:  caller must ensure `env` is valid for `len` elements.
    pub unsafe fn from_raw(env: *const f64, len: usize) -> Self {
        Env { data: std::slice::from_raw_parts(env, len) }
    }

    #[inline] pub fn get(&self, idx: usize) -> f64 { self.data[idx] }

    /// Coordinates of atom `iatom` (3 consecutive f64 at atm[iatom].PTR_COORD).
    pub fn coords(&self, ptr: usize) -> [f64; 3] {
        [self.data[ptr], self.data[ptr+1], self.data[ptr+2]]
    }

    /// Slice of primitive exponents for a shell.
    pub fn exps(&self, ptr: usize, nprim: usize) -> &[f64] {
        &self.data[ptr..ptr+nprim]
    }

    /// Slice of contraction coefficients for a shell (column-major: nprim × nctr).
    pub fn coeffs(&self, ptr: usize, nprim: usize, nctr: usize) -> &[f64] {
        &self.data[ptr..ptr+nprim*nctr]
    }

    pub fn raw(&self) -> &[f64] { self.data }
    pub fn ptr(&self) -> *const f64 { self.data.as_ptr() }
    pub fn range_omega(&self) -> f64 { self.data[PTR_RANGE_OMEGA] }
    pub fn expcutoff(&self) -> f64 {
        let v = self.data[PTR_EXPCUTOFF];
        if v == 0.0 { EXPCUTOFF } else { v.max(MIN_EXPCUTOFF) + 1.0 }
    }
}

/// Computation-state struct passed through the integral engine.
/// Rust equivalent of `CINTEnvVars`.
#[derive(Debug, Clone)]
pub struct EnvVars {
    // Angular momenta of the four shells
    pub i_l: usize,
    pub j_l: usize,
    pub k_l: usize,
    pub l_l: usize,

    // Number of Cartesian components: nfi = (i_l+1)(i_l+2)/2
    pub nfi: usize,
    pub nfj: usize,
    pub nfk: usize,
    pub nfl: usize,
    /// Product nfi*nfj*nfk*nfl
    pub nf: usize,

    /// Number of contraction coefficients per shell
    pub x_ctr: [usize; 4],

    /// Angular-momentum ceilings (= l + ng[IINC] etc.)
    pub li_ceil: usize,
    pub lj_ceil: usize,
    pub lk_ceil: usize,
    pub ll_ceil: usize,

    pub nrys_roots: usize,
    pub rys_order:  usize,

    /// Strides in the flat g-array (in units of one f64)
    pub g_stride_i: usize,
    pub g_stride_k: usize,
    pub g_stride_l: usize,
    pub g_stride_j: usize,
    /// Total size of one xyz component of g
    pub g_size:     usize,

    /// Which 2D/4D algorithm variant to use
    pub g2d_ijmax: usize,
    pub g2d_klmax: usize,

    /// ibase / kbase flags
    pub ibase: bool,
    pub kbase: bool,

    /// Common overall factor: (π³)·2/√π · fac_sp(i)·fac_sp(j)·fac_sp(k)·fac_sp(l)
    pub common_factor: f64,
    pub expcutoff: f64,

    // Current primitive exponents (updated per primitive loop iteration)
    pub ai: f64,
    pub aj: f64,
    pub ak: f64,
    pub al: f64,

    /// Overall scale factor for current primitive combo: fac × exp(...) × coefficients
    pub fac: f64,

    /// R_ij = R_i - R_j  (or R_j - R_i depending on ibase)
    pub rirj: [f64; 3],
    /// R_kl = R_k - R_l  (or R_l - R_k depending on kbase)
    pub rkrl: [f64; 3],

    /// Weighted center of ij pair: R_ij = (ai·Ri + aj·Rj)/(ai+aj)
    pub rij: [f64; 3],
    /// Weighted center of kl pair
    pub rkl: [f64; 3],

    /// Pointer (offset into env) to atom coordinates for each shell
    pub ri: [f64; 3],
    pub rj: [f64; 3],
    pub rk: [f64; 3],
    pub rl: [f64; 3],

    /// rx_in_rijrx = ri or rj (depending on ibase); similar for kl
    pub rx_in_rijrx: [f64; 3],
    pub rx_in_rklrx: [f64; 3],

    // ng parameters
    pub ncomp_e1: usize,
    pub ncomp_e2: usize,
    pub ncomp_tensor: usize,
    pub gbits: usize,
}

impl EnvVars {
    /// Return number of spherical components for shell i: 2*l+1
    #[inline] pub fn nsi(&self) -> usize { 2*self.i_l + 1 }
    #[inline] pub fn nsj(&self) -> usize { 2*self.j_l + 1 }
    #[inline] pub fn nsk(&self) -> usize { 2*self.k_l + 1 }
    #[inline] pub fn nsl(&self) -> usize { 2*self.l_l + 1 }

    /// Number of output elements per contraction block (Cartesian)
    pub fn nc(&self) -> usize {
        self.nf * self.x_ctr[0] * self.x_ctr[1] * self.x_ctr[2] * self.x_ctr[3]
    }
}

/// l-dependent normalisation factor for s and p orbitals.
/// For l≥2 the factor is absorbed into the cart2sph coefficients.
pub fn common_fac_sp(l: usize) -> f64 {
    match l {
        0 => 0.282094791773878143,  // 1/sqrt(4π)
        1 => 0.488602511902919921,  // sqrt(3/(4π))
        _ => 1.0,
    }
}

/// Number of Cartesian GTOs for angular momentum l: (l+1)(l+2)/2
#[inline]
pub fn ncart(l: usize) -> usize { (l + 1) * (l + 2) / 2 }

/// Fill `nx`, `ny`, `nz` with (nx,ny,nz) tuples for all Cartesian components at ang l.
/// Order: highest x first, consistent with libcint convention.
///
/// Note: prefer `cart_comp_l` for hot loops (returns `&'static [i32]`, no allocation).
pub fn cart_comp(nx: &mut Vec<i32>, ny: &mut Vec<i32>, nz: &mut Vec<i32>, l: usize) {
    nx.clear(); ny.clear(); nz.clear();
    for ix in (0..=l as i32).rev() {
        for iy in (0..=(l as i32 - ix)).rev() {
            let iz = l as i32 - ix - iy;
            nx.push(ix);
            ny.push(iy);
            nz.push(iz);
        }
    }
}

// Static Cartesian component tables for l = 0..=4, generated by build.rs.
// Provides `cart_comp_l(l) -> (&'static [i32], &'static [i32], &'static [i32])`.
include!(concat!(env!("OUT_DIR"), "/cart_tables.rs"));
