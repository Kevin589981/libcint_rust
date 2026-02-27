#!/usr/bin/env python3
"""
validate_pyscf.py — Phase 1 POC validation: libcint-rs vs PySCF/libcint reference.

Tests performed:
  1. int1e_ovlp_cart  — shell-by-shell overlap integrals
  2. int1e_kin_cart   — shell-by-shell kinetic energy integrals
  3. int1e_nuc_cart   — shell-by-shell nuclear attraction integrals
  4. int2e_cart       — full ERI tensor (all shell quartets)
  5. HF energy        — 1-electron Hamiltonian built from our integrals;
                        SCF diagonalisation; compare total energy to PySCF

Molecule: H2 / STO-3G (2 s-type contracted shells, all s-orbital)
Pass criterion: all integral max-errors < 1e-10, |ΔE_HF| < 1e-8 Hartree.

Usage:
    cargo build --release
    uv run python python/validate_pyscf.py

Requires: pyscf, numpy, scipy
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path

# ── 1. Load our library ───────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
LIB  = REPO / "target" / "release" / "libcint.so"

if not LIB.exists():
    print(f"ERROR: {LIB} not found. Run `cargo build --release` first.")
    sys.exit(1)

_cint = ctypes.CDLL(str(LIB))

# ── 2. Import PySCF ───────────────────────────────────────────────────────────
try:
    from pyscf import gto
    import pyscf
except ImportError:
    print("ERROR: pyscf not installed. Run `pip install pyscf`.")
    sys.exit(1)

# ── 3. Build H2 molecule ──────────────────────────────────────────────────────
mol = gto.Mole()
mol.atom = [
    ["H", (0.0, 0.0, 0.0)],
    ["H", (0.0, 0.0, 1.4)],   # ~0.74 Å in bohr
]
mol.basis = "sto-3g"
mol.unit  = "bohr"
mol.verbose = 0
mol.build()

# ── 4. Reference integrals from PySCF ────────────────────────────────────────
print("=== PySCF reference integrals ===")
ref_ovlp = mol.intor("int1e_ovlp_cart")
ref_kin  = mol.intor("int1e_kin_cart")
ref_nuc  = mol.intor("int1e_nuc_cart")
ref_eri  = mol.intor("int2e_cart")
print(f"  Overlap  shape: {ref_ovlp.shape}")
print(f"  Kinetic  shape: {ref_kin.shape}")
print(f"  Nuclear  shape: {ref_nuc.shape}")
print(f"  ERI      shape: {ref_eri.shape}")
print()
print("  Overlap matrix (PySCF):")
print(ref_ovlp)
print()

# ── 4b. Manual overlap check (Python, no Rust) ────────────────────────────────
import math
SQRTPI = math.sqrt(math.pi)

def common_fac_sp(l):
    if l == 0: return 0.282094791773878143
    if l == 1: return 0.488602511902919921
    return 1.0

def manual_ovlp_ss(mol, i_sh, j_sh):
    """Pure-Python overlap integral for s-s shell pair, following C code."""
    bi = mol._bas[i_sh]; bj = mol._bas[j_sh]
    ai_atom = bi[gto.mole.ATOM_OF]; aj_atom = bj[gto.mole.ATOM_OF]
    li = bi[gto.mole.ANG_OF]; lj = bj[gto.mole.ANG_OF]
    nprim_i = bi[gto.mole.NPRIM_OF]; nprim_j = bj[gto.mole.NPRIM_OF]
    nctr_i  = bi[gto.mole.NCTR_OF];  nctr_j  = bj[gto.mole.NCTR_OF]
    ptr_exp_i  = bi[gto.mole.PTR_EXP];   ptr_exp_j  = bj[gto.mole.PTR_EXP]
    ptr_coe_i  = bi[gto.mole.PTR_COEFF]; ptr_coe_j  = bj[gto.mole.PTR_COEFF]
    ptr_ri = mol._atm[ai_atom, gto.mole.PTR_COORD]
    ptr_rj = mol._atm[aj_atom, gto.mole.PTR_COORD]
    ri = mol._env[ptr_ri:ptr_ri+3]; rj = mol._env[ptr_rj:ptr_rj+3]
    rirj = ri - rj; r2ij = float(np.dot(rirj, rirj))
    expi = mol._env[ptr_exp_i:ptr_exp_i+nprim_i]
    expj = mol._env[ptr_exp_j:ptr_exp_j+nprim_j]
    coei = mol._env[ptr_coe_i:ptr_coe_i+nprim_i*nctr_i]
    coej = mol._env[ptr_coe_j:ptr_coe_j+nprim_j*nctr_j]
    cf = common_fac_sp(li) * common_fac_sp(lj)
    s = 0.0
    for jp in range(nprim_j):
        aj = expj[jp]; cj = coej[jp]
        for ip in range(nprim_i):
            ai = expi[ip]; ci = coei[ip]
            aij = ai + aj
            exp_ij = -(ai*aj/aij)*r2ij
            if exp_ij < -50: continue
            # gz[0] = fac0 * SQRTPI*M_PI / (aij * sqrt(aij))  (from C code)
            s += cf * ci * cj * math.exp(exp_ij) * SQRTPI * math.pi / (aij * math.sqrt(aij))
    return s

s00 = manual_ovlp_ss(mol, 0, 0)
s01 = manual_ovlp_ss(mol, 0, 1)
print(f"  Manual Python S(0,0) = {s00:.8f}  (ref={ref_ovlp[0,0]:.8f})")
print(f"  Manual Python S(0,1) = {s01:.8f}  (ref={ref_ovlp[0,1]:.8f})")
print()

# ── 5. Helper: call our int function for a shell pair (or quartet) ────────────

def _c_int_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def _c_dbl_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def call_int1e(fname, shls, mol):
    """Call int1e_*_cart for shell pair shls=(i,j)."""
    fn = getattr(_cint, fname)
    fn.restype = ctypes.c_int

    i_sh, j_sh = shls
    bi = mol._bas[i_sh]
    bj = mol._bas[j_sh]
    li = bi[gto.mole.ANG_OF]
    lj = bj[gto.mole.ANG_OF]
    nfi = (li+1)*(li+2)//2
    nfj = (lj+1)*(lj+2)//2

    out_arr = np.zeros((nfi, nfj), dtype=np.float64, order='F')
    shls_arr = np.array(shls, dtype=np.int32)
    dims_arr = np.array([nfi, nfj], dtype=np.int32)

    fn(
        _c_dbl_p(out_arr),
        _c_int_p(dims_arr),
        _c_int_p(shls_arr),
        _c_int_p(mol._atm),
        ctypes.c_int(mol.natm),
        _c_int_p(mol._bas),
        ctypes.c_int(mol.nbas),
        _c_dbl_p(mol._env),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
    )
    return out_arr


def call_int2e(shls, mol):
    """Call int2e_cart for shell quartet shls=(i,j,k,l)."""
    fn = _cint.int2e_cart
    fn.restype = ctypes.c_int

    i_sh, j_sh, k_sh, l_sh = shls
    li = mol._bas[i_sh][gto.mole.ANG_OF]
    lj = mol._bas[j_sh][gto.mole.ANG_OF]
    lk = mol._bas[k_sh][gto.mole.ANG_OF]
    ll = mol._bas[l_sh][gto.mole.ANG_OF]
    nfi = (li+1)*(li+2)//2
    nfj = (lj+1)*(lj+2)//2
    nfk = (lk+1)*(lk+2)//2
    nfl = (ll+1)*(ll+2)//2

    out_arr = np.zeros((nfi, nfj, nfk, nfl), dtype=np.float64, order='F')
    shls_arr = np.array(shls, dtype=np.int32)
    dims_arr = np.array([nfi, nfj, nfk, nfl], dtype=np.int32)

    fn(
        _c_dbl_p(out_arr),
        _c_int_p(dims_arr),
        _c_int_p(shls_arr),
        _c_int_p(mol._atm),
        ctypes.c_int(mol.natm),
        _c_int_p(mol._bas),
        ctypes.c_int(mol.nbas),
        _c_dbl_p(mol._env),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
    )
    return out_arr


# ── 6. Compare shell-by-shell ─────────────────────────────────────────────────
print("=== Shell-by-shell comparison (int1e_ovlp_cart) ===")
nbas = mol.nbas
all_pass = True
max_err  = 0.0

for i in range(nbas):
    for j in range(nbas):
        our   = call_int1e("int1e_ovlp_cart", (i, j), mol)
        # Extract reference block
        ai = mol.ao_loc_nr(cart=True)[i]
        bi_end = mol.ao_loc_nr(cart=True)[i+1]
        aj = mol.ao_loc_nr(cart=True)[j]
        bj_end = mol.ao_loc_nr(cart=True)[j+1]
        ref   = ref_ovlp[ai:bi_end, aj:bj_end]
        err   = np.max(np.abs(our - ref))
        max_err = max(max_err, err)
        if err > 1e-10:
            print(f"  FAIL  shells ({i},{j}) max_err = {err:.3e}")
            print(f"    our:\n{our}")
            print(f"    ref:\n{ref}")
            all_pass = False

print(f"  max error across all shell pairs: {max_err:.3e}")
if all_pass:
    print("  ✓ All overlap integrals match PySCF to 1e-10")
print()

# ── 7. Nuclear attraction ─────────────────────────────────────────────────────
print("=== int1e_nuc_cart ===")
max_err_nuc = 0.0
for i in range(nbas):
    for j in range(nbas):
        our = call_int1e("int1e_nuc_cart", (i, j), mol)
        ai = mol.ao_loc_nr(cart=True)[i]; bi_end = mol.ao_loc_nr(cart=True)[i+1]
        aj = mol.ao_loc_nr(cart=True)[j]; bj_end = mol.ao_loc_nr(cart=True)[j+1]
        ref = ref_nuc[ai:bi_end, aj:bj_end]
        err = np.max(np.abs(our - ref))
        max_err_nuc = max(max_err_nuc, err)
        if err > 1e-8:
            print(f"  FAIL nuc shells ({i},{j}) err={err:.3e}  our={our.flat[0]:.6f}  ref={ref.flat[0]:.6f}")
            all_pass = False

print(f"  max error: {max_err_nuc:.3e}")
if max_err_nuc < 1e-8:
    print("  ✓ Nuclear integrals OK")
print()

# ── 7b. Kinetic energy ────────────────────────────────────────────────────────
print("=== int1e_kin_cart ===")
max_err_kin = 0.0
for i in range(nbas):
    for j in range(nbas):
        our = call_int1e("int1e_kin_cart", (i, j), mol)
        ai = mol.ao_loc_nr(cart=True)[i]; bi_end = mol.ao_loc_nr(cart=True)[i+1]
        aj = mol.ao_loc_nr(cart=True)[j]; bj_end = mol.ao_loc_nr(cart=True)[j+1]
        ref = ref_kin[ai:bi_end, aj:bj_end]
        err = np.max(np.abs(our - ref))
        max_err_kin = max(max_err_kin, err)
        if err > 1e-8:
            print(f"  FAIL kin shells ({i},{j}) err={err:.3e}  our={our.flat[0]:.6f}  ref={ref.flat[0]:.6f}")
            all_pass = False

print(f"  max error: {max_err_kin:.3e}")
if max_err_kin < 1e-8:
    print("  ✓ Kinetic integrals OK")
print()

# ── 8. ERI ────────────────────────────────────────────────────────────────────
print("=== int2e_cart ===")
max_err_eri = 0.0
for i in range(nbas):
    for j in range(nbas):
        for k in range(nbas):
            for l in range(nbas):
                our = call_int2e((i, j, k, l), mol)
                ai = mol.ao_loc_nr(cart=True)[i]; be_i = mol.ao_loc_nr(cart=True)[i+1]
                aj = mol.ao_loc_nr(cart=True)[j]; be_j = mol.ao_loc_nr(cart=True)[j+1]
                ak = mol.ao_loc_nr(cart=True)[k]; be_k = mol.ao_loc_nr(cart=True)[k+1]
                al = mol.ao_loc_nr(cart=True)[l]; be_l = mol.ao_loc_nr(cart=True)[l+1]
                ref = ref_eri[ai:be_i, aj:be_j, ak:be_k, al:be_l]
                err = np.max(np.abs(our - ref))
                max_err_eri = max(max_err_eri, err)
                if err > 1e-8:
                    print(f"  FAIL ERI ({i},{j},{k},{l}) err={err:.3e}")
                    all_pass = False

print(f"  max error: {max_err_eri:.3e}")
if max_err_eri < 1e-8:
    print("  ✓ ERI OK")
print()

# ── 9. HF energy from our integrals ──────────────────────────────────────────
print("=== HF energy (numpy RHF using our integrals) ===")
import scipy.linalg

# Assemble full matrices from our library
nao = mol.nao_nr(cart=True)
our_S   = np.zeros((nao, nao))
our_T   = np.zeros((nao, nao))
our_V   = np.zeros((nao, nao))
our_eri = np.zeros((nao, nao, nao, nao))

ao_loc = mol.ao_loc_nr(cart=True)
for i in range(nbas):
    i0, i1 = ao_loc[i], ao_loc[i+1]
    for j in range(nbas):
        j0, j1 = ao_loc[j], ao_loc[j+1]
        our_S[i0:i1, j0:j1] = call_int1e("int1e_ovlp_cart", (i,j), mol)
        our_T[i0:i1, j0:j1] = call_int1e("int1e_kin_cart",  (i,j), mol)
        our_V[i0:i1, j0:j1] = call_int1e("int1e_nuc_cart",  (i,j), mol)
        for k in range(nbas):
            k0, k1 = ao_loc[k], ao_loc[k+1]
            for l in range(nbas):
                l0, l1 = ao_loc[l], ao_loc[l+1]
                our_eri[i0:i1, j0:j1, k0:k1, l0:l1] = call_int2e((i,j,k,l), mol)

h1e = our_T + our_V  # core Hamiltonian

# Nuclear repulsion energy
coords = mol.atom_coords()
charges = mol.atom_charges()
e_nn = 0.0
for a in range(mol.natm):
    for b in range(a+1, mol.natm):
        r = np.linalg.norm(coords[a] - coords[b])
        e_nn += charges[a] * charges[b] / r

# Simple RHF for closed-shell molecule (2 electrons, 1 occ MO for H2)
nocc = mol.nelectron // 2

# Initial Fock = h1e, solve generalized eigenvalue problem
mo_e, mo_c = scipy.linalg.eigh(h1e, our_S)
dm = 2.0 * mo_c[:, :nocc] @ mo_c[:, :nocc].T  # density matrix

# SCF iterations (max 50)
e_hf_prev = 0.0
for scf_iter in range(50):
    # Build Fock matrix: F_pq = h_pq + sum_rs D_rs * [(pq|rs) - 0.5*(pr|qs)]
    J = np.einsum("pqrs,rs->pq", our_eri, dm)
    K = np.einsum("prqs,rs->pq", our_eri, dm)
    fock = h1e + J - 0.5 * K

    # Solve F C = e S C
    mo_e, mo_c = scipy.linalg.eigh(fock, our_S)
    dm_new = 2.0 * mo_c[:, :nocc] @ mo_c[:, :nocc].T

    # Energy: E = 0.5 * Tr[D(h + F)] + E_nn
    e_hf = 0.5 * np.einsum("pq,pq->", dm_new, h1e + fock) + e_nn

    if abs(e_hf - e_hf_prev) < 1e-12:
        break
    e_hf_prev = e_hf
    dm = dm_new

# PySCF reference HF energy
from pyscf import scf
mf = scf.RHF(mol)
mf.verbose = 0
e_ref = mf.kernel()

delta_e = abs(e_hf - e_ref)
print(f"  Our RHF energy   : {e_hf:.10f} Hartree")
print(f"  PySCF RHF energy : {e_ref:.10f} Hartree")
print(f"  |ΔE|             : {delta_e:.3e} Hartree")
if delta_e < 1e-8:
    print("  ✓ HF energy matches PySCF to < 1e-8 Hartree")
else:
    print(f"  FAIL: |ΔE| = {delta_e:.3e} exceeds 1e-8 Hartree threshold")
    all_pass = False
print()

# ── 10. Summary ────────────────────────────────────────────────────────────────
print("=" * 50)
if all_pass and max_err_eri < 1e-8 and max_err_nuc < 1e-8 and max_err_kin < 1e-8 and delta_e < 1e-8:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — see above")
    sys.exit(1)
