# libcint-rs

**A Rust reimplementation of the [libcint](https://github.com/sunqm/libcint) quantum chemistry integral engine.**

Provides a 100 % C-ABI-compatible shared library (`libcint.so`) that PySCF can load via `ctypes` as a drop-in replacement for the original C library — without any changes to PySCF source code.

> **Status: Phase 2 — CINTOpt Cauchy-Schwarz screening + rayon parallel ERI fill complete. All 8 unit tests pass; H₂/STO-3G |ΔE_HF| = 1.8 × 10⁻¹⁴ Hartree.**

---

## Table of Contents / 目录

- [English](#english)
- [中文](#中文)

---

## English

### Project Goal

Replace `libcint` with a safe, modern Rust implementation that:

1. Exports identical C-ABI symbols (`int1e_ovlp_sph`, `int2e_sph`, …)
2. Is loadable by PySCF via `ctypes` with zero Python-side changes
3. Eventually outperforms `libcint + OpenMP` on multi-core hardware via `rayon`

### Quick Start

```bash
# Build the shared library
cargo build --release

# The drop-in .so is at:
ls target/release/libcint.so

# Run unit tests
cargo test
```

### Exported C-ABI Symbols

All functions match the libcint signature exactly:

```c
CACHE_SIZE_T int_xxx(double *out, int *dims, int *shls,
    int *atm, int natm, int *bas, int nbas, double *env,
    CINTOpt *opt, double *cache);
```

| Symbol | Status |
|---|---|
| `int1e_ovlp_cart` | ✅ Implemented |
| `int1e_ovlp_sph`  | ✅ Implemented (identity for l ≤ 1) |
| `int1e_kin_cart`  | ✅ Implemented |
| `int1e_kin_sph`   | ✅ Implemented (identity for l ≤ 1) |
| `int1e_nuc_cart`  | ✅ Implemented |
| `int1e_nuc_sph`   | ✅ Implemented (identity for l ≤ 1) |
| `int2e_cart`      | ✅ Implemented |
| `int2e_sph`       | ✅ Implemented (identity for l ≤ 1) |
| `int1e_*_optimizer`    | ✅ Stub (returns NULL, no screening needed) |
| `int2e_optimizer`      | ✅ Builds Cauchy-Schwarz screening table (`CINTOpt`) |
| `CINTdel_optimizer`    | ✅ Frees `CINTOpt` heap allocation |
| `int2e_fill_cart`      | ✅ Parallel batch fill (rayon, all CPU cores) |
| `int1e_kin_cart` + gradient variants | ❌ Not yet |
| `int2e_ip1_sph` (gradient ERI) | ❌ Not yet |
| 3-center / 4-center integrals | ❌ Not yet |
| `int1e_grids` (DFT grid integrals) | ❌ Not yet |

### Implemented Modules

| Module | File | Description | Status |
|---|---|---|---|
| Data structures | `src/types.rs` | `AtmSlot`, `BasSlot`, `Env`, `EnvVars` | ✅ |
| Boys function | `src/rys/mod.rs` | $F_m(t)$ via Taylor / asymptotic expansion + Gauss quadrature | ✅ |
| Rys roots/weights | `src/rys/mod.rs` | Wheeler algorithm, implicit-shift QL eigensolver, orders 1–6 | ✅ |
| Primitive 2e kernel | `src/recur/g0_2e.rs` | Rys2eT intermediates, g-buffer init | ✅ |
| OS recursion (2D+4D) | `src/recur/g0_2d4d.rs` | Obara-Saika up-recursion for s/p/d shells | ✅ |
| Cart→Sph transform | `src/transform/cart2sph.rs` | l = 0,1 (identity); l = 2 (6→5 matrix) | ✅ l≤2 / ❌ l≥3 |
| Overlap integral | `src/int1e/overlap.rs` | `<i\|j>` any contracted Cartesian shells | ✅ |
| Kinetic integral | `src/int1e/kinetic.rs` | `<i\|−½∇²\|j>` via OS formula | ✅ |
| Nuclear attraction | `src/int1e/nuclear.rs` | `<i\|Σ Z_A/r_A\|j>` via Boys function | ✅ |
| ERI (bare) | `src/int2e/eri.rs` | `(ij\|kl)` Cartesian 4-center 2e, no screening | ✅ |
| ERI driver | `src/int2e/driver.rs` | Single-quartet CS screening + rayon batch fill | ✅ |
| CS pre-screener | `src/optimizer.rs` | `CINTOpt`: sqrt-Schwarz table, `passes(i,j,k,l)` | ✅ |
| C-ABI exports | `src/lib.rs` | 19 exported symbols | ✅ |

### Not Yet Implemented

| Component | Corresponding libcint file | Phase |
|---|---|---|
| Cart→Sph for l ≥ 3 (f, g, h shells) | `cart2sph.c` | Phase 3 |
| Kinetic + nuclear gradient integrals | `autocode/grad1.c` | Phase 3 |
| ERI gradient `int2e_ip1` | `autocode/grad2.c` | Phase 3 |
| 3-center 2e integrals (DF/RI) | `cint3c2e.c` | Phase 3 |
| 2-center 2e integrals | `cint2c2e.c` | Phase 3 |
| Grid integrals (DFT) | `cint1e_grids.c` | Phase 3 |
| Breit / Gaunt interaction | `breit.c` | Phase 3 |
| F12 / STG integrals | `cint2e_f12.c` | Phase 3 |
| Hessian integrals | `autocode/hess.c` | Phase 3 |
| SIMD vectorisation | `gout2e_simd.c` | Phase 2 |
| `build.rs` codegen for l = 0…4 | `scripts/gen-code.cl` | Phase 2 |
| PySCF end-to-end validation | `python/validate_pyscf.py` | ✅ Done (Phase 1) |

### Architecture

```
src/
├── lib.rs            # C-ABI exports (19 symbols)
├── types.rs          # AtmSlot / BasSlot / Env / EnvVars
├── optimizer.rs      # CINTOpt: Cauchy-Schwarz screening table
├── rys/
│   └── mod.rs        # Boys function + Rys quadrature
├── recur/
│   ├── g0_2e.rs      # Primitive 2e kernel (Rys2eT)
│   └── g0_2d4d.rs    # 2D/4D Obara-Saika recursion
├── transform/
│   └── cart2sph.rs   # Cartesian → spherical (l ≤ 2)
├── int1e/
│   ├── overlap.rs    # <i|j>
│   ├── kinetic.rs    # <i|−½∇²|j>
│   └── nuclear.rs    # <i|V_nuc|j>
└── int2e/
    ├── eri.rs        # (ij|kl) bare kernel
    └── driver.rs     # CS-screened single call + rayon batch fill
```

### Validation Results

| Integral | Max error vs PySCF |
|---|---|
| `int1e_ovlp` | 1.1 × 10⁻¹⁶ |
| `int1e_kin` | 8.3 × 10⁻¹⁷ |
| `int1e_nuc` | 1.6 × 10⁻¹⁴ |
| `int2e` | 5.6 × 10⁻¹⁵ |
| HF energy (H₂/STO-3G) | \|ΔE\| = 1.8 × 10⁻¹⁴ Hartree |

### Precision Goals

| Level | Target |
|---|---|
| All integrals vs libcint | relative error < 1 × 10⁻¹² |
| SCF energy (HF/DFT) | |ΔE| < 1 × 10⁻⁸ Hartree |

### License

Apache-2.0 — same as PySCF.

---

## 中文

### 项目目标

用安全、现代的 Rust 重新实现 `libcint`，使其：

1. 导出与原版完全相同的 C-ABI 符号（`int1e_ovlp_sph`、`int2e_sph` 等）
2. 可由 PySCF 通过 `ctypes` 直接加载，**无需修改任何 PySCF 代码**
3. 通过 `rayon` 在多核上最终超越 `libcint + OpenMP` 的整体吞吐量

### 快速开始

```bash
# 构建动态库
cargo build --release

# 产物路径
ls target/release/libcint.so

# 运行单元测试
cargo test
```

### 已导出的 C-ABI 符号

所有函数签名与 libcint 完全一致：

```c
CACHE_SIZE_T int_xxx(double *out, int *dims, int *shls,
    int *atm, int natm, int *bas, int nbas, double *env,
    CINTOpt *opt, double *cache);
```

| 符号 | 状态 |
|---|---|
| `int1e_ovlp_cart` | ✅ 已实现 |
| `int1e_ovlp_sph`  | ✅ 已实现（l ≤ 1 为恒等变换） |
| `int1e_kin_cart`  | ✅ 已实现 |
| `int1e_kin_sph`   | ✅ 已实现（l ≤ 1 为恒等变换） |
| `int1e_nuc_cart`  | ✅ 已实现 |
| `int1e_nuc_sph`   | ✅ 已实现（l ≤ 1 为恒等变换） |
| `int2e_cart`      | ✅ 已实现 |
| `int2e_sph`       | ✅ 已实现（l ≤ 1 为恒等变换） |
| `int1e_*_optimizer` | ✅ Stub（返回 NULL，1e 积分无需筛选） |
| `int2e_optimizer`   | ✅ 构建 Cauchy-Schwarz 筛选表（`CINTOpt`） |
| `CINTdel_optimizer` | ✅ 释放 `CINTOpt` 堆内存 |
| `int2e_fill_cart`   | ✅ rayon 并行批量填充完整 ERI 张量 |
| 梯度积分变体 `_ip1` 等 | ❌ 尚未实现 |
| 3 中心 / 4 中心积分 | ❌ 尚未实现 |
| 格点积分 `int1e_grids` | ❌ 尚未实现 |

### 已实现模块

| 模块 | 文件 | 功能描述 | 状态 |
|---|---|---|---|
| 数据结构 | `src/types.rs` | `AtmSlot`、`BasSlot`、`Env`、`EnvVars` | ✅ |
| Boys 函数 | `src/rys/mod.rs` | $F_m(t)$，Taylor/渐近展开 + 高斯求积 | ✅ |
| Rys 根与权重 | `src/rys/mod.rs` | Wheeler 算法 + 隐位移 QL 特征求解，1~6 阶 | ✅ |
| 2e 原始积分核 | `src/recur/g0_2e.rs` | Rys2eT 中间量、g 缓冲初始化 | ✅ |
| OS 递推（2D+4D） | `src/recur/g0_2d4d.rs` | s/p/d 壳的 Obara-Saika 向上递推 | ✅ |
| 笛卡尔→球谐变换 | `src/transform/cart2sph.rs` | l=0,1（恒等）；l=2（6→5 矩阵） | ✅ l≤2 / ❌ l≥3 |
| 重叠积分 | `src/int1e/overlap.rs` | `<i\|j>`，任意收缩笛卡尔壳 | ✅ |
| 动能积分 | `src/int1e/kinetic.rs` | `<i\|−½∇²\|j>`，OS 公式 | ✅ |
| 核势积分 | `src/int1e/nuclear.rs` | `<i\|Σ Z_A/r_A\|j>`，Boys 函数 | ✅ |
| ERI 裸核 | `src/int2e/eri.rs` | `(ij\|kl)` 笛卡尔 4 中心 2 电子，无筛选 | ✅ |
| ERI 驱动层 | `src/int2e/driver.rs` | 单四元组 CS 筛选 + rayon 并行批量填充 | ✅ |
| CS 预筛选器 | `src/optimizer.rs` | `CINTOpt`：sqrt-Schwarz 表、`passes(i,j,k,l)` | ✅ |
| C-ABI 导出 | `src/lib.rs` | 19 个导出符号 | ✅ |

### 尚未实现

| 组件 | 对应 libcint 文件 | 阶段 |
|---|---|---|
| l ≥ 3 球谐变换（f/g/h 轨道） | `cart2sph.c` | 第三阶段 |
| 动能/核势梯度积分 | `autocode/grad1.c` | 第三阶段 |
| ERI 梯度 `int2e_ip1` | `autocode/grad2.c` | 第三阶段 |
| 3 中心 2 电子积分（DF/RI 加速） | `cint3c2e.c` | 第三阶段 |
| 2 中心 2 电子积分 | `cint2c2e.c` | 第三阶段 |
| 格点积分（DFT 用） | `cint1e_grids.c` | 第三阶段 |
| Breit/Gaunt 相互作用 | `breit.c` | 第三阶段 |
| F12/STG 显式相关积分 | `cint2e_f12.c` | 第三阶段 |
| Hessian 积分 | `autocode/hess.c` | 第三阶段 |
| SIMD 向量化 | `gout2e_simd.c` | 第二阶段 |
| `build.rs` 代码生成（l = 0…4 展开） | `scripts/gen-code.cl` | 第二阶段 |
| PySCF 端到端验证（H₂O/STO-3G HF） | `python/validate_pyscf.py` | ✅ 已完成（第一阶段） |

### 代码结构

```
src/
├── lib.rs            # C-ABI 导出（19 个符号）
├── types.rs          # AtmSlot / BasSlot / Env / EnvVars
├── optimizer.rs      # CINTOpt：Cauchy-Schwarz 筛选表
├── rys/
│   └── mod.rs        # Boys 函数 + Rys 求积
├── recur/
│   ├── g0_2e.rs      # 原始 2e 核（Rys2eT）
│   └── g0_2d4d.rs    # 2D/4D Obara-Saika 递推
├── transform/
│   └── cart2sph.rs   # 笛卡尔 → 球谐（l ≤ 2）
├── int1e/
│   ├── overlap.rs    # <i|j>
│   ├── kinetic.rs    # <i|−½∇²|j>
│   └── nuclear.rs    # <i|V_nuc|j>
└── int2e/
    ├── eri.rs        # (ij|kl) 裸核
    └── driver.rs     # CS 筛选调用 + rayon 并行批量填充
```

### 精度目标

| 层级 | 目标 |
|---|---|
| 所有积分对比 libcint | 相对误差 < 1 × 10⁻¹² |
| HF/DFT SCF 收敛能量 | \|ΔE\| < 1 × 10⁻⁸ Hartree |

### 开发计划

```
第一阶段（POC）✅ 已完成
  ✅ 脚手架 & 数据结构
  ✅ Boys 函数 & Rys 求积
  ✅ Obara-Saika 递推
  ✅ 单电子积分（重叠、动能、核势）
  ✅ 双电子 ERI
  ✅ C-ABI 导出（16 符号）
  ✅ PySCF 端到端验证（H₂/STO-3G，|ΔE_HF| = 1.8×10⁻¹⁴ Hartree）

第二阶段（性能优化）⏳ 进行中
  ✅ CINTOpt Cauchy-Schwarz 预筛选（optimizer.rs）
  ✅ rayon 并行 ERI 批量填充（int2e/driver.rs）
  ✅ C-ABI 符号扩展至 19 个（CINTdel_optimizer、int2e_fill_cart）
  ☐ SIMD 向量化
  ☐ build.rs 代码生成（l=0…4 静态展开）

第三阶段（功能扩展）
  ☐ l ≥ 3 球谐变换
  ☐ 梯度积分（∇_A）
  ☐ 3c/2c 积分（DF/RI）
  ☐ 高角动量支持（f/g/h）
```

### 许可证

Apache-2.0 — 与 PySCF 保持一致。
