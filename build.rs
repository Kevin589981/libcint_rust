/// build.rs — Phase 2 Step 10: generate static Cartesian component tables.
///
/// Emits `$OUT_DIR/cart_tables.rs` which contains:
///   - `CART_NX_L`, `CART_NY_L`, `CART_NZ_L` for L = 0..=4 (static [i32; N])
///   - `cart_comp_l(l) -> (&'static [i32], &'static [i32], &'static [i32])`
///
/// This replaces the runtime `cart_comp` Vec-filling call in every primitive
/// integral loop, eliminating heap allocation in the hot path.

use std::io::Write;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = std::path::Path::new(&out_dir).join("cart_tables.rs");
    let mut f = std::io::BufWriter::new(std::fs::File::create(dest).unwrap());

    // For each angular momentum l generate tuples (nx, ny, nz) in libcint
    // order: ix descending, then iy descending within each ix block.
    for l in 0usize..=4 {
        let mut nx: Vec<i32> = Vec::new();
        let mut ny: Vec<i32> = Vec::new();
        let mut nz: Vec<i32> = Vec::new();
        for ix in (0..=l as i32).rev() {
            for iy in (0..=(l as i32 - ix)).rev() {
                nx.push(ix);
                ny.push(iy);
                nz.push(l as i32 - ix - iy);
            }
        }
        let n = nx.len();
        let fmt = |v: &[i32]| -> String {
            v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
        };
        writeln!(f, "const CART_NX_{l}: [i32; {n}] = [{}];", fmt(&nx)).unwrap();
        writeln!(f, "const CART_NY_{l}: [i32; {n}] = [{}];", fmt(&ny)).unwrap();
        writeln!(f, "const CART_NZ_{l}: [i32; {n}] = [{}];\n", fmt(&nz)).unwrap();
    }

    // Dispatch function — zero-allocation replacement for cart_comp().
    writeln!(f, "/// Return static (nx, ny, nz) Cartesian component tables for l = 0..=4.").unwrap();
    writeln!(f, "/// Zero-allocation hot-path replacement for `cart_comp`.").unwrap();
    writeln!(f, "#[inline]").unwrap();
    writeln!(f, "pub fn cart_comp_l(l: usize) -> (&'static [i32], &'static [i32], &'static [i32]) {{").unwrap();
    writeln!(f, "    match l {{").unwrap();
    for l in 0usize..=4 {
        writeln!(f, "        {l} => (&CART_NX_{l}, &CART_NY_{l}, &CART_NZ_{l}),").unwrap();
    }
    writeln!(f, "        _ => panic!(\"cart_comp_l: l={{l}} not supported (only l ≤ 4)\"),").unwrap();
    writeln!(f, "    }}").unwrap();
    writeln!(f, "}}").unwrap();
}

