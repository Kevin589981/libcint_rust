//! Cartesian → spherical-harmonic transformation module.
//!
//! For s (l=0) and p (l=1) shells the Cartesian and spherical bases have
//! the same dimension (1 and 3 respectively), so the transforms are identity
//! or simple sign-only reorderings.  For the POC we only need l ≤ 1.

pub mod cart2sph;

pub use cart2sph::{cart2sph_inplace, cart2sph_block};
