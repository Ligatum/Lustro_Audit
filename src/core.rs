// ====================================================
// LUSTRO CORE V1 (SSOT) REFERENCE IMPLEMENTATION
// ====================================================

const PHI_64: u64 = 0x9E3779B97F4A7C15;

const PHI_64_ROT_23: u64 = PHI_64.rotate_left(23);
const PHI_64_ROT_17: u64 = PHI_64.rotate_left(17);

// ====================================================
// IDM (SCALAR) (COMPONENT OF LUSTRO CORE, SSOT)
// ====================================================

// IDM (INITIAL DIFFUSION MODULE)
#[inline]
pub(crate) fn idm_scalar_ref(s0: u128, s1: u128) -> (u128, u128) {

    let a = (s0 >> 64) as u64;
    let b = s0 as u64;
    let c = (s1 >> 64) as u64;
    let d = s1 as u64;

    let a = a.wrapping_add(b);
    let c = c.wrapping_add(d);
    let b = b ^ a.rotate_left(11);
    let d = d ^ c.rotate_left(19);

    let a = a.wrapping_add(PHI_64);
    let c = c.wrapping_add(PHI_64_ROT_23);
    let a = a ^ c.rotate_left(37);
    let d = d ^ b.rotate_left(43);

    let b = b.wrapping_add(a);
    let d = d.wrapping_add(c);
    let a = a ^ b.rotate_left(31);
    let c = c ^ d.rotate_left(47);

    let m = (a ^ c).wrapping_mul(PHI_64);
    let b = b ^ m;
    let d = d ^ m.rotate_left(33);

    let mix0 = a ^ c;
    let m0 = (mix0 ^ mix0.rotate_left(32)).wrapping_mul(PHI_64);
    let m1 = (b ^ d).wrapping_mul(PHI_64_ROT_17);

    let a = a ^ m0;
    let c = c ^ m0.rotate_left(23);
    let b = b ^ m1;
    let d = d ^ m1.rotate_left(41);

    let a = a.wrapping_add(b);
    let d = d ^ a.rotate_left(33);
    let b = b ^ (c ^ d).rotate_left(21);

    let a = a.wrapping_add(d);
    let c = c.wrapping_add(b);
    let d = d ^ a.rotate_left(41);
    let b = b ^ c.rotate_left(23);

    let b = b.wrapping_add(a);
    let d = d.wrapping_add(c);
    let a = a ^ b.rotate_left(17);
    let c = c ^ d.rotate_left(31);

    (
        ((a as u128) << 64) | (b as u128),
        ((c as u128) << 64) | (d as u128),
    )
}

// ====================================================
// ERD ROUND (SCALAR) (COMPONENT OF LUSTRO CORE, SSOT)
// ====================================================

// ERD (EVOLVING REPRESENTATION DYNAMICS)
// OUTPUT: STATE-DERIVED ROTATION AMOUNTS
#[inline]
fn erd(
    s0: u128,
    s1: u128
) -> (u32, u32) {

    let mut v0_base = s0.wrapping_add(s1);
    let mut v1_base = s0 ^ s1.rotate_left(42);

    let t0 = v0_base;
    let t1 = v1_base;
    v0_base ^= (t0 >> 64) ^ t0.rotate_left(37);
    v1_base ^= (t1 >> 64) ^ t1.rotate_left(43);

    let mut v0 = v0_base as u64;
    let mut v1 = v1_base as u64;

    v0 = v0.wrapping_add(v1.rotate_left(31));
    v1 ^= v0.rotate_left(27);

    v0 = v0.wrapping_add(v1.rotate_left(17));
    v1 ^= v0.rotate_left(29);

    let r0 = v1.rotate_left(11);
    let r1 = v0.rotate_left(13);

    let mut x0 = v0 ^ r0;
    x0 ^= x0 >> 32;
    x0 = x0.wrapping_mul(x0 | 1);

    let mut x1 = v1 ^ r1;
    x1 ^= x1 >> 32;
    x1 = x1.wrapping_mul(x1 | 1);

    let k0 = ((x0 >> 58) as u32) | 1;
    let k1 = ((x1 >> 58) as u32) | 1;

    v0 = v0.wrapping_add(v1.rotate_left(k0) ^ (v1 >> 7));
    v1 ^= (v0 >> 7) ^ v0.rotate_left(k1);

    v0 = v0.wrapping_add(v1.rotate_left(17));
    v1 ^= v0.rotate_left(40);

    v0 ^= v0 >> 29;

    // PROJECTION
    let rot0 = (((v0 as u128).wrapping_mul(127) >> 64) as u32) + 1;
    let rot1 = (((v1 as u128).wrapping_mul(113) >> 64) as u32) + 1;

    (rot0, rot1)
}

// Y+ TEMPORAL PERTURBATION + ERD PERMUTATION (COUNTER-BASED)
#[inline]
pub(crate) fn erd_round_scalar_ref(
    mut s0: u128,
    mut s1: u128,
    step: u64
) -> (u128, u128) {

    // WEYL RC
    let current_rc = PHI_64.wrapping_mul(step.wrapping_add(2));

    // 128-BIT RC MASK
    let rc128 =
        ((current_rc.rotate_left(17) as u128) << 64)
        | (current_rc as u128);

    // ASYMMETRIC PERTURBATION
    s0 ^= rc128;
    s1 ^= rc128.rotate_left(17);

    // ERD ROTATION GENERATION
    let (rot0, rot1) = erd(s0, s1);

    // ERD-DRIVEN STATE PERMUTATION
    s0 = s0.rotate_left(rot0);
    s1 = s1.rotate_left(rot1);

    (s0, s1)
}