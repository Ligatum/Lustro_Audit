"""
================================================================
LUSTRO V1 — PUBLIC BINARY VALIDATOR
================================================================
Verifies that your local Lustro build produces bit-exact output
against the reference implementation.
================================================================
"""

import hashlib
import sys
import numpy as np

# ================================================================
# ENGINE IMPORT
# ================================================================

try:
    from lustro_rust import LustroCoreV1Py
except ImportError:
    sys.exit(
        "ERROR: lustro_rust not found.\n"
        "Install the Lustro scalar engine and try again."
    )

# ================================================================
# GOLDEN VECTORS
# 31 input/output pairs generated from the reference build.
# Each entry: (s0_in, s1_in, s0_out, s1_out)
# ================================================================

GOLDEN_VECTORS = [
    # (s0_in, s1_in, s0_out, s1_out)
    # [00] zero / zero
    (
        0x00000000000000000000000000000000,  # s0_in
        0x00000000000000000000000000000000,  # s1_in
        0x0257939FEDFEE524EA28CFF110AA301D,  # s0_out
        0xF3EA3A8E0C0094B7A4BCF38E6A737A93,  # s1_out
    ),
    # [01] all-ones / all-ones
    (
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,  # s0_in
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,  # s1_in
        0x204C76949F3E38973CD49123B518CABF,  # s0_out
        0x585D5252868A26BD3839C3ECE9029EEB,  # s1_out
    ),
    # [02] all-ones / zero
    (
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,  # s0_in
        0x00000000000000000000000000000000,  # s1_in
        0xB31E2EB85EEBD242F241AB6A4915DFEC,  # s0_out
        0xE097B4EB9A42EBD715F03D0DB035C5D0,  # s1_out
    ),
    # [03] zero / all-ones
    (
        0x00000000000000000000000000000000,  # s0_in
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,  # s1_in
        0xEC5F6A6957106E41459747178C1EDF1F,  # s0_out
        0xAD015E54A77805231819ADEB2D2E2AFC,  # s1_out
    ),
    # [04] 1 / 0
    (
        0x00000000000000000000000000000001,  # s0_in
        0x00000000000000000000000000000000,  # s1_in
        0xC5BF3266AC9A30B755713D4B51D3E1DD,  # s0_out
        0xD93999BDE2A9F6429B56DCE356370ACB,  # s1_out
    ),
    # [05] 0 / 1
    (
        0x00000000000000000000000000000000,  # s0_in
        0x00000000000000000000000000000001,  # s1_in
        0x60804FA84D973E54E27720D4105F88CC,  # s0_out
        0x435394A48770A15BAE811349AE61F8DC,  # s1_out
    ),
    # [06] MSB s0 / zero
    (
        0x80000000000000000000000000000000,  # s0_in
        0x00000000000000000000000000000000,  # s1_in
        0x96731F1745AB19C16092173DBA1C8E21,  # s0_out
        0x811447A93C7B0D4E01FD09290D8ECE95,  # s1_out
    ),
    # [07] zero / MSB s1
    (
        0x00000000000000000000000000000000,  # s0_in
        0x80000000000000000000000000000000,  # s1_in
        0x40F48D92AA38289E94FE224D2E980684,  # s0_out
        0xCA81B548B40385DEED9A499383374257,  # s1_out
    ),
    # [08] ALT_A / ALT_B
    (
        0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,  # s0_in
        0x55555555555555555555555555555555,  # s1_in
        0xBE10779D566A587B14316E6214720B2C,  # s0_out
        0x0444BDD9F60BBA6B7ECF7012CACECF51,  # s1_out
    ),
    # [09] ALT_B / ALT_A
    (
        0x55555555555555555555555555555555,  # s0_in
        0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,  # s1_in
        0xCBBBAC6BC679E0F6BBB1300F4F76F2FC,  # s0_out
        0xC9834B52B4731C0F6F199F1656EA44A5,  # s1_out
    ),
    # [10] ALT_A / ALT_A
    (
        0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,  # s0_in
        0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,  # s1_in
        0x369568912C4E9E71CA9023E53D3C2F12,  # s0_out
        0x14AB577579D8C660BAD24CFAFF7FC06D,  # s1_out
    ),
    # [11] ALT_B / ALT_B
    (
        0x55555555555555555555555555555555,  # s0_in
        0x55555555555555555555555555555555,  # s1_in
        0x94F473694BE12F0ED7A9108FFCD6D2ED,  # s0_out
        0x5DB9B2AAB29DC8758241F026C41A05CB,  # s1_out
    ),
    # [12] single-bit walk 1
    (
        0x00000000000000000000000000000001,  # s0_in
        0x00000000000000010000000000000000,  # s1_in
        0x4ADC7C921E9D5B1C4EFD2E5109673368,  # s0_out
        0xD7D4FB31D5D2ECAAAAD687CD2AA082D8,  # s1_out
    ),
    # [13] single-bit walk 2
    (
        0x00000000000000000000000100000000,  # s0_in
        0x00000001000000000000000000000000,  # s1_in
        0xBE66B02B105C7436F5E5BD44F294E2C0,  # s0_out
        0x03F4FD4DBD1C0FDFDDCBFBEAE2C6F5BE,  # s1_out
    ),
    # [14] single-bit walk 3
    (
        0x00000000000000008000000000000000,  # s0_in
        0x00000000000000008000000000000000,  # s1_in
        0x8BFC9EA2F57F5E8ED4D4D6D484E17765,  # s0_out
        0xA7C51157DD0C2643E78F531C08EBD1E8,  # s1_out
    ),
    # [15] single-bit walk 4
    (
        0x00000000000000010000000000000000,  # s0_in
        0x00000000000000000000000000000001,  # s1_in
        0xE465F34D07D3D8ED1D12005ED048738B,  # s0_out
        0xF908C62FA02EE06026FE386AB92B55A5,  # s1_out
    ),
    # [16] PHI / PHI^MASK
    (
        0x9E3779B97F4A7C159E3779B97F4A7C15,  # s0_in
        0x61C8864680B583EA61C8864680B583EA,  # s1_in
        0xD18F3B9881E65681F2B8A1EFF03A3A9C,  # s0_out
        0x07C432DD96C8372DCAAD62F3636C746E,  # s1_out
    ),
    # [17] PHI / zero
    (
        0x9E3779B97F4A7C159E3779B97F4A7C15,  # s0_in
        0x00000000000000000000000000000000,  # s1_in
        0x89D39DABC9A169D70EE377807C52AA99,  # s0_out
        0x635108492B10FE13623D7080028F7E1F,  # s1_out
    ),
    # [18] zero / PHI
    (
        0x00000000000000000000000000000000,  # s0_in
        0x9E3779B97F4A7C159E3779B97F4A7C15,  # s1_in
        0x774DD51B5C207598A18E3021C21AECC1,  # s0_out
        0x90D6754444A27568FFD5C24BFC4BF86E,  # s1_out
    ),
    # [19] fixed pseudo-random 1
    (
        0x0123456789ABCDEF0123456789ABCDEF,  # s0_in
        0xFEDCBA9876543210FEDCBA9876543210,  # s1_in
        0x56D48CD7279CE5A2417BC02D24943F7D,  # s0_out
        0xF244A16EB2040D581A584AE9FD49AF03,  # s1_out
    ),
    # [20] fixed pseudo-random 2
    (
        0xDEADBEEFCAFEBABEDEADBEEFCAFEBABE,  # s0_in
        0x0102030405060708090A0B0C0D0E0F10,  # s1_in
        0xF417829CC606FE8877ED76F255C57164,  # s0_out
        0xCF53BF031D43982C446060C96BC2FC45,  # s1_out
    ),
    # [21] fixed pseudo-random 3
    (
        0xFFFFFFFF00000000FFFFFFFF00000000,  # s0_in
        0x00000000FFFFFFFF00000000FFFFFFFF,  # s1_in
        0xA18307A830A7A36E2B319E530192E226,  # s0_out
        0xB6EAD6F58F739A9FC0DFA896FFE16C20,  # s1_out
    ),
    # [22] fixed pseudo-random 4
    (
        0x123456789ABCDEF0123456789ABCDEF0,  # s0_in
        0xF0EDCBA987654321F0EDCBA987654321,  # s1_in
        0x2F20BCF6389636C80AAF464E2F3C3CC7,  # s0_out
        0x415AF513A11B23D654307994E5ED6424,  # s1_out
    ),
    # [23] fixed pseudo-random 5
    (
        0xA5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5,  # s0_in
        0x5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A,  # s1_in
        0x781E6A8BC798F33432E763261BDBDDC7,  # s0_out
        0x8C8CCA47630F674B829DC8A2D6655C8C,  # s1_out
    ),
    # [24] fixed pseudo-random 6
    (
        0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F,  # s0_in
        0xF0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0,  # s1_in
        0xAB32DB389C3EE5DF68E616BF3C434965,  # s0_out
        0x3F3B47F0B47CC733BC216DB8C44BA6E6,  # s1_out
    ),
    # [25] fixed pseudo-random 7
    (
        0x13579BDF02468ACE13579BDF02468ACE,  # s0_in
        0xECA8642031975BDFECA8642031975BDF,  # s1_in
        0x2E4DF04828CBBEDEC0631A178333A0C3,  # s0_out
        0xB96BEF302847C657EDE2D8A0A258F94B,  # s1_out
    ),
    # [26] fixed pseudo-random 8
    (
        0x80000000000000018000000000000001,  # s0_in
        0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,  # s1_in
        0xAC4A7285CC4EA06FFFCFBD913CF01E17,  # s0_out
        0x650E9D7E640D353BCB28445A31C49F25,  # s1_out
    ),
    # [27] fixed pseudo-random 9
    (
        0xC0FFEE00C0FFEE00C0FFEE00C0FFEE00,  # s0_in
        0x1234ABCD5678EF001234ABCD5678EF00,  # s1_in
        0x0625DF58DCEB4C6F1AA39DC3BED0DECF,  # s0_out
        0x2827B54AB666992A51A24FCFB4C24BF6,  # s1_out
    ),
    # [28] fixed pseudo-random 10
    (
        0xFFFF0000FFFF0000FFFF0000FFFF0000,  # s0_in
        0x0000FFFF0000FFFF0000FFFF0000FFFF,  # s1_in
        0xF8D52DF6A070FC106A17F27CBAC3C9E3,  # s0_out
        0xBD324782C9C242BBF0531A741C4BCD06,  # s1_out
    ),
    # [29] fixed pseudo-random 11
    (
        0x9E3779B97F4A7C159E3779B97F4A7C15,  # s0_in
        0x06C62272E07BB01426890DC4DF6FDAE9,  # s1_in
        0x002885810D0FB8B9F50D871AFD716125,  # s0_out
        0x1AB1CA7150A9C47669127AC1FADFF334,  # s1_out
    ),
    # [30] fixed pseudo-random 12
    (
        0x517CC1B727220A95517CC1B727220A95,  # s0_in
        0xA88388C0A3352A95A88388C0A3352A95,  # s1_in
        0x6CC5E41201EF7868657C2426A9008321,  # s0_out
        0x7B0F13BBB90F5511EAF50AC41542C363,  # s1_out
    ),
]

GOLDEN_FINGERPRINT = "ab5b1cb816f8479d5a2f7bf5b4ab814a7afb01bdf0095fcdde6234722f2e387c"

# ================================================================
# VALIDATOR
# ================================================================

MASK_64 = 0xFFFFFFFFFFFFFFFF


def run_vector(core, s0_in: int, s1_in: int):
    """Run evaluate_inplace on a single (s0, s1) state. LE ABI."""
    buf = np.zeros(4, dtype=np.uint64)
    buf[0] =  s0_in        & MASK_64
    buf[1] = (s0_in >> 64) & MASK_64
    buf[2] =  s1_in        & MASK_64
    buf[3] = (s1_in >> 64) & MASK_64

    core.evaluate_inplace(buf)

    out_s0 = (int(buf[1]) << 64) | int(buf[0])
    out_s1 = (int(buf[3]) << 64) | int(buf[2])
    return out_s0, out_s1


def main():
    width = 72

    print()
    print("=" * width)
    print(f"{'LUSTRO V1 — BINARY VALIDATOR':^{width}}")
    print("=" * width)

    core = LustroCoreV1Py()

    passed  = 0
    failed  = 0
    sha     = hashlib.sha256()

    print(f"\n  Running {len(GOLDEN_VECTORS)} golden vector checks...\n")

    for idx, (s0_in, s1_in, s0_exp, s1_exp) in enumerate(GOLDEN_VECTORS):

        s0_got, s1_got = run_vector(core, s0_in, s1_in)

        sha.update(s0_got.to_bytes(16, "little"))
        sha.update(s1_got.to_bytes(16, "little"))

        if s0_got == s0_exp and s1_got == s1_exp:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL  vector [{idx:02d}]")
            print(f"        s0_in  = 0x{s0_in:032X}")
            print(f"        s1_in  = 0x{s1_in:032X}")
            print(f"        s0_exp = 0x{s0_exp:032X}")
            print(f"        s0_got = 0x{s0_got:032X}")
            print(f"        s1_exp = 0x{s1_exp:032X}")
            print(f"        s1_got = 0x{s1_got:032X}")
            print()

    # ================================================================
    # FINGERPRINT CHECK
    # ================================================================

    fingerprint_got = sha.hexdigest()
    fingerprint_ok  = (fingerprint_got == GOLDEN_FINGERPRINT)

    print(f"  Vectors  : {passed}/{len(GOLDEN_VECTORS)} passed")
    print()
    print(f"  Expected fingerprint : {GOLDEN_FINGERPRINT}")
    print(f"  Computed fingerprint : {fingerprint_got}")
    print(f"  Fingerprint match    : {'YES' if fingerprint_ok else 'NO'}")

    # ================================================================
    # VERDICT
    # ================================================================

    print()
    print("=" * width)

    all_ok = (failed == 0) and fingerprint_ok

    if all_ok:
        print(f"{'PASS — build is 100% identical to the reference':^{width}}")
    else:
        print(f"{'FAIL — this build does not match the reference':^{width}}")
        if failed > 0:
            print(f"{'Vector mismatches: ' + str(failed):^{width}}")
        if not fingerprint_ok:
            print(f"{'Fingerprint mismatch':^{width}}")

    print("=" * width)
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()