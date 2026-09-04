import sys
import time
import multiprocessing
import numpy as np
import re

try:
    from lustro_rust import LustroCoreV1Py
except ImportError:
    sys.stderr.write("Error: Could not find lustro_rust. Please compile the engine (cargo build --release).\n")
    sys.exit(1)

# ==============================================================================
# HELPERS
# ==============================================================================

def generate_states_u64(n, rng):
    """Random states as ndarray(n, 4) uint64.
    Row layout: [s0_lo, s0_hi, s1_lo, s1_hi] — matches dispatch::process_scalar."""
    return rng.integers(0, 2**64, size=(n, 4), dtype=np.uint64)

def _b53_random_masks(n_masks, rng):
    """Random full-weight masks — global screening.
    shape: (n_masks, 8) uint64"""
    return rng.integers(0, 2 ** 64, size=(n_masks, 8), dtype=np.uint64)

def _b53_single_bit_masks(n_masks, rng):
    """Structured masks: 1 active input bit, 1 active output bit.
    Sampled WITHOUT replacement from the full 256x256 (in_bit, out_bit)
    space — guarantees no repeated pair within a single call, unlike
    independent per-column sampling with replacement.
    shape: (n_masks, 8) uint64"""
    space = 256 * 256
    if n_masks > space:
        raise ValueError(f"n_masks ({n_masks}) exceeds unique pair space ({space})")

    flat_idx = rng.choice(space, size=n_masks, replace=False)
    in_bit   = flat_idx // 256
    out_bit  = flat_idx %  256

    masks = np.zeros((n_masks, 8), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(n_masks):
        iw, ib = divmod(int(in_bit[i]), 64)
        ow, ob = divmod(int(out_bit[i]), 64)
        masks[i, iw]     = one << np.uint64(ib)
        masks[i, 4 + ow] = one << np.uint64(ob)

    return masks

def _b53_spectrum_analysis(sums_flat, n_samples):
    """Walsh spectrum analysis over sampled masks.
    Z_j = S_j / sqrt(N) ~ N(0,1) under H0 (independent uniform input/output)."""
    import math
    import warnings
    from scipy import stats as sp

    z = np.asarray(sums_flat, dtype=np.float64) / math.sqrt(n_samples)
    n = len(z)

    tails = {}
    for sigma in [3.0, 4.0, 5.0]:
        obs = int(np.sum(np.abs(z) > sigma))
        exp = 2.0 * sp.norm.sf(sigma) * n
        tails[sigma] = (obs, float(exp))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ad = sp.anderson(z, dist='norm')

    ad_mild_5pct = bool(ad.statistic > ad.critical_values[2])
    ad_reject_1pct = bool(ad.statistic > ad.critical_values[4])

    ks_stat, ks_p = sp.kstest(z, 'norm')

    return {
        "n": n,
        "max_abs_z": float(np.max(np.abs(z))),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "std_z": float(np.std(z)),
        "tails": tails,
        "ad_stat": float(ad.statistic),
        "ad_crit_5pct": float(ad.critical_values[2]),
        "ad_crit_1pct": float(ad.critical_values[4]),
        "ad_mild_5pct": ad_mild_5pct,
        "ad_reject_1pct": ad_reject_1pct,
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }

def _b53_evt_thresholds(n_masks_total):
    """Exact EVT thresholds for max|Z| over n_masks_total iid N(0,1) variables.
    P(max|Z| <= x) = (1 - 2*sf(x))^M
    Solved analytically via inverse survival function — zero sampling cost."""
    from scipy.stats import norm as _norm
    M = n_masks_total
    thresholds = {}
    for pct in [0.99, 0.999]:
        # Invert the exact CDF of max|Z| over M independent N(0,1) samples.
        sf_val = (1.0 - pct ** (1.0 / M)) / 2.0
        thresholds[pct] = float(_norm.isf(sf_val))
    # Asymptotic Gumbel-mode estimate for max|Z|.
    import math
    thresholds["mean_approx"] = math.sqrt(2.0 * math.log(max(2 * M, 2)))
    thresholds["M"] = M
    return thresholds

def _b53_print_spectrum(label, spec, mc_spec=None, mc_evt=None):
    """Print Walsh spectrum analysis for one branch."""
    mc_ref = f"  (MC: {mc_spec['max_abs_z']:.4f}σ)" if mc_spec else ""
    print(f"\n  ── {label} ──")
    print(f"  {'n Z-scores':<24}: {spec['n']:,}")
    print(f"  {'max |Z|':<24}: {spec['max_abs_z']:8.4f}σ{mc_ref}")
    print(f"  {'mean |Z|':<24}: {spec['mean_abs_z']:8.4f}σ"
          + (f"  (MC: {mc_spec['mean_abs_z']:.4f}σ)" if mc_spec else ""))
    print(f"  {'std Z':<24}: {spec['std_z']:8.4f}  (expected ~1.0000)")

    print(f"\n  Tail counts  (observed  expected  ratio):")
    for sigma, (obs, exp) in spec["tails"].items():
        ratio = obs / max(exp, 0.01)
        flag = (
            "  [!!]" if ratio > 3.0 and exp >= 1.0
            else "  [!]" if ratio > 2.0 and exp >= 1.0
            else ""
        )
        print(f"    |Z| > {sigma:.0f}σ :  "
              f"{obs:6d} obs  {exp:8.2f} exp  ratio = {ratio:5.2f}{flag}")

    ad_flag = (
        "  [!!] REJECT @ 1%" if spec["ad_reject_1pct"]
        else "  [~] mild 5% deviation" if spec["ad_mild_5pct"]
        else "  OK"
    )
    print(f"\n  Anderson-Darling : stat = {spec['ad_stat']:.4f}  "
          f"crit(5%) = {spec['ad_crit_5pct']:.4f}  "
          f"crit(1%) = {spec['ad_crit_1pct']:.4f}{ad_flag}")

    ks_flag = "  [!] p < 0.01" if spec["ks_p"] < 0.01 else ""
    print(f"  KS test          : stat = {spec['ks_stat']:.4f}  "
          f"p = {spec['ks_p']:.4f}{ks_flag}")
    if mc_evt is not None:
        p99 = mc_evt[0.99]
        p999 = mc_evt[0.999]
        mean_approx = mc_evt["mean_approx"]
        flag = ""
        if spec["max_abs_z"] > p999:
            flag = "  [!!] above EVT p99.9"
        elif spec["max_abs_z"] > p99:
            flag = "  [!]  above EVT p99"
        print(f"  EVT thresholds   : p99={p99:.4f}σ  p99.9={p999:.4f}σ  "
              f"(M={mc_evt['M']}, E[max]≈{mean_approx:.4f}σ)")
        if mc_spec is not None:
            delta = spec["max_abs_z"] - mc_spec["max_abs_z"]
            print(f"  Engine max|Z|    : {spec['max_abs_z']:.4f}σ{flag}")
            print(f"  MC max|Z|        : {mc_spec['max_abs_z']:.4f}σ")
            print(f"  Δ vs MC max      : {delta:+.4f}σ")
        else:
            print(f"  MC max|Z|        : {spec['max_abs_z']:.4f}σ")

def _b53_spec_fails(spec, mc_evt=None):
    tail4_obs, tail4_exp = spec["tails"][4.0]
    tail5_obs, tail5_exp = spec["tails"][5.0]
    fail_theoretical = (
        spec["ad_reject_1pct"]
        or tail4_obs > 4 * max(tail4_exp, 0.1)
        or (tail5_obs > 0 and tail5_exp < 0.5)
    )
    if mc_evt is not None:
        fail_mc = spec["max_abs_z"] > mc_evt[0.999]
        return fail_theoretical or fail_mc
    return fail_theoretical

def _b53_spec_warns(spec, mc_evt=None):
    tail3_obs, tail3_exp = spec["tails"][3.0]
    tail4_obs, tail4_exp = spec["tails"][4.0]
    warn_tail3 = tail3_obs > 2 * max(tail3_exp, 1.0)
    warn_tail4 = (tail4_obs >= 2) if tail4_exp < 1.0 else (tail4_obs > 2 * tail4_exp)
    warn_theoretical = spec["ad_mild_5pct"] or warn_tail3 or warn_tail4
    if mc_evt is not None:
        warn_mc = spec["max_abs_z"] > mc_evt[0.99]
        return warn_theoretical or warn_mc
    return warn_theoretical

# ==============================================================================
# WORKERS
# ==============================================================================

# ==========================
# B2 AVALANCHE / SAC
# ==========================

def worker_b2_full_flip(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b2_evaluate_flipped
        chunk, eval_chunk, flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi = args
        orig_matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        eval_matrix = np.ascontiguousarray(eval_chunk, dtype=np.uint64)
        sac_flat, av_hist = b2_evaluate_flipped(
            orig_matrix, eval_matrix,
            flip_s0_lo, flip_s0_hi,
            flip_s1_lo, flip_s1_hi,
        )
        return (sac_flat, av_hist)
    finally:
        gc.enable()

def worker_b2_single_flip(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b2_evaluate_flipped_lanes
        chunk, eval_chunk, flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi = args
        orig_matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        eval_matrix = np.ascontiguousarray(eval_chunk, dtype=np.uint64)
        sac_s0o0, sac_s0o1, sac_s1o0, sac_s1o1, sac_full, av_hist = b2_evaluate_flipped_lanes(
            orig_matrix, eval_matrix,
            flip_s0_lo, flip_s0_hi,
            flip_s1_lo, flip_s1_hi,
        )
        return (sac_s0o0, sac_s0o1, sac_s1o0, sac_s1o1, sac_full, av_hist)
    finally:
        gc.enable()

def worker_b2_conditional(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b2_conditional_sac
        chunk, eval_chunk, flip_bit, flip_s0_lo, flip_s0_hi, flip_s1_lo, flip_s1_hi = args
        orig_matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        eval_matrix = np.ascontiguousarray(eval_chunk, dtype=np.uint64)
        cond0_sac, cond1_sac, cond0_cnt, cond1_cnt = b2_conditional_sac(
            orig_matrix, eval_matrix,
            flip_bit,
            flip_s0_lo, flip_s0_hi,
            flip_s1_lo, flip_s1_hi,
        )
        return cond0_sac, cond1_sac, cond0_cnt, cond1_cnt
    finally:
        gc.enable()


# ==========================
# B13 STRUCTURED & TRUNCATED HIGHER-ORDER DIFFERENTIAL AUDIT
# ==========================

def worker_b13_st(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b13_structured_truncated
        chunk, eval_chunk = args
        orig = np.ascontiguousarray(chunk, dtype=np.uint64)
        evld = np.ascontiguousarray(eval_chunk, dtype=np.uint64)
        hw_sum, hw_sq_sum, zero_cnt, bit_counts, phase_b = b13_structured_truncated(
            orig, evld
        )
        return hw_sum, hw_sq_sum, zero_cnt, bit_counts, phase_b, orig.shape[0]
    finally:
        gc.enable()

# ==========================
# B16 JACOBIAN / INFLUENCE MATRIX
# ==========================

def worker_b16_jacobian(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b16_jacobian_diff
        chunk, = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        jacobian, row_hw_sum, row_hw_sq_sum, row_min_hw, row_count = b16_jacobian_diff(
            matrix
        )
        return jacobian, row_hw_sum, row_hw_sq_sum, row_min_hw, row_count
    finally:
        gc.enable()

# ==========================
# B22 ROTATIONAL BIT CORRELATION + CHAIN
# ==========================

def worker_b22_rot_chain(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b22_rot_chain
        chunk, rotation, steps = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        ones, chain_lt112, chain_lt104, chain_lt96, chain_lt88, chain_lt80 = b22_rot_chain(
            matrix, rotation, steps
        )
        return ones, int(chain_lt112), int(chain_lt104), int(chain_lt96), int(chain_lt88), int(chain_lt80), matrix.shape[0]
    finally:
        gc.enable()

def worker_b22_rot_chain_256(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b22_rot_chain_256
        chunk, rotation, steps = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        ones, chain_lt112, chain_lt104, chain_lt96, chain_lt88, chain_lt80 = b22_rot_chain_256(
            matrix, rotation, steps
        )
        return ones, int(chain_lt112), int(chain_lt104), int(chain_lt96), int(chain_lt88), int(chain_lt80), matrix.shape[0]
    finally:
        gc.enable()

# ==========================
# B32 GLOBAL CONVERGENCE
# ==========================

def worker_b32_global_convergence(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b32_find_dp
        chunk, max_steps, dp_bits, fp_bits = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        arr = b32_find_dp(matrix, max_steps, dp_bits, fp_bits)
        return np.array(arr, dtype=np.uint64), matrix.shape[0]
    finally:
        gc.enable()

def worker_b32_isotropy(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b32_isotropy
        chunk, steps, bucket_bits = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        counts = b32_isotropy(matrix, steps, bucket_bits)
        return np.array(counts, dtype=np.uint64)
    finally:
        gc.enable()

def worker_b32_cycle_signature(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b32_cycle_signature
        chunk, max_orbit, fp_bits = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        out = b32_cycle_signature(matrix, max_orbit, fp_bits)
        return np.array(out, dtype=np.uint64)
    finally:
        gc.enable()

def worker_b32_orbit_mixing(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b32_orbit_mixing
        chunk, max_steps = args
        matrix = np.ascontiguousarray(chunk, dtype=np.uint64)
        out = b32_orbit_mixing(matrix, max_steps)
        return np.array(out, dtype=np.uint64)
    finally:
        gc.enable()

# ==========================
# B51 ALGEBRAIC DEGREE TEST
# ==========================

def worker_b51_exact(args):
    import gc
    gc.disable()
    try:
        from lustro_rust import b51_exact_degree
        base_state, variable_bits, output_bit, batch_states = args
        result = b51_exact_degree(
            base_state,
            variable_bits,
            output_bit,
            batch_states,
        )
        return bool(result)
    finally:
        gc.enable()

def worker_b51_prob(args):
    import gc
    gc.disable()
    try:
        from lustro_rust import b51_prob_degree
        base_state, variable_bits, cubes, batch_states, seed = args
        result = b51_prob_degree(
            base_state,
            variable_bits,
            cubes,
            batch_states,
            seed,
        )
        return int(result)
    finally:
        gc.enable()

# ==========================
# B53 LINEAR CORRELATION / WALSH SPECTRUM TEST
# ==========================

def worker_b53_walsh(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b53_linear_correlation
        n_samples, n_masks, mask_kind, seed_states, seed_mask = args

        rng_states = np.random.default_rng(seed_states)
        rng_mask   = np.random.default_rng(seed_mask)

        states = rng_states.integers(0, 2**64, size=(n_samples, 4), dtype=np.uint64)
        masks  = (_b53_random_masks(n_masks, rng_mask) if mask_kind == "random"
                  else _b53_single_bit_masks(n_masks, rng_mask))

        sums = b53_linear_correlation(
            np.ascontiguousarray(states),
            np.ascontiguousarray(masks),
        )
        return np.array(sums, dtype=np.int64)

    except Exception as e:
        raise RuntimeError(f"B53 worker failure: {e}")
    finally:
        gc.enable()

def worker_b53_walsh_mc(args):
    import gc
    gc.disable()
    try:
        import numpy as np
        from lustro_rust import b53_walsh_precomputed
        n_samples, n_masks, mask_kind, seed_in, seed_out, seed_mask = args

        rng_in   = np.random.default_rng(seed_in)
        rng_out  = np.random.default_rng(seed_out)
        rng_mask = np.random.default_rng(seed_mask)

        s_in  = rng_in.integers(0, 2**64, size=(n_samples, 4), dtype=np.uint64)
        s_out = rng_out.integers(0, 2**64, size=(n_samples, 4), dtype=np.uint64)

        masks = (_b53_random_masks(n_masks, rng_mask) if mask_kind == "random"
                 else _b53_single_bit_masks(n_masks, rng_mask))

        sums = b53_walsh_precomputed(
            np.ascontiguousarray(s_in),
            np.ascontiguousarray(s_out),
            np.ascontiguousarray(masks),
        )
        return np.array(sums, dtype=np.int64)

    except Exception as e:
        raise RuntimeError(f"B53 MC worker failure: {e}")
    finally:
        gc.enable()

# ==============================================================================
# LustroAuditSuite
# ==============================================================================

class LustroAuditSuite:

    def __init__(self, profile_name="AUDIT"):
        self.profile_name = profile_name

        self.cpu_count = multiprocessing.cpu_count()

        self.master_seed = 42

        self.test_seeds = {
            "B2":  0xB2B2B2B2B2B2,
            "B13": 0xB13B13B13B13,
            "B16": 0xB16B16B16B16,
            "B22": 0xB22B22B22B22,
            "B32": 0xB32B32B32B32,
            "B51": 0xB51B51B51B51,
            "B53": 0xB53B53B53B53,
        }

        self.report = {}

    def fresh_pool(self):
        return multiprocessing.Pool(processes=self.cpu_count)

    def shutdown_pool(self, pool):
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass

    def shutdown(self):
        pass

    def test_rng(self, test_id: str, stream: str = "main"):
        """
        Deterministic RNG:
        - per test
        - per stream
        - order independent
        """

        stream_seeds = {
            "main": 0x1111111111111111,
            "states": 0x2222222222222222,
            "mc": 0x3333333333333333,
            "workers": 0x4444444444444444,
            "masks": 0x5555555555555555,
            "walsh": 0x6666666666666666,
            "subspace": 0x7777777777777777,
            "benchmark": 0x8888888888888888,
            "affine": 0x9999999999999999,
        }

        if test_id not in self.test_seeds:
            raise ValueError(f"unknown test_id={test_id}")

        if stream not in stream_seeds:
            raise ValueError(f"unknown stream={stream}")

        seed = (
                       self.master_seed
                       ^ self.test_seeds[test_id]
                       ^ stream_seeds[stream]
               ) & 0xFFFFFFFFFFFFFFFF

        return np.random.default_rng(seed)

    def chunkify(self, data):
        """
        Deterministic chunking.
        """

        n = len(data)

        k = self.cpu_count * 2

        chunk_size = (n + k - 1) // k

        chunks = []

        for i in range(0, n, chunk_size):
            chunk = data[i:i + chunk_size]

            chunks.append(chunk)

        return chunks

    def pool_map(self, pool, worker_fn, args):
        """
        Deterministic multiprocessing wrapper.
        """

        return pool.imap(
            worker_fn,
            args,
            chunksize=1
        )

    def run_test(self, test_id):
        test_map = {
            "B2": self._run_b2,
            "B13": self._run_b13,
            "B16": self._run_b16,
            "B22": self._run_b22,
            "B32": self._run_b32,
            "B51": self._run_b51,
            "B53": self._run_b53,
        }

        test_id = test_id.upper()
        if test_id not in test_map:
            print(f"[!] Unknown test: {test_id}")
            return

        print(f"[*] Running test: {test_id}...")
        pool = self.fresh_pool()
        t0 = time.perf_counter()
        try:
            test_map[test_id](pool)
        finally:
            self.shutdown_pool(pool)
        t1 = time.perf_counter()

        print(f"[OK] Test {test_id} finished in {t1 - t0:.2f}s.\n")

    def run_all(self):
        print(f"\n{'=' * 65}")
        print(f"{'LUSTRO CORE V1 - PREMIERE AUDIT':^65}")
        print(f"{'=' * 65}\n")


        temp_map = {"B2": None, "B13": None, "B16": None, "B22": None, "B32": None, "B51": None, "B53": None,}


        active_tests = sorted(
            temp_map.keys(),
            key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x)]
        )

        for t_id in active_tests:
            self.run_test(t_id)

# ==============================================================================
# TEST TEMPLATE
# ==============================================================================
# RNG:      self.test_rng("BX", "stream")  — one stream per domain
# Seeds:    self.worker_seed("BX", idx)    — never derive from execution order
# MP:       self.pool_map(pool, fn, args)  — ordered, chunksize=1
# Batch:    np.ascontiguousarray(arr)      — ndarray APIs only, not tuple APIs

# ==============================================================================
# RUNNERS
# ==============================================================================

# ==========================
# B2 AVALANCHE / SAC
# ==========================
# B2 tests avalanche behavior by measuring output-bit flip probabilities
# under single-bit, higher-order, structured, and conditional input differences.
# The ideal SAC baseline is P(output bit flips) = 0.5.
# ==========================

    def _b2_structured_masks(self):
        # Generate masks targeting progressively larger contiguous regions.
        masks = []

        word_labels = ["s0_lo", "s0_hi", "s1_lo", "s1_hi"]

        for word_idx, word_label in enumerate(word_labels):

            def make_mask(word_idx, value):
                s0_lo = value if word_idx == 0 else 0
                s0_hi = value if word_idx == 1 else 0
                s1_lo = value if word_idx == 2 else 0
                s1_hi = value if word_idx == 3 else 0
                return s0_lo, s0_hi, s1_lo, s1_hi

            for nibble_pos in range(16):
                value = 0xF << (nibble_pos * 4)
                masks.append((f"nibble_{word_label}[{nibble_pos}]",) + make_mask(word_idx, value))

            for byte_pos in range(8):
                value = 0xFF << (byte_pos * 8)
                masks.append((f"byte_{word_label}[{byte_pos}]",) + make_mask(word_idx, value))

            masks.append((f"half_lo_{word_label}",) + make_mask(word_idx, 0x00000000FFFFFFFF))
            masks.append((f"half_hi_{word_label}",) + make_mask(word_idx, 0xFFFFFFFF00000000))
            masks.append((f"full_{word_label}",)    + make_mask(word_idx, 0xFFFFFFFFFFFFFFFF))

        return masks

    def _run_b2(self, pool):
        import math
        from lustro_rust import LustroCoreV1Py

        LOCAL_SAMPLES        = 5_000_000
        NUM_BITS             = 256
        SECOND_ORDER_PAIRS   = 512
        NUM_OUTPUT_BITS      = 256

        print(f"\n[>>>] B2: AVALANCHE / SAC")
        print(f"[>>>] Samples: {LOCAL_SAMPLES:,}")
        print(f"[>>>] Single-bit flips: {NUM_BITS}")
        print(f"[>>>] Second-order pairs: {SECOND_ORDER_PAIRS}")

        rng_states = self.test_rng("B2", "states")
        states     = generate_states_u64(LOCAL_SAMPLES, rng_states)
        chunks     = self.chunkify(states)

        # Map the logical 256-bit index onto the four uint64 state words.
        def bit_to_flip_mask(bit):
            s0_lo = s0_hi = s1_lo = s1_hi = np.uint64(0)
            if bit < 64:
                s0_lo = np.uint64(1) << np.uint64(bit)
            elif bit < 128:
                s0_hi = np.uint64(1) << np.uint64(bit - 64)
            elif bit < 192:
                s1_lo = np.uint64(1) << np.uint64(bit - 128)
            else:
                s1_hi = np.uint64(1) << np.uint64(bit - 192)
            return int(s0_lo), int(s0_hi), int(s1_lo), int(s1_hi)

        # Evaluate the baseline states once and reuse the results across all differential tests.
        eval_chunks = []
        engine = LustroCoreV1Py()
        engine.lustro_init()
        for chunk in chunks:
            eval_chunk = np.array(chunk, dtype=np.uint64, copy=True)
            flat = eval_chunk.ravel()
            engine.evaluate_inplace(flat)
            eval_chunks.append(flat.reshape(chunk.shape[0], 4))


        sac_full    = np.zeros((NUM_BITS, NUM_OUTPUT_BITS), dtype=np.uint64)
        av_full     = np.zeros(257, dtype=np.uint64)

        lane_s0o0   = np.zeros((128, 128), dtype=np.uint64)
        lane_s0o1   = np.zeros((128, 128), dtype=np.uint64)
        lane_s1o0   = np.zeros((128, 128), dtype=np.uint64)
        lane_s1o1   = np.zeros((128, 128), dtype=np.uint64)

        sac_second  = np.zeros((SECOND_ORDER_PAIRS, NUM_OUTPUT_BITS), dtype=np.uint64)
        av_second   = np.zeros(257, dtype=np.uint64)

        structured_masks = self._b2_structured_masks()
        sac_struct  = np.zeros((len(structured_masks), NUM_OUTPUT_BITS), dtype=np.uint64)
        av_struct   = np.zeros(257, dtype=np.uint64)

        cond0_total   = np.zeros((NUM_BITS, 256), dtype=np.uint64)
        cond1_total   = np.zeros((NUM_BITS, 256), dtype=np.uint64)
        cond0_cnt_arr = np.zeros(NUM_BITS, dtype=np.uint64)
        cond1_cnt_arr = np.zeros(NUM_BITS, dtype=np.uint64)

        # Measure first-order SAC for every input bit, including lane-separated influence.
        print(f"  [1/5] single-bit flips...")

        for bit in range(NUM_BITS):
            s0_lo, s0_hi, s1_lo, s1_hi = bit_to_flip_mask(bit)
            args_lanes = [
                (chunk, eval_chunk, s0_lo, s0_hi, s1_lo, s1_hi)
                for chunk, eval_chunk in zip(chunks, eval_chunks)
            ]
            for result in self.pool_map(pool, worker_b2_single_flip, args_lanes):
                s0o0, s0o1, s1o0, s1o1, sac_flat, av_hist = result
                sac_full[bit] += sac_flat
                av_full += av_hist
                if bit < 128:
                    lane_s0o0[bit] += s0o0
                    lane_s0o1[bit] += s0o1
                else:
                    lane_s1o0[bit - 128] += s1o0
                    lane_s1o1[bit - 128] += s1o1

        # Measure output diffusion under random two-bit input differences.
        print(f"  [2/5] second-order SAC ({SECOND_ORDER_PAIRS} pairs)...")

        rng_so = self.test_rng("B2", "workers")

        for pair_idx in range(SECOND_ORDER_PAIRS):
            bit_a = int(rng_so.integers(0, NUM_BITS))
            bit_b = int(rng_so.integers(0, NUM_BITS))
            while bit_b == bit_a:
                bit_b = int(rng_so.integers(0, NUM_BITS))

            ma = bit_to_flip_mask(bit_a)
            mb = bit_to_flip_mask(bit_b)

            s0_lo = ma[0] ^ mb[0]
            s0_hi = ma[1] ^ mb[1]
            s1_lo = ma[2] ^ mb[2]
            s1_hi = ma[3] ^ mb[3]

            args_so = [
                (chunk, eval_chunk, s0_lo, s0_hi, s1_lo, s1_hi)
                for chunk, eval_chunk in zip(chunks, eval_chunks)
            ]
            for result in self.pool_map(pool, worker_b2_full_flip, args_so):
                sac_flat, av_hist = result
                sac_second[pair_idx] += sac_flat
                av_second += av_hist

        # Probe diffusion for structured differences spanning nibble, byte, half-word, and full-word regions.
        print(f"  [3/5] structured flips ({len(structured_masks)} masks)...")

        for mask_idx, (label, s0_lo, s0_hi, s1_lo, s1_hi) in enumerate(structured_masks):
            args_st = [
                (chunk, eval_chunk, s0_lo, s0_hi, s1_lo, s1_hi)
                for chunk, eval_chunk in zip(chunks, eval_chunks)
            ]
            for result in self.pool_map(pool, worker_b2_full_flip, args_st):
                sac_flat, av_hist = result
                sac_struct[mask_idx] += sac_flat
                av_struct += av_hist

        # Compare differential behavior conditioned on the original value of each flipped bit.
        print(f"  [4/5] conditional SAC ({NUM_BITS} bits)...")

        for bit in range(NUM_BITS):
            s0_lo, s0_hi, s1_lo, s1_hi = bit_to_flip_mask(bit)

            args_cond = [
                (chunk, eval_chunk, bit, s0_lo, s0_hi, s1_lo, s1_hi)
                for chunk, eval_chunk in zip(chunks, eval_chunks)
            ]
            for c0_sac, c1_sac, c0_cnt, c1_cnt in self.pool_map(
                    pool, worker_b2_conditional, args_cond):
                cond0_total[bit]   += c0_sac
                cond1_total[bit]   += c1_sac
                cond0_cnt_arr[bit] += c0_cnt
                cond1_cnt_arr[bit] += c1_cnt

        # Evaluate SAC and Hamming-weight deviations against the random baseline.
        print(f"  [5/5] analysis...")

        def sac_stats(sac_matrix, n_samples, label):
            probs         = sac_matrix.astype(np.float64) / n_samples
            devs          = np.abs(probs - 0.5)
            max_dev       = float(np.max(devs))
            mean_dev      = float(np.mean(devs))
            rms_dev       = float(np.sqrt(np.mean(devs ** 2)))
            sigma         = math.sqrt(0.25 / n_samples)
            z_raw         = max_dev / sigma
            z_ev_baseline = math.sqrt(2 * math.log(max(sac_matrix.size, 2)))
            z_ev_norm     = z_raw - z_ev_baseline
            worst_cell    = np.unravel_index(np.argmax(devs), devs.shape)

            print(f"\n  --- {label} ---")
            print(f"  Max deviation    : {max_dev:.6f}  (ideal 0.0)")
            print(f"  Mean deviation   : {mean_dev:.6f}")
            print(f"  RMS deviation    : {rms_dev:.6f}")
            print(f"  Worst cell       : {worst_cell}")
            print(f"  EV-normalized z  : {z_ev_norm:.2f}σ")

            if z_ev_norm > 3.0:
                print(f"  [!!] bias detected (EV-corrected)")
            elif z_ev_norm > 1.5:
                print(f"  [!]  weak signal")
            else:
                print(f"  [OK] uniform")

            return {
                "max_dev": max_dev, "mean_dev": mean_dev,
                "rms_dev": rms_dev, "z_ev_norm": z_ev_norm,
                "worst_cell": worst_cell,
            }

        def hw_stats(av_hist, label):
            weights     = np.arange(257, dtype=np.float64)
            total_f     = float(np.sum(av_hist))
            avg_hw      = float(np.sum(weights * av_hist) / total_f)
            var_hw      = float(np.sum(av_hist * (weights - avg_hw) ** 2) / total_f)

            print(f"\n  --- HW: {label} ---")
            print(f"  Avg HW           : {avg_hw:.4f}  (ideal 128.0)")
            print(f"  Variance HW      : {var_hw:.4f}  (ideal ~64.0)")
            print(f"  Avg HW deviation : {abs(avg_hw - 128.0):.4f}")

            if abs(avg_hw - 128.0) > 4.0:
                print(f"  [!!] poor diffusion (mean HW)")
            elif abs(avg_hw - 128.0) > 2.0:
                print(f"  [!]  slight diffusion bias")
            else:
                print(f"  [OK] no diffusion anomaly detected")

            # Under the Bernoulli(0.5) baseline, Var(HW) = 64; the estimator's CLT scale
            # provides the reference z-score for the observed variance.
            hw_var_sigma = math.sqrt(2) * 64.0 / math.sqrt(LOCAL_SAMPLES)
            hw_var_z = abs(var_hw - 64.0) / hw_var_sigma
            if hw_var_z > 5.0:
                print(f"  [!!] abnormal HW variance ({hw_var_z:.1f}σ from ideal)")
            elif hw_var_z > 3.0:
                print(f"  [!]  slight HW variance anomaly ({hw_var_z:.1f}σ from ideal)")
            else:
                print(f"  [OK] HW variance normal ({hw_var_z:.1f}σ from ideal)")

            return {"avg_hw": avg_hw, "var_hw": var_hw}

        print(f"\n[B2] RESULTS (samples={LOCAL_SAMPLES:,})")

        r_full   = sac_stats(sac_full,   LOCAL_SAMPLES, "SAC MATRIX (single-bit)")
        r_s0o0   = sac_stats(lane_s0o0,  LOCAL_SAMPLES, "LANE s0 -> o0")
        r_s0o1   = sac_stats(lane_s0o1,  LOCAL_SAMPLES, "LANE s0 -> o1")
        r_s1o0   = sac_stats(lane_s1o0,  LOCAL_SAMPLES, "LANE s1 -> o0")
        r_s1o1   = sac_stats(lane_s1o1,  LOCAL_SAMPLES, "LANE s1 -> o1")
        r_second = sac_stats(sac_second, LOCAL_SAMPLES, "SECOND-ORDER SAC")
        r_struct = sac_stats(sac_struct, LOCAL_SAMPLES, "STRUCTURED FLIPS")

        hw_full   = hw_stats(av_full,   "single-bit flips")
        hw_second = hw_stats(av_second, "second-order flips")
        hw_struct = hw_stats(av_struct, "structured flips")


        # Conditional SAC analysis
        cond_max_diff  = np.zeros(NUM_BITS, dtype=np.float64)
        cond_mean_diff = np.zeros(NUM_BITS, dtype=np.float64)
        cond_rms_diff  = np.zeros(NUM_BITS, dtype=np.float64)
        cond_z_ev      = np.zeros(NUM_BITS, dtype=np.float64)

        z_ev_baseline_cond = math.sqrt(2 * math.log(max(NUM_BITS * NUM_OUTPUT_BITS, 2)))

        for bit in range(NUM_BITS):
            c0_cnt = int(cond0_cnt_arr[bit])
            c1_cnt = int(cond1_cnt_arr[bit])
            if c0_cnt == 0 or c1_cnt == 0:
                continue

            p0   = cond0_total[bit].astype(np.float64) / c0_cnt
            p1   = cond1_total[bit].astype(np.float64) / c1_cnt
            diff = np.abs(p0 - p1)

            cond_max_diff[bit]  = float(np.max(diff))
            cond_mean_diff[bit] = float(np.mean(diff))
            cond_rms_diff[bit]  = float(np.sqrt(np.mean(diff ** 2)))

            sigma_cond    = math.sqrt(0.25 / c0_cnt + 0.25 / c1_cnt)
            cond_z_ev[bit] = (cond_max_diff[bit] / sigma_cond) - z_ev_baseline_cond

        worst_cond_bit  = int(np.argmax(cond_z_ev))
        overall_cond_max = float(cond_max_diff[worst_cond_bit])
        overall_z_ev    = float(cond_z_ev[worst_cond_bit])
        top_cond_bits   = np.argsort(cond_z_ev)[::-1][:10]

        print(f"\n  --- CONDITIONAL SAC ---")
        print(f"  Worst input bit  : {worst_cond_bit}")
        print(f"  Max |P0-P1|      : {overall_cond_max:.6f}")
        print(f"  EV-normalized z  : {overall_z_ev:.2f}σ")
        print(f"\n  Top 10 asymmetric input bits:")
        for b in top_cond_bits:
            print(f"    bit {b:3d} : max|P0-P1| = {cond_max_diff[b]:.6f}"
                  f"  mean = {cond_mean_diff[b]:.6f}"
                  f"  RMS = {cond_rms_diff[b]:.6f}"
                  f"  EV-z = {cond_z_ev[b]:.2f}σ")

        if overall_z_ev > 5.0:
            print(f"\n  [!!] conditional bias detected (EV-corrected)")
        elif overall_z_ev > 3.0:
            print(f"\n  [!]  weak conditional signal")
        else:
            print(f"\n  [OK] no conditional bias")

        self.report["B2"] = {
            "total":           LOCAL_SAMPLES,
            "sac_full":        r_full,
            "lane_s0o0":       r_s0o0,
            "lane_s0o1":       r_s0o1,
            "lane_s1o0":       r_s1o0,
            "lane_s1o1":       r_s1o1,
            "sac_second":      r_second,
            "sac_struct":      r_struct,
            "hw_full":         hw_full,
            "hw_second":       hw_second,
            "hw_struct":       hw_struct,
            "cond_max":        overall_cond_max,
            "cond_z_ev":       overall_z_ev,
            "worst_cond_bit":  worst_cond_bit,
        }

# ==========================
# B13 STRUCTURED & TRUNCATED HIGHER-ORDER DIFFERENTIAL AUDIT
# ==========================
# B13 probes second-order differentials on 13 architecture-motivated bit pairs.
# Phase A tests full-state Δ² Hamming weight against Bin(256, 0.5).
# Phase B tests truncated output projections against uniformity using KL divergence
# and an EVT-corrected deviation score.

    def _run_b13(self, pool):
        import math
        from lustro_rust import LustroCoreV1Py, b13_st_pairs_meta, b13_st_layout

        LOCAL_SAMPLES = 200_000_000

        NUM_PAIRS, SOURCES, LOW8, LOW12, LOW16, PHASE_B_LEN = b13_st_layout()

        # Preserve the Rust source ordering for histogram reconstruction.
        SOURCE_LABELS = ["d_w0", "d_w1", "d_w2", "d_w3", "fold"]

        pairs_raw = b13_st_pairs_meta()
        pairs_raw = pairs_raw.reshape(NUM_PAIRS, 2)
        PAIRS = [(int(pairs_raw[i, 0]), int(pairs_raw[i, 1])) for i in range(NUM_PAIRS)]

        _PAIR_CLASSES = {
            (63, 64): ("lane boundary", "lane"),
            (127, 128): ("lane boundary", "lane"),
            (191, 192): ("lane boundary", "lane"),
            (0, 1): ("carry chain", "carry"),
            (62, 63): ("carry chain", "carry"),
            (64, 65): ("carry chain", "carry"),
            (126, 127): ("carry chain", "carry"),
            (0, 32): ("rotational", "rotational"),
            (0, 64): ("rotational", "rotational"),
            (63, 127): ("rotational", "rotational"),
            (0, 128): ("cross-lane", "cross"),
            (127, 191): ("cross-lane", "cross"),
            (0, 255): ("mirror", "mirror"),
        }

        assert len(_PAIR_CLASSES) == NUM_PAIRS, \
            f"PAIR_CLASSES count {len(_PAIR_CLASSES)} != Rust NUM_PAIRS {NUM_PAIRS}"
        assert all(tuple(p) in _PAIR_CLASSES for p in PAIRS), \
            f"Rust pairs not covered by PAIR_CLASSES: {[p for p in PAIRS if tuple(p) not in _PAIR_CLASSES]}"

        PAIR_META = [(pair, *_PAIR_CLASSES[tuple(pair)]) for pair in PAIRS]

        print(f"\n[>>>] B13: STRUCTURED & TRUNCATED HIGHER-ORDER DIFFERENTIAL")
        print(f"[>>>] Samples: {LOCAL_SAMPLES:,}")
        print(f"[>>>] Pairs: {NUM_PAIRS} | Sources: {SOURCES} | Projections: LOW8/LOW12/LOW16")

        rng_states = self.test_rng("B13", "states")
        states     = generate_states_u64(LOCAL_SAMPLES, rng_states)
        chunks     = self.chunkify(states)

        # Evaluate baseline states once and reuse them for every structural pair.
        eval_chunks = []
        engine = LustroCoreV1Py()
        engine.lustro_init()
        for chunk in chunks:
            eval_chunk = np.array(chunk, dtype=np.uint64, copy=True)
            flat = eval_chunk.ravel()
            engine.evaluate_inplace(flat)
            eval_chunks.append(flat.reshape(chunk.shape[0], 4))


        hw_sum_acc    = np.zeros(NUM_PAIRS,            dtype=np.uint64)
        hw_sq_sum_acc = np.zeros(NUM_PAIRS,            dtype=np.uint64)
        zero_cnt_acc  = np.zeros(NUM_PAIRS,            dtype=np.uint64)
        bit_counts_acc = np.zeros(NUM_PAIRS * 256,     dtype=np.uint64)
        phase_b_acc   = np.zeros(NUM_PAIRS * PHASE_B_LEN, dtype=np.uint64)
        total         = 0

        args = [(chunk, eval_chunk) for chunk, eval_chunk in zip(chunks, eval_chunks)]

        for hw_sum, hw_sq_sum, zero_cnt, bit_counts, phase_b, n in \
                self.pool_map(pool, worker_b13_st, args):
            hw_sum_acc    += hw_sum
            hw_sq_sum_acc += hw_sq_sum
            zero_cnt_acc  += zero_cnt
            bit_counts_acc += bit_counts
            phase_b_acc   += phase_b
            total         += n

        n_f = float(total)

        # Correct the largest pair-wise deviation for multiple structural comparisons.
        evt_z_pairs = math.sqrt(2 * math.log(max(NUM_PAIRS, 2)))

        # Phase A analysis — per pair
        print(f"\n[>>>] B13 PHASE A — STRUCTURAL Δ² (HW distribution per pair)")
        print(f"\n  {'pair':<12}  {'class':<12}  {'avg_hw':>8}  {'var_hw':>8}  "
              f"{'z_hw':>7}  {'z_ev':>7}  {'zeros':>6}  {'max_dev':>8}  verdict")
        print(f"  {'-' * 100}")

        phase_a_results = []
        any_fail_a = False
        any_warn_a  = False

        for p, ((b1, b2), label, cls) in enumerate(PAIR_META):
            avg_hw = float(hw_sum_acc[p])    / n_f
            var_hw = float(hw_sq_sum_acc[p]) / n_f - avg_hw ** 2
            zeros  = int(zero_cnt_acc[p])

            sigma_hw = math.sqrt(64.0 / n_f)
            z_hw     = abs(avg_hw - 128.0) / sigma_hw
            z_hw_ev  = z_hw - evt_z_pairs

            # Measure the largest deviation of output-bit flip probability from 0.5.
            bc = bit_counts_acc[p * 256:(p + 1) * 256].astype(np.float64) / n_f
            max_dev = float(np.max(np.abs(bc - 0.5)))

            if zeros > 0 or z_hw_ev > 3.0:
                verdict    = "[!!]"
                any_fail_a = True
            elif z_hw_ev > 1.5:
                verdict   = "[! ]"
                any_warn_a = True
            else:
                verdict = "[OK]"

            pair_label = f"({b1},{b2})"
            print(f"  {pair_label:<12}  {cls:<12}  {avg_hw:>8.4f}  {var_hw:>8.4f}  "
                  f"{z_hw:>6.2f}σ  {z_hw_ev:>+6.2f}σ  {zeros:>6}  {max_dev:>8.6f}  {verdict}")

            phase_a_results.append({
                "b1": b1, "b2": b2, "cls": cls, "label": label,
                "avg_hw": avg_hw, "var_hw": var_hw,
                "z_hw": z_hw, "z_hw_ev": z_hw_ev,
                "zeros": zeros, "max_dev": max_dev,
            })

        # class summary Phase A
        print(f"\n  --- CLASS SUMMARY (Phase A) ---")
        for cls_name in ["lane", "carry", "rotational", "cross", "mirror"]:
            cls_r = [r for r in phase_a_results if r["cls"] == cls_name]
            if not cls_r:
                continue
            cls_avg    = float(np.mean([r["avg_hw"]  for r in cls_r]))
            cls_z      = float(np.max([r["z_hw"]     for r in cls_r]))
            cls_z_ev   = cls_z - evt_z_pairs
            cls_zeros  = sum(r["zeros"] for r in cls_r)
            flag = "[!!]" if cls_z_ev > 3.0 or cls_zeros > 0 else \
                   "[! ]" if cls_z_ev > 1.5 else "[OK]"
            print(f"  {cls_name:<12} : avg_hw={cls_avg:.4f}  "
                  f"max_z={cls_z:.2f}σ  ev_corr={cls_z_ev:+.2f}σ  zeros={cls_zeros}  {flag}")

        # Phase B analysis — truncated projections per pair per source
        print(f"\n[>>>] B13 PHASE B — TRUNCATED Δ² PROJECTIONS (KL vs uniform)")

        # KL divergence from the uniform distribution.
        def kl_uniform(hist, space):
            n_total  = float(np.sum(hist))
            if n_total == 0:
                return 0.0
            expected = 1.0 / space
            probs    = hist.astype(np.float64) / n_total
            mask     = probs > 0
            return float(np.sum(probs[mask] * np.log2(probs[mask] / expected)))

        def proj_z_ev(hist, space):
            n_total       = float(np.sum(hist))
            if n_total == 0:
                return 0.0
            expected      = 1.0 / space
            probs         = hist.astype(np.float64) / n_total
            max_dev       = float(np.max(np.abs(probs - expected)))
            sigma         = math.sqrt(expected * (1.0 - expected) / n_total)
            z_raw         = max_dev / sigma
            z_ev_baseline = math.sqrt(2 * math.log(max(space, 2)))
            return z_raw - z_ev_baseline

        # Offsets within PER_SOURCE_LEN block
        OFF_LOW8  = 0
        OFF_LOW12 = LOW8
        OFF_LOW16 = LOW8 + LOW12

        proj_widths = [
            ("LOW8",  LOW8,  OFF_LOW8),
            ("LOW12", LOW12, OFF_LOW12),
            ("LOW16", LOW16, OFF_LOW16),
        ]

        any_fail_b = False
        any_warn_b  = False

        # PER_SOURCE_LEN from Rust layout
        PER_SRC = LOW8 + LOW12 + LOW16

        print(f"\n  {'pair':<12}  {'source':<6}  {'proj':<6}  "
              f"{'z_ev':>7}  {'KL':>10}  verdict")
        print(f"  {'-' * 58}")

        phase_b_results = []

        for p, ((b1, b2), label, cls) in enumerate(PAIR_META):
            pb_block = phase_b_acc[p * PHASE_B_LEN:(p + 1) * PHASE_B_LEN]
            pair_label = f"({b1},{b2})"
            pair_entry = {"b1": b1, "b2": b2, "sources": {}}

            for s, src_label in enumerate(SOURCE_LABELS):
                src_block = pb_block[s * PER_SRC:(s + 1) * PER_SRC]
                src_entry = {}

                for proj_name, space, offset in proj_widths:
                    hist = src_block[offset:offset + space]
                    kl   = kl_uniform(hist, space)
                    zev  = proj_z_ev(hist, space)

                    if zev > 3.0:
                        verdict    = "[!!]"
                        any_fail_b = True
                    elif zev > 1.5:
                        verdict   = "[! ]"
                        any_warn_b = True
                    else:
                        verdict = "[OK]"

                    print(f"  {pair_label:<12}  {src_label:<6}  {proj_name:<6}  "
                          f"{zev:>+7.2f}σ  {kl:>10.6f}  {verdict}")

                    src_entry[proj_name] = {"kl": kl, "z_ev": zev}

                pair_entry["sources"][src_label] = src_entry

            phase_b_results.append(pair_entry)
            print()

        # Verdict
        print(f"  --- VERDICT ---")

        any_fail = any_fail_a or any_fail_b
        any_warn = (not any_fail) and (any_warn_a or any_warn_b)

        if any_fail_a:
            print(f"  [!!] Phase A: structural Δ² anomaly detected")
        if any_fail_b:
            print(f"  [!!] Phase B: truncated projection anomaly detected")
        if any_warn_a and not any_fail:
            print(f"  [!]  Phase A: weak structural signal")
        if any_warn_b and not any_fail:
            print(f"  [!]  Phase B: weak projection signal")
        if not any_fail and not any_warn:
            print(f"  [OK] no second-order structural bias detected at tested pairs")
            print(f"  [OK] no truncated projection signal above detection threshold")

        self.report["B13"] = {
            "total":        total,
            "any_fail":     any_fail,
            "any_warn":     any_warn,
            "any_fail_a":   any_fail_a,
            "any_warn_a":   any_warn_a,
            "any_fail_b":   any_fail_b,
            "any_warn_b":   any_warn_b,
            "phase_a":      phase_a_results,
            "phase_b":      phase_b_results,
        }

# ==========================
# B16 — JACOBIAN / INFLUENCE MATRIX
# ==========================
# B16 estimates the 256×256 bit-influence matrix by assigning one input-bit flip
# to each sample in round-robin order. This avoids evaluating all 256 flips per sample.
# EVT and Monte Carlo baselines identify unusually stiff input bits and concentrated
# output influence.

    def _run_b16(self, pool):
        import os
        import math as _math

        LOCAL_SAMPLES = 500_000_000

        print(f"\n[>>>] B16: RANDOMIZED JACOBIAN ESTIMATION")
        print(f"[>>>] Samples: {LOCAL_SAMPLES:,}")

        rng_states = self.test_rng("B16", "states")

        states = generate_states_u64(LOCAL_SAMPLES, rng_states)
        chunks = self.chunkify(states)

        args = [(chunk,) for chunk in chunks]

        jacobian_acc = np.zeros(256 * 256, dtype=np.uint64)
        row_hw_sum_acc = np.zeros(256, dtype=np.uint64)
        row_hw_sq_sum_acc = np.zeros(256, dtype=np.uint64)
        row_min_hw_acc = np.full(256, 256, dtype=np.uint64)
        row_count_acc = np.zeros(256, dtype=np.uint64)

        for jacobian, row_hw_sum, row_hw_sq_sum, row_min_hw, row_count in \
                self.pool_map(pool, worker_b16_jacobian, args):
            jacobian_acc += jacobian
            row_hw_sum_acc += row_hw_sum
            row_hw_sq_sum_acc += row_hw_sq_sum
            row_min_hw_acc = np.minimum(row_min_hw_acc, row_min_hw)
            row_count_acc += row_count


        safe_count = np.where(row_count_acc > 0, row_count_acc, 1).astype(np.float64)
        J = jacobian_acc.reshape(256, 256).astype(np.float64)
        P = J / safe_count[:, None]

        active = row_count_acc > 0
        P_active = P[active]

        def domain(bit):
            if bit < 64:
                return "s0_lo"
            elif bit < 128:
                return "s0_hi"
            elif bit < 192:
                return "s1_lo"
            else:
                return "s1_hi"

        # GLOBAL METRICS
        abs_dev_mat = np.abs(P - 0.5)
        active_rows = abs_dev_mat[active, :]

        mean_dev = float(np.mean(active_rows))
        max_dev = float(np.max(active_rows))
        rmse = float(np.sqrt(np.mean((P_active - 0.5) ** 2)))
        frac_good = float(np.mean(active_rows < 0.05))

        print(f"\n  --- GLOBAL METRICS ---")
        print(f"  Active input bits    : {int(np.sum(active))}")
        print(f"  Total samples        : {int(np.sum(row_count_acc)):,}")
        print(f"  Mean |P - 0.5|       : {mean_dev:.6f}  (ideal 0.000000)")
        print(f"  Max  |P - 0.5|       : {max_dev:.6f}  (ideal 0.000000)")
        print(f"  RMSE                 : {rmse:.6f}  (ideal 0.000000)")
        print(f"  Frac cells < 0.05    : {frac_good:.4f}  (ideal 1.0000)")

        # WORST 10 CELLS
        dev_full = abs_dev_mat.copy()
        dev_full[~active] = 0.0
        top20 = np.argsort(dev_full.ravel())[::-1][:10]

        print(f"\n  --- WORST 10 CELLS ---")
        print(f"  {'#':<3}  {'in':>3}  {'in_dom':<8}  {'out':>3}  {'out_dom':<8}  "
              f"{'P':>7}  {'|P-0.5|':>8}  {'count':>9}")
        print(f"  {'-' * 65}")

        for k, idx in enumerate(top20):
            i, j = divmod(int(idx), 256)
            print(f"  {k + 1:<3}  {i:>3}  {domain(i):<8}  {j:>3}  {domain(j):<8}  "
                  f"{P[i, j]:>7.4f}  {dev_full[i, j]:>8.4f}  {int(row_count_acc[i]):>9,}")

        # INPUT INFLUENCE
        row_abs_dev = np.mean(abs_dev_mat, axis=1)

        eps = 1e-15
        P_c = np.clip(P, eps, 1.0 - eps)
        row_entropy = np.mean(
            -(P_c * np.log2(P_c) + (1 - P_c) * np.log2(1 - P_c)),
            axis=1
        )

        hw_means = row_hw_sum_acc.astype(np.float64) / safe_count
        hw_mins = row_min_hw_acc.astype(np.float64)
        hw_vars = (row_hw_sq_sum_acc.astype(np.float64) / safe_count) - hw_means ** 2
        hw_stds = np.sqrt(np.maximum(hw_vars, 0.0))

        worst_input = np.argsort(row_abs_dev)[::-1][:10]

        print(f"\n  --- INPUT INFLUENCE (top 10 by row_abs_dev) ---")
        print(f"  {'bit':>3}  {'domain':<8}  {'count':>9}  {'row_abs_dev':>12}  "
              f"{'row_entropy':>12}  {'hw_mean':>8}  {'hw_std':>7}  {'hw_min':>7}")
        print(f"  {'-' * 84}")

        for i in worst_input:
            if not active[i]:
                continue
            print(f"  {i:>3}  {domain(i):<8}  {int(row_count_acc[i]):>9,}  "
                  f"{row_abs_dev[i]:>12.6f}  {row_entropy[i]:>12.6f}  "
                  f"{hw_means[i]:>8.2f}  {hw_stds[i]:>7.3f}  {hw_mins[i]:>7.0f}")

        low_entropy_input = np.argsort(
            np.where(active, row_entropy, np.inf)
        )[:10]

        print(f"\n  --- INPUT INFLUENCE (top 10 lowest entropy — corridors) ---")
        print(f"  {'bit':>3}  {'domain':<8}  {'row_entropy':>12}  {'row_abs_dev':>12}")
        print(f"  {'-' * 45}")

        for i in low_entropy_input:
            if not active[i]:
                continue
            print(f"  {i:>3}  {domain(i):<8}  {row_entropy[i]:>12.6f}  "
                  f"{row_abs_dev[i]:>12.6f}")

        # OUTPUT SENSITIVITY
        col_abs_dev = np.mean(abs_dev_mat[active], axis=0)
        col_entropy = np.mean(
            -(P_c[active] * np.log2(P_c[active])
              + (1 - P_c[active]) * np.log2(1 - P_c[active])),
            axis=0
        )

        worst_output = np.argsort(col_abs_dev)[::-1][:10]
        low_entropy_out = np.argsort(col_entropy)[:10]

        print(f"\n  --- OUTPUT SENSITIVITY (top 10 by col_abs_dev) ---")
        print(f"  {'bit':>3}  {'domain':<8}  {'col_entropy':>12}  {'col_abs_dev':>12}")
        print(f"  {'-' * 45}")

        for j in worst_output:
            print(f"  {j:>3}  {domain(j):<8}  {col_entropy[j]:>12.6f}  "
                  f"{col_abs_dev[j]:>12.6f}")

        print(f"\n  --- OUTPUT SENSITIVITY (top 10 lowest entropy — dead zones) ---")
        print(f"  {'bit':>3}  {'domain':<8}  {'col_entropy':>12}  {'col_abs_dev':>12}")
        print(f"  {'-' * 45}")
        for j in low_entropy_out:
            print(f"  {j:>3}  {domain(j):<8}  {col_entropy[j]:>12.6f}  "
                  f"{col_abs_dev[j]:>12.6f}")

        # LANE BIAS
        domains_labels = ["s0_lo", "s0_hi", "s1_lo", "s1_hi"]
        slices = [(0, 64), (64, 128), (128, 192), (192, 256)]

        print(f"\n  --- LANE BIAS (mean |P-0.5| per 64-bit block) ---")
        print(f"  {'':10}" + "".join(f"{'→' + d:>11}" for d in domains_labels))
        print("  " + "-" * (10 + 11 * 4))

        lane_max = 0.0
        for (r0, r1), d_in in zip(slices, domains_labels):
            row_str = f"  {d_in + '↓':<10}"
            for (c0, c1), _ in zip(slices, domains_labels):
                val = float(np.mean(np.abs(P[r0:r1, c0:c1] - 0.5)))
                row_str += f"{val:>11.6f}"
                lane_max = max(lane_max, val)
            print(row_str)

        # MONTE CARLO BASELINE
        MC_SAMPLES = 50_000
        rng_mc = self.test_rng("B16", "mc")

        # Use the mean per-row sample count as the Binomial MC baseline.
        cell_n = float(np.mean(safe_count[active]))
        cell_sigma = _math.sqrt(0.25 / cell_n)
        cell_n_int = max(round(cell_n), 1)

        # Bootstrap rows from the SAC null model: Binomial(cell_n, 0.5).
        mc_rows = rng_mc.binomial(cell_n_int, 0.5, size=(MC_SAMPLES, 256)).astype(np.float64)
        mc_p = mc_rows / cell_n_int

        # Empirical null distribution of the maximum cell deviation within one row.
        mc_row_max_z = np.max(np.abs(mc_p - 0.5) / cell_sigma, axis=1)
        mc_max_z_mean = float(np.mean(mc_row_max_z))
        mc_max_z_std = float(np.std(mc_row_max_z))
        mc_max_z_p99 = float(np.percentile(mc_row_max_z, 99))
        mc_max_z_p999 = float(np.percentile(mc_row_max_z, 99.9))

        # Bootstrap the global maximum across all active rows.
        n_active = int(np.sum(active))
        mc_global_max_z = np.max(
            rng_mc.choice(mc_row_max_z, size=(MC_SAMPLES, n_active), replace=True),
            axis=1
        )
        mc_global_mean = float(np.mean(mc_global_max_z))
        mc_global_std = float(np.std(mc_global_max_z))
        mc_global_p99 = float(np.percentile(mc_global_max_z, 99))
        mc_global_p999 = float(np.percentile(mc_global_max_z, 99.9))

        eps_mc = 1e-15
        mc_p_clip = np.clip(mc_p, eps_mc, 1.0 - eps_mc)
        mc_row_entropy = np.mean(
            -(mc_p_clip * np.log2(mc_p_clip) + (1 - mc_p_clip) * np.log2(1 - mc_p_clip)),
            axis=1
        )
        mc_entropy_mean = float(np.mean(mc_row_entropy))
        mc_entropy_std = float(np.std(mc_row_entropy))
        mc_entropy_p001 = float(np.percentile(mc_row_entropy, 0.1))

        # Scale the mean-row-entropy null distribution by the CLT across active rows.
        mc_mean_entropy_std = mc_entropy_std / _math.sqrt(max(n_active, 1))
        mc_mean_entropy_p001 = mc_entropy_mean - 3.1 * mc_mean_entropy_std

        # Bootstrap the lower tail of row entropy across active rows.
        mc_min_entropy = np.min(
            rng_mc.choice(mc_row_entropy, size=(MC_SAMPLES, n_active), replace=True),
            axis=1
        )
        mc_min_entropy_p01 = float(np.percentile(mc_min_entropy, 1.0))
        mc_min_entropy_p001 = float(np.percentile(mc_min_entropy, 0.1))

        # Observed metrics
        z_max_raw = max_dev / cell_sigma
        # Global z relative to MC global distribution
        z_max_mc = (z_max_raw - mc_global_mean) / max(mc_global_std, 1e-12)

        # Per-row metrics
        row_max_z = np.where(
            active,
            np.max(abs_dev_mat, axis=1) / cell_sigma,
            0.0
        )
        col_max_z = np.max(abs_dev_mat[active], axis=0) / cell_sigma

        # Row entropy z-score relative to MC distribution
        row_entropy_z = (row_entropy - mc_entropy_mean) / max(mc_entropy_std, 1e-12)
        # Col entropy z-score — approximation using row MC baseline
        col_entropy_z = (col_entropy - mc_entropy_mean) / max(mc_entropy_std, 1e-12)

        # Worst rows/cols
        top_stiff_z = np.argsort(row_max_z)[::-1][:10]
        top_weak_z = np.argsort(col_max_z)[::-1][:10]
        top_entropy_z = np.argsort(row_entropy_z)[:10]

        print(f"\n  --- MONTE CARLO BASELINE (n={MC_SAMPLES:,} simulations) ---")
        print(f"  Cell N (mean)        : {cell_n:.0f}")
        print(f"  Cell σ               : {cell_sigma:.6f}")
        print(f"  MC row max|z|        : mean={mc_max_z_mean:.2f}σ  "
              f"p99={mc_max_z_p99:.2f}σ  p99.9={mc_max_z_p999:.2f}σ")
        print(f"  MC global max|z|     : mean={mc_global_mean:.2f}σ  "
              f"p99={mc_global_p99:.2f}σ  p99.9={mc_global_p999:.2f}σ")
        print(f"  MC row entropy       : mean={mc_entropy_mean:.8f}  "
              f"std={mc_entropy_std:.2e}  p0.1={mc_entropy_p001:.8f}")
        print(f"  MC min row entropy   : p1={mc_min_entropy_p01:.8f}  "
              f"p0.1={mc_min_entropy_p001:.8f}")
        print(f"\n  Observed:")
        print(f"  max |z| raw          : {z_max_raw:.2f}σ")
        print(f"  MC-normalized z      : {z_max_mc:+.2f}σ  "
              f"(p99={mc_global_p99:.2f}σ, p99.9={mc_global_p999:.2f}σ)")

        if z_max_raw > mc_global_p999:
            print(f"  [!!] global max above MC p99.9")
        elif z_max_raw > mc_global_p99:
            print(f"  [!]  global max above MC p99")
        else:
            print(f"  [OK] global max within MC expectation")

        print(f"\n  Top 10 input bits by maximum cell z-score (per-row):")
        print(f"  {'bit':>3}  {'domain':<8}  {'row_entropy':>12}  {'max|z|':>8}  {'H_z(MC)':>10}")
        print(f"  {'-' * 55}")
        for i in top_stiff_z:
            if not active[i]:
                continue
            print(f"  {i:>3}  {domain(i):<8}  "
                  f"{row_entropy[i]:>12.6f}  {row_max_z[i]:>8.2f}σ  {row_entropy_z[i]:>10.2f}σ")

        print(f"\n  Top 10 output bits by maximum cell z-score (per-col):")
        print(f"  {'bit':>3}  {'domain':<8}  {'col_entropy':>12}  {'max|z|':>8}  {'H_z(MC)':>10}")
        print(f"  {'-' * 55}")
        for j in top_weak_z:
            print(f"  {j:>3}  {domain(j):<8}  "
                  f"{col_entropy[j]:>12.6f}  {col_max_z[j]:>8.2f}σ  {col_entropy_z[j]:>10.2f}σ")

        print(f"\n  Top 10 lowest entropy rows (corridor detection):")
        print(f"  {'bit':>3}  {'domain':<8}  {'row_entropy':>12}  {'max|z|':>8}  {'H_z(MC)':>10}")
        print(f"  {'-' * 57}")
        for i in top_entropy_z:
            if not active[i]:
                continue
            print(f"  {i:>3}  {domain(i):<8}  {row_entropy[i]:>12.6f}  "
                  f"{row_max_z[i]:>8.2f}σ  {row_entropy_z[i]:>10.2f}σ")

        # HEATMAP
        heatmap_path = "jacobian_heatmap_b16.png"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle(
                f"B16 Jacobian — samples={LOCAL_SAMPLES:,}",
                fontsize=13
            )

            im0 = axes[0].imshow(
                P - 0.5,
                cmap="seismic",
                vmin=-max_dev, vmax=max_dev,
                aspect="auto",
                interpolation="nearest"
            )
            axes[0].set_title(f"P[i,j] − 0.5  (diverging, ±{max_dev:.4f})", fontsize=11)
            axes[0].set_xlabel("output bit")
            axes[0].set_ylabel("input bit")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(
                np.abs(P - 0.5),
                cmap="hot",
                vmin=0.0, vmax=max_dev,
                aspect="auto",
                interpolation="nearest"
            )
            axes[1].set_title(f"|P[i,j] − 0.5|  (magnitude, max={max_dev:.4f})", fontsize=11)
            axes[1].set_xlabel("output bit")
            axes[1].set_ylabel("input bit")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            for ax in axes:
                for boundary in [64, 128, 192]:
                    ax.axhline(boundary - 0.5, color="lime", linewidth=0.6, alpha=0.7)
                    ax.axvline(boundary - 0.5, color="lime", linewidth=0.6, alpha=0.7)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(64))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(64))

            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  [OK] Heatmap saved → {os.path.abspath(heatmap_path)}")

        except ImportError:
            print(f"\n  [!] matplotlib not available — heatmap skipped")

        # SUMMARY
        mean_entropy_row = float(np.mean(row_entropy[active]))
        mean_entropy_col = float(np.mean(col_entropy))

        print(f"\n  --- VERDICT ---")
        print(f"  Mean row entropy     : {mean_entropy_row:.6f}  (ideal 1.000000)")
        print(f"  Mean col entropy     : {mean_entropy_col:.6f}  (ideal 1.000000)")

        # min_entropy_z relative to MC min distribution
        min_entropy_obs = float(np.min(row_entropy[active]))
        min_entropy_z_mc = (min_entropy_obs - float(np.mean(mc_min_entropy))) / max(
            float(np.std(mc_min_entropy)), 1e-12
        )

        any_fail = (
                z_max_raw > mc_global_p999
        )

        any_warn = (
                (not any_fail)
                and
                (z_max_raw > mc_global_p99)
        )

        if any_fail:
            print(
                f"  [!!] Jacobian anomaly  "f"(max|z|={z_max_raw:.2f}σ vs MC p99.9={mc_global_p999:.2f}σ)")
        elif any_warn:
            print(
                f"  [!]  slight non-uniformity  "f"(max|z|={z_max_raw:.2f}σ vs MC p99={mc_global_p99:.2f}σ)")
        else:
            print(f"  [OK] corridor metric below detection threshold at current sample size")

        self.report["B16"] = {
            "total": int(np.sum(row_count_acc)),
            "mean_dev": mean_dev,
            "max_dev": max_dev,
            "rmse": rmse,
            "frac_good": frac_good,
            "lane_max": lane_max,
            "mean_entropy_row": mean_entropy_row,
            "mean_entropy_col": mean_entropy_col,
            "z_max_raw": z_max_raw,
            "z_max_mc": z_max_mc,
            "mc_global_p99": mc_global_p99,
            "mc_global_p999": mc_global_p999,
            "mc_entropy_mean": mc_entropy_mean,
            "mc_entropy_std": mc_entropy_std,
            "mc_mean_entropy_std": mc_mean_entropy_std,
            "mc_mean_entropy_p001": mc_mean_entropy_p001,
            "min_entropy_obs":        min_entropy_obs,
            "min_entropy_z_mc":       min_entropy_z_mc,
            "mc_min_entropy_p01":     mc_min_entropy_p01,
            "mc_min_entropy_p001":    mc_min_entropy_p001,
            "any_fail": any_fail,
            "any_warn": any_warn,
            "heatmap": heatmap_path,
        }

# ==========================
# B22 — ROTATIONAL BIT CORRELATION + CHAIN
# ==========================
# B22 tests rotational symmetry by comparing f(rotl(x)) with rotl(f(x)).
# The first-step difference exposes per-bit bias; repeated iterations test
# whether unusually low Hamming-weight differences persist through the chain.

    def _run_b22(self, pool):
        import math
        from scipy.stats import binom as scipy_binom

        LOCAL_SAMPLES  = 200_000_000
        STEPS          = 3
        ROTATIONS_128  = [1, 3, 7, 13, 16, 32, 47, 63]
        ROTATIONS_256  = [1, 7, 13, 32, 64, 96, 128, 192]
        NUM_BITS       = 256

        print(f"\n[>>>] B22: ROTATIONAL PER-BIT BIAS + CHAIN PERSISTENCE")
        print(f"[>>>] Samples: {LOCAL_SAMPLES:,} | Chain steps: {STEPS}")
        print(f"[>>>] Rotations 2x128: {ROTATIONS_128}")
        print(f"[>>>] Rotations 256:   {ROTATIONS_256}")

        rng_states = self.test_rng("B22", "states")
        states     = generate_states_u64(LOCAL_SAMPLES, rng_states)
        chunks     = self.chunkify(states)

        n_rot_128   = len(ROTATIONS_128)
        n_rot_256   = len(ROTATIONS_256)
        evt_z_128 = math.sqrt(2 * math.log(max(n_rot_128 * NUM_BITS, 2)))
        evt_z_256 = math.sqrt(2 * math.log(max(n_rot_256 * NUM_BITS, 2)))
        evt_z_per_rot = math.sqrt(2 * math.log(max(NUM_BITS, 2)))

        thresholds = [112, 104, 96, 88, 80]
        expected_chain = {}
        for thr in thresholds:
            p_single = float(scipy_binom.cdf(thr - 1, 256, 0.5))
            p_chain  = p_single ** STEPS
            expected_chain[thr] = LOCAL_SAMPLES * p_chain

        # Chain persistence
        def run_chain(worker_fn, rotations, evt_z_baseline, mode_label):
            import math as _math
            from scipy.stats import poisson as _poisson

            def _poisson_p(obs, exp):
                if obs == 0:
                    return 1.0
                return float(_poisson.sf(obs - 1, max(exp, 1e-12)))

            chain_results = {}
            bias_results = {}
            any_fail_chain = False
            any_warn_chain = False
            any_fail_bias = False
            any_warn_bias = False

            for rot in rotations:
                args = [(chunk, rot, STEPS) for chunk in chunks]

                total_ones = np.zeros(NUM_BITS, dtype=np.uint64)
                total_lt112 = 0
                total_lt104 = 0
                total_lt96 = 0
                total_lt88 = 0
                total_lt80 = 0
                total = 0

                for ones, lt112, lt104, lt96, lt88, lt80, n in self.pool_map(pool, worker_fn, args):
                    total_ones += ones
                    total_lt112 += lt112
                    total_lt104 += lt104
                    total_lt96 += lt96
                    total_lt88 += lt88
                    total_lt80 += lt80
                    total += n

                # --- chain ---
                chain_results[rot] = {
                    "lt112": total_lt112,
                    "lt104": total_lt104,
                    "lt96": total_lt96,
                    "lt88": total_lt88,
                    "lt80": total_lt80,
                    "p_lt96": total_lt96 / total if total > 0 else 0.0,
                }

                p_lt96 = _poisson_p(total_lt96, expected_chain[96])
                p_lt104 = _poisson_p(total_lt104, expected_chain[104])
                p_lt112 = _poisson_p(total_lt112, expected_chain[112])

                if p_lt96 < 1e-6:
                    any_fail_chain = True
                elif p_lt104 < 1e-3 or p_lt112 < 1e-3:
                    any_warn_chain = True

                # First-step bit-bias statistic.
                probs = total_ones.astype(np.float64) / total
                devs = np.abs(probs - 0.5)
                sigma = math.sqrt(0.25 / total)
                z_scores = devs / sigma
                max_z = float(np.max(z_scores))
                mean_z = float(np.mean(z_scores))
                worst = int(np.argmax(z_scores))
                z_ev = max_z - evt_z_baseline

                bias_results[rot] = {
                    "max_z": max_z, "mean_z": mean_z,
                    "z_ev": z_ev, "worst_bit": worst,
                }

                if z_ev > 3.0:
                    any_fail_bias = True
                elif z_ev > 1.5:
                    any_warn_bias = True

            def _fmt_exp(val):
                # >= 0.1: plain float; below: 10^X notation (log10-safe)
                if val >= 0.1:
                    return f"{val:.1f}"
                import math as _m
                return f"10^{_m.log10(max(val, 1e-300)):.1f}"

            # Reference scale: HW ~ Binomial(256, 0.5), with μ=128 and σ=8.
            sigma_labels = {112: "-2σ", 104: "-3σ", 96: "-4σ", 88: "-5σ", 80: "-6σ"}

            print(f"\n  --- {mode_label} CHAIN PERSISTENCE (steps={STEPS}) ---")
            print(f"  Expected baseline ({LOCAL_SAMPLES // 1_000_000}M samples):"
                  f"  fail: p96 < 1e-6  |  warn: p104 or p112 < 1e-3")
            for thr in thresholds:
                slabel = sigma_labels.get(thr, "")
                print(f"  {'<' + str(thr):>5} ({slabel:>3}) : {_fmt_exp(expected_chain[thr])}")
            print(f"\n  {'rot':>4}  {'lt112':>8}  {'lt104':>8}  {'lt96':>8}  "
                  f"{'lt88':>8}  {'lt80':>8}  verdict")
            print(f"  {'-' * 65}")
            for rot in rotations:
                r    = chain_results[rot]
                p96  = _poisson_p(r["lt96"],  expected_chain[96])
                p104 = _poisson_p(r["lt104"], expected_chain[104])
                p112 = _poisson_p(r["lt112"], expected_chain[112])
                if p96 < 1e-6:
                    v = f"[!!] p96={p96:.2e}  obs={r['lt96']}  exp={_fmt_exp(expected_chain[96])}"
                elif p104 < 1e-3:
                    v = f"[! ] p104={p104:.2e}  obs={r['lt104']}  exp={_fmt_exp(expected_chain[104])}"
                elif p112 < 1e-3:
                    v = f"[! ] p112={p112:.2e}  obs={r['lt112']}  exp={_fmt_exp(expected_chain[112])}"
                else:
                    v = "[OK]"
                print(f"  {rot:>4}  {r['lt112']:>8}  {r['lt104']:>8}  {r['lt96']:>8}  "
                      f"{r['lt88']:>8}  {r['lt80']:>8}  {v}")

            print(f"\n  --- {mode_label} PER-BIT BIAS (from chain step 1) ---")
            print(f"  EVT per-rot: {evt_z_per_rot:.2f}σ  |  "
                  f"EVT global: {evt_z_baseline:.2f}σ  "
                  f"({len(rotations)} rot × {NUM_BITS} bits)")
            print(f"  {'rot':>4}  {'max_z':>7}  {'mean_z':>7}  "
                  f"{'z_ev':>7}  {'worst_bit':>9}  verdict")
            print(f"  {'-' * 57}")
            for rot in rotations:
                r = bias_results[rot]
                v = "[!!]" if r["z_ev"] > 3.0 else "[! ]" if r["z_ev"] > 1.5 else "[OK]"
                print(f"  {rot:>4}  {r['max_z']:>6.2f}σ  {r['mean_z']:>6.2f}σ  "
                      f"{r['z_ev']:>+6.2f}σ  {r['worst_bit']:>9}  {v}")

            return chain_results, bias_results, any_fail_chain, any_warn_chain, any_fail_bias, any_warn_bias

        print(f"\n  [1/2] 2x128 chain + bias...")
        res_chain_128, res_bias_128, fail_chain_128, warn_chain_128, fail_bias_128, warn_bias_128 = run_chain(
            worker_b22_rot_chain, ROTATIONS_128, evt_z_128, "2x128"
        )
        print(f"\n  [2/2] 256-bit chain + bias...")
        res_chain_256, res_bias_256, fail_chain_256, warn_chain_256, fail_bias_256, warn_bias_256 = run_chain(
            worker_b22_rot_chain_256, ROTATIONS_256, evt_z_256, "256-BIT"
        )

        # SUMMARY
        any_fail_bias = fail_bias_128 or fail_bias_256
        any_warn_bias = warn_bias_128 or warn_bias_256
        any_fail_chain = fail_chain_128 or fail_chain_256
        any_warn_chain = warn_chain_128 or warn_chain_256

        print(f"\n  --- VERDICT ---")

        if any_fail_chain:
            print(f"  [!!] ROTATIONAL CHAIN ANOMALY DETECTED")
        elif any_warn_chain:
            print(f"  [!]  weak chain signal (lt112 above baseline)")
        elif any_fail_bias:
            print(f"  [!!] rotational per-bit bias detected (EV-corrected)")
        elif any_warn_bias:
            print(f"  [!]  weak rotational per-bit signal")
        else:
            print(f"  [OK] no rotational bias at tested rotations")

        self.report["B22"] = {
            "total":           LOCAL_SAMPLES,
            "steps":           STEPS,
            "any_fail_bias":   any_fail_bias,
            "any_warn_bias":   any_warn_bias,
            "any_fail_chain":  any_fail_chain,
            "any_warn_chain":  any_warn_chain,
        }

# ==========================
# B32 GLOBAL CONVERGENCE
# ==========================

    # ==========================
    # B32A — GLOBAL CONVERGENCE (DISTINGUISHED POINTS)
    # ==========================
    # B32A uses a rho-style walk to distinguished points.
    # Collision counts are compared with the birthday bound, while the
    # distinguished-point hit rate is compared with the corresponding geometric tail.

    def _run_b32(self, _pool=None):
        import math

        LOCAL_SAMPLES = 2_000_000
        MAX_STEPS     = 65_536
        DP_BITS       = 16
        FP_BITS       = 32

        print(f"\n[>>>] B32A: GLOBAL CONVERGENCE TEST")
        print(f"[>>>] Samples: {LOCAL_SAMPLES:,}")
        print(f"[>>>] DP bits: {DP_BITS} | FP bits: {FP_BITS}")
        print(f"[>>>] Max steps: {MAX_STEPS:,}")

        # DATA
        rng_states = self.test_rng("B32", "states")
        states = generate_states_u64(LOCAL_SAMPLES, rng_states)

        from lustro_rust import b32_find_dp
        arr = b32_find_dp(
            np.ascontiguousarray(states, dtype=np.uint64),
            MAX_STEPS, DP_BITS, FP_BITS
        )

        fps = arr[:, 0]
        steps = arr[:, 1]
        reason = arr[:, 2]

        total = LOCAL_SAMPLES
        total_steps = int(np.sum(steps))
        max_steps_seen = int(np.max(steps))
        hit_mask = reason == 0
        dp_hits = int(np.sum(hit_mask))
        dp_steps_total = int(np.sum(steps[hit_mask]))
        runaways = int(np.sum(~hit_mask))

        fp_vals, fp_counts = np.unique(fps[hit_mask], return_counts=True)
        unique_fp = len(fp_vals)
        max_bucket = int(fp_counts.max()) if unique_fp > 0 else 0
        collisions = int(np.sum(fp_counts * (fp_counts - 1) // 2))
        avg_steps = total_steps / total if total > 0 else 0.0
        avg_dp_steps = dp_steps_total / dp_hits if dp_hits > 0 else 0.0

        # RANDOM BASELINE
        N        = dp_hits
        M        = 2 ** FP_BITS
        expected = N * (N - 1) / (2 * M) if M > 0 else 0.0
        std      = math.sqrt(expected) if expected > 0 else 1.0
        z        = (collisions - expected) / std if std > 0 else 0.0

        runaway_rate     = runaways / total if total > 0 else 0.0
        expected_runaway = math.exp(-MAX_STEPS / float(2 ** DP_BITS))
        runaway_diff     = runaway_rate - expected_runaway

        # ENTROPY
        entropy = 0.0
        if dp_hits > 0:
            probs = fp_counts.astype(np.float64) / dp_hits
            entropy = float(-np.sum(probs * np.log2(probs)))

        max_entropy  = min(FP_BITS, math.log2(max(dp_hits, 1)))
        entropy_loss = max_entropy - entropy

        # OUTPUT
        print(f"\n  --- RESULTS ---")
        print(f"  Total samples      : {total:,}")
        print(f"  DP hits            : {dp_hits:,}")
        print(f"  Runaways           : {runaways:,}")
        print(f"  Runaway rate       : {runaway_rate:.6e}")
        print(f"  Expected runaway   : {expected_runaway:.6e}")
        print(f"  Runaway deviation  : {runaway_diff:+.6e}")
        print(f"\n  [FINGERPRINT SPACE]")
        print(f"  Unique fingerprints: {unique_fp:,}")
        print(f"  Collisions         : {collisions:,}")
        print(f"  Expected           : {expected:.2f}")
        print(f"  Z-score            : {z:+.2f}σ")
        print(f"  Max bucket         : {max_bucket}")
        print(f"\n  [TRAJECTORY STATS]")
        print(f"  Avg steps (all)    : {avg_steps:.2f}")
        print(f"  Avg steps (DP only): {avg_dp_steps:.2f}")
        print(f"  Max steps observed : {max_steps_seen:,}")
        print(f"\n  [ENTROPY]")
        print(f"  Entropy            : {entropy:.4f} bits")
        print(f"  Max entropy        : {max_entropy:.4f} bits")
        print(f"  Entropy loss       : {entropy_loss:.4f}")

        # VERDICT
        print(f"\n  --- VERDICT ---")
        any_fail = False
        any_warn = False

        if abs(z) > 5.0:
            print(f"  [!!] structural convergence detected")
            any_fail = True
        elif abs(z) > 3.0:
            print(f"  [!]  weak convergence signal")
            any_warn = True
        else:
            print(f"  [OK] fingerprint distribution normal")

        # Entropy loss is reported as a diagnostic rather than a standalone verdict.
        print(f"  [diagnostic] entropy_loss={entropy_loss:.4f}  "
              f"(informational only)")

        runaway_sigma = math.sqrt(
            expected_runaway * (1.0 - expected_runaway) / total
        ) if total > 0 else 1.0
        runaway_z = runaway_diff / runaway_sigma if runaway_sigma > 0 else 0.0

        print(f"  Runaway z-score    : {runaway_z:+.2f}σ")
        if abs(runaway_z) > 5.0:
            print(f"  [!!] runaway dynamics inconsistent with random walk")
            any_fail = True
        elif abs(runaway_z) > 3.0:
            print(f"  [!]  slight runaway deviation")
            any_warn = True
        else:
            print(f"  [OK] runaway dynamics match theory")

        self.report["B32"] = {
            "total":            total,
            "dp_hits":          dp_hits,
            "runaways":         runaways,
            "expected_runaway": expected_runaway,
            "runaway_diff":     runaway_diff,
            "runaway_z":        runaway_z,
            "collisions":       collisions,
            "expected":         expected,
            "z":                z,
            "entropy":          entropy,
            "entropy_loss":     entropy_loss,
            "max_bucket":       max_bucket,
            "avg_steps":        avg_steps,
            "avg_dp_steps":     avg_dp_steps,
            "max_steps_seen":   max_steps_seen,
            "any_fail":         any_fail,
            "any_warn":         any_warn,
        }

        # ==========================
        # B32B — XOR-PROJECTION OCCUPANCY
        # ==========================
        # B32B projects each iterated state onto a reduced XOR fingerprint and
        # tests bucket occupancy. Entropy and the EVT-corrected maximum occupancy
        # are primary signals; chi-square is diagnostic as orbit visits are correlated.

        ISO_SAMPLES = 10_000_000 # 20_000_000
        ISO_STEPS = 128 # 1024
        BUCKET_BITS = 16
        N_BUCKETS = 1 << BUCKET_BITS

        print(f"\n[>>>] B32B: XOR-PROJECTION OCCUPANCY")
        print(f"[>>>] Samples: {ISO_SAMPLES:,} | Steps/state: {ISO_STEPS} | Buckets: {N_BUCKETS:,}")
        print(f"[>>>] Projection: upper {BUCKET_BITS} bits of s0^s1^s2^s3")

        rng_iso = self.test_rng("B32", "mc")
        iso_states = generate_states_u64(ISO_SAMPLES, rng_iso)

        from lustro_rust import b32_isotropy
        bucket_acc = b32_isotropy(
            np.ascontiguousarray(iso_states, dtype=np.uint64),
            ISO_STEPS, BUCKET_BITS
        )

        total_visits = float(np.sum(bucket_acc))
        expected_iso = total_visits / N_BUCKETS
        probs_iso = bucket_acc.astype(np.float64) / total_visits

        import math as _math
        chi2 = float(np.sum((bucket_acc.astype(np.float64) - expected_iso) ** 2 / expected_iso))
        chi2_dof = N_BUCKETS - 1
        chi2_sigma = _math.sqrt(2 * chi2_dof)
        chi2_z = (chi2 - chi2_dof) / chi2_sigma

        # Entropy
        p_clip = np.clip(probs_iso, 1e-15, 1.0)
        iso_entropy = float(-np.sum(p_clip * np.log2(p_clip)))
        max_entropy = _math.log2(N_BUCKETS)
        entropy_loss_iso = max_entropy - iso_entropy

        # Max occupancy
        max_bucket_iso = int(np.max(bucket_acc))
        max_bucket_z = (max_bucket_iso - expected_iso) / _math.sqrt(expected_iso)

        # EVT reference for the largest bucket deviation.
        evt_baseline_bucket = _math.sqrt(2 * _math.log(max(N_BUCKETS, 2)))
        bucket_z_ev = max_bucket_z - evt_baseline_bucket

        print(f"\n  --- XOR-PROJECTION OCCUPANCY RESULTS ---")
        print(f"  Total visits         : {int(total_visits):,}")
        print(f"  Expected per bucket  : {expected_iso:.2f}")
        print(f"  EVT baseline (max z) : {evt_baseline_bucket:.2f}σ")
        print(f"  Max bucket visits    : {max_bucket_iso:,}  "
              f"(z={max_bucket_z:+.2f}σ, EV-corr={bucket_z_ev:+.2f}σ)")
        print(f"  Chi-square           : {chi2:.2f}  (dof={chi2_dof}, z={chi2_z:+.2f}σ)")
        print(f"  Bucket entropy       : {iso_entropy:.6f} / {max_entropy:.6f} bits")
        print(f"  Entropy loss         : {entropy_loss_iso:.6f}")

        # Within-orbit correlation inflates chi-square; entropy loss is the primary signal.
        iso_fail = entropy_loss_iso > 0.05 or bucket_z_ev > 5.0
        iso_warn = (not iso_fail) and (entropy_loss_iso > 0.005 or bucket_z_ev > 3.0)

        if iso_fail:
            print(f"  [!!] XOR-projection non-uniformity detected  "
                  f"(entropy_loss={entropy_loss_iso:.6f}, chi2_z={chi2_z:+.2f}σ)")
        elif iso_warn:
            print(f"  [!]  mild XOR-projection deviation  "
                  f"(entropy_loss={entropy_loss_iso:.6f}, chi2_z={chi2_z:+.2f}σ)")
        else:
            print(f"  [OK] XOR-projection uniform — no concentration detected")

        self.report["B32B"] = {
            "total_visits": int(total_visits),
            "chi2": chi2,
            "chi2_z": chi2_z,
            "iso_entropy": iso_entropy,
            "entropy_loss": entropy_loss_iso,
            "max_bucket": max_bucket_iso,
            "max_bucket_z": max_bucket_z,
            "evt_baseline_bucket": evt_baseline_bucket,
            "bucket_z_ev": bucket_z_ev,
            "any_fail": iso_fail,
            "any_warn": iso_warn,
        }

        # ==========================
        # B32C — ORBIT MIXING
        # ==========================
        # B32C tracks per-step diffusion metrics along deterministic orbits:
        # global/per-word Hamming weight, per-bit density, and HW autocorrelation.
        # The results are diagnostic only; no independent H0 model is assumed
        # due to successive orbit states being deterministically related.

        import math as _math

        ORBIT_SAMPLES = 10_000_000 # 20_000_000
        ORBIT_STEPS = 128 # 1024

        print(f"\n[>>>] B32C: ORBIT MIXING UNDER ITERATION")
        print(f"[>>>] Samples: {ORBIT_SAMPLES:,} | Steps: {ORBIT_STEPS}")

        rng_orbit = self.test_rng("B32", "affine")
        orbit_states = generate_states_u64(ORBIT_SAMPLES, rng_orbit)

        from lustro_rust import b32_orbit_mixing
        STRIDE = 5 + 256
        orbit_acc = b32_orbit_mixing(
            np.ascontiguousarray(orbit_states, dtype=np.uint64),
            ORBIT_STEPS
        ).reshape(ORBIT_STEPS, STRIDE)

        hw_mean = orbit_acc[:, 0].astype(np.float64) / ORBIT_SAMPLES
        w0_mean = orbit_acc[:, 1].astype(np.float64) / ORBIT_SAMPLES
        w1_mean = orbit_acc[:, 2].astype(np.float64) / ORBIT_SAMPLES
        w2_mean = orbit_acc[:, 3].astype(np.float64) / ORBIT_SAMPLES
        w3_mean = orbit_acc[:, 4].astype(np.float64) / ORBIT_SAMPLES

        # Per-bit density: shape (ORBIT_STEPS, 256) — mean fraction per bit per step
        bit_density = orbit_acc[:, 5:].astype(np.float64) / ORBIT_SAMPLES  # ideal: 0.5 per bit

        # Random-mixing reference: HW=128 overall and 32 per 64-bit word.
        hw_dev = np.abs(hw_mean - 128.0)
        w_dev_max = np.max(np.abs(np.stack([w0_mean, w1_mean, w2_mean, w3_mean], axis=0) - 32.0), axis=0)

        # CLT scale for the sample mean under the Binomial(256, 0.5) baseline.
        sigma_hw = _math.sqrt(64.0 / ORBIT_SAMPLES)
        sigma_w = _math.sqrt(16.0 / ORBIT_SAMPLES)  # per-word: 64 bits, p=0.5

        hw_z_max = float(np.max(hw_dev / sigma_hw))
        w_z_max = float(np.max(w_dev_max / sigma_w))

        ## Measure temporal structure in the orbit-averaged HW series, not per-orbit dynamics.
        hw_centered = hw_mean - np.mean(hw_mean)
        hw_var = float(np.var(hw_mean))
        autocorr = {}
        if hw_var > 0:
            for lag in range(1, 6):
                ac = float(np.mean(hw_centered[lag:] * hw_centered[:-lag])) / hw_var
                autocorr[lag] = ac

        # Per-bit max deviation from 0.5 across all steps
        bit_dev     = np.abs(bit_density - 0.5)
        bit_max_dev = float(np.max(bit_dev))
        sigma_bit   = _math.sqrt(0.25 / ORBIT_SAMPLES)
        bit_z_max   = bit_max_dev / sigma_bit

        evt_baseline_bits = _math.sqrt(2 * _math.log(max(ORBIT_STEPS * 256, 2)))
        bit_z_ev    = bit_z_max - evt_baseline_bits

        # Worst bit across all steps
        worst_step, worst_bit_idx = np.unravel_index(np.argmax(bit_dev), bit_dev.shape)

        print(f"\n  Per-bit density analysis:")
        print(f"  Max |density - 0.5|  : {bit_max_dev:.6f}  "
              f"(bit {worst_bit_idx}, step {worst_step})")
        print(f"  Bit z_raw            : {bit_z_max:.2f}σ")
        print(f"  Bit z_ev             : {bit_z_ev:+.2f}σ  "
              f"(EVT baseline {evt_baseline_bits:.2f}σ)")

        print(f"\n  --- ORBIT MIXING RESULTS ---")
        print(f"  σ(HW mean)           : {sigma_hw:.4f}")
        print(f"  σ(word mean)         : {sigma_w:.4f}")
        print(f"  Max |HW - 128| / σ   : {hw_z_max:.2f}σ  (over {ORBIT_STEPS} steps)")
        print(f"  Max |word - 32| / σ  : {w_z_max:.2f}σ")

        print(f"\n  HW mean by step (first 16 and last 4):")
        print(f"  {'step':>5}  {'hw_mean':>8}  {'dev':>8}  {'z':>7}")
        print(f"  {'-' * 35}")
        show_steps = list(range(min(16, ORBIT_STEPS))) + list(range(max(0, ORBIT_STEPS - 4), ORBIT_STEPS))
        shown = set()
        for s in show_steps:
            if s in shown:
                continue
            shown.add(s)
            print(f"  {s:>5}  {hw_mean[s]:>8.4f}  {hw_dev[s]:>8.4f}  {hw_dev[s] / sigma_hw:>7.2f}σ")

        print(f"\n  Autocorrelation of HW series:")
        for lag, ac in autocorr.items():
            print(f"  lag {lag}  :  {ac:+.6f}")

        # EVT reference treating checkpoint statistics as independent tests.
        evt_baseline_orbit = _math.sqrt(2 * _math.log(max(ORBIT_STEPS, 2)))
        hw_z_ev = hw_z_max - evt_baseline_orbit
        w_z_ev = w_z_max - evt_baseline_orbit

        max_autocorr = max(abs(v) for v in autocorr.values()) if autocorr else 0.0

        # White-noise reference scale for sample autocorrelation.
        ac_sigma = 1.0 / _math.sqrt(max(ORBIT_STEPS, 1))
        ac_z = max_autocorr / ac_sigma

        # Diagnostic metrics only; deterministic orbit dependence prevents a simple iid H0.
        orbit_fail = False
        orbit_warn = False

        print(f"\n  [diagnostic] hw_z_ev={hw_z_ev:+.2f}σ  w_z_ev={w_z_ev:+.2f}σ  "
              f"bit_z_ev={bit_z_ev:+.2f}σ  (EVT assumes independence — informational only)")
        print(f"  [diagnostic] ac_z={ac_z:.2f}σ  "
              f"(H0 distribution not derived — autocorrelation informational only)")

        self.report["B32C"] = {
            "samples": ORBIT_SAMPLES,
            "steps": ORBIT_STEPS,
            "hw_z_max": hw_z_max,
            "w_z_max": w_z_max,
            "hw_z_ev": hw_z_ev,
            "w_z_ev": w_z_ev,
            "bit_z_max": bit_z_max,
            "bit_z_ev": bit_z_ev,
            "max_autocorr": max_autocorr,
            "autocorr": autocorr,
            "ac_sigma": ac_sigma,
            "ac_z": ac_z,
            "any_fail": orbit_fail,
            "any_warn": orbit_warn,
        }

        # ==========================
        # B32D — ORBIT FINGERPRINT RECURRENCE
        # ==========================
        # B32D searches each orbit for the first repeated fingerprint.
        # The H0 uses the identical recurrence algorithm with iid uniform fingerprints,
        # enabling a direct comparison of recurrence rates and gap distributions.

        CYCLE_SAMPLES = 2_000_000 # 4_000_000
        MAX_ORBIT = 8_192 # 65_536
        CYCLE_FP_BITS = 32 # 16
        H0_N_REP = 50_000

        print(f"\n[>>>] B32D: ORBIT FINGERPRINT RECURRENCE")
        print(f"[>>>] Samples: {CYCLE_SAMPLES:,} | Max orbit: {MAX_ORBIT:,} | FP bits: {CYCLE_FP_BITS}")

        rng_cycle = self.test_rng("B32", "workers")
        cycle_states = generate_states_u64(CYCLE_SAMPLES, rng_cycle)

        from lustro_rust import b32_cycle_signature
        cycle_out = b32_cycle_signature(
            np.ascontiguousarray(cycle_states, dtype=np.uint64),
            MAX_ORBIT, CYCLE_FP_BITS
        ).reshape(CYCLE_SAMPLES, 2)
        rec_steps = cycle_out[:, 0]
        rec_fps = cycle_out[:, 1]
        found_mask = rec_steps > 0
        n_found = int(np.sum(found_mask))
        n_no_recur = CYCLE_SAMPLES - n_found
        found_rate = n_found / CYCLE_SAMPLES

        from lustro_rust import b32_simulate_h0

        rng_h0_seed = self.test_rng("B32", "benchmark")
        h0_seed = int(rng_h0_seed.integers(2 ** 63))

        print(f"  Building H0 baseline ({H0_N_REP:,} simulated orbits, Rust)...")
        h0_raw = b32_simulate_h0(
            H0_N_REP, MAX_ORBIT, CYCLE_FP_BITS, h0_seed
        ).reshape(H0_N_REP, 3)

        h0_found_mask = h0_raw[:, 0] == 1
        h0_n_found = int(np.sum(h0_found_mask))
        h0_rate = h0_n_found / H0_N_REP
        h0_gaps = h0_raw[h0_found_mask, 1].astype(np.float64)
        h0_mean_gap = float(np.mean(h0_gaps)) if h0_n_found > 0 else 0.0

        # Use the combined variance of the engine and H0 estimates.
        var_engine = found_rate * (1.0 - found_rate) / CYCLE_SAMPLES
        var_h0 = h0_rate * (1.0 - h0_rate) / H0_N_REP
        rate_sigma = _math.sqrt(var_engine + var_h0)
        rate_z = (found_rate - h0_rate) / max(rate_sigma, 1e-12)

        FP_SPACE = 1 << CYCLE_FP_BITS
        fp_space_f = float(FP_SPACE)

        # Birthday-paradox expectation is a reference, not part of the verdict.
        e_first_recur = _math.sqrt(_math.pi * fp_space_f / 2.0)

        print(f"\n  --- H0 BASELINE (uniform iid fp(t), Rust, n={H0_N_REP:,}) ---")
        print(f"  FP space             : 2^{CYCLE_FP_BITS} = {FP_SPACE:,}")
        print(f"  H0 recur. rate       : {h0_rate:.6e}  (n={H0_N_REP:,} orbits)")
        print(f"  H0 mean gap          : {h0_mean_gap:.2f}  (conditional on recurrence)")
        print(f"  E[first recurrence]  : {e_first_recur:.2e} steps  "
              f"(birthday paradox reference, absolute t2 — not used in verdict)")

        print(f"\n  --- ORBIT FINGERPRINT RECURRENCE RESULTS ---")
        print(f"  States sampled       : {CYCLE_SAMPLES:,}")
        print(f"  Recurrences found    : {n_found:,}  ({found_rate:.4%})")
        print(f"  H0 expected rate     : {h0_rate:.4%}  (z={rate_z:+.2f}σ)")
        print(f"  No recurrence        : {n_no_recur:,}  (orbit > {MAX_ORBIT:,})")

        if n_found > 0:
            found_steps = rec_steps[found_mask].astype(np.float64)
            min_rec_dist = int(np.min(found_steps))
            max_rec_dist = int(np.max(found_steps))
            mean_rec_dist = float(np.mean(found_steps))
            median_rec_dist = float(np.median(found_steps))

            print(f"  Min recurrence dist  : {min_rec_dist:,}")
            print(f"  Max recurrence dist  : {max_rec_dist:,}")
            print(f"  Mean recurrence dist : {mean_rec_dist:.2f}  (conditional on recurrence found)")
            print(f"  Median recur. dist   : {median_rec_dist:.2f}")

            # Gap histogram — H0 baseline from simulation
            short_bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, MAX_ORBIT + 1]
            print(f"\n  Recurrence gap distribution  (H0 baseline from Rust simulation):")
            print(f"  {'range':<20}  {'obs':>8}  {'obs_frac':>10}  "
                  f"{'h0_frac':>10}  {'ratio':>7}")
            print(f"  {'-' * 62}")

            for lo, hi in zip(short_bins[:-1], short_bins[1:]):
                obs_mask = (found_steps >= lo) & (found_steps < hi)
                cnt = int(np.sum(obs_mask))
                obs_frac = cnt / CYCLE_SAMPLES

                if h0_n_found > 0:
                    h0_mask = (h0_gaps >= lo) & (h0_gaps < hi)
                    h0_frac = float(np.sum(h0_mask)) / H0_N_REP
                else:
                    h0_frac = 0.0

                ratio = (obs_frac / h0_frac) if h0_frac > 1e-12 else float('nan')
                label = f"[{lo}, {hi})"
                print(f"  {label:<20}  {cnt:>8,}  {obs_frac:>10.6f}  "
                      f"{h0_frac:>10.6f}  {ratio:>7.2f}")

            # Adjacent-step collisions — diagnostic only
            dist1_obs = int(np.sum(found_steps == 1))
            print(f"\n  Adjacent-step FP collisions (distance=1)  [diagnostic only]:")
            print(f"  Observed             : {dist1_obs:,}")

            unique_fps = int(np.unique(rec_fps[found_mask]).size)
            fp_collisions = n_found - unique_fps
            print(f"  FP collisions        : {fp_collisions:,}  "
                  f"(unique={unique_fps:,} / total={n_found:,})")

            rec_fail = rate_z > 5.0
            rec_warn = (not rec_fail) and rate_z > 3.0

        else:
            min_rec_dist = 0
            max_rec_dist = 0
            mean_rec_dist = 0.0
            median_rec_dist = 0.0
            fp_collisions = 0
            dist1_obs = 0
            rec_fail = False
            rec_warn = False

        if rec_fail:
            print(f"\n  [!!] FP recurrence anomaly vs H0 baseline  "
                  f"(rate_z={rate_z:+.2f}σ)")
        elif rec_warn:
            print(f"\n  [!]  mild FP recurrence signal  "
                  f"(rate_z={rate_z:+.2f}σ)")
        else:
            print(f"\n  [OK] FP recurrence consistent with H0 baseline")

        self.report["B32D"] = {
            "samples": CYCLE_SAMPLES,
            "max_orbit": MAX_ORBIT,
            "fp_bits": CYCLE_FP_BITS,
            "n_found": n_found,
            "found_rate": found_rate,
            "h0_rate": h0_rate,
            "h0_mean_gap": h0_mean_gap,
            "h0_n_rep": H0_N_REP,
            "rate_z": rate_z,
            "e_first_recur": e_first_recur,
            "min_rec_dist": min_rec_dist,
            "max_rec_dist": max_rec_dist,
            "mean_rec_dist": mean_rec_dist,
            "median_rec_dist": median_rec_dist,
            "dist1_obs": dist1_obs,
            "fp_collisions": fp_collisions,
            "any_fail": rec_fail,
            "any_warn": rec_warn,
        }

        # ==========================
        # B32E — STATE-SPACE PROFILE
        # ==========================
        # B32E profiles state-space occupancy across iteration checkpoints by
        # partitioning the 256-bit state into equal windows. Entropy loss is corrected
        # with Miller-Madow, and a Spearman trend tests whether compression grows with depth.

        import math as _math
        from scipy.stats import spearmanr

        B32E_SAMPLES = 10_000_000
        B32E_CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096] # , 8192, 16384, 32768, 65536]
        B32E_N_WINDOWS = 16 # 32
        B32E_N_CP = len(B32E_CHECKPOINTS)
        B32E_N_BUCKETS = 1 << (256 // B32E_N_WINDOWS)
        B32E_MAX_ENTROPY = float(256 // B32E_N_WINDOWS)

        # Miller-Madow correction for plug-in entropy:
        # E[H_hat] ≈ H - (K-1)/(2N ln 2).
        B32E_MM_BIAS = (B32E_N_BUCKETS - 1) / (2 * B32E_SAMPLES * _math.log(2))

        B32E_EVT_BASELINE = _math.sqrt(2 * _math.log(max(B32E_N_BUCKETS, 2)))

        # Occupancy reference under uniform sampling.
        B32E_LAMBDA = B32E_SAMPLES / B32E_N_BUCKETS
        B32E_EXPECTED_OCC = B32E_N_BUCKETS * (1.0 - _math.exp(-B32E_LAMBDA))

        print(f"\n[>>>] B32E: STATE-SPACE PROFILE")
        print(f"[>>>] Samples: {B32E_SAMPLES:,} | Checkpoints: {B32E_CHECKPOINTS}")
        print(f"[>>>] Windows: {B32E_N_WINDOWS} | Buckets/window: {B32E_N_BUCKETS:,}")
        print(f"[>>>] λ = N/K = {B32E_LAMBDA:.2f} | MM bias = {B32E_MM_BIAS:.6f}")

        rng_b32e = self.test_rng("B32", "subspace")
        b32e_states = generate_states_u64(B32E_SAMPLES, rng_b32e)

        from lustro_rust import b32e_state_space_profile
        raw = b32e_state_space_profile(
            np.ascontiguousarray(b32e_states, dtype=np.uint64),
            B32E_CHECKPOINTS,
            B32E_N_WINDOWS,
        )

        hist = raw.reshape(B32E_N_CP, B32E_N_WINDOWS, B32E_N_BUCKETS)
        hist_f64 = hist.astype(np.float64)

        # SANITY
        totals = hist_f64.sum(axis=2)
        sanity_ok = bool(np.all(totals == B32E_SAMPLES))
        if not sanity_ok:
            print(f"  [!!] SANITY FAILED — histogram totals inconsistent")
            bad = np.argwhere(totals != B32E_SAMPLES)
            for cp_idx, w_idx in bad[:5]:
                print(f"       cp={B32E_CHECKPOINTS[cp_idx]}  w={w_idx}  "
                      f"total={int(totals[cp_idx, w_idx])}")

        # ENTROPY — vectorized, log2(0) safe
        probs = hist_f64 / B32E_SAMPLES
        mask = probs > 0
        log_p = np.zeros_like(probs)
        log_p[mask] = np.log2(probs[mask])
        entropy = -np.sum(probs * log_p, axis=2)  # (N_CP, N_WINDOWS)
        entropy_loss = B32E_MAX_ENTROPY - entropy  # (N_CP, N_WINDOWS)

        mean_el_per_cp = entropy_loss.mean(axis=1)  # (N_CP,)
        std_el_per_cp = entropy_loss.std(axis=1)  # (N_CP,) — window-to-window variance

        # Delta entropy loss — direct compression signal
        delta_el = float(mean_el_per_cp[-1] - mean_el_per_cp[0])
        el_growth = float(mean_el_per_cp.max() - mean_el_per_cp.min())

        # MAX BUCKET Z + EVT
        max_buckets = hist_f64.max(axis=2)  # (N_CP, N_WINDOWS)
        max_bkt_z = (max_buckets - B32E_LAMBDA) / _math.sqrt(B32E_LAMBDA)
        max_bkt_zev = max_bkt_z - B32E_EVT_BASELINE

        global_max_occ = int(hist.max())
        global_max_occ_z = (global_max_occ - B32E_LAMBDA) / _math.sqrt(B32E_LAMBDA)
        global_max_occ_zev = global_max_occ_z - B32E_EVT_BASELINE

        # OCCUPANCY SPECTRUM — per (cp, window)
        # np.bincount requires 1D — loop justified
        spectrum_results = []
        for cp_idx in range(B32E_N_CP):
            cp_spectra = []
            for w_idx in range(B32E_N_WINDOWS):
                counts_1d = hist[cp_idx, w_idx]
                occupied = int(np.count_nonzero(counts_1d))

                # Tail percentiles
                p999 = float(np.percentile(counts_1d, 99.9))
                p9999 = float(np.percentile(counts_1d, 99.99))

                cp_spectra.append({
                    "occupied": occupied,
                    "p999": p999,
                    "p9999": p9999,
                    "max_occ": int(counts_1d.max()),
                })
            spectrum_results.append(cp_spectra)

        # Summary across all windows per checkpoint
        mean_occupied_per_cp = np.array([
            float(np.mean([s["occupied"] for s in spectrum_results[ci]]))
            for ci in range(B32E_N_CP)
        ])

        # TREND — Spearman
        log_cp = np.log2(np.array(B32E_CHECKPOINTS, dtype=np.float64))
        if mean_el_per_cp.std() > 0:
            spearman_r, spearman_p = spearmanr(log_cp, mean_el_per_cp)
        else:
            spearman_r, spearman_p = 0.0, 1.0

        # OUTPUT
        print(f"\n  --- MILLER-MADOW FINITE-SAMPLE BASELINE ---")
        print(f"  Theoretical MM bias  : {B32E_MM_BIAS:.6f}")
        print(f"  Observed range       : "
              f"{float(mean_el_per_cp.min()):.6f} .. {float(mean_el_per_cp.max()):.6f}")
        print(f"  Deviation from MM    : "
              f"{float(mean_el_per_cp.min() - B32E_MM_BIAS):+.2e} .. "
              f"{float(mean_el_per_cp.max() - B32E_MM_BIAS):+.2e}")

        print(f"\n  --- ENTROPY LOSS PER CHECKPOINT (mean ± std over {B32E_N_WINDOWS} windows) ---")
        print(f"  {'checkpoint':>12}  {'mean_el':>12}  {'std_el':>10}  "
              f"{'max_bkt_zev':>12}  {'occupied':>10}  verdict")
        print(f"  {'-' * 76}")

        cp_any_fail = False
        cp_any_warn = False
        cp_results = []

        for cp_idx, cp in enumerate(B32E_CHECKPOINTS):
            mel = float(mean_el_per_cp[cp_idx])
            sel = float(std_el_per_cp[cp_idx])
            mbzev = float(max_bkt_zev[cp_idx].max())
            occ = float(mean_occupied_per_cp[cp_idx])

            # Verdict relative to MM baseline — deviation from expected finite-sample bias
            mel_dev = mel - B32E_MM_BIAS

            if abs(mel_dev) > 0.005 or mbzev > 5.0:
                v = "[!!]"
                cp_any_fail = True
            elif abs(mel_dev) > 0.001 or mbzev > 3.0:
                v = "[! ]"
                cp_any_warn = True
            else:
                v = "[OK]"

            print(f"  {cp:>12}  {mel:>12.6f}  {sel:>10.2e}  "
                  f"{mbzev:>+12.2f}σ  {occ:>10.1f}  {v}")

            cp_results.append({
                "checkpoint": cp,
                "mean_el": mel,
                "std_el": sel,
                "mel_dev_mm": mel_dev,
                "max_bkt_zev": mbzev,
                "mean_occupied": occ,
            })

        print(f"\n  --- DELTA ENTROPY LOSS ---")
        print(f"  Δ entropy_loss (cp4096 - cp1) : {delta_el:+.6f}")
        print(f"  max - min across checkpoints  : {el_growth:.6f}")

        print(f"\n  --- GLOBAL MAX BUCKET ---")
        print(f"  Global max occupancy : {global_max_occ:,}  "
              f"(z={global_max_occ_z:+.2f}σ, EV-corr={global_max_occ_zev:+.2f}σ)")
        print(f"  Expected λ           : {B32E_LAMBDA:.2f}")
        print(f"  Expected occupied    : {B32E_EXPECTED_OCC:.1f} / {B32E_N_BUCKETS:,}")

        # Fano factor tests occupancy dispersion relative to the Poisson baseline;
        # values above 1 indicate over-dispersion and possible concentration.
        fano = hist_f64.var(axis=2) / (hist_f64.mean(axis=2) + 1e-15)  # (N_CP, N_WINDOWS)
        mean_fano_per_cp = fano.mean(axis=1)  # (N_CP,)
        max_fano_per_cp = fano.max(axis=1)  # (N_CP,)

        # Worst (cp, window) by entropy_loss
        worst_flat = np.argmax(max_bkt_zev)
        worst_cp_idx, worst_w_idx = np.unravel_index(worst_flat, max_bkt_zev.shape)
        worst_ref = spectrum_results[worst_cp_idx][worst_w_idx]
        worst_cp = B32E_CHECKPOINTS[worst_cp_idx]

        # Inspect the local occupancy spectrum around the Poisson mode.
        sigma_bkt = _math.sqrt(B32E_LAMBDA)
        k_lo = max(0, int(B32E_LAMBDA - sigma_bkt))
        k_hi = int(B32E_LAMBDA + sigma_bkt)

        worst_counts = hist[worst_cp_idx, worst_w_idx]
        worst_spec = np.bincount(worst_counts)

        print(f"\n  --- FANO FACTOR (Var/Mean, ideal = 1.0) ---")
        print(f"  {'checkpoint':>12}  {'mean_fano':>12}  {'max_fano':>12}  [diagnostic only]")
        print(f"  {'-' * 50}")
        for cp_idx, cp in enumerate(B32E_CHECKPOINTS):
            mf = float(mean_fano_per_cp[cp_idx])
            xf = float(max_fano_per_cp[cp_idx])
            print(f"  {cp:>12}  {mf:>12.6f}  {xf:>12.6f}")

        print(f"\n  --- OCCUPANCY SPECTRUM — worst (cp={worst_cp}, window={worst_w_idx}) ---")
        print(f"  Entropy loss         : {float(entropy_loss[worst_cp_idx, worst_w_idx]):.6f}")
        print(f"  Fano factor          : {float(fano[worst_cp_idx, worst_w_idx]):.6f}")
        print(f"  λ = {B32E_LAMBDA:.1f}  σ = {sigma_bkt:.1f}  spectrum around mode:")

        _rows = []
        for k in range(k_lo, k_hi + 1):
            obs_k = int(worst_spec[k]) if k < len(worst_spec) else 0
            gauss_exp = (B32E_N_BUCKETS *
                         _math.exp(-0.5 * ((k - B32E_LAMBDA) / sigma_bkt) ** 2) /
                         (_math.sqrt(2 * _math.pi) * sigma_bkt))
            z_k = (obs_k - gauss_exp) / max(_math.sqrt(gauss_exp), 1.0)
            _rows.append((k, obs_k, gauss_exp, z_k))

        def _fmt_row(k, obs_k, gauss_exp, z_k):
            sign = '+' if z_k >= 0 else ''
            z_str = f"{sign}{z_k:.2f}σ"
            return f"{k:>6}  {obs_k:>10,}  {gauss_exp:>9.1f}  {z_str:>8}"

        _COLS = 3
        _chunk = (len(_rows) + _COLS - 1) // _COLS
        _hdr = f"{'k':>6}  {'observed':>10}  {'gauss_exp':>9}  {'z':>8}"
        _cw = len(_hdr)
        print(f"  {_hdr}   {_hdr}   {_hdr}")
        print(f"  {'-' * _cw}   {'-' * _cw}   {'-' * _cw}")
        for i in range(_chunk):
            parts = []
            for c in range(_COLS):
                idx = c * _chunk + i
                if idx < len(_rows):
                    parts.append(_fmt_row(*_rows[idx]))
                else:
                    parts.append(" " * _cw)
            print(f"  {'   '.join(parts)}")

        print(f"  p99.9  : {worst_ref['p999']:.1f}  |  p99.99 : {worst_ref['p9999']:.1f}")

        print(f"\n  --- TREND ANALYSIS (Spearman: log2(cp) vs mean_entropy_loss) ---")
        print(f"  Spearman r           : {spearman_r:+.4f}")
        print(f"  p-value              : {spearman_p:.4f}")

        max_abs_dev = float(np.max(np.abs(mean_el_per_cp - B32E_MM_BIAS)))
        trend_fail = spearman_r > 0.85 and max_abs_dev > 0.005
        trend_warn = (not trend_fail) and spearman_r > 0.65 and max_abs_dev > 0.001

        if trend_fail:
            print(f"  [!!] monotonic concentration trend detected  "
                  f"(r={spearman_r:+.4f})")
        elif trend_warn:
            print(f"  [!]  weak concentration trend  "
                  f"(r={spearman_r:+.4f})")
        else:
            print(f"  [OK] no systematic entropy loss trend")

        # VERDICT
        any_fail_b32e = (not sanity_ok) or cp_any_fail or trend_fail
        any_warn_b32e = (not any_fail_b32e) and (cp_any_warn or trend_warn)

        print(f"\n  --- VERDICT ---")
        if cp_any_fail:
            print(f"  [!!] entropy loss / bucket anomaly above MM-corrected threshold")
        if trend_fail:
            print(f"  [!!] state-space compression trend confirmed")
        if cp_any_warn and not any_fail_b32e:
            print(f"  [!]  mild deviation above MM baseline")
        if trend_warn and not any_fail_b32e:
            print(f"  [!]  weak compression trend")
        if not any_fail_b32e and not any_warn_b32e:
            print(f"  [OK] entropy loss consistent with Miller-Madow finite-sample baseline")
            print(f"  [OK] no concentration trend detected")

        self.report["B32E"] = {
            "samples": B32E_SAMPLES,
            "checkpoints": B32E_CHECKPOINTS,
            "mm_bias": B32E_MM_BIAS,
            "sanity_ok": sanity_ok,
            "mean_el_per_cp": mean_el_per_cp.tolist(),
            "std_el_per_cp": std_el_per_cp.tolist(),
            "delta_el": delta_el,
            "el_growth": el_growth,
            "global_max_occ": global_max_occ,
            "global_max_occ_zev": global_max_occ_zev,
            "worst_bucket_zev": float(max_bkt_zev.max()),
            "worst_cp": int(worst_cp),
            "worst_window": int(worst_w_idx),
            "mean_fano_per_cp": mean_fano_per_cp.tolist(),
            "max_fano_per_cp": max_fano_per_cp.tolist(),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "trend_fail": trend_fail,
            "trend_warn": trend_warn,
            "cp_results": cp_results,
            "any_fail": any_fail_b32e,
            "any_warn": any_warn_b32e,
        }

        # ==========================
        # B32F — NEAR-NEIGHBOUR DISTANCE RETENTION
        # ==========================
        # B32F tracks whether a single-bit perturbation retains unusually small
        # Hamming distance during iteration. Per-checkpoint HD uses a Binomial(256, 0.5)
        # random-mixing reference; min-HD uses an iid order-statistics reference only,
        # since actual trajectory samples are correlated.

        import math as _math
        from scipy.stats import spearmanr, binom as scipy_binom

        B32F_SAMPLES = 102_400  # multiple of 256 — round-robin coverage
        B32F_MAX_STEPS = 4_096 # 65_536
        B32F_CHECKPOINTS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]  # , 8192, 16384, 32768, 65536]

        from lustro_rust import b32f_distance_retention
        HD_BINS = 257
        N_CP = len(B32F_CHECKPOINTS_LIST)
        HD_HIST_LEN = N_CP * HD_BINS
        OUT_LEN = HD_HIST_LEN + HD_BINS

        # Random-mixing reference: HD is approximately Binomial(256, 0.5) after mixing.
        H0_MU = 128.0
        H0_SIGMA = 8.0

        print(f"\n[>>>] B32F: NEAR-NEIGHBOUR DISTANCE RETENTION")
        print(f"[>>>] Pairs: {B32F_SAMPLES:,} | Max steps: {B32F_MAX_STEPS:,}")
        print(f"[>>>] Checkpoints: {B32F_CHECKPOINTS_LIST}")
        print(f"[>>>] H0 reference: HD ~ Binomial(256, 0.5)  μ={H0_MU}  σ={H0_SIGMA}")

        rng_b32f = self.test_rng("B32", "walsh")
        b32f_states = generate_states_u64(B32F_SAMPLES, rng_b32f)

        raw = b32f_distance_retention(
            np.ascontiguousarray(b32f_states, dtype=np.uint64),
            B32F_MAX_STEPS,
            B32F_CHECKPOINTS_LIST,
        )

        hd_hist = raw[:HD_HIST_LEN].reshape(N_CP, HD_BINS).astype(np.float64)
        min_hd_hist = raw[HD_HIST_LEN:].astype(np.float64)

        # SANITY
        row_sums = hd_hist.sum(axis=1)
        sanity_ok_f = bool(np.all(row_sums == B32F_SAMPLES))
        if not sanity_ok_f:
            print(f"  [!!] SANITY FAILED — checkpoint row sums inconsistent")
            for ci, s in enumerate(row_sums):
                if s != B32F_SAMPLES:
                    print(f"       cp={B32F_CHECKPOINTS_LIST[ci]}  sum={int(s)}")

        # HD STATISTICS PER CHECKPOINT
        hd_values = np.arange(HD_BINS, dtype=np.float64)  # 0..256

        mean_hd = (hd_hist * hd_values[None, :]).sum(axis=1) / B32F_SAMPLES
        var_hd = (hd_hist * (hd_values[None, :] - mean_hd[:, None]) ** 2).sum(axis=1) / B32F_SAMPLES
        std_hd = np.sqrt(var_hd)

        cdf = np.cumsum(hd_hist, axis=1) / B32F_SAMPLES

        def quantile_from_cdf(cdf_row, q):
            return int(np.searchsorted(cdf_row, q))

        p01 = np.array([quantile_from_cdf(cdf[ci], 0.01) for ci in range(N_CP)])
        p05 = np.array([quantile_from_cdf(cdf[ci], 0.05) for ci in range(N_CP)])
        median = np.array([quantile_from_cdf(cdf[ci], 0.50) for ci in range(N_CP)])
        p95 = np.array([quantile_from_cdf(cdf[ci], 0.95) for ci in range(N_CP)])
        p99 = np.array([quantile_from_cdf(cdf[ci], 0.99) for ci in range(N_CP)])

        # Tail probabilities — empirical
        p_lt_8 = hd_hist[:, :8].sum(axis=1) / B32F_SAMPLES
        p_lt_16 = hd_hist[:, :16].sum(axis=1) / B32F_SAMPLES
        p_lt_32 = hd_hist[:, :32].sum(axis=1) / B32F_SAMPLES
        p_lt_64 = hd_hist[:, :64].sum(axis=1) / B32F_SAMPLES
        p_lt_96 = hd_hist[:, :96].sum(axis=1) / B32F_SAMPLES
        p_lt_104 = hd_hist[:, :104].sum(axis=1) / B32F_SAMPLES
        p_lt_112 = hd_hist[:, :112].sum(axis=1) / B32F_SAMPLES

        # Compare lower tails against the Binomial(256, 0.5) reference.
        h0_p_lt_8 = float(scipy_binom.cdf(7, 256, 0.5))
        h0_p_lt_16 = float(scipy_binom.cdf(15, 256, 0.5))
        h0_p_lt_32 = float(scipy_binom.cdf(31, 256, 0.5))
        h0_p_lt_64 = float(scipy_binom.cdf(63, 256, 0.5))
        h0_p_lt_96 = float(scipy_binom.cdf(95, 256, 0.5))
        h0_p_lt_104 = float(scipy_binom.cdf(103, 256, 0.5))
        h0_p_lt_112 = float(scipy_binom.cdf(111, 256, 0.5))

        # TREND — Spearman: log2(cp) vs mean_hd
        log_cp = np.log2(np.array(B32F_CHECKPOINTS_LIST, dtype=np.float64))
        if mean_hd.std() > 0:
            spearman_r_f, spearman_p_f = spearmanr(log_cp, mean_hd)
        else:
            spearman_r_f, spearman_p_f = 0.0, 1.0


        # min_HD ANALYSIS
        min_cdf = np.cumsum(min_hd_hist) / B32F_SAMPLES

        # Empirical P(min_HD <= k)
        p_min_8 = float(min_cdf[8])
        p_min_16 = float(min_cdf[16])
        p_min_32 = float(min_cdf[32])
        p_min_64 = float(min_cdf[64])

        # IID order-statistics reference for the trajectory minimum:
        # P(min_t HD_t <= k) = 1 - (1 - F_H0(k))^T.
        # This is not an exact bound because orbit samples are correlated.
        T = B32F_MAX_STEPS

        def min_hd_bound(k):
            p_single = float(scipy_binom.cdf(k, 256, 0.5))
            return 1.0 - (1.0 - p_single) ** T

        h0_min_8 = min_hd_bound(8)
        h0_min_16 = min_hd_bound(16)
        h0_min_32 = min_hd_bound(32)
        h0_min_64 = min_hd_bound(64)

        # OUTPUT
        # z-score of mean_hd per checkpoint — diagnostic only
        sigma_mean = H0_SIGMA / _math.sqrt(B32F_SAMPLES)

        print(f"\n  --- HD PER CHECKPOINT ---")
        print(f"  H0 analytic tails (Binomial(256,0.5), approximate baseline):")
        print(f"  P(HD<8)={h0_p_lt_8:.2e}  P(HD<16)={h0_p_lt_16:.2e}"
              f"  P(HD<32)={h0_p_lt_32:.2e}  P(HD<64)={h0_p_lt_64:.2e}")
        print(f"  P(HD<96)={h0_p_lt_96:.4f}  P(HD<104)={h0_p_lt_104:.4f}"
              f"  P(HD<112)={h0_p_lt_112:.4f}")
        print()
        print(f"  {'cp':>6}  {'mean':>7}  {'std':>6}  {'p01':>4}  {'p05':>4}  "
              f"{'med':>4}  {'p95':>4}  {'p99':>4}  "
              f"{'P<96':>10}  {'P<104':>10}  {'P<112':>10}  {'z_mean':>8}")
        print(f"  {'-' * 96}")

        cp_results_f = []
        for ci, cp in enumerate(B32F_CHECKPOINTS_LIST):
            z_m = (float(mean_hd[ci]) - H0_MU) / sigma_mean
            print(f"  {cp:>6}  {mean_hd[ci]:>7.3f}  {std_hd[ci]:>6.3f}"
                  f"  {p01[ci]:>4}  {p05[ci]:>4}  {median[ci]:>4}"
                  f"  {p95[ci]:>4}  {p99[ci]:>4}"
                  f"  {p_lt_96[ci]:>10.4f}  {p_lt_104[ci]:>10.4f}"
                  f"  {p_lt_112[ci]:>10.4f}  {z_m:>+8.2f}σ")
            cp_results_f.append({
                "checkpoint": cp,
                "mean_hd": float(mean_hd[ci]),
                "std_hd": float(std_hd[ci]),
                "p01": int(p01[ci]), "p05": int(p05[ci]),
                "median": int(median[ci]),
                "p95": int(p95[ci]), "p99": int(p99[ci]),
                "p_lt_8": float(p_lt_8[ci]),
                "p_lt_16": float(p_lt_16[ci]),
                "p_lt_32": float(p_lt_32[ci]),
                "p_lt_64":  float(p_lt_64[ci]),
                "p_lt_96":  float(p_lt_96[ci]),
                "p_lt_104": float(p_lt_104[ci]),
                "p_lt_112": float(p_lt_112[ci]),
                "z_mean": float(z_m),
            })

        print(f"\n  --- min_HD ACROSS FULL TRAJECTORY (T={T}) ---")
        print("Independent-sample reference (iid order-statistics model)")
        print(f"  H0 P(min<=8)={h0_min_8:.2e}  P(min<=16)={h0_min_16:.2e}"
              f"  P(min<=32)={h0_min_32:.2e}  P(min<=64)={h0_min_64:.2e}")
        print(f"\n  Observed:")
        print(f"  P(min_HD<=8) ={p_min_8 :.2e}  "
              f"P(min_HD<=16)={p_min_16:.2e}  "
              f"P(min_HD<=32)={p_min_32:.2e}  "
              f"P(min_HD<=64)={p_min_64:.2e}")

        print(f"\n  --- TREND (Spearman: log2(cp) vs mean_HD) ---")
        print(f"  Spearman r = {spearman_r_f:+.4f}  p = {spearman_p_f:.4f}")
        if spearman_r_f < 0:
            print(f"  [note] negative r — trajectories tend to converge over time")

        # VERDICT
        # Flag convergence when mean HD trends downward strongly and falls below 120.
        trend_fail_f = spearman_r_f < -0.85 and float(mean_hd[-1]) < 120.0
        trend_warn_f = False  # diagnostic only — 12 points insufficient for reliable verdict

        # Treat persistent mass in the extreme lower HD tail as a structural signal.
        # Expected extreme-tail count under the Binomial H0.
        expected_tail_32 = B32F_SAMPLES * N_CP * h0_p_lt_32
        tail_count_32 = float(hd_hist[:, :32].sum())
        # Require a substantial excess over the H0 expectation to avoid false alarms
        # when the sample size is large enough for the expected tail count to be non-negligible.
        from scipy.stats import poisson as _poisson
        # Evaluate the upper-tail probability for the observed count under the H0 Poisson model.
        if tail_count_32 > 0 and expected_tail_32 > 0:
            tail_p = float(_poisson.sf(int(tail_count_32) - 1, expected_tail_32))
        elif tail_count_32 > 0:
            tail_p = 0.0  # expected ~ 0 but observed > 0 — extreme
        else:
            tail_p = 1.0
        tail_fail_f = tail_p < 1e-6
        # P(min_HD <= 16) — compared to iid bound; actual trajectories are correlated
        # so empirical value may exceed iid bound without structural anomaly
        min_tail_warn_f = p_min_16 > h0_min_16 * 10.0  # 10x excess over iid reference

        any_fail_b32f = (not sanity_ok_f) or trend_fail_f or tail_fail_f
        any_warn_b32f = (not any_fail_b32f) and min_tail_warn_f

        print(f"\n  --- VERDICT ---")
        if not sanity_ok:
            print(f"  [!!] histogram sanity failure")
        if trend_fail_f:
            print(f"  [!!] long-term trajectory convergence detected"
                  f"  (r={spearman_r_f:+.4f}, mean_hd[-1]={float(mean_hd[-1]):.3f})")
        if tail_fail_f:
            print(f"  [!!] anomalous close-orbit population detected"
                  f"  (tail_count={int(tail_count_32)}, p={tail_p:.2e})")
        if min_tail_warn_f and not any_fail_b32f:
            print(f"  [!]  min_HD tail excess vs iid reference"
                  f"  (P(min<=16)={p_min_16:.2e} vs H0={h0_min_16:.2e})")
        if not any_fail_b32f and not any_warn_b32f:
            print(f"  [OK] no trajectory convergence detected")
            print(f"  [OK] close-orbit population consistent with H0 reference")

        self.report["B32F"] = {
            "samples": B32F_SAMPLES,
            "max_steps": B32F_MAX_STEPS,
            "sanity_ok": sanity_ok_f,
            "spearman_r": float(spearman_r_f),
            "spearman_p": float(spearman_p_f),
            "trend_fail": trend_fail_f,
            "tail_fail": tail_fail_f,
            "tail_count_32": float(tail_count_32),
            "tail_p": float(tail_p),
            "expected_tail_32": float(expected_tail_32),
            "min_tail_warn": min_tail_warn_f,
            "p_min_8": p_min_8,
            "p_min_16": p_min_16,
            "p_min_32": p_min_32,
            "p_min_64": p_min_64,
            "h0_min_8": h0_min_8,
            "h0_min_16": h0_min_16,
            "h0_min_32": h0_min_32,
            "h0_min_64": h0_min_64,
            "h0_p_lt_8": h0_p_lt_8,
            "h0_p_lt_16": h0_p_lt_16,
            "h0_p_lt_32": h0_p_lt_32,
            "h0_p_lt_64": h0_p_lt_64,
            "h0_p_lt_96": h0_p_lt_96,
            "h0_p_lt_104": h0_p_lt_104,
            "h0_p_lt_112": h0_p_lt_112,
            "cp_results": cp_results_f,
            "any_fail": any_fail_b32f,
            "any_warn": any_warn_b32f,
        }

# ==========================
# B51 — ALGEBRAIC DEGREE TEST
# ==========================
# B51 tests algebraic degree via cube sums (Möbius derivatives).
# EXACT evaluates a single output bit over the full affine cube;
# PROB tests the full output vector across randomized affine cubes.

    def _run_b51(self, pool):

        # CONFIG
        MIN_DEGREE_EXACT = 17
        MAX_DEGREE_EXACT = 24
        MAX_DEGREE_PROB = 31
        DEGREES_EXACT = list(range(MIN_DEGREE_EXACT, MAX_DEGREE_EXACT + 1))
        DEGREES_PROB = list(range(MAX_DEGREE_EXACT + 1, MAX_DEGREE_PROB + 1))
        EXACT_TRIALS     = 64
        CUBES_PER_DEGREE = 16
        CUBES_PER_TASK   = 8
        N_TASKS          = max(1, CUBES_PER_DEGREE // CUBES_PER_TASK)
        BATCH_STATES     = 65_536

        print(f"\n[>>>] B51: ALGEBRAIC DEGREE TEST")
        print(f"[>>>] EXACT  — degrees: {MIN_DEGREE_EXACT}..{MAX_DEGREE_EXACT}"
              f"  | trials per degree: {EXACT_TRIALS}")
        print(f"[>>>] PROB   — degrees: {DEGREES_PROB[0]}..{DEGREES_PROB[-1]}"
              f"  | cubes: {CUBES_PER_DEGREE}"
              f"  ({N_TASKS} tasks × {CUBES_PER_TASK} cubes)")
        print(f"[>>>] Batch states: {BATCH_STATES:,}")
        print(f"[>>>] PROB tests full 256-bit output vector: nonzero = at least one"
              f" output bit at degree d")

        rng = self.test_rng("B51", "states")

        # HELPER — random base_state i variable_bits
        def random_base_and_bits(degree, rng_inst):
            base = rng_inst.integers(0, 2**64, size=4, dtype=np.uint64)
            all_bits = rng_inst.choice(256, size=degree, replace=False)
            variable_bits = [int(b) for b in all_bits]
            output_bit    = int(rng_inst.integers(0, 256))
            return base, variable_bits, output_bit


        # PHASE 1 — EXACT
        print(f"\n  [1/2] exact degree test...")
        print(f"  {'degree':>8}  {'nonzero_trials':>15}  {'total_trials':>13}"
              f"  {'result':>8}  verdict")
        print(f"  {'-' * 62}")

        # Under the random-output baseline, the 64-trial nonzero count is Binomial(64, 0.5).
        # Strong lower-tail depletion is interpreted as evidence of derivative collapse;
        # B51 is not intended to detect small distributional bias.
        EXACT_LOW_FAIL = 8

        exact_results = {}
        any_fail_exact = False
        first_exact_anomaly = None

        for degree in DEGREES_EXACT:
            args_list = []
            for _ in range(EXACT_TRIALS):
                base, vbits, obit = random_base_and_bits(degree, rng)
                args_list.append((
                    np.ascontiguousarray(base),
                    vbits,
                    obit,
                    BATCH_STATES,
                ))

            results_trial = list(
                pool.imap(worker_b51_exact, args_list, chunksize=1)
            )

            nonzero = sum(1 for r in results_trial if r)
            total   = len(results_trial)

            if nonzero < EXACT_LOW_FAIL:
                verdict = "[!!]"
                any_fail_exact = True
                if first_exact_anomaly is None:
                    first_exact_anomaly = degree
            else:
                verdict = "[OK]"

            exact_results[degree] = {
                "nonzero": nonzero,
                "total":   total,
                "verdict": verdict,
            }

            print(f"  {degree:>8}  {nonzero:>15}  {total:>13}"
                  f"  {'nonzero' if nonzero > 0 else 'ALL ZERO':>8}  {verdict}")

        # PHASE 2 — PROB
        print(f"\n  [2/2] probabilistic degree test...")
        print(f"  {'degree':>8}  {'nonzero_cubes':>14}  {'cubes':>6}"
              f"  {'result':>8}  verdict")
        print(f"  {'-' * 55}")

        prob_results = {}
        any_fail_prob = False
        first_prob_anomaly = None

        for degree in DEGREES_PROB:
            args_list = []
            for _ in range(N_TASKS):
                base, vbits, _ = random_base_and_bits(degree, rng)
                seed_val = int(rng.integers(1, 2 ** 63))
                args_list.append((
                    np.ascontiguousarray(base),
                    vbits,
                    np.uint32(CUBES_PER_TASK),
                    BATCH_STATES,
                    np.uint64(seed_val),
                ))

            results_trial = list(
                pool.imap(worker_b51_prob, args_list, chunksize=1)
            )

            nonzero = int(sum(results_trial))
            total = N_TASKS * CUBES_PER_TASK

            if nonzero < total:
                zero_count = total - nonzero
                verdict = "[!!]"
                any_fail_prob = True
                if first_prob_anomaly is None:
                    first_prob_anomaly = degree
                print(f"  [!!] {zero_count} zero cube(s) at degree {degree}"
                      f" — algebraic degeneracy")
            else:
                verdict = "[OK]"

            prob_results[degree] = {
                "nonzero": nonzero,
                "total":   total,
                "verdict": verdict,
            }

            print(f"  {degree:>8}  {nonzero:>14}  {total:>6}"
                  f"  {'nonzero' if nonzero > 0 else 'ALL ZERO':>8}  {verdict}")

        # VERDICT
        any_fail = any_fail_exact or any_fail_prob

        print(f"\n  --- VERDICT ---")

        if any_fail_exact:
            print(f"  [!!] exact phase anomaly at degree {first_exact_anomaly}"
                  f" (lower-tail, B(32,0.5) reference)")
            print(f"  [!!] ALGEBRAIC DEGENERACY SIGNAL DETECTED")

        if any_fail_prob:
            print(f"  [!!] zero cube detected in PROB phase at degree"
                  f" {first_prob_anomaly}")
            print(f"  [!!] ALGEBRAIC DEGENERACY SIGNAL DETECTED")

        if not any_fail:
            print(f"  [OK] at least one output component has a nonzero"
                  f" degree-{MAX_DEGREE_PROB} cube sum")
            print(f"  [OK] no algebraic degeneracy detected")

        self.report["B51"] = {
            "max_degree_exact": MAX_DEGREE_EXACT,
            "max_degree_prob": MAX_DEGREE_PROB,
            "first_exact_anomaly": first_exact_anomaly,
            "first_prob_anomaly": first_prob_anomaly,
            "any_fail": any_fail,
            "exact": exact_results,
            "prob": prob_results,
        }

# ==========================
# B53 — LINEAR CORRELATION / WALSH SPECTRUM TEST
# ==========================
# B53 tests linear correlation through Walsh-Hadamard sums.
# For each input/output mask pair, Z = S / sqrt(N) is approximately N(0,1) under H0.
# Random and single-bit mask families are compared with an independent MC baseline
# to calibrate EVT, Anderson-Darling, KS, and tail-count diagnostics.

    def _run_b53(self, pool):
        import math
        import numpy as np

        LOCAL_SAMPLES = 15_000_000

        # Use independent state samples per mask batch to reduce cross-mask dependence,
        # improving the independence approximation used by EVT and Anderson-Darling.
        MASKS_PER_DRAW_RAND = 64
        TOTAL_MASKS_RAND = 8192
        N_MASKS_RAND = MASKS_PER_DRAW_RAND
        N_BATCHES_RAND = TOTAL_MASKS_RAND // MASKS_PER_DRAW_RAND

        MASKS_PER_DRAW_SBIT = 32
        TOTAL_MASKS_SBIT = 4096
        N_MASKS_SBIT = MASKS_PER_DRAW_SBIT
        N_BATCHES_SBIT = TOTAL_MASKS_SBIT // MASKS_PER_DRAW_SBIT

        eff_rand = N_MASKS_RAND * N_BATCHES_RAND
        eff_sbit = N_MASKS_SBIT * N_BATCHES_SBIT
        sigma0   = 1.0 / math.sqrt(LOCAL_SAMPLES)

        print(f"\n[>>>] B53: LINEAR CORRELATION / WALSH SPECTRUM TEST")
        print(f"[>>>] Samples / batch     : {LOCAL_SAMPLES:,}")
        print(f"[>>>] σ_single (1/√N)     : {sigma0:.6f}")
        print(f"[>>>] [A] Random masks    : "
              f"{N_MASKS_RAND} × {N_BATCHES_RAND} = {eff_rand:,} trials")
        print(f"[>>>] [B] Single-bit masks: "
              f"{N_MASKS_SBIT} × {N_BATCHES_SBIT} = {eff_sbit:,} trials")

        _base_states = self.test_rng("B53", "states")
        _base_masks  = self.test_rng("B53", "masks")
        _base_mc     = self.test_rng("B53", "mc")
        _base_perm   = self.test_rng("B53", "subspace")  # seed for MC shuffle

        def _child(base_rng):
            return np.random.default_rng(int(base_rng.integers(2**63)))

        rng_states_rand   = _child(_base_states)
        rng_states_sbit   = _child(_base_states)
        rng_masks_mc_rand = _child(_base_masks)
        rng_masks_eng_rand= _child(_base_masks)
        rng_sbit_mc       = _child(_base_masks)
        rng_sbit_eng      = _child(_base_masks)
        rng_mc_rand_st    = _child(_base_mc)
        rng_mc_sbit_st    = _child(_base_mc)
        rng_mc_rand_perm  = _child(_base_perm)
        rng_mc_sbit_perm  = _child(_base_perm)

        # Phase 1 — MC baseline, random masks
        print(f"\n  [1/4] MC baseline — random masks...")

        def batch_gen_mc_rand():
            for _ in range(N_BATCHES_RAND):
                seed_in   = int(rng_mc_rand_st.integers(2**63))
                seed_out  = int(rng_mc_rand_perm.integers(2**63))
                seed_mask = int(rng_masks_mc_rand.integers(2**63))
                yield (LOCAL_SAMPLES, N_MASKS_RAND, "random", seed_in, seed_out, seed_mask)

        mc_rand_sums = []
        for sums in self.pool_map(pool, worker_b53_walsh_mc, batch_gen_mc_rand()):
            mc_rand_sums.append(sums)

        mc_rand_flat = np.concatenate(mc_rand_sums)
        mc_rand_spec = _b53_spectrum_analysis(mc_rand_flat, LOCAL_SAMPLES)
        mc_rand_evt = _b53_evt_thresholds(N_MASKS_RAND * N_BATCHES_RAND)
        _b53_print_spectrum("MC baseline — random masks", mc_rand_spec,
                            mc_evt=mc_rand_evt)

        # Phase 2 — engine, random masks
        print(f"\n  [2/4] Engine — random masks...")

        def batch_gen_rand():
            for _ in range(N_BATCHES_RAND):
                seed_states = int(rng_states_rand.integers(2**63))
                seed_mask   = int(rng_masks_eng_rand.integers(2**63))
                yield (LOCAL_SAMPLES, N_MASKS_RAND, "random", seed_states, seed_mask)

        eng_rand_sums = []
        for sums in self.pool_map(pool, worker_b53_walsh, batch_gen_rand()):
            eng_rand_sums.append(sums)

        eng_rand_flat = np.concatenate(eng_rand_sums)
        eng_rand_spec = _b53_spectrum_analysis(eng_rand_flat, LOCAL_SAMPLES)
        _b53_print_spectrum("Engine — random masks", eng_rand_spec,
                            mc_spec=mc_rand_spec, mc_evt=mc_rand_evt)

        # Phase 3 — MC baseline, single-bit masks
        print(f"\n  [3/4] MC baseline — single-bit masks...")

        def batch_gen_mc_sbit():
            for _ in range(N_BATCHES_SBIT):
                seed_in   = int(rng_mc_sbit_st.integers(2**63))
                seed_out  = int(rng_mc_sbit_perm.integers(2**63))
                seed_mask = int(rng_sbit_mc.integers(2**63))
                yield (LOCAL_SAMPLES, N_MASKS_SBIT, "single_bit", seed_in, seed_out, seed_mask)

        mc_sbit_sums = []
        for sums in self.pool_map(pool, worker_b53_walsh_mc, batch_gen_mc_sbit()):
            mc_sbit_sums.append(sums)

        mc_sbit_flat = np.concatenate(mc_sbit_sums)
        mc_sbit_spec = _b53_spectrum_analysis(mc_sbit_flat, LOCAL_SAMPLES)
        mc_sbit_evt = _b53_evt_thresholds(N_MASKS_SBIT * N_BATCHES_SBIT)
        _b53_print_spectrum("MC baseline — single-bit masks", mc_sbit_spec,
                            mc_evt=mc_sbit_evt)


        # Phase 4 — engine, single-bit masks
        print(f"\n  [4/4] Engine — single-bit masks...")

        def batch_gen_sbit():
            for _ in range(N_BATCHES_SBIT):
                seed_states = int(rng_states_sbit.integers(2**63))
                seed_mask   = int(rng_sbit_eng.integers(2**63))
                yield (LOCAL_SAMPLES, N_MASKS_SBIT, "single_bit", seed_states, seed_mask)

        eng_sbit_sums = []
        for sums in self.pool_map(pool, worker_b53_walsh, batch_gen_sbit()):
            eng_sbit_sums.append(sums)

        eng_sbit_flat = np.concatenate(eng_sbit_sums)
        eng_sbit_spec = _b53_spectrum_analysis(eng_sbit_flat, LOCAL_SAMPLES)
        _b53_print_spectrum("Engine — single-bit masks", eng_sbit_spec,
                            mc_spec=mc_sbit_spec, mc_evt=mc_sbit_evt)

        # TABLE
        print(f"\n  --- RESULTS SUMMARY ---")
        print(f"  {'branch':<28}  {'max|Z|':>8}  {'AD stat':>8}  "
              f"{'tail4σ obs':>10}  {'tail4σ exp':>10}")
        print(f"  {'-' * 72}")

        for label, spec in [
            ("MC — random masks",    mc_rand_spec),
            ("Engine — random masks", eng_rand_spec),
            ("MC — single-bit",      mc_sbit_spec),
            ("Engine — single-bit",  eng_sbit_spec),
        ]:
            t4o, t4e = spec["tails"][4.0]
            print(f"  {label:<28}  {spec['max_abs_z']:8.4f}  "
                  f"{spec['ad_stat']:8.4f}  {t4o:10d}  {t4e:10.2f}")

        # SUMMARY
        fail_rand = _b53_spec_fails(eng_rand_spec, mc_evt=mc_rand_evt)
        warn_rand = (not fail_rand) and _b53_spec_warns(eng_rand_spec, mc_evt=mc_rand_evt)
        fail_sbit = _b53_spec_fails(eng_sbit_spec, mc_evt=mc_sbit_evt)
        warn_sbit = (not fail_sbit) and _b53_spec_warns(eng_sbit_spec, mc_evt=mc_sbit_evt)

        any_fail = fail_rand or fail_sbit
        any_warn = (not any_fail) and (warn_rand or warn_sbit)

        print(f"\n  --- VERDICT ---")
        print(f"  [A] Random masks    : {'FAIL' if fail_rand else 'WARN' if warn_rand else 'OK'}")
        print(f"  [B] Single-bit masks: {'FAIL' if fail_sbit else 'WARN' if warn_sbit else 'OK'}")

        if fail_rand:
            print(f"  [!!] Leakage detected — random mask spectrum (global)")
        if fail_sbit:
            print(f"  [!!] Leakage detected — single-bit (LAT-level, interpretable)")
        if warn_rand:
            print(f"  [!]  Weak anomaly — random mask spectrum")
        if warn_sbit:
            print(f"  [!]  Weak anomaly — single-bit spectrum")
        if not any_fail and not any_warn:
            print(f"  [OK] Walsh spectrum consistent with random-permutation baseline")
            print(f"  [OK] No linear bias above detection threshold")

        self.report["B53"] = {
            "samples": LOCAL_SAMPLES,
            "random_masks": {"n_masks": N_MASKS_RAND, "n_batches": N_BATCHES_RAND,
                             "mc_spec": mc_rand_spec, "eng_spec": eng_rand_spec,
                             "evt_p99": mc_rand_evt[0.99],
                             "evt_p999": mc_rand_evt[0.999],
                             "max_abs_z": eng_rand_spec["max_abs_z"]},
            "singlebit_masks": {"n_masks": N_MASKS_SBIT, "n_batches": N_BATCHES_SBIT,
                                "mc_spec": mc_sbit_spec, "eng_spec": eng_sbit_spec,
                                "evt_p99": mc_sbit_evt[0.99],
                                "evt_p999": mc_sbit_evt[0.999],
                                "max_abs_z": eng_sbit_spec["max_abs_z"]},
            "any_fail": any_fail,
            "any_warn": any_warn,
        }

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("\n[?] Chose your test option:")
    print("     FULL AUDIT - Type 0, ALL, FULL")
    print("     [ID] Specific Test (B2, B13, B16, B22, B32, B51, B53)")

    try:
        choice = input("\nYour choice > ").strip().upper()
        if not choice:
            choice = "0"
    except EOFError:
        choice = "0"

    suite = LustroAuditSuite()
    try:
        if choice in ["0", "FULL", "ALL"]:
            suite.run_all()
        else:
            suite.run_test(choice)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        suite.shutdown()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()