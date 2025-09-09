## oneAPI Hash Cracker: Parallel Meet-in-the-Middle (MITM)

**Overview:**
This project recovers an input array of `L` character indices (0–36) that produces a specific 4-element hash using a custom hash function. The implementation uses C++ and oneAPI (SYCL) for parallelism.

---

### Problem Statement
- **Goal:** Given a 4-element hash, find an input array of length `L` (indices 0–36) that hashes to this value.
- **Parallelism:** Use SYCL to exploit device parallelism.
- **Input Length:** `L` is chosen at compile time to ensure a high probability (e.g., 99%) of finding a solution.

---

### Hash Function (see `include/unhash/hash.hpp`)
- **Input:** `std::array<unsigned char, L>`
- **Computation:** For each of 4 segments, sum precomputed values from a table `T[4][L][37]` (indexed by segment, position, and character), then reduce modulo `MA = M * A`.
  - `M` is the segment's original modulus (e.g., 10000).
  - `A` is a scaling factor (may be floating-point) to convert floating-point hash contributions to integers. `MA` must be an integer.
- **Output:** For each segment, output is `sum / A` (may be floating-point division), yielding a 4-element array in `[0, M)`.

**Code Template:**
```cpp
std::array<uint32_t, 4> hashify(const std::array<unsigned char, L>& input) {
    std::array<uint64_t, 4> sum = {0, 0, 0, 0};
    for (size_t i = 0; i < L; ++i)
        for (size_t s = 0; s < 4; ++s)
            sum[s] = (sum[s] + T[s][i][input[i]]) % MA;
    std::array<uint32_t, 4> hash;
    for (size_t s = 0; s < 4; ++s)
        hash[s] = static_cast<uint32_t>(sum[s] / A); // A may be floating-point
    return hash;
}
```

---

### Precomputed Table T
- **Purpose:** Store all possible contributions for each segment, position, and character.
- **How to generate:**
```cpp
for (size_t s = 0; s < 4; ++s)
    for (size_t i = 0; i < L; ++i)
        for (size_t c = 0; c < 37; ++c)
            T[s][i][c] = contribution(s, i, c); // See hash.hpp
```

---

### Bucketed Sums Optimization
- **Purpose:** Reduce memory and the number of probes in the search.
- **Method:**
  - Store Phase 1 partial sums quantized into buckets of size `k` (power of two, e.g., 256 or 1024).
  - Instead of storing every `partial_sum % MA`, store `partial_sum / k` (integer division).
  - In Phase 2, only probe those `k_s` values that are multiples of `k` (i.e., `k_s = 0, k, 2k, ..., A-k`), reducing probes per segment from `A` to `A/k`.
  - This introduces false negatives (some solutions may be missed), but is acceptable if you only need any solution and can increase `L` to compensate.
  - For best performance, use a power of two for `k` (enables fast bitwise division/modulo). `A` may be floating-point, but `MA` must be an integer.

**Code Template:**
```cpp
// Compute bucketed sum for Phase 1
std::array<uint32_t, 4> bucketed_sum;
for (size_t s = 0; s < 4; ++s)
    bucketed_sum[s] = partial_sum[s] / k; // k must be a power of two
```

**Note:**
- If you choose `A` so that `MA = M * A` is a power of two, you can use bitwise operations for modulo and division, which may make bucketing unnecessary for performance. However, bucketing can still reduce memory and probe count if the search space is very large.
- If memory and probe count are not a bottleneck, and `MA` is a power of two, you may skip bucketing and store all possible sums directly.

---

### Algorithm Structure
1. **Split Input:**
   - Divide `L` into `L1` and `L2` (e.g., `L1 = L/2`).
   - Choose `L1` based on available device memory for Phase 1 data (bucketed sum vectors and prefixes) and the filter.
2. **Phase 1 (Prefix Generation):**
   - Generate all `NCHARS^L1` prefixes and their partial hash sums in parallel.
   - Store `Phase1Entry {bucketed_sum_vector, input_prefix}` in device memory, where `bucketed_sum_vector[i] = partial_sum[i] / k`.
   - Build an Approximate Membership Query (AMQ) filter (e.g., Binary Fuse Filter) from all unique `bucketed_sum_vector`s. The filter resides in device memory.
3. **Phase 2 (Suffix Search):**
   - For all `NCHARS^L2` suffixes, compute their partial hash sums in parallel.
   - For each suffix and each `k_s` in `{0, k, 2k, ..., A-k}` (i.e., `A/k` probes), calculate the required Phase 1 bucketed sum to match the target hash:
     - For each segment `i`:
       - `target_sum = target_hash[i] * A + k_s`
       - `target_S1_sum = (target_sum + MA - S2_sum_vector[i]) % MA`
       - `target_S1_bucket = target_S1_sum / k`
   - For each candidate bucketed sum vector:
     - Query the AMQ filter (O(1) check).
     - If the filter returns true, perform an exact lookup in Phase 1 data to retrieve the corresponding prefix.
     - If a valid prefix is found, atomically store the solution and signal early termination.
4. **Early Termination:**
   - Kernels periodically check an atomic flag and terminate if a solution is found.

---

### Data Structures
- `Phase1Entry`: `{std::array<uint_fast32_t, 4> bucketed_sum, std::array<unsigned char, L1> input_prefix}`
- `BinaryFuseFilter`: AMQ filter built from Phase 1 bucketed sums (device memory)
- `T[4][L][37]`: Precomputed lookup table (device memory)
- Atomic flag for early termination (device memory)
- Solution buffer: `std::array<unsigned char, L>` (device memory, copied to host on success)

---

### Implementation Checklist
- [ ] Precompute and upload lookup table `T` once.
- [ ] Dynamically select `L1` to fit both Phase 1 entries and the AMQ filter in device memory.
- [ ] Use a power of two for `k` for fast bucketed sums, unless you skip bucketing because `MA` is a power of two and memory/probe count is not a bottleneck.
- [ ] Use AMQ (Binary Fuse Filter) for O(1) lookups in Phase 2.
- [ ] Use early termination via atomic flag.
- [ ] Optimize memory layout (e.g., Structure of Arrays for Phase1Entry if beneficial).
- [ ] Tune kernel launch parameters for device occupancy.
- [ ] Batch Phase 2 suffixes for occupancy.
- [ ] Minimize synchronization and host-device transfers.

---

### Example Usage
```cpp
constexpr double probability_threshold = 0.99; // for L determination
constexpr int L_value = compute_minimal_L(probability_threshold);
constexpr uint32_t k = 1024;  // power of two for bucketed sums
std::array<uint32_t, 4> target_hash = {/* ... */};
// HashCracker cracker; // Internally manages L1, AMQ, etc.
// std::optional<std::array<unsigned char, L_value>> solution = cracker.find_input(target_hash);
```

---

### Analysis
- **Time Complexity:**
  - Phase 1 (Prefix Generation): O(NCHARS^L1)
  - Phase 1 (AMQ Build): O(NCHARS^L1)
  - Phase 2 (Suffix Search): O(NCHARS^L2 * (A/k) * (AMQ_lookup_cost + Exact_lookup_cost_on_hit))
  - Overall: Dominated by O(NCHARS^L1 + NCHARS^L2 * (A/k))
- **Space Complexity:**
  - O(NCHARS^L1 * (sizeof(L1_prefix) + sizeof(bucketed_sum)) + AMQ_filter_size)
- **Bottlenecks:**
  - Phase 2 iteration count (NCHARS^L2)
  - Memory capacity for Phase 1 entries and AMQ

---

### Precision Note
- All 4 segments of the hash must be matched for a valid solution. Using fewer segments will result in many false positives and is not recommended.
