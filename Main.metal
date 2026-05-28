#include <metal_stdlib>
using namespace metal;

// ==========================================
// CONSTANTS & CONFIGURATION
// ==========================================

// Hash table sizing
#define MAP_SIZE_WORDS 268435456u               // 1 GB map size in 32-bit words
#define BUILD_THREADS (MAP_SIZE_WORDS / 8u)     // 33,554,432 threads for map building
#define HASH_MASK 0x0FFFFFFFu                   // 28-bit mask for hash table

// Hash table state
#define EMPTY_BUCKET 0xFFFFFFFFu                // Sentinel value for empty slots

// Search space sizing
#define B_INNER_SIZE 1024u                      // Size of innermost search dimension
#define B_SIDE_SIZE 1024u                       // Size of outer search dimensions
#define THREADGROUP_DIM 32u                     // Threadgroup size for L1 cache work
#define CANDIDATES_CAP 1000u                    // Maximum matches to return per dispatch

// Hash table constants
#define HASH_LEN 16u                            // 16 hash rows per character (5-bit index into 32)
#define HASH_CHARS 32u                          // 32 character encodings (5-bit alphabet)

// ==========================================
// HASH COMPUTATION
// ==========================================

/// Mixes a 128-bit key across 28-bit space for distribution
/// @param key Four-component hash from character position sums
/// @return 28-bit hash suitable for HASH_MASK indexing
static inline uint compute_hash(uint4 key)
{
    return ((key.x << 20) ^ (key.y << 14) ^ (key.z << 7) ^ key.w) & HASH_MASK;
}

/// Reconstructs the full 128-bit hash value for a character sequence on Side A
/// Sums hash contributions from 5 sequential characters at fixed positions
/// @param htable Hash lookup table (512 entries: 16 positions × 32 characters)
/// @param a_id Packed 5-character ID (5 bits per character, 25 bits total)
/// @return Sum of character hashes for the sequence
static inline uint4 sum_from_a(constant uint4 *htable, uint a_id)
{
    return htable[a_id & 31] +
           htable[32 + ((a_id >> 5) & 31)] +
           htable[64 + ((a_id >> 10) & 31)] +
           htable[96 + ((a_id >> 15) & 31)] +
           htable[128 + ((a_id >> 20) & 31)];
}

// ==========================================
// KERNEL 1: INITIALIZE MAP
// ==========================================

/// Initializes the entire hash map to empty state
/// Lock-free: each thread independently marks one bucket as empty
/// @param thread_id Linear thread index in grid
/// @param map Device buffer containing 268M hash table slots
kernel void init_map(uint thread_id [[thread_position_in_grid]],
                     device atomic_uint *map [[buffer(0)]])
{
    if (thread_id >= MAP_SIZE_WORDS)
        return;
    atomic_store_explicit(&map[thread_id], EMPTY_BUCKET, memory_order_relaxed);
}

// ==========================================
// KERNEL 2: BUILD MAP
// ==========================================

/// Builds the hash map from all 33M possible 5-character prefixes
/// Each thread represents one 5-character sequence on Side A
/// Uses linear probing with atomic CAS for lock-free concurrent insertion
/// Guarantees 100% insertion (no hash collisions lost)
/// @param thread_id Encodes 5 characters as 5-bit indices
/// @param map Device hash table to populate
/// @param htable Character hash lookup table (16 positions × 32 characters)
kernel void build_map(uint thread_id [[thread_position_in_grid]],
                      device atomic_uint *map [[buffer(0)]],
                      constant uint4 *htable [[buffer(1)]])
{
    if (thread_id >= BUILD_THREADS)
        return;
    
    uint a_id = thread_id;

    // Compute hash value for this 5-character sequence
    uint4 sumA = sum_from_a(htable, a_id);
    uint4 key = sumA >> 24;
    uint h = compute_hash(key);

    // LINEAR PROBING with atomic CAS: guarantees insertion despite hash clustering
    while (true)
    {
        uint expected = EMPTY_BUCKET;
        if (atomic_compare_exchange_weak_explicit(&map[h], &expected, a_id, 
                                                  memory_order_relaxed, 
                                                  memory_order_relaxed))
        {
            break;
        }
        // Wrap around hash table with HASH_MASK
        h = (h + 1) & HASH_MASK;
    }
}

// ==========================================
// KERNEL 3: SIEVE
// ==========================================

/// Probes hash table using 1024×1024×1024 3D grid of character combinations (Side B)
/// Each thread searches for matching Side A sequences in the hash map
/// Uses L1 cache optimization for the innermost dimension (Table_B_Inner)
/// Implements boundary checking to handle hash value quantization
/// @param thread_id 2D coordinates in the search space
/// @param thread_in_group Local position within threadgroup for cache loading
/// @param map Hash table built in build_map kernel
/// @param offset_target Adjusted target hash after subtracting suffix
/// @param htable Character hash table (Side A lookup)
/// @param Table_B_Outer Pre-computed outer dimension hashes
/// @param Table_B_Mid Pre-computed middle dimension hashes
/// @param Table_B_Inner Pre-computed inner dimension hashes (cached)
/// @param candidates_A Output array for matching Side A IDs
/// @param candidates_B Output array for matching Side B IDs
/// @param counter Atomic counter for output index
/// @param VQ Quantization tolerance threshold
kernel void sieve(uint2 thread_id [[thread_position_in_grid]],
                  uint2 thread_in_group [[thread_position_in_threadgroup]],
                  device const atomic_uint *map [[buffer(0)]],
                  constant uint4 &offset_target [[buffer(1)]],
                  constant uint4 *htable [[buffer(2)]],
                  constant uint4 *Table_B_Outer [[buffer(3)]],
                  constant uint4 *Table_B_Mid [[buffer(4)]],
                  constant uint4 *Table_B_Inner [[buffer(5)]],
                  device uint *candidates_A [[buffer(6)]],
                  device uint *candidates_B [[buffer(7)]],
                  device atomic_uint *counter [[buffer(8)]],
                  constant uint &VQ [[buffer(9)]])
{
    uint id_outer = thread_id.x;
    uint id_mid = thread_id.y;

    // ==========================================
    // L1 CACHE OPTIMIZATION: Load inner table into threadgroup memory
    // ==========================================
    threadgroup uint4 local_Inner[B_INNER_SIZE];
    uint linear_tid = thread_in_group.y * THREADGROUP_DIM + thread_in_group.x;
    if (linear_tid < B_INNER_SIZE)
        local_Inner[linear_tid] = Table_B_Inner[linear_tid];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bounds checking
    if (id_outer >= B_SIDE_SIZE || id_mid >= B_SIDE_SIZE)
        return;

    // Pre-compute partial sums to reduce work in inner loop
    uint4 target_minus_outer = offset_target - Table_B_Outer[id_outer];
    uint4 target_minus_mid = target_minus_outer - Table_B_Mid[id_mid];

    // Inner loop: iterate through all 1024 inner dimension values
    for (uint b_inner = 0; b_inner < B_INNER_SIZE; b_inner++)
    {
        uint4 needed_A = target_minus_mid - local_Inner[b_inner];

        // ==========================================
        // BOUNDARY CHECKING: Account for hash quantization at byte boundaries
        // ==========================================
        uint4 diff;
        diff.x = ((needed_A.x + VQ) >> 24) != (needed_A.x >> 24) ? 1 : 0;
        diff.y = ((needed_A.y + VQ) >> 24) != (needed_A.y >> 24) ? 1 : 0;
        diff.z = ((needed_A.z + VQ) >> 24) != (needed_A.z >> 24) ? 1 : 0;
        diff.w = ((needed_A.w + VQ) >> 24) != (needed_A.w >> 24) ? 1 : 0;

        // Exhaustively probe all boundary-crossing hash buckets
        for (uint ix = 0; ix <= diff.x; ix++)
            for (uint iy = 0; iy <= diff.y; iy++)
                for (uint iz = 0; iz <= diff.z; iz++)
                    for (uint iw = 0; iw <= diff.w; iw++)
                    {
                        // Construct 4-byte key from needed_A high bytes
                        uint4 key;
                        key.x = ((needed_A.x >> 24) + ix) & 0xFF;
                        key.y = ((needed_A.y >> 24) + iy) & 0xFF;
                        key.z = ((needed_A.z >> 24) + iz) & 0xFF;
                        key.w = ((needed_A.w >> 24) + iw) & 0xFF;

                        uint h = compute_hash(key);

                        // ==========================================
                        // HASH TABLE SEARCH: Linear probing with early termination
                        // ==========================================
                        while (true)
                        {
                            uint a_id = atomic_load_explicit(&map[h], memory_order_relaxed);

                            if (a_id == EMPTY_BUCKET)
                                break; // Empty bucket: no match possible

                            // Reconstruct Side A hash and verify match quality
                            uint4 sumA = sum_from_a(htable, a_id);
                            uint4 sumB = Table_B_Outer[id_outer] + 
                                         Table_B_Mid[id_mid] + 
                                         local_Inner[b_inner];
                            uint4 con = sumA + sumB - offset_target;

                            // Check if constraint is satisfied (all components within VQ tolerance)
                            if (con.x <= VQ && con.y <= VQ && con.z <= VQ && con.w <= VQ)
                            {
                                // Valid match found: store and move to next probe
                                uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
                                if (idx < CANDIDATES_CAP)
                                {
                                    candidates_A[idx] = a_id;
                                    // Pack 3D coordinates into 30-bit value
                                    candidates_B[idx] = (id_outer << 20) | (id_mid << 10) | b_inner;
                                }
                                break;
                            }

                            // Probe next bucket in linear sequence
                            h = (h + 1) & HASH_MASK;
                        }
                    }
    }
}