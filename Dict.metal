#include <metal_stdlib>
using namespace metal;

// ==========================================
// CONSTANTS & CONFIGURATION
// ==========================================

// Hash table configuration
#define HASH_MASK 0xFFFFFFFu                    // 268M buckets (~1 GB RAM)
#define EMPTY_BUCKET 0xFFFFFFFFu               // Sentinel value for empty slots

// Buffer constraints
#define MAX_TOTAL_LEN 128u                      // Maximum string length in bytes
#define N CONST_N                               // Dictionary size (injected at compile time)

// Bit-packing for ID storage
#define ID_BITS_16 0xFFFFu                     // Mask for 16-bit ID in 32-bit word
#define ID_SHIFT_16 16u                         // Shift for second 16-bit ID

// ==========================================
// HASH COMPUTATION
// ==========================================

/// Computes a 28-bit hash from a 128-bit key (four uint32 components)
/// Uses mixing across all four components with different shift values
/// @param key Four-component key derived from character hash sums
/// @return 28-bit hash value in range [0, HASH_MASK]
static inline uint compute_hash(uint4 key)
{
    return ((key.x << 20) ^ (key.y << 14) ^ (key.z << 7) ^ key.w) & HASH_MASK;
}

// ==========================================
// KERNEL 1: INITIALIZE MAP
// ==========================================

/// Initializes the entire hash map to empty state
/// Each thread marks one bucket as empty for lock-free concurrent inserts
kernel void init_map(uint thread_id [[thread_position_in_grid]],
                     device atomic_uint *map [[buffer(0)]])
{
    atomic_store_explicit(&map[thread_id], EMPTY_BUCKET, memory_order_relaxed);
}

// ==========================================
// KERNEL 2: BUILD MAP (Side A: Exactly 2 Words)
// ==========================================

/// Builds hash table from all possible 2-word combinations from the dictionary
/// Each 2D thread processes one pair (id0, id1) and inserts the packed ID pair
/// Uses linear probing with atomic CAS for lock-free concurrent inserts
kernel void build_map(uint2 thread_id [[thread_position_in_grid]],
                      device atomic_uint *map [[buffer(0)]],
                      constant uint *lenS [[buffer(1)]],
                      constant uint4 *hashS [[buffer(2)]])
{
    uint id0 = thread_id.x;
    uint id1 = thread_id.y;

    // Bounds checking
    if (id0 >= N || id1 >= N)
        return;

    // Accumulate hash values for both words
    uint pos = 0;
    uint4 sumA = uint4(0);

    // Word 1: indexed by id0
    sumA += hashS[(id0 << 7) + pos];
    pos += lenS[id0];

    // Word 2: indexed by id1
    sumA += hashS[(id1 << 7) + pos];
    pos += lenS[id1];

    // Abort if combined length exceeds buffer size
    if (pos >= MAX_TOTAL_LEN)
        return;

    // Pack both 16-bit word IDs into a single 32-bit value
    uint a_id = id0 | (id1 << ID_SHIFT_16);

    // Compute initial hash bucket
    uint h = compute_hash(sumA >> 24);

    // Linear probing: insert with lock-free atomic compare-and-swap
    while (true)
    {
        uint expected = EMPTY_BUCKET;
        if (atomic_compare_exchange_weak_explicit(&map[h], &expected, a_id, 
                                                  memory_order_relaxed, 
                                                  memory_order_relaxed))
            break;
        
        // Move to next bucket, wrapping around with HASH_MASK
        h = (h + 1) & HASH_MASK;
    }
}

// ==========================================
// KERNEL 3: SIEVE MAP (Side B: Up to 4 Words)
// ==========================================

/// Probes the hash table for matches using combinations from dictionary
/// Processes 3D grid of word combinations (up to 4 words on Side B)
/// For each candidate, searches hash table with boundary checking on hash values
kernel void sieve_map(uint3 thread_id [[thread_position_in_grid]],
                      device const atomic_uint *map [[buffer(0)]],
                      constant uint *lenS [[buffer(1)]],
                      constant uint *lenN [[buffer(2)]],
                      constant uint4 *hashS [[buffer(3)]],
                      constant uint4 *hashN [[buffer(4)]],
                      constant uint &num_words [[buffer(5)]],
                      constant uint &offset_A [[buffer(6)]],
                      constant uint4 &TARGET [[buffer(7)]],
                      device uint *cands_A [[buffer(8)]],
                      device ulong *cands_B [[buffer(9)]],
                      device atomic_uint *counter [[buffer(10)]],
                      constant uint &VQ [[buffer(11)]],
                      constant uint &fixed_id3 [[buffer(12)]])
{
    uint id0 = thread_id.x;
    uint id1 = thread_id.y;
    uint id2 = thread_id.z;

    // Bounds checking: always check id0, conditionally check others
    if (id0 >= N)
        return;
    if (num_words >= 2 && id1 >= N)
        return;
    if (num_words >= 3 && id2 >= N)
        return;

    // Gather word IDs for up to 4-word combination
    uint ids[4];
    ids[0] = id0;
    if (num_words >= 2)
        ids[1] = id1;
    if (num_words >= 3)
        ids[2] = id2;
    if (num_words >= 4)
        ids[3] = fixed_id3;

    // Accumulate hash values starting from offset_A
    uint pos = offset_A;
    uint4 sumB = uint4(0);

    for (uint i = 0; i < num_words; i++)
    {
        uint w_id = ids[i];
        if (pos >= MAX_TOTAL_LEN)
            return;

        // Last word uses "no space" hashes; others use "space" hashes
        if (i == num_words - 1)
        {
            sumB += hashN[(w_id << 7) + pos];
            pos += lenN[w_id];
        }
        else
        {
            sumB += hashS[(w_id << 7) + pos];
            pos += lenS[w_id];
        }
    }

    // Validate final position: must be within [12, MAX_TOTAL_LEN]
    if (pos > MAX_TOTAL_LEN || pos < 12)
        return;

    // Compute required Side A hash: TARGET - sumB
    uint4 needed_A = TARGET - sumB;

    // Boundary checking: detect if needed_A crosses 8-bit boundaries when VQ is added
    uint4 diff;
    diff.x = ((needed_A.x + VQ) >> 24) != (needed_A.x >> 24) ? 1 : 0;
    diff.y = ((needed_A.y + VQ) >> 24) != (needed_A.y >> 24) ? 1 : 0;
    diff.z = ((needed_A.z + VQ) >> 24) != (needed_A.z >> 24) ? 1 : 0;
    diff.w = ((needed_A.w + VQ) >> 24) != (needed_A.w >> 24) ? 1 : 0;

    // Exhaustively search all hash bucket boundaries
    for (uint ix = 0; ix <= diff.x; ix++)
        for (uint iy = 0; iy <= diff.y; iy++)
            for (uint iz = 0; iz <= diff.z; iz++)
                for (uint iw = 0; iw <= diff.w; iw++)
                {
                    // Construct search key from top byte of each component
                    uint4 key;
                    key.x = ((needed_A.x >> 24) + ix) & 0xFF;
                    key.y = ((needed_A.y >> 24) + iy) & 0xFF;
                    key.z = ((needed_A.z >> 24) + iz) & 0xFF;
                    key.w = ((needed_A.w >> 24) + iw) & 0xFF;

                    uint h = compute_hash(key);

                    // Linear probe: search for matching Side A combination
                    while (true)
                    {
                        uint a_val = atomic_load_explicit(&map[h], memory_order_relaxed);
                        if (a_val == EMPTY_BUCKET)
                            break;

                        // Unpack 16-bit IDs: a_id0 in lower 16 bits, a_id1 in upper 16 bits
                        uint a_id0 = a_val & ID_BITS_16;
                        uint a_id1 = a_val >> ID_SHIFT_16;

                        // Recompute Side A hash to verify
                        uint a_pos = 0;
                        uint4 sumA = uint4(0);

                        sumA += hashS[(a_id0 << 7) + a_pos];
                        a_pos += lenS[a_id0];
                        sumA += hashS[(a_id1 << 7) + a_pos];
                        a_pos += lenS[a_id1];

                        // Verify correct offset and check constraint
                        if (a_pos == offset_A)
                        {
                            uint4 con = sumA + sumB - TARGET;
                            if (con.x <= VQ && con.y <= VQ && con.z <= VQ && con.w <= VQ)
                            {
                                uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
                                if (idx < 1000)
                                {
                                    cands_A[idx] = a_val;

                                    // Pack Side B IDs into 64-bit integer (16 bits each)
                                    ulong packedB = id0;
                                    if (num_words >= 2)
                                        packedB |= ((ulong)id1 << 16);
                                    if (num_words >= 3)
                                        packedB |= ((ulong)id2 << 32);
                                    if (num_words >= 4)
                                        packedB |= ((ulong)fixed_id3 << 48);
                                    cands_B[idx] = packedB;
                                }
                                break;
                            }
                        }
                        h = (h + 1) & HASH_MASK;
                    }
                }
}