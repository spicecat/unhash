#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <limits>
#include <climits>
#include <algorithm>

namespace unhash::filter {

class Filter {
    static constexpr uint8_t  kFingerprintBits   = 8;        // one byte / key
    static constexpr double   kLoadFactor        = 1.15;     // filter size / n
    static constexpr uint32_t kMaxAttempts       = 64;       // seeds to try

    // Internal:
    struct Edge {
        uint32_t h[4];   // four positions in the fingerprint array
        uint8_t  fp;     // 8‑bit fingerprint for the key
    };

public:
    explicit Filter(const std::vector<uint64_t>& keys)
    {
        if (keys.empty()) return;             // trivial empty set
        build(keys);
    }

    bool contains(uint64_t key) const noexcept
    {
        if (fingerprints_.empty()) return false;
        uint64_t hash = mix(key ^ seed_);
        uint8_t  fp   = fingerprint(hash);
        uint32_t h0, h1, h2, h3;
        split_hash(hash, h0, h1, h2, h3);
        uint8_t res = fp ^ fingerprints_[h0] ^ fingerprints_[h1]
                         ^ fingerprints_[h2] ^ fingerprints_[h3];
        return res == 0;
    }

    size_t bytes() const noexcept
    {
        return fingerprints_.size();
    }

private:
    // ------------------------- Construction -------------------------
    void build(const std::vector<uint64_t>& keys)
    {
        const size_t n = keys.size();
        const size_t m = round_up_power_of_two(
            static_cast<size_t>(std::ceil(n * kLoadFactor)));

        fingerprints_.assign(m, 0);

        // Temporary structures
        std::vector<uint32_t> degree(m, 0);
        std::vector<Edge>     edges(n);
        std::vector<uint32_t> stack;
        stack.reserve(n);

        std::mt19937_64 rng{0xF00DFACECAFEBEEFULL};

        for (uint32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
            seed_ = rng();
            std::fill(degree.begin(), degree.end(), 0);

            // 1. Generate edges & compute vertex degrees
            for (size_t i = 0; i < n; ++i) {
                uint64_t h = mix(keys[i] ^ seed_);
                edges[i].fp = fingerprint(h);
                split_hash(h, edges[i].h[0], edges[i].h[1],
                              edges[i].h[2], edges[i].h[3], m);
                ++degree[edges[i].h[0]];
                ++degree[edges[i].h[1]];
                ++degree[edges[i].h[2]];
                ++degree[edges[i].h[3]];
            }

            // 2. Peel: enqueue vertices with degree 1
            std::queue<uint32_t> q;
            for (uint32_t v = 0; v < m; ++v)
                if (degree[v] == 1) q.push(v);

            std::vector<size_t> peel_order;
            peel_order.reserve(n);
            std::vector<uint8_t> edge_use(n, 0);

            while (!q.empty()) {
                uint32_t v = q.front(); q.pop();
                // Visit all incident edges (linear scan – O(n) total)
                for (size_t e = 0; e < n; ++e) {
                    if (edge_use[e]) continue;
                    const auto& edge = edges[e];
                    if (edge.h[0] != v && edge.h[1] != v &&
                        edge.h[2] != v && edge.h[3] != v) continue;
                    edge_use[e] = 1;
                    peel_order.push_back(e);
                    // decrement degrees of other vertices
                    for (int k = 0; k < 4; ++k) {
                        uint32_t u = edge.h[k];
                        if (--degree[u] == 1) q.push(u);
                    }
                    break;
                }
            }

            if (peel_order.size() != n) {
                // not peelable – try another seed
                continue;
            }

            // 3. Assign fingerprints in reverse peel order
            std::fill(fingerprints_.begin(), fingerprints_.end(), 0);
            for (auto it = peel_order.rbegin(); it != peel_order.rend(); ++it) {
                const Edge& edge = edges[*it];
                uint8_t val = edge.fp
                            ^ fingerprints_[edge.h[0]]
                            ^ fingerprints_[edge.h[1]]
                            ^ fingerprints_[edge.h[2]]
                            ^ fingerprints_[edge.h[3]];
                // Store val in the unique vertex which currently has 0
                for (int k = 0; k < 4; ++k) {
                    uint32_t idx = edge.h[k];
                    if (fingerprints_[idx] == 0) {
                        fingerprints_[idx] = val;
                        break;
                    }
                }
            }
            return; // success
        }
        throw std::runtime_error(
            "Filter construction failed after multiple attempts");
    }

    // ------------------------- Hash helpers -------------------------
    static inline uint64_t mix(uint64_t x) noexcept
    {
        // SplitMix64 finaliser
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    static inline uint8_t fingerprint(uint64_t h) noexcept
    {
        uint8_t fp = static_cast<uint8_t>(h & ((1u << kFingerprintBits) - 1));
        return fp ? fp : 1; // avoid zero which would break contains()
    }

    static inline void split_hash(uint64_t h,
                                  uint32_t& h0, uint32_t& h1,
                                  uint32_t& h2, uint32_t& h3,
                                  uint32_t mod_mask) noexcept
    {
        // Use different bit slices, then mix again for diffusion
        h0 = static_cast<uint32_t>(h) & mod_mask;
        h  = mix(h);
        h1 = static_cast<uint32_t>(h) & mod_mask;
        h  = mix(h);
        h2 = static_cast<uint32_t>(h) & mod_mask;
        h  = mix(h);
        h3 = static_cast<uint32_t>(h) & mod_mask;
    }

    inline void split_hash(uint64_t h,
                           uint32_t& h0, uint32_t& h1,
                           uint32_t& h2, uint32_t& h3) const noexcept
    {
        split_hash(h, h0, h1, h2, h3, static_cast<uint32_t>(fingerprints_.size() - 1));
    }

    // Round up to power of two
    static size_t round_up_power_of_two(size_t x)
    {
        if (x <= 1) return 1;
        --x;
        for (size_t i = 1; i < sizeof(size_t) * CHAR_BIT; i <<= 1)
            x |= x >> i;
        return x + 1;
    }

    // data
    std::vector<uint8_t> fingerprints_;
    uint64_t             seed_{0};
};

} // namespace unhash::filter
