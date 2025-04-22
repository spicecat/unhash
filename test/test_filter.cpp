#include "unhash/filter.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <cstdint>

using unhash::filter::Filter;

// Helper: generate a vector of unique random 64‑bit integers
static std::vector<uint64_t> generate_keys(size_t n, std::mt19937_64 &rng)
{
    std::unordered_set<uint64_t> seen;
    seen.reserve(n * 2);
    std::vector<uint64_t> out;
    out.reserve(n);
    while (out.size() < n)
    {
        uint64_t x = rng();
        if (seen.insert(x).second)
            out.push_back(x);
    }
    return out;
}

// ----------------------------- Unit tests -----------------------------

TEST(FilterTest, EmptyFilterReturnsFalse)
{
    Filter filter({});
    EXPECT_FALSE(filter.contains(0ULL));
    EXPECT_FALSE(filter.contains(123456789ULL));
}

TEST(FilterTest, ContainsInsertedKeys)
{
    std::vector<uint64_t> keys = {1ULL, 2ULL, 3ULL, 42ULL, 0xDEADBEEFCAFEBEEFULL};
    Filter filter(keys);
    for (uint64_t k : keys)
        EXPECT_TRUE(filter.contains(k)) << "Key " << k << " should be present";
}

TEST(FilterTest, AcceptableFalsePositiveRate)
{
    constexpr size_t n_keys = 100000;
    constexpr size_t n_queries = 200000;
    std::mt19937_64 rng{12345};

    auto keys = generate_keys(n_keys, rng);
    Filter filter(keys);

    size_t fp = 0;
    for (size_t i = 0; i < n_queries; ++i)
    {
        uint64_t candidate;
        do
            candidate = rng();
        while (std::binary_search(keys.begin(), keys.end(), candidate));

        if (filter.contains(candidate))
            ++fp;
    }
    double fpr = static_cast<double>(fp) / n_queries;
    EXPECT_LT(fpr, 0.008) << "False positive rate " << fpr;
}

TEST(FilterTest, LargeRandomSetBuildsAndQueries)
{
    constexpr size_t n_keys = 1'000'000;
    std::mt19937_64 rng{67890};
    auto keys = generate_keys(n_keys, rng);
    Filter filter(keys);

    // spot‑check 1000 random true queries
    for (size_t i = 0; i < 1000; ++i)
    {
        uint64_t k = keys[rng() % n_keys];
        EXPECT_TRUE(filter.contains(k));
    }
}
 