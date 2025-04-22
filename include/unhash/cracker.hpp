#pragma once

#include "ext/xorfilter.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_group.h>
#include <sycl/sycl.hpp>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>

namespace unhash
{
    inline int choose_L1(size_t entry_bytes, const sycl::device &dev, int L_max)
    {
        const size_t max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        int L1 = 1;
        while (L1 < L_max &&
               std::pow(static_cast<double>(SIGMA), L1 + 1) * entry_bytes <
                   0.85 * max_alloc)
            ++L1;
        return L1;
    }

    template <int L>
    inline sycl::ext::oneapi::experimental::array<char, L> idx_to_word(u64 idx)
    {
        sycl::ext::oneapi::experimental::array<char, L> w{};
#pragma unroll
        for (int i = 0; i < L; ++i)
        {
            w[i] = static_cast<char>(idx % SIGMA);
            idx /= SIGMA;
        }
        return w;
    }

    static inline u32 bucketize(u32 sum)
    {
        return sycl::mul_hi(sum, 100000u);
    }

    template <int L1, int L>
    class PrefixKernel;

    template <int L1, int L>
    static void build_prefix_table(sycl::queue &Q,
                                   const LookupTable<L> &T,
                                   const std::array<u32, 4> &target,
                                   u64 slice_start, u64 count, u64 stride,
                                   u32 *bucket_out, u64 *idx_out)
    {
        Q.submit([&](sycl::handler &h)
                 { h.parallel_for<PrefixKernel<L1, L>>(sycl::range<1>(count), [=](sycl::id<1> tid)
                                                       {
            const u64 logical = slice_start + tid[0] * stride;
            auto w = idx_to_word<L1>(logical);
            u32 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
            for (int i = 0; i < L1; ++i) {
                const char c = w[i];
                s0 += T.T[0][i][c];
                s1 += T.T[1][i][c];
                s2 += T.T[2][i][c];
                s3 += T.T[3][i][c];
            }
            const u32 b0 = bucketize(s0);
            const u32 b1 = bucketize(s1);
            const u32 b2 = bucketize(s2);
            const u32 b3 = bucketize(s3);
            // store complement wrt target
            sycl::vec<u32,4> comp = {
                static_cast<u32>((target[0] + 100000u - b0) % 100000u),
                static_cast<u32>((target[1] + 100000u - b1) % 100000u),
                static_cast<u32>((target[2] + 100000u - b2) % 100000u),
                static_cast<u32>((target[3] + 100000u - b3) % 100000u)};
            // single 128‑bit global store
            sycl::store(comp, reinterpret_cast<sycl::vec<u32,4>*>(bucket_out) + tid[0]);
            idx_out[tid[0]] = logical; }); });
    }

    /* ---------------------------------------------------------------------------
     * BinaryFuse8 helpers (pack 4×17‑bit buckets → 64‑bit key)
     * ------------------------------------------------------------------------ */
    using BF8 = xorfilter::BinaryFuse8;

    static inline uint64_t pack_key(const u32 *b)
    {
        uint64_t k = (static_cast<uint64_t>(b[0])) |
                     (static_cast<uint64_t>(b[1]) << 17) |
                     (static_cast<uint64_t>(b[2]) << 34);
        k ^= static_cast<uint64_t>(b[3]) * 0x9E3779B185EBCA87ULL;
        return k;
    }

    /* ---------------------------------------------------------------------------
     * CPU suffix phase (host threads)
     * ------------------------------------------------------------------------ */

    template <int L1, int L>
    static bool cpu_suffix_phase(const LookupTable<L> &T,
                                 const BF8 &filter,
                                 const std::vector<u32> &buckets,
                                 const std::vector<u64> &indices,
                                 std::atomic<u64> &found_pre,
                                 std::atomic<u64> &found_suf,
                                 std::atomic<bool> &found_flag)
    {
        constexpr int L2 = L - L1;
        const u64 total_suffixes = static_cast<u64>(std::pow(SIGMA, L2));

        tbb::parallel_for(tbb::blocked_range<u64>(0, total_suffixes, 1 << 15),
                          [&](const tbb::blocked_range<u64> &r)
                          {
                              if (found_flag.load(std::memory_order_acquire))
                                  return;
                              for (u64 sid = r.begin(); sid != r.end(); ++sid)
                              {
                                  if (found_flag.load(std::memory_order_relaxed))
                                      return;
                                  auto suf = idx_to_word<L2>(sid);
                                  u32 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
                                  for (int j = 0; j < L2; ++j)
                                  {
                                      const char c = suf[j];
                                      s0 += T.T[0][L1 + j][c];
                                      s1 += T.T[1][L1 + j][c];
                                      s2 += T.T[2][L1 + j][c];
                                      s3 += T.T[3][L1 + j][c];
                                  }
                                  const u32 b0 = bucketize(s0);
                                  const u32 b1 = bucketize(s1);
                                  const u32 b2 = bucketize(s2);
                                  const u32 b3 = bucketize(s3);
                                  const u32 key4[4] = {b0, b1, b2, b3};
                                  const uint64_t k = pack_key(key4);
                                  if (!filter.contain(k))
                                      continue; // 99.7 % rejected

                                  // tiny candidate set: linear match
                                  for (std::size_t p = 0; p < indices.size(); ++p)
                                  {
                                      const u32 *comp = &buckets[4 * p1];
                                      if (comp[0] == b0 && comp[1] == b1 && comp[2] == b2 && comp[3] == b3)
                                      {
                                          found_pre.store(indices[p], std::memory_order_release);
                                          found_suf.store(sid, std::memory_order_release);
                                          found_flag.store(true, std::memory_order_release);
                                          return;
                                      }
                                  }
                              }
                          });
        return found_flag.load();
    }

    /* ---------------------------------------------------------------------------
     * Public API – multi‑device crack (template)
     * ------------------------------------------------------------------------ */

    template <int L>
    std::optional<std::array<char, L>> crack_multi_device(const LookupTable<L> &T,
                                                          const std::array<u32, 4> &target,
                                                          int requested_L1)
    {
        std::vector<sycl::device> devs = sycl::device::get_devices();
        if (devs.empty())
            devs.emplace_back(sycl::device(sycl::cpu_selector_v));

        std::atomic<bool> found{false};
        std::atomic<u64> found_pre{0}, found_suf{0};
        std::atomic<int> solved_L1{-1};

        tbb::task_group tg;
        for (size_t did = 0; did < devs.size(); ++did)
        {
            tg.run([&, did]
                   {
            sycl::device dev = devs[did];
            sycl::queue Q(dev, sycl::property::queue::in_order{});

            constexpr std::size_t entry_bytes = 24;  // 4×u32 + index
            const int L1 = (requested_L1 >= 0) ? requested_L1
                                               : choose_L1(entry_bytes, dev, L);
            const u64 P_total = static_cast<u64>(std::pow(SIGMA, L1));

            const std::size_t max_alloc =
                dev.get_info<sycl::info::device::max_mem_alloc_size>();
            const u64 batch_entries = static_cast<u64>(0.75 * max_alloc / entry_bytes);
            const std::size_t stride = devs.size();

            std::vector<u32> h_buckets;
            std::vector<u64> h_indices;

            for (u64 slice = did; slice < P_total && !found.load(); slice += stride * batch_entries) {
                const u64 remaining = P_total - slice;
                const u64 count = std::min<u64>(batch_entries, remaining / stride);
                if (!count) break;

                u32 *d_buckets = sycl::malloc_device<u32>(count * 4, Q);
                u64 *d_indices = sycl::malloc_device<u64>(count, Q);

                build_prefix_table<L1, L>(Q, T, target, slice, count, stride,
                                          d_buckets, d_indices);
                Q.wait_and_throw();

                h_buckets.resize(count * 4);
                h_indices.resize(count);
                Q.memcpy(h_buckets.data(), d_buckets, count * 4 * sizeof(u32)).wait();
                Q.memcpy(h_indices.data(), d_indices, count * sizeof(u64)).wait();
                sycl::free(d_buckets, Q);
                sycl::free(d_indices, Q);

                // Build BinaryFuse8 filter on host
                std::vector<uint64_t> keys(count);
                for (u64 i = 0; i < count; ++i)
                    keys[i] = pack_key(&h_buckets[4 * i]);

                BF8 filter;
                xorfilter::xor8_build(keys.size(), keys.data(), &filter);

                if (cpu_suffix_phase<L1, L>(T, filter, h_buckets, h_indices,
                                             found_pre, found_suf, found)) {
                    solved_L1.compare_exchange_strong(requested_L1, L1);
                    break;
                }
            } });
        }
        tg.wait();

        if (!found.load())
            return std::nullopt;

        const int L1_final = (solved_L1.load() > 0) ? solved_L1.load() : ((requested_L1 > 0) ? requested_L1 : 1);
        const int L2_final = L - L1_final;

        u64 pre_idx = found_pre.load();
        u64 suf_idx = found_suf.load();

        auto pre_word = idx_to_word<L>(pre_idx); // large template, will use first L1
        auto suf_word = idx_to_word<L>(suf_idx); // size L2; we'll copy subset

        std::array<char, L> result{};
        for (int i = 0; i < L1_final; ++i)
            result[i] = pre_word[i];
        for (int j = 0; j < L2_final; ++j)
            result[L1_final + j] = suf_word[j];
        return result;
    }

#define INSTANTIATE(L)                                                 \
    template std::optional<std::array<char, L>> crack_multi_device<L>( \
        const LookupTable<L> &, const std::array<u32, 4> &, int);
    INSTANTIATE(9)
    INSTANTIATE(10)
    INSTANTIATE(11)
    INSTANTIATE(12)
    INSTANTIATE(13)
    INSTANTIATE(14)
#undef INSTANTIATE

} // namespace hc
