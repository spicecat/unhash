#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>
#include <sycl/sycl.hpp>

#include "unhash/hash.hpp"

using std::array;
using std::vector;
using namespace std::chrono;
using namespace sycl;

constexpr size_t num_repetitions = 500;
constexpr size_t N = 10000;

template <typename T>
nanoseconds::rep test_vector_add(queue &q, const T &type)
{
    array<T, N> a, b, sum_sequential, sum_parallel;

    std::mt19937 rng(0);
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                          std::numeric_limits<T>::max());
    std::cout << "Size: " << sizeof(T) << " Min: " << dist.min() << " Max: " << dist.max() << std::endl;

    T mask = 0;
    for (size_t i = 0; i < sizeof(T) / 2 - 1; i++)
        mask |= static_cast<T>(1) << (16 * i + 15);

    for (size_t i = 0; i < N; i++)
    {
        a.at(i) = dist(rng);
        b.at(i) = dist(rng);
        sum_sequential.at(i) = a.at(i) + b.at(i) - ((a.at(i) & b.at(i) & mask) << 1);
    }

    duration best_time = steady_clock::duration::max();

    range<1> num_items{N};

    buffer a_buf(a);
    buffer b_buf(b);
    buffer sum_buf(sum_parallel.data(), num_items);

    for (size_t i = 0; i < num_repetitions; i++)
    {
        steady_clock::time_point start = steady_clock::now();
        for (size_t i = 0; i < 8 / sizeof(T); i++)
            q.submit([&](handler &h)
                     {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor sum(sum_buf, h, write_only, no_init);
            h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i] - ((a[i]&b[i]&mask) << 1); }); });

        q.wait();
        steady_clock::time_point end = steady_clock::now();

        if (end - start < best_time)
            best_time = end - start;
        for (size_t i = 0; i < N; i++)
            if (sum_parallel.at(i) != sum_sequential.at(i))
            {
                std::cout << "Vector add failed " << i << " " << sum_parallel.at(i) << " " << sum_sequential.at(i) << std::endl;
                return -1;
            }
    }
    std::cout << typeid(T).name() << " Best time: "
              << best_time.count() << " nanoseconds" << std::endl;
    return best_time.count();
}

int main(int argc, char *argv[])
{
    const vector<device> devices = device::get_devices();
    std::cout << "Found " << devices.size() << " devices.\n";

    vector<queue> queues;
    for (const device &dev : devices)
        queues.emplace_back(queue(dev));

    for (queue &q : queues)
    {
        device dev = q.get_device();
        std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl
                  << "Memory: " << dev.get_info<info::device::global_mem_size>() << " bytes\n";

        std::cout << "Native vector width for int: " << dev.get_info<info::device::native_vector_width_int>() << std::endl;
        std::cout << "Native vector width for short: " << dev.get_info<info::device::native_vector_width_short>() << std::endl;
        std::cout << "Native vector width for long: " << dev.get_info<info::device::native_vector_width_long>() << std::endl;

        std::cout << "Vector size: " << N << "\n";
        std::cout << "Repetitions: " << num_repetitions << "\n";

        test_vector_add(q, uint16_t{});
        test_vector_add(q, uint32_t{});
        test_vector_add(q, uint64_t{});
        test_vector_add(q, uint_fast16_t{});
        test_vector_add(q, uint_fast32_t{});
        test_vector_add(q, uint_fast64_t{});
    }
    return 0;
}
