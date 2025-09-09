#include <oneapi/tbb.h>
#include <sycl/sycl.hpp>
#include "binfuse/sharded_filter.hpp"
#include <iostream>
#include <filesystem>

using std::cout;
using std::endl;
using std::vector;
using sycl::accessor;
using sycl::buffer;
using sycl::device;
using sycl::handler;
using sycl::host_accessor;
using sycl::queue;
using sycl::range;
using sycl::read_only;

int main(int argc, char **argv)
{
    std::cout << "Hello, World!!" << std::endl;
    std::filesystem::remove_all("tmp");
    std::filesystem::create_directories("tmp");

    binfuse::filter16 tiny_low(std::vector<std::uint64_t>{
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000002,
    });

    binfuse::filter16 tiny_high(std::vector<std::uint64_t>{
        0x8000000000000000,
        0x8000000000000001,
        0x8000000000000002,
    });

    binfuse::sharded_filter16_sink sink("tmp/sharded_filter16_tiny.bin", 1);
    sink.add_shard(tiny_low, 0);
    sink.add_shard(tiny_high, 1);

    binfuse::sharded_filter16_source source("tmp/sharded_filter16_tiny.bin", 1);

    std::cout << source.contains(0x0000000000000000) << std::endl;
    std::cout << source.shards() << std::endl;

    int num_threads = tbb::info::default_concurrency();
    const vector<device> devices = device::get_devices();
    cout << "Default number of threads: " << num_threads << endl;
    cout << "Found " << devices.size() << " devices." << endl;
    vector<queue> queues;
    for (const device &dev : devices)
        queues.emplace_back(queue(dev));

    for (queue &q : queues)
    {
        device dev = q.get_device();
        cout << "Device: " << dev.get_info<sycl::info::device::name>() << endl;
        cout << "Vendor: " << dev.get_info<sycl::info::device::vendor>() << endl;
        cout << "Max Compute Units: " << dev.get_info<sycl::info::device::max_compute_units>() << endl;
        cout << "Max Work Group Size: " << dev.get_info<sycl::info::device::max_work_group_size>() << endl;
        cout << "Max Clock Frequency: " << dev.get_info<sycl::info::device::max_clock_frequency>() << " MHz" << endl;
        cout << "Global Memory Size: " << dev.get_info<sycl::info::device::global_mem_size>() << " bytes" << endl;
        cout << "Local Memory Size: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << endl;
        cout << "Max Memory Alloc Size: " << dev.get_info<sycl::info::device::max_mem_alloc_size>() << " bytes" << endl;
    }

    return 0;
}