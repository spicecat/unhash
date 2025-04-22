#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cxxopts.hpp>
#include <oneapi/tbb.h>
#include <sycl/sycl.hpp>

#include "unhash/chunk.hpp"

using namespace unhash::hash;
using namespace unhash::offset;
using namespace unhash::chunk;
using std::cout;
using std::endl;
using std::pow;
using std::string;
using std::vector;
using sycl::accessor;
using sycl::buffer;
using sycl::device;
using sycl::handler;
using sycl::host_accessor;
using sycl::queue;
using sycl::range;
using sycl::read_only;
using unhash::chunk::chunk;

int main(int argc, char **argv)
{
    cxxopts::Options options("Unhash", "Unhash TheLogFather project");
    options.add_options()("L", "Target length", cxxopts::value<idx>()->default_value("8"));
    options.add_options()("C", "Chunk length", cxxopts::value<idx>()->default_value("2"));
    options.add_options()("CL,chunk", "Chunk filename", cxxopts::value<string>()->default_value("data/chunk.bin"));
    options.add_options()("h,help", "Print usage");
    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        cout << options.help() << endl;
        return 0;
    }

    auto t_begin = std::chrono::steady_clock::now();

    idx L = result["L"].as<idx>(), C = result["C"].as<idx>();
    string chunk_file = result["CL"].as<string>();

    chunk CHUNK;
    if (fs::exists(chunk_file))
        CHUNK = load_chunk(chunk_file);
    else
    {
        CHUNK = build_chunk(C);
        save_chunk(chunk_file, CHUNK);
    }
    buffer<off> CHUNK_L_BUF(CHUNK.data(), CHUNK.size());

    cout << "Chunk L size: " << CHUNK.size() << endl;

    const vector<device> devices = device::get_devices();
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

    int num_threads = tbb::info::default_concurrency();
    cout << "Default number of threads: " << num_threads << endl;

    auto t_end = std::chrono::steady_clock::now();
    auto t_total = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_begin).count();

    std::cout << " Total time: " << t_total << " seconds" << endl;
    return 0;
}