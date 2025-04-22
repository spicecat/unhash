#include <oneapi/tbb.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cxxopts.hpp>

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
    options.add_options()("CL,chunk-left", "Chunk left filename", cxxopts::value<string>()->default_value("data/chunk_l.bin"));
    options.add_options()("CR,chunk-right", "Chunk right filename", cxxopts::value<string>()->default_value("data/chunk_r.bin"));
    options.add_options()("h,help", "Print usage");
    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        cout << options.help() << endl;
        return 0;
    }

    auto t_begin = std::chrono::steady_clock::now();

    idx L = result["L"].as<idx>(), C = result["C"].as<idx>();
    string chunk_l_file = result["CL"].as<string>(), chunk_r_file = result["CR"].as<string>();

    chunk CHUNK_L, CHUNK_R;
    if (fs::exists(chunk_l_file))
        CHUNK_L = load_chunk(chunk_l_file);
    else
    {
        CHUNK_L = build_chunk(C);
        save_chunk(chunk_l_file, CHUNK_L);
    }
    if (fs::exists(chunk_r_file))
        CHUNK_R = load_chunk(chunk_r_file);
    else
    {
        CHUNK_R = build_chunk(C, C);
        save_chunk(chunk_r_file, CHUNK_R);
    }

    cout << "Chunk L size: " << CHUNK_L.size() << endl;
    cout << "Chunk R size: " << CHUNK_R.size() << endl;

    int num_threads = tbb::info::default_concurrency();
    const vector<device> devices = device::get_devices();
    cout << "Default number of threads: " << num_threads << endl;
    cout << "Found " << devices.size() << " devices." << endl;
    vector<queue> queues;
    for (const device &dev : devices)
        queues.emplace_back(queue(dev));

    buffer<off> CHUNK_L_BUF(CHUNK_L.data(), CHUNK_L.size()), CHUNK_R_BUF(CHUNK_R.data(), CHUNK_R.size());
    buffer<orow> OFFSETS_BUF(OFFSETS.data(), OFFSETS.size());
    int host_intersegment_found = 0;
    buffer<int, 1> buf_result(&host_intersegment_found, range<1>(1));

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

        q.submit([&](handler &h)
                 {
                     accessor chunk_l_acc(CHUNK_L_BUF, h, read_only);
                     accessor chunk_r_acc(CHUNK_R_BUF, h, read_only);
                     // accessor acc_result_atomic(buf_result, h, sycl::access::mode::read_write, sycl::access::target::global_buffer);

                     // cout << "Chunk L: " << chunk_l_acc[0] << endl;
                     // cout << "Chunk R: " << chunk_r_acc[0] << endl;
                     // h.parallel_for(sycl::range<1>(M), [=](sycl::id<1> item_id)
                     //                { size_t offset_idx = item_id[0];
                 });
    }

    auto t_end = std::chrono::steady_clock::now();
    auto t_total = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_begin).count();

    std::cout << " Total time: " << t_total << " seconds" << endl;
    return 0;
}