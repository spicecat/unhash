#pragma once

#include "offset.hpp"
#include <oneapi/tbb.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace unhash::chunk
{
    const size_t K = 1<<9;

    namespace fs = std::filesystem;
    using offset::operator<<;
    using offset::operator+;
    using offset::off;
    using offset::orow;
    using tchunk = tbb::concurrent_unordered_set<off>; // Chunk builder
    using chunk = std::vector<off>;                    // Chunk
    // using filter = 

    std::ostream &operator<<(std::ostream &os, const tchunk &c)
    {
        os << "[";
        for (const off &o : c)
            os << std::endl
               << o;
        os << "]";
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const chunk &c)
    {
        os << "[";
        for (const off &o : c)
            os << std::endl
               << o;
        os << "]";
        return os;
    }

    tchunk operator+(const tchunk &a, const off &b)
    {
        tchunk c{};
        c.reserve(a.size());
        for (const off &o : a)
            c.insert(o + b);
        return c;
    }

    tchunk operator*(tchunk &a, const orow &b)
    {
        tchunk c{};
        c.reserve(a.size() * b.size());
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, b.size()),
            [&](const tbb::blocked_range<size_t> &range)
            {
                for (size_t i = range.begin(); i != range.end(); ++i)
                {
                    const tchunk nrow = a + b[i];
                    c.insert(nrow.begin(), nrow.end());
                }
            });
        return c;
    }

    chunk operator+(const chunk &a, const off &b)
    {
        chunk c(a.size());
        for (const off &o : a)
            c.emplace_back(o + b);
        return c;
    }

    chunk build_chunk(const hash::idx size, const hash::idx start = 0)
    {
        tchunk c = {{}};
        for (auto i = start; i < start + size; ++i)
            c = c * offset::OFFSETS[i];
        chunk bc(c.begin(), c.end());
        tbb::parallel_sort(bc.begin(), bc.end());
        return bc;
    }

    void save_chunk(const fs::path &filepath, const chunk &c)
    {
        if (auto parent = filepath.parent_path(); !parent.empty() && !fs::exists(parent))
            fs::create_directories(parent);
        std::ofstream outfile(filepath, std::ios::binary | std::ios::trunc);
        size_t n = c.size();
        outfile.write(reinterpret_cast<const char *>(&n), sizeof(n));
        outfile.write(reinterpret_cast<const char *>(c.data()), n * sizeof(off));
    }

    chunk load_chunk(const fs::path &filepath)
    {
        std::ifstream infile(filepath, std::ios::binary);
        size_t n = 0;
        infile.read(reinterpret_cast<char *>(&n), sizeof(n));
        chunk c(n);
        infile.read(reinterpret_cast<char *>(c.data()), n * sizeof(off));
        return c;
    }
}