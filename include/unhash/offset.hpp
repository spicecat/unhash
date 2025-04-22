#pragma once

#include "hash.hpp"
#include <array>
#include <functional> // For std::hash
#include <iostream>

namespace unhash::offset
{

    // --- Constants and Type Aliases ---

    using hash::chr;
    using hash::cont;
    using hash::idx;
    using hash::NCHARS;
    using hash::P;
    using hash::SEGMENTS;

    // Helper function to determine MAX_L at compile time
    // Finds the maximum value of P[i][1] across all segments.
    inline constexpr idx get_max_L()
    {
        idx max_L = 0;
        for (idx i = 0; i < P.size(); ++i)
            if (P[i][1] > max_L)
                max_L = P[i][1];
        return max_L;
    }

    // Maximum length considered for precomputed offsets, derived from P parameters.
    constexpr idx MAX_L = get_max_L();

    // --- Offset Specific Types ---

    using oc = uint32_t;                    // Offset Contribution (scaled to fit the range of oc)
    using off = std::array<oc, SEGMENTS>;   // Offset
    using orow = std::array<off, NCHARS>;   // Offset Row (offsets for all possible chars at a specific index)
    using otable = std::array<orow, MAX_L>; // Offset Table (all rows up to MAX_L)

    // --- Output Stream Operators ---

    // Print a single offset `off`
    constexpr std::ostream &operator<<(std::ostream &os, const off &o)
    {
        os << "o(";
        for (size_t s = 0; s < SEGMENTS; ++s)
        {
            os << o[s] << (s == SEGMENTS - 1 ? "" : " "); // No space after last element
        }
        os << ")";
        return os;
    }

    // Print the entire offset table `otable`
    std::ostream &operator<<(std::ostream &os, const otable &ot)
    {
        for (const orow &current_orow : ot)
        {
            for (const off &o : current_orow)
            {
                os << o << " ";
            }
            os << std::endl;
        }
        return os;
    }

    // --- Arithmetic Operators for Offset (`off`) ---

    // Addition operator for offsets
    constexpr off operator+(const off &a, const off &b)
    {
        off result{};
        for (size_t s = 0; s < SEGMENTS; ++s)
            result[s] = a[s] + b[s];
        return result;
    }

    // Addition assignment operator for offsets
    constexpr off &operator+=(off &a, const off &b)
    {
        for (size_t s = 0; s < SEGMENTS; ++s)
            a[s] += b[s];
        return a;
    }

    // --- Packing Functions ---

    // Scales a hash contribution segment (`cont`) to cover the range of the offset contribution type (`oc`).
    constexpr oc pack(const hash::cont &co)
    {
        return static_cast<oc>(co * (1ULL << 8 * sizeof(oc)) / hash::MA);
    }

    // Packs a full hash value (`hash::h`) into an offset (`off`) by packing each segment.
    constexpr off pack(const hash::h &hash)
    {
        off o{};
        for (size_t s = 0; s < SEGMENTS; ++s)
            o[s] = pack(hash[s]);
        return o;
    }

    // --- Offset Table Generation ---

    // Builds the complete offset table by calculating and packing contributions
    // for each character at each position up to MAX_L.
    inline otable build_offset_table()
    {
        otable table{};
        for (hash::idx i = 0; i < MAX_L; ++i)
            for (hash::chr c = 0; c < NCHARS; ++c)
            {
                // Calculate the contribution hash for char 'c' at index 'i'
                hash::h contribution_hash = hash::contribution(i, c);
                // Pack the hash into an offset and store it in the table
                table[i][c] = pack(contribution_hash);
            }
        return table;
    }

    // Initialize the global offset table.
    inline otable OFFSETS = build_offset_table();

} // namespace unhash::offset

// --- std::hash Specialization ---

// Specialization of std::hash for the `unhash::offset::off` type
// Allows `off` objects to be used as keys in standard hash containers (e.g., unordered_map).
template <>
struct std::hash<unhash::offset::off>
{
    size_t operator()(const unhash::offset::off &o) const noexcept
    {
        size_t seed = 0;
        constexpr size_t total_bits = sizeof(size_t) * 8;          // Bits in the result hash value
        constexpr size_t oc_bits = sizeof(unhash::offset::oc) * 8; // Bits in one packed offset segment
        // Bits to try and extract from each `oc` segment for the final hash.
        // Distribute total_bits as evenly as possible among SEGMENTS.
        constexpr size_t bits_per_segment = total_bits / unhash::offset::SEGMENTS;
        constexpr size_t right_shift_amount = oc_bits - bits_per_segment;

        for (size_t s = 0; s < unhash::offset::SEGMENTS; ++s)
        {
            // 1. Extract the top `bits_per_segment` from the current offset segment `o[s]`.
            size_t extracted_bits = static_cast<size_t>(o[s] >> right_shift_amount);

            // 2. Shift these extracted bits to their designated position within the final `seed`.
            // This prevents bits from different segments from overlapping too much, assuming
            // bits_per_segment * SEGMENTS <= total_bits.
            size_t shifted_bits = extracted_bits << s * bits_per_segment;
            seed ^= shifted_bits;
        }
        return seed;
    }
};