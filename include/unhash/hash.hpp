#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace unhash::hash
{

    using std::string_view;

    // Character set allowed in input strings
    constexpr string_view CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 ";

    // --- Type Aliases ---
    using chr = uint_fast8_t;   // Represents an encoded character (index in CHARS)
    using seg = uint_fast8_t;   // Represents a hash segment index
    using idx = uint_fast8_t;   // Represents a character index within the input
    using cont = uint_fast32_t; // Represents a hash segment contribution value

    // --- Constants ---
    constexpr seg SEGMENTS = 4;            // Number of segments in final hash
    constexpr chr NCHARS = CHARS.length(); // Number of characters in CHARS
    constexpr cont M = 100000;             // Hash segment range
    constexpr double A = 20000.0;          // Multiplier used contribution precision
    constexpr cont MA = M * A;             // Hash segment contribution range

    // Parameters for the contribution calculation per segment
    constexpr std::array<std::array<cont, 5>, SEGMENTS> P = {{
        {{11, 29, 53, 52, 0}},  // Parameters for segment 0
        {{17, 31, 67, 12, 90}}, // Parameters for segment 1
        {{7, 17, 103, 24, 0}},  // Parameters for segment 2
        {{5, 13, 47, 30, 90}}   // Parameters for segment 3
    }};

    // --- Data Structures ---
    using enc = std::vector<chr>;         // Represents the encoded input string (vector of character indices)
    using h = std::array<cont, SEGMENTS>; // Represents the final hash (array of contributions per segment)

    // --- Output Stream Operators ---

    // Print encoded representation
    std::ostream &operator<<(std::ostream &os, const enc &encoding)
    {
        for (chr c : encoding)
            os << static_cast<int>(c) << " ";
        return os;
    }

    // Print hash representation
    std::ostream &operator<<(std::ostream &os, const h &hash)
    {
        os << "h(";
        for (seg s = 0; s < SEGMENTS; ++s)
            os << hash[s] << (s == SEGMENTS - 1 ? "" : " ");
        os << ")";
        return os;
    }

    // --- Core Functions ---

    // Encodes a string_view into a vector of character indices (chr)
    // Characters not found in CHARS are ignored.
    enc encode(const string_view &input)
    {
        enc encoding{};
        encoding.reserve(input.length());
        for (char ch : input)
            if (size_t index = CHARS.find(ch); index != string_view::npos)
                encoding.emplace_back(static_cast<chr>(index));
        return encoding;
    }

    // Calculates the contribution of a single character 'c' at index 'i' for a segment 's'.
    cont contribution(seg s, idx i, chr c)
    {
        double angle =
            (P[s][0] * (static_cast<cont>(i) + 1) % P[s][1] + P[s][2]) *
                (static_cast<cont>(c) + P[s][3]) +
            P[s][4];
        double sin_val = std::sin(angle * M_PI / 180.0);
        double intermediate = std::fmod(static_cast<double>(M) + sin_val, 0.2);
        double scaled_val = intermediate * MA * 5.0;
        return static_cast<cont>(std::round(scaled_val));
    }

    // Calculates the hash contributions for all segments for a single character 'c' at index 'i'.
    h contribution(idx i, chr c)
    {
        h char_hash{};
        for (seg s = 0; s < SEGMENTS; ++s)
            char_hash[s] = contribution(s, i, c);
        return char_hash;
    }

    // Computes the final hash from an encoded input.
    h hashify(const enc &encoding)
    {
        h hash{};

        // Accumulate contributions for each character
        for (idx i = 0; i < encoding.size(); ++i)
            for (seg s = 0; s < SEGMENTS; ++s)
                hash[s] = (hash[s] + contribution(s, i, encoding[i])) % MA;

        // Rescale each segment
        for (seg s = 0; s < SEGMENTS; ++s)
            hash[s] /= A;
        return hash;
    }

    // Computes the hash directly from a string_view input.
    h hashify(const string_view &input)
    {
        enc encoding = encode(input);
        return hashify(encoding);
    }

} // namespace unhash::hash