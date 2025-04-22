#include "unhash/chunk.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace unhash::chunk;
using std::cout;
using std::endl;

TEST(ChunkTest, Addition)
{
    tchunk c = {{}, {1, 2, 3, 1234 + (1U << 31)}};
    unhash::offset::off o = {0, 1, 5, 1U << 31};
    tchunk result = c + o;
    cout << result << endl;
    tchunk expected = {{0, 1, 5, 1U << 31}, {1, 3, 8, 1234}};
    ASSERT_EQ(result, expected);
}

TEST(ChunkTest, Multiplication)
{
    tchunk c = {{}, {1, 2, 3, 1234 + (1U << 31)}};
    orow o = unhash::offset::OFFSETS[0];
    tchunk result = c * o;
    // cout << result << endl;
    ASSERT_EQ(result.size(), 2 * 37);
}

TEST(ChunkTest, BuildChunk)
{
    chunk c = build_chunk(2);
    std::cout << c.size() << std::endl;
    // std::cout << c << std::endl;
    ASSERT_EQ(c.size(), 37 * 37);
}

TEST(ChunkTest, SaveLoadChunk)
{
    chunk c = build_chunk(0, 2);
    save_chunk("data/chunk_l_2.bin", c);
    chunk cc = load_chunk("data/chunk_l_2.bin");
    std::cout << c.size() << ' ' << cc.size() << std::endl;
    ASSERT_EQ(c, cc);
}