#include <cfloat>
#include <gtest/gtest.h>
#include <iostream>
#include "unhash/offset.hpp"

using namespace unhash::offset;
using std::cout;
using std::endl;

TEST(OffsetTest, Pack)
{
    off o = pack({333356628U, 1324207265U, 1401060374U, 1672518790U});
    cout << o << endl;
    off expected = {715877907U, 2843713448U, 3008754243U, 3591706752U};
    EXPECT_EQ(o, expected);
}

TEST(OffsetTest, OffsetTable)
{
    cout << OFFSETS << endl;
    EXPECT_EQ(OFFSETS[0].size(), 37);
}