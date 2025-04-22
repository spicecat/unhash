#include "unhash/hash.hpp"
#include <gtest/gtest.h>

using namespace unhash::hash;
using std::cout;
using std::endl;

TEST(HashTest, Encode)
{
    enc encoding = encode("hello world!");
    cout << encoding << endl;
    enc expected = {7, 4, 11, 11, 14, 36, 22, 14, 17, 11, 3};
    EXPECT_EQ(encoding, expected);
}

TEST(HashTest, Contributions)
{
    cout << contribution(0, 0, 4) << endl;
    cout << contribution(0, 123, 36) << endl;
    cout << contribution(1, 123, 36) << endl;
    EXPECT_EQ(contribution(0, 0, 4), 1243626442U);
    EXPECT_EQ(contribution(0, 123, 36), 1510565163U);
    EXPECT_EQ(contribution(1, 123, 36), 1135454576U);
}

TEST(HashTest, Hashify)
{
    h hash = hashify("hello world!");
    cout << hash << endl;
    h expected = {69554U, 35256U, 40190U, 1363U};
    for (seg s = 0; s < SEGMENTS; ++s)
        EXPECT_EQ(hash[s], expected[s]);
}