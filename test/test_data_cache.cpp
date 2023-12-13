#include "gtest/gtest.h"
#include <DataCache.hpp>

using namespace EllipticForest;

TEST(DataCache, init_and_operator) {

    DataCache<int> cache;

    int var1 = 1;
    int var2 = 2;
    cache["var1"] = 1;
    cache["var2"] = 2;

    EXPECT_EQ(var1, cache["var1"]);
    EXPECT_EQ(var2, cache["var2"]);

}

TEST(DataCache, contains) {

    DataCache<int> cache;

    int var1 = 1;
    int var2 = 2;
    cache["var1"] = 1;
    cache["var2"] = 2;

    EXPECT_TRUE(cache.contains("var1"));
    EXPECT_TRUE(cache.contains("var2"));
    EXPECT_FALSE(cache.contains("var3"));

}