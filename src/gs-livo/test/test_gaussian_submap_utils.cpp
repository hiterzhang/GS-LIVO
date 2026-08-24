#include <gtest/gtest.h>

#include "gaussian_submap_utils.h"

TEST(GaussianSubmapUtils, KeepsSingletonWithoutModuloByZero)
{
  EXPECT_EQ(1u, gaussianKeepCount(1u, 42u));
}

TEST(GaussianSubmapUtils, PreservesExistingRandomHalfSelection)
{
  EXPECT_EQ(2u, gaussianKeepCount(10u, 7u));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
