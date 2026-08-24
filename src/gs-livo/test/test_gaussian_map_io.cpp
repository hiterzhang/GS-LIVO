#include "gaussian_map_io.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

TEST(GaussianMapIo, WritesRequiredThreeDimensionalGaussianProperties)
{
  GS_point point{};
  point._points = {1.0f, 2.0f, 3.0f};
  point._normals = {0.0f, 0.0f, 1.0f};
  point._distance = {0.1f, 0.2f, 0.3f};
  point._quaternion = {1.0f, 0.0f, 0.0f, 0.0f};
  point._colors = {255.0f, 128.0f, 0.0f};
  point._opacity = 1.0f;

  const auto path = std::filesystem::temp_directory_path() / "gs_livo_gaussian_io_test.ply";
  const auto result = writeGaussianPly(path, std::vector<GS_point>{point});
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.count, 1u);

  std::ifstream input(path);
  std::stringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();
  EXPECT_NE(text.find("element vertex 1"), std::string::npos);
  EXPECT_NE(text.find("property float f_dc_0"), std::string::npos);
  EXPECT_NE(text.find("property float opacity"), std::string::npos);
  EXPECT_NE(text.find("property float scale_2"), std::string::npos);
  EXPECT_NE(text.find("property float rot_3"), std::string::npos);

  std::filesystem::remove(path);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
