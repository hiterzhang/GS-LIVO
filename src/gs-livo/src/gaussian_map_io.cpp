#include "gaussian_map_io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace
{
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kScaleFloor = 1e-8f;

float rgbToSh(float value)
{
  const float normalized = std::clamp(value / 255.0f, 0.0f, 1.0f);
  return (normalized - 0.5f) / kShC0;
}
}  // namespace

GaussianPlyWriteResult writeGaussianPly(
    const std::filesystem::path &path,
    const std::vector<GS_point> &points)
{
  GaussianPlyWriteResult result;
  result.count = points.size();

  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
  {
    result.error = error.message();
    return result;
  }

  std::ofstream output(path);
  if (!output)
  {
    result.error = "failed to open Gaussian PLY output";
    return result;
  }

  output << "ply\nformat ascii 1.0\n";
  output << "element vertex " << points.size() << "\n";
  output << "property float x\nproperty float y\nproperty float z\n";
  output << "property float nx\nproperty float ny\nproperty float nz\n";
  output << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
  output << "property float opacity\n";
  output << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
  output << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
  output << "end_header\n";
  output << std::setprecision(9);

  for (const auto &point : points)
  {
    output << point._points.x << ' ' << point._points.y << ' ' << point._points.z << ' '
           << point._normals.x << ' ' << point._normals.y << ' ' << point._normals.z << ' '
           << rgbToSh(point._colors.r) << ' ' << rgbToSh(point._colors.g) << ' '
           << rgbToSh(point._colors.b) << ' ' << point._opacity << ' '
           << std::log(std::max(point._distance.r1, kScaleFloor)) << ' '
           << std::log(std::max(point._distance.r2, kScaleFloor)) << ' '
           << std::log(std::max(point._distance.r3, kScaleFloor)) << ' '
           << point._quaternion.qw << ' ' << point._quaternion.qx << ' '
           << point._quaternion.qy << ' ' << point._quaternion.qz << '\n';
  }

  if (!output.good())
  {
    result.error = "failed while writing Gaussian PLY";
    return result;
  }
  result.ok = true;
  return result;
}
