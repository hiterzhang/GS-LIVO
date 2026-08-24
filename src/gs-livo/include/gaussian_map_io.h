#pragma once

#include "point_cloud.cuh"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

struct GaussianPlyWriteResult
{
  bool ok = false;
  std::size_t count = 0;
  std::string error;
};

GaussianPlyWriteResult writeGaussianPly(
    const std::filesystem::path &path,
    const std::vector<GS_point> &points);
