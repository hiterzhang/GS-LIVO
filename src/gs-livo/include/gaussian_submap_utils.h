#pragma once

#include <cstddef>

inline std::size_t gaussianKeepCount(
    const std::size_t submap_size,
    const unsigned int random_value)
{
  if (submap_size < 2) return submap_size;
  const std::size_t half_size = submap_size / 2;
  return random_value % half_size;
}
