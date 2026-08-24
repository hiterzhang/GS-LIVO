cmake_minimum_required(VERSION 3.18)
project(simple_knn LANGUAGES CUDA CXX)

add_library(simple-knn SHARED simple_knn.cu simple_knn.h)
target_include_directories(simple-knn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(simple-knn PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES native
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
