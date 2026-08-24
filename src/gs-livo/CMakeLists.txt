cmake_minimum_required(VERSION 2.8.3)
project(fast_livo CUDA CXX)

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found! Install CUDA Toolkit.")
endif()
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)

set(PYTHON_EXECUTABLE /usr/bin/python3)
find_package(PythonLibs REQUIRED)

set(GS_LIVO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "GS-LIVO repository root")
set(LIBTORCH_DIR "${GS_LIVO_REPO_ROOT}/third_party/libtorch" CACHE PATH "LibTorch root")
set(GS_LIVO_LIB3DGS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../lib3dgs" CACHE PATH "lib3dgs source root")
set(GS_LIVO_LIB3DGS_LIBRARY "${GS_LIVO_LIB3DGS_ROOT}/build/lib3dgs_lib.so" CACHE FILEPATH "lib3dgs shared library")
list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

# Specific settings for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Detect CPU architecture
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    # 32-bit ARM optimizations with NEON support
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops") #-flto
  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops") 
  add_definitions(-DX86_ARCH)
endif()

# Define project root directory
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Detect CPU core count for potential multithreading optimization
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

# Set the number of cores for multithreading
if(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
else()
  add_definitions(-DMP_PROC_NUM=1)
  message(STATUS "Single core detected. Multithreading disabled.")
endif()

set(PYTHON_EXECUTABLE /usr/bin/python3)


include_directories(${LIBTORCH_DIR}/include)
link_directories(${LIBTORCH_DIR}/lib)

set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS "${LIBTORCH_DIR}/share/cmake/Torch" NO_DEFAULT_PATH)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")  # 添加Torch标志
include_directories(${TORCH_INCLUDE_DIRS})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  #${JSONCPP_LIBRARIES}
  include)


find_package(PythonLibs REQUIRED)

# Check for OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
else()
  message(STATUS "OpenMP not found, proceeding without it")
endif()

# Check for mimalloc support
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

# Find catkin and required dependencies
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

set(Sophus_LIBRARIES libSophus.so)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

# Include directories for dependencies
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GS_LIVO_LIB3DGS_ROOT}/includes
  ${GS_LIVO_LIB3DGS_ROOT}/cuda_rasterizer
  ${CMAKE_CURRENT_SOURCE_DIR}/../simple-knn
  ${CMAKE_CURRENT_SOURCE_DIR}/../tinyply/source
  ${CMAKE_CURRENT_SOURCE_DIR}/../json/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../glm
  ${LIBTORCH_DIR}/include
  ${LIBTORCH_DIR}/include/torch/csrc/api/include
  include
)

# Add libraries
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp src/gaussian_map_io.cpp)

target_link_libraries(vio
  ${GS_LIVO_LIB3DGS_LIBRARY}
  ${TORCH_LIBRARIES}
)

target_include_directories(vio PUBLIC
  ${GS_LIVO_LIB3DGS_ROOT}/includes
  ${GS_LIVO_LIB3DGS_ROOT}/cuda_rasterizer
  ${CMAKE_CURRENT_SOURCE_DIR}/../simple-knn
  ${CMAKE_CURRENT_SOURCE_DIR}/../tinyply/source
  ${LIBTORCH_DIR}/include
  ${LIBTORCH_DIR}/include/torch/csrc/api/include)

set_property(TARGET vio APPEND PROPERTY BUILD_RPATH "${LIBTORCH_DIR}/lib;${GS_LIVO_LIB3DGS_ROOT}/build")


add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)

# Add the main executable
add_executable(fastlivo_mapping src/main.cpp)

# Link libraries to the executable
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${TORCH_LIBRARIES}  # 添加
)

set_property(TARGET fastlivo_mapping APPEND PROPERTY BUILD_RPATH "${LIBTORCH_DIR}/lib;${GS_LIVO_LIB3DGS_ROOT}/build")



# Link mimalloc if found
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_gaussian_map_io test/test_gaussian_map_io.cpp src/gaussian_map_io.cpp)
  if(TARGET test_gaussian_map_io)
    target_include_directories(test_gaussian_map_io PRIVATE include ${GS_LIVO_LIB3DGS_ROOT}/includes)
  endif()
endif()
