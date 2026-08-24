#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_jobs=${GS_LIVO_BUILD_JOBS:-4}

if [[ ${1:-} == "--dry-run" ]]; then
  printf 'cmake lib3dgs with CUDA 11.8 and local LibTorch\n'
  printf 'catkin_make -j%s\n' "$build_jobs"
  exit 0
fi

source /opt/ros/noetic/setup.bash
export CUDACXX=/usr/local/cuda-11.8/bin/nvcc
mkdir -p "$repo_root/build_logs"
build_log="$repo_root/build_logs/$(date +%Y%m%d-%H%M%S).log"
trap 'cp "$build_log" "$repo_root/build_logs/latest.log"' EXIT
exec > >(tee "$build_log") 2>&1
"$repo_root/tools/bootstrap_dependencies.sh"

cmake -S "$repo_root/src/lib3dgs" -B "$repo_root/src/lib3dgs/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGS_LIVO_REPO_ROOT="$repo_root" \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$repo_root/src/lib3dgs/build" --parallel "$build_jobs"

if [[ ! -e "$repo_root/src/CMakeLists.txt" ]]; then
  (cd "$repo_root/src" && catkin_init_workspace)
fi

(cd "$repo_root" && catkin_make \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGS_LIVO_REPO_ROOT="$repo_root" \
  -DSophus_INCLUDE_DIRS=/usr/local/include \
  -DSophus_LIBRARIES=/usr/local/lib/libSophus.so \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -j"$build_jobs")
