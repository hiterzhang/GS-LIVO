# GS-LIVO HKairport01 Native Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the published GS-LIVO code natively on `HKairport01.bag`, producing a validated trajectory, colored point clouds, Gaussian PLY map, logs, metadata, and visualization artifacts.

**Architecture:** Keep the upstream ROS/catkin source layout and pin all non-system dependencies locally under `third_party/`. Apply narrowly scoped build-path and output-observability repairs, add a dedicated MARS-LVIG launch/configuration, and use wrapper tools to execute short, medium, and complete runs into versioned result directories on the data disk.

**Tech Stack:** Ubuntu 20.04, ROS Noetic/catkin, C++17, CUDA 11.8, LibTorch C++ API, OpenCV, PCL, Eigen, Sophus, rpg_vikit, Bash, Python 3 standard library, RViz.

---

## File Structure

The implementation creates or changes these units:

- `.gitignore`: excludes local dependencies, catkin outputs, run links, and generated reports.
- `tools/check_environment.py`: read-only preflight checks and machine-readable environment report.
- `tools/bootstrap_dependencies.sh`: pins and installs rpg_vikit and LibTorch locally.
- `tools/build_native.sh`: builds lib3dgs and the catkin workspace with reproducible paths.
- `tools/run_hkairport01.sh`: orchestrates launch, rosbag playback, logging, telemetry, cleanup, and validation.
- `tools/write_manifest.py`: writes run metadata as JSON-compatible YAML.
- `tools/validate_artifacts.py`: validates trajectory, PCD, Gaussian PLY, images, and fatal-log patterns.
- `tools/visualize_gaussian_ply.py`: creates an interactive HTML view of exported Gaussian centers and colors.
- `tests/`: Python unit tests for preflight, configuration, wrappers, manifests, and validators.
- `src/simple-knn/CMakeLists.txt`: provides the missing native CUDA library target.
- `src/lib3dgs/CMakeLists.txt`: consumes project-local dependencies and removes `/usr/local` assumptions.
- `src/gs-livo/CMakeLists.txt`: links the published lib3dgs tree by relative/configurable path.
- `src/gs-livo/config/HKairport01.yaml`: MARS-LVIG sensor and GS-LIVO runtime parameters.
- `src/gs-livo/config/camera_MARS_LVIG.yaml`: official HKairport camera model.
- `src/gs-livo/launch/mapping_hkairport01.launch`: dataset-specific launch entry point.
- `src/gs-livo/rviz_cfg/HKairport01.rviz`: focused path, odometry, RGB cloud, and image visualization.
- `src/gs-livo/include/gaussian_map_io.h` and `src/gs-livo/src/gaussian_map_io.cpp`: standalone standard 3DGS-style PLY writer.
- `src/gs-livo/test/test_gaussian_map_io.cpp`: C++ regression test for PLY schema and values.
- `src/gs-livo/include/vio.h` and `src/gs-livo/src/vio.cpp`: expose a snapshot of the published global/in-window Gaussian state and publish rendered images.
- `src/gs-livo/include/LIVMapper.h` and `src/gs-livo/src/LIVMapper.cpp`: configurable output roots and graceful artifact flushing.
- `REPRODUCTION.md`: final commands, configuration deviations, results, and known upstream limitations.

### Task 1: Add repository hygiene and an environment preflight

**Files:**
- Create: `.gitignore`
- Create: `tools/check_environment.py`
- Create: `tests/test_environment.py`

- [ ] **Step 1: Write the failing parser and required-topic tests**

Create `tests/test_environment.py`:

```python
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_environment", ROOT / "tools" / "check_environment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EnvironmentCheckTests(unittest.TestCase):
    def test_parse_rosbag_topics(self):
        text = """
topics:
    - topic: /left_camera/image/compressed
      type: sensor_msgs/CompressedImage
    - topic: /livox/imu
      type: sensor_msgs/Imu
    - topic: /livox/lidar
      type: livox_ros_driver/CustomMsg
"""
        self.assertEqual(
            MODULE.parse_rosbag_topics(text),
            {
                "/left_camera/image/compressed",
                "/livox/imu",
                "/livox/lidar",
            },
        )

    def test_missing_required_topics(self):
        missing = MODULE.missing_required_topics({"/livox/imu"})
        self.assertEqual(
            missing,
            ["/left_camera/image/compressed", "/livox/lidar"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_environment -v
```

Expected: `FileNotFoundError` for `tools/check_environment.py`.

- [ ] **Step 3: Implement the preflight tool**

Create `tools/check_environment.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys


REQUIRED_TOPICS = {
    "/left_camera/image/compressed",
    "/livox/imu",
    "/livox/lidar",
}
REQUIRED_COMMANDS = (
    "cmake",
    "g++",
    "git",
    "nvidia-smi",
    "nvcc",
    "rosbag",
    "roscore",
    "roslaunch",
)


def parse_rosbag_topics(text):
    return set(re.findall(r"^\s*- topic: (\S+)$", text, re.MULTILINE))


def missing_required_topics(topics):
    return sorted(REQUIRED_TOPICS - set(topics))


def command_output(command):
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.returncode, completed.stdout.strip()


def disk_free_gib(path):
    return round(shutil.disk_usage(path).free / (1024 ** 3), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/media/zzh/data/LVIO_and_LVIO_GS/HKairport01.bag",
    )
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    dataset = pathlib.Path(args.dataset)
    commands = {name: shutil.which(name) for name in REQUIRED_COMMANDS}
    bag_code, bag_info = command_output(["rosbag", "info", "--yaml", str(dataset)])
    topics = parse_rosbag_topics(bag_info) if bag_code == 0 else set()
    gpu_code, gpu_info = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )

    report = {
        "dataset": {
            "path": str(dataset),
            "exists": dataset.is_file(),
            "size_bytes": dataset.stat().st_size if dataset.is_file() else 0,
            "topics": sorted(topics),
            "missing_topics": missing_required_topics(topics),
        },
        "commands": commands,
        "gpu": {"ok": gpu_code == 0, "description": gpu_info},
        "disk_free_gib": {
            "workspace": disk_free_gib(pathlib.Path(__file__).resolve().parents[1]),
            "data": disk_free_gib(dataset.parent),
        },
    }
    report["ok"] = all(commands.values()) and all(
        (
            report["dataset"]["exists"],
            not report["dataset"]["missing_topics"],
            report["gpu"]["ok"],
            report["disk_free_gib"]["workspace"] >= 20,
            report["disk_free_gib"]["data"] >= 50,
        )
    )

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        pathlib.Path(args.output).write_text(payload, encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add generated-file exclusions**

Create `.gitignore`:

```gitignore
/build/
/devel/
/third_party/
/build_logs/
/results
/preflight.json
/src/CMakeLists.txt
/src/gs-livo/Log/
__pycache__/
*.pyc
```

- [ ] **Step 5: Run unit tests and the real preflight**

Run:

```bash
python3 -m unittest tests.test_environment -v
python3 tools/check_environment.py --output preflight.json
python3 -m json.tool preflight.json >/dev/null
```

Expected: 2 tests pass; the preflight exits `0`; `preflight.json` reports all three required topics and an RTX 4060.

- [ ] **Step 6: Commit the preflight**

```bash
git add .gitignore tools/check_environment.py tests/test_environment.py
git commit -m "test: add GS-LIVO environment preflight"
```

### Task 2: Bootstrap pinned dependencies and repair native build integration

**Files:**
- Create: `tests/test_build_tooling.py`
- Create: `tools/bootstrap_dependencies.sh`
- Create: `tools/build_native.sh`
- Create: `src/simple-knn/CMakeLists.txt`
- Modify: `src/lib3dgs/CMakeLists.txt`
- Modify: `src/gs-livo/CMakeLists.txt`

- [ ] **Step 1: Write failing static build-tool tests**

Create `tests/test_build_tooling.py`:

```python
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class BuildToolingTests(unittest.TestCase):
    def test_bootstrap_pins_dependencies(self):
        text = (ROOT / "tools" / "bootstrap_dependencies.sh").read_text()
        self.assertIn("6c886c8e5d83997806e00294826d528cea3581dd", text)
        self.assertIn("libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip", text)

    def test_cmake_has_no_author_absolute_path(self):
        text = (ROOT / "src" / "gs-livo" / "CMakeLists.txt").read_text()
        self.assertNotIn("/home/sheng", text)
        self.assertIn("GS_LIVO_LIB3DGS_ROOT", text)

    def test_lib3dgs_uses_local_source_targets(self):
        text = (ROOT / "src" / "lib3dgs" / "CMakeLists.txt").read_text()
        self.assertIn("add_subdirectory(${SIMPLE_KNN_SOURCE_DIR}", text)
        self.assertIn("add_subdirectory(${TINYPLY_SOURCE_DIR}", text)
        self.assertNotIn('set(SIMPLE_KNN_DIR "/usr/local")', text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_build_tooling -v
```

Expected: missing bootstrap script and assertions exposing `/home/sheng` and `/usr/local`.

- [ ] **Step 3: Create the missing simple-knn native target**

Create `src/simple-knn/CMakeLists.txt`:

```cmake
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
```

- [ ] **Step 4: Replace lib3dgs path assumptions with project-local targets**

In `src/lib3dgs/CMakeLists.txt`, remove the `/usr/local` simple-knn block and the relative `./libtorch` block. Add these cache paths immediately after `project(...)`:

```cmake
set(GS_LIVO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "GS-LIVO repository root")
set(LIBTORCH_DIR "${GS_LIVO_REPO_ROOT}/third_party/libtorch" CACHE PATH "LibTorch root")
set(SIMPLE_KNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../simple-knn" CACHE PATH "simple-knn source")
set(TINYPLY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tinyply" CACHE PATH "tinyply source")
set(NLOHMANN_JSON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../json" CACHE PATH "nlohmann-json source")
set(GLM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../glm" CACHE PATH "GLM source")
set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../eigen" CACHE PATH "Eigen source")

list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
add_subdirectory(${SIMPLE_KNN_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/simple-knn)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SHARED_LIB ON CACHE BOOL "" FORCE)
add_subdirectory(${TINYPLY_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/tinyply)
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${NLOHMANN_JSON_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/json)
add_subdirectory(${GLM_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/glm)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
add_subdirectory(${EIGEN_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/eigen)
```

Replace the Torch lookup with:

```cmake
find_package(Torch REQUIRED PATHS "${LIBTORCH_DIR}/share/cmake/Torch" NO_DEFAULT_PATH)
```

Delete the later `find_package(tinyply REQUIRED)`, both `find_package(Eigen3 ...)`, `find_package(nlohmann_json REQUIRED)`, and `find_package(glm REQUIRED)` calls. Keep the existing `target_link_libraries(3dgs_lib ...)` list, because the locally added targets provide `simple-knn`, `tinyply`, `Eigen3::Eigen`, `nlohmann_json::nlohmann_json`, and `glm::glm`.

Replace the existing lib3dgs include directory line with:

```cmake
target_include_directories(3dgs_lib PRIVATE
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/cuda_rasterizer
  ${TINYPLY_SOURCE_DIR}/source)
```

- [ ] **Step 5: Repair the ROS package CMake layout**

In `src/gs-livo/CMakeLists.txt`:

1. Delete `add_subdirectory(rpg_vikit)` because rpg_vikit is a sibling catkin source package.
2. Delete the `SIMPLE_KNN_DIR "/usr/local"`, `include_directories(${SIMPLE_KNN_DIR}...)`, and `link_directories(${SIMPLE_KNN_DIR}...)` block.
3. Replace both `LIBTORCH_DIR "./libtorch"` assignments with:

```cmake
set(GS_LIVO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "GS-LIVO repository root")
set(LIBTORCH_DIR "${GS_LIVO_REPO_ROOT}/third_party/libtorch" CACHE PATH "LibTorch root")
set(GS_LIVO_LIB3DGS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../lib3dgs" CACHE PATH "lib3dgs source root")
set(GS_LIVO_LIB3DGS_LIBRARY "${GS_LIVO_LIB3DGS_ROOT}/build/lib3dgs_lib.so" CACHE FILEPATH "lib3dgs shared library")
list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
```

4. Replace `find_package(Torch REQUIRED)` with:

```cmake
find_package(Torch REQUIRED PATHS "${LIBTORCH_DIR}/share/cmake/Torch" NO_DEFAULT_PATH)
```

5. Replace the `vio` target link/include blocks with:

```cmake
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)

target_link_libraries(vio
  ${GS_LIVO_LIB3DGS_LIBRARY}
  ${TORCH_LIBRARIES})

target_include_directories(vio PUBLIC
  ${GS_LIVO_LIB3DGS_ROOT}/includes
  ${GS_LIVO_LIB3DGS_ROOT}/cuda_rasterizer
  ${CMAKE_CURRENT_SOURCE_DIR}/../tinyply/source
  ${LIBTORCH_DIR}/include
  ${LIBTORCH_DIR}/include/torch/csrc/api/include)
```

6. Add runtime search paths:

```cmake
set_property(TARGET vio APPEND PROPERTY BUILD_RPATH "${LIBTORCH_DIR}/lib;${GS_LIVO_LIB3DGS_ROOT}/build")
set_property(TARGET fastlivo_mapping APPEND PROPERTY BUILD_RPATH "${LIBTORCH_DIR}/lib;${GS_LIVO_LIB3DGS_ROOT}/build")
```

- [ ] **Step 6: Implement pinned dependency bootstrap**

Create `tools/bootstrap_dependencies.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
third_party_dir="$repo_root/third_party"
download_dir="$third_party_dir/downloads"
vikit_dir="$repo_root/src/rpg_vikit"
vikit_commit="6c886c8e5d83997806e00294826d528cea3581dd"
libtorch_url="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
libtorch_zip="$download_dir/libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip"

if [[ ${1:-} == "--dry-run" ]]; then
  printf 'rpg_vikit=%s@%s\n' "https://github.com/xuankuzcr/rpg_vikit.git" "$vikit_commit"
  printf 'libtorch=%s\n' "$libtorch_url"
  exit 0
fi

mkdir -p "$download_dir"

if [[ ! -d "$vikit_dir/.git" ]]; then
  git clone https://github.com/xuankuzcr/rpg_vikit.git "$vikit_dir"
fi
git -C "$vikit_dir" fetch --depth=1 origin "$vikit_commit"
git -C "$vikit_dir" checkout --detach "$vikit_commit"

if [[ ! -d "$third_party_dir/libtorch" ]]; then
  if [[ ! -f "$libtorch_zip" ]]; then
    curl --fail --location --retry 3 --output "$libtorch_zip" "$libtorch_url"
  fi
  unzip -q "$libtorch_zip" -d "$third_party_dir"
fi

sha256sum "$libtorch_zip" > "$libtorch_zip.sha256"
git -C "$vikit_dir" rev-parse HEAD
```

- [ ] **Step 7: Implement the two-stage native build script**

Create `tools/build_native.sh`:

```bash
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
mkdir -p "$repo_root/build_logs"
build_log="$repo_root/build_logs/$(date +%Y%m%d-%H%M%S).log"
trap 'cp "$build_log" "$repo_root/build_logs/latest.log"' EXIT
exec > >(tee "$build_log") 2>&1
"$repo_root/tools/bootstrap_dependencies.sh"

cmake -S "$repo_root/src/lib3dgs" -B "$repo_root/src/lib3dgs/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGS_LIVO_REPO_ROOT="$repo_root" \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$repo_root/src/lib3dgs/build" --parallel "$build_jobs"

if [[ ! -e "$repo_root/src/CMakeLists.txt" ]]; then
  (cd "$repo_root/src" && catkin_init_workspace)
fi

(cd "$repo_root" && catkin_make \
  -DCMAKE_BUILD_TYPE=Release \
  -DGS_LIVO_REPO_ROOT="$repo_root" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -j"$build_jobs")
```

Make both scripts executable using `chmod +x tools/bootstrap_dependencies.sh tools/build_native.sh`.

- [ ] **Step 8: Run static tests and dry runs**

Run:

```bash
python3 -m unittest tests.test_build_tooling -v
tools/bootstrap_dependencies.sh --dry-run
tools/build_native.sh --dry-run
git diff --check
```

Expected: 3 tests pass; dry-run output contains the pinned commit, LibTorch URL, and `catkin_make -j4`.

- [ ] **Step 9: Commit build integration**

```bash
git add tests/test_build_tooling.py tools/bootstrap_dependencies.sh tools/build_native.sh src/simple-knn/CMakeLists.txt src/lib3dgs/CMakeLists.txt src/gs-livo/CMakeLists.txt
git commit -m "build: support native project-local GS-LIVO dependencies"
```

### Task 3: Add the HKairport01 configuration and launch entry point

**Files:**
- Create: `tests/test_hkairport_config.py`
- Create: `src/gs-livo/config/HKairport01.yaml`
- Create: `src/gs-livo/config/camera_MARS_LVIG.yaml`
- Create: `src/gs-livo/launch/mapping_hkairport01.launch`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_hkairport_config.py`:

```python
import pathlib
import unittest
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class HKairportConfigTests(unittest.TestCase):
    def test_sensor_topics_and_time_offset(self):
        config = yaml.safe_load(
            (ROOT / "src/gs-livo/config/HKairport01.yaml").read_text()
        )
        self.assertEqual(config["common"]["img_topic"], "/left_camera/image")
        self.assertEqual(config["common"]["lid_topic"], "/livox/lidar")
        self.assertEqual(config["common"]["imu_topic"], "/livox/imu")
        self.assertEqual(config["time_offset"]["img_time_offset"], 0.1)
        self.assertEqual(config["evo"]["seq_name"], "HKairport01")
        self.assertTrue(config["evo"]["pose_output_en"])
        self.assertTrue(config["pcd_save"]["pcd_save_en"])
        self.assertTrue(config["output"]["gaussian_save_en"])

    def test_camera_intrinsics(self):
        camera = yaml.safe_load(
            (ROOT / "src/gs-livo/config/camera_MARS_LVIG.yaml").read_text()
        )
        self.assertEqual(camera["cam_width"], 2448)
        self.assertEqual(camera["cam_height"], 2048)
        self.assertEqual(camera["scale"], 0.25)
        self.assertAlmostEqual(camera["cam_fx"], 1444.431662789634)
        self.assertAlmostEqual(camera["cam_cy"], 1043.601026568268)

    def test_launch_loads_dataset_files(self):
        text = (ROOT / "src/gs-livo/launch/mapping_hkairport01.launch").read_text()
        self.assertIn("HKairport01.yaml", text)
        self.assertIn("camera_MARS_LVIG.yaml", text)
        self.assertIn("output/root_dir", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_hkairport_config -v
```

Expected: three errors for missing configuration and launch files.

- [ ] **Step 3: Add the official camera model**

Create `src/gs-livo/config/camera_MARS_LVIG.yaml`:

```yaml
cam_model: Pinhole
cam_width: 2448
cam_height: 2048
scale: 0.25
cam_fx: 1444.431662789634
cam_fy: 1444.343536688358
cam_cx: 1177.801079401826
cam_cy: 1043.601026568268
cam_d0: -0.05729528706141188
cam_d1: 0.1210407244166642
cam_d2: 0.001274128378760289
cam_d3: 0.0004389741530109464
```

- [ ] **Step 4: Add the HKairport01 runtime configuration**

Create `src/gs-livo/config/HKairport01.yaml`:

```yaml
common:
  img_topic: "/left_camera/image"
  lid_topic: "/livox/lidar"
  imu_topic: "/livox/imu"
  img_en: 1
  lidar_en: 1
  ros_driver_bug_fix: false

extrin_calib:
  extrinsic_T: [0.04165, 0.02326, -0.0284]
  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  Rcl: [0.00438814, -0.999807, -0.0191582,
        -0.00978695, 0.0191145, -0.999769,
        0.999942, 0.00457463, -0.00970118]
  Pcl: [0.016069, 0.0871753, -0.0718021]

time_offset:
  imu_time_offset: 0.0
  img_time_offset: 0.1
  exposure_time_init: 0.0

preprocess:
  point_filter_num: 1
  filter_size_surf: 0.1
  lidar_type: 1
  scan_line: 6
  blind: 0.8

vio:
  max_iterations: 5
  outlier_threshold: 1000
  img_point_cov: 1000
  patch_size: 8
  patch_pyrimid_level: 4
  normal_en: true
  raycast_en: false
  inverse_composition_en: false
  exposure_estimate_en: true
  inv_expo_cov: 0.1

imu:
  imu_en: true
  imu_int_frame: 30
  acc_cov: 2.0
  gyr_cov: 0.1
  b_acc_cov: 0.0001
  b_gyr_cov: 0.0001

lio:
  max_iterations: 5
  dept_err: 0.02
  beam_err: 0.05
  min_eigen_value: 0.005
  voxel_size: 2.0
  max_layer: 2
  max_points_num: 50
  layer_init_num: [5, 5, 5, 5, 5]

local_map:
  map_sliding_en: false
  half_map_size: 100
  sliding_thresh: 8

uav:
  imu_rate_odom: false
  gravity_align_en: false

publish:
  dense_map_en: true
  pub_effect_point_en: false
  pub_plane_en: false
  pub_scan_num: 1
  blind_rgb_points: 0.0

evo:
  seq_name: "HKairport01"
  pose_output_en: true

pcd_save:
  pcd_save_en: true
  colmap_output_en: false
  filter_size_pcd: 0.15
  interval: -1

output:
  root_dir: "/tmp/gs-livo-hkairport01"
  gaussian_save_en: true
  visualization_save_en: true

gs:
  map_voxel_size: 3.0
  normal_rejecter: 0.0
  gs_iterations: 40
  border_gs: 4
  plot_gs_render: 1
  gs_position_lr: 0.01
  gs_feature_lr: 0.01
  gs_opacity_lr: 0.1
  gs_scaling_lr: 1.8
  gs_rotation_lr: 0.01

scale_factor: 0.035
root_voxel_size: 0.01
octree_max_level: 3
```

- [ ] **Step 5: Add the dataset launch file**

Create `src/gs-livo/launch/mapping_hkairport01.launch`:

```xml
<launch>
  <arg name="rviz" default="true" />
  <arg name="output_root" default="/tmp/gs-livo-hkairport01" />

  <rosparam command="load" file="$(find fast_livo)/config/HKairport01.yaml" />
  <param name="output/root_dir" value="$(arg output_root)" />

  <node pkg="fast_livo" type="fastlivo_mapping" name="laserMapping" output="screen">
    <rosparam file="$(find fast_livo)/config/camera_MARS_LVIG.yaml" />
  </node>

  <group if="$(arg rviz)">
    <node pkg="rviz" type="rviz" name="rviz"
          args="-d $(find fast_livo)/rviz_cfg/HKairport01.rviz" />
  </group>

  <node pkg="image_transport" type="republish" name="republish_hkairport01"
        args="compressed in:=/left_camera/image raw out:=/left_camera/image"
        output="screen" respawn="true" />
</launch>
```

- [ ] **Step 6: Run the configuration tests**

Run:

```bash
python3 -m unittest tests.test_hkairport_config -v
```

Expected: 3 tests pass.

- [ ] **Step 7: Commit dataset configuration**

```bash
git add tests/test_hkairport_config.py src/gs-livo/config/HKairport01.yaml src/gs-livo/config/camera_MARS_LVIG.yaml src/gs-livo/launch/mapping_hkairport01.launch
git commit -m "feat: add HKairport01 dataset configuration"
```

### Task 4: Add a tested Gaussian-map exporter and configurable outputs

**Files:**
- Create: `src/gs-livo/include/gaussian_map_io.h`
- Create: `src/gs-livo/src/gaussian_map_io.cpp`
- Create: `src/gs-livo/test/test_gaussian_map_io.cpp`
- Modify: `src/gs-livo/include/vio.h`
- Modify: `src/gs-livo/src/vio.cpp`
- Modify: `src/gs-livo/include/LIVMapper.h`
- Modify: `src/gs-livo/src/LIVMapper.cpp`
- Modify: `src/gs-livo/CMakeLists.txt`

- [ ] **Step 1: Write the failing C++ PLY regression test**

Create `src/gs-livo/test/test_gaussian_map_io.cpp`:

```cpp
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
```

- [ ] **Step 2: Add the test target and verify compilation fails**

Append to `src/gs-livo/CMakeLists.txt`:

```cmake
if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_gaussian_map_io test/test_gaussian_map_io.cpp src/gaussian_map_io.cpp)
  if(TARGET test_gaussian_map_io)
    target_include_directories(test_gaussian_map_io PRIVATE include ${GS_LIVO_LIB3DGS_ROOT}/includes)
  endif()
endif()
```

Run after dependencies are bootstrapped:

```bash
tools/build_native.sh
```

Expected: configuration or compilation fails because the Gaussian map I/O header and source do not exist.

- [ ] **Step 3: Implement the standalone Gaussian PLY writer**

Create `src/gs-livo/include/gaussian_map_io.h`:

```cpp
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
```

Create `src/gs-livo/src/gaussian_map_io.cpp`:

```cpp
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
```

Append `src/gaussian_map_io.cpp` to the existing `vio` source list in `src/gs-livo/CMakeLists.txt`:

```cmake
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp src/gaussian_map_io.cpp)
```

- [ ] **Step 4: Expose a snapshot of published Gaussian state**

Add to the public method section of `VIOManager` in `src/gs-livo/include/vio.h`:

```cpp
std::vector<GS_point> snapshotGaussianMap();
```

Add to `src/gs-livo/src/vio.cpp`:

```cpp
std::vector<GS_point> VIOManager::snapshotGaussianMap()
{
  std::vector<GS_point> points(sub_GSMap.begin(), sub_GSMap.end());
  if (!gsmap_manager) return points;

  for (auto &entry : gsmap_manager->gs_map_)
  {
    if (entry.second == nullptr) continue;
    std::vector<GS_point *> voxel_points;
    entry.second->get_all_gs_points(voxel_points);
    for (const auto *point : voxel_points)
    {
      if (point != nullptr) points.push_back(*point);
    }
  }
  return points;
}
```

This intentionally exports the data structures that the public code retains. It does not claim to restore unpublished persistent Gaussian optimization.

- [ ] **Step 5: Wire a single configurable GS map voxel size**

Add `double gs_map_voxel_size = 3.0;` to both `LIVMapper` and `VIOManager` fields.

In `LIVMapper.h`, also initialize the existing GS fields so parameter loading never reads indeterminate values:

```cpp
double scale_factor = 3.4;
double scale_factor2 = 3.4;
double normal_rejecter = 0.0;
int save_GS_iter = 0;
```

In `LIVMapper::readParameters`, add:

```cpp
nh.param<double>("gs/map_voxel_size", gs_map_voxel_size, 3.0);
nh.param<double>("gs/normal_rejecter", normal_rejecter, 0.0);
```

In `LIVMapper::initializeComponents`, set:

```cpp
vio_manager->gs_map_voxel_size = gs_map_voxel_size;
vio_manager->border_gs = border_gs;
```

In `VIOManager::initializeVIO`, replace the hard-coded manager construction with:

```cpp
gsmap_manager.reset(new GSMapManager(gs_octree, gs_map_voxel_size, octree_max_level));
```

In `VIOManager::retrieveFrom_GS_Map2`, replace:

```cpp
float voxel_size = root_voxel_size;
```

with:

```cpp
float voxel_size = gsmap_manager->voxel_size_;
```

- [ ] **Step 6: Add configurable output methods to LIVMapper**

In `src/gs-livo/include/LIVMapper.h`, include `<filesystem>` and add:

```cpp
void saveGaussianMap();
void saveVisualizationFrames();
std::filesystem::path outputPath(const std::string &category, const std::string &name) const;

std::string output_root_dir;
bool gaussian_save_en = true;
bool visualization_save_en = true;
```

In `LIVMapper::readParameters`, add:

```cpp
nh.param<std::string>("output/root_dir", output_root_dir, std::string(ROOT_DIR) + "Log");
nh.param<bool>("output/gaussian_save_en", gaussian_save_en, true);
nh.param<bool>("output/visualization_save_en", visualization_save_en, true);
```

Add these implementations to `src/gs-livo/src/LIVMapper.cpp`:

```cpp
std::filesystem::path LIVMapper::outputPath(
    const std::string &category,
    const std::string &name) const
{
  return std::filesystem::path(output_root_dir) / category / name;
}

void LIVMapper::saveGaussianMap()
{
  if (!gaussian_save_en || !vio_manager) return;
  const auto points = vio_manager->snapshotGaussianMap();
  const auto result = writeGaussianPly(
      outputPath("gaussian", "global_gaussians.ply"), points);
  if (!result.ok)
  {
    ROS_ERROR_STREAM("Gaussian map export failed: " << result.error);
    return;
  }
  ROS_INFO_STREAM("Gaussian map saved with " << result.count << " vertices");
}

void LIVMapper::saveVisualizationFrames()
{
  if (!visualization_save_en || !vio_manager) return;
  std::filesystem::create_directories(outputPath("visualization", ""));
  if (!vio_manager->img_undistort.empty())
    cv::imwrite(outputPath("visualization", "input.png").string(), vio_manager->img_undistort);
  if (!vio_manager->img_rendered.empty())
    cv::imwrite(outputPath("visualization", "rendered.png").string(), vio_manager->img_rendered);
}
```

Add `#include "gaussian_map_io.h"` to `LIVMapper.cpp`.

- [ ] **Step 7: Redirect existing trajectory and PCD outputs**

At the start of `LIVMapper::initializeFiles`, create:

```cpp
for (const auto *directory : {"trajectory", "pointcloud", "gaussian", "logs", "visualization"})
  std::filesystem::create_directories(outputPath(directory, ""));
```

Use these replacements in `LIVMapper.cpp`:

```cpp
fout_pre.open(outputPath("logs", "mat_pre.txt"), std::ios::out);
fout_out.open(outputPath("logs", "mat_out.txt"), std::ios::out);
```

```cpp
const auto trajectory_file = outputPath("trajectory", seq_name + ".txt");
```

Use `trajectory_file.string()` for both trajectory open modes.

In `savePCD`, use:

```cpp
const auto raw_points_dir = outputPath("pointcloud", "all_raw_points.pcd");
const auto downsampled_points_dir = outputPath("pointcloud", "all_downsampled_points.pcd");
```

Pass `.string()` to `pcl::PCDWriter::writeBinary`.

At the end of `LIVMapper::run`, replace the single flush with:

```cpp
savePCD();
saveGaussianMap();
saveVisualizationFrames();
```

- [ ] **Step 8: Publish the rendered image as a ROS topic**

Add `image_transport::Publisher pubRenderedImage;` to `LIVMapper`.

In `initializeSubscribersAndPublishers`, add:

```cpp
pubRenderedImage = it.advertise("/gs_rendered_image", 1);
```

At the end of `publish_img_rgb`, add:

```cpp
if (!vio_manager->img_rendered.empty())
{
  cv_bridge::CvImage rendered_msg;
  rendered_msg.header.stamp = ros::Time::now();
  rendered_msg.encoding = sensor_msgs::image_encodings::BGR8;
  rendered_msg.image = vio_manager->img_rendered;
  pubRenderedImage.publish(rendered_msg.toImageMsg());
}
```

- [ ] **Step 9: Rebuild and run the Gaussian I/O test**

Run:

```bash
tools/build_native.sh
source devel/setup.bash
catkin_make run_tests_fast_livo
catkin_test_results --verbose
```

Expected: build exits `0`; `gaussian_map_io` test passes; catkin reports zero failures.

- [ ] **Step 10: Commit output support**

```bash
git add src/gs-livo/include/gaussian_map_io.h src/gs-livo/src/gaussian_map_io.cpp src/gs-livo/test/test_gaussian_map_io.cpp src/gs-livo/include/vio.h src/gs-livo/src/vio.cpp src/gs-livo/include/LIVMapper.h src/gs-livo/src/LIVMapper.cpp src/gs-livo/CMakeLists.txt
git commit -m "feat: export GS-LIVO trajectory maps and rendered frames"
```

### Task 5: Add run metadata and artifact validation tools

**Files:**
- Create: `tests/test_artifact_tools.py`
- Create: `tools/write_manifest.py`
- Create: `tools/validate_artifacts.py`
- Create: `tools/visualize_gaussian_ply.py`

- [ ] **Step 1: Write failing manifest and artifact parser tests**

Create `tests/test_artifact_tools.py`:

```python
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = load("validate_artifacts", ROOT / "tools/validate_artifacts.py")
VISUALIZE = load("visualize_gaussian_ply", ROOT / "tools/visualize_gaussian_ply.py")


class ArtifactToolTests(unittest.TestCase):
    def test_tum_validation_rejects_non_monotonic_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trajectory.txt"
            path.write_text(
                "2.0 0 0 0 0 0 0 1\n1.0 0 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            result = VALIDATE.validate_tum(path)
            self.assertFalse(result["ok"])

    def test_pcd_and_ply_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pcd = root / "map.pcd"
            pcd.write_text("VERSION .7\nPOINTS 12\nDATA ascii\n", encoding="utf-8")
            ply = root / "map.ply"
            ply.write_text(
                "ply\nformat ascii 1.0\nelement vertex 7\n"
                "property float x\nproperty float f_dc_0\nproperty float opacity\n"
                "property float scale_0\nproperty float rot_0\nend_header\n",
                encoding="utf-8",
            )
            self.assertEqual(VALIDATE.read_pcd_points(pcd), 12)
            self.assertEqual(VALIDATE.read_ply_vertex_count(ply), 7)
            self.assertTrue(VALIDATE.validate_ply(ply)["ok"])

    def test_gaussian_visualizer_recovers_position_and_color(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "map.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                "end_header\n1 2 3 0 0 0\n",
                encoding="utf-8",
            )
            xyz, rgb = VISUALIZE.read_gaussian_ply(path)
            self.assertEqual(xyz.tolist(), [[1.0, 2.0, 3.0]])
            self.assertEqual(rgb.tolist(), [[128, 128, 128]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify missing tools fail**

Run:

```bash
python3 -m unittest tests.test_artifact_tools -v
```

Expected: `FileNotFoundError` for `tools/validate_artifacts.py`.

- [ ] **Step 3: Implement the manifest writer**

Create `tools/write_manifest.py`:

```python
#!/usr/bin/env python3
import argparse
import datetime
import json
import pathlib
import platform
import subprocess


def output(command):
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--duration", required=True)
    parser.add_argument("--bag-status", type=int, required=True)
    parser.add_argument("--launch-status", type=int, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    dataset = pathlib.Path(args.dataset)
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            "url": output(["git", "-C", str(repo), "remote", "get-url", "origin"]),
            "commit": output(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        },
        "system": {
            "platform": platform.platform(),
            "cmake": output(["cmake", "--version"]).splitlines()[0],
            "compiler": output(["g++", "--version"]).splitlines()[0],
            "cuda": output(["nvcc", "--version"]).splitlines()[-1],
            "gpu": output([
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]),
            "ros_distro": "noetic",
        },
        "dataset": {
            "path": str(dataset),
            "size_bytes": dataset.stat().st_size,
            "sha256": args.dataset_sha256,
        },
        "dependencies": {
            "rpg_vikit_commit": output(["git", "-C", str(repo / "src/rpg_vikit"), "rev-parse", "HEAD"]),
            "libtorch": "2.0.1+cu118-cxx11-abi",
            "libtorch_archive_sha256": (repo / "third_party/downloads/libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip.sha256").read_text().split()[0],
        },
        "run": {
            "phase": args.phase,
            "rate": args.rate,
            "duration": args.duration,
            "start_offset_seconds": 75,
            "bag_status": args.bag_status,
            "launch_status": args.launch_status,
        },
    }
    path = pathlib.Path(args.run_dir) / "run_manifest.yaml"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement artifact validation**

Create `tools/validate_artifacts.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import re


FATAL_PATTERNS = (
    "CUDA out of memory",
    "Segmentation fault",
    "terminate called",
    "[FATAL]",
)
REQUIRED_PLY_PROPERTIES = {
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
}


def validate_tum(path):
    previous = -math.inf
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = [float(value) for value in line.split()]
        if len(fields) != 8 or not all(math.isfinite(value) for value in fields):
            return {"ok": False, "count": count, "reason": "invalid TUM row"}
        if fields[0] <= previous:
            return {"ok": False, "count": count, "reason": "timestamps not increasing"}
        previous = fields[0]
        count += 1
    return {"ok": count > 0, "count": count, "reason": "" if count else "empty"}


def read_pcd_points(path):
    with path.open("rb") as handle:
        header = handle.read(16384).decode("ascii", errors="ignore")
    match = re.search(r"^POINTS\s+(\d+)\s*$", header, re.MULTILINE)
    return int(match.group(1)) if match else 0


def read_ply_header(path):
    lines = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            lines.append(line)
            if line == "end_header":
                break
    return lines


def read_ply_vertex_count(path):
    for line in read_ply_header(path):
        match = re.fullmatch(r"element vertex (\d+)", line)
        if match:
            return int(match.group(1))
    return 0


def validate_ply(path):
    header = read_ply_header(path)
    properties = {
        line.split()[-1] for line in header if line.startswith("property ")
    }
    missing = sorted(REQUIRED_PLY_PROPERTIES - properties)
    count = read_ply_vertex_count(path)
    return {"ok": count > 0 and not missing, "count": count, "missing": missing}


def validate_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = [pattern for pattern in FATAL_PATTERNS if pattern in text]
    return {"ok": not matches, "fatal_patterns": matches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    root = pathlib.Path(args.run_dir)

    artifacts = {
        "trajectory": validate_tum(root / "trajectory/HKairport01.txt"),
        "raw_pcd": {"count": read_pcd_points(root / "pointcloud/all_raw_points.pcd")},
        "downsampled_pcd": {"count": read_pcd_points(root / "pointcloud/all_downsampled_points.pcd")},
        "gaussian_ply": validate_ply(root / "gaussian/global_gaussians.ply"),
    "launch_log": validate_log(root / "logs/launch.log"),
        "path_topic": {"bytes": (root / "logs/topic_path.txt").stat().st_size},
        "cloud_topic": {"bytes": (root / "logs/topic_cloud_registered.txt").stat().st_size},
        "render_topic": {"bytes": (root / "logs/topic_gs_rendered_image.txt").stat().st_size},
        "input_image": {"bytes": (root / "visualization/input.png").stat().st_size},
        "rendered_image": {"bytes": (root / "visualization/rendered.png").stat().st_size},
        "gaussian_html": {"bytes": (root / "visualization/gaussian_map.html").stat().st_size},
    }
    artifacts["raw_pcd"]["ok"] = artifacts["raw_pcd"]["count"] > 0
    artifacts["downsampled_pcd"]["ok"] = artifacts["downsampled_pcd"]["count"] > 0
    artifacts["path_topic"]["ok"] = artifacts["path_topic"]["bytes"] > 0
    artifacts["cloud_topic"]["ok"] = artifacts["cloud_topic"]["bytes"] > 0
    artifacts["render_topic"]["ok"] = artifacts["render_topic"]["bytes"] > 0
    artifacts["input_image"]["ok"] = artifacts["input_image"]["bytes"] > 0
    artifacts["rendered_image"]["ok"] = artifacts["rendered_image"]["bytes"] > 0
    artifacts["gaussian_html"]["ok"] = artifacts["gaussian_html"]["bytes"] > 0
    manifest = json.loads((root / "run_manifest.yaml").read_text())
    if manifest["run"]["phase"] == "full":
        rviz_path = root / "visualization/rviz.png"
        artifacts["rviz_image"] = {
            "bytes": rviz_path.stat().st_size,
            "ok": rviz_path.stat().st_size > 0,
        }
    report = {"ok": all(item["ok"] for item in artifacts.values()), "artifacts": artifacts}
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Implement interactive Gaussian-map visualization**

Create `tools/visualize_gaussian_ply.py`:

```python
#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import plotly.graph_objects as go


SH_C0 = 0.28209479177387814


def read_gaussian_ply(path):
    path = pathlib.Path(path)
    properties = []
    vertex_count = 0
    header_lines = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, ndmin=2)
    indices = {name: index for index, name in enumerate(properties)}
    xyz = data[:, [indices["x"], indices["y"], indices["z"]]]
    sh = data[:, [indices["f_dc_0"], indices["f_dc_1"], indices["f_dc_2"]]]
    rgb = np.clip(np.rint((sh * SH_C0 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return xyz, rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-points", type=int, default=100000)
    args = parser.parse_args()

    xyz, rgb = read_gaussian_ply(args.ply)
    if len(xyz) > args.max_points:
        indices = np.linspace(0, len(xyz) - 1, args.max_points, dtype=int)
        xyz = xyz[indices]
        rgb = rgb[indices]
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    figure = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker={"size": 1.5, "color": colors, "opacity": 0.8},
    ))
    figure.update_layout(
        title="GS-LIVO HKairport01 Gaussian Map",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs="cdn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Make all three tools executable.

- [ ] **Step 6: Run the artifact-tool tests**

Run:

```bash
python3 -m unittest tests.test_artifact_tools -v
```

Expected: 3 tests pass.

- [ ] **Step 7: Commit metadata, validation, and visualization tools**

```bash
git add tests/test_artifact_tools.py tools/write_manifest.py tools/validate_artifacts.py tools/visualize_gaussian_ply.py
git commit -m "test: validate GS-LIVO run artifacts"
```

### Task 6: Add runtime orchestration and focused RViz visualization

**Files:**
- Create: `tests/test_run_tooling.py`
- Create: `tools/run_hkairport01.sh`
- Create: `src/gs-livo/rviz_cfg/HKairport01.rviz`

- [ ] **Step 1: Write a failing run-script dry-run test**

Create `tests/test_run_tooling.py`:

```python
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RunToolingTests(unittest.TestCase):
    def test_short_dry_run(self):
        completed = subprocess.run(
            [str(ROOT / "tools/run_hkairport01.sh"), "short", "--dry-run"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("offset=75", completed.stdout)
        self.assertIn("duration=20", completed.stdout)
        self.assertIn("rate=0.25", completed.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_run_tooling -v
```

Expected: missing `tools/run_hkairport01.sh`.

- [ ] **Step 3: Add a focused RViz configuration**

Create `src/gs-livo/rviz_cfg/HKairport01.rviz`:

```yaml
Panels:
  - Class: rviz/Displays
    Name: Displays
  - Class: rviz/Views
    Name: Views
Visualization Manager:
  Class: ""
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: camera_init
    Frame Rate: 30
  Displays:
    - Class: rviz/Grid
      Name: Grid
      Enabled: true
      Cell Size: 5
      Plane Cell Count: 40
    - Class: rviz/PointCloud2
      Name: Registered RGB Cloud
      Enabled: true
      Topic: /cloud_registered
      Queue Size: 5
      Style: Points
      Size (Pixels): 2
      Color Transformer: RGB8
      Position Transformer: XYZ
    - Class: rviz/Path
      Name: Estimated Path
      Enabled: true
      Topic: /path
      Queue Size: 10
      Line Style: Lines
      Line Width: 0.05
      Color: 25; 255; 255
    - Class: rviz/Odometry
      Name: Odometry
      Enabled: true
      Topic: /aft_mapped_to_init
      Keep: 50
      Shape:
        Value: Axes
        Axes Length: 1
        Axes Radius: 0.1
    - Class: rviz/Image
      Name: Input Image
      Enabled: true
      Image Topic: /rgb_img
      Queue Size: 2
    - Class: rviz/Image
      Name: Gaussian Render
      Enabled: true
      Image Topic: /gs_rendered_image
      Queue Size: 2
  Tools:
    - Class: rviz/Interact
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 80
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Pitch: 0.5
      Yaw: 0.8
Window Geometry:
  Width: 1600
  Height: 900
```

- [ ] **Step 4: Implement phase-based run orchestration**

Create `tools/run_hkairport01.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dataset=${GS_LIVO_DATASET:-/media/zzh/data/LVIO_and_LVIO_GS/HKairport01.bag}
result_root=${GS_LIVO_RESULT_ROOT:-/media/zzh/data/LVIO_and_LVIO_GS/GS-LIVO-results/HKairport01}
phase=${1:-short}
dry_run=${2:-}
offset=75

case "$phase" in
  short) duration=20; rate=0.25; rviz=false ;;
  medium) duration=120; rate=0.25; rviz=false ;;
  full) duration=full; rate=${GS_LIVO_FULL_RATE:-0.25}; rviz=true ;;
  *) printf 'usage: %s {short|medium|full} [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

if [[ "$dry_run" == "--dry-run" ]]; then
  printf 'phase=%s offset=%s duration=%s rate=%s rviz=%s\n' "$phase" "$offset" "$duration" "$rate" "$rviz"
  exit 0
fi

source /opt/ros/noetic/setup.bash
source "$repo_root/devel/setup.bash"

run_id=$(date +%Y%m%d-%H%M%S)-$phase
run_dir="$result_root/$run_id"
mkdir -p "$run_dir"/{trajectory,pointcloud,gaussian,logs,visualization,config}
ln -sfn "$result_root" "$repo_root/results"
cp "$repo_root/build_logs/latest.log" "$run_dir/logs/build.log"
cp "$repo_root/src/gs-livo/config/HKairport01.yaml" "$run_dir/config/"
cp "$repo_root/src/gs-livo/config/camera_MARS_LVIG.yaml" "$run_dir/config/"

launch_pid=""
gpu_pid=""
capture_pid=""
probe_pids=()

cleanup() {
  if [[ -n "$capture_pid" ]]; then kill "$capture_pid" 2>/dev/null || true; fi
  for probe_pid in "${probe_pids[@]}"; do kill "$probe_pid" 2>/dev/null || true; done
  if [[ -n "$launch_pid" ]]; then kill -INT "$launch_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_pid" ]]; then kill "$gpu_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 1
  done
) > "$run_dir/logs/gpu.csv" 2>&1 &
gpu_pid=$!

roslaunch fast_livo mapping_hkairport01.launch \
  output_root:="$run_dir" rviz:="$rviz" \
  > "$run_dir/logs/launch.log" 2>&1 &
launch_pid=$!

for _ in $(seq 1 30); do
  if rostopic list >/dev/null 2>&1; then break; fi
  sleep 1
done
rostopic list > "$run_dir/logs/topics.txt"

timeout 90 rostopic echo -n1 /path > "$run_dir/logs/topic_path.txt" 2>&1 &
probe_pids+=("$!")
timeout 90 rostopic echo -n1 /cloud_registered > "$run_dir/logs/topic_cloud_registered.txt" 2>&1 &
probe_pids+=("$!")
timeout 90 rostopic echo -n1 /gs_rendered_image > "$run_dir/logs/topic_gs_rendered_image.txt" 2>&1 &
probe_pids+=("$!")

if [[ "$rviz" == "true" ]]; then
  (sleep 30; gnome-screenshot -f "$run_dir/visualization/rviz.png") &
  capture_pid=$!
fi

bag_command=(rosbag play "$dataset" -s "$offset" -r "$rate")
if [[ "$duration" != "full" ]]; then
  bag_command+=(--duration "$duration")
fi
"${bag_command[@]}" > "$run_dir/logs/rosbag.log" 2>&1
bag_status=$?

sleep 5
for probe_pid in "${probe_pids[@]}"; do wait "$probe_pid" 2>/dev/null || true; done
kill -INT "$launch_pid" 2>/dev/null || true
wait "$launch_pid"
launch_status=$?
if [[ "$launch_status" -eq 130 ]]; then launch_status=0; fi
launch_pid=""
kill "$gpu_pid" 2>/dev/null || true
wait "$gpu_pid" 2>/dev/null || true
gpu_pid=""
if [[ -n "$capture_pid" ]]; then wait "$capture_pid" 2>/dev/null || true; capture_pid=""; fi

dataset_sha256="not-computed-for-validation-run"
if [[ "$phase" == "full" ]]; then
  sha256sum "$dataset" > "$run_dir/logs/dataset.sha256"
  dataset_sha256=$(cut -d' ' -f1 "$run_dir/logs/dataset.sha256")
fi

python3 "$repo_root/tools/write_manifest.py" \
  --run-dir "$run_dir" --dataset "$dataset" --phase "$phase" \
  --rate "$rate" --duration "$duration" \
  --bag-status "$bag_status" --launch-status "$launch_status" \
  --dataset-sha256 "$dataset_sha256"

python3 "$repo_root/tools/visualize_gaussian_ply.py" \
  "$run_dir/gaussian/global_gaussians.ply" \
  --output "$run_dir/visualization/gaussian_map.html"

python3 "$repo_root/tools/validate_artifacts.py" "$run_dir"
validation_status=$?

(cd "$run_dir" && find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
printf '%s\n' "$run_dir"

if [[ "$bag_status" -ne 0 || "$launch_status" -ne 0 || "$validation_status" -ne 0 ]]; then
  exit 1
fi
```

Make it executable with `chmod +x tools/run_hkairport01.sh`.

- [ ] **Step 5: Run dry-run and complete Python tests**

Run:

```bash
python3 -m unittest discover -s tests -v
tools/run_hkairport01.sh short --dry-run
```

Expected: all Python tests pass; dry-run prints `offset=75 duration=20 rate=0.25`.

- [ ] **Step 6: Commit runtime orchestration**

```bash
git add tests/test_run_tooling.py tools/run_hkairport01.sh src/gs-livo/rviz_cfg/HKairport01.rviz
git commit -m "feat: orchestrate reproducible HKairport01 runs"
```

### Task 7: Build, test, and perform startup verification

**Files:**
- No planned tracked-file changes.

- [ ] **Step 1: Record a clean pre-build environment report**

Run:

```bash
python3 tools/check_environment.py --output preflight.json
python3 -m json.tool preflight.json | tee /tmp/gs_livo_preflight.txt
```

Expected: preflight `ok` is `true`.

- [ ] **Step 2: Bootstrap dependencies**

Run:

```bash
tools/bootstrap_dependencies.sh 2>&1 | tee /tmp/gs_livo_bootstrap.log
test "$(git -C src/rpg_vikit rev-parse HEAD)" = "6c886c8e5d83997806e00294826d528cea3581dd"
test -f third_party/libtorch/lib/libtorch.so
```

Expected: both `test` commands exit `0`.

- [ ] **Step 3: Build lib3dgs and the ROS workspace**

Run:

```bash
tools/build_native.sh 2>&1 | tee /tmp/gs_livo_build.log
test -f src/lib3dgs/build/lib3dgs_lib.so
test -x devel/lib/fast_livo/fastlivo_mapping
```

Expected: build exits `0` and both binaries exist.

- [ ] **Step 4: Run all unit tests**

Run:

```bash
python3 -m unittest discover -s tests -v
source devel/setup.bash
catkin_make run_tests_fast_livo
catkin_test_results --verbose
```

Expected: all Python and C++ tests pass with zero catkin failures.

- [ ] **Step 5: Launch without a bag and verify publishers**

Run in terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch fast_livo mapping_hkairport01.launch rviz:=false output_root:=/tmp/gs-livo-startup
```

Run in terminal 2:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rostopic list | rg '^/(aft_mapped_to_init|cloud_registered|gs_rendered_image|path|rgb_img)$'
```

Expected: all five topics are listed. Stop terminal 1 with Ctrl-C and confirm it exits without a segmentation fault.

### Task 8: Execute short, medium, and complete HKairport01 runs

**Files:**
- Create through execution: `/media/zzh/data/LVIO_and_LVIO_GS/GS-LIVO-results/HKairport01/<run-id>/...`

- [ ] **Step 1: Run the 20-second smoke test**

Run:

```bash
tools/run_hkairport01.sh short | tee /tmp/gs_livo_short_run_path.txt
short_run=$(tail -n 1 /tmp/gs_livo_short_run_path.txt)
python3 -m json.tool "$short_run/validation.json"
```

Expected: validation `ok` is `true`; trajectory, both PCDs, Gaussian PLY, and both images are non-empty.

- [ ] **Step 2: Run the 120-second medium test**

Run:

```bash
tools/run_hkairport01.sh medium | tee /tmp/gs_livo_medium_run_path.txt
medium_run=$(tail -n 1 /tmp/gs_livo_medium_run_path.txt)
python3 -m json.tool "$medium_run/validation.json"
```

Expected: validation `ok` is `true`; GPU telemetry contains at least 100 samples; launch log contains no fatal pattern.

- [ ] **Step 3: Choose the fastest stable full-run rate**

Read the medium run duration and callback behavior. Use `0.25` unless queues grow without recovery or GPU memory approaches 8 GB. If unstable, set one lower rate:

```bash
export GS_LIVO_FULL_RATE=0.1
```

Record the selected rate in the final report. Playback rate changes timing of data delivery only; no sensor timestamps or algorithm parameters are changed.

- [ ] **Step 4: Run the complete remaining sequence**

Run:

```bash
tools/run_hkairport01.sh full | tee /tmp/gs_livo_full_run_path.txt
full_run=$(tail -n 1 /tmp/gs_livo_full_run_path.txt)
python3 -m json.tool "$full_run/validation.json"
```

Expected: validation `ok` is `true`; RViz screenshot exists; complete-run trajectory and maps exceed the short-run data counts.

- [ ] **Step 5: Perform independent artifact checks**

Run:

```bash
wc -l "$full_run/trajectory/HKairport01.txt"
pcl_viewer "$full_run/pointcloud/all_downsampled_points.pcd"
head -40 "$full_run/gaussian/global_gaussians.ply"
(cd "$full_run" && sha256sum -c checksums.sha256)
```

Expected: trajectory has multiple poses; PCL viewer opens the cloud; PLY header contains the required Gaussian fields and a positive vertex count; all checksums pass.

- [ ] **Step 6: Capture or replace the representative RViz screenshot**

If the timed screenshot does not clearly show trajectory and cloud, replay the saved result in RViz and run:

```bash
gnome-screenshot -f "$full_run/visualization/rviz.png"
```

Expected: the PNG is non-empty and visibly includes the estimated path and colored cloud.

### Task 9: Document the reproduction and run final verification

**Files:**
- Create: `tests/test_reproduction_report.py`
- Create: `tools/write_reproduction_report.py`
- Create: `REPRODUCTION.md`

- [ ] **Step 1: Write a failing measured-report test**

Create `tests/test_reproduction_report.py`:

```python
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "write_reproduction_report", ROOT / "tools/write_reproduction_report.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReproductionReportTests(unittest.TestCase):
    def test_report_contains_measured_counts_and_peak_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            (run_dir / "logs").mkdir()
            (run_dir / "run_manifest.yaml").write_text(json.dumps({
                "source": {"commit": "abc123"},
                "system": {"gpu": "RTX 4060, 535.230.02, 8188 MiB"},
                "dataset": {"path": "/data/HKairport01.bag"},
                "run": {"rate": 0.25, "start_offset_seconds": 75},
            }))
            (run_dir / "validation.json").write_text(json.dumps({
                "ok": True,
                "artifacts": {
                    "trajectory": {"count": 10},
                    "raw_pcd": {"count": 20},
                    "downsampled_pcd": {"count": 15},
                    "gaussian_ply": {"count": 7},
                },
            }))
            (run_dir / "logs/gpu.csv").write_text(
                "2026/08/24 20:00:00, 90, 4096, 8188, 70, 90\n"
            )
            report = MODULE.render_report(ROOT, run_dir)
            self.assertIn("Trajectory poses: 10", report)
            self.assertIn("Exported Gaussian vertices: 7", report)
            self.assertIn("Peak GPU memory: 4096 MiB", report)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_reproduction_report -v
```

Expected: missing `tools/write_reproduction_report.py`.

- [ ] **Step 3: Implement the measured reproduction report generator**

Create `tools/write_reproduction_report.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import pathlib


def peak_gpu_memory(path):
    peak = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            try:
                peak = max(peak, int(float(fields[2])))
            except ValueError:
                pass
    return peak


def render_report(repo, run_dir):
    manifest = json.loads((run_dir / "run_manifest.yaml").read_text())
    validation = json.loads((run_dir / "validation.json").read_text())
    artifacts = validation["artifacts"]
    cloud_path = run_dir / "pointcloud/all_downsampled_points.pcd"
    fence = chr(96) * 3
    return f"""# GS-LIVO HKairport01 Reproduction

## Source

- Upstream: `https://github.com/HKUST-Aerial-Robotics/GS-LIVO`
- Upstream base commit: `cc65db279bf0d7c4df12d21710cc4053753dd133`
- Local reproduction commit: `{manifest['source']['commit']}`

## Environment

- Ubuntu 20.04, ROS Noetic
- CUDA 11.8
- GPU and driver: `{manifest['system']['gpu']}`
- LibTorch: 2.0.1+cu118, C++11 ABI

## Dataset and command

- Dataset: `{manifest['dataset']['path']}`
- Start offset: {manifest['run']['start_offset_seconds']} seconds
- Playback rate: {manifest['run']['rate']}
- Command: `tools/run_hkairport01.sh full`

## Result

- Result directory: `{run_dir}`
- Trajectory poses: {artifacts['trajectory']['count']}
- Raw colored points: {artifacts['raw_pcd']['count']}
- Downsampled colored points: {artifacts['downsampled_pcd']['count']}
- Exported Gaussian vertices: {artifacts['gaussian_ply']['count']}
- Peak GPU memory: {peak_gpu_memory(run_dir / 'logs/gpu.csv')} MiB
- Validation: {'passed' if validation['ok'] else 'failed'}

## Visualize

{fence}bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rviz -d src/gs-livo/rviz_cfg/HKairport01.rviz
pcl_viewer {cloud_path}
{fence}

Interactive Gaussian-map view: `{run_dir / 'visualization/gaussian_map.html'}`

## Public-code limitation

The exported PLY faithfully represents the global and active-window Gaussian data retained by the published implementation. The public code does not clearly persist every optimized per-frame Gaussian back into the global map, so this result is not presented as proof that unpublished paper-level Gaussian maintenance has been reproduced.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="REPRODUCTION.md")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    run_dir = pathlib.Path(args.run_dir).resolve()
    pathlib.Path(args.output).write_text(render_report(repo, run_dir), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Make it executable with `chmod +x tools/write_reproduction_report.py`.

- [ ] **Step 4: Generate and verify the measured report**

Run:

```bash
python3 -m unittest tests.test_reproduction_report -v
python3 tools/write_reproduction_report.py "$full_run" --output REPRODUCTION.md
rg -n 'Trajectory poses|Raw colored points|Exported Gaussian vertices|Peak GPU memory|Validation: passed' REPRODUCTION.md
```

Expected: the test passes and every measured-result line is present.

- [ ] **Step 5: Run final fresh verification**

Run:

```bash
python3 -m unittest discover -s tests -v
source devel/setup.bash
catkin_make run_tests_fast_livo
catkin_test_results --verbose
python3 tools/validate_artifacts.py "$full_run"
(cd "$full_run" && sha256sum -c checksums.sha256)
git diff --check
```

Expected: all commands exit `0`; Python and catkin report zero failures; artifact validation is true; all checksums pass.

- [ ] **Step 6: Commit the measured reproduction report and generator**

```bash
git add tests/test_reproduction_report.py tools/write_reproduction_report.py REPRODUCTION.md
git commit -m "docs: record HKairport01 GS-LIVO reproduction"
```

- [ ] **Step 7: Record final repository state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -10
```

Expected: no uncommitted tracked changes; local generated directories remain ignored; recent history shows the preflight, build integration, dataset configuration, output export, runtime tooling, and reproduction report commits.
