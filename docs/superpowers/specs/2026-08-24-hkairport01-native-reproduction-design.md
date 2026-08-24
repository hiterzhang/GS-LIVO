# GS-LIVO HKairport01 Native Reproduction Design

## Objective

Reproduce the public GS-LIVO implementation natively on the local Ubuntu 20.04 and ROS Noetic workstation, using `/media/zzh/data/LVIO_and_LVIO_GS/HKairport01.bag`. The first milestone is a complete run that produces a trajectory, colored point clouds, an exported Gaussian map, reproducible logs, and live/offline visualization.

The implementation is pinned to upstream commit `cc65db279bf0d7c4df12d21710cc4053753dd133` from `HKUST-Aerial-Robotics/GS-LIVO`.

## Success Criteria

A reproduction run is accepted when all of the following are true:

1. The GS-LIVO ROS node consumes LiDAR, IMU, and camera messages from `HKairport01.bag` without a fatal ROS, C++, CUDA, or LibTorch error.
2. `/path`, `/aft_mapped_to_init`, `/cloud_registered`, and the image/render visualization update during playback.
3. A non-empty TUM-format trajectory is written with strictly increasing timestamps and finite poses.
4. Raw and downsampled colored PCD maps are written and can be opened by PCL or RViz.
5. A non-empty Gaussian PLY is written with position, color or spherical-harmonic features, opacity, scale, and rotation properties.
6. RViz shows the estimated trajectory and registered colored point cloud, while the input and Gaussian-rendered images are visible during the run.
7. Build logs, launch logs, rosbag logs, GPU telemetry, configuration, source revision, commands, output statistics, and checksums are retained.

## Selected Approach

Use a native ROS catkin workspace inside `GS-LVIO`, with project-local third-party dependencies and narrowly scoped source/build repairs.

This approach is preferred over Docker because the machine already provides Ubuntu 20.04, ROS Noetic, CUDA 11.8, GCC 9.4, OpenCV 4.2, PCL 1.10, and a working NVIDIA RTX 4060. Native execution also avoids X11, RViz, ROS networking, and NVIDIA-container integration overhead.

FAST-LIVO2 may be used only as a diagnostic reference if sensor synchronization or calibration cannot be validated directly. It is not a substitute for the GS-LIVO acceptance run.

## Source and Dependency Layout

The upstream repository remains the Git root. Its ROS workspace source tree is retained under `src/`.

Project-local generated or downloaded dependencies will live under a gitignored `third_party/` directory. Build artifacts will live under gitignored `build/` and `devel/` directories. No system package installation that requires `sudo` is assumed.

The build integration will repair, without changing algorithm behavior:

- the hard-coded `/home/sheng/.../lib3dgs_lib.so` link path;
- incorrect `3dgs` include paths that do not match the published `src/lib3dgs` layout;
- local discovery of LibTorch, `simple-knn`, `tinyply`, Eigen, GLM, nlohmann-json, TBB, YAML-CPP, Sophus, and rpg_vikit;
- CUDA architecture selection for the RTX 4060;
- creation and discovery of required runtime output directories.

The initial LibTorch candidate will be a C++11-ABI CUDA 11.8 build compatible with GCC 9 and the published Torch C++ APIs. Its exact version and archive checksum will be recorded in the run manifest. If compilation proves an API incompatibility, only the LibTorch version will change before any source-level Torch adaptation is considered.

## Dataset Configuration

The run will use the official FAST-LIVO2 MARS-LVIG calibration as the primary calibration reference rather than the generic GS-LIVO `avia.yaml` camera values.

The dedicated `HKairport01` configuration will contain:

- image topic `/left_camera/image`;
- LiDAR topic `/livox/lidar`;
- IMU topic `/livox/imu`;
- Livox Avia preprocessing;
- MARS-LVIG HKairport camera intrinsics at 2448 by 2048 pixels with scale `0.25`;
- MARS-LVIG camera-LiDAR and IMU-LiDAR extrinsics;
- image time offset `+0.1` seconds;
- sequence name `HKairport01`;
- trajectory and point-cloud saving enabled.

The bag will start at 75 seconds to skip the unsuitable initial segment documented by the FAST-LIVO2 MARS-LVIG configuration. Playback speed is an execution parameter, not an algorithm parameter, and may be reduced to keep processing synchronized on the 8 GB GPU.

Primary configuration references:

- <https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/config/MARS_LVIG.yaml>
- <https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/config/camera_MARS_LVIG.yaml>
- <https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/launch/mapping_avia_marslvig.launch>

## Runtime Data Flow

The runtime pipeline is:

1. `rosbag play` publishes the source bag from offset 75 seconds.
2. `image_transport republish` converts `/left_camera/image/compressed` to `/left_camera/image`.
3. GS-LIVO synchronizes Livox LiDAR, IMU, and camera measurements.
4. LIO and visual/Gaussian updates estimate state and update the published maps.
5. ROS publishers provide odometry, path, registered cloud, and image products for visualization.
6. Shutdown hooks flush trajectory, point-cloud, Gaussian-map, and run-record files.

Before the complete run, the same pipeline will be exercised on a short segment and then a medium segment. This separates build/runtime faults from long-run performance or memory faults.

## Output Layout

Large outputs will be stored on the data disk at:

`/media/zzh/data/LVIO_and_LVIO_GS/GS-LIVO-results/HKairport01/<run-id>/`

`GS-LVIO/results` will provide a convenient project-local entry to that result root.

Each run directory will contain:

```text
trajectory/HKairport01.txt
pointcloud/all_raw_points.pcd
pointcloud/all_downsampled_points.pcd
gaussian/global_gaussians.ply
logs/build.log
logs/launch.log
logs/rosbag.log
logs/gpu.csv
logs/topics.txt
visualization/rviz.png
visualization/input.png
visualization/rendered.png
config/HKairport01.yaml
config/camera_MARS_LVIG.yaml
run_manifest.yaml
checksums.sha256
```

Short and medium validation runs use distinct run IDs so they cannot overwrite the complete-run artifacts.

## Gaussian Map Export Scope

The published source contains Gaussian PLY writing utilities but includes hard-coded author paths and does not expose a reliable final global-map export workflow. The reproduction will add a configurable exporter that serializes the global GS data structure owned by the running process during shutdown.

The exporter is an observability/output addition. It must not change state estimation, Gaussian initialization, optimization iteration order, or map-update decisions.

There is an upstream concern that optimized per-frame Gaussians may not be persisted fully into the global map. The reproduction will not claim to fix or reproduce unpublished behavior. The manifest and final report will distinguish between:

- successful execution of the published GS-LIVO code;
- the exact global Gaussian state exposed by that code;
- any known discrepancy between the public implementation and the paper description.

Relevant upstream reports:

- <https://github.com/HKUST-Aerial-Robotics/GS-LIVO/issues/4>
- <https://github.com/HKUST-Aerial-Robotics/GS-LIVO/issues/7>

## Visualization

Live visualization consists of:

- RViz using a dedicated HKairport01 configuration with path, odometry, registered point cloud, and available map topics;
- the upstream OpenCV windows for undistorted input and Gaussian-rendered images.

Offline visualization consists of loading the saved trajectory and PCD in RViz and inspecting the Gaussian PLY with a compatible PLY/3DGS viewer. At least one representative RViz frame and one input/render pair will be retained in the result directory.

## Logging and Reproducibility

Wrapper scripts will save stdout and stderr while preserving the true exit status of each process. GPU telemetry will record timestamp, utilization, memory use, temperature, and power at a regular interval during validation and complete runs.

`run_manifest.yaml` will record:

- upstream repository URL and commit;
- local patch commit;
- operating system, ROS, compiler, CMake, CUDA, NVIDIA driver, and GPU;
- LibTorch and other locally built dependency versions;
- dataset absolute path, size, and checksum;
- playback offset and rate;
- launch file and effective configuration files;
- start/end times and exit statuses;
- output file sizes and data counts;
- deviations from the initial upstream or MARS-LVIG configuration.

## Validation Strategy

Validation proceeds in increasing cost:

1. Static environment check: required commands, libraries, disk capacity, GPU visibility, dataset topics, and configuration consistency.
2. Dependency builds: compile and link each non-catkin CUDA/C++ dependency before the ROS package.
3. Catkin build: build from a clean configuration and record the full log and exit status.
4. Node startup test: launch without rosbag and confirm initialization and publishers.
5. Short playback test: process a small segment from offset 75 seconds and verify all three sensor streams, path/cloud publishers, rendered frames, and absence of fatal errors.
6. Medium playback test: exercise map growth, output writing, shutdown, and GPU memory behavior.
7. Complete playback: process the remaining HKairport01 sequence at the fastest stable playback rate.
8. Artifact validation: parse the trajectory, PCD headers/data counts, Gaussian PLY header/properties, manifests, logs, and checksums.
9. Visualization validation: open saved ROS/PCD products in RViz and capture representative screenshots.

## Failure Handling

Failures will be addressed in this order:

1. Correct missing or incorrectly located build dependencies.
2. Correct topic, image republishing, camera model, calibration, or time-offset configuration.
3. Reduce rosbag playback rate if callbacks cannot keep pace.
4. Resolve runtime output-directory or shutdown-flush issues.
5. If CUDA runs out of memory, first reduce Gaussian iterations or image scale, changing only one parameter at a time and recording the deviation.
6. If the public global Gaussian structure cannot represent the paper's claimed persistent optimization, export the structure faithfully and document the limitation instead of fabricating an equivalent result.

No paper-level accuracy or real-time performance claim is part of this first milestone.

