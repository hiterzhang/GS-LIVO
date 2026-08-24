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
