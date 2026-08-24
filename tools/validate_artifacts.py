#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import re

import cv2
import numpy as np


FATAL_PATTERNS = (
    "CUDA out of memory",
    "Segmentation fault",
    "terminate called",
    "[FATAL]",
    "process has died",
    "return value -11",
    "exit code -8",
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


def validate_rendered_image(path, minimum_coverage=0.05):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "bytes": 0, "coverage": 0.0, "reason": "unreadable"}
    coverage = float(np.mean(np.any(image < 245, axis=2)))
    return {
        "ok": coverage >= minimum_coverage,
        "bytes": path.stat().st_size,
        "coverage": coverage,
        "reason": "" if coverage >= minimum_coverage else "nearly blank",
    }


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
        "rendered_image": validate_rendered_image(root / "visualization/rendered.png"),
        "gaussian_html": {"bytes": (root / "visualization/gaussian_map.html").stat().st_size},
    }
    artifacts["raw_pcd"]["ok"] = artifacts["raw_pcd"]["count"] > 0
    artifacts["downsampled_pcd"]["ok"] = artifacts["downsampled_pcd"]["count"] > 0
    artifacts["path_topic"]["ok"] = artifacts["path_topic"]["bytes"] > 0
    artifacts["cloud_topic"]["ok"] = artifacts["cloud_topic"]["bytes"] > 0
    artifacts["render_topic"]["ok"] = artifacts["render_topic"]["bytes"] > 0
    artifacts["input_image"]["ok"] = artifacts["input_image"]["bytes"] > 0
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
