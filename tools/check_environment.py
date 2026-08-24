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
