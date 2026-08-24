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
            "rpg_vikit_commit": output([
                "git", "-C", str(repo / "src/rpg_vikit"), "rev-parse", "HEAD"
            ]),
            "libtorch": "2.0.1+cu118-cxx11-abi",
            "libtorch_archive_sha256": (
                repo / "third_party/downloads/libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip.sha256"
            ).read_text().split()[0],
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
