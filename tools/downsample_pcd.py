#!/usr/bin/env python3
import argparse
import math
import pathlib
import subprocess

import numpy as np


PCL_GRID_LIMIT = 2**31 - 1


def voxel_cell_count(ranges, leaf_size):
    return math.prod(math.floor(max(0.0, value) / leaf_size) + 1 for value in ranges)


def safe_leaf_size(ranges, requested_leaf_size):
    leaf_size = requested_leaf_size
    while voxel_cell_count(ranges, leaf_size) > PCL_GRID_LIMIT:
        leaf_size *= 1.05
    return leaf_size


def binary_xyz_bounds(path):
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PCD header has no DATA line")
            text = line.decode("ascii").strip()
            header.append(text)
            if text.startswith("DATA "):
                if text != "DATA binary":
                    raise ValueError(f"expected DATA binary, got {text}")
                break
        fields = next(line for line in header if line.startswith("FIELDS ")).split()[1:]
        sizes = list(map(int, next(line for line in header if line.startswith("SIZE ")).split()[1:]))
        types = next(line for line in header if line.startswith("TYPE ")).split()[1:]
        if fields[:3] != ["x", "y", "z"] or sizes[:3] != [4, 4, 4] or types[:3] != ["F", "F", "F"]:
            raise ValueError("expected leading float32 x y z fields")
        point_step = sum(sizes)
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        while chunk := handle.read(point_step * 1_000_000):
            usable = len(chunk) - len(chunk) % point_step
            points = np.ndarray(
                shape=(usable // point_step, 3),
                dtype=np.float32,
                buffer=chunk[:usable],
                strides=(point_step, 4),
            )
            finite = points[np.isfinite(points).all(axis=1)]
            if finite.size:
                mins = np.minimum(mins, finite.min(axis=0))
                maxs = np.maximum(maxs, finite.max(axis=0))
    if not np.isfinite(mins).all():
        raise ValueError("PCD contains no finite xyz points")
    return tuple((maxs - mins).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--leaf", type=float, default=0.15)
    args = parser.parse_args()

    ranges = binary_xyz_bounds(args.input)
    leaf = safe_leaf_size(ranges, args.leaf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pcl_voxel_grid", str(args.input), str(args.output), "-leaf", f"{leaf},{leaf},{leaf}"],
        check=True,
    )
    print(f"requested_leaf={args.leaf} effective_leaf={leaf} ranges={ranges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
