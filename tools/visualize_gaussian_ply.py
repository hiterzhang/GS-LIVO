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
