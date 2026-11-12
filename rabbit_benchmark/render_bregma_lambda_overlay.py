"""Render the Plan B rabbit benchmark result as a PNG and offline 3D HTML."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


DIRECTORY = Path(__file__).resolve().parent
REFERENCE_COLOR = "#286792"
SCAN_COLOR = "#ba5509"
INK = "#172333"
LANDMARKS = (
    ("Bregma_candidate", "Bregma", "#00766d", "o", "circle"),
    ("Lambda_candidate", "Lambda", "#941957", "D", "diamond"),
)


def read_cloud(name):
    points = np.asarray(o3d.io.read_point_cloud(str(DIRECTORY / name)).points).copy()
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Missing or invalid point cloud: {name}")
    return points


def render_png(reference, scan, points, rotation, center, subtitle, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.055, right=0.96, top=0.80, bottom=0.12, wspace=0.18)
    fig.text(0.055, 0.932, "Plan B | Rabbit skull registration", fontsize=23,
             weight="bold", color=INK)
    fig.text(0.055, 0.883, subtitle,
             fontsize=12, color="#4b5665")
    fig.legend(handles=[
        Line2D([], [], color=REFERENCE_COLOR, marker="o", linestyle="none",
               markersize=6, label="Aligned reference"),
        Line2D([], [], color=SCAN_COLOR, marker=".", linestyle="none",
               markersize=9, label="Simulated scan"),
    ], loc="upper right", bbox_to_anchor=(0.96, 0.936), frameon=False,
        labelcolor=INK, fontsize=10)

    for ax, name, angle in zip(axes, ("Dorsal view", "Oblique view"), (0, 45)):
        theta = np.deg2rad(angle)
        horizontal = rotation @ [np.cos(theta), np.sin(theta), 0]
        projection = np.column_stack((horizontal, rotation[:, 2]))
        ref_2d = (reference - center) @ projection
        scan_2d = (scan - center) @ projection
        landmarks_2d = (points - center) @ projection
        ax.set_facecolor("#f5f7f9")
        ax.scatter(*ref_2d.T, s=1.15, color=REFERENCE_COLOR, alpha=0.42, linewidths=0)
        ax.scatter(*scan_2d.T, s=2.25, color=SCAN_COLOR, alpha=0.54, linewidths=0)
        for xy, (_, label, color, marker, _) in zip(landmarks_2d, LANDMARKS):
            is_bregma = label == "Bregma"
            offset = (-14, 19) if is_bregma else (13, -15)
            ax.scatter(*xy, s=68, marker=marker, color=color, edgecolors="white",
                       linewidths=1.6, zorder=5)
            ax.annotate(label, xy, xytext=offset, textcoords="offset points",
                        ha="right" if is_bregma else "left", va="center",
                        color=color, weight="bold", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.94),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.05), zorder=6)
        ax.set(xlim=(-35, 35), ylim=(-40, 40), aspect="equal", xticks=[], yticks=[])
        ax.set_title(name, loc="left", fontsize=13, weight="bold", color=INK, pad=10)
        ax.text(-32, 35.5, "Anterior ↑", fontsize=9, color=INK)
        ax.plot([-31.5, -21.5], [-35, -35], color=INK, lw=2, solid_capstyle="butt")
        ax.text(-26.5, -37, "10 units", ha="center", va="top", fontsize=8, color=INK)
        for spine in ax.spines.values():
            spine.set_color("#dce3e9")
    fig.text(0.055, 0.075, "Candidate landmarks · Orthographic projections in the scan frame · "
             "Distances in STL model units", fontsize=9, color="#4b5665")
    fig.text(0.055, 0.044, "NIH 3D 3DPX-004799 v3 · SMART Neuro at Penn Vet · CC BY-NC-SA 4.0",
             fontsize=8, color="#4b5665")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def render_html(reference, scan, points, rotation, translation, output):
    figure = go.Figure()
    for xyz, name, color, size, opacity, symbol in (
        (reference, "Aligned reference cloud", REFERENCE_COLOR, 2, 0.38, "circle-open"),
        (scan, "Simulated scan", SCAN_COLOR, 2.6, 0.64, "circle"),
    ):
        figure.add_scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=name,
                             mode="markers", hoverinfo="skip",
                             marker=dict(size=size, color=color, opacity=opacity, symbol=symbol))

    mesh = trimesh.load_mesh(DIRECTORY / "krakoski_1-nih3d.stl")
    xyz = mesh.vertices @ rotation.T + translation
    figure.add_mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                      i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                      name="Aligned full STL (optional)", color="#8594a1", opacity=0.22,
                      showlegend=True, visible="legendonly", hoverinfo="skip")

    for xyz, (_, label, color, _, symbol) in zip(points, LANDMARKS):
        x, y, z = xyz.tolist()
        figure.add_scatter3d(x=[x], y=[y], z=[z], name=f"{label} candidate",
                             mode="markers+text", text=[label], textposition="middle right",
                             textfont=dict(size=15, color=color),
                             marker=dict(size=8, color=color, symbol=symbol,
                                         line=dict(width=2, color="white")),
                             hovertemplate=f"{label} candidate<br>" "X %{x:.2f}<br>Y %{y:.2f}<br>"
                                           "Z %{z:.2f}<extra></extra>")
    eye = rotation @ [0.18, -2.05, 0.28]
    up = rotation[:, 2]
    figure.update_layout(
        title=dict(text="Plan B | Rabbit skull point-cloud overlay", x=0.03),
        font=dict(family="Arial, sans-serif", color=INK),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)"),
        paper_bgcolor="white", margin=dict(l=0, r=0, t=66, b=36),
        scene=dict(
            aspectmode="data",
            xaxis_title="Scan X (model units)",
            yaxis_title="Scan Y (model units)",
            zaxis_title="Scan Z (model units)",
            camera=dict(eye=dict(zip("xyz", eye)), up=dict(zip("xyz", up)),
                        projection=dict(type="orthographic")),
        ),
        annotations=[dict(xref="paper", yref="paper", x=0.02, y=0.01, showarrow=False,
                          text="NIH 3D 3DPX-004799 v3 · SMART Neuro at Penn Vet · CC BY-NC-SA 4.0",
                          font=dict(size=11, color="#4b5665"))],
    )
    figure.write_html(output, include_plotlyjs=True,
                      config=dict(displaylogo=False, responsive=True),
                      default_height="95vh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("full", "cropped"), default="full")
    parser.add_argument("--png-only", action="store_true")
    args = parser.parse_args()

    result_path = DIRECTORY / f"bregma_lambda_{args.case}_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    transform = np.asarray(result["reference_to_scan_mm"])
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    reference = read_cloud("reference_dorsal.ply") @ rotation.T + translation
    scan = read_cloud(f"{args.case}_scan.ply")
    points = np.asarray([result["landmarks"][name]["xyz_mm"] for name, *_ in LANDMARKS])
    if points.shape != (len(LANDMARKS), 3) or not np.isfinite(points).all():
        raise ValueError("Both candidate landmarks must have valid scan coordinates")
    center = (reference.min(axis=0) + reference.max(axis=0)) / 2
    case_label = "Full" if args.case == "full" else "Cropped"
    subtitle = f"{case_label} simulated scan and aligned dorsal reference"
    stem = "bregma_lambda_overlay" if args.case == "full" else "bregma_lambda_cropped_overlay"
    render_png(reference, scan, points, rotation, center, subtitle, DIRECTORY / f"{stem}.png")
    if not args.png_only:
        render_html(reference, scan, points, rotation, translation, DIRECTORY / f"{stem}.html")


if __name__ == "__main__":
    main()
