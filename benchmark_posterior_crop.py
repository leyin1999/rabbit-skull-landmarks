"""Crop the existing rabbit scan without changing the retained samples."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmark_rabbit import write_cloud, write_json
from rabbit_benchmark.render_bregma_lambda_overlay import (
    INK, LANDMARKS, REFERENCE_COLOR, SCAN_COLOR, read_cloud, plt,
)
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rabbit_benchmark"
CUT_Z = 41.5


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def render_overlay(reference, scan, landmarks):
    fig, (overview, detail) = plt.subplots(1, 2, figsize=(10.8, 6.5), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.95, top=0.79, bottom=0.19, wspace=0.24)
    fig.text(0.075, 0.937, "Plan B | Posterior crop test", fontsize=23, weight="bold", color=INK)
    fig.text(0.075, 0.887, "Same simulated pose and noise · Scan cropped at reference Z = 41.5",
             fontsize=11, color="#4b5665")
    fig.legend(handles=[
        Line2D([], [], color=REFERENCE_COLOR, marker="o", ls="none", label="Aligned reference"),
        Line2D([], [], color=SCAN_COLOR, marker=".", ls="none", label="Retained scan"),
        Line2D([], [], color="#3e4a59", ls="--", label="Crop boundary"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False,
        fontsize=10, labelcolor=INK)

    for ax in (overview, detail):
        ax.set_facecolor("#f5f7f9")
        ax.scatter(reference[:, 0], reference[:, 2], s=1.6, color=REFERENCE_COLOR,
                   alpha=0.42, linewidths=0)
        ax.scatter(scan[:, 0], scan[:, 2], s=3.2, color=SCAN_COLOR,
                   alpha=0.60, linewidths=0)
        ax.axhline(CUT_Z, ls="--", color="#3e4a59", lw=1.4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9, colors=INK)
        ax.set_xlabel("Reference X (model units)", fontsize=9, color=INK)
        for spine in ax.spines.values():
            spine.set_color("#dce3e9")

    overview.set(xlim=(24, 78), ylim=(30, 102), ylabel="Reference Z (model units)")
    detail.set(xlim=(46.5, 53.5), ylim=(38.2, 44.7))
    overview.set_title("Dorsal view", loc="left", weight="bold", color=INK)
    detail.set_title("Lambda at the cut edge", loc="left", weight="bold", color=INK)
    detail.text(46.75, 43.95, "Inside scan", fontsize=9, color=INK)
    detail.text(46.75, 38.5, "Outside scan", fontsize=9, color=INK)
    for key, label, color, marker, _ in LANDMARKS:
        point = landmarks.get(key)
        if point is None:
            continue
        panels = (overview,) if label == "Bregma" else (overview, detail)
        for ax in panels:
            xy = point[[0, 2]]
            offset = (-8, 13) if label == "Bregma" else (14, -17)
            ax.scatter(*xy, s=72, marker=marker, color=color, edgecolor="white", lw=1.4, zorder=5)
            ax.annotate(label + " returned", xy, xytext=offset, textcoords="offset points",
                        ha="right" if label == "Bregma" else "left", va="center",
                        fontsize=9, weight="bold", color=color,
                        bbox=dict(fc="white", ec="none", alpha=0.95, pad=2),
                        arrowprops=dict(arrowstyle="-", color=color), zorder=6)
    fig.text(0.075, 0.025, "NIH 3D 3DPX-004799 v3 · SMART Neuro at Penn Vet · CC BY-NC-SA 4.0",
             fontsize=8, color="#4b5665")
    fig.savefig(DATA / "bregma_lambda_posterior_crop_overlay.png", dpi=160)
    plt.close(fig)


def main():
    truth = np.asarray(read_json("summary.json")["cases"]["cropped"]["known_reference_to_scan"])
    rotation, translation = truth[:3, :3], truth[:3, 3]
    scan = read_cloud("cropped_scan.ply")
    in_reference = (scan - translation) @ rotation
    keep = in_reference[:, 2] >= CUT_Z
    scan_name = "posterior_crop_scan.ply"
    write_cloud(DATA / scan_name, scan[keep])
    np.testing.assert_array_equal(read_cloud(scan_name), scan[keep])

    result_name = "bregma_lambda_posterior_crop_result.json"
    run = subprocess.run([
        sys.executable, str(ROOT / "recognize_landmarks.py"),
        "--scan", str(DATA / scan_name), "--scan-unit", "mm",
        "--reference", str(DATA / "reference_dorsal.ply"), "--reference-unit", "mm",
        "--landmarks", str(DATA / "bregma_lambda_candidates.json"),
        "--output", str(DATA / result_name),
    ], check=False)
    if run.returncode not in (0, 2):
        raise RuntimeError(f"Recognition failed with exit code {run.returncode}")
    result = read_json(result_name)
    baseline = read_json("bregma_lambda_cropped_result.json")
    comparison = {}
    mapped_reference = {}
    for name, point in read_json("bregma_lambda_candidates.json").items():
        expected = np.asarray(point) @ rotation.T + translation
        before = baseline["landmarks"][name]["xyz_mm"]
        after = result["landmarks"].get(name, {}).get("xyz_mm")
        error_before = float(np.linalg.norm(before - expected))
        error_after = float(np.linalg.norm(after - expected)) if after is not None else None
        within_crop = point[2] >= CUT_Z
        comparison[name] = {
            "expected_inside_crop": within_crop,
            "reported": after is not None,
            "false_positive": not within_crop and after is not None,
            "error_before": error_before,
            "error_after": error_after,
            "error_change": error_after - error_before if error_after is not None else None,
            "distance_to_crop_boundary": abs(point[2] - CUT_Z),
        }
        if after is not None:
            mapped_reference[name] = (after - translation) @ rotation

    write_json(DATA / "bregma_lambda_posterior_crop_comparison.json", {
        "source_scan": "cropped_scan.ply",
        "reference_cut_z": CUT_Z,
        "retained_region": "reference Z >= cut",
        "known_reference_to_scan": truth.tolist(),
        "points_before": len(scan),
        "points_retained": int(keep.sum()),
        "coordinate_unit": "STL model unit",
        "landmarks": comparison,
    })
    if "reference_to_scan_mm" in result:
        fit = np.asarray(result["reference_to_scan_mm"])
        aligned = read_cloud("reference_dorsal.ply") @ fit[:3, :3].T + fit[:3, 3]
        render_overlay((aligned - translation) @ rotation, in_reference[keep], mapped_reference)


if __name__ == "__main__":
    main()
