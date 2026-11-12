"""Test Plan B across posterior crop boundaries without regenerating the scan."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from benchmark_posterior_crop import DATA, ROOT, read_json
from benchmark_rabbit import write_cloud, write_json
from rabbit_benchmark.render_bregma_lambda_overlay import INK, read_cloud, plt


BREGMA = "Bregma_candidate"
LAMBDA = "Lambda_candidate"
BLUE = "#286792"
MAGENTA = "#941957"
BOUNDARY_TOLERANCE = 0.01


def measure(cut, scratch, scan, reference_z, marks, expected, truth, with_boundary):
    keep = reference_z >= cut
    scan_path = scratch / f"scan_{cut:.1f}.ply"
    result_path = scratch / f"result_{cut:.1f}.json"
    write_cloud(scan_path, scan[keep])
    np.testing.assert_array_equal(read_cloud(scan_path), scan[keep])
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    command = [
        sys.executable, "-B", str(ROOT / "recognize_landmarks.py"),
        "--scan", str(scan_path), "--scan-unit", "mm",
        "--reference", str(DATA / "reference_dorsal.ply"), "--reference-unit", "mm",
        "--landmarks", str(DATA / "bregma_lambda_candidates.json"),
        "--output", str(result_path),
    ]
    if with_boundary:
        normal = truth[:3, 2]
        offset = -cut - normal @ truth[:3, 3]
        boundary_path = scratch / f"boundary_{cut:.1f}.json"
        write_json(boundary_path, {
            "planes": [[*normal.tolist(), float(offset)]],
            "tolerance_mm": BOUNDARY_TOLERANCE,
        })
        command.extend(["--coverage-planes", str(boundary_path)])
    process = subprocess.run(command, capture_output=True, text=True, env=env)
    if process.returncode not in (0, 2):
        raise RuntimeError(f"Recognition failed at cut Z={cut}: {process.stderr}")

    report = read_json(result_path)
    landmarks = {}
    for name, point in marks.items():
        status = report["landmarks"].get(name, {})
        mapped = status.get("xyz_mm")
        position = "inside" if point[2] > cut else "outside" if point[2] < cut else "boundary"
        landmarks[name] = {
            "position_relative_to_cut": position,
            "status": status.get("status", "registration_rejected"),
            "xyz_scan": mapped,
            "nearest_scan_distance": status.get("nearest_scan_distance_mm"),
            "coverage_boundary_distance": status.get("coverage_boundary_distance_mm"),
            "mapping_error": float(np.linalg.norm(mapped - expected[name])) if mapped is not None else None,
        }
    return {
        "cut_reference_z": float(cut),
        "retained_points": int(keep.sum()),
        "registration_status": report["status"],
        "coverage_boundary_source": report.get("coverage_boundary_source"),
        "coverage_boundary_tolerance": report.get("coverage_boundary_tolerance_mm"),
        "landmarks": landmarks,
    }


def render(sweep, output, previous=None):
    rows = sweep["cuts"]
    relative_cut = np.array([row["cut_reference_z"] - sweep["lambda_reference_z"] for row in rows])
    lambda_results = [row["landmarks"][LAMBDA] for row in rows]
    mapped = np.array([result["xyz_scan"] is not None for result in lambda_results])
    inside = relative_cut < 0
    outside = relative_cut > 0
    boundary = relative_cut == 0
    bregma_errors = np.array([row["landmarks"][BREGMA]["mapping_error"] for row in rows], dtype=float)

    fig, (status_ax, error_ax) = plt.subplots(
        2, 1, figsize=(10.3, 6.8), sharex=True, facecolor="white",
        gridspec_kw={"height_ratios": (1, 1.35)},
    )
    fig.subplots_adjust(left=0.16, right=0.95, top=0.77, bottom=0.13, hspace=0.44)
    title = (
        "Plan B | Known scan boundary" if "coverage_boundary_tolerance" in sweep
        else "Plan B | Boundary estimated from point clouds"
    )
    fig.text(0.075, 0.94, title, fontsize=23,
             color=INK, weight="bold")
    fig.text(0.075, 0.895, "21 cuts of the same scan · Retained coordinates preserve the pose and noise",
             fontsize=11, color="#4b5665")
    for ax in (status_ax, error_ax):
        ax.axvspan(-1.06, 0, color="#eff6f9", zorder=0)
        ax.axvspan(0, 1.06, color="#f8f4f0", zorder=0)
        ax.axvline(0, color="#576576", lw=1.2, ls="--", zorder=2)
        ax.set_xlim(-1.06, 1.06)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce4e9", linewidth=0.8)
        ax.tick_params(colors=INK, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#dce3e9")

    status_ax.text(-0.52, 1.11, "Lambda inside scan", transform=status_ax.get_xaxis_transform(),
                   ha="center", fontsize=10, color=BLUE)
    status_ax.text(0.52, 1.11, "Lambda outside scan", transform=status_ax.get_xaxis_transform(),
                   ha="center", fontsize=10, color="#7a4a28")
    correct = (inside & mapped) | (outside & ~mapped)
    status_ax.scatter(relative_cut[correct], mapped[correct], color=BLUE, s=45, zorder=4,
                      label="Matches coverage")
    mismatch = (inside & ~mapped) | (outside & mapped)
    if np.any(mismatch):
        status_ax.scatter(relative_cut[mismatch], mapped[mismatch], color=MAGENTA, marker="X",
                          s=78, zorder=5, label="Coverage mismatch")
    if previous is not None:
        previous_mapped = np.array([
            row["landmarks"][LAMBDA]["xyz_scan"] is not None for row in previous["cuts"]
        ])
        corrected = outside & previous_mapped & ~mapped
        status_ax.scatter(relative_cut[corrected], previous_mapped[corrected], color=MAGENTA,
                          alpha=0.65, marker="X", s=78, zorder=5, label="Previous false positive")
    status_ax.scatter(relative_cut[boundary], mapped[boundary], facecolor="white", edgecolor=INK,
                      marker="D", s=56, linewidth=1.4, zorder=5, label="On boundary")
    status_ax.set(ylim=(-0.45, 1.45), yticks=[0, 1], yticklabels=["Rejected", "Returned"],
                  ylabel="Lambda result")
    status_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                     frameon=False, fontsize=9, labelcolor=INK)

    error_ax.plot(relative_cut, bregma_errors, color="#00766d", marker="o", ms=4.6, lw=1.45)
    error_ax.axhline(sweep["uncut_bregma_error"], color="#687585", ls=":", lw=1.5,
                     label="Before additional crop")
    error_ax.set_ylim(bottom=0)
    error_ax.set(ylabel="Bregma mapping error\n(STL model units)",
                 xlabel="Cut Z relative to Lambda (STL model units)")
    error_ax.set_xticks(np.linspace(-1, 1, 11))
    error_ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.075, 0.036, "NIH 3D 3DPX-004799 v3 · SMART Neuro at Penn Vet · CC BY-NC-SA 4.0",
             fontsize=8, color="#4b5665")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-boundary", action="store_true",
                        help="Provide the known scan-space crop plane at each step")
    args = parser.parse_args()
    experiment = read_json("bregma_lambda_posterior_crop_comparison.json")
    truth = np.asarray(experiment["known_reference_to_scan"])
    marks = read_json("bregma_lambda_candidates.json")
    scan = read_cloud("cropped_scan.ply")
    reference_z = (scan - truth[:3, 3]) @ truth[:3, 2]
    expected = {name: truth[:3, :3] @ point + truth[:3, 3] for name, point in marks.items()}
    baseline = read_json("bregma_lambda_cropped_result.json")["landmarks"][BREGMA]["xyz_mm"]
    cuts = np.round(marks[LAMBDA][2] + np.arange(-10, 11) / 10, 1)
    with tempfile.TemporaryDirectory(prefix="plan-b-crop-sweep-") as folder:
        scratch = Path(folder)
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(
                lambda cut: measure(
                    cut, scratch, scan, reference_z, marks, expected, truth, args.with_boundary
                ), cuts
            ))
    sweep = {
        "source_scan": "cropped_scan.ply",
        "retained_region": "reference Z >= cut",
        "unit": "STL model unit",
        "known_reference_to_scan": truth.tolist(),
        "lambda_reference_z": marks[LAMBDA][2],
        "uncut_bregma_error": float(np.linalg.norm(baseline - expected[BREGMA])),
        "cuts": rows,
    }
    stem = "bregma_lambda_crop_sweep"
    previous = read_json(f"{stem}.json")
    if args.with_boundary:
        stem += "_boundary"
        sweep["coverage_boundary_tolerance"] = BOUNDARY_TOLERANCE
    else:
        stem += "_estimated"
    write_json(DATA / f"{stem}.json", sweep)
    render(sweep, DATA / f"{stem}.png", previous)


if __name__ == "__main__":
    main()
