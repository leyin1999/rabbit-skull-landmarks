"""Remove growing disks from a fixed scan to test coverage under occlusion."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from benchmark_posterior_crop import DATA, ROOT, read_json
from benchmark_rabbit import write_cloud, write_json
from rabbit_benchmark.render_bregma_lambda_overlay import (
    INK, LANDMARKS, REFERENCE_COLOR, SCAN_COLOR, read_cloud, plt,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


BREGMA, LAMBDA = (landmark[0] for landmark in LANDMARKS)


def measure(radius, scan, radial_distance, scratch, expected):
    retained = scan[radial_distance >= radius]
    scan_path = scratch / f"scan_{radius:.2f}.ply"
    result_path = scratch / f"result_{radius:.2f}.json"
    write_cloud(scan_path, retained)
    np.testing.assert_array_equal(read_cloud(scan_path), retained)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    process = subprocess.run([
        sys.executable, "-B", str(ROOT / "recognize_landmarks.py"),
        "--scan", str(scan_path), "--scan-unit", "mm",
        "--reference", str(DATA / "reference_dorsal.ply"), "--reference-unit", "mm",
        "--landmarks", str(DATA / "bregma_lambda_candidates.json"),
        "--output", str(result_path),
    ], capture_output=True, text=True, env=env)
    if process.returncode not in (0, 2):
        raise RuntimeError(f"Recognition failed at radius {radius}: {process.stderr}")

    report = read_json(result_path)
    errors = {}
    for name, truth in expected.items():
        xyz = report["landmarks"].get(name, {}).get("xyz_mm")
        errors[name] = float(np.linalg.norm(xyz - truth)) if xyz is not None else None
    return {
        "radius": float(radius),
        "retained_points": len(retained),
        "removed_points": len(scan) - len(retained),
        "mapping_errors": errors,
        "report": report,
    }


def draw_detail(ax, row, reference, scan, radial_distance, center, rotation, translation,
                label, extent):
    report = row["report"]
    radius = row["radius"]
    aligned_transform = np.asarray(report["reference_to_scan_mm"])
    aligned = reference @ aligned_transform[:3, :3].T + aligned_transform[:3, 3]
    aligned = ((aligned - translation) @ rotation)[:, [0, 2]] - center
    local_scan = scan[radial_distance >= radius][:, [0, 2]] - center
    ax.scatter(*aligned.T, s=3.5, color=REFERENCE_COLOR, alpha=0.45, linewidths=0)
    ax.scatter(*local_scan.T, s=7.5, color=SCAN_COLOR, alpha=0.75, linewidths=0)
    if radius:
        ax.add_patch(Circle((0, 0), radius, fill=False, ls="--", lw=1.3, ec="#576576"))
    status = report["landmarks"][LAMBDA]
    color = LANDMARKS[1][2]
    if status["xyz_mm"] is None:
        position = np.zeros(2)
        face = "white"
        result = "Rejected"
    else:
        point = (np.asarray(status["xyz_mm"]) - translation) @ rotation
        position = point[[0, 2]] - center
        face = color
        result = "Returned"
    ax.scatter(*position, s=72, marker="D", facecolor=face,
               edgecolor=color if face == "white" else "white", lw=1.6, zorder=5)
    ax.set_title(f"{label} | {result}", fontsize=11, color=INK, weight="bold")
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.tick_params(labelsize=8, colors=INK)


def render(sweep, previous, reference, scan, radial_distance, center, rotation, translation):
    rows = sweep["radii"]
    previous_rows = previous["radii"]
    accepted = np.array([
        row["report"]["landmarks"].get(LAMBDA, {}).get("xyz_mm") is not None for row in rows
    ])
    radii = np.array([row["radius"] for row in rows])
    errors = [row["mapping_errors"][BREGMA] for row in rows]
    previous_accepted = np.array([
        row["report"]["landmarks"].get(LAMBDA, {}).get("xyz_mm") is not None
        for row in previous_rows
    ])
    previous_errors = [row["mapping_errors"][BREGMA] for row in previous_rows]
    first_changed = next(iter(np.flatnonzero(previous_accepted & ~accepted)), len(rows) - 1)
    radius = radii[first_changed]
    extent = max(1.15, radius + 0.35)
    selections = (
        (rows[0], "No hole"),
        (previous_rows[first_changed], f"Radius {radius:.2f} before"),
        (rows[first_changed], f"Radius {radius:.2f} after"),
    )

    fig = plt.figure(figsize=(11, 8), facecolor="white")
    grid = fig.add_gridspec(3, 3, height_ratios=(2.25, 1, 1.25), hspace=0.66,
                           left=0.125, right=0.955, top=0.83, bottom=0.12, wspace=0.22)
    fig.text(0.075, 0.943, "Plan B | Local hole detection", fontsize=23,
             color=INK, weight="bold")
    fig.text(0.075, 0.902, "One simulated scan · Pose and noise held fixed · Boundary inferred from points",
             fontsize=11, color="#4b5665")
    for col, (row, label) in enumerate(selections):
        ax = fig.add_subplot(grid[0, col])
        draw_detail(ax, row, reference, scan, radial_distance, center, rotation, translation,
                    label, extent)
        if col == 0:
            ax.set_ylabel("Relative Z (model units)", fontsize=9, color=INK)
        if col == 1:
            ax.set_xlabel("Relative X (model units)", fontsize=9, color=INK)
    fig.legend(handles=[
        Line2D([], [], color=REFERENCE_COLOR, marker="o", ls="none", label="Aligned reference"),
        Line2D([], [], color=SCAN_COLOR, marker=".", ls="none", label="Retained scan"),
        Line2D([], [], color="#576576", ls="--", label="Removed disk"),
    ], loc="upper center", bbox_to_anchor=(0.55, 0.887), ncol=3,
        frameon=False, fontsize=9, labelcolor=INK)

    status_ax = fig.add_subplot(grid[1, :])
    error_ax = fig.add_subplot(grid[2, :], sharex=status_ax)
    color = LANDMARKS[1][2]
    for level, mask in enumerate((accepted, previous_accepted)):
        status_ax.scatter(radii[mask], np.full(mask.sum(), level), s=42, color=color,
                          label="Returned" if level == 0 else None)
        status_ax.scatter(radii[~mask], np.full((~mask).sum(), level), s=42,
                          facecolors="white", edgecolors=color, linewidths=1.4, marker="D",
                          label="Rejected" if level == 0 else None)
    status_ax.set(ylim=(-0.35, 1.35), yticks=(0, 1), yticklabels=("After", "Before"), ylabel="Lambda")
    status_ax.tick_params(labelbottom=False)
    status_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=2,
                     frameon=False, fontsize=9)
    error_ax.plot(radii, errors, color="#00766d", marker="o", ms=4.5, lw=1.4)
    if np.allclose(np.array(errors, dtype=float), np.array(previous_errors, dtype=float),
                   rtol=0, atol=1e-8, equal_nan=True):
        error_ax.lines[-1].set_label("Before and after overlap")
    else:
        error_ax.lines[-1].set_label("After")
        error_ax.plot(radii, previous_errors, color="#687585", ls=":", label="Before")
    error_ax.set(ylim=(0, 1.25 * max(error for error in errors + previous_errors if error is not None)),
                 ylabel="Bregma error\n(model units)",
                 xlabel="Hole radius in the dorsal projection (STL model units)", xticks=radii)
    error_ax.legend(loc="lower right", frameon=False, fontsize=9)
    for ax in (status_ax, error_ax):
        ax.set_xlim(-0.08, radii[-1] + 0.08)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce4e9", linewidth=0.8)
        ax.tick_params(colors=INK, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#dce3e9")
    fig.text(0.075, 0.035, "NIH 3D 3DPX-004799 v3 · SMART Neuro at Penn Vet · CC BY-NC-SA 4.0",
             fontsize=8, color="#4b5665")
    fig.savefig(DATA / "bregma_lambda_occlusion_sweep_local_hole.png", dpi=160)
    plt.close(fig)


def main():
    truth = np.asarray(read_json("summary.json")["cases"]["cropped"]["known_reference_to_scan"])
    rotation, translation = truth[:3, :3], truth[:3, 3]
    marks = read_json("bregma_lambda_candidates.json")
    expected = {name: rotation @ point + translation for name, point in marks.items()}
    scan = read_cloud("cropped_scan.ply")
    in_reference = (scan - translation) @ rotation
    center = np.asarray(marks[LAMBDA])[[0, 2]]
    radial_distance = np.linalg.norm(in_reference[:, [0, 2]] - center, axis=1)
    previous = read_json("bregma_lambda_occlusion_sweep.json")
    radii = [row["radius"] for row in previous["radii"]]
    with tempfile.TemporaryDirectory(prefix="plan-b-occlusion-") as folder:
        scratch = Path(folder)
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(
                lambda radius: measure(radius, scan, radial_distance, scratch, expected), radii
            ))
    sweep = {
        "source_scan": "cropped_scan.ply",
        "unit": "STL model unit",
        "known_reference_to_scan": truth.tolist(),
        "occlusion_center_reference_xz": center.tolist(),
        "radii": rows,
    }
    write_json(DATA / "bregma_lambda_occlusion_sweep_local_hole.json", sweep)
    render(sweep, previous, read_cloud("reference_dorsal.ply"), in_reference,
           radial_distance, center, rotation, translation)


if __name__ == "__main__":
    main()
