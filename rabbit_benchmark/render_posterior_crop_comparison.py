"""Compare the saved posterior-crop results using the same view and scale."""

import json

import numpy as np

from render_bregma_lambda_overlay import (
    DIRECTORY, INK, LANDMARKS, REFERENCE_COLOR, SCAN_COLOR, read_cloud, plt,
)
from matplotlib.lines import Line2D


def read_json(name):
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


def draw_clouds(ax, reference, scan, cut, limits, zoom):
    ax.set_facecolor("#f5f7f9")
    ax.scatter(reference[:, 0], reference[:, 2], s=3.5 if zoom else 1.4,
               color=REFERENCE_COLOR, alpha=0.46, linewidths=0)
    ax.scatter(scan[:, 0], scan[:, 2], s=8 if zoom else 3,
               color=SCAN_COLOR, alpha=0.70, linewidths=0)
    ax.axhline(cut, ls="--", color="#3e4a59", lw=1.4)
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.tick_params(labelsize=8, colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#dce3e9")


def mark(ax, point, name, color, shape, rejected=False):
    xy = point[[0, 2]]
    ax.scatter(*xy, s=72, marker=shape, facecolor="white" if rejected else color,
               edgecolor=color if rejected else "white", linewidth=1.6, zorder=5)
    offset = (-13, 14) if shape == "o" else (13, -13)
    ax.annotate(name, xy, xytext=offset, textcoords="offset points",
                ha="right" if shape == "o" else "left", va="center", fontsize=9,
                weight="bold", color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.96),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.1,
                                linestyle="--" if rejected else "-"))


def main():
    experiment = read_json("bregma_lambda_posterior_crop_comparison.json")
    truth = np.asarray(experiment["known_reference_to_scan"])
    rotation, translation = truth[:3, :3], truth[:3, 3]
    cut = experiment["reference_cut_z"]
    scan = (read_cloud("posterior_crop_scan.ply") - translation) @ rotation
    if len(scan) != experiment["points_retained"] or np.any(scan[:, 2] < cut):
        raise ValueError("The input cloud does not match the saved posterior crop")
    reference = read_cloud("reference_dorsal.ply")
    lambda_ref = np.asarray(read_json("bregma_lambda_candidates.json")["Lambda_candidate"])
    before = read_json("bregma_lambda_posterior_crop_result.json")
    after = read_json("bregma_lambda_posterior_crop_coverage_result.json")
    if (before["landmarks"]["Lambda_candidate"]["status"] != "mapped"
            or after["landmarks"]["Lambda_candidate"]["status"] != "outside_scan_coverage"
            or after["landmarks"]["Lambda_candidate"]["xyz_mm"] is not None):
        raise ValueError("Expected the saved false positive and the subsequent rejection")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.7), facecolor="white",
                             gridspec_kw={"height_ratios": (1.5, 1)})
    fig.subplots_adjust(left=0.105, right=0.955, top=0.83, bottom=0.15,
                        hspace=0.29, wspace=0.15)
    fig.text(0.075, 0.942, "Plan B | Scan coverage check", fontsize=23, weight="bold", color=INK)
    fig.text(0.075, 0.902, "Same posterior crop in both columns · Shared dorsal view and scale",
             fontsize=11, color="#4b5665")
    fig.legend(handles=[
        Line2D([], [], color=REFERENCE_COLOR, marker="o", ls="none", label="Aligned reference"),
        Line2D([], [], color=SCAN_COLOR, marker=".", ls="none", label="Retained scan"),
        Line2D([], [], color="#3e4a59", ls="--", label="Crop boundary (Z = 41.5)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False,
        fontsize=10, labelcolor=INK)

    for col, (title, result, rejected) in enumerate((
        ("Before coverage check", before, False), ("With coverage check", after, True),
    )):
        transform = np.asarray(result["reference_to_scan_mm"])
        aligned_scan_frame = reference @ transform[:3, :3].T + transform[:3, 3]
        aligned = (aligned_scan_frame - translation) @ rotation
        bregma = (np.asarray(result["landmarks"]["Bregma_candidate"]["xyz_mm"])
                  - translation) @ rotation
        lambda_scan_frame = (
            lambda_ref @ transform[:3, :3].T + transform[:3, 3]
            if rejected else np.asarray(result["landmarks"]["Lambda_candidate"]["xyz_mm"])
        )
        lambda_point = (lambda_scan_frame - translation) @ rotation
        if not lambda_point[2] < cut < bregma[2]:
            raise ValueError("Expected Lambda and Bregma on opposite sides of the crop")
        overview, detail = axes[:, col]
        draw_clouds(overview, aligned, scan, cut, ((24, 76), (31, 101)), zoom=False)
        draw_clouds(detail, aligned, scan, cut, ((46.2, 53.8), (39.2, 43.8)), zoom=True)
        overview.set_title(title, fontsize=14, weight="bold", color=INK, pad=9)
        detail.set_title("Lambda at the crop edge", fontsize=10, color=INK, pad=7)
        detail.set_xlabel("Reference X (model units)", fontsize=9, color=INK)
        if col == 0:
            for ax in (overview, detail):
                ax.set_ylabel("Reference Z (model units)", fontsize=9, color=INK)
        mark(overview, bregma, "Bregma\nLocated", LANDMARKS[0][2], LANDMARKS[0][3])
        status = "Rejected" if rejected else "Reported"
        location = "Reference position" if rejected else "Outside scan"
        mark(overview, lambda_point, f"Lambda\n{status}", LANDMARKS[1][2],
             LANDMARKS[1][3], rejected)
        mark(detail, lambda_point, f"Lambda {status.lower()}\n{location}", LANDMARKS[1][2],
             LANDMARKS[1][3], rejected)
    fig.text(0.075, 0.029, "NIH 3D 3DPX-004799 v3 · SMART Neuro at Penn Vet · CC BY-NC-SA 4.0",
             fontsize=8, color="#4b5665")
    fig.savefig(DIRECTORY / "bregma_lambda_posterior_crop_before_after.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
