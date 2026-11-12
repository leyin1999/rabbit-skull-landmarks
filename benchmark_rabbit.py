"""Exercise Plan B on NIH 3D 3DPX-004799 v3 with known scan poses."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import open3d as o3d
import trimesh


SOURCE_SHA256 = "b5c4e459e206a2832ec2e4883937d8a6b5f2bc9a86bbd181c7997217de172a75"
SOURCE_URL = "https://3d.nih.gov/entries/3DPX-004799?version=3"


def dorsal_surface(scene, x_bounds, z_bounds, spacing, tilt=0):
    # In this STL the dorsal surface faces -Y. Cast through an XZ grid and
    # keep only the first intersection so internal surfaces are excluded.
    x, z = np.meshgrid(np.arange(*x_bounds, spacing), np.arange(*z_bounds, spacing))
    origins = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    directions = np.tile((tilt, 1.0, 0.0), (x.size, 1))
    rays = np.column_stack((origins, directions)).astype(np.float32)
    distances = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    hits = np.isfinite(distances)
    return origins[hits] + distances[hits, None] * directions[hits]


def write_cloud(path, points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f"Could not write {path}")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def benchmark(mesh_path, output):
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Use the NIH 3D v3 STL listed in the README. This crop is model-specific.")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(mesh.vertices.astype(np.float32)),
        o3d.core.Tensor(mesh.faces.astype(np.uint32)),
    )
    output.mkdir(parents=True, exist_ok=True)
    reference = dorsal_surface(scene, (28, 73), (32, 100), 0.25)
    reference_path = output / "reference_dorsal.ply"
    write_cloud(reference_path, reference)

    # Points on the model used to check geometry. They have no anatomical labels.
    marks = {}
    for name, x, z in (("test_A", 51, 46), ("test_B", 50, 67), ("test_C", 49, 83)):
        index = np.argmin(np.linalg.norm(reference[:, [0, 2]] - (x, z), axis=1))
        marks[name] = reference[index].tolist()
    marks_path = output / "reference_points.json"
    write_json(marks_path, marks)

    summary = {
        "source": SOURCE_URL,
        "source_sha256": digest,
        "scan_type": "simulated depth from the same public mesh",
        "coordinate_unit": "STL model unit, treated as mm by the CLI",
        "cases": {},
    }
    rng = np.random.default_rng(13)
    cases = (
        ("full", (28.1, 72), (32.1, 100), 0.01, (0.16, -0.21, 0.31), (28, -19, 140)),
        ("cropped", (33.2, 69), (38.1, 93), -0.04, (-0.2, 0.14, -0.35), (-15, 22, 180)),
        ("missing_point", (32, 70), (39, 76), 0.02, (0.28, 0.17, -0.22), (8, 13, 151)),
    )
    failed = []
    for name, x_bounds, z_bounds, tilt, angles, translation in cases:
        sample = dorsal_surface(scene, x_bounds, z_bounds, 0.40, tilt=tilt)
        rotation = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        scan = sample @ rotation.T + translation + rng.normal(0, 0.1, sample.shape)
        scan_path = output / f"{name}_scan.ply"
        write_cloud(scan_path, scan)

        result_path = output / f"{name}_result.json"
        runner = Path(__file__).with_name("recognize_landmarks.py")
        run = subprocess.run([
            sys.executable, str(runner),
            "--reference", str(reference_path), "--reference-unit", "mm",
            "--scan", str(scan_path), "--scan-unit", "mm",
            "--landmarks", str(marks_path), "--output", str(result_path),
        ], check=False)
        if run.returncode not in (0, 2):
            raise RuntimeError(f"Recognition failed for {name} with exit code {run.returncode}")
        result = json.loads(result_path.read_text(encoding="utf-8"))

        errors = {}
        for label, point in marks.items():
            mapped = result["landmarks"].get(label, {}).get("xyz_mm")
            expected = np.asarray(point) @ rotation.T + translation
            errors[label] = float(np.linalg.norm(mapped - expected)) if mapped is not None else None
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        summary["cases"][name] = {
            "status": result["status"],
            "known_reference_to_scan": pose.tolist(),
            "registration": result.get("registration"),
            "landmark_errors": errors,
        }
        expected_labels = {"test_A", "test_B"} if name == "missing_point" else set(marks)
        within_tolerance = {label for label, error in errors.items() if error is not None and error <= 0.5}
        mapped_labels = {label for label, error in errors.items() if error is not None}
        if within_tolerance != expected_labels or mapped_labels != expected_labels:
            failed.append(name)

    write_json(output / "summary.json", summary)
    if failed:
        raise SystemExit(f"Unexpected landmark results: {', '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="Original NIH 3D v3 STL")
    parser.add_argument("output", type=Path, help="Directory for inputs and results")
    args = parser.parse_args()
    benchmark(args.mesh, args.output)
