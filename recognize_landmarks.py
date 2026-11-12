"""Register a dorsal skull scan and transfer landmarks from its reference."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, QhullError, cKDTree
import trimesh


registration = o3d.pipelines.registration
MIN_POINTS = 50
UNIT_TO_MM = {"mm": 1.0, "m": 1000.0}


def load_cloud(path, unit, allow_mesh=False):
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if allow_mesh and suffix in {".stl", ".obj"}:
        # Open3D's mesh importer can fail on non-ASCII paths on Windows.
        with path.open("rb") as stream:
            source_mesh = trimesh.load_mesh(stream, file_type=suffix[1:], process=False)
        if not np.isfinite(source_mesh.area) or source_mesh.area <= 0:
            raise ValueError(f"No finite surface area in reference mesh: {path}")
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(source_mesh.vertices),
            o3d.utility.Vector3iVector(source_mesh.faces),
        )
        cloud = mesh.sample_points_uniformly(number_of_points=150_000)
    else:
        cloud = o3d.io.read_point_cloud(
            str(path), remove_nan_points=True, remove_infinite_points=True
        )
    if len(cloud.points) < MIN_POINTS or not np.isfinite(cloud.points).all():
        raise ValueError(f"Need at least {MIN_POINTS} finite points in {path}")
    cloud.scale(UNIT_TO_MM[unit], center=(0, 0, 0))
    return cloud


def load_landmarks(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Landmarks must be a nonempty JSON object of names and XYZ arrays")
    names = list(data)
    if any(not name.strip() for name in names):
        raise ValueError("Landmark names must be nonempty")
    points = np.asarray(list(data.values()), dtype=float)
    if points.shape != (len(names), 3) or not np.isfinite(points).all():
        raise ValueError("Every landmark must have three finite coordinates")
    return names, points


def load_initial_pose(path):
    matrix = np.asarray(json.loads(path.read_text(encoding="utf-8-sig")), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Initial pose must be a finite 4×4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("Initial pose must be a rigid scan-to-reference transform in mm")
    return matrix


def load_coverage_planes(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "planes" not in data:
        raise ValueError("Coverage JSON must contain a planes array")
    planes = np.asarray(data["planes"], dtype=float)
    if planes.ndim != 2 or planes.shape[1] != 4 or not len(planes) or not np.isfinite(planes).all():
        raise ValueError("Coverage planes must be nonempty finite [nx, ny, nz, d] arrays")
    lengths = np.linalg.norm(planes[:, :3], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths < 1e-12):
        raise ValueError("Each coverage plane must have a finite, nonzero normal")
    tolerance = float(data.get("tolerance_mm", 0.0))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Coverage tolerance_mm must be finite and nonnegative")
    planes = planes / lengths[:, None]
    if not np.isfinite(planes).all():
        raise ValueError("Coverage plane offsets must be finite after normalization")
    return planes, tolerance


def downsample(cloud, voxel, normals=False):
    reduced = cloud.voxel_down_sample(voxel)
    if len(reduced.points) < MIN_POINTS:
        raise ValueError(f"Fewer than {MIN_POINTS} points at {voxel:g} mm voxel size")
    if normals:
        # Ignore normals stored by the scanner or model so both use the same estimator.
        reduced.normals = o3d.utility.Vector3dVector()
        reduced.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(2.5 * voxel, 40))
    return reduced


def initial_poses(scan, reference, voxel):
    # FPFH depends on normal sign. Try both relative orientations on the surface.
    scan.orient_normals_consistent_tangent_plane(15)
    reference.orient_normals_consistent_tangent_plane(15)
    search = o3d.geometry.KDTreeSearchParamHybrid(5 * voxel, 100)
    reference_features = registration.compute_fpfh_feature(reference, search)
    distance = 1.5 * voxel
    for _ in range(2):
        scan_features = registration.compute_fpfh_feature(scan, search)
        result = registration.registration_ransac_based_on_feature_matching(
            scan, reference, scan_features, reference_features, True, distance,
            registration.TransformationEstimationPointToPoint(False), 3,
            [
                registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                registration.CorrespondenceCheckerBasedOnDistance(distance),
            ],
            registration.RANSACConvergenceCriteria(200_000, 0.999),
        )
        if len(result.correspondence_set) >= MIN_POINTS:
            yield result.transformation
        scan.normals = o3d.utility.Vector3dVector(-np.asarray(scan.normals))


def refine(scan_coarse, reference_coarse, scan_fine, reference_fine, initial, voxel):
    pose = initial
    for source, target, distance in (
        (scan_coarse, reference_coarse, 2 * voxel),
        (scan_fine, reference_fine, voxel),
    ):
        result = registration.registration_icp(
            source, target, distance, pose,
            registration.TransformationEstimationPointToPlane(
                registration.TukeyLoss(distance)
            ),
            registration.ICPConvergenceCriteria(max_iteration=60),
        )
        if len(result.correspondence_set) < MIN_POINTS:
            return None
        pose = result.transformation
    return pose


def transform_points(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def tangent_plane_axes(reference_offsets):
    if len(reference_offsets) < 3:
        return None
    centered = reference_offsets - reference_offsets.mean(axis=0)
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered)
    if eigenvalues[1] <= 1e-8 * eigenvalues[2]:
        return None
    return axes[:, 1:]


def surrounds_landmark(projected):
    """Check whether projected scan samples surround the candidate at the origin."""
    projected = projected[np.linalg.norm(projected, axis=1) > 1e-9]
    if len(projected) < 3:
        return False
    angles = np.sort(np.arctan2(projected[:, 1], projected[:, 0]))
    gaps = np.diff(angles, append=angles[0] + 2 * np.pi)
    # A gap of at least 180 degrees puts every neighbor on one side of
    # a line through the candidate. The candidate lies at or beyond the edge.
    return bool(np.max(gaps) < np.pi - 1e-9)


def has_enclosed_hole(projected):
    """Detect a gap at the origin whose unsupported triangles cannot reach the hull."""
    points = np.unique(projected, axis=0)
    if len(points) < 4:
        return False
    try:
        mesh = Delaunay(points)
    except QhullError:
        return False
    containing = int(mesh.find_simplex(np.zeros(2)).item())
    if containing < 0:
        return False

    # The second neighbor is less sensitive to near-duplicates than the closest one.
    distances, _ = cKDTree(points).query(points, k=3)
    spacing = np.median(distances[:, 2])
    triangles = points[mesh.simplices]
    edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    unsupported = edges.max(axis=1) > 2.25 * spacing
    if not unsupported[containing]:
        return False

    visited = {containing}
    pending = [containing]
    while pending:
        for neighbor in mesh.neighbors[pending.pop()]:
            if neighbor < 0:
                return False
            if unsupported[neighbor] and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return True


def estimate_coverage_planes(scan):
    """Keep broad convex-hull faces supported by nearby scan samples."""
    points = np.asarray(scan.points)
    centered = points - points.mean(axis=0)
    spread = np.linalg.eigvalsh(centered.T @ centered)
    if spread[0] <= 1e-10 * spread[-1]:
        return None, 0.0
    try:
        hull, _ = scan.compute_convex_hull()
    except RuntimeError:
        return None, 0.0

    vertices = np.asarray(hull.vertices)
    corners = vertices[np.asarray(hull.triangles)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_areas = np.linalg.norm(normals, axis=1)
    broad = (double_areas > 0) & (double_areas >= 0.01 * double_areas.sum())
    if not np.any(broad):
        return None, 0.0
    normals = normals[broad] / double_areas[broad, None]
    anchors = corners[broad, 0]
    # Orient the normals toward the interior to match the supplied-plane convention.
    interior_direction = np.einsum("ij,ij->i", normals, vertices.mean(axis=0) - anchors)
    normals[interior_direction < 0] *= -1
    offsets = -np.einsum("ij,ij->i", normals, anchors)

    distances = np.asarray(scan.compute_nearest_neighbor_distance())
    spacing = distances[np.isfinite(distances) & (distances > 0)]
    if not len(spacing):
        return None, 0.0
    tolerance = float(np.median(spacing) / 4)
    min_support = max(6, int(np.ceil(0.001 * len(points))))
    supported = [
        [*normal, offset] for normal, offset in zip(normals, offsets)
        if np.count_nonzero(np.abs(points @ normal + offset) <= tolerance) >= min_support
    ]
    return (np.asarray(supported), tolerance) if supported else (None, 0.0)


def recognize(args):
    planes, tolerance = (None, 0.0)
    if args.coverage_planes is not None:
        planes, tolerance = load_coverage_planes(args.coverage_planes)
    o3d.utility.random.seed(0)
    scan = load_cloud(args.scan, args.scan_unit)
    reference = load_cloud(args.reference, args.reference_unit, allow_mesh=True)
    names, landmarks = load_landmarks(args.landmarks)
    landmarks *= UNIT_TO_MM[args.reference_unit]

    voxel = args.voxel_mm
    scan_coarse = downsample(scan, voxel, normals=True)
    ref_coarse = downsample(reference, voxel, normals=True)
    scan_fine = downsample(scan, voxel / 2)
    ref_fine = downsample(reference, voxel / 2, normals=True)
    reference_tree = o3d.geometry.KDTreeFlann(reference)
    for name, point in zip(names, landmarks):
        _, _, distances = reference_tree.search_knn_vector_3d(point, 1)
        if distances[0] > args.landmark_radius_mm ** 2:
            raise ValueError(f"Landmark {name!r} is too far from the reference surface")

    if args.initial_pose:
        poses = [load_initial_pose(args.initial_pose)]
    else:
        poses = list(initial_poses(scan_coarse, ref_coarse, voxel))

    report = {
        "status": "rejected",
        "unit": "mm",
        "landmark_frame": "scan",
        "voxel_mm": voxel,
        "evaluation_distance_mm": args.inlier_distance_mm,
        "landmarks": {},
    }
    candidates = []
    for initial in poses:
        pose = refine(scan_coarse, ref_coarse, scan_fine, ref_fine, initial, voxel)
        if pose is None:
            continue
        inverse = np.linalg.inv(pose)
        forward = registration.evaluate_registration(
            scan_fine, ref_fine, args.inlier_distance_mm, pose
        )
        backward = registration.evaluate_registration(
            ref_fine, scan_fine, args.inlier_distance_mm, inverse
        )
        metrics = {
            "scan_overlap": forward.fitness,
            "reference_overlap": backward.fitness,
            "inlier_rmse_mm": forward.inlier_rmse if forward.fitness else None,
        }
        # Prefer high coverage in both directions so a small matched patch
        # cannot dominate the ranking.
        score = (forward.fitness * backward.fitness) ** 0.5
        passes = (
            forward.fitness >= args.min_scan_overlap
            and backward.fitness >= args.min_reference_overlap
            and metrics["inlier_rmse_mm"] is not None
            and metrics["inlier_rmse_mm"] <= args.max_rmse_mm
        )
        candidates.append((passes, score, inverse, metrics))

    if not candidates:
        report["reason"] = "no_registration"
        return report
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    passed, best_score, reference_to_scan, report["registration"] = candidates[0]
    if not passed:
        report["reason"] = "registration_thresholds"
        return report

    mapped = transform_points(landmarks, reference_to_scan)
    for passed, score, alternative, _ in candidates[1:]:
        separation = np.linalg.norm(mapped - transform_points(landmarks, alternative), axis=1)
        if passed and score >= 0.95 * best_score and np.any(separation > args.landmark_radius_mm):
            report["reason"] = "ambiguous_registration"
            return report

    report["reference_to_scan_mm"] = reference_to_scan.tolist()
    if planes is None:
        planes, tolerance = estimate_coverage_planes(scan)
    boundary_distances = (
        (mapped @ planes[:, :3].T + planes[:, 3]).min(axis=1)
        if planes is not None else [None] * len(mapped)
    )
    if planes is not None:
        report["coverage_boundary_tolerance_mm"] = tolerance
        report["coverage_boundary_source"] = (
            "provided_planes" if args.coverage_planes is not None else "point_cloud_hull"
        )
    scan_tree = o3d.geometry.KDTreeFlann(scan)
    scan_points = np.asarray(scan.points)
    reference_points = np.asarray(reference.points)
    radius = args.landmark_radius_mm
    rotation = reference_to_scan[:3, :3]
    for name, reference_point, point, boundary_distance in zip(
        names, landmarks, mapped, boundary_distances
    ):
        _, _, distances = scan_tree.search_knn_vector_3d(point, 1)
        distance = float(np.sqrt(distances[0]))
        if boundary_distance is not None and boundary_distance < -tolerance:
            status = "outside_scan_coverage"
        elif distance > radius:
            status = "no_nearby_scan_point"
        else:
            _, scan_indices, _ = scan_tree.search_radius_vector_3d(point, radius)
            _, ref_indices, _ = reference_tree.search_radius_vector_3d(reference_point, radius)
            axes = tangent_plane_axes(reference_points[ref_indices] - reference_point)
            status = "outside_scan_coverage"
            if axes is not None:
                projection = rotation @ axes
                local = (scan_points[scan_indices] - point) @ projection
                if surrounds_landmark(local):
                    # A wider neighborhood lets a missing patch reach an open edge.
                    _, nearby, _ = scan_tree.search_radius_vector_3d(point, 2 * radius)
                    projected = (scan_points[nearby] - point) @ projection
                    status = "local_scan_hole" if has_enclosed_hole(projected) else "mapped"
        point_report = {
            "status": status,
            "xyz_mm": point.tolist() if status == "mapped" else None,
            "nearest_scan_distance_mm": distance,
        }
        if boundary_distance is not None:
            point_report["coverage_boundary_distance_mm"] = float(boundary_distance)
        report["landmarks"][name] = point_report
    mapped_count = sum(point["status"] == "mapped" for point in report["landmarks"].values())
    if mapped_count == len(names):
        report["status"] = "accepted"
    elif mapped_count:
        report["status"] = "partial"
    else:
        report["reason"] = "landmarks_outside_scan"
    return report


def write_report(path, report):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-unit", choices=UNIT_TO_MM, required=True)
    parser.add_argument("--reference-unit", choices=UNIT_TO_MM, required=True)
    parser.add_argument("--initial-pose", type=Path, help="Scan-to-reference 4×4 matrix in mm")
    parser.add_argument("--coverage-planes", type=Path, help="Known scan-space coverage planes in mm")
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--inlier-distance-mm", type=float, default=1.5)
    parser.add_argument("--max-rmse-mm", type=float, default=0.9)
    parser.add_argument("--min-scan-overlap", type=float, default=0.65)
    parser.add_argument("--min-reference-overlap", type=float, default=0.35)
    parser.add_argument("--landmark-radius-mm", type=float, default=2.0)
    args = parser.parse_args()

    for value in (args.voxel_mm, args.inlier_distance_mm, args.max_rmse_mm, args.landmark_radius_mm):
        if not np.isfinite(value) or value <= 0:
            parser.error("Distance parameters must be finite and positive")
    if not all(0 < value <= 1 for value in (args.min_scan_overlap, args.min_reference_overlap)):
        parser.error("Overlap thresholds must be in (0, 1]")
    inputs = [args.scan, args.reference, args.landmarks, args.initial_pose, args.coverage_planes]
    if args.output.resolve() in {path.resolve() for path in inputs if path is not None}:
        parser.error("Output must be different from the input files")

    try:
        report = recognize(args)
        write_report(args.output, report)
    except (OSError, ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
