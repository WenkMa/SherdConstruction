#!/usr/bin/env python3
"""
Register the inner side point cloud (bottom.ply) to the outer side point cloud (top.ply).

The preferred path uses TEASER++/Open3D when they are installed.  The fallback is
fully self-contained and uses boundary extraction, PCA global initialization and
trimmed point-to-point ICP, which works well for two sides of the same shard
because their outlines are usually the most reliable common signal.
C:/Users/57746/anaconda3/envs/mapany/python.exe regretion/register_front_back_teaser.py --method fallback --source regretion/bottom.ply --target regretion/top.ply --common-region outline --normal-orientation opposite --voxel 0.0025 --max-points 9000 --trim-fraction 0.80
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"workspace": str(SCRIPT_DIR)}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        try:
            import yaml

            config = yaml.safe_load(f) or {}
        except Exception:
            config = {}
    apply_batch_override(config)
    expand_config_templates(config)
    workspace = Path(config.get("workspace") or SCRIPT_DIR).resolve()
    batch_name = detect_batch_name(config)
    output_root = workspace / "output" / batch_name if batch_name else workspace / "output"
    config["workspace"] = str(workspace)
    config["batch_name"] = batch_name
    config["output_root"] = str(output_root)
    return config


def apply_batch_override(config: dict) -> None:
    batch_number = os.environ.get("PIPELINE_BATCH_NUMBER", "")
    if not batch_number:
        return
    config["batch_number"] = int(batch_number)
    config["batch_num"] = int(batch_number)
    config.pop("OUTPUT_BATCH", None)
    config.pop("batch_name", None)


def expand_config_templates(config: dict) -> None:
    values = {
        "batch_number": config.get("batch_number", config.get("batch_num", "")),
        "batch_num": config.get("batch_num", config.get("batch_number", "")),
    }

    def expand(value):
        if isinstance(value, str):
            for key, item in values.items():
                value = value.replace("${" + key + "}", str(item))
            return value
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = expand(item)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = expand(item)
        return value

    expand(config)


def detect_batch_name(config: dict) -> str:
    explicit = config.get("OUTPUT_BATCH") or config.get("batch_name")
    if explicit:
        return str(explicit)
    candidates = []
    side_inputs = config.get("SIDE_INPUTS") or {}
    if isinstance(side_inputs, dict):
        for item in side_inputs.values():
            if not isinstance(item, dict):
                continue
            for key in ("images", "masks"):
                value = item.get(key)
                if not value:
                    continue
                for part in Path(value).parts:
                    if part.lower().startswith("batch"):
                        candidates.append(part)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else ""


def read_obj(path: Path):
    vertices = []
    normals = []
    lines = []
    vertex_line_ids = []
    normal_line_ids = []
    faces = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            lines.append(line)
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                vertex_line_ids.append(len(lines) - 1)
            elif line.startswith("vn "):
                parts = line.strip().split()
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                normal_line_ids.append(len(lines) - 1)
            elif line.startswith("f "):
                idx = []
                for token in line.strip().split()[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    vi = int(raw)
                    if vi < 0:
                        vi = len(vertices) + vi + 1
                    idx.append(vi - 1)
                if len(idx) >= 3:
                    faces.append(idx)

    if not vertices:
        raise ValueError(f"No vertices found in {path}")

    return {
        "kind": "mesh",
        "path": path,
        "vertices": np.asarray(vertices, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "colors": np.empty((0, 3), dtype=np.float64),
        "lines": lines,
        "vertex_line_ids": vertex_line_ids,
        "normal_line_ids": normal_line_ids,
        "faces": faces,
    }


def read_point_cloud(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No points found in {path}")

    vertices = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else np.empty((0, 3))
    colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else np.empty((0, 3))

    return {
        "kind": "point_cloud",
        "path": path,
        "vertices": vertices,
        "normals": normals,
        "colors": colors,
        "lines": [],
        "vertex_line_ids": [],
        "normal_line_ids": [],
        "faces": [],
    }


def read_geometry(path: Path):
    if path.suffix.lower() == ".obj":
        return read_obj(path)
    return read_point_cloud(path)


def to_open3d_point_cloud(geometry, transform: np.ndarray | None = None, preview_color=None):
    points = geometry["vertices"]
    if transform is not None:
        points = apply_transform(points, transform)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    normals = geometry.get("normals")
    if normals is not None and len(normals) == len(points):
        if transform is not None:
            normals = normals @ transform_rotation(transform).T
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        pcd.normals = o3d.utility.Vector3dVector(normals)

    colors = geometry.get("colors")
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    elif preview_color is not None:
        color = np.asarray(preview_color, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color[None, :], (len(points), 1)))

    return pcd


def write_transformed_obj(mesh, transform: np.ndarray, out_path: Path):
    vertices = mesh["vertices"]
    transformed = apply_transform(vertices, transform)
    lines = list(mesh["lines"])

    for row, line_id in zip(transformed, mesh["vertex_line_ids"]):
        parts = lines[line_id].strip().split()
        extras = parts[4:]
        coords = [f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"]
        lines[line_id] = "v " + " ".join(coords + extras) + "\n"

    normals = mesh.get("normals")
    normal_line_ids = mesh.get("normal_line_ids", [])
    if normals is not None and len(normals) == len(normal_line_ids):
        rotated = normals @ transform_rotation(transform).T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        rotated = rotated / np.maximum(norms, 1e-12)
        for row, line_id in zip(rotated, normal_line_ids):
            lines[line_id] = f"vn {row[0]:.9f} {row[1]:.9f} {row[2]:.9f}\n"

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)


def write_transformed_point_cloud(point_cloud, transform: np.ndarray, out_path: Path):
    pcd = to_open3d_point_cloud(point_cloud, transform=transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False):
        raise RuntimeError(f"Failed to write registered point cloud: {out_path}")


def write_transformed_geometry(geometry, transform: np.ndarray, out_path: Path):
    if geometry["kind"] == "mesh" and out_path.suffix.lower() == ".obj":
        write_transformed_obj(geometry, transform, out_path)
    else:
        write_transformed_point_cloud(geometry, transform, out_path)


def write_merged_obj(target_mesh, source_mesh, transform: np.ndarray, out_path: Path):
    target_v = target_mesh["vertices"]
    source_v = apply_transform(source_mesh["vertices"], transform)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Registered pair: target top11 + transformed bottommo\n")
        handle.write("o target_top11\n")
        for v in target_v:
            handle.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f} 0.800000 0.360000 0.220000\n")
        for face in target_mesh["faces"]:
            handle.write("f " + " ".join(str(i + 1) for i in face) + "\n")

        offset = len(target_v)
        handle.write("o source_bottommo_registered\n")
        for v in source_v:
            handle.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f} 0.180000 0.520000 0.900000\n")
        for face in source_mesh["faces"]:
            handle.write("f " + " ".join(str(i + 1 + offset) for i in face) + "\n")


def write_merged_point_cloud(target_geometry, source_geometry, transform: np.ndarray, out_path: Path):
    target_pcd = to_open3d_point_cloud(target_geometry, preview_color=(0.80, 0.36, 0.22))
    source_pcd = to_open3d_point_cloud(
        source_geometry,
        transform=transform,
        preview_color=(0.18, 0.52, 0.90),
    )
    merged = target_pcd + source_pcd
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(out_path), merged, write_ascii=False):
        raise RuntimeError(f"Failed to write merged point cloud: {out_path}")


def write_merged_geometry(target_geometry, source_geometry, transform: np.ndarray, out_path: Path):
    if (
        target_geometry["kind"] == "mesh"
        and source_geometry["kind"] == "mesh"
        and out_path.suffix.lower() == ".obj"
    ):
        write_merged_obj(target_geometry, source_geometry, transform, out_path)
    else:
        write_merged_point_cloud(target_geometry, source_geometry, transform, out_path)


def parse_face_edges(faces):
    edges = defaultdict(int)
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges[(a, b)] += 1
    return edges


def boundary_points(mesh):
    faces = mesh["faces"]
    vertices = mesh["vertices"]
    if not faces:
        return vertices

    edges = parse_face_edges(faces)
    ids = sorted({i for edge, count in edges.items() if count == 1 for i in edge})
    if len(ids) < 32:
        return vertices
    return vertices[np.asarray(ids, dtype=np.int64)]


def side_points_from_normals(mesh, side_fraction: float):
    vertices = mesh["vertices"]
    normals = mesh.get("normals")
    if normals is None or len(normals) != len(vertices):
        return vertices

    _, frame = pca_frame(vertices)
    thin_normal = frame[:, -1]
    unit_normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normal_dot = np.abs(unit_normals @ thin_normal)

    fraction = min(max(side_fraction, 0.01), 0.50)
    threshold = float(np.quantile(normal_dot, fraction))
    ids = np.flatnonzero(normal_dot <= threshold)
    if len(ids) < 64:
        ids = np.argsort(normal_dot)[:64]
    return vertices[ids]


def projected_outline_points(mesh, outline_bins: int, outline_points_per_bin: int):
    vertices = mesh["vertices"]
    if len(vertices) <= outline_bins * outline_points_per_bin:
        return vertices

    center, frame = pca_frame(vertices)
    uv = (vertices - center) @ frame[:, :2]
    uv_center = np.median(uv, axis=0)
    uv = uv - uv_center
    radius = np.linalg.norm(uv, axis=1)
    angle = np.arctan2(uv[:, 1], uv[:, 0])

    bins = max(32, int(outline_bins))
    per_bin = max(1, int(outline_points_per_bin))
    bin_ids = np.floor((angle + np.pi) / (2.0 * np.pi) * bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    selected = []
    for bin_id in range(bins):
        ids = np.flatnonzero(bin_ids == bin_id)
        if len(ids) == 0:
            continue
        keep = min(per_bin, len(ids))
        local = np.argpartition(radius[ids], len(ids) - keep)[-keep:]
        selected.append(ids[local])

    if not selected:
        return vertices
    selected_ids = np.unique(np.concatenate(selected))
    if len(selected_ids) < 64:
        return side_points_from_normals(mesh, 0.15)
    return vertices[selected_ids]


def projected_fracture_edge_points(
    mesh,
    outline_bins: int,
    edge_points_per_bin: int,
    edge_band: float,
    edge_quantile: float,
):
    vertices = mesh["vertices"]
    if len(vertices) <= outline_bins * edge_points_per_bin:
        return vertices

    center, frame = pca_frame(vertices)
    uv = (vertices - center) @ frame[:, :2]
    uv_center = np.median(uv, axis=0)
    uv = uv - uv_center
    radius = np.linalg.norm(uv, axis=1)
    angle = np.arctan2(uv[:, 1], uv[:, 0])

    bins = max(32, int(outline_bins))
    per_bin = max(1, int(edge_points_per_bin))
    quantile = min(max(float(edge_quantile), 0.50), 0.999)
    bin_ids = np.floor((angle + np.pi) / (2.0 * np.pi) * bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    selected = []
    for bin_id in range(bins):
        ids = np.flatnonzero(bin_ids == bin_id)
        if len(ids) == 0:
            continue
        local_radius = radius[ids]
        if edge_band > 0:
            threshold = float(local_radius.max() - edge_band)
        else:
            threshold = float(np.quantile(local_radius, quantile))
        edge_ids = ids[local_radius >= threshold]
        if len(edge_ids) == 0:
            continue
        keep = min(per_bin, len(edge_ids))
        local = np.argpartition(radius[edge_ids], len(edge_ids) - keep)[-keep:]
        selected.append(edge_ids[local])

    if not selected:
        return projected_outline_points(mesh, outline_bins, edge_points_per_bin)

    selected_ids = np.unique(np.concatenate(selected))
    if len(selected_ids) < 64:
        return projected_outline_points(mesh, outline_bins, edge_points_per_bin)
    return vertices[selected_ids]


def common_region_points(mesh, side_fraction: float, common_region: str, outline_bins: int, outline_points_per_bin: int):
    if common_region == "all":
        return mesh["vertices"], "all_points"

    if mesh["faces"] and common_region in {"auto", "mesh-boundary"}:
        return boundary_points(mesh), "mesh_boundary"

    if common_region == "normal-side":
        return side_points_from_normals(mesh, side_fraction), "normal_side_region"

    return (
        projected_outline_points(mesh, outline_bins, outline_points_per_bin),
        "projected_outline",
    )


def voxel_downsample(points: np.ndarray, voxel: float | None, max_points: int):
    if len(points) <= max_points and not voxel:
        return points.copy()

    pts = points
    if voxel and voxel > 0:
        keys = np.floor((pts - pts.min(axis=0)) / voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        pts = pts[np.sort(first)]

    if len(pts) > max_points:
        rng = np.random.default_rng(7)
        ids = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[np.sort(ids)]
    return pts


def pca_frame(points: np.ndarray):
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    frame = vecs[:, ::-1]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1.0
    return center, frame


def nearest_neighbors(src: np.ndarray, dst: np.ndarray, chunk_size: int = 1024):
    if cKDTree is not None:
        distances, ids = cKDTree(dst).query(src, k=1, workers=-1)
        return ids.astype(np.int64), distances.astype(np.float64)

    all_ids = np.empty(len(src), dtype=np.int64)
    all_dist2 = np.empty(len(src), dtype=np.float64)

    dst2 = np.sum(dst * dst, axis=1)
    for start in range(0, len(src), chunk_size):
        chunk = src[start : start + chunk_size]
        dist2 = np.sum(chunk * chunk, axis=1)[:, None] + dst2[None, :] - 2.0 * chunk @ dst.T
        ids = np.argmin(dist2, axis=1)
        all_ids[start : start + len(chunk)] = ids
        all_dist2[start : start + len(chunk)] = dist2[np.arange(len(chunk)), ids]
    np.maximum(all_dist2, 0.0, out=all_dist2)
    return all_ids, np.sqrt(all_dist2)


def rigid_from_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    allow_reflection: bool = False,
    estimate_scale: bool = False,
):
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    h = xs.T @ xd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if not allow_reflection and np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    scale = 1.0
    if estimate_scale:
        rotated = xs @ r.T
        denom = float(np.sum(rotated * rotated))
        if denom > 1e-12:
            scale = float(np.sum(rotated * xd) / denom)
        if scale <= 0 and not allow_reflection:
            scale = 1.0

    t = cd - scale * (r @ cs)
    transform = np.eye(4)
    transform[:3, :3] = scale * r
    transform[:3, 3] = t
    return transform


def compose(a: np.ndarray, b: np.ndarray):
    return a @ b


def transform_scale(transform: np.ndarray):
    return float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))


def transform_rotation(transform: np.ndarray):
    scale = transform_scale(transform)
    if scale <= 1e-12:
        return transform[:3, :3]
    return transform[:3, :3] / scale


def apply_transform(points: np.ndarray, transform: np.ndarray):
    return points @ transform[:3, :3].T + transform[:3, 3]


def representative_normal(geometry):
    normals = geometry.get("normals")
    if normals is not None and len(normals) == len(geometry["vertices"]):
        valid = normals[np.linalg.norm(normals, axis=1) > 1e-9]
        if len(valid) > 0:
            # Normal signs from dense reconstruction can contain outliers, so use
            # the dominant PCA direction of the normal cloud instead of a raw mean.
            _, _, vt = np.linalg.svd(valid, full_matrices=False)
            normal = vt[0]
            mean = valid.mean(axis=0)
            if np.dot(normal, mean) < 0:
                normal *= -1.0
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                return normal / norm

    _, frame = pca_frame(geometry["vertices"])
    return frame[:, -1]


def normal_orientation_dot(source_geometry, target_geometry, transform: np.ndarray):
    src_normal = representative_normal(source_geometry)
    dst_normal = representative_normal(target_geometry)
    moved_src_normal = transform_rotation(transform) @ src_normal
    return float(np.dot(moved_src_normal, dst_normal))


def robust_scale_from_points(src: np.ndarray, dst: np.ndarray):
    src_centered = src - np.median(src, axis=0)
    dst_centered = dst - np.median(dst, axis=0)
    src_radius = np.linalg.norm(src_centered, axis=1)
    dst_radius = np.linalg.norm(dst_centered, axis=1)

    src_scale = float(np.percentile(src_radius, 90))
    dst_scale = float(np.percentile(dst_radius, 90))
    if src_scale <= 1e-12 or dst_scale <= 1e-12:
        return 1.0
    return dst_scale / src_scale


def pca_initial_transforms(
    src: np.ndarray,
    dst: np.ndarray,
    allow_reflection: bool,
    initial_scale: float = 1.0,
):
    cs, fs = pca_frame(src)
    cd, fd = pca_frame(dst)

    transforms = []
    signs = (-1.0, 1.0)
    for sx in signs:
        for sy in signs:
            for sz in signs:
                sign = np.diag([sx, sy, sz])
                r = fd @ sign @ fs.T
                det = np.linalg.det(r)
                if allow_reflection:
                    if abs(abs(det) - 1.0) > 1e-6:
                        continue
                elif det < 0:
                    continue
                t = cd - initial_scale * (r @ cs)
                tf = np.eye(4)
                tf[:3, :3] = initial_scale * r
                tf[:3, 3] = t
                transforms.append(tf)
    return transforms


def trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    init: np.ndarray,
    iterations: int,
    trim_fraction: float,
    allow_reflection: bool,
    estimate_scale: bool,
    scale_min: float,
    scale_max: float,
):
    transform = init.copy()
    best_transform = transform.copy()
    best_error = math.inf

    keep_count = max(12, int(len(source) * trim_fraction))
    keep_count = min(keep_count, len(source))

    for _ in range(iterations):
        moved = apply_transform(source, transform)
        nn_ids, distances = nearest_neighbors(moved, target)
        keep_ids = np.argpartition(distances, keep_count - 1)[:keep_count]
        error = float(np.sqrt(np.mean(distances[keep_ids] ** 2)))
        if error < best_error:
            best_error = error
            best_transform = transform.copy()

        delta = rigid_from_correspondences(
            moved[keep_ids],
            target[nn_ids[keep_ids]],
            allow_reflection,
            estimate_scale=estimate_scale,
        )
        transform = compose(delta, transform)
        scale = transform_scale(transform)
        if scale < scale_min or scale > scale_max:
            clamped = min(max(scale, scale_min), scale_max)
            if scale > 1e-12:
                transform[:3, :3] *= clamped / scale

        step_r = np.linalg.norm(delta[:3, :3] - np.eye(3))
        step_t = np.linalg.norm(delta[:3, 3])
        if step_r < 1e-8 and step_t < 1e-8:
            break

    return best_transform, best_error


def refine_on_fracture_edges(
    source_mesh,
    target_mesh,
    initial_transform: np.ndarray,
    voxel: float,
    max_points: int,
    iterations: int,
    trim_fraction: float,
    allow_reflection: bool,
    estimate_scale: bool,
    scale_min: float,
    scale_max: float,
    outline_bins: int,
    edge_points_per_bin: int,
    edge_band: float,
    edge_quantile: float,
):
    src_edge = projected_fracture_edge_points(
        source_mesh,
        outline_bins=outline_bins,
        edge_points_per_bin=edge_points_per_bin,
        edge_band=edge_band,
        edge_quantile=edge_quantile,
    )
    dst_edge = projected_fracture_edge_points(
        target_mesh,
        outline_bins=outline_bins,
        edge_points_per_bin=edge_points_per_bin,
        edge_band=edge_band,
        edge_quantile=edge_quantile,
    )

    src = voxel_downsample(src_edge, voxel, max_points)
    dst = voxel_downsample(dst_edge, voxel, max_points)
    if len(src) < 12 or len(dst) < 12:
        return initial_transform, math.inf, {
            "enabled": True,
            "skipped": True,
            "reason": "not_enough_edge_points",
            "source_edge_points": len(src_edge),
            "target_edge_points": len(dst_edge),
            "source_used_points": len(src),
            "target_used_points": len(dst),
        }

    transform, error = trimmed_icp(
        src,
        dst,
        initial_transform,
        iterations=iterations,
        trim_fraction=trim_fraction,
        allow_reflection=allow_reflection,
        estimate_scale=estimate_scale,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return transform, error, {
        "enabled": True,
        "skipped": False,
        "source_edge_points": len(src_edge),
        "target_edge_points": len(dst_edge),
        "source_used_points": len(src),
        "target_used_points": len(dst),
        "voxel": voxel,
        "iterations": iterations,
        "trim_fraction": trim_fraction,
        "edge_points_per_bin": edge_points_per_bin,
        "edge_band": edge_band,
        "edge_quantile": edge_quantile,
        "rmse": error,
    }


def robust_boundary_registration(
    source_mesh,
    target_mesh,
    max_points: int,
    voxel: float | None,
    icp_iterations: int,
    trim_fraction: float,
    allow_reflection: bool,
    side_fraction: float,
    common_region: str,
    outline_bins: int,
    outline_points_per_bin: int,
    normal_orientation: str,
    normal_weight: float,
    estimate_scale: bool,
    refine_scale: bool,
    initial_scale_hint: float | None,
    scale_min: float,
    scale_max: float,
    fine_edge_refine: bool,
    fine_icp_iterations: int,
    fine_trim_fraction: float,
    fine_max_points: int,
    fine_voxel_scale: float,
    edge_points_per_bin: int,
    edge_band: float,
    edge_quantile: float,
):
    src_boundary, src_region = common_region_points(
        source_mesh,
        side_fraction,
        common_region,
        outline_bins,
        outline_points_per_bin,
    )
    dst_boundary, dst_region = common_region_points(
        target_mesh,
        side_fraction,
        common_region,
        outline_bins,
        outline_points_per_bin,
    )

    diag = np.linalg.norm(np.ptp(np.vstack([src_boundary, dst_boundary]), axis=0))
    if voxel is None:
        voxel = diag / 180.0

    src = voxel_downsample(src_boundary, voxel, max_points)
    dst = voxel_downsample(dst_boundary, voxel, max_points)

    if initial_scale_hint is not None and initial_scale_hint > 1e-12:
        initial_scale = float(initial_scale_hint)
        scale_source = "metadata_hint"
    elif estimate_scale:
        initial_scale = robust_scale_from_points(src, dst)
        scale_source = "robust_region_estimate"
    else:
        initial_scale = 1.0
        scale_source = "fixed_identity"
    initial_scale = min(max(initial_scale, scale_min), scale_max)
    candidates = pca_initial_transforms(
        src,
        dst,
        allow_reflection=allow_reflection,
        initial_scale=initial_scale,
    )
    if not candidates:
        raise RuntimeError("No valid PCA initial transforms were generated.")

    best_transform = None
    best_error = math.inf
    best_score = math.inf
    best_normal_dot = math.nan
    for init in candidates:
        transform, error = trimmed_icp(
            src,
            dst,
            init,
            iterations=icp_iterations,
            trim_fraction=trim_fraction,
            allow_reflection=allow_reflection,
            estimate_scale=refine_scale,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        normal_dot = normal_orientation_dot(source_mesh, target_mesh, transform)
        score = error
        if normal_orientation == "opposite" and normal_dot > 0:
            score += abs(normal_dot) * diag * normal_weight
        elif normal_orientation == "same" and normal_dot < 0:
            score += abs(normal_dot) * diag * normal_weight

        if score < best_score:
            best_score = score
            best_error = error
            best_transform = transform
            best_normal_dot = normal_dot

    fine_info = {"enabled": False}
    if fine_edge_refine:
        fine_voxel = voxel * max(float(fine_voxel_scale), 1e-6)
        best_transform, fine_error, fine_info = refine_on_fracture_edges(
            source_mesh,
            target_mesh,
            initial_transform=best_transform,
            voxel=fine_voxel,
            max_points=fine_max_points,
            iterations=fine_icp_iterations,
            trim_fraction=fine_trim_fraction,
            allow_reflection=allow_reflection,
            estimate_scale=refine_scale,
            scale_min=scale_min,
            scale_max=scale_max,
            outline_bins=outline_bins,
            edge_points_per_bin=edge_points_per_bin,
            edge_band=edge_band,
            edge_quantile=edge_quantile,
        )
        if not fine_info.get("skipped"):
            best_error = fine_error
            best_normal_dot = normal_orientation_dot(source_mesh, target_mesh, best_transform)

    return best_transform, best_error, {
        "source_boundary_points": len(src_boundary),
        "target_boundary_points": len(dst_boundary),
        "source_used_points": len(src),
        "target_used_points": len(dst),
        "voxel": voxel,
        "source_region": src_region,
        "target_region": dst_region,
        "side_fraction": side_fraction,
        "common_region": common_region,
        "outline_bins": outline_bins,
        "outline_points_per_bin": outline_points_per_bin,
        "normal_orientation": normal_orientation,
        "normal_orientation_dot": best_normal_dot,
        "normal_weight": normal_weight,
        "estimate_scale": bool(estimate_scale),
        "refine_scale": bool(refine_scale),
        "initial_scale": initial_scale,
        "initial_scale_hint": initial_scale_hint,
        "initial_scale_source": scale_source,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale": transform_scale(best_transform),
        "fine_edge_refine": fine_info,
        "method": "numpy_common_region_pca_trimmed_icp",
    }


def load_open3d_as_point_cloud(path: Path):
    if path.suffix.lower() == ".obj":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.is_empty() and len(mesh.triangles) > 0:
            return mesh.sample_points_uniformly(number_of_points=80000)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise RuntimeError(f"Could not read geometry: {path}")
        if len(mesh.triangles) > 0:
            return mesh.sample_points_uniformly(number_of_points=80000)
        pcd.points = mesh.vertices
    return pcd


def try_teaser_open3d(source_path: Path, target_path: Path, voxel: float | None):
    try:
        import teaserpp_python
    except Exception:
        return None

    src_pcd = load_open3d_as_point_cloud(source_path)
    dst_pcd = load_open3d_as_point_cloud(target_path)

    if len(src_pcd.points) > 80000:
        src_pcd = src_pcd.farthest_point_down_sample(80000)
    if len(dst_pcd.points) > 80000:
        dst_pcd = dst_pcd.farthest_point_down_sample(80000)

    src_extent = np.linalg.norm(np.ptp(np.asarray(src_pcd.points), axis=0))
    dst_extent = np.linalg.norm(np.ptp(np.asarray(dst_pcd.points), axis=0))
    if voxel is None:
        voxel = max(src_extent, dst_extent) / 120.0

    def preprocess(pcd):
        down = pcd.voxel_down_sample(voxel)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=40)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=120),
        )
        return down, np.asarray(fpfh.data).T

    src_down, src_feat = preprocess(src_pcd)
    dst_down, dst_feat = preprocess(dst_pcd)
    src_points = np.asarray(src_down.points)
    dst_points = np.asarray(dst_down.points)

    if cKDTree is not None:
        _, feat_ids = cKDTree(dst_feat).query(src_feat, k=1, workers=-1)
    else:
        feat_ids = []
        dst_feat2 = np.sum(dst_feat * dst_feat, axis=1)
        for feature in src_feat:
            dist2 = np.sum(feature * feature) + dst_feat2 - 2.0 * dst_feat @ feature
            feat_ids.append(int(np.argmin(dist2)))
        feat_ids = np.asarray(feat_ids, dtype=np.int64)

    src_corr = []
    dst_corr = []
    for i, dst_id in enumerate(feat_ids):
        src_corr.append(src_points[i])
        dst_corr.append(dst_points[int(dst_id)])

    if len(src_corr) < 12:
        return None

    src_corr = np.asarray(src_corr).T
    dst_corr = np.asarray(dst_corr).T

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.cbar2 = 1.0
    params.noise_bound = voxel * 1.5
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(src_corr, dst_corr)
    solution = solver.getSolution()

    init = np.eye(4)
    init[:3, :3] = solution.rotation
    init[:3, 3] = solution.translation

    result = o3d.pipelines.registration.registration_icp(
        src_down,
        dst_down,
        voxel * 2.0,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )

    return np.asarray(result.transformation), float(result.inlier_rmse), {
        "source_used_points": len(src_points),
        "target_used_points": len(dst_points),
        "voxel": voxel,
        "method": "teaserpp_open3d_fpfh_icp",
    }


def save_transform(
    path: Path,
    transform: np.ndarray,
    error: float,
    info: dict,
    source_name: str,
    target_name: str,
):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# Transform maps source/{source_name} into target/{target_name} coordinates.\n")
        handle.write(f"# method: {info.get('method')}\n")
        handle.write(f"# trimmed/inlier RMSE: {error:.9f}\n")
        for key, value in info.items():
            if key != "method":
                handle.write(f"# {key}: {value}\n")
        handle.write(f"# transform_scale: {transform_scale(transform):.12g}\n")
        for row in transform:
            handle.write(" ".join(f"{v:.12g}" for v in row) + "\n")


def save_registration_preview(
    path: Path,
    source_geometry,
    target_geometry,
    transform: np.ndarray,
    title: str,
    max_points: int = 20000,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] registration preview skipped: {exc}", flush=True)
        return

    target = target_geometry["vertices"]
    source = apply_transform(source_geometry["vertices"], transform)
    points = np.vstack([target, source])
    center, frame = pca_frame(points)
    target_uv = (target - center) @ frame[:, :2]
    source_uv = (source - center) @ frame[:, :2]

    rng = np.random.default_rng(7)
    if len(target_uv) > max_points:
        ids = rng.choice(len(target_uv), size=max_points, replace=False)
        target_uv = target_uv[np.sort(ids)]
    if len(source_uv) > max_points:
        ids = rng.choice(len(source_uv), size=max_points, replace=False)
        source_uv = source_uv[np.sort(ids)]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.scatter(target_uv[:, 0], target_uv[:, 1], s=0.2, color="#1f77b4", alpha=0.55, label="top target")
    ax.scatter(source_uv[:, 0], source_uv[:, 1], s=0.2, color="#d62728", alpha=0.55, label="bottom registered")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(markerscale=12, fontsize=8, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Register bottom.ply (moving/source) to top.ply (fixed/target)."
    )
    parser.add_argument("--workspace", default=".", help="Workspace root. Relative paths are resolved from config.yaml workspace.")
    parser.add_argument(
        "--pairs-dir",
        default="",
        help="Directory containing pair_XXX/top.ply and pair_XXX/bottom.ply from 3_match_fragments.py.",
    )
    parser.add_argument(
        "--registration-dir",
        default="",
        help="Directory for registration outputs. Each pair is written under registration-dir/pair_XXX.",
    )
    parser.add_argument(
        "--pair",
        default="",
        help="Only register one pair folder name, for example pair_001. Empty means all pairs.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Use explicit --source and --target instead of scanning --pairs-dir.",
    )
    parser.add_argument("--source", default="bottom.ply", help="Moving/source PLY or OBJ for --single mode.")
    parser.add_argument("--target", default="top.ply", help="Fixed/target PLY or OBJ for --single mode.")
    parser.add_argument("--out", default="bottom_registered_to_top.ply")
    parser.add_argument("--merged", default="registered_pair.ply")
    parser.add_argument("--transform", default="registration_transform.txt")
    parser.add_argument(
        "--method",
        choices=["auto", "teaser", "fallback"],
        default="auto",
        help="auto tries TEASER++ first, then the self-contained fallback.",
    )
    parser.add_argument("--voxel", type=float, default=None, help="Downsample voxel size.")
    parser.add_argument("--max-points", type=int, default=4500)
    parser.add_argument("--icp-iterations", type=int, default=80)
    parser.add_argument("--trim-fraction", type=float, default=0.70)
    parser.add_argument(
        "--no-fine-edge-refine",
        action="store_true",
        help="Disable second-stage ICP on projected fracture edge points.",
    )
    parser.add_argument("--fine-icp-iterations", type=int, default=60)
    parser.add_argument("--fine-trim-fraction", type=float, default=0.80)
    parser.add_argument("--fine-max-points", type=int, default=9000)
    parser.add_argument(
        "--fine-voxel-scale",
        type=float,
        default=0.5,
        help="Fine edge ICP voxel size = coarse voxel * this value.",
    )
    parser.add_argument(
        "--edge-points-per-bin",
        type=int,
        default=16,
        help="Farthest edge-band points kept per angular bin for fine edge ICP.",
    )
    parser.add_argument(
        "--edge-band",
        type=float,
        default=0.0,
        help="Absolute radial band width for fracture edge points. 0 uses --edge-quantile.",
    )
    parser.add_argument(
        "--edge-quantile",
        type=float,
        default=0.92,
        help="Keep points above this per-bin radial quantile as fracture edge candidates.",
    )
    parser.add_argument(
        "--common-region",
        choices=["auto", "outline", "normal-side", "mesh-boundary", "all"],
        default="outline",
        help="Region used for registration. outline is recommended for top/bottom with no shared face.",
    )
    parser.add_argument(
        "--outline-bins",
        type=int,
        default=360,
        help="Angular bins for projected outline extraction from point clouds.",
    )
    parser.add_argument(
        "--outline-points-per-bin",
        type=int,
        default=8,
        help="Number of farthest projected points kept per outline bin.",
    )
    parser.add_argument(
        "--normal-orientation",
        choices=["opposite", "same", "ignore"],
        default="opposite",
        help="Expected top/bottom surface normal relation after registration.",
    )
    parser.add_argument(
        "--normal-weight",
        type=float,
        default=0.15,
        help="Penalty weight for transforms with the wrong normal orientation.",
    )
    parser.add_argument(
        "--side-fraction",
        type=float,
        default=0.15,
        help="For point-cloud OBJ files, use this lowest-normal-dot fraction as shared side/edge points.",
    )
    parser.add_argument(
        "--allow-reflection",
        action="store_true",
        help="Allow mirror transforms for diagnostic use. Leave off for physical rigid registration.",
    )
    parser.add_argument(
        "--estimate-scale",
        action="store_true",
        help="Estimate a single global scale together with rotation/translation.",
    )
    parser.add_argument(
        "--refine-scale",
        action="store_true",
        help="Continue refining scale during ICP. By default --estimate-scale only sets the initial global scale.",
    )
    parser.add_argument(
        "--use-match-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-pair metadata.json scale from matching as the initial registration scale.",
    )
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=0.0,
        help="Manual initial scale override. 0 means use metadata or automatic/default scale.",
    )
    parser.add_argument("--scale-min", type=float, default=0.10, help="Minimum allowed global scale.")
    parser.add_argument("--scale-max", type=float, default=1.50, help="Maximum allowed global scale.")
    return parser


def resolve_input_path(value: str):
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path.resolve()

    script_dir = Path(__file__).resolve().parent
    script_relative = script_dir / path
    if script_relative.exists():
        return script_relative

    workspace = script_dir.parent
    data_candidates = {
        "bottom.ply": workspace / "data" / "bottom" / "bottom.ply",
        "top.ply": workspace / "data" / "top" / "top.ply",
    }
    candidate = data_candidates.get(path.name.lower())
    if candidate is not None and candidate.exists():
        return candidate

    return path.resolve()


def resolve_output_path(value: str, default_dir: Path):
    path = Path(value)
    if path.is_absolute() or path.parent != Path("."):
        return path.resolve()
    return (default_dir / path).resolve()


def load_match_scale(pair_dir: Path, args: argparse.Namespace) -> float | None:
    if args.initial_scale > 0:
        return float(args.initial_scale)
    if not args.use_match_scale:
        return None

    metadata_path = pair_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    scale = metadata.get("scale")
    if scale is None:
        return None
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return scale


def resolve_workspace(value: str) -> Path:
    config = load_config()
    workspace = Path(value)
    if not workspace.is_absolute():
        workspace = Path(config["workspace"]) / workspace
    return workspace.resolve()


def resolve_workspace_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (workspace / path).resolve()


def iter_pair_dirs(pairs_root: Path, pair_name: str = ""):
    if pair_name:
        pair_dir = pairs_root / pair_name
        if not pair_dir.is_dir():
            raise FileNotFoundError(f"Pair folder not found: {pair_dir}")
        yield pair_dir
        return

    found = False
    for pair_dir in sorted(pairs_root.glob("pair_*")):
        if pair_dir.is_dir():
            found = True
            yield pair_dir
    if not found:
        raise FileNotFoundError(f"No pair_* folders found: {pairs_root}")


def register_pair(
    source_path: Path,
    target_path: Path,
    args: argparse.Namespace,
    output_dir: Path | None = None,
    initial_scale_hint: float | None = None,
):
    source_geometry = read_geometry(source_path)
    target_geometry = read_geometry(target_path)

    if output_dir is None:
        output_dir = source_path.parent
    out_path = resolve_output_path(args.out, output_dir)
    merged_path = resolve_output_path(args.merged, output_dir)
    transform_path = resolve_output_path(args.transform, output_dir)

    result = None
    if args.method in {"auto", "teaser"}:
        result = try_teaser_open3d(source_path, target_path, args.voxel)
        if result is None and args.method == "teaser":
            raise RuntimeError("TEASER++/Open3D path is unavailable or failed.")

    if result is None:
        result = robust_boundary_registration(
            source_geometry,
            target_geometry,
            max_points=args.max_points,
            voxel=args.voxel,
            icp_iterations=args.icp_iterations,
            trim_fraction=args.trim_fraction,
            allow_reflection=args.allow_reflection,
            estimate_scale=args.estimate_scale,
            side_fraction=args.side_fraction,
            common_region=args.common_region,
            outline_bins=args.outline_bins,
            outline_points_per_bin=args.outline_points_per_bin,
            normal_orientation=args.normal_orientation,
            normal_weight=args.normal_weight,
            refine_scale=args.refine_scale,
            initial_scale_hint=initial_scale_hint,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            fine_edge_refine=not args.no_fine_edge_refine,
            fine_icp_iterations=args.fine_icp_iterations,
            fine_trim_fraction=args.fine_trim_fraction,
            fine_max_points=args.fine_max_points,
            fine_voxel_scale=args.fine_voxel_scale,
            edge_points_per_bin=args.edge_points_per_bin,
            edge_band=args.edge_band,
            edge_quantile=args.edge_quantile,
        )

    transform, error, info = result
    write_transformed_geometry(source_geometry, transform, out_path)
    write_merged_geometry(target_geometry, source_geometry, transform, merged_path)
    preview_path = output_dir / "registration_preview.png"
    save_registration_preview(
        preview_path,
        source_geometry,
        target_geometry,
        transform,
        title=f"{source_path.stem} registered to {target_path.stem}",
    )
    save_transform(
        transform_path,
        transform,
        error,
        info,
        source_path.name,
        target_path.name,
    )

    print(f"method: {info.get('method')}")
    print(f"RMSE: {error:.9f}")
    print(f"scale: {transform_scale(transform):.9f}")
    print(f"source: {source_path}")
    print(f"target: {target_path}")
    print(f"source -> target transform saved to: {transform_path}")
    print(f"registered source point cloud saved to: {out_path}")
    print(f"merged preview point cloud saved to: {merged_path}")
    print(f"registration preview saved to: {preview_path}")

    return {
        "source": str(source_path),
        "target": str(target_path),
        "out": str(out_path),
        "merged": str(merged_path),
        "transform": str(transform_path),
        "preview": str(preview_path),
        "method": info.get("method"),
        "rmse": error,
        "scale": transform_scale(transform),
        "initial_scale_hint": initial_scale_hint,
        "info": info,
    }


def register_pair_folder(pair_dir: Path, output_pair_dir: Path, args: argparse.Namespace):
    source_path = pair_dir / "bottom.ply"
    target_path = pair_dir / "top.ply"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source bottom.ply: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target top.ply: {target_path}")

    print(f"[pair] {pair_dir.name}")
    output_pair_dir.mkdir(parents=True, exist_ok=True)
    initial_scale_hint = load_match_scale(pair_dir, args)
    result = register_pair(
        source_path,
        target_path,
        args,
        output_dir=output_pair_dir,
        initial_scale_hint=initial_scale_hint,
    )
    result["pair"] = pair_dir.name
    result["input_pair_dir"] = str(pair_dir)
    result["output_pair_dir"] = str(output_pair_dir)

    summary_path = output_pair_dir / "registration_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"registration summary saved to: {summary_path}")
    return result


def main():
    args = build_arg_parser().parse_args()
    config = load_config()

    if args.single:
        source_path = resolve_input_path(args.source)
        target_path = resolve_input_path(args.target)
        initial_scale_hint = float(args.initial_scale) if args.initial_scale > 0 else None
        register_pair(source_path, target_path, args, initial_scale_hint=initial_scale_hint)
        return

    workspace = resolve_workspace(args.workspace)
    output_root = Path(config["output_root"])
    default_pairs_dir = output_root / "matching" / "pairs"
    default_registration_dir = output_root / "registration" / "pairs"
    args.pairs_dir = args.pairs_dir or str(default_pairs_dir.relative_to(workspace))
    args.registration_dir = args.registration_dir or str(default_registration_dir.relative_to(workspace))
    pairs_root = resolve_workspace_path(args.pairs_dir, workspace)
    registration_root = resolve_workspace_path(args.registration_dir, workspace)
    registration_root.mkdir(parents=True, exist_ok=True)
    results = []
    for pair_dir in iter_pair_dirs(pairs_root, args.pair):
        output_pair_dir = registration_root / pair_dir.name
        results.append(register_pair_folder(pair_dir, output_pair_dir, args))

    batch_summary = {
        "workspace": str(workspace),
        "batch_name": config.get("batch_name"),
        "output_root": str(output_root),
        "input_pairs_dir": str(pairs_root),
        "registration_dir": str(registration_root),
        "count": len(results),
        "results": results,
    }
    batch_summary_path = registration_root / "registration_batch_summary.json"
    with batch_summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] registered {len(results)} pair(s)")
    print(f"[OK] batch summary saved to: {batch_summary_path}")


if __name__ == "__main__":
    main()
