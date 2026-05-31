#!/usr/bin/env python3
"""
Convert registered front/back point-cloud pairs into meshes with Poisson reconstruction.

Default input:
  output/<batch>/registration/pairs/pair_XXX/registered_pair.ply

Default output:
  output/<batch>/mesh/registered_pairs/pair_XXX/registered_pair_mesh.ply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

try:
    from scipy import ndimage
except Exception:
    ndimage = None


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


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


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"workspace": str(SCRIPT_DIR), "batch_name": "", "output_root": str(SCRIPT_DIR / "output")}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert registered point-cloud pairs to triangle meshes with Poisson reconstruction."
    )
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument(
        "--registration-dir",
        default="",
        help="Directory containing pair_XXX registration outputs. Empty uses output/<batch>/registration/pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated meshes. Empty uses output/<batch>/mesh/registered_pairs.",
    )
    parser.add_argument(
        "--pair",
        default="",
        help="Only process one pair folder, for example pair_001. Empty means all pairs.",
    )
    parser.add_argument(
        "--input-name",
        default="registered_pair.ply",
        choices=["registered_pair.ply", "bottom_registered_to_top.ply", "top.ply", "bottom.ply"],
        help="Point cloud to mesh inside each pair folder.",
    )
    parser.add_argument(
        "--copy-components",
        action="store_true",
        help="Also mesh top.ply and bottom_registered_to_top.ply separately when available.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0008,
        help="Voxel downsample size. 0 means auto estimate from point-cloud extent.",
    )
    parser.add_argument(
        "--auto-voxel-divisor",
        type=float,
        default=450.0,
        help="Auto voxel = bbox diagonal / this value.",
    )
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--normal-radius-multiplier", type=float, default=4.0)
    parser.add_argument("--normal-max-nn", type=int, default=40)
    parser.add_argument("--normal-orient-k", type=int, default=50)
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.0,
        help="Remove vertices below this density quantile after Poisson reconstruction.",
    )
    parser.add_argument(
        "--point-outlier-filter",
        choices=["off", "radius", "statistical", "both"],
        default="radius",
        help="Filter isolated input points before Poisson reconstruction.",
    )
    parser.add_argument(
        "--filter-interior",
        action="store_true",
        help="Also filter interior points. By default only edge-band outliers are removed.",
    )
    parser.add_argument(
        "--edge-band-multiplier",
        type=float,
        default=10.0,
        help="Edge band width = voxel_size * this value when --filter-interior is not used.",
    )
    parser.add_argument(
        "--outlier-radius-multiplier",
        type=float,
        default=4.0,
        help="Radius outlier search radius = voxel_size * this value.",
    )
    parser.add_argument(
        "--outlier-min-neighbors",
        type=int,
        default=6,
        help="Minimum neighbors inside the radius for keeping a point.",
    )
    parser.add_argument(
        "--stat-nb-neighbors",
        type=int,
        default=30,
        help="Neighbor count for statistical outlier filtering.",
    )
    parser.add_argument(
        "--stat-std-ratio",
        type=float,
        default=2.5,
        help="Standard deviation ratio for statistical outlier filtering.",
    )
    parser.add_argument(
        "--outlier-max-remove-ratio",
        type=float,
        default=0.1,
        help="Skip a point outlier filter step if it would remove more than this fraction of points.",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=1,
        help="Laplacian smoothing iterations after cleanup. 0 disables smoothing.",
    )
    parser.add_argument(
        "--save-downsampled",
        action="store_true",
        help="Save the downsampled and filtered point cloud used for reconstruction.",
    )
    return parser.parse_args()


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


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_pair_dirs(registration_root: Path, pair_name: str = ""):
    if pair_name:
        pair_dir = registration_root / pair_name
        if not pair_dir.is_dir():
            raise FileNotFoundError(f"Pair folder not found: {pair_dir}")
        yield pair_dir
        return

    found = False
    for pair_dir in sorted(registration_root.glob("pair_*")):
        if pair_dir.is_dir():
            found = True
            yield pair_dir
    if not found:
        raise FileNotFoundError(f"No pair_* folders found: {registration_root}")


def auto_voxel_size(pcd: o3d.geometry.PointCloud, divisor: float) -> float:
    extent = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    divisor = max(float(divisor), 1.0)
    voxel = extent / divisor
    return max(float(voxel), 1e-6)


def ensure_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
    orient_k: int,
) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    if orient_k > 0 and len(pcd.points) > orient_k:
        pcd.orient_normals_consistent_tangent_plane(orient_k)
    return pcd


def principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    frame = vecs[:, ::-1]
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1
    return center, frame


def edge_band_mask(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    edge_band_multiplier: float,
) -> tuple[np.ndarray | None, dict]:
    points = np.asarray(pcd.points)
    info = {
        "enabled": False,
        "available": False,
        "reason": "",
        "edge_band_width": voxel_size * edge_band_multiplier,
        "edge_points": 0,
        "edge_ratio": 0.0,
    }
    if edge_band_multiplier <= 0:
        info["reason"] = "edge_band_multiplier_disabled"
        return None, info
    if ndimage is None:
        info["reason"] = "scipy_ndimage_unavailable"
        return None, info
    if len(points) < 3:
        info["reason"] = "not_enough_points"
        return None, info

    center, frame = principal_frame(points)
    uv = (points - center) @ frame[:, :2]
    grid_size = max(voxel_size * 2.0, 1e-6)
    margin = max(voxel_size * edge_band_multiplier * 2.0, grid_size * 2.0)
    uv_min = uv.min(axis=0) - margin
    uv_max = uv.max(axis=0) + margin
    shape_xy = np.ceil((uv_max - uv_min) / grid_size).astype(np.int64) + 1
    if np.any(shape_xy <= 0):
        info["reason"] = "invalid_grid"
        return None, info

    max_cells = 4_000_000
    cells = int(shape_xy[0] * shape_xy[1])
    if cells > max_cells:
        scale = float(np.sqrt(cells / max_cells))
        grid_size *= scale
        shape_xy = np.ceil((uv_max - uv_min) / grid_size).astype(np.int64) + 1
        cells = int(shape_xy[0] * shape_xy[1])

    ij = np.floor((uv - uv_min) / grid_size).astype(np.int64)
    valid = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < shape_xy[0])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < shape_xy[1])
    )
    occupied = np.zeros((shape_xy[1], shape_xy[0]), dtype=bool)
    occupied[ij[valid, 1], ij[valid, 0]] = True
    if not np.any(occupied):
        info["reason"] = "empty_occupancy"
        return None, info

    distance_to_background = ndimage.distance_transform_edt(occupied) * grid_size
    edge_band_width = voxel_size * edge_band_multiplier
    edge_grid = occupied & (distance_to_background <= edge_band_width)
    edge_mask = np.zeros(len(points), dtype=bool)
    edge_mask[valid] = edge_grid[ij[valid, 1], ij[valid, 0]]

    edge_points = int(np.count_nonzero(edge_mask))
    info.update(
        {
            "enabled": True,
            "available": True,
            "grid_size": grid_size,
            "grid_width": int(shape_xy[0]),
            "grid_height": int(shape_xy[1]),
            "grid_cells": cells,
            "edge_band_width": edge_band_width,
            "edge_points": edge_points,
            "edge_ratio": edge_points / len(points),
        }
    )
    return edge_mask, info


def guarded_filter_step(
    pcd: o3d.geometry.PointCloud,
    name: str,
    filtered: o3d.geometry.PointCloud,
    max_remove_ratio: float,
    extra: dict,
) -> tuple[o3d.geometry.PointCloud, dict]:
    before = len(pcd.points)
    after = len(filtered.points)
    removed = before - after
    remove_ratio = removed / before if before > 0 else 0.0
    skipped = after <= 0 or remove_ratio > max_remove_ratio
    step = {
        "name": name,
        "before": before,
        "after": before if skipped else after,
        "candidate_after": after,
        "removed": 0 if skipped else removed,
        "candidate_removed": removed,
        "remove_ratio": 0.0 if skipped else remove_ratio,
        "candidate_remove_ratio": remove_ratio,
        "skipped": skipped,
        "skip_reason": "too_many_points_removed" if skipped and after > 0 else "empty_result" if skipped else "",
    }
    step.update(extra)
    return (pcd if skipped else filtered), step


def guarded_index_filter_step(
    pcd: o3d.geometry.PointCloud,
    name: str,
    remove_mask: np.ndarray,
    max_remove_ratio: float,
    extra: dict,
) -> tuple[o3d.geometry.PointCloud, dict]:
    before = len(pcd.points)
    remove_mask = np.asarray(remove_mask, dtype=bool)
    removed = int(np.count_nonzero(remove_mask))
    after = before - removed
    remove_ratio = removed / before if before > 0 else 0.0
    skipped = after <= 0 or remove_ratio > max_remove_ratio
    step = {
        "name": name,
        "before": before,
        "after": before if skipped else after,
        "candidate_after": after,
        "removed": 0 if skipped else removed,
        "candidate_removed": removed,
        "remove_ratio": 0.0 if skipped else remove_ratio,
        "candidate_remove_ratio": remove_ratio,
        "skipped": skipped,
        "skip_reason": "too_many_points_removed" if skipped and after > 0 else "empty_result" if skipped else "",
    }
    step.update(extra)
    if skipped or removed == 0:
        return pcd, step
    keep_indices = np.flatnonzero(~remove_mask)
    return pcd.select_by_index(keep_indices), step


def make_edge_limited_remove_mask(
    before_count: int,
    kept_indices: list[int],
    edge_mask: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    keep_mask = np.zeros(before_count, dtype=bool)
    keep_mask[np.asarray(kept_indices, dtype=np.int64)] = True
    candidate_remove_mask = ~keep_mask
    candidate_removed = int(np.count_nonzero(candidate_remove_mask))
    if edge_mask is None:
        return candidate_remove_mask, candidate_removed
    return candidate_remove_mask & edge_mask, candidate_removed


def filter_point_cloud_outliers(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    args: argparse.Namespace,
) -> tuple[o3d.geometry.PointCloud, dict]:
    mode = args.point_outlier_filter
    info = {
        "mode": mode,
        "before": len(pcd.points),
        "after": len(pcd.points),
        "removed": 0,
        "max_remove_ratio": args.outlier_max_remove_ratio,
        "steps": [],
        "filter_interior": args.filter_interior,
        "edge_band": None,
    }
    if mode == "off" or len(pcd.points) == 0:
        return pcd, info

    max_remove_ratio = min(max(float(args.outlier_max_remove_ratio), 0.0), 1.0)
    edge_mask = None
    if not args.filter_interior:
        edge_mask, edge_info = edge_band_mask(pcd, voxel_size, args.edge_band_multiplier)
        info["edge_band"] = edge_info
        if edge_mask is None:
            print(
                f"[WARN] edge-band filter unavailable ({edge_info.get('reason', 'unknown')}); "
                "falling back to full-cloud outlier filtering.",
                flush=True,
            )
    else:
        info["edge_band"] = {"enabled": False, "reason": "filter_interior_enabled"}

    if mode in ("radius", "both"):
        radius = voxel_size * args.outlier_radius_multiplier
        if radius > 0 and args.outlier_min_neighbors > 0 and len(pcd.points) > args.outlier_min_neighbors:
            before = len(pcd.points)
            _, kept_indices = pcd.remove_radius_outlier(
                nb_points=args.outlier_min_neighbors,
                radius=radius,
            )
            remove_mask, candidate_removed = make_edge_limited_remove_mask(before, kept_indices, edge_mask)
            pcd, step = guarded_index_filter_step(
                pcd,
                "radius",
                remove_mask,
                max_remove_ratio,
                {
                    "radius": radius,
                    "radius_multiplier": args.outlier_radius_multiplier,
                    "min_neighbors": args.outlier_min_neighbors,
                    "full_cloud_candidate_removed": candidate_removed,
                    "edge_limited": edge_mask is not None,
                },
            )
            info["steps"].append(step)
            if edge_mask is not None and not step["skipped"]:
                edge_mask = edge_mask[~remove_mask]

    if mode in ("statistical", "both"):
        if args.stat_nb_neighbors > 0 and len(pcd.points) > args.stat_nb_neighbors:
            before = len(pcd.points)
            _, kept_indices = pcd.remove_statistical_outlier(
                nb_neighbors=args.stat_nb_neighbors,
                std_ratio=args.stat_std_ratio,
            )
            remove_mask, candidate_removed = make_edge_limited_remove_mask(before, kept_indices, edge_mask)
            pcd, step = guarded_index_filter_step(
                pcd,
                "statistical",
                remove_mask,
                max_remove_ratio,
                {
                    "nb_neighbors": args.stat_nb_neighbors,
                    "std_ratio": args.stat_std_ratio,
                    "full_cloud_candidate_removed": candidate_removed,
                    "edge_limited": edge_mask is not None,
                },
            )
            info["steps"].append(step)
            if edge_mask is not None and not step["skipped"]:
                edge_mask = edge_mask[~remove_mask]

    info["after"] = len(pcd.points)
    info["removed"] = info["before"] - info["after"]
    return pcd, info


def remove_non_finite_vertices(mesh: o3d.geometry.TriangleMesh) -> int:
    if len(mesh.vertices) == 0:
        return 0
    vertices = np.asarray(mesh.vertices)
    mask = ~np.all(np.isfinite(vertices), axis=1)
    removed = int(np.count_nonzero(mask))
    if removed > 0:
        mesh.remove_vertices_by_mask(mask)
    return removed


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
    densities: np.ndarray,
    density_quantile: float,
    smooth_iterations: int,
) -> tuple[o3d.geometry.TriangleMesh, float | None, int]:
    non_finite_removed = remove_non_finite_vertices(mesh)
    threshold = None
    if density_quantile > 0 and len(densities) == len(mesh.vertices):
        threshold = float(np.quantile(densities, density_quantile))
        mesh.remove_vertices_by_mask(densities < threshold)

    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    non_finite_removed += remove_non_finite_vertices(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if smooth_iterations > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)
        non_finite_removed += remove_non_finite_vertices(mesh)
    mesh.compute_vertex_normals()
    return mesh, threshold, non_finite_removed


def mesh_quality(mesh: o3d.geometry.TriangleMesh) -> dict:
    if len(mesh.triangles) > 250000:
        return {
            "is_watertight": "skipped_large_mesh",
            "is_edge_manifold_allow_boundary": "skipped_large_mesh",
            "is_edge_manifold_no_boundary": "skipped_large_mesh",
            "is_vertex_manifold": "skipped_large_mesh",
            "is_self_intersecting": "skipped_large_mesh",
        }
    checks = {
        "is_watertight": lambda: mesh.is_watertight(),
        "is_edge_manifold_allow_boundary": lambda: mesh.is_edge_manifold(allow_boundary_edges=True),
        "is_edge_manifold_no_boundary": lambda: mesh.is_edge_manifold(allow_boundary_edges=False),
        "is_vertex_manifold": lambda: mesh.is_vertex_manifold(),
        "is_self_intersecting": lambda: mesh.is_self_intersecting(),
    }
    result = {}
    for name, func in checks.items():
        try:
            result[name] = bool(func())
        except Exception:
            result[name] = None
    return result


def save_mesh_preview(path: Path, mesh: o3d.geometry.TriangleMesh, title: str, max_points: int = 50000) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] mesh preview skipped: {exc}", flush=True)
        return

    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return
    valid = np.all(np.isfinite(vertices), axis=1)
    if not np.any(valid):
        return
    vertices = vertices[valid]
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    if colors is not None and len(colors) == len(valid):
        colors = colors[valid]

    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = centered.T @ centered / max(len(vertices) - 1, 1)
    try:
        _, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError as exc:
        print(f"[WARN] mesh preview skipped: {exc}", flush=True)
        return
    frame = vecs[:, ::-1]
    uv = centered @ frame[:, :2]

    rng = np.random.default_rng(7)
    if len(uv) > max_points:
        ids = rng.choice(len(uv), size=max_points, replace=False)
        ids = np.sort(ids)
        uv = uv[ids]
        if colors is not None and len(colors) == len(vertices):
            colors = colors[ids]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    if colors is not None and len(colors) == len(uv):
        ax.scatter(uv[:, 0], uv[:, 1], s=0.2, c=np.clip(colors, 0, 1), alpha=0.8)
    else:
        ax.scatter(uv[:, 0], uv[:, 1], s=0.2, color="#666666", alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_mesh(input_path: Path, output_path: Path, args: argparse.Namespace) -> dict:
    print(f"[mesh] reading {input_path}", flush=True)
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {input_path}")

    original_points = len(pcd.points)
    original_has_colors = pcd.has_colors()
    voxel_size = args.voxel_size if args.voxel_size > 0 else auto_voxel_size(pcd, args.auto_voxel_divisor)

    pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points_before_filter = len(pcd.points)
    pcd, outlier_info = filter_point_cloud_outliers(pcd, voxel_size, args)
    print(
        f"[mesh] points={original_points} downsampled={downsampled_points_before_filter} "
        f"filtered={len(pcd.points)} voxel={voxel_size:.9f}",
        flush=True,
    )
    pcd = ensure_normals(
        pcd,
        radius=voxel_size * args.normal_radius_multiplier,
        max_nn=args.normal_max_nn,
        orient_k=args.normal_orient_k,
    )

    print(f"[mesh] poisson depth={args.poisson_depth}", flush=True)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.poisson_depth,
    )
    raw_vertices = len(mesh.vertices)
    raw_triangles = len(mesh.triangles)
    mesh, density_threshold, non_finite_removed = clean_mesh(
        mesh,
        pcd,
        np.asarray(densities),
        args.density_quantile,
        args.smooth_iterations,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write mesh: {output_path}")
    preview_path = output_path.with_name(output_path.stem + "_preview.png")
    save_mesh_preview(preview_path, mesh, title=output_path.stem)

    downsampled_path = None
    if args.save_downsampled:
        downsampled_path = output_path.with_name(output_path.stem + "_downsampled.ply")
        if not o3d.io.write_point_cloud(str(downsampled_path), pcd, write_ascii=False):
            raise RuntimeError(f"Failed to write downsampled point cloud: {downsampled_path}")

    result = {
        "method": "poisson",
        "input": str(input_path),
        "mesh": str(output_path),
        "preview": str(preview_path),
        "downsampled": str(downsampled_path) if downsampled_path else None,
        "original_points": original_points,
        "downsampled_points_before_filter": downsampled_points_before_filter,
        "downsampled_points": len(pcd.points),
        "point_outlier_filter": outlier_info,
        "has_colors": original_has_colors,
        "voxel_size": voxel_size,
        "poisson_depth": args.poisson_depth,
        "density_quantile": args.density_quantile,
        "density_threshold": density_threshold,
        "smooth_iterations": args.smooth_iterations,
        "non_finite_vertices_removed": non_finite_removed,
        "raw_vertices": raw_vertices,
        "raw_triangles": raw_triangles,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "quality": mesh_quality(mesh),
    }
    print(
        f"[mesh] wrote {output_path} vertices={result['vertices']} triangles={result['triangles']}",
        flush=True,
    )
    return result


def input_candidates(pair_dir: Path, output_pair_dir: Path, args: argparse.Namespace):
    items = [
        (
            pair_dir / args.input_name,
            output_pair_dir / f"{Path(args.input_name).stem}_mesh.ply",
        )
    ]
    if args.copy_components:
        items.extend(
            [
                (pair_dir / "bottom_registered_to_top.ply", output_pair_dir / "bottom_registered_to_top_mesh.ply"),
                (pair_dir / "top.ply", output_pair_dir / "top_mesh.ply"),
            ]
        )
    return items


def process_pair(pair_dir: Path, output_root: Path, args: argparse.Namespace) -> dict:
    output_pair_dir = output_root / pair_dir.name
    output_pair_dir.mkdir(parents=True, exist_ok=True)

    meshes = []
    for input_path, output_path in input_candidates(pair_dir, output_pair_dir, args):
        if not input_path.exists():
            print(f"[skip] missing {input_path}", flush=True)
            continue
        meshes.append(build_mesh(input_path, output_path, args))

    if not meshes:
        raise FileNotFoundError(f"No mesh inputs found in {pair_dir}")

    result = {
        "pair": pair_dir.name,
        "mesh_method": "poisson",
        "input_pair_dir": str(pair_dir),
        "output_pair_dir": str(output_pair_dir),
        "meshes": meshes,
    }
    summary_path = output_pair_dir / "mesh_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[mesh] summary {summary_path}", flush=True)
    return result


def main():
    args = parse_args()
    config = load_config()
    workspace = resolve_workspace(args.workspace)
    output_root = Path(config["output_root"])
    default_registration_dir = output_root / "registration" / "pairs"
    default_output_dir = output_root / "mesh" / "registered_pairs"
    args.registration_dir = args.registration_dir or relative_or_absolute(default_registration_dir, workspace)
    args.output_dir = args.output_dir or relative_or_absolute(default_output_dir, workspace)
    registration_root = resolve_workspace_path(args.registration_dir, workspace)
    mesh_output_root = resolve_workspace_path(args.output_dir, workspace)
    mesh_output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for pair_dir in iter_pair_dirs(registration_root, args.pair):
        print(f"[pair] {pair_dir.name}", flush=True)
        results.append(process_pair(pair_dir, mesh_output_root, args))

    batch = {
        "workspace": str(workspace),
        "batch_name": config.get("batch_name"),
        "output_root": str(output_root),
        "registration_dir": str(registration_root),
        "output_dir": str(mesh_output_root),
        "count": len(results),
        "results": results,
    }
    batch_path = mesh_output_root / "mesh_batch_summary.json"
    with batch_path.open("w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)
    print(f"[OK] meshed {len(results)} pair(s)", flush=True)
    print(f"[OK] batch summary {batch_path}", flush=True)


if __name__ == "__main__":
    main()
