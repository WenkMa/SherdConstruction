#!/usr/bin/env python3
"""
Match top/bottom ceramic fragments by projected outline shape.

The matcher uses only geometry available after Euclidean split:

  output/<batch>/fragments/top/raw/frag_*.ply
  output/<batch>/fragments/bottom/raw/frag_*.ply

For each fragment it computes a PCA plane, extracts a polar ordered outline,
resamples it to a fixed length, then compares every top-bottom pair while
searching circular shift and mirrored orientation. The final one-to-one pairing
is solved with the Hungarian algorithm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


@dataclass
class FragmentDescriptor:
    side: str
    fragment_id: str
    path: Path
    points: int
    bbox_diag: float
    rms_radius: float
    contour: np.ndarray
    outline_points: int


def load_config() -> dict:
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


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    axes = vecs[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return center, axes


def resample_closed(points: np.ndarray, samples: int) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg_len.sum())
    if total <= 1e-12:
        raise ValueError("Degenerate contour")
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, total, samples, endpoint=False)
    out = np.empty((samples, points.shape[1]), dtype=np.float64)
    for i, target in enumerate(targets):
        seg = int(np.searchsorted(cumulative, target, side="right") - 1)
        seg = min(seg, len(seg_len) - 1)
        t = 0.0 if seg_len[seg] <= 1e-12 else (target - cumulative[seg]) / seg_len[seg]
        out[i] = closed[seg] * (1.0 - t) + closed[seg + 1] * t
    return out


def normalize_contour(contour: np.ndarray) -> np.ndarray:
    centered = contour - contour.mean(axis=0)
    rms = math.sqrt(float(np.mean(np.sum(centered * centered, axis=1))))
    if rms <= 1e-12:
        return centered
    return centered / rms


def projected_outline(
    points: np.ndarray,
    outline_bins: int,
    points_per_bin: int,
    samples: int,
    method: str = "alpha",
    grid_size: float = 0.006,
    alpha_radius: float = 0.0,
) -> tuple[np.ndarray, int]:
    center, axes = pca_frame(points)
    plane = (points - center) @ axes
    uv = plane[:, :2]

    if method == "alpha":
        try:
            outline = alpha_shape_outline(uv, grid_size=grid_size, alpha_radius=alpha_radius)
            return resample_closed(outline, samples), len(outline)
        except Exception as exc:
            print(f"[WARN] alpha outline failed, using radial outline: {exc}", flush=True)

    return radial_outline(uv, outline_bins, points_per_bin, samples)


def radial_outline(
    uv: np.ndarray,
    outline_bins: int,
    points_per_bin: int,
    samples: int,
) -> tuple[np.ndarray, int]:
    uv_center = np.median(uv, axis=0)
    rel = uv - uv_center
    radius = np.linalg.norm(rel, axis=1)
    angle = np.arctan2(rel[:, 1], rel[:, 0])

    bin_ids = np.floor((angle + np.pi) / (2.0 * np.pi) * outline_bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, outline_bins - 1)
    selected = []
    for bin_id in range(outline_bins):
        ids = np.flatnonzero(bin_ids == bin_id)
        if len(ids) == 0:
            continue
        keep = min(points_per_bin, len(ids))
        local = np.argpartition(radius[ids], len(ids) - keep)[-keep:]
        selected.append(ids[local])
    if not selected:
        raise RuntimeError("No outline points selected")

    outline = uv[np.unique(np.concatenate(selected))]
    outline_center = np.median(outline, axis=0)
    order = np.argsort(np.arctan2(outline[:, 1] - outline_center[1], outline[:, 0] - outline_center[0]))
    ordered = outline[order]
    return resample_closed(ordered, samples), len(outline)


def grid_sample_2d(points: np.ndarray, grid_size: float) -> np.ndarray:
    grid_size = max(float(grid_size), 1e-9)
    lo = points.min(axis=0)
    keys = np.floor((points - lo) / grid_size).astype(np.int64)
    unique_keys = np.unique(keys, axis=0)
    return lo + (unique_keys.astype(np.float64) + 0.5) * grid_size


def triangle_circumradius(triangles: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    ab = triangles[:, 1] - triangles[:, 0]
    ac = triangles[:, 2] - triangles[:, 0]
    area2 = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    return (a * b * c) / np.maximum(2.0 * area2, 1e-12)


def convex_hull_outline(points: np.ndarray) -> np.ndarray:
    hull = ConvexHull(points)
    return points[hull.vertices]


def alpha_shape_outline(uv: np.ndarray, grid_size: float, alpha_radius: float) -> np.ndarray:
    sampled = grid_sample_2d(uv, grid_size=grid_size)
    if len(sampled) < 4:
        raise ValueError("Too few points for alpha outline")

    radius = float(alpha_radius) if alpha_radius > 0 else float(grid_size) * 4.0
    triangulation = Delaunay(sampled)
    simplices = triangulation.simplices
    tri_pts = sampled[simplices]
    keep = triangle_circumradius(tri_pts) <= radius
    kept = simplices[keep]
    if len(kept) == 0:
        return convex_hull_outline(sampled)

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in kept:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(a), int(b)) if a < b else (int(b), int(a))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if len(boundary_edges) < 3:
        return convex_hull_outline(sampled)

    graph: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    component = largest_boundary_component(graph, sampled)
    ordered = order_boundary_component(component, graph, sampled)
    if len(ordered) < 3:
        return convex_hull_outline(sampled)
    return sampled[np.asarray(ordered, dtype=np.int64)]


def largest_boundary_component(graph: dict[int, list[int]], points: np.ndarray) -> list[int]:
    visited = set()
    best: list[int] = []
    best_perimeter = -1.0
    for start in graph:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        perimeter = 0.0
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in graph[item]:
                perimeter += float(np.linalg.norm(points[item] - points[neighbor]))
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        if perimeter > best_perimeter:
            best_perimeter = perimeter
            best = component
    return best


def order_boundary_component(component: list[int], graph: dict[int, list[int]], points: np.ndarray) -> list[int]:
    component_set = set(component)
    endpoints = [idx for idx in component if len([n for n in graph[idx] if n in component_set]) == 1]
    start = endpoints[0] if endpoints else min(component)
    ordered = []
    previous = None
    current = start

    for _ in range(len(component) + 2):
        ordered.append(current)
        candidates = [n for n in graph[current] if n in component_set and n != previous]
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            if previous is None:
                nxt = min(candidates)
            else:
                direction = points[current] - points[previous]
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                scores = []
                for candidate in candidates:
                    candidate_dir = points[candidate] - points[current]
                    candidate_dir /= max(float(np.linalg.norm(candidate_dir)), 1e-12)
                    scores.append(float(np.dot(direction, candidate_dir)))
                nxt = candidates[int(np.argmax(scores))]
        if nxt == start:
            break
        previous, current = current, nxt

    return ordered


def load_descriptor(path: Path, side: str, args: argparse.Namespace) -> FragmentDescriptor:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty fragment: {path}")
    points = np.asarray(pcd.points, dtype=np.float64)
    extent = np.ptp(points, axis=0)
    center = points.mean(axis=0)
    contour, outline_points = projected_outline(
        points,
        outline_bins=args.outline_bins,
        points_per_bin=args.outline_points_per_bin,
        samples=args.samples,
        method=args.outline_method,
        grid_size=args.outline_grid_size,
        alpha_radius=args.outline_alpha_radius,
    )
    return FragmentDescriptor(
        side=side,
        fragment_id=path.stem,
        path=path,
        points=len(points),
        bbox_diag=float(np.linalg.norm(extent)),
        rms_radius=float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))),
        contour=contour,
        outline_points=outline_points,
    )


def estimate_2d_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cs = source.mean(axis=0)
    ct = target.mean(axis=0)
    xs = source - cs
    xt = target - ct
    u, _, vt = np.linalg.svd(xs.T @ xt)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    rotated = xs @ r.T
    denom = float(np.sum(rotated * rotated))
    scale = 1.0 if denom <= 1e-12 else float(np.sum(rotated * xt) / denom)
    return rotated * scale + ct


def contour_variants(contour: np.ndarray, allow_mirror: bool) -> list[tuple[str, np.ndarray]]:
    contour_n = normalize_contour(contour)
    variants: list[tuple[str, np.ndarray]] = [
        ("direct", contour_n),
        ("reversed", contour_n[::-1]),
    ]
    if allow_mirror:
        mirrored = contour_n.copy()
        mirrored[:, 0] *= -1.0
        variants.extend([
            ("mirror_x", mirrored),
            ("mirror_x_reversed", mirrored[::-1]),
        ])
    return variants


def aligned_contours(
    top: np.ndarray,
    bottom: np.ndarray,
    shift: int,
    variant: str,
    allow_mirror: bool,
) -> tuple[np.ndarray, np.ndarray]:
    top_n = normalize_contour(top)
    variants = dict(contour_variants(bottom, allow_mirror=allow_mirror))
    if variant not in variants:
        raise ValueError(f"Unknown contour variant: {variant}")
    shifted = np.roll(variants[variant], -shift, axis=0)
    return top_n, estimate_2d_similarity(shifted, top_n)


def compare_contours(top: np.ndarray, bottom: np.ndarray, allow_mirror: bool) -> dict:
    top_n = normalize_contour(top)
    variants = contour_variants(bottom, allow_mirror=allow_mirror)

    best = {"distance": math.inf, "shift": 0, "mirror": 0, "variant": ""}
    n = len(top_n)
    for variant_name, candidate in variants:
        for shift in range(n):
            shifted = np.roll(candidate, -shift, axis=0)
            moved = estimate_2d_similarity(shifted, top_n)
            distance = float(np.sqrt(np.mean(np.sum((moved - top_n) ** 2, axis=1))))
            if distance < best["distance"]:
                best = {
                    "distance": distance,
                    "shift": shift,
                    "mirror": int(variant_name.startswith("mirror")),
                    "variant": variant_name,
                }
    return best


def load_fragments(workspace: Path, side: str, args: argparse.Namespace) -> list[FragmentDescriptor]:
    raw_dir = workspace / args.fragments_dir / side / "raw"
    files = sorted(raw_dir.glob("frag_*.ply"))
    if args.max_fragments > 0:
        files = files[: args.max_fragments]
    if not files:
        raise FileNotFoundError(f"No fragments found: {raw_dir}")
    result = []
    for path in files:
        print(f"[load] {side}/{path.name}", flush=True)
        result.append(load_descriptor(path, side, args))
    return result


def save_alignment_plot(
    path: Path,
    top_contour: np.ndarray,
    bottom_contour: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    top_closed = np.vstack([top_contour, top_contour[0]])
    bottom_closed = np.vstack([bottom_contour, bottom_contour[0]])

    ax.plot(top_closed[:, 0], top_closed[:, 1], color="#1f77b4", linewidth=2.0, label="top")
    ax.plot(bottom_closed[:, 0], bottom_closed[:, 1], color="#d62728", linewidth=2.0, label="bottom aligned")
    ax.scatter(top_contour[0, 0], top_contour[0, 1], color="#1f77b4", s=24)
    ax.scatter(bottom_contour[0, 0], bottom_contour[0, 1], color="#d62728", s=24)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_match_pair(
    pair_dir: Path,
    pair_index: int,
    top: FragmentDescriptor,
    bottom: FragmentDescriptor,
    info: dict,
    allow_mirror: bool,
) -> dict:
    pair_dir.mkdir(parents=True, exist_ok=True)
    top_out = pair_dir / "top.ply"
    bottom_out = pair_dir / "bottom.ply"
    shutil.copy2(top.path, top_out)
    shutil.copy2(bottom.path, bottom_out)

    top_aligned, bottom_aligned = aligned_contours(
        top.contour,
        bottom.contour,
        shift=int(info["shift"]),
        variant=str(info["variant"]),
        allow_mirror=allow_mirror,
    )
    image_out = pair_dir / "alignment_2d.png"
    save_alignment_plot(
        image_out,
        top_aligned,
        bottom_aligned,
        title=(
            f"pair_{pair_index:03d}: {top.fragment_id} <-> {bottom.fragment_id} "
            f"score={info['score']:.6f}"
        ),
    )

    metadata = {
        "pair": pair_index,
        "top_source": str(top.path),
        "bottom_source": str(bottom.path),
        "top_file": str(top_out),
        "bottom_file": str(bottom_out),
        "alignment_image": str(image_out),
        "top_fragment": top.fragment_id,
        "bottom_fragment": bottom.fragment_id,
        "distance": info["distance"],
        "score": info["score"],
        "shift": int(info["shift"]),
        "mirror": int(info["mirror"]),
        "variant": info["variant"],
        "scale": info["scale"],
        "size_penalty": info["size_penalty"],
    }
    with (pair_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match top/bottom split fragments by outline")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--top-side", default="top")
    parser.add_argument("--bottom-side", default="bottom")
    parser.add_argument("--fragments-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--pairs-dir", default="")
    parser.add_argument("--max-fragments", type=int, default=0)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--outline-bins", type=int, default=720)
    parser.add_argument("--outline-points-per-bin", type=int, default=8)
    parser.add_argument("--outline-method", choices=["alpha", "radial"], default="alpha")
    parser.add_argument("--outline-grid-size", type=float, default=0.006)
    parser.add_argument(
        "--outline-alpha-radius",
        type=float,
        default=0.0,
        help="Alpha-shape circumradius threshold. 0 uses outline-grid-size * 4.",
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--size-weight",
        type=float,
        default=0.15,
        help="Penalty weight for log RMS-size mismatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = Path(config["workspace"]) / workspace
    workspace = workspace.resolve()
    output_root = Path(config["output_root"])
    default_output = output_root / "matching" / "match_results.csv"
    default_summary = output_root / "matching" / "match_summary.json"
    default_pairs = output_root / "matching" / "pairs"
    default_fragments = output_root / "fragments"
    args.output = args.output or str(default_output.relative_to(workspace))
    args.summary = args.summary or str(default_summary.relative_to(workspace))
    args.pairs_dir = args.pairs_dir or str(default_pairs.relative_to(workspace))
    args.fragments_dir = args.fragments_dir or str(default_fragments.relative_to(workspace))

    top_items = load_fragments(workspace, args.top_side, args)
    bottom_items = load_fragments(workspace, args.bottom_side, args)

    cost = np.zeros((len(top_items), len(bottom_items)), dtype=np.float64)
    pair_info: dict[tuple[int, int], dict] = {}
    for i, top in enumerate(top_items):
        for j, bottom in enumerate(bottom_items):
            info = compare_contours(top.contour, bottom.contour, allow_mirror=not args.no_mirror)
            scale = top.rms_radius / bottom.rms_radius if bottom.rms_radius > 1e-12 else 1.0
            size_penalty = abs(math.log(max(scale, 1e-12))) * args.size_weight
            score = info["distance"] + size_penalty
            cost[i, j] = score
            pair_info[(i, j)] = {**info, "scale": scale, "score": score, "size_penalty": size_penalty}

    rows, cols = linear_sum_assignment(cost)
    output_path = (workspace / args.output).resolve()
    summary_path = (workspace / args.summary).resolve()
    pairs_root = (workspace / args.pairs_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)

    matches = []
    pair_exports = []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Pair",
                "Top",
                "Bottom",
                "Distance",
                "Score",
                "Shift",
                "Mirror",
                "Variant",
                "Scale",
                "TopPoints",
                "BottomPoints",
                "PairDir",
                "AlignmentImage",
            ],
        )
        writer.writeheader()
        for pair_index, (r, c) in enumerate(zip(rows, cols), start=1):
            top = top_items[int(r)]
            bottom = bottom_items[int(c)]
            info = pair_info[(int(r), int(c))]
            pair_dir = pairs_root / f"pair_{pair_index:03d}"
            export_info = export_match_pair(
                pair_dir,
                pair_index,
                top,
                bottom,
                info,
                allow_mirror=not args.no_mirror,
            )
            row = {
                "Pair": f"pair_{pair_index:03d}",
                "Top": top.fragment_id,
                "Bottom": bottom.fragment_id,
                "Distance": f"{info['distance']:.9f}",
                "Score": f"{info['score']:.9f}",
                "Shift": int(info["shift"]),
                "Mirror": int(info["mirror"]),
                "Variant": info["variant"],
                "Scale": f"{info['scale']:.9f}",
                "TopPoints": top.points,
                "BottomPoints": bottom.points,
                "PairDir": str(pair_dir),
                "AlignmentImage": str(export_info["alignment_image"]),
            }
            writer.writerow(row)
            matches.append(row)
            pair_exports.append(export_info)

    summary = {
        "workspace": str(workspace),
        "batch_name": config.get("batch_name"),
        "output_root": str(output_root),
        "top_side": args.top_side,
        "bottom_side": args.bottom_side,
        "samples": args.samples,
        "outline_bins": args.outline_bins,
        "outline_points_per_bin": args.outline_points_per_bin,
        "outline_method": args.outline_method,
        "outline_grid_size": args.outline_grid_size,
        "outline_alpha_radius": args.outline_alpha_radius,
        "size_weight": args.size_weight,
        "pairs_dir": str(pairs_root),
        "matches": matches,
        "pair_exports": pair_exports,
        "cost_matrix": cost.tolist(),
        "top_fragments": [
            {
                "id": item.fragment_id,
                "path": str(item.path),
                "points": item.points,
                "bbox_diag": item.bbox_diag,
                "rms_radius": item.rms_radius,
                "outline_points": item.outline_points,
            }
            for item in top_items
        ],
        "bottom_fragments": [
            {
                "id": item.fragment_id,
                "path": str(item.path),
                "points": item.points,
                "bbox_diag": item.bbox_diag,
                "rms_radius": item.rms_radius,
                "outline_points": item.outline_points,
            }
            for item in bottom_items
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {output_path}", flush=True)
    for item in matches:
        print(
            f"  {item['Top']} <-> {item['Bottom']} "
            f"score={item['Score']} distance={item['Distance']} "
            f"shift={item['Shift']} mirror={item['Mirror']} scale={item['Scale']}",
            flush=True,
        )
    print(f"[OK] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
