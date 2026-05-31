#!/usr/bin/env python3
"""
Export a rotating GIF/MP4 preview from a triangle mesh.

Default input:
  output/batch0/mesh/registered_pairs/pair_002/registered_pair_mesh.ply

The input must contain triangle faces.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d

try:
    import cv2
except Exception:
    cv2 = None

try:
    from numba import njit
except Exception:
    njit = None


DEFAULT_INPUT = Path(r"C:\Users\57746\Desktop\newpipeline\output\batch1\mesh\registered_pairs\pair_004\registered_pair_mesh.ply")


def parse_color(value: str) -> np.ndarray:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise argparse.ArgumentTypeError("color must be like #ffffff")
    try:
        return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be like #ffffff") from exc


def resolve_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")

    preferred = path / "registered_pair_mesh.ply"
    if preferred.exists():
        return preferred

    ply_files = sorted(path.glob("*.ply"))
    if len(ply_files) == 1:
        return ply_files[0]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {path}")
    raise RuntimeError(f"Multiple .ply files found in {path}; pass the file path explicitly.")


def default_output_path(input_path: Path) -> Path:
    for parent in [input_path.parent, *input_path.parents]:
        if parent.name.lower() == "output":
            return parent / "gif" / f"{input_path.stem}_rotate.gif"
    return input_path.parent / "gif" / f"{input_path.stem}_rotate.gif"


def load_mesh(input_path: Path) -> o3d.geometry.TriangleMesh | None:
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if len(mesh.triangles) == 0:
        return None
    return mesh


def camera_basis(yaw: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ce = math.cos(elevation)
    forward = np.array([ce * math.cos(yaw), ce * math.sin(yaw), math.sin(elevation)])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-12)
    return right, up, forward


def prepare_mesh_arrays(
    mesh: o3d.geometry.TriangleMesh,
    triangle_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if mesh.has_vertex_colors() else None

    finite = np.all(np.isfinite(vertices), axis=1)
    if not np.all(finite):
        valid_ids = np.full(len(vertices), -1, dtype=np.int64)
        valid_ids[finite] = np.arange(np.count_nonzero(finite))
        keep_triangles = np.all(finite[triangles], axis=1)
        triangles = valid_ids[triangles[keep_triangles]].astype(np.int32)
        vertices = vertices[finite]
        if colors is not None and len(colors) == len(finite):
            colors = colors[finite]

    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError("Mesh has no valid renderable geometry.")

    if triangle_limit > 0 and len(triangles) > triangle_limit:
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(triangles), size=triangle_limit, replace=False)
        ids.sort()
        triangles = triangles[ids]
        print(f"[gif] sampled mesh faces: {len(mesh.triangles)} -> {len(triangles)} triangles", flush=True)

    if colors is not None and len(colors) == len(vertices):
        colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors_u8 = np.full((len(vertices), 3), 180, dtype=np.uint8)

    return vertices, triangles, colors_u8


def _rasterize_mesh_python(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
    image: np.ndarray,
    zbuffer: np.ndarray,
) -> None:
    height, width = zbuffer.shape
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        x0, y0, z0 = xs[i0], ys[i0], zs[i0]
        x1, y1, z1 = xs[i1], ys[i1], zs[i1]
        x2, y2, z2 = xs[i2], ys[i2], zs[i2]
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-9:
            continue
        min_x = max(int(math.floor(min(x0, x1, x2))), 0)
        max_x = min(int(math.ceil(max(x0, x1, x2))), width - 1)
        min_y = max(int(math.floor(min(y0, y1, y2))), 0)
        max_y = min(int(math.ceil(max(y0, y1, y2))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        c0 = colors[i0]
        c1 = colors[i1]
        c2 = colors[i2]
        inv_area = 1.0 / area
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                sx = px + 0.5
                sy = py + 0.5
                w0 = ((x1 - sx) * (y2 - sy) - (y1 - sy) * (x2 - sx)) * inv_area
                w1 = ((x2 - sx) * (y0 - sy) - (y2 - sy) * (x0 - sx)) * inv_area
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z <= zbuffer[py, px]:
                    continue
                zbuffer[py, px] = z
                image[py, px, 0] = int(w0 * c0[0] + w1 * c1[0] + w2 * c2[0])
                image[py, px, 1] = int(w0 * c0[1] + w1 * c1[1] + w2 * c2[1])
                image[py, px, 2] = int(w0 * c0[2] + w1 * c1[2] + w2 * c2[2])


if njit is not None:
    rasterize_mesh = njit(cache=True)(_rasterize_mesh_python)
else:
    rasterize_mesh = _rasterize_mesh_python


def render_mesh_frame_cpu(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    scale: float,
    yaw: float,
    elevation: float,
    width: int,
    height: int,
    background: np.ndarray,
) -> np.ndarray:
    right, up, forward = camera_basis(yaw, elevation)
    centered = vertices - center
    xs = width * 0.5 + centered @ right * scale
    ys = height * 0.5 - centered @ up * scale
    zs = centered @ forward

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = background
    zbuffer = np.full((height, width), -np.inf, dtype=np.float64)
    rasterize_mesh(xs, ys, zs, triangles, colors, image, zbuffer)
    return image


def render_mesh_animation(args: argparse.Namespace, input_path: Path, output_path: Path) -> Path:
    mesh = load_mesh(input_path)
    if mesh is None:
        raise RuntimeError(f"Input has no triangles and cannot be rendered as a mesh: {input_path}")

    vertices, triangles, colors = prepare_mesh_arrays(mesh, args.triangle_limit, args.seed)
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Mesh has invalid bounds.")

    background = parse_color(args.background)
    scale = min(args.width, args.height) * float(args.padding) / (2.0 * radius)

    suffix = output_path.suffix.lower()
    print(
        f"[gif] rendering mesh {input_path} triangles={len(triangles)} frames={args.frames} "
        f"size={args.width}x{args.height} fps={args.fps}",
        flush=True,
    )

    def frame_at(frame_id: int) -> np.ndarray:
        yaw = math.radians(args.start_angle) + 2.0 * math.pi * frame_id / args.frames
        elevation = math.radians(args.elevation)
        return render_mesh_frame_cpu(
            vertices,
            triangles,
            colors,
            center,
            scale,
            yaw,
            elevation,
            args.width,
            args.height,
            background,
        )

    if suffix in {".mp4", ".m4v", ".mov"}:
        if cv2 is None:
            raise RuntimeError("MP4 output needs opencv-python or imageio-ffmpeg in this environment.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(args.fps), (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open MP4 writer: {output_path}")
        try:
            for frame_id in range(args.frames):
                writer.write(cv2.cvtColor(frame_at(frame_id), cv2.COLOR_RGB2BGR))
                report_frame_progress(frame_id, args.frames)
        finally:
            writer.release()
    else:
        writer_kwargs = {"duration": 1.0 / max(args.fps, 1), "loop": 0}
        with imageio.get_writer(str(output_path), mode="I", **writer_kwargs) as writer:
            for frame_id in range(args.frames):
                writer.append_data(frame_at(frame_id))
                report_frame_progress(frame_id, args.frames)

    print(f"[gif] saved: {output_path}", flush=True)
    return output_path


def save_animation(args: argparse.Namespace) -> Path:
    input_path = resolve_input(Path(args.input))
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return render_mesh_animation(args, input_path, output_path)


def report_frame_progress(frame_id: int, frames: int) -> None:
    if frame_id == 0 or (frame_id + 1) % max(1, frames // 6) == 0:
        print(f"[gif] frame {frame_id + 1}/{frames}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a rotating triangle-mesh GIF or MP4.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input triangle-mesh .ply file.",
    )
    parser.add_argument("--output", default="", help="Output .gif or .mp4 path. Empty writes next to input .ply.")
    parser.add_argument("--frames", type=int, default=72, help="Number of frames in one rotation.")
    parser.add_argument("--fps", type=int, default=18, help="Playback frames per second.")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--triangle-limit", type=int, default=0, help="Maximum mesh triangles to render. 0 renders all triangles.")
    parser.add_argument("--padding", type=float, default=0.86, help="Fraction of image used by the mesh.")
    parser.add_argument("--elevation", type=float, default=25.0, help="Camera elevation in degrees.")
    parser.add_argument("--start-angle", type=float, default=0.0, help="Starting yaw angle in degrees.")
    parser.add_argument("--background", default="#ffffff", help="Background color, for example #ffffff or #111111.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used only when --triangle-limit samples faces.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    save_animation(args)


if __name__ == "__main__":
    main()
