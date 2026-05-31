import os
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    apply_batch_override(config)
    expand_config_templates(config)
    workspace = config.get("workspace") or os.path.normpath(os.path.join(script_dir, "../.."))
    config["workspace"] = workspace
    batch_name = detect_batch_name(config)
    output_root = os.path.join(workspace, "output", batch_name) if batch_name else os.path.join(workspace, "output")
    config["batch_name"] = batch_name
    config["output_root"] = output_root
    config["data_root"] = os.path.join(output_root, "data")
    config["frag_root"] = os.path.join(output_root, "fragments")
    return config


def apply_batch_override(config):
    batch_number = os.environ.get("PIPELINE_BATCH_NUMBER", "")
    if not batch_number:
        return
    config["batch_number"] = int(batch_number)
    config["batch_num"] = int(batch_number)
    config.pop("OUTPUT_BATCH", None)
    config.pop("batch_name", None)


def expand_config_templates(config):
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


def detect_batch_name(config):
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


CONFIG = load_config()
MIN_POINTS = int(CONFIG["MIN_POINTS"])
TOP_K_FRAGMENTS = max(1, int(CONFIG["TOP_K_FRAGMENTS"]))
SPLIT_GRID_SIZE = float(CONFIG.get("SPLIT_GRID_SIZE", 0.01))
SPLIT_GRID_CONNECTIVITY = int(CONFIG.get("SPLIT_GRID_CONNECTIVITY", 8))
SPLIT_GRID_MIN_CELL_POINTS = max(1, int(CONFIG.get("SPLIT_GRID_MIN_CELL_POINTS", 1)))
SPLIT_PREVIEW_MAX_POINTS = int(CONFIG.get("SPLIT_PREVIEW_MAX_POINTS", 80000))

DATA_ROOT = CONFIG["data_root"]
FRAG_ROOT = CONFIG["frag_root"]
SIDES = CONFIG["sides"]
WHITE_FILTER_ENABLE = bool(CONFIG.get("WHITE_FILTER_ENABLE", False))


def prepare_clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def pca_project_xy(points):
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    frame = vecs[:, ::-1]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1.0
    uv = centered @ frame[:, :2]
    return uv


def grid_connected_component_split(pcd, cell_size, min_points, connectivity, min_cell_points):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return []

    cell_size = max(float(cell_size), 1e-9)
    uv = pca_project_xy(points)
    uv_min = uv.min(axis=0)
    cells = np.floor((uv - uv_min) / cell_size).astype(np.int64)

    unique_cells, inverse, cell_counts = np.unique(
        cells,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    valid_cell_mask = cell_counts >= int(min_cell_points)
    valid_cell_ids = np.flatnonzero(valid_cell_mask)
    valid_lookup = {tuple(unique_cells[cell_id]): int(cell_id) for cell_id in valid_cell_ids}

    print(
        f"[INFO] grid connected components: points={len(points)} cells={len(unique_cells)} "
        f"valid_cells={len(valid_cell_ids)} cell_size={cell_size:.6f} connectivity={connectivity}"
    )
    if len(valid_cell_ids) == 0:
        return []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    visited = set()
    components = []
    for start in valid_cell_ids:
        start_key = tuple(unique_cells[start])
        if start_key in visited:
            continue

        stack = [start_key]
        visited.add(start_key)
        component_cells = []
        total_points = 0
        while stack:
            key = stack.pop()
            cell_id = valid_lookup[key]
            component_cells.append(cell_id)
            total_points += int(cell_counts[cell_id])
            x, y = key
            for dx, dy in neighbors:
                next_key = (x + dx, y + dy)
                if next_key in visited or next_key not in valid_lookup:
                    continue
                visited.add(next_key)
                stack.append(next_key)

        if total_points >= int(min_points):
            components.append((component_cells, total_points))

    components.sort(key=lambda item: item[1], reverse=True)
    print(f"[INFO] grid components above min_points: {len(components)}")

    fragments = []
    for component_id, (component_cells, _) in enumerate(components):
        component_cell_mask = np.zeros(len(unique_cells), dtype=bool)
        component_cell_mask[np.asarray(component_cells, dtype=np.int64)] = True
        point_mask = component_cell_mask[inverse]
        global_idx = np.flatnonzero(point_mask)
        frag = pcd.select_by_index(global_idx)
        print(
            f"[DEBUG] grid component {component_id}: cells={len(component_cells)} "
            f"points={len(global_idx)}"
        )
        fragments.append((frag, global_idx))

    return fragments


def process_batch(input_ply, output_raw_dir):
    prepare_clean_dir(output_raw_dir)

    print(f"[INFO] Loading {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    if len(pcd.points) == 0:
        print("  [WARNING] empty point cloud")
        return

    frags = grid_connected_component_split(
        pcd,
        cell_size=SPLIT_GRID_SIZE,
        min_points=MIN_POINTS,
        connectivity=SPLIT_GRID_CONNECTIVITY,
        min_cell_points=SPLIT_GRID_MIN_CELL_POINTS,
    )
    if not frags:
        print("[DONE] 0 fragments")
        return

    frags.sort(key=lambda x: len(x[0].points), reverse=True)
    detected_n = len(frags)
    keep_n = min(TOP_K_FRAGMENTS, detected_n)
    frags = frags[:keep_n]

    print(f"[INFO] detected {detected_n} components; saving top {keep_n}")

    for i, (frag, global_idx) in enumerate(frags):
        frag_id = f"frag_{i:03d}"
        out_ply = os.path.join(output_raw_dir, f"{frag_id}.ply")
        out_indices = os.path.join(output_raw_dir, f"{frag_id}_global_indices.npy")

        o3d.io.write_point_cloud(out_ply, frag)
        np.save(out_indices, global_idx)
        print(f"  saved {out_ply} ({len(frag.points)} pts)")
        print(f"  saved {out_indices} (global indices)")

    save_fragments_preview(frags, output_raw_dir)
    print(f"[DONE] saved {len(frags)} fragments")


def save_fragments_preview(frags, output_raw_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] fragments preview skipped: {exc}")
        return

    if not frags:
        return

    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(frags))))

    for i, (frag, _) in enumerate(frags):
        pts = np.asarray(frag.points)
        if len(pts) == 0:
            continue
        if SPLIT_PREVIEW_MAX_POINTS > 0 and len(pts) > SPLIT_PREVIEW_MAX_POINTS:
            ids = rng.choice(len(pts), size=SPLIT_PREVIEW_MAX_POINTS, replace=False)
            pts = pts[np.sort(ids)]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.2, color=colors[i % len(colors)], label=f"frag_{i:03d}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Fragment split preview")
    ax.legend(markerscale=12, fontsize=8, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()

    preview_path = os.path.join(os.path.dirname(output_raw_dir), "fragments_preview.png")
    fig.savefig(preview_path)
    plt.close(fig)
    print(f"  saved {preview_path} (preview)")


if __name__ == "__main__":
    for side in SIDES:
        print(f"\n=== {side.upper()} batch ===")

        filtered_ply = os.path.join(DATA_ROOT, side, f"{side}_white_filtered.ply")
        raw_ply = os.path.join(DATA_ROOT, side, f"{side}.ply")

        if WHITE_FILTER_ENABLE and os.path.exists(filtered_ply):
            input_ply = filtered_ply
            print(f"[INFO] using white-filtered point cloud: {filtered_ply}")
        else:
            input_ply = raw_ply
            print(f"[INFO] using raw point cloud: {raw_ply}")

        output_raw = os.path.join(FRAG_ROOT, side, "raw")
        output_raw_clean = os.path.join(FRAG_ROOT, side, "raw_clean")

        if not os.path.exists(input_ply):
            print(f"  [ERROR] point cloud file not found: {input_ply}")
            continue

        if os.path.isdir(output_raw_clean):
            shutil.rmtree(output_raw_clean)
            print(f"[INFO] removed stale raw_clean: {output_raw_clean}")

        process_batch(input_ply, output_raw)
        print(f"[DEBUG] {side} batch done")

    print("\n[DEBUG] all batches done")
