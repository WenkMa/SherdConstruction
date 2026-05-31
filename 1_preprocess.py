"""
Step 0 preprocess powered by Agisoft Metashape.

This replaces the old external preprocess while preserving the legacy workspace
contract expected by the remaining core pipeline:

  data/<side>/<side>.ply
  output/<batch>/data/<side>/<side>.ply
  output/<batch>/preprocess/<side>/dense/scene_dense.ply
  output/<batch>/preprocess/<side>/sfm/sparse/0/{cameras,images,points3D}.txt
  output/<batch>/preprocess/<side>/sfm/images_for_pipeline.tsv
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE = SCRIPT_DIR
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
METASHAPE_SCRIPT = SCRIPT_DIR / "metashape_reconstruct.py"
DEFAULT_METASHAPE_EXE = Path(r"D:\Program Files\Agisoft\Metashape Pro\metashape.exe")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    apply_batch_override(config)
    expand_config_templates(config)
    workspace = Path(config.get("workspace") or WORKSPACE).resolve()
    config["workspace"] = str(workspace)
    batch_name = detect_batch_name(config)
    output_root = workspace / "output" / batch_name if batch_name else workspace / "output"
    config["batch_name"] = batch_name
    config["output_root"] = str(output_root)
    config["image_root"] = str(Path(config.get("IMAGE_ROOT") or workspace / "images").resolve())
    config["input_root"] = str(Path(config.get("INPUT_ROOT") or workspace / "newimages").resolve())
    config["data_root"] = str(output_root / "data")
    config["interim_pre"] = str(output_root / "preprocess")
    config["log_root"] = str(output_root / "logs")

    config.setdefault("METASHAPE_EXE", str(DEFAULT_METASHAPE_EXE))
    config.setdefault("METASHAPE_MATCH_DOWNSCALE", 1)
    config.setdefault("METASHAPE_DEPTH_DOWNSCALE", 4)
    config.setdefault("METASHAPE_DEPTH_FILTER", "mild")
    config.setdefault("METASHAPE_KEYPOINT_LIMIT", 40000)
    config.setdefault("METASHAPE_KEYPOINT_LIMIT_PER_MPX", 1000)
    config.setdefault("METASHAPE_TIEPOINT_LIMIT", 4000)
    config.setdefault("METASHAPE_MAX_NEIGHBORS", 16)
    config.setdefault("METASHAPE_POINT_CLOUD_MAX_NEIGHBORS", 100)
    config.setdefault("METASHAPE_LAYOUT", "prefix-sensors")
    config.setdefault("METASHAPE_BUILD_MODEL", False)
    config.setdefault("METASHAPE_USE_MASKS", True)
    config.setdefault("METASHAPE_MASK_MODE", "white-valid")
    config.setdefault("METASHAPE_MASK_MATCHING", False)
    config.setdefault("METASHAPE_MASK_TIEPOINTS", False)
    config.setdefault("METASHAPE_REUSE", False)
    config.setdefault("AUTO_PREPARE_INPUT", True)
    config.setdefault("PREPROCESS_PARALLEL_SIDES", False)
    config.setdefault("PREPROCESS_PARALLEL_WORKERS", 2)
    config.setdefault("PREPROCESS_REUSE_DENSE", False)
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


def ensure_file(path: Path, desc: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{desc} not found: {path}")


def ensure_dir(path: Path, desc: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{desc} not found: {path}")


def has_images(path: Path) -> bool:
    return path.is_dir() and any(
        item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        for item in path.iterdir()
    )


def list_image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    images = [
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: p.name.lower())
    return images


def mask_for_image(mask_dir: Path, image_path: Path) -> Path | None:
    candidates = [
        mask_dir / f"{image_path.name}.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.name}_mask.png",
        mask_dir / f"{image_path.stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def validate_mask_coverage(image_dir: Path, mask_dir: Path, side: str, min_ratio: float) -> None:
    images = list_image_files(image_dir)
    if not images:
        raise RuntimeError(f"{side}: no images found in {image_dir}")

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"{side} masks not found: {mask_dir}")

    matched = sum(1 for image in images if mask_for_image(mask_dir, image))
    ratio = matched / len(images)
    log_info(f"{side}: masks matched {matched}/{len(images)} in {mask_dir}")
    if ratio < min_ratio:
        raise RuntimeError(
            f"{side}: mask coverage too low: {matched}/{len(images)} matched, "
            f"required >= {min_ratio:.0%}"
        )


def batch_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []

    dirs = [root]
    batches = [
        item for item in root.iterdir()
        if item.is_dir() and item.name.lower().startswith("batch")
    ]

    def sort_key(path: Path) -> list[int | str]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

    dirs.extend(sorted(batches, key=sort_key))
    return dirs


def side_aliases(side: str) -> list[str]:
    lowered = side.lower()
    aliases = [side, lowered]
    stripped = re.sub(r"\d+$", "", lowered)
    if stripped and stripped not in aliases:
        aliases.append(stripped)

    if "front" in lowered or "top" in lowered:
        aliases.extend(["front", "top"])
    if "back" in lowered or "bottom" in lowered:
        aliases.extend(["back", "bottom"])

    result = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            result.append(alias)
            seen.add(alias)
    return result


def candidate_dirs(input_root: Path, side: str, suffix: str) -> list[Path]:
    candidates = []
    for alias in side_aliases(side):
        for root in batch_dirs(input_root):
            candidates.extend(
                [
                    root / f"{alias}_{suffix}",
                    root / alias / suffix,
                    root / alias,
                    root / "0" / f"{alias}_{suffix}",
                    root / "0" / alias / suffix,
                    root / "0" / alias,
                ]
            )
    return candidates


def pick_source_dir(input_root: Path, side: str, suffix: str) -> Path | None:
    for candidate in candidate_dirs(input_root, side, suffix):
        if has_images(candidate):
            return candidate.resolve()
    return None


def explicit_side_input(config: dict, side: str) -> tuple[Path, Path | None] | None:
    side_inputs = config.get("SIDE_INPUTS") or {}
    item = side_inputs.get(side)
    if not isinstance(item, dict):
        return None

    image_dir = item.get("images")
    if not image_dir:
        return None

    mask_dir = item.get("masks")
    image_path = Path(image_dir).resolve()
    mask_path = Path(mask_dir).resolve() if mask_dir else None
    return image_path, mask_path


def link_or_copy_dir(src: Path, dst: Path, desc: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if result.returncode == 0:
            log_ok(f"Linked {desc}: {dst} -> {src}")
            return
        log_warn(f"Junction failed for {desc}; copying instead: {result.stderr.strip()}")
    except Exception as exc:
        log_warn(f"Junction failed for {desc}; copying instead: {exc}")

    shutil.copytree(src, dst)
    log_ok(f"Copied {desc}: {src} -> {dst}")


def ensure_side_input(config: dict, side: str) -> Path:
    image_root = Path(config["image_root"])
    input_root = Path(config["input_root"])
    side_root = image_root / side

    explicit = explicit_side_input(config, side)
    if explicit:
        image_src, mask_src = explicit
        ensure_dir(image_src, f"{side} images")
        if not has_images(side_root / "images"):
            link_or_copy_dir(image_src, side_root / "images", f"{side} images")
        if mask_src:
            validate_mask_coverage(
                image_src,
                mask_src,
                side,
                float(config.get("MASK_MATCH_MIN_RATIO", 0.95)),
            )
            if not (side_root / "masks").exists():
                link_or_copy_dir(mask_src, side_root / "masks", f"{side} masks")
        return side_root

    if has_images(side_root / "images"):
        return side_root

    image_src = pick_source_dir(image_root, side, "images")
    mask_src = pick_source_dir(image_root, side, "masks")
    if image_src:
        link_or_copy_dir(image_src, side_root / "images", f"{side} images")
        if mask_src:
            link_or_copy_dir(mask_src, side_root / "masks", f"{side} masks")
        return side_root
    if not bool(config.get("AUTO_PREPARE_INPUT", True)):
        return side_root
    if not input_root.is_dir():
        return side_root

    image_src = pick_source_dir(input_root, side, "images")
    if image_src:
        link_or_copy_dir(image_src, side_root / "images", f"{side} images")

    mask_src = pick_source_dir(input_root, side, "masks")
    if mask_src:
        link_or_copy_dir(mask_src, side_root / "masks", f"{side} masks")

    return side_root


def run_cmd(cmd: list[str], desc: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    log_info(desc)
    log_info("CMD: " + " ".join(str(x) for x in cmd))
    start = time.perf_counter()
    log_dir = Path(load_config()["log_root"])
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", desc).strip("_").lower()
    log_path = log_dir / f"{safe_name}.log"

    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as log_file:
        log_file.write("CMD: " + " ".join(str(x) for x in cmd) + "\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd or SCRIPT_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        returncode = process.wait()

    result = subprocess.CompletedProcess(
        cmd,
        returncode,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}; see log: {log_path}")
    log_ok(f"{desc} finished in {elapsed:.1f}s")
    log_info(f"Log: {log_path}")
    return result


def side_is_complete(config: dict, side: str) -> bool:
    sparse_dir = Path(config["interim_pre"]) / side / "sfm" / "sparse" / "0"
    data_ply = Path(config["data_root"]) / side / f"{side}.ply"
    required = [
        data_ply,
        sparse_dir / "cameras.txt",
        sparse_dir / "images.txt",
        sparse_dir / "points3D.txt",
    ]
    return all(path.is_file() for path in required)


def clean_side_outputs(config: dict, side: str) -> None:
    targets = [
        Path(config["interim_pre"]) / side,
        Path(config["data_root"]) / side,
    ]
    for target in targets:
        if target.exists():
            try:
                shutil.rmtree(target)
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot clean {target}. Close Metashape or any program using files in this directory, "
                    "then run preprocess again."
                ) from exc


def metashape_args(config: dict, side: str) -> list[str]:
    workspace = Path(config["workspace"])
    explicit = explicit_side_input(config, side)
    if explicit:
        image_dir, mask_dir = explicit
        ensure_dir(image_dir, f"{side} images")
        if mask_dir:
            validate_mask_coverage(
                image_dir,
                mask_dir,
                side,
                float(config.get("MASK_MATCH_MIN_RATIO", 0.95)),
            )
        root = workspace
        images_arg = str(image_dir)
        masks_arg = str(mask_dir) if mask_dir else "masks"
    else:
        root = ensure_side_input(config, side)
        ensure_dir(root, f"{side} image root")
        ensure_dir(root / "images", f"{side} images")
        images_arg = "images"
        masks_arg = "masks"

    args = [
        str(config["METASHAPE_EXE"]),
        "-r",
        str(METASHAPE_SCRIPT),
        "--root",
        str(root),
        "--images",
        images_arg,
        "--masks",
        masks_arg,
        "--output",
        str(Path(config["interim_pre"]) / side),
        "--label",
        side,
        "--layout",
        str(config.get("METASHAPE_LAYOUT", "prefix-sensors")),
        "--match-downscale",
        str(int(config.get("METASHAPE_MATCH_DOWNSCALE", 1))),
        "--depth-downscale",
        str(int(config.get("METASHAPE_DEPTH_DOWNSCALE", 4))),
        "--depth-filter",
        str(config.get("METASHAPE_DEPTH_FILTER", "mild")),
        "--keypoint-limit",
        str(int(config.get("METASHAPE_KEYPOINT_LIMIT", 40000))),
        "--keypoint-limit-per-mpx",
        str(int(config.get("METASHAPE_KEYPOINT_LIMIT_PER_MPX", 1000))),
        "--tiepoint-limit",
        str(int(config.get("METASHAPE_TIEPOINT_LIMIT", 4000))),
        "--max-neighbors",
        str(int(config.get("METASHAPE_MAX_NEIGHBORS", 16))),
        "--point-cloud-max-neighbors",
        str(int(config.get("METASHAPE_POINT_CLOUD_MAX_NEIGHBORS", 100))),
        "--direct-legacy-output",
        "--data-root",
        str(Path(config["data_root"])),
    ]

    if not bool(config.get("METASHAPE_BUILD_MODEL", False)):
        args.append("--no-model")
    if bool(config.get("METASHAPE_USE_MASKS", False)):
        args.append("--use-masks-for-metashape")
        args.extend(["--mask-mode", str(config.get("METASHAPE_MASK_MODE", "white-valid"))])
        args.extend(["--mask-match-min-ratio", str(float(config.get("MASK_MATCH_MIN_RATIO", 0.95)))])
        if bool(config.get("METASHAPE_MASK_MATCHING", False)):
            args.append("--mask-matching")
        if bool(config.get("METASHAPE_MASK_TIEPOINTS", False)):
            args.append("--mask-tiepoints")
    if bool(config.get("METASHAPE_REUSE", False)):
        args.append("--reuse")

    return args


def process_side(config: dict, side: str) -> dict:
    banner(f"Metashape preprocess - {side.upper()}")
    metashape_exe = Path(str(config["METASHAPE_EXE"]))
    ensure_file(metashape_exe, "Metashape executable")
    ensure_file(METASHAPE_SCRIPT, "Metashape reconstruction script")

    if bool(config.get("PREPROCESS_REUSE_DENSE", False)) and side_is_complete(config, side):
        log_ok(f"{side}: existing Metashape outputs reused")
        return {"side": side, "status": "reused", "elapsed": 0.0}

    if not bool(config.get("METASHAPE_REUSE", False)):
        clean_side_outputs(config, side)

    start = time.perf_counter()
    run_cmd(metashape_args(config, side), f"Metashape reconstruction - {side}")
    elapsed = time.perf_counter() - start

    data_ply = Path(config["data_root"]) / side / f"{side}.ply"
    sparse_dir = Path(config["interim_pre"]) / side / "sfm" / "sparse" / "0"
    ensure_file(data_ply, f"{side} dense point cloud")
    for name in ["cameras.txt", "images.txt", "points3D.txt"]:
        ensure_file(sparse_dir / name, f"{side} {name}")

    log_ok(f"{side}: preprocess output ready ({elapsed:.1f}s)")
    return {"side": side, "status": "ok", "elapsed": elapsed}


def main() -> None:
    config = load_config()
    sides = config.get("sides") or []
    if not sides:
        raise RuntimeError("No sides configured in config.yaml")

    banner("Metashape preprocess")
    log_info(f"Workspace: {config['workspace']}")
    log_info(f"Batch: {config.get('batch_name') or '(none)'}")
    log_info(f"Output root: {config['output_root']}")
    log_info(f"Image root: {config['image_root']}")
    log_info(f"Input root: {config['input_root']}")
    log_info(f"Sides: {', '.join(sides)}")
    log_info(f"Metashape: {config['METASHAPE_EXE']}")
    log_info(f"Depth downscale: {config.get('METASHAPE_DEPTH_DOWNSCALE', 4)}")
    log_info(f"Build model: {bool(config.get('METASHAPE_BUILD_MODEL', False))}")
    log_info(f"Use masks inside Metashape: {bool(config.get('METASHAPE_USE_MASKS', False))}")
    log_info(f"Use masks during matching: {bool(config.get('METASHAPE_MASK_MATCHING', False))}")

    start = time.perf_counter()
    results = []

    if bool(config.get("PREPROCESS_PARALLEL_SIDES", False)) and len(sides) > 1:
        workers = max(1, min(int(config.get("PREPROCESS_PARALLEL_WORKERS", 2)), len(sides)))
        log_warn(f"Parallel sides enabled, workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_side, config, side): side for side in sides}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for side in sides:
            results.append(process_side(config, side))

    total = time.perf_counter() - start
    banner("Metashape preprocess summary")
    for item in results:
        print(f"  {item['side']}: {item['status']} ({item['elapsed']:.1f}s)")
    print(f"  total: {total:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise
