import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import Metashape


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def log(message):
    print(f"[metashape_reconstruct] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align photos and reconstruct point cloud/model with Agisoft Metashape."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Dataset root containing images/ and optional masks/.",
    )
    parser.add_argument("--images", default="images", help="Image directory, relative to root or absolute.")
    parser.add_argument("--masks", default="masks", help="Mask directory, relative to root or absolute.")
    parser.add_argument("--output", default="output", help="Output directory, relative to root or absolute.")
    parser.add_argument("--label", default="re_con", help="Dataset label used for legacy-style exports.")
    parser.add_argument(
        "--direct-legacy-output",
        action="store_true",
        help=(
            "Write sfm/ and dense/ directly under --output, matching interim/preprocess/<side>. "
            "By default they are written under --output/legacy/<label> for safer testing."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Directory for data/<label>/<label>.ply. Defaults to --output/data for safe testing; "
            "use the workspace data directory when replacing preprocess."
        ),
    )
    parser.add_argument("--project-name", default="metashape_reconstruction.psx")
    parser.add_argument("--match-downscale", type=int, default=1, help="0 highest, 1 high, 2 medium, 4 low.")
    parser.add_argument("--depth-downscale", type=int, default=1, help="1 ultra, 2 high, 4 medium, 8 low.")
    parser.add_argument(
        "--depth-filter",
        choices=["none", "mild", "moderate", "aggressive"],
        default="moderate",
        help="Depth map filtering strength.",
    )
    parser.add_argument("--keypoint-limit", type=int, default=40000)
    parser.add_argument("--keypoint-limit-per-mpx", type=int, default=1000)
    parser.add_argument("--tiepoint-limit", type=int, default=4000)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--point-cloud-max-neighbors", type=int, default=100)
    parser.add_argument(
        "--layout",
        choices=["auto", "flat", "prefix-sensors", "multiplane"],
        default="auto",
        help=(
            "Photo layout. auto uses prefix-sensors when names look like "
            "0_IMG_0001.JPG, 1_IMG_0001.JPG, ..."
        ),
    )
    parser.add_argument("--point-spacing", type=float, default=0.1, help="Used only with --uniform-sampling.")
    parser.add_argument(
        "--uniform-sampling",
        action="store_true",
        help="Use Metashape uniform point sampling. By default this is off for denser object point clouds.",
    )
    parser.add_argument("--no-model", action="store_true", help="Skip mesh model generation.")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing project if present.")
    parser.add_argument("--no-masks", action="store_true", help="Ignore masks even if masks/ exists.")
    parser.add_argument(
        "--mask-mode",
        choices=["white-valid", "metashape"],
        default="white-valid",
        help=(
            "Mask convention for input files. metashape passes masks through unchanged. "
            "white-valid inverts masks before assigning to Metashape."
        ),
    )
    parser.add_argument(
        "--use-masks-for-metashape",
        action="store_true",
        help=(
            "Load masks into Metashape for depth/model generation. Masks are always copied "
            "to the legacy output when present, even when this is disabled."
        ),
    )
    parser.add_argument(
        "--mask-matching",
        action="store_true",
        help=(
            "Use loaded masks during key point detection. "
            "This is off by default because small object masks can break alignment."
        ),
    )
    parser.add_argument(
        "--mask-tiepoints",
        action="store_true",
        help=(
            "Also apply mask filtering to tie points. This is stricter than key point "
            "masking and can remove all tracks when masks are too tight or inverted."
        ),
    )
    parser.add_argument(
        "--mask-match-min-ratio",
        type=float,
        default=0.95,
        help="Minimum mask coverage required when --use-masks-for-metashape is enabled.",
    )
    return parser.parse_args()


def resolve_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def depth_filter_mode(name):
    modes = {
        "none": Metashape.FilterMode.NoFiltering,
        "mild": Metashape.FilterMode.MildFiltering,
        "moderate": Metashape.FilterMode.ModerateFiltering,
        "aggressive": Metashape.FilterMode.AggressiveFiltering,
    }
    return modes[name]


def list_images(image_dir):
    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    images.sort(key=lambda p: p.name.lower())
    if not images:
        raise RuntimeError(f"No images found: {image_dir}")
    return images


def camera_group_from_name(image_path):
    match = re.match(r"^([0-9]+)_(.+)$", image_path.name)
    if match:
        return f"cam{match.group(1)}", match.group(2)
    return "cam0", image_path.name


def detect_multiplane_groups(images):
    grouped = {}
    camera_ids = set()
    for image in images:
        match = re.match(r"^([0-9]+)_(.+)$", image.name)
        if not match:
            return None
        camera_id, frame_name = match.groups()
        grouped.setdefault(frame_name, {})[camera_id] = image
        camera_ids.add(camera_id)

    def sort_key(value):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

    def camera_sort_key(value):
        return int(value) if value.isdigit() else value

    camera_order = sorted(camera_ids, key=camera_sort_key)
    frame_order = sorted(grouped, key=sort_key)
    if len(camera_order) < 2 or not frame_order:
        return None

    for frame_name in frame_order:
        if sorted(grouped[frame_name], key=camera_sort_key) != camera_order:
            return None

    ordered_images = []
    filegroups = []
    for frame_name in frame_order:
        for camera_id in camera_order:
            ordered_images.append(grouped[frame_name][camera_id])
        filegroups.append(len(camera_order))

    return {
        "camera_order": camera_order,
        "frame_order": frame_order,
        "ordered_images": ordered_images,
        "filegroups": filegroups,
    }


def add_photos(chunk, images, layout):
    multiplane = detect_multiplane_groups(images)
    if layout == "auto":
        layout = "prefix-sensors" if multiplane else "flat"

    if layout == "multiplane":
        if not multiplane:
            raise RuntimeError("Requested multiplane layout, but image names are not complete prefix groups.")
        log(
            "Adding photos as multiplane layout: "
            f"{len(multiplane['frame_order'])} stations x {len(multiplane['camera_order'])} cameras "
            f"(prefixes: {', '.join(multiplane['camera_order'])})"
        )
        chunk.addPhotos(
            filenames=[str(path) for path in multiplane["ordered_images"]],
            filegroups=multiplane["filegroups"],
            layout=Metashape.ImageLayout.MultiplaneLayout,
            strip_extensions=False,
        )
        return multiplane["ordered_images"]

    log("Adding photos as flat layout")
    chunk.addPhotos([str(p) for p in images], strip_extensions=False)
    if layout == "prefix-sensors":
        assign_prefix_sensors(chunk)
    return images


def assign_prefix_sensors(chunk):
    cameras = list(chunk.cameras)
    if not cameras:
        return

    original_sensor = cameras[0].sensor
    sensors_by_group = {}
    assigned = 0

    for camera in cameras:
        name = camera_name_keys(camera)[0] if camera_name_keys(camera) else camera.label
        match = re.match(r"^([0-9]+)_", Path(name).name)
        if not match:
            continue

        group = match.group(1)
        if group not in sensors_by_group:
            sensor = chunk.addSensor()
            sensor.label = f"cam{group}"
            sensor.type = original_sensor.type
            sensor.width = original_sensor.width
            sensor.height = original_sensor.height
            sensor.user_calib = original_sensor.calibration.copy()
            sensors_by_group[group] = sensor

        camera.sensor = sensors_by_group[group]
        assigned += 1

    log(f"Assigned prefix sensors: {len(sensors_by_group)} sensors for {assigned}/{len(cameras)} cameras")


def mask_for_image(mask_dir, image_path):
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


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_image_manifest(images, sparse_dir):
    manifest = sparse_dir.parent.parent / "images_for_pipeline.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("src_rel\tpipeline_rel\tcamera_group\tbase_name\n")
        for image in images:
            camera_group, base_name = camera_group_from_name(image)
            f.write(f"{image.name}\t{image.name}\t{camera_group}\t{base_name}\n")
    return manifest


def copy_masks_for_pipeline(images, mask_dir, sparse_root):
    pipeline_mask_dir = sparse_root.parent / "masks_pipeline"
    ensure_clean_dir(pipeline_mask_dir)
    matched = 0
    for image in images:
        mask = mask_for_image(mask_dir, image)
        if not mask:
            continue
        shutil.copy2(mask, pipeline_mask_dir / f"{image.name}.png")
        matched += 1
    return pipeline_mask_dir, matched


def add_masks(chunk, images, mask_dir, mask_mode):
    matched = 0
    by_label = {}
    for camera in iter_chunk_cameras(chunk):
        for key in camera_name_keys(camera):
            by_label.setdefault(key, camera)

    for image in images:
        camera = by_label.get(image.stem) or by_label.get(image.name)
        if camera is None:
            continue
        mask_path = mask_for_image(mask_dir, image)
        if not mask_path:
            continue
        mask = Metashape.Mask()
        mask.load(str(mask_path))
        if mask_mode == "white-valid":
            inverted = mask.invert()
            if inverted is not None:
                mask = inverted
        camera.mask = mask
        matched += 1
    return matched


def iter_chunk_cameras(chunk):
    seen = set()

    def visit(camera):
        key = getattr(camera, "key", id(camera))
        if key in seen:
            return
        seen.add(key)
        yield camera
        for plane in getattr(camera, "planes", []) or []:
            yield from visit(plane)
        for frame in getattr(camera, "frames", []) or []:
            yield from visit(frame)

    for camera in chunk.cameras:
        yield from visit(camera)


def camera_name_keys(camera):
    keys = []
    label = getattr(camera, "label", None)
    if label:
        keys.append(label)
        keys.append(Path(label).stem)

    photo = getattr(camera, "photo", None)
    path = getattr(photo, "path", None) if photo else None
    if path:
        name = Path(path).name
        keys.append(name)
        keys.append(Path(name).stem)

    return [key for key in keys if key]


def export_camera_text_model(chunk, sparse_dir):
    sparse_dir.mkdir(parents=True, exist_ok=True)
    camera_export_path = sparse_dir / "cameras.txt"
    chunk.exportCameras(
        path=str(camera_export_path),
        # Metashape exposes the required text camera export through this API enum.
        format=Metashape.CamerasFormat.CamerasFormatColmap,
        save_points=True,
        save_images=False,
        save_masks=False,
        binary=False,
    )

    required = ["cameras.txt", "images.txt", "points3D.txt"]
    nested_dir = sparse_dir / "sparse" / "0"
    if nested_dir.is_dir() and all((nested_dir / name).is_file() for name in required):
        for name in required:
            shutil.copy2(nested_dir / name, sparse_dir / name)

    missing = [name for name in required if not (sparse_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Camera text export missing files in {sparse_dir}: {missing}")


def normalize_camera_text_for_pipeline(sparse_dir, images):
    """
    Metashape 2.2.x may export FULL_OPENCV cameras and generated image names
    such as image_0.jpg. The downstream pipeline expects OPENCV and original names.
    """
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"

    normalized_camera_lines = []
    with cameras_txt.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                normalized_camera_lines.append(line)
                continue

            parts = line.strip().split()
            if len(parts) >= 16 and parts[1] == "FULL_OPENCV":
                camera_id, _, width, height = parts[:4]
                opencv_params = parts[4:12]
                normalized_camera_lines.append(
                    " ".join([camera_id, "OPENCV", width, height, *opencv_params]) + "\n"
                )
            else:
                normalized_camera_lines.append(line)

    with cameras_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(normalized_camera_lines)

    image_names = [p.name for p in images]
    image_by_stem = {p.stem.lower(): p.name for p in images}

    def normalize_export_name(name, fallback_index):
        stem = Path(name).stem.lower()
        if stem in image_by_stem:
            return image_by_stem[stem]
        # Metashape 2.2.x often exports "1_IMG_0001_0.jpg".
        stem_without_suffix = re.sub(r"_[0-9]+$", "", stem)
        if stem_without_suffix in image_by_stem:
            return image_by_stem[stem_without_suffix]
        if fallback_index < len(image_names):
            return image_names[fallback_index]
        return name

    data_index = 0
    normalized_image_lines = []
    with images_txt.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                normalized_image_lines.append(line)
                continue

            # In this camera text format every image record is followed by one POINTS2D line.
            if data_index % 2 == 0:
                image_record_index = data_index // 2
                parts = line.rstrip("\n").split()
                if len(parts) >= 10 and image_record_index < len(image_names):
                    parts[9] = normalize_export_name(parts[9], image_record_index)
                    line = " ".join(parts) + "\n"

            normalized_image_lines.append(line)
            data_index += 1

    with images_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(normalized_image_lines)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    image_dir = resolve_path(root, args.images)
    mask_dir = resolve_path(root, args.masks)
    out_dir = resolve_path(root, args.output)

    images = list_images(image_dir)
    processing_images = images
    masks_present = mask_dir.is_dir() and not args.no_masks
    masks_available = masks_present and args.use_masks_for_metashape

    project_dir = out_dir / "project"
    legacy_root = out_dir if args.direct_legacy_output else out_dir / "legacy" / args.label
    sparse_dir = legacy_root / "sfm" / "sparse" / "0"
    dense_dir = legacy_root / "dense"
    model_dir = out_dir / "model"
    data_root = resolve_path(root, args.data_root) if args.data_root else out_dir / "data"
    data_dir = data_root / args.label

    if out_dir.exists() and not args.reuse:
        shutil.rmtree(out_dir)
    for directory in [project_dir, sparse_dir, dense_dir, model_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    project_path = project_dir / args.project_name
    log(f"Metashape version: {Metashape.app.version}")
    log(f"Root: {root}")
    log(f"Images: {len(images)} from {image_dir}")
    log(f"Legacy output: {legacy_root}")
    log(f"Data output: {data_dir}")
    log(
        "Quality: "
        f"match downscale={args.match_downscale}, "
        f"depth downscale={args.depth_downscale}, "
        f"depth filter={args.depth_filter}, "
        f"depth max neighbors={args.max_neighbors}, "
        f"point cloud max neighbors={args.point_cloud_max_neighbors}"
    )
    if masks_available:
        log(f"Masks: enabled ({args.mask_mode} mode)")
        if args.mask_mode == "white-valid":
            log("Input masks are inverted before assigning to Metashape")
        else:
            log("Input masks are assigned to Metashape without inversion")
        log(f"Use masks during matching: {args.mask_matching}")
    else:
        if masks_present:
            log("Masks: copied for legacy output, not used by Metashape")
        else:
            log("Masks: disabled")
    log(f"Output: {out_dir}")

    doc = Metashape.Document()
    if args.reuse and project_path.is_file():
        log(f"Opening existing project: {project_path}")
        doc.open(str(project_path))
        chunk = doc.chunk
        if chunk is None:
            raise RuntimeError(f"Project has no active chunk: {project_path}")
    else:
        chunk = doc.addChunk()
        chunk.label = args.label
        processing_images = add_photos(chunk, images, args.layout)
        doc.save(str(project_path))

    if masks_available:
        matched_masks = add_masks(chunk, processing_images, mask_dir, args.mask_mode)
        log(f"Loaded masks: {matched_masks}/{len(processing_images)}")
        mask_ratio = matched_masks / len(processing_images) if processing_images else 0.0
        if mask_ratio < args.mask_match_min_ratio:
            raise RuntimeError(
                f"Mask coverage too low: {matched_masks}/{len(processing_images)} "
                f"matched, required >= {args.mask_match_min_ratio:.0%}"
            )

    log("Matching photos")
    use_masks_for_matching = masks_available and args.mask_matching
    use_masks_for_tiepoints = masks_available and args.mask_tiepoints
    chunk.matchPhotos(
        downscale=args.match_downscale,
        generic_preselection=True,
        reference_preselection=False,
        filter_mask=use_masks_for_matching,
        mask_tiepoints=use_masks_for_tiepoints,
        keypoint_limit=args.keypoint_limit,
        keypoint_limit_per_mpx=args.keypoint_limit_per_mpx,
        tiepoint_limit=args.tiepoint_limit,
        guided_matching=True,
        reset_matches=True,
    )

    log("Aligning cameras")
    chunk.alignCameras(reset_alignment=True, adaptive_fitting=False)
    aligned = [camera for camera in chunk.cameras if camera.transform]
    log(f"Aligned cameras: {len(aligned)}/{len(chunk.cameras)}")
    if not aligned:
        raise RuntimeError("No cameras aligned; cannot reconstruct.")

    log("Exporting camera text model")
    export_camera_text_model(chunk, sparse_dir)
    normalize_camera_text_for_pipeline(sparse_dir, processing_images)
    manifest = save_image_manifest(processing_images, sparse_dir)
    log(f"Wrote image manifest: {manifest}")
    if masks_present:
        pipeline_mask_dir, matched = copy_masks_for_pipeline(processing_images, mask_dir, sparse_dir.parent)
        log(f"Wrote pipeline masks: {matched}/{len(processing_images)} -> {pipeline_mask_dir}")

    doc.save()

    log("Building depth maps")
    chunk.buildDepthMaps(
        downscale=args.depth_downscale,
        filter_mode=depth_filter_mode(args.depth_filter),
        max_neighbors=args.max_neighbors,
        reuse_depth=False,
    )
    doc.save()

    log("Building dense point cloud")
    try:
        chunk.buildPointCloud(
            source_data=Metashape.DataSource.DepthMapsData,
            point_colors=True,
            point_confidence=True,
            keep_depth=True,
            max_neighbors=args.point_cloud_max_neighbors,
            uniform_sampling=args.uniform_sampling,
            points_spacing=args.point_spacing,
            replace_asset=True,
        )
    except Exception as exc:
        if "Zero resolution" in str(exc):
            raise RuntimeError(
                "Dense point cloud failed because no usable depth maps were selected. "
                "Alignment likely produced cameras without valid neighbors; try disabling "
                "--mask-matching or using a less restrictive mask/matching setup."
            ) from exc
        raise
    doc.save()

    dense_ply = dense_dir / "scene_dense.ply"
    data_ply = data_dir / f"{args.label}.ply"
    log(f"Exporting dense point cloud: {dense_ply}")
    chunk.exportPointCloud(
        path=str(dense_ply),
        source_data=Metashape.DataSource.PointCloudData,
        binary=True,
        save_point_color=True,
        save_point_normal=True,
        save_point_confidence=True,
        format=Metashape.PointCloudFormat.PointCloudFormatPLY,
        clip_to_boundary=False,
        clip_to_region=False,
    )
    shutil.copy2(dense_ply, data_ply)
    log(f"Wrote legacy data point cloud: {data_ply}")

    if not args.no_model:
        model_ply = model_dir / f"{args.label}_model.ply"
        log("Building mesh model")
        chunk.buildModel(
            surface_type=Metashape.SurfaceType.Arbitrary,
            interpolation=Metashape.Interpolation.EnabledInterpolation,
            face_count=Metashape.FaceCount.MediumFaceCount,
            source_data=Metashape.DataSource.DepthMapsData,
            vertex_colors=True,
            vertex_confidence=False,
            volumetric_masks=masks_available,
            keep_depth=True,
            replace_asset=True,
            build_texture=False,
        )
        doc.save()
        log(f"Exporting model: {model_ply}")
        chunk.exportModel(
            path=str(model_ply),
            binary=True,
            save_texture=False,
            save_uv=False,
            save_normals=True,
            save_colors=True,
            format=Metashape.ModelFormat.ModelFormatPLY,
            clip_to_boundary=False,
            clip_to_region=False,
        )

    doc.save()
    log("Done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise
