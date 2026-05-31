"""
直接运行：
python images/auto_label.py --save-white-bg(不要加，除非你了解它的作用)
如果以后只想单独跑某个文件夹，也还能这样用：

python images/auto_label.py images/batch3/top_images --output images/batch3/top_masks
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import Sam3Model, Sam3Processor

MODEL_PATH = r"D:\mwk\sam3"
# 只修改下面的就可以
BATCH_DIR = Path(r"C:\Users\57746\Desktop\newpipeline\images\batch3")



SIDE_DIRS = {
    "top": (BATCH_DIR / "top_images", BATCH_DIR / "top_masks"),
    "bottom": (BATCH_DIR / "bottom_images", BATCH_DIR / "bottom_masks"),
}
INPUT_DIR = str(SIDE_DIRS["top"][0])
OUTPUT_DIR = str(SIDE_DIRS["top"][1])

# Keep this as a target noun phrase. Long negative instructions are less stable
# for text-prompted segmentation than a concise description of the object class.
TEXT_PROMPT = "broken pottery shards"
TOP_K = 12
THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
RANK_BY = "area"
MIN_AREA = 500
MIN_MASK_RATIO = 0.005
MAX_MASK_RATIO = 0.5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate white-foreground / black-background pottery shard masks."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="",
        help="Optional image file or folder. Empty processes both top_images and bottom_images.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output mask folder. Required only when processing a custom input.",
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Local SAM3 model path.")
    parser.add_argument("--prompt", default=TEXT_PROMPT, help="Text prompt for segmentation.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Maximum instances to merge per image.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--rank-by", choices=["area", "score"], default=RANK_BY)
    parser.add_argument(
        "--save-white-bg",
        action="store_true",
        help="Also save preview images with the background replaced by white.",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_paths(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def mask_area(mask_tensor):
    return int(mask_tensor.sum().item())


def to_white_fg_black_bg_mask(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy()
    binary_mask = (mask_np > 0).astype(np.uint8)
    return binary_mask * 255


def rank_instances(masks, scores, rank_by):
    items = []
    for i in range(len(masks)):
        score = float(scores[i].item()) if hasattr(scores[i], "item") else float(scores[i])
        area = mask_area(masks[i])
        items.append((i, score, area))

    if rank_by == "score":
        items.sort(key=lambda x: x[1], reverse=True)
    else:
        items.sort(key=lambda x: x[2], reverse=True)

    return [x[0] for x in items]


def replace_background_with_white(image, mask_array):
    img_array = np.array(image)
    white_bg = np.ones_like(img_array) * 255
    mask_normalized = (mask_array / 255.0)[:, :, np.newaxis]
    result = img_array * mask_normalized + white_bg * (1 - mask_normalized)
    return Image.fromarray(result.astype(np.uint8))


def process_one_image(image_path, model, processor, device, args):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=[(h, w)],
    )[0]

    masks = results.get("masks", [])
    scores = results.get("scores", [])
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if masks is None or len(masks) == 0:
        return combined_mask, 0, 0, image

    valid_indices = [i for i in range(len(masks)) if mask_area(masks[i]) >= args.min_area]
    if not valid_indices:
        return combined_mask, len(masks), 0, image

    filtered_masks = [masks[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    selected_indices = rank_instances(filtered_masks, filtered_scores, args.rank_by)[: args.top_k]

    for idx in selected_indices:
        mask_img = to_white_fg_black_bg_mask(filtered_masks[idx])
        combined_mask = np.maximum(combined_mask, mask_img)

    white_bg_image = replace_background_with_white(image, combined_mask)
    return combined_mask, len(masks), len(selected_indices), white_bg_image


def mask_ratio(mask_array):
    return float((mask_array > 127).mean())


def default_jobs():
    return [(side, input_dir, output_dir) for side, (input_dir, output_dir) in SIDE_DIRS.items()]


def resolve_jobs(args):
    if args.input:
        input_path = Path(args.input)
        output_dir = Path(args.output) if args.output else Path(OUTPUT_DIR)
        return [(input_path.stem if input_path.is_file() else input_path.name, input_path, output_dir)]
    return default_jobs()


def process_folder(label, input_path, output_dir, model, processor, device, args):
    ensure_dir(output_dir)
    image_paths = load_image_paths(input_path)
    if not image_paths:
        print(f"No images found: {input_path}")
        return []

    preview_dir = output_dir / "white_bg_preview"
    if args.save_white_bg:
        ensure_dir(preview_dir)

    print(f"\n=== {label} ===")
    print(f"Input: {input_path}")
    print(f"Output masks: {output_dir}")
    print(f"Found {len(image_paths)} image(s).")

    warnings = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {image_path.name}")
        try:
            combined_mask, found_num, kept_num, white_bg_image = process_one_image(
                image_path=image_path,
                model=model,
                processor=processor,
                device=device,
                args=args,
            )

            output_path = output_dir / f"{image_path.stem}_mask.png"
            Image.fromarray(combined_mask).save(output_path)

            ratio = mask_ratio(combined_mask)
            status = "OK"
            if ratio < MIN_MASK_RATIO or ratio > MAX_MASK_RATIO:
                status = "WARN"
                warnings.append((image_path.name, ratio))

            if args.save_white_bg:
                white_bg_image.save(preview_dir / f"{image_path.stem}_white_bg.jpg", quality=95)

            print(
                f"  found={found_num}, kept={kept_num}, mask_ratio={ratio:.4f}, "
                f"status={status}, saved={output_path}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            warnings.append((image_path.name, None))

    if warnings:
        print(f"\nWarnings for {label}:")
        for name, ratio in warnings:
            if ratio is None:
                print(f"  {name}: failed")
            else:
                print(f"  {name}: mask_ratio={ratio:.4f}")

    return warnings


def main():
    args = parse_args()
    jobs = resolve_jobs(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model}")
    model = Sam3Model.from_pretrained(args.model).to(device)
    processor = Sam3Processor.from_pretrained(args.model)
    model.eval()

    all_warnings = []
    for label, input_path, output_dir in jobs:
        warnings = process_folder(label, Path(input_path), Path(output_dir), model, processor, device, args)
        all_warnings.extend((label, name, ratio) for name, ratio in warnings)

    print(f"\nDone. Processed {len(jobs)} job(s).")
    if all_warnings:
        print("Warnings summary:")
        for label, name, ratio in all_warnings:
            value = "failed" if ratio is None else f"mask_ratio={ratio:.4f}"
            print(f"  {label}/{name}: {value}")


if __name__ == "__main__":
    main()
