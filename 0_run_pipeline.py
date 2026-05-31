#!/usr/bin/env python3
"""
你现在可以这样批量跑：
C:/Users/57746/anaconda3/envs/mapany/python.exe 0_run_pipeline.py --batches 5-22
也可以只跑几个：
C:/Users/57746/anaconda3/envs/mapany/python.exe 0_run_pipeline.py --batches 5,6,8
只跑单个 batch：
C:/Users/57746/anaconda3/envs/mapany/python.exe 0_run_pipeline.py --batch 5
也可以结合阶段参数，比如只跑预处理到分割：
C:/Users/57746/anaconda3/envs/mapany/python.exe 0_run_pipeline.py --batches 5-22 --start-at preprocess --stop-after split
或者只跑第五步：
C:/Users/57746/anaconda3/envs/mapany/python.exe 0_run_pipeline.py --batches 5-22 --only mesh



Run the complete ceramic fragment pipeline and record per-stage timing.

Stages:
  1_preprocess.py
  2_euclidean_split.py
  3_match_fragments.py
  4_register_front_back_teaser.py
  5_pointcloud_to_mesh.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


STAGES = [
    ("preprocess", "1_preprocess.py"),
    ("split", "2_euclidean_split.py"),
    ("match", "3_match_fragments.py"),
    ("register", "4_register_front_back_teaser.py"),
    ("mesh", "5_pointcloud_to_mesh.py"),
]


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


def apply_batch_override(config: dict, batch_number: int | str | None) -> None:
    if batch_number in (None, ""):
        env_batch = os.environ.get("PIPELINE_BATCH_NUMBER", "")
        batch_number = env_batch if env_batch else None
    if batch_number in (None, ""):
        return
    config["batch_number"] = int(batch_number)
    config["batch_num"] = int(batch_number)
    config.pop("OUTPUT_BATCH", None)
    config.pop("batch_name", None)


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


def load_config(batch_number: int | str | None = None) -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    apply_batch_override(config, batch_number)
    expand_config_templates(config)

    workspace = Path(config.get("workspace") or SCRIPT_DIR).resolve()
    batch_name = detect_batch_name(config)
    output_root = workspace / "output" / batch_name if batch_name else workspace / "output"
    config["workspace"] = str(workspace)
    config["batch_name"] = batch_name
    config["output_root"] = str(output_root)
    return config


def parse_batch_list(value: str) -> list[int]:
    result: list[int] = []
    seen = set()
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())
            step = 1 if start <= stop else -1
            values = range(start, stop + step, step)
        else:
            values = [int(part)]
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
    if not result:
        raise ValueError("No valid batch numbers were provided.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all pipeline stages with timing.")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Run one batch without editing config.yaml, for example --batch 5.",
    )
    parser.add_argument(
        "--batches",
        default="",
        help="Run multiple batches, for example 5,6,8 or 5-22.",
    )
    parser.add_argument(
        "--start-at",
        choices=[name for name, _ in STAGES],
        default="preprocess",
        help="First stage to run.",
    )
    parser.add_argument(
        "--stop-after",
        choices=[name for name, _ in STAGES],
        default="mesh",
        help="Last stage to run.",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _ in STAGES],
        default="",
        help="Run only one stage.",
    )
    parser.add_argument(
        "--python",
        default="",
        help="Python executable for stages 2-5. Defaults to PIPELINE_PYTHON from config, then current Python.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later stages after a failure. Normally disabled.",
    )
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.only:
        return [stage for stage in STAGES if stage[0] == args.only]

    names = [name for name, _ in STAGES]
    start = names.index(args.start_at)
    stop = names.index(args.stop_after)
    if start > stop:
        raise ValueError("--start-at must be before or equal to --stop-after")
    return STAGES[start : stop + 1]


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_stage(
    stage_name: str,
    script_name: str,
    python_exe: str,
    log_file,
    batch_number: int | None = None,
) -> dict:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing stage script: {script_path}")

    cmd = [python_exe, str(script_path)]
    print(f"\n===== {stage_name.upper()} START =====", flush=True)
    print("CMD: " + " ".join(cmd), flush=True)
    log_file.write(f"\n===== {stage_name.upper()} START =====\n")
    log_file.write("CMD: " + " ".join(cmd) + "\n")
    log_file.flush()

    start_wall = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    env = os.environ.copy()
    if batch_number is not None:
        env["PIPELINE_BATCH_NUMBER"] = str(batch_number)

    process = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        env=env,
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
    elapsed = time.perf_counter() - start
    end_wall = datetime.now().isoformat(timespec="seconds")

    status = "ok" if returncode == 0 else "failed"
    print(
        f"===== {stage_name.upper()} {status.upper()} "
        f"elapsed={format_seconds(elapsed)} returncode={returncode} =====",
        flush=True,
    )
    log_file.write(
        f"===== {stage_name.upper()} {status.upper()} "
        f"elapsed={format_seconds(elapsed)} returncode={returncode} =====\n"
    )
    log_file.flush()

    return {
        "stage": stage_name,
        "script": script_name,
        "command": cmd,
        "status": status,
        "returncode": returncode,
        "start_time": start_wall,
        "end_time": end_wall,
        "elapsed_seconds": elapsed,
        "elapsed": format_seconds(elapsed),
    }


def run_one_batch(args: argparse.Namespace, batch_number: int | None = None) -> tuple[int, dict]:
    config = load_config(batch_number)
    workspace = Path(config["workspace"])
    output_root = Path(config["output_root"])
    log_root = output_root / "pipeline_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    python_exe = args.python or str(config.get("PIPELINE_PYTHON") or sys.executable)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_root / f"pipeline_{run_id}.log"

    stages = selected_stages(args)
    start = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    results = []

    print("Pipeline workspace:", workspace)
    print("Pipeline batch:", config.get("batch_name") or "(none)")
    print("Pipeline output:", output_root)
    print("Pipeline python:", python_exe)
    print("Pipeline log:", log_path)

    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as log_file:
        log_file.write(f"workspace: {workspace}\n")
        log_file.write(f"batch: {config.get('batch_name') or '(none)'}\n")
        log_file.write(f"output_root: {output_root}\n")
        log_file.write(f"python: {python_exe}\n")

        for stage_name, script_name in stages:
            result = run_stage(stage_name, script_name, python_exe, log_file, batch_number=batch_number)
            results.append(result)
            if result["returncode"] != 0 and not args.continue_on_error:
                break

    total_elapsed = time.perf_counter() - start
    summary = {
        "workspace": str(workspace),
        "batch_name": config.get("batch_name"),
        "batch_number": config.get("batch_number"),
        "output_root": str(output_root),
        "pipeline_log": str(log_path),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed": format_seconds(total_elapsed),
        "stages": results,
    }
    write_timing(output_root, run_id, summary)

    print("\n===== PIPELINE SUMMARY =====")
    for item in results:
        print(f"{item['stage']:<12} {item['status']:<8} {item['elapsed']}")
    print(f"{'total':<12} {'':<8} {format_seconds(total_elapsed)}")

    return (0 if all(item["returncode"] == 0 for item in results) else 1), summary


def write_timing(output_root: Path, run_id: str, summary: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "pipeline_timing.json"
    run_json_path = output_root / f"pipeline_timing_{run_id}.json"
    csv_path = output_root / "pipeline_timing.csv"

    for path in (json_path, run_json_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "status",
                "returncode",
                "elapsed_seconds",
                "elapsed",
                "start_time",
                "end_time",
                "script",
            ],
        )
        writer.writeheader()
        for item in summary["stages"]:
            writer.writerow({
                "stage": item["stage"],
                "status": item["status"],
                "returncode": item["returncode"],
                "elapsed_seconds": f"{item['elapsed_seconds']:.3f}",
                "elapsed": item["elapsed"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "script": item["script"],
            })

    print(f"[OK] timing json: {json_path}")
    print(f"[OK] timing csv: {csv_path}")


def main() -> int:
    args = parse_args()
    if args.batch is not None and args.batches:
        raise ValueError("Use either --batch or --batches, not both.")

    if args.batches:
        batch_numbers = parse_batch_list(args.batches)
    elif args.batch is not None:
        batch_numbers = [args.batch]
    else:
        batch_numbers = [None]

    exit_codes = []
    summaries = []
    for batch_number in batch_numbers:
        if batch_number is not None:
            print(f"\n######## BATCH {batch_number} ########", flush=True)
        code, summary = run_one_batch(args, batch_number=batch_number)
        exit_codes.append(code)
        summaries.append(summary)
        if code != 0 and not args.continue_on_error:
            break

    if len(summaries) > 1:
        print("\n===== MULTI-BATCH SUMMARY =====")
        for summary, code in zip(summaries, exit_codes):
            status = "ok" if code == 0 else "failed"
            print(f"{summary.get('batch_name'):<10} {status:<8} {summary.get('total_elapsed')}")

    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
