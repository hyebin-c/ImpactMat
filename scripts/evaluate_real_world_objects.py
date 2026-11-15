from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apcl.audio_crop import load_cropped_clip
from apcl.materials import MATERIAL_NAMES, MATERIAL_TO_INDEX, build_anchor_table
from test import build_model_from_checkpoint, load_checkpoint
from train import prediction_key_from_config, target_names_for_config


OBJECT_ID_RE = re.compile(r"^(?P<object_id>\d+)(?:_|$)")


def object_id_from_name(name: str) -> str:
    match = OBJECT_ID_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse object id from {name!r}")
    return match.group("object_id")


def load_material_map(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row["object_id"])
            material = str(row["material"])
            if material not in MATERIAL_TO_INDEX:
                continue
            rows[object_id] = {
                "object_id": object_id,
                "canonical_name": str(row.get("name", "")),
                "material": material,
            }
    return rows


def infer_realimpact_material(object_name: str) -> str | None:
    lowered = object_name.lower()
    token_map = [
        ("polycarbonate", "Polycarbonate"),
        ("ceramic", "Ceramic"),
        ("plastic", "Plastic"),
        ("wood", "Wood"),
        ("iron", "Iron"),
        ("glass", "Glass"),
        ("goblet", "Glass"),
        ("metal", "Steel"),
        ("steel", "Steel"),
    ]
    for token, material in token_map:
        if token in lowered:
            return material
    return None


def choose_paths(paths: list[Path], k: int, rng: random.Random) -> list[Path]:
    if not paths:
        raise ValueError("Cannot choose from an empty path list.")
    if len(paths) >= k:
        return rng.sample(paths, k)
    chosen = list(paths)
    chosen.extend(rng.choices(paths, k=k - len(paths)))
    rng.shuffle(chosen)
    return chosen


def collect_objectfolder_groups(root: Path, material_map: dict[str, dict[str, str]], k: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups = []
    audio_root = root / "audio"
    for object_dir in sorted(audio_root.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else 10**9):
        if not object_dir.is_dir() or not object_dir.name.isdigit():
            continue
        meta = material_map.get(object_dir.name)
        if meta is None:
            continue
        paths = sorted((object_dir / "audio").glob("*/mic.wav"))
        if not paths:
            continue
        groups.append(
            {
                "dataset": "ObjectFolder_Real",
                "object": object_dir.name,
                "object_id": object_dir.name,
                "canonical_name": meta["canonical_name"],
                "material": meta["material"],
                "material_label_source": "object_materials.csv",
                "num_available_wavs": len(paths),
                "wav_paths": choose_paths(paths, k, rng),
            }
        )
    return groups


def collect_realimpact_groups(root: Path, material_map: dict[str, dict[str, str]], k: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    metadata_path = root / "metadata.csv"
    paths_by_object: dict[str, list[Path]] = defaultdict(list)
    if metadata_path.exists():
        with open(metadata_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                object_name = str(row["object"])
                wav_path = Path(row["wav_path"])
                if wav_path.exists():
                    paths_by_object[object_name].append(wav_path)
    else:
        for object_dir in root.iterdir():
            if object_dir.is_dir():
                paths_by_object[object_dir.name].extend(sorted(object_dir.glob("*.wav")))

    groups = []
    for object_name in sorted(paths_by_object, key=lambda value: int(object_id_from_name(value))):
        object_id = object_id_from_name(object_name)
        meta = material_map.get(object_id)
        inferred_material = infer_realimpact_material(object_name)
        if meta is None and inferred_material is None:
            continue
        material = inferred_material or str(meta["material"])
        canonical_name = str(meta["canonical_name"]) if meta is not None else ""
        paths = sorted(paths_by_object[object_name])
        if not paths:
            continue
        groups.append(
            {
                "dataset": "RealImpact_wavs_n8",
                "object": object_name,
                "object_id": object_id,
                "canonical_name": canonical_name,
                "material": material,
                "material_label_source": "object_name" if inferred_material is not None else "object_materials.csv",
                "num_available_wavs": len(paths),
                "wav_paths": choose_paths(paths, k, rng),
            }
        )
    return groups


def denormalize(theta_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return theta_norm * std.to(theta_norm.device) + mean.to(theta_norm.device)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_table(handle, rows: list[dict[str, Any]], columns: list[str]) -> None:
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |\n")


def summarize_rows(rows: list[dict[str, Any]], target_names: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_objects": len(rows),
        "top1_acc": sum(int(row["top1_correct"]) for row in rows) / max(len(rows), 1),
        "closest_anchor_acc": sum(int(row["closest_anchor_correct"]) for row in rows) / max(len(rows), 1),
        "mean_top1_prob": sum(float(row["softmax_top1_prob"]) for row in rows) / max(len(rows), 1),
        "mean_gt_prob": sum(float(row["gt_material_prob"]) for row in rows) / max(len(rows), 1),
        "mean_loss_reg": sum(float(row["loss_reg"]) for row in rows) / max(len(rows), 1),
        "mean_nmae": sum(float(row["nmae"]) for row in rows) / max(len(rows), 1),
        "top1_counts": dict(Counter(str(row["softmax_top1"]) for row in rows)),
        "gt_counts": dict(Counter(str(row["gt_material"]) for row in rows)),
    }
    for name in target_names:
        summary[f"mean_mae_norm_{name}"] = (
            sum(float(row[f"mae_norm_{name}"]) for row in rows) / max(len(rows), 1)
        )
    return summary


def summarize_by_material(rows: list[dict[str, Any]], target_names: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for material in MATERIAL_NAMES:
        material_rows = [row for row in rows if row["gt_material"] == material]
        if not material_rows:
            continue
        row = {
            "material": material,
            **summarize_rows(material_rows, target_names),
        }
        out.append(row)
    return out


@torch.no_grad()
def evaluate_groups(
    groups: list[dict[str, Any]],
    run_dir: Path,
    checkpoint_name: str,
    device_name: str,
    batch_size: int,
    crop_mode: str,
    crop_pre_seconds: float,
    clip_seconds_override: float | None,
    external_event_time_by_path: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(str(run_dir / checkpoint_name), device)
    model, target_table_norm, _, config = build_model_from_checkpoint(checkpoint, device)
    prediction_key = prediction_key_from_config(config)
    target_names = target_names_for_config(config)

    target_mean = config["_runtime_tensors"]["target_mean"].float()
    target_std = config["_runtime_tensors"]["target_std"].float()
    anchor_raw = build_anchor_table(
        target_names=target_names,
        eta_reference_hz=float(config["model"]["eta_reference_hz"]),
    )
    anchor_norm = ((anchor_raw - target_mean) / target_std.clamp_min(1e-8)).to(device)
    anchor_raw = anchor_raw.to(device)

    sample_rate = int(config["data"]["sample_rate"])
    clip_seconds = float(clip_seconds_override or config["data"]["clip_seconds"])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(groups), batch_size):
        batch_groups = groups[start : start + batch_size]
        batch_crop_infos: list[list[dict[str, Any]]] = []
        batch_clips: list[torch.Tensor] = []
        for group in batch_groups:
            group_clips = []
            group_infos = []
            for path in group["wav_paths"]:
                external_event_time_sec = None
                if external_event_time_by_path is not None:
                    external_event_time_sec = external_event_time_by_path.get(str(path))
                    if external_event_time_sec is None:
                        raise KeyError(f"Missing external event time for {path}")
                clip, crop_info = load_cropped_clip(
                    str(path),
                    sample_rate=sample_rate,
                    clip_seconds=clip_seconds,
                    crop_mode=crop_mode,
                    crop_pre_seconds=crop_pre_seconds,
                    external_event_time_sec=external_event_time_sec,
                )
                group_clips.append(clip)
                group_infos.append(crop_info)
            batch_clips.append(torch.stack(group_clips, dim=0))
            batch_crop_infos.append(group_infos)
        waveforms = torch.stack(
            batch_clips,
            dim=0,
        ).to(device)
        hit_mask = torch.ones(waveforms.shape[:2], dtype=torch.bool, device=device)
        out = model(waveforms, hit_mask)
        probs = out["material_probs"].detach()
        pred_norm = out[prediction_key].detach()
        pred_raw = denormalize(pred_norm, target_mean.to(device), target_std.to(device))
        dists = torch.cdist(pred_norm, anchor_norm, p=2)

        for idx, group in enumerate(batch_groups):
            gt_material = str(group["material"])
            gt_idx = MATERIAL_TO_INDEX[gt_material]
            top_idx = int(probs[idx].argmax().item())
            closest_idx = int(dists[idx].argmin().item())
            gt_norm = anchor_norm[gt_idx]
            gt_raw = anchor_raw[gt_idx]
            abs_norm = (pred_norm[idx] - gt_norm).abs()
            loss_reg = F.smooth_l1_loss(pred_norm[idx], gt_norm, reduction="none").mean()
            row: dict[str, Any] = {
                "dataset": group["dataset"],
                "object": group["object"],
                "object_id": group["object_id"],
                "canonical_name": group["canonical_name"],
                "gt_material": gt_material,
                "material_label_source": group["material_label_source"],
                "num_available_wavs": int(group["num_available_wavs"]),
                "k": int(waveforms.shape[1]),
                "samples": ";".join(path.name for path in group["wav_paths"]),
                "sample_paths": ";".join(str(path) for path in group["wav_paths"]),
                "crop_mode": crop_mode,
                "crop_pre_seconds": float(crop_pre_seconds),
                "clip_seconds": float(clip_seconds),
                "crop_start_times_sec": ";".join(format_value(info["crop_start_time_sec"]) for info in batch_crop_infos[idx]),
                "crop_end_times_sec": ";".join(format_value(info["crop_end_time_sec"]) for info in batch_crop_infos[idx]),
                "crop_event_times_sec": ";".join(format_value(info["crop_event_time_sec"]) for info in batch_crop_infos[idx]),
                "external_event_times_sec": ";".join(
                    "" if info.get("external_event_time_sec") is None else format_value(info["external_event_time_sec"])
                    for info in batch_crop_infos[idx]
                ),
                "softmax_top1": MATERIAL_NAMES[top_idx],
                "softmax_top1_prob": float(probs[idx, top_idx].cpu()),
                "gt_material_prob": float(probs[idx, gt_idx].cpu()),
                "top1_correct": int(top_idx == gt_idx),
                "closest_anchor_l2": MATERIAL_NAMES[closest_idx],
                "closest_anchor_l2_dist": float(dists[idx, closest_idx].cpu()),
                "closest_anchor_correct": int(closest_idx == gt_idx),
                "loss_reg": float(loss_reg.cpu()),
                "nmae": float(abs_norm.mean().cpu()),
            }
            for material_idx, material in enumerate(MATERIAL_NAMES):
                row[f"prob_{material}"] = float(probs[idx, material_idx].cpu())
                row[f"dist_l2_{material}"] = float(dists[idx, material_idx].cpu())
            for target_idx, target_name in enumerate(target_names):
                row[f"est_{target_name}"] = float(pred_raw[idx, target_idx].cpu())
                row[f"gt_{target_name}"] = float(gt_raw[target_idx].cpu())
                row[f"mae_norm_{target_name}"] = float(abs_norm[target_idx].cpu())
            rows.append(row)

    summary = summarize_rows(rows, target_names)
    summary["crop_mode"] = crop_mode
    summary["crop_pre_seconds"] = float(crop_pre_seconds)
    summary["clip_seconds"] = float(clip_seconds)
    summary["by_material"] = summarize_by_material(rows, target_names)
    return rows, summary, target_names


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    target_names: tuple[str, ...],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "object_predictions.json", "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "objects": rows}, handle, indent=2)

    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "dataset",
        "object",
        "object_id",
        "canonical_name",
        "gt_material",
        "material_label_source",
        "k",
        "num_available_wavs",
        "crop_mode",
        "clip_seconds",
        "crop_start_times_sec",
        "crop_event_times_sec",
        "softmax_top1",
        "softmax_top1_prob",
        "gt_material_prob",
        "top1_correct",
        "closest_anchor_l2",
        "closest_anchor_l2_dist",
        "closest_anchor_correct",
        "loss_reg",
        "nmae",
        *[f"est_{name}" for name in target_names],
        *[f"mae_norm_{name}" for name in target_names],
        *[f"prob_{name}" for name in MATERIAL_NAMES],
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(output_dir / "object_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_material = summary.get("by_material", [])
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# Real-World Object Inference\n\n")
        summary_rows = [
            {
                "num_objects": summary["num_objects"],
                "crop_mode": summary.get("crop_mode", "first"),
                "clip_seconds": summary.get("clip_seconds", ""),
                "top1_acc": summary["top1_acc"],
                "closest_anchor_acc": summary["closest_anchor_acc"],
                "mean_top1_prob": summary["mean_top1_prob"],
                "mean_gt_prob": summary["mean_gt_prob"],
                "mean_loss_reg": summary["mean_loss_reg"],
                "mean_nmae": summary["mean_nmae"],
            }
        ]
        write_table(
            handle,
            summary_rows,
            ["num_objects", "crop_mode", "clip_seconds", "top1_acc", "closest_anchor_acc", "mean_top1_prob", "mean_gt_prob", "mean_loss_reg", "mean_nmae"],
        )
        handle.write("\n## Material Breakdown\n\n")
        write_table(
            handle,
            by_material,
            ["material", "num_objects", "top1_acc", "closest_anchor_acc", "mean_gt_prob", "mean_loss_reg", "mean_nmae"],
        )
        handle.write("\n## Object Predictions\n\n")
        write_table(
            handle,
            rows,
            [
                "object",
                "canonical_name",
                "gt_material",
                "material_label_source",
                "crop_mode",
                "softmax_top1",
                "softmax_top1_prob",
                "gt_material_prob",
                "closest_anchor_l2",
                "loss_reg",
                "nmae",
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["objectfolder", "realimpact"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--material-csv", default="/mnt/bear2/users/hbcho/ObjectFolder_Real/object_materials.csv")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-mode", choices=["first", "onset", "peak", "rms_peak", "external_start"], default="first")
    parser.add_argument("--crop-pre-seconds", type=float, default=0.1)
    parser.add_argument(
        "--clip-seconds-override",
        type=float,
        default=None,
        help="Override config data.clip_seconds at inference time, e.g. 6.0 for full ObjectFolder clips.",
    )
    args = parser.parse_args()

    material_map = load_material_map(Path(args.material_csv))
    if args.dataset == "objectfolder":
        groups = collect_objectfolder_groups(Path(args.root), material_map, int(args.k), int(args.seed))
    else:
        groups = collect_realimpact_groups(Path(args.root), material_map, int(args.k), int(args.seed))
    if not groups:
        raise RuntimeError(f"No groups found for dataset={args.dataset} root={args.root}")

    rows, summary, target_names = evaluate_groups(
        groups=groups,
        run_dir=Path(args.run_dir),
        checkpoint_name=str(args.checkpoint_name),
        device_name=str(args.device),
        batch_size=int(args.batch_size),
        crop_mode=str(args.crop_mode),
        crop_pre_seconds=float(args.crop_pre_seconds),
        clip_seconds_override=args.clip_seconds_override,
    )
    write_outputs(rows, summary, target_names, Path(args.output_dir))
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
