from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
from pathlib import Path
from typing import Any

import torch

from apcl.audio_crop import load_cropped_clip
from apcl.materials import MATERIAL_NAMES, MATERIAL_TO_INDEX, TARGET_NAMES, build_anchor_table, normalize_anchor_table
from apcl.model import APCLModel


DEFAULT_INFERENCE_CONFIG: dict[str, Any] = {
    "data": {
        "dataset": "dataset_a",
        "sample_rate": 32000,
        "clip_seconds": 2.0,
        "embedding_cache": {"enabled": False, "root": None, "embedding_dim": 768},
        "material_holdout": {
            "enabled": False,
            "held_out_materials": [],
            "exclude_from_normalization": True,
            "remove_from_anchor_candidates": False,
            "remove_from_class_space": False,
        },
        "material_class_aliases": {},
    },
    "model": {
        "audio_encoder": {
            "type": "pann",
            "repo_root": "third_party/audioset_tagging_cnn",
            "checkpoint_path": "checkpoints/Cnn14_mAP=0.431.pth",
            "freeze": True,
        },
        "pann_repo_root": "third_party/audioset_tagging_cnn",
        "pann_checkpoint_path": "checkpoints/Cnn14_mAP=0.431.pth",
        "freeze_audio_encoder": True,
        "audio_dim": 512,
        "projection_dim": 128,
        "audio_projection_hidden_dim": 256,
        "param_encoder_hidden_dim": 64,
        "prediction_mode": "anchor_residual",
        "dropout": 0.1,
        "eta_reference_hz": 1000.0,
        "target_names": list(TARGET_NAMES),
        "predict_uncertainty": False,
        "predict_vae": False,
        "vae_latent_dim": None,
        "vae_hidden_dim": None,
    },
}


PREDICTION_KEYS = {
    "lookup": "theta_anchor",
    "direct": "theta_direct",
    "anchor_residual": "theta",
    "vae": "theta_vae",
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _target_names_for_config(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("model", {}).get("target_names") or TARGET_NAMES)


def _prediction_key_from_config(config: dict[str, Any]) -> str:
    mode = str(config["model"].get("prediction_mode", "anchor_residual"))
    if mode not in PREDICTION_KEYS:
        raise ValueError(f"Unsupported model.prediction_mode={mode!r}")
    return PREDICTION_KEYS[mode]


def _audio_encoder_config_for_model(config: dict[str, Any]) -> dict[str, Any] | None:
    cache_cfg = config["data"].get("embedding_cache", {})
    if cache_cfg.get("enabled", False):
        return {
            "type": "precomputed",
            "embedding_dim": int(cache_cfg.get("embedding_dim", 768)),
        }
    return config["model"].get("audio_encoder")


def _get_held_out_materials(config: dict[str, Any]) -> list[str]:
    holdout_cfg = config["data"].get("material_holdout", {})
    held_out = holdout_cfg.get("held_out_materials", [])
    if isinstance(held_out, str):
        held_out = [held_out]
    if not holdout_cfg.get("enabled", False) and not held_out:
        return []
    unknown = sorted(set(held_out) - set(MATERIAL_NAMES))
    if unknown:
        raise ValueError(f"Unknown held-out material names: {unknown}. Choose from {MATERIAL_NAMES}.")
    return [name for name in MATERIAL_NAMES if name in set(held_out)]


def _get_seen_materials(config: dict[str, Any]) -> list[str]:
    held_out = set(_get_held_out_materials(config))
    return [name for name in MATERIAL_NAMES if name not in held_out]


def _normalize_anchor_table_for_config(
    raw_table: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    holdout_cfg = config["data"].get("material_holdout", {})
    if _get_held_out_materials(config) and holdout_cfg.get("exclude_from_normalization", True):
        return normalize_anchor_table(raw_table, materials=_get_seen_materials(config))
    return normalize_anchor_table(raw_table)


def _material_class_aliases_for_config(config: dict[str, Any]) -> dict[str, str]:
    aliases = dict(config["data"].get("material_class_aliases", {}))
    unknown = sorted((set(aliases) | set(aliases.values())) - set(MATERIAL_NAMES))
    if unknown:
        raise ValueError(f"Unknown material_class_aliases names: {unknown}. Choose from {MATERIAL_NAMES}.")
    return aliases


def _material_class_name_for_config(material: str, config: dict[str, Any]) -> str:
    return _material_class_aliases_for_config(config).get(material, material)


def _class_space_materials_for_config(config: dict[str, Any]) -> list[str]:
    if config["data"].get("material_holdout", {}).get("remove_from_class_space", False):
        return _get_seen_materials(config)
    return list(MATERIAL_NAMES)


def _class_materials_for_config(config: dict[str, Any]) -> list[str]:
    aliases = _material_class_aliases_for_config(config)
    class_materials = []
    for material in _class_space_materials_for_config(config):
        if material in aliases:
            continue
        class_name = aliases.get(material, material)
        if class_name not in class_materials:
            class_materials.append(class_name)
    return class_materials


def _class_target_indices_for_config(config: dict[str, Any]) -> torch.Tensor:
    class_materials = _class_materials_for_config(config)
    class_to_index = {name: idx for idx, name in enumerate(class_materials)}
    return torch.tensor(
        [class_to_index.get(_material_class_name_for_config(name, config), -100) for name in MATERIAL_NAMES],
        dtype=torch.long,
    )


def _anchor_candidate_mask_for_config(config: dict[str, Any]) -> torch.Tensor:
    mask = torch.ones(len(_class_materials_for_config(config)), dtype=torch.bool)
    holdout_cfg = config["data"].get("material_holdout", {})
    if _get_held_out_materials(config) and holdout_cfg.get("remove_from_anchor_candidates", False):
        held_out = set(_get_held_out_materials(config))
        for idx, name in enumerate(_class_materials_for_config(config)):
            if name in held_out:
                mask[idx] = False
    if not bool(mask.any()):
        raise ValueError("material_holdout.remove_from_anchor_candidates removed every anchor candidate.")
    return mask


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _patch_encoder_paths(checkpoint: dict[str, Any], pann_repo_root: str | None, pann_checkpoint_path: str | None) -> None:
    if pann_repo_root is None and pann_checkpoint_path is None:
        return
    config = checkpoint.setdefault("config", {})
    model_cfg = config.setdefault("model", {})
    encoder_cfg = model_cfg.setdefault("audio_encoder", {})
    if pann_repo_root is not None:
        model_cfg["pann_repo_root"] = str(pann_repo_root)
        encoder_cfg["repo_root"] = str(pann_repo_root)
    if pann_checkpoint_path is not None:
        model_cfg["pann_checkpoint_path"] = str(pann_checkpoint_path)
        encoder_cfg["checkpoint_path"] = str(pann_checkpoint_path)


def _load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not look like an ImpactMat training checkpoint.")
    return checkpoint


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[APCLModel, torch.Tensor, torch.Tensor, dict[str, Any]]:
    config = _deep_update(DEFAULT_INFERENCE_CONFIG, checkpoint.get("config", {}))
    raw_table = checkpoint.get("anchor_table_raw")
    if raw_table is None:
        raw_table = build_anchor_table(
            target_names=_target_names_for_config(config),
            eta_reference_hz=float(config["model"]["eta_reference_hz"]),
        )
    anchor_table_norm = checkpoint.get("anchor_table_norm")
    if anchor_table_norm is None:
        anchor_table_norm, _, _ = _normalize_anchor_table_for_config(raw_table, config)
    target_table_norm = checkpoint.get("target_table_norm")
    if target_table_norm is None:
        target_table_norm = anchor_table_norm
    anchor_table_norm = anchor_table_norm.to(device)
    target_table_norm = target_table_norm.to(device)
    class_target_indices = checkpoint.get("class_target_indices")
    if class_target_indices is None:
        class_target_indices = _class_target_indices_for_config(config)
    class_target_indices = class_target_indices.to(device)

    model = APCLModel(
        pann_repo_root=config["model"]["pann_repo_root"],
        pann_checkpoint_path=config["model"]["pann_checkpoint_path"],
        anchor_table_norm=anchor_table_norm,
        sample_rate=int(config["data"]["sample_rate"]),
        audio_dim=int(config["model"]["audio_dim"]),
        projection_dim=int(config["model"]["projection_dim"]),
        audio_projection_hidden_dim=int(config["model"].get("audio_projection_hidden_dim", 256)),
        param_encoder_hidden_dim=int(config["model"].get("param_encoder_hidden_dim", 64)),
        freeze_audio_encoder=bool(config["model"]["freeze_audio_encoder"]),
        audio_encoder_config=_audio_encoder_config_for_model(config),
        anchor_candidate_mask=_anchor_candidate_mask_for_config(config),
        dropout=float(config["model"]["dropout"]),
        predict_uncertainty=bool(config["model"].get("predict_uncertainty", False)),
        predict_vae=bool(config["model"].get("predict_vae", False)),
        vae_latent_dim=config["model"].get("vae_latent_dim"),
        vae_hidden_dim=config["model"].get("vae_hidden_dim"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")
    if target_mean is not None and target_std is not None:
        config["_runtime_tensors"] = {
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
            "class_anchor_table_raw": raw_table.detach().cpu(),
        }
    return model, target_table_norm, class_target_indices, config


def _load_model(
    checkpoint_path: Path,
    device_name: str,
    pann_repo_root: str | None,
    pann_checkpoint_path: str | None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(str(checkpoint_path), device)
    _patch_encoder_paths(checkpoint, pann_repo_root, pann_checkpoint_path)
    model, _, _, config = _build_model_from_checkpoint(checkpoint, device)
    model.eval()
    return model, config, device


def _expand_wavs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing wav file(s): {missing[:5]}")
    return paths


def _groups_from_wavs(values: list[str], object_id: str, material: str | None) -> list[dict[str, Any]]:
    paths = _expand_wavs(values)
    return [
        {
            "object_id": object_id,
            "object": object_id,
            "material": material or "",
            "wav_paths": paths,
        }
    ]


def _groups_from_manifest(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "wav_path" not in reader.fieldnames:
            raise ValueError("Manifest CSV must contain a wav_path column.")
        for row_idx, row in enumerate(reader):
            group_id = str(row.get("group_id") or row.get("object_id") or row.get("object") or row_idx)
            entry = grouped.setdefault(
                group_id,
                {
                    "object_id": str(row.get("object_id") or group_id),
                    "object": str(row.get("object") or group_id),
                    "material": str(row.get("material") or ""),
                    "wav_paths": [],
                },
            )
            wav_path = Path(str(row["wav_path"]))
            if not wav_path.is_file():
                raise FileNotFoundError(f"Missing wav file in manifest row {row_idx + 2}: {wav_path}")
            entry["wav_paths"].append(wav_path)
    return list(grouped.values())


def _denormalize(theta_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return theta_norm * std.to(theta_norm.device) + mean.to(theta_norm.device)


@torch.no_grad()
def infer_groups(
    groups: list[dict[str, Any]],
    checkpoint_path: Path,
    device_name: str,
    pann_repo_root: str | None,
    pann_checkpoint_path: str | None,
    crop_mode: str,
    crop_pre_seconds: float,
    clip_seconds_override: float | None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    model, config, device = _load_model(checkpoint_path, device_name, pann_repo_root, pann_checkpoint_path)
    prediction_key = _prediction_key_from_config(config)
    target_names = _target_names_for_config(config)
    target_mean = config["_runtime_tensors"]["target_mean"].float()
    target_std = config["_runtime_tensors"]["target_std"].float()
    anchor_raw = build_anchor_table(
        target_names=target_names,
        eta_reference_hz=float(config["model"]["eta_reference_hz"]),
    ).to(device)
    anchor_norm = ((anchor_raw - target_mean.to(device)) / target_std.to(device).clamp_min(1e-8)).to(device)
    sample_rate = int(config["data"]["sample_rate"])
    clip_seconds = float(clip_seconds_override or config["data"]["clip_seconds"])

    rows: list[dict[str, Any]] = []
    for group in groups:
        clips = []
        crop_infos = []
        for path in group["wav_paths"]:
            clip, crop_info = load_cropped_clip(
                str(path),
                sample_rate=sample_rate,
                clip_seconds=clip_seconds,
                crop_mode=crop_mode,
                crop_pre_seconds=crop_pre_seconds,
            )
            clips.append(clip)
            crop_infos.append(crop_info)
        waveforms = torch.stack(clips, dim=0).unsqueeze(0).to(device)
        hit_mask = torch.ones(waveforms.shape[:2], dtype=torch.bool, device=device)
        out = model(waveforms, hit_mask)
        probs = out["material_probs"][0].detach()
        pred_norm = out[prediction_key][0].detach()
        pred_raw = _denormalize(pred_norm, target_mean.to(device), target_std.to(device))
        dists = torch.cdist(pred_norm.unsqueeze(0), anchor_norm, p=2)[0]
        top_idx = int(probs.argmax().item())
        closest_idx = int(dists.argmin().item())

        row: dict[str, Any] = {
            "object_id": group["object_id"],
            "object": group["object"],
            "k": len(group["wav_paths"]),
            "samples": ";".join(Path(path).name for path in group["wav_paths"]),
            "sample_paths": ";".join(str(path) for path in group["wav_paths"]),
            "crop_mode": crop_mode,
            "clip_seconds": clip_seconds,
            "crop_start_times_sec": ";".join(_format_float(float(info["crop_start_time_sec"])) for info in crop_infos),
            "crop_event_times_sec": ";".join(_format_float(float(info["crop_event_time_sec"])) for info in crop_infos),
            "softmax_top1": MATERIAL_NAMES[top_idx],
            "softmax_top1_prob": float(probs[top_idx].cpu()),
            "closest_anchor_l2": MATERIAL_NAMES[closest_idx],
            "closest_anchor_l2_dist": float(dists[closest_idx].cpu()),
        }
        material = str(group.get("material") or "")
        if material in MATERIAL_TO_INDEX:
            gt_idx = MATERIAL_TO_INDEX[material]
            gt_norm = anchor_norm[gt_idx]
            abs_norm = (pred_norm - gt_norm).abs()
            row.update(
                {
                    "gt_material": material,
                    "gt_material_prob": float(probs[gt_idx].cpu()),
                    "top1_correct": int(top_idx == gt_idx),
                    "closest_anchor_correct": int(closest_idx == gt_idx),
                    "nmae": float(abs_norm.mean().cpu()),
                }
            )
            for target_idx, target_name in enumerate(target_names):
                row[f"mae_norm_{target_name}"] = float(abs_norm[target_idx].cpu())
        for target_idx, target_name in enumerate(target_names):
            row[f"est_{target_name}"] = float(pred_raw[target_idx].cpu())
        for material_idx, material_name in enumerate(MATERIAL_NAMES):
            row[f"prob_{material_name}"] = float(probs[material_idx].cpu())
            row[f"dist_l2_{material_name}"] = float(dists[material_idx].cpu())
        rows.append(row)
    return rows, target_names


def write_outputs(rows: list[dict[str, Any]], target_names: tuple[str, ...], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as handle:
        json.dump({"target_names": list(target_names), "predictions": rows}, handle, indent=2)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "object_id",
        "object",
        "gt_material",
        "k",
        "crop_mode",
        "softmax_top1",
        "softmax_top1_prob",
        "gt_material_prob",
        "top1_correct",
        "closest_anchor_l2",
        "closest_anchor_l2_dist",
        "nmae",
        *[f"est_{name}" for name in target_names],
        *[f"prob_{name}" for name in MATERIAL_NAMES],
        "samples",
        "sample_paths",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ImpactMat checkpoint inference on one or more wav files.")
    parser.add_argument("--checkpoint", required=True, help="Path to an ImpactMat checkpoint.")
    parser.add_argument("--wav", action="append", default=[], help="Input wav path or shell glob. Repeat for K-hit inference.")
    parser.add_argument("--manifest", default=None, help="CSV with wav_path and optional group_id/object_id/object/material columns.")
    parser.add_argument("--object-id", default="input", help="Object id used when --wav is supplied.")
    parser.add_argument("--material", default=None, help=f"Optional ground-truth material: {', '.join(MATERIAL_NAMES)}")
    parser.add_argument("--output-dir", default="inference_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pann-repo-root", default=None, help="Override checkpoint's PANN repo path.")
    parser.add_argument("--pann-checkpoint-path", default=None, help="Override checkpoint's PANN CNN14 checkpoint path.")
    parser.add_argument("--crop-mode", choices=["first", "onset", "peak", "rms_peak"], default="first")
    parser.add_argument("--crop-pre-seconds", type=float, default=0.1)
    parser.add_argument("--clip-seconds-override", type=float, default=None)
    args = parser.parse_args()

    if bool(args.manifest) == bool(args.wav):
        raise ValueError("Provide exactly one of --wav or --manifest.")
    if args.material is not None and args.material not in MATERIAL_TO_INDEX:
        raise ValueError(f"Unknown material {args.material!r}. Choose from {MATERIAL_NAMES}.")

    groups = (
        _groups_from_manifest(Path(args.manifest))
        if args.manifest
        else _groups_from_wavs(args.wav, object_id=str(args.object_id), material=args.material)
    )
    rows, target_names = infer_groups(
        groups=groups,
        checkpoint_path=Path(args.checkpoint),
        device_name=str(args.device),
        pann_repo_root=args.pann_repo_root,
        pann_checkpoint_path=args.pann_checkpoint_path,
        crop_mode=str(args.crop_mode),
        crop_pre_seconds=float(args.crop_pre_seconds),
        clip_seconds_override=args.clip_seconds_override,
    )
    write_outputs(rows, target_names, Path(args.output_dir))
    print(json.dumps({"output_dir": str(args.output_dir), "num_groups": len(rows), "predictions": rows}, indent=2))


if __name__ == "__main__":
    main()
