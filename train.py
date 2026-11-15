from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from apcl.data import (
    DatasetAImpactGroups,
    DatasetBImpactGroups,
    collate_impact_groups,
    discover_dataset_a_groups,
    discover_dataset_b_groups,
    split_groups_by_object,
)
from apcl.losses import latent_interpolation_loss, multi_impact_consistency, multi_positive_cross_modal_loss, soft_parameter_contrastive
from apcl.materials import MATERIAL_NAMES, MATERIAL_TO_INDEX, TARGET_NAMES, build_anchor_table, normalize_anchor_table
from apcl.model import APCLModel

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "dataset_a_root": "data/impactmat",
        "dataset": "dataset_a",
        "dataset_b_root": "data/impactmat",
        "include_rand": True,
        "include_blend": False,
        "complete_object_ids_file": None,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "split_seed": 42,
        "train_k": 4,
        "train_k_values": [1, 2, 4, 8],
        "eval_k_values": [1, 2, 4, 8],
        "sample_rate": 32000,
        "clip_seconds": 2.0,
        "embedding_cache": {
            "enabled": False,
            "root": None,
            "embedding_dim": 768,
        },
        "material_holdout": {
            "enabled": False,
            "held_out_materials": [],
            "exclude_from_normalization": True,
            "remove_from_anchor_candidates": False,
            "remove_from_class_space": False,
        },
        "material_class_aliases": {},
        "source_sampling": {
            "name": "natural",
            "rand_weight": 1.0,
            "blend_weight": 1.0,
            "balance_blend_ratios": False,
            "balance_blend_pairs": False,
            "schedule": [],
        },
    },
    "model": {
        "audio_encoder": {
            "type": "pann",
            "repo_root": "third_party/audioset_tagging_cnn",
            "checkpoint_path": "checkpoints/Cnn14_mAP=0.431.pth",
            "freeze": True,
            "pretrained_model_name": "",
            "sample_rate": 16000,
            "target_length": 1024,
            "norm_mean": -4.268,
            "norm_std": 4.569,
            "pooling": "mean",
            "trust_remote_code": True,
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
    "optim": {
        "epochs": 100,
        "batch_size": 16,
        "lr": 3e-4,
        "head_lr": None,
        "audio_encoder_lr": None,
        "weight_decay": 1e-4,
        "num_workers": 4,
        "min_lr": 5e-6,
    },
    "loss": {
        "reg_weight": 1.0,
        "reg_target_weights": {},
        "cls_weight": 0.2,
        "cls_weight_schedule": [],
        "anchor_reg_weight": 0.0,
        "direct_anchor_consistency_weight": 0.0,
        "direct_anchor_consistency_detach_anchor": True,
        "blend_latent_consistency_weight": 0.0,
        "blend_param_consistency_weight": 0.0,
        "blend_prediction_consistency_weight": 0.0,
        "rand_endpoint_consistency_weight": 0.0,
        "uncertainty_regression_weight": 0.0,
        "uncertainty_log_var_min": -5.0,
        "uncertainty_log_var_max": 2.0,
        "vae_kl_weight": 0.0,
        "vae_kl_warmup_epochs": 0,
        "sample_reweighting": {
            "enabled": False,
            "start_epoch": 0,
            "blend_only": False,
            "min_weight": 0.5,
            "max_weight": 2.0,
            "current_error_gamma": 0.0,
            "uncertainty_gamma": 0.0,
            "source_weights": {},
            "blend_ratio_weights": {},
            "blend_pair_weights": {},
            "blend_material_weights": {},
        },
        "param_contrast_weight": 0.05,
        "audio_param_weight": 0.05,
        "latent_mix_weight": 0.05,
        "multi_impact_weight": 0.05,
        "warmup_epochs": 10,
        "contrast_sigma": 1.0,
        "contrast_temperature": 0.1,
        "soft_label_loss": {
            "source_aware": False,
            "soft_ce_weight": 1.0,
            "rand_ce_weight": 1.0,
            "blend_soft_ce_weight": 1.0,
            "blend_endpoint_support_weight": 0.0,
            "blend_ratio_weight": 0.0,
            "rand_entropy_weight": 0.0,
        },
        "anchor_mixture": {
            "enabled": False,
            "space": "raw_then_log",
            "loss_weight": 0.0,
        },
    },
    "runtime": {
        "seed": 42,
        "output_root": "runs",
        "run_name": "dataset_a_apcl",
        "device": "cuda",
        "use_wandb": False,
        "wandb_entity": None,
        "wandb_project": "apcl",
        "wandb_run_name": None,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return deep_update(DEFAULT_CONFIG, json.load(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_k_values(data_cfg: dict[str, Any]) -> list[int]:
    train_k = data_cfg.get("train_k", 4)
    if isinstance(train_k, str):
        if train_k.lower() != "variable":
            raise ValueError("data.train_k must be an integer or 'variable'.")
        return [int(k) for k in data_cfg.get("train_k_values", [1, 2, 4, 8])]
    return [int(train_k)]


def get_eval_k_values(data_cfg: dict[str, Any]) -> list[int]:
    return [int(k) for k in data_cfg.get("eval_k_values", [data_cfg.get("train_k", 4)])]


PREDICTION_KEYS = {
    "lookup": "theta_anchor",
    "direct": "theta_direct",
    "anchor_residual": "theta",
    "vae": "theta_vae",
}


def prediction_key_from_config(config: dict[str, Any]) -> str:
    mode = str(config["model"].get("prediction_mode", "anchor_residual"))
    if mode not in PREDICTION_KEYS:
        raise ValueError(f"Unsupported model.prediction_mode: {mode}. Choose from {sorted(PREDICTION_KEYS)}.")
    return PREDICTION_KEYS[mode]


def target_names_for_config(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("model", {}).get("target_names") or TARGET_NAMES)


def target_loss_weights(config: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weight_cfg = config.get("loss", {}).get("reg_target_weights", {})
    target_names = target_names_for_config(config)
    weights = torch.tensor(
        [float(weight_cfg.get(name, 1.0)) for name in target_names],
        device=device,
        dtype=dtype,
    )
    return weights / weights.mean().clamp_min(1e-8)


def regression_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    per_target = F.smooth_l1_loss(pred, target, reduction="none")
    weights = target_loss_weights(config, pred.device, pred.dtype)
    return (per_target * weights.unsqueeze(0)).mean(dim=1)


def regression_loss(pred: torch.Tensor, target: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    return regression_loss_per_sample(pred, target, config).mean()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def uncertainty_regression_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    loss_cfg = config.get("loss", {})
    min_log_var = float(loss_cfg.get("uncertainty_log_var_min", -5.0))
    max_log_var = float(loss_cfg.get("uncertainty_log_var_max", 2.0))
    log_var = log_var.clamp(min=min_log_var, max=max_log_var)
    per_target = 0.5 * torch.exp(-log_var) * (pred - target).pow(2) + 0.5 * log_var
    weights = target_loss_weights(config, pred.device, pred.dtype)
    return (per_target * weights.unsqueeze(0)).mean(dim=1)


def scheduled_loss_weight(config: dict[str, Any], key: str, epoch: int) -> float:
    loss_cfg = config.get("loss", {})
    weight = float(loss_cfg.get(key, 0.0))
    schedule = loss_cfg.get(f"{key}_schedule", [])
    if not schedule:
        return weight
    for item in sorted(schedule, key=lambda row: int(row.get("epoch", row.get("start_epoch", 0)))):
        start_epoch = int(item.get("epoch", item.get("start_epoch", 0)))
        if epoch >= start_epoch:
            weight = float(item["value"])
    return weight


def _dict_weight(weight_cfg: dict[str, Any], key: str, default: float = 1.0) -> float:
    if key in weight_cfg:
        return float(weight_cfg[key])
    if key.startswith("w") and key[1:] in weight_cfg:
        return float(weight_cfg[key[1:]])
    if f"w{key}" in weight_cfg:
        return float(weight_cfg[f"w{key}"])
    return float(default)


def sample_loss_weights(
    batch: dict[str, Any],
    per_sample_reg: torch.Tensor,
    config: dict[str, Any],
    log_var: torch.Tensor | None = None,
    epoch: int = 0,
) -> torch.Tensor | None:
    cfg = config.get("loss", {}).get("sample_reweighting", {})
    if not bool(cfg.get("enabled", False)):
        return None
    if int(epoch) < int(cfg.get("start_epoch", 0)):
        return None

    weights = torch.ones_like(per_sample_reg)
    sources = [str(source) for source in batch.get("source", [])]
    if len(sources) == int(per_sample_reg.numel()):
        source_cfg = cfg.get("source_weights", {})
        for idx, source in enumerate(sources):
            weights[idx] *= float(source_cfg.get(source, 1.0))

    blend_mask = torch.tensor(
        [source == "blend" for source in sources],
        device=per_sample_reg.device,
        dtype=torch.bool,
    ) if len(sources) == int(per_sample_reg.numel()) else torch.zeros_like(per_sample_reg, dtype=torch.bool)

    if bool(blend_mask.any()):
        ratio_cfg = cfg.get("blend_ratio_weights", {})
        pair_cfg = cfg.get("blend_pair_weights", {})
        material_cfg = cfg.get("blend_material_weights", {})
        blend_w1 = batch.get("blend_w1")
        blend_m1 = [str(value) for value in batch.get("blend_m1", [])]
        blend_m2 = [str(value) for value in batch.get("blend_m2", [])]
        for idx in torch.where(blend_mask)[0].tolist():
            if torch.is_tensor(blend_w1):
                ratio = int(blend_w1[idx].detach().cpu())
                weights[idx] *= _dict_weight(ratio_cfg, f"w{ratio}", 1.0)
            if idx < len(blend_m1) and idx < len(blend_m2) and blend_m1[idx] and blend_m2[idx]:
                pair_key = "_".join(sorted([blend_m1[idx], blend_m2[idx]]))
                ordered_pair_key = f"{blend_m1[idx]}_{blend_m2[idx]}"
                pair_weight = float(pair_cfg.get(ordered_pair_key, pair_cfg.get(pair_key, 1.0)))
                m1_weight = float(material_cfg.get(blend_m1[idx], 1.0))
                m2_weight = float(material_cfg.get(blend_m2[idx], 1.0))
                weights[idx] *= pair_weight * 0.5 * (m1_weight + m2_weight)

    hard_mask = blend_mask if bool(cfg.get("blend_only", False)) else torch.ones_like(per_sample_reg, dtype=torch.bool)
    if bool(hard_mask.any()):
        error_gamma = float(cfg.get("current_error_gamma", 0.0))
        if error_gamma > 0:
            detached_error = per_sample_reg.detach().clamp_min(1e-8)
            baseline = detached_error[hard_mask].mean().clamp_min(1e-8)
            dynamic = (detached_error / baseline).pow(error_gamma)
            weights = torch.where(hard_mask, weights * dynamic, weights)

        uncertainty_gamma = float(cfg.get("uncertainty_gamma", 0.0))
        if uncertainty_gamma > 0 and log_var is not None:
            mean_log_var = log_var.detach().mean(dim=1).clamp(min=-10.0, max=10.0)
            confidence = torch.exp(-mean_log_var)
            confidence = confidence / confidence[hard_mask].mean().clamp_min(1e-8)
            weights = torch.where(hard_mask, weights * confidence.pow(uncertainty_gamma), weights)

    weights = weights.clamp(
        min=float(cfg.get("min_weight", 0.5)),
        max=float(cfg.get("max_weight", 2.0)),
    )
    return weights / weights.mean().clamp_min(1e-8)


def project_class_probs_for_config(class_probs: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if class_probs.shape[-1] == len(class_materials_for_config(config)):
        return class_probs
    projected = class_probs.new_zeros((class_probs.shape[0], len(class_materials_for_config(config))))
    class_target_indices = class_target_indices_for_config(config).to(class_probs.device)
    for source_idx, target_idx in enumerate(class_target_indices.tolist()):
        if target_idx >= 0:
            projected[:, int(target_idx)] += class_probs[:, source_idx]
    row_sum = projected.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0.0, projected / row_sum.clamp_min(1e-8), projected)


def source_mask(batch: dict[str, Any], source_name: str, device: torch.device) -> torch.Tensor:
    sources = batch.get("source")
    if sources is None:
        return torch.zeros(0, device=device, dtype=torch.bool)
    return torch.tensor([source == source_name for source in sources], device=device, dtype=torch.bool)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0 or not bool(mask.any()):
        return values.new_tensor(0.0)
    return values[mask].mean()


def compute_soft_label_classification(
    material_logits: torch.Tensor,
    material_probs: torch.Tensor,
    class_probs: torch.Tensor,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = config["loss"].get("soft_label_loss", {})
    log_probs = F.log_softmax(material_logits, dim=1)
    class_probs = class_probs.to(log_probs.device)
    soft_ce_per_sample = -(class_probs * log_probs).sum(dim=1)
    rand_mask = source_mask(batch, "rand", log_probs.device)
    blend_mask = source_mask(batch, "blend", log_probs.device)
    if rand_mask.numel() != soft_ce_per_sample.numel():
        rand_mask = torch.zeros_like(soft_ce_per_sample, dtype=torch.bool)
        blend_mask = torch.zeros_like(soft_ce_per_sample, dtype=torch.bool)

    if bool(loss_cfg.get("source_aware", False)):
        rand_ce = masked_mean(soft_ce_per_sample, rand_mask)
        blend_soft_ce = masked_mean(soft_ce_per_sample, blend_mask)
        cls = (
            float(loss_cfg.get("rand_ce_weight", 1.0)) * rand_ce
            + float(loss_cfg.get("blend_soft_ce_weight", 1.0)) * blend_soft_ce
        )
    else:
        rand_ce = masked_mean(soft_ce_per_sample, rand_mask)
        blend_soft_ce = masked_mean(soft_ce_per_sample, blend_mask)
        cls = float(loss_cfg.get("soft_ce_weight", 1.0)) * soft_ce_per_sample.mean()

    support_mask = class_probs > 0.0
    support_mass = (material_probs * support_mask.to(material_probs.dtype)).sum(dim=1).clamp_min(1e-8)
    endpoint_support = masked_mean(-torch.log(support_mass), blend_mask)

    pair_probs = material_probs * support_mask.to(material_probs.dtype)
    pair_probs = pair_probs / pair_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pair_targets = class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ratio_loss_per_sample = -(pair_targets * torch.log(pair_probs.clamp_min(1e-8))).sum(dim=1)
    blend_ratio = masked_mean(ratio_loss_per_sample, blend_mask)
    ratio_mae_per_sample = (pair_probs - pair_targets).abs().sum(dim=1) * 0.5
    blend_ratio_mae = masked_mean(ratio_mae_per_sample, blend_mask)

    entropy_per_sample = -(material_probs * torch.log(material_probs.clamp_min(1e-8))).sum(dim=1)
    rand_entropy = masked_mean(entropy_per_sample, rand_mask)

    endpoint_support_weight = float(loss_cfg.get("blend_endpoint_support_weight", 0.0))
    blend_ratio_weight = float(loss_cfg.get("blend_ratio_weight", 0.0))
    rand_entropy_weight = float(loss_cfg.get("rand_entropy_weight", 0.0))
    if endpoint_support_weight > 0:
        cls = cls + endpoint_support_weight * endpoint_support
    if blend_ratio_weight > 0:
        cls = cls + blend_ratio_weight * blend_ratio
    if rand_entropy_weight > 0:
        cls = cls + rand_entropy_weight * rand_entropy

    top2 = material_probs.topk(k=min(2, material_probs.shape[1]), dim=1).indices
    top2_hits = support_mask.gather(1, top2).all(dim=1).float()
    endpoint_recall_at_2 = masked_mean(top2_hits, blend_mask)
    off_support_mass = masked_mean(1.0 - support_mass, blend_mask)

    metrics = {
        "loss_cls_soft_ce": float(soft_ce_per_sample.mean().detach().cpu()),
        "loss_cls_rand_ce": float(rand_ce.detach().cpu()),
        "loss_cls_blend_soft_ce": float(blend_soft_ce.detach().cpu()),
        "loss_blend_endpoint_support": float(endpoint_support.detach().cpu()),
        "loss_blend_ratio": float(blend_ratio.detach().cpu()),
        "loss_rand_entropy": float(rand_entropy.detach().cpu()),
        "blend_off_support_mass": float(off_support_mass.detach().cpu()),
        "blend_endpoint_recall_at_2": float(endpoint_recall_at_2.detach().cpu()),
        "blend_ratio_mae": float(blend_ratio_mae.detach().cpu()),
    }
    return cls, metrics


def raw_space_anchor_mixture_norm(material_probs: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    runtime_tensors = config.get("_runtime_tensors", {})
    anchor_raw = runtime_tensors["class_anchor_table_raw"].to(material_probs.device)
    target_mean = runtime_tensors["target_mean"].to(material_probs.device)
    target_std = runtime_tensors["target_std"].to(material_probs.device).clamp_min(1e-8)
    columns = []
    for idx, name in enumerate(target_names_for_config(config)):
        anchor_column = anchor_raw[:, idx]
        if name.startswith("log_"):
            mixed = material_probs @ anchor_column.exp()
            columns.append(torch.log(mixed.clamp_min(1e-12)))
        else:
            columns.append(material_probs @ anchor_column)
    mixture_raw = torch.stack(columns, dim=1)
    return (mixture_raw - target_mean) / target_std


def audio_encoder_config_for_model(config: dict[str, Any]) -> dict[str, Any] | None:
    cache_cfg = config["data"].get("embedding_cache", {})
    if cache_cfg.get("enabled", False):
        return {
            "type": "precomputed",
            "embedding_dim": int(cache_cfg.get("embedding_dim", 768)),
        }
    return config["model"].get("audio_encoder")


def get_held_out_materials(config: dict[str, Any]) -> list[str]:
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


def get_seen_materials(config: dict[str, Any]) -> list[str]:
    held_out = set(get_held_out_materials(config))
    return [name for name in MATERIAL_NAMES if name not in held_out]


def class_space_materials_for_config(config: dict[str, Any]) -> list[str]:
    if config["data"].get("material_holdout", {}).get("remove_from_class_space", False):
        return get_seen_materials(config)
    return list(MATERIAL_NAMES)


def filter_groups_by_material(groups: Sequence[Any], *, include: Sequence[str] | None = None, exclude: Sequence[str] | None = None) -> list[Any]:
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()
    return [
        group
        for group in groups
        if (include_set is None or group.material in include_set) and group.material not in exclude_set
    ]


def group_material_set(group: Any) -> set[str]:
    blend_m1 = getattr(group, "blend_m1", None)
    blend_m2 = getattr(group, "blend_m2", None)
    if blend_m1 is not None and blend_m2 is not None:
        return {str(blend_m1), str(blend_m2)}
    material = str(getattr(group, "material", ""))
    if material in MATERIAL_NAMES:
        return {material}
    return {part for part in material.split("_") if part in MATERIAL_NAMES}


def group_has_any_material(group: Any, materials: set[str]) -> bool:
    return bool(group_material_set(group) & materials)


def group_all_materials_in(group: Any, materials: set[str]) -> bool:
    group_materials = group_material_set(group)
    return bool(group_materials) and group_materials <= materials


def split_groups_for_material_holdout(
    groups: Sequence[Any],
    config: dict[str, Any],
) -> tuple[list[Any], list[Any], dict[str, list[Any]], dict[str, Any]]:
    data_cfg = config["data"]
    train_groups, val_groups, test_groups = split_groups_by_object(
        groups,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(data_cfg["split_seed"]),
    )
    held_out = get_held_out_materials(config)
    seen = get_seen_materials(config)
    if not held_out:
        counts = {"groups": len(groups), "train": len(train_groups), "val": len(val_groups), "test": len(test_groups)}
        return train_groups, val_groups, {"test": test_groups}, counts

    held_out_set = set(held_out)
    seen_set = set(seen)
    train_seen = [group for group in train_groups if not group_has_any_material(group, held_out_set)]
    val_seen = [group for group in val_groups if not group_has_any_material(group, held_out_set)]
    test_seen = [group for group in test_groups if group_all_materials_in(group, seen_set)]
    test_unseen = [group for group in test_groups if group_has_any_material(group, held_out_set)]
    counts = {
        "groups": len(groups),
        "train_seen": len(train_seen),
        "val_seen": len(val_seen),
        "test_all": len(test_groups),
        "test_seen": len(test_seen),
        "test_unseen": len(test_unseen),
        "seen_materials": seen,
        "held_out_materials": held_out,
    }
    return train_seen, val_seen, {"test_seen": test_seen, "test_unseen": test_unseen, "test_all": list(test_groups)}, counts


def normalize_anchor_table_for_config(
    raw_table: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    holdout_cfg = config["data"].get("material_holdout", {})
    if get_held_out_materials(config) and holdout_cfg.get("exclude_from_normalization", True):
        return normalize_anchor_table(raw_table, materials=get_seen_materials(config))
    return normalize_anchor_table(raw_table)


def material_class_aliases_for_config(config: dict[str, Any]) -> dict[str, str]:
    aliases = dict(config["data"].get("material_class_aliases", {}))
    unknown = sorted((set(aliases) | set(aliases.values())) - set(MATERIAL_NAMES))
    if unknown:
        raise ValueError(f"Unknown material_class_aliases names: {unknown}. Choose from {MATERIAL_NAMES}.")
    return aliases


def material_class_name_for_config(material: str, config: dict[str, Any]) -> str:
    return material_class_aliases_for_config(config).get(material, material)


def class_materials_for_config(config: dict[str, Any]) -> list[str]:
    aliases = material_class_aliases_for_config(config)
    class_materials = []
    for material in class_space_materials_for_config(config):
        if material in aliases:
            continue
        class_name = aliases.get(material, material)
        if class_name not in class_materials:
            class_materials.append(class_name)
    return class_materials


def class_target_indices_for_config(config: dict[str, Any]) -> torch.Tensor:
    class_materials = class_materials_for_config(config)
    class_to_index = {name: idx for idx, name in enumerate(class_materials)}
    return torch.tensor(
        [class_to_index.get(material_class_name_for_config(name, config), -100) for name in MATERIAL_NAMES],
        dtype=torch.long,
    )


def class_anchor_indices_for_config(config: dict[str, Any]) -> torch.Tensor:
    return torch.tensor([MATERIAL_TO_INDEX[name] for name in class_materials_for_config(config)], dtype=torch.long)


def build_class_anchor_tables(
    raw_table: torch.Tensor,
    target_table_norm: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    indices = class_anchor_indices_for_config(config)
    class_materials = class_materials_for_config(config)
    return raw_table.index_select(0, indices), target_table_norm.index_select(0, indices), class_materials


def anchor_candidate_mask_for_config(config: dict[str, Any]) -> torch.Tensor:
    mask = torch.ones(len(class_materials_for_config(config)), dtype=torch.bool)
    holdout_cfg = config["data"].get("material_holdout", {})
    if get_held_out_materials(config) and holdout_cfg.get("remove_from_anchor_candidates", False):
        held_out = set(get_held_out_materials(config))
        for idx, name in enumerate(class_materials_for_config(config)):
            if name in held_out:
                mask[idx] = False
    if not bool(mask.any()):
        raise ValueError("material_holdout.remove_from_anchor_candidates removed every anchor candidate.")
    return mask


def _make_loader(
    groups,
    config: dict[str, Any],
    random_hits: bool,
    k_choices: list[int] | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    data_cfg = config["data"]
    cache_cfg = data_cfg.get("embedding_cache", {})
    embedding_cache_root = cache_cfg.get("root") if cache_cfg.get("enabled", False) else None
    dataset = DatasetAImpactGroups(
        groups,
        k_choices=k_choices or get_train_k_values(data_cfg),
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        random_hits=random_hits,
        embedding_cache_root=embedding_cache_root,
        audio_augmentation=data_cfg.get("audio_augmentation") if shuffle and embedding_cache_root is None else None,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(config["optim"]["num_workers"]),
        collate_fn=collate_impact_groups,
        pin_memory=torch.cuda.is_available(),
    )


def _read_object_ids_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


class SourceWeightedSampler(Sampler[int]):
    def __init__(self, groups, sampling_cfg: dict[str, Any], num_samples: int | None = None) -> None:
        self.groups = list(groups)
        self.sampling_cfg = sampling_cfg
        self.num_samples = int(num_samples or len(self.groups))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _source_weights_for_epoch(self) -> dict[str, float]:
        name = str(self.sampling_cfg.get("name", "natural")).lower()
        if name == "source_curriculum":
            schedule = list(self.sampling_cfg.get("schedule", []))
            if schedule:
                selected = schedule[-1]
                for item in schedule:
                    if self.epoch < int(item.get("end_epoch", self.epoch + 1)):
                        selected = item
                        break
                return {
                    "rand": float(selected.get("rand_weight", 1.0)),
                    "blend": float(selected.get("blend_weight", 1.0)),
                }
        return {
            "rand": float(self.sampling_cfg.get("rand_weight", 1.0)),
            "blend": float(self.sampling_cfg.get("blend_weight", 1.0)),
        }

    def __iter__(self):
        source_weights = self._source_weights_for_epoch()
        balance_blend_ratios = bool(self.sampling_cfg.get("balance_blend_ratios", False))
        balance_blend_pairs = bool(self.sampling_cfg.get("balance_blend_pairs", False))

        def bucket_for_group(group: Any) -> str:
            source = str(getattr(group, "source", "rand"))
            if source != "blend":
                return source
            parts = []
            if balance_blend_pairs and getattr(group, "blend_m1", None) and getattr(group, "blend_m2", None):
                parts.append("pair_" + "_".join(sorted([str(getattr(group, "blend_m1")), str(getattr(group, "blend_m2"))])))
            if balance_blend_ratios and getattr(group, "blend_w1", None) is not None:
                parts.append(f"w{int(getattr(group, 'blend_w1'))}")
            return "blend:" + "|".join(parts) if parts else source

        counts: dict[str, int] = {}
        source_bucket_names: dict[str, set[str]] = {}
        for group in self.groups:
            source = str(getattr(group, "source", "rand"))
            bucket = bucket_for_group(group)
            counts[bucket] = counts.get(bucket, 0) + 1
            source_bucket_names.setdefault(source, set()).add(bucket)
        weights = []
        for group in self.groups:
            source = str(getattr(group, "source", "rand"))
            source_weight = max(float(source_weights.get(source, 1.0)), 0.0)
            bucket = bucket_for_group(group)
            count = max(int(counts.get(bucket, 1)), 1)
            bucket_weight = source_weight / max(len(source_bucket_names.get(source, {bucket})), 1)
            weights.append(bucket_weight / count)
        weight_tensor = torch.tensor(weights, dtype=torch.double)
        if float(weight_tensor.sum()) <= 0.0:
            weight_tensor = torch.ones(len(self.groups), dtype=torch.double)
        generator = torch.Generator()
        generator.manual_seed(int(self.sampling_cfg.get("seed", 0)) + self.epoch)
        indices = torch.multinomial(weight_tensor, self.num_samples, replacement=True, generator=generator)
        return iter(indices.tolist())


def _dataset_b_train_sampler(groups, config: dict[str, Any]) -> SourceWeightedSampler | None:
    sampling_cfg = config["data"].get("source_sampling", {})
    name = str(sampling_cfg.get("name", "natural")).lower()
    if name in {"natural", "none", "off", "disabled"}:
        return None
    if name not in {"source_balanced", "source_curriculum", "source_ratio_balanced"}:
        raise ValueError(f"Unsupported data.source_sampling.name: {name}")
    return SourceWeightedSampler(groups, sampling_cfg, num_samples=len(groups))


def _make_dataset_b_loader(
    groups,
    config: dict[str, Any],
    random_hits: bool,
    k_choices: list[int] | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    data_cfg = config["data"]
    cache_cfg = data_cfg.get("embedding_cache", {})
    embedding_cache_root = cache_cfg.get("root") if cache_cfg.get("enabled", False) else None
    dataset = DatasetBImpactGroups(
        groups,
        k_choices=k_choices or get_train_k_values(data_cfg),
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        random_hits=random_hits,
        embedding_cache_root=embedding_cache_root,
        audio_augmentation=data_cfg.get("audio_augmentation") if shuffle and embedding_cache_root is None else None,
    )
    sampler = _dataset_b_train_sampler(groups, config) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=int(config["optim"]["num_workers"]),
        collate_fn=collate_impact_groups,
        pin_memory=torch.cuda.is_available(),
    )


def build_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, DataLoader], dict[str, Any]]:
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("dataset", "dataset_a")).lower()
    if dataset_name in {"dataset_b", "b", "rand_blend", "blend"}:
        complete_ids = _read_object_ids_file(data_cfg.get("complete_object_ids_file"))
        groups = discover_dataset_b_groups(
            data_cfg.get("dataset_b_root") or data_cfg["dataset_a_root"],
            include_rand=bool(data_cfg.get("include_rand", True)),
            include_blend=bool(data_cfg.get("include_blend", True)),
            complete_object_ids=complete_ids,
            eta_reference_hz=float(config["model"]["eta_reference_hz"]),
            target_names=target_names_for_config(config),
        )
        if not groups:
            raise RuntimeError(f"No dataset_b groups found under {data_cfg.get('dataset_b_root') or data_cfg['dataset_a_root']}")
        train_groups, val_groups, test_groups_by_name, split_counts = split_groups_for_material_holdout(groups, config)
        counts = {
            "dataset": "dataset_b",
            "groups": len(groups),
            "complete_object_ids_file": data_cfg.get("complete_object_ids_file"),
            "source_counts": {
                "rand": len([group for group in groups if getattr(group, "source", "") == "rand"]),
                "blend": len([group for group in groups if getattr(group, "source", "") == "blend"]),
            },
            **split_counts,
        }
        test_loaders = {
            f"{split_name}/K{int(k)}": _make_dataset_b_loader(split_groups, config, random_hits=False, k_choices=[int(k)])
            for split_name, split_groups in test_groups_by_name.items()
            for k in get_eval_k_values(data_cfg)
        }
        return (
            _make_dataset_b_loader(train_groups, config, random_hits=True, shuffle=True, drop_last=True),
            _make_dataset_b_loader(val_groups, config, random_hits=False),
            test_loaders,
            counts,
        )

    groups = discover_dataset_a_groups(data_cfg["dataset_a_root"])
    if not groups:
        raise RuntimeError(f"No rand material wav groups found under {data_cfg['dataset_a_root']}")
    train_groups, val_groups, test_groups_by_name, counts = split_groups_for_material_holdout(groups, config)
    if not train_groups:
        raise RuntimeError("No training groups remain after material holdout filtering.")

    test_loaders = {
        f"{split_name}/K{int(k)}": _make_loader(split_groups, config, random_hits=False, k_choices=[int(k)])
        for split_name, split_groups in test_groups_by_name.items()
        for k in get_eval_k_values(data_cfg)
    }
    return (
        _make_loader(train_groups, config, random_hits=True, shuffle=True, drop_last=True),
        _make_loader(val_groups, config, random_hits=False),
        test_loaders,
        counts,
    )


def compute_losses(model: APCLModel, batch: dict[str, Any], target_table_norm: torch.Tensor, class_target_indices: torch.Tensor, config: dict[str, Any], epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    original_labels = batch["material_index"]
    valid_class_mask = original_labels >= 0
    class_labels = torch.full_like(original_labels, fill_value=-100)
    if bool(valid_class_mask.any()):
        mapped_labels = class_target_indices.index_select(0, original_labels[valid_class_mask])
        class_labels[valid_class_mask] = mapped_labels
        valid_class_mask = valid_class_mask & (class_labels >= 0)
    class_probs = batch.get("class_probs")
    if "theta_raw" in batch:
        target_mean = config["_runtime_tensors"]["target_mean"].to(batch["theta_raw"].device)
        target_std = config["_runtime_tensors"]["target_std"].to(batch["theta_raw"].device)
        theta = (batch["theta_raw"] - target_mean) / target_std
    else:
        theta = target_table_norm.index_select(0, original_labels)
    out = model(batch.get("waveforms"), batch["hit_mask"], clip_embeddings=batch.get("clip_embeddings"))

    prediction_key = prediction_key_from_config(config)
    if prediction_key not in out:
        raise ValueError(
            f"model.prediction_mode={config['model'].get('prediction_mode')} requires output '{prediction_key}'. "
            "Check the corresponding model flag, e.g. model.predict_vae=true for VAE."
        )
    pred_theta = out[prediction_key]
    per_sample_reg = regression_loss_per_sample(pred_theta, theta, config)
    log_var = out.get("theta_log_var")
    sample_weights = sample_loss_weights(batch, per_sample_reg, config, log_var=log_var, epoch=epoch)
    reg_plain = per_sample_reg.mean()
    reg_objective = weighted_mean(per_sample_reg, sample_weights)
    vae_kl_weight_base = float(config["loss"].get("vae_kl_weight", 0.0))
    vae_kl_warmup_epochs = int(config["loss"].get("vae_kl_warmup_epochs", 0))
    vae_kl_weight = vae_kl_weight_base
    if vae_kl_warmup_epochs > 0:
        vae_kl_weight *= min(1.0, float(epoch + 1) / float(vae_kl_warmup_epochs))
    vae_kl = pred_theta.new_tensor(0.0)
    if vae_kl_weight_base > 0:
        if "vae_mu" not in out or "vae_logvar" not in out:
            raise ValueError("loss.vae_kl_weight requires model.predict_vae=true.")
        vae_logvar = out["vae_logvar"].clamp(min=-10.0, max=10.0)
        vae_kl_per_sample = -0.5 * (1.0 + vae_logvar - out["vae_mu"].pow(2) - vae_logvar.exp()).mean(dim=1)
        vae_kl = weighted_mean(vae_kl_per_sample, sample_weights)
    uncertainty_weight = float(config["loss"].get("uncertainty_regression_weight", 0.0))
    uncertainty_reg = pred_theta.new_tensor(0.0)
    if uncertainty_weight > 0:
        if log_var is None:
            raise ValueError("loss.uncertainty_regression_weight requires model.predict_uncertainty=true.")
        uncertainty_per_sample = uncertainty_regression_loss_per_sample(pred_theta, theta, log_var, config)
        uncertainty_reg = weighted_mean(uncertainty_per_sample, sample_weights)
    if class_probs is not None:
        class_probs = project_class_probs_for_config(class_probs, config)
        if class_probs.shape[-1] != out["material_logits"].shape[-1]:
            raise ValueError(
                "Dataset-provided class_probs must match material_logits dimension. "
                "Disable material_class_aliases for dataset_b soft-label classification."
            )
        cls, cls_metrics = compute_soft_label_classification(
            out["material_logits"],
            out["material_probs"],
            class_probs,
            batch,
            config,
        )
        class_argmax = class_probs.to(out["material_logits"].device).argmax(dim=1)
        valid_acc = (out["material_logits"].argmax(dim=1) == class_argmax).float().mean()
        soft_target_mass = (
            out["material_probs"] * class_probs.to(out["material_probs"].device)
        ).sum(dim=1).mean()
    elif bool(valid_class_mask.any()):
        cls = F.cross_entropy(out["material_logits"], class_labels, ignore_index=-100)
        cls_metrics = {}
        valid_acc = (
            (out["material_logits"].argmax(dim=1)[valid_class_mask] == class_labels[valid_class_mask]).float().mean()
            if bool(valid_class_mask.any())
            else torch.tensor(0.0, device=out["material_logits"].device)
        )
        soft_target_mass = torch.tensor(0.0, device=out["material_logits"].device)
    else:
        cls = out["material_logits"].sum() * 0.0
        cls_metrics = {}
        valid_acc = torch.tensor(0.0, device=out["material_logits"].device)
        soft_target_mass = torch.tensor(0.0, device=out["material_logits"].device)
    reg_weight = float(config["loss"].get("reg_weight", 1.0))
    cls_weight = scheduled_loss_weight(config, "cls_weight", epoch)
    anchor_reg = regression_loss(out["theta_anchor"], theta, config)
    anchor_mixture_cfg = config["loss"].get("anchor_mixture", {})
    raw_anchor_mixture_reg = pred_theta.new_tensor(0.0)
    if bool(anchor_mixture_cfg.get("enabled", False)):
        if str(anchor_mixture_cfg.get("space", "raw_then_log")).lower() != "raw_then_log":
            raise ValueError("loss.anchor_mixture.space currently supports only 'raw_then_log'.")
        raw_anchor_mixture = raw_space_anchor_mixture_norm(out["material_probs"], config)
        raw_anchor_mixture_reg = regression_loss(raw_anchor_mixture, theta, config)
    anchor_reg_weight = float(config["loss"].get("anchor_reg_weight", 0.0))
    consistency_weight = float(config["loss"].get("direct_anchor_consistency_weight", 0.0))
    consistency_anchor = out["theta_anchor"]
    if bool(config["loss"].get("direct_anchor_consistency_detach_anchor", True)):
        consistency_anchor = consistency_anchor.detach()
    direct_anchor_consistency = regression_loss(out["theta_direct"], consistency_anchor, config)

    total = reg_weight * reg_objective + cls_weight * cls
    if vae_kl_weight > 0:
        total = total + vae_kl_weight * vae_kl
    if uncertainty_weight > 0:
        total = total + uncertainty_weight * uncertainty_reg
    blend_prediction_weight = float(config["loss"].get("blend_prediction_consistency_weight", 0.0))
    blend_prediction = pred_theta.new_tensor(0.0)
    if blend_prediction_weight > 0 and "source" in batch:
        blend_mask = torch.tensor(
            [source == "blend" for source in batch["source"]],
            device=pred_theta.device,
            dtype=torch.bool,
        )
        if bool(blend_mask.any()):
            blend_weights = sample_weights[blend_mask] if sample_weights is not None else None
            blend_prediction = weighted_mean(per_sample_reg[blend_mask], blend_weights)
            total = total + blend_prediction_weight * blend_prediction
    rand_endpoint_weight = float(config["loss"].get("rand_endpoint_consistency_weight", 0.0))
    rand_endpoint = pred_theta.new_tensor(0.0)
    if rand_endpoint_weight > 0 and "source" in batch:
        rand_mask = torch.tensor(
            [source == "rand" for source in batch["source"]],
            device=pred_theta.device,
            dtype=torch.bool,
        )
        if bool(rand_mask.any()):
            rand_weights = sample_weights[rand_mask] if sample_weights is not None else None
            rand_endpoint = weighted_mean(per_sample_reg[rand_mask], rand_weights)
            total = total + rand_endpoint_weight * rand_endpoint
    anchor_mixture_weight = float(anchor_mixture_cfg.get("loss_weight", 0.0))
    if anchor_mixture_weight > 0:
        total = total + anchor_mixture_weight * raw_anchor_mixture_reg
    if anchor_reg_weight > 0:
        total = total + anchor_reg_weight * anchor_reg
    if consistency_weight > 0:
        total = total + consistency_weight * direct_anchor_consistency
    metrics = {
        "loss_reg": float(reg_plain.detach().cpu()),
        "loss_reg_objective": float(reg_objective.detach().cpu()),
        "loss_reg_lookup": float(anchor_reg.detach().cpu()),
        "loss_reg_direct": float(regression_loss(out["theta_direct"], theta, config).detach().cpu()),
        "loss_reg_anchor_residual": float(regression_loss(out["theta"], theta, config).detach().cpu()),
        "loss_anchor_reg": float(anchor_reg.detach().cpu()),
        "loss_direct_anchor_consistency": float(direct_anchor_consistency.detach().cpu()),
        "loss_raw_anchor_mixture": float(raw_anchor_mixture_reg.detach().cpu()),
        "loss_blend_prediction_consistency": float(blend_prediction.detach().cpu()),
        "loss_rand_endpoint_consistency": float(rand_endpoint.detach().cpu()),
        "loss_vae_kl": float(vae_kl.detach().cpu()),
        "loss_vae_kl_weight": float(vae_kl_weight),
        "loss_uncertainty_reg": float(uncertainty_reg.detach().cpu()),
        "sample_weight_mean": float((sample_weights.mean() if sample_weights is not None else torch.tensor(1.0)).detach().cpu()),
        "sample_weight_max": float((sample_weights.max() if sample_weights is not None else torch.tensor(1.0)).detach().cpu()),
        "loss_cls": float(cls.detach().cpu()),
        "loss_cls_weight": float(cls_weight),
        "acc": float(valid_acc.detach().cpu()),
        "soft_target_mass": float(soft_target_mass.detach().cpu()),
        "class_fraction": float(valid_class_mask.float().mean().detach().cpu()),
    }
    metrics.update(cls_metrics)
    if log_var is not None:
        clipped_log_var = log_var.detach().clamp(
            min=float(config["loss"].get("uncertainty_log_var_min", -5.0)),
            max=float(config["loss"].get("uncertainty_log_var_max", 2.0)),
        )
        metrics["uncertainty_log_var_mean"] = float(clipped_log_var.mean().cpu())
        metrics["uncertainty_log_var_min"] = float(clipped_log_var.min().cpu())
        metrics["uncertainty_log_var_max"] = float(clipped_log_var.max().cpu())

    if epoch >= int(config["loss"]["warmup_epochs"]):
        sigma = float(config["loss"]["contrast_sigma"])
        temp = float(config["loss"]["contrast_temperature"])
        param_contrast_weight = float(config["loss"]["param_contrast_weight"])
        audio_param_weight = float(config["loss"]["audio_param_weight"])
        latent_mix_weight = float(config["loss"]["latent_mix_weight"])
        multi_impact_weight = float(config["loss"]["multi_impact_weight"])
        blend_latent_weight = float(config["loss"].get("blend_latent_consistency_weight", 0.0))
        blend_param_weight = float(config["loss"].get("blend_param_consistency_weight", 0.0))
        if param_contrast_weight > 0:
            pc = soft_parameter_contrastive(out["q_audio"], theta, sigma=sigma, temperature=temp)
            total = total + param_contrast_weight * pc
            metrics["loss_param_contrast"] = float(pc.detach().cpu())
        if audio_param_weight > 0:
            ap = multi_positive_cross_modal_loss(out["q_audio"], model.encode_params(theta), theta, sigma=sigma, temperature=temp)
            total = total + audio_param_weight * ap
            metrics["loss_audio_param"] = float(ap.detach().cpu())
        if latent_mix_weight > 0:
            mix = latent_interpolation_loss(model, out["set_embedding"], theta, prediction_key=prediction_key)
            total = total + latent_mix_weight * mix
            metrics["loss_latent_mix"] = float(mix.detach().cpu())
        if multi_impact_weight > 0:
            mi = multi_impact_consistency(out["clip_embeddings"], batch["hit_mask"])
            total = total + multi_impact_weight * mi
            metrics["loss_multi_impact"] = float(mi.detach().cpu())
        if (blend_latent_weight > 0 or blend_param_weight > 0) and class_probs is not None and "source" in batch:
            blend_mask = torch.tensor(
                [source == "blend" for source in batch["source"]],
                device=out["q_audio"].device,
                dtype=torch.bool,
            )
            if bool(blend_mask.any()):
                class_probs_device = class_probs.to(out["q_audio"].device)
                endpoint_q = model.encode_params(model.anchor_table_norm)
                blend_q_target = F.normalize(class_probs_device @ endpoint_q, dim=-1)
                blend_latent = (1.0 - (out["q_audio"][blend_mask] * blend_q_target[blend_mask]).sum(dim=-1)).mean()

                endpoint_theta_target = class_probs_device @ model.anchor_table_norm
                blend_param = F.smooth_l1_loss(pred_theta[blend_mask], endpoint_theta_target[blend_mask])
            else:
                blend_latent = total.new_tensor(0.0)
                blend_param = total.new_tensor(0.0)
            if blend_latent_weight > 0:
                total = total + blend_latent_weight * blend_latent
                metrics["loss_blend_latent_consistency"] = float(blend_latent.detach().cpu())
            if blend_param_weight > 0:
                total = total + blend_param_weight * blend_param
                metrics["loss_blend_param_consistency"] = float(blend_param.detach().cpu())
    metrics["loss"] = float(total.detach().cpu())
    return total, metrics


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def maybe_init_wandb(config: dict[str, Any], run_dir: str):
    if not config["runtime"].get("use_wandb", False):
        return None
    if wandb is None:
        raise ImportError("wandb is not installed in the active environment.")
    run_name = config["runtime"].get("wandb_run_name") or config["runtime"].get("run_name")
    return wandb.init(
        entity=config["runtime"].get("wandb_entity"),
        project=config["runtime"].get("wandb_project", "apcl"),
        name=run_name,
        config=config,
        dir=run_dir,
    )


def build_wandb_payload(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr_metrics: dict[str, float],
    best_val: float,
    best_val_cls_soft_ce: float,
) -> dict[str, float]:
    payload = {f"train/{key}": value for key, value in train_metrics.items()}
    payload.update({f"val/{key}": value for key, value in val_metrics.items()})
    payload.update({f"optim/{key}": value for key, value in lr_metrics.items()})
    payload["best/val_loss_reg"] = float(best_val)
    payload["best/val_loss_cls_soft_ce"] = float(best_val_cls_soft_ce)
    return payload


def save_checkpoint(
    output_path: str,
    model: APCLModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    target_table_raw: torch.Tensor,
    target_table_norm: torch.Tensor,
    class_anchor_table_raw: torch.Tensor,
    class_anchor_table_norm: torch.Tensor,
    class_materials: list[str],
    class_target_indices: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    history: list[dict[str, Any]],
    epoch: int,
    best_val: float,
    best_val_cls_soft_ce: float = float("inf"),
) -> None:
    config_for_save = copy.deepcopy(config)
    config_for_save.pop("_runtime_tensors", None)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config_for_save,
            "anchor_table_raw": class_anchor_table_raw,
            "anchor_table_norm": class_anchor_table_norm.detach().cpu(),
            "target_table_raw": target_table_raw,
            "target_table_norm": target_table_norm.detach().cpu(),
            "class_anchor_materials": class_materials,
            "class_target_indices": class_target_indices.detach().cpu(),
            "target_mean": target_mean,
            "target_std": target_std,
            "history": history,
            "best_val_loss_reg": best_val,
            "best_val_loss_cls_soft_ce": best_val_cls_soft_ce,
        },
        output_path,
    )


def build_optimizer(model: APCLModel, optim_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg["weight_decay"])
    head_lr_cfg = optim_cfg.get("head_lr")
    audio_lr_cfg = optim_cfg.get("audio_encoder_lr")

    if head_lr_cfg is None and audio_lr_cfg is None:
        params = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    head_lr = lr if head_lr_cfg is None else float(head_lr_cfg)
    audio_lr = lr if audio_lr_cfg is None else float(audio_lr_cfg)
    head_params = []
    audio_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("audio_encoder."):
            audio_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    if audio_params:
        param_groups.append({"params": audio_params, "lr": audio_lr, "name": "audio_encoder"})
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def get_lr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    metrics = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group{idx}"))
        metrics[f"lr_{name}"] = float(group["lr"])
    if optimizer.param_groups:
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    return metrics


@torch.no_grad()
def evaluate(model: APCLModel, loader: DataLoader, target_table_norm: torch.Tensor, class_target_indices: torch.Tensor, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    prediction_key = prediction_key_from_config(config)
    for batch in loader:
        batch = move_batch(batch, device)
        _, metrics = compute_losses(model, batch, target_table_norm, class_target_indices, config, epoch=10**9)
        if "theta_raw" in batch:
            target_mean = config["_runtime_tensors"]["target_mean"].to(batch["theta_raw"].device)
            target_std = config["_runtime_tensors"]["target_std"].to(batch["theta_raw"].device)
            theta = (batch["theta_raw"] - target_mean) / target_std
        else:
            labels = batch["material_index"]
            theta = target_table_norm.index_select(0, labels)
        pred = model(batch.get("waveforms"), batch["hit_mask"], clip_embeddings=batch.get("clip_embeddings"))[prediction_key]
        mae = (pred - theta).abs().mean(dim=0)
        for idx, name in enumerate(target_names_for_config(config)):
            metrics[f"mae_norm_{name}"] = float(mae[idx].detach().cpu())
        rows.append(metrics)
    return summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/impactmat/default.json")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint saved by train.py, usually latest.pt.")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        resume_checkpoint = torch.load(args.resume, map_location=map_location)
        config = deep_update(DEFAULT_CONFIG, resume_checkpoint.get("config", config))
    if args.use_wandb:
        config["runtime"]["use_wandb"] = True
    if args.wandb_entity is not None:
        config["runtime"]["wandb_entity"] = args.wandb_entity
    if args.wandb_project is not None:
        config["runtime"]["wandb_project"] = args.wandb_project
    if args.wandb_run_name is not None:
        config["runtime"]["wandb_run_name"] = args.wandb_run_name
    if args.run_name is not None:
        config["runtime"]["run_name"] = args.run_name
    if args.epochs is not None:
        config["optim"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["optim"]["batch_size"] = int(args.batch_size)
    set_seed(int(config["runtime"]["seed"]))

    device_name = config["runtime"]["device"]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(config["runtime"]["output_root"], config["runtime"]["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    train_loader, val_loader, test_loaders, counts = build_loaders(config)
    target_table_raw = build_anchor_table(
        target_names=target_names_for_config(config),
        eta_reference_hz=float(config["model"]["eta_reference_hz"]),
    )
    target_table_norm, target_mean, target_std = normalize_anchor_table_for_config(target_table_raw, config)
    class_anchor_table_raw, class_anchor_table_norm, class_materials = build_class_anchor_tables(target_table_raw, target_table_norm, config)
    class_target_indices = class_target_indices_for_config(config)
    if resume_checkpoint is not None:
        target_table_raw = resume_checkpoint.get("target_table_raw", target_table_raw)
        target_table_norm = resume_checkpoint.get("target_table_norm", target_table_norm)
        class_anchor_table_raw = resume_checkpoint.get("anchor_table_raw", class_anchor_table_raw)
        class_anchor_table_norm = resume_checkpoint.get("anchor_table_norm", class_anchor_table_norm)
        class_materials = list(resume_checkpoint.get("class_anchor_materials", class_materials))
        class_target_indices = resume_checkpoint.get("class_target_indices", class_target_indices)
        target_mean = resume_checkpoint.get("target_mean", target_mean)
        target_std = resume_checkpoint.get("target_std", target_std)
    target_table_norm = target_table_norm.to(device)
    class_anchor_table_norm = class_anchor_table_norm.to(device)
    class_target_indices = class_target_indices.to(device)

    model = APCLModel(
        pann_repo_root=config["model"]["pann_repo_root"],
        pann_checkpoint_path=config["model"]["pann_checkpoint_path"],
        anchor_table_norm=class_anchor_table_norm,
        sample_rate=int(config["data"]["sample_rate"]),
        audio_dim=int(config["model"]["audio_dim"]),
        projection_dim=int(config["model"]["projection_dim"]),
        audio_projection_hidden_dim=int(config["model"]["audio_projection_hidden_dim"]),
        param_encoder_hidden_dim=int(config["model"]["param_encoder_hidden_dim"]),
        freeze_audio_encoder=bool(config["model"]["freeze_audio_encoder"]),
        audio_encoder_config=audio_encoder_config_for_model(config),
        anchor_candidate_mask=anchor_candidate_mask_for_config(config),
        dropout=float(config["model"]["dropout"]),
        predict_uncertainty=bool(config["model"].get("predict_uncertainty", False)),
        predict_vae=bool(config["model"].get("predict_vae", False)),
        vae_latent_dim=config["model"].get("vae_latent_dim"),
        vae_hidden_dim=config["model"].get("vae_hidden_dim"),
    ).to(device)
    optimizer = build_optimizer(model, config["optim"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["optim"]["epochs"])),
        eta_min=float(config["optim"]["min_lr"]),
    )

    best_val = float("inf")
    best_val_cls_soft_ce = float("inf")
    history = []
    start_epoch = 0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        history = list(resume_checkpoint.get("history", []))
        best_val = float(resume_checkpoint.get("best_val_loss_reg", best_val))
        best_val_cls_soft_ce = float(resume_checkpoint.get("best_val_loss_cls_soft_ce", best_val_cls_soft_ce))
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        print(
            f"Resuming from {args.resume}: "
            f"start_epoch={start_epoch}, "
            f"best_val_loss_reg={best_val:.6g}, "
            f"best_val_loss_cls_soft_ce={best_val_cls_soft_ce:.6g}"
        )
    print(f"Dataset A groups: {counts}")
    print(f"Materials: {', '.join(MATERIAL_NAMES)}")
    print(f"Target parameters: {', '.join(target_names_for_config(config))}")
    print(f"Class anchors: {', '.join(class_materials)}")
    aliases = material_class_aliases_for_config(config)
    if aliases:
        print(f"Material class aliases: {aliases}")
        print("Regression targets: original material parameters")
    held_out_materials = get_held_out_materials(config)
    if held_out_materials:
        print(f"Seen materials: {', '.join(get_seen_materials(config))}")
        print(f"Held-out materials: {', '.join(held_out_materials)}")
        print(f"Remove held-out anchors from theta_anchor candidates: {bool(config['data'].get('material_holdout', {}).get('remove_from_anchor_candidates', False))}")
    wandb_run = maybe_init_wandb(config, run_dir)
    config["_runtime_tensors"] = {
        "target_mean": target_mean.detach(),
        "target_std": target_std.detach(),
        "class_anchor_table_raw": class_anchor_table_raw.detach(),
    }
    for epoch in range(start_epoch, int(config["optim"]["epochs"])):
        model.train()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        train_rows = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_losses(model, batch, target_table_norm, class_target_indices, config, epoch)
            loss.backward()
            optimizer.step()
            train_rows.append(metrics)
        scheduler.step()

        train_metrics = summarize(train_rows)
        val_metrics = evaluate(model, val_loader, target_table_norm, class_target_indices, config, device) if len(val_loader.dataset) else {}
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        val_key = val_metrics.get("loss_reg", train_metrics["loss_reg"])
        val_soft_ce_key = val_metrics.get("loss_cls_soft_ce", val_metrics.get("loss_cls", float("inf")))
        improved_soft_ce = val_soft_ce_key < best_val_cls_soft_ce
        if improved_soft_ce:
            best_val_cls_soft_ce = val_soft_ce_key
        if val_key < best_val:
            best_val = val_key
            save_checkpoint(
                os.path.join(run_dir, "best.pt"),
                model,
                optimizer,
                scheduler,
                config,
                target_table_raw,
                target_table_norm,
                class_anchor_table_raw,
                class_anchor_table_norm,
                class_materials,
                class_target_indices,
                target_mean,
                target_std,
                history,
                epoch,
                best_val,
                best_val_cls_soft_ce,
            )
        if improved_soft_ce:
            save_checkpoint(
                os.path.join(run_dir, "best_soft_ce.pt"),
                model,
                optimizer,
                scheduler,
                config,
                target_table_raw,
                target_table_norm,
                class_anchor_table_raw,
                class_anchor_table_norm,
                class_materials,
                class_target_indices,
                target_mean,
                target_std,
                history,
                epoch,
                best_val,
                best_val_cls_soft_ce,
            )
        save_checkpoint(
            os.path.join(run_dir, "latest.pt"),
            model,
            optimizer,
            scheduler,
            config,
            target_table_raw,
            target_table_norm,
            class_anchor_table_raw,
            class_anchor_table_norm,
            class_materials,
            class_target_indices,
            target_mean,
            target_std,
            history,
            epoch,
            best_val,
            best_val_cls_soft_ce,
        )
        if wandb_run is not None:
            wandb_run.log(
                build_wandb_payload(
                    train_metrics,
                    val_metrics,
                    get_lr_metrics(optimizer),
                    best_val,
                    best_val_cls_soft_ce,
                ),
                step=epoch,
            )
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} val_reg={val_key:.4f} acc={train_metrics['acc']:.3f}")

    test_metrics = {
        split_name: evaluate(model, loader, target_table_norm, class_target_indices, config, device) if len(loader.dataset) else {}
        for split_name, loader in test_loaders.items()
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    if wandb_run is not None:
        test_payload = {}
        for split_name, metrics in test_metrics.items():
            test_payload.update({f"test/{split_name}/{key}": value for key, value in metrics.items()})
        wandb_run.log(test_payload, step=int(config["optim"]["epochs"]))
        wandb_run.finish()
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
