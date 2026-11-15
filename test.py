from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from apcl.data import DatasetAImpactGroups, DatasetBImpactGroups, collate_impact_groups, discover_dataset_a_groups, discover_dataset_b_groups, split_groups_by_object
from apcl.materials import TARGET_NAMES, build_anchor_table
from apcl.model import APCLModel
from train import (
    DEFAULT_CONFIG,
    audio_encoder_config_for_model,
    anchor_candidate_mask_for_config,
    class_target_indices_for_config,
    deep_update,
    evaluate,
    get_eval_k_values,
    load_config,
    normalize_anchor_table_for_config,
    split_groups_for_material_holdout,
    target_names_for_config,
)


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[APCLModel, torch.Tensor, torch.Tensor, dict[str, Any]]:
    config = deep_update(DEFAULT_CONFIG, checkpoint.get("config", {}))
    raw_table = checkpoint.get("anchor_table_raw")
    if raw_table is None:
        raw_table = build_anchor_table(
            target_names=target_names_for_config(config),
            eta_reference_hz=float(config["model"]["eta_reference_hz"]),
        )
    anchor_table_norm = checkpoint.get("anchor_table_norm")
    if anchor_table_norm is None:
        anchor_table_norm, _, _ = normalize_anchor_table_for_config(raw_table, config)
    target_table_norm = checkpoint.get("target_table_norm")
    if target_table_norm is None:
        target_table_norm = anchor_table_norm
    anchor_table_norm = anchor_table_norm.to(device)
    target_table_norm = target_table_norm.to(device)
    class_target_indices = checkpoint.get("class_target_indices")
    if class_target_indices is None:
        class_target_indices = class_target_indices_for_config(config)
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
        audio_encoder_config=audio_encoder_config_for_model(config),
        anchor_candidate_mask=anchor_candidate_mask_for_config(config),
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


def make_eval_loader(groups, config: dict[str, Any], k: int) -> DataLoader:
    cache_cfg = config["data"].get("embedding_cache", {})
    embedding_cache_root = cache_cfg.get("root") if cache_cfg.get("enabled", False) else None
    dataset_kind = str(config["data"].get("dataset", "dataset_a")).lower()
    dataset_cls = DatasetBImpactGroups if dataset_kind in {"dataset_b", "b", "rand_blend", "blend"} else DatasetAImpactGroups
    dataset = dataset_cls(
        groups,
        k_choices=[int(k)],
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
        random_hits=False,
        embedding_cache_root=embedding_cache_root,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["optim"]["num_workers"]),
        collate_fn=collate_impact_groups,
        pin_memory=torch.cuda.is_available(),
    )


def dataset_a_eval_groups(root: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    groups = discover_dataset_a_groups(root)
    _, _, test_groups_by_name, _ = split_groups_for_material_holdout(groups, config)
    return test_groups_by_name


def read_object_ids_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def dataset_b_eval_groups(
    root: str,
    config: dict[str, Any],
) -> dict[str, list[Any]]:
    data_cfg = config["data"]
    groups = discover_dataset_b_groups(
        root,
        include_rand=bool(data_cfg.get("include_rand", True)),
        include_blend=bool(data_cfg.get("include_blend", True)),
        complete_object_ids=read_object_ids_file(data_cfg.get("complete_object_ids_file")),
        eta_reference_hz=float(config["model"]["eta_reference_hz"]),
        target_names=target_names_for_config(config),
    )
    _, _, test_groups_by_name, _ = split_groups_for_material_holdout(groups, config)
    return test_groups_by_name


def filter_dataset_b_eval_sources(
    eval_groups: dict[str, list[Any]],
    sources: list[str] | None,
) -> dict[str, list[Any]]:
    if sources is None:
        return eval_groups

    selected: dict[str, list[Any]] = {}
    for split_name, groups in eval_groups.items():
        for source in sources:
            if source == "all":
                selected[split_name] = groups
                continue
            selected[f"{split_name}_{source}"] = [
                group for group in groups if getattr(group, "source", None) == source
            ]
    return selected


def evaluate_dataset(
    dataset_name: str,
    groups,
    model: APCLModel,
    target_table_norm: torch.Tensor,
    class_target_indices: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    metrics_by_k = {}
    for k in get_eval_k_values(config["data"]):
        loader = make_eval_loader(groups, config, int(k))
        metrics_by_k[f"K{k}"] = evaluate(model, loader, target_table_norm, class_target_indices, config, device) if len(loader.dataset) else {}
    return {
        "dataset": dataset_name,
        "num_groups": len(groups),
        "metrics": metrics_by_k,
    }


def flatten_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset_name, dataset_result in results.items():
        for k_name, metrics in dataset_result["metrics"].items():
            row = {
                "dataset": dataset_name,
                "k": k_name,
                "num_groups": dataset_result["num_groups"],
            }
            row.update(metrics)
            rows.append(row)
    return rows


def default_dataset_name(dataset_kind: str, split_name: str) -> str:
    if dataset_kind in {"dataset_b", "b", "rand_blend", "blend"}:
        return f"dataset_b_{split_name}"
    return "dataset_a" if split_name == "test" else f"dataset_a_{split_name}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["dataset", "k", "num_groups", "loss", "loss_reg", "loss_cls", "acc", "soft_target_mass"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "dataset",
        "k",
        "num_groups",
        "loss_reg",
        "loss_cls",
        "loss_cls_soft_ce",
        "loss",
        "acc",
        "soft_target_mass",
        "blend_endpoint_recall_at_2",
        "blend_off_support_mass",
        "blend_ratio_mae",
    ]
    columns.extend(
        sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("mae_norm_")
            }
        )
    )
    columns = [column for column in columns if any(column in row for row in rows)]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_b_08_no_holdout/f_v1_blend_cls025_contrast001_k4_b128_e200.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--dataset-a-root", default=None)
    parser.add_argument("--dataset-b-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-k-values", nargs="+", type=int, default=None)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=None,
        choices=["test", "test_seen", "test_unseen", "test_all", "test_rand", "test_blend"],
    )
    parser.add_argument(
        "--dataset-b-sources",
        nargs="+",
        default=None,
        choices=["all", "rand", "blend"],
        help="Dataset B only: evaluate all groups, rand-only groups, and/or blend-only groups.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    test_cfg = config.get("test", {})
    run_dir = args.run_dir or os.path.join(config["runtime"]["output_root"], config["runtime"]["run_name"])
    checkpoint_name = args.checkpoint_name or test_cfg.get("checkpoint_name", "best.pt")
    checkpoint_path = args.checkpoint or os.path.join(run_dir, checkpoint_name)
    device_name = args.device or config["runtime"]["device"]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model, target_table_norm, class_target_indices, train_config = build_model_from_checkpoint(checkpoint, device)
    # Keep the checkpoint's training/model configuration as the source of truth.
    # CLI config is only used for test-time options such as dataset_b_root or
    # output_dir, otherwise EAT/SSLAM checkpoints can be accidentally rebuilt as
    # the default PANN model.
    eval_config = train_config
    if args.eval_k_values is not None:
        eval_config = deep_update(eval_config, {"data": {"eval_k_values": [int(k) for k in args.eval_k_values]}})
    if args.num_workers is not None:
        eval_config = deep_update(eval_config, {"optim": {"num_workers": int(args.num_workers)}})

    dataset_a_root = args.dataset_a_root or test_cfg.get("dataset_a_root") or eval_config["data"]["dataset_a_root"]
    dataset_b_root = args.dataset_b_root if args.dataset_b_root is not None else test_cfg.get("dataset_b_root")

    results = {}
    dataset_kind = str(eval_config["data"].get("dataset", "dataset_a")).lower()
    if dataset_kind in {"dataset_b", "b", "rand_blend", "blend"}:
        eval_groups = dataset_b_eval_groups(dataset_b_root or dataset_a_root, eval_config)
    else:
        eval_groups = dataset_a_eval_groups(dataset_a_root, eval_config)
    if args.eval_splits is not None:
        requested = set(args.eval_splits)
        eval_groups = {name: groups for name, groups in eval_groups.items() if name in requested}
    if dataset_kind in {"dataset_b", "b", "rand_blend", "blend"}:
        eval_groups = filter_dataset_b_eval_sources(eval_groups, args.dataset_b_sources)
    if not eval_groups:
        raise ValueError(f"No dataset A eval groups selected. Available splits: {sorted(dataset_a_eval_groups(dataset_a_root, eval_config))}")
    for split_name, groups in eval_groups.items():
        dataset_name = default_dataset_name(dataset_kind, split_name)
        results[dataset_name] = evaluate_dataset(
            dataset_name,
            groups,
            model,
            target_table_norm,
            class_target_indices,
            eval_config,
            device,
        )
    if dataset_b_root and dataset_kind not in {"dataset_b", "b", "rand_blend", "blend"}:
        dataset_b_groups = discover_dataset_a_groups(dataset_b_root)
        results["dataset_b"] = evaluate_dataset(
            "dataset_b",
            dataset_b_groups,
            model,
            target_table_norm,
            class_target_indices,
            eval_config,
            device,
        )

    output_dir = Path(args.output_dir or os.path.join(run_dir, test_cfg.get("output_dir", "test_evaluation")))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    rows = flatten_rows(results)
    write_csv(rows, output_dir / "summary_table.csv")
    write_markdown(rows, output_dir / "summary_table.md")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
