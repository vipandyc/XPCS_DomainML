"""Supplementary retraining controls for XPCS parameter inference.

The routines in this file create controlled dataset manifests, retrain existing
model families, and write prediction CSVs that can be plotted from
``notebooks/supplementary_plotting.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata

from inference import plot_phase_diagrams
from train_adv_coral_surrogate import (
    XPCSDataset as CoralSurrogateDataset,
    XPCSNetCoral,
    compute_nonequilibrium_measure,
    train as train_coral_surrogate,
)
from train_vanilla_no_T import (
    VanillaXPCSNet,
    denorm_from_meta,
    load_model as load_vanilla_model,
    train as train_vanilla,
)


PARAM_COLUMNS = ["gamma", "D", "GB_conc"]
NONEQ_COLUMNS = ["nonequilibrium_measure_raw", "nonequilibrium_measure", "unequilibrium_measure"]


def r2_score_np(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(true) & np.isfinite(pred)
    true = true[finite]
    pred = pred[finite]
    if true.size < 2:
        return float("nan")
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def mae_np(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(true) & np.isfinite(pred)
    if not finite.any():
        return float("nan")
    return float(np.mean(np.abs(pred[finite] - true[finite])))


def project_relative(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def write_manifest(root: Path, manifest: pd.DataFrame, project_root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = manifest.copy()
    manifest.to_csv(root / "manifest.csv", index=False)
    if any(col in manifest.columns for col in NONEQ_COLUMNS):
        keep = [col for col in ["id", *PARAM_COLUMNS, "T", "path", "domain", *NONEQ_COLUMNS] if col in manifest.columns]
        manifest[keep].to_csv(root / "manifest_with_non_equ.csv", index=False)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "num_rows": int(len(manifest)),
    }
    (root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")
    return root


def newest_checkpoint(model_dir: Path, pattern: str, existing: set[Path]) -> Path | None:
    candidates = sorted(model_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate not in existing:
            return candidate
    return candidates[0] if candidates else None


def load_simulation_phase_table(sim_root: Path) -> pd.DataFrame:
    candidates = [sim_root / "manifest.csv", sim_root / "manifest_with_non_equ.csv"]
    manifest = pd.read_csv(candidates[0])
    if "nonequilibrium_measure_raw" in manifest.columns or "nonequilibrium_measure" in manifest.columns:
        return manifest
    noneq_path = sim_root / "manifest_with_non_equ.csv"
    if noneq_path.exists():
        noneq = pd.read_csv(noneq_path)
        if len(noneq) == len(manifest):
            for col in NONEQ_COLUMNS:
                if col in noneq.columns:
                    manifest[col] = noneq[col].to_numpy()
        elif "id" in manifest.columns and "id" in noneq.columns:
            add_cols = ["id", *[col for col in NONEQ_COLUMNS if col in noneq.columns]]
            manifest = manifest.merge(noneq[add_cols], on="id", how="left")
    return manifest


def interpolate_predicted_noneq(predictions: pd.DataFrame, sim_root: Path) -> np.ndarray:
    phase = load_simulation_phase_table(sim_root)
    noneq_col = "nonequilibrium_measure_raw" if "nonequilibrium_measure_raw" in phase.columns else "nonequilibrium_measure"
    required = [*PARAM_COLUMNS, noneq_col]
    if not set(required).issubset(phase.columns):
        return np.full(len(predictions), np.nan)

    phase = phase.dropna(subset=required).copy()
    phase = phase.loc[phase["D"] > 0]
    points = np.column_stack(
        [
            phase["gamma"].to_numpy(dtype=float),
            np.log10(phase["D"].to_numpy(dtype=float)),
            phase["GB_conc"].to_numpy(dtype=float),
        ]
    )
    values = phase[noneq_col].to_numpy(dtype=float)
    query_d = np.clip(
        predictions["D_pred"].to_numpy(dtype=float),
        float(phase["D"].min()),
        float(phase["D"].max()),
    )
    query = np.column_stack(
        [
            predictions["gamma_pred"].to_numpy(dtype=float),
            np.log10(query_d),
            predictions["GB_conc_pred"].to_numpy(dtype=float),
        ]
    )
    linear = griddata(points, values, query, method="linear", rescale=True)
    nearest = griddata(points, values, query, method="nearest", rescale=True)
    return np.where(np.isfinite(linear), linear, nearest)


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset_root: Path,
    sim_root_for_noneq: Path,
    output_csv: Path,
    device: torch.device,
) -> pd.DataFrame:
    dataset = CoralSurrogateDataset(dataset_root)
    manifest = dataset.manifest.reset_index(drop=True)
    norm_meta = dataset.norm_meta
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        x, _, y_raw, temperature = sample[:4]
        pred_norm = model(x.unsqueeze(0).to(device), temperature.unsqueeze(0).to(device))
        pred_raw = denorm_from_meta(pred_norm.squeeze(0), norm_meta, device=device).detach().cpu()
        manifest_row = manifest.iloc[index]
        rows.append(
            {
                "dataset_index": index,
                "id": manifest_row.get("id", index),
                "path": manifest_row.get("path"),
                "T": float(temperature.item()),
                "gamma_true": float(y_raw[0].item()),
                "D_true": float(y_raw[1].item()),
                "GB_conc_true": float(y_raw[2].item()),
                "gamma_pred": float(pred_raw[0].item()),
                "D_pred": float(pred_raw[1].item()),
                "GB_conc_pred": float(pred_raw[2].item()),
                "S_noneq_true": float(
                    manifest_row.get(
                        "nonequilibrium_measure_raw",
                        manifest_row.get("nonequilibrium_measure", np.nan),
                    )
                ),
            }
        )
    predictions = pd.DataFrame(rows)
    predictions["S_noneq_pred"] = interpolate_predicted_noneq(predictions, sim_root_for_noneq)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    return predictions


def summarize_predictions(frame: pd.DataFrame, output_csv: Path, label: str) -> pd.DataFrame:
    specs = [
        ("gamma", "gamma_true", "gamma_pred", 1e-18),
        ("D", "D_true", "D_pred", 1e22),
        ("GB_conc", "GB_conc_true", "GB_conc_pred", 1.0),
        ("S_noneq", "S_noneq_true", "S_noneq_pred", 1.0),
    ]
    rows = []
    for quantity, true_col, pred_col, scale in specs:
        if true_col not in frame.columns or pred_col not in frame.columns:
            continue
        rows.append(
            {
                "label": label,
                "quantity": quantity,
                "R2": r2_score_np(frame[true_col].to_numpy() * scale, frame[pred_col].to_numpy() * scale),
                "MAE": mae_np(frame[true_col].to_numpy() * scale, frame[pred_col].to_numpy() * scale),
                "n": int(len(frame)),
            }
        )
    summary = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary


def make_label_permutation_dataset(sim_root: Path, output_root: Path, seed: int, project_root: Path) -> Path:
    rng = np.random.default_rng(seed)
    manifest = pd.read_csv(sim_root / "manifest.csv")
    shuffled = manifest.copy()
    permutation = rng.permutation(len(shuffled))
    shuffled[[f"{col}_original" for col in PARAM_COLUMNS]] = shuffled[PARAM_COLUMNS]
    shuffled[PARAM_COLUMNS] = manifest.iloc[permutation][PARAM_COLUMNS].to_numpy()
    shuffled["label_permutation_source_index"] = permutation
    dataset_root = output_root / f"simulation_label_permutation_seed{seed}"
    write_manifest(dataset_root, shuffled, project_root)
    shuffled[["id", "label_permutation_source_index", *PARAM_COLUMNS, *[f"{col}_original" for col in PARAM_COLUMNS]]].to_csv(
        dataset_root / "label_permutation_map.csv",
        index=False,
    )
    return dataset_root


def make_snoneq_shuffle_dataset(exp_root: Path, output_root: Path, seed: int, project_root: Path) -> Path:
    rng = np.random.default_rng(seed)
    manifest = pd.read_csv(exp_root / "manifest.csv")
    shuffled = manifest.copy()
    for col in NONEQ_COLUMNS:
        if col in shuffled.columns:
            shuffled[f"{col}_original"] = shuffled[col]
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    dataset_root = output_root / f"experiment_snoneq_shuffle_seed{seed}"
    write_manifest(dataset_root, shuffled, project_root)
    return dataset_root


def holdout_mask_from_args(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.holdout_query:
        selected_index = manifest.query(args.holdout_query, engine="python").index
        return manifest.index.isin(selected_index)
    values = manifest[args.holdout_column].astype(float)
    low = float(values.quantile(args.holdout_lower_quantile))
    high = float(values.quantile(args.holdout_upper_quantile))
    return values.between(low, high, inclusive="both")


def make_ood_datasets(sim_root: Path, output_root: Path, args: argparse.Namespace, project_root: Path) -> tuple[Path, Path]:
    manifest = pd.read_csv(sim_root / "manifest.csv")
    mask = holdout_mask_from_args(manifest, args)
    if mask.sum() == 0 or (~mask).sum() == 0:
        raise ValueError("OOD split must leave at least one training row and one holdout row")
    split_root = output_root / f"simulation_ood_{args.holdout_column}_q{args.holdout_lower_quantile:g}-{args.holdout_upper_quantile:g}_seed{args.seed}"
    train_root = split_root / "train_in_domain"
    holdout_root = split_root / "holdout"
    train_manifest = manifest.loc[~mask].copy()
    holdout_manifest = manifest.loc[mask].copy()
    train_manifest["ood_split"] = "train_in_domain"
    holdout_manifest["ood_split"] = "holdout"
    write_manifest(train_root, train_manifest, project_root)
    write_manifest(holdout_root, holdout_manifest, project_root)
    pd.DataFrame(
        {
            "split": ["train_in_domain", "holdout"],
            "count": [int((~mask).sum()), int(mask.sum())],
            "holdout_column": args.holdout_column,
            "holdout_lower_quantile": args.holdout_lower_quantile,
            "holdout_upper_quantile": args.holdout_upper_quantile,
            "holdout_query": args.holdout_query,
        }
    ).to_csv(split_root / "split_summary.csv", index=False)
    return train_root, holdout_root


def corrupt_tensor(
    tensor: torch.Tensor,
    rng: np.random.Generator,
    noise_std: float,
    background: float,
    gradient: float,
    smear_kernel: int,
    smear_alpha: float,
) -> torch.Tensor:
    x = tensor.to(torch.float32).clone()
    if x.ndim == 3 and x.shape[0] == 1:
        x2 = x.squeeze(0)
    else:
        x2 = x
    if smear_kernel > 1 and smear_alpha > 0:
        if smear_kernel % 2 == 0:
            smear_kernel += 1
        pad = smear_kernel // 2
        blurred = F.avg_pool2d(
            F.pad(x2[None, None], (pad, pad, pad, pad), mode="reflect"),
            kernel_size=smear_kernel,
            stride=1,
        ).squeeze(0).squeeze(0)
        x2 = (1.0 - smear_alpha) * x2 + smear_alpha * blurred
    if background:
        x2 = x2 + float(background)
    if gradient:
        coords = torch.linspace(-1.0, 1.0, x2.shape[-1], dtype=x2.dtype)
        ramp = coords[None, :] + coords[:, None]
        x2 = x2 + float(gradient) * x2.std().clamp_min(1e-8) * ramp
    if noise_std:
        noise = torch.from_numpy(rng.normal(0.0, float(noise_std), size=tuple(x2.shape))).to(dtype=x2.dtype)
        x2 = x2 + noise * x2.std().clamp_min(1e-8)
    finite = x2[torch.isfinite(x2)]
    if finite.numel() == 0:
        x2 = torch.zeros_like(x2)
    else:
        x2 = torch.nan_to_num(
            x2,
            nan=float(finite.mean()),
            posinf=float(finite.max()),
            neginf=float(finite.min()),
        )
    return x2.unsqueeze(0)


def make_synthetic_shift_dataset(sim_root: Path, output_root: Path, args: argparse.Namespace, project_root: Path) -> Path:
    rng = np.random.default_rng(args.seed)
    manifest = pd.read_csv(sim_root / "manifest.csv")
    if args.synthetic_count is not None and args.synthetic_count < len(manifest):
        selected = manifest.sample(n=args.synthetic_count, random_state=args.seed).sort_index().copy()
    else:
        selected = manifest.copy()
    dataset_root = output_root / f"synthetic_shift_seed{args.seed}"
    tensor_dir = dataset_root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_index, (_, row) in enumerate(selected.iterrows()):
        source_path = project_root / str(row["path"]) if not Path(str(row["path"])).is_absolute() else Path(str(row["path"]))
        clean = torch.load(source_path, map_location="cpu", weights_only=True)
        shifted = corrupt_tensor(
            clean,
            rng=rng,
            noise_std=args.synthetic_noise_std,
            background=args.synthetic_background,
            gradient=args.synthetic_gradient,
            smear_kernel=args.synthetic_smear_kernel,
            smear_alpha=args.synthetic_smear_alpha,
        )
        out_path = tensor_dir / f"{output_index:06d}.pt"
        torch.save(shifted, out_path)
        out_row = row.copy()
        out_row["source_sim_path"] = row["path"]
        out_row["source_sim_id"] = row.get("id", output_index)
        out_row["path"] = project_relative(out_path, project_root)
        out_row["domain"] = "experiment"
        out_row["id"] = output_index
        out_row["nonequilibrium_measure_raw"] = compute_nonequilibrium_measure(shifted)
        out_row["nonequilibrium_measure"] = out_row["nonequilibrium_measure_raw"]
        rows.append(out_row)
    synthetic_manifest = pd.DataFrame(rows)
    write_manifest(dataset_root, synthetic_manifest, project_root)
    (dataset_root / "synthetic_shift_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source_sim_root": str(sim_root),
                "synthetic_count": int(len(synthetic_manifest)),
                "noise_std": args.synthetic_noise_std,
                "background": args.synthetic_background,
                "gradient": args.synthetic_gradient,
                "smear_kernel": args.synthetic_smear_kernel,
                "smear_alpha": args.synthetic_smear_alpha,
            },
            indent=2,
        ),
        encoding="ascii",
    )
    return dataset_root


def train_vanilla_control(dataset_root: Path, args: argparse.Namespace, run_root: Path) -> tuple[VanillaXPCSNet, Path | None]:
    model_dir = run_root / "models"
    runs_dir = run_root / "runs"
    existing = set(model_dir.glob("Vanilla_XPCS_no_T_best_*.pt"))
    model = VanillaXPCSNet(predictor_output_activation="sigmoid")
    trained = train_vanilla(
        model,
        sim_root=dataset_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.vanilla_learning_rate,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        num_workers=args.num_workers,
        device=torch.device(args.device),
        log_pardir=runs_dir,
        model_path=model_dir,
    )
    return trained, newest_checkpoint(model_dir, "Vanilla_XPCS_no_T_best_*.pt", existing)


def train_coral_surrogate_control(
    sim_root: Path,
    exp_root: Path,
    args: argparse.Namespace,
    run_root: Path,
    init_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[XPCSNetCoral, Path | None]:
    model_dir = run_root / "models"
    runs_dir = run_root / "runs"
    existing = set(model_dir.glob("XPCS_coral_surrogate_no_T_best_*.pt"))
    model = XPCSNetCoral(predictor_output_activation="sigmoid")
    trained = train_coral_surrogate(
        model,
        sim_root=sim_root,
        exp_root=exp_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.coral_learning_rate,
        coral_weight=args.coral_weight,
        surrogate_weight=args.surrogate_weight,
        surrogate_learning_rate=args.surrogate_learning_rate,
        surrogate_pretrain_epochs=args.surrogate_pretrain_epochs,
        surrogate_pretrain_patience=args.surrogate_pretrain_patience,
        surrogate_checkpoint_path=args.surrogate_checkpoint_path,
        force_surrogate_retrain=args.force_surrogate_retrain,
        surrogate_loss_type=args.surrogate_loss,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        num_workers=args.num_workers,
        init_state_dict=init_state_dict,
        device=torch.device(args.device),
        log_pardir=runs_dir,
        model_path=model_dir,
    )
    return trained, newest_checkpoint(model_dir, "XPCS_coral_surrogate_no_T_best_*.pt", existing)


def evaluate_on_raw_experiments(
    model: XPCSNetCoral,
    args: argparse.Namespace,
    results_dir: Path,
    project_root: Path,
    model_name: str = "adv",
) -> None:
    from run_all import (
        PreparedExperimentDataset,
        filter_files,
        iter_raw_experiment_files,
        maybe_limit_dataset,
        prepare_experiment_shots,
        predict_samples,
        write_embedding_diagnostics,
        write_results_csvs,
    )
    from train_adv_no_T import XPCSDataset as EvalSimulationDataset
    from utils import calc_umap, plot_cluster

    raw_files = filter_files(iter_raw_experiment_files(args.exp_data_dir), args.files)
    if not raw_files:
        print("[evaluate-exp] no raw experiment files selected")
        return
    samples = prepare_experiment_shots(
        raw_files=raw_files,
        shot_indices=args.shot_indices,
        crop_size=args.crop_size,
        coarse_size=args.coarse_size,
        crop_step=args.crop_step,
        crop_policy=args.eval_crop_policy,
        no_t=True,
        cache_dir=args.experiment_shot_cache_dir,
    )
    exp_dataset = PreparedExperimentDataset(samples)
    device = torch.device(args.device)
    predictions = predict_samples(
        model=model,
        dataset=exp_dataset,
        batch_size=args.eval_batch_size,
        device=device,
        denorm_fn=denorm_from_meta,
    )
    adv_coords = None
    adv_sim_count = 0
    if not args.skip_umap:
        sim_dataset = maybe_limit_dataset(EvalSimulationDataset(args.sim_root), args.umap_sim_limit)
        coords, domain_labels = calc_umap(
            model,
            sim_dataset,
            exp_dataset,
            device=device,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            init=args.umap_init,
            random_state=args.umap_random_state,
        )
        adv_coords = coords
        adv_sim_count = len(sim_dataset)
        plot_cluster(coords, domain_labels, results_dir / "UMAP.pdf")
        write_embedding_diagnostics(coords, domain_labels, samples, results_dir, model_suffix=model_name)
    write_results_csvs(
        samples=samples,
        adv_predictions=predictions,
        vanilla_predictions=None,
        adv_coords=adv_coords,
        adv_sim_count=adv_sim_count,
        vanilla_coords=None,
        vanilla_sim_count=0,
        results_dir=results_dir,
        skip_umap=args.skip_umap,
        crop_policy=args.eval_crop_policy,
        crop_aggregation=args.eval_crop_aggregation,
    )
    if args.write_phase_diagrams:
        plot_phase_diagrams(
            results_dir=results_dir,
            simulation_manifest=args.phase_diagram_sim_manifest,
            output_dir=results_dir / "phase_diagrams_shot",
            model_names=[model_name],
            aggregate_by="shot",
            shot_index=args.phase_diagram_shot_index,
            split_by_material=False,
            raw_data_dir=args.exp_data_dir,
            crop_size=args.crop_size,
            coarse_size=args.coarse_size,
            range_mode=args.phase_diagram_range_mode,
        )
    os.chdir(project_root)


def load_optional_vanilla_state(args: argparse.Namespace) -> dict[str, torch.Tensor] | None:
    if args.init_vanilla_model_path is None:
        return None
    return torch.load(args.init_vanilla_model_path, weights_only=True, map_location="cpu")


def run_label_permutation(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    dataset_root = make_label_permutation_dataset(args.sim_root, args.control_dataset_root, args.seed, project_root)
    run_root = args.output_root / f"label_permutation_seed{args.seed}"
    vanilla_model, vanilla_checkpoint = train_vanilla_control(dataset_root, args, run_root / "vanilla")
    vanilla_predictions = predict_dataset(
        vanilla_model,
        dataset_root=args.sim_root,
        sim_root_for_noneq=args.sim_root,
        output_csv=run_root / "predictions_clean_sim_truth_vanilla.csv",
        device=torch.device(args.device),
    )
    summaries = [
        summarize_predictions(
            vanilla_predictions,
            run_root / "metrics_clean_sim_truth_vanilla.csv",
            "label_permutation_vanilla",
        )
    ]
    if not args.skip_coral_surrogate:
        coral_model, coral_checkpoint = train_coral_surrogate_control(
            sim_root=dataset_root,
            exp_root=args.exp_root,
            args=args,
            run_root=run_root / "coral_surrogate",
            init_state_dict=None,
        )
        coral_predictions = predict_dataset(
            coral_model,
            dataset_root=args.sim_root,
            sim_root_for_noneq=args.sim_root,
            output_csv=run_root / "predictions_clean_sim_truth.csv",
            device=torch.device(args.device),
        )
        summaries.append(
            summarize_predictions(
                coral_predictions,
                run_root / "metrics_clean_sim_truth_coral_surrogate.csv",
                "label_permutation_coral_surrogate",
            )
        )
    else:
        coral_checkpoint = None
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(run_root / "metrics_clean_sim_truth.csv", index=False)
    metadata = {
        "control": "label_permutation",
        "dataset_root": str(dataset_root),
        "vanilla_checkpoint": str(vanilla_checkpoint),
        "coral_surrogate_checkpoint": str(coral_checkpoint),
    }
    (run_root / "control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")
    print(summary)


def run_snoneq_shuffle(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    exp_root = make_snoneq_shuffle_dataset(args.exp_root, args.control_dataset_root, args.seed, project_root)
    run_root = args.output_root / f"experiment_snoneq_shuffle_seed{args.seed}"
    model, checkpoint = train_coral_surrogate_control(
        sim_root=args.sim_root,
        exp_root=exp_root,
        args=args,
        run_root=run_root,
        init_state_dict=load_optional_vanilla_state(args),
    )
    predictions = predict_dataset(
        model,
        dataset_root=args.sim_root,
        sim_root_for_noneq=args.sim_root,
        output_csv=run_root / "predictions_clean_sim.csv",
        device=torch.device(args.device),
    )
    summarize_predictions(predictions, run_root / "metrics_clean_sim.csv", "experiment_snoneq_shuffle")
    if args.evaluate_experiment:
        evaluate_on_raw_experiments(model, args, run_root / "experiment_inference", project_root)
    metadata = {"control": "experiment_snoneq_shuffle", "dataset_root": str(exp_root), "checkpoint": str(checkpoint)}
    (run_root / "control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")


def run_ood_holdout(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    train_root, holdout_root = make_ood_datasets(args.sim_root, args.control_dataset_root, args, project_root)
    run_root = args.output_root / f"ood_holdout_{args.holdout_column}_seed{args.seed}"
    model, checkpoint = train_vanilla_control(train_root, args, run_root)
    id_predictions = predict_dataset(
        model,
        dataset_root=train_root,
        sim_root_for_noneq=args.sim_root,
        output_csv=run_root / "predictions_in_domain.csv",
        device=torch.device(args.device),
    )
    holdout_predictions = predict_dataset(
        model,
        dataset_root=holdout_root,
        sim_root_for_noneq=args.sim_root,
        output_csv=run_root / "predictions_holdout.csv",
        device=torch.device(args.device),
    )
    summaries = pd.concat(
        [
            summarize_predictions(id_predictions, run_root / "metrics_in_domain.csv", "in_domain"),
            summarize_predictions(holdout_predictions, run_root / "metrics_holdout.csv", "holdout"),
        ],
        ignore_index=True,
    )
    summaries.to_csv(run_root / "metrics_combined.csv", index=False)
    metadata = {"control": "ood_holdout", "train_root": str(train_root), "holdout_root": str(holdout_root), "checkpoint": str(checkpoint)}
    (run_root / "control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")
    print(summaries)


def run_synthetic_shift(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    synthetic_root = make_synthetic_shift_dataset(args.sim_root, args.control_dataset_root, args, project_root)
    run_root = args.output_root / f"synthetic_shift_seed{args.seed}"

    if args.vanilla_model_path is not None:
        vanilla_model = load_vanilla_model(args.vanilla_model_path, device=torch.device(args.device))
        vanilla_checkpoint = args.vanilla_model_path
        init_state = torch.load(args.vanilla_model_path, weights_only=True, map_location="cpu")
    else:
        vanilla_model, vanilla_checkpoint = train_vanilla_control(args.sim_root, args, run_root / "clean_vanilla")
        init_state = vanilla_model.state_dict()

    adapted_model, adapted_checkpoint = train_coral_surrogate_control(
        sim_root=args.sim_root,
        exp_root=synthetic_root,
        args=args,
        run_root=run_root / "adapted_coral_surrogate",
        init_state_dict=init_state,
    )

    vanilla_predictions = predict_dataset(
        vanilla_model,
        dataset_root=synthetic_root,
        sim_root_for_noneq=args.sim_root,
        output_csv=run_root / "synthetic_shift_vanilla_predictions.csv",
        device=torch.device(args.device),
    )
    adapted_predictions = predict_dataset(
        adapted_model,
        dataset_root=synthetic_root,
        sim_root_for_noneq=args.sim_root,
        output_csv=run_root / "synthetic_shift_adapted_predictions.csv",
        device=torch.device(args.device),
    )
    summaries = pd.concat(
        [
            summarize_predictions(vanilla_predictions, run_root / "synthetic_shift_vanilla_metrics.csv", "vanilla_clean_sim"),
            summarize_predictions(adapted_predictions, run_root / "synthetic_shift_adapted_metrics.csv", "coral_surrogate_synthetic_shift"),
        ],
        ignore_index=True,
    )
    summaries.to_csv(run_root / "synthetic_shift_metrics_combined.csv", index=False)
    metadata = {
        "control": "synthetic_shift",
        "synthetic_root": str(synthetic_root),
        "vanilla_checkpoint": str(vanilla_checkpoint),
        "adapted_checkpoint": str(adapted_checkpoint),
    }
    (run_root / "control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")
    print(summaries)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sim-root", type=Path, default=Path("dataset/simulation"))
    parser.add_argument("--exp-root", type=Path, default=Path("dataset/experiment"))
    parser.add_argument("--control-dataset-root", type=Path, default=Path("dataset/supplementary_controls"))
    parser.add_argument("--output-root", type=Path, default=Path("results/supplementary_controls"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--vanilla-learning-rate", type=float, default=1e-3)
    parser.add_argument("--coral-learning-rate", type=float, default=3e-4)
    parser.add_argument("--coral-weight", type=float, default=1.0)
    parser.add_argument("--surrogate-weight", type=float, default=1.2)
    parser.add_argument("--surrogate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--surrogate-pretrain-epochs", type=int, default=200)
    parser.add_argument("--surrogate-pretrain-patience", type=int, default=20)
    parser.add_argument("--surrogate-checkpoint-path", type=Path, default=Path("models/XPCS_noneq_surrogate_no_T.pt"))
    parser.add_argument("--force-surrogate-retrain", action="store_true")
    parser.add_argument("--surrogate-loss", choices=["smooth-l1", "mse"], default="mse")
    parser.add_argument("--init-vanilla-model-path", type=Path, default=None)


def add_experiment_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluate-experiment", action="store_true")
    parser.add_argument("--exp-data-dir", type=Path, default=Path("exp_data"))
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--shot-indices", nargs="*", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=2500)
    parser.add_argument("--coarse-size", type=int, default=256)
    parser.add_argument("--crop-step", type=int, default=100)
    parser.add_argument("--eval-crop-policy", choices=["top-left", "all-diagonal"], default="all-diagonal")
    parser.add_argument("--eval-crop-aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--experiment-shot-cache-dir", type=Path, default=Path("dataset/experiment_eval_cache"))
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--umap-sim-limit", type=int, default=500)
    parser.add_argument("--umap-neighbors", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-init", default="spectral")
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument("--write-phase-diagrams", action="store_true")
    parser.add_argument("--phase-diagram-sim-manifest", type=Path, default=Path("dataset/simulation/manifest_with_non_equ.csv"))
    parser.add_argument("--phase-diagram-shot-index", type=int, default=0)
    parser.add_argument("--phase-diagram-range-mode", choices=["stats", "data", "fixed"], default="stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser("label-permutation")
    add_common_args(label_parser)
    label_parser.add_argument("--skip-coral-surrogate", action="store_true")
    label_parser.set_defaults(func=run_label_permutation)

    snoneq_parser = subparsers.add_parser("snoneq-shuffle")
    add_common_args(snoneq_parser)
    add_experiment_eval_args(snoneq_parser)
    snoneq_parser.set_defaults(func=run_snoneq_shuffle)

    ood_parser = subparsers.add_parser("ood-holdout")
    add_common_args(ood_parser)
    ood_parser.add_argument("--holdout-column", choices=PARAM_COLUMNS + ["T", "nonequilibrium_measure_raw", "nonequilibrium_measure"], default="D")
    ood_parser.add_argument("--holdout-lower-quantile", type=float, default=0.8)
    ood_parser.add_argument("--holdout-upper-quantile", type=float, default=1.0)
    ood_parser.add_argument("--holdout-query", default=None)
    ood_parser.set_defaults(func=run_ood_holdout)

    synthetic_parser = subparsers.add_parser("synthetic-shift")
    add_common_args(synthetic_parser)
    synthetic_parser.add_argument("--vanilla-model-path", type=Path, default=Path("models/Vanilla_XPCS_no_T_best_20260414-202028.pt"))
    synthetic_parser.add_argument("--synthetic-count", type=int, default=512)
    synthetic_parser.add_argument("--synthetic-noise-std", type=float, default=0.35)
    synthetic_parser.add_argument("--synthetic-background", type=float, default=0.0025)
    synthetic_parser.add_argument("--synthetic-gradient", type=float, default=0.25)
    synthetic_parser.add_argument("--synthetic-smear-kernel", type=int, default=5)
    synthetic_parser.add_argument("--synthetic-smear-alpha", type=float, default=0.35)
    synthetic_parser.set_defaults(func=run_synthetic_shift)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.sim_root = (args.project_root / args.sim_root).resolve() if not args.sim_root.is_absolute() else args.sim_root
    args.exp_root = (args.project_root / args.exp_root).resolve() if not args.exp_root.is_absolute() else args.exp_root
    args.control_dataset_root = (args.project_root / args.control_dataset_root).resolve() if not args.control_dataset_root.is_absolute() else args.control_dataset_root
    args.output_root = (args.project_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    for attr in [
        "surrogate_checkpoint_path",
        "init_vanilla_model_path",
        "vanilla_model_path",
        "exp_data_dir",
        "experiment_shot_cache_dir",
        "phase_diagram_sim_manifest",
    ]:
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None and not value.is_absolute():
                setattr(args, attr, (args.project_root / value).resolve())
    args.func(args)


if __name__ == "__main__":
    main()
