"""
Per-material fine-tuning for a pretrained CORAL-surrogate no-T model.

This script starts from one globally trained CORAL-surrogate model, clones it
once per material-dose group, and fine-tunes that child model only on the
selected experimental shots for that group. The fine-tuning objective aligns
the frozen params->noneq surrogate prediction with the measured experiment
nonequilibrium measure.

Outputs intentionally follow the existing evaluation layout:

    results/<run_name>/<material_dose>/<raw_file_stem>.csv

Those CSVs contain `gamma_adv`, `D_adv`, and `lambda_GB_adv`, so the existing
phase-diagram plotting code can overlay each material-specific child model.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from inference import plot_phase_diagrams
from run_all import (
    ShotSample,
    filter_files,
    iter_raw_experiment_files,
    parse_result_group,
    prepare_experiment_shots,
    resolve_phase_diagram_sim_manifest,
    write_results_csvs,
)
from train_adv_coral_surrogate import (
    XPCSNetCoral,
    compute_noneq_surrogate_loss,
    compute_regression_r2,
    denorm_from_meta,
    load_model as load_coral_surrogate_model,
    set_module_requires_grad,
)
from train_vanilla_no_T import set_global_seed


@dataclass
class FineTuneMetrics:
    material: str
    num_files: int
    num_crops: int
    num_shot_rows: int
    initial_loss: float
    initial_mae: float
    initial_r2: float
    final_loss: float
    final_mae: float
    final_r2: float
    best_epoch: int
    epochs_completed: int


class MaterialFineTuneDataset(Dataset):
    """Small dataset wrapper over prepared experimental shot/crop samples."""

    def __init__(self, samples: Sequence[ShotSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        temperature = torch.tensor([sample.temperature_k], dtype=torch.float32)
        noneq = torch.tensor(sample.nonequilibrium_measure, dtype=torch.float32)
        return sample.x, temperature, noneq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune one child CORAL-surrogate model per material-dose group "
            "to align selected experimental non-eq measures, then draw per-group "
            "and combined phase diagrams."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Pretrained CORAL-surrogate no-T checkpoint. If omitted, loads the "
            "latest XPCS_coral_surrogate_no_T_best_*.pt from models/."
        ),
    )
    parser.add_argument(
        "--exp-data-dir",
        type=Path,
        default=Path("exp_data"),
        help="Directory containing top-level raw .npz/.npy experiment files.",
    )
    parser.add_argument(
        "--simulation-dataset-dir",
        type=Path,
        default=Path("dataset/simulation"),
        help="Simulation dataset directory used to resolve the phase-map manifest.",
    )
    parser.add_argument(
        "--phase-diagram-sim-manifest",
        type=Path,
        default=None,
        help=(
            "Optional simulation manifest for phase diagrams. Defaults to the "
            "nonequilibrium-enriched manifest under --simulation-dataset-dir."
        ),
    )
    parser.add_argument(
        "--experiment-shot-cache-dir",
        type=Path,
        default=Path("dataset/experiment_eval_cache"),
        help="Cache directory for prepared raw experiment shots.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Parent directory for the run output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output run directory before writing.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional raw file stems or filenames to include.",
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        default=None,
        help=(
            "Optional material-dose group keys to include, e.g. 030BM_L_dose2. "
            "If omitted, all parseable groups are used."
        ),
    )
    parser.add_argument(
        "--shot-index",
        type=int,
        default=None,
        help="Convenience option for selecting one raw shot index.",
    )
    parser.add_argument(
        "--shot-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of raw shot indices to fine-tune/evaluate.",
    )
    parser.add_argument(
        "--all-shots",
        action="store_true",
        help="Use all shots instead of the default shot 0.",
    )
    parser.add_argument("--crop-size", type=int, default=2500)
    parser.add_argument("--coarse-size", type=int, default=256)
    parser.add_argument("--crop-step", type=int, default=100)
    parser.add_argument(
        "--crop-policy",
        choices=["top-left", "all-diagonal"],
        default="all-diagonal",
        help="Raw-shot crop policy used before fine-tuning/evaluation.",
    )
    parser.add_argument(
        "--crop-aggregation",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate multiple crops into one shot-level CSV row.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=["smooth-l1", "mse"],
        default="mse",
        help="Non-eq alignment loss used for material fine-tuning.",
    )
    parser.add_argument(
        "--tune-scope",
        choices=["predictor", "encoder-predictor", "all"],
        default="predictor",
        help=(
            "Which child-model parameters are trainable. The noneq surrogate is "
            "always frozen."
        ),
    )
    parser.add_argument(
        "--prediction-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Optional penalty on normalized params drifting from the pretrained "
            "global model predictions. Default 0 disables the anchor."
        ),
    )
    parser.add_argument(
        "--save-material-models",
        action="store_true",
        help="Save each material child model checkpoint under its output folder.",
    )
    parser.add_argument(
        "--skip-phase-diagrams",
        action="store_true",
        help="Only fine-tune and write CSVs; skip phase-diagram PDFs.",
    )
    return parser.parse_args()


def resolve_selected_shots(args: argparse.Namespace) -> list[int] | None:
    if args.all_shots:
        if args.shot_index is not None or args.shot_indices is not None:
            raise ValueError("--all-shots cannot be combined with --shot-index/--shot-indices")
        return None
    if args.shot_index is not None and args.shot_indices is not None:
        raise ValueError("Use either --shot-index or --shot-indices, not both")
    if args.shot_index is not None:
        return [args.shot_index]
    if args.shot_indices is not None:
        selected = sorted(set(args.shot_indices))
        if not selected:
            raise ValueError("--shot-indices was provided but no indices were listed")
        return selected
    return [0]


def shot_label(shot_indices: list[int] | None) -> str:
    if shot_indices is None:
        return "allshots"
    if len(shot_indices) == 1:
        return f"shot{shot_indices[0]}"
    joined = "-".join(str(idx) for idx in shot_indices)
    return f"shots{joined}"


def group_raw_files_by_material(
    raw_files: Sequence[Path],
    material_filter: Sequence[str] | None,
) -> dict[str, list[Path]]:
    wanted = None if not material_filter else set(material_filter)
    groups: dict[str, list[Path]] = {}
    skipped: list[str] = []
    for raw_file in raw_files:
        try:
            group_key, _ = parse_result_group(Path(f"{raw_file.stem}.csv"))
        except ValueError:
            skipped.append(raw_file.name)
            continue
        if wanted is not None and group_key not in wanted:
            continue
        groups.setdefault(group_key, []).append(raw_file)

    if wanted is not None:
        missing = sorted(wanted - set(groups))
        if missing:
            raise FileNotFoundError(f"Requested material groups were not found: {missing}")
    if skipped:
        print(f"[select] skipped {len(skipped)} file(s) with unsupported names")
    return {key: sorted(value) for key, value in sorted(groups.items())}


def configure_trainable_parameters(
    model: XPCSNetCoral,
    tune_scope: str,
) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)

    modules: list[torch.nn.Module]
    if tune_scope == "predictor":
        modules = [model.xpcs_predictor]
    elif tune_scope == "encoder-predictor":
        modules = [model.conv_net, model.xpcs_predictor]
    elif tune_scope == "all":
        modules = [model]
    else:
        raise ValueError(f"Unsupported tune scope: {tune_scope}")

    for module in modules:
        set_module_requires_grad(module, True)
    set_module_requires_grad(model.noneq_surrogate, False)
    model.noneq_surrogate.eval()

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters for tune scope {tune_scope}")
    return trainable


@torch.no_grad()
def evaluate_material_noneq(
    model: XPCSNetCoral,
    dataset: MaterialFineTuneDataset,
    device: torch.device,
    batch_size: int,
    loss_type: str,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    model.noneq_surrogate.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, temperature, noneq in loader:
        x = x.to(device)
        temperature = temperature.to(device)
        noneq = noneq.to(device)
        pred_params = model(x, temperature)
        pred_noneq = model.predict_nonequilibrium_from_params(pred_params)
        total_loss += compute_noneq_surrogate_loss(
            pred_noneq,
            noneq,
            loss_type=loss_type,
            reduction="sum",
        ).item()
        total_abs_error += torch.abs(pred_noneq - noneq).sum().item()
        total_count += x.size(0)
        predictions.append(pred_noneq.detach().cpu().numpy())
        targets.append(noneq.detach().cpu().numpy())

    if total_count == 0:
        return {"loss": float("nan"), "mae": float("nan"), "r2": float("nan")}

    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    return {
        "loss": float(total_loss / total_count),
        "mae": float(total_abs_error / total_count),
        "r2": compute_regression_r2(target_array, pred_array),
    }


def fine_tune_one_material(
    material: str,
    base_model: XPCSNetCoral,
    reference_model: XPCSNetCoral | None,
    samples: Sequence[ShotSample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[XPCSNetCoral, FineTuneMetrics]:
    dataset = MaterialFineTuneDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, max(1, len(dataset))),
        shuffle=True,
        num_workers=0,
    )

    child_model = copy.deepcopy(base_model).to(device)
    trainable_params = configure_trainable_parameters(child_model, args.tune_scope)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    initial = evaluate_material_noneq(
        child_model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        loss_type=args.loss,
    )
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in child_model.state_dict().items()
    }
    best_loss = initial["loss"]
    best_epoch = 0
    bad_epochs = 0
    epochs_completed = 0

    for epoch in range(1, args.epochs + 1):
        child_model.train()
        child_model.noneq_surrogate.eval()
        epoch_loss_sum = 0.0
        epoch_count = 0
        for x, temperature, noneq in loader:
            x = x.to(device)
            temperature = temperature.to(device)
            noneq = noneq.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_params = child_model(x, temperature)
            pred_noneq = child_model.predict_nonequilibrium_from_params(pred_params)
            loss = compute_noneq_surrogate_loss(
                pred_noneq,
                noneq,
                loss_type=args.loss,
            )
            if args.prediction_anchor_weight > 0.0:
                if reference_model is None:
                    raise RuntimeError("reference_model is required for prediction anchoring")
                with torch.no_grad():
                    reference_params = reference_model(x, temperature)
                anchor_loss = torch.nn.functional.mse_loss(pred_params, reference_params)
                loss = loss + (args.prediction_anchor_weight * anchor_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * x.size(0)
            epoch_count += x.size(0)

        epochs_completed = epoch
        metrics = evaluate_material_noneq(
            child_model,
            dataset,
            device=device,
            batch_size=args.batch_size,
            loss_type=args.loss,
        )
        if metrics["loss"] < best_loss - 1e-8:
            best_loss = metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in child_model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 25 == 0 or bad_epochs == 0:
            avg_train_loss = epoch_loss_sum / max(1, epoch_count)
            print(
                f"[fine-tune:{material}] epoch {epoch:04d} "
                f"train {avg_train_loss:.6f} eval {metrics['loss']:.6f} "
                f"mae {metrics['mae']:.6f} R2 {metrics['r2']:.4f}"
            )
        if bad_epochs >= args.patience:
            print(
                f"[fine-tune:{material}] early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, loss {best_loss:.6f})"
            )
            break

    child_model.load_state_dict(best_state)
    child_model.eval()
    final = evaluate_material_noneq(
        child_model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        loss_type=args.loss,
    )
    grouped_shots = {
        (sample.file_stem, sample.shot_index)
        for sample in samples
    }
    metrics = FineTuneMetrics(
        material=material,
        num_files=len({sample.file_stem for sample in samples}),
        num_crops=len(samples),
        num_shot_rows=len(grouped_shots),
        initial_loss=initial["loss"],
        initial_mae=initial["mae"],
        initial_r2=initial["r2"],
        final_loss=final["loss"],
        final_mae=final["mae"],
        final_r2=final["r2"],
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
    )
    print(
        f"[fine-tune:{material}] loss {initial['loss']:.6f} -> {final['loss']:.6f}, "
        f"MAE {initial['mae']:.6f} -> {final['mae']:.6f}, "
        f"R2 {initial['r2']:.4f} -> {final['r2']:.4f}"
    )
    return child_model.cpu(), metrics


@torch.no_grad()
def predict_raw_params(
    model: XPCSNetCoral,
    samples: Sequence[ShotSample],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = MaterialFineTuneDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    predictions = []
    for x, temperature, _ in loader:
        x = x.to(device)
        temperature = temperature.to(device)
        pred_params_norm = model(x, temperature)
        pred_params_raw = denorm_from_meta(
            pred_params_norm,
            model.norm_meta,
            device=device,
        )
        predictions.append(pred_params_raw.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def write_material_summary(
    material_dir: Path,
    material: str,
    metrics: FineTuneMetrics,
) -> None:
    result_csvs = []
    for csv_path in sorted(material_dir.glob("*.csv")):
        try:
            parse_result_group(csv_path)
        except ValueError:
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "material", material)
        df.insert(1, "source_csv", csv_path.name)
        result_csvs.append(df)

    if result_csvs:
        summary_df = pd.concat(result_csvs, ignore_index=True)
        summary_path = material_dir / "material_finetune_predictions.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[write:{material}] wrote {summary_path}")

    metrics_path = material_dir / "material_finetune_metrics.json"
    with open(metrics_path, "w", encoding="ascii") as f:
        json.dump(asdict(metrics), f, indent=2)
    print(f"[write:{material}] wrote {metrics_path}")


def prepare_output_dir(results_dir: Path, overwrite: bool) -> None:
    if results_dir.exists() and overwrite:
        shutil.rmtree(results_dir)
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {results_dir}. "
            "Use --overwrite or rerun to create a fresh timestamped directory."
        )
    results_dir.mkdir(parents=True, exist_ok=True)


def build_default_results_run_name(
    args: argparse.Namespace,
    selected_shots: list[int] | None,
) -> str:
    """
    Build a readable timestamped run directory name.

    The user only chooses the parent `--results-dir`; this helper owns the
    run-specific folder name so repeated ablations remain easy to compare.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    labels = [
        "coral-surrogate-material-ft",
        "noT",
        shot_label(selected_shots),
        args.tune_scope,
        args.loss,
        f"seed{args.seed}",
        args.crop_policy,
        stamp,
    ]
    return "_".join(labels)


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    selected_shots: list[int] | None,
    model_path: Path | None,
    materials: Sequence[str],
) -> None:
    metadata = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "method": "material_coral_surrogate_finetune",
        "model_path": None if model_path is None else str(model_path),
        "selected_shots": "all" if selected_shots is None else selected_shots,
        "materials": list(materials),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with open(path, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)


def single_phase_shot_index(selected_shots: list[int] | None) -> int | None:
    if selected_shots is not None and len(selected_shots) == 1:
        return selected_shots[0]
    return None


def main() -> None:
    args = parse_args()
    selected_shots = resolve_selected_shots(args)
    set_global_seed(args.seed, deterministic=True)
    device = torch.device(args.device)

    run_name = build_default_results_run_name(args, selected_shots)
    run_dir = args.results_dir / run_name
    prepare_output_dir(run_dir, overwrite=args.overwrite)
    print(f"[results] writing run outputs to {run_dir}")

    raw_files = filter_files(iter_raw_experiment_files(args.exp_data_dir), args.files)
    groups = group_raw_files_by_material(raw_files, args.materials)
    if not groups:
        raise RuntimeError("No material groups selected for fine-tuning")

    base_model = load_coral_surrogate_model(args.model_path, device=torch.device("cpu"))
    base_model.eval()
    reference_model = None
    if args.prediction_anchor_weight > 0.0:
        reference_model = copy.deepcopy(base_model).to(device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    write_run_metadata(
        run_dir / "material_finetune_metadata.json",
        args=args,
        selected_shots=selected_shots,
        model_path=args.model_path,
        materials=groups.keys(),
    )

    all_metrics: list[FineTuneMetrics] = []
    for material, material_files in groups.items():
        print(
            f"\n[material:{material}] preparing {len(material_files)} raw file(s), "
            f"shots={selected_shots if selected_shots is not None else 'all'}"
        )
        samples = prepare_experiment_shots(
            raw_files=material_files,
            shot_indices=selected_shots,
            crop_size=args.crop_size,
            coarse_size=args.coarse_size,
            crop_step=args.crop_step,
            crop_policy=args.crop_policy,
            no_t=True,
            cache_dir=args.experiment_shot_cache_dir,
        )
        child_model, metrics = fine_tune_one_material(
            material=material,
            base_model=base_model,
            reference_model=reference_model,
            samples=samples,
            args=args,
            device=device,
        )
        predictions = predict_raw_params(
            child_model,
            samples=samples,
            batch_size=args.batch_size,
            device=device,
        )
        write_results_csvs(
            samples=samples,
            adv_predictions=predictions,
            vanilla_predictions=None,
            adv_coords=None,
            adv_sim_count=0,
            vanilla_coords=None,
            vanilla_sim_count=0,
            results_dir=run_dir,
            skip_umap=True,
            crop_policy=args.crop_policy,
            crop_aggregation=args.crop_aggregation,
        )
        material_dir = run_dir / material
        if args.save_material_models:
            model_path = material_dir / "XPCS_coral_surrogate_no_T_material_finetuned.pt"
            torch.save(child_model.state_dict(), model_path)
            print(f"[write:{material}] wrote {model_path}")
        write_material_summary(material_dir, material=material, metrics=metrics)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame([asdict(metric) for metric in all_metrics])
    metrics_csv = run_dir / "material_finetune_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[write] wrote {metrics_csv}")

    if args.skip_phase_diagrams:
        print("[phase] skipped by --skip-phase-diagrams")
        return

    sim_manifest = resolve_phase_diagram_sim_manifest(
        simulation_dataset_dir=args.simulation_dataset_dir,
        requested_manifest=args.phase_diagram_sim_manifest,
    )
    phase_shot_index = single_phase_shot_index(selected_shots)

    print("[phase] writing per-material child phase diagrams")
    plot_phase_diagrams(
        results_dir=run_dir,
        simulation_manifest=sim_manifest,
        output_dir=None,
        model_names=["adv"],
        aggregate_by="shot",
        shot_index=phase_shot_index,
        split_by_material=True,
        raw_data_dir=args.exp_data_dir,
        crop_size=args.crop_size,
        coarse_size=args.coarse_size,
    )

    global_phase_dir = run_dir / f"phase_diagrams_material_finetuned_{shot_label(selected_shots)}"
    print(f"[phase] writing combined child-model phase diagram to {global_phase_dir}")
    plot_phase_diagrams(
        results_dir=run_dir,
        simulation_manifest=sim_manifest,
        output_dir=global_phase_dir,
        model_names=["adv"],
        aggregate_by="shot",
        shot_index=phase_shot_index,
        split_by_material=False,
        raw_data_dir=args.exp_data_dir,
        crop_size=args.crop_size,
        coarse_size=args.coarse_size,
    )


if __name__ == "__main__":
    main()
