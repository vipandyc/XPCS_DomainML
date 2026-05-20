from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from produce_data import coarse_grain_g2, normalize_g2, simulate_xpcs


PARAMETER_KEYS = ("gamma", "D", "GB_conc")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        return json.load(f)


def compute_constant_baselines(
    metadata_path: Path,
) -> dict[str, Any]:
    """
    Compute simple constant baselines on the saved vanilla test split.

    The three baselines are:
    - train-split mean predictor
    - train-split median predictor
    - normalization-range center predictor
    """
    metadata = load_json(metadata_path)
    manifest_path = Path(metadata["dataset_root"]) / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    train_ids = set(metadata["train_ids"])
    test_ids = set(metadata["test_ids"])
    train_df = manifest_df[manifest_df["id"].isin(train_ids)].copy()
    test_df = manifest_df[manifest_df["id"].isin(test_ids)].copy()

    baselines = {
        "train_mean": {
            key: float(train_df[key].mean()) for key in PARAMETER_KEYS
        },
        "train_median": {
            key: float(train_df[key].median()) for key in PARAMETER_KEYS
        },
        "range_center": {
            "gamma": 0.5 * (2e18 + 5e18),
            "D": float(np.sqrt(1e-23 * 1e-21)),
            "GB_conc": 0.15,
        },
    }

    metrics: dict[str, dict[str, float]] = {}
    for baseline_name, predictor in baselines.items():
        baseline_metrics: dict[str, float] = {}
        for key in PARAMETER_KEYS:
            err = test_df[key].to_numpy(dtype=float) - predictor[key]
            baseline_metrics[f"{key}_mae"] = float(np.mean(np.abs(err)))
            baseline_metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(err**2)))
        metrics[baseline_name] = baseline_metrics

    return {
        "metadata_path": str(metadata_path),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "best_val_loss": metadata.get("best_val_loss"),
        "baselines": baselines,
        "metrics": metrics,
    }


def flatten_baseline_report(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for baseline_name, predictor in report["baselines"].items():
        metric_row = report["metrics"][baseline_name]
        rows.append({
            "baseline": baseline_name,
            "gamma_pred": predictor["gamma"],
            "D_pred": predictor["D"],
            "GB_conc_pred": predictor["GB_conc"],
            **metric_row,
        })
    return pd.DataFrame(rows)


def load_reconstruction_summary(summary_path: Path) -> pd.DataFrame:
    return pd.read_csv(summary_path)


def summarize_prediction_collapse(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in PARAMETER_KEYS:
        rows.append({
            "parameter": key,
            "pred_mean": float(summary_df[f"{key}_pred"].mean()),
            "pred_std": float(summary_df[f"{key}_pred"].std(ddof=1)),
            "true_mean": float(summary_df[f"{key}_true"].mean()),
            "true_std": float(summary_df[f"{key}_true"].std(ddof=1)),
            "abs_error_mean": float(summary_df[f"{key}_abs_error"].mean()),
        })
    return pd.DataFrame(rows)


def simulate_true_and_pred_full_resolution(
    summary_row: pd.Series,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Re-simulate the true and predicted spectra at full resolution.

    Because dataset generation used a fixed seed of 42, re-running the true
    parameters with the same seed recovers the original pre-coarsening sample.
    """
    temperature_k = float(summary_row["temperature_k"])
    true_full = simulate_xpcs(
        gamma=float(summary_row["gamma_true"]),
        D=float(summary_row["D_true"]),
        GB_conc=float(summary_row["GB_conc_true"]),
        T=temperature_k,
        seed=seed,
        coarse=False,
    )
    pred_full = simulate_xpcs(
        gamma=float(summary_row["gamma_pred"]),
        D=float(summary_row["D_pred"]),
        GB_conc=float(summary_row["GB_conc_pred"]),
        T=temperature_k,
        seed=seed,
        coarse=False,
    )
    if true_full is None or pred_full is None:
        raise RuntimeError("Simulation became numerically unstable")
    return {
        "true_full": true_full.to(torch.float32),
        "pred_full": pred_full.to(torch.float32),
    }


def build_multiscale_reconstruction_bundle(
    summary_row: pd.Series,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Build full-resolution and coarse-grained diagnostics for one summary row.
    """
    full = simulate_true_and_pred_full_resolution(summary_row, seed=seed)
    true_full = full["true_full"]
    pred_full = full["pred_full"]
    true_coarse = coarse_grain_g2(true_full, target_size=(256, 256))
    pred_coarse = coarse_grain_g2(pred_full, target_size=(256, 256))
    original_coarse = torch.load(
        summary_row["path"],
        weights_only=True,
    ).to(torch.float32).squeeze(0)
    return {
        "true_full_raw": true_full,
        "pred_full_raw": pred_full,
        "true_full_norm": normalize_g2(true_full, min_val=1.0, max_val=1.2),
        "pred_full_norm": normalize_g2(pred_full, min_val=1.0, max_val=1.2),
        "true_coarse_norm": normalize_g2(true_coarse, min_val=1.0, max_val=1.2),
        "pred_coarse_norm": normalize_g2(pred_coarse, min_val=1.0, max_val=1.2),
        "original_coarse_norm": normalize_g2(original_coarse, min_val=1.0, max_val=1.2),
    }


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).item())


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).item())
