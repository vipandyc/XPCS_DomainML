from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from CNN_diagnostics.diagnostics import (
    build_multiscale_reconstruction_bundle,
    load_reconstruction_summary,
    mae,
    mse,
)
from utils import plot_auto_correlation_comparison, plot_g2_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run expensive full-resolution forward simulations for the selected "
            "rows in a reconstruction summary CSV."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to `sim_reconstruction_vanilla/summary.csv`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where full-resolution diagnostics will be written.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        type=int,
        default=None,
        help="Optional sample_rank values to run. Defaults to every row in the summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Simulation seed passed to `simulate_xpcs`. Use 42 to match dataset generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = load_reconstruction_summary(args.summary)
    if args.samples:
        summary_df = summary_df[summary_df["sample_rank"].isin(args.samples)].copy()
    if summary_df.empty:
        raise ValueError("No summary rows selected for full-resolution reconstruction")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for row in summary_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        sample_rank = int(row_series["sample_rank"])
        sample_dir = output_dir / f"sample_{sample_rank:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[full-res] running sample {sample_rank}")

        bundle = build_multiscale_reconstruction_bundle(
            row_series,
            seed=args.seed,
        )

        torch.save(bundle["true_full_raw"], sample_dir / "true_full_raw.pt")
        torch.save(bundle["pred_full_raw"], sample_dir / "pred_full_raw.pt")
        torch.save(bundle["true_full_norm"], sample_dir / "true_full_norm.pt")
        torch.save(bundle["pred_full_norm"], sample_dir / "pred_full_norm.pt")
        torch.save(bundle["true_coarse_norm"], sample_dir / "true_coarse_norm.pt")
        torch.save(bundle["pred_coarse_norm"], sample_dir / "pred_coarse_norm.pt")
        torch.save(bundle["original_coarse_norm"], sample_dir / "original_coarse_norm.pt")

        plot_g2_side_by_side(
            bundle["true_full_norm"],
            bundle["pred_full_norm"],
            save_path=sample_dir / "full_res_true_vs_pred.pdf",
            left_title="True params | full res",
            right_title="Pred params | full res",
        )
        plot_g2_side_by_side(
            bundle["original_coarse_norm"],
            bundle["pred_coarse_norm"],
            save_path=sample_dir / "original_coarse_vs_pred_coarse.pdf",
            left_title="Original dataset coarse",
            right_title="Pred params -> full res -> coarse",
        )
        plot_g2_side_by_side(
            bundle["true_coarse_norm"],
            bundle["pred_coarse_norm"],
            save_path=sample_dir / "true_coarse_vs_pred_coarse.pdf",
            left_title="True params -> full res -> coarse",
            right_title="Pred params -> full res -> coarse",
        )
        plot_auto_correlation_comparison(
            bundle["true_full_norm"],
            bundle["pred_full_norm"],
            save_path=sample_dir / "full_res_autocorrelation_true_vs_pred.pdf",
            title=f"Sample {sample_rank}: full-resolution reconstruction",
        )

        metrics_rows.append({
            "sample_rank": sample_rank,
            "temperature_k": float(row_series["temperature_k"]),
            "nonequilibrium_measure": row_series.get("nonequilibrium_measure"),
            "full_res_mse_true_vs_pred": mse(
                bundle["true_full_norm"],
                bundle["pred_full_norm"],
            ),
            "full_res_mae_true_vs_pred": mae(
                bundle["true_full_norm"],
                bundle["pred_full_norm"],
            ),
            "coarse_mse_true_vs_pred": mse(
                bundle["true_coarse_norm"],
                bundle["pred_coarse_norm"],
            ),
            "coarse_mae_true_vs_pred": mae(
                bundle["true_coarse_norm"],
                bundle["pred_coarse_norm"],
            ),
            "coarse_mse_original_vs_true": mse(
                bundle["original_coarse_norm"],
                bundle["true_coarse_norm"],
            ),
            "coarse_mae_original_vs_true": mae(
                bundle["original_coarse_norm"],
                bundle["true_coarse_norm"],
            ),
            "coarse_mse_original_vs_pred": mse(
                bundle["original_coarse_norm"],
                bundle["pred_coarse_norm"],
            ),
            "coarse_mae_original_vs_pred": mae(
                bundle["original_coarse_norm"],
                bundle["pred_coarse_norm"],
            ),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("sample_rank")
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[full-res] wrote {metrics_path}")


if __name__ == "__main__":
    main()
