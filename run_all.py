import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from process_exp_data import crop_data, merge_data
from produce_data import coarse_grain_g2, normalize_g2, simulate_xpcs
from inference import (
    detect_available_models as detect_phase_diagram_models,
    iter_result_csvs as iter_phase_diagram_result_csvs,
    plot_phase_diagrams as inference_plot_phase_diagrams,
)
from train_adv import (
    XPCSDataset as XPCSDatasetWithT,
    XPCSNet,
    denorm_from_meta as denorm_from_meta_with_t,
    load_model as load_adv_model,
    train as train_adv_model,
)
from train_adv_no_T import (
    INPUT_MEAN as INPUT_MEAN_NO_T,
    INPUT_STD as INPUT_STD_NO_T,
    XPCSDataset as XPCSDatasetNoT,
    XPCSNet as XPCSNetNoT,
    denorm_from_meta as denorm_from_meta_no_t,
    load_model as load_adv_model_no_t,
    train as train_adv_model_no_t,
)
from train_adv_coral_distill import (
    XPCSNetCoral as XPCSNetCoralNoT,
    load_model as load_coral_model_no_t,
    train as train_coral_model_no_t,
)
from train_adv_coral_surrogate import (
    XPCSNetCoral as XPCSNetCoralSurrogateNoT,
    load_model as load_coral_surrogate_model_no_t,
    train as train_coral_surrogate_model_no_t,
)
from train_vanilla import (
    VanillaXPCSNet,
    load_model as load_vanilla_model,
    set_global_seed as set_vanilla_global_seed,
    train as train_vanilla_model,
)
from train_vanilla_no_T import (
    VanillaXPCSNet as VanillaXPCSNetNoT,
    load_model as load_vanilla_model_no_t,
    train as train_vanilla_model_no_t,
)
from utils import (
    calc_umap,
    nonequilibrium_measure,
    plot_cluster,
    plot_auto_correlation_comparison,
    plot_experiment_metadata_embedding,
    plot_g2_side_by_side,
    plot_multi_bar_v2,
    plot_single_model_multi_bar_v2,
    plot_parameter_pair_comparison,
)


RAW_SUFFIXES = {".npz", ".npy"}
MODEL_SELECTION_CHOICES = [
    "both",
    "vanilla",
    "adv",
    "coral",
    "coral-surrogate",
    "none",
]
VanillaModel = VanillaXPCSNet | VanillaXPCSNetNoT
AdvModel = XPCSNet | XPCSNetNoT | XPCSNetCoralNoT | XPCSNetCoralSurrogateNoT


def resolve_vanilla_components(
    no_t: bool,
) -> tuple[type[VanillaModel], object, object]:
    """
    Resolve the vanilla model class and train/load functions for the selected
    temperature-input setting.
    """
    if no_t:
        return VanillaXPCSNetNoT, train_vanilla_model_no_t, load_vanilla_model_no_t
    return VanillaXPCSNet, train_vanilla_model, load_vanilla_model


def resolve_adv_components(
    no_t: bool,
) -> tuple[type[AdvModel], object, object]:
    """
    Resolve the adversarial model class and train/load functions for the
    selected temperature-input setting.
    """
    if no_t:
        return XPCSNetNoT, train_adv_model_no_t, load_adv_model_no_t
    return XPCSNet, train_adv_model, load_adv_model


def resolve_coral_components(
    no_t: bool,
) -> tuple[type[AdvModel], object, object]:
    """
    Resolve the CORAL+distillation model class and train/load functions.
    """
    if no_t:
        return XPCSNetCoralNoT, train_coral_model_no_t, load_coral_model_no_t
    raise NotImplementedError("CORAL training with T not yet implemented")


def resolve_coral_surrogate_components(
    no_t: bool,
) -> tuple[type[AdvModel], object, object]:
    """
    Resolve the CORAL+surrogate model class and train/load functions.
    """
    if no_t:
        return (
            XPCSNetCoralSurrogateNoT,
            train_coral_surrogate_model_no_t,
            load_coral_surrogate_model_no_t,
        )
    raise NotImplementedError("CORAL surrogate training with T not yet implemented")


def resolve_eval_dataset_components(
    no_t: bool,
) -> tuple[type[Dataset], object]:
    """
    Resolve the simulation dataset preprocessing and parameter denormalization
    helpers used during evaluation.
    """
    if no_t:
        return XPCSDatasetNoT, denorm_from_meta_no_t
    return XPCSDatasetWithT, denorm_from_meta_with_t


@dataclass
class ShotSample:
    """
    Container for one raw experimental XPCS shot after preprocessing for inference.

    Attributes:
        file_name: Original raw data filename.
        file_stem: Filename stem used for result subdirectory naming.
        shot_index: Slice index within the raw 3D XPCS array.
        crop_start: Diagonal crop offset used to produce this model input.
        temperature_k: Temperature parsed from the filename, in Kelvin.
        x: Normalized and masked model input tensor of shape [1, 256, 256].
        nonequilibrium_measure: Scalar nonequilibrium metric computed from the shot.
    """
    file_name: str
    file_stem: str
    shot_index: int
    crop_start: int
    temperature_k: float
    x: torch.Tensor
    nonequilibrium_measure: float


class PreparedExperimentDataset(Dataset):
    """
    Lightweight dataset wrapper for a list of preprocessed experimental shots.

    The returned sample format matches the structure expected by the existing
    training and feature-extraction utilities:
    `(x, y_norm, y_raw, T, label)`.
    """

    def __init__(self, samples: Sequence[ShotSample]):
        """
        Args:
            samples: Sequence of already prepared experimental shot records.
        """
        self.samples = list(samples)

    def __len__(self) -> int:
        """Return the number of prepared experimental shots."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Return one prepared experimental sample in model-compatible dataset format.

        Args:
            idx: Sample index.

        Returns:
            x: Preprocessed XPCS tensor of shape [1, 256, 256].
            y_norm: Dummy target tensor of shape [3].
            y_raw: Dummy raw-target tensor of shape [3].
            temperature: Temperature tensor of shape [1].
            label: Domain label, always 1 for experiment.
        """
        sample = self.samples[idx]
        dummy = torch.zeros(3, dtype=torch.float32)
        temperature = torch.tensor([sample.temperature_k], dtype=torch.float32)
        return sample.x, dummy, dummy, temperature, 1


def resolve_experiment_shot_cache_path(
    raw_files: Sequence[Path],
    shot_indices: Sequence[int] | None,
    crop_size: int,
    coarse_size: int,
    crop_step: int,
    crop_policy: str,
    no_t: bool,
    cache_dir: Path,
) -> Path:
    """
    Build a stable cache path for prepared experimental evaluation shots.

    The cache key includes the selected raw files, their mtimes/sizes, and the
    evaluation preprocessing parameters so cached tensors are reused only when
    the full preparation recipe matches.
    """
    payload = {
        "raw_files": [
            {
                "path": str(path.resolve()),
                "mtime_ns": path.stat().st_mtime_ns,
                "size": path.stat().st_size,
            }
            for path in raw_files
        ],
        "shot_indices": None if shot_indices is None else list(shot_indices),
        "crop_size": crop_size,
        "coarse_size": coarse_size,
        "crop_step": crop_step,
        "crop_policy": crop_policy,
        "no_t": no_t,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"prepared_shots_{digest}.pt"


def load_prepared_experiment_shots(cache_path: Path) -> list[ShotSample]:
    """Load cached experimental evaluation shots from disk."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return [
        ShotSample(
            file_name=str(item["file_name"]),
            file_stem=str(item["file_stem"]),
            shot_index=int(item["shot_index"]),
            crop_start=int(item["crop_start"]),
            temperature_k=float(item["temperature_k"]),
            x=item["x"].to(torch.float32),
            nonequilibrium_measure=float(item["nonequilibrium_measure"]),
        )
        for item in payload["samples"]
    ]


def save_prepared_experiment_shots(
    cache_path: Path,
    samples: Sequence[ShotSample],
) -> None:
    """Persist prepared experimental evaluation shots for future reuse."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": [
                {
                    "file_name": sample.file_name,
                    "file_stem": sample.file_stem,
                    "shot_index": sample.shot_index,
                    "crop_start": sample.crop_start,
                    "temperature_k": sample.temperature_k,
                    "x": sample.x.cpu(),
                    "nonequilibrium_measure": sample.nonequilibrium_measure,
                }
                for sample in samples
            ]
        },
        cache_path,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the end-to-end experiment pipeline.

    Returns:
        args: Parsed CLI arguments controlling preprocessing, training, and
            evaluation behavior.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the experiment dataset, retrain the vanilla/adversarial models, "
            "and evaluate raw experiment shots with one global UMAP run."
        )
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "preprocess",
            "train",
            "evaluate",
            "plot-bars",
            "plot-global-scatter",
            "phase-diagram",
        ],
        default=["train", "evaluate", "plot-bars", "plot-global-scatter"],
        help="Workflow steps to run.",
    )
    parser.add_argument(
        "--train-models",
        choices=MODEL_SELECTION_CHOICES,
        default="both",
        help="Which model(s) to train in the `train` step.",
    )
    parser.add_argument(
        "--eval-models",
        choices=MODEL_SELECTION_CHOICES,
        default="both",
        help="Which model(s) to use in the `evaluate` step.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional file stems or filenames to restrict preprocessing/evaluation to.",
    )
    parser.add_argument(
        "--shot-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional raw shot indices to evaluate within each selected file.",
    )
    parser.add_argument(
        "--sim-reconstruction-samples",
        type=int,
        default=4,
        help=(
            "Number of simulation samples to reconstruct from predicted vanilla "
            "parameters during evaluation. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--sim-reconstruction-manifest",
        type=Path,
        default=None,
        help=(
            "Optional simulation manifest CSV for selecting reconstruction "
            "diagnostic samples. Defaults to a nonequilibrium-enriched manifest "
            "when available."
        ),
    )
    parser.add_argument(
        "--exp-data-dir",
        type=Path,
        default=Path("exp_data"),
        help="Directory containing raw top-level .npz/.npy experiment files.",
    )
    parser.add_argument(
        "--simulation-dataset-dir",
        type=Path,
        default=Path("dataset/simulation_2"),
        help="Directory containing the processed simulation dataset.",
    )
    parser.add_argument(
        "--experiment-dataset-dir",
        type=Path,
        default=Path("dataset/experiment"),
        help="Directory where the rebuilt processed experiment dataset will be stored.",
    )
    parser.add_argument(
        "--experiment-shot-cache-dir",
        type=Path,
        default=Path("dataset/experiment_eval_cache"),
        help=(
            "Directory for caching prepared raw experimental evaluation shots so "
            "repeated evaluate/UMAP runs can reuse the expensive crop+coarse step."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Parent directory for evaluation-output run folders.",
    )
    parser.add_argument(
        "--results-run-name",
        default=None,
        help="Optional subdirectory name under `--results-dir` for one evaluation run.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing saved model checkpoints.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--keep-existing-experiment-dir",
        action="store_true",
        help="When preprocessing is requested, keep the existing processed experiment dataset instead of clearing it first.",
    )
    parser.add_argument(
        "--keep-existing-results-dir",
        action="store_true",
        help="Keep the existing results directory instead of clearing it first.",
    )
    parser.add_argument("--crop-size", type=int, default=2500)
    parser.add_argument("--coarse-size", type=int, default=256)
    parser.add_argument("--crop-step", type=int, default=100)
    parser.add_argument(
        "--no-T",
        dest="no_t",
        action="store_true",
        help=(
            "Use the no-temperature model variants from `train_vanilla_no_T.py` "
            "and `train_adv_no_T.py`."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--vanilla-seed",
        type=int,
        default=42,
        help="Seed used for vanilla initialization, dataset splits, and DataLoader shuffling.",
    )
    parser.add_argument(
        "--vanilla-num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for deterministic vanilla training.",
    )
    parser.add_argument(
        "--adv-seed",
        type=int,
        default=42,
        help="Seed used for adversarial initialization, splits, and DataLoader shuffling.",
    )
    parser.add_argument(
        "--adv-num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for deterministic adversarial training.",
    )
    parser.add_argument("--vanilla-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--vanilla-shared-feature-dim",
        type=int,
        default=128,
        help=(
            "Output dimension of the shared XPCS+temperature feature mixer "
            "used by vanilla training."
        ),
    )
    parser.add_argument(
        "--vanilla-shared-feature-mixer-hidden-dim",
        type=int,
        default=512,
        help=(
            "Hidden dimension of the shared XPCS+temperature feature mixer "
            "used by vanilla training."
        ),
    )
    parser.add_argument("--adv-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--adv-domain-learning-rate",
        type=float,
        default=1e-4,
        help=(
            "Learning rate for the domain-classifier optimizer. "
            "Defaults to `1e-4`, which matched the standalone probe experiments."
        ),
    )
    parser.add_argument(
        "--adaptation-rate",
        type=float,
        default=1.2,
        help=(
            "Maximum GRL strength applied to encoder features during "
            "adversarial training. The domain-classification loss itself "
            "remains full strength."
        ),
    )
    parser.add_argument(
        "--adv-warmup-epochs",
        type=int,
        default=20,
        help=(
            "Number of epochs over which the DANN-style adversarial GRL "
            "schedule saturates to its maximum value."
        ),
    )
    parser.add_argument(
        "--adv-domain-pretrain-epochs",
        type=int,
        default=10,
        help=(
            "Number of classifier-only warm-start epochs on frozen shared "
            "features before adversarial training begins."
        ),
    )
    parser.add_argument(
        "--adv-domain-steps-per-iteration",
        type=int,
        default=5,
        help=(
            "Number of discriminator mini-batch updates per outer minimax "
            "iteration during adversarial training."
        ),
    )
    parser.add_argument(
        "--adv-prediction-steps-per-iteration",
        type=int,
        default=1,
        help=(
            "Number of encoder/predictor mini-batch updates per outer minimax "
            "iteration during adversarial training."
        ),
    )
    parser.add_argument(
        "--adv-domain-only-passes",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility knob. Any positive value is added on top "
            "of `--adv-domain-steps-per-iteration`."
        ),
    )
    parser.add_argument(
        "--adv-use-shared-feature-mixer",
        "--adv-use-prediction-feature-mixer",
        dest="adv_use_shared_feature_mixer",
        action="store_true",
        help=(
            "Insert an FFN after concatenating XPCS and temperature features "
            "to build one shared representation for both the predictor and "
            "the domain classifier."
        ),
    )
    parser.add_argument(
        "--adv-shared-feature-dim",
        "--adv-prediction-feature-dim",
        dest="adv_shared_feature_dim",
        type=int,
        default=128,
        help=(
            "Output dimension of the optional shared XPCS+temperature feature "
            "mixer."
        ),
    )
    parser.add_argument(
        "--adv-shared-feature-mixer-hidden-dim",
        "--adv-prediction-feature-mixer-hidden-dim",
        dest="adv_shared_feature_mixer_hidden_dim",
        type=int,
        default=512,
        help=(
            "Hidden dimension of the optional shared XPCS+temperature feature "
            "mixer."
        ),
    )
    parser.add_argument(
        "--non-deterministic-vanilla",
        action="store_true",
        help="Allow faster but less reproducible vanilla training behavior.",
    )
    parser.add_argument(
        "--non-deterministic-adv",
        action="store_true",
        help="Allow faster but less reproducible adversarial training behavior.",
    )
    parser.add_argument(
        "--adv-init-vanilla-model-path",
        type=Path,
        default=None,
        help=(
            "Optional vanilla checkpoint used to initialize the adversarial "
            "encoder and predictor before domain adaptation."
        ),
    )
    parser.add_argument(
        "--adv-init-from-vanilla",
        action="store_true",
        help=(
            "Initialize adversarial training from vanilla weights. If a vanilla "
            "model was trained in the same run, use it; otherwise fall back to "
            "the latest vanilla checkpoint unless `--adv-init-vanilla-model-path` "
            "is provided."
        ),
    )

    # --- CORAL + distillation arguments ---
    parser.add_argument(
        "--coral-weight",
        type=float,
        default=1.0,
        help="Weight on the CORAL alignment loss (default: 1.0).",
    )
    parser.add_argument(
        "--coral-init-model-path",
        type=Path,
        default=None,
        help=(
            "Optional CORAL checkpoint used to initialize CORAL training "
            "before continuing finetuning."
        ),
    )
    parser.add_argument(
        "--coral-distill-weight",
        type=float,
        default=0.5,
        help="Weight on the feature distillation loss (default: 0.5).",
    )
    parser.add_argument(
        "--coral-vanilla-anchor-path",
        type=Path,
        default=None,
        help=(
            "Vanilla no-T checkpoint used as the frozen distillation anchor. "
            "If not specified, the latest vanilla no-T checkpoint is loaded."
        ),
    )
    parser.add_argument(
        "--coral-contrastive-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the nonequilibrium-aware contrastive loss that shapes "
            "the shared feature space (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--coral-contrastive-loss",
        choices=["margin", "soft-infonce"],
        default="margin",
        help=(
            "Nonequilibrium-aware contrastive objective for CORAL training. "
            "`margin` preserves the existing pairwise pull/push loss; "
            "`soft-infonce` uses a soft-label supervised InfoNCE target from "
            "continuous noneq distances (default: margin)."
        ),
    )
    parser.add_argument(
        "--coral-contrastive-bandwidth",
        type=float,
        default=0.1,
        help=(
            "Bandwidth in normalized nonequilibrium space for deciding which "
            "pairs should be pulled together most strongly (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--coral-contrastive-margin",
        type=float,
        default=1.0,
        help=(
            "Margin in unit-normalized feature space for pushing apart pairs "
            "with dissimilar nonequilibrium measures (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--coral-infonce-temperature",
        type=float,
        default=0.1,
        help=(
            "Softmax temperature for `--coral-contrastive-loss soft-infonce` "
            "(default: 0.1)."
        ),
    )
    parser.add_argument(
        "--coral-surrogate-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the experiment-side nonequilibrium surrogate loss used "
            "by `coral-surrogate` training (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--coral-surrogate-loss",
        choices=["smooth-l1", "mse"],
        default="smooth-l1",
        help=(
            "Loss used for the nonequilibrium surrogate objective in "
            "`coral-surrogate` training. `mse` is the squared L2 ablation "
            "(default: smooth-l1)."
        ),
    )
    parser.add_argument(
        "--coral-surrogate-learning-rate",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for pretraining the params->noneq surrogate used by "
            "`coral-surrogate` training (default: 1e-3)."
        ),
    )
    parser.add_argument(
        "--coral-surrogate-pretrain-epochs",
        type=int,
        default=200,
        help=(
            "Maximum number of simulation-only warmup epochs for the "
            "params->noneq surrogate used by `coral-surrogate` training "
            "(default: 200)."
        ),
    )
    parser.add_argument(
        "--coral-surrogate-pretrain-patience",
        type=int,
        default=20,
        help=(
            "Early-stopping patience for surrogate pretraining in "
            "`coral-surrogate` training (default: 20)."
        ),
    )
    parser.add_argument(
        "--coral-surrogate-checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Reusable params->noneq surrogate checkpoint. If omitted, defaults "
            "to `models/XPCS_noneq_surrogate_no_T.pt`."
        ),
    )
    parser.add_argument(
        "--force-coral-surrogate-retrain",
        action="store_true",
        help=(
            "Retrain and overwrite the reusable params->noneq surrogate even "
            "when `--coral-surrogate-checkpoint-path` already exists."
        ),
    )
    parser.add_argument(
        "--eval-crop-policy",
        choices=["auto", "top-left", "all-diagonal"],
        default="auto",
        help=(
            "How to crop raw experiment arrays during evaluation. `auto` keeps "
            "vanilla-only runs on the legacy top-left crop and switches any "
            "adversarial evaluation to multi-crop diagonal averaging."
        ),
    )
    parser.add_argument(
        "--eval-crop-aggregation",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate multiple crop predictions for one raw shot.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string, for example cpu or cuda.",
    )
    parser.add_argument(
        "--phase-diagram-sim-manifest",
        type=Path,
        default=None,
        help=(
            "Optional simulation manifest CSV used as the background for phase "
            "diagrams. Defaults to a nonequilibrium-enriched manifest when "
            "available."
        ),
    )
    parser.add_argument(
        "--phase-diagram-model",
        choices=["auto", "adv", "vanilla", "both"],
        default="auto",
        help=(
            "Which prediction columns to overlay on the phase diagram. `auto` "
            "uses whatever model columns are present in the results."
        ),
    )
    parser.add_argument(
        "--phase-diagram-aggregate-by",
        choices=["shot", "file"],
        default="shot",
        help=(
            "How to aggregate experimental prediction overlays for phase "
            "diagrams."
        ),
    )
    parser.add_argument(
        "--phase-diagram-shot-index",
        type=int,
        default=None,
        help=(
            "Optional shot index to keep when `--phase-diagram-aggregate-by shot` "
            "is used, for example 0 for fixed-q shot-zero phase diagrams."
        ),
    )
    parser.add_argument(
        "--phase-diagram-range-mode",
        choices=["stats", "data", "fixed"],
        default="stats",
        help=(
            "Coordinate ranges for phase diagrams. `stats` uses specs from "
            "the simulation dataset stats.json; `data` uses the sampled "
            "simulation manifest domain; `fixed` uses the legacy hard-coded "
            "plot ranges."
        ),
    )
    parser.add_argument(
        "--phase-diagram-split-by-material",
        action="store_true",
        help=(
            "Write per-material-dose phase diagrams in addition to the full "
            "combined phase diagrams."
        ),
    )
    parser.add_argument(
        "--phase-diagram-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for phase diagrams. Defaults to "
            "`<results-run>/phase_diagram`."
        ),
    )
    parser.add_argument(
        "--vanilla-model-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint path for vanilla evaluation. If provided, this overrides "
            "both the just-trained vanilla model and the default latest checkpoint."
        ),
    )
    parser.add_argument(
        "--adv-model-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint path for adversarial evaluation. If provided, this overrides "
            "both the just-trained adversarial model and the default latest checkpoint."
        ),
    )
    parser.add_argument(
        "--umap-sim-limit",
        type=int,
        default=None,
        help="Optional cap on the number of simulation samples used for UMAP.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-init", default="spectral")
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP embedding/plot generation and write CSV predictions only.",
    )
    return parser.parse_args()


def iter_raw_experiment_files(exp_data_dir: Path) -> list[Path]:
    """
    Collect all top-level raw experimental data files.

    Only files ending with `.npz` or `.npy` are included; nested directories are
    ignored.

    Args:
        exp_data_dir: Directory containing raw experimental files.

    Returns:
        raw_files: Sorted list of matching raw data paths.
    """
    if not exp_data_dir.exists():
        raise FileNotFoundError(f"Raw experiment directory not found: {exp_data_dir}")
    return sorted(
        path
        for path in exp_data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in RAW_SUFFIXES
    )


def filter_files(paths: Sequence[Path], selections: Sequence[str] | None) -> list[Path]:
    """
    Restrict a list of raw files to user-selected filenames or stems.

    Args:
        paths: Candidate raw data paths.
        selections: Optional sequence of filename stems or full filenames.

    Returns:
        selected: Filtered list of paths. If `selections` is None, returns all
            input paths.
    """
    if not selections:
        return list(paths)
    wanted = set(selections)
    selected = [path for path in paths if path.name in wanted or path.stem in wanted]
    missing = sorted(
        wanted
        - {path.name for path in selected}
        - {path.stem for path in selected}
    )
    if missing:
        raise FileNotFoundError(f"Requested raw experiment files were not found: {missing}")
    return selected


def parse_temperature_from_name(path: Path) -> float:
    """
    Parse the temperature encoded in a raw experiment filename.

    The filename is expected to contain a token like `T26C` or `T98.5C`.

    Args:
        path: Raw experiment file path.

    Returns:
        temperature_k: Temperature in Kelvin.
    """
    match = re.search(r"T(-?\d+(?:\.\d+)?)C", path.stem)
    if match is None:
        raise ValueError(f"Could not parse temperature from filename: {path.name}")
    return 273.15 + float(match.group(1))


def load_raw_experiment_array(path: Path) -> np.ndarray:
    """
    Load one raw experimental XPCS array from disk.

    `.npz` files are read from the `g12` key, while `.npy` files are loaded
    directly. Two-dimensional arrays are promoted to shape `[H, W, 1]`.

    Args:
        path: Path to a raw `.npz` or `.npy` experiment file.

    Returns:
        array: Experimental data array of shape `[H, W, B]`, where `B` is the
            number of shots.
    """
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "g12" not in data:
                raise KeyError(f"Missing 'g12' array in {path}")
            array = data["g12"]
    else:
        array = np.load(path)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array for {path.name}, got shape {array.shape}")
    return array


def build_diag_mask(size: int) -> torch.Tensor:
    """
    Create a diagonal mask used to hide the `t1 == t2` pixels from the models.

    Args:
        size: Spatial size of the square XPCS tensor.

    Returns:
        mask: Tensor of shape `[1, size, size]` with zeros on the diagonal and
            ones elsewhere.
    """
    mask = torch.ones(1, size, size, dtype=torch.float32)
    mask[0, range(size), range(size)] = 0.0
    return mask


def maybe_limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    """
    Optionally subsample a dataset for faster UMAP computation.

    Args:
        dataset: Input dataset.
        limit: Maximum number of samples to keep. If None or not restrictive,
            the original dataset is returned.

    Returns:
        dataset_out: Either the original dataset or a subset with evenly spaced
            indices.
    """
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, num=limit, dtype=int).tolist()
    return Subset(dataset, indices)


def parse_result_group(path: Path) -> tuple[str, float]:
    """
    Parse the material-dose group key and temperature from a result CSV filename.

    The filename is expected to follow the same pattern as the raw experiment
    file stem, for example `030BM_L_dose3_T96C.csv`.

    Args:
        path: Path to a result CSV file.

    Returns:
        group_key: Material-and-dose identifier, without the temperature suffix.
        temperature_c: Temperature in Celsius parsed from the filename.
    """
    match = re.match(r"^(.*_dose\d+)_T(-?\d+(?:\.\d+)?)C$", path.stem)
    if match is None:
        raise ValueError(f"Unexpected results filename format: {path.name}")
    return match.group(1), float(match.group(2))


def format_temperature_label(temperature_c: float) -> str:
    """
    Format a Kelvin temperature label for plots.

    Args:
        temperature_c: Temperature in Celsius.

    Returns:
        label: Legend-friendly label such as `299 K` or `371.6 K`.
    """
    temperature_k = temperature_c + 273.15
    if abs(temperature_k - round(temperature_k)) < 1e-6:
        return f"{int(round(temperature_k))} K"
    return f"{temperature_k:.1f} K"


def resolve_eval_crop_policy(
    requested_policy: str,
    eval_models: str,
) -> str:
    """
    Resolve the effective raw-shot crop policy for evaluation.

    Args:
        requested_policy: CLI crop policy value.
        eval_models: Which model variants are being evaluated.

    Returns:
        policy: Concrete crop policy, either `top-left` or `all-diagonal`.
    """
    if requested_policy != "auto":
        return requested_policy
    if eval_models in {"adv", "both", "coral-surrogate"}:
        return "all-diagonal"
    return "top-left"


def build_default_results_run_name(args: argparse.Namespace) -> str:
    """
    Build a readable per-run results directory name.

    Args:
        args: Parsed CLI arguments.

    Returns:
        run_name: Timestamped run directory name under `args.results_dir`.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    labels = [args.eval_models]
    if args.no_t:
        labels.append("noT")
    if args.eval_models in {"both", "vanilla"}:
        labels.append(f"vanseed{args.vanilla_seed}")
    if args.eval_models in {"both", "adv", "coral-surrogate"}:
        labels.append(f"advseed{args.adv_seed}")
    if args.train_models == "coral" or args.eval_models == "coral":
        if args.coral_contrastive_weight > 0.0:
            labels.append(f"contrastive-{args.coral_contrastive_loss}")
            labels.append(f"cw{args.coral_contrastive_weight:g}")
    labels.append(resolve_eval_crop_policy(args.eval_crop_policy, args.eval_models))
    labels.append(stamp)
    return "_".join(labels)


def resolve_evaluation_results_dir(args: argparse.Namespace) -> Path:
    """
    Resolve the output directory for one evaluation run.

    Args:
        args: Parsed CLI arguments.

    Returns:
        results_dir: Run-specific directory where this evaluation should write.
    """
    run_name = args.results_run_name or build_default_results_run_name(args)
    return args.results_dir / run_name


def resolve_sim_reconstruction_manifest(
    simulation_dataset_dir: Path,
    requested_manifest: Path | None,
) -> Path:
    """
    Resolve which simulation manifest to use for reconstruction diagnostics.

    Preference is given to manifests that already include a
    `nonequilibrium_measure` column so the selected samples can span the
    nonequilibrium distribution.
    """
    if requested_manifest is not None:
        return requested_manifest
    candidate_names = [
        "manifest_with_non_equ_1.csv",
        "manifest_with_non_equ.csv",
        "manifest_with_unequ.csv",
        "manifest.csv",
    ]
    for name in candidate_names:
        candidate = simulation_dataset_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No simulation manifest found under {simulation_dataset_dir}"
    )


def resolve_phase_diagram_sim_manifest(
    simulation_dataset_dir: Path,
    requested_manifest: Path | None,
) -> Path:
    """
    Resolve which simulation manifest should provide the background phase map.

    Preference is given to manifests that already include
    `nonequilibrium_measure`.
    """
    return resolve_sim_reconstruction_manifest(
        simulation_dataset_dir=simulation_dataset_dir,
        requested_manifest=requested_manifest,
    )
def generate_phase_diagrams(
    args: argparse.Namespace,
    results_dir: Path,
) -> None:
    """
    Generate simulation nonequilibrium phase diagrams overlaid with experiment
    inference results.
    """
    if args.phase_diagram_shot_index is not None and args.phase_diagram_aggregate_by != "shot":
        raise ValueError("--phase-diagram-shot-index requires --phase-diagram-aggregate-by shot")

    sim_manifest_path = resolve_phase_diagram_sim_manifest(
        simulation_dataset_dir=args.simulation_dataset_dir,
        requested_manifest=args.phase_diagram_sim_manifest,
    )
    simulation_df = pd.read_csv(sim_manifest_path)
    if "nonequilibrium_measure" not in simulation_df.columns:
        raise ValueError(
            f"Simulation manifest {sim_manifest_path} must contain `nonequilibrium_measure`"
        )

    result_csvs = iter_phase_diagram_result_csvs(results_dir)
    if not result_csvs:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")

    available_models = detect_phase_diagram_models(result_csvs)
    if args.phase_diagram_model == "auto":
        model_names = available_models
    elif args.phase_diagram_model == "both":
        model_names = [name for name in ["adv", "vanilla"] if name in available_models]
    else:
        model_names = [args.phase_diagram_model] if args.phase_diagram_model in available_models else []
    if not model_names:
        print("[phase] no matching model prediction columns available")
        return

    default_output_name = f"phase_diagrams_{args.phase_diagram_aggregate_by}"
    if args.phase_diagram_shot_index is not None:
        default_output_name = f"{default_output_name}_idx{args.phase_diagram_shot_index}"
    output_dir = (
        args.phase_diagram_output_dir
        if args.phase_diagram_output_dir is not None
        else results_dir / default_output_name
    )

    inference_plot_phase_diagrams(
        results_dir=results_dir,
        simulation_manifest=sim_manifest_path,
        output_dir=output_dir,
        model_names=model_names,
        aggregate_by=args.phase_diagram_aggregate_by,
        shot_index=args.phase_diagram_shot_index,
        split_by_material=False,
        raw_data_dir=args.exp_data_dir,
        crop_size=args.crop_size,
        coarse_size=args.coarse_size,
        range_mode=args.phase_diagram_range_mode,
    )

    if args.phase_diagram_split_by_material:
        by_material_output_dir = output_dir / "by_material"
        inference_plot_phase_diagrams(
            results_dir=results_dir,
            simulation_manifest=sim_manifest_path,
            output_dir=by_material_output_dir,
            model_names=model_names,
            aggregate_by=args.phase_diagram_aggregate_by,
            shot_index=args.phase_diagram_shot_index,
            split_by_material=True,
            raw_data_dir=args.exp_data_dir,
            crop_size=args.crop_size,
            coarse_size=args.coarse_size,
            range_mode=args.phase_diagram_range_mode,
        )


def select_reconstruction_rows(
    manifest_df: pd.DataFrame,
    num_samples: int,
) -> pd.DataFrame:
    """
    Select representative simulation rows for reconstruction diagnostics.

    When `nonequilibrium_measure` is available, the rows are sorted by that
    value and one high-nonequilibrium representative is taken from each
    quantile bin. Otherwise, evenly spaced rows are selected by index.
    """
    if num_samples <= 0:
        return manifest_df.iloc[0:0].copy()
    if manifest_df.empty:
        raise ValueError("Cannot select reconstruction rows from an empty manifest")

    df = manifest_df.reset_index(drop=True).copy()
    if "nonequilibrium_measure" in df.columns:
        df = df.sort_values("nonequilibrium_measure", ascending=True).reset_index(drop=True)
        boundaries = np.linspace(0, len(df), num_samples + 1, dtype=int)
        selected_indices: list[int] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            selected_indices.append(end - 1)
        if len(selected_indices) < num_samples:
            fallback_indices = np.linspace(0, len(df) - 1, num_samples, dtype=int)
            selected_indices = fallback_indices.tolist()
        selected = df.iloc[selected_indices].copy()
        selected["selection_mode"] = "nonequilibrium_quantiles"
        selected["selection_bin"] = range(1, len(selected) + 1)
        return selected.reset_index(drop=True)

    selected_indices = np.linspace(0, len(df) - 1, num_samples, dtype=int)
    selected = df.iloc[selected_indices].copy()
    selected["selection_mode"] = "uniform_index_spacing"
    selected["selection_bin"] = range(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def migrate_flat_result_csvs(results_dir: Path) -> None:
    """
    Move legacy flat result CSV files into material-dose subdirectories.

    Older runs stored all result CSV files directly under `results_dir`. This
    helper relocates each such CSV into `results_dir / <material_dose> /`.

    Args:
        results_dir: Root results directory.
    """
    for csv_path in sorted(results_dir.glob("*.csv")):
        group_key, _ = parse_result_group(csv_path)
        target_dir = results_dir / group_key
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / csv_path.name
        if csv_path == target_path:
            continue
        shutil.move(str(csv_path), str(target_path))
        print(f"[results] moved {csv_path.name} -> {target_dir.name}/")


def generate_grouped_bar_plots(
    results_dir: Path,
) -> None:
    """
    Generate grouped temperature-comparison plots from per-file result CSVs.

    For each material-dose group, this function averages the shot-level
    predictions within each result CSV, then plots parameter trends across
    temperatures. If both vanilla and adversarial predictions are present, it
    also writes direct comparison plots and pairwise scatter plots.

    Args:
        results_dir: Root results directory containing grouped result CSVs, or
            legacy flat CSVs that can be migrated automatically.
    """
    migrate_flat_result_csvs(results_dir)

    group_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    result_csvs = []
    for group_dir in group_dirs:
        result_csvs.extend(sorted(group_dir.glob("*.csv")))

    if not result_csvs:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")

    grouped_records: dict[str, list[dict[str, float | str]]] = {}
    grouped_shot_frames: dict[str, list[pd.DataFrame]] = {}
    for csv_path in result_csvs:
        try:
            group_key, temperature_c = parse_result_group(csv_path)
        except ValueError:
            print(f"[plot-bars] skipping {csv_path.name}: unsupported filename format")
            continue
        df = pd.read_csv(csv_path)
        df = df.copy()
        df["temperature_c"] = temperature_c
        df["temperature_label"] = format_temperature_label(temperature_c)
        record: dict[str, float | str] = {
            "temperature_c": temperature_c,
            "temperature_label": format_temperature_label(temperature_c),
        }
        has_predictions = False
        if {"D_adv", "gamma_adv", "lambda_GB_adv"}.issubset(df.columns):
            record.update({
                "D_adv": float(df["D_adv"].mean()),
                "gamma_adv": float(df["gamma_adv"].mean()),
                "lambda_GB_adv": float(df["lambda_GB_adv"].mean()),
            })
            has_predictions = True
        if {"D_vanilla", "gamma_vanilla", "lambda_GB_vanilla"}.issubset(df.columns):
            record.update({
                "D_vanilla": float(df["D_vanilla"].mean()),
                "gamma_vanilla": float(df["gamma_vanilla"].mean()),
                "lambda_GB_vanilla": float(df["lambda_GB_vanilla"].mean()),
            })
            has_predictions = True
        if not has_predictions:
            print(f"[plot-bars] skipping {csv_path.name}: no compatible prediction columns")
            continue
        if {"D_adv", "gamma_adv", "lambda_GB_adv", "D_vanilla", "gamma_vanilla", "lambda_GB_vanilla"}.issubset(df.columns):
            grouped_shot_frames.setdefault(group_key, []).append(df)
        grouped_records.setdefault(group_key, []).append(record)

    if not grouped_records:
        print("[plot-bars] no compatible result CSVs found for plotting")
        return

    for group_key, records in grouped_records.items():
        records.sort(key=lambda item: float(item["temperature_c"]))
        group_dir = results_dir / group_key
        group_dir.mkdir(parents=True, exist_ok=True)

        has_adv = all({"D_adv", "gamma_adv", "lambda_GB_adv"}.issubset(record) for record in records)
        has_vanilla = all({"D_vanilla", "gamma_vanilla", "lambda_GB_vanilla"}.issubset(record) for record in records)

        if has_adv:
            d_adv = {str(r["temperature_label"]): float(r["D_adv"]) * 1e23 for r in records}
            gamma_adv = {str(r["temperature_label"]): float(r["gamma_adv"]) * 1e-18 for r in records}
            gb_adv = {str(r["temperature_label"]): float(r["lambda_GB_adv"]) for r in records}
            plot_single_model_multi_bar_v2(
                params=d_adv,
                max_param=max(d_adv.values()),
                min_param=min(d_adv.values()),
                xname="Diffusitivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
                save_path=group_dir / "parameter_D_adv.pdf",
                model_name="Adversarial",
            )
            plot_single_model_multi_bar_v2(
                params=gamma_adv,
                max_param=max(gamma_adv.values()),
                min_param=min(gamma_adv.values()),
                xname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
                save_path=group_dir / "parameter_gamma_adv.pdf",
                model_name="Adversarial",
            )
            plot_single_model_multi_bar_v2(
                params=gb_adv,
                max_param=max(gb_adv.values()),
                min_param=min(gb_adv.values()),
                xname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
                save_path=group_dir / "parameter_GB_adv.pdf",
                model_name="Adversarial",
            )

        if has_vanilla:
            d_van = {str(r["temperature_label"]): float(r["D_vanilla"]) * 1e23 for r in records}
            gamma_van = {str(r["temperature_label"]): float(r["gamma_vanilla"]) * 1e-18 for r in records}
            gb_van = {str(r["temperature_label"]): float(r["lambda_GB_vanilla"]) for r in records}
            plot_single_model_multi_bar_v2(
                params=d_van,
                max_param=max(d_van.values()),
                min_param=min(d_van.values()),
                xname="Diffusitivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
                save_path=group_dir / "parameter_D_vanilla.pdf",
                model_name="Vanilla",
            )
            plot_single_model_multi_bar_v2(
                params=gamma_van,
                max_param=max(gamma_van.values()),
                min_param=min(gamma_van.values()),
                xname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
                save_path=group_dir / "parameter_gamma_vanilla.pdf",
                model_name="Vanilla",
            )
            plot_single_model_multi_bar_v2(
                params=gb_van,
                max_param=max(gb_van.values()),
                min_param=min(gb_van.values()),
                xname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
                save_path=group_dir / "parameter_GB_vanilla.pdf",
                model_name="Vanilla",
            )

        if has_adv and has_vanilla:
            all_d = list(d_adv.values()) + list(d_van.values())
            all_gamma = list(gamma_adv.values()) + list(gamma_van.values())
            all_gb = list(gb_adv.values()) + list(gb_van.values())

            plot_multi_bar_v2(
                params_adv=d_adv,
                params_van=d_van,
                max_param=max(all_d),
                min_param=min(all_d),
                xname="Diffusitivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
                save_path=group_dir / "parameter_D_comparison.pdf",
            )
            plot_multi_bar_v2(
                params_adv=gamma_adv,
                params_van=gamma_van,
                max_param=max(all_gamma),
                min_param=min(all_gamma),
                xname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
                save_path=group_dir / "parameter_gamma_comparison.pdf",
            )
            plot_multi_bar_v2(
                params_adv=gb_adv,
                params_van=gb_van,
                max_param=max(all_gb),
                min_param=min(all_gb),
                xname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
                save_path=group_dir / "parameter_GB_comparison.pdf",
            )
            shots_df = pd.concat(grouped_shot_frames[group_key], ignore_index=True)
            plot_parameter_pair_comparison(
                df=shots_df,
                save_path=group_dir / "parameter_scatter_D_gamma.pdf",
                x_adv="D_adv",
                y_adv="gamma_adv",
                x_van="D_vanilla",
                y_van="gamma_vanilla",
                xname="Diffusivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
                yname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
                x_scale_factor=1e23,
                y_scale_factor=1e-18,
            )
            plot_parameter_pair_comparison(
                df=shots_df,
                save_path=group_dir / "parameter_scatter_gamma_GB.pdf",
                x_adv="gamma_adv",
                y_adv="lambda_GB_adv",
                x_van="gamma_vanilla",
                y_van="lambda_GB_vanilla",
                xname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
                yname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
                x_scale_factor=1e-18,
            )
            plot_parameter_pair_comparison(
                df=shots_df,
                save_path=group_dir / "parameter_scatter_D_GB.pdf",
                x_adv="D_adv",
                y_adv="lambda_GB_adv",
                x_van="D_vanilla",
                y_van="lambda_GB_vanilla",
                xname="Diffusivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
                yname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
                x_scale_factor=1e23,
            )
        print(f"[plot-bars] wrote grouped plots under {group_dir}")


def generate_global_fixed_q_scatter_plots(
    results_dir: Path,
) -> None:
    """
    Generate one global scatter set across all result folders using only the
    fixed-q row (`shot_index == 0`) from each result CSV.

    Args:
        results_dir: Root results directory containing grouped result CSVs, or
            legacy flat CSVs that can be migrated automatically.
    """
    migrate_flat_result_csvs(results_dir)

    group_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    result_csvs = []
    for group_dir in group_dirs:
        result_csvs.extend(sorted(group_dir.glob("*.csv")))

    if not result_csvs:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")

    required_columns = {
        "shot_index",
        "D_adv", "gamma_adv", "lambda_GB_adv",
        "D_vanilla", "gamma_vanilla", "lambda_GB_vanilla",
    }
    global_rows: list[pd.DataFrame] = []
    for csv_path in result_csvs:
        try:
            group_key, temperature_c = parse_result_group(csv_path)
        except ValueError:
            print(f"[plot-bars] skipping {csv_path.name}: unsupported filename format")
            continue
        df = pd.read_csv(csv_path)
        if not required_columns.issubset(df.columns):
            continue
        fixed_q_df = df.loc[df["shot_index"] == 0].copy()
        if fixed_q_df.empty:
            continue
        fixed_q_df["group_key"] = group_key
        fixed_q_df["temperature_c"] = temperature_c
        fixed_q_df["temperature_label"] = format_temperature_label(temperature_c)
        global_rows.append(fixed_q_df)

    if not global_rows:
        print("[plot-bars] no fixed-q rows found for global scatter comparison")
        return

    global_df = pd.concat(global_rows, ignore_index=True)
    plot_parameter_pair_comparison(
        df=global_df,
        save_path=results_dir / "global_parameter_scatter_D_gamma_q0.pdf",
        x_adv="D_adv",
        y_adv="gamma_adv",
        x_van="D_vanilla",
        y_van="gamma_vanilla",
        xname="Diffusivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
        yname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
        x_scale_factor=1e23,
        y_scale_factor=1e-18,
        title="All experiment samples, fixed-q channel 0",
    )
    plot_parameter_pair_comparison(
        df=global_df,
        save_path=results_dir / "global_parameter_scatter_gamma_GB_q0.pdf",
        x_adv="gamma_adv",
        y_adv="lambda_GB_adv",
        x_van="gamma_vanilla",
        y_van="lambda_GB_vanilla",
        xname="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
        yname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
        x_scale_factor=1e-18,
        title="All experiment samples, fixed-q channel 0",
    )
    plot_parameter_pair_comparison(
        df=global_df,
        save_path=results_dir / "global_parameter_scatter_D_GB_q0.pdf",
        x_adv="D_adv",
        y_adv="lambda_GB_adv",
        x_van="D_vanilla",
        y_van="lambda_GB_vanilla",
        xname="Diffusivity " r"$D$ ($10^{-23}$ cm$^2$/s)",
        yname="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
        x_scale_factor=1e23,
        title="All experiment samples, fixed-q channel 0",
    )
    print(f"[plot-bars] wrote global fixed-q scatter plots under {results_dir}")


def build_diagonal_crop_starts(
    array_shape: tuple[int, int],
    crop_size: int,
    crop_step: int,
    crop_policy: str,
) -> list[int]:
    """
    Build the list of diagonal crop offsets used for one raw experimental shot.

    Args:
        array_shape: Spatial `(height, width)` of the raw XPCS matrix.
        crop_size: Side length of the square crop.
        crop_step: Step size between consecutive diagonal crops.
        crop_policy: Concrete crop policy, either `top-left` or `all-diagonal`.

    Returns:
        crop_starts: Ordered list of diagonal crop offsets.
    """
    height, width = array_shape
    if height != width:
        raise ValueError(f"Expected square raw experiment arrays, got {array_shape}")
    if height <= crop_size:
        # Raw data is already at or below crop size — use the full array
        return [0]
    if crop_policy == "top-left":
        return [0]
    if crop_policy != "all-diagonal":
        raise ValueError(f"Unsupported evaluation crop policy: {crop_policy}")
    if crop_step <= 0:
        raise ValueError(f"Crop step must be positive, got {crop_step}")
    return list(range(0, height - crop_size + 1, crop_step))


def aggregate_rows(values: np.ndarray, aggregation: str) -> np.ndarray:
    """
    Aggregate one or more per-crop vectors into one per-shot vector.

    Args:
        values: Array of shape `[N, D]`.
        aggregation: Reduction name, either `mean` or `median`.

    Returns:
        reduced: Array of shape `[D]`.
    """
    if aggregation == "mean":
        return values.mean(axis=0)
    if aggregation == "median":
        return np.median(values, axis=0)
    raise ValueError(f"Unsupported crop aggregation: {aggregation}")


def write_embedding_diagnostics(
    coords: np.ndarray,
    domain_labels: np.ndarray,
    samples: Sequence[ShotSample],
    results_dir: Path,
    model_suffix: str,
) -> None:
    """
    Write embedding diagnostics that color experimental points by metadata which
    can reveal shortcut structure such as crop position or source-file identity.

    Args:
        coords: Embedding coordinates for simulation followed by experiment rows.
        domain_labels: Domain labels aligned with `coords`.
        samples: Experimental samples in the same order used for evaluation.
        results_dir: Root directory for evaluation outputs.
        model_suffix: Filename suffix such as `adv` or `vanilla`.
    """
    crop_starts = [sample.crop_start for sample in samples]
    if len(set(crop_starts)) > 1:
        crop_start_path = results_dir / f"UMAP_{model_suffix}_crop_start.pdf"
        plot_experiment_metadata_embedding(
            X=coords,
            domain_labels=domain_labels,
            experiment_values=crop_starts,
            save_path=crop_start_path,
            value_label="crop_start",
            title="Experiment points colored by crop_start",
            categorical=False,
        )
        print(f"[evaluate] wrote {crop_start_path}")

    source_files = [sample.file_name for sample in samples]
    if len(set(source_files)) > 1:
        source_path = results_dir / f"UMAP_{model_suffix}_source_file.pdf"
        plot_experiment_metadata_embedding(
            X=coords,
            domain_labels=domain_labels,
            experiment_values=source_files,
            save_path=source_path,
            value_label="source_file",
            title="Experiment points colored by source file",
            categorical=True,
        )
        print(f"[evaluate] wrote {source_path}")

    noneq_values = [sample.nonequilibrium_measure for sample in samples]
    noneq_path = results_dir / f"UMAP_{model_suffix}_nonequilibrium.pdf"
    plot_experiment_metadata_embedding(
        X=coords,
        domain_labels=domain_labels,
        experiment_values=noneq_values,
        save_path=noneq_path,
        value_label="nonequilibrium_measure",
        title="Experiment points colored by nonequilibrium measure",
        categorical=False,
    )
    print(f"[evaluate] wrote {noneq_path}")


def preprocess_experiment_dataset(
    raw_files: Sequence[Path],
    output_dir: Path,
    crop_size: int,
    coarse_size: int,
    crop_step: int,
    clean_output_dir: bool,
) -> None:
    """
    Rebuild the processed experiment tensor dataset from raw experiment files.

    Each raw file is cropped along its diagonal into multiple patches, then
    coarse-grained and merged into one flat processed dataset directory with a
    unified manifest.

    Args:
        raw_files: Raw experiment files to process.
        output_dir: Destination directory for merged processed tensors.
        crop_size: Square crop size taken from the raw data.
        coarse_size: Final coarse-grained tensor size.
        crop_step: Step size between diagonal crops.
        clean_output_dir: Whether to delete the existing processed directory
            before rebuilding it.
    """
    if clean_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)

    temp_dir = output_dir.parent / f"{output_dir.name}_rebuild_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in raw_files:
        data = load_raw_experiment_array(raw_file)
        temperature = parse_temperature_from_name(raw_file)
        crop_data(
            data=data,
            temperature=temperature,
            crop_size=crop_size,
            coarse_size=coarse_size,
            step=crop_step,
            save_path=temp_dir,
            source_name=raw_file.name,
            sample_name=raw_file.stem,
        )
        print(f"[preprocess] processed {raw_file.name}")

    merge_data(
        dataset_path=temp_dir,
        output_data_dir=output_dir,
        remove_original=True,
    )
    print(f"[preprocess] rebuilt experiment dataset at {output_dir}")


def prepare_experiment_shots(
    raw_files: Sequence[Path],
    shot_indices: Sequence[int] | None,
    crop_size: int,
    coarse_size: int,
    crop_step: int,
    crop_policy: str,
    no_t: bool,
    cache_dir: Path | None = None,
) -> list[ShotSample]:
    """
    Convert raw experimental files into shot-level inputs for model evaluation.

    Args:
        raw_files: Raw experiment files to evaluate.
        shot_indices: Optional subset of shot indices to keep for each file.
        crop_size: Spatial crop size taken from the raw data.
        coarse_size: Final coarse-grained tensor size for model input.
        crop_step: Step size between diagonal crops when multi-crop evaluation
            is enabled.
        crop_policy: Concrete crop policy, either `top-left` or
            `all-diagonal`.

    Returns:
        samples: Flat list of prepared shot records across all selected files.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = resolve_experiment_shot_cache_path(
            raw_files=raw_files,
            shot_indices=shot_indices,
            crop_size=crop_size,
            coarse_size=coarse_size,
            crop_step=crop_step,
            crop_policy=crop_policy,
            no_t=no_t,
            cache_dir=cache_dir,
        )
        if cache_path.exists():
            print(f"[evaluate] loading prepared experiment shots from cache: {cache_path}")
            return load_prepared_experiment_shots(cache_path)

    diag_mask = build_diag_mask(coarse_size)
    selected_indices = None if shot_indices is None else set(shot_indices)
    samples: list[ShotSample] = []

    for raw_file in raw_files:
        data = load_raw_experiment_array(raw_file)
        temperature = parse_temperature_from_name(raw_file)
        crop_starts = build_diagonal_crop_starts(
            array_shape=data.shape[:2],
            crop_size=crop_size,
            crop_step=crop_step,
            crop_policy=crop_policy,
        )
        n_shots = data.shape[-1]
        if selected_indices is None:
            selected_shots = list(range(n_shots))
        else:
            selected_shots = [idx for idx in sorted(selected_indices) if 0 <= idx < n_shots]
        for shot_index in selected_shots:
            for crop_start in crop_starts:
                shot = torch.tensor(
                    data[
                        crop_start:crop_start + crop_size,
                        crop_start:crop_start + crop_size,
                        shot_index,
                    ],
                    dtype=torch.float32,
                )
                g2 = coarse_grain_g2(
                    shot,
                    target_size=(coarse_size, coarse_size),
                ).to(torch.float32)
                if no_t:
                    g2 = (g2 - INPUT_MEAN_NO_T) / (INPUT_STD_NO_T + 1e-6)
                else:
                    g2 = normalize_g2(g2, min_val=1.0, max_val=1.2)
                samples.append(
                    ShotSample(
                        file_name=raw_file.name,
                        file_stem=raw_file.stem,
                        shot_index=shot_index,
                        crop_start=crop_start,
                        temperature_k=temperature,
                        x=g2.unsqueeze(0) * diag_mask,
                        nonequilibrium_measure=nonequilibrium_measure(g2),
                    )
                )
        print(
            f"[evaluate] prepared {raw_file.name} with {len(selected_shots)} "
            f"selected shot(s) and {len(crop_starts)} crop(s) per shot"
        )

    if not samples:
        raise RuntimeError("No experiment shots were selected for evaluation.")
    if cache_path is not None:
        save_prepared_experiment_shots(cache_path, samples)
        print(f"[evaluate] cached prepared experiment shots at {cache_path}")
    return samples


def run_training(args: argparse.Namespace) -> tuple[VanillaModel | None, AdvModel | None]:
    """
    Train the requested model variants using the existing training modules.

    Args:
        args: Parsed CLI arguments controlling which models to train and with
            what hyperparameters.

    Returns:
        vanilla_model: Trained vanilla model, if requested.
        adv_model: Trained adversarial model, if requested.
    """
    device = torch.device(args.device)
    vanilla_model = None
    adv_model = None
    vanilla_model_class, vanilla_train_fn, vanilla_load_fn = resolve_vanilla_components(
        args.no_t
    )
    adv_model_class, adv_train_fn, _ = resolve_adv_components(args.no_t)

    if args.train_models == "none":
        print("[train] skipped because --train-models none")
        return vanilla_model, adv_model

    if args.train_models in {"both", "vanilla"}:
        print("[train] training vanilla model")
        set_vanilla_global_seed(
            args.vanilla_seed,
            deterministic=not args.non_deterministic_vanilla,
        )
        if args.no_t:
            vanilla_model_instance = vanilla_model_class(
                predictor_output_activation="sigmoid",
            )
        else:
            vanilla_model_instance = vanilla_model_class(
                use_shared_feature_mixer=True,
                shared_feature_dim=args.vanilla_shared_feature_dim,
                shared_feature_mixer_hidden_dim=(
                    args.vanilla_shared_feature_mixer_hidden_dim
                ),
                predictor_output_activation="sigmoid",
            )
        vanilla_model = vanilla_train_fn(
            vanilla_model_instance,
            sim_root=args.simulation_dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.vanilla_learning_rate,
            seed=args.vanilla_seed,
            deterministic=not args.non_deterministic_vanilla,
            num_workers=args.vanilla_num_workers,
            device=device,
            log_pardir=args.runs_dir,
            model_path=args.models_dir,
        )

    if args.train_models in {"both", "adv"}:
        print("[train] training adversarial model")
        adv_init_state_dict = None
        if args.adv_init_vanilla_model_path is not None:
            adv_init_state_dict = torch.load(
                args.adv_init_vanilla_model_path,
                weights_only=True,
                map_location="cpu",
            )
            print(
                "[train] initializing adversarial model from vanilla checkpoint "
                f"{args.adv_init_vanilla_model_path}"
            )
        elif args.adv_init_from_vanilla and vanilla_model is not None:
            adv_init_state_dict = vanilla_model.state_dict()
            print("[train] initializing adversarial model from the current vanilla run")
        elif args.adv_init_from_vanilla:
            try:
                latest_vanilla_model = vanilla_load_fn(
                    None,
                    device=torch.device("cpu"),
                )
                adv_init_state_dict = latest_vanilla_model.state_dict()
                print(
                    "[train] initializing adversarial model from the latest "
                    "available vanilla checkpoint"
                )
            except FileNotFoundError:
                print(
                    "[train] no vanilla checkpoint found for adversarial "
                    "initialization; training from scratch"
                )
        else:
            print("[train] adversarial model will train from scratch")
        set_vanilla_global_seed(
            args.adv_seed,
            deterministic=not args.non_deterministic_adv,
        )
        if args.no_t:
            adv_model_instance = adv_model_class(
                predictor_output_activation="sigmoid",
            )
        else:
            adv_model_instance = adv_model_class(
                use_shared_feature_mixer=args.adv_use_shared_feature_mixer,
                shared_feature_dim=args.adv_shared_feature_dim,
                shared_feature_mixer_hidden_dim=args.adv_shared_feature_mixer_hidden_dim,
                predictor_output_activation="sigmoid",
            )
        adv_model = adv_train_fn(
            adv_model_instance,
            sim_root=args.simulation_dataset_dir,
            exp_root=args.experiment_dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.adv_learning_rate,
            domain_learning_rate=args.adv_domain_learning_rate,
            adaptation_rate=args.adaptation_rate,
            seed=args.adv_seed,
            deterministic=not args.non_deterministic_adv,
            num_workers=args.adv_num_workers,
            warmup_epochs=args.adv_warmup_epochs,
            domain_pretrain_epochs=args.adv_domain_pretrain_epochs,
            domain_steps_per_iteration=args.adv_domain_steps_per_iteration,
            prediction_steps_per_iteration=args.adv_prediction_steps_per_iteration,
            domain_only_passes=args.adv_domain_only_passes,
            init_state_dict=adv_init_state_dict,
            device=device,
            log_pardir=args.runs_dir,
            model_path=args.models_dir,
        )

    if args.train_models == "coral":
        print("[train] training CORAL + distillation model")
        if not args.no_t:
            raise NotImplementedError("CORAL training with T not yet implemented")
        coral_model_class, coral_train_fn, _ = resolve_coral_components(args.no_t)
        coral_init_state_dict = None
        if args.coral_init_model_path is not None:
            coral_init_state_dict = torch.load(
                args.coral_init_model_path,
                weights_only=True,
                map_location="cpu",
            )
            print(
                "[train] initializing CORAL model from CORAL checkpoint "
                f"{args.coral_init_model_path}"
            )
        elif args.adv_init_vanilla_model_path is not None:
            coral_init_state_dict = torch.load(
                args.adv_init_vanilla_model_path,
                weights_only=True,
                map_location="cpu",
            )
            print(
                "[train] initializing CORAL model from vanilla checkpoint "
                f"{args.adv_init_vanilla_model_path}"
            )
        elif args.adv_init_from_vanilla:
            try:
                latest_vanilla_model = vanilla_load_fn(
                    None,
                    device=torch.device("cpu"),
                )
                coral_init_state_dict = latest_vanilla_model.state_dict()
                print(
                    "[train] initializing CORAL model from the latest "
                    "available vanilla checkpoint"
                )
            except FileNotFoundError:
                print(
                    "[train] no vanilla checkpoint found for CORAL "
                    "initialization; training from scratch"
                )
        set_vanilla_global_seed(
            args.adv_seed,
            deterministic=not args.non_deterministic_adv,
        )
        coral_model_instance = coral_model_class(
            predictor_output_activation="sigmoid",
        )
        adv_model = coral_train_fn(
            coral_model_instance,
            sim_root=args.simulation_dataset_dir,
            exp_root=args.experiment_dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.adv_learning_rate,
            coral_weight=args.coral_weight,
            contrastive_weight=args.coral_contrastive_weight,
            contrastive_loss_type=args.coral_contrastive_loss,
            contrastive_bandwidth=args.coral_contrastive_bandwidth,
            contrastive_feature_margin=args.coral_contrastive_margin,
            contrastive_infonce_temperature=args.coral_infonce_temperature,
            seed=args.adv_seed,
            deterministic=not args.non_deterministic_adv,
            num_workers=args.adv_num_workers,
            init_state_dict=coral_init_state_dict,
            device=device,
            log_pardir=args.runs_dir,
            model_path=args.models_dir,
        )

    if args.train_models == "coral-surrogate":
        print("[train] training CORAL + surrogate model")
        if not args.no_t:
            raise NotImplementedError("CORAL surrogate training with T not yet implemented")
        coral_model_class, coral_train_fn, _ = resolve_coral_surrogate_components(args.no_t)
        coral_init_state_dict = None
        if args.coral_init_model_path is not None:
            coral_init_state_dict = torch.load(
                args.coral_init_model_path,
                weights_only=True,
                map_location="cpu",
            )
            print(
                "[train] initializing CORAL surrogate model from checkpoint "
                f"{args.coral_init_model_path}"
            )
        elif args.adv_init_vanilla_model_path is not None:
            coral_init_state_dict = torch.load(
                args.adv_init_vanilla_model_path,
                weights_only=True,
                map_location="cpu",
            )
            print(
                "[train] initializing CORAL surrogate model from vanilla checkpoint "
                f"{args.adv_init_vanilla_model_path}"
            )
        elif args.adv_init_from_vanilla:
            try:
                latest_vanilla_model = vanilla_load_fn(
                    None,
                    device=torch.device("cpu"),
                )
                coral_init_state_dict = latest_vanilla_model.state_dict()
                print(
                    "[train] initializing CORAL surrogate model from the latest "
                    "available vanilla checkpoint"
                )
            except FileNotFoundError:
                print(
                    "[train] no vanilla checkpoint found for CORAL surrogate "
                    "initialization; training from scratch"
                )
        set_vanilla_global_seed(
            args.adv_seed,
            deterministic=not args.non_deterministic_adv,
        )
        coral_model_instance = coral_model_class(
            predictor_output_activation="sigmoid",
        )
        adv_model = coral_train_fn(
            coral_model_instance,
            sim_root=args.simulation_dataset_dir,
            exp_root=args.experiment_dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.adv_learning_rate,
            coral_weight=args.coral_weight,
            surrogate_weight=args.coral_surrogate_weight,
            surrogate_loss_type=args.coral_surrogate_loss,
            surrogate_learning_rate=args.coral_surrogate_learning_rate,
            surrogate_pretrain_epochs=args.coral_surrogate_pretrain_epochs,
            surrogate_pretrain_patience=args.coral_surrogate_pretrain_patience,
            surrogate_checkpoint_path=args.coral_surrogate_checkpoint_path,
            force_surrogate_retrain=args.force_coral_surrogate_retrain,
            seed=args.adv_seed,
            deterministic=not args.non_deterministic_adv,
            num_workers=args.adv_num_workers,
            init_state_dict=coral_init_state_dict,
            device=device,
            log_pardir=args.runs_dir,
            model_path=args.models_dir,
        )

    return vanilla_model, adv_model


def ensure_eval_models(
    args: argparse.Namespace,
    trained_vanilla: VanillaModel | None,
    trained_adv: AdvModel | None,
) -> tuple[VanillaModel | None, AdvModel | None]:
    """
    Resolve the models to use for evaluation.

    Explicit checkpoint paths take precedence. Otherwise, preference is given
    to models trained in the current run, then to the latest matching
    checkpoint on disk.

    Args:
        args: Parsed CLI arguments.
        trained_vanilla: Vanilla model produced in the current run, if any.
        trained_adv: Adversarial model produced in the current run, if any.

    Returns:
        vanilla_model: Model to use for vanilla evaluation, if requested.
        adv_model: Model to use for adversarial evaluation, if requested.
    """
    device = torch.device(args.device)
    vanilla_model = None
    adv_model = None
    _, _, vanilla_load_fn = resolve_vanilla_components(args.no_t)
    _, _, adv_load_fn = resolve_adv_components(args.no_t)

    if args.eval_models in {"both", "vanilla"}:
        if args.vanilla_model_path is not None:
            vanilla_model = vanilla_load_fn(args.vanilla_model_path, device=device)
        elif trained_vanilla is not None:
            vanilla_model = trained_vanilla
        else:
            vanilla_model = vanilla_load_fn(None, device=device)

    if args.eval_models in {"both", "adv"}:
        if args.adv_model_path is not None:
            adv_model = adv_load_fn(args.adv_model_path, device=device)
        elif trained_adv is not None:
            adv_model = trained_adv
        else:
            adv_model = adv_load_fn(None, device=device)

    if args.eval_models == "coral":
        if trained_adv is not None:
            adv_model = trained_adv
        else:
            _, _, coral_load_fn = resolve_coral_components(args.no_t)
            if args.adv_model_path is not None:
                adv_model = coral_load_fn(args.adv_model_path, device=device)
            else:
                adv_model = coral_load_fn(None, device=device)

    if args.eval_models == "coral-surrogate":
        if trained_adv is not None:
            adv_model = trained_adv
        else:
            _, _, coral_surrogate_load_fn = resolve_coral_surrogate_components(args.no_t)
            if args.adv_model_path is not None:
                adv_model = coral_surrogate_load_fn(args.adv_model_path, device=device)
            else:
                adv_model = coral_surrogate_load_fn(None, device=device)

    return vanilla_model, adv_model


def predict_samples(
    model: VanillaModel | AdvModel,
    dataset: PreparedExperimentDataset,
    batch_size: int,
    device: torch.device,
    denorm_fn: object,
) -> np.ndarray:
    """
    Run batched inference on prepared experimental shots.

    Args:
        model: Trained vanilla or adversarial model.
        dataset: Prepared experimental shot dataset.
        batch_size: Evaluation batch size.
        device: Device used for inference.

    Returns:
        predictions: Array of de-normalized predictions with shape `[N, 3]`,
            ordered as `(gamma, D, GB_conc)`.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    if hasattr(model, "on_pred_mode") and hasattr(model, "off_class_mode"):
        model.on_pred_mode().off_class_mode().set_grl_alpha(1.0)

    predictions = []
    with torch.no_grad():
        for x, _, _, temperature, _ in loader:
            x = x.to(device)
            temperature = temperature.to(device)
            pred_params_norm = model(x, temperature)
            pred_params_raw = denorm_fn(
                pred_params_norm,
                model.norm_meta,
                device=device,
            )
            predictions.append(pred_params_raw.cpu().numpy())
    return np.concatenate(predictions, axis=0)

def write_results_csvs(
    samples: Sequence[ShotSample],
    adv_predictions: np.ndarray | None,
    vanilla_predictions: np.ndarray | None,
    adv_coords: np.ndarray | None,
    adv_sim_count: int,
    vanilla_coords: np.ndarray | None,
    vanilla_sim_count: int,
    results_dir: Path,
    skip_umap: bool,
    crop_policy: str,
    crop_aggregation: str,
) -> None:
    """
    Write one prediction CSV per raw experiment file into its material-dose
    subdirectory under `results_dir`.

    The CSV filename matches the raw experiment filename stem. If UMAP was
    computed, the per-shot coordinates are also included as columns.

    Args:
        samples: Flat list of prepared shot records.
        adv_predictions: Adversarial predictions of shape `[N, 3]`, if available.
        vanilla_predictions: Vanilla predictions of shape `[N, 3]`, if available.
        adv_coords: Adversarial UMAP coordinates, if computed.
        adv_sim_count: Number of simulation rows in `adv_coords`.
        vanilla_coords: Vanilla UMAP coordinates, if computed.
        vanilla_sim_count: Number of simulation rows in `vanilla_coords`.
        results_dir: Root directory for evaluation outputs.
        skip_umap: Whether UMAP generation was skipped.
        crop_policy: Effective evaluation crop policy.
        crop_aggregation: Aggregation used to reduce multiple crops per shot.
    """
    del skip_umap
    grouped_samples: dict[str, dict[int, list[tuple[int, ShotSample]]]] = {}
    for idx, sample in enumerate(samples):
        grouped_samples.setdefault(sample.file_stem, {}).setdefault(
            sample.shot_index,
            [],
        ).append((idx, sample))

    for file_stem, grouped_by_shot in grouped_samples.items():
        rows = []
        for shot_index, indexed_samples in sorted(grouped_by_shot.items()):
            sample_indices = np.array([global_idx for global_idx, _ in indexed_samples], dtype=int)
            first_sample = indexed_samples[0][1]
            crop_starts = [sample.crop_start for _, sample in indexed_samples]
            row = {
                "file_name": first_sample.file_name,
                "shot_index": shot_index,
                "temperature_k": first_sample.temperature_k,
                "num_crops": len(indexed_samples),
                "crop_policy": crop_policy,
                "crop_aggregation": crop_aggregation,
                "crop_start_min": min(crop_starts),
                "crop_start_max": max(crop_starts),
                "nonequilibrium_measure": float(aggregate_rows(
                    np.array(
                        [[sample.nonequilibrium_measure] for _, sample in indexed_samples],
                        dtype=np.float64,
                    ),
                    crop_aggregation,
                )[0]),
            }
            if adv_predictions is not None:
                adv_values = aggregate_rows(adv_predictions[sample_indices], crop_aggregation)
                row.update({
                    "D_adv": float(adv_values[1]),
                    "gamma_adv": float(adv_values[0]),
                    "lambda_GB_adv": float(adv_values[2]),
                })
            if vanilla_predictions is not None:
                vanilla_values = aggregate_rows(
                    vanilla_predictions[sample_indices],
                    crop_aggregation,
                )
                row.update({
                    "D_vanilla": float(vanilla_values[1]),
                    "gamma_vanilla": float(vanilla_values[0]),
                    "lambda_GB_vanilla": float(vanilla_values[2]),
                })
            if adv_coords is not None:
                adv_coord_values = aggregate_rows(
                    adv_coords[adv_sim_count + sample_indices],
                    crop_aggregation,
                )
                row.update({
                    "umap_adv_x": float(adv_coord_values[0]),
                    "umap_adv_y": float(adv_coord_values[1]),
                })
            if vanilla_coords is not None:
                vanilla_coord_values = aggregate_rows(
                    vanilla_coords[vanilla_sim_count + sample_indices],
                    crop_aggregation,
                )
                row.update({
                    "umap_vanilla_x": float(vanilla_coord_values[0]),
                    "umap_vanilla_y": float(vanilla_coord_values[1]),
                })
            rows.append(row)

        group_key, _ = parse_result_group(Path(f"{file_stem}.csv"))
        group_dir = results_dir / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        output_path = group_dir / f"{file_stem}.csv"
        pd.DataFrame(rows).sort_values("shot_index").to_csv(
            output_path,
            index=False,
        )
        print(f"[evaluate] wrote {output_path}")


def write_sim_reconstruction_diagnostics(
    model: VanillaModel,
    simulation_dataset_dir: Path,
    results_dir: Path,
    device: torch.device,
    num_samples: int,
    manifest_path: Path | None,
    no_t: bool,
) -> None:
    """
    Reconstruct a small set of simulation spectra from vanilla model
    predictions and compare the reconstructed `g2` against the original stored
    simulation spectra.

    The selected rows span the nonequilibrium distribution when an enriched
    simulation manifest is available.
    """
    if num_samples <= 0:
        return

    resolved_manifest_path = resolve_sim_reconstruction_manifest(
        simulation_dataset_dir,
        manifest_path,
    )
    manifest_df = pd.read_csv(resolved_manifest_path)
    selected_rows = select_reconstruction_rows(manifest_df, num_samples)
    if selected_rows.empty:
        print("[evaluate] no simulation rows selected for reconstruction diagnostics")
        return

    sim_dataset_cls, denorm_fn = resolve_eval_dataset_components(no_t)
    sim_dataset = sim_dataset_cls(simulation_dataset_dir)
    if "path" not in sim_dataset.manifest.columns:
        raise ValueError("Simulation manifest must contain a `path` column")
    dataset_index_by_path = {
        str(path): idx for idx, path in enumerate(sim_dataset.manifest["path"])
    }

    diagnostics_dir = results_dir / "sim_reconstruction_vanilla"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    model.eval()
    if hasattr(model, "on_pred_mode") and hasattr(model, "off_class_mode"):
        model.on_pred_mode().off_class_mode().set_grl_alpha(1.0)

    summary_rows = []
    for sample_rank, row in enumerate(selected_rows.itertuples(index=False), start=1):
        row_dict = row._asdict()
        row_path = str(row_dict["path"])
        dataset_idx = dataset_index_by_path.get(row_path)
        if dataset_idx is None:
            print(f"[evaluate] skipping reconstruction sample not found in dataset: {row_path}")
            continue

        x, _, y_raw, temperature, _ = sim_dataset[dataset_idx]
        original_coarse = torch.load(row_path, weights_only=True).to(torch.float32).squeeze(0)
        original_g2 = normalize_g2(original_coarse, min_val=1.0, max_val=1.2)

        with torch.no_grad():
            pred_norm = model(
                x.unsqueeze(0).to(device),
                temperature.unsqueeze(0).to(device),
            )
            pred_raw = denorm_fn(pred_norm, model.norm_meta, device=device).squeeze(0).cpu()

        reconstructed = simulate_xpcs(
            gamma=float(pred_raw[0].item()),
            D=float(pred_raw[1].item()),
            GB_conc=float(pred_raw[2].item()),
            T=float(temperature.item()),
            seed=42,
            coarse=True,
        )
        if reconstructed is None:
            print(
                "[evaluate] skipped unstable reconstruction for simulation sample "
                f"{sample_rank}"
            )
            continue
        reconstructed_g2 = normalize_g2(
            reconstructed.to(torch.float32),
            min_val=1.0,
            max_val=1.2,
        )

        recon_mse = float(torch.mean((original_g2 - reconstructed_g2) ** 2).item())
        recon_mae = float(torch.mean(torch.abs(original_g2 - reconstructed_g2)).item())
        sample_prefix = f"sample_{sample_rank:02d}"
        noneq_value = row_dict.get("nonequilibrium_measure")
        noneq_text = (
            f"noneq={float(noneq_value):.4f}"
            if noneq_value is not None and not pd.isna(noneq_value)
            else "noneq=n/a"
        )
        g2_path = diagnostics_dir / f"{sample_prefix}_g2_comparison.pdf"
        autocorr_path = diagnostics_dir / f"{sample_prefix}_autocorrelation_comparison.pdf"
        plot_g2_side_by_side(
            original_g2,
            reconstructed_g2,
            save_path=g2_path,
            left_title=f"Original | {noneq_text}",
            right_title=f"Reconstructed | MSE={recon_mse:.4e}",
        )
        plot_auto_correlation_comparison(
            original_g2,
            reconstructed_g2,
            save_path=autocorr_path,
            title=f"Sample {sample_rank}: original vs reconstructed",
        )

        summary_rows.append({
            "sample_rank": sample_rank,
            "selection_mode": row_dict.get("selection_mode"),
            "selection_bin": row_dict.get("selection_bin"),
            "dataset_index": dataset_idx,
            "path": row_path,
            "id": row_dict.get("id"),
            "nonequilibrium_measure": row_dict.get("nonequilibrium_measure"),
            "temperature_k": float(temperature.item()),
            "gamma_true": float(y_raw[0].item()),
            "D_true": float(y_raw[1].item()),
            "GB_conc_true": float(y_raw[2].item()),
            "gamma_pred": float(pred_raw[0].item()),
            "D_pred": float(pred_raw[1].item()),
            "GB_conc_pred": float(pred_raw[2].item()),
            "gamma_abs_error": abs(float(pred_raw[0].item() - y_raw[0].item())),
            "D_abs_error": abs(float(pred_raw[1].item() - y_raw[1].item())),
            "GB_conc_abs_error": abs(float(pred_raw[2].item() - y_raw[2].item())),
            "reconstruction_mse": recon_mse,
            "reconstruction_mae": recon_mae,
            "g2_comparison_path": str(g2_path),
            "autocorrelation_comparison_path": str(autocorr_path),
        })

    if not summary_rows:
        print("[evaluate] no simulation reconstruction diagnostics were written")
        return

    summary_path = diagnostics_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[evaluate] wrote {summary_path}")


def evaluate_models(
    args: argparse.Namespace,
    raw_files: Sequence[Path],
    vanilla_model: VanillaModel | None,
    adv_model: AdvModel | None,
) -> Path:
    """
    Evaluate the requested trained models on raw experimental shots.

    This step prepares shot-level tensors from raw files, runs inference,
    optionally computes UMAP embeddings against the simulation dataset, and
    writes the final per-file outputs.

    Args:
        args: Parsed CLI arguments.
        raw_files: Raw experiment files selected for evaluation.
        vanilla_model: Vanilla model to evaluate, if requested.
        adv_model: Adversarial model to evaluate, if requested.

    Returns:
        results_dir: Run-specific results directory used for this evaluation.
    """
    results_dir = resolve_evaluation_results_dir(args)
    if (not args.keep_existing_results_dir) and results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] writing outputs under {results_dir}")
    eval_crop_policy = resolve_eval_crop_policy(args.eval_crop_policy, args.eval_models)
    print(
        "[evaluate] crop policy: "
        f"{eval_crop_policy} (aggregation={args.eval_crop_aggregation})"
    )

    samples = prepare_experiment_shots(
        raw_files=raw_files,
        shot_indices=args.shot_indices,
        crop_size=args.crop_size,
        coarse_size=args.coarse_size,
        crop_step=args.crop_step,
        crop_policy=eval_crop_policy,
        no_t=args.no_t,
        cache_dir=args.experiment_shot_cache_dir,
    )
    exp_dataset = PreparedExperimentDataset(samples)
    device = torch.device(args.device)
    sim_dataset_cls, denorm_fn = resolve_eval_dataset_components(args.no_t)

    vanilla_predictions = None
    adv_predictions = None
    if vanilla_model is not None:
        print("[evaluate] running vanilla predictions")
        vanilla_predictions = predict_samples(
            model=vanilla_model,
            dataset=exp_dataset,
            batch_size=args.eval_batch_size,
            device=device,
            denorm_fn=denorm_fn,
        )
    if adv_model is not None:
        print("[evaluate] running adversarial predictions")
        adv_predictions = predict_samples(
            model=adv_model,
            dataset=exp_dataset,
            batch_size=args.eval_batch_size,
            device=device,
            denorm_fn=denorm_fn,
        )

    adv_coords = None
    vanilla_coords = None
    adv_sim_count = 0
    vanilla_sim_count = 0
    adv_domain_labels = None
    vanilla_domain_labels = None
    if not args.skip_umap:
        sim_dataset = maybe_limit_dataset(
            sim_dataset_cls(args.simulation_dataset_dir),
            args.umap_sim_limit,
        )
        total_points = len(sim_dataset) + len(exp_dataset)
        n_neighbors = max(2, min(args.umap_neighbors, total_points - 1))

        if adv_model is not None:
            print("[evaluate] computing adversarial UMAP")
            adv_coords, adv_domain_labels = calc_umap(
                adv_model,
                sim_dataset,
                exp_dataset,
                device=device,
                n_neighbors=n_neighbors,
                min_dist=args.umap_min_dist,
                init=args.umap_init,
                random_state=args.umap_random_state,
            )
            adv_sim_count = len(sim_dataset)
        if vanilla_model is not None:
            print("[evaluate] computing vanilla UMAP")
            vanilla_coords, vanilla_domain_labels = calc_umap(
                vanilla_model,
                sim_dataset,
                exp_dataset,
                device=device,
                n_neighbors=n_neighbors,
                min_dist=args.umap_min_dist,
                init=args.umap_init,
                random_state=args.umap_random_state,
            )
            vanilla_sim_count = len(sim_dataset)

        if adv_coords is not None:
            plot_cluster(
                adv_coords,
                adv_domain_labels,
                results_dir / "UMAP.pdf",
                sim_marker="o",
                exp_marker="o",
                sim_marker_size=600,
                exp_marker_size=600,
            )
            print(f"[evaluate] wrote {results_dir / 'UMAP.pdf'}")
            write_embedding_diagnostics(
                coords=adv_coords,
                domain_labels=adv_domain_labels,
                samples=samples,
                results_dir=results_dir,
                model_suffix="adv",
            )
        if vanilla_coords is not None:
            plot_cluster(
                vanilla_coords,
                vanilla_domain_labels,
                results_dir / "UMAP_vanilla.pdf",
                sim_marker="s",
                exp_marker="^",
                sim_marker_size=600,
                exp_marker_size=600,
            )
            print(f"[evaluate] wrote {results_dir / 'UMAP_vanilla.pdf'}")
            write_embedding_diagnostics(
                coords=vanilla_coords,
                domain_labels=vanilla_domain_labels,
                samples=samples,
                results_dir=results_dir,
                model_suffix="vanilla",
            )

    write_results_csvs(
        samples=samples,
        adv_predictions=adv_predictions,
        vanilla_predictions=vanilla_predictions,
        adv_coords=adv_coords,
        adv_sim_count=adv_sim_count,
        vanilla_coords=vanilla_coords,
        vanilla_sim_count=vanilla_sim_count,
        results_dir=results_dir,
        skip_umap=args.skip_umap,
        crop_policy=eval_crop_policy,
        crop_aggregation=args.eval_crop_aggregation,
    )
    if vanilla_model is not None:
        print("[evaluate] writing vanilla simulation reconstruction diagnostics")
        write_sim_reconstruction_diagnostics(
            model=vanilla_model,
            simulation_dataset_dir=args.simulation_dataset_dir,
            results_dir=results_dir,
            device=device,
            num_samples=args.sim_reconstruction_samples,
            manifest_path=args.sim_reconstruction_manifest,
            no_t=args.no_t,
        )
    return results_dir


def main() -> None:
    """
    Run the pipeline according to the requested CLI steps.

    Depending on `--steps`, this may rebuild the processed experiment dataset,
    retrain the models, evaluate raw experiment files, or perform any subset of
    those actions.
    """
    args = parse_args()
    raw_files = filter_files(iter_raw_experiment_files(args.exp_data_dir), args.files)
    print(f"[setup] selected {len(raw_files)} raw experiment file(s)")

    if "preprocess" in args.steps:
        preprocess_experiment_dataset(
            raw_files=raw_files,
            output_dir=args.experiment_dataset_dir,
            crop_size=args.crop_size,
            coarse_size=args.coarse_size,
            crop_step=args.crop_step,
            clean_output_dir=not args.keep_existing_experiment_dir,
        )

    trained_vanilla = None
    trained_adv = None
    plot_results_dir = (
        args.results_dir / args.results_run_name
        if args.results_run_name is not None
        else args.results_dir
    )
    if "train" in args.steps:
        trained_vanilla, trained_adv = run_training(args)

    if "evaluate" in args.steps:
        if args.eval_models == "none":
            print("[evaluate] skipped because --eval-models none")
        else:
            vanilla_model, adv_model = ensure_eval_models(
                args=args,
                trained_vanilla=trained_vanilla,
                trained_adv=trained_adv,
            )
            plot_results_dir = evaluate_models(
                args=args,
                raw_files=raw_files,
                vanilla_model=vanilla_model,
                adv_model=adv_model,
            )
    
    if "plot-bars" in args.steps:
        generate_grouped_bar_plots(plot_results_dir)

    if "plot-global-scatter" in args.steps:
        generate_global_fixed_q_scatter_plots(plot_results_dir)

    if "phase-diagram" in args.steps:
        generate_phase_diagrams(args, plot_results_dir)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"[done] total runtime: {time.time() - start:.1f}s")
