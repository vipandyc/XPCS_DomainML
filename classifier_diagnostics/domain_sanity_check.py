import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset, TensorDataset
import umap.umap_ as umap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_adv import XPCSDataset, build_combined_domain_splits
from train_vanilla import load_model as load_vanilla_model
from train_adv import load_model as load_adv_model


def resolve_repo_relative_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_optional_float(value: str) -> Optional[float]:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check whether frozen encoder features separate simulation "
            "and experiment, using sklearn and simple Torch probes."
        )
    )
    parser.add_argument(
        "--model-type",
        choices=["vanilla", "adv"],
        default="vanilla",
        help="Which checkpoint family to load.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Defaults to the latest matching model.",
    )
    parser.add_argument(
        "--simulation-dataset-dir",
        type=Path,
        default=Path("dataset/simulation"),
        help="Directory containing the processed simulation manifest and tensors.",
    )
    parser.add_argument(
        "--experiment-dataset-dir",
        type=Path,
        default=Path("dataset/experiment"),
        help="Directory containing the processed experiment manifest and tensors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split/probe seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for feature extraction and Torch probes.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for feature extraction.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap per split for faster smoke tests.",
    )
    parser.add_argument(
        "--torch-probe-epochs",
        type=int,
        default=50,
        help="Number of epochs for Torch-based probes.",
    )
    parser.add_argument(
        "--torch-probe-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Torch-based probes.",
    )
    parser.add_argument(
        "--torch-probe-hidden-dim",
        type=int,
        default=64,
        help="Hidden width for the MLP probe. The linear probe always uses one layer.",
    )
    parser.add_argument(
        "--torch-probe-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit hidden widths for the MLP probe, e.g. "
            "`--torch-probe-hidden-dims 32` or `--torch-probe-hidden-dims 64 32`. "
            "Defaults to two layers of `--torch-probe-hidden-dim`."
        ),
    )
    parser.add_argument(
        "--torch-probe-activation",
        choices=["relu", "leaky-relu", "gelu"],
        default="relu",
        help="Hidden activation used by the MLP probe.",
    )
    parser.add_argument(
        "--torch-probe-leaky-relu-slope",
        type=float,
        default=0.01,
        help="Negative slope used when `--torch-probe-activation leaky-relu`.",
    )
    parser.add_argument(
        "--torch-probe-layer-norm",
        action="store_true",
        help="Apply LayerNorm before each hidden linear layer in the MLP probe.",
    )
    parser.add_argument(
        "--torch-probe-standardize",
        action="store_true",
        help="Standardize features using train-split mean/std before fitting any probe.",
    )
    parser.add_argument(
        "--torch-probe-class-weighting",
        choices=["balanced", "none"],
        default="balanced",
        help="Whether to use balanced class weights in the Torch probe loss.",
    )
    parser.add_argument(
        "--torch-probe-optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adamw",
        help="Optimizer used for Torch probes.",
    )
    parser.add_argument(
        "--torch-probe-weight-decay",
        type=parse_optional_float,
        default=0.01,
        help="Weight decay used by the Torch probe optimizer. Use `none` to disable it.",
    )
    parser.add_argument(
        "--torch-probe-momentum",
        type=float,
        default=0.9,
        help="Momentum used when `--torch-probe-optimizer sgd`.",
    )
    parser.add_argument(
        "--torch-probe-show-steps",
        action="store_true",
        help="Print per-epoch training progress for Torch probes.",
    )
    parser.add_argument(
        "--torch-probe-log-interval",
        type=int,
        default=1,
        help="Epoch interval for printing Torch probe progress when `--torch-probe-show-steps` is enabled.",
    )
    parser.add_argument(
        "--torch-probe-save-history",
        action="store_true",
        help="Include per-epoch Torch probe history in the optional JSON report.",
    )
    parser.add_argument(
        "--torch-probe-umap-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for per-epoch Torch-probe UMAP frames. "
            "If set, the script saves one fixed embedding plus epoch-by-epoch "
            "prediction/probability plots for each selected Torch probe."
        ),
    )
    parser.add_argument(
        "--torch-probe-umap-split",
        choices=["train", "val", "test"],
        default="val",
        help="Which split to visualize on the fixed UMAP embedding.",
    )
    parser.add_argument(
        "--torch-probe-umap-interval",
        type=int,
        default=1,
        help="Epoch interval for saving Torch-probe UMAP frames.",
    )
    parser.add_argument(
        "--torch-probe-umap-representative-points",
        type=int,
        default=8,
        help="How many fixed representative UMAP points to annotate in each frame.",
    )
    parser.add_argument(
        "--torch-probe-umap-gif-duration-ms",
        type=int,
        default=350,
        help="Frame duration in milliseconds for the saved Torch-probe GIF animation.",
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=["sklearn-linear", "torch-linear", "torch-mlp"],
        default=["sklearn-linear", "torch-linear", "torch-mlp"],
        help=(
            "Which probes to run. Defaults to all probes. Example: "
            "`--probes sklearn-linear` or `--probes torch-linear torch-mlp`."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file to save the full report.",
    )
    return parser.parse_args(argv)


def maybe_limit_indices(indices: np.ndarray, max_samples: Optional[int], seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(selected)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    features = []
    labels = []
    for x, _, _, _, batch_labels in loader:
        x = x.to(device)
        features.append(model.conv_net(x).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def compute_binary_classification_metrics(confusion: np.ndarray) -> dict[str, float]:
    total = float(confusion.sum())
    if total == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "recall_sim": float("nan"),
            "recall_exp": float("nan"),
            "predicted_exp_fraction": float("nan"),
        }
    recall_sim = float(confusion[0, 0]) / float(confusion[0].sum()) if confusion[0].sum() > 0 else float("nan")
    recall_exp = float(confusion[1, 1]) / float(confusion[1].sum()) if confusion[1].sum() > 0 else float("nan")
    return {
        "accuracy": float((confusion[0, 0] + confusion[1, 1]) / total),
        "balanced_accuracy": float(0.5 * (recall_sim + recall_exp)),
        "recall_sim": recall_sim,
        "recall_exp": recall_exp,
        "predicted_exp_fraction": float((confusion[0, 1] + confusion[1, 1]) / total),
    }


def evaluate_predictions(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        confusion[int(true_label), int(pred_label)] += 1
    return compute_binary_classification_metrics(confusion)


def resolve_model_path(model_type: str, model_path: Optional[Path]) -> Path:
    if model_path is not None:
        return resolve_repo_relative_path(model_path)
    model_dir = REPO_ROOT / "models"
    pattern = "Vanilla_XPCS_best_*.pt" if model_type == "vanilla" else "XPCS_best_*.pt"
    model_files = list(model_dir.glob(pattern))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir} matching {pattern}")
    model_files.sort(key=lambda path: path.stem.split("_")[-1], reverse=True)
    return model_files[0]


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(param.numel() for param in model.parameters())),
        "trainable": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }


def load_checkpoint_metadata(model_path: Path) -> tuple[Optional[Path], Optional[dict[str, object]]]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return None, None
    with open(metadata_path, "r", encoding="ascii") as handle:
        return metadata_path, json.load(handle)


def summarize_encoder_model(
    model: nn.Module,
    model_type: str,
    resolved_model_path: Path,
) -> dict[str, object]:
    parameter_counts = count_parameters(model)
    child_modules = {name: child.__class__.__name__ for name, child in model.named_children()}
    metadata_path, checkpoint_metadata = load_checkpoint_metadata(resolved_model_path)
    summary: dict[str, object] = {
        "model_type": model_type,
        "class_name": model.__class__.__name__,
        "checkpoint_path": str(resolved_model_path),
        "checkpoint_exists": resolved_model_path.exists(),
        "parameter_counts": parameter_counts,
        "child_modules": child_modules,
        "state_dict_num_tensors": len(model.state_dict()),
        "device": str(next(model.parameters()).device),
    }
    if hasattr(model, "grl_alpha"):
        summary["grl_alpha"] = float(model.grl_alpha)
    if hasattr(model, "pred_mode"):
        summary["pred_mode"] = bool(model.pred_mode)
    if hasattr(model, "class_mode"):
        summary["class_mode"] = bool(model.class_mode)
    if metadata_path is not None:
        summary["checkpoint_metadata_path"] = str(metadata_path)
        summary["checkpoint_metadata"] = checkpoint_metadata
    return summary


def summarize_probe_model(
    probe_type: str,
    model: Optional[nn.Module] = None,
    text_architecture: Optional[str] = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "probe_type": probe_type,
    }
    if model is not None:
        summary["class_name"] = model.__class__.__name__
        summary["parameter_counts"] = count_parameters(model)
        summary["module_repr"] = str(model)
        summary["architecture"] = str(model.classifier) if hasattr(model, "classifier") else str(model)
    elif text_architecture is not None:
        summary["architecture"] = text_architecture
    return summary


def build_activation(
    activation_name: str,
    leaky_relu_slope: float,
) -> nn.Module:
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leaky-relu":
        return nn.LeakyReLU(leaky_relu_slope)
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation_name}")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation_name: str = "relu",
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if use_layer_norm:
                layers.append(nn.LayerNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(build_activation(activation_name, leaky_relu_slope))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def build_feature_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_feature_splits(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    adjusted_std = np.where(std < 1e-8, 1.0, std)
    train_standardized = (train_features - mean) / adjusted_std
    val_standardized = (val_features - mean) / adjusted_std
    test_standardized = (test_features - mean) / adjusted_std
    summary = {
        "mean_abs_train_mean": float(np.mean(np.abs(train_standardized.mean(axis=0)))),
        "mean_train_std": float(np.mean(train_standardized.std(axis=0))),
        "num_near_constant_features": int(np.sum(std < 1e-8)),
    }
    return train_standardized, val_standardized, test_standardized, summary


def absolutize_dataset_manifest_paths(dataset: XPCSDataset) -> XPCSDataset:
    if "path" not in dataset.manifest.columns:
        return dataset
    dataset.manifest = dataset.manifest.copy()
    dataset.manifest["path"] = dataset.manifest["path"].map(
        lambda path: str(resolve_repo_relative_path(Path(path)))
    )
    return dataset


def build_class_weights(
    train_labels: np.ndarray,
    weighting: str,
) -> torch.Tensor:
    if weighting == "none":
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    class_counts = np.bincount(train_labels, minlength=2)
    return torch.tensor(
        [
            len(train_labels) / (2.0 * max(1, int(class_counts[0]))),
            len(train_labels) / (2.0 * max(1, int(class_counts[1]))),
        ],
        dtype=torch.float32,
    )


def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: Optional[float],
    momentum: float,
):
    optimizer_kwargs = {"lr": learning_rate}
    if weight_decay is not None:
        optimizer_kwargs["weight_decay"] = weight_decay
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = momentum
        return torch.optim.SGD(
            model.parameters(),
            **optimizer_kwargs,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def collect_loader_outputs(
    probe_model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, dict[str, object]]:
    total_loss = 0.0
    probs_exp = []
    true = []
    logits_all = []
    probe_model.eval()
    with torch.no_grad():
        for features_batch, labels_batch in loader:
            logits = probe_model(features_batch)
            loss = F.cross_entropy(logits, labels_batch, weight=class_weights)
            total_loss += loss.item() * features_batch.size(0)
            logits_all.append(logits.cpu().numpy())
            probs_exp.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            true.append(labels_batch.cpu().numpy())
    logits_np = np.concatenate(logits_all, axis=0)
    probs_exp_np = np.concatenate(probs_exp, axis=0)
    true_np = np.concatenate(true, axis=0)
    preds_np = (probs_exp_np >= 0.5).astype(np.int64)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(true_np, preds_np):
        confusion[int(true_label), int(pred_label)] += 1
    metrics = compute_binary_classification_metrics(confusion)
    mean_loss = total_loss / max(1, len(loader.dataset))
    summary = {
        "confusion_matrix": confusion.tolist(),
        "predicted_counts": {
            "simulation": int(np.sum(preds_np == 0)),
            "experiment": int(np.sum(preds_np == 1)),
        },
        "mean_prob_exp": float(np.mean(probs_exp_np)),
        "mean_prob_exp_by_true": {
            "simulation": float(np.mean(probs_exp_np[true_np == 0])) if np.any(true_np == 0) else float("nan"),
            "experiment": float(np.mean(probs_exp_np[true_np == 1])) if np.any(true_np == 1) else float("nan"),
        },
        "mean_logits": {
            "class_0": float(np.mean(logits_np[:, 0])),
            "class_1": float(np.mean(logits_np[:, 1])),
        },
        "mean_logits_by_true": {
            "simulation": {
                "class_0": float(np.mean(logits_np[true_np == 0, 0])) if np.any(true_np == 0) else float("nan"),
                "class_1": float(np.mean(logits_np[true_np == 0, 1])) if np.any(true_np == 0) else float("nan"),
            },
            "experiment": {
                "class_0": float(np.mean(logits_np[true_np == 1, 0])) if np.any(true_np == 1) else float("nan"),
                "class_1": float(np.mean(logits_np[true_np == 1, 1])) if np.any(true_np == 1) else float("nan"),
            },
        },
    }
    return mean_loss, metrics, preds_np, probs_exp_np, summary


def farthest_point_sampling(coords: np.ndarray, num_points: int) -> np.ndarray:
    if len(coords) == 0 or num_points <= 0:
        return np.array([], dtype=np.int64)
    num_points = min(num_points, len(coords))
    selected = [0]
    distances = np.full(len(coords), np.inf)
    for _ in range(1, num_points):
        last = coords[selected[-1]]
        distances = np.minimum(distances, np.linalg.norm(coords - last, axis=1))
        selected.append(int(np.argmax(distances)))
    return np.array(selected, dtype=np.int64)


def select_representative_points(
    coords: np.ndarray,
    labels: np.ndarray,
    num_points: int,
) -> np.ndarray:
    if num_points <= 0:
        return np.array([], dtype=np.int64)
    sim_idx = np.where(labels == 0)[0]
    exp_idx = np.where(labels == 1)[0]
    sim_quota = min(len(sim_idx), num_points // 2)
    exp_quota = min(len(exp_idx), num_points - sim_quota)
    if sim_quota + exp_quota < num_points:
        remaining = num_points - (sim_quota + exp_quota)
        extra_sim = min(len(sim_idx) - sim_quota, remaining)
        sim_quota += max(0, extra_sim)
        remaining -= max(0, extra_sim)
        exp_quota += min(len(exp_idx) - exp_quota, remaining)

    chosen = []
    if sim_quota > 0:
        sim_local = farthest_point_sampling(coords[sim_idx], sim_quota)
        chosen.extend(sim_idx[sim_local].tolist())
    if exp_quota > 0:
        exp_local = farthest_point_sampling(coords[exp_idx], exp_quota)
        chosen.extend(exp_idx[exp_local].tolist())
    if len(chosen) < min(num_points, len(coords)):
        remaining_idx = np.array(sorted(set(range(len(coords))) - set(chosen)), dtype=np.int64)
        extra_local = farthest_point_sampling(coords[remaining_idx], min(num_points - len(chosen), len(remaining_idx)))
        chosen.extend(remaining_idx[extra_local].tolist())
    return np.array(chosen[: min(num_points, len(coords))], dtype=np.int64)


def compute_umap_embedding(features: np.ndarray, seed: int) -> np.ndarray:
    if umap is None:
        raise ModuleNotFoundError(
            "UMAP visualization requires the 'umap-learn' package."
        )
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def save_true_domain_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sim_mask = labels == 0
    exp_mask = labels == 1
    ax.scatter(coords[sim_mask, 0], coords[sim_mask, 1], s=12, c="#7f7f7f", alpha=0.6, label="Simulation")
    ax.scatter(coords[exp_mask, 0], coords[exp_mask, 1], s=12, c="#e67e22", alpha=0.6, label="Experiment")
    ax.set_title("Frozen-feature UMAP colored by true domain")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_probability_boundary(
    axis: plt.Axes,
    coords: np.ndarray,
    pred_probs_exp: np.ndarray,
) -> None:
    """
    Draw an approximate projected decision boundary on the 2D UMAP.

    This is the 0.5 iso-contour of the predicted experiment probability
    interpolated over the 2D embedding, not the true decision surface in the
    original 128-D feature space.
    """
    if len(coords) < 3:
        return
    try:
        triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1])
        axis.tricontour(
            triangulation,
            pred_probs_exp,
            levels=[0.5],
            colors="black",
            linewidths=1.6,
            linestyles="--",
            alpha=0.9,
        )
    except Exception:
        return


def draw_probability_heatmap(
    axis: plt.Axes,
    coords: np.ndarray,
    pred_probs_exp: np.ndarray,
):
    """
    Draw a smooth-ish probability field over the fixed 2D UMAP.

    This is an interpolation in UMAP space, so it is only a visualization aid
    for how the current classifier behaves after projection.
    """
    if len(coords) < 3:
        return None
    try:
        triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1])
        return axis.tripcolor(
            triangulation,
            pred_probs_exp,
            shading="gouraud",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )
    except Exception:
        return None


def save_gif(
    frame_paths: list[Path],
    output_path: Path,
    duration_ms: int,
) -> None:
    if not frame_paths:
        return
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    first_image, remaining_images = images[0], images[1:]
    first_image.save(
        output_path,
        save_all=True,
        append_images=remaining_images,
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()


def save_probe_umap_frame(
    coords: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs_exp: np.ndarray,
    representative_indices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    true_fill_colors = np.where(true_labels == 1, "#f3a65a", "#7da7d9")
    pred_edge_colors = np.where(pred_labels == 1, "#b22222", "#13294b")
    marker_map = {0: "s", 1: "o"}

    for label_value, legend_label in [(0, "True sim"), (1, "True exp")]:
        mask = true_labels == label_value
        axes[0].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=30,
            c=true_fill_colors[mask],
            edgecolors=pred_edge_colors[mask],
            marker=marker_map[label_value],
            alpha=0.85,
            linewidths=1.0,
        )
    draw_probability_boundary(axes[0], coords, pred_probs_exp)
    axes[0].set_title("Fixed UMAP\nfill=true domain, edge=predicted class")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    left_legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#7da7d9", markeredgecolor="black", label="True sim"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f3a65a", markeredgecolor="black", label="True exp"),
        Line2D([0], [0], marker="o", color="#13294b", markerfacecolor="white", label="Pred sim edge"),
        Line2D([0], [0], marker="o", color="#b22222", markerfacecolor="white", label="Pred exp edge"),
        Line2D([0], [0], color="black", linestyle="--", label="Projected p=0.5 contour"),
    ]
    axes[0].legend(handles=left_legend, loc="best", fontsize=9)

    heatmap = draw_probability_heatmap(axes[1], coords, pred_probs_exp)
    for label_value in [0, 1]:
        mask = true_labels == label_value
        axes[1].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=32,
            facecolors="#f8d8b0" if label_value == 1 else "#d9e8fb",
            edgecolors="#8c4f13" if label_value == 1 else "#274c77",
            marker=marker_map[label_value],
            linewidths=0.9,
            alpha=0.9,
        )
    draw_probability_boundary(axes[1], coords, pred_probs_exp)
    axes[1].set_title("Predicted experiment probability\nbackground heatmap + true-domain markers")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    if heatmap is not None:
        fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04, label="p(experiment)")
    right_legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d9e8fb", markeredgecolor="#274c77", label="True sim"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f8d8b0", markeredgecolor="#8c4f13", label="True exp"),
        Line2D([0], [0], color="black", linestyle="--", label="Projected p=0.5 contour"),
    ]
    axes[1].legend(handles=right_legend, loc="best", fontsize=9)

    for rep_number, idx in enumerate(representative_indices, start=1):
        x_coord, y_coord = coords[idx]
        for axis in axes:
            axis.scatter(
                [x_coord],
                [y_coord],
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )
        axes[1].annotate(
            f"{rep_number}: {pred_probs_exp[idx]:.2f}",
            (x_coord, y_coord),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_torch_probe(
    probe_name: str,
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    optimizer_name: str = "adamw",
    weight_decay: Optional[float] = 0.01,
    momentum: float = 0.9,
    class_weighting: str = "balanced",
    show_steps: bool = False,
    log_interval: int = 1,
    save_history: bool = False,
    umap_features: Optional[np.ndarray] = None,
    umap_labels: Optional[np.ndarray] = None,
    umap_coords: Optional[np.ndarray] = None,
    umap_output_dir: Optional[Path] = None,
    umap_interval: int = 1,
    representative_indices: Optional[np.ndarray] = None,
    umap_gif_duration_ms: int = 350,
) -> dict[str, object]:
    seed_torch(seed)

    train_loader = build_feature_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = build_feature_loader(val_features, val_labels, batch_size=batch_size, shuffle=False)
    test_loader = build_feature_loader(test_features, test_labels, batch_size=batch_size, shuffle=False)

    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = build_class_weights(train_labels, class_weighting)

    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    best_state = None
    best_val_balanced = -float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    saved_frame_paths: list[Path] = []

    if (
        umap_output_dir is not None
        and umap_coords is not None
        and umap_labels is not None
        and umap_features is not None
        and representative_indices is not None
    ):
        umap_loader = build_feature_loader(
            umap_features,
            umap_labels,
            batch_size=batch_size,
            shuffle=False,
        )
        _, _, umap_pred_labels, umap_pred_probs, _ = collect_loader_outputs(model, umap_loader, class_weights)
        init_frame_path = umap_output_dir / f"{probe_name}_epoch_init.png"
        save_probe_umap_frame(
            coords=umap_coords,
            true_labels=umap_labels,
            pred_labels=umap_pred_labels,
            pred_probs_exp=umap_pred_probs,
            representative_indices=representative_indices,
            output_path=init_frame_path,
            title=f"{probe_name} init (before training)",
        )
        saved_frame_paths.append(init_frame_path)

    for epoch in range(epochs):
        model.train()
        for features_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            logits = model(features_batch)
            loss = F.cross_entropy(logits, labels_batch, weight=class_weights)
            loss.backward()
            optimizer.step()

        train_loss, train_metrics, _, _, train_summary = collect_loader_outputs(model, train_loader, class_weights)
        val_loss, val_metrics, _, _, val_summary = collect_loader_outputs(model, val_loader, class_weights)
        if save_history:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_recall_sim": train_metrics["recall_sim"],
                    "train_recall_exp": train_metrics["recall_exp"],
                    "train_predicted_exp_fraction": train_metrics["predicted_exp_fraction"],
                    "train_confusion_matrix": train_summary["confusion_matrix"],
                    "train_predicted_counts": train_summary["predicted_counts"],
                    "train_mean_prob_exp_by_true": train_summary["mean_prob_exp_by_true"],
                    "train_mean_logits_by_true": train_summary["mean_logits_by_true"],
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_recall_sim": val_metrics["recall_sim"],
                    "val_recall_exp": val_metrics["recall_exp"],
                    "val_predicted_exp_fraction": val_metrics["predicted_exp_fraction"],
                    "val_confusion_matrix": val_summary["confusion_matrix"],
                    "val_predicted_counts": val_summary["predicted_counts"],
                    "val_mean_prob_exp_by_true": val_summary["mean_prob_exp_by_true"],
                    "val_mean_logits_by_true": val_summary["mean_logits_by_true"],
                }
            )
        if (
            umap_output_dir is not None
            and umap_coords is not None
            and umap_labels is not None
            and umap_features is not None
            and representative_indices is not None
            and (epoch % max(1, umap_interval) == 0 or epoch == epochs - 1)
        ):
            umap_loader = build_feature_loader(
                umap_features,
                umap_labels,
                batch_size=batch_size,
                shuffle=False,
            )
            _, _, umap_pred_labels, umap_pred_probs, _ = collect_loader_outputs(model, umap_loader, class_weights)
            frame_path = umap_output_dir / f"{probe_name}_epoch_{epoch:03d}.png"
            save_probe_umap_frame(
                coords=umap_coords,
                true_labels=umap_labels,
                pred_labels=umap_pred_labels,
                pred_probs_exp=umap_pred_probs,
                representative_indices=representative_indices,
                output_path=frame_path,
                title=f"{probe_name} epoch {epoch+1}/{epochs}",
            )
            saved_frame_paths.append(frame_path)
        if show_steps and (epoch % max(1, log_interval) == 0 or epoch == epochs - 1):
            print(
                f"[{probe_name} epoch {epoch+1}/{epochs}] "
                f"train_loss={train_loss:.6f} "
                f"train_bal={train_metrics['balanced_accuracy']:.3f} "
                f"train_pred_exp={train_metrics['predicted_exp_fraction']:.3f} "
                f"val_loss={val_loss:.6f} "
                f"val_bal={val_metrics['balanced_accuracy']:.3f} "
                f"val_pred_exp={val_metrics['predicted_exp_fraction']:.3f} "
                f"val_p(exp|sim)={val_summary['mean_prob_exp_by_true']['simulation']:.3f} "
                f"val_p(exp|exp)={val_summary['mean_prob_exp_by_true']['experiment']:.3f}"
            )
        if val_metrics["balanced_accuracy"] > best_val_balanced:
            best_val_balanced = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    report: dict[str, object] = {
        "best_val_epoch": best_epoch,
        "class_counts_train": {
            "simulation": int(class_counts[0]),
            "experiment": int(class_counts[1]),
        },
        "probe_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "class_weighting": class_weighting,
            "seed": seed,
            "show_steps": show_steps,
            "log_interval": log_interval,
            "save_history": save_history,
            "umap_interval": umap_interval,
            "umap_gif_duration_ms": umap_gif_duration_ms,
        },
    }
    if save_history:
        report["history"] = history
    if saved_frame_paths:
        gif_path = umap_output_dir / f"{probe_name}.gif"
        save_gif(saved_frame_paths, gif_path, duration_ms=umap_gif_duration_ms)
        report["umap_gif"] = str(gif_path)
    model.eval()
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        _, metrics, _, _, summary = collect_loader_outputs(model, loader, class_weights)
        report[split_name] = {**metrics, **summary}
    return report


def load_encoder(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, str, Path]:
    resolved_model_path = resolve_model_path(args.model_type, args.model_path)
    if args.model_type == "vanilla":
        model = load_vanilla_model(resolved_model_path, device=device)
        model_label = str(resolved_model_path)
        return model, model_label, resolved_model_path
    model = load_adv_model(resolved_model_path, device=device)
    model_label = str(resolved_model_path)
    return model, model_label, resolved_model_path


def run_probe_analysis(args: argparse.Namespace) -> dict[str, object]:
    args = argparse.Namespace(**vars(args))
    args.model_path = resolve_repo_relative_path(args.model_path)
    args.simulation_dataset_dir = resolve_repo_relative_path(args.simulation_dataset_dir)
    args.experiment_dataset_dir = resolve_repo_relative_path(args.experiment_dataset_dir)
    args.torch_probe_umap_dir = resolve_repo_relative_path(args.torch_probe_umap_dir)
    args.output_json = resolve_repo_relative_path(args.output_json)

    device = torch.device(args.device)
    model, model_label, resolved_model_path = load_encoder(args, device)
    sim_dataset = absolutize_dataset_manifest_paths(XPCSDataset(args.simulation_dataset_dir))
    exp_dataset = absolutize_dataset_manifest_paths(XPCSDataset(args.experiment_dataset_dir))
    full_dataset = absolutize_dataset_manifest_paths(
        XPCSDataset([args.simulation_dataset_dir, args.experiment_dataset_dir])
    )

    _, _, _, train_idx, val_idx, test_idx = build_combined_domain_splits(
        len(sim_dataset),
        len(exp_dataset),
        seed=args.seed,
    )
    train_idx = maybe_limit_indices(train_idx, args.max_samples_per_split, args.seed)
    val_idx = maybe_limit_indices(val_idx, args.max_samples_per_split, args.seed + 1)
    test_idx = maybe_limit_indices(test_idx, args.max_samples_per_split, args.seed + 2)

    train_features, train_labels = extract_features(model, Subset(full_dataset, train_idx), device, args.batch_size)
    val_features, val_labels = extract_features(model, Subset(full_dataset, val_idx), device, args.batch_size)
    test_features, test_labels = extract_features(model, Subset(full_dataset, test_idx), device, args.batch_size)
    standardization_summary = None
    if args.torch_probe_standardize:
        train_features, val_features, test_features, standardization_summary = standardize_feature_splits(
            train_features,
            val_features,
            test_features,
        )

    split_feature_map = {
        "train": (train_features, train_labels),
        "val": (val_features, val_labels),
        "test": (test_features, test_labels),
    }
    umap_features = None
    umap_labels = None
    umap_coords = None
    representative_indices = None
    if args.torch_probe_umap_dir is not None:
        umap_features, umap_labels = split_feature_map[args.torch_probe_umap_split]
        umap_coords = compute_umap_embedding(umap_features, seed=args.seed)
        representative_indices = select_representative_points(
            umap_coords,
            umap_labels,
            num_points=args.torch_probe_umap_representative_points,
        )
        args.torch_probe_umap_dir.mkdir(parents=True, exist_ok=True)
        save_true_domain_umap(
            umap_coords,
            umap_labels,
            args.torch_probe_umap_dir / f"{args.torch_probe_umap_split}_true_domain.png",
        )

    report: dict[str, object] = {
        "model_type": args.model_type,
        "model_label": model_label,
        "seed": args.seed,
        "probes_run": args.probes,
        "feature_dim": int(train_features.shape[1]),
        "run_config": {
            "script": str(Path(__file__).resolve()),
            "argv": sys.argv,
            "model_type": args.model_type,
            "model_path": str(args.model_path) if args.model_path is not None else None,
            "resolved_model_label": model_label,
            "resolved_model_path": str(resolved_model_path),
            "device": str(device),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_samples_per_split": args.max_samples_per_split,
            "simulation_dataset_dir": str(args.simulation_dataset_dir),
            "experiment_dataset_dir": str(args.experiment_dataset_dir),
            "output_json": str(args.output_json) if args.output_json is not None else None,
        },
        "encoder_model_info": summarize_encoder_model(
            model=model,
            model_type=args.model_type,
            resolved_model_path=resolved_model_path,
        ),
        "split_config": {
            "split_seed": args.seed,
            "train_limit_seed": args.seed,
            "val_limit_seed": args.seed + 1,
            "test_limit_seed": args.seed + 2,
        },
        "split_sizes": {
            "train": int(train_features.shape[0]),
            "val": int(val_features.shape[0]),
            "test": int(test_features.shape[0]),
        },
        "torch_probe_config": {
            "standardize": args.torch_probe_standardize,
            "optimizer": args.torch_probe_optimizer,
            "learning_rate": args.torch_probe_learning_rate,
            "weight_decay": args.torch_probe_weight_decay,
            "class_weighting": args.torch_probe_class_weighting,
        },
    }
    if standardization_summary is not None:
        report["standardization_summary"] = standardization_summary

    probe_label_map = {
        "sklearn-linear": "sklearn_linear_probe",
        "torch-linear": "torch_linear_probe",
        "torch-mlp": "torch_mlp_probe",
    }

    if "sklearn-linear" in args.probes:
        sklearn_probe = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=args.seed,
        )
        sklearn_probe.fit(train_features, train_labels)
        sklearn_report: dict[str, object] = {
            "probe_config": {
                "probe_type": "sklearn_linear",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": args.seed,
            },
            "probe_model_info": summarize_probe_model(
                probe_type="sklearn_linear",
                text_architecture=(
                    "LogisticRegression(class_weight='balanced', "
                    f"max_iter=1000, random_state={args.seed})"
                ),
            ),
        }
        for split_name, features, labels in [
            ("train", train_features, train_labels),
            ("val", val_features, val_labels),
            ("test", test_features, test_labels),
        ]:
            preds = sklearn_probe.predict(features)
            sklearn_report[split_name] = evaluate_predictions(labels, preds)
        report["sklearn_linear_probe"] = sklearn_report

    if "torch-linear" in args.probes:
        seed_torch(args.seed)
        linear_probe_model = LinearProbe(train_features.shape[1])
        linear_probe_report = train_torch_probe(
            probe_name="torch_linear",
            model=linear_probe_model,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            test_features=test_features,
            test_labels=test_labels,
            batch_size=args.batch_size,
            learning_rate=args.torch_probe_learning_rate,
            epochs=args.torch_probe_epochs,
            seed=args.seed,
            optimizer_name=args.torch_probe_optimizer,
            weight_decay=args.torch_probe_weight_decay,
            momentum=args.torch_probe_momentum,
            class_weighting=args.torch_probe_class_weighting,
            show_steps=args.torch_probe_show_steps,
            log_interval=args.torch_probe_log_interval,
            save_history=args.torch_probe_save_history,
            umap_features=umap_features,
            umap_labels=umap_labels,
            umap_coords=umap_coords,
            umap_output_dir=args.torch_probe_umap_dir,
            umap_interval=args.torch_probe_umap_interval,
            representative_indices=representative_indices,
            umap_gif_duration_ms=args.torch_probe_umap_gif_duration_ms,
        )
        linear_probe_report["probe_model_info"] = summarize_probe_model(
            probe_type="torch_linear",
            model=linear_probe_model,
        )
        report["torch_linear_probe"] = linear_probe_report

    if "torch-mlp" in args.probes:
        seed_torch(args.seed)
        mlp_hidden_dims = (
            list(args.torch_probe_hidden_dims)
            if args.torch_probe_hidden_dims is not None
            else [args.torch_probe_hidden_dim, args.torch_probe_hidden_dim]
        )
        mlp_probe_model = MLPProbe(
            train_features.shape[1],
            hidden_dims=mlp_hidden_dims,
            activation_name=args.torch_probe_activation,
            leaky_relu_slope=args.torch_probe_leaky_relu_slope,
            use_layer_norm=args.torch_probe_layer_norm,
        )
        mlp_probe_report = train_torch_probe(
            probe_name="torch_mlp",
            model=mlp_probe_model,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            test_features=test_features,
            test_labels=test_labels,
            batch_size=args.batch_size,
            learning_rate=args.torch_probe_learning_rate,
            epochs=args.torch_probe_epochs,
            seed=args.seed,
            optimizer_name=args.torch_probe_optimizer,
            weight_decay=args.torch_probe_weight_decay,
            momentum=args.torch_probe_momentum,
            class_weighting=args.torch_probe_class_weighting,
            show_steps=args.torch_probe_show_steps,
            log_interval=args.torch_probe_log_interval,
            save_history=args.torch_probe_save_history,
            umap_features=umap_features,
            umap_labels=umap_labels,
            umap_coords=umap_coords,
            umap_output_dir=args.torch_probe_umap_dir,
            umap_interval=args.torch_probe_umap_interval,
            representative_indices=representative_indices,
            umap_gif_duration_ms=args.torch_probe_umap_gif_duration_ms,
        )
        mlp_probe_report["probe_model_info"] = summarize_probe_model(
            probe_type="torch_mlp",
            model=mlp_probe_model,
        )
        report["torch_mlp_probe"] = mlp_probe_report

    return report


def print_report_summary(report: dict[str, object]) -> None:
    probe_label_map = {
        "sklearn-linear": "sklearn_linear_probe",
        "torch-linear": "torch_linear_probe",
        "torch-mlp": "torch_mlp_probe",
    }

    print(f"Model: {report['model_type']} ({report['model_label']})")
    print(f"Frozen feature dimension: {report['feature_dim']}")
    split_sizes = report["split_sizes"]
    print(
        "Split sizes: "
        f"train={split_sizes['train']}, "
        f"val={split_sizes['val']}, "
        f"test={split_sizes['test']}"
    )
    for probe_name in report["probes_run"]:
        key = probe_label_map[probe_name]
        section = report[key]
        print(f"\n[{key}]")
        if isinstance(section, dict) and "best_val_epoch" in section:
            print(f"best_val_epoch: {section['best_val_epoch']}")
        for split_name in ["train", "val", "test"]:
            metrics = section[split_name]
            print(
                f"{split_name}: "
                f"acc={metrics['accuracy']:.3f}, "
                f"bal={metrics['balanced_accuracy']:.3f}, "
                f"recall_sim={metrics['recall_sim']:.3f}, "
                f"recall_exp={metrics['recall_exp']:.3f}, "
                f"pred_exp={metrics['predicted_exp_fraction']:.3f}"
            )

    output_json = report.get("run_config", {}).get("output_json")
    if output_json is not None:
        print(f"\nSaved report to {output_json}")


def save_report(report: dict[str, object], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="ascii") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    args = parse_args()
    report = run_probe_analysis(args)
    if args.output_json is not None:
        save_report(report, args.output_json)
    print_report_summary(report)


if __name__ == "__main__":
    main()
