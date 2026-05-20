"""
CORAL + Distillation domain adaptation for XPCS parameter extraction (no-T variant).

Replaces the DANN/GRL adversarial framework with:
  - CORAL loss: aligns second-order statistics (covariance) of sim vs exp features
  - Distillation loss: anchors features to a frozen vanilla model to prevent collapse

No discriminator, no gradient reversal, no minimax dynamics.
"""

import json
import math
import time
from typing import List, Tuple, Dict, Union, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, LeakyReLU, Dropout, Linear
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def close(self):
            pass
from sklearn.linear_model import LogisticRegression

from produce_data import coarse_grain_g2
from train_vanilla_no_T import (
    compute_prediction_loss,
    save_training_metadata,
    seed_dataloader_worker,
    set_global_seed,
)

# --- Split configuration ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
INPUT_SIZE = 256
INPUT_MEAN = 1.1315594972968084
INPUT_STD = 0.011241361147397289
CONTRASTIVE_LOSS_CHOICES = ("margin", "soft-infonce")

# ---------------------------------------------------------------------------
# Normalization helpers (mirrored from train_adv_no_T for self-containment)
# ---------------------------------------------------------------------------


def norm_from_meta(
    y_raw: torch.Tensor,
    norm_meta: Dict[str, Dict[str, Union[float, str]]],
    device: torch.device = None,
    eps=1e-300,
) -> torch.Tensor:
    """Normalize raw parameters (gamma, D, GB_conc) to [0, 1]."""
    if device is None:
        device = y_raw.device
    param_order = ["gamma", "D", "GB_conc"]
    y_norm = torch.zeros_like(y_raw, device=device)
    for i, name in enumerate(param_order):
        spec = norm_meta[name]
        low, high, scale = spec["low"], spec["high"], spec["scale"]
        if scale == "log":
            val = torch.log10(y_raw[..., i].clamp(min=eps))
            low_log = math.log10(max(low, eps))
            high_log = math.log10(max(high, eps))
            y_norm[..., i] = (val - low_log) / (high_log - low_log + eps)
        else:
            y_norm[..., i] = (y_raw[..., i] - low) / (high - low + eps)
    return y_norm.clamp(0.0, 1.0)


def denorm_from_meta(
    y_norm: torch.Tensor,
    norm_meta: Dict[str, Dict[str, Union[float, str]]],
    device: torch.device = None,
    eps=1e-300,
) -> torch.Tensor:
    """Denormalize [0, 1] parameters back to physical units."""
    if device is None:
        device = y_norm.device
    param_order = ["gamma", "D", "GB_conc"]
    y_raw = torch.zeros_like(y_norm, device=device)
    for i, name in enumerate(param_order):
        spec = norm_meta[name]
        low, high, scale = spec["low"], spec["high"], spec["scale"]
        if scale == "log":
            low_log = math.log10(max(low, eps))
            high_log = math.log10(max(high, eps))
            val_log = y_norm[..., i] * (high_log - low_log) + low_log
            y_raw[..., i] = 10.0 ** val_log
        else:
            y_raw[..., i] = y_norm[..., i] * (high - low) + low
    return y_raw


# ---------------------------------------------------------------------------
# Data splitting helpers (mirrored from train_adv_no_T)
# ---------------------------------------------------------------------------


def create_random_splits(
    dataset_size: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    return (
        indices[:train_size],
        indices[train_size : train_size + val_size],
        indices[train_size + val_size :],
    )


def build_combined_domain_splits(
    sim_dataset_size: int,
    exp_dataset_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_sim, val_sim, test_sim = create_random_splits(sim_dataset_size, seed=seed)
    train_exp, val_exp, test_exp = create_random_splits(exp_dataset_size, seed=seed + 1)
    exp_offset = sim_dataset_size
    return (
        train_sim,
        val_sim,
        test_sim,
        np.concatenate([train_sim, exp_offset + train_exp]),
        np.concatenate([val_sim, exp_offset + val_exp]),
        np.concatenate([test_sim, exp_offset + test_exp]),
    )


# ---------------------------------------------------------------------------
# Domain probe helpers (mirrored from train_adv_no_T)
# ---------------------------------------------------------------------------


def compute_binary_classification_metrics(confusion: np.ndarray) -> dict[str, float]:
    if confusion.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got {confusion.shape}")
    total = float(confusion.sum())
    if total == 0:
        return {k: float("nan") for k in [
            "accuracy", "balanced_accuracy", "recall_sim", "recall_exp",
            "predicted_exp_fraction",
        ]}
    recall_sim = (
        float(confusion[0, 0]) / float(confusion[0].sum())
        if confusion[0].sum() > 0
        else float("nan")
    )
    recall_exp = (
        float(confusion[1, 1]) / float(confusion[1].sum())
        if confusion[1].sum() > 0
        else float("nan")
    )
    return {
        "accuracy": float((confusion[0, 0] + confusion[1, 1]) / total),
        "balanced_accuracy": float(0.5 * (recall_sim + recall_exp)),
        "recall_sim": recall_sim,
        "recall_exp": recall_exp,
        "predicted_exp_fraction": float((confusion[0, 1] + confusion[1, 1]) / total),
    }


@torch.no_grad()
def extract_domain_probe_features(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    features_list = []
    labels_list = []
    for x, _, _, T, batch_labels, _ in loader:
        x, T = x.to(device), T.to(device)
        xpcs_features, temp_features = model.extract_features(x, T)
        shared_features = model.build_shared_features(xpcs_features, temp_features)
        features_list.append(shared_features.cpu().numpy())
        labels_list.append(batch_labels.cpu().numpy())
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def run_domain_probe(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    device: torch.device,
    seed: int,
) -> Optional[dict[str, float]]:
    if LogisticRegression is None:
        return None
    train_features, train_labels = extract_domain_probe_features(model, train_dataset, device=device)
    val_features, val_labels = extract_domain_probe_features(model, val_dataset, device=device)
    test_features, test_labels = extract_domain_probe_features(model, test_dataset, device=device)
    probe = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    probe.fit(train_features, train_labels)
    results: dict[str, float] = {}
    for split_name, features, labels in [
        ("train", train_features, train_labels),
        ("val", val_features, val_labels),
        ("test", test_features, test_labels),
    ]:
        preds = probe.predict(features)
        confusion = np.zeros((2, 2), dtype=np.int64)
        for true_label, pred_label in zip(labels, preds):
            confusion[int(true_label), int(pred_label)] += 1
        metrics = compute_binary_classification_metrics(confusion)
        for k, v in metrics.items():
            results[f"{split_name}_{k}"] = v
    return results


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------


def next_loader_batch(
    loader: DataLoader,
    iterator: Optional[Any],
) -> tuple[tuple[torch.Tensor, ...], Any]:
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


# ---------------------------------------------------------------------------
# State-dict loading helper
# ---------------------------------------------------------------------------


def load_matching_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(filtered_state))
    model.load_state_dict(filtered_state, strict=False)
    return {
        "loaded": sorted(filtered_state.keys()),
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CORAL loss
# ---------------------------------------------------------------------------


def coral_loss(feat_sim: torch.Tensor, feat_exp: torch.Tensor) -> torch.Tensor:
    """
    Compute a distribution alignment loss combining:

    1. **Mean alignment**: Per-dimension mean gap, using both the average and
       the max across dimensions.  The max term prevents the model from
       keeping a few "discriminator-friendly" dimensions with large gaps while
       averaging out the penalty.
    2. **Covariance alignment (CORAL)**: Squared Frobenius norm of the
       difference between covariance matrices, computed after joint
       standardization.

    All statistics are computed in pooled-standardized space so every feature
    dimension contributes equally regardless of raw scale.

    Args:
        feat_sim: Simulation features of shape [N_sim, D].
        feat_exp: Experiment features of shape [N_exp, D].

    Returns:
        loss: Scalar alignment loss.
    """
    d = feat_sim.shape[1]
    n_sim = feat_sim.shape[0]
    n_exp = feat_exp.shape[0]
    if n_sim < 2 or n_exp < 2:
        return torch.tensor(0.0, device=feat_sim.device)

    eps = 1e-6

    # --- Pooled statistics (joint across both domains) ---
    all_feats = torch.cat([feat_sim, feat_exp], dim=0)
    pooled_mean = all_feats.mean(0)
    pooled_std = all_feats.std(0).clamp(min=eps)

    # Standardize both domains with the SAME reference
    sim_normed = (feat_sim - pooled_mean) / pooled_std
    exp_normed = (feat_exp - pooled_mean) / pooled_std

    # --- Mean alignment (first-order) ---
    per_dim_gap = (sim_normed.mean(0) - exp_normed.mean(0)).pow(2)
    mean_loss = per_dim_gap.mean() + per_dim_gap.max()

    # --- Covariance alignment (second-order, CORAL) ---
    cs = sim_normed.T @ sim_normed / (n_sim - 1)
    ct = exp_normed.T @ exp_normed / (n_exp - 1)
    cov_loss = (cs - ct).pow(2).sum() / (4 * d * d)

    return mean_loss + cov_loss


def continuous_noneq_contrastive_loss(
    features: torch.Tensor,
    noneq_values: torch.Tensor,
    measure_bandwidth: float = 0.1,
    feature_margin: float = 1.0,
) -> torch.Tensor:
    """
    Encourage the latent geometry to reflect the continuous nonequilibrium
    labels shared by simulation and experiment samples.

    Pairs with similar nonequilibrium measures are pulled together, while
    pairs with distant measures are pushed apart up to ``feature_margin`` in
    the L2 distance of unit-normalized feature space.
    """
    num_samples = features.shape[0]
    if num_samples < 2:
        return features.new_zeros(())

    bandwidth = max(float(measure_bandwidth), 1e-6)
    margin = max(float(feature_margin), 0.0)

    normalized_features = F.normalize(features, p=2, dim=1)
    feature_distances = torch.cdist(normalized_features, normalized_features, p=2)
    noneq_distances = torch.cdist(
        noneq_values.reshape(-1, 1),
        noneq_values.reshape(-1, 1),
        p=1,
    )

    positive_weights = torch.exp(-(noneq_distances / bandwidth).pow(2))
    negative_weights = 1.0 - positive_weights
    pair_mask = torch.triu(
        torch.ones_like(feature_distances, dtype=torch.bool),
        diagonal=1,
    )
    positive_weights = positive_weights[pair_mask]
    negative_weights = negative_weights[pair_mask]
    feature_distances = feature_distances[pair_mask]

    positive_loss = (
        positive_weights * feature_distances.pow(2)
    ).sum() / positive_weights.sum().clamp(min=1e-6)
    negative_loss = (
        negative_weights * F.relu(margin - feature_distances).pow(2)
    ).sum() / negative_weights.sum().clamp(min=1e-6)
    return positive_loss + negative_loss


def soft_supervised_noneq_infonce_loss(
    features: torch.Tensor,
    noneq_values: torch.Tensor,
    measure_bandwidth: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Soft-label supervised InfoNCE for continuous nonequilibrium labels.

    For each anchor, the model distribution is the softmax over cosine
    similarities to all other samples in the batch. The target distribution is
    a Gaussian kernel over nonequilibrium-measure distances, so samples with
    similar noneq values receive more probability mass without requiring hard
    positive/negative thresholds.
    """
    num_samples = features.shape[0]
    if num_samples < 2:
        return features.new_zeros(())

    bandwidth = max(float(measure_bandwidth), 1e-6)
    tau = max(float(temperature), 1e-6)

    normalized_features = F.normalize(features, p=2, dim=1)
    logits = normalized_features @ normalized_features.T
    logits = logits / tau

    noneq_distances = torch.cdist(
        noneq_values.reshape(-1, 1),
        noneq_values.reshape(-1, 1),
        p=1,
    )
    target_weights = torch.exp(-(noneq_distances / bandwidth).pow(2))

    self_mask = torch.eye(num_samples, dtype=torch.bool, device=features.device)
    off_diag_mask = ~self_mask
    logits = logits.masked_fill(self_mask, float("-inf"))
    target_weights = target_weights.masked_fill(self_mask, 0.0)
    target_weights = target_weights + off_diag_mask.to(target_weights.dtype) * 1e-12
    target_probs = target_weights / target_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)

    log_probs = F.log_softmax(logits, dim=1).masked_fill(self_mask, 0.0)
    return -(target_probs * log_probs).sum(dim=1).mean()


def compute_noneq_contrastive_loss(
    features: torch.Tensor,
    noneq_values: torch.Tensor,
    loss_type: str = "margin",
    measure_bandwidth: float = 0.1,
    feature_margin: float = 1.0,
    infonce_temperature: float = 0.1,
) -> torch.Tensor:
    if loss_type == "margin":
        return continuous_noneq_contrastive_loss(
            features,
            noneq_values,
            measure_bandwidth=measure_bandwidth,
            feature_margin=feature_margin,
        )
    if loss_type == "soft-infonce":
        return soft_supervised_noneq_infonce_loss(
            features,
            noneq_values,
            measure_bandwidth=measure_bandwidth,
            temperature=infonce_temperature,
        )
    raise ValueError(
        f"Unsupported contrastive loss {loss_type!r}; "
        f"expected one of {CONTRASTIVE_LOSS_CHOICES}"
    )


def resolve_nonequilibrium_column(manifest: pd.DataFrame) -> str | None:
    if "nonequilibrium_measure" in manifest.columns:
        return "nonequilibrium_measure"
    if "unequilibrium_measure" in manifest.columns:
        return "unequilibrium_measure"
    return None


def avg_diagonal(g2: torch.Tensor) -> torch.Tensor:
    if g2.dim() == 3 and g2.shape[0] == 1:
        g2 = g2.squeeze(0)
    n = g2.shape[0]
    offsets = torch.arange(n).reshape(1, n) - torch.arange(n).reshape(n, 1)
    diagonal_means = torch.zeros(2 * n - 1, dtype=g2.dtype)
    for offset in range(-n + 1, n):
        diagonal_means[offset + n - 1] = g2[offsets == offset].mean()
    return diagonal_means[offsets + n - 1]


def compute_nonequilibrium_measure(g2: torch.Tensor) -> float:
    if g2.dim() == 3 and g2.shape[0] == 1:
        g2 = g2.squeeze(0)
    g2 = g2.to(torch.float32)
    diagonal_average = avg_diagonal(g2)
    numerator = torch.linalg.vector_norm(g2 - diagonal_average)
    denominator = torch.linalg.vector_norm(g2 - g2.mean())
    if denominator <= 0:
        return 0.0
    return float((numerator / denominator).item())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class XPCSDataset(Dataset):
    """
    XPCS g2 correlation dataset for both simulation and experiment domains.
    Identical to the one in ``train_adv_no_T.py`` — duplicated here so the
    module stays self-contained.
    """
    PARAM_KEYS = ["gamma", "D", "GB_conc"]

    def __init__(self, paths: Union[Path, List[Path]]):
        self.paths = paths if isinstance(paths, list) else [paths]
        self.manifest = pd.concat(
            [pd.read_csv(Path(p) / "manifest.csv") for p in self.paths],
            ignore_index=True,
        )
        self.noneq_column = resolve_nonequilibrium_column(self.manifest)
        self.noneq_cache: list[Optional[float]] = [None] * len(self.manifest)
        self.norm_meta = {
            "gamma": {"low": 2e18, "high": 5e18, "scale": "linear"},
            "D": {"low": 1e-23, "high": 1e-21, "scale": "log"},
            "GB_conc": {"low": 0.0, "high": 0.3, "scale": "linear"},
            "T": {"low": 300, "high": 500, "scale": "linear"},
        }
        self.diag_mask = torch.ones(1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        self.diag_mask[0, range(INPUT_SIZE), range(INPUT_SIZE)] = 0.0

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        x_raw = torch.load(row["path"], weights_only=True).to(torch.float32).squeeze(0)
        x = x_raw
        x = (x - INPUT_MEAN) / (INPUT_STD + 1e-6)
        x = x * self.diag_mask
        y_raw = torch.tensor([row[k] for k in self.PARAM_KEYS], dtype=torch.float32)
        y_norm = norm_from_meta(y_raw, self.norm_meta, device=y_raw.device)
        T = torch.tensor([row["T"]], dtype=torch.float32)
        domain = row["domain"] if "domain" in row.index else (
            "simulation" if "simulation" in str(row["path"]) else "experiment"
        )
        label = 0 if domain == "simulation" else 1
        if self.noneq_column is not None and pd.notna(row[self.noneq_column]):
            noneq_value = float(row[self.noneq_column])
        else:
            cached_value = self.noneq_cache[idx]
            if cached_value is None:
                cached_value = compute_nonequilibrium_measure(x_raw)
                self.noneq_cache[idx] = cached_value
            noneq_value = cached_value
        noneq_tensor = torch.tensor([noneq_value], dtype=torch.float32)
        return x, y_norm, y_raw, T, label, noneq_tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class XPCSNetCoral(nn.Module):
    """
    CNN encoder + predictor head for XPCS parameter extraction.

    Same architecture as the DANN ``XPCSNet`` in ``train_adv_no_T.py`` minus
    the domain classifier, feature standardizer, and gradient reversal layer.
    Compatibility stubs (``on_pred_mode``, ``off_class_mode``, ``set_grl_alpha``)
    are included so ``run_all.py:predict_samples()`` works unchanged.
    """

    def __init__(
        self,
        predictor_output_activation: str = "sigmoid",
    ):
        super().__init__()
        if predictor_output_activation not in {"linear", "sigmoid"}:
            raise ValueError("predictor_output_activation must be 'linear' or 'sigmoid'")
        self.predictor_output_activation = predictor_output_activation
        self.norm_meta = {
            "gamma": {"low": 2e18, "high": 5e18, "scale": "linear"},
            "D": {"low": 1e-23, "high": 1e-21, "scale": "log"},
            "GB_conc": {"low": 0.0, "high": 0.3, "scale": "linear"},
            "T": {"low": 300, "high": 500, "scale": "linear"},
        }
        self.xpcs_feature_dim = 128
        self.shared_feature_dim = self.xpcs_feature_dim

        self.conv_net = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=2),
            MaxPool2d(kernel_size=3, stride=2),
            LeakyReLU(0.01),
            Conv2d(32, 64, kernel_size=3, padding=2),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU(),
            Conv2d(64, 128, kernel_size=3, padding=2),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=2),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.dropout = Dropout(0.0)
        self.xpcs_predictor = Sequential(
            Linear(self.shared_feature_dim, 64),
            ReLU(),
            self.dropout,
            nn.Linear(64, 64),
            ReLU(),
            self.dropout,
            nn.Linear(64, 3),
        )

    # -- Interface expected by run_all.py:predict_samples() --

    def get_architecture_config(self) -> Dict[str, Any]:
        return {"predictor_output_activation": self.predictor_output_activation}

    def extract_features(
        self, x: torch.Tensor, T: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        del T
        return self.conv_net(x), None

    def build_shared_features(
        self, xpcs_features: torch.Tensor, temp_features: None,
    ) -> torch.Tensor:
        del temp_features
        return xpcs_features

    def forward_predictor_from_shared_features(
        self, shared_features: torch.Tensor, return_logits: bool = False,
    ) -> torch.Tensor:
        pred_logits = self.xpcs_predictor(shared_features)
        if return_logits:
            return pred_logits
        if self.predictor_output_activation == "sigmoid":
            return torch.sigmoid(pred_logits)
        return pred_logits

    def forward(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        xpcs_features, temp_features = self.extract_features(x, T)
        shared_features = self.build_shared_features(xpcs_features, temp_features)
        return self.forward_predictor_from_shared_features(shared_features)

    # -- Compatibility stubs for the eval pipeline --

    def on_pred_mode(self):
        return self

    def off_class_mode(self):
        return self

    def set_grl_alpha(self, alpha: float):
        return self


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    model: XPCSNetCoral,
    sim_root: Path,
    exp_root: Path,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    coral_weight: float = 1.0,
    contrastive_weight: float = 0.0,
    contrastive_loss_type: str = "margin",
    contrastive_bandwidth: float = 0.1,
    contrastive_feature_margin: float = 1.0,
    contrastive_infonce_temperature: float = 0.1,
    seed: int = RANDOM_SEED,
    deterministic: bool = True,
    num_workers: int = 0,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    log_pardir: Path = Path("runs"),
    model_path: Path = Path("models"),
) -> XPCSNetCoral:
    """
    Train ``XPCSNetCoral`` with CORAL alignment.

    Args:
        model: Freshly instantiated model.
        sim_root: Simulation dataset directory.
        exp_root: Experiment dataset directory.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        learning_rate: Optimizer learning rate.
        coral_weight: Multiplier on the CORAL alignment loss.
        contrastive_weight: Multiplier on the nonequilibrium-aware contrastive loss.
        contrastive_loss_type: Contrastive objective, either margin or soft-infonce.
        contrastive_bandwidth: Soft positive-pair bandwidth in normalized noneq space.
        contrastive_feature_margin: Minimum target distance for dissimilar pairs.
        contrastive_infonce_temperature: Softmax temperature for soft InfoNCE.
        seed: Global random seed.
        deterministic: Whether to enforce deterministic behaviour.
        num_workers: DataLoader workers.
        init_state_dict: Optional checkpoint for warm-start initialization.
        device: Torch device.
        log_pardir: Parent directory for TensorBoard logs.
        model_path: Directory for saved checkpoints.

    Returns:
        best_model: Best model selected by validation prediction loss.
    """
    set_global_seed(seed, deterministic=deterministic)
    if contrastive_loss_type not in CONTRASTIVE_LOSS_CHOICES:
        raise ValueError(
            f"Unsupported contrastive_loss_type {contrastive_loss_type!r}; "
            f"expected one of {CONTRASTIVE_LOSS_CHOICES}"
        )

    # --- Logging ---
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = log_pardir / f"xpcs_coral_no_T_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[logger] TensorBoard log dir: {log_dir}")
    print(f"[seed] CORAL training seed: {seed} (deterministic={deterministic})")
    print(
        "[contrastive] "
        f"loss={contrastive_loss_type}, bandwidth={contrastive_bandwidth}, "
        f"margin={contrastive_feature_margin}, "
        f"infonce_temperature={contrastive_infonce_temperature}"
    )

    # --- Datasets and splits ---
    sim_dataset = XPCSDataset(sim_root)
    exp_dataset = XPCSDataset(exp_root)
    full_dataset = XPCSDataset([sim_root, exp_root])
    norm_meta = sim_dataset.norm_meta

    (
        train_indices_sim,
        val_indices_sim,
        test_indices_sim,
        train_indices_full,
        val_indices_full,
        test_indices_full,
    ) = build_combined_domain_splits(
        sim_dataset_size=len(sim_dataset),
        exp_dataset_size=len(exp_dataset),
        seed=seed,
    )
    train_set_sim = Subset(sim_dataset, train_indices_sim)
    train_set_full = Subset(full_dataset, train_indices_full)
    val_set_sim = Subset(sim_dataset, val_indices_sim)
    val_set_full = Subset(full_dataset, val_indices_full)
    test_set_sim = Subset(sim_dataset, test_indices_sim)
    test_set_full = Subset(full_dataset, test_indices_full)

    train_loader_sim = DataLoader(
        train_set_sim,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        drop_last=True,
    )
    train_loader_exp = DataLoader(
        Subset(exp_dataset, train_indices_full[len(train_indices_sim):] - len(sim_dataset)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        drop_last=True,
    )
    val_loader_sim = DataLoader(
        val_set_sim, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    # --- Model initialization ---
    model = model.to(device)
    if init_state_dict is not None:
        summary = load_matching_state_dict(model, init_state_dict)
        print(
            f"[init] Loaded matching initialization weights: "
            f"{len(summary['loaded'])} tensors, skipped {len(summary['skipped'])}"
        )

    # --- Metadata ---
    training_metadata: Dict[str, Any] = {
        "timestamp": stamp,
        "method": "coral",
        "seed": seed,
        "deterministic": deterministic,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "coral_weight": coral_weight,
        "contrastive_weight": contrastive_weight,
        "contrastive_loss_type": contrastive_loss_type,
        "contrastive_bandwidth": contrastive_bandwidth,
        "contrastive_feature_margin": contrastive_feature_margin,
        "contrastive_infonce_temperature": contrastive_infonce_temperature,
        "architecture": model.get_architecture_config(),
        "initialized_from_matching_state": init_state_dict is not None,
        "input_normalization": "global_zscore",
        "input_mean": INPUT_MEAN,
        "input_std": INPUT_STD,
        "simulation_dataset_root": str(sim_root),
        "experiment_dataset_root": str(exp_root),
        "simulation_dataset_size": len(sim_dataset),
        "experiment_dataset_size": len(exp_dataset),
        "full_dataset_size": len(full_dataset),
        "train_sim_count": len(train_indices_sim),
        "train_exp_count": len(train_indices_full) - len(train_indices_sim),
        "num_workers": num_workers,
    }
    metadata_path = log_dir / "training_metadata.json"
    checkpoint_metadata_path = model_path / f"XPCS_coral_no_T_best_{stamp}.json"
    save_training_metadata(metadata_path, training_metadata)
    writer.add_text(
        "config",
        json.dumps(training_metadata, indent=2, default=str),
        global_step=0,
    )

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    patience = 20
    bad_epochs = 0

    # --- Training loop ---
    exp_iterator = None
    for epoch in range(epochs):
        model.train()
        train_pred_sum = 0.0
        train_coral_sum = 0.0
        train_contrastive_sum = 0.0
        train_total_sum = 0.0
        train_steps = 0

        for sim_batch in train_loader_sim:
            x_sim, y_norm, _, T_sim, _, noneq_sim = sim_batch
            x_sim = x_sim.to(device)
            y_norm = y_norm.to(device)
            T_sim = T_sim.to(device)
            noneq_sim = noneq_sim.to(device)

            # Fetch an experiment batch (cycling)
            exp_batch, exp_iterator = next_loader_batch(train_loader_exp, exp_iterator)
            x_exp, _, _, T_exp, _, noneq_exp = exp_batch
            x_exp = x_exp.to(device)
            T_exp = T_exp.to(device)
            noneq_exp = noneq_exp.to(device)

            optimizer.zero_grad()

            # --- Sim features ---
            sim_feats, _ = model.extract_features(x_sim, T_sim)
            sim_shared = model.build_shared_features(sim_feats, None)

            # --- Prediction loss (sim only) ---
            pred_logits = model.forward_predictor_from_shared_features(
                sim_shared, return_logits=True,
            )
            pred_loss, _, _ = compute_prediction_loss(
                pred_logits, y_norm, model.predictor_output_activation,
            )

            # --- Exp features ---
            exp_feats, _ = model.extract_features(x_exp, T_exp)
            exp_shared = model.build_shared_features(exp_feats, None)

            # --- CORAL alignment loss ---
            c_loss = coral_loss(sim_shared, exp_shared)

            # --- Nonequilibrium-aware contrastive loss ---
            contrastive_loss = compute_noneq_contrastive_loss(
                torch.cat([sim_shared, exp_shared], dim=0),
                torch.cat([noneq_sim, noneq_exp], dim=0).squeeze(1),
                loss_type=contrastive_loss_type,
                measure_bandwidth=contrastive_bandwidth,
                feature_margin=contrastive_feature_margin,
                infonce_temperature=contrastive_infonce_temperature,
            )

            # --- Combined loss ---
            weighted_coral_loss = coral_weight * c_loss
            weighted_contrastive_loss = contrastive_weight * contrastive_loss
            total_loss = (
                pred_loss
                + weighted_coral_loss
                + weighted_contrastive_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_pred_sum += pred_loss.item()
            train_coral_sum += c_loss.item()
            train_contrastive_sum += contrastive_loss.item()
            train_total_sum += total_loss.item()
            train_steps += 1

        # --- Epoch averages ---
        avg_pred = train_pred_sum / max(1, train_steps)
        avg_coral = train_coral_sum / max(1, train_steps)
        avg_contrastive = train_contrastive_sum / max(1, train_steps)
        avg_total = train_total_sum / max(1, train_steps)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_total:.6f} "
            f"(Pred: {avg_pred:.6f}, CORAL: {avg_coral:.6f}, "
            f"Contrastive: {avg_contrastive:.6f})"
        )
        writer.add_scalar("train/total_loss", avg_total, epoch)
        writer.add_scalar("train/pred_loss", avg_pred, epoch)
        writer.add_scalar("train/coral_loss", avg_coral, epoch)
        writer.add_scalar("train/contrastive_loss", avg_contrastive, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # --- Validation ---
        model.eval()
        val_loss_pred = 0.0
        val_mae = torch.zeros(3, device=device)
        with torch.no_grad():
            for x, y_norm_v, y_raw_v, T, _, _ in val_loader_sim:
                x, y_norm_v, y_raw_v, T = (
                    x.to(device), y_norm_v.to(device), y_raw_v.to(device), T.to(device),
                )
                feats, _ = model.extract_features(x, T)
                shared = model.build_shared_features(feats, None)
                pred_logits = model.forward_predictor_from_shared_features(
                    shared, return_logits=True,
                )
                pred_loss_v, _, pred_params = compute_prediction_loss(
                    pred_logits, y_norm_v, model.predictor_output_activation,
                )
                pred_params_raw = denorm_from_meta(pred_params, norm_meta, device=device)
                val_mae += (pred_params_raw - y_raw_v).abs().sum(dim=0)
                val_loss_pred += pred_loss_v.item() * x.size(0)
        val_loss_pred /= len(val_set_sim)
        val_mae /= len(val_set_sim)
        print(
            f"Val Pred Loss: {val_loss_pred:.6f} "
            f"Per-parameter MAE: "
            f"gamma: {val_mae[0]:.4e}, D: {val_mae[1]:.4e}, GB_conc: {val_mae[2]:.4e}"
        )
        writer.add_scalar("val/pred_loss", val_loss_pred, epoch)
        for i, name in enumerate(["gamma", "D", "GB_conc"]):
            writer.add_scalar(f"mae_raw/{name}", float(val_mae[i]), epoch)

        # --- Checkpoint best ---
        if val_loss_pred < best_val_loss - 1e-6:
            best_val_loss = val_loss_pred
            bad_epochs = 0
            model_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                model_path / f"XPCS_coral_no_T_best_{stamp}.pt",
            )
            print(
                f"Saved best checkpoint (val {best_val_loss:.4f}) "
                f"-> XPCS_coral_no_T_best_{stamp}.pt"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val_loss:.4f})")
                break
        scheduler.step()

    # --- Test evaluation ---
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    best_model = XPCSNetCoral(**model.get_architecture_config())
    load_matching_state_dict(
        best_model,
        torch.load(
            model_path / f"XPCS_coral_no_T_best_{stamp}.pt",
            weights_only=True,
            map_location=device,
        ),
    )
    best_model = best_model.to(device)
    best_model.eval()

    test_loss_pred = 0.0
    test_mae = torch.zeros(3, device=device)
    test_loader_sim = DataLoader(
        test_set_sim, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    with torch.no_grad():
        for x, y_norm_t, y_raw_t, T, _, _ in test_loader_sim:
            x, y_norm_t, y_raw_t, T = (
                x.to(device), y_norm_t.to(device), y_raw_t.to(device), T.to(device),
            )
            feats, _ = best_model.extract_features(x, T)
            shared = best_model.build_shared_features(feats, None)
            pred_logits = best_model.forward_predictor_from_shared_features(
                shared, return_logits=True,
            )
            pred_loss_t, _, pred_params = compute_prediction_loss(
                pred_logits, y_norm_t, best_model.predictor_output_activation,
            )
            pred_params_raw = denorm_from_meta(pred_params, norm_meta, device=device)
            test_mae += (pred_params_raw - y_raw_t).abs().sum(dim=0)
            test_loss_pred += pred_loss_t.item() * x.size(0)
    test_loss_pred /= len(test_set_sim)
    test_mae /= len(test_set_sim)
    print(f"Test MAE [gamma]: {test_mae[0]:.3e}")
    print(f"Test MAE [D]: {test_mae[1]:.3e}")
    print(f"Test MAE [GB_conc]: {test_mae[2]:.3e}")

    # --- Domain probe ---
    probe_metrics = run_domain_probe(
        best_model,
        train_dataset=train_set_full,
        val_dataset=val_set_full,
        test_dataset=test_set_full,
        device=device,
        seed=seed,
    )
    if probe_metrics is not None:
        print(
            f"Frozen-feature domain probe: "
            f"train acc/bal {probe_metrics['train_accuracy']:.3f}/"
            f"{probe_metrics['train_balanced_accuracy']:.3f}, "
            f"val acc/bal {probe_metrics['val_accuracy']:.3f}/"
            f"{probe_metrics['val_balanced_accuracy']:.3f}, "
            f"test acc/bal {probe_metrics['test_accuracy']:.3f}/"
            f"{probe_metrics['test_balanced_accuracy']:.3f}"
        )
        for key, value in probe_metrics.items():
            writer.add_scalar(f"probe/{key}", value, 0)

    training_metadata["domain_probe"] = probe_metrics
    save_training_metadata(metadata_path, training_metadata)
    save_training_metadata(checkpoint_metadata_path, training_metadata)

    writer.close()
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")
    print("Training complete.")
    return best_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def inference_sim(
    model: XPCSNetCoral,
    indices: Optional[List[int]] = None,
    sim_root: Path = Path("dataset/simulation"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> pd.DataFrame:
    sim_dataset = XPCSDataset(sim_root)
    norm_meta = sim_dataset.norm_meta
    if indices is not None:
        sim_dataset = Subset(sim_dataset, indices)
    model = model.to(device)
    model.eval()
    results = []
    for i in range(len(sim_dataset)):
        x, _, y_raw, T, _, _ = sim_dataset[i]
        x = x.unsqueeze(0).to(device)
        T = T.unsqueeze(0).to(device)
        pred_params_norm = model(x, T)
        pred_params_raw = denorm_from_meta(pred_params_norm.squeeze(0), norm_meta, device=device)
        results.append({
            "T": T.item(),
            "gamma_true": y_raw[0].item(),
            "D_true": y_raw[1].item(),
            "GB_conc_true": y_raw[2].item(),
            "gamma_pred": pred_params_raw[0].item(),
            "D_pred": pred_params_raw[1].item(),
            "GB_conc_pred": pred_params_raw[2].item(),
        })
    return pd.DataFrame(results)


@torch.no_grad()
def inference_exp(
    model: XPCSNetCoral,
    exp_root: Path = Path("exp_data"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    select_batches: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    norm_meta = model.norm_meta
    model = model.to(device)
    model.eval()
    results_dfs: Dict[str, pd.DataFrame] = {}
    for data_file in sorted(Path(exp_root).iterdir()):
        if data_file.suffix not in [".npy", ".npz"]:
            continue
        if data_file.suffix == ".npy":
            data = np.load(data_file)
        else:
            data = np.load(data_file)["g12"]
        T_val = 273.15 + float(str(data_file.name).split("T")[-1].split("C")[0])
        data_tensor = torch.tensor(data, dtype=torch.float32)
        results = []
        for i in range(data_tensor.size(-1)):
            if select_batches is not None and i not in select_batches:
                continue
            x = data_tensor[:2500, :2500, i]
            x = coarse_grain_g2(x, 256).unsqueeze(0).unsqueeze(0).to(device)
            x = (x - INPUT_MEAN) / (INPUT_STD + 1e-6)
            T = torch.tensor([[T_val]], dtype=torch.float32).to(device)
            pred_params_norm = model(x, T)
            pred_params_raw = denorm_from_meta(
                pred_params_norm.squeeze(0), norm_meta, device=device,
            )
            results.append({
                "T": T.item(),
                "gamma": pred_params_raw[0].item(),
                "D": pred_params_raw[1].item(),
                "GB_conc": pred_params_raw[2].item(),
            })
        results_dfs[data_file.stem] = pd.DataFrame(results)
    return results_dfs


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def build_model_from_checkpoint_metadata(model_path: Path) -> XPCSNetCoral:
    metadata_path = model_path.with_suffix(".json")
    architecture_config: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="ascii") as f:
            metadata = json.load(f)
        architecture_config = dict(metadata.get("architecture") or {})
        if architecture_config:
            print(
                f"[load] Reconstructing CORAL no-T architecture from {metadata_path.name}"
            )
    return XPCSNetCoral(**architecture_config)


def load_model(
    model_path: Optional[Path] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> XPCSNetCoral:
    if model_path is None:
        model_dir = Path("models")
        model_files = list(model_dir.glob("XPCS_coral_no_T_best_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No CORAL model files found in {model_dir}")
        model_files.sort(key=lambda x: x.stem.split("_")[-1], reverse=True)
        model_path = model_files[0]
        print(f"No model path specified, loading the most recent model: {model_path}")
    else:
        model_path = Path(model_path)
        print(f"Loading the model: {model_path}")
    model = build_model_from_checkpoint_metadata(model_path)
    load_matching_state_dict(
        model,
        torch.load(model_path, weights_only=True, map_location=device),
    )
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    set_global_seed(RANDOM_SEED, deterministic=True)
    m = XPCSNetCoral(predictor_output_activation="sigmoid")
    best = train(
        m,
        sim_root=Path("dataset/simulation"),
        exp_root=Path("dataset/experiment"),
        coral_weight=1.0,
        contrastive_weight=0.2,
        seed=RANDOM_SEED,
        deterministic=True,
    )
