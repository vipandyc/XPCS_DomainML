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

"""
The goal is to train an adversarial network for domain adaptation.
The network is comprised of three parts:
1. A feature extractor (CNN) that extracts features from input data.
2. A predictor that predicts the gamma, D and GB_conc from the extracted features.
3. A domain discriminator that tries to distinguish between simulated and experimental domains based on the extracted features.
"""

# --- Auxiliary Functions ---

def norm_from_meta(
    y_raw: torch.Tensor,
    norm_meta: Dict[str, Dict[str, Union[float, str]]],
    device: torch.device = None,
    eps=1e-300,  # to keep log10 defined
) -> torch.Tensor:
    """
    Normalize the raw parameters (gamma, D, GB_conc) to [0, 1] using the normalization
    meta info.
    
    Args:
        y_raw: [B, 3] or [3], listed in the order of (gamma, D, GB_conc).
        norm_meta: The normalization meta info per parameter, where each key is a parameter 
            name and each value is a dict with keys: 'low', 'high', 'scale' (either 'linear' 
            or 'log').
        device: The device to perform computation on. If None, use y_raw's device.
        eps: A small value to avoid log10(0).
        
    Returns:
        y_out: [B, 3] or [3], normalized parameters in [0, 1].
    """
    single = False
    if y_raw.ndim == 1:
        y_raw = y_raw.unsqueeze(0)
        single = True
    if device is None:
        device = y_raw.device
    y_out = torch.empty_like(y_raw, device=device)
    for j, key in enumerate(["gamma", "D", "GB_conc"]):
        meta = norm_meta[key]
        low, high = float(meta["low"]), float(meta["high"])
        low = torch.tensor(low, dtype=torch.float32, device=device)
        high = torch.tensor(high, dtype=torch.float32, device=device)
        scale = meta.get("scale", "linear")
        if scale == "log":
            y_out[:, j] = (
                (torch.log10(torch.clamp(y_raw[:, j], min=eps)) - torch.log10(low)) \
                / (torch.log10(high) - torch.log10(low))
            )
        else:
            y_out[:, j] = (y_raw[:, j] - low) / (high - low)
    if single:
        y_out = y_out.squeeze(0)
    return y_out.to(device)

def denorm_from_meta(
    y_norm: torch.Tensor,
    norm_meta: Dict[str, Dict[str, Union[float, str]]],
    device: torch.device = None,
) -> torch.Tensor:
    """
    De-normalize the predicted parameters (gamma, D, GB_conc) back to raw units using the
    normalization meta info.
    
    Args:
        y_norm: [B, 3] or [3], listed in the order of (gamma, D, GB_conc).
        norm_meta: The normalization meta info per parameter, where each key is a parameter 
            name and each value is a dict with keys: 'low', 'high', 'scale' (either 'linear' 
            or 'log').
        device: The device to perform computation on. If None, use y_norm's device.
        
    Returns:
        y_out: [B, 3] or [3], de-normalized parameters in raw physical units.
    """
    single = False
    if y_norm.ndim == 1:
        y_norm = y_norm.unsqueeze(0)
        single = True
    if device is None:
        device = y_norm.device
    y_out = torch.empty_like(y_norm, device=device)
    for j, key in enumerate(["gamma", "D", "GB_conc"]):
        meta = norm_meta[key]
        low, high = float(meta["low"]), float(meta["high"])
        low = torch.tensor(low, dtype=torch.float32, device=device)
        high = torch.tensor(high, dtype=torch.float32, device=device)
        scale = meta.get("scale", "linear")
        if scale == "log":
            y_out[:, j] = low * 10 ** (y_norm[:, j] * (torch.log10(high) - torch.log10(low)))
        else:
            y_out[:, j] = low + y_norm[:, j] * (high - low)
    if single:
        y_out = y_out.squeeze(0)
    return y_out.to(device)

def create_random_splits(
    dataset_size: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create random train/val/test splits from dataset indices.
    
    Args:
        dataset_size: Total number of samples in the dataset.
        train_ratio: Proportion of samples for training set.
        val_ratio: Proportion of samples for validation set.
        test_ratio: Proportion of samples for test set.
        seed: Random seed for reproducibility.
        
    Returns:
        train_indices: Indices for training set.
        val_indices: Indices for validation set.
        test_indices: Indices for test set.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def build_combined_domain_splits(
    sim_dataset_size: int,
    exp_dataset_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build disjoint train/val/test splits for the simulation-only and combined
    domain-classification datasets without leaking simulation val/test rows into
    adversarial training.

    Args:
        sim_dataset_size: Number of simulation samples.
        exp_dataset_size: Number of experiment samples.
        seed: Base reproducibility seed.

    Returns:
        train_indices_sim: Simulation-only train indices.
        val_indices_sim: Simulation-only validation indices.
        test_indices_sim: Simulation-only test indices.
        train_indices_full: Combined-domain train indices.
        val_indices_full: Combined-domain validation indices.
        test_indices_full: Combined-domain test indices.
    """
    train_indices_sim, val_indices_sim, test_indices_sim = create_random_splits(
        sim_dataset_size,
        seed=seed,
    )
    train_indices_exp, val_indices_exp, test_indices_exp = create_random_splits(
        exp_dataset_size,
        seed=seed + 1,
    )
    exp_offset = sim_dataset_size
    train_indices_full = np.concatenate([train_indices_sim, exp_offset + train_indices_exp])
    val_indices_full = np.concatenate([val_indices_sim, exp_offset + val_indices_exp])
    test_indices_full = np.concatenate([test_indices_sim, exp_offset + test_indices_exp])
    return (
        train_indices_sim,
        val_indices_sim,
        test_indices_sim,
        train_indices_full,
        val_indices_full,
        test_indices_full,
    )


def compute_grl_alpha(progress: float) -> float:
    """
    Compute the DANN-style adversarial weight from normalized training
    progress.

    The schedule starts near zero so the encoder can first learn predictive
    structure, then smoothly ramps toward one as training progresses:

        lambda(p) = 2 / (1 + exp(-10 p)) - 1

    Args:
        progress: Normalized progress in [0, 1].

    Returns:
        alpha: GRL scaling factor in [0, 1).
    """
    clipped_progress = min(max(float(progress), 0.0), 1.0)
    return float((2.0 / (1.0 + math.exp(-10.0 * clipped_progress))) - 1.0)


def compute_binary_classification_metrics(
    confusion: np.ndarray,
) -> dict[str, float]:
    """
    Summarize a 2x2 confusion matrix for binary domain classification.

    Rows correspond to true labels and columns correspond to predicted labels.

    Args:
        confusion: Integer confusion matrix with shape [2, 2].

    Returns:
        metrics: Accuracy, balanced accuracy, per-class recall, and the fraction
            of predictions assigned to the experimental class.
    """
    if confusion.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, got {confusion.shape}")
    total = float(confusion.sum())
    if total == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "recall_sim": float("nan"),
            "recall_exp": float("nan"),
            "predicted_exp_fraction": float("nan"),
        }
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


def update_binary_confusion(
    confusion: np.ndarray,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """
    Accumulate predictions into a binary confusion matrix in-place.

    Args:
        confusion: Running confusion matrix with shape [2, 2].
        logits: Unnormalized classifier outputs of shape [B, 2].
        labels: True labels of shape [B].
    """
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    for true_label, pred_label in zip(labels_np, preds):
        confusion[int(true_label), int(pred_label)] += 1


@torch.no_grad()
def extract_domain_probe_features(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract frozen shared features and domain labels for the linear probe.

    Args:
        model: Trained XPCS model.
        dataset: Dataset returning `(x, ..., label)`.
        device: Device used for feature extraction.
        batch_size: DataLoader batch size.

    Returns:
        features: Array of shape [N, latent_dim].
        labels: Binary domain labels of shape [N].
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    features = []
    labels = []
    for x, _, _, T, batch_labels in loader:
        x = x.to(device)
        T = T.to(device)
        xpcs_features, temp_features = model.extract_features(x, T)
        shared_features = model.build_shared_features(xpcs_features, temp_features)
        features.append(shared_features.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def run_domain_probe(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    device: torch.device,
    seed: int,
) -> Optional[dict[str, float]]:
    """
    Train a balanced linear probe on frozen shared features to measure the
    residual domain separability of the learned feature space.

    Args:
        model: Trained XPCS model.
        train_dataset: Domain-classification training subset.
        val_dataset: Domain-classification validation subset.
        test_dataset: Domain-classification test subset.
        device: Device used to extract features.
        seed: Reproducibility seed for the linear probe.

    Returns:
        results: Flat metrics dictionary, or None if scikit-learn is unavailable.
    """
    if LogisticRegression is None:
        return None

    train_features, train_labels = extract_domain_probe_features(
        model,
        train_dataset,
        device=device,
    )
    val_features, val_labels = extract_domain_probe_features(
        model,
        val_dataset,
        device=device,
    )
    test_features, test_labels = extract_domain_probe_features(
        model,
        test_dataset,
        device=device,
    )

    probe = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
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
        results[f"{split_name}_accuracy"] = metrics["accuracy"]
        results[f"{split_name}_balanced_accuracy"] = metrics["balanced_accuracy"]
        results[f"{split_name}_recall_sim"] = metrics["recall_sim"]
        results[f"{split_name}_recall_exp"] = metrics["recall_exp"]
        results[f"{split_name}_predicted_exp_fraction"] = metrics["predicted_exp_fraction"]
    return results


def run_domain_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    domain_class_weights: torch.Tensor,
    grl_alpha: float,
    domain_optimizer: Optional[torch.optim.Optimizer] = None,
    encoder_optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, np.ndarray]:
    """
    Run one full pass of domain classification.

    Args:
        model: Adversarial model.
        loader: Domain-classification DataLoader.
        device: Torch device.
        domain_class_weights: Class weights for the domain CE loss.
        grl_alpha: GRL strength applied to encoder gradients.
        domain_optimizer: Optimizer for the domain head. If omitted, no training
            step is taken.
        encoder_optimizer: Optimizer for encoder-side parameters. If provided,
            the reversed GRL gradient updates the encoder on this pass.

    Returns:
        mean_loss: Mean cross-entropy loss over the full loader.
        confusion: 2x2 confusion matrix accumulated over the pass.
    """
    mean_loss = 0.0
    confusion = np.zeros((2, 2), dtype=np.int64)
    is_training = domain_optimizer is not None or encoder_optimizer is not None
    model.off_pred_mode().on_class_mode().set_grl_alpha(grl_alpha)

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for x, _, _, T, labels in loader:
            x, T, labels = x.to(device), T.to(device), labels.to(device)
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            if domain_optimizer is not None:
                domain_optimizer.zero_grad()

            domain_out = model(x, T)
            class_loss = F.cross_entropy(
                domain_out,
                labels,
                weight=domain_class_weights,
            )

            if is_training:
                class_loss.backward()
                if encoder_optimizer is not None:
                    encoder_optimizer.step()
                if domain_optimizer is not None:
                    domain_optimizer.step()

            update_binary_confusion(confusion, domain_out, labels)
            mean_loss += class_loss.item() * x.size(0)

    mean_loss /= max(1, len(loader.dataset))
    return mean_loss, confusion


def next_loader_batch(
    loader: DataLoader,
    iterator: Optional[Any],
) -> tuple[tuple[torch.Tensor, ...], Any]:
    """
    Fetch the next mini-batch from a DataLoader, restarting the iterator after
    exhaustion so alternating updates can keep cycling smoothly.

    Args:
        loader: DataLoader to draw from.
        iterator: Existing iterator or None.

    Returns:
        batch: Next mini-batch from `loader`.
        iterator: Iterator positioned after the returned batch.
    """
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Toggle gradient tracking for every parameter in a module.

    Args:
        module: Module whose parameters should be frozen or unfrozen.
        requires_grad: Whether gradients should be tracked.
    """
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def run_discriminator_step(
    model: "XPCSNet",
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    domain_class_weights: torch.Tensor,
    domain_optimizer: torch.optim.Optimizer,
) -> tuple[float, np.ndarray, int]:
    """
    Update only the domain discriminator on one combined-domain mini-batch while
    keeping the shared feature extractor frozen.

    Args:
        model: Adversarial model.
        batch: `(x, _, _, T, domain_label)` mini-batch from the combined loader.
        device: Torch device.
        domain_class_weights: Class weights for binary domain CE.
        domain_optimizer: Optimizer for the domain classifier.

    Returns:
        loss_value: Scalar discriminator loss for the mini-batch.
        confusion: 2x2 confusion matrix for this mini-batch.
        batch_size: Number of domain examples consumed.
    """
    x, _, _, T, labels = batch
    x, T, labels = x.to(device), T.to(device), labels.to(device)
    domain_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        xpcs_features, temp_features = model.extract_features(x, T)
        shared_features = model.build_shared_features(xpcs_features, temp_features)
    domain_out = model.forward_domain_logits(shared_features, apply_grl=False)
    class_loss = F.cross_entropy(domain_out, labels, weight=domain_class_weights)
    class_loss.backward()
    domain_optimizer.step()

    confusion = np.zeros((2, 2), dtype=np.int64)
    update_binary_confusion(confusion, domain_out, labels)
    return class_loss.item(), confusion, x.size(0)


def run_generator_predictor_step(
    model: "XPCSNet",
    sim_batch: tuple[torch.Tensor, ...],
    domain_batch: tuple[torch.Tensor, ...],
    device: torch.device,
    prediction_optimizer: torch.optim.Optimizer,
    domain_optimizer: torch.optim.Optimizer,
    domain_class_weights: torch.Tensor,
    grl_alpha: float,
) -> tuple[float, float, float, np.ndarray, int, int]:
    """
    Update the encoder/predictor on one simulation mini-batch plus one domain
    mini-batch while keeping the discriminator fixed.

    Args:
        model: Adversarial model.
        sim_batch: `(x, y_norm, _, T, _)` mini-batch from the simulation loader.
        domain_batch: `(x, _, _, T, domain_label)` mini-batch from the combined
            domain loader.
        device: Torch device.
        prediction_optimizer: Optimizer for encoder/predictor parameters.
        domain_optimizer: Discriminator optimizer, zeroed here so stale
            gradients do not accumulate while D is frozen.
        domain_class_weights: Class weights for binary domain CE.
        grl_alpha: GRL strength applied to encoder gradients.

    Returns:
        pred_loss_value: Scalar regression loss on the simulation mini-batch.
        domain_loss_value: Scalar domain CE on the combined mini-batch.
        total_loss_value: Combined loss used for the encoder/predictor update.
        confusion: 2x2 confusion matrix for the domain mini-batch.
        sim_batch_size: Number of simulation examples consumed.
        domain_batch_size: Number of domain examples consumed.
    """
    x_sim, y_norm, _, T_sim, _ = sim_batch
    x_dom, _, _, T_dom, labels_dom = domain_batch
    x_sim = x_sim.to(device)
    y_norm = y_norm.to(device)
    T_sim = T_sim.to(device)
    x_dom = x_dom.to(device)
    T_dom = T_dom.to(device)
    labels_dom = labels_dom.to(device)

    prediction_optimizer.zero_grad(set_to_none=True)
    domain_optimizer.zero_grad(set_to_none=True)

    standardizer_was_training = model.domain_feature_standardizer.training
    classifier_was_training = model.domain_classifier.training
    model.domain_feature_standardizer.eval()
    model.domain_classifier.eval()
    set_module_requires_grad(model.domain_classifier, False)
    try:
        xpcs_features_sim, temp_features_sim = model.extract_features(x_sim, T_sim)
        shared_features_sim = model.build_shared_features(
            xpcs_features_sim,
            temp_features_sim,
        )
        pred_logits = model.forward_predictor_from_shared_features(
            shared_features_sim,
            return_logits=True,
        )
        pred_loss, _, _ = compute_prediction_loss(
            pred_logits,
            y_norm,
            model.predictor_output_activation,
        )

        xpcs_features_dom, temp_features_dom = model.extract_features(x_dom, T_dom)
        shared_features_dom = model.build_shared_features(
            xpcs_features_dom,
            temp_features_dom,
        )
        domain_out = model.forward_domain_logits(
            shared_features_dom,
            apply_grl=True,
            grl_alpha=grl_alpha,
        )
        domain_loss = F.cross_entropy(
            domain_out,
            labels_dom,
            weight=domain_class_weights,
        )

        total_loss = pred_loss + domain_loss
        total_loss.backward()
    finally:
        set_module_requires_grad(model.domain_classifier, True)
        model.domain_feature_standardizer.train(standardizer_was_training)
        model.domain_classifier.train(classifier_was_training)

    prediction_optimizer.step()

    confusion = np.zeros((2, 2), dtype=np.int64)
    update_binary_confusion(confusion, domain_out, labels_dom)
    return (
        pred_loss.item(),
        domain_loss.item(),
        total_loss.item(),
        confusion,
        x_sim.size(0),
        x_dom.size(0),
    )


class FeatureStandardizer(nn.Module):
    """
    Standardize latent features per dimension using running dataset statistics.

    This mirrors the train-set standardization that stabilized the standalone
    domain-classifier experiments, while remaining usable inside the live
    adversarial loop where features evolve over time.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected [batch, {self.num_features}] features, got {tuple(x.shape)}"
            )

        use_batch_stats = self.training and x.shape[0] > 1
        if use_batch_stats:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            self.running_mean.lerp_(batch_mean.detach(), self.momentum)
            self.running_var.lerp_(batch_var.detach(), self.momentum)
            self.num_batches_tracked += 1
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        return (x - mean) / torch.sqrt(var + self.eps)


def pretrain_domain_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    domain_class_weights: torch.Tensor,
    domain_optimizer: torch.optim.Optimizer,
    epochs: int,
    writer: SummaryWriter,
) -> None:
    """
    Warm-start the domain head on frozen shared features before adversarial
    training begins.

    Args:
        model: Adversarial model.
        train_loader: Domain-classification training DataLoader.
        val_loader: Domain-classification validation DataLoader.
        device: Torch device.
        domain_class_weights: Class weights for domain CE.
        domain_optimizer: Optimizer that updates only the domain head.
        epochs: Number of classifier warm-start epochs.
        writer: TensorBoard writer.
    """
    if epochs <= 0:
        return

    print(
        "[domain-pretrain] Warming up the domain classifier on frozen shared "
        f"features for {epochs} epoch(s)"
    )
    for epoch in range(epochs):
        model.train()
        train_loss, train_confusion = run_domain_pass(
            model,
            train_loader,
            device=device,
            domain_class_weights=domain_class_weights,
            grl_alpha=0.0,
            domain_optimizer=domain_optimizer,
            encoder_optimizer=None,
        )
        train_metrics = compute_binary_classification_metrics(train_confusion)

        model.eval()
        val_loss, val_confusion = run_domain_pass(
            model,
            val_loader,
            device=device,
            domain_class_weights=domain_class_weights,
            grl_alpha=0.0,
            domain_optimizer=None,
            encoder_optimizer=None,
        )
        val_metrics = compute_binary_classification_metrics(val_confusion)

        print(
            f"[domain-pretrain {epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"(Acc: {train_metrics['accuracy']:.3f}, "
            f"Bal: {train_metrics['balanced_accuracy']:.3f}, "
            f"PredExp: {train_metrics['predicted_exp_fraction']:.3f}) "
            f"Val Loss: {val_loss:.6f} "
            f"(Acc: {val_metrics['accuracy']:.3f}, "
            f"Bal: {val_metrics['balanced_accuracy']:.3f}, "
            f"PredExp: {val_metrics['predicted_exp_fraction']:.3f})"
        )
        writer.add_scalar("domain_pretrain/train_loss", train_loss, epoch)
        writer.add_scalar("domain_pretrain/train_accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar(
            "domain_pretrain/train_balanced_accuracy",
            train_metrics["balanced_accuracy"],
            epoch,
        )
        writer.add_scalar(
            "domain_pretrain/train_predicted_exp_fraction",
            train_metrics["predicted_exp_fraction"],
            epoch,
        )
        writer.add_scalar("domain_pretrain/val_loss", val_loss, epoch)
        writer.add_scalar("domain_pretrain/val_accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar(
            "domain_pretrain/val_balanced_accuracy",
            val_metrics["balanced_accuracy"],
            epoch,
        )
        writer.add_scalar(
            "domain_pretrain/val_predicted_exp_fraction",
            val_metrics["predicted_exp_fraction"],
            epoch,
        )


def load_matching_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """
    Load only the checkpoint tensors whose names and shapes match the current
    model, which keeps older checkpoints usable after architecture changes.

    Args:
        model: Model to receive compatible weights.
        state_dict: Checkpoint state dictionary.

    Returns:
        summary: Short bookkeeping about what was loaded or skipped.
    """
    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(filtered_state))
    missing_before_load = sorted(set(model_state) - set(filtered_state))
    model.load_state_dict(filtered_state, strict=False)
    return {
        "loaded": sorted(filtered_state.keys()),
        "skipped": skipped,
        "missing": missing_before_load,
    }


def canonicalize_architecture_config(
    architecture_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize saved architecture metadata onto the current `XPCSNet`
    constructor names while keeping older checkpoints loadable.

    Args:
        architecture_config: Raw architecture dictionary from metadata.

    Returns:
        normalized_config: Constructor kwargs understood by the current model.
    """
    normalized_config = dict(architecture_config)
    if "use_prediction_feature_mixer" in normalized_config:
        normalized_config["use_shared_feature_mixer"] = normalized_config.pop(
            "use_prediction_feature_mixer"
        )
    if "prediction_feature_dim" in normalized_config:
        normalized_config["shared_feature_dim"] = normalized_config.pop(
            "prediction_feature_dim"
        )
    if "prediction_feature_mixer_hidden_dim" in normalized_config:
        normalized_config["shared_feature_mixer_hidden_dim"] = normalized_config.pop(
            "prediction_feature_mixer_hidden_dim"
        )
    return normalized_config

# --- Dataset ---

class XPCSDataset(Dataset):
    """
    Loads samples from dataset based on either one manifest.csv and stats.json, 
    or multiple manifests. Splits are dynamically handled.
    Returns (x_norm, y_norm, y_raw, T, label) where:
      - x_norm: torch.FloatTensor [1, 256, 256], normalized with `normalize_g2`
        and masked on the diagonal
      - y_norm: torch.FloatTensor [3], targets scaled to [0,1] per-parameter
      - y_raw : torch.FloatTensor [3], raw targets in physical units
      - T     : torch.FloatTensor [1], temperature (not normalized)
      - label : int, 0 for simulated, 1 for experimental
      
    Args:
        paths: The root directories, each containing a manifest.csv.
    """
    PARAM_KEYS = ["gamma", "D", "GB_conc"]

    def __init__(self, paths: Union[Path, List[Path]]):
        self.paths = paths if isinstance(paths, list) else [paths]
        self.manifest = pd.concat([
            pd.read_csv(Path(path) / "manifest.csv") for path in self.paths
        ], ignore_index=True)

        self.norm_meta = {
            "gamma": {"low": 2e18, "high": 5e18, "scale": "linear"},
            "D": {"low": 1e-23, "high": 1e-21, "scale": "log"},
            "GB_conc": {"low": 0.0, "high": 0.3, "scale": "linear"},
            "T": {"low": 300, "high": 500, "scale": "linear"}
        }

        # precompute a [1, 256, 256] diagonal mask (zeros on diag)
        # so the network never sees diagonal pixels
        # TODO: ablation needed!
        self.diag_mask = torch.ones(1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        self.diag_mask[0, range(INPUT_SIZE), range(INPUT_SIZE)] = 0.0

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        x = torch.load(row["path"], weights_only=True).to(torch.float32).squeeze(0)  # [1, 256, 256]
        x = (x - INPUT_MEAN) / (INPUT_STD + 1e-6)
        x = x * self.diag_mask

        # raw targets (only predicted parameters)
        y_raw = torch.tensor([row[k] for k in self.PARAM_KEYS], dtype=torch.float32)
        y_norm = norm_from_meta(y_raw, self.norm_meta, device=y_raw.device)
        
        # temperature (not predicted, passed as additional input)
        T = torch.tensor([row["T"]], dtype=torch.float32)
        
        # domain label
        domain = row["domain"] if "domain" in row.index else (
            "simulation" if "simulation" in str(row["path"]) else "experiment"
        )
        label = 0 if domain == "simulation" else 1
        
        return x, y_norm, y_raw, T, label

# --- Model ---
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha=1.0):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output = grad_output.neg() * ctx.alpha
        return output, None

class XPCSNet(nn.Module):
    """
    Implement the XPCSNet model. The input XPCS prediction shape: (B, 1, 256, 256).
    
    If prediction_mode is True, return the predicted parameters (gamma, D, 
    GB_conc). If classification_mode is True, return the domain classification output.
    """
    # TODO: 
    # 1. diag_mask (done)
    # 2. compare the two model designs: one with batchnorm and one with maxpool2d
    # 3. compare adaptiveavgpool2d(1) and adaptiveavgpool2d((5,5))
    # 4. temperature projection (done)
    # 5. dropout? (no dropout)
    
    def __init__(
        self,
        prediction_mode: bool = True,
        classification_mode: bool = False,
        predictor_output_activation: str = "linear",
    ):
        super().__init__()
        if predictor_output_activation not in {"linear", "sigmoid"}:
            raise ValueError(
                "predictor_output_activation must be 'linear' or 'sigmoid'"
            )
        self.norm_meta = {
            "gamma": {"low": 2e18, "high": 5e18, "scale": "linear"},
            "D": {"low": 1e-23, "high": 1e-21, "scale": "log"},
            "GB_conc": {"low": 0.0, "high": 0.3, "scale": "linear"},
            "T": {"low": 300, "high": 500, "scale": "linear"}
        }
        self.prediction_mode = prediction_mode
        self.classification_mode = classification_mode
        self.predictor_output_activation = predictor_output_activation
        self.grl_alpha = 1.0
        self.xpcs_feature_dim = 128
        self.shared_feature_dim = self.xpcs_feature_dim
        self.conv_net = Sequential(
            Conv2d(
                in_channels=1, out_channels=32, kernel_size=3,
                stride=1, padding=2
            ),
            MaxPool2d(kernel_size=3, stride=2),
            LeakyReLU(negative_slope=0.01),   
            Conv2d(
                in_channels=32, out_channels=64, kernel_size=3,
                stride=1, padding=2
            ),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU(),        
            Conv2d(
                in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=2
            ),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU(),
            Conv2d(
                in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=2
            ),
            MaxPool2d(kernel_size=3, stride=2),
            ReLU(),
            # nn.AdaptiveAvgPool2d(output_size=(5, 5)),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
        )   # Output shape: (B, 128)

        # Probe experiments showed conditioning, not regularization, was the
        # main classifier bottleneck, so dropout stays disabled by default.
        self.dropout = Dropout(0.0)
        self.domain_feature_standardizer = FeatureStandardizer(self.shared_feature_dim)

        self.domain_classifier = Sequential(
            Linear(self.shared_feature_dim, 64),
            ReLU(),
            self.dropout,
            nn.Linear(64, 64),
            ReLU(),
            self.dropout,
            nn.Linear(64, 2),
        )
        
        self.xpcs_predictor = Sequential(
            Linear(self.shared_feature_dim, 64),
            ReLU(),
            self.dropout,
            nn.Linear(64, 64),
            ReLU(),
            self.dropout,
            nn.Linear(64, 3),
        )

    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Export the architecture knobs needed to reconstruct this model.

        Returns:
            config: Keyword arguments that can be passed back into `XPCSNet`.
        """
        return {
            "predictor_output_activation": self.predictor_output_activation,
        }

    def extract_features(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """
        Extract the XPCS feature branch and ignore explicit temperature input.

        Args:
            x: Input XPCS tensor of shape `(B, 1, 256, 256)`.
            T: Temperature tensor of shape `(B, 1)`, ignored in the no-T model.

        Returns:
            xpcs_features: CNN features of shape `(B, 128)`.
            temp_features: Placeholder `None` kept for API compatibility.
        """
        del T
        xpcs_features = self.conv_net(x)
        return xpcs_features, None

    def build_shared_features(
        self,
        xpcs_features: torch.Tensor,
        temp_features: None,
    ) -> torch.Tensor:
        """
        Use the XPCS branch directly as the shared representation.

        Args:
            xpcs_features: CNN features of shape `(B, 128)`.
            temp_features: Placeholder `None`, ignored.

        Returns:
            shared_features: Shared head input features.
        """
        del temp_features
        return xpcs_features

    def forward_predictor_from_shared_features(
        self,
        shared_features: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Predict normalized `(gamma, D, GB_conc)` values from the shared mixed
        representation.

        Args:
            shared_features: Shared features consumed by both heads.

        Returns:
            predicted_params: Normalized target predictions.
        """
        pred_logits = self.xpcs_predictor(shared_features)
        if return_logits:
            return pred_logits
        if self.predictor_output_activation == "sigmoid":
            return torch.sigmoid(pred_logits)
        return pred_logits

    def forward_domain_logits(
        self,
        shared_features: torch.Tensor,
        apply_grl: bool = True,
        grl_alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run the domain classifier on the shared mixed representation.

        Args:
            shared_features: Shared mixed features consumed by both heads.
            apply_grl: Whether to reverse the encoder gradient.
            grl_alpha: Optional override for the GRL strength.

        Returns:
            domain_out: Domain-classification logits of shape `(B, 2)`.
        """
        standardized_domain_features = self.domain_feature_standardizer(
            shared_features
        )
        if apply_grl:
            effective_grl_alpha = self.grl_alpha if grl_alpha is None else float(grl_alpha)
            standardized_domain_features = ReverseLayerF.apply(
                standardized_domain_features,
                effective_grl_alpha,
            )
        return self.domain_classifier(standardized_domain_features)

    def forward(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of XPCSNet. The predicted parameters are normalized parameters
        (gamma, D, GB_conc) in [0, 1].
        
        Parameters:
            x: Input XPCS data of shape (B, 1, 256, 256)
            T: Temperature data of shape (B, 1)
            
        Returns:
            If both prediction_mode and classification_mode are True, returns a 
            tuple of (predicted_params, domain_out).
            If only prediction_mode is True, return the predicted parameters.
            If only classification_mode is True, returns the domain classification 
            output.
        """
        xpcs_features, temp_features = self.extract_features(x, T)
        shared_features = self.build_shared_features(
            xpcs_features,
            temp_features,
        )
        
        returned_values = ()
        
        if self.prediction_mode:
            predicted_params = self.forward_predictor_from_shared_features(
                shared_features,
            )
            returned_values += (predicted_params,)
        if self.classification_mode:
            domain_out = self.forward_domain_logits(
                shared_features,
                apply_grl=True,
            )
            returned_values += (domain_out,)
            
        return returned_values if len(returned_values) > 1 else returned_values[0]
       
    def on_pred_mode(self):
        """
        Enable prediction mode. Return the predicted parameters.
        """
        self.prediction_mode = True
        return self
        
    def off_pred_mode(self):
        """
        Disable prediction mode. Do not return predicted parameters.
        """
        self.prediction_mode = False
        return self
        
    def on_class_mode(self):
        """
        Enable classification mode. Return domain classification output.
        """
        self.classification_mode = True
        return self
        
    def off_class_mode(self):
        """
        Disable classification mode. Do not return domain classification output.
        """
        self.classification_mode = False
        return self

    def set_grl_alpha(self, alpha: float):
        """
        Update the gradient-reversal scaling used by the domain head.

        Args:
            alpha: Non-negative GRL multiplier.
        """
        self.grl_alpha = float(alpha)
        return self

# --- Training Loop ---

def train(
    model: XPCSNet,
    sim_root: Path = Path("dataset"),
    exp_root: Path = Path("dataset"),
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    adaptation_rate: float = 1.2,
    domain_learning_rate: Optional[float] = None,
    seed: int = RANDOM_SEED,
    deterministic: bool = True,
    num_workers: int = 0,
    warmup_epochs: int = 20,
    domain_pretrain_epochs: int = 10,
    domain_steps_per_iteration: int = 5,
    prediction_steps_per_iteration: int = 1,
    domain_only_passes: int = 0,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    log_pardir: Path = Path("runs"),
    model_path: Path = Path("models"), 
) -> XPCSNet:
    """
    Train the XPCSNet model with adversarial training for domain adaptation.
    
    Args:
        model: The XPCSNet model to be trained.
        sim_root: Path to the simulated dataset root directory.
        exp_root: Path to the experimental dataset root directory.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        adaptation_rate: Maximum GRL strength applied to the encoder during
            adversarial alignment. The domain-classification loss itself stays
            at full strength.
        domain_learning_rate: Learning rate for the domain-classifier
            optimizer. If None, default to `1e-4` based on the standalone
            probe experiments.
        seed: Global random seed used for initialization, splits, and shuffling.
        deterministic: Whether to enforce deterministic Torch behavior.
        num_workers: Number of DataLoader workers.
        warmup_epochs: Number of epochs over which the DANN-style GRL schedule
            saturates to its maximum value. If non-positive, the schedule spans
            the full training run.
        domain_pretrain_epochs: Number of classifier-only warm-start epochs on
            frozen shared features before adversarial training begins.
        domain_steps_per_iteration: Number of discriminator mini-batch updates
            per outer minimax iteration.
        prediction_steps_per_iteration: Number of encoder/predictor mini-batch
            updates per outer minimax iteration.
        domain_only_passes: Deprecated compatibility knob. Any positive value is
            added on top of `domain_steps_per_iteration`.
        init_state_dict: Optional checkpoint state used to initialize matching
            layers before adversarial training starts.
        device: Device to perform training on (CPU or GPU).
        log_pardir: Parent directory for TensorBoard logs.
        model_path: Directory to save the best model checkpoints.
        
    Returns:
        best_model: The best-performing XPCSNet model on the validation set.
    """
    set_global_seed(seed, deterministic=deterministic)
    domain_learning_rate = 1e-4 if domain_learning_rate is None else domain_learning_rate
    if domain_steps_per_iteration <= 0:
        raise ValueError("domain_steps_per_iteration must be positive")
    if prediction_steps_per_iteration <= 0:
        raise ValueError("prediction_steps_per_iteration must be positive")
    effective_domain_steps_per_iteration = (
        domain_steps_per_iteration + max(0, domain_only_passes)
    )
    if domain_only_passes > 0:
        print(
            "[schedule] `domain_only_passes` is deprecated; folding it into "
            f"`domain_steps_per_iteration`, giving k={effective_domain_steps_per_iteration}"
        )

    # Set up logging
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = log_pardir / f"xpcs_no_T_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[logger] TensorBoard log dir: {log_dir}")
    print(f"[seed] Adversarial training seed: {seed} (deterministic={deterministic})")
    
    # Load datasets
    sim_dataset = XPCSDataset(sim_root)
    exp_dataset = XPCSDataset(exp_root)
    full_dataset = XPCSDataset([sim_root, exp_root])
    norm_meta = sim_dataset.norm_meta
    
    # Create splits and loaders
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
    train_sim_count = len(train_indices_sim)
    train_exp_count = len(train_indices_full) - train_sim_count
    val_sim_count = len(val_indices_sim)
    val_exp_count = len(val_indices_full) - val_sim_count
    test_sim_count = len(test_indices_sim)
    test_exp_count = len(test_indices_full) - test_sim_count
    domain_class_weights = torch.tensor(
        [
            len(train_indices_full) / (2.0 * max(1, train_sim_count)),
            len(train_indices_full) / (2.0 * max(1, train_exp_count)),
        ],
        dtype=torch.float32,
        device=device,
    )

    training_metadata: Dict[str, Any] = {
        "timestamp": stamp,
        "seed": seed,
        "deterministic": deterministic,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "domain_learning_rate": domain_learning_rate,
        "adaptation_rate": adaptation_rate,
        "warmup_epochs": warmup_epochs,
        "grl_schedule": {
            "type": "dann_logistic",
            "slope": 10.0,
            "warmup_epochs": warmup_epochs,
            "max_scale": adaptation_rate,
        },
        "domain_pretrain_epochs": domain_pretrain_epochs,
        "domain_steps_per_iteration": domain_steps_per_iteration,
        "prediction_steps_per_iteration": prediction_steps_per_iteration,
        "domain_only_passes_deprecated": domain_only_passes,
        "effective_domain_steps_per_iteration": effective_domain_steps_per_iteration,
        "epoch_anchor": "simulation_prediction_loader",
        "architecture": model.get_architecture_config(),
        "domain_feature_standardization": {
            "enabled": True,
            "type": "running_feature_standardizer",
            "eps": model.domain_feature_standardizer.eps,
            "momentum": model.domain_feature_standardizer.momentum,
        },
        "initialized_from_matching_state": init_state_dict is not None,
        "input_normalization": "global_zscore",
        "input_mean": INPUT_MEAN,
        "input_std": INPUT_STD,
        "simulation_dataset_root": str(sim_root),
        "experiment_dataset_root": str(exp_root),
        "simulation_dataset_size": len(sim_dataset),
        "experiment_dataset_size": len(exp_dataset),
        "full_dataset_size": len(full_dataset),
        "train_domain_counts": {"simulation": train_sim_count, "experiment": train_exp_count},
        "val_domain_counts": {"simulation": val_sim_count, "experiment": val_exp_count},
        "test_domain_counts": {"simulation": test_sim_count, "experiment": test_exp_count},
        "domain_class_weights": {
            "simulation": float(domain_class_weights[0].item()),
            "experiment": float(domain_class_weights[1].item()),
        },
        "train_indices_sim": train_indices_sim.tolist(),
        "val_indices_sim": val_indices_sim.tolist(),
        "test_indices_sim": test_indices_sim.tolist(),
        "train_indices_full": train_indices_full.tolist(),
        "val_indices_full": val_indices_full.tolist(),
        "test_indices_full": test_indices_full.tolist(),
    }
    if "id" in sim_dataset.manifest.columns:
        training_metadata["train_ids_sim"] = sim_dataset.manifest.iloc[train_indices_sim]["id"].astype(int).tolist()
        training_metadata["val_ids_sim"] = sim_dataset.manifest.iloc[val_indices_sim]["id"].astype(int).tolist()
        training_metadata["test_ids_sim"] = sim_dataset.manifest.iloc[test_indices_sim]["id"].astype(int).tolist()
    if "id" in exp_dataset.manifest.columns:
        exp_train_indices, exp_val_indices, exp_test_indices = create_random_splits(
            len(exp_dataset),
            seed=seed + 1,
        )
        training_metadata["train_ids_exp"] = exp_dataset.manifest.iloc[exp_train_indices]["id"].astype(int).tolist()
        training_metadata["val_ids_exp"] = exp_dataset.manifest.iloc[exp_val_indices]["id"].astype(int).tolist()
        training_metadata["test_ids_exp"] = exp_dataset.manifest.iloc[exp_test_indices]["id"].astype(int).tolist()
    metadata_path = log_dir / "training_metadata.json"
    checkpoint_metadata_path = model_path / f"XPCS_no_T_best_{stamp}.json"
    save_training_metadata(metadata_path, training_metadata)
    save_training_metadata(checkpoint_metadata_path, training_metadata)
    print(f"[repro] Saved training metadata -> {metadata_path}")

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_dataloader_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_sim_generator = torch.Generator()
    train_sim_generator.manual_seed(seed)
    train_full_generator = torch.Generator()
    train_full_generator.manual_seed(seed + 1)

    train_loader_sim = DataLoader(
        train_set_sim,
        shuffle=True,
        generator=train_sim_generator,
        **loader_kwargs,
    )
    train_loader_full = DataLoader(
        train_set_full,
        shuffle=True,
        generator=train_full_generator,
        **loader_kwargs,
    )
    val_loader_sim = DataLoader(
        val_set_sim,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader_full = DataLoader(
        val_set_full,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader_sim = DataLoader(
        test_set_sim,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader_full = DataLoader(
        test_set_full,
        shuffle=False,
        **loader_kwargs,
    )
    
    # Set up optimizer and scheduler
    model = model.to(device)
    if init_state_dict is not None:
        init_summary = load_matching_state_dict(model, init_state_dict)
        print(
            "[init] Loaded matching initialization weights: "
            f"{len(init_summary['loaded'])} tensors, "
            f"skipped {len(init_summary['skipped'])}"
        )
    prediction_parameters = (
        list(model.conv_net.parameters())
        + list(model.xpcs_predictor.parameters())
    )
    domain_parameters = list(model.domain_classifier.parameters())
    prediction_optimizer = torch.optim.Adam(prediction_parameters, lr=learning_rate)
    domain_optimizer = torch.optim.Adam(domain_parameters, lr=domain_learning_rate)
    prediction_scheduler = CosineAnnealingLR(prediction_optimizer, T_max=epochs)
    best_val_loss = float("inf")
    patience = 20
    bad_epochs = 0

    pretrain_domain_classifier(
        model,
        train_loader=train_loader_full,
        val_loader=val_loader_full,
        device=device,
        domain_class_weights=domain_class_weights,
        domain_optimizer=domain_optimizer,
        epochs=domain_pretrain_epochs,
        writer=writer,
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_pred_sum = 0.0
        train_discriminator_sum = 0.0
        train_adv_domain_sum = 0.0
        train_adv_total_sum = 0.0
        train_pred_examples = 0
        train_discriminator_examples = 0
        train_adv_domain_examples = 0
        train_adv_step_count = 0
        train_grl_alpha_sum = 0.0
        train_discriminator_confusion = np.zeros((2, 2), dtype=np.int64)
        train_adv_domain_confusion = np.zeros((2, 2), dtype=np.int64)
        outer_iterations = max(
            1,
            math.ceil(len(train_loader_sim) / prediction_steps_per_iteration),
        )
        prediction_updates_per_epoch = outer_iterations * prediction_steps_per_iteration
        total_prediction_updates = max(1, epochs * prediction_updates_per_epoch)
        total_warmup_prediction_updates = (
            total_prediction_updates
            if warmup_epochs <= 0
            else max(
                1,
                min(warmup_epochs, epochs) * prediction_updates_per_epoch,
            )
        )
        sim_iterator = None
        full_iterator = None

        for outer_iteration_idx in range(outer_iterations):
            for _ in range(effective_domain_steps_per_iteration):
                domain_batch, full_iterator = next_loader_batch(
                    train_loader_full,
                    full_iterator,
                )
                disc_loss_value, disc_confusion, disc_batch_size = run_discriminator_step(
                    model,
                    domain_batch,
                    device=device,
                    domain_class_weights=domain_class_weights,
                    domain_optimizer=domain_optimizer,
                )
                train_discriminator_sum += disc_loss_value * disc_batch_size
                train_discriminator_examples += disc_batch_size
                train_discriminator_confusion += disc_confusion

            for prediction_step_idx in range(prediction_steps_per_iteration):
                global_prediction_update = (
                    epoch * prediction_updates_per_epoch
                    + outer_iteration_idx * prediction_steps_per_iteration
                    + prediction_step_idx
                )
                progress = min(
                    global_prediction_update / max(1, total_warmup_prediction_updates - 1),
                    1.0,
                )
                current_grl_alpha = adaptation_rate * compute_grl_alpha(progress)
                model.set_grl_alpha(current_grl_alpha)
                sim_batch, sim_iterator = next_loader_batch(
                    train_loader_sim,
                    sim_iterator,
                )
                domain_batch, full_iterator = next_loader_batch(
                    train_loader_full,
                    full_iterator,
                )
                (
                    pred_loss_value,
                    adv_domain_loss_value,
                    adv_total_loss_value,
                    adv_confusion,
                    sim_batch_size,
                    adv_domain_batch_size,
                ) = run_generator_predictor_step(
                    model,
                    sim_batch,
                    domain_batch,
                    device=device,
                    prediction_optimizer=prediction_optimizer,
                    domain_optimizer=domain_optimizer,
                    domain_class_weights=domain_class_weights,
                    grl_alpha=current_grl_alpha,
                )
                train_pred_sum += pred_loss_value * sim_batch_size
                train_pred_examples += sim_batch_size
                train_adv_domain_sum += adv_domain_loss_value * adv_domain_batch_size
                train_adv_domain_examples += adv_domain_batch_size
                train_adv_total_sum += adv_total_loss_value
                train_adv_step_count += 1
                train_grl_alpha_sum += current_grl_alpha
                train_adv_domain_confusion += adv_confusion

        loss_pred = train_pred_sum / max(1, train_pred_examples)
        loss_class = train_discriminator_sum / max(1, train_discriminator_examples)
        loss_adv_domain = train_adv_domain_sum / max(1, train_adv_domain_examples)
        loss_adv_total = train_adv_total_sum / max(1, train_adv_step_count)
        current_grl_alpha = train_grl_alpha_sum / max(1, train_adv_step_count)
        loss = loss_pred + loss_class + loss_adv_domain
        train_discriminator_metrics = compute_binary_classification_metrics(
            train_discriminator_confusion
        )
        train_adv_domain_metrics = compute_binary_classification_metrics(
            train_adv_domain_confusion
        )
        train_discriminator_pass_equivalent = (
            train_discriminator_examples / max(1, len(train_set_full))
        )
        train_prediction_pass_equivalent = (
            train_pred_examples / max(1, len(train_set_sim))
        )
        train_adv_domain_pass_equivalent = (
            train_adv_domain_examples / max(1, len(train_set_full))
        )
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {loss:.6f} "
            f"(Pred: {loss_pred:.6f}, D: {loss_class:.6f}, "
            f"AdvDom: {loss_adv_domain:.6f}, AdvStep: {loss_adv_total:.6f}, "
            f"D Acc: {train_discriminator_metrics['accuracy']:.3f}, "
            f"D Bal: {train_discriminator_metrics['balanced_accuracy']:.3f}, "
            f"Adv Acc: {train_adv_domain_metrics['accuracy']:.3f}, "
            f"Adv Bal: {train_adv_domain_metrics['balanced_accuracy']:.3f}, "
            f"AvgGRL: {current_grl_alpha:.3f}, "
            f"k/l: {effective_domain_steps_per_iteration}/{prediction_steps_per_iteration})"
        )
        writer.add_scalar("train/total_loss", loss, epoch)
        writer.add_scalar("train/pred_loss", loss_pred, epoch)
        writer.add_scalar("train/discriminator_loss", loss_class, epoch)
        writer.add_scalar("train/adversarial_domain_loss", loss_adv_domain, epoch)
        writer.add_scalar("train/adversarial_step_loss", loss_adv_total, epoch)
        writer.add_scalar("train/grl_alpha", current_grl_alpha, epoch)
        writer.add_scalar(
            "train/domain_steps_per_iteration",
            effective_domain_steps_per_iteration,
            epoch,
        )
        writer.add_scalar(
            "train/prediction_steps_per_iteration",
            prediction_steps_per_iteration,
            epoch,
        )
        writer.add_scalar(
            "train/discriminator_accuracy",
            train_discriminator_metrics["accuracy"],
            epoch,
        )
        writer.add_scalar(
            "train/discriminator_balanced_accuracy",
            train_discriminator_metrics["balanced_accuracy"],
            epoch,
        )
        writer.add_scalar(
            "train/discriminator_recall_sim",
            train_discriminator_metrics["recall_sim"],
            epoch,
        )
        writer.add_scalar(
            "train/discriminator_recall_exp",
            train_discriminator_metrics["recall_exp"],
            epoch,
        )
        writer.add_scalar(
            "train/discriminator_predicted_exp_fraction",
            train_discriminator_metrics["predicted_exp_fraction"],
            epoch,
        )
        writer.add_scalar(
            "train/adversarial_domain_accuracy",
            train_adv_domain_metrics["accuracy"],
            epoch,
        )
        writer.add_scalar(
            "train/adversarial_domain_balanced_accuracy",
            train_adv_domain_metrics["balanced_accuracy"],
            epoch,
        )
        writer.add_scalar(
            "train/adversarial_domain_recall_sim",
            train_adv_domain_metrics["recall_sim"],
            epoch,
        )
        writer.add_scalar(
            "train/adversarial_domain_recall_exp",
            train_adv_domain_metrics["recall_exp"],
            epoch,
        )
        writer.add_scalar(
            "train/adversarial_domain_predicted_exp_fraction",
            train_adv_domain_metrics["predicted_exp_fraction"],
            epoch,
        )
        writer.add_scalar(
            "train/discriminator_pass_equivalent",
            train_discriminator_pass_equivalent,
            epoch,
        )
        writer.add_scalar(
            "train/prediction_pass_equivalent",
            train_prediction_pass_equivalent,
            epoch,
        )
        writer.add_scalar(
            "train/adversarial_domain_pass_equivalent",
            train_adv_domain_pass_equivalent,
            epoch,
        )
        writer.add_scalar("lr", prediction_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("lr/prediction", prediction_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("lr/domain", domain_optimizer.param_groups[0]["lr"], epoch)
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_pred = 0.0
        val_loss_class = 0.0
        val_mae = torch.zeros(3, device=device) # per-parameter MAE
        with torch.no_grad():
            for x, y_norm, y_raw, T, _ in val_loader_sim:
                x, y_norm, y_raw, T = x.to(device), y_norm.to(device), y_raw.to(device), T.to(device)
                model.on_pred_mode().off_class_mode()
                xpcs_features, temp_features = model.extract_features(x, T)
                shared_features = model.build_shared_features(xpcs_features, temp_features)
                pred_logits = model.forward_predictor_from_shared_features(
                    shared_features,
                    return_logits=True,
                )
                pred_loss, _, pred_params = compute_prediction_loss(
                    pred_logits,
                    y_norm,
                    model.predictor_output_activation,
                )
                pred_params_raw = denorm_from_meta(pred_params, norm_meta, device=device)
                val_mae += (pred_params_raw - y_raw).abs().sum(dim=0)
                val_loss += pred_loss.item() * x.size(0)
                val_loss_pred += pred_loss.item() * x.size(0)
        val_loss_class, val_domain_confusion = run_domain_pass(
            model,
            val_loader_full,
            device=device,
            domain_class_weights=domain_class_weights,
            grl_alpha=0.0,
            domain_optimizer=None,
            encoder_optimizer=None,
        )
        val_loss += val_loss_class * len(val_set_full)
        val_loss /= len(val_set_sim) + len(val_set_full)
        val_loss_pred /= len(val_set_sim)
        val_mae /= len(val_set_sim)
        val_selection_loss = val_loss_pred - adaptation_rate * val_loss_class
        val_domain_metrics = compute_binary_classification_metrics(val_domain_confusion)
        print(
            f"Val Loss: {val_loss:.6f} "
            f"(Pred: {val_loss_pred:.6f}, Class: {val_loss_class:.6f}, "
            f"Select: {val_selection_loss:.6f}, "
            f"Domain Acc: {val_domain_metrics['accuracy']:.3f}, "
            f"Bal: {val_domain_metrics['balanced_accuracy']:.3f}, "
            f"PredExp: {val_domain_metrics['predicted_exp_fraction']:.3f}) "
            f"Per-parameter MAE: "
            f"gamma: {val_mae[0]:.4e}, D: {val_mae[1]:.4e}, GB_conc: {val_mae[2]:.4e}"
        )
        writer.add_scalar("val/total_loss", val_loss, epoch)
        writer.add_scalar("val/selection_loss", val_selection_loss, epoch)
        writer.add_scalar("val/domain_accuracy", val_domain_metrics["accuracy"], epoch)
        writer.add_scalar(
            "val/domain_balanced_accuracy",
            val_domain_metrics["balanced_accuracy"],
            epoch,
        )
        writer.add_scalar("val/domain_recall_sim", val_domain_metrics["recall_sim"], epoch)
        writer.add_scalar("val/domain_recall_exp", val_domain_metrics["recall_exp"], epoch)
        writer.add_scalar(
            "val/domain_predicted_exp_fraction",
            val_domain_metrics["predicted_exp_fraction"],
            epoch,
        )
        for i, name in enumerate(["gamma", "D", "GB_conc"]):
            writer.add_scalar(f"mae_raw/{name}", float(val_mae[i]), epoch)

        # checkpoint best
        if val_selection_loss < best_val_loss - 1e-6:
            best_val_loss = val_selection_loss
            bad_epochs = 0
            model_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path / f"XPCS_no_T_best_{stamp}.pt")
            print(f"Saved best checkpoint (val {best_val_loss:.4f}) -> XPCS_no_T_best_{stamp}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val_loss:.4f})")
                break
        prediction_scheduler.step()
        
    # Fetch the best model for testing
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    best_model = XPCSNet(**model.get_architecture_config())
    load_matching_state_dict(
        best_model,
        torch.load(
            model_path / f"XPCS_no_T_best_{stamp}.pt",
            weights_only=True,
            map_location=device,
        ),
    )
    best_model = best_model.to(device)
    best_model.eval()
    best_model.set_grl_alpha(1.0)
    test_loss = 0.0
    test_loss_pred = 0.0
    test_loss_class = 0.0
    test_mae = torch.zeros(3, device=device)
    with torch.no_grad():
        for x, y_norm, y_raw, T, _ in test_loader_sim:
            x, y_norm, y_raw, T = x.to(device), y_norm.to(device), y_raw.to(device), T.to(device)
            best_model.on_pred_mode().off_class_mode()
            xpcs_features, temp_features = best_model.extract_features(x, T)
            shared_features = best_model.build_shared_features(xpcs_features, temp_features)
            pred_logits = best_model.forward_predictor_from_shared_features(
                shared_features,
                return_logits=True,
            )
            pred_loss, _, pred_params = compute_prediction_loss(
                pred_logits,
                y_norm,
                best_model.predictor_output_activation,
            )
            pred_params_raw = denorm_from_meta(pred_params, norm_meta, device=device)
            test_mae += (pred_params_raw - y_raw).abs().sum(dim=0)
            test_loss += pred_loss.item() * x.size(0)
            test_loss_pred += pred_loss.item() * x.size(0)
    test_loss_class, test_domain_confusion = run_domain_pass(
        best_model,
        test_loader_full,
        device=device,
        domain_class_weights=domain_class_weights,
        grl_alpha=0.0,
        domain_optimizer=None,
        encoder_optimizer=None,
    )
    test_loss += test_loss_class * len(test_set_full)
    test_loss /= len(test_set_sim) + len(test_set_full)
    test_loss_pred /= len(test_set_sim)
    test_mae /= len(test_set_sim)
    test_domain_metrics = compute_binary_classification_metrics(test_domain_confusion)
    
    for i, name in enumerate(full_dataset.PARAM_KEYS):
        print(f"Test MAE [{name}]: {test_mae[i].item():.3e}")
    print(
        "Test Domain Metrics: "
        f"acc={test_domain_metrics['accuracy']:.3f}, "
        f"bal={test_domain_metrics['balanced_accuracy']:.3f}, "
        f"recall_sim={test_domain_metrics['recall_sim']:.3f}, "
        f"recall_exp={test_domain_metrics['recall_exp']:.3f}, "
        f"pred_exp={test_domain_metrics['predicted_exp_fraction']:.3f}"
    )
    writer.add_scalar("test/domain_accuracy", test_domain_metrics["accuracy"], 0)
    writer.add_scalar(
        "test/domain_balanced_accuracy",
        test_domain_metrics["balanced_accuracy"],
        0,
    )
    writer.add_scalar("test/domain_recall_sim", test_domain_metrics["recall_sim"], 0)
    writer.add_scalar("test/domain_recall_exp", test_domain_metrics["recall_exp"], 0)
    writer.add_scalar(
        "test/domain_predicted_exp_fraction",
        test_domain_metrics["predicted_exp_fraction"],
        0,
    )

    probe_metrics = run_domain_probe(
        best_model,
        train_dataset=train_set_full,
        val_dataset=val_set_full,
        test_dataset=test_set_full,
        device=device,
        seed=seed,
    )
    if probe_metrics is None:
        print("[probe] scikit-learn not available; skipping frozen domain probe")
    else:
        print(
            "[probe] Frozen-feature domain probe: "
            f"train acc/bal={probe_metrics['train_accuracy']:.3f}/{probe_metrics['train_balanced_accuracy']:.3f}, "
            f"val acc/bal={probe_metrics['val_accuracy']:.3f}/{probe_metrics['val_balanced_accuracy']:.3f}, "
            f"test acc/bal={probe_metrics['test_accuracy']:.3f}/{probe_metrics['test_balanced_accuracy']:.3f}"
        )
        for key, value in probe_metrics.items():
            writer.add_scalar(f"probe/{key}", value, 0)

    training_metadata["final_domain_metrics"] = {
        "test_accuracy": test_domain_metrics["accuracy"],
        "test_balanced_accuracy": test_domain_metrics["balanced_accuracy"],
        "test_recall_sim": test_domain_metrics["recall_sim"],
        "test_recall_exp": test_domain_metrics["recall_exp"],
        "test_predicted_exp_fraction": test_domain_metrics["predicted_exp_fraction"],
    }
    if probe_metrics is not None:
        training_metadata["domain_probe"] = probe_metrics
    save_training_metadata(metadata_path, training_metadata)
    save_training_metadata(checkpoint_metadata_path, training_metadata)
        
    writer.close()
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")
    print("Training complete.")
    
    best_model.on_pred_mode().off_class_mode().set_grl_alpha(1.0)
    return best_model

@torch.no_grad()
def inference_sim(
    model: XPCSNet,
    indices: Optional[List[int]] = None,
    sim_root: Path = Path("dataset/simulation"),
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> pd.DataFrame:
    """
    Perform inference on simulated data using the trained model.
    
    Args:
        model: The trained XPCSNet model.
        indices: Optional list of dataset indices to infer. If None, infer the entire dataset.
        sim_root: Path to the simulated dataset root directory.
        device: Device to perform inference on (CPU or GPU).
        
    Returns:
        results_df: A DataFrame containing the inferred parameters for each sample, 
            along with the ground truth parameters.
    """
    sim_dataset = XPCSDataset(sim_root)
    norm_meta = sim_dataset.norm_meta
    if indices is not None:
        sim_dataset = Subset(sim_dataset, indices)
    
    model = model.to(device)
    model.eval()
    model.on_pred_mode().off_class_mode().set_grl_alpha(1.0)
    results = []
    
    for i in range(len(sim_dataset)):
        x, _, y_raw, T, _ = sim_dataset[i]
        x = x.unsqueeze(0).to(device)
        T = T.unsqueeze(0).to(device)
        pred_params_norm = model(x, T)      # (1, 3)
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

    results_df = pd.DataFrame(results)
    return results_df
    
@torch.no_grad()
def inference_exp(
    model: XPCSNet,
    exp_root: Path = Path("exp_data"),
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    select_batches: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Perform inference on raw experimental data using the trained model.
    All the experimental data are located in the `exp_root` directory, stored as .npy files
    or .npz files (the data key being 'g12'). Each file contains multiple batches (size, 
    size, B). The function returns a dictionary of DataFrames, each containing the inferred 
    parameters for every batch in the corresponding file.
    
    If `select_batches` is provided, only those batch indices will be inferred.
    
    Args:
        model: The trained XPCSNet model.
        exp_root: Path to the experimental dataset root directory.
        device: Device to perform inference on (CPU or GPU).
        select_batches: Optional list of batch indices to infer. If None, infer all batches.
        
    Returns:
        results_dfs: A dictionary where each key is a filename and each value is a DataFrame
            containing the inferred parameters for each batch in that file.
    """
    model = model.to(device)
    model.eval()
    model.on_pred_mode().off_class_mode().set_grl_alpha(1.0)
    results_dfs = {}
    
    for data_file in sorted(exp_root.glob("*")):
        if data_file.suffix not in [".npy", ".npz"]:
            continue
        if data_file.suffix == ".npy":
            data = np.load(data_file)
        else:
            data = np.load(data_file)['g12']
        T = 273.15 + float(str(data_file.name).split("T")[-1].split("C")[0])
        data_tensor = torch.tensor(data, dtype=torch.float32)  # (size, size, B)
        results = []
        for i in range(data_tensor.size(-1)):
            if select_batches is not None and i not in select_batches:
                continue
            x = data_tensor[:2500, :2500, i]  # (2500, 2500)
            x = coarse_grain_g2(x, 256).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 256, 256)
            x = (x - INPUT_MEAN) / (INPUT_STD + 1e-6)
            T = torch.tensor([[T]], dtype=torch.float32).to(device) # (1, 1)
            pred_params_norm = model(x, T)      # (1, 3)
            pred_params_raw = denorm_from_meta(pred_params_norm.squeeze(0), model.norm_meta, device=device)
            results.append({
                "T": T.item(),
                "gamma": pred_params_raw[0].item(),
                "D": pred_params_raw[1].item(),
                "GB_conc": pred_params_raw[2].item(),
            })
        results_df = pd.DataFrame(results)
        results_dfs[data_file.stem] = results_df
    
    return results_dfs


def build_model_from_checkpoint_metadata(model_path: Path) -> XPCSNet:
    """
    Reconstruct the saved adversarial architecture using the sibling metadata
    JSON when it is available.

    Args:
        model_path: Path to a checkpoint file.

    Returns:
        model: Newly instantiated XPCSNet with matching architecture knobs.
    """
    metadata_path = model_path.with_suffix(".json")
    architecture_config: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="ascii") as f:
            metadata = json.load(f)
        architecture_config = dict(metadata.get("architecture") or {})
        if architecture_config:
            print(
                "[load] Reconstructing adversarial no-T architecture from "
                f"{metadata_path.name}"
            )
    return XPCSNet(**architecture_config)

def load_model(
    model_path: Optional[Path] = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> XPCSNet:
    """
    Load a trained XPCSNet model from the specified path. If no path is provided,
    load the most recent model from the "models" directory. The model name is expected
    to follow the format "XPCS_no_T_best_{timestamp}.pt", where {timestamp} is a datetime string
    in the format "%Y%m%d-%H%M%S".
    
    Args:
        model_path: Path to the trained model file. If None, load the most recent model.
        device: Device to load the model onto (CPU or GPU).
        
    Returns:
        model: The loaded XPCSNet model.
    """
    if model_path is None:
        model_dir = Path("models")
        model_files = list(model_dir.glob("XPCS_no_T_best_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        # Sort by timestamp in filename
        model_files.sort(key=lambda x: x.stem.split("_")[-1], reverse=True)
        model_path = model_files[0]
        print(f"No model path specified, loading the most recent model: {model_path}")
    else:
        model_path = Path(model_path)
        print(f"Loading the model: {model_path}")
        
    model = build_model_from_checkpoint_metadata(model_path)
    load_summary = load_matching_state_dict(
        model,
        torch.load(model_path, weights_only=True, map_location=device),
    )
    if load_summary["skipped"]:
        print(
            "[load] Skipped incompatible checkpoint tensors: "
            f"{len(load_summary['skipped'])}"
        )
    model = model.to(device)
    model.eval()
    model.on_pred_mode().off_class_mode().set_grl_alpha(1.0)
    
    return model

if __name__ == "__main__":
    set_global_seed(RANDOM_SEED, deterministic=True)
    model = XPCSNet(predictor_output_activation="sigmoid")
    best_model = train(
        model,
        sim_root=Path("dataset/simulation"),
        exp_root=Path("dataset/experiment"),
        adaptation_rate=1.2,
        seed=RANDOM_SEED,
        deterministic=True,
    )
    # best_model = load_model(model_path="models/XPCS_no_T_best_20251114-000648.pt")
    # results = inference_exp(best_model)
    # combined_df = pd.DataFrame()
    # for filename, df in results.items():
    #     print(f"Results for {filename}:")
    #     print(df)
    #     df['name'] = filename
    #     combined_df = pd.concat([combined_df, df], ignore_index=True)
    # combined_df.to_csv("inference_results.csv", index=False)
    # results_df = inference_sim(
    #     best_model,
    #     indices=np.random.choice(2000, size=10, replace=False).tolist(),
    #     sim_root=Path("dataset/simulation"),
    # )
    # print(results_df)
    # results_df.to_csv("inference_sim_results.csv", index=False)
    
