import json
import os
import random
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
from produce_data import coarse_grain_g2

# --- Split configuration ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
INPUT_SIZE = 256
INPUT_MEAN = 1.1315594972968084
INPUT_STD = 0.011241361147397289
PREDICTION_SMOOTH_L1_WEIGHT = 0.25

"""
The goal is to provide a comparison baseline using a vanilla CNN model without
domain adaptation. The code structure is similar to train_adv_no_T.py but without the
domain discriminator and adversarial training.
The network is comprised of only two parts:
1. A feature extractor (CNN) that extracts features from input data.
2. A predictor that predicts the gamma, D and GB_conc from the extracted features.
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
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
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


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and Torch so retraining can be reproduced exactly.

    Args:
        seed: Base random seed.
        deterministic: Whether to prefer deterministic Torch/CUDA kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by a subset of deterministic CUDA kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def seed_dataloader_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker from Torch's per-worker seed.

    Args:
        worker_id: PyTorch worker index. Included for the worker-init signature.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def save_training_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    """
    Write reproducibility metadata alongside a run or checkpoint.

    Args:
        path: JSON output path.
        metadata: Serializable metadata dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def summarize_parameter_losses(
    component_losses: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Summarize per-parameter loss magnitudes and their relative shares.

    Args:
        component_losses: Tensor of shape `[3]` containing the epoch-averaged
            normalized MSE for `(gamma, D, GB_conc)`.
        eps: Small constant to avoid division by zero.

    Returns:
        component_losses: The input tensor, preserved for convenience.
        component_shares: Relative contribution of each parameter to the sum of
            component losses. These shares sum to one.
    """
    total_component_loss = torch.clamp(component_losses.sum(), min=eps)
    component_shares = component_losses / total_component_loss
    return component_losses, component_shares


def compute_prediction_loss(
    pred_logits: torch.Tensor,
    y_norm: torch.Tensor,
    predictor_output_activation: str,
    smooth_l1_weight: float = PREDICTION_SMOOTH_L1_WEIGHT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the supervised predictor loss from raw predictor logits.

    When the predictor is configured with sigmoid-bounded outputs, train on the
    logits with BCE-with-logits plus a smaller SmoothL1 term on the bounded
    predictions. This keeps outputs in range while penalizing confidently wrong
    saturated predictions more strongly than plain MSE.

    Args:
        pred_logits: Raw predictor outputs of shape `[B, 3]`.
        y_norm: Normalized targets in `[0, 1]` with shape `[B, 3]`.
        predictor_output_activation: Predictor output activation mode.
        smooth_l1_weight: Relative weight of the bounded SmoothL1 term.

    Returns:
        total_loss: Mean scalar predictor loss.
        component_loss: Per-parameter loss vector of shape `[3]`.
        pred_params: Bounded predictions used for metrics and denormalization.
    """
    if predictor_output_activation == "sigmoid":
        pred_params = torch.sigmoid(pred_logits)
        bce_components = F.binary_cross_entropy_with_logits(
            pred_logits,
            y_norm,
            reduction="none",
        ).mean(dim=0)
        smooth_l1_components = F.smooth_l1_loss(
            pred_params,
            y_norm,
            reduction="none",
        ).mean(dim=0)
        component_loss = bce_components + smooth_l1_weight * smooth_l1_components
    else:
        pred_params = pred_logits
        component_loss = F.mse_loss(
            pred_params,
            y_norm,
            reduction="none",
        ).mean(dim=0)
    total_loss = component_loss.mean()
    return total_loss, component_loss, pred_params

# --- Dataset ---

class XPCSDataset(Dataset):
    """
    Loads samples from dataset based on either one manifest.csv and stats.json, 
    or multiple manifests. Splits are dynamically handled.
    Returns (x_norm, y_norm, y_raw, T, label) where:
      - x_norm: torch.FloatTensor [1, 256, 256], normalized using train mean/std
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

        self.mean = INPUT_MEAN
        self.std = INPUT_STD

        self.norm_meta = {
            "gamma": {"low": 2e18, "high": 5e18, "scale": "linear"},
            "D": {"low": 1e-23, "high": 1e-21, "scale": "log"},
            "GB_conc": {"low": 0.0, "high": 0.3, "scale": "linear"},
            "T": {"low": 300, "high": 500, "scale": "linear"}
        }

        # precompute a [1, 256, 256] diagonal mask (zeros on diag)
        # so the network never sees diagonal pixels
        self.diag_mask = torch.ones(1, 256, 256, dtype=torch.float32)
        self.diag_mask[0, range(256), range(256)] = 0.0

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        x = torch.load(row["path"], weights_only=True).to(torch.float32).squeeze(0)  # [1, 256, 256]
        x = (x - self.mean) / (self.std + 1e-6)
        x = x * self.diag_mask.to(x.device)

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
class VanillaXPCSNet(nn.Module):
    """
    Implement the XPCSNet model *without* domain adaptation. 
    The input XPCS prediction shape: (B, 1, 256, 256).
    
    If prediction_mode is True, return the predicted parameters (gamma, D, 
    GB_conc). If classification_mode is True, return the domain classification output.
    """
    
    def __init__(self, predictor_output_activation: str = "linear"):
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
        self.predictor_output_activation = predictor_output_activation
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

    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Export the architecture knobs needed to reconstruct this model.

        Returns:
            config: Keyword arguments that can be passed back into
                `VanillaXPCSNet`.
        """
        return {
            "predictor_output_activation": self.predictor_output_activation,
        }

    def extract_features(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Extract the XPCS feature branch and ignore explicit temperature input.
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
        Keep the no-temperature shared representation equal to the XPCS branch.
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
        """
        pred_logits = self.xpcs_predictor(shared_features)
        if return_logits:
            return pred_logits
        if self.predictor_output_activation == "sigmoid":
            return torch.sigmoid(pred_logits)
        return pred_logits

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
            predicted_params: Predicted normalized parameters of shape (B, 3)
        """
        xpcs_features, temp_features = self.extract_features(x, T)
        shared_features = self.build_shared_features(xpcs_features, temp_features)
        predicted_params = self.forward_predictor_from_shared_features(
            shared_features
        )
        return predicted_params

# --- Training Loop ---

def train(
    model: VanillaXPCSNet,
    sim_root: Path = Path("dataset"),
    # exp_root: Path = Path("dataset"),
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    seed: int = RANDOM_SEED,
    deterministic: bool = True,
    num_workers: int = 0,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    log_pardir: Path = Path("runs"),
    model_path: Path = Path("models"), 
) -> VanillaXPCSNet:
    """
    Train the XPCSNet model without domain adaptation.
    
    Args:
        model: The XPCSNet model to be trained.
        sim_root: Path to the simulated dataset root directory.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        seed: Global random seed used for initialization, splits, and shuffling.
        deterministic: Whether to enforce deterministic Torch behavior.
        num_workers: Number of DataLoader workers.
        device: Device to perform training on (CPU or GPU).
        log_pardir: Parent directory for TensorBoard logs.
        model_path: Directory to save the best model checkpoints.
        
    Returns:
        best_model: The best-performing Vanilla XPCSNet model on the validation set.
    """   
    set_global_seed(seed, deterministic=deterministic)

    # Set up logging
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = log_pardir / f"vanilla_xpcs_no_T_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[logger] TensorBoard log dir: {log_dir}")
    print(f"[seed] Vanilla training seed: {seed} (deterministic={deterministic})")
    
    # Load datasets
    sim_dataset = XPCSDataset(sim_root)
    norm_meta = sim_dataset.norm_meta
    
    # Create splits and loaders
    train_indices_sim, val_indices_sim, test_indices_sim = create_random_splits(
        len(sim_dataset),
        seed=seed,
    )
    train_set_sim = Subset(sim_dataset, train_indices_sim)
    val_set_sim = Subset(sim_dataset, val_indices_sim)
    test_set_sim = Subset(sim_dataset, test_indices_sim)

    sim_manifest = sim_dataset.manifest
    training_metadata: Dict[str, Any] = {
        "timestamp": stamp,
        "seed": seed,
        "deterministic": deterministic,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dataset_root": str(sim_root),
        "dataset_size": len(sim_dataset),
        "input_normalization": "global_zscore",
        "input_mean": sim_dataset.mean,
        "input_std": sim_dataset.std,
        "train_indices": train_indices_sim.tolist(),
        "val_indices": val_indices_sim.tolist(),
        "test_indices": test_indices_sim.tolist(),
        "architecture": model.get_architecture_config(),
    }
    if "id" in sim_manifest.columns:
        training_metadata["train_ids"] = sim_manifest.iloc[train_indices_sim]["id"].astype(int).tolist()
        training_metadata["val_ids"] = sim_manifest.iloc[val_indices_sim]["id"].astype(int).tolist()
        training_metadata["test_ids"] = sim_manifest.iloc[test_indices_sim]["id"].astype(int).tolist()

    metadata_path = log_dir / "training_metadata.json"
    checkpoint_metadata_path = model_path / f"Vanilla_XPCS_no_T_best_{stamp}.json"
    save_training_metadata(metadata_path, training_metadata)
    save_training_metadata(checkpoint_metadata_path, training_metadata)
    print(f"[repro] Saved training metadata -> {metadata_path}")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_dataloader_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader_sim = DataLoader(
        train_set_sim,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader_sim = DataLoader(
        val_set_sim,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader_sim = DataLoader(
        test_set_sim,
        shuffle=False,
        **loader_kwargs,
    )

    # Set up optimizer and scheduler
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    patience = 15
    bad_epochs = 0
    
    # Training loop
    stopped_epoch = epochs
    parameter_names = sim_dataset.PARAM_KEYS
    for epoch in range(epochs):
        model.train()
        loss = 0.0
        train_component_loss = torch.zeros(3, device=device)
        for x, y_norm, _, T, _ in train_loader_sim:
            x, y_norm, T = x.to(device), y_norm.to(device), T.to(device)
            optimizer.zero_grad()
            xpcs_features, temp_features = model.extract_features(x, T)
            shared_features = model.build_shared_features(xpcs_features, temp_features)
            pred_logits = model.forward_predictor_from_shared_features(
                shared_features,
                return_logits=True,
            )
            pred_loss, per_parameter_loss, _ = compute_prediction_loss(
                pred_logits,
                y_norm,
                model.predictor_output_activation,
            )
            pred_loss.backward()
            optimizer.step()
            batch_size_actual = x.size(0)
            loss += pred_loss.item() * batch_size_actual
            train_component_loss += per_parameter_loss.detach() * batch_size_actual
        loss /= len(train_set_sim)
        train_component_loss /= len(train_set_sim)
        _, train_component_share = summarize_parameter_losses(train_component_loss)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {loss:.6f} "
            f"| Component Loss: "
            f"gamma={train_component_loss[0]:.6f}, "
            f"D={train_component_loss[1]:.6f}, "
            f"GB_conc={train_component_loss[2]:.6f} "
            f"| Share: "
            f"gamma={100.0 * train_component_share[0]:.1f}%, "
            f"D={100.0 * train_component_share[1]:.1f}%, "
            f"GB_conc={100.0 * train_component_share[2]:.1f}%"
        )
        writer.add_scalar("train/total_loss", loss, epoch)
        for i, name in enumerate(parameter_names):
            writer.add_scalar(
                f"train/loss_components/{name}",
                float(train_component_loss[i]),
                epoch,
            )
            writer.add_scalar(
                f"train/loss_shares/{name}",
                float(train_component_share[i]),
                epoch,
            )
        # Validation
        model.eval()
        val_loss = 0.0
        val_component_loss = torch.zeros(3, device=device)
        val_mae = torch.zeros(3, device=device) # per-parameter MAE
        with torch.no_grad():
            for x, y_norm, y_raw, T, _ in val_loader_sim:
                x, y_norm, y_raw, T = x.to(device), y_norm.to(device), y_raw.to(device), T.to(device)
                xpcs_features, temp_features = model.extract_features(x, T)
                shared_features = model.build_shared_features(xpcs_features, temp_features)
                pred_logits = model.forward_predictor_from_shared_features(
                    shared_features,
                    return_logits=True,
                )
                pred_loss, per_parameter_loss, pred_params = compute_prediction_loss(
                    pred_logits,
                    y_norm,
                    model.predictor_output_activation,
                )
                pred_params_raw = denorm_from_meta(pred_params, norm_meta, device=device)
                batch_size_actual = x.size(0)
                val_mae += (pred_params_raw - y_raw).abs().sum(dim=0)
                val_loss += pred_loss.item() * batch_size_actual
                val_component_loss += per_parameter_loss * batch_size_actual
        val_loss /= len(val_set_sim)
        val_component_loss /= len(val_set_sim)
        val_mae /= len(val_set_sim)
        _, val_component_share = summarize_parameter_losses(val_component_loss)
        print(
            f"Val Loss: {val_loss:.6f} "
            f"| Component Loss: "
            f"gamma={val_component_loss[0]:.6f}, "
            f"D={val_component_loss[1]:.6f}, "
            f"GB_conc={val_component_loss[2]:.6f} "
            f"| Share: "
            f"gamma={100.0 * val_component_share[0]:.1f}%, "
            f"D={100.0 * val_component_share[1]:.1f}%, "
            f"GB_conc={100.0 * val_component_share[2]:.1f}% "
            f"| "
            f"Per-parameter MAE: "
            f"gamma: {val_mae[0]:.4e}, D: {val_mae[1]:.4e}, GB_conc: {val_mae[2]:.4e}"
        )
        writer.add_scalar("val/total_loss", val_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        for i, name in enumerate(parameter_names):
            writer.add_scalar(
                f"val/loss_components/{name}",
                float(val_component_loss[i]),
                epoch,
            )
            writer.add_scalar(
                f"val/loss_shares/{name}",
                float(val_component_share[i]),
                epoch,
            )
            writer.add_scalar(f"mae_raw/{name}", float(val_mae[i]), epoch)

        # checkpoint best
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            model_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path / f"Vanilla_XPCS_no_T_best_{stamp}.pt")
            print(f"Saved best checkpoint (val {best_val_loss:.4f}) -> Vanilla_XPCS_no_T_best_{stamp}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val_loss:.4f})")
                stopped_epoch = epoch + 1
                break
        scheduler.step()
    else:
        stopped_epoch = epochs

    training_metadata.update({
        "completed_epochs": stopped_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(model_path / f"Vanilla_XPCS_no_T_best_{stamp}.pt"),
    })
    save_training_metadata(metadata_path, training_metadata)
    save_training_metadata(checkpoint_metadata_path, training_metadata)
        
    # Fetch the best model for testing
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    best_model = VanillaXPCSNet(**model.get_architecture_config())
    best_model.load_state_dict(
        torch.load(
            model_path / f"Vanilla_XPCS_no_T_best_{stamp}.pt",
            weights_only=True,
            map_location=device,
        )
    )
    best_model = best_model.to(device)
    best_model.eval()
    test_loss = 0.0
    test_component_loss = torch.zeros(3, device=device)
    test_mae = torch.zeros(3, device=device)
    with torch.no_grad():
        for x, y_norm, y_raw, T, _ in test_loader_sim:
            x, y_norm, y_raw, T = x.to(device), y_norm.to(device), y_raw.to(device), T.to(device)
            xpcs_features, temp_features = best_model.extract_features(x, T)
            shared_features = best_model.build_shared_features(xpcs_features, temp_features)
            pred_logits = best_model.forward_predictor_from_shared_features(
                shared_features,
                return_logits=True,
            )
            pred_loss, per_parameter_loss, pred_params = compute_prediction_loss(
                pred_logits,
                y_norm,
                best_model.predictor_output_activation,
            )
            pred_params_raw = denorm_from_meta(pred_params, norm_meta, device=device)
            batch_size_actual = x.size(0)
            test_mae += (pred_params_raw - y_raw).abs().sum(dim=0)
            test_loss += pred_loss.item() * batch_size_actual
            test_component_loss += per_parameter_loss * batch_size_actual
    test_loss /= len(test_set_sim)
    test_component_loss /= len(test_set_sim)
    test_mae /= len(test_set_sim)
    _, test_component_share = summarize_parameter_losses(test_component_loss)
    print(
        f"Test Loss: {test_loss:.6f} "
        f"| Component Loss: "
        f"gamma={test_component_loss[0]:.6f}, "
        f"D={test_component_loss[1]:.6f}, "
        f"GB_conc={test_component_loss[2]:.6f} "
        f"| Share: "
        f"gamma={100.0 * test_component_share[0]:.1f}%, "
        f"D={100.0 * test_component_share[1]:.1f}%, "
        f"GB_conc={100.0 * test_component_share[2]:.1f}%"
    )
    
    for i, name in enumerate(parameter_names):
        print(f"Test MAE [{name}]: {test_mae[i].item():.3e}")
        
    writer.close()
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")
    print("Training complete.")
    
    return best_model

@torch.no_grad()
def inference_sim(
    model: VanillaXPCSNet,
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
    model: VanillaXPCSNet,
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


def build_model_from_checkpoint_metadata(model_path: Path) -> VanillaXPCSNet:
    """
    Reconstruct the saved vanilla architecture using the sibling metadata JSON
    when it is available.

    Args:
        model_path: Path to a checkpoint file.

    Returns:
        model: Newly instantiated VanillaXPCSNet with matching architecture
            knobs.
    """
    architecture_config: Dict[str, Any] = {}
    metadata_path = model_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="ascii") as f:
            metadata = json.load(f)
        architecture_config = dict(metadata.get("architecture") or {})
        if architecture_config:
            print(
                "[load] Reconstructing vanilla no-T architecture from "
                f"{metadata_path.name}"
            )
    return VanillaXPCSNet(**architecture_config)

def load_model(
    model_path: Optional[Path] = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> VanillaXPCSNet:
    """
    Load a trained XPCSNet model from the specified path. If no path is provided,
    load the most recent model from the "models" directory. The model name is expected
    to follow the format "Vanilla_XPCS_no_T_best_{timestamp}.pt", where {timestamp} is a datetime 
    string in the format "%Y%m%d-%H%M%S".
    
    Args:
        model_path: Path to the trained model file. If None, load the most recent model.
        device: Device to load the model onto (CPU or GPU).
        
    Returns:
        model: The loaded XPCSNet model.
    """
    if model_path is None:
        model_dir = Path("models")
        model_files = list(model_dir.glob("Vanilla_XPCS_no_T_best_*.pt"))
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
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    set_global_seed(RANDOM_SEED, deterministic=True)
    model = VanillaXPCSNet(predictor_output_activation="sigmoid")
    best_model = train(
        model,
        sim_root=Path("dataset/simulation"),
        learning_rate=1e-3,
    )
    # best_model = load_model(model_path="models/Vanilla_XPCS_no_T_best_20251120-232700.pt")
    results = inference_exp(best_model)
    combined_df = pd.DataFrame()
    for filename, df in results.items():
        print(f"Results for {filename}:")
        print(df)
        df['name'] = filename
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df.to_csv("inference_results_vanilla_no_T.csv", index=False)
    results_df = inference_sim(
        best_model,
        indices=np.random.choice(2000, size=10, replace=False).tolist(),
        sim_root=Path("dataset/simulation"),
    )
    print(results_df)
    results_df.to_csv("inference_sim_results_vanilla_no_T.csv", index=False)
    
    
