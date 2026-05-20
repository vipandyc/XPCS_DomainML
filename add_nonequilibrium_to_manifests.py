from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


_DIAGONAL_CACHE: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def diagonal_cache(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, device)
    cached = _DIAGONAL_CACHE.get(key)
    if cached is not None:
        return cached

    row = torch.arange(n, device=device).reshape(n, 1)
    col = torch.arange(n, device=device).reshape(1, n)
    diagonal_ids = col - row + n - 1
    counts = torch.bincount(diagonal_ids.reshape(-1), minlength=2 * n - 1).to(
        torch.float32
    )
    cached = (diagonal_ids.to(torch.long), counts)
    _DIAGONAL_CACHE[key] = cached
    return cached


def avg_diagonal(g2: torch.Tensor) -> torch.Tensor:
    if g2.dim() == 3 and g2.shape[0] == 1:
        g2 = g2.squeeze(0)
    n = g2.shape[0]
    diagonal_ids, counts = diagonal_cache(n, g2.device)
    diagonal_sums = torch.zeros(2 * n - 1, dtype=torch.float32, device=g2.device)
    diagonal_sums.scatter_add_(0, diagonal_ids.reshape(-1), g2.reshape(-1).to(torch.float32))
    diagonal_means = diagonal_sums / counts.to(g2.device)
    return diagonal_means[diagonal_ids]


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


def find_neighbor_noneq_values(dataset_root: Path) -> dict[str, float]:
    candidates = [
        dataset_root / "manifest_with_non_equ.csv",
        dataset_root / "manifest_with_non_equ_1.csv",
        dataset_root / "manifest_with_unequ.csv",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)
        column = None
        if "nonequilibrium_measure" in df.columns:
            column = "nonequilibrium_measure"
        elif "unequilibrium_measure" in df.columns:
            column = "unequilibrium_measure"
        if column is None or "path" not in df.columns:
            continue
        return {
            str(path): float(value)
            for path, value in zip(df["path"], df[column])
            if pd.notna(value)
        }
    return {}


def compute_manifest_raw_noneq(
    manifest_path: Path,
    prefer_existing: bool = True,
    progress_interval: int = 50,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    dataset_root = manifest_path.parent
    existing_by_path = (
        find_neighbor_noneq_values(dataset_root) if prefer_existing else {}
    )
    raw_values: list[float] = []

    total = len(manifest)
    for index, row in manifest.iterrows():
        sample_path = str(row["path"])
        if sample_path in existing_by_path:
            raw_values.append(existing_by_path[sample_path])
        else:
            g2 = torch.load(sample_path, map_location="cpu", weights_only=True)
            raw_values.append(compute_nonequilibrium_measure(g2))

        processed = index + 1
        if progress_interval > 0 and (
            processed == total or processed % progress_interval == 0
        ):
            print(f"[noneq] {manifest_path}: {processed}/{total}", flush=True)

    manifest["nonequilibrium_measure_raw"] = raw_values
    return manifest


def normalize_jointly(frames: Iterable[pd.DataFrame]) -> tuple[list[pd.DataFrame], float, float]:
    frames = [frame.copy() for frame in frames]
    all_values = pd.concat(
        [frame["nonequilibrium_measure_raw"] for frame in frames],
        ignore_index=True,
    )
    min_value = float(all_values.min())
    max_value = float(all_values.max())
    scale = max(max_value - min_value, 1e-12)
    for frame in frames:
        normalized = (frame["nonequilibrium_measure_raw"] - min_value) / scale
        frame["nonequilibrium_measure"] = normalized.clip(0.0, 1.0)
    return frames, min_value, max_value


def default_manifest_paths() -> list[Path]:
    candidates = [
        Path("dataset/simulation/manifest.csv"),
        Path("dataset/experiment/manifest.csv"),
        Path("dataset/experiment_sub/manifest.csv"),
    ]
    return [path for path in candidates if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add raw and jointly normalized nonequilibrium_measure columns to "
            "dataset manifest CSV files."
        )
    )
    parser.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        default=default_manifest_paths(),
        help="Manifest CSV files to update in-place.",
    )
    parser.add_argument(
        "--no-prefer-existing",
        action="store_true",
        help="Recompute every sample instead of reusing any neighboring manifest.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N rows while computing raw measures. Use 0 to disable.",
    )
    args = parser.parse_args()

    manifest_paths = [path.resolve() for path in args.manifests]
    if not manifest_paths:
        raise SystemExit("No manifest paths found to update.")

    raw_frames = []
    for manifest_path in manifest_paths:
        print(f"[noneq] scanning {manifest_path}")
        raw_frames.append(
            compute_manifest_raw_noneq(
                manifest_path,
                prefer_existing=not args.no_prefer_existing,
                progress_interval=args.progress_interval,
            )
        )

    normalized_frames, min_value, max_value = normalize_jointly(raw_frames)
    print(
        "[noneq] joint normalization "
        f"min={min_value:.6g}, max={max_value:.6g}, "
        f"count={sum(len(frame) for frame in normalized_frames)}"
    )

    for manifest_path, frame in zip(manifest_paths, normalized_frames):
        frame.to_csv(manifest_path, index=False)
        print(
            f"[noneq] wrote {manifest_path} "
            f"(rows={len(frame)}, "
            f"norm_min={frame['nonequilibrium_measure'].min():.6f}, "
            f"norm_max={frame['nonequilibrium_measure'].max():.6f})"
        )


if __name__ == "__main__":
    main()
