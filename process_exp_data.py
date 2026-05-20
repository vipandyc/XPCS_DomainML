from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
import torch
from produce_data import coarse_grain_g2

def crop_data(
    data: np.ndarray,   # [x, x, channel_dim], where x >= crop_size
    temperature: float,
    crop_size: int = 2500,
    coarse_size: int = 256,
    step: int = 100,
    save_path: Path = Path("dataset/experiment"),
    channel_indices: int | list[int] = None,
    source_name: str | None = None,
    sample_name: str | None = None,
) -> None:
    """
    Cropping the experimental data to a bunch of smaller patches on the diagonal line
    of size `coarse_size` for every `step`, before coarse-graining to coarse_size.
    Save the patches as individual .pt files under save_path. This cropping is applied to 
    each channel independently, unless specified the target channel indices.
    
    To avoid conflicting file names while keeping rebuilds reproducible, the
    patches for each raw sample are stored under a deterministic subfolder
    derived from `sample_name` or `source_name`. Under the same subfolder, we
    also save a manifest file `manifest.csv` recording the temperature (T) and
    data path for reference. To match the manifest structure of the simulation
    data (id,gamma,D,GB_conc,T,path), we also add these columns with dummy
    values.
    
    Args:
        data: np.ndarray of shape [x, x, channel_dim], where x >= crop_size.
            The last dimension corresponds to the channel dimension.
        temperature: float, temperature value to record in the manifest
        crop_size: int, size of the cropped patches
        coarse_size: int, size after coarse-graining
        step: int, step size for cropping along the diagonal
        save_path: Path, directory to save the cropped patches
        channel_indices: int or list of int, specifying which channels to process.
            If None, process all channels.
        source_name: Optional raw source filename recorded in the manifest for
            traceability.
        sample_name: Optional deterministic directory name for this raw sample.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    sample_key = sample_name or source_name or "sample"
    sample_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_key).strip("._")
    if not sample_key:
        sample_key = "sample"
    sample_path = save_path / sample_key
    if sample_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing cropped sample directory: {sample_path}"
        )
    sample_path.mkdir(parents=True, exist_ok=True)
    h, w, c = data.shape
    assert h == w and h >= crop_size, "Data must be square and larger than crop_size"
    
    patch_rows = []
    count = 0
    if channel_indices is None:
        channel_indices = list(range(c))
    elif isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    for ch in range(c):
        if ch not in channel_indices:
            continue
        channel_data = data[:, :, ch]
        for start in range(0, h - crop_size + 1, step):
            patch = channel_data[start:start + crop_size, start:start + crop_size]
            patch_tensor = torch.tensor(patch, dtype=torch.float32)
            coarse_patch = coarse_grain_g2(
                patch_tensor,
                target_size=(coarse_size, coarse_size),
            ).unsqueeze(0)  # add channel dim; shape (1, coarse_size, coarse_size)
            patch_path = sample_path / f"patch_{count:06d}.pt"
            torch.save(coarse_patch, patch_path)
            patch_rows.append({
                "gamma": -1.0,
                "D": -1.0,
                "GB_conc": -1.0,
                "T": temperature,
                "path": str(patch_path),
                "domain": "experiment",
                "source_file": source_name or "",
                "source_channel": ch,
                "crop_start": start,
            })
            count += 1
    print(f"Saved {count} patches under {sample_path}")

    for idx, row in enumerate(patch_rows):
        row["id"] = idx

    manifest_df = pd.DataFrame(patch_rows)
    manifest_path = sample_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
        
    
def merge_data(
    dataset_path: Path = Path("dataset/experiment"),
    output_data_dir: Path = Path("dataset/experiment"),
    remove_original: bool = False,
) -> None:
    """
    Merge all cropped data patches in dataset_path into output_data_dir.
    In `dataset_path`, each subfolder corresponds to one original data sample,
    containing multiple cropped patches labeled as patch_xxx.pt. This function collects
    all these patches, renames them to avoid conflicts, and saves them directly under 
    output_data_dir. Moreover, there is a manifest.csv file under each subfolder; all
    these manifests are merged into a single manifest.csv under output_data_dir.
    
    If `remove_original` is True, the original `dataset_path` directory is deleted.
    
    Args:
        dataset_path: Path, directory containing subfolders of cropped patches
        output_data_dir: Path, directory to save merged patches
        remove_original: bool, whether to delete the original dataset_path after merging
    """
    same_dir = False
    if output_data_dir == dataset_path:
        # To avoid conflict, create a new, separate directory first
        # After merging is complete, the original directory can be deleted and renamed
        same_dir = True
        output_data_dir = dataset_path.parent / (dataset_path.name + "_merged")
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    all_manifests = []
    updated_paths = []
    updated_ids = []
    
    for file in sorted(dataset_path.iterdir()):
        assert file.is_dir(), f"Unexpected file {file} in dataset_path"
        manifest_file = file / "manifest.csv"
        assert manifest_file.exists(), f"Missing manifest in {file}"
        df = pd.read_csv(manifest_file)
        all_manifests.append(df)
        
    merged_manifest = pd.concat(all_manifests, ignore_index=True)
    count = 0
    
    for _, row in merged_manifest.iterrows():
        old_path = Path(row['path'])
        new_filename = f"{count:06d}.pt"
        new_path = output_data_dir / new_filename
        
        # Copy the file
        if old_path.exists():
            shutil.copy2(old_path, new_path)
            updated_paths.append(str(new_path) if not same_dir else str(dataset_path / new_filename))
            updated_ids.append(count)
            count += 1
            
    merged_manifest['path'] = updated_paths
    merged_manifest['id'] = updated_ids
    if "domain" not in merged_manifest.columns:
        merged_manifest["domain"] = "experiment"
    merged_manifest.to_csv(output_data_dir / "manifest.csv", index=False)
    
    if same_dir:
        shutil.rmtree(dataset_path)
        output_data_dir.rename(dataset_path)
    elif remove_original:
        shutil.rmtree(dataset_path)
    
if __name__ == "__main__":
    raw_exp_data_path = Path("exp_data")
    for data_file in raw_exp_data_path.iterdir():
        if data_file.suffix == ".npy":
            data = np.load(data_file)
        elif data_file.suffix == ".npz":
            data = np.load(data_file)["g12"]
        else:
            continue
        T = 273.15 + float(str(data_file.name).split("T")[-1].split("C")[0])
        crop_data(data, T, source_name=data_file.name, sample_name=data_file.stem)
        print(f"Processed {data_file.name}")
    merge_data(
        dataset_path=Path("dataset/experiment"),
        output_data_dir=Path("dataset/experiment"),
        remove_original=True,
    )
