import os, json, numpy as np
import torch
import pandas as pd
import shutil
from pathlib import Path
import math

def merge_data_and_manifests(
    specs: dict = {},
    dataset_path="dataset",
    output_data_dir="dataset/data",
    output_manifest="dataset/manifest.csv",
    output_json="dataset/stats.json"
) -> None:
    """
    Merge all data_xxxx directories and manifest_xxxx.csv files into unified structure
    """
    dataset_path = Path(dataset_path)
    output_data_dir = Path(output_data_dir)
    
    original_manifest = None
    original_json = None
    
    mean = 0.0
    m2 = 0.0
    n_count = 0
    
    if os.path.exists(output_manifest):
        original_manifest = pd.read_csv(output_manifest)
        assert os.path.exists(output_json), f"Existing manifest but no existing json"
        with open(output_json) as f:
            original_json = json.load(f)
        specs = original_json.get("specs", {})
        mean = original_json.get("mean", 0.0)
        std = original_json.get("std", 1.0)
        assert output_data_dir.exists(), f"No existing data but with existing manifest and json"
        n_count = len(original_manifest) * 256 ** 2
        m2 = (std**2) * n_count
    
    # Create output data directory
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all data_xxxx directories and manifest files
    data_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("data_")])
    manifest_files = sorted([f for f in dataset_path.iterdir() if f.name.startswith("manifest_") and f.suffix == ".csv"])
    
    print(f"Found {len(data_dirs)} data directories and {len(manifest_files)} manifest files")
    
    # Read and combine all manifest files
    all_manifests = []
    for manifest_file in manifest_files:
        df = pd.read_csv(manifest_file)
        all_manifests.append(df)
    
    # Concatenate all manifests
    if all_manifests:
        merged_manifest = pd.concat(all_manifests, ignore_index=True)
    else:
        print("No new manifest files to merge")
        return
    
    # Copy all .pt files with sequential naming and update paths
    file_counter = len(original_manifest) if os.path.exists(output_manifest) else 0
    updated_paths = []
    updated_ids = []
    
    for _, row in merged_manifest.iterrows():
        old_path = Path(row['path'])
        new_filename = f"{file_counter:06d}.pt"
        new_path = output_data_dir / new_filename
        
        # Copy the file
        if old_path.exists():
            shutil.copy2(old_path, new_path)
            updated_paths.append(str(new_path))
            updated_ids.append(file_counter)
            print(f"Copied {old_path} -> {new_path}")
        else:
            raise FileNotFoundError(f"File {old_path} does not exist")
        
        file_counter += 1
        
        # Compute the statistics on the dataset
        xi = torch.load(new_path, map_location="cpu", weights_only=True).to(torch.float32)  # [1, 256, 256]
        n = xi.numel()
        x_mean = xi.mean().item()
        x_var  = xi.var(correction=0).item()
        # combine stats
        delta = x_mean - mean
        total = n_count + n
        mean += delta * n / total
        m2 += x_var * n + (delta**2) * n_count * n / total
        n_count = total
    
    # Update the path and id columns in merged manifest
    merged_manifest['id'] = updated_ids
    merged_manifest['path'] = updated_paths
    
    # Save merged manifest
    if original_manifest is not None:
        merged_manifest = pd.concat([original_manifest, merged_manifest], ignore_index=True)
    merged_manifest.to_csv(output_manifest, index=False)
    print(f"Merged manifest saved to {output_manifest}")
    print(f"Total files processed: {len(merged_manifest)}")

    std = math.sqrt(m2 / n_count)

    with open(output_json, "w") as f:
        json.dump({"mean": mean, "std": std, "specs": specs}, f)

if __name__ == "__main__":
    PATH = "dataset"
    specs = {
        "gamma":   {"low": 1e18,   "high": 6e18,   "scale": "linear"},
        "D":       {"low": 3e-24,  "high": 3e-21,  "scale": "log"},
        "GB_conc": {"low": 0.0,    "high": 0.5,    "scale": "linear"},
        "T":       {"low": 300,    "high": 500,    "scale": "linear"}, 
    }
    merge_data_and_manifests(
        specs=specs,
        dataset_path="dataset",
        output_data_dir="dataset/simulation_2",
    )
    # # Remove all .pt files in the given directory
    # target_dir = Path("dataset/data")
    # if target_dir.exists():
    #     for pdf_file in target_dir.glob("*.pdf"):
    #         pdf_file.unlink()
    #         print(f"Deleted {pdf_file}")
