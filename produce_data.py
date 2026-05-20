import csv, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.constants import Boltzmann
from scipy.stats import qmc, beta as beta_dist
from typing import Optional
from tqdm import tqdm
import warnings

### Global Parameters ###
OUT = Path("dataset")
steps = 1250           # Number of time steps; since dt = 2s, total time = 2500s.
# Reminder: we'll be coarse-graining to 256x256 later!

def normalize_g2(g2: torch.Tensor, min_val=1.0, max_val=1.15) -> torch.Tensor:
    """Linearly normalize to [min_val, max_val]"""
    g2_min, g2_max = g2.min(), g2.max()
    return (g2 - g2_min) / (g2_max - g2_min + 1e-8) * (max_val - min_val) + min_val

def coarse_grain_g2(g2: torch.Tensor, target_size=(256, 256)) -> torch.Tensor:
    """Downsample using bilinear interpolation"""
    g2 = g2.unsqueeze(0).unsqueeze(0)  # add batch and channel
    return F.interpolate(g2, size=target_size, mode='bilinear', align_corners=False).squeeze()

def simulate_xpcs(gamma, D, GB_conc, T, q_vec=0.045, dt=2, steps=steps, seed: Optional[int] = None, coarse=False,) -> Optional[torch.Tensor]:
    """
    `gamma`, `D`, `GB_conc`, `T`, `q_vec` --> `g2_matrix`
    
    Ranges:
        - gamma: 3e17 ~ 5e18
        - D: 1e-23 ~ 1e-21
        - GB_conc: 0.0 ~ 0.3
        - T: 300 ~ 500
        - q_vec: fixed at 0.045 Å⁻¹

    Returns the g2 tensor (and optionally, with coarse-graining), or None if numerical 
    instability is detected.
    """
    N = 50
    kT = T * Boltzmann  # Convert to J
    k = 0.25           # J/m²
    # dt = 2             # s
    q = np.array([q_vec * 1e10])  # wavevector in m⁻¹
    A = 0             # dimensionless
    dx = 1e-9         # m
    beta = 0.14
    mechanics = 0
    
    L = (N - 1) * dx
    r = np.zeros((steps, N, 1))
    x0 = np.linspace(0, 1, N)
    r[0, :, 0] = A * x0 * (1 - x0) * L
    r[:, 0] = r[0, 0]
    r[:, -1] = r[0, -1]

    sqrt_2T_dt = np.sqrt(2 * kT * dt / gamma) / dx

    # per-sample RNG for reproducibility
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Langevin Dynamics
    for t in range(1, steps):
        r[t] = r[t - 1].copy()
        for i in range(1, N - 1):
            F = k * (r[t - 1, i + 1] + r[t - 1, i - 1] - 2 * r[t - 1, i]) / dx**2
            noise = rng.normal(0, 1, 1) * sqrt_2T_dt
            # The term `dt / gamma` acts as a mobility factor. `gamma` itself is a friction/viscosity coefficient.
            r[t, i] += dt / gamma * (F + mechanics) + noise

        if np.any(np.isnan(r[t])) or np.any(np.isinf(r[t])):
            print(f"Numerical instability at step {t}")
            break

        max_abs = float(np.max(np.abs(r[t])))
        if max_abs > 1e-6:
            warnings.warn(
                f"simulate_xpcs early exit at step {t}: max |r| = {max_abs:.2e} m; skipping this sample",
                RuntimeWarning
            )
            return None

    # g1 Matrix Calculation
    t_list = np.linspace(0, dt * steps, steps)
    g1_matrix = np.zeros((steps, steps), dtype=complex)

    for i, t1 in enumerate(t_list):
        R1 = r[i]
        for j, t2 in enumerate(t_list):
            R2 = r[j]
            diffs = R1[:, None, :] - R2[None, :, :]
            phases = np.einsum('ijk,k->ij', diffs, q) # shape (N, N)
            g1_matrix[i, j] = np.mean(GB_conc * np.exp(-1j * phases)) + \
                              (1 - GB_conc) * np.exp(-D * float(np.linalg.norm(q))**2 * abs(t2 - t1))
    
    g2_matrix = 1 + beta * np.abs(g1_matrix)**2
    return (
        torch.tensor(g2_matrix, dtype=torch.float32) if not coarse
        else coarse_grain_g2(torch.tensor(g2_matrix, dtype=torch.float32), target_size=(256, 256))
    )

def sample_params_lhs(n, specs, seed=123):
    """
    Args:
        n: number of samples
        specs: parameter specifications (above)
    
    Returns:
        pars: parameter dictionary of arrays, each of shape (n,)
            ({"gamma": [...], "D": [...], "GB_conc": [...], "T": [...]})
    """
    keys = list(specs.keys())
    d = len(keys)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    U = sampler.random(n)  # shape (n, d)
    
    X = {}
    for j, k in enumerate(keys):
        s = specs[k]
        if s.get("scale", "linear") == "log": # Log scale
            lo, hi = np.log10(s["low"]), np.log10(s["high"])
            X[k] = 10 ** (lo + U[:, j] * (hi - lo))
        else: # Linear scale
            X[k] = s["low"] + U[:, j] * (s["high"] - s["low"])
    return {k: X[k] for k in keys}

if __name__ == "__main__":
    N = 500                # total samples
    dtype = torch.float32  # or float16
    seed = random.randint(1000, 10000)
    while OUT.exists() and (OUT/f"data_{seed}").is_dir() and (OUT/f"data_{seed}").exists():
        seed = random.randint(1000, 10000)
    print(f"Using random seed {seed}")
    (OUT/f"data_{seed}").mkdir(parents=True, exist_ok=True)

    # specs = {
    #     "gamma":   {"low": 2e18,   "high": 5e18,   "scale": "linear"},
    #     "D":       {"low": 1e-23,  "high": 4e-23,  "scale": "log"},
    #     "GB_conc": {"low": 0.15,    "high": 0.3,    "scale": "linear"},
    #     "T":       {"low": 300,    "high": 500,    "scale": "linear"}, 
    # }
    
    specs = {
        "gamma":   {"low": 1e18,   "high": 6e18,   "scale": "linear"},
        "D":       {"low": 3e-24,  "high": 3e-21,  "scale": "log"},
        "GB_conc": {"low": 0.0,    "high": 0.5,    "scale": "linear"},
        "T":       {"low": 300,    "high": 500,    "scale": "linear"}, 
    }
    
    # Simulate and save data
    pars = sample_params_lhs(N, specs, seed=seed)
    rows = []
    skipped = 0
    mean = 0.0
    m2 = 0.0
    count = 0
    for i in tqdm(range(N), desc="Simulating samples"):
        gamma = pars["gamma"][i]
        D = pars["D"][i]
        GB_conc = pars["GB_conc"][i]
        T = pars["T"][i]
        
        g2 = simulate_xpcs(gamma, D, GB_conc, T, seed=42, coarse=False) # Fixed seed for simulation! Update: no coarse-graining here
        if g2 is None:
            skipped += 1
            continue
        g2 = g2.unsqueeze(0)  # add channel dim: (1, 256, 256)

        id_ = i
        path = OUT/f"data_{seed}"/f"{id_:06d}.pt"
        # path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(g2, path)
        
        rows.append({
            "id": id_,
            "gamma": gamma,
            "D": D,
            "GB_conc": GB_conc,
            "T": T,
            "path": str(path),
        })
        
        # online mean/st
        n = g2.numel()
        x_mean = g2.mean().item()
        x_var  = g2.var(correction=0).item()
        delta = x_mean - mean
        total = count + n
        mean += delta * n / total
        m2 += x_var * n + (delta**2) * count * n / total
        count = total

    print(f"Saved {len(rows)} samples; skipped {skipped} due to instability.")

    # Save manifest
    if not rows:
        raise RuntimeError("No samples were generated; all simulations were unstable. Adjust ranges or integration settings.")
    with open(OUT/f"manifest_{seed}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x["id"]))

    import math
    std = math.sqrt(m2 / count)

    with open(OUT/f"stats_{seed}.json", "w") as f:
        json.dump({"mean": mean, "std": std, "specs": specs}, f)