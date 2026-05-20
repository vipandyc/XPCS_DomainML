import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from produce_data import coarse_grain_g2, normalize_g2, simulate_xpcs
from utils import (
    nonequilibrium_measure,
    plot_g2,
    plot_g2_side_by_side,
    plot_nonequilibrium_measure,
)


DEFAULT_INFERENCE_OUTPUT_DIR = Path("inference_outputs")
MODEL_TYPE_CHOICES = ["vanilla", "adv", "coral", "coral-surrogate"]
PHASE_RANGE_MODE_CHOICES = ["stats", "data", "fixed"]


PHASE_PLOT_SPECS = [
    {
        "x": "D",
        "y": "gamma",
        "xscale": "log",
        "yscale": "linear",
        "xlabel": "Diffusivity " r"$D$ (cm$^2$ $\cdot$s$^{-1}$)",
        "ylabel": "GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
        "xticks": [1e-23, 1e-22, 1e-21],
        "yticks": [2e18, 2.5e18, 3e18, 3.5e18, 4e18, 4.5e18, 5e18],
        "xlabels": ["$10^{-23}$", "$10^{-22}$", "$10^{-21}$"],
        "ylabels": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "filename": "phase_map_D_gamma",
        "x_range": (1e-23, 1e-21),
        "y_range": (2e18, 5e18),
    },
    {
        "x": "gamma",
        "y": "GB_conc",
        "xscale": "linear",
        "yscale": "linear",
        "xlabel": "GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
        "ylabel": "Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
        "xticks": [2e18, 3e18, 4e18, 5e18],
        "yticks": [0.0, 0.1, 0.2, 0.3],
        "xlabels": [2, 3, 4, 5],
        "ylabels": [0.0, 0.1, 0.2, 0.3],
        "filename": "phase_map_gamma_GB",
        "x_range": (2e18, 5e18),
        "y_range": (0.0, 0.3),
    },
    {
        "x": "D",
        "y": "GB_conc",
        "xscale": "log",
        "yscale": "linear",
        "xlabel": "Diffusivity " r"$D$ (cm$^2$ $\cdot$s$^{-1}$)",
        "ylabel": "Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
        "xticks": [1e-23, 1e-22, 1e-21],
        "yticks": [0.0, 0.1, 0.2, 0.3],
        "xlabels": ["$10^{-23}$", "$10^{-22}$", "$10^{-21}$"],
        "ylabels": [0.0, 0.1, 0.2, 0.3],
        "filename": "phase_map_D_GB",
        "x_range": (1e-23, 1e-21),
        "y_range": (0.0, 0.3),
    },
]

MODEL_COLUMN_MAP = {
    "adv": {
        "D": "D_adv",
        "gamma": "gamma_adv",
        "GB_conc": "lambda_GB_adv",
    },
    "vanilla": {
        "D": "D_vanilla",
        "gamma": "gamma_vanilla",
        "GB_conc": "lambda_GB_vanilla",
    },
}

TEMPERATURE_MARKERS = (
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
    "*",
    "h",
    "8",
)


def infer_no_t_from_checkpoint_name(
    model_path: Path | None,
    model_type: str,
) -> bool:
    """Infer whether a checkpoint uses the no-temperature model variant."""
    if model_type in {"coral", "coral-surrogate"}:
        return True
    if model_path is None:
        return False
    lowered = model_path.name.lower()
    return ("no_t" in lowered) or ("no-t" in lowered)


def resolve_model_components(
    model_type: str,
    model_path: Path | None = None,
    no_t: bool | None = None,
) -> tuple[object, object, bool]:
    """Resolve the checkpoint loader and de-normalization helper."""
    resolved_no_t = infer_no_t_from_checkpoint_name(model_path, model_type) if no_t is None else no_t

    if model_type == "vanilla":
        if resolved_no_t:
            from train_vanilla_no_T import (
                denorm_from_meta as denorm_from_meta_vanilla_no_t,
                load_model as load_vanilla_model_no_t,
            )

            return load_vanilla_model_no_t, denorm_from_meta_vanilla_no_t, True
        from train_vanilla import (
            denorm_from_meta as denorm_from_meta_vanilla_with_t,
            load_model as load_vanilla_model,
        )

        return load_vanilla_model, denorm_from_meta_vanilla_with_t, False
    if model_type == "adv":
        if resolved_no_t:
            from train_adv_no_T import (
                denorm_from_meta as denorm_from_meta_adv_no_t,
                load_model as load_adv_model_no_t,
            )

            return load_adv_model_no_t, denorm_from_meta_adv_no_t, True
        from train_adv import (
            denorm_from_meta as denorm_from_meta_adv_with_t,
            load_model as load_adv_model,
        )

        return load_adv_model, denorm_from_meta_adv_with_t, False
    if model_type == "coral":
        if not resolved_no_t:
            raise ValueError("CORAL checkpoints are only supported for the no-T model variant")
        from train_adv_coral_distill import (
            denorm_from_meta as denorm_from_meta_coral_no_t,
            load_model as load_coral_model_no_t,
        )

        return load_coral_model_no_t, denorm_from_meta_coral_no_t, True
    if model_type == "coral-surrogate":
        if not resolved_no_t:
            raise ValueError(
                "CORAL surrogate checkpoints are only supported for the no-T model variant"
            )
        from train_adv_coral_surrogate import (
            denorm_from_meta as denorm_from_meta_coral_surrogate_no_t,
            load_model as load_coral_surrogate_model_no_t,
        )

        return (
            load_coral_surrogate_model_no_t,
            denorm_from_meta_coral_surrogate_no_t,
            True,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def resolve_sim_dataset_class(no_t: bool):
    """Return the simulation dataset class matching the checkpoint preprocessing."""
    if no_t:
        from train_adv_no_T import XPCSDataset as XPCSDatasetNoT

        return XPCSDatasetNoT
    from train_adv import XPCSDataset as XPCSDatasetWithT

    return XPCSDatasetWithT


def load_simulation_manifest(sim_root: Path) -> pd.DataFrame:
    """
    Load the base simulation manifest and merge the nonequilibrium column when
    the enriched manifest is available.
    """
    manifest_path = sim_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find simulation manifest: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)

    noneq_manifest_path = sim_root / "manifest_with_non_equ.csv"
    if noneq_manifest_path.exists():
        noneq_df = pd.read_csv(noneq_manifest_path)
        if "nonequilibrium_measure" in noneq_df.columns:
            if len(noneq_df) == len(manifest_df):
                manifest_df["nonequilibrium_measure"] = noneq_df["nonequilibrium_measure"].to_numpy()
            elif "id" in manifest_df.columns and "id" in noneq_df.columns:
                manifest_df = manifest_df.merge(
                    noneq_df[["id", "nonequilibrium_measure"]],
                    on="id",
                    how="left",
                )
    return manifest_df


def parse_temperature_from_name(path: Path) -> float:
    """Parse a raw experiment filename token like `T26C` into Kelvin."""
    match = re.search(r"T(-?\d+(?:\.\d+)?)C", path.stem)
    if match is None:
        raise ValueError(f"Could not parse temperature from filename: {path.name}")
    return 273.15 + float(match.group(1))


def build_diagonal_crop_starts(
    array_shape: tuple[int, int],
    crop_size: int,
    crop_step: int,
    crop_policy: str,
) -> list[int]:
    """Return the diagonal crop offsets used to build one experiment display g2."""
    height, width = array_shape
    if height != width:
        raise ValueError(f"Expected square raw experiment arrays, got {array_shape}")
    if height <= crop_size:
        return [0]
    if crop_policy == "top-left":
        return [0]
    if crop_policy != "all-diagonal":
        raise ValueError(f"Unsupported crop policy: {crop_policy}")
    if crop_step <= 0:
        raise ValueError(f"Crop step must be positive, got {crop_step}")
    return list(range(0, height - crop_size + 1, crop_step))


def default_plot_path(
    source: str,
    identifier: str,
    output_path: Path | None = None,
) -> Path:
    """Build a default output path for one saved g2 figure."""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    save_path = DEFAULT_INFERENCE_OUTPUT_DIR / f"{source}_{identifier}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def ensure_nonequilibrium_column(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate `nonequilibrium_measure` for simulation rows when the enriched
    manifest is unavailable.
    """
    if "nonequilibrium_measure" in sim_df.columns and sim_df["nonequilibrium_measure"].notna().all():
        return sim_df
    if "path" not in sim_df.columns:
        raise ValueError("Simulation manifest must contain a `path` column")

    sim_df = sim_df.copy()
    measures = []
    for path_str in sim_df["path"]:
        g2 = torch.load(path_str, weights_only=True).to(torch.float32).squeeze(0)
        g2 = normalize_g2(g2, min_val=1.0, max_val=1.2)
        measures.append(float(nonequilibrium_measure(g2)))
    sim_df["nonequilibrium_measure"] = measures
    return sim_df


def format_temperature_label(temperature_c: float) -> str:
    return f"{float(temperature_c):g} C"


def build_temperature_style_lookup(temperatures_c: pd.Series) -> dict[float, dict[str, object]]:
    unique_temperatures = sorted(
        {float(value) for value in temperatures_c.to_numpy(dtype=np.float64) if np.isfinite(value)}
    )
    if not unique_temperatures:
        return {}

    if len(unique_temperatures) <= 10:
        cmap = plt.get_cmap("tab10")
    elif len(unique_temperatures) <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("viridis")

    if len(unique_temperatures) == 1:
        color_positions = [0.0]
    else:
        color_positions = np.linspace(0.0, 1.0, num=len(unique_temperatures))

    lookup: dict[float, dict[str, object]] = {}
    for index, (temperature_c, color_position) in enumerate(
        zip(unique_temperatures, color_positions, strict=False)
    ):
        lookup[temperature_c] = {
            "label": format_temperature_label(temperature_c),
            "color": to_hex(cmap(float(color_position))),
            "marker": TEMPERATURE_MARKERS[index % len(TEMPERATURE_MARKERS)],
            "order": index,
        }
    return lookup


def iter_temperature_styles(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    style_columns = [
        "temperature_c",
        "temperature_order",
        "temperature_label",
        "temperature_color",
        "temperature_marker",
    ]
    style_df = (
        df[style_columns]
        .drop_duplicates()
        .sort_values(["temperature_order", "temperature_c"])
        .reset_index(drop=True)
    )
    return style_df.to_dict("records")


def temperature_legend_ncols(num_items: int) -> int:
    if num_items > 10:
        return 3
    if num_items > 5:
        return 2
    return 1


def add_overlay_metadata(
    overlay_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    overlay_df = overlay_df.copy()
    temperature_style_lookup = build_temperature_style_lookup(overlay_df["temperature_c"])
    overlay_df["temperature_label"] = overlay_df["temperature_c"].map(
        lambda value: temperature_style_lookup[float(value)]["label"]
    )
    overlay_df["temperature_color"] = overlay_df["temperature_c"].map(
        lambda value: temperature_style_lookup[float(value)]["color"]
    )
    overlay_df["temperature_marker"] = overlay_df["temperature_c"].map(
        lambda value: temperature_style_lookup[float(value)]["marker"]
    )
    overlay_df["temperature_order"] = overlay_df["temperature_c"].map(
        lambda value: temperature_style_lookup[float(value)]["order"]
    )

    for axis_name in ("D", "gamma", "GB_conc"):
        sim_min = float(simulation_df[axis_name].min())
        sim_max = float(simulation_df[axis_name].max())
        overlay_df[f"{axis_name}_in_sim_range"] = overlay_df[axis_name].between(sim_min, sim_max)

    overlay_df["within_simulation_domain"] = overlay_df[
        ["D_in_sim_range", "gamma_in_sim_range", "GB_conc_in_sim_range"]
    ].all(axis=1)
    return overlay_df


def interpolate_phase_measure(
    df: pd.DataFrame,
    query_df: pd.DataFrame,
    x: str,
    y: str,
    xscale: str,
    yscale: str,
) -> np.ndarray:
    """
    Evaluate the projected simulation nonequilibrium phase map at query points.

    Linear interpolation is used first, with nearest-neighbor fallback so points
    inside the axis bounds but outside the convex hull still get a projected
    value.
    """
    X = df[x].to_numpy(dtype=np.float64)
    Y = df[y].to_numpy(dtype=np.float64)
    Z = df["nonequilibrium_measure"].to_numpy(dtype=np.float64)

    query_x = query_df[x].to_numpy(dtype=np.float64)
    query_y = query_df[y].to_numpy(dtype=np.float64)

    x_transformed = np.log10(X) if xscale == "log" else X
    y_transformed = np.log10(Y) if yscale == "log" else Y
    query_x_transformed = np.log10(query_x) if xscale == "log" else query_x
    query_y_transformed = np.log10(query_y) if yscale == "log" else query_y

    points = np.column_stack([x_transformed, y_transformed])
    xi = np.column_stack([query_x_transformed, query_y_transformed])
    linear = griddata(
        points=points,
        values=Z,
        xi=xi,
        method="linear",
        rescale=True,
    )
    nearest = griddata(
        points=points,
        values=Z,
        xi=xi,
        method="nearest",
        rescale=True,
    )
    linear = np.asarray(linear, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.float64)
    return np.where(np.isfinite(linear), linear, nearest)


def interpolate_phase_measure_3d(
    df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> np.ndarray:
    """
    Evaluate the simulation nonequilibrium field at inferred `(D, gamma, GB_conc)`
    points using 3D interpolation in parameter space.
    """
    sim_D = df["D"].to_numpy(dtype=np.float64)
    sim_gamma = df["gamma"].to_numpy(dtype=np.float64)
    sim_gb = df["GB_conc"].to_numpy(dtype=np.float64)
    sim_noneq = df["nonequilibrium_measure"].to_numpy(dtype=np.float64)

    query_D = query_df["D"].to_numpy(dtype=np.float64)
    query_gamma = query_df["gamma"].to_numpy(dtype=np.float64)
    query_gb = query_df["GB_conc"].to_numpy(dtype=np.float64)

    sim_points = np.column_stack([np.log10(sim_D), sim_gamma, sim_gb])
    query_points = np.column_stack([np.log10(query_D), query_gamma, query_gb])

    linear = griddata(
        points=sim_points,
        values=sim_noneq,
        xi=query_points,
        method="linear",
        rescale=True,
    )
    nearest = griddata(
        points=sim_points,
        values=sim_noneq,
        xi=query_points,
        method="nearest",
        rescale=True,
    )
    linear = np.asarray(linear, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.float64)
    return np.where(np.isfinite(linear), linear, nearest)


def plot_nonequilibrium_phase_map(
    df: pd.DataFrame,
    save_path: Path,
    x: str,
    y: str,
    xscale: str,
    yscale: str,
    xlabel: str,
    ylabel: str,
    overlay_df: pd.DataFrame | None = None,
    overlay_x: str | None = None,
    overlay_y: str | None = None,
    overlay_label: str = "Experiment inference",
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> dict[str, int]:
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "arial"
    plt.rcParams["mathtext.it"] = "arial:italic"
    plt.rcParams["mathtext.bf"] = "arial:bold"

    X = df[x].to_numpy(dtype=np.float64)
    Y = df[y].to_numpy(dtype=np.float64)
    Z = df["nonequilibrium_measure"].to_numpy(dtype=np.float64)

    x_transformed = np.log10(X) if xscale == "log" else X
    y_transformed = np.log10(Y) if yscale == "log" else Y

    if x_range is not None:
        xt_lo = np.log10(x_range[0]) if xscale == "log" else x_range[0]
        xt_hi = np.log10(x_range[1]) if xscale == "log" else x_range[1]
    else:
        xt_lo, xt_hi = x_transformed.min(), x_transformed.max()
    if y_range is not None:
        yt_lo = np.log10(y_range[0]) if yscale == "log" else y_range[0]
        yt_hi = np.log10(y_range[1]) if yscale == "log" else y_range[1]
    else:
        yt_lo, yt_hi = y_transformed.min(), y_transformed.max()

    grid_x_transformed = np.linspace(xt_lo, xt_hi, 400)
    grid_y_transformed = np.linspace(yt_lo, yt_hi, 400)
    grid_x_t, grid_y_t = np.meshgrid(grid_x_transformed, grid_y_transformed)

    grid_z = griddata(
        points=np.column_stack([x_transformed, y_transformed]),
        values=Z,
        xi=(grid_x_t, grid_y_t),
        method="linear",
        rescale=True,
    )
    nearest_grid_z = griddata(
        points=np.column_stack([x_transformed, y_transformed]),
        values=Z,
        xi=(grid_x_t, grid_y_t),
        method="nearest",
        rescale=True,
    )
    grid_z = np.where(np.isfinite(grid_z), grid_z, nearest_grid_z)
    # grid_z = gaussian_filter(grid_z, sigma=1.0)

    grid_x = np.power(10.0, grid_x_t) if xscale == "log" else grid_x_t
    grid_y = np.power(10.0, grid_y_t) if yscale == "log" else grid_y_t
    x_min = x_range[0] if x_range is not None else float(np.min(X))
    x_max = x_range[1] if x_range is not None else float(np.max(X))
    y_min = y_range[0] if y_range is not None else float(np.min(Y))
    y_max = y_range[1] if y_range is not None else float(np.max(Y))

    fig, ax = plt.subplots(figsize=(9, 7))
    cntr = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=80,
        cmap="plasma",
    )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks, xlabels if xlabels is not None else xticks)
    if yticks is not None:
        ax.set_yticks(yticks, ylabels if ylabels is not None else yticks)

    plotted_count = 0
    omitted_count = 0
    if (
        overlay_df is not None
        and overlay_x is not None
        and overlay_y is not None
        and not overlay_df.empty
    ):
        overlay_x_values = overlay_df[overlay_x].to_numpy(dtype=np.float64)
        overlay_y_values = overlay_df[overlay_y].to_numpy(dtype=np.float64)
        visible_mask = (
            np.isfinite(overlay_x_values)
            & np.isfinite(overlay_y_values)
            & (overlay_x_values >= x_min)
            & (overlay_x_values <= x_max)
            & (overlay_y_values >= y_min)
            & (overlay_y_values <= y_max)
        )
        visible_df = overlay_df.loc[visible_mask].copy()
        plotted_count = int(len(visible_df))
        omitted_count = int(len(overlay_df) - plotted_count)

        temperature_styles = iter_temperature_styles(visible_df)
        for style in temperature_styles:
            group_df = visible_df.loc[visible_df["temperature_c"] == style["temperature_c"]]
            if group_df.empty:
                continue
            ax.scatter(
                group_df[overlay_x],
                group_df[overlay_y],
                s=110,
                marker=str(style["temperature_marker"]),
                color=str(style["temperature_color"]),
                edgecolors="white",
                linewidths=1.0,
                alpha=0.95,
                clip_on=True,
                zorder=4,
                label=str(style["temperature_label"]),
            )

        if plotted_count:
            ax.legend(
                loc="upper right",
                fontsize=12,
                title=overlay_label,
                title_fontsize=13,
                framealpha=0.95,
                ncol=temperature_legend_ncols(len(temperature_styles)),
            )
        if omitted_count:
            ax.text(
                0.02,
                0.02,
                f"Showing {plotted_count}/{len(overlay_df)} points\nwithin simulation bounds",
                transform=ax.transAxes,
                fontsize=11,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
            )

    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("Nonequilibrium Measure")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return {
        "plotted_overlay_points": plotted_count,
        "omitted_overlay_points": omitted_count,
    }


def plot_nonequilibrium_phase_map_exp_overlay(
    df: pd.DataFrame,
    save_path: Path,
    x: str,
    y: str,
    xscale: str,
    yscale: str,
    xlabel: str,
    ylabel: str,
    overlay_df: pd.DataFrame,
    overlay_x: str,
    overlay_y: str,
    overlay_value_column: str = "experimental_nonequilibrium_measure",
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> dict[str, int]:
    """
    Plot the phase-map background and color experiment points by their measured
    nonequilibrium value.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "arial"
    plt.rcParams["mathtext.it"] = "arial:italic"
    plt.rcParams["mathtext.bf"] = "arial:bold"

    X = df[x].to_numpy(dtype=np.float64)
    Y = df[y].to_numpy(dtype=np.float64)
    Z = df["nonequilibrium_measure"].to_numpy(dtype=np.float64)

    x_transformed = np.log10(X) if xscale == "log" else X
    y_transformed = np.log10(Y) if yscale == "log" else Y

    if x_range is not None:
        xt_lo = np.log10(x_range[0]) if xscale == "log" else x_range[0]
        xt_hi = np.log10(x_range[1]) if xscale == "log" else x_range[1]
    else:
        xt_lo, xt_hi = x_transformed.min(), x_transformed.max()
    if y_range is not None:
        yt_lo = np.log10(y_range[0]) if yscale == "log" else y_range[0]
        yt_hi = np.log10(y_range[1]) if yscale == "log" else y_range[1]
    else:
        yt_lo, yt_hi = y_transformed.min(), y_transformed.max()

    grid_x_transformed = np.linspace(xt_lo, xt_hi, 400)
    grid_y_transformed = np.linspace(yt_lo, yt_hi, 400)
    grid_x_t, grid_y_t = np.meshgrid(grid_x_transformed, grid_y_transformed)

    grid_z = griddata(
        points=np.column_stack([x_transformed, y_transformed]),
        values=Z,
        xi=(grid_x_t, grid_y_t),
        method="linear",
        rescale=True,
    )
    nearest_grid_z = griddata(
        points=np.column_stack([x_transformed, y_transformed]),
        values=Z,
        xi=(grid_x_t, grid_y_t),
        method="nearest",
        rescale=True,
    )
    grid_z = np.where(np.isfinite(grid_z), grid_z, nearest_grid_z)

    grid_x = np.power(10.0, grid_x_t) if xscale == "log" else grid_x_t
    grid_y = np.power(10.0, grid_y_t) if yscale == "log" else grid_y_t
    x_min = x_range[0] if x_range is not None else float(np.min(X))
    x_max = x_range[1] if x_range is not None else float(np.max(X))
    y_min = y_range[0] if y_range is not None else float(np.min(Y))
    y_max = y_range[1] if y_range is not None else float(np.max(Y))

    overlay_value_all = overlay_df[overlay_value_column].to_numpy(dtype=np.float64)
    combined_for_norm = np.concatenate(
        [Z[np.isfinite(Z)], overlay_value_all[np.isfinite(overlay_value_all)]]
    )
    norm = Normalize(
        vmin=float(np.nanmin(combined_for_norm)),
        vmax=float(np.nanmax(combined_for_norm)),
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cntr = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=np.linspace(norm.vmin, norm.vmax, 80),
        cmap="plasma",
        norm=norm,
    )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks, xlabels if xlabels is not None else xticks)
    if yticks is not None:
        ax.set_yticks(yticks, ylabels if ylabels is not None else yticks)

    overlay_x_values = overlay_df[overlay_x].to_numpy(dtype=np.float64)
    overlay_y_values = overlay_df[overlay_y].to_numpy(dtype=np.float64)
    overlay_value = overlay_df[overlay_value_column].to_numpy(dtype=np.float64)
    visible_mask = (
        np.isfinite(overlay_x_values)
        & np.isfinite(overlay_y_values)
        & np.isfinite(overlay_value)
        & (overlay_x_values >= x_min)
        & (overlay_x_values <= x_max)
        & (overlay_y_values >= y_min)
        & (overlay_y_values <= y_max)
    )
    visible_df = overlay_df.loc[visible_mask].copy()
    plotted_count = int(len(visible_df))
    omitted_count = int(len(overlay_df) - plotted_count)

    legend_handles: list[Line2D] = []
    if plotted_count:
        temperature_styles = iter_temperature_styles(visible_df)
        for style in temperature_styles:
            group_df = visible_df.loc[visible_df["temperature_c"] == style["temperature_c"]]
            if group_df.empty:
                continue
            ax.scatter(
                group_df[overlay_x],
                group_df[overlay_y],
                c=group_df[overlay_value_column],
                cmap="plasma",
                norm=norm,
                s=120,
                marker=str(style["temperature_marker"]),
                edgecolors=str(style["temperature_color"]),
                linewidths=1.4,
                alpha=0.95,
                zorder=4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=str(style["temperature_marker"]),
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor=str(style["temperature_color"]),
                    markersize=10,
                    linewidth=0,
                    label=str(style["temperature_label"]),
                )
            )

    if omitted_count:
        ax.text(
            0.02,
            0.02,
            f"Showing {plotted_count}/{len(overlay_df)} points\nwithin simulation bounds",
            transform=ax.transAxes,
            fontsize=11,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=10,
            title="Temperature",
            title_fontsize=11,
            framealpha=0.95,
            ncol=temperature_legend_ncols(len(legend_handles)),
        )

    cbar = plt.colorbar(cntr, ax=ax, pad=0.04)
    cbar.set_label("Nonequilibrium Measure")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return {
        "plotted_overlay_points": plotted_count,
        "omitted_overlay_points": omitted_count,
    }


def resolve_simulation_stats_path(simulation_manifest: Path) -> Path | None:
    """
    Locate the stats.json associated with a simulation manifest.

    Older preprocessing runs wrote `dataset/stats.json` even when the tensor
    files lived in `dataset/simulation_2`, while newer layouts may keep
    stats.json next to the manifest. Support both.
    """
    candidates = [
        simulation_manifest.parent / "stats.json",
        simulation_manifest.parent.parent / "stats.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_phase_ranges_from_stats(
    simulation_manifest: Path,
) -> dict[str, tuple[float, float]]:
    stats_path = resolve_simulation_stats_path(simulation_manifest)
    if stats_path is None:
        print(
            "[phase] no stats.json found next to simulation manifest; "
            "falling back to simulation data ranges"
        )
        return {}

    with stats_path.open() as handle:
        stats = json.load(handle)
    specs = stats.get("specs", {})
    ranges: dict[str, tuple[float, float]] = {}
    for key in ("D", "gamma", "GB_conc"):
        spec = specs.get(key)
        if not isinstance(spec, dict) or "low" not in spec or "high" not in spec:
            continue
        low = float(spec["low"])
        high = float(spec["high"])
        ranges[key] = (min(low, high), max(low, high))
    print(f"[phase] using parameter ranges from {stats_path}")
    return ranges


def plot_nonequilibrium_comparison(
    overlay_df: pd.DataFrame,
    save_path: Path,
    title: str,
    phase_value_column: str = "phase_map_nonequilibrium_measure",
) -> dict[str, int]:
    """
    Compare measured experiment nonequilibrium values against the projected
    phase-map value at each inferred parameter point.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 18

    required_columns = {
        "experimental_nonequilibrium_measure",
        phase_value_column,
        "temperature_label",
        "temperature_color",
        "temperature_marker",
    }
    if not required_columns.issubset(overlay_df.columns):
        missing = sorted(required_columns - set(overlay_df.columns))
        raise ValueError(f"Missing comparison columns: {missing}")

    compare_df = overlay_df.loc[
        np.isfinite(overlay_df["experimental_nonequilibrium_measure"].to_numpy(dtype=np.float64))
        & np.isfinite(overlay_df[phase_value_column].to_numpy(dtype=np.float64))
    ].copy()

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    temperature_styles = iter_temperature_styles(compare_df)
    for style in temperature_styles:
        group_df = compare_df.loc[compare_df["temperature_c"] == style["temperature_c"]]
        if group_df.empty:
            continue
        ax.scatter(
            group_df["experimental_nonequilibrium_measure"],
            group_df[phase_value_column],
            s=85,
            marker=str(style["temperature_marker"]),
            color=str(style["temperature_color"]),
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            label=str(style["temperature_label"]),
        )

    if not compare_df.empty:
        combined = np.concatenate(
            [
                compare_df["experimental_nonequilibrium_measure"].to_numpy(dtype=np.float64),
                compare_df[phase_value_column].to_numpy(dtype=np.float64),
            ]
        )
        axis_min = float(np.nanmin(combined))
        axis_max = float(np.nanmax(combined))
        if axis_max > axis_min:
            padding = 0.05 * (axis_max - axis_min)
            axis_min -= padding
            axis_max += padding
            ax.plot(
                [axis_min, axis_max],
                [axis_min, axis_max],
                linestyle="--",
                color="black",
                linewidth=1.2,
                alpha=0.8,
            )
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)

    ax.set_xlabel("Experiment Nonequilibrium Measure")
    ax.set_ylabel("Phase-Map Nonequilibrium Measure")
    ax.set_title(title)
    if not compare_df.empty:
        ax.legend(
            loc="upper left",
            fontsize=10,
            framealpha=0.95,
            title="Temperature",
            title_fontsize=11,
            ncol=temperature_legend_ncols(len(temperature_styles)),
        )
    ax.grid(alpha=0.20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return {
        "plotted_overlay_points": int(len(compare_df)),
        "omitted_overlay_points": int(len(overlay_df) - len(compare_df)),
    }


def prepare_experiment_display_g2(
    raw_data_dir: Path,
    file_name: str,
    shot_index: int,
    crop_size: int,
    coarse_size: int,
    crop_step: int,
    crop_policy: str,
    crop_aggregation: str,
) -> dict[str, object]:
    """Prepare one experimental shot for visualization without inference."""
    raw_path = resolve_raw_data_file(raw_data_dir, file_name)
    raw_data = load_raw_experiment_array(raw_path)
    if shot_index < 0 or shot_index >= raw_data.shape[-1]:
        raise IndexError(
            f"Shot index {shot_index} is out of bounds for {raw_path.name} "
            f"with {raw_data.shape[-1]} shot(s)"
        )

    crop_starts = build_diagonal_crop_starts(
        array_shape=raw_data.shape[:2],
        crop_size=crop_size,
        crop_step=crop_step,
        crop_policy=crop_policy,
    )
    crops: list[torch.Tensor] = []
    for crop_start in crop_starts:
        crop_end = crop_start + crop_size
        if crop_end > raw_data.shape[0] or crop_end > raw_data.shape[1]:
            raise ValueError(
                f"Crop [{crop_start}:{crop_end}] is out of bounds for {raw_path.name} "
                f"with shape {raw_data.shape}"
            )
        crop = torch.tensor(
            raw_data[crop_start:crop_end, crop_start:crop_end, shot_index],
            dtype=torch.float32,
        )
        coarse_crop = coarse_grain_g2(
            crop,
            target_size=(coarse_size, coarse_size),
        ).to(torch.float32)
        crops.append(normalize_g2(coarse_crop, min_val=1.0, max_val=1.2))

    g2 = aggregate_g2_crops(crops, aggregation=crop_aggregation)
    return {
        "raw_path": raw_path,
        "temperature_k": parse_temperature_from_name(raw_path),
        "crop_starts": crop_starts,
        "g2": g2,
        "nonequilibrium_measure": float(nonequilibrium_measure(g2)),
    }


@torch.no_grad()
def run_infer_sim_command(args: argparse.Namespace) -> None:
    """Infer parameters on one simulation index and optionally forward-plot g2."""
    device = torch.device(args.device)
    load_fn, denorm_fn, resolved_no_t = resolve_model_components(
        model_type=args.model_type,
        model_path=args.model_path,
        no_t=args.no_t,
    )
    model = load_fn(args.model_path, device=device)
    dataset_class = resolve_sim_dataset_class(resolved_no_t)
    sim_dataset = dataset_class(args.sim_root)
    sim_df = load_simulation_manifest(args.sim_root)

    if args.index < 0 or args.index >= len(sim_dataset):
        raise IndexError(
            f"Simulation index {args.index} is out of bounds for {len(sim_dataset)} samples"
        )
    if len(sim_df) != len(sim_dataset):
        raise ValueError(
            "Simulation manifest length does not match dataset length; "
            "please check dataset/simulation/manifest.csv"
        )

    row = sim_df.iloc[args.index]
    x, _, y_raw, temperature, _ = sim_dataset[args.index]
    pred_norm = model(
        x.unsqueeze(0).to(device),
        temperature.unsqueeze(0).to(device),
    )
    pred_raw = denorm_fn(pred_norm, model.norm_meta, device=device).squeeze(0).cpu()

    noneq_text = "n/a"
    if "nonequilibrium_measure" in row.index and not pd.isna(row["nonequilibrium_measure"]):
        noneq_text = f"{float(row['nonequilibrium_measure']):.6f}"
    print(f"[infer-sim] model_type={args.model_type} no_t={resolved_no_t}")
    print(f"[infer-sim] index={args.index} id={row.get('id', 'n/a')} path={row['path']}")
    print(
        "[infer-sim] "
        f"T={float(temperature.item()):.4f} K "
        f"| noneq={noneq_text}"
    )
    print(
        "[infer-sim] true    "
        f"gamma={float(y_raw[0]):.6e} "
        f"D={float(y_raw[1]):.6e} "
        f"GB_conc={float(y_raw[2]):.6f}"
    )
    print(
        "[infer-sim] pred    "
        f"gamma={float(pred_raw[0]):.6e} "
        f"D={float(pred_raw[1]):.6e} "
        f"GB_conc={float(pred_raw[2]):.6f}"
    )
    print(
        "[infer-sim] abs err "
        f"gamma={abs(float(pred_raw[0] - y_raw[0])):.6e} "
        f"D={abs(float(pred_raw[1] - y_raw[1])):.6e} "
        f"GB_conc={abs(float(pred_raw[2] - y_raw[2])):.6f}"
    )

    if not args.forward_g2:
        return

    output_dir = args.output_dir or (
        DEFAULT_INFERENCE_OUTPUT_DIR / f"infer_sim_{args.model_type}_idx{args.index:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_g2 = torch.load(row["path"], weights_only=True).to(torch.float32).squeeze(0)
    original_g2 = normalize_g2(original_g2, min_val=1.0, max_val=1.2)
    forward_g2 = simulate_xpcs(
        gamma=float(pred_raw[0].item()),
        D=float(pred_raw[1].item()),
        GB_conc=float(pred_raw[2].item()),
        T=float(temperature.item()),
        seed=args.forward_seed,
        coarse=True,
    )
    if forward_g2 is None:
        raise RuntimeError("simulate_xpcs returned None for the predicted parameters")
    forward_g2 = normalize_g2(forward_g2.to(torch.float32), min_val=1.0, max_val=1.2)

    original_path = output_dir / "sim_original_g2.pdf"
    forward_path = output_dir / "sim_forward_predicted_g2.pdf"
    comparison_path = output_dir / "sim_forward_comparison.pdf"
    original_noneq_path = output_dir / "sim_original_nonequilibrium.pdf"
    forward_noneq_path = output_dir / "sim_forward_nonequilibrium.pdf"

    plot_g2(
        original_g2,
        save_path=original_path,
        title=f"Simulation index {args.index} | original",
    )
    plot_g2(
        forward_g2,
        save_path=forward_path,
        title=f"Simulation index {args.index} | forward from predicted params",
    )
    plot_g2_side_by_side(
        original_g2,
        forward_g2,
        save_path=comparison_path,
        left_title="Original simulation g2",
        right_title="Forward g2 from prediction",
    )
    original_forward_noneq = plot_nonequilibrium_measure(
        original_g2,
        save_path=original_noneq_path,
    )
    predicted_forward_noneq = plot_nonequilibrium_measure(
        forward_g2,
        save_path=forward_noneq_path,
    )
    print(f"[infer-sim] wrote {original_path}")
    print(f"[infer-sim] wrote {forward_path}")
    print(f"[infer-sim] wrote {comparison_path}")
    print(f"[infer-sim] wrote {original_noneq_path}")
    print(f"[infer-sim] wrote {forward_noneq_path}")
    print(
        "[infer-sim] "
        f"forward noneq original={original_forward_noneq:.6f} "
        f"predicted={predicted_forward_noneq:.6f}"
    )


def run_plot_g2_command(args: argparse.Namespace) -> None:
    """Plot one simulation or experiment g2 without running inference."""
    if args.source == "sim":
        if args.index is None:
            raise ValueError("--index is required when --source sim")
        sim_df = load_simulation_manifest(args.sim_root)
        if args.index < 0 or args.index >= len(sim_df):
            raise IndexError(
                f"Simulation index {args.index} is out of bounds for {len(sim_df)} samples"
            )
        row = sim_df.iloc[args.index]
        g2 = torch.load(row["path"], weights_only=True).to(torch.float32).squeeze(0)
        g2 = normalize_g2(g2, min_val=1.0, max_val=1.2)
        noneq_value = float(nonequilibrium_measure(g2))
        save_path = default_plot_path(
            source="sim_g2",
            identifier=f"idx{args.index:06d}",
            output_path=args.output_path,
        )
        plot_g2(
            g2,
            save_path=save_path,
            title=f"Simulation index {args.index} | noneq={noneq_value:.6f}",
        )
        print(f"[plot-g2] wrote {save_path}")
        print(
            "[plot-g2] "
            f"index={args.index} id={row.get('id', 'n/a')} "
            f"gamma={float(row['gamma']):.6e} "
            f"D={float(row['D']):.6e} "
            f"GB_conc={float(row['GB_conc']):.6f} "
            f"T={float(row['T']):.4f} "
            f"noneq={noneq_value:.6f}"
        )
        return

    if args.name is None:
        raise ValueError("--name is required when --source exp")
    if args.shot_index is None:
        raise ValueError("--shot-index is required when --source exp")
    exp_result = prepare_experiment_display_g2(
        raw_data_dir=args.raw_data_dir,
        file_name=args.name,
        shot_index=args.shot_index,
        crop_size=args.crop_size,
        coarse_size=args.coarse_size,
        crop_step=args.crop_step,
        crop_policy=args.crop_policy,
        crop_aggregation=args.crop_aggregation,
    )
    save_path = default_plot_path(
        source="exp_g2",
        identifier=f"{Path(str(args.name)).stem}_shot{args.shot_index:02d}",
        output_path=args.output_path,
    )
    plot_g2(
        exp_result["g2"],
        save_path=save_path,
        title=(
            f"{Path(str(args.name)).stem} | shot {args.shot_index} | "
            f"noneq={float(exp_result['nonequilibrium_measure']):.6f}"
        ),
    )
    print(f"[plot-g2] wrote {save_path}")
    print(
        "[plot-g2] "
        f"name={Path(str(args.name)).stem} "
        f"shot={args.shot_index} "
        f"T={float(exp_result['temperature_k']):.4f} "
        f"noneq={float(exp_result['nonequilibrium_measure']):.6f} "
        f"crop_starts={exp_result['crop_starts']}"
    )


def run_sample_sim_quartiles_command(args: argparse.Namespace) -> None:
    """Sample one simulation index from each nonequilibrium quartile."""
    sim_df = ensure_nonequilibrium_column(load_simulation_manifest(args.sim_root))
    quartile_codes = pd.qcut(
        sim_df["nonequilibrium_measure"],
        q=4,
        labels=False,
        duplicates="drop",
    )
    if quartile_codes.nunique(dropna=True) != 4:
        raise RuntimeError(
            "Could not construct four non-empty nonequilibrium quartiles from the simulation manifest"
        )
    sim_df = sim_df.copy()
    sim_df["quartile"] = quartile_codes.astype(int) + 1

    rng = np.random.default_rng(args.seed)
    selected_rows: list[dict[str, object]] = []
    for quartile in range(1, 5):
        quartile_df = sim_df.loc[sim_df["quartile"] == quartile]
        selected_idx = int(rng.integers(0, len(quartile_df)))
        selected_row = quartile_df.iloc[selected_idx]
        selected_rows.append(
            {
                "quartile": quartile,
                "dataset_index": int(selected_row.name),
                "id": int(selected_row["id"]) if "id" in selected_row.index else None,
                "nonequilibrium_measure": float(selected_row["nonequilibrium_measure"]),
                "gamma": float(selected_row["gamma"]),
                "D": float(selected_row["D"]),
                "GB_conc": float(selected_row["GB_conc"]),
                "T": float(selected_row["T"]),
                "path": str(selected_row["path"]),
            }
        )

    selected_df = pd.DataFrame(selected_rows).sort_values("quartile").reset_index(drop=True)
    print(selected_df.to_string(index=False, float_format=lambda value: f"{value:.6e}"))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(args.output_csv, index=False)
        print(f"[sim-quartiles] wrote {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone inference utilities for plotting phase diagrams and related diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_sim_parser = subparsers.add_parser(
        "infer-sim",
        help="Load a vanilla/XPCS checkpoint, infer parameters on one simulation index, and optionally forward-plot g2.",
    )
    infer_sim_parser.add_argument(
        "--model-type",
        choices=MODEL_TYPE_CHOICES,
        required=True,
        help="Checkpoint family to load.",
    )
    infer_sim_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to the most recent checkpoint of the requested type.",
    )
    infer_temp_group = infer_sim_parser.add_mutually_exclusive_group()
    infer_temp_group.add_argument(
        "--no-t",
        dest="no_t",
        action="store_true",
        help="Force the no-temperature checkpoint variant.",
    )
    infer_temp_group.add_argument(
        "--with-t",
        dest="no_t",
        action="store_false",
        help="Force the temperature-input checkpoint variant.",
    )
    infer_sim_parser.set_defaults(no_t=None)
    infer_sim_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Simulation dataset index to evaluate.",
    )
    infer_sim_parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("dataset/simulation"),
        help="Processed simulation dataset directory.",
    )
    infer_sim_parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string, for example cpu or cuda.",
    )
    infer_sim_parser.add_argument(
        "--forward-g2",
        action="store_true",
        help="Forward-simulate g2 from the predicted parameters and save comparison plots.",
    )
    infer_sim_parser.add_argument(
        "--forward-seed",
        type=int,
        default=42,
        help="Random seed passed to simulate_xpcs when --forward-g2 is enabled.",
    )
    infer_sim_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the forward-g2 plots. Defaults to inference_outputs/...",
    )

    plot_g2_parser = subparsers.add_parser(
        "plot-g2",
        help="Plot a simulation or experiment g2 directly, without running inference.",
    )
    plot_g2_parser.add_argument(
        "--source",
        choices=["sim", "exp"],
        required=True,
        help="Whether to load a processed simulation sample or a raw experiment shot.",
    )
    plot_g2_parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Simulation dataset index when --source sim is used.",
    )
    plot_g2_parser.add_argument(
        "--name",
        default=None,
        help="Experiment filename or filename stem when --source exp is used.",
    )
    plot_g2_parser.add_argument(
        "--shot-index",
        type=int,
        default=None,
        help="Raw experiment shot index when --source exp is used.",
    )
    plot_g2_parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("dataset/simulation"),
        help="Processed simulation dataset directory.",
    )
    plot_g2_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("exp_data"),
        help="Directory containing raw experiment .npy/.npz files.",
    )
    plot_g2_parser.add_argument("--crop-size", type=int, default=2500)
    plot_g2_parser.add_argument("--coarse-size", type=int, default=256)
    plot_g2_parser.add_argument("--crop-step", type=int, default=100)
    plot_g2_parser.add_argument(
        "--crop-policy",
        choices=["top-left", "all-diagonal"],
        default="top-left",
        help="How to crop raw experiment data before plotting.",
    )
    plot_g2_parser.add_argument(
        "--crop-aggregation",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate multiple diagonal crops when --crop-policy all-diagonal is used.",
    )
    plot_g2_parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to inference_outputs/...",
    )

    quartile_parser = subparsers.add_parser(
        "sample-sim-quartiles",
        help="Split the simulation set into four nonequilibrium quartiles and sample one index from each.",
    )
    quartile_parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("dataset/simulation"),
        help="Processed simulation dataset directory.",
    )
    quartile_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling one row per quartile.",
    )
    quartile_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the sampled quartile summary.",
    )

    phase_parser = subparsers.add_parser(
        "phase-diagrams",
        help="Plot simulation nonequilibrium phase diagrams and overlay experiment inference results.",
    )
    phase_parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Evaluation results directory containing per-file inference CSVs.",
    )
    phase_parser.add_argument(
        "--simulation-manifest",
        type=Path,
        default=Path("dataset/simulation/manifest_with_non_equ.csv"),
        help="Simulation manifest containing gamma, D, GB_conc, and nonequilibrium_measure.",
    )
    phase_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for phase-diagram PDFs. Defaults to <results-dir>/phase_diagrams.",
    )
    phase_parser.add_argument(
        "--model",
        choices=["auto", "adv", "vanilla", "both"],
        default="auto",
        help="Which prediction columns to overlay from the results directory.",
    )
    phase_parser.add_argument(
        "--aggregate-by",
        choices=["file", "shot"],
        default="file",
        help="Whether to overlay one point per result CSV or one point per shot row.",
    )
    phase_parser.add_argument(
        "--shot-index",
        type=int,
        default=None,
        help="Optional shot index to keep when --aggregate-by shot is used.",
    )
    phase_parser.add_argument(
        "--range-mode",
        choices=PHASE_RANGE_MODE_CHOICES,
        default="stats",
        help=(
            "Coordinate ranges for phase diagrams. `stats` uses specs from "
            "the simulation dataset stats.json; `data` uses the sampled "
            "simulation manifest domain; `fixed` uses the legacy hard-coded "
            "plot ranges."
        ),
    )
    phase_parser.add_argument(
        "--split-by-material",
        action="store_true",
        help=(
            "Write separate phase-diagram outputs per result subdirectory "
            "(for example, one material-dose folder at a time)."
        ),
    )
    phase_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("exp_data"),
        help="Raw experimental data directory used to reconstruct shot-level g2 diagnostics.",
    )
    phase_parser.add_argument(
        "--crop-size",
        type=int,
        default=2500,
        help="Raw crop size used during experiment evaluation when rebuilding representative g2 plots.",
    )
    phase_parser.add_argument(
        "--coarse-size",
        type=int,
        default=256,
        help="Coarse-grained size used during experiment evaluation when rebuilding representative g2 plots.",
    )

    return parser.parse_args()


def parse_result_temperature(path: Path) -> float:
    match = re.match(r"^(.*_dose\d+)_T(-?\d+(?:\.\d+)?)C$", path.stem)
    if match is None:
        raise ValueError(f"Unexpected results filename format: {path.name}")
    return float(match.group(2))


def iter_result_csvs(results_dir: Path) -> list[Path]:
    csv_paths = sorted(path for path in results_dir.glob("*/*.csv"))
    valid_paths: list[Path] = []
    for csv_path in csv_paths:
        try:
            parse_result_temperature(csv_path)
        except ValueError:
            continue
        valid_paths.append(csv_path)
    return valid_paths


def detect_available_models(result_csvs: list[Path]) -> list[str]:
    available: list[str] = []
    for model_name, column_map in MODEL_COLUMN_MAP.items():
        required_columns = set(column_map.values())
        for csv_path in result_csvs:
            df = pd.read_csv(csv_path, nrows=1)
            if required_columns.issubset(df.columns):
                available.append(model_name)
                break
    return available


def load_overlay_points(
    results_dir: Path,
    model_name: str,
    aggregate_by: str,
    shot_index: int | None = None,
) -> pd.DataFrame:
    column_map = MODEL_COLUMN_MAP[model_name]
    rows: list[dict[str, object]] = []

    for csv_path in iter_result_csvs(results_dir):
        df = pd.read_csv(csv_path)
        if not set(column_map.values()).issubset(df.columns):
            continue

        df = df.copy()
        df["source_csv"] = str(csv_path)
        df["temperature_c"] = parse_result_temperature(csv_path)
        df["material_subdir"] = csv_path.parent.name

        if aggregate_by == "file":
            rows.append(
                {
                    "source_csv": str(csv_path),
                    "material_subdir": csv_path.parent.name,
                    "file_name": df["file_name"].iloc[0] if "file_name" in df.columns else csv_path.name,
                    "temperature_k": float(df["temperature_k"].mean()) if "temperature_k" in df.columns else float(df["temperature_c"].mean() + 273.15),
                    "temperature_c": float(df["temperature_c"].mean()),
                    "gamma": float(df[column_map["gamma"]].mean()),
                    "D": float(df[column_map["D"]].mean()),
                    "GB_conc": float(df[column_map["GB_conc"]].mean()),
                    "experimental_nonequilibrium_measure": float(df["nonequilibrium_measure"].mean()) if "nonequilibrium_measure" in df.columns else float("nan"),
                    "num_rows": int(len(df)),
                }
            )
        else:
            renamed = df.rename(
                columns={
                    column_map["gamma"]: "gamma",
                    column_map["D"]: "D",
                    column_map["GB_conc"]: "GB_conc",
                    "nonequilibrium_measure": "experimental_nonequilibrium_measure",
                }
            )
            if shot_index is not None:
                if "shot_index" not in renamed.columns:
                    raise ValueError(
                        "Shot filtering requires shot-level result CSVs with a 'shot_index' column"
                    )
                renamed = renamed.loc[renamed["shot_index"] == shot_index].copy()
            if renamed.empty:
                continue
            keep_columns = [
                col for col in [
                    "source_csv",
                    "material_subdir",
                    "file_name",
                    "shot_index",
                    "temperature_k",
                    "temperature_c",
                    "num_crops",
                    "crop_policy",
                    "crop_aggregation",
                    "crop_start_min",
                    "crop_start_max",
                    "gamma",
                    "D",
                    "GB_conc",
                    "experimental_nonequilibrium_measure",
                ]
                if col in renamed.columns
            ]
            rows.extend(renamed[keep_columns].to_dict("records"))

    return pd.DataFrame(rows)


def load_raw_experiment_array(path: Path) -> np.ndarray:
    """
    Load one raw experimental XPCS array from disk.

    `.npz` files are read from the `g12` key, while `.npy` files are loaded
    directly. Two-dimensional arrays are promoted to shape `[H, W, 1]`.
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
    return np.asarray(array, dtype=np.float32)


def resolve_raw_data_file(raw_data_dir: Path, file_name: str) -> Path:
    """
    Resolve one result row's raw experiment filename under `raw_data_dir`.
    """
    direct_path = raw_data_dir / file_name
    if direct_path.exists():
        return direct_path

    stem = Path(file_name).stem
    matches = sorted(raw_data_dir.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find raw experiment file for {file_name} in {raw_data_dir}"
        )
    if len(matches) > 1:
        preferred_suffixes = [".npz", ".npy"]
        for suffix in preferred_suffixes:
            for match in matches:
                if match.suffix.lower() == suffix:
                    return match
    return matches[0]


def infer_crop_starts(row: pd.Series) -> list[int]:
    """
    Reconstruct the crop starts represented by one shot-level result row.

    The evaluation path stores only the min/max crop offsets and the crop count,
    so we rebuild the evenly spaced diagonal offsets used during inference.
    """
    num_crops = int(row.get("num_crops", 1))
    crop_start_min = int(round(float(row.get("crop_start_min", 0))))
    crop_start_max = int(round(float(row.get("crop_start_max", crop_start_min))))
    if num_crops <= 1 or crop_start_max <= crop_start_min:
        return [crop_start_min]
    if num_crops == 2:
        return [crop_start_min, crop_start_max]
    return [
        int(round(value))
        for value in np.linspace(crop_start_min, crop_start_max, num=num_crops)
    ]


def aggregate_g2_crops(
    crops: list[torch.Tensor],
    aggregation: str,
) -> torch.Tensor:
    """
    Aggregate multiple coarse-grained crop tensors into one representative `g2`.
    """
    if not crops:
        raise ValueError("Expected at least one crop tensor to aggregate")
    if len(crops) == 1:
        return crops[0]

    stacked = torch.stack(crops, dim=0)
    if aggregation == "mean":
        return stacked.mean(dim=0)
    if aggregation == "median":
        return torch.median(stacked, dim=0).values
    raise ValueError(f"Unsupported crop aggregation: {aggregation}")


def reconstruct_shot_g2(
    row: pd.Series,
    raw_data_dir: Path,
    crop_size: int,
    coarse_size: int,
) -> dict[str, object]:
    """
    Rebuild a representative coarse-grained `g2` image for one shot-level row.

    The representative `g2` is formed from the same diagonal crop locations used
    during evaluation, with the row's crop aggregation applied after
    coarse-graining and visual normalization.
    """
    raw_path = resolve_raw_data_file(raw_data_dir, str(row["file_name"]))
    raw_data = load_raw_experiment_array(raw_path)
    shot_index = int(row["shot_index"])
    if shot_index < 0 or shot_index >= raw_data.shape[-1]:
        raise IndexError(
            f"Shot index {shot_index} is out of bounds for {raw_path.name} "
            f"with {raw_data.shape[-1]} shot(s)"
        )

    spatial_height, spatial_width = raw_data.shape[:2]
    already_cropped = spatial_height <= crop_size or spatial_width <= crop_size
    crop_starts = [0] if already_cropped else infer_crop_starts(row)
    crop_tensors: list[torch.Tensor] = []
    for crop_start in crop_starts:
        if already_cropped:
            crop_end_h = spatial_height
            crop_end_w = spatial_width
        else:
            crop_end_h = crop_start + crop_size
            crop_end_w = crop_start + crop_size
        if (
            crop_start < 0
            or crop_end_h > spatial_height
            or crop_end_w > spatial_width
        ):
            raise ValueError(
                f"Crop [{crop_start}:{crop_end_h}] is out of bounds for {raw_path.name} "
                f"with shape {raw_data.shape}"
            )
        crop = torch.tensor(
            raw_data[crop_start:crop_end_h, crop_start:crop_end_w, shot_index],
            dtype=torch.float32,
        )
        if tuple(crop.shape) == (coarse_size, coarse_size):
            coarse_crop = crop.to(torch.float32)
        else:
            coarse_crop = coarse_grain_g2(
                crop,
                target_size=(coarse_size, coarse_size),
            ).to(torch.float32)
        crop_tensors.append(normalize_g2(coarse_crop, min_val=1.0, max_val=1.2))

    aggregation = str(row.get("crop_aggregation", "mean"))
    g2 = aggregate_g2_crops(crop_tensors, aggregation=aggregation)
    return {
        "raw_path": raw_path,
        "crop_starts": crop_starts,
        "g2": g2,
        "recomputed_nonequilibrium_measure": float(nonequilibrium_measure(g2)),
    }


def write_shot_g2_diagnostics(
    overlay_df: pd.DataFrame,
    output_dir: Path,
    raw_data_dir: Path,
    crop_size: int,
    coarse_size: int,
) -> None:
    """
    Save representative `g2` and nonequilibrium plots for the selected shot rows.
    """
    if overlay_df.empty or "shot_index" not in overlay_df.columns:
        return

    diagnostics_dir = output_dir / "shot_g2_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    sort_columns = [
        col
        for col in ["temperature_k", "temperature_c", "file_name", "shot_index"]
        if col in overlay_df.columns
    ]
    ordered_df = overlay_df.sort_values(sort_columns).reset_index(drop=True)

    for _, row in ordered_df.iterrows():
        reconstructed = reconstruct_shot_g2(
            row=row,
            raw_data_dir=raw_data_dir,
            crop_size=crop_size,
            coarse_size=coarse_size,
        )
        file_stem = Path(str(row["file_name"])).stem
        shot_index = int(row["shot_index"])
        prefix = f"{file_stem}_shot{shot_index:02d}"
        g2_path = diagnostics_dir / f"{prefix}_g2.pdf"
        noneq_path = diagnostics_dir / f"{prefix}_nonequilibrium_measure.pdf"
        reported_noneq = row.get("experimental_nonequilibrium_measure", float("nan"))
        title = (
            f"{file_stem} | shot {shot_index} | "
            f"reported non-eq={float(reported_noneq):.4f}"
        )
        plot_g2(
            reconstructed["g2"],
            save_path=g2_path,
            title=title,
        )
        recomputed_noneq = plot_nonequilibrium_measure(
            reconstructed["g2"],
            save_path=noneq_path,
        )

        summary_rows.append(
            {
                "material_subdir": row.get("material_subdir"),
                "file_name": row["file_name"],
                "shot_index": shot_index,
                "temperature_k": row.get("temperature_k"),
                "temperature_c": row.get("temperature_c"),
                "raw_path": str(reconstructed["raw_path"]),
                "crop_starts": ",".join(str(value) for value in reconstructed["crop_starts"]),
                "crop_aggregation": row.get("crop_aggregation"),
                "reported_nonequilibrium_measure": reported_noneq,
                "recomputed_nonequilibrium_measure": recomputed_noneq,
                "g2_path": str(g2_path),
                "nonequilibrium_plot_path": str(noneq_path),
            }
        )

    summary_path = diagnostics_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[phase] wrote {summary_path}")


def enrich_overlay_points(
    overlay_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add metadata and interpolated phase-map values to the overlay rows.
    """
    overlay_df = add_overlay_metadata(overlay_df, sim_df)
    overlay_df["phase_map_nonequilibrium_measure"] = interpolate_phase_measure_3d(
        df=sim_df,
        query_df=overlay_df,
    )
    if "experimental_nonequilibrium_measure" in overlay_df.columns:
        overlay_df["nonequilibrium_measure_delta"] = (
            overlay_df["experimental_nonequilibrium_measure"]
            - overlay_df["phase_map_nonequilibrium_measure"]
        )
    return overlay_df


def write_phase_outputs(
    sim_df: pd.DataFrame,
    overlay_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    aggregate_by: str,
    range_mode: str = "stats",
    stats_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Write the overlay CSV plus the phase-map figures for one output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_csv_path = output_dir / f"overlay_points_{model_name}_{aggregate_by}.csv"
    overlay_df.to_csv(overlay_csv_path, index=False)
    print(f"[phase] wrote {overlay_csv_path}")
    total_points = len(overlay_df)
    in_domain_points = int(overlay_df["within_simulation_domain"].sum())
    print(
        "[phase] "
        f"{model_name}: {in_domain_points}/{total_points} overlay points are fully inside the "
        "simulation parameter domain"
    )

    comparison_path = output_dir / f"nonequilibrium_comparison_{model_name}_{aggregate_by}.pdf"
    comparison_stats = plot_nonequilibrium_comparison(
        overlay_df=overlay_df,
        save_path=comparison_path,
        phase_value_column="phase_map_nonequilibrium_measure",
        title=f"{model_name} inference: experiment vs phase-map non-eq",
    )
    print(
        "[phase] "
        f"{model_name} noneq comparison: "
        f"plotted {comparison_stats['plotted_overlay_points']} comparable points, "
        f"omitted {comparison_stats['omitted_overlay_points']} points"
    )
    print(f"[phase] wrote {comparison_path}")

    for spec in PHASE_PLOT_SPECS:
        if range_mode == "fixed":
            x_range = spec.get("x_range")
            y_range = spec.get("y_range")
        elif range_mode == "stats" and stats_ranges is not None:
            x_range = stats_ranges.get(spec["x"])
            y_range = stats_ranges.get(spec["y"])
        else:
            x_range = None
            y_range = None
        xticks = spec["xticks"] if range_mode == "fixed" else None
        yticks = spec["yticks"] if range_mode == "fixed" else None
        xlabels = spec["xlabels"] if range_mode == "fixed" else None
        ylabels = spec["ylabels"] if range_mode == "fixed" else None
        save_path = output_dir / f"{spec['filename']}_{model_name}.pdf"
        plot_stats = plot_nonequilibrium_phase_map(
            df=sim_df,
            save_path=save_path,
            x=spec["x"],
            y=spec["y"],
            xscale=spec["xscale"],
            yscale=spec["yscale"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            overlay_df=overlay_df,
            overlay_x=spec["x"],
            overlay_y=spec["y"],
            overlay_label=f"{model_name} inference",
            xticks=xticks,
            yticks=yticks,
            xlabels=xlabels,
            ylabels=ylabels,
            x_range=x_range,
            y_range=y_range,
        )
        print(
            "[phase] "
            f"{model_name} {spec['x']}-{spec['y']}: "
            f"plotted {plot_stats['plotted_overlay_points']} in-range overlay points, "
            f"omitted {plot_stats['omitted_overlay_points']} out-of-range points"
        )
        print(f"[phase] wrote {save_path}")

        exp_overlay_path = output_dir / (
            f"{spec['filename']}_{model_name}_experiment_nonequilibrium.pdf"
        )
        exp_overlay_stats = plot_nonequilibrium_phase_map_exp_overlay(
            df=sim_df,
            save_path=exp_overlay_path,
            x=spec["x"],
            y=spec["y"],
            xscale=spec["xscale"],
            yscale=spec["yscale"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            overlay_df=overlay_df,
            overlay_x=spec["x"],
            overlay_y=spec["y"],
            xticks=xticks,
            yticks=yticks,
            xlabels=xlabels,
            ylabels=ylabels,
            x_range=x_range,
            y_range=y_range,
        )
        print(
            "[phase] "
            f"{model_name} {spec['x']}-{spec['y']} exp-noneq overlay: "
            f"plotted {exp_overlay_stats['plotted_overlay_points']} in-range overlay points, "
            f"omitted {exp_overlay_stats['omitted_overlay_points']} out-of-range points"
        )
        print(f"[phase] wrote {exp_overlay_path}")


def plot_phase_diagrams(
    results_dir: Path,
    simulation_manifest: Path,
    output_dir: Path | None,
    model_names: list[str],
    aggregate_by: str,
    shot_index: int | None = None,
    split_by_material: bool = False,
    raw_data_dir: Path = Path("exp_data"),
    crop_size: int = 2500,
    coarse_size: int = 256,
    range_mode: str = "stats",
) -> None:
    sim_df = pd.read_csv(simulation_manifest)
    required_sim_columns = {"gamma", "D", "GB_conc", "nonequilibrium_measure"}
    if not required_sim_columns.issubset(sim_df.columns):
        raise ValueError(
            f"Simulation manifest must contain {sorted(required_sim_columns)}, got {sim_df.columns.tolist()}"
        )
    stats_ranges = (
        load_phase_ranges_from_stats(simulation_manifest)
        if range_mode == "stats"
        else None
    )

    for model_name in model_names:
        overlay_df = load_overlay_points(
            results_dir=results_dir,
            model_name=model_name,
            aggregate_by=aggregate_by,
            shot_index=shot_index,
        )
        if overlay_df.empty:
            print(f"[phase] no overlay points found for model={model_name} in {results_dir}")
            continue
        overlay_df = enrich_overlay_points(overlay_df, sim_df)

        if split_by_material:
            material_output_name = f"phase_diagrams_{aggregate_by}"
            if shot_index is not None:
                material_output_name = f"{material_output_name}_idx{shot_index}"
            for material_subdir, material_df in overlay_df.groupby("material_subdir", sort=True):
                if output_dir is None:
                    material_output_dir = results_dir / material_subdir / material_output_name
                else:
                    material_output_dir = output_dir / material_subdir
                write_phase_outputs(
                    sim_df=sim_df,
                    overlay_df=material_df.copy(),
                    output_dir=material_output_dir,
                    model_name=model_name,
                    aggregate_by=aggregate_by,
                    range_mode=range_mode,
                    stats_ranges=stats_ranges,
                )
                if shot_index is not None:
                    write_shot_g2_diagnostics(
                        overlay_df=material_df.copy(),
                        output_dir=material_output_dir,
                        raw_data_dir=raw_data_dir,
                        crop_size=crop_size,
                        coarse_size=coarse_size,
                    )
        else:
            if output_dir is None:
                raise ValueError("output_dir must be provided when split_by_material is False")
            write_phase_outputs(
                sim_df=sim_df,
                overlay_df=overlay_df,
                output_dir=output_dir,
                model_name=model_name,
                aggregate_by=aggregate_by,
                range_mode=range_mode,
                stats_ranges=stats_ranges,
            )


def main() -> None:
    args = parse_args()

    if args.command == "infer-sim":
        run_infer_sim_command(args)
        return

    if args.command == "plot-g2":
        run_plot_g2_command(args)
        return

    if args.command == "sample-sim-quartiles":
        run_sample_sim_quartiles_command(args)
        return

    if args.command == "phase-diagrams":
        results_dir = args.results_dir
        if args.shot_index is not None and args.aggregate_by != "shot":
            raise ValueError("--shot-index requires --aggregate-by shot")
        if args.split_by_material:
            output_dir = args.output_dir
        else:
            default_output_name = f"phase_diagrams_{args.aggregate_by}"
            if args.shot_index is not None:
                default_output_name = f"{default_output_name}_idx{args.shot_index}"
            output_dir = args.output_dir or (results_dir / default_output_name)
        result_csvs = iter_result_csvs(results_dir)
        if not result_csvs:
            raise FileNotFoundError(f"No compatible result CSVs found in {results_dir}")

        available_models = detect_available_models(result_csvs)
        if args.model == "auto":
            model_names = available_models
        elif args.model == "both":
            model_names = [name for name in ["adv", "vanilla"] if name in available_models]
        else:
            model_names = [args.model]

        if not model_names:
            raise RuntimeError(
                f"No matching prediction columns found in {results_dir}; available models: {available_models}"
            )

        plot_phase_diagrams(
            results_dir=results_dir,
            simulation_manifest=args.simulation_manifest,
            output_dir=output_dir,
            model_names=model_names,
            aggregate_by=args.aggregate_by,
            shot_index=args.shot_index,
            split_by_material=args.split_by_material,
            raw_data_dir=args.raw_data_dir,
            crop_size=args.crop_size,
            coarse_size=args.coarse_size,
            range_mode=args.range_mode,
        )


if __name__ == "__main__":
    main()
