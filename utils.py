import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
try:
    import umap.umap_ as umap
except ModuleNotFoundError:
    umap = None
from pathlib import Path
from typing import (
    List, Tuple, Dict, Optional, Callable, Literal, Iterable,
)
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import Normalize, to_rgba

class HandlerMultiSquare(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        colors = orig_handle              # tuple/list of color strings
        n = len(colors)
        w = width / n                     # each square gets 1/n of the width
        artists = []
        for i, c in enumerate(colors):
            r = Rectangle((xdescent + i*w, ydescent),
                          w, height,
                          transform=trans,
                          facecolor=c,
                          edgecolor='none')
            artists.append(r)
        return artists

def _rescale(
    data: np.ndarray,
    min_val: float,
    max_val: float,
    min_ratio: float = 0.50,
    max_ratio: float = 0.92,
) -> np.ndarray:
    """Rescale the data to a specified range [min_ratio, max_ratio]
    based on the given min_val and max_val."""
    data_rescaled = (data - min_val) / (max_val - min_val)
    data_rescaled = data_rescaled * (max_ratio - min_ratio) + min_ratio
    return data_rescaled

def _normalize(
    data: np.ndarray,
    scale: Literal['linear', 'log'] = 'linear',
) -> np.ndarray:
    """Normalize the data to [0, 1] range."""
    if scale == 'linear':
        min_val = data.min()
        max_val = data.max()
        data_normalized = (data - min_val) / (max_val - min_val)
    elif scale == 'log':
        log_data = np.log(data)
        min_val = log_data.min()
        max_val = log_data.max()
        data_normalized = (log_data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    return data_normalized

def _denormalize(
    data_normalized: np.ndarray,
    original_min: float,
    original_max: float,
    scale: Literal['linear', 'log'] = 'linear',
) -> np.ndarray:
    """Denormalize the data from [0, 1] range to original range."""
    if scale == 'linear':
        data = data_normalized * (original_max - original_min) + original_min
    elif scale == 'log':
        log_data = data_normalized * (np.log(original_max) - np.log(original_min)) + np.log(original_min)
        data = np.exp(log_data)
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    return data

def plot_bar(
    params: Dict[str, float],
    max_param: float,
    xname: str,
    save_path: Path,
    title: str = "",
) -> None:
    """
    Plot the bar chart of parameters.
    
    Args:
        params (dict[str, float]): The parameter values.
        max_param (float): The maximum value for normalization.
        xname (str): The name of the x-axis.
        save_path (Path): The path to save the plot.
        title (str): The title of the plot.
    """
    plt.rcParams["font.family"] = "arial"  # the label docations
    plt.rcParams["font.size"] = 24
    fig, ax = plt.subplots(figsize=(6, 4))
    keys, values = zip(*params.items())
    colors = ["#96ced3", "#e9c54e", "#e64b35"]
    rects = ax.bar(keys, np.array(values) / max_param, label=keys, color=colors)
    values = [f"{v:.3f}" for v in values]
    ax.bar_label(rects, labels=values, padding=3, fmt="%.3f")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(xname)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
def plot_multi_bar(
    params_adv: Dict[str, float],
    params_van: Dict[str, float],
    max_param: float,
    min_param: float,
    xname: str,
    save_path: Path,
    title: str = "",
    legend: bool = True,
) -> None:
    """
    Plot the bar chart of parameters for adversarial and vanilla model.
    
    Args:
        params_adv (dict[str, float]): The parameter values for adversarial model.
        params_van (dict[str, float]): The parameter values for vanilla model.
        max_param (float): The maximum value for normalization.
        min_param (float): The minimum value for normalization.
        xname (str): The name of the x-axis.
        save_path (Path): The path to save the plot.
        title (str): The title of the plot.
    """
    plt.rcParams["font.family"] = "arial"  
    plt.rcParams["font.size"] = 24
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'arial'
    # plt.rcParams['mathtext.it'] = 'arial:italic'
    # plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    width = 0.4
    colors = ("#96ced3", "#e9c54e", "#e64b35")
    colors_alpha = tuple(to_rgba(c, alpha=0.3) for c in colors)
    x = np.arange(len(params_adv))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Vanilla model
    keys, values_van = zip(*params_van.items())
    norm_values_van = _rescale(np.array(values_van), min_param, max_param)
    rects_van = ax.bar(x + width, norm_values_van, width, label="Vanilla", color=colors_alpha)
    values_van = [f"{v:.3f}" for v in values_van]
    ax.bar_label(rects_van, labels=values_van, padding=3, fmt="%.3f")
    # Adversarial model
    keys, values_adv = zip(*params_adv.items())
    norm_values_adv = _rescale(np.array(values_adv), min_param, max_param)
    rects_adv = ax.bar(x, norm_values_adv, width, label="Adversarial", color=colors)
    values_adv = [f"{v:.3f}" for v in values_adv]
    ax.bar_label(rects_adv, labels=values_adv, padding=3, fmt="%.3f")
    
    ax.set_xticks(x + width / 2, keys)
    ax.set_yticks([])
    ax.set_xlabel(xname)
    ax.set_ylim(0.0, 1.0)
    if legend:
        ax.legend(
            [colors, colors_alpha],
            ["Adversarial", "Vanilla"],
            handler_map={tuple: HandlerMultiSquare()},
            loc='upper right', bbox_to_anchor=(1.3, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
def plot_multi_bar_v2(
    params_adv: Dict[str, float],
    params_van: Dict[str, float],
    max_param: float,
    min_param: float,
    xname: str,
    save_path: Path,
    title: str = "",
    legend: bool = True,
    tight_layout: bool = True,
) -> None:
    """
    Plot the bar chart of parameters for adversarial and vanilla model.
    Compared to plot_multi_bar, the bars are plotted such that data from the same
    model (vanilla or adversarial) are grouped together, rather than by temperature.
    
    Args:
        params_adv (dict[str, float]): The parameter values for adversarial model.
        params_van (dict[str, float]): The parameter values for vanilla model.
        max_param (float): The maximum value for normalization.
        min_param (float): The minimum value for normalization.
        xname (str): The name of the x-axis.
        save_path (Path): The path to save the plot.
        title (str): The title of the plot.
    """
    plt.rcParams["font.family"] = "arial"  
    plt.rcParams["font.size"] = 24
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'arial'
    # plt.rcParams['mathtext.it'] = 'arial:italic'
    # plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    width = 0.35
    colors = ("#96ced3", "#e9c54e", "#e64b35")
    colors_alpha = tuple(to_rgba(c, alpha=0.3) for c in colors)
    x = np.array([0.0, 1.2])
    
    fig, ax = plt.subplots(figsize=(11, 6))
    temps = list(params_adv.keys())
    adv_values = list(params_adv.values())
    van_values = list(params_van.values())
    norm_values_adv = _rescale(np.array(adv_values), min_param, max_param)
    norm_values_van = _rescale(np.array(van_values), min_param, max_param)
    for i, (temp, norm_adv, norm_van, adv_value, van_value, color, color_alpha) in enumerate(
        zip(temps, norm_values_adv, norm_values_van, adv_values, van_values, colors, colors_alpha)
    ):
        rects = ax.bar(x + i * width, [norm_van, norm_adv], width, color=[color_alpha, color], label=temp)
        values = [f"{van_value:.3f}", f"{adv_value:.3f}"]
        ax.bar_label(rects, labels=values, padding=3, fmt="%.3f")
        
    ax.set_xticks(x + width, ["Vanilla", "Adversarial"])
    ax.set_yticks([])
    ax.set_xlabel(xname, fontsize=24)
    ax.set_ylim(0.0, 1.1)
    if legend:
        ax.legend(
            [(color_alpha, color) for color, color_alpha in zip(colors, colors_alpha)],
            temps,
            handler_map={tuple: HandlerMultiSquare()},
            loc='upper right', bbox_to_anchor=(1.35, 1)
        )
    if title:
        plt.title(title)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_single_model_multi_bar_v2(
    params: Dict[str, float],
    max_param: float,
    min_param: float,
    xname: str,
    save_path: Path,
    model_name: str,
    title: str = "",
    legend: bool = True,
    tight_layout: bool = True,
) -> None:
    """
    Plot temperature-grouped bars for a single model across one parameter.

    Args:
        params: Temperature label to parameter value mapping.
        max_param: Maximum raw value across plotted temperatures.
        min_param: Minimum raw value across plotted temperatures.
        xname: X-axis label.
        save_path: Output figure path.
        model_name: Display label for the model, for example `Vanilla`.
        title: Optional figure title.
        legend: Whether to draw the temperature legend.
        tight_layout: Whether to call `plt.tight_layout()`.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24

    width = 0.35
    colors = ("#96ced3", "#e9c54e", "#e64b35")
    x = np.array([0.0])

    fig, ax = plt.subplots(figsize=(8, 6))
    temps = list(params.keys())
    values = list(params.values())
    if np.isclose(max_param, min_param):
        norm_values = np.full(len(values), 0.8)
    else:
        norm_values = _rescale(np.array(values), min_param, max_param)

    for i, (temp, norm_value, value) in enumerate(zip(temps, norm_values, values)):
        color = colors[i % len(colors)]
        rects = ax.bar(x + i * width, [norm_value], width, color=[color], label=temp)
        ax.bar_label(rects, labels=[f"{value:.3f}"], padding=3, fmt="%.3f")

    center = x + width * max(len(temps) - 1, 0) / 2
    ax.set_xticks(center, [model_name])
    ax.set_yticks([])
    ax.set_xlabel(xname, fontsize=24)
    ax.set_ylim(0.0, 1.1)
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
    if title:
        plt.title(title)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_parameter_pair_comparison(
    df: pd.DataFrame,
    save_path: Path,
    x_adv: str,
    y_adv: str,
    x_van: str,
    y_van: str,
    xname: str,
    yname: str,
    x_scale_factor: float = 1.0,
    y_scale_factor: float = 1.0,
    title: str = "",
    show_colorbar: bool = True,
) -> None:
    """
    Plot shot-level parameter scatters for vanilla and adversarial models
    side by side, colored by experimental temperature.

    The same axis limits are used for both panels so the clustering/trend
    differences can be compared directly. Temperature is encoded with one
    continuous color scale shared across both panels.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 22
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'

    required_columns = {x_adv, y_adv, x_van, y_van}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns for scatter comparison: {missing_columns}")
    if "temperature_k" in df.columns:
        temperature_values = df["temperature_k"].to_numpy(dtype=float)
    elif "temperature_c" in df.columns:
        temperature_values = df["temperature_c"].to_numpy(dtype=float) + 273.15
    else:
        raise ValueError("Missing temperature column for scatter comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    model_specs = (
        ("Vanilla", axes[0], x_van, y_van),
        ("Adversarial", axes[1], x_adv, y_adv),
    )
    temp_min = float(np.min(temperature_values))
    temp_max = float(np.max(temperature_values))
    if np.isclose(temp_min, temp_max):
        temp_min -= 0.5
        temp_max += 0.5
    temp_norm = Normalize(vmin=temp_min, vmax=temp_max)
    temp_cmap = plt.cm.plasma

    x_all = np.concatenate([
        df[x_adv].to_numpy(dtype=float) * x_scale_factor,
        df[x_van].to_numpy(dtype=float) * x_scale_factor,
    ])
    y_all = np.concatenate([
        df[y_adv].to_numpy(dtype=float) * y_scale_factor,
        df[y_van].to_numpy(dtype=float) * y_scale_factor,
    ])

    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = 0.08 * (x_max - x_min) if x_max > x_min else max(abs(x_min) * 0.08, 0.1)
    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else max(abs(y_min) * 0.08, 0.1)

    for model_name, ax, x_col, y_col in model_specs:
        for temperature_k in np.sort(np.unique(temperature_values)):
            temp_mask = np.isclose(temperature_values, temperature_k)
            temp_df = df.loc[temp_mask]
            x_values = temp_df[x_col].to_numpy(dtype=float) * x_scale_factor
            y_values = temp_df[y_col].to_numpy(dtype=float) * y_scale_factor
            ax.scatter(
                x_values,
                y_values,
                s=140,
                alpha=0.85,
                c=np.full(len(temp_df), float(temperature_k)),
                cmap=temp_cmap,
                norm=temp_norm,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

        ax.set_title(model_name)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.2, linewidth=0.8)

    axes[0].set_ylabel(yname)
    for ax in axes:
        ax.set_xlabel(xname)

    if show_colorbar:
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=temp_norm, cmap=temp_cmap),
            ax=axes,
            fraction=0.046,
            pad=0.08,
        )
        colorbar.set_label("Temperature (K)")

    plt.tight_layout()

    fig.subplots_adjust(
        right=0.84 if show_colorbar else 0.96,
        top=0.92,
        wspace=0.18,
    )
    plt.savefig(save_path)
    plt.close(fig)

def plot_grouped_bar(
    types: Tuple[str, ...],
    params: Dict[str, Tuple[float, ...]],
    max_params: Tuple[float, ...],
    save_path: Path,
    title: str = "",
) -> None:
    """
    Plot the grouped bar chart of parameters.
    
    Args:
        types (Tuple[str, ...]): The names of the parameters.
        params (dict[str, Tuple[float, ...]]): The parameter values for different samples.
        max_params (Tuple[float, ...]): The maximum values for each parameter type, used for normalization.
        save_path (Path): The path to save the plot.
        title (str): The title of the plot.
    """
    plt.rcParams["font.family"] = "arial"  # the label docations
    plt.rcParams["font.size"] = 24
    x = np.arange(len(types)) * 1.2
    width = 0.35  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#96ced3", "#e9c54e", "#e64b35"]
    max_params = np.array(max_params)
    
    for i, (attr_name, values) in enumerate(params.items()):
        color = colors[i % len(colors)]
        offset = width * multiplier
        norm_values = np.array(values) / max_params
        rects = ax.bar(x + offset, norm_values, width, label=attr_name, color=color)
        values = [f"{v:.3f}" for v in values]
        ax.bar_label(rects, labels=values, padding=3, fmt="%.3f")
        multiplier += 1
        
    ax.set_yticks([])
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x + width, types)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    

def plot_g2(
    g2: torch.Tensor, save_path: Path, title: str = "",
    xlabel: str = "", ylabel: str = "",
    vmin: float = 1.0, vmax: float = 1.2,
    xticks=None, yticks=None, xlabels=None, ylabels=None,
    colorbar: bool = True, cbar_ticks: List[float] = None,
    cbar_label: str = r"$g_2(t_1, t_2)$",
    cmap: str = 'viridis',
) -> None:
    """Plot and save the g2 matrix as a heatmap."""
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    plt.figure(figsize=(10, 6))
    plt.imshow(g2, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    if colorbar:
        cbar = plt.colorbar(label=f"{cbar_label}")
        if cbar_ticks is None:
            cbar.set_ticks([1.0, 1.1, 1.2])
        else:
            cbar.set_ticks(cbar_ticks)
    if title:
        plt.title(f"{title}")
    plt.xlabel(xlabel if xlabel else r"Time $t_1$ (s)")
    plt.ylabel(ylabel if ylabel else r"Time $t_2$ (s)")
    if xticks is not None:
        plt.xticks(xticks, xlabels if xlabels is not None else xticks)
    else:
        plt.xticks([])
    if yticks is not None:
        plt.yticks(yticks, ylabels if ylabels is not None else yticks)
    else:
        plt.yticks([])
    plt.tick_params(axis='both', which='both', length=0, pad=10)   # hide tick marks
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_g2_side_by_side(
    g2_left: torch.Tensor | np.ndarray,
    g2_right: torch.Tensor | np.ndarray,
    save_path: Path,
    left_title: str = "Original",
    right_title: str = "Reconstructed",
    vmin: float = 1.0,
    vmax: float = 1.2,
    cmap: str = "viridis",
    colorbar_label: str = r"$g_2(t_1, t_2)$",
) -> None:
    """
    Plot two `g2` heatmaps side by side using the same color scale.

    Args:
        g2_left: Left heatmap values.
        g2_right: Right heatmap values.
        save_path: Output figure path.
        left_title: Title for the left panel.
        right_title: Title for the right panel.
        vmin: Shared lower color limit.
        vmax: Shared upper color limit.
        cmap: Matplotlib colormap name.
        colorbar_label: Label for the shared colorbar.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 18
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'

    left = np.asarray(g2_left, dtype=np.float32)
    right = np.asarray(g2_right, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    images = []
    for ax, image_data, title in zip(
        axes,
        (left, right),
        (left_title, right_title),
    ):
        image = ax.imshow(
            image_data,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"Time $t_1$ (s)")
    axes[0].set_ylabel(r"Time $t_2$ (s)")
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
def avg_diagonal(g2: torch.Tensor) -> torch.Tensor:
    n = g2.size(0)
    k = torch.arange(n).reshape(1, n) - torch.arange(n).reshape(n, 1) # [n, n]
    diagnoal_means = torch.zeros(2*n-1, dtype=g2.dtype)
    for i in range(-n+1, n):
        diagnoal_means[i+n-1] = g2[k == i].mean()
    avg_diagonal_g2 = diagnoal_means[k + n - 1] # [n, n]
    return avg_diagonal_g2

def nonequilibrium_measure(g2: torch.Tensor, metric: Optional[Callable] = None) -> float:
    """
    **Deprecated**. Use plot_nonequilibrium_measure instead.
    """
    if g2.dim() == 3 and g2.shape[0] == 1:
        g2 = g2.squeeze(0)
    g2 = g2.to(torch.float32)
    avg_diagonal_g2 = avg_diagonal(g2)
    if metric is None:
        metric = lambda x: (x * x).sum().sqrt().item()
    return metric(g2 - avg_diagonal_g2) / metric(g2 - g2.mean())

def plot_nonequilibrium_measure(
    g2: torch.Tensor,
    save_path: Path,
    metric: Optional[Callable] = None,
    xticks=None, yticks=None, xlabels=None, ylabels=None,
) -> float:
    """Compute and return the nonequilibrium measure of the g2 matrix."""
    if g2.dim() == 3 and g2.shape[0] == 1:
        g2 = g2.squeeze(0)
    g2 = g2.to(torch.float32)
    avg_diagonal_g2 = avg_diagonal(g2)
    nonequilibrium_value = nonequilibrium_measure(g2, metric=metric)
    plot_g2(
        g2 - avg_diagonal_g2,
        save_path=save_path,
        vmin=-0.1, vmax=0.1,
        cbar_ticks=[-0.1, 0.0, 0.1],
        xticks=xticks, yticks=yticks,
        xlabels=xlabels, ylabels=ylabels,
        xlabel="Time $t_1$ (s)\n\nNonequilibrium measure: "f"{nonequilibrium_value:.4f}",
        cmap='PiYG',
        cbar_label=r"$g_2(t_1, t_2) - \langle g_2 \rangle_{\mathrm{diag}}$",
    )
    return nonequilibrium_value
    

def plot_auto_correlation(
    g2: np.ndarray | torch.Tensor,
    save_path: Path,
    n_curves: int = 5,
    title: str = "",
    legend: bool = True,
    xticks=None, yticks=None, xlabels=None, ylabels=None,
    smooth: bool = False,
) -> None:
    """Plot and save the auto-correlation curves extracted from the g2 matrix."""
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    n = g2.shape[0]
    step = n // n_curves
    fig, ax = plt.subplots(figsize=(6, 6))
    
    colors = ['#48bcbc', '#5496ce', '#b778b3', '#dc6463', '#f29742', '#e9c54e']
    times = [0, 500, 1000, 1500, 2000]
    
    for i, time in zip(range(n_curves), times):
        idx = i * step
        curve = g2[idx, idx:]
        times = np.arange(idx, n) - idx
        if smooth:
            curve = gaussian_filter1d(curve, sigma=5)
        ax.plot(times, curve, label=rf"$t_1={time}$ s", lw=2, color=colors[i % len(colors)])
    
    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(r"$f(\tau; t_1) = g_2(t_1, t_1 + \tau)$")
    if title:
        ax.set_title(f"{title}")
    if legend:
        ax.legend()
    if xticks is not None:
        ax.set_xticks(xticks, xlabels if xlabels is not None else xticks)
    else:
        ax.set_xticks([0, 1250, 2500])
    if yticks is not None:
        ax.set_yticks(yticks, ylabels if ylabels is not None else yticks)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_auto_correlation_comparison(
    g2_target: torch.Tensor | np.ndarray,
    g2_pred: torch.Tensor | np.ndarray,
    save_path: Path,
    n_curves: int = 5,
    title: str = "",
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
) -> None:
    """
    Plot auto-correlation curves for the original and reconstructed `g2`
    matrices on the same axes.

    Solid lines correspond to the target and dashed lines correspond to the
    reconstruction, with matching colors for the same `t_1`.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 22
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'

    target = np.asarray(g2_target, dtype=np.float32)
    pred = np.asarray(g2_pred, dtype=np.float32)
    n = target.shape[0]
    step = max(1, n // n_curves)
    indices = [min(i * step, n - 1) for i in range(n_curves)]
    colors = ['#48bcbc', '#5496ce', '#b778b3', '#dc6463', '#f29742', '#e9c54e']

    fig, ax = plt.subplots(figsize=(7, 6))
    for curve_idx, idx in enumerate(indices):
        curve_target = target[idx, idx:]
        curve_pred = pred[idx, idx:]
        tau = np.arange(idx, n) - idx
        color = colors[curve_idx % len(colors)]
        ax.plot(
            tau,
            curve_target,
            lw=2.2,
            color=color,
            label=rf"$t_1={2 * idx}$ s target",
        )
        ax.plot(
            tau,
            curve_pred,
            lw=2.2,
            ls="--",
            color=color,
            label=rf"$t_1={2 * idx}$ s recon",
        )

    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(r"$f(\tau; t_1) = g_2(t_1, t_1 + \tau)$")
    if title:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks, xlabels if xlabels is not None else xticks)
    else:
        ax.set_xticks([0, 1250, 2500])
    if yticks is not None:
        ax.set_yticks(yticks, ylabels if ylabels is not None else yticks)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
def plot_nonequilibrium_distribution(
    df: pd.DataFrame,
    save_path: Path,
    x: str,
    y: str,
    xscale: str,
    yscale: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot and save the nonequilibrium distribution, with respect to
    the given x and y axes."""
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'arial'
    # plt.rcParams['mathtext.it'] = 'arial:italic'
    # plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    assert x in df.columns, f"{x} not in dataframe columns"
    assert y in df.columns, f"{y} not in dataframe columns"
    sc = ax.scatter(df[x], df[y], alpha=0.6, s=100, c=df['nonequilibrium_measure'], cmap='viridis')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Nonequilibrium Measure')
    
    plt.tight_layout()
    plt.savefig(save_path)
    
def plot_nonequilibrium_distribution_v2(
    df: pd.DataFrame,
    save_path: Path,
    x: str,
    y: str,
    xscale: str,
    yscale: str,
    xlabel: str,
    ylabel: str,
    xticks=None, yticks=None, xlabels=None, ylabels=None,
) -> None:
    """Plot and save the nonequilibrium distribution, with respect to
    the given x and y axes. Compared to the original version, this function
    uses griddata to create a smooth contour plot."""
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    assert x in df.columns, f"{x} not in dataframe columns"
    assert y in df.columns, f"{y} not in dataframe columns"
    X = df[x].values
    Y = df[y].values
    Z = df["nonequilibrium_measure"].values
    
    # xmin, xmax = X.min(), X.max()
    # ymin, ymax = Y.min(), Y.max()
    # X = _normalize(X, scale=xscale)
    # Y = _normalize(Y, scale=yscale)
    
    # grid_x, grid_y = np.mgrid[
    #     0:1:400j,
    #     0:1:400j
    # ]
    
    grid_x, grid_y = np.mgrid[
        X.min():X.max():400j,
        Y.min():Y.max():400j
    ]
    
    grid_z = griddata(
        points=np.column_stack([X, Y]),
        values=Z,
        xi=(grid_x, grid_y),
        method="nearest",     # “linear”, “cubic”, “nearest”
        rescale=True,
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cntr = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=200,       # smooth
        cmap="plasma"
    )
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks, xlabels if xlabels is not None else xticks)
    if yticks is not None:
        ax.set_yticks(yticks, ylabels if ylabels is not None else yticks)
    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_ticks([0.00, 0.20, 0.40, 0.60, 0.80])
    cbar.set_label("Nonequilibrium Measure")

    plt.tight_layout()
    plt.savefig(save_path)


def plot_nonequilibrium_phase_map(
    df: pd.DataFrame,
    save_path: Path,
    x: str,
    y: str,
    xscale: str,
    yscale: str,
    xlabel: str,
    ylabel: str,
    overlay_df: Optional[pd.DataFrame] = None,
    overlay_x: Optional[str] = None,
    overlay_y: Optional[str] = None,
    overlay_label: str = "Experiment inference",
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
) -> None:
    """
    Plot a simulation nonequilibrium phase map for one parameter pair and
    optionally overlay inferred experimental points.

    The background is interpolated from simulation samples, while the overlay
    is typically an aggregated set of experimental inference results.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "arial"
    plt.rcParams["mathtext.it"] = "arial:italic"
    plt.rcParams["mathtext.bf"] = "arial:bold"

    assert x in df.columns, f"{x} not in dataframe columns"
    assert y in df.columns, f"{y} not in dataframe columns"
    assert "nonequilibrium_measure" in df.columns, "nonequilibrium_measure missing from dataframe"

    X = df[x].to_numpy(dtype=np.float64)
    Y = df[y].to_numpy(dtype=np.float64)
    Z = df["nonequilibrium_measure"].to_numpy(dtype=np.float64)

    x_transformed = np.log10(X) if xscale == "log" else X
    y_transformed = np.log10(Y) if yscale == "log" else Y

    grid_x_transformed = np.linspace(x_transformed.min(), x_transformed.max(), 400)
    grid_y_transformed = np.linspace(y_transformed.min(), y_transformed.max(), 400)
    grid_x_t, grid_y_t = np.meshgrid(grid_x_transformed, grid_y_transformed)

    grid_z = griddata(
        points=np.column_stack([x_transformed, y_transformed]),
        values=Z,
        xi=(grid_x_t, grid_y_t),
        method="nearest",
        rescale=True,
    )

    grid_x = np.power(10.0, grid_x_t) if xscale == "log" else grid_x_t
    grid_y = np.power(10.0, grid_y_t) if yscale == "log" else grid_y_t

    fig, ax = plt.subplots(figsize=(9, 7))
    cntr = ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=200,
        cmap="plasma",
    )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks, xlabels if xlabels is not None else xticks)
    if yticks is not None:
        ax.set_yticks(yticks, ylabels if ylabels is not None else yticks)

    if (
        overlay_df is not None
        and overlay_x is not None
        and overlay_y is not None
        and not overlay_df.empty
    ):
        ax.scatter(
            overlay_df[overlay_x],
            overlay_df[overlay_y],
            s=90,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=1.5,
            alpha=0.95,
            label=overlay_label,
        )
        ax.legend(loc="upper right", fontsize=14)

    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("Nonequilibrium Measure")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    
def feature_extractor(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Extract latent features from the model for the given data loader."""
    model.eval()
    features = []
    with torch.no_grad():
        for x, _, _, T, _ in data_loader:
            x = x.to(device)
            T = T.to(device)
            if hasattr(model, "extract_features") and hasattr(model, "build_shared_features"):
                xpcs_features, temp_features = model.extract_features(x, T)
                latents = model.build_shared_features(xpcs_features, temp_features)
            else:
                latents = model.conv_net(x)
            features.append(latents.cpu())
    features = torch.cat(features, dim=0)
    return features

def calc_tSNE(
    model: torch.nn.Module,
    simulation_dataset: torch.utils.data.Dataset,
    experiment_dataset: torch.utils.data.Dataset,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    perplexity: int = 5,
    max_iter: int = 1000,
    init: Literal['random', 'pca'] = 'random',
    random_state: int = 42,
):
    """Perform t-SNE visualization of the latent space representations
    learned by the model for both simulation and experimental datasets."""
    model.eval()
    model.to(device)
    simulation_loader = torch.utils.data.DataLoader(
        simulation_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    experiment_loader = torch.utils.data.DataLoader(
        experiment_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    sim_features = feature_extractor(model, simulation_loader, device)
    exp_features = feature_extractor(model, experiment_loader, device)
    
    all_features = torch.cat([sim_features, exp_features], dim=0).numpy()
    domain_labels = np.concatenate([
        np.zeros(sim_features.size(0), dtype=int),
        np.ones(exp_features.size(0), dtype=int),
    ])  # 0: simulation, 1: experiment
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        max_iter=max_iter,
        init=init,
        random_state=random_state,
    )
    
    X_tsne = tsne.fit_transform(all_features)    # [N, 2]
    return X_tsne, domain_labels

def calc_umap(
    model: torch.nn.Module,
    simulation_dataset: torch.utils.data.Dataset,
    experiment_dataset: torch.utils.data.Dataset,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    init: str = "spectral",
    random_state: int = 42,
):
    """Perform UMAP visualization of the latent space representations
    learned by the model for both simulation and experimental datasets."""
    if umap is None:
        raise ModuleNotFoundError(
            "UMAP evaluation requires the 'umap-learn' package. "
            "Install it to enable calc_umap."
        )
    model.eval()
    model.to(device)
    simulation_loader = torch.utils.data.DataLoader(
        simulation_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    experiment_loader = torch.utils.data.DataLoader(
        experiment_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    sim_features = feature_extractor(model, simulation_loader, device)
    exp_features = feature_extractor(model, experiment_loader, device)
    
    all_features = torch.cat([sim_features, exp_features], dim=0).numpy()
    domain_labels = np.concatenate([
        np.zeros(sim_features.size(0), dtype=int),
        np.ones(exp_features.size(0), dtype=int),
    ])  # 0: simulation, 1: experiment
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        init=init,
        random_state=random_state,
    )
    
    X_umap = reducer.fit_transform(all_features)    # [N, 2]
    return X_umap, domain_labels

def calc_pca(
    model: torch.nn.Module,
    simulation_dataset: torch.utils.data.Dataset,
    experiment_dataset: torch.utils.data.Dataset,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Perform PCA visualization of the latent space representations
    learned by the model for both simulation and experimental datasets."""
    model.eval()
    model.to(device)
    simulation_loader = torch.utils.data.DataLoader(
        simulation_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    experiment_loader = torch.utils.data.DataLoader(
        experiment_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    sim_features = feature_extractor(model, simulation_loader, device)
    exp_features = feature_extractor(model, experiment_loader, device)
    
    all_features = torch.cat([sim_features, exp_features], dim=0).numpy()
    domain_labels = np.concatenate([
        np.zeros(sim_features.size(0), dtype=int),
        np.ones(exp_features.size(0), dtype=int),
    ])  # 0: simulation, 1: experiment
    
    pca = PCA(n_components=2)
    
    X_pca = pca.fit_transform(all_features)    # [N, 2]
    return X_pca, domain_labels

def plot_cluster(
    X: np.ndarray,
    domain_labels: np.ndarray,
    save_path: Path,
    legend: bool = False,
    sim_marker: str = 's',
    exp_marker: str = '^',
    sim_marker_size: int = 300,
    exp_marker_size: int = 300,
    sim_marker_alpha: float = 1.0,
    exp_marker_alpha: float = 1.0,
) -> None:
    """Plot and save the t-SNE / UMAP visualization."""
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 24
    plt.figure(figsize=(12, 4))

    mask_sim = (domain_labels == 0)
    mask_exp = (domain_labels == 1)

    plt.scatter(
        X[mask_sim, 0],
        X[mask_sim, 1],
        alpha=sim_marker_alpha,
        marker=sim_marker,
        label="Simulation",
        s=sim_marker_size,
        color='#4DBBD5FF',
    )
    plt.scatter(
        X[mask_exp, 0],
        X[mask_exp, 1],
        alpha=exp_marker_alpha,
        marker=exp_marker,
        label="Experiment",
        s=exp_marker_size,
        color='#F39B7FFF',
    )
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.xticks([])
    plt.yticks([])
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    plt.xlim(xmin - 0.10 * (xmax - xmin), xmax + 0.10 * (xmax - xmin))
    plt.ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_experiment_metadata_embedding(
    X: np.ndarray,
    domain_labels: np.ndarray,
    experiment_values: Iterable[float | str],
    save_path: Path,
    value_label: str,
    title: str = "",
    categorical: bool = False,
) -> None:
    """
    Plot one embedding where simulation points are shown in the background and
    experimental points are colored by one diagnostic metadata field.

    Args:
        X: Embedding coordinates of shape [N, 2].
        domain_labels: Domain labels aligned with `X`, where 0 is simulation and
            1 is experiment.
        experiment_values: Metadata values for the experimental rows only, in
            the same order as `X[domain_labels == 1]`.
        save_path: PDF output path.
        value_label: Label used for the legend or colorbar.
        title: Optional figure title.
        categorical: Whether to treat `experiment_values` as categories.
    """
    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 18

    exp_values = np.asarray(list(experiment_values))
    exp_mask = (domain_labels == 1)
    sim_mask = (domain_labels == 0)
    exp_coords = X[exp_mask]
    sim_coords = X[sim_mask]
    if exp_coords.shape[0] != exp_values.shape[0]:
        raise ValueError(
            "Experiment metadata length must match the number of experimental embedding rows"
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(
        sim_coords[:, 0],
        sim_coords[:, 1],
        alpha=0.20,
        s=120,
        color="#B3B3B3",
        label="Simulation",
    )

    if categorical:
        categories = list(dict.fromkeys(exp_values.tolist()))
        cmap = plt.get_cmap("tab20", max(len(categories), 1))
        for idx, category in enumerate(categories):
            category_mask = (exp_values == category)
            label = str(category)
            if label.endswith(".npz") or label.endswith(".npy"):
                label = Path(label).stem
            ax.scatter(
                exp_coords[category_mask, 0],
                exp_coords[category_mask, 1],
                alpha=0.85,
                s=140,
                color=cmap(idx),
                label=label,
            )
        legend_cols = 1 if len(categories) <= 10 else 2
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=9,
            ncol=legend_cols,
        )
    else:
        numeric_values = exp_values.astype(float)
        scatter = ax.scatter(
            exp_coords[:, 0],
            exp_coords[:, 1],
            alpha=0.90,
            s=140,
            c=numeric_values,
            cmap="viridis",
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        colorbar.set_label(value_label, fontsize=12)
        ax.legend(loc="upper left", frameon=False, fontsize=11)

    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    ax.set_xlim(xmin - 0.10 * (xmax - xmin), xmax + 0.10 * (xmax - xmin))
    ax.set_ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    
