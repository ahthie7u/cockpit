"""Benchmark Bar Plot of the Overhead of Individual Instruments."""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import run_individual
import seaborn as sns
from benchmark_utils import _fix_dev_naming, _fix_tp_naming, _quantity_naming, read_data

from cockpit.experiments.utils import _get_plot_size, _set_plotting_params

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "fig_individual/")
os.makedirs(SAVEDIR, exist_ok=True)


def plot_data(df, show=True, save=True, title=False):
    """Create a bar plot from the benchmarking data.

    The bar plot shows the relative run time compared to an empty cockpit for
    individual instruments. The run time is averaged over multiple seeds.

    Args:
        df (pandas.DataFrame): DataFrame holding the benchmark data.
    """
    # Plotting Params #
    _set_plotting_params()
    # Smaller font size for quantities
    plt.rcParams.update({"xtick.labelsize": 6})
    width_capsize = 0.25
    width_errorbars = 0.75
    ci = "sd"
    hline_color = "gray"
    hline_style = ":"
    color_palette = "husl"  # "rocket_r", "tab10" "Set2"

    fig, ax = plt.subplots(
        figsize=_get_plot_size(textwidth="cvpr", fraction=0.65, height_ratio=0.4)
    )

    drop = [
        # Remove cockpit_configurations
        "full",
        "business",
        "economy",
        "MaxEV",
        "BatchGradHistogram2d",
    ]

    for d in drop:
        df.drop(df[(df.quantities == d)].index, inplace=True)

    # Compute mean time for basline
    mean_baseline = df.loc[df["quantities"] == "baseline"].mean(axis=0).time_per_step
    df["relative_overhead"] = df["time_per_step"].div(mean_baseline)

    # Verify that the data is from a single test problem and use it as a title
    testproblem_set = df.testproblem.unique()
    assert len(testproblem_set) == 1
    tp_name = str(testproblem_set[0])
    tp_name_fixed = _fix_tp_naming(tp_name)

    device_set = df.device.unique()
    assert len(device_set) == 1
    dev_name = str(device_set[0])
    dev_name_fixed = _fix_dev_naming(dev_name)

    # Order from smallest to largest
    grp_order = df.groupby("quantities").time_per_step.agg("mean").sort_values().index
    # but put "baseline" always in front:
    idx_baseline = np.where(grp_order._index_data == "baseline")[0][0]
    order_list = list(grp_order._index_data)
    order_list.insert(0, order_list.pop(idx_baseline))
    grp_order._index_data = order_list
    grp_order._data = order_list

    sns.barplot(
        x="quantities",
        y="relative_overhead",
        data=df,
        order=grp_order,
        ax=ax,
        capsize=width_capsize,
        errwidth=width_errorbars,
        ci=ci,
        estimator=np.mean,
        palette=color_palette,
    )

    # Line at 1
    ax.axhline(
        y=1,
        color=hline_color,
        linestyle=hline_style,
    )

    if title:
        ax.set_title(
            f"Computational Overhead for {tp_name_fixed} ({dev_name_fixed})",
            fontweight="bold",
        )
    ax.set_xlabel("")
    ax.set_ylabel("Run Time Overhead")
    ax.set_xticklabels(_quantity_naming(x.get_text()) for x in ax.get_xticklabels())
    plt.tight_layout()

    # Fix to make the bar plot for the paper a bit more appealing
    ylims = list(ax.get_ylim())
    ylims[1] = max(3.0, ylims[1])
    ax.set_ylim(ylims)

    if save:
        savepath = os.path.join(SAVEDIR, f"bar_plot_{tp_name}_{dev_name}_{title}.pdf")
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


def plot_expensive_data(df, show=True, save=True, title=False):
    """Create a bar plot from the expensive instruments.

    The bar plot shows the relative run time compared to an empty cockpit for
    individual instruments. The run time is averaged over multiple seeds.

    Args:
        df (pandas.DataFrame): DataFrame holding the benchmark data.
    """
    # Plotting Params #
    _set_plotting_params()
    # Smaller font size for quantities
    plt.rcParams.update({"xtick.labelsize": 6})
    width_capsize = 0.25
    width_errorbars = 0.75
    ci = "sd"
    hline_color = "gray"
    hline_style = ":"
    color_palette = "husl"  # "rocket_r", "tab10" "Set2"

    fig, ax = plt.subplots(
        figsize=_get_plot_size(
            textwidth="cvpr",
            fraction=0.65 / 0.65 * 0.35,
            height_ratio=0.4 * 0.65 / 0.35,
        )
    )

    keep = [
        "baseline",
        "MaxEV",
        "BatchGradHistogram2d",
    ]
    drop = [c for c in set(df.quantities) if c not in keep]

    for d in drop:
        df.drop(df[(df.quantities == d)].index, inplace=True)

    # Compute mean time for basline
    mean_baseline = df.loc[df["quantities"] == "baseline"].mean(axis=0).time_per_step
    df["relative_overhead"] = df["time_per_step"].div(mean_baseline)

    # Verify that the data is from a single test problem and use it as a title
    testproblem_set = df.testproblem.unique()
    assert len(testproblem_set) == 1
    tp_name = str(testproblem_set[0])
    tp_name_fixed = _fix_tp_naming(tp_name)

    device_set = df.device.unique()
    assert len(device_set) == 1
    dev_name = str(device_set[0])
    dev_name_fixed = _fix_dev_naming(dev_name)

    # Order from smallest to largest
    grp_order = df.groupby("quantities").time_per_step.agg("mean").sort_values().index
    # but put "baseline" always in front:
    idx_baseline = np.where(grp_order._index_data == "baseline")[0][0]
    order_list = list(grp_order._index_data)
    order_list.insert(0, order_list.pop(idx_baseline))
    grp_order._index_data = order_list
    grp_order._data = order_list

    sns.barplot(
        x="quantities",
        y="relative_overhead",
        data=df,
        order=grp_order,
        ax=ax,
        capsize=width_capsize,
        errwidth=width_errorbars,
        ci=ci,
        estimator=np.mean,
        palette=color_palette,
    )

    # Line at 1
    ax.axhline(
        y=1,
        color=hline_color,
        linestyle=hline_style,
    )

    if title:
        ax.set_title(
            f"Computational Overhead for {tp_name_fixed} ({dev_name_fixed})",
            fontweight="bold",
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(_quantity_naming(x.get_text()) for x in ax.get_xticklabels())
    plt.tight_layout()

    # Fix to make the bar plot for the paper a bit more appealing
    ylims = list(ax.get_ylim())
    ylims[1] = max(3.0, ylims[1])
    ax.set_ylim(ylims)

    if save:
        savepath = os.path.join(
            SAVEDIR, f"bar_plot_expensive_{tp_name}_{dev_name}_{title}.pdf"
        )
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":

    for (testproblem, device) in run_individual.RUNS:

        filepath = run_individual.get_savefile(testproblem, device)
        df, testproblem_set = read_data(filepath)

        for tp in testproblem_set:
            plot_data(copy.deepcopy(df[tp]), show=False)
            plot_expensive_data(copy.deepcopy(df[tp]), show=False)
