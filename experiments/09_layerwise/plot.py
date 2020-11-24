import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from run import PROBLEMS

from cockpit import CockpitPlotter, instruments
from cockpit.experiments.plot import TikzExport
from cockpit.instruments.histogram_2d_gauge import _get_xmargin_histogram_data

mpl.use("Agg")

HERE = os.path.abspath(__file__)
FIG_DIR = os.path.join(os.path.dirname(HERE), "fig")

os.makedirs(FIG_DIR, exist_ok=True)

base = os.path.join(os.path.dirname(HERE), "results")


def get_filepath(problem):
    # get log path
    probpath = os.path.join(base, problem, "SGD")
    pattern = os.path.join(probpath, "*", "*__log.json")

    filepath = glob.glob(pattern)
    if len(filepath) != 1:
        raise ValueError(f"Found no or multiple files: {filepath}, pattern: {pattern}")
    filepath = filepath[0].replace(".json", "")
    print(filepath)

    return filepath


def get_cockpit_plotter(filepath, global_step=0):
    cp = CockpitPlotter(filepath)
    cp._read_tracking_results()
    # drop all data except first step
    clean_tracking_data = cp.tracking_data.loc[
        cp.tracking_data["iteration"] == global_step
    ]
    cp.tracking_data = clean_tracking_data

    return cp


def plot_preview(problem, global_step=0):
    """Create plots while exploring interesting problems."""
    filepath = get_filepath(problem)
    cp = get_cockpit_plotter(filepath, global_step=global_step)

    # plot full network #
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(1, 1)
    sub_gs = gs[0, 0].subgridspec(1, 1)

    instruments.histogram_2d_gauge(cp, fig, sub_gs[0, 0])

    plt.savefig(os.path.join(FIG_DIR, f"{problem}.png"))
    plt.close()

    # plot layerwise #
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(1, 1)

    cp._plot_layerwise(gs[0, 0], fig=fig)

    plt.savefig(os.path.join(FIG_DIR, f"layerwise-{problem}.png"))
    plt.close()


def plot_net_paper(problem, color, global_step=0):
    """Create TikZ plots for the paper."""
    filepath = get_filepath(problem)
    cp = get_cockpit_plotter(filepath, global_step=global_step)

    vals, mid_points, bin_size = _get_xmargin_histogram_data(cp.tracking_data)
    start_points = [x - bin_size / 2 for x in mid_points]

    plot_histogram(start_points, vals, bin_size, color)

    savepath = get_net_out_file(problem)
    TikzExport().save_fig(savepath)
    plt.close()


def get_net_out_file(problem):
    return os.path.join(FIG_DIR, f"net-grad-margin-{problem}")


def plot_params_paper(problem, color, global_step=0):
    """Create TikZ plots for the paper."""
    filepath = get_filepath(problem)
    cp = get_cockpit_plotter(filepath, global_step=global_step)

    idx = 0

    while True:
        try:
            vals, mid_points, bin_size = _get_xmargin_histogram_data(
                cp.tracking_data, idx=idx
            )
        except KeyError:
            break

        start_points = [x - bin_size / 2 for x in mid_points]

        plot_histogram(start_points, vals, bin_size, color)

        savepath = get_params_out_file(problem, idx)
        TikzExport().save_fig(savepath)
        plt.close()

        idx += 1


def get_params_out_file(problem, idx):
    return os.path.join(FIG_DIR, f"param-{idx}-grad-margin-{problem}")


def plot_histogram(start_points, vals, bin_size, color):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_facecolor("white")
    ax.barh(
        start_points,
        vals,
        height=bin_size,
        color=color,
        linewidth=0.1,
        log=True,
        left=0.9,
        align="edge",
    )
    ax.set_ylim([min(start_points), max(start_points) + bin_size])

    return fig, ax


if __name__ == "__main__":
    # for problem in PROBLEMS:
    # plot_preview(problem)
    PROBLEMS = [p for p in PROBLEMS if p in ["cifar10_3c3d", "cifar10_3c3dsig"]]

    # plot the full net and copy to tex dir
    COLORS = sns.color_palette("tab10")[:2]

    for problem, color in zip(PROBLEMS, COLORS):
        plot_net_paper(problem, color)

    # plot the layers and copy to tex dir
    PARAM_COLORS = COLORS

    for problem, color in zip(PROBLEMS, PARAM_COLORS):
        plot_params_paper(problem, color)

    PARAM_IDX = [0, 4, 10]
