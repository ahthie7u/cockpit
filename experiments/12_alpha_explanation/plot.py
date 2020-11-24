"""Plot three subplots explaining the alpha quantitiy."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cockpit.experiments.utils import _get_plot_size, _set_plotting_params
from cockpit.quantities.alpha import _fit_quadratic

COLORS = sns.color_palette("tab10")

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "output/")
os.makedirs(SAVEDIR, exist_ok=True)

np.random.seed(1)


def plot_alpha_explanation():
    fig, axs = plt.subplots(
        1,
        3,
        figsize=_get_plot_size(textwidth="cvpr", height_ratio=0.9, subplots=(1, 3)),
        sharey=True,
    )

    plot_observations(
        axs[0],
        f0=1,
        f1=0.51,
        df0=-0.5,
        df1=-0.4,
        title=r"Understepping: $\alpha < 0$",
        ylabel="Loss",
    )
    plot_observations(
        axs[1],
        f0=1,
        f1=0.5,
        df0=-0.75,
        df1=0,
        title=r"Minimizing: $\alpha \approx 0$",
        # xlabel="Step Size",
    )
    plot_observations(
        axs[2],
        f0=1,
        f1=1,
        df0=-1,
        df1=1.1,
        pos_f1=2.0,
        title=r"Overshooting: $\alpha > 0$",
    )


def plot_observations(
    ax, f0, f1, df0, df1, pos_f0=0.0, pos_f1=1.0, title=None, ylabel=None, xlabel=None
):
    dt = 0.15
    var_fs = [1e-1, 1e-1]
    var_dfs = [2e-1, 2e-1]
    xlims = [-0.4, 2.4]
    ylims = [0.0, 1.2]
    num_ind = 3

    # Loss values
    ax.plot(pos_f0, f0, marker="o", markersize=6, color=COLORS[1], zorder=10)
    ax.plot(pos_f1, f1, marker="o", markersize=6, color=COLORS[1], zorder=10)

    # Slopes
    ax.plot(
        [pos_f0 - dt, pos_f0 + dt], [f0 - dt * df0, f0 + dt * df0], "#32414b", zorder=8
    )
    ax.plot(
        [pos_f1 - dt, pos_f1 + dt], [f1 - dt * df1, f1 + dt * df1], "#32414b", zorder=8
    )

    # Plot "individual values"
    for _ in range(num_ind):
        f0_ind = f0 + np.random.normal(scale=var_fs[0])
        f1_ind = f1 + np.random.normal(scale=var_fs[1])
        df0_ind = df0 + np.random.normal(scale=var_dfs[0])
        df1_ind = df1 + np.random.normal(scale=var_dfs[1])
        ax.plot(pos_f0, f0_ind, marker="o", markersize=6, color="#FFE5CE", zorder=5)
        ax.plot(
            [pos_f0 - dt, pos_f0 + dt],
            [f0_ind - dt * df0_ind, f0_ind + dt * df0_ind],
            "#32414b",
            alpha=0.25,
        )
        ax.plot(pos_f1, f1_ind, marker="o", markersize=6, color="#FFE5CE", zorder=5)
        ax.plot(
            [pos_f1 - dt, pos_f1 + dt],
            [f1_ind - dt * df1_ind, f1_ind + dt * df1_ind],
            "#32414b",
            alpha=0.25,
        )

    mu = _fit_quadratic(pos_f1, [f0, f1], [df0, df1], var_fs, var_dfs)
    # alpha = _get_alpha(mu, pos_f1 - pos_f0)

    xs = np.linspace(xlims[0], xlims[1], 100)
    ys = mu[2] * xs ** 2 + mu[1] * xs + mu[0]
    ax.plot(xs, ys, color=COLORS[0])

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.set_xticks([pos_f0, pos_f1])
    ax.set_xticklabels([r"$\theta_t$", r"$\theta_{t+1}$"])

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


if __name__ == "__main__":
    _set_plotting_params()

    plot_alpha_explanation()
    plt.tight_layout()
    # plt.show()

    savepath = os.path.join(SAVEDIR, "alpha_explanation.pdf")
    plt.savefig(savepath, bbox_inches="tight")
