"""Plot the three subplots showing that the Loss Is Not Enough (LINE)."""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy import stats

from cockpit import CockpitPlotter
from cockpit.experiments.utils import _get_plot_size, _set_plotting_params

COLORS = sns.color_palette("tab10")
COLOR_PARABOLA = "Gray"

LABEL_PADDING = 10


def read_data(path):
    """Read the data and return the necessary quantities.

    Args:
        path (str): Path to the json logfile

    Returns:
        tuple: Consisting of the optimizer's trajectory, its losses and alphas
    """
    plotter = CockpitPlotter(path)
    plotter._read_tracking_results()
    losses = plotter.tracking_data.mini_batch_loss.dropna().to_numpy()
    traj = np.concatenate(plotter.tracking_data.params.dropna().to_numpy(), axis=0)
    alphas = plotter.tracking_data.alpha.dropna().to_numpy()

    return traj, losses, alphas


class two_d_quadratic_problem:
    """Simple class for plotting a 2D Quadratic Problem."""

    def __init__(self, rotate=30):
        """Initialize the class.

        Args:
            rotate (float, optional): Rotate the Hessian. Defaults to 30.
        """
        rotate = rotate * np.pi / 180
        Q = np.diag([0.5, 10.0])
        R = np.array(
            [[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]]
        )

        self.Hessian = R.dot(Q).dot(R.T)

    def __call__(self, x1, x2):
        """Evaluate the Quadratic at a certain point.

        Args:
            x1 (float): First parameter
            x2 (float): Second parameter

        Returns:
            float: Value of the quadratic problem at this point.
        """
        x = np.array([x1, x2])
        return 1 / 2 * x.T.dot(self.Hessian.dot(x))

    def plot(
        self,
        ax,
        paths,
        xlow=-0.7,
        xhigh=1.6,
        ylow=-0.75,
        yhigh=1.1,
        resolution=200,
        resolution_contour=20,
    ):
        """Plot the quadratic problem and optimizer trajectories.

        Args:
            ax (Matplotib.axes): Axis where the plot should be created.
            paths (list): The optimizers trajectory.
            xlow (float, optional): Lower limit of the x-Axis. Defaults to -0.5.
            xhigh (float, optional): Upper limit of the x-Axis. Defaults to 1.4.
            ylow (float, optional): Lower limit of the y-Axis. Defaults to -0.75.
            yhigh (float, optional): Upper limit of the y-Axis. Defaults to 1.1.
            resolution (int, optional): Resolution of the quadratic. Defaults to 200.
            resolution_contour (int, optional): Number of contour lines. Defaults to 20.
        """
        Z = [
            [self(x1, x2) for x1 in np.linspace(xlow, xhigh, resolution)]
            for x2 in np.linspace(ylow, yhigh, resolution)
        ]

        ax.imshow(
            Z,
            cmap=mpl.cm.Greys,
            extent=(1.1 * xlow, 1.1 * xhigh, 1.1 * ylow, 1.1 * yhigh),
            origin="lower",
        )
        # adding the Contour lines with labels
        ax.contour(
            Z,
            levels=resolution_contour,
            linewidths=1,
            colors="Gray",
            alpha=0.6,
            extent=(xlow, xhigh, ylow, yhigh),
        )

        # Highlight Minimizer
        ax.plot(0.0, 0.0, "k*", ms=5, mew=0)
        # Highlight Start Point
        ax.plot(1.0, 1.0, "kx", ms=3.5)

        for idx, path in enumerate(paths):
            path = path.T
            ax.quiver(
                path[0, :-1],
                path[1, :-1],
                path[0, 1:] - path[0, :-1],
                path[1, 1:] - path[1, :-1],
                scale_units="xy",
                angles="xy",
                scale=1,
                color=COLORS[idx],
                zorder=100,
                width=0.0115,  # 1.2,
                edgecolors=COLORS[idx],
                alpha=1,
                headwidth=4,
            )

        # Set xlim ylim
        ax.set_xlim(xlow, xhigh)
        ax.set_ylim(ylow, yhigh)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax.set_title("Loss Landscape", fontweight="bold")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.tick_params(axis="both", which="both", length=0)
        # ax.xaxis.labelpad = LABEL_PADDING

        # Flip plot
        ax.invert_xaxis()
        ax.invert_yaxis()

        plt.tight_layout()


def plot_loss(ax, data):
    """Plot the loss.

    Args:
        ax (Matplotib.axes): Axis where the plot should be created.
        data ([float]): List of loss values.
    """
    for idx, d in enumerate(data):
        ax.plot(d, linewidth=2, color=COLORS[idx])

    # Set xlim ylim
    ax.set_xlim(-0.2, len(d) - 1 + 0.2)

    ax.set_title("Loss Curve", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mini-Batch Training Loss")
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    # ax.xaxis.labelpad = LABEL_PADDING

    plt.tight_layout()


def plot_alpha(ax, alphas):
    """Plot a simpler version of the alpha gauge for two runs.

    Args:
        ax (Matplotib.axes): Axis where the plot should be created.
        alphas ([float]): List of alpha values.
    """
    xlim = [-1.25, 1.25]
    ylim = [0, 1.75]

    # Plot unit parabola
    x = np.linspace(xlim[0], xlim[1], 100)
    y = x ** 2
    ax.plot(x, y, linewidth=1.5, color=COLOR_PARABOLA)
    # Adding Zone Lines
    # ax.axvline(0, ls="-", color="#ababba", linewidth=0.5, zorder=0)
    # ax.axvline(-1, ls="-", color="#ababba", linewidth=0.5, zorder=0)
    # ax.axvline(1, ls="-", color="#ababba", linewidth=0.5, zorder=0)
    # ax.axhline(0, ls="-", color="#ababba", linewidth=0.5, zorder=0)
    # ax.axhline(1, ls="-", color="#ababba", linewidth=0.5, zorder=0)

    # Alpha Histogram
    ax2 = ax.twinx()
    for idx, alpha in enumerate(alphas):
        # All alphas
        sns.distplot(
            alpha,
            ax=ax2,
            norm_hist=True,
            bins=6,
            fit=stats.norm,
            kde=False,
            color=COLORS[idx],
            fit_kws={"color": COLORS[idx]},
            hist_kws={"linewidth": 0, "alpha": 0.25},
        )

    ax.yaxis.set_major_locator(MaxNLocator(2, integer=True))
    ax.tick_params(axis="both", which="both", length=0)
    ax2.tick_params(axis="both", which="both", length=0)
    ax2.set_yticks([])

    ax.set_title("Alpha Distribution", fontweight="bold")
    ax.set_xlabel("Normalized Step Length")
    ax.set_ylabel("Normalized Loss")
    ax2.set_ylabel("Density")

    ax2.set_ylim(0, 30)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.xaxis.labelpad = LABEL_PADDING

    plt.tight_layout()


if __name__ == "__main__":
    # Abuse the CockpitPlotter to read data
    basepath = os.path.join(
        sys.path[0],
        "results/two_d_quadratic/SGD/",
        "num_epochs__20__batch_size__128__l2_reg__0.e+00__lr__1.e+00__momentum"
        "__0.e+00__nesterov__False",
    )
    # This refers so specific files.
    # Running 02_run_Line.py will produce them, but they will be called
    # slightly different
    large_path = "random_seed__42__2020-11-23-14-30-33__log"
    short_path = "random_seed__42__2020-11-23-14-30-47__log"

    large_traj, large_losses, large_alphas = read_data(
        os.path.join(basepath, large_path)
    )
    short_traj, short_losses, short_alphas = read_data(
        os.path.join(basepath, short_path)
    )

    _set_plotting_params()
    # Update params with individual changes
    # Matplotlib settings (using tex font)
    update = {
        # Less space between label and axis
        "xtick.major.pad": 1.0,
        "ytick.major.pad": 1.0,
    }
    plt.rcParams.update(update)

    fig, axs = plt.subplots(
        1,
        3,
        figsize=_get_plot_size(textwidth="cvpr", height_ratio=0.9, subplots=(1, 3)),
    )

    qp = two_d_quadratic_problem()

    # Create the three subplots
    plot_loss(axs[0], [large_losses, short_losses])
    qp.plot(axs[1], paths=[large_traj, short_traj])
    plot_alpha(axs[2], [large_alphas, short_alphas])

    # Save plot
    savepath = os.path.join(sys.path[0], "output/")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(savepath + "LINE.pdf", bbox_inches="tight")
    plt.show()
