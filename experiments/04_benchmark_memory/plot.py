import os

import baseline
import expensive
import matplotlib.pyplot as plt
import numpy
import optimized
import pandas
from palettable.colorbrewer.qualitative import Set1_3
from shared import FIG_DIR, parse

from cockpit.experiments.plot import TikzExport

# colors
COLOR_EXPENSIVE = Set1_3.mpl_colors[0]
COLOR_BASELINE = Set1_3.mpl_colors[1]
COLOR_OPTIMIZED = Set1_3.mpl_colors[2]

# number of standard deviations shown around mean
NUM_STD = 2
# opacity of standard deviations
SHADE = 0.4


def out_file(testproblem):
    return os.path.join(FIG_DIR, f"{testproblem}")


def get_timespan(testproblem):
    min_time = 0
    max_time = 0
    for f in expensive.get_out_files(testproblem) + optimized.get_out_files(
        testproblem
    ):
        data = pandas.read_csv(f)

        f_min_time = data["time"].min()
        if f_min_time < min_time:
            min_time = f_min_time

        f_max_time = data["time"].max()
        if f_max_time > max_time:
            max_time = f_max_time

    return numpy.linspace(min_time, max_time)


def get_max_usage_mean_std(testproblem, which):
    if which == "expensive":
        files = expensive.get_out_files(testproblem)
    elif which == "optimized":
        files = optimized.get_out_files(testproblem)
    elif which == "baseline":
        files = baseline.get_out_files(testproblem)
    else:
        raise ValueError("Arg 'which' must be 'expensive', 'optimized' or 'baseline")

    max_usage = []

    for f in files:
        data = pandas.read_csv(f)
        max_usage.append(data["usage"].max())

    max_usage = numpy.array(max_usage)
    mean_max_usage = numpy.mean(max_usage)
    std_max_usage = numpy.std(max_usage)

    return mean_max_usage, std_max_usage


def compute_markevery(data, max_points=200):
    """Compute number of points that will be dropped to compress the plot."""
    num_points = len(data)
    markevery = max(num_points // max_points, 1)

    return markevery


def plot(testproblem):
    plt.figure()
    plt.title(r"\texttt{" + testproblem.replace("_", r"\_") + "}")
    plt.xlabel("Time [s]")
    plt.ylabel("Memory [MB]")

    # line plots
    for f in expensive.get_out_files(testproblem):
        data = pandas.read_csv(f)
        markevery = compute_markevery(data["time"])
        plt.plot(
            data["time"][::markevery],
            data["usage"][::markevery],
            color=COLOR_EXPENSIVE,
            ls="dotted",
        )

    for f in optimized.get_out_files(testproblem):
        data = pandas.read_csv(f)
        markevery = compute_markevery(data["time"])
        plt.plot(
            data["time"][::markevery],
            data["usage"][::markevery],
            color=COLOR_OPTIMIZED,
            ls="dotted",
        )

    # error bars
    times = get_timespan(testproblem)
    # first and last point sufficient
    times = [times[0], times[-1]]

    mean_max_usage, std_max_usage = get_max_usage_mean_std(testproblem, "expensive")
    label = (
        f"expensive: ${mean_max_usage:.0f}" + r" \pm " + f"{std_max_usage:.0f}$" + " MB"
    )
    plt.plot(
        times,
        len(times) * [mean_max_usage],
        color=COLOR_EXPENSIVE,
        ls="dashed",
        label=label,
    )
    plt.fill_between(
        times,
        y1=mean_max_usage - NUM_STD * std_max_usage,
        y2=mean_max_usage + NUM_STD * std_max_usage,
        color=COLOR_EXPENSIVE,
        alpha=SHADE,
    )

    mean_max_usage, std_max_usage = get_max_usage_mean_std(testproblem, "optimized")
    label = (
        f"optimized: ${mean_max_usage:.0f}" + r" \pm " + f"{std_max_usage:.0f}$" + " MB"
    )
    plt.plot(
        times,
        len(times) * [mean_max_usage],
        color=COLOR_OPTIMIZED,
        ls="dashed",
        label=label,
    )
    plt.fill_between(
        times,
        y1=mean_max_usage - NUM_STD * std_max_usage,
        y2=mean_max_usage + NUM_STD * std_max_usage,
        color=COLOR_OPTIMIZED,
        alpha=SHADE,
    )

    mean_max_usage, std_max_usage = get_max_usage_mean_std(testproblem, "baseline")
    label = (
        f"baseline: ${mean_max_usage:.0f}" + r" \pm " + f"{std_max_usage:.0f}$" + " MB"
    )
    plt.plot(
        times,
        len(times) * [mean_max_usage],
        color=COLOR_BASELINE,
        ls="dashed",
        label=label,
    )
    plt.fill_between(
        times,
        y1=mean_max_usage - NUM_STD * std_max_usage,
        y2=mean_max_usage + NUM_STD * std_max_usage,
        color=COLOR_BASELINE,
        alpha=SHADE,
    )

    plt.legend()

    TikzExport().save_fig(out_file(testproblem))
    plt.close()


if __name__ == "__main__":
    testproblem, _ = parse()
    print(f"Plot memory benchmark on {testproblem}")
    plot(testproblem)
