"""Utility Functions for the Benchmark Plotting."""

import matplotlib as mpl
import pandas as pd
import seaborn as sns

from deepobs.config import DATA_SET_NAMING, TP_NAMING


def read_data(filepath):
    """Read the benchmarking data to a pandas DataFrame.

    Args:
        filepath (str): Path to the .csv file holding the data.

    Returns:
        pandas.DataFrame: DataFrame holding the individual runs of the benchmark.
    """
    # CSV file starts with soft- & hardware info that is filtered out
    df = pd.read_csv(filepath, comment="#", index_col=[0])

    # Split by testproblem
    # create unique list of names
    testproblem_set = df.testproblem.unique()
    # create a data frame dictionary to store your data frames
    df_dict = {elem: pd.DataFrame for elem in testproblem_set}
    for key in df_dict.keys():
        df_dict[key] = df[:][df.testproblem == key]

    return df_dict, testproblem_set


def _set_plotting_params():
    # Settings:
    plot_size_default = [16, 8]
    plot_scale = 1.0
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale=1.2)
    # Apply the settings
    mpl.rcParams["figure.figsize"] = [plot_scale * e for e in plot_size_default]


def _fix_tp_naming(tp):
    dataset = tp.split("_", 1)[0]
    problem = tp.split("_", 1)[1]

    return DATA_SET_NAMING[dataset] + " " + TP_NAMING[problem]


def _fix_dev_naming(dev):
    mapping = {
        "cuda": "GPU",
        "cpu": "CPU",
    }
    return mapping[dev]


def _quantity_naming(quantity):
    quantity_naming = {
        "InnerProductTest": "Inner\nProduct Test",
        "OrthogonalityTest": "Orthogonality\nTest",
        "NormTest": "Norm Test",
        "baseline": "Baseline",
        "TICDiag": "TIC",
        "AlphaOptimized": "Alpha",
        "BatchGradHistogram1d": "1D\nHistogram",
        "BatchGradHistogram2d": "2D\nHistogram",
        "GradNorm": "Gradient\nNorm",
    }

    if quantity in quantity_naming:
        return quantity_naming[quantity]
    else:
        return quantity
