"""Compute the 2d histogram for different data pre-processing steps."""

import glob
import os

from run_samples import make_and_register_tproblems
from torch.optim import SGD

from cockpit import quantities
from cockpit.runners.scheduled_runner import ScheduleCockpitRunner
from cockpit.utils import fix_deepobs_data_dir

fix_deepobs_data_dir()

optimizer_class = SGD
hyperparams = {"lr": {"type": float, "default": 0.001}}

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)


def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.0


def plot_schedule(global_step):
    return global_step == 0


def make_quantities():
    def track_schedule(global_step):
        return global_step == 0

    def adapt_schedule(global_step):
        return False

    quants = [
        quantities.BatchGradHistogram2d(
            xmin=-1.5,
            xmax=1.5,
            ymin=-0.2,
            ymax=0.2,
            track_schedule=track_schedule,
            adapt_schedule=adapt_schedule,
            keep_individual=False,
        )
    ]

    return quants


def get_out_file(tproblem):
    probpath = os.path.join(HEREDIR, "results", tproblem, "SGD")
    pattern = os.path.join(probpath, "*", "*__log.json")

    filepath = glob.glob(pattern)
    if len(filepath) != 1:
        raise ValueError(f"Found no or multiple files: {filepath}, pattern: {pattern}")
    filepath = filepath[0].replace(".json", "")

    return filepath


make_and_register_tproblems()

PROBLEMS = ["cifar10raw_3c3d", "cifar10scale255_3c3d"]

if __name__ == "__main__":

    for problem in PROBLEMS:
        runner = ScheduleCockpitRunner(
            optimizer_class,
            hyperparams,
            quantities=make_quantities(),
            plot_schedule=plot_schedule,
        )

        runner.run(
            testproblem=problem,
            l2_reg=0.0,  # necessary for backobs!
            num_epochs=1,
            batch_size=128,
            track_interval=1,
            plot_interval=1,
            show_plots=False,
            save_plots=True,
            save_final_plot=False,
            save_animation=False,
            lr_schedule=lr_schedule,
            plot_schedule=plot_schedule,
            skip_if_exists=True,
        )
