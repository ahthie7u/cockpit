"""Compare memory footprint w/o individual gradient transformations."""

import os
import sys
import warnings

import pandas
from memory_profiler import memory_usage
from torch.optim import SGD

from cockpit.runners.scheduled_runner import _ScheduleCockpitRunner

HERE = os.path.abspath(__file__)
DIR = os.path.join(os.path.dirname(HERE), "data")
FIG_DIR = os.path.join(os.path.dirname(HERE), "fig")

os.makedirs(DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def set_up():
    from cockpit.utils import fix_deepobs_data_dir
    from deepobs.pytorch.config import set_default_device

    fix_deepobs_data_dir()

    FORCE_CPU = True
    if FORCE_CPU:
        set_default_device("cpu")


INTERVAL = 0.01


def report_memory(f):
    mem_usage = memory_usage(f, interval=INTERVAL)
    mem_time = [INTERVAL * idx for idx in range(len(mem_usage))]

    return pandas.DataFrame(data={"time": mem_time, "usage": mem_usage})


def lr_schedule(num_epochs):
    """Some Learning rate schedule.

    Example:
        >>> # Halving the learning rate every epoch:
        >>> lambda epoch: 0.5 ** epoch
        >>> # A less aggressive decay:
        >>> lambda epoch: 0.9 ** epoch
        >>> # Constant learning rate (using init lr):
        >>> lambda epoch: 1.0
    """
    return lambda epoch: 1.0


def run(quants, testproblem):
    optimizer_class = SGD
    hyperparams = {
        "lr": {"type": float, "default": 0.001},
        "momentum": {"type": float, "default": 0.0},
        "nesterov": {"type": bool, "default": False},
    }

    def plot_schedule(global_step):
        return False

    runner = MemoryBenchmarkRunner(
        optimizer_class,
        hyperparams,
        quantities=quants,
        plot=False,
        plot_schedule=plot_schedule,
    )

    runner.run(
        testproblem=testproblem,
        num_epochs=1,
        l2_reg=0.0,  # necessary for backobs!
        track_interval=1,
        plot_interval=1,
        show_plots=False,
        save_plots=False,
        save_final_plot=False,
        save_animation=False,
        lr_schedule=lr_schedule,
    )


class MemoryBenchmarkRunner(_ScheduleCockpitRunner):
    """Run first forward-backward pass and update step of training, then quit.

    Note:
        Disables DeepOBS' additional metrics. Performs one step per epoch.
    """

    STOP_BATCH_COUNT_PER_EPOCH = 1

    def _maybe_stop_iteration(self, global_step, batch_count):
        """Stop after first step of each epoch."""
        if batch_count == self.STOP_BATCH_COUNT_PER_EPOCH:
            warnings.warn(
                "The memory benchmark runner performs only "
                + f"{self.STOP_BATCH_COUNT_PER_EPOCH} steps per epoch."
            )
            raise StopIteration

    def _should_eval(self):
        """Disable DeepOBS' evaluation of test/train/valid losses and accuracies."""
        return False


def hotfix_deepobs_argparse():
    """Truncate command line arguments from pytest call to make DeepOBS arparse work.

    TODO Think about good alternatives.
    """
    sys.argv = sys.argv[:1]


def parse():
    testproblem = sys.argv[1]

    try:
        num_run = int(sys.argv[2])
    except IndexError:
        num_run = None

    hotfix_deepobs_argparse()
    return testproblem, num_run


def skip_if_exists(filename):
    if os.path.exists(filename):
        print(f"Skipping as file already exists: {filename}")
        sys.exit(0)
