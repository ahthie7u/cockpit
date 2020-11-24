"""Run SGD on MNIST using different learning rates."""

import matplotlib.pyplot as plt
from torch.optim import SGD

from cockpit.cockpit import configured_quantities
from cockpit.runners.scheduled_runner import ScheduleCockpitRunner

optimizer_class = SGD


def const_schedule(num_epochs):
    """Constant schedule with a small decay at the end."""
    return lambda epoch: 1.0


quants = configured_quantities("business")

lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1]

for lr in lrs:
    hyperparams = {"lr": {"type": float, "default": lr}}
    runner = ScheduleCockpitRunner(
        optimizer_class, hyperparams, quantities=quants, plot=True
    )
    runner.run(
        testproblem="mnist_mlp",
        l2_reg=0.0,  # necessary for backobs!
        track_interval=1,
        plot_interval=1,
        show_plots=False,
        save_plots=True,
        save_final_plot=True,
        save_animation=True,
        lr_schedule=const_schedule,
    )
    plt.close("all")
