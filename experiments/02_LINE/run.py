"""Run SGD on the local quadratic problem."""

import two_d_quadratic
from torch.optim import SGD

from cockpit import quantities
from cockpit.runners.scheduled_runner import ScheduleCockpitRunner

two_d_quadratic.register()

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 1.0},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.2


cp_quantities = [
    quantities.parameters.Parameters,
    quantities.alpha.AlphaOptimized,
    quantities.loss.Loss,
    quantities.distance.Distance,
    quantities.grad_norm.GradNorm,
    quantities.time.Time,
]

runner = ScheduleCockpitRunner(
    optimizer_class, hyperparams, quantities=cp_quantities, plot=False
)
runner.run(
    testproblem="two_d_quadratic",
    l2_reg=0.0,  # necessary for backobs!
    num_epochs=20,
    batch_size=128,
    track_interval=1,
    plot_interval=10,
    show_plots=True,
    save_plots=False,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=lr_schedule,
)


def lr_schedule_decay(num_epochs):
    """A manually tuned schedule to emulate the loss curve of the other run."""
    r = 0.87
    a = 7.2 / 1000
    return lambda epoch: a * r ** epoch


runner = ScheduleCockpitRunner(
    optimizer_class, hyperparams, quantities=cp_quantities, plot=False
)
runner.run(
    testproblem="two_d_quadratic",
    l2_reg=0.0,  # necessary for backobs!
    num_epochs=20,
    batch_size=128,
    track_interval=1,
    plot_interval=10,
    show_plots=True,
    save_plots=False,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=lr_schedule_decay,
)
