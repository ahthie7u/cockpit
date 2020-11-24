"""Run SGD on noisy quadratic with a cyclic LR schedule."""

import math

import torch
from animate import animate
from torch.optim import SGD

from cockpit import quantities
from cockpit.cockpit import configured_quantities
from cockpit.runners.scheduled_runner import ScheduleCockpitRunner
from cockpit.utils import fix_deepobs_data_dir
from deepobs.pytorch.config import set_default_device

###############################################################################
#                            Experimental settings                            #
###############################################################################

fix_deepobs_data_dir()

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.1},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def cosine_decay_restarts(
    steps_for_cycle,
    max_epochs,
    increase_restart_interval_factor=2,
    min_lr=0.0,
    restart_discount=0.0,
):
    """Cyclic LR schedule with restarts."""
    lr_factors = []

    step = 0
    cycle = 0

    for _ in range(0, max_epochs + 1):
        step += 1
        completed_fraction = step / steps_for_cycle
        cosine_decayed = 0.5 * (1 + math.cos(math.pi * completed_fraction))
        lr_factors.append(cosine_decayed)

        if completed_fraction == 1:
            step = 0
            cycle += 1
            steps_for_cycle = steps_for_cycle * increase_restart_interval_factor

    return lr_factors


USE_COSINE_DECAY_RESTARTS = True

if USE_COSINE_DECAY_RESTARTS:

    def lr_schedule(num_epochs):
        factors = cosine_decay_restarts(steps_for_cycle=7, max_epochs=num_epochs)

        return lambda epoch: factors[epoch]


else:

    def lr_schedule(num_epochs):
        return lambda epoch: 1.0


def logarithmic_schedule(start=0, end=19, steps=300, base=2):
    track_at = torch.logspace(start, end, steps=steps, base=base).int()

    def schedule(global_step):
        return global_step in track_at or global_step == 0

    return schedule


def adapt_schedule(global_step):
    return global_step == 0


plot_schedule = logarithmic_schedule()
track_schedule = plot_schedule

# initialize the quantities manually
quants = []
for q in configured_quantities("full"):
    if q == quantities.BatchGradHistogram1d:
        quants.append(
            q(
                track_schedule=track_schedule,
                adapt_schedule=adapt_schedule,
                verbose=True,
            )
        )
    elif q == quantities.BatchGradHistogram2d:
        pass
        quants.append(
            q(
                track_schedule=track_schedule,
                adapt_schedule=adapt_schedule,
                verbose=True,
                keep_individual=False,
            )
        )
    else:
        quants.append(q(track_schedule=track_schedule, verbose=True))

# device
FORCE_CPU = False
if FORCE_CPU:
    set_default_device("cpu")

runner = ScheduleCockpitRunner(
    optimizer_class, hyperparams, quantities=quants, plot_schedule=plot_schedule
)

# DeepOBS metric evaluation every epoch
DEEPOBS_EVAL = True
if not DEEPOBS_EVAL:

    def hotfix_disable_eval():
        return False

    runner._should_eval = hotfix_disable_eval

runner.run(
    testproblem="quadratic_deep",
    l2_reg=0.0,  # necessary for backobs!
    track_interval=1,
    plot_interval=1,
    show_plots=False,
    save_plots=True,
    save_final_plot=True,
    save_animation=True,
    lr_schedule=lr_schedule,
    skip_if_exists=True,
)

animate("quadratic_deep")
