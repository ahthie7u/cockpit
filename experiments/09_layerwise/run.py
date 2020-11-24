"""Compute the 2d histogram for different data pre-processing steps."""

from torch.optim import SGD

from cockpit import quantities
from cockpit.experiments.cifar10_3c3d import (
    cifar10_3c3dsig,
    make_cifar10transform_3c3d,
    make_cifar10transform_3c3dsig,
)
from cockpit.experiments.utils import register
from cockpit.runners.scheduled_runner import ScheduleCockpitRunner
from cockpit.utils import fix_deepobs_data_dir

# from torchvision import transforms


register(cifar10_3c3dsig)
# register(cifar10_3c3dtanh)

fix_deepobs_data_dir()

optimizer_class = SGD
hyperparams = {"lr": {"type": float, "default": 0.001}}


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
            keep_individual=True,
        )
    ]

    return quants


def scale254(tensor, verbose=False):
    scaled = 254.0 * tensor

    if verbose:
        print(f"Raw min: {tensor.min():6.4f}, raw max: {tensor.max():6.4f}")
        print(f"    min: {scaled.min():6.4f},     max: {scaled.max():6.4f}")

    return scaled


TRANSFORMS = {
    # "raw": transforms.Compose([transforms.ToTensor()]),
    # "scale254": transforms.Compose([transforms.ToTensor(), scale254]),
    # "gray": transforms.Compose(
    #     [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
    # ),
    # "graymean": transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         lambda tensor: tensor.mean(0, keepdim=True).repeat(3, 1, 1),
    #     ]
    # ),
    # "graysum": transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         lambda tensor: tensor.sum(0, keepdim=True).repeat(3, 1, 1),
    #     ]
    # ),
    # "normal": transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             (0.49139968, 0.48215841, 0.44653091),
    #             (0.24703223, 0.24348513, 0.26158784),
    #         ),
    #     ]
    # ),
}


# def make_scale_fn(scale):
#     def scale_fn(tensor):
#         print(f"Scaling by: {scale}")
#         return tensor * scale

#     return scale_fn


# for scale in [0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5]:
#     scale_str = f"{scale:05.2f}".replace(".", "c")

#     transform_name = f"normalscale{scale_str}"

#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 (0.49139968, 0.48215841, 0.44653091),
#                 (0.24703223, 0.24348513, 0.26158784),
#             ),
#             make_scale_fn(scale),
#         ]
#     )
#     TRANSFORMS[transform_name] = transform


PROBLEMS = []

# build and register testproblems with ReLU
PROBLEMS += ["cifar10_3c3d"]

for trafo_name, trafo in TRANSFORMS.items():
    make_cifar10transform_3c3d(trafo, trafo_name)

PROBLEMS += [f"cifar10{trafo_name}_3c3d" for trafo_name in TRANSFORMS.keys()]

# build and register testproblems with Sigmoids
PROBLEMS += ["cifar10_3c3dsig"]

for trafo_name, trafo in TRANSFORMS.items():
    make_cifar10transform_3c3dsig(trafo, trafo_name)

PROBLEMS += [f"cifar10{trafo_name}_3c3dsig" for trafo_name in TRANSFORMS.keys()]

# build and register testproblems with Tanh
# PROBLEMS += ["cifar10_3c3dtanh"]

# for trafo_name, trafo in TRANSFORMS.items():
#     make_cifar10transform_3c3dtanh(trafo, trafo_name)

# PROBLEMS += [f"cifar10{trafo_name}_3c3dtanh" for trafo_name in TRANSFORMS.keys()]


if __name__ == "__main__":

    for problem in PROBLEMS:
        runner = ScheduleCockpitRunner(
            optimizer_class,
            hyperparams,
            quantities=make_quantities(),
            secondary_screen=True,
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
