"""Compute the 2d histogram for different data pre-processing steps."""

import os
import random

import numpy
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from cockpit.experiments.cifar10_3c3d import make_cifar10transform_3c3d
from cockpit.utils import fix_deepobs_data_dir

fix_deepobs_data_dir()

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "fig_samples")
os.makedirs(SAVEDIR, exist_ok=True)


def set_deepobs_seed(seed=0):
    """Set all seeds used by DeepOBS."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def imshow(x):
    if x.max() > 1:
        x = x.int()
    # channel-first to channel-last
    plt.imshow(numpy.transpose(x.numpy(), (1, 2, 0)), aspect="auto")


def visualize_inputs(tproblem_cls, num_images):
    set_deepobs_seed(42)
    tproblem = tproblem_cls(batch_size=num_images, l2_reg=0.0)

    tproblem.set_up()
    tproblem.train_init_op()

    X, _ = tproblem._get_next_batch()

    width, height = X.shape[-2:]

    for idx in range(num_images):
        savepath = get_out_file(tproblem_cls, idx)

        # turn off axes and get resolution right (https://stackoverflow.com/a/821888)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(width, height)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        imshow(X[idx])

        plt.savefig(savepath, dpi=1)
        plt.close()
        plt.axis("on")


def get_out_file(tproblem_cls, sample_idx):
    return os.path.join(SAVEDIR, f"{tproblem_cls.__name__}_sample_{sample_idx:02d}.png")


def make_and_register_tproblems():
    def raw(tensor, verbose=False):
        if verbose:
            print(f"[raw] min: {tensor.min():6.4f}, max: {tensor.max():6.4f}")

        return tensor

    def scale255(tensor, verbose=False):
        scaled = 255 * tensor

        if verbose:
            print(f"[scale255] min: {scaled.min():6.4f}, max: {scaled.max():6.4f}")

        return scaled

    TRANSFORMS = {
        "raw": transforms.Compose([transforms.ToTensor(), raw]),
        "scale255": transforms.Compose([transforms.ToTensor(), scale255]),
    }

    for trafo_name, trafo in TRANSFORMS.items():
        make_cifar10transform_3c3d(trafo, trafo_name)


make_and_register_tproblems()

if __name__ == "__main__":
    from deepobs.pytorch.testproblems import cifar10raw_3c3d, cifar10scale255_3c3d

    # number of images to be visualized
    num_images = 4
    for problem_cls in [cifar10raw_3c3d, cifar10scale255_3c3d]:
        visualize_inputs(problem_cls, num_images)
