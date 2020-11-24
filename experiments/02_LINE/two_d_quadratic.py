# -*- coding: utf-8 -*-
"""A simple N-Dimensional Noisy Quadratic Problem with Deep Learning eigenvalues."""

import numpy as np
import torch

import backobs
import deepobs
from deepobs.pytorch.datasets.quadratic import quadratic
from deepobs.pytorch.testproblems.testproblem import UnregularizedTestproblem
from deepobs.pytorch.testproblems.testproblems_modules import net_quadratic_deep


class two_d_quadratic(UnregularizedTestproblem):
    r"""DeepOBS test problem class for a 2D quadratic test problem.

    Args:
      batch_size (int): Batch size to use.
      l2_reg (float): No L2-Regularization (weight decay) is used in this
          test problem. Defaults to ``None`` and any input here is ignored.

    Attributes:
        data: The DeepOBS data set class for the quadratic problem.
        loss_function: None. The output of the model is the loss.
        net: The DeepOBS subclass of torch.nn.Module that is trained for this
            tesproblem (net_quadratic_deep).
    """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new quadratic deep test problem instance.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-Regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(two_d_quadratic, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the quadratic test problem."""
        hessian = self._make_hessian()
        self._hessian = hessian
        self.net = net_quadratic_deep(hessian)
        self.data = quadratic(
            self._batch_size, dim=2, noise_level=0.0, train_size=self._batch_size
        )
        self.net.to(self._device)
        self.loss_function = torch.nn.MSELoss
        self.regularization_groups = self.get_regularization_groups()

    @staticmethod
    def _make_hessian(rotate=30):
        rotate = rotate * np.pi / 180
        Q = np.diag([0.5, 10.0])

        R = np.array(
            [[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]]
        )
        Hessian = R.dot(Q).dot(R.T)

        return torch.from_numpy(Hessian).to(torch.float32)

    def get_batch_loss_and_accuracy_func(
        self, reduction="mean", add_regularization_if_available=True
    ):
        """Get new batch and create forward function that calculates loss on that batch.

        Args:
            reduction (str): The reduction that is used for returning the loss.
                Can be 'mean', 'sum' or 'none' in which case each indivual loss
                in the mini-batch is returned as a tensor.
            add_regularization_if_available (bool): If true, regularization is
                added to the loss.

        Returns:
            callable:  The function that calculates the loss/accuracy on the
                current batch.
        """
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        def forward_func():
            # in evaluation phase is no gradient needed
            if self.phase in ["train_eval", "test", "valid"]:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    loss = self.loss_function(reduction=reduction)(outputs, labels)
            else:
                outputs = self.net(inputs)
                loss = self.loss_function(reduction=reduction)(outputs, labels)

            accuracy = 0.0

            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                regularizer_loss = torch.tensor(0.0, device=torch.device(self._device))

            return loss + regularizer_loss, accuracy

        return forward_func


def register():
    """Let DeepOBS and BackOBS know about the existence of the toy problem."""
    # DeepOBS
    deepobs.pytorch.testproblems.two_d_quadratic = two_d_quadratic

    # for CockpitPlotter
    if "scalar" in deepobs.config.DATA_SET_NAMING.keys():
        assert deepobs.config.DATA_SET_NAMING["scalar"] == "Scalar"
    else:
        deepobs.config.DATA_SET_NAMING["scalar"] = "Scalar"

    if "deep" in deepobs.config.TP_NAMING.keys():
        assert deepobs.config.TP_NAMING["deep"] == "Deep"
    else:
        deepobs.config.TP_NAMING["deep"] = "deep"

    # BackOBS
    backobs.utils.ALL += (two_d_quadratic,)
    backobs.utils.REGRESSION += (two_d_quadratic,)
    backobs.utils.SUPPORTED += (two_d_quadratic,)
    backobs.integration.SUPPORTED += (two_d_quadratic,)
