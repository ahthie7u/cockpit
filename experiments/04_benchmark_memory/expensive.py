import glob
import os

from shared import DIR, parse, report_memory, run, set_up, skip_if_exists

from backpack import extensions
from cockpit.quantities import BatchGradHistogram1d
from cockpit.quantities.utils_hists import transform_grad_batch_abs_max


class BatchGradHistogram1dExpensive(BatchGradHistogram1d):
    """One-dimensional histogram of individual gradient elements.

    Computes all individual gradients before creating a histogram.
    """

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.is_active(global_step) or self._adapt_schedule(global_step):
            ext.append(extensions.BatchGrad())

        return ext

    def compute(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        self.__pre_compute(global_step, params, batch_loss)
        return super().compute(global_step, params, batch_loss)

    def __pre_compute(self, global_step, params, batch_loss):
        """Perform computations that optimized version performs during backprop."""
        if self.is_active(global_step):
            for p in params:
                p.grad_batch_transforms = {}
                p.grad_batch_transforms["hist_1d"] = self._compute_histogram(
                    p.grad_batch
                )

                if self._adapt_schedule(global_step):
                    p.grad_batch_transforms[
                        "grad_batch_abs_max"
                    ] = transform_grad_batch_abs_max(p)


def get_out_files(testproblem):
    """Return all available output files for a test problem."""
    pattern = os.path.join(DIR, f"{testproblem}_expensive_*.csv")
    return glob.glob(pattern)


def out_file(testproblem, num_run):
    """Return save path for a specific run of a test problem"""
    return os.path.join(DIR, f"{testproblem}_expensive_{num_run:03d}.csv")


if __name__ == "__main__":
    set_up()

    testproblem, num_run = parse()
    filename = out_file(testproblem, num_run)

    skip_if_exists(filename)

    def benchmark_fn():
        quants = [BatchGradHistogram1dExpensive(remove_outliers=True)]
        run(quants, testproblem)

    data = report_memory(benchmark_fn)
    data.to_csv(filename)
