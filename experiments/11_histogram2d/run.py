"""Own histogram2d versus third-party histogramdd."""

import json
import os
import time

import numpy
import torch

from cockpit.quantities.utils_hists import histogram2d, histogramdd


def make_uniform(w):
    params_3c3d = 895210
    batch_size = 128
    size = (2, params_3c3d * batch_size)

    uniform = torch.from_numpy(numpy.random.uniform(low=-w, high=w, size=size))
    # per default float 64, need to cast
    return uniform.to(torch.float32)


def time_numpy_histogram2d_uniform(w):
    uniform = make_uniform(w).numpy()
    x, y = uniform[0], uniform[1]

    bins = (60, 40)
    range = ((-1.0, 1.0), (-1.0, 1.0))

    start = time.time()

    numpy.histogram2d(x, y, bins, range)

    end = time.time()

    return end - start


def time_histogram2d_uniform(w, hist_func, device):
    uniform = make_uniform(w).to(device)

    bins = (60, 40)
    range = ((-1.0, 1.0), (-1.0, 1.0))

    return time_histogram2d(hist_func, uniform, bins, range, device)


def time_histogram2d(hist_func, data, bins, range, device):
    start = time.time()

    hist_func(data, bins, range)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    return end - start


def get_out_file(hist_func, device):
    HERE = os.path.abspath(__file__)
    HEREDIR = os.path.dirname(HERE)

    RESULTSDIR = os.path.join(HEREDIR, "results")
    os.makedirs(RESULTSDIR, exist_ok=True)

    savename = f"{hist_func.__name__}_{device}.json"

    return os.path.join(RESULTSDIR, savename)


def get_numpy_out_file():
    HERE = os.path.abspath(__file__)
    HEREDIR = os.path.dirname(HERE)

    RESULTSDIR = os.path.join(HEREDIR, "results")
    os.makedirs(RESULTSDIR, exist_ok=True)

    savename = "numpy_histogram2d.json"

    return os.path.join(RESULTSDIR, savename)


epsilon = 1e-2
widths = numpy.linspace(epsilon, 1 - epsilon, 20)
num_runs = 10

devices = ["cpu", "cuda"]
hist_funcs = [histogram2d, histogramdd]

if __name__ == "__main__":  # noqa: C901

    # in torch
    for hist_func in hist_funcs:
        for device in devices:
            # try load existing data
            savepath = get_out_file(hist_func, device)

            if os.path.isfile(savepath):
                with open(savepath) as json_file:
                    benchmark = json.load(json_file)
            else:
                benchmark = {}

            # iterate over runs, skip existing, update json
            for width in widths:
                if str(width) not in benchmark.keys():
                    benchmark[str(width)] = {}

                for num_run in range(num_runs):
                    if str(num_run) in benchmark[str(width)].keys():
                        print(
                            f"Setting {hist_func.__name__}, {device}, w={width},"
                            + f" run={num_run} already exists. Skipping."
                        )
                    else:
                        print(
                            f"Running {hist_func.__name__}, {device}, w={width},"
                            + f" run={num_run}"
                        )
                        this_runtime = time_histogram2d_uniform(
                            width, hist_func, device
                        )
                        print(f"Took {this_runtime:.5f}s")

                        benchmark[str(width)][str(num_run)] = this_runtime

                        with open(savepath, "w") as json_file:
                            json.dump(benchmark, json_file)

    # comparison with numpy (single threaded)
    # try load existing data
    savepath = get_numpy_out_file()

    if os.path.isfile(savepath):
        with open(savepath) as json_file:
            benchmark = json.load(json_file)
    else:
        benchmark = {}

    # iterate over runs, skip existing, update json
    for width in widths:
        if str(width) not in benchmark.keys():
            benchmark[str(width)] = {}

        for num_run in range(num_runs):
            if str(num_run) in benchmark[str(width)].keys():
                print(
                    f"Numpy setting w={width}, run={num_run} already exists. Skipping."
                )
            else:
                print(f"Running numpy, w={width}, run={num_run}")
                this_runtime = time_numpy_histogram2d_uniform(width)
                print(f"Took {this_runtime:.5f}s")

                benchmark[str(width)][str(num_run)] = this_runtime

                with open(savepath, "w") as json_file:
                    json.dump(benchmark, json_file)
