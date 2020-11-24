"""Utility functions for benchmarks."""

import subprocess

from pytest_benchmark.plugin import (
    get_commit_info,
    pytest_benchmark_generate_machine_info,
)

from cockpit.cockpit import configured_quantities
from cockpit.quantities import Time


def settings_individual():
    """Return benchmark quantities to measure overhead of individual quantities."""
    configs = {}

    for q in configured_quantities("full"):
        if not q == Time:
            configs[q.__name__] = [q, Time]

    return configs


def settings_configured():
    configs = {}

    for label in ["economy", "business", "full"]:
        configs[label] = configured_quantities(label)

    return configs


def settings_baseline():
    return {"baseline": [Time]}


def get_sys_info():
    """Return system information for benchmark."""
    machine_info = pytest_benchmark_generate_machine_info()
    machine_info.pop("node")

    info = {
        "machine": machine_info,
        "commit": get_commit_info(),
    }
    try:
        info["gpu"] = _get_gpu_info()
    except Exception:
        info["gpu"] = "Unknown"

    return info


def _get_gpu_info(keys=("Product Name", "CUDA Version")):
    """Parse output of nvidia-smi into a python dictionary.

    Link:
        - https://gist.github.com/telegraphic/ecb8161aedb02d3a09e39f9585e91735
    """
    sp = subprocess.Popen(
        ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_list = sp.communicate()[0].decode("utf-8").split("\n")

    info = {}

    for item in out_list:
        try:
            key, val = item.split(":")
            key, val = key.strip(), val.strip()
            if key in keys:
                info[key] = val
        except Exception:
            pass

    return info
