"""Shared stats: mean/std over lists (flat or nested) and from log files."""
import json
import os
from typing import List, Union
import numpy as np


def mean_std(values: List[Union[float, int]]) -> tuple:
    """Compute (nanmean, nanstd) over a flat list of numbers."""
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def get_avg_from_lists(list_val: list) -> list:
    """
    Flatten list_val (if nested) and return [mean, std].
    Used by attn_flow_test and ft_attn_flow_test for type1--type4 aggregates.
    """
    if all(isinstance(item, list) for item in list_val):
        flat = [x for xs in list_val for x in xs]
    elif all(not isinstance(item, list) for item in list_val):
        flat = list_val
    else:
        flat = list_val
    m, s = mean_std(flat)
    return [m, s]


def read_log_and_compute_stats(output_dir: str, sub_dir: str, output_name: str) -> tuple:
    """Read a JSON log (list of lists), return (mean, std) over all values."""
    path = os.path.join(output_dir, sub_dir, output_name)
    with open(path) as f:
        list_vals = json.load(f)
    flat = [x for xs in list_vals for x in xs]
    return mean_std(flat)
