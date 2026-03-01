"""
Aggregate attention-flow statistics from JSON log files (type1/type2/type3).
Usage: python stat_analysis.py
  or:  get_avg_from_lists(output_dir, sub_dir, output_name) for custom paths.
"""
import os
import json
import argparse

import numpy as np


def get_avg_from_lists(output_dir, sub_dir, output_name):
    """Read a JSON log (list of lists), return (mean, std) over all values."""
    path = os.path.join(output_dir, sub_dir, output_name)
    with open(path) as f:
        list_vals = json.load(f)
    flat = [x for xs in list_vals for x in xs]
    arr = np.array(flat, dtype=float)
    return np.nanmean(arr), np.nanstd(arr)


def main():
    parser = argparse.ArgumentParser(description="Compute mean/std from attention flow logs")
    parser.add_argument("--output_dir", default="log_entity_attn", help="Log directory (same as attn_flow_test.py output)")
    parser.add_argument("--sub_dir", default="microsoft-Phi-3.5-mini-instruct", help="Model subdir (e.g. microsoft-Phi-3.5-mini-instruct)")
    args = parser.parse_args()

    for name in ("type1.log", "type2.log", "type3.log"):
        try:
            avg, std = get_avg_from_lists(args.output_dir, args.sub_dir, name)
            print("{} Avg(Std): {} ({})".format(name.replace(".log", ""), avg, std))
        except FileNotFoundError:
            print("{} not found at {}/{}".format(name, args.output_dir, args.sub_dir))


if __name__ == "__main__":
    main()
