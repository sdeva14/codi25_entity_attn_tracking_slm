"""
Aggregate attention-flow statistics from JSON log files (type1/type2/type3).
Usage: python stat_analysis.py [--output_dir DIR] [--sub_dir SUB]
  or:  read_log_and_compute_stats(output_dir, sub_dir, output_name) for custom paths.
"""
import argparse
from utils.stats import read_log_and_compute_stats


def main():
    parser = argparse.ArgumentParser(description="Compute mean/std from attention flow logs")
    parser.add_argument("--output_dir", default="log_entity_attn", help="Log directory (same as attn_flow_test.py output)")
    parser.add_argument("--sub_dir", default="microsoft-Phi-3.5-mini-instruct", help="Model subdir (e.g. microsoft-Phi-3.5-mini-instruct)")
    args = parser.parse_args()

    for name in ("type1.log", "type2.log", "type3.log"):
        try:
            avg, std = read_log_and_compute_stats(args.output_dir, args.sub_dir, name)
            print("{} Avg(Std): {} ({})".format(name.replace(".log", ""), avg, std))
        except FileNotFoundError:
            print("{} not found at {}/{}".format(name, args.output_dir, args.sub_dir))


if __name__ == "__main__":
    main()
