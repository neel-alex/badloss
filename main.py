#!/usr/bin/env python
"""Run one backdoor experiment: poison the training set with the chosen attack(s), apply a defense, and report
clean accuracy, attack success rates and detection metrics (also written to <output_dir>/results.json)."""
import os
import sys

import torch

import config
from defenses import get_defense
from experiment import Experiment, TimeLimitReached

RESUME_EXIT_CODE = 75  # EX_TEMPFAIL: rerun the same command to resume


def main():
    # Bitwise-reproducible runs: deterministic kernels only (raises if an op has no deterministic implementation)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    args = config.parse_args()
    exp = Experiment(args)
    try:
        get_defense(args.defense).run(exp)
    except TimeLimitReached as e:
        print(f"Time limit reached: {e}. Rerun the same command to resume.")
        sys.exit(RESUME_EXIT_CODE)
    exp.finish()


if __name__ == "__main__":
    main()
