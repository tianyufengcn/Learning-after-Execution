#!/usr/bin/env python3
from __future__ import annotations

import argparse

from i2t_workflow_grpo.config import load_config
from i2t_workflow_grpo.trainer import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Two-stage on-policy Image-to-TikZ workflow GRPO")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    run_training(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
