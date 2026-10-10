#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.config import load_config
from src.trainer import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Learning after Execution with action-specific credit")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    run_training(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
