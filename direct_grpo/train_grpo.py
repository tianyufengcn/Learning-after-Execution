#!/usr/bin/env python3
from __future__ import annotations

import argparse

from direct_grpo.config import load_config
from direct_grpo.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-to-TikZ render-aware GRPO training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a YAML value; may be repeated")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    run_training(cfg)


if __name__ == "__main__":
    main()
