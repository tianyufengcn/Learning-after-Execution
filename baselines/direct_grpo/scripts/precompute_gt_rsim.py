#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from i2t_grpo.config import load_config
from i2t_grpo.rewards.rsim import RSimV2Scorer


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute persistent RSim GT features for the fixed GRPO cohort")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    from datasets import load_from_disk

    ds = load_from_disk(cfg["data"]["dataset_path"])
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--num-shards must be >= 1 and --shard-index in [0, num-shards)")
    # Deterministic, re-entrant sharding: sample_index % num_shards == shard_index.
    paths = [
        str(x) for i, x in enumerate(ds["gt_image_path"])
        if i % args.num_shards == args.shard_index
    ]
    r = cfg["reward"]
    scorer = RSimV2Scorer(
        model_path=r["model_path"],
        cache_dir=r["feature_cache_dir"],
        detikzify_repo=r.get("detikzify_repo"),
        batch_size=r.get("score_batch_size", 16),
        mem_cache_size=r.get("score_mem_cache_size", 4096),
        emd_workers=r.get("emd_workers", 4),
        device=r.get("device"),
    )
    start = time.monotonic()
    stats = scorer.precompute_gt(paths)
    stats["elapsed_s"] = time.monotonic() - start
    summary_name = "precompute_summary.json" if args.num_shards == 1 else f"precompute_summary_shard{args.shard_index}.json"
    out = Path(r["feature_cache_dir"]) / summary_name
    stats["num_shards"] = args.num_shards
    stats["shard_index"] = args.shard_index
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[ok] summary -> {out}")


if __name__ == "__main__":
    main()
