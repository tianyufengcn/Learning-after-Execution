#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Image as HFImage, load_dataset, load_from_disk
from PIL import Image

from i2t_grpo.prompt_contract import DIRECT_I2T_PROMPT, IMAGE_SIZE, PROMPT_VERSION, contract_dict, direct_prompt_messages


def load_source(path: str):
    p = Path(path)
    if p.is_dir():
        return load_from_disk(str(p))
    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(p), split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(p), split="train")
    raise ValueError(f"Unsupported source: {path}")


def pick(row: dict, explicit: str | None, candidates: list[str]):
    if explicit:
        return row.get(explicit)
    for key in candidates:
        if row.get(key) is not None:
            return row[key]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Adapt an already selected Direct-SFT subset/manifest into the TRL VLM-GRPO dataset format. "
            "For the formal GRPO-1K cohort, perform length/source-balanced sampling with the historical sampler first, "
            "then point this adapter at that fixed 1K manifest."
        )
    )
    ap.add_argument("--source", required=True, help="Fixed Direct-4K/GRPO-1K JSONL/JSON/parquet/HF dataset")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = keep all rows in the supplied fixed cohort")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--shuffle", action="store_true", help="Optional only for smoke/pilot; formal fixed cohort should normally stay ordered")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--image-column", default=None)
    ap.add_argument("--code-column", default=None)
    ap.add_argument("--id-column", default=None)
    ap.add_argument("--require-448", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--export-decoded-images", action="store_true", help="Only use if the source has PIL images but no stable image paths")
    args = ap.parse_args()

    source = load_source(args.source)
    if args.shuffle:
        source = source.shuffle(seed=args.seed)
    root = Path(args.image_root).resolve() if args.image_root else None
    rows = []
    seen_ids: set[str] = set()
    stats = {"missing_image": 0, "bad_size": 0, "duplicate_id": 0, "exported_image": 0}

    for idx, row in enumerate(source):
        image = pick(row, args.image_column, ["gt_image_path", "image_path", "image", "rendered_image", "target_image", "png_path"])
        code = pick(row, args.code_column, ["gt_tikz", "tikz", "code", "target", "label"])
        if not isinstance(code, str) or not code.strip():
            code_path = pick(row, args.code_column, [
                "reference_tikz_path", "gt_tikz_path", "tikz_path", "code_path", "generated_tikz_path",
            ])
            if code_path and Path(code_path).is_file():
                code = Path(code_path).read_text(encoding="utf-8", errors="replace")
        sample_id = pick(row, args.id_column, ["sample_id", "id", "file_id", "uid"])
        sample_id = str(sample_id if sample_id is not None else idx)
        if sample_id in seen_ids:
            stats["duplicate_id"] += 1
            continue

        if isinstance(image, dict):
            image = image.get("path")
        if not isinstance(image, str):
            if not args.export_decoded_images:
                stats["missing_image"] += 1
                continue
            try:
                out_img_dir = Path(args.output).parent / (Path(args.output).name + "_images")
                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_img_dir / f"{sample_id}.png"
                image.save(out_path)
                image = str(out_path)
                stats["exported_image"] += 1
            except Exception:
                stats["missing_image"] += 1
                continue

        path = Path(image)
        if not path.is_absolute() and root is not None:
            path = root / path
        path = path.resolve()
        if not path.exists():
            stats["missing_image"] += 1
            continue
        if args.require_448:
            try:
                with Image.open(path) as im:
                    if im.size != (IMAGE_SIZE, IMAGE_SIZE):
                        stats["bad_size"] += 1
                        continue
            except Exception:
                stats["missing_image"] += 1
                continue

        seen_ids.add(sample_id)
        out_row = {
            "sample_id": sample_id,
            # Current TRL VLM-GRPO injects the separate `image` column before
            # the first user text block. This yields exactly Direct-SFT's order:
            # target image first, exact Direct prompt second.
            "prompt": direct_prompt_messages(),
            "prompt_version": PROMPT_VERSION,
            "image": str(path),
            "gt_image_path": str(path),
            "gt_tikz": code if isinstance(code, str) else "",
            "gt_tikz_chars": len(code) if isinstance(code, str) else 0,
        }
        # Preserve useful provenance fields when they already exist; no new
        # semantic labels are invented by this adapter.
        for key in ("source", "category", "tikz_type", "parquet", "parquet_path", "parent_id", "code_tokens", "code_length"):
            if row.get(key) is not None:
                out_row[key] = row[key]
        rows.append(out_row)
        if args.limit > 0 and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit("No valid image rows found. Pass --image-column/--image-root explicitly.")
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", HFImage())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    manifest = {
        "source": str(Path(args.source).resolve()),
        "rows": len(ds),
        "seed": args.seed,
        "shuffle": bool(args.shuffle),
        "limit": args.limit,
        "contract": contract_dict(),
        "adapter_stats": stats,
    }
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] saved {len(ds)} GRPO prompts -> {out}")
    print("[contract]", json.dumps(manifest["contract"], ensure_ascii=False))
    print("[columns]", ds.column_names)
    print("[adapter_stats]", stats)


if __name__ == "__main__":
    main()
