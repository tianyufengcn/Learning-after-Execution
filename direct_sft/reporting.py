from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8")


def write_training_summary(output_dir: Path, meta: dict[str,Any], metrics: dict[str,Any]):
    write_json(output_dir/"reports"/"training_summary.json",{"meta":meta,"metrics":metrics})
    lines=[f"# Direct Image-to-TikZ SFT — {meta.get('name','run')}","","## Configuration"]
    for k,v in meta.items(): lines.append(f"- **{k}**: `{v}`")
    lines += ["","## Final trainer metrics"]
    for k,v in metrics.items(): lines.append(f"- **{k}**: `{v}`")
    (output_dir/"reports").mkdir(parents=True,exist_ok=True)
    (output_dir/"reports"/"training_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
