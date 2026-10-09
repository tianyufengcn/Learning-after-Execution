from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IMAGE_SIZE = 448
PROMPT_VERSION = "workflow_grpo_v0.1.0"
CURRENT_TIKZ_PLACEHOLDER = "<<CURRENT_TIKZ_SOURCE>>"
FAILURE_STATUS_PLACEHOLDER = "<<STAGE0_FAILURE_STATUS>>"

# Copied verbatim from the Direct-SFT / Direct-GRPO contract.
DIRECT_I2T_PROMPT = r"""You are an expert Image-to-TikZ model and LaTeX/TikZ developer.
Study the provided target image carefully and reconstruct it as a high-quality
TikZ figure. Focus on the visible objects, geometry, topology, text and
mathematical labels, colors, line and arrow styles, relative sizes, alignment,
and overall layout. Choose the native TikZ, PGFPlots, circuitikz, tikz-cd, or
other LaTeX drawing constructs that best reproduce the image. Do not invent or
omit important visible content when it can be represented faithfully.

OUTPUT REQUIREMENTS

- Return only raw LaTeX source code. Do not output Markdown fences,
  explanations, review prose, or any text outside the document.
- Output exactly one complete standalone document. Start with
  \documentclass[tikz]{standalone}; only when a small margin is needed to
  prevent visible clipping, use border=1pt, border=2pt, or border=3pt in the
  same document class.
- Include every package and TikZ library required by the generated code in the
  preamble. The document must be self-contained, use native TikZ/PGF drawing,
  and compile successfully with pdflatex with shell escape disabled. Do not
  load external files or assets.
- Reconstruct the target image as faithfully as possible while preserving
  compilability and document completeness.
- Include exactly one \begin{document} and one matching \end{document}. The
  final non-whitespace content must be \end{document}; stop immediately after
  it."""

# Copied from Qwen3VL8B-i2t-revision-sft-v0.5.1. The only change in this repo
# is that the current state is generated on-policy at every workflow rollout.
REVISION_PROMPT_TEMPLATE = rf"""You are an expert Image-to-TikZ revision model and LaTeX/TikZ developer.

You are given three aligned inputs describing one revision state:
1. TARGET / REFERENCE IMAGE: the first image is the visual ground truth that the final TikZ should reproduce.
2. CURRENT RENDERED IMAGE: the second image is the rendering produced by the current TikZ program below.
3. CURRENT TIKZ PROGRAM: the editable source code that produced the current rendered image.

The current rendered image and current TikZ program describe the same current state. Compare that state against the target image and revise the current TikZ program so that the new rendered result matches the target image as faithfully as possible. Use the target image as the authority for visible content. Use the current rendered image as execution evidence for what the current program actually draws, and use the current TikZ as the program to edit.

Focus on visible objects, geometry, topology, text and mathematical labels, colors, line and arrow styles, relative sizes, alignment, spacing, and overall layout. Preserve correct parts when appropriate, while fixing missing, extra, misplaced, malformed, or visually incorrect elements. Do not add details that are not visually supported by the target image merely because they appear in the current code.

CURRENT TIKZ PROGRAM
{CURRENT_TIKZ_PLACEHOLDER}

OUTPUT REQUIREMENTS

- Return only the complete revised raw LaTeX source code. Do not output Markdown fences, explanations, review prose, analysis, or any text outside the document.
- Output exactly one complete standalone document. Start with \documentclass[tikz]{{standalone}}; only when a small margin is needed to prevent visible clipping, use border=1pt, border=2pt, or border=3pt in the same document class.
- Include every package and TikZ library required by the revised code in the preamble. The document must be self-contained, use native TikZ/PGF drawing, and compile successfully with pdflatex with shell escape disabled. Do not load external files or assets.
- The revised program may keep, modify, reorganize, or replace parts of the current program as needed to improve visual fidelity to the target image. Return the full document, not a patch or diff.
- Include exactly one \begin{{document}} and one matching \end{{document}}. The final non-whitespace content must be \end{{document}}; stop immediately after it."""

# Stage-1 fallback when Stage-0 has no valid render. Keep this deliberately
# minimal: no reviewer, no line-level diagnosis, no compiler log dump.
REPAIR_PROMPT_TEMPLATE = rf"""You are an expert Image-to-TikZ model and LaTeX/TikZ developer.

The first-stage TikZ attempt did not produce a valid rendered image. Use the target image as visual ground truth and the current TikZ attempt as editable state. Complete or repair the program into one complete, self-contained, compilable TikZ document that matches the target image as faithfully as possible.

STAGE-0 STATUS
{FAILURE_STATUS_PLACEHOLDER}

CURRENT TIKZ ATTEMPT
{CURRENT_TIKZ_PLACEHOLDER}

OUTPUT REQUIREMENTS

- Return only raw LaTeX source code. Do not output Markdown fences, explanations, review prose, analysis, or text outside the document.
- Return one complete standalone document, not a patch or diff.
- Preserve useful existing content when appropriate, but freely repair, complete, reorganize, or replace the current attempt when needed.
- The document must compile with shell escape disabled and must not depend on external assets.
- Include exactly one \begin{{document}} and one matching \end{{document}}. The final non-whitespace content must be \end{{document}}; stop immediately after it."""


def _replace_once(template: str, placeholder: str, value: str) -> str:
    if placeholder not in template:
        raise ValueError(f"Template missing placeholder {placeholder!r}")
    return template.replace(placeholder, value, 1)


def direct_messages(target_image: Any) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": target_image},
            {"type": "text", "text": DIRECT_I2T_PROMPT},
        ],
    }]


def revision_messages(target_image: Any, current_image: Any, current_tikz: str) -> list[dict[str, Any]]:
    prompt = _replace_once(REVISION_PROMPT_TEMPLATE, CURRENT_TIKZ_PLACEHOLDER, current_tikz)
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "TARGET / REFERENCE IMAGE (visual ground truth):"},
            {"type": "image", "image": target_image},
            {"type": "text", "text": "CURRENT RENDERED IMAGE (produced by the current TikZ program):"},
            {"type": "image", "image": current_image},
            {"type": "text", "text": prompt},
        ],
    }]


def repair_messages(target_image: Any, current_tikz: str, failure_status: str) -> list[dict[str, Any]]:
    prompt = _replace_once(REPAIR_PROMPT_TEMPLATE, CURRENT_TIKZ_PLACEHOLDER, current_tikz)
    prompt = _replace_once(prompt, FAILURE_STATUS_PLACEHOLDER, failure_status)
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "TARGET / REFERENCE IMAGE (visual ground truth):"},
            {"type": "image", "image": target_image},
            {"type": "text", "text": prompt},
        ],
    }]


def contract_dict() -> dict[str, Any]:
    return {
        "version": PROMPT_VERSION,
        "image_size": IMAGE_SIZE,
        "routing": {
            "stage0_render_success": "visual_revision",
            "stage0_render_failure": "repair_or_complete",
        },
    }
