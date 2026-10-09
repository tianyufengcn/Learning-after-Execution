from __future__ import annotations

"""Frozen Image-to-TikZ task contract reused from Direct-SFT.

The prompt below is copied verbatim from
Qwen3VL8B-i2t-revision-sft-v0.5.1/src/i2t_revision_sft/prompting.py, where it is
itself documented as copied verbatim from direct_sft.

Keeping the task contract here makes the GRPO experiment change the optimization
signal only; it does not silently change the Image-to-TikZ prompt or image protocol.
"""

PROMPT_VERSION = "direct_i2t_v0.2.0"
IMAGE_SIZE = 448

# Historical data/eval protocol metadata. These values are intentionally metadata
# rather than implicit tokenizer logic so that runs can record the exact contract.
SFT_GT_TIKZ_TOKEN_CAP = 2048
FORMAL_DIRECT_MAX_NEW_TOKENS = 8192
FORMAL_DIRECT_TEMPERATURE = 0.7
FORMAL_DIRECT_TOP_P = 0.9

# GRPO keeps the Direct task contract but increases sampling exploration, following
# TikZilla's released RL recipe.
GRPO_TEMPERATURE = 1.0
GRPO_TOP_P = 0.9

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


def direct_prompt_messages() -> list[dict[str, str]]:
    """Return the textual chat used by TRL VLM-GRPO.

    TRL's multimodal preprocessing injects the dataset's ``image`` column before
    the first user text block. Therefore this yields the same effective message
    order as Direct-SFT: image first, then the exact Direct prompt.
    """

    return [{"role": "user", "content": DIRECT_I2T_PROMPT}]


def contract_dict() -> dict[str, object]:
    return {
        "prompt_version": PROMPT_VERSION,
        "image_size": IMAGE_SIZE,
        "sft_gt_tikz_token_cap": SFT_GT_TIKZ_TOKEN_CAP,
        "formal_direct_max_new_tokens": FORMAL_DIRECT_MAX_NEW_TOKENS,
        "formal_direct_temperature": FORMAL_DIRECT_TEMPERATURE,
        "formal_direct_top_p": FORMAL_DIRECT_TOP_P,
        "grpo_temperature": GRPO_TEMPERATURE,
        "grpo_top_p": GRPO_TOP_P,
    }
