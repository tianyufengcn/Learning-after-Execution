from __future__ import annotations

import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:latex|tex|tikz)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DOC_RE = re.compile(r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}.*?\\end\{document\})", re.DOTALL)
_TIKZ_RE = re.compile(r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})", re.DOTALL)


def completion_to_text(completion: Any) -> str:
    """Normalize TRL plain-text or conversational completions to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            )
        return str(content)
    if isinstance(completion, list):
        # Conversational reward input is commonly a list containing one assistant message.
        pieces = [completion_to_text(x) for x in completion]
        return "\n".join(x for x in pieces if x)
    return str(completion)


def extract_tikz_document(text: str, wrap_tikzpicture: bool = True) -> str | None:
    """Extract a compilable-looking LaTeX document from a model completion.

    Preference order:
      1. complete ``\\documentclass ... \\end{document}`` document;
      2. fenced block containing a complete document;
      3. bare tikzpicture wrapped in a standalone document.

    This intentionally does *not* repair semantic/syntax errors. Compilation remains the
    environment/verifier, matching the render-aware RL setup.
    """
    if not isinstance(text, str):
        return None
    cleaned = text.replace("\x00", "").strip()

    # Complete document anywhere in the completion.
    match = _DOC_RE.search(cleaned)
    if match:
        return match.group(1).strip()

    # Some models put the program in a fenced block after a short sentence.
    fence = _FENCE_RE.search(cleaned)
    if fence:
        inner = fence.group(1).strip()
        match = _DOC_RE.search(inner)
        if match:
            return match.group(1).strip()
        cleaned = inner

    if wrap_tikzpicture:
        tikz = _TIKZ_RE.search(cleaned)
        if tikz:
            return (
                "\\documentclass[tikz]{standalone}\n"
                "\\usepackage{tikz}\n"
                "\\begin{document}\n"
                f"{tikz.group(1).strip()}\n"
                "\\end{document}"
            )
    return None
