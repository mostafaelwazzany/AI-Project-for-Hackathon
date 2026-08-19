"""Text utilities shared across the pipeline."""

from __future__ import annotations

import re

import arabic_reshaper
from bidi.algorithm import get_display


def is_arabic(text: str) -> bool:
    # Regex101: [\u0600-\u06FF]
    return bool(re.search(r"[\u0600-\u06FF]", text))


def terminal_display(text: str) -> str:
    """Make Arabic readable in VS Code's LTR-only integrated terminal.

    VS Code's terminal renderer does not consistently apply Arabic shaping or
    bidirectional layout.  This affects display only: generate() still returns
    normal Unicode Arabic for a future web interface or file output.
    """
    if not is_arabic(text):
        return text
    return "\n".join(
        get_display(arabic_reshaper.reshape(line)) if line else ""
        for line in text.splitlines()
    )
