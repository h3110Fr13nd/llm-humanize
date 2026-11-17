"""Text-normalisation helpers that sit *before* generation.

`verbalise_numerals` exists because of Experiment 04's diffusion collapse. The
0.6 B masked-diffusion checkpoint (`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`)
falls into a comma+digit attractor whenever the conditioning text contains
numerals, and a single digit list is enough to overwrite a whole canvas:

    source: "... a 2.4-meter primary mirror ... 540 kilometers ... April 24, 1990"
    output: ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"

Measured causally in `experiments/04_text_diffusion_global_rewrite/outputs/
numeral_probe.json`:

    doc 01, raw numerals   -> PPL 2770.3, 1 word, 63 commas
    doc 01, verbalised     -> PPL   26.8, 22 words, 0 commas
    doc 03, few numerals   -> PPL    4.9, 15 words, 0 commas
    doc 03, numerals added -> sudden "is,, in,,,,,, and beyond. Values: 1,2,3,..."

Spelling the numbers out removes the attractor. In a real pipeline you would
restore the original digits afterwards by alignment, since the rewrite is
supposed to keep its factual anchors.
"""

from __future__ import annotations

import re

_SMALL = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
         70: "seventy", 80: "eighty", 90: "ninety"}


def _under_thousand(n: int) -> str:
    if n < 20:
        return _SMALL[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens * 10] + (f"-{_SMALL[rest]}" if rest else "")
    hundreds, rest = divmod(n, 100)
    out = f"{_SMALL[hundreds]} hundred"
    return out if not rest else f"{out} {_under_thousand(rest)}"


def number_to_words(n: int) -> str:
    """0..999_999_999 to English words (enough for the corpora in this repo)."""
    if n < 1000:
        return _under_thousand(n)
    for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        if n >= size:
            head, rest = divmod(n, size)
            out = f"{number_to_words(head)} {name}"
            if rest:
                out += f" {number_to_words(rest)}" if rest >= 100 else f" {_under_thousand(rest)}"
            return out
    raise ValueError(n)


def verbalise_numerals(text: str, *, keep_digits: bool = False) -> str:
    """Spell out numerals so a masked-diffusion canvas keeps its footing.

    ``1,234.5``   -> ``one thousand two hundred thirty-four point five``
    ``1990``      -> ``one thousand nine hundred ninety``
    ``2.4-meter`` -> ``two point four meter``

    ``keep_digits=True`` leaves any digit the word table cannot render (it can
    render everything, so this only matters for exotic input) in place; the
    default drops stray digits entirely, which is the strongest form of the
    anti-attractor fix.
    """

    def repl(m: re.Match[str]) -> str:
        raw = m.group(0).replace(",", "")
        if "." in raw:
            whole, _, frac = raw.partition(".")
            left = number_to_words(int(whole)) if whole else "zero"
            right = " ".join(_SMALL[int(c)] if int(c) < 20 else c for c in frac)
            return f"{left} point {right}"
        return number_to_words(int(raw))

    out = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", repl, text)
    return out if keep_digits else re.sub(r"\d", "", out)
