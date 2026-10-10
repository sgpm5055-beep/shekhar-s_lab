"""Band bucketing, the NTC predicate and the pincode pattern (PRD v1.1 section 8.1): SEED of ``derive/range.py``.

Task 5 ships ONLY what ``surepass/adapters.py`` needs at fact-construction time (``ScoreFacts`` / ``ReportFacts`` require
a band), so the adapters can import it without a circular dependency. Task 6 EXTENDS this same file with the rest of
the PRD layout: the annuity range / ceiling / displayed-FOIR math, half-up display rounding (K6), age at application
and maturity, and candidate pincodes. Do not create a second range module; add to this one and keep the two names
below (and their semantics) stable.

NTC (v1.1, FIXED): a score is "no file" ONLY when the response was HTTP 200 + ``success: true`` and its
``credit_score`` is not a three-digit string. ``success: false`` is a service failure (the client raises), never NTC.
"""

from __future__ import annotations

import re
from typing import Final

from eligibility_agent.graph.state import PINCODE_RE_STR, Band

THREE_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]{3}")
PIN_RE: Final[re.Pattern[str]] = re.compile(PINCODE_RE_STR)


def is_ntc_value(credit_score: object) -> bool:
    """True unless ``str(credit_score).strip()`` is exactly three ASCII digits (None / missing key included).

    ``''``, ``'NH'`` and ``'0745'`` are NTC; ``'745'`` is not. Total over any object: it never raises.
    """
    if credit_score is None:
        return True
    return THREE_DIGIT_RE.fullmatch(str(credit_score).strip()) is None


def band_from_score(score: int | None) -> Band:
    """Bucket a score into the internal band. ``None`` -> ``'NTC'``; buckets are left-closed / right-open.

    ``< 650`` -> ``'<650'``; ``[650, 720)`` -> ``'650-720'``; ``[720, 780)`` -> ``'720-780'``; ``>= 780`` -> ``'780+'``.
    A three-digit value outside 300-900 maps to the nearest bucket (``299`` -> ``'<650'``, ``950`` -> ``'780+'``); the
    internal ``score_out_of_range`` error for the review trail is the redaction / errors layer's concern, not ours.
    """
    if score is None:
        return "NTC"
    if score < 650:
        return "<650"
    if score < 720:
        return "650-720"
    if score < 780:
        return "720-780"
    return "780+"
