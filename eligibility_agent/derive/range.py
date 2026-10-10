"""Band bucketing, NTC predicate, display rounding, annuity range maths, age and candidate pincodes (PRD v1.1 8.1/8.4-8.6).

One module, one home for every derived number the LC sees. Pure and total: nothing here does I/O, reads a clock, or
raises on any criteria / income / account value the models can hold.

* 8.1 band: ``is_ntc_value`` / ``band_from_score`` (Task-5 seed, unchanged). NTC (v1.1, FIXED) is ONLY an HTTP 200 +
  ``success: true`` response whose ``credit_score`` is not a three-digit string; ``success: false`` is a service failure
  (the client raises), never NTC.
* K6 rounding: ``round_to_nearest_1000`` (HALF-UP) is the ONLY rounding helper in the repo. ``lender_numbers`` applies
  it exactly once and hands the display values to ``LenderNumbers``; ``output/lc_view.py`` consumes them without
  re-rounding. ``*_raw`` fields carry the UNCLIPPED, unrounded annuity values.
* 8.4 range / ceiling / displayed FOIR (K22): annuity maths with ``r = annual_pct / 100 / 12`` (the ``r == 0`` case is
  its own branch). The displayed FOIR is the inverse-identity reading: ``(existing_emi + emi(P_wc, r_max, n_max)) /
  income`` with ``P_wc = min(loan_max(A, r_max, n_max), loan_max)``, which equals the lender's FOIR limit exactly when
  the lender's loan cap does not clip and falls below it when it does (so safety's ``<= limit + FOIR_EPS`` holds by
  construction).
* 8.5 age (integer-month maths, no float fuzz) and 8.6 candidate pincodes.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Final, NamedTuple

from eligibility_agent.graph.state import PINCODE_RE_STR, Band, Identity, Kind, LenderCriteria, LenderNumbers, ReportFacts

THREE_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]{3}")
PIN_RE: Final[re.Pattern[str]] = re.compile(PINCODE_RE_STR)

FOIR_EPS: Final[float] = 1e-9  # tolerance for FOIR identities and for "zero headroom" (shared with lenders/safety.py)
MAX_CRITERIA_TENURE_MONTHS: Final[int] = 1200  # lender tenure criteria outside 1..1200 months are unusable, not maths
MAX_CRITERIA_LOAN_AMOUNT: Final[float] = 1e15  # loan bounds above this (rupees) are admin junk; keeps display ints finite


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


# --------------------------------------------------------------------------- K6: display rounding (the ONE home)


def round_to_nearest_1000(v: float) -> int:
    """HALF-UP to the nearest 1,000 (``731500 -> 732000``, ``222749.104 -> 223000``, ``499.9 -> 0``).

    The only rounding helper in the repo: ``lender_numbers`` calls it once; nothing downstream re-rounds. ``floor`` of
    ``v / 1000 + 0.5`` is ``int(v / 1000 + 0.5)`` for ``v >= 0`` and stays half-up for negatives. ``v`` must be finite
    (``lender_numbers`` only ever passes clipped, finite bounds).
    """
    return math.floor(v / 1000.0 + 0.5) * 1000


# --------------------------------------------------------------------------- 8.5 age


def age_years_and_months(dob: date, on: date) -> tuple[int, int]:
    """``(whole years, whole months)`` between ``dob`` and ``on``; a month is complete only once the day of month is
    reached (``dob 1998-04-12`` on ``2026-10-10`` -> 341 months -> ``(28, 341)``). Clamped at 0 for a future dob."""
    months = (on.year - dob.year) * 12 + (on.month - dob.month) - (1 if on.day < dob.day else 0)
    months = max(0, months)
    return months // 12, months


def age_at_maturity_years(age_months_at_application: int, tenure_max_months: int) -> int:
    """Whole years at loan maturity: ``(age_months + tenure_max_months) // 12``. Integer months, no float fuzz."""
    return (age_months_at_application + tenure_max_months) // 12


# --------------------------------------------------------------------------- 8.6 candidate pincodes


def candidate_pincodes(identity: Identity, report: ReportFacts | None, *, kind: Kind) -> list[str]:
    """Quick: ``[pan_pincode]`` when set. Full: the PAN pincode UNION ``report.bureau_pincodes_12m`` (the adapter already
    restricted those to OPEN accounts reported within 12 months of the report date). Every value is re-checked against
    the registry pattern, then the result is sorted and deduplicated. The LC can never enter or override a pincode."""
    pins: set[str] = set()
    if identity.pan_pincode is not None:
        pins.add(identity.pan_pincode)
    if kind == "full" and report is not None:
        pins.update(report.bureau_pincodes_12m)
    return sorted(p for p in pins if PIN_RE.fullmatch(p) is not None)


# --------------------------------------------------------------------------- 8.4 annuity maths


def available_emi(foir_limit: float, income: float, existing_emi: float) -> float:
    """``max(0, foir_limit * income - existing_emi)``. Headroom within ``FOIR_EPS * income`` of zero (binary-float fuzz
    such as ``0.55 * 10500 - 5775 == 9e-13``) counts as exactly 0.0, so "no headroom" classifies the same at the boundary
    as it does arithmetically. 0.0 means "not fundable alone" with this lender."""
    headroom = foir_limit * income - existing_emi
    return headroom if headroom > FOIR_EPS * max(income, 0.0) else 0.0


def annuity_loan_max(a_emi: float, annual_rate_pct: float, n_months: int) -> float:
    """Principal a monthly EMI of ``a_emi`` repays over ``n_months``: ``A * ((1+r)^n - 1) / (r * (1+r)^n)`` with
    ``r = annual_rate_pct / 100 / 12``.

    Contract: ``annual_rate_pct >= 0`` (``lender_numbers`` screens rates and tenures). Total over that domain: ``r == 0``
    is ``A * n``; when ``(1+r)^n`` overflows a float the perpetuity limit ``A / r`` is returned; when ``r`` is so small
    that ``1 + r == 1.0`` the ``r -> 0`` limit ``A * n`` is returned; ``n_months <= 0`` repays nothing (0.0).
    """
    if n_months <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0.0:
        return a_emi * n_months
    try:
        f = (1.0 + r) ** n_months
    except OverflowError:
        return a_emi / r
    if f == 1.0:
        return a_emi * n_months
    denominator = r * f
    if not math.isfinite(denominator):
        return a_emi / r
    return a_emi * (f - 1.0) / denominator


def annuity_emi(principal: float, annual_rate_pct: float, n_months: int) -> float:
    """Monthly EMI that repays ``principal`` over ``n_months``: ``P * r * (1+r)^n / ((1+r)^n - 1)``; the inverse of
    ``annuity_loan_max``. Same contract and totality: ``r == 0`` is ``P / n``; an overflowing ``(1+r)^n`` gives the
    interest-only limit ``P * r``; ``n_months <= 0`` has no instalment (0.0)."""
    if n_months <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0.0:
        return principal / n_months
    try:
        f = (1.0 + r) ** n_months
    except OverflowError:
        return principal * r
    if f == 1.0:
        return principal / n_months
    numerator = principal * r * f
    if not math.isfinite(numerator) or not math.isfinite(f):
        return principal * r
    return numerator / (f - 1.0)


class _Usable(NamedTuple):
    foir_limit: float
    rate_min_pct: float
    rate_max_pct: float
    tenure_min_months: int
    tenure_max_months: int
    loan_min: float
    loan_max: float


def _usable_criteria(c: LenderCriteria) -> _Usable | None:
    """The seven criteria the money maths needs, or None when any is missing or nonsensical (non-finite, a negative
    rate / loan bound / FOIR, a loan bound above 1e15, a tenure outside 1..1200). The gate (Task 8) guarantees they are known; this keeps
    ``lender_numbers`` total regardless."""
    if (
        c.foir_limit is None
        or c.rate_min_pct is None
        or c.rate_max_pct is None
        or c.tenure_min_months is None
        or c.tenure_max_months is None
        or c.loan_min is None
        or c.loan_max is None
    ):
        return None
    floats = (c.foir_limit, c.rate_min_pct, c.rate_max_pct, c.loan_min, c.loan_max)
    if not all(math.isfinite(x) and x >= 0.0 for x in floats):
        return None
    if c.loan_min > MAX_CRITERIA_LOAN_AMOUNT or c.loan_max > MAX_CRITERIA_LOAN_AMOUNT:
        return None
    if not all(1 <= t <= MAX_CRITERIA_TENURE_MONTHS for t in (c.tenure_min_months, c.tenure_max_months)):
        return None
    return _Usable(
        c.foir_limit, c.rate_min_pct, c.rate_max_pct, c.tenure_min_months, c.tenure_max_months, c.loan_min, c.loan_max
    )


def lender_numbers(
    c: LenderCriteria, *, lender_id: str, income: float | None, existing_emi: float, age_months: int | None
) -> LenderNumbers:
    """Range, ceiling, displayed FOIR and age at maturity for ONE gated lender.

    * No usable income (None / not finite / <= 0), a non-finite ``existing_emi``, or any needed criterion missing or
      nonsensical: every MONEY field is None (headline + lender-list view only). ``age_at_maturity_years`` does not
      depend on money and is still computed when ``age_months`` and ``tenure_max_months`` are known.
    * ``A = available_emi(foir_limit, income, existing_emi)``. ``A == 0.0``: ``not_fundable_alone=True``, ``available_emi``
      0.0, the bounds / ceiling / displayed FOIR None.
    * Otherwise ``upper_raw = loan_max(A, rate_min, tenure_max)`` (lender MIN rate, MAX tenure) and
      ``lower_raw = loan_max(A, rate_max, tenure_min)`` (MAX rate, MIN tenure); both are stored UNCLIPPED in ``*_raw``.
      Clip: ``upper_c = min(upper_raw, loan_max)``; ``lower_c = min(max(lower_raw, loan_min), upper_c)``. A bound is
      never raised above what the applicant can afford, and a ceiling below ``loan_min`` is left for safety rule 4.
      Display = ``round_to_nearest_1000`` of the clipped bound, computed once; ``ceiling == loan_max_upper``.
    * ``clipped``: ``'upper'`` / ``'lower'`` / ``'both'`` for the bounds whose clipped value differs from their raw
      value (the lower bound differs when raised to ``loan_min`` or capped at the upper), else ``'none'``.
    * Displayed FOIR (K22): see the module docstring.
    """
    maturity = (
        age_at_maturity_years(age_months, c.tenure_max_months)
        if age_months is not None and c.tenure_max_months is not None
        else None
    )
    unknown = LenderNumbers(
        lender_id=lender_id,
        available_emi=None,
        loan_max_lower=None,
        loan_max_upper=None,
        loan_max_lower_raw=None,
        loan_max_upper_raw=None,
        ceiling=None,
        displayed_foir=None,
        age_at_maturity_years=maturity,
        not_fundable_alone=False,
        clipped="none",
    )
    p = _usable_criteria(c)
    if p is None or income is None or not math.isfinite(income) or income <= 0.0 or not math.isfinite(existing_emi):
        return unknown

    a = available_emi(p.foir_limit, income, existing_emi)
    if a == 0.0:
        return unknown.model_copy(update={"available_emi": 0.0, "not_fundable_alone": True})

    upper_raw = annuity_loan_max(a, p.rate_min_pct, p.tenure_max_months)
    lower_raw = annuity_loan_max(a, p.rate_max_pct, p.tenure_min_months)
    upper_c = min(upper_raw, p.loan_max)
    lower_c = min(max(lower_raw, p.loan_min), upper_c)
    upper_clipped = upper_c < upper_raw
    lower_clipped = lower_c != lower_raw
    clipped = "both" if upper_clipped and lower_clipped else "upper" if upper_clipped else "lower" if lower_clipped else "none"

    worst_case_principal = min(annuity_loan_max(a, p.rate_max_pct, p.tenure_max_months), p.loan_max)
    displayed_foir = (existing_emi + annuity_emi(worst_case_principal, p.rate_max_pct, p.tenure_max_months)) / income

    upper_display = float(round_to_nearest_1000(upper_c))
    return unknown.model_copy(
        update={
            "available_emi": a,
            "loan_max_lower": float(round_to_nearest_1000(lower_c)),
            "loan_max_upper": upper_display,
            "loan_max_lower_raw": lower_raw,
            "loan_max_upper_raw": upper_raw,
            "ceiling": upper_display,
            "displayed_foir": displayed_foir,
            "clipped": clipped,
        }
    )
