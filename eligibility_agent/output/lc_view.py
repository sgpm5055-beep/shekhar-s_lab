"""The redaction boundary (PRD v1.1 sections 6 / 10 / 12): the ONLY place an ``LcView`` is built.

Everything an LC ever sees leaves through the four builders below. Raw envelopes, storage keys, band strings, bureau
names, scores, enquiry counts and account identifiers have no field to live in (``extra='forbid'`` on the wire models)
and every free string is scanned here.

Two tiers of leak scan, both by WHOLE-LINE drop (an offending line is removed, never partially masked):

* LLM-prose tier ``LLM_LEAK_PATTERNS`` = (``BUREAU_TERM_RE`` incl. the generic ``bureaus?``, ``STANDALONE_SCORE_RE``,
  ``ACCOUNT_ID_RE``): rationale, conditions, items_to_confirm and coapplicant_reason, the only free text a model or a
  derivation can author;
* global tier ``GLOBAL_LEAK_PATTERNS`` = (``BUREAU_NAME_RE`` names only, ``STANDALONE_SCORE_RE``, ``ACCOUNT_ID_RE``):
  every fixed string and every admin-authored leaf (lender name, rate band, tenure range). PRD v1.1 mandates the copy
  "bureau report on <date>" and "No bureau file was found on <date>", so the generic word cannot be banned globally;
  the rule that holds everywhere is "never show a bureau NAME".

Tests and evals import the compiled patterns from here; nobody keeps a copy (the C11 rule).

Contracts honoured here (controller K-rulings):

* K13: exactly four builders plus the fixed strings. The headline comes from ``lenders.safety.compute_headline`` and is
  passed the TRUE wire status (``service_failure`` + ``OPENAI_FAILURE`` on the K7 path, never a synthetic ``'ok'``).
* K1: ``pending_verification`` carries ``code=None``; its message rides in ``label``.
* K6: LenderView numbers are the ``LenderNumbers`` display values verbatim. The half-up display rounding lives in
  ``derive.range`` and is applied once upstream; nothing here rounds.
* K7: the OPENAI_FAILURE view is the documented exception to the low-confidence nulls: confidence low, band-derived
  headline, no top lender, at most two code-built number-free alternates whose floor the band clears.
* K15: the band-read (NO_PINCODE_ON_PAN) and pan_not_verified paths have ``confidence=None``; they branch off BEFORE
  ``final_confidence`` is ever computed.
* K20: ``label`` is the one human line (provenance on ok, the section-10 message on failures, the refusal message on
  refusals); it is selected inside the builders.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

from eligibility_agent.derive.range import band_from_score
from eligibility_agent.graph.state import (
    BAND_ORDER,
    Band,
    CheckState,
    Code,
    GatedLender,
    Headline,
    Kind,
    LcStatus,
    LcView,
    LenderCriteria,
    LenderNumbers,
    LenderView,
    LoanRange,
    ModelOutput,
    Pick,
)
from eligibility_agent.lenders.safety import compute_headline, final_confidence

__all__ = [
    "AADHAAR_WARNING",
    "ACCOUNT_ID_RE",
    "BUREAU_NAME_RE",
    "BUREAU_TERM_RE",
    "GLOBAL_LEAK_PATTERNS",
    "LABEL_FULL_FMT",
    "LABEL_QUICK",
    "LABEL_QUICK_NO_EMI",
    "LLM_LEAK_PATTERNS",
    "MSG_BUREAU_SERVICE_FAILURE",
    "MSG_INTERNAL",
    "MSG_NO_PINCODE",
    "MSG_OPENAI_FAILURE",
    "MSG_PAN_NOT_VERIFIED",
    "MSG_PAN_SERVICE_FAILURE",
    "MSG_PENDING_VERIFICATION",
    "RATIONALE_RANKING_UNAVAILABLE",
    "REFUSAL_MESSAGES",
    "STANDALONE_SCORE_RE",
    "WARN_NTC_RECENT",
    "RedactionContext",
    "build_error_lc_view",
    "build_failure_lc_view",
    "build_lc_view",
    "build_refusal_lc_view",
    "line_leaks",
    "scrub_lines",
]

# --------------------------------------------------------------------------- leak patterns (single home)

# PRD 7.3 pattern plus the optional plural: a strict superset of the FIXED rule. LLM-prose tier only.
BUREAU_TERM_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:cibil|experian|equifax|crif|bureaus?)\b", re.IGNORECASE)

# Names only. The global tier uses this one so the PRD-mandated "bureau report on <date>" copy survives.
BUREAU_NAME_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:cibil|experian|equifax|crif)\b", re.IGNORECASE)

# A standalone integer 300-899. Two fixed-width lookbehinds and two lookaheads: not digit-adjacent (1736, 7360), not a
# grouped or decimal amount (3,500 / 12,34,567 / 0.736 / 736.50 / 736,000); Rs.736 and the rupee-sign form still hit.
STANDALONE_SCORE_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d)")

# "No account list in any LC-facing string" (PRD 12 no_leak). Three alternatives:
#   1. an account-number label followed by its value: "account number 1234", "a/c no: SYNTH0001", "acct # 99"
#      (``no`` needs a word boundary so "account normally" is prose, not a label);
#   2. a masked id: "XXXX1234", "****1234", "xxxx-1234" (the lookbehind, not \b, so a leading '*' can anchor);
#   3. any bare run of 9 or more digits (account-like, never legitimate prose: amounts are grouped and short). The
#      dossier capped this at 18 digits, which let a 19+ digit run escape; there is no upper bound here.
# The bare word "account" is deliberately NOT banned ("An account last updated <date>" is mandated copy).
ACCOUNT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:a/?c|acct|account)\s*(?:no\b\.?|number\b|#)\s*[:\-]?\s*\S+"
    r"|(?<!\w)[x*]{2,}[\s-]?\d{3,}\b"
    r"|\b\d{9,}\b",
    re.IGNORECASE,
)

LLM_LEAK_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (BUREAU_TERM_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE)
GLOBAL_LEAK_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (BUREAU_NAME_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE)


def line_leaks(line: str, patterns: Sequence[re.Pattern[str]] = LLM_LEAK_PATTERNS) -> bool:
    """True when ``line`` matches any pattern. Defaults to the strict LLM-prose tier."""
    return any(p.search(line) is not None for p in patterns)


def scrub_lines(lines: Sequence[str]) -> list[str]:
    """LLM-prose tier, whole-line drop: an offending line is removed, never partially edited."""
    return [line for line in lines if not line_leaks(line)]


# --------------------------------------------------------------------------- fixed strings (every one passes the global tier)

LABEL_QUICK: Final[str] = "Indicative, based on what the learner told us"
LABEL_QUICK_NO_EMI: Final[str] = "Indicative, based on what the learner told us, assuming no other EMIs"  # income given, EMI not
LABEL_FULL_FMT: Final[str] = "bureau report on {report_date}, income as declared"  # report_date = YYYY-MM-DD, PRD-verbatim casing
AADHAAR_WARNING: Final[str] = "PAN not linked to Aadhaar; lenders will ask for this"  # EXACT PRD 6 string
MSG_PAN_SERVICE_FAILURE: Final[str] = "Couldn't verify PAN, retry"  # PRD 10 verbatim
MSG_PAN_NOT_VERIFIED: Final[str] = (
    "PAN not verified. Correct the PAN with the learner and run the check again; no quota was used."
)
MSG_BUREAU_SERVICE_FAILURE: Final[str] = "Credit check couldn't complete, retry"  # PRD 10 verbatim (score and report)
MSG_PENDING_VERIFICATION: Final[str] = (
    "Pending verification: the credit check didn't complete. This learner is locked for 10 minutes "
    "while the result is reconciled; please try again after that."
)
MSG_OPENAI_FAILURE: Final[str] = "Couldn't generate a recommendation, retry"  # PRD 10 verbatim
MSG_INTERNAL: Final[str] = "The check could not be completed automatically. Please raise it for manual review."
MSG_NO_PINCODE: Final[str] = "No pincode on the PAN record; lender matching will be available after the full check"  # PRD 9.1
# K7: the code-built rationale of a gate survivor when the ranking model failed. Number-free by construction.
RATIONALE_RANKING_UNAVAILABLE: Final[str] = "Automatic ranking unavailable; this lender serves the learner's area"
# PRD 5.14 verbatim; ``.format(date=<YYYY-MM-DD>)`` at build time. Passes the global tier (generic 'bureau' allowed there).
WARN_NTC_RECENT: Final[str] = "No bureau file was found on {date}; a full report will most likely return nothing. Proceed?"

# The ten refusal codes: Code minus NO_PINCODE_ON_PAN (an ok band-read), SUREPASS_FAILURE and OPENAI_FAILURE (failures).
REFUSAL_MESSAGES: Final[dict[Code, str]] = {
    "CONSENT_MISSING": "Consent was not recorded. Take the learner's verbal consent, tick the consent box and run the check again.",
    "QUOTA_EXHAUSTED": "You have used all your quick checks for today. The limit resets at midnight; saved results stay available.",
    "PAN_DAILY_CAP": "This learner's checks have reached today's safety limit. The record is locked until tomorrow and has been sent for review.",
    "STAGE_TOO_EARLY": "A full check unlocks once this opportunity reaches the Offer stage. Until then, use the quick check.",
    "REPORT_WINDOW_OPEN": "This learner's full report is still current. Refresh unlocks once it is 30 days old.",
    "IN_FLIGHT": "A check for this learner is already running. Please wait for it to finish.",
    "PAN_LOCKED": "This learner's record is locked for a few minutes while an earlier check is reconciled. Please try again shortly.",
    "WARN_NTC_RECENT": WARN_NTC_RECENT,
    "PAN_CHANGE_NEEDS_MANAGER": "A report already exists for this opportunity, so changing the PAN needs a manager. This has been flagged for review.",
    "LIVE_CALLS_DISABLED": "Eligibility checks are temporarily switched off on the server. Please try again later or contact the platform team.",
}


# --------------------------------------------------------------------------- redaction context


@dataclass(frozen=True, slots=True)
class RedactionContext:
    """Facts the redact node supplies so the builders stay pure functions of their arguments.

    ``quick_checks_left_today`` is only ever shown on quick views; ``checked_on`` is the IST date of the raw pull that
    produced the facts (the original date on cache hits); ``refresh_available_on`` is that pull date + 30 days for the
    governing cache window. A refusal carries ``refresh_available_on`` only when the caller has cache context for it.
    """

    quick_checks_left_today: int | None
    checked_on: date | None
    from_cache: bool
    refresh_available_on: date | None


# --------------------------------------------------------------------------- small helpers


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def _quick_left(kind: Kind, left: int | None) -> int | None:
    return left if kind == "quick" else None


def _display_int(value: float | None) -> int | None:
    """A ``LenderNumbers`` display value as the wire int, verbatim. Non-finite (never produced upstream) becomes None."""
    if value is None or not math.isfinite(value):
        return None
    return int(value)


def _guard_admin(value: str, fallback: str) -> str:
    """Admin-authored strings (lender name, rate band, tenure range) pass the global tier or fall back."""
    return fallback if line_leaks(value, GLOBAL_LEAK_PATTERNS) else value


def _scrub_prose(text: str) -> str:
    """One LLM-authored string: the whole thing, or nothing."""
    return "" if line_leaks(text) else text


def _band_of(state: CheckState) -> Band | None:
    """The applicant's band: ``derived.band``, else the newer of the score / report facts (the derive node's rule)."""
    derived = state.get("derived")
    if derived is not None:
        return derived.band
    score = state.get("score")
    report = state.get("report")
    if report is not None and score is not None:
        return report.band if (report.report_date or date.min) >= score.pulled_on else score.band
    if report is not None:
        return report.band
    if score is not None:
        return score.band
    return None


def _band_clears_floor(band: Band, criteria: LenderCriteria) -> bool:
    """Safety rule 1 (``band_floor``) read as a predicate: NTC needs ``ntc_accepted is True``; a scored band must sit
    in a bucket at or above the bucket of ``score_floor`` (an unknown floor never excludes a scored band)."""
    if band == "NTC":
        return criteria.ntc_accepted is True
    if criteria.score_floor is None:
        return True
    return BAND_ORDER[band] >= BAND_ORDER[band_from_score(criteria.score_floor)]


def _lender_view(
    lender_id: str,
    name: str,
    rate_band: str,
    tenure_range: str,
    nums: LenderNumbers | None,
    *,
    rationale: str,
    conditions: list[str],
) -> LenderView:
    ceiling: int | None = None
    loan_range: LoanRange | None = None
    if nums is not None:
        ceiling = _display_int(nums.ceiling)
        lower = _display_int(nums.loan_max_lower)
        upper = _display_int(nums.loan_max_upper)
        if lower is not None and upper is not None:
            loan_range = LoanRange(lower=lower, upper=upper)
    return LenderView(
        id=lender_id,
        name=_guard_admin(name, lender_id),
        rate_band=_guard_admin(rate_band, ""),
        tenure_range=_guard_admin(tenure_range, ""),
        ceiling=ceiling,
        range=loan_range,
        rationale=rationale,
        conditions=conditions,
    )


def _pick_view(
    pick: Pick, gated_by_id: dict[str, GatedLender], numbers: dict[str, LenderNumbers]
) -> LenderView | None:
    """The LenderView of one safety-surviving pick, or None when the lender is not in the gate (fail closed)."""
    lender = gated_by_id.get(pick.lender_id)
    if lender is None:
        return None
    return _lender_view(
        lender.id,
        lender.name,
        lender.rate_band_label,
        lender.tenure_label,
        numbers.get(lender.id),
        rationale=_scrub_prose(pick.rationale),
        conditions=scrub_lines(pick.conditions),
    )


def _aadhaar_warning(state: CheckState) -> str | None:
    """The exact warning string iff ``aadhaar_linked`` is literally False; True, None and no identity give None."""
    identity = state.get("identity")
    return AADHAAR_WARNING if identity is not None and identity.aadhaar_linked is False else None


def _items_to_confirm(state: CheckState) -> list[str]:
    """Derived confirm prompts (stale / past-tenure accounts), scrubbed on the LLM-prose tier."""
    derived = state.get("derived")
    if derived is None:
        return []
    return scrub_lines([item.confirm_item for item in derived.emi.items if item.confirm_item is not None])


def _provenance_label(state: CheckState, ctx: RedactionContext, kind: Kind) -> str:
    """Section 3 / 6 provenance label on an ok view (K20)."""
    if kind == "full":
        report = state.get("report")
        report_date = report.report_date if report is not None else None
        when = _iso(report_date or ctx.checked_on)
        if when is None:
            raise ValueError("a full ok view needs a report date (neither the report facts nor the context carry one)")
        return LABEL_FULL_FMT.format(report_date=when)
    if state.get("monthly_income") is not None and state.get("existing_emi_declared") is None:
        return LABEL_QUICK_NO_EMI
    return LABEL_QUICK


def _coapplicant(
    headline: Headline | None, model: ModelOutput | None
) -> tuple[bool, str | None]:
    """``(suggested, reason)``. 'unlikely' and 'not_fundable_alone' force the suggestion (PRD 10: 'Unlikely alone,
    co-applicant suggested'); otherwise the model's own suggestion passes through. The reason is the model's prose,
    scrubbed whole, and only shown when a co-applicant is actually suggested."""
    forced = headline in ("unlikely", "not_fundable_alone")
    suggested = forced or (model is not None and model.co_applicant_suggested)
    reason: str | None = None
    if suggested and model is not None and model.co_applicant_reason is not None:
        reason = _scrub_prose(model.co_applicant_reason) or None
    return suggested, reason


# --------------------------------------------------------------------------- builder 1: state -> view


def build_lc_view(state: CheckState, ctx: RedactionContext) -> LcView:
    """Build the LcView for a finished (or band-read / failed-rank / pan-unverified) check.

    Handles exactly four state shapes; every other (status, code) pair is a programming error upstream (refusals,
    pending_verification and SUREPASS failures have their own builders) and is answered with the internal-error view
    rather than guessed at:

    * ``ok`` + no code: the recommendation view;
    * ``ok`` + ``NO_PINCODE_ON_PAN``: the band read (K15: confidence None);
    * ``service_failure`` + ``OPENAI_FAILURE`` (state status may be unset or ``'ok'``): the K7 view;
    * ``pan_not_verified`` (also when ``identity.pan_valid`` is False): K15, confidence None.

    Raises ``ValueError`` when the state has no ``check_id`` / ``kind``, or a full ok view has no date for its label.
    """
    check_id = state.get("check_id")
    kind = state.get("kind")
    if check_id is None:
        raise ValueError("state has no check_id")
    if kind is None:
        raise ValueError("state has no kind")

    status: LcStatus = state.get("status") or "ok"
    code = state.get("code")
    identity = state.get("identity")
    if status == "ok" and identity is not None and not identity.pan_valid:
        status, code = "pan_not_verified", None  # never emit an 'ok' view for an unverified PAN
    elif status == "ok" and code == "OPENAI_FAILURE":
        status = "service_failure"  # the code implies the wire status (K7)

    if status == "pan_not_verified" and code is None:
        return _pan_not_verified_view(check_id, kind, ctx)
    if status == "ok" and code == "NO_PINCODE_ON_PAN":
        return _band_read_view(state, ctx, check_id, kind)
    if status == "service_failure" and code == "OPENAI_FAILURE":
        return _openai_failure_view(state, ctx, check_id, kind)
    if status == "ok" and code is None:
        return _recommendation_view(state, ctx, check_id, kind)
    return build_error_lc_view(check_id=check_id, kind=kind)


def _pan_not_verified_view(check_id: str, kind: Kind, ctx: RedactionContext) -> LcView:
    """PAN 200 valid-negative: nothing cached, nothing marked checked, no quota used (so the counter is unchanged)."""
    return LcView(
        check_id=check_id,
        kind=kind,
        status="pan_not_verified",
        code=None,
        headline=None,
        label=MSG_PAN_NOT_VERIFIED,
        confidence=None,
        quick_checks_left_today=_quick_left(kind, ctx.quick_checks_left_today),
    )


def _band_read_view(state: CheckState, ctx: RedactionContext, check_id: str, kind: Kind) -> LcView:
    """NO_PINCODE_ON_PAN: the band read only. Rank and safety never ran, so there is no confidence and no lender."""
    headline = compute_headline(
        status="ok",
        band=_band_of(state),
        final_conf=None,
        safety=None,
        gated=None,
        code="NO_PINCODE_ON_PAN",
        model_output=None,
    )
    suggested, _ = _coapplicant(headline, None)
    return LcView(
        check_id=check_id,
        kind=kind,
        status="ok",
        code="NO_PINCODE_ON_PAN",
        headline=headline,
        label=MSG_NO_PINCODE,
        confidence=None,
        coapplicant_suggested=suggested,
        pan_aadhaar_link_warning=_aadhaar_warning(state),
        checked_on=_iso(ctx.checked_on),
        from_cache=ctx.from_cache,
        refresh_available_on=_iso(ctx.refresh_available_on),
        quick_checks_left_today=_quick_left(kind, ctx.quick_checks_left_today),
    )


def _openai_failure_view(state: CheckState, ctx: RedactionContext, check_id: str, kind: Kind) -> LcView:
    """K7. The ranking model failed twice: the band read plus up to two code-built gate survivors, with no numbers.

    The documented exception to 'low confidence means no lenders'. Model output, if the state holds any, is ignored:
    none of its prose is trusted or shown.
    """
    band = _band_of(state)
    headline = compute_headline(
        status="service_failure",  # the TRUE wire status, never a synthetic 'ok'
        band=band,
        final_conf="low",
        safety=state.get("safety"),
        gated=state.get("gated_lenders"),
        code="OPENAI_FAILURE",
        model_output=None,
    )
    survivors: list[GatedLender] = []
    if band is not None:
        cleared = [g for g in state.get("gated_lenders") or [] if _band_clears_floor(band, g.criteria)]
        survivors = sorted(cleared, key=lambda g: g.id)[:2]
    alternates = [
        _lender_view(
            g.id, g.name, g.rate_band_label, g.tenure_label, None, rationale=RATIONALE_RANKING_UNAVAILABLE, conditions=[]
        )
        for g in survivors
    ]
    suggested, _ = _coapplicant(headline, None)
    return LcView(
        check_id=check_id,
        kind=kind,
        status="service_failure",
        code="OPENAI_FAILURE",
        headline=headline,
        label=MSG_OPENAI_FAILURE,
        confidence="low",
        top_lender=None,
        alternates=alternates,
        coapplicant_suggested=suggested,
        pan_aadhaar_link_warning=_aadhaar_warning(state),
        checked_on=_iso(ctx.checked_on),
        from_cache=ctx.from_cache,
        refresh_available_on=_iso(ctx.refresh_available_on),
        quick_checks_left_today=_quick_left(kind, ctx.quick_checks_left_today),
    )


def _recommendation_view(state: CheckState, ctx: RedactionContext, check_id: str, kind: Kind) -> LcView:
    """The normal ok view. Picks come from the safety survivors, never straight from the model; they are shown only at
    medium or high final confidence (PRD 6: low confidence is headline-only)."""
    model = state.get("model_output")
    safety = state.get("safety")
    final_conf = final_confidence(state.get("code_confidence"), model, safety=safety, code=None)
    headline = compute_headline(
        status="ok",
        band=_band_of(state),
        final_conf=final_conf,
        safety=safety,
        gated=state.get("gated_lenders"),
        code=None,
        model_output=model,
    )
    top: LenderView | None = None
    alternates: list[LenderView] = []
    if safety is not None and final_conf in ("high", "medium"):
        gated_by_id = {g.id: g for g in state.get("gated_lenders") or []}
        numbers = state.get("lender_numbers") or {}
        if safety.top is not None:
            top = _pick_view(safety.top, gated_by_id, numbers)
        alternates = [v for a in safety.alternates if (v := _pick_view(a, gated_by_id, numbers)) is not None][:2]
    suggested, reason = _coapplicant(headline, model)
    return LcView(
        check_id=check_id,
        kind=kind,
        status="ok",
        code=None,
        headline=headline,
        label=_provenance_label(state, ctx, kind),
        confidence=final_conf,
        top_lender=top,
        alternates=alternates,
        coapplicant_suggested=suggested,
        coapplicant_reason=reason,
        pan_aadhaar_link_warning=_aadhaar_warning(state),
        items_to_confirm=_items_to_confirm(state),
        checked_on=_iso(ctx.checked_on),
        from_cache=ctx.from_cache,
        refresh_available_on=_iso(ctx.refresh_available_on),
        quick_checks_left_today=_quick_left(kind, ctx.quick_checks_left_today),
    )


# --------------------------------------------------------------------------- builder 2: refusals


def build_refusal_lc_view(
    *, check_id: str, kind: Kind, code: Code, ctx: RedactionContext, warn_date: date | None = None
) -> LcView:
    """An admission refusal: status ``refused``, the refusal message in ``label``, no headline, no lenders.

    ``warn_date`` is the NTC-found date for WARN_NTC_RECENT (the context's pull date stands in when it is absent).
    Cache context (``checked_on`` / ``refresh_available_on``) passes through only when the context carries it, and a
    refusal is never ``from_cache``: nothing was served. Raises ``ValueError`` for a code that is not one of the ten
    refusal codes, or for WARN_NTC_RECENT with no date at all.
    """
    message = REFUSAL_MESSAGES.get(code)
    if message is None:
        raise ValueError(f"{code} is not a refusal code")
    if code == "WARN_NTC_RECENT":
        when = _iso(warn_date or ctx.checked_on)
        if when is None:
            raise ValueError("WARN_NTC_RECENT needs the NTC-found date (warn_date or the context pull date)")
        message = message.format(date=when)
    return LcView(
        check_id=check_id,
        kind=kind,
        status="refused",
        code=code,
        headline=None,
        label=message,
        confidence=None,
        checked_on=_iso(ctx.checked_on),
        from_cache=False,
        refresh_available_on=_iso(ctx.refresh_available_on),
        quick_checks_left_today=_quick_left(kind, ctx.quick_checks_left_today),
    )


# --------------------------------------------------------------------------- builder 3: service failures


def build_failure_lc_view(
    *,
    check_id: str,
    kind: Kind,
    status: Literal["pending_verification", "service_failure"],
    code: Code | None,
    label: str,
    quick_checks_left_today: int | None,
) -> LcView:
    """A section-10 failure (PAN / score / report service failure, or a pending-verification timeout).

    ``label`` is the caller's fixed message; it must pass the global tier or the internal-error message is shown in
    its place. ``pending_verification`` always carries ``code=None`` (K1), whatever the caller passes. No headline:
    no band is known on these paths (the OPENAI_FAILURE band read is ``build_lc_view``'s K7 view).
    """
    return LcView(
        check_id=check_id,
        kind=kind,
        status=status,
        code=None if status == "pending_verification" else code,
        headline=None,
        label=_guard_admin(label, MSG_INTERNAL),
        confidence=None,
        quick_checks_left_today=_quick_left(kind, quick_checks_left_today),
    )


# --------------------------------------------------------------------------- builder 4: internal error


def build_error_lc_view(*, check_id: str, kind: Kind) -> LcView:
    """An unexpected crash or route timeout: ``service_failure``, no code, the manual-review message."""
    return LcView(
        check_id=check_id,
        kind=kind,
        status="service_failure",
        code=None,
        headline=None,
        label=MSG_INTERNAL,
        confidence=None,
    )
