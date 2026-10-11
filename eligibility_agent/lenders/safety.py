"""Safety check, confidence and headline: the code fence around the single LLM node (PRD v1.1 9.3 / 9.4).

The gate (``lenders/gate.py``) decides who the model may rank; this module demotes whatever the model gets wrong,
computes confidence that can only ever go DOWN, and derives the headline. All of it is pure: arguments in, value out,
no repo, no clock, no LLM.

* ``check_pick`` / ``run_safety`` (9.3): rules in order, first breach wins; survivors keep model order, so a demoted top
  promotes the first surviving alternate. ``review`` is True only when nothing survived or the model's own
  recommendation was demoted (alternate-only demotions are logged, not reviewed).
* Rule 4 (``min_loan_amount``, K-ruling 1 / Task-6 hand-off): compares ``min(loan_max_upper_raw, loan_max)`` with
  ``loan_min``. It NEVER reads the half-up display ``ceiling`` (rounding can lift a below-minimum raw over the floor:
  income 20,000 / EMI 8,160 on a 25,000-minimum lender is raw 24,577 but displays 25,000), and ``not_fundable_alone``
  (no EMI headroom) is itself a rule-4 breach.
* ``compute_code_confidence`` (9.4, K16) before the model runs; ``final_confidence`` (K15) applies the post-safety floor
  cap and ``min()``s with the model's own confidence. A model that RAISES confidence is ignored.
* ``compute_headline`` (K13) is the merged headline table, code only; the model never influences it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from eligibility_agent.derive.range import FOIR_EPS, band_from_score
from eligibility_agent.graph.state import (
    BAND_ORDER,
    CONFIDENCE_ORDER,
    Band,
    Code,
    Confidence,
    Derived,
    GatedLender,
    Headline,
    Identity,
    Kind,
    LcStatus,
    LenderCriteria,
    LenderNumbers,
    ModelOutput,
    Pick,
    SafetyBreach,
    SafetyResult,
)
from eligibility_agent.lenders.gate import uncertain_required_fields

__all__ = [
    "NO_HEADROOM_DETAIL",
    "check_pick",
    "compute_code_confidence",
    "compute_headline",
    "final_confidence",
    "run_safety",
]

# The rule-4 detail for a lender with no EMI headroom (``LenderNumbers.not_fundable_alone``). The headline table keys off
# it to tell "cannot afford this lender at all" from "ceiling below its minimum loan", so it is a shared constant.
NO_HEADROOM_DETAIL: Final[str] = "no EMI headroom"


# --------------------------------------------------------------------------- 9.3 safety


def _floor_clearance(band: Band, criteria: LenderCriteria) -> int | None:
    """Buckets by which the band clears the lender's score floor (``BAND_ORDER[band] - BAND_ORDER[floor bucket]``).

    None when the floor is unknown, or on the NTC path (no score to clear a floor with).
    """
    if band == "NTC" or criteria.score_floor is None:
        return None
    return BAND_ORDER[band] - BAND_ORDER[band_from_score(criteria.score_floor)]


def check_pick(
    pick: Pick,
    lender: GatedLender | None,
    nums: LenderNumbers | None,
    *,
    band: Band,
    age_known: bool,
    age_app_years: int | None,
) -> SafetyBreach | None:
    """Check ONE model pick against the gate, the criteria and the computed numbers; the first breach wins.

    Rules, in order (``detail`` is internal only and may contain numbers):

    0. ``not_in_gate``: the pick has no ``GatedLender`` (the model invented it or it is outside the gate), or there are no
       computed numbers to verify it against (fail closed).
    1. ``band_floor``: NTC needs ``ntc_accepted is True``; a scored band must sit in a bucket at or above the bucket of
       ``score_floor`` (``band_from_score``). An unknown floor on a scored band never breaches.
    2. ``foir_limit``: ``displayed_foir > foir_limit + FOIR_EPS``. Skipped when ``displayed_foir`` is None (no income).
    3. ``age_bounds`` (ONLY when dob is known): applicant age below ``min_age_years`` or age at maturity above
       ``max_age_at_maturity_years``; each limit is checked only when set.
    4. ``min_loan_amount``: ``not_fundable_alone`` is a breach (detail ``NO_HEADROOM_DETAIL``); otherwise
       ``min(loan_max_upper_raw, loan_max) < loan_min`` is, on the UNROUNDED raw, never the display ``ceiling``.
    """
    if lender is None or nums is None:
        why = "pick is not in the gated lender list" if lender is None else "no computed numbers for the lender"
        return SafetyBreach(lender_id=pick.lender_id, rule="not_in_gate", detail=why)

    lender_id = pick.lender_id
    c = lender.criteria

    if band == "NTC":
        if c.ntc_accepted is not True:
            return SafetyBreach(lender_id=lender_id, rule="band_floor", detail="NTC applicant and lender does not accept NTC")
    elif c.score_floor is not None:
        floor_band = band_from_score(c.score_floor)
        if BAND_ORDER[band] < BAND_ORDER[floor_band]:
            return SafetyBreach(
                lender_id=lender_id, rule="band_floor", detail=f"band {band} is below the lender floor bucket {floor_band}"
            )

    if nums.displayed_foir is not None and c.foir_limit is not None and nums.displayed_foir > c.foir_limit + FOIR_EPS:
        return SafetyBreach(
            lender_id=lender_id,
            rule="foir_limit",
            detail=f"displayed FOIR {nums.displayed_foir:.6f} exceeds the limit {c.foir_limit:.6f}",
        )

    if age_known and age_app_years is not None:
        if c.min_age_years is not None and age_app_years < c.min_age_years:
            return SafetyBreach(
                lender_id=lender_id, rule="age_bounds", detail=f"age {age_app_years} is below the minimum {c.min_age_years}"
            )
        if (
            c.max_age_at_maturity_years is not None
            and nums.age_at_maturity_years is not None
            and nums.age_at_maturity_years > c.max_age_at_maturity_years
        ):
            return SafetyBreach(
                lender_id=lender_id,
                rule="age_bounds",
                detail=f"age at maturity {nums.age_at_maturity_years} exceeds the maximum {c.max_age_at_maturity_years}",
            )

    if nums.not_fundable_alone:
        return SafetyBreach(lender_id=lender_id, rule="min_loan_amount", detail=NO_HEADROOM_DETAIL)
    if nums.loan_max_upper_raw is not None and c.loan_min is not None:
        usable = nums.loan_max_upper_raw if c.loan_max is None else min(nums.loan_max_upper_raw, c.loan_max)
        if usable < c.loan_min:
            return SafetyBreach(
                lender_id=lender_id,
                rule="min_loan_amount",
                detail=f"affordable ceiling {usable:.2f} is below the lender minimum loan {c.loan_min:.2f}",
            )
    return None


def run_safety(
    model_output: ModelOutput | None,
    gated: Sequence[GatedLender],
    numbers: dict[str, LenderNumbers],
    *,
    band: Band,
    age_app_years: int | None,
    age_known: bool,
) -> SafetyResult:
    """Safety-check the model's picks and rebuild the ranking from the survivors.

    Candidates are ``[recommendation] + alternates`` (``model_output`` None, or no picks at all, means no candidates:
    ``none_survived=True``, ``review=True``). Each candidate is checked in model order; a repeat of a lender that already
    survived is a ``duplicate`` breach. Survivors keep model order: the first is the top (a demoted top therefore
    promotes the first surviving alternate), the next at most two are the alternates. ``review`` is True only when
    nothing survived or the model's own recommendation was demoted; alternate-only demotions are logged in ``demoted``.
    ``top_clears_floor_by_one`` describes the surviving top (None when it has no top, an unknown floor, or is on the NTC
    path).
    """
    candidates: list[Pick] = []
    recommendation_given = model_output is not None and model_output.recommendation is not None
    if model_output is not None:
        if model_output.recommendation is not None:
            candidates.append(model_output.recommendation)
        candidates.extend(model_output.alternates)

    by_id = {g.id: g for g in gated}
    survivors: list[Pick] = []
    surviving_ids: set[str] = set()
    demoted: list[SafetyBreach] = []
    recommendation_demoted = False
    for index, candidate in enumerate(candidates):
        if candidate.lender_id in surviving_ids:
            breach: SafetyBreach | None = SafetyBreach(
                lender_id=candidate.lender_id, rule="duplicate", detail="lender already picked earlier in the model output"
            )
        else:
            breach = check_pick(
                candidate,
                by_id.get(candidate.lender_id),
                numbers.get(candidate.lender_id),
                band=band,
                age_known=age_known,
                age_app_years=age_app_years,
            )
        if breach is None:
            survivors.append(candidate)
            surviving_ids.add(candidate.lender_id)
            continue
        demoted.append(breach)
        if index == 0 and recommendation_given:
            recommendation_demoted = True

    top = survivors[0] if survivors else None
    alternates = survivors[1:3]
    none_survived = top is None
    clears: bool | None = None
    if top is not None:
        top_lender = by_id.get(top.lender_id)
        clearance = _floor_clearance(band, top_lender.criteria) if top_lender is not None else None
        clears = None if clearance is None else clearance >= 1
    return SafetyResult(
        top=top,
        alternates=alternates,
        demoted=demoted,
        none_survived=none_survived,
        review=none_survived or recommendation_demoted,
        top_clears_floor_by_one=clears,
    )


# --------------------------------------------------------------------------- 9.4 confidence


def compute_code_confidence(
    *,
    kind: Kind,
    identity: Identity,
    derived: Derived,
    gated: Sequence[GatedLender],
    numbers: dict[str, LenderNumbers],
    openai_failed: bool,
) -> Confidence:
    """Code confidence, computed BEFORE the model runs (the decision table; re-capped after safety in ``final_confidence``).

    | row | condition                                                                       | result |
    | L1  | no lender survives the gate (``gated == []``)                                    | low    |
    | L2  | ``openai_failed`` (the model was invalid twice)                                  | low    |
    | L3  | safety demoted every pick: applied by ``final_confidence``, not here             | low    |
    | H   | ALL of: ``identity.pan_valid``; ``kind == 'full'``; EMI fully reported            | high   |
    |     | (``estimated_total == 0``, no ``stale_excluded`` item, no past-tenure item);      |        |
    |     | ``emi.used_source == 'derived'``; dob known (age known on ``derived`` too);       |        |
    |     | no ``'uncertain'`` required criteria field on any gated lender; and at least one  |        |
    |     | gated lender whose score floor the band clears by >= 1 bucket                     |        |
    | M   | anything else (quick checks ALWAYS land here or lower)                            | medium |

    ``numbers`` is accepted for signature parity with the dossier; no table row reads it.
    """
    if not gated:
        return "low"
    if openai_failed:
        return "low"
    if kind != "full" or not identity.pan_valid:
        return "medium"

    emi = derived.emi
    fully_reported = (
        emi.estimated_total == 0.0
        and not any(item.kind == "stale_excluded" or item.past_tenure for item in emi.items)
        and emi.used_source == "derived"
    )
    dob_known = identity.dob is not None and derived.age_at_application_years is not None
    any_uncertain = any(uncertain_required_fields(g.criteria) for g in gated)
    clears_a_floor = any((c := _floor_clearance(derived.band, g.criteria)) is not None and c >= 1 for g in gated)
    if fully_reported and dob_known and not any_uncertain and clears_a_floor:
        return "high"
    return "medium"


def final_confidence(
    code_conf: Confidence | None,
    model_output: ModelOutput | None,
    *,
    safety: SafetyResult | None,
    code: Code | None,
) -> Confidence | None:
    """Final confidence = code confidence, lowered by the post-safety facts and by the model; never raised.

    ``code_conf`` None (the band-read / ``pan_not_verified`` paths never compute one) gives None. Otherwise ``'low'`` when
    ``code == 'OPENAI_FAILURE'``, when safety left no survivor, or when ``code_conf`` is already low. Else the base is
    ``code_conf``, capped to ``'medium'`` when the surviving top clears its floor by exactly 0 buckets
    (``safety.top_clears_floor_by_one is False``), then ``min()``-ed with the model's own confidence by
    ``CONFIDENCE_ORDER``: a model that raises its confidence changes nothing.
    """
    if code_conf is None:
        return None
    if code == "OPENAI_FAILURE" or code_conf == "low" or (safety is not None and safety.none_survived):
        return "low"
    base: Confidence = code_conf
    if safety is not None and safety.top_clears_floor_by_one is False and base == "high":
        base = "medium"
    if model_output is not None and CONFIDENCE_ORDER[model_output.confidence] < CONFIDENCE_ORDER[base]:
        base = model_output.confidence
    return base


# --------------------------------------------------------------------------- headline


def _band_derived_headline(band: Band | None) -> Headline:
    """Headline from the band alone (no ranking was possible): NTC / '<650' -> unlikely, any other band -> possible.

    An unknown band cannot occur on a status-ok path; it is answered ``under_review`` rather than guessed.
    """
    if band is None:
        return "under_review"
    return "unlikely" if band in ("NTC", "<650") else "possible"


def _every_demotion_is_floor_or_headroom(safety: SafetyResult | None) -> bool:
    """True when safety demoted at least one pick and EVERY demotion was ``band_floor`` or the no-headroom breach."""
    if safety is None or not safety.demoted:
        return False
    return all(
        d.rule == "band_floor" or (d.rule == "min_loan_amount" and d.detail == NO_HEADROOM_DETAIL) for d in safety.demoted
    )


def _band_fails_every_gated_floor(band: Band | None, gated: Sequence[GatedLender] | None) -> bool:
    """PRD 9.4 "no lender surviving the gate OR THE BAND": the gate is non-empty and the band fails EVERY gated lender's
    floor, each floor KNOWN. A floor is known when its value is present and the admin did not mark it ``'uncertain'``:
    a scored band needs ``score_floor`` below the band's bucket; NTC needs ``ntc_accepted`` explicitly False. An unknown
    (None) or uncertain floor on ANY gated lender, an empty gate or an unknown band is False: the claim "nothing fits"
    is only made on facts the criteria support."""
    if band is None or not gated:
        return False
    for lender in gated:
        c = lender.criteria
        if band == "NTC":
            known, failed = c.ntc_accepted is not None, c.ntc_accepted is False
            status = c.field_status.get("ntc_accepted")
        else:
            floor = c.score_floor
            known = floor is not None
            failed = floor is not None and BAND_ORDER[band] < BAND_ORDER[band_from_score(floor)]
            status = c.field_status.get("score_floor")
        if not known or status == "uncertain" or not failed:
            return False
    return True


def compute_headline(
    *,
    status: LcStatus,
    band: Band | None,
    final_conf: Confidence | None,
    safety: SafetyResult | None,
    gated: Sequence[GatedLender] | None,
    code: Code | None,
    model_output: ModelOutput | None,
) -> Headline | None:
    """The headline, in code (K13 merged table; first match wins; the model never influences it).

    ``status`` is the true LcView wire status: an OPENAI_FAILURE run is ``status='service_failure'`` (K7), and callers pass
    it as-is (never a synthetic ``'ok'``).

    | #   | condition                                                                        | headline             |
    | a   | ``status != 'ok'``, EXCEPT ``status == 'service_failure'`` with ``code ==         | None                 |
    |     | 'OPENAI_FAILURE'`` (the single exemption, which falls through to row e)           |                      |
    | b   | ``code == 'NO_PINCODE_ON_PAN'`` (band read): band NTC / '<650'; any other band    | unlikely; possible   |
    | e   | ``code == 'OPENAI_FAILURE'``: the same band-derived mapping (K7, never review)    | unlikely; possible   |
    | c   | final low AND (gate empty, OR nothing survived AND every demotion was band_floor  | not_fundable_alone   |
    |     | / no-headroom, OR nothing survived AND the gate is non-empty AND the band fails   |                      |
    |     | every gated lender's KNOWN floor: PRD 9.4 "no lender surviving the gate or the    |                      |
    |     | band is one path", whether the model declined them or safety demoted them)        |                      |
    | d   | final low, nothing survived, any other demotion mix / nothing returned / no       | under_review         |
    |     | safety result                                                                     |                      |
    | f   | band NTC and a pick survived (an NTC-accepting lender)                            | unlikely             |
    | g   | final high and a top survived                                                    | fundable             |
    | h   | final medium and a top survived                                                  | possible             |
    | i   | anything else, incl. final low with a surviving top at a scored band (defensive)  | under_review         |

    Rows b and e map identically and come BEFORE the generic low rows (an OPENAI_FAILURE run is final-low but must show
    the band-derived headline). The demotion-driven rows (c's demotion clause, d) need ``safety.top is None``: a surviving
    top whose alternates were demoted falls through to rows f..i. Forcing the co-applicant flag on ``not_fundable_alone`` /
    NTC ``unlikely`` is the view builder's job. ``model_output`` is accepted for signature parity with the dossier; no
    row reads it.
    """
    if status != "ok" and not (status == "service_failure" and code == "OPENAI_FAILURE"):
        return None
    if code == "NO_PINCODE_ON_PAN" or code == "OPENAI_FAILURE":
        return _band_derived_headline(band)
    survivor = safety is not None and safety.top is not None
    if final_conf == "low":
        if not gated:
            return "not_fundable_alone"
        if not survivor:
            if _every_demotion_is_floor_or_headroom(safety) or _band_fails_every_gated_floor(band, gated):
                return "not_fundable_alone"
            return "under_review"
    if band == "NTC" and survivor:
        return "unlikely"
    if survivor and final_conf == "high":
        return "fundable"
    if survivor and final_conf == "medium":
        return "possible"
    return "under_review"
