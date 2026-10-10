"""Task 8 (part 2): lenders/safety.py, the safety check (9.3), code confidence + final confidence (9.4), headline.

Everything here is code that fences the single LLM node: the gate decides who the model may rank, safety demotes
whatever it gets wrong, confidence can only go DOWN, and the headline is pure code. No repo, no clock, no LLM.

BINDING pins exercised below:
  * K-ruling 1 (Task-6 hand-off): rule 4 (min_loan_amount) compares ``min(loan_max_upper_raw, loan_max)`` against
    ``loan_min`` using the UNROUNDED raw, never the half-up display ceiling; ``not_fundable_alone`` is itself a breach.
  * K13: ``compute_headline`` lives here with the MERGED table (OPENAI_FAILURE band-derived row BEFORE the generic low rows).
  * K15: ``final_confidence`` is None-in / None-out.
  * K16: ``compute_code_confidence`` high row needs the full battery; quick checks cap at medium.
"""

from __future__ import annotations

import ast
import copy
import inspect
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pytest

from eligibility_agent.derive.range import FOIR_EPS, lender_numbers
from eligibility_agent.graph.state import (
    Band,
    Code,
    Confidence,
    Derived,
    EmiDerivation,
    EmiItem,
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
from eligibility_agent.lenders import safety as safety_module
from eligibility_agent.lenders.safety import (
    NO_HEADROOM_DETAIL,
    check_pick,
    compute_code_confidence,
    compute_headline,
    final_confidence,
    run_safety,
)

# --------------------------------------------------------------------------- builders


def crit(**overrides: Any) -> LenderCriteria:
    """Seed lender L1: FOIR 0.55, 11-14%, 12-48m, loans 50,000-1,500,000, floor 650 (bucket '650-720')."""
    fields: dict[str, Any] = {
        "score_floor": 650,
        "ntc_accepted": False,
        "foir_limit": 0.55,
        "min_income": 25000.0,
        "min_age_years": 21,
        "max_age_at_maturity_years": 58,
        "rate_min_pct": 11.0,
        "rate_max_pct": 14.0,
        "tenure_min_months": 12,
        "tenure_max_months": 48,
        "loan_min": 50000.0,
        "loan_max": 1500000.0,
        "policy_notes": None,
        "field_status": {},
    }
    fields.update(overrides)
    return LenderCriteria(**fields)


def gl(lender_id: str = "L1", **criteria_overrides: Any) -> GatedLender:
    return GatedLender(
        id=lender_id,
        name=f"Lender {lender_id}",
        criteria=crit(**criteria_overrides),
        matched_pincodes=["560001"],
        rate_band_label="11 to 14%",
        tenure_label="12 to 48 months",
    )


def num(lender_id: str = "L1", **overrides: Any) -> LenderNumbers:
    """Healthy numbers for the seed lender: nothing breaches unless an override says so."""
    fields: dict[str, Any] = {
        "lender_id": lender_id,
        "available_emi": 20000.0,
        "loan_max_lower": 300000.0,
        "loan_max_upper": 800000.0,
        "loan_max_lower_raw": 300400.0,
        "loan_max_upper_raw": 800100.0,
        "ceiling": 800000.0,
        "displayed_foir": 0.5,
        "age_at_maturity_years": 33,
        "not_fundable_alone": False,
        "clipped": "none",
    }
    fields.update(overrides)
    return LenderNumbers(**fields)


def pick(lender_id: str, rank: int = 1) -> Pick:
    return Pick(lender_id=lender_id, rank=rank, rationale=f"why {lender_id}", conditions=[])


def mo(
    recommendation: str | None,
    alternates: tuple[str, ...] = (),
    *,
    confidence: Confidence = "high",
) -> ModelOutput:
    return ModelOutput(
        recommendation=pick(recommendation, 1) if recommendation is not None else None,
        alternates=[pick(a, i + 2) for i, a in enumerate(alternates)],
        confidence=confidence,
        co_applicant_suggested=False,
        co_applicant_reason=None,
        review_flags=[],
    )


def check(
    p: Pick,
    lender: GatedLender | None,
    nums: LenderNumbers | None,
    *,
    band: Band = "720-780",
    age_known: bool = True,
    age_app_years: int | None = 30,
) -> SafetyBreach | None:
    return check_pick(p, lender, nums, band=band, age_known=age_known, age_app_years=age_app_years)


def world(*lender_ids: str, **crit_by_all: Any) -> tuple[list[GatedLender], dict[str, LenderNumbers]]:
    return [gl(i, **crit_by_all) for i in lender_ids], {i: num(i) for i in lender_ids}


# --------------------------------------------------------------------------- check_pick: rule 0 not_in_gate


def test_a_healthy_pick_has_no_breach() -> None:
    assert check(pick("L1"), gl("L1"), num("L1")) is None


def test_rule0_not_in_gate_when_the_lender_is_not_gated() -> None:
    breach = check(pick("L9"), None, None)
    assert breach is not None
    assert (breach.lender_id, breach.rule) == ("L9", "not_in_gate")


def test_rule0_not_in_gate_when_there_are_no_numbers_for_the_lender() -> None:
    breach = check(pick("L1"), gl("L1"), None)
    assert breach is not None and breach.rule == "not_in_gate"  # fail closed: nothing to verify the pick against


# --------------------------------------------------------------------------- rule 1 band_floor


@pytest.mark.parametrize(
    ("band", "floor", "breach"),
    [
        ("<650", 650, True),  # below the floor's bucket
        ("<650", 600, False),  # floor 600 is in the '<650' bucket: same bucket clears
        ("650-720", 650, False),
        ("650-720", 719, False),  # floor 719 is still the 650-720 bucket
        ("650-720", 720, True),  # floor 720 is the 720-780 bucket
        ("650-720", 780, True),
        ("720-780", 700, False),
        ("720-780", 780, True),
        ("720-780", 779, False),
        ("780+", 900, False),
        ("780+", 780, False),
        ("<650", 780, True),
    ],
)
def test_rule1_band_floor_compares_buckets_not_scores(band: Band, floor: int, breach: bool) -> None:
    result = check(pick("L1"), gl("L1", score_floor=floor), num("L1"), band=band)
    if breach:
        assert result is not None and result.rule == "band_floor"
    else:
        assert result is None


def test_rule1_floor_unknown_never_breaches_a_scored_band() -> None:
    assert check(pick("L1"), gl("L1", score_floor=None, ntc_accepted=True), num("L1"), band="<650") is None


def test_rule1_ntc_is_a_breach_unless_the_lender_accepts_ntc() -> None:
    for accepted in (False, None):
        result = check(pick("L1"), gl("L1", score_floor=None, ntc_accepted=accepted), num("L1"), band="NTC")
        assert result is not None and result.rule == "band_floor"
    # ntc_accepted True survives (score_floor is irrelevant on the NTC path)
    assert check(pick("L1"), gl("L1", score_floor=780, ntc_accepted=True), num("L1"), band="NTC") is None


def test_rule1_ntc_accepted_true_does_not_waive_later_rules() -> None:
    result = check(
        pick("L1"),
        gl("L1", ntc_accepted=True, foir_limit=0.4),
        num("L1", displayed_foir=0.5),
        band="NTC",
    )
    assert result is not None and result.rule == "foir_limit"


def test_first_breach_wins_in_rule_order() -> None:
    # breaches band_floor (<650 vs floor 650), foir_limit, age and min_loan at once: band_floor is rule 1
    everything_wrong = num("L1", displayed_foir=0.9, age_at_maturity_years=99, loan_max_upper_raw=10.0, ceiling=0.0)
    result = check(pick("L1"), gl("L1"), everything_wrong, band="<650", age_app_years=18)
    assert result is not None and result.rule == "band_floor"
    # clear the band; foir (rule 2) is next
    result = check(pick("L1"), gl("L1"), everything_wrong, band="720-780", age_app_years=18)
    assert result is not None and result.rule == "foir_limit"
    # clear foir; age (rule 3) is next
    result = check(pick("L1"), gl("L1"), num("L1", age_at_maturity_years=99, loan_max_upper_raw=10.0), age_app_years=18)
    assert result is not None and result.rule == "age_bounds"
    # clear age; min_loan (rule 4) is last
    result = check(pick("L1"), gl("L1"), num("L1", loan_max_upper_raw=10.0))
    assert result is not None and result.rule == "min_loan_amount"


def test_breach_carries_the_lender_id_and_an_internal_detail() -> None:
    result = check(pick("L1"), gl("L1", score_floor=780), num("L1"), band="650-720")
    assert result is not None
    assert result.lender_id == "L1"
    assert result.detail  # non-empty, internal only


# --------------------------------------------------------------------------- rule 2 foir_limit (FOIR_EPS)


def test_rule2_displayed_foir_exactly_at_the_limit_passes() -> None:
    assert check(pick("L1"), gl("L1", foir_limit=0.55), num("L1", displayed_foir=0.55)) is None


def test_rule2_within_foir_eps_passes_and_beyond_it_breaches() -> None:
    assert FOIR_EPS == 1e-9
    assert check(pick("L1"), gl("L1", foir_limit=0.55), num("L1", displayed_foir=0.55 + 0.5 * FOIR_EPS)) is None
    result = check(pick("L1"), gl("L1", foir_limit=0.55), num("L1", displayed_foir=0.55 + 2 * FOIR_EPS))
    assert result is not None and result.rule == "foir_limit"


def test_rule2_clearly_over_the_limit_breaches() -> None:
    result = check(pick("L1"), gl("L1", foir_limit=0.5), num("L1", displayed_foir=0.62))
    assert result is not None and result.rule == "foir_limit"


def test_rule2_is_skipped_when_displayed_foir_is_none() -> None:
    assert check(pick("L1"), gl("L1"), num("L1", displayed_foir=None)) is None


def test_rule2_the_range_node_identity_passes_by_construction() -> None:
    # delta-2 L2 worked example: displayed FOIR == foir_limit exactly when unclipped
    c = crit(foir_limit=0.55, rate_min_pct=12.5, rate_max_pct=16.0, tenure_min_months=24, tenure_max_months=48,
             loan_min=100000.0, loan_max=2000000.0)
    n = lender_numbers(c, lender_id="L1", income=60000.0, existing_emi=10000.0, age_months=341)
    g = GatedLender(id="L1", name="n", criteria=c, matched_pincodes=["560001"], rate_band_label="", tenure_label="")
    assert check(pick("L1"), g, n) is None


# --------------------------------------------------------------------------- rule 3 age bounds


def test_rule3_skipped_entirely_when_dob_unknown() -> None:
    n = num("L1", age_at_maturity_years=99)
    assert check(pick("L1"), gl("L1"), n, age_known=False, age_app_years=None) is None
    # even a stray age value is ignored when age_known is False
    assert check(pick("L1"), gl("L1"), n, age_known=False, age_app_years=10) is None


def test_rule3_min_age_breach_and_boundary() -> None:
    result = check(pick("L1"), gl("L1", min_age_years=21), num("L1"), age_app_years=20)
    assert result is not None and result.rule == "age_bounds"
    assert check(pick("L1"), gl("L1", min_age_years=21), num("L1"), age_app_years=21) is None  # exactly the minimum


def test_rule3_maturity_breach_and_boundary() -> None:
    result = check(pick("L1"), gl("L1", max_age_at_maturity_years=58), num("L1", age_at_maturity_years=59))
    assert result is not None and result.rule == "age_bounds"
    assert check(pick("L1"), gl("L1", max_age_at_maturity_years=58), num("L1", age_at_maturity_years=58)) is None


def test_rule3_unset_limits_are_not_checked() -> None:
    n = num("L1", age_at_maturity_years=99)
    g = gl("L1", min_age_years=None, max_age_at_maturity_years=None)
    assert check(pick("L1"), g, n, age_app_years=5) is None


def test_rule3_unknown_maturity_age_is_skipped() -> None:
    assert check(pick("L1"), gl("L1", max_age_at_maturity_years=58), num("L1", age_at_maturity_years=None)) is None


# --------------------------------------------------------------------------- rule 4 min_loan_amount (BINDING ruling 1)


def test_rule4_probe_l2_20000_8160_is_demoted_on_the_raw_even_though_the_display_ceiling_reaches_the_floor() -> None:
    """The brief's exact probe: lender L2 (FOIR 0.45, 14-18%, 12-36m, loans 25,000-500,000), income 20,000,
    existing EMI 8,160. A = 0.45 * 20000 - 8160 = 840; raw upper = 24,577.48 < loan_min 25,000, but half-up display
    rounds the ceiling to 25,000. Keying off the display would let the pick through; the raw must demote it."""
    c = crit(
        score_floor=650, ntc_accepted=True, foir_limit=0.45, min_income=20000.0, max_age_at_maturity_years=60,
        rate_min_pct=14.0, rate_max_pct=18.0, tenure_min_months=12, tenure_max_months=36,
        loan_min=25000.0, loan_max=500000.0,
    )
    n = lender_numbers(c, lender_id="L2", income=20000.0, existing_emi=8160.0, age_months=341)
    assert n.loan_max_upper_raw == pytest.approx(24577.48, abs=0.01)
    assert n.loan_max_upper_raw is not None and n.loan_max_upper_raw < 25000.0  # the raw is below the floor
    assert n.ceiling == 25000.0 and n.loan_max_upper == 25000.0  # ... but the display is not
    assert n.not_fundable_alone is False
    g = GatedLender(id="L2", name="n", criteria=c, matched_pincodes=["560001"], rate_band_label="", tenure_label="")
    breach = check(pick("L2"), g, n)
    assert breach is not None
    assert (breach.lender_id, breach.rule) == ("L2", "min_loan_amount")


def test_rule4_an_affordable_ceiling_passes() -> None:
    assert check(pick("L1"), gl("L1", loan_min=50000.0), num("L1", loan_max_upper_raw=800100.0)) is None


def test_rule4_raw_exactly_at_loan_min_passes_and_just_below_breaches() -> None:
    assert check(pick("L1"), gl("L1", loan_min=50000.0), num("L1", loan_max_upper_raw=50000.0)) is None
    result = check(pick("L1"), gl("L1", loan_min=50000.0), num("L1", loan_max_upper_raw=49999.99))
    assert result is not None and result.rule == "min_loan_amount"


def test_rule4_raw_is_clamped_by_loan_max_before_the_comparison() -> None:
    # raw 30,000 is above loan_min 25,000, but the lender's own cap (24,000) clamps the real ceiling below it
    g = gl("L1", loan_min=25000.0, loan_max=24000.0)
    result = check(pick("L1"), g, num("L1", loan_max_upper_raw=30000.0, ceiling=24000.0, loan_max_upper=24000.0))
    assert result is not None and result.rule == "min_loan_amount"
    # and a cap at/above loan_min leaves the raw to decide
    ok = check(pick("L1"), gl("L1", loan_min=25000.0, loan_max=26000.0), num("L1", loan_max_upper_raw=30000.0))
    assert ok is None


def test_rule4_a_huge_raw_clipped_by_loan_max_still_passes() -> None:
    # delta-2 L1: raw 919,860.68 clipped to the 500,000 cap; the clamp lands above loan_min
    g = gl("L1", loan_min=50000.0, loan_max=500000.0)
    assert check(pick("L1"), g, num("L1", loan_max_upper_raw=919860.68, ceiling=500000.0)) is None


def test_rule4_ceiling_none_alone_is_never_a_breach() -> None:
    # no income (quick without income): every money field None and not_fundable_alone False -> nothing to compare
    n = num(
        "L1", available_emi=None, loan_max_lower=None, loan_max_upper=None, loan_max_lower_raw=None,
        loan_max_upper_raw=None, ceiling=None, displayed_foir=None,
    )
    assert check(pick("L1"), gl("L1"), n) is None


def test_rule4_ceiling_zero_alone_is_never_the_signal() -> None:
    # never key off the display ceiling: a 0 ceiling with a healthy raw is not a breach (the flag/raw decide)
    assert check(pick("L1"), gl("L1"), num("L1", ceiling=0.0, loan_max_upper=0.0, loan_max_upper_raw=800100.0)) is None


def test_rule4_not_fundable_alone_is_a_breach_with_the_no_headroom_detail() -> None:
    """income 30,000, existing EMI 16,000, FOIR 0.50 -> A = 0.0 -> not_fundable_alone with this lender."""
    c = crit(foir_limit=0.50)
    n = lender_numbers(c, lender_id="L1", income=30000.0, existing_emi=16000.0, age_months=341)
    assert n.not_fundable_alone is True and n.available_emi == 0.0
    assert n.ceiling is None and n.loan_max_upper_raw is None  # nothing else to key off
    g = GatedLender(id="L1", name="n", criteria=c, matched_pincodes=["560001"], rate_band_label="", tenure_label="")
    breach = check(pick("L1"), g, n)
    assert breach is not None
    assert (breach.lender_id, breach.rule) == ("L1", "min_loan_amount")
    assert breach.detail == NO_HEADROOM_DETAIL == "no EMI headroom"


def test_rule4_not_fundable_alone_keys_off_the_flag_not_the_ceiling() -> None:
    # a hand-built numbers object whose flag is set but whose ceiling/raw look fine: the flag alone demotes
    n = num("L1", not_fundable_alone=True, available_emi=0.0)
    breach = check(pick("L1"), gl("L1"), n)
    assert breach is not None and breach.rule == "min_loan_amount" and breach.detail == "no EMI headroom"


def test_rule4_no_headroom_detail_differs_from_the_below_min_detail() -> None:
    below = check(pick("L1"), gl("L1", loan_min=50000.0), num("L1", loan_max_upper_raw=1000.0))
    assert below is not None and below.rule == "min_loan_amount" and below.detail != NO_HEADROOM_DETAIL


# --------------------------------------------------------------------------- run_safety


def test_run_safety_none_model_output_means_none_survived_and_review() -> None:
    gated, numbers = world("L1", "L2")
    r = run_safety(None, gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert r == SafetyResult(
        top=None, alternates=[], demoted=[], none_survived=True, review=True, top_clears_floor_by_one=None
    )


def test_run_safety_model_with_no_candidates_means_none_survived_and_review() -> None:
    gated, numbers = world("L1")
    r = run_safety(mo(None), gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert r.top is None and r.alternates == [] and r.demoted == []
    assert r.none_survived is True and r.review is True and r.top_clears_floor_by_one is None


def test_run_safety_happy_path_keeps_top_and_alternates_in_order() -> None:
    gated, numbers = world("L1", "L2", "L3")
    r = run_safety(mo("L2", ("L1", "L3")), gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L2"
    assert [a.lender_id for a in r.alternates] == ["L1", "L3"]
    assert r.demoted == []
    assert r.none_survived is False and r.review is False


def test_run_safety_does_not_rewrite_the_picks() -> None:
    gated, numbers = world("L1", "L2")
    out = mo("L1", ("L2",))
    r = run_safety(out, gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert r.top == out.recommendation
    assert r.alternates == out.alternates


def test_run_safety_promotion_in_model_order_when_the_top_is_demoted() -> None:
    gated, numbers = world("L1", "L2", "L3")
    gated[0] = gl("L1", score_floor=780)  # L1 needs 780+, applicant is '720-780' -> demoted
    r = run_safety(mo("L1", ("L2", "L3")), gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L2"  # FIRST alternate promoted
    assert [a.lender_id for a in r.alternates] == ["L3"]
    assert [(d.lender_id, d.rule) for d in r.demoted] == [("L1", "band_floor")]
    assert r.none_survived is False
    assert r.review is True  # the original recommendation was demoted


def test_run_safety_promotion_skips_a_demoted_first_alternate() -> None:
    gated, numbers = world("L1", "L2", "L3")
    gated[0] = gl("L1", score_floor=780)
    gated[1] = gl("L2", score_floor=780)
    r = run_safety(mo("L1", ("L2", "L3")), gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L3"
    assert r.alternates == []
    assert [d.lender_id for d in r.demoted] == ["L1", "L2"]
    assert r.review is True


def test_run_safety_alternate_only_demotion_is_logged_but_does_not_trigger_review() -> None:
    gated, numbers = world("L1", "L2", "L3")
    gated[1] = gl("L2", score_floor=780)
    r = run_safety(mo("L1", ("L2", "L3")), gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L1"
    assert [a.lender_id for a in r.alternates] == ["L3"]
    assert [(d.lender_id, d.rule) for d in r.demoted] == [("L2", "band_floor")]
    assert r.review is False  # review ONLY when none survived or the original recommendation was demoted


def test_run_safety_all_demoted_is_none_survived_and_review() -> None:
    gated = [gl("L1", score_floor=780), gl("L2", score_floor=780)]
    numbers = {"L1": num("L1"), "L2": num("L2")}
    r = run_safety(mo("L1", ("L2",)), gated, numbers, band="650-720", age_app_years=30, age_known=True)
    assert r.top is None and r.alternates == []
    assert r.none_survived is True and r.review is True
    assert [d.lender_id for d in r.demoted] == ["L1", "L2"]
    assert r.top_clears_floor_by_one is None


def test_run_safety_out_of_gate_pick_is_demoted_not_in_gate() -> None:
    gated, numbers = world("L1")
    r = run_safety(mo("LX", ("L1",)), gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert [(d.lender_id, d.rule) for d in r.demoted] == [("LX", "not_in_gate")]
    assert r.top is not None and r.top.lender_id == "L1"
    assert r.review is True  # the model's own recommendation was the one demoted


def test_run_safety_only_out_of_gate_picks_means_none_survived() -> None:
    gated, numbers = world("L1")
    r = run_safety(mo("LX", ("LY",)), gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert r.none_survived is True and r.review is True
    assert [d.rule for d in r.demoted] == ["not_in_gate", "not_in_gate"]


def test_run_safety_lender_with_no_numbers_entry_is_demoted() -> None:
    gated = [gl("L1"), gl("L2")]
    r = run_safety(mo("L1", ("L2",)), gated, {"L2": num("L2")}, band="780+", age_app_years=30, age_known=True)
    assert [(d.lender_id, d.rule) for d in r.demoted] == [("L1", "not_in_gate")]
    assert r.top is not None and r.top.lender_id == "L2"


def test_run_safety_duplicate_pick_is_a_duplicate_breach_and_the_first_occurrence_survives() -> None:
    gated, numbers = world("L1", "L2")
    out = mo("L1", ("L1", "L2"))
    r = run_safety(out, gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L1"
    assert [a.lender_id for a in r.alternates] == ["L2"]
    assert [(d.lender_id, d.rule) for d in r.demoted] == [("L1", "duplicate")]
    assert r.review is False  # the demoted one was an alternate repeat, not the recommendation


def test_run_safety_a_repeat_of_an_already_demoted_lender_is_not_relabelled_duplicate() -> None:
    gated = [gl("L1", score_floor=780), gl("L2")]
    numbers = {"L1": num("L1"), "L2": num("L2")}
    r = run_safety(mo("L1", ("L1", "L2")), gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert [(d.lender_id, d.rule) for d in r.demoted] == [("L1", "band_floor"), ("L1", "band_floor")]
    assert r.top is not None and r.top.lender_id == "L2"


def test_run_safety_without_a_recommendation_promotes_the_first_surviving_alternate() -> None:
    gated, numbers = world("L1", "L2", "L3")
    r = run_safety(mo(None, ("L1", "L2")), gated, numbers, band="780+", age_app_years=30, age_known=True)
    # no recommendation: the first surviving alternate is promoted to top, the next becomes the alternate
    assert r.top is not None and r.top.lender_id == "L1"
    assert [a.lender_id for a in r.alternates] == ["L2"]
    assert r.review is False  # nothing was demoted


def test_run_safety_caps_the_survivors_at_one_top_plus_two_alternates() -> None:
    """ModelOutput validates ``alternates`` to at most 2 (so a validated output has at most 3 candidates); the cap in
    run_safety is a defensive second line, exercised here with an unvalidated over-long output."""
    gated, numbers = world("L1", "L2", "L3", "L4")
    over_long = ModelOutput.model_construct(
        recommendation=pick("L1", 1),
        alternates=[pick("L2", 2), pick("L3", 3), pick("L4", 4)],
        confidence="high",
        co_applicant_suggested=False,
        co_applicant_reason=None,
        review_flags=[],
    )
    r = run_safety(over_long, gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L1"
    assert [a.lender_id for a in r.alternates] == ["L2", "L3"]  # L4 is the third survivor past the cap: dropped
    assert r.demoted == []  # dropped by the cap, not demoted
    assert r.review is False


def test_run_safety_uses_the_dob_flags() -> None:
    gated = [gl("L1", min_age_years=25), gl("L2")]
    numbers = {"L1": num("L1"), "L2": num("L2")}
    young = run_safety(mo("L1", ("L2",)), gated, numbers, band="780+", age_app_years=22, age_known=True)
    assert [(d.lender_id, d.rule) for d in young.demoted] == [("L1", "age_bounds")]
    unknown = run_safety(mo("L1", ("L2",)), gated, numbers, band="780+", age_app_years=None, age_known=False)
    assert unknown.demoted == [] and unknown.top is not None and unknown.top.lender_id == "L1"


def test_run_safety_not_fundable_alone_pick_is_demoted_with_no_headroom_detail() -> None:
    gated, numbers = world("L1", "L2")
    numbers["L1"] = num("L1", not_fundable_alone=True, available_emi=0.0, ceiling=None, loan_max_upper=None,
                        loan_max_lower=None, loan_max_upper_raw=None, loan_max_lower_raw=None, displayed_foir=None)
    r = run_safety(mo("L1", ("L2",)), gated, numbers, band="780+", age_app_years=30, age_known=True)
    assert [(d.lender_id, d.rule, d.detail) for d in r.demoted] == [("L1", "min_loan_amount", "no EMI headroom")]
    assert r.top is not None and r.top.lender_id == "L2"


# --------------------------------------------------------------------------- top_clears_floor_by_one


@pytest.mark.parametrize(
    ("band", "floor", "expected"),
    [
        ("780+", 650, True),  # 4 - 2
        ("720-780", 650, True),  # 3 - 2 = 1
        ("650-720", 650, False),  # cleared by exactly 0 buckets
        ("720-780", 720, False),
        ("780+", 780, False),
        ("780+", 700, True),
    ],
)
def test_top_clears_floor_by_one_for_the_surviving_top(band: Band, floor: int, expected: bool) -> None:
    gated = [gl("L1", score_floor=floor)]
    r = run_safety(mo("L1"), gated, {"L1": num("L1")}, band=band, age_app_years=30, age_known=True)
    assert r.top is not None
    assert r.top_clears_floor_by_one is expected


def test_top_clears_floor_by_one_is_none_when_the_floor_is_unknown() -> None:
    gated = [gl("L1", score_floor=None, ntc_accepted=True)]
    r = run_safety(mo("L1"), gated, {"L1": num("L1")}, band="780+", age_app_years=30, age_known=True)
    assert r.top is not None and r.top_clears_floor_by_one is None


def test_top_clears_floor_by_one_is_none_on_the_ntc_accepted_path() -> None:
    gated = [gl("L1", score_floor=650, ntc_accepted=True)]
    r = run_safety(mo("L1"), gated, {"L1": num("L1")}, band="NTC", age_app_years=30, age_known=True)
    assert r.top is not None  # NTC-accepting lender survives
    assert r.top_clears_floor_by_one is None


def test_top_clears_floor_by_one_uses_the_promoted_top_not_the_original() -> None:
    gated = [gl("L1", score_floor=780), gl("L2", score_floor=650)]
    numbers = {"L1": num("L1"), "L2": num("L2")}
    r = run_safety(mo("L1", ("L2",)), gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert r.top is not None and r.top.lender_id == "L2"
    assert r.top_clears_floor_by_one is True  # 720-780 vs the 650-720 bucket


def test_run_safety_is_pure_and_deterministic() -> None:
    gated, numbers = world("L1", "L2", "L3")
    gated[0] = gl("L1", score_floor=780)
    out = mo("L1", ("L2", "L3"))
    before = (copy.deepcopy(gated), copy.deepcopy(numbers), out.model_copy(deep=True))
    first = run_safety(out, gated, numbers, band="720-780", age_app_years=30, age_known=True)
    second = run_safety(out, gated, numbers, band="720-780", age_app_years=30, age_known=True)
    assert first == second
    assert (gated, numbers, out) == before


# --------------------------------------------------------------------------- compute_code_confidence


def identity(*, pan_valid: bool = True, dob: date | None = date(1990, 5, 1)) -> Identity:
    return Identity(
        pan_valid=pan_valid, status="valid", category="person", full_name="A B", dob=dob,
        aadhaar_linked=True, pan_pincode="560001", surepass_client_id=None,
    )


def emi_item(
    kind: Literal["reported", "estimated", "stale_excluded"] = "reported", *, past_tenure: bool = False, key: str = "acct_0"
) -> EmiItem:
    return EmiItem(
        account_key=key,
        kind=kind,
        monthly_amount=0.0 if kind == "stale_excluded" else 5000.0,
        method=None if kind == "stale_excluded" else "reported",
        past_tenure=past_tenure,
        confirm_item=None,
    )


def emi(
    *,
    estimated_total: float = 0.0,
    used_source: Literal["derived", "declared", "declared_zero_assumed"] = "derived",
    items: list[EmiItem] | None = None,
) -> EmiDerivation:
    return EmiDerivation(
        reported_total=12500.0,
        estimated_total=estimated_total,
        derived_total=12500.0 + estimated_total,
        declared=None,
        used=12500.0 + estimated_total,
        used_source=used_source,
        emi_floor_flag=estimated_total > 0.0,
        items=items if items is not None else [emi_item("reported")],
    )


def derived(*, band: Band = "780+", age: int | None = 36, emi_value: EmiDerivation | None = None) -> Derived:
    return Derived(
        band=band,
        emi=emi_value if emi_value is not None else emi(),
        current_foir=0.2,
        age_at_application_years=age,
        age_months_at_application=age * 12 + 3 if age is not None else None,
        candidate_pincodes=["560001"],
    )


def cc(
    *,
    kind: Kind = "full",
    ident: Identity | None = None,
    der: Derived | None = None,
    gated: list[GatedLender] | None = None,
    openai_failed: bool = False,
) -> Confidence:
    g = gated if gated is not None else [gl("L1", score_floor=650)]
    return compute_code_confidence(
        kind=kind,
        identity=ident if ident is not None else identity(),
        derived=der if der is not None else derived(),
        gated=g,
        numbers={x.id: num(x.id) for x in g},
        openai_failed=openai_failed,
    )


def test_code_confidence_L1_no_gated_lender_is_low() -> None:
    assert cc(gated=[]) == "low"


def test_code_confidence_L1_beats_every_high_condition() -> None:
    # empty gate with an otherwise-perfect full check
    assert cc(kind="full", gated=[]) == "low"


def test_code_confidence_L2_openai_failed_is_low_even_when_everything_else_is_high() -> None:
    assert cc() == "high"  # sanity: this world IS high
    assert cc(openai_failed=True) == "low"


def test_code_confidence_H_full_check_with_every_condition_is_high() -> None:
    assert cc() == "high"


def test_code_confidence_quick_always_caps_at_medium_even_with_every_high_condition() -> None:
    assert cc(kind="quick") == "medium"


def test_code_confidence_quick_with_no_gated_lender_is_still_low() -> None:
    assert cc(kind="quick", gated=[]) == "low"


@pytest.mark.parametrize(
    "case",
    [
        "pan_not_valid",
        "estimated_emi",
        "stale_excluded_item",
        "past_tenure_item",
        "declared_governs",
        "declared_zero_assumed",
        "dob_unknown",
        "age_unknown_on_derived",
        "uncertain_required_field",
        "uncertain_floor_field",
        "floor_cleared_by_zero",
        "floor_unknown_everywhere",
        "ntc_band",
    ],
)
def test_code_confidence_each_high_condition_failing_lands_on_medium(case: str) -> None:
    ident = identity()
    der = derived()
    gated = [gl("L1", score_floor=650)]
    if case == "pan_not_valid":
        ident = identity(pan_valid=False)
    elif case == "estimated_emi":
        der = derived(emi_value=emi(estimated_total=3000.0, items=[emi_item("reported"), emi_item("estimated", key="a1")]))
    elif case == "stale_excluded_item":
        der = derived(emi_value=emi(items=[emi_item("reported"), emi_item("stale_excluded", key="a1")]))
    elif case == "past_tenure_item":
        der = derived(emi_value=emi(items=[emi_item("reported", past_tenure=True)]))
    elif case == "declared_governs":
        der = derived(emi_value=emi(used_source="declared"))
    elif case == "declared_zero_assumed":
        der = derived(emi_value=emi(used_source="declared_zero_assumed"))
    elif case == "dob_unknown":
        ident = identity(dob=None)
        der = derived(age=None)
    elif case == "age_unknown_on_derived":
        der = derived(age=None)
    elif case == "uncertain_required_field":
        gated = [gl("L1", score_floor=650, field_status={"foir_limit": "uncertain"})]
    elif case == "uncertain_floor_field":
        gated = [gl("L1", score_floor=650, field_status={"score_floor": "uncertain"})]
    elif case == "floor_cleared_by_zero":
        der = derived(band="650-720")
    elif case == "floor_unknown_everywhere":
        gated = [gl("L1", score_floor=None, ntc_accepted=True)]
    elif case == "ntc_band":
        der = derived(band="NTC")
        gated = [gl("L1", score_floor=None, ntc_accepted=True)]
    assert cc(ident=ident, der=der, gated=gated) == "medium"


def test_code_confidence_optional_uncertain_field_does_not_block_high() -> None:
    gated = [gl("L1", score_floor=650, field_status={"min_income": "uncertain", "min_age_years": "uncertain"})]
    assert cc(gated=gated) == "high"


def test_code_confidence_any_gated_lender_with_an_uncertain_required_field_blocks_high() -> None:
    gated = [gl("L1", score_floor=650), gl("L2", score_floor=650, field_status={"loan_max": "uncertain"})]
    assert cc(gated=gated) == "medium"


def test_code_confidence_high_needs_only_one_lender_whose_floor_clears_by_one() -> None:
    # L1 floor 780 -> '780+' band clears by 0; L2 floor 650 -> clears by 2: one such lender is enough
    gated = [gl("L1", score_floor=780), gl("L2", score_floor=650)]
    assert cc(gated=gated) == "high"
    assert cc(gated=[gl("L1", score_floor=780)]) == "medium"  # only the zero-clearance lender


def test_code_confidence_floor_clearance_boundary_is_exactly_one_bucket() -> None:
    assert cc(der=derived(band="720-780"), gated=[gl("L1", score_floor=650)]) == "high"  # 3 - 2 = 1
    assert cc(der=derived(band="720-780"), gated=[gl("L1", score_floor=720)]) == "medium"  # 3 - 3 = 0
    assert cc(der=derived(band="650-720"), gated=[gl("L1", score_floor=600)]) == "high"  # 2 - 1 = 1 ('<650' bucket)


def test_code_confidence_ntc_accepting_only_lender_never_counts_as_floor_clearance() -> None:
    gated = [gl("L1", score_floor=None, ntc_accepted=True)]
    assert cc(der=derived(band="780+"), gated=gated) == "medium"


def test_code_confidence_is_pure_and_deterministic() -> None:
    assert cc() == cc() == "high"


# --------------------------------------------------------------------------- final_confidence


def safety_result(
    *,
    top: str | None = "L1",
    none_survived: bool = False,
    clears: bool | None = True,
    demoted: list[SafetyBreach] | None = None,
    review: bool = False,
) -> SafetyResult:
    return SafetyResult(
        top=pick(top) if top is not None else None,
        alternates=[],
        demoted=demoted if demoted is not None else [],
        none_survived=none_survived,
        review=review,
        top_clears_floor_by_one=clears,
    )


def test_final_confidence_none_in_is_none_out() -> None:
    assert final_confidence(None, None, safety=None, code=None) is None
    assert final_confidence(None, mo("L1", confidence="high"), safety=safety_result(), code=None) is None
    # band-read / pan_not_verified paths never produce a confidence, whatever else is true
    assert final_confidence(None, None, safety=None, code="NO_PINCODE_ON_PAN") is None
    assert final_confidence(None, None, safety=safety_result(none_survived=True), code="OPENAI_FAILURE") is None


@pytest.mark.parametrize(
    ("code_conf", "model_conf", "expected"),
    [
        ("high", "high", "high"),
        ("high", "medium", "medium"),  # model lowers
        ("high", "low", "low"),
        ("medium", "high", "medium"),  # model RAISES: ignored by min()
        ("medium", "medium", "medium"),
        ("medium", "low", "low"),
        ("low", "high", "low"),
        ("low", "medium", "low"),
        ("low", "low", "low"),
    ],
)
def test_final_confidence_is_min_of_code_and_model_in_both_directions(
    code_conf: Confidence, model_conf: Confidence, expected: Confidence
) -> None:
    out = final_confidence(code_conf, mo("L1", confidence=model_conf), safety=safety_result(), code=None)
    assert out == expected


def test_final_confidence_without_a_model_output_is_the_code_confidence() -> None:
    assert final_confidence("high", None, safety=None, code=None) == "high"
    assert final_confidence("medium", None, safety=safety_result(), code=None) == "medium"


def test_final_confidence_openai_failure_is_low() -> None:
    assert final_confidence("high", None, safety=None, code="OPENAI_FAILURE") == "low"
    assert final_confidence("medium", mo("L1", confidence="high"), safety=safety_result(), code="OPENAI_FAILURE") == "low"


def test_final_confidence_none_survived_is_low() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None)
    assert final_confidence("high", mo("L1", confidence="high"), safety=sr, code=None) == "low"


def test_final_confidence_code_low_stays_low_whatever_the_model_says() -> None:
    assert final_confidence("low", mo("L1", confidence="high"), safety=safety_result(), code=None) == "low"


def test_final_confidence_caps_high_to_medium_when_the_top_clears_the_floor_by_zero() -> None:
    sr = safety_result(clears=False)
    assert final_confidence("high", mo("L1", confidence="high"), safety=sr, code=None) == "medium"
    assert final_confidence("high", None, safety=sr, code=None) == "medium"


def test_final_confidence_floor_cap_does_not_apply_when_cleared_or_unknown() -> None:
    assert final_confidence("high", mo("L1", confidence="high"), safety=safety_result(clears=True), code=None) == "high"
    assert final_confidence("high", mo("L1", confidence="high"), safety=safety_result(clears=None), code=None) == "high"


def test_final_confidence_floor_cap_only_ever_lowers() -> None:
    sr = safety_result(clears=False)
    assert final_confidence("medium", mo("L1", confidence="high"), safety=sr, code=None) == "medium"
    assert final_confidence("high", mo("L1", confidence="low"), safety=sr, code=None) == "low"


def test_final_confidence_is_never_above_the_code_confidence() -> None:
    order = {"low": 0, "medium": 1, "high": 2}
    levels: tuple[Confidence, ...] = ("high", "medium", "low")
    for code_conf in levels:
        for model_conf in levels:
            for clears in (True, False, None):
                out = final_confidence(code_conf, mo("L1", confidence=model_conf), safety=safety_result(clears=clears), code=None)
                assert out is not None
                assert order[out] <= order[code_conf]
                assert order[out] <= order[model_conf]


# --------------------------------------------------------------------------- compute_headline (merged table)


_DEFAULT_SAFETY = safety_result()  # a surviving top 'L1' that clears its floor; frozen, so safe to share


def headline(
    *,
    status: LcStatus = "ok",
    band: Band | None = "720-780",
    final_conf: Confidence | None = "high",
    safety: SafetyResult | None = _DEFAULT_SAFETY,
    gated: list[GatedLender] | None = None,
    code: Code | None = None,
    model_output: ModelOutput | None = None,
) -> Headline | None:
    """Defaults describe a clean fundable run; ``safety=None`` / ``gated=[]`` are passed through as-is."""
    return compute_headline(
        status=status,
        band=band,
        final_conf=final_conf,
        safety=safety,
        gated=gated if gated is not None else [gl("L1")],
        code=code,
        model_output=model_output,
    )


Rule = Literal["not_in_gate", "band_floor", "foir_limit", "age_bounds", "min_loan_amount", "duplicate"]


def breach(rule: Rule, detail: str = "x", lender_id: str = "L1") -> SafetyBreach:
    return SafetyBreach(lender_id=lender_id, rule=rule, detail=detail)


NO_HEADROOM = breach("min_loan_amount", NO_HEADROOM_DETAIL)


# ---- (a) status != 'ok' -> None, with ONE exemption: service_failure + OPENAI_FAILURE (K7) -> band-derived (row e)

NON_OK_STATUSES: tuple[LcStatus, ...] = ("refused", "pan_not_verified", "pending_verification", "service_failure")
ALL_CODES: tuple[Code | None, ...] = ("NO_PINCODE_ON_PAN", "OPENAI_FAILURE", "SUREPASS_FAILURE", "CONSENT_MISSING", None)
K7_PAIR: tuple[LcStatus, Code] = ("service_failure", "OPENAI_FAILURE")


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_headline_a_non_ok_status_is_none(status: LcStatus) -> None:
    assert headline(status=status) is None


@pytest.mark.parametrize(
    ("status", "code"), [(s, c) for s in NON_OK_STATUSES for c in ALL_CODES if (s, c) != K7_PAIR]
)
def test_headline_a_non_ok_status_wins_over_every_code_except_the_k7_pair(status: LcStatus, code: Code | None) -> None:
    assert headline(status=status, code=code, band="780+", final_conf="high") is None


def test_headline_a_service_failure_with_surepass_failure_is_none() -> None:
    """The exemption is for OPENAI_FAILURE only: a SurePass service failure has no usable band and shows no headline."""
    assert headline(status="service_failure", code="SUREPASS_FAILURE", band="780+", final_conf="high") is None
    assert headline(status="service_failure", code="SUREPASS_FAILURE", band="NTC", final_conf=None, safety=None) is None


@pytest.mark.parametrize(
    ("band", "expected"),
    [("NTC", "unlikely"), ("<650", "unlikely"), ("650-720", "possible"), ("720-780", "possible"), ("780+", "possible")],
)
def test_headline_a_k7_service_failure_with_openai_failure_is_band_derived(band: Band, expected: Headline) -> None:
    """K7's real wire status for an OPENAI_FAILURE run is 'service_failure'; it must reach the band-derived row."""
    got = headline(status="service_failure", code="OPENAI_FAILURE", band=band, final_conf="low", safety=None)
    assert got == expected


def test_headline_a_k7_pair_is_identical_to_the_status_ok_openai_failure_result() -> None:
    for band in ("NTC", "<650", "650-720", "720-780", "780+"):
        wire = headline(status="service_failure", code="OPENAI_FAILURE", band=band, final_conf="low", safety=None)
        ok = headline(status="ok", code="OPENAI_FAILURE", band=band, final_conf="low", safety=None)
        assert wire == ok is not None


def test_headline_a_k7_pair_still_precedes_the_generic_low_rows() -> None:
    sr = safety_result(top=None, none_survived=True, demoted=[breach("band_floor")])
    assert headline(status="service_failure", code="OPENAI_FAILURE", band="720-780", final_conf="low", safety=sr) == "possible"
    assert headline(status="service_failure", code="OPENAI_FAILURE", band="720-780", final_conf="low", safety=None, gated=[]) == "possible"
    assert headline(status="service_failure", code="OPENAI_FAILURE", band=None, final_conf="low", safety=None) == "under_review"  # defensive: no band


@pytest.mark.parametrize("status", ["refused", "pan_not_verified", "pending_verification"])
def test_headline_a_the_exemption_is_exactly_one_pair_other_statuses_with_openai_failure_stay_none(status: LcStatus) -> None:
    assert headline(status=status, code="OPENAI_FAILURE", band="780+", final_conf="low", safety=None) is None


# ---- (b) NO_PINCODE_ON_PAN: band-read


@pytest.mark.parametrize(("band", "expected"), [("NTC", "unlikely"), ("<650", "unlikely")])
def test_headline_b_no_pincode_weak_bands_are_unlikely(band: Band, expected: Headline) -> None:
    assert headline(code="NO_PINCODE_ON_PAN", band=band, final_conf=None, safety=None, gated=[]) == expected


@pytest.mark.parametrize("band", ["650-720", "720-780", "780+"])
def test_headline_b_no_pincode_other_bands_are_possible(band: Band) -> None:
    assert headline(code="NO_PINCODE_ON_PAN", band=band, final_conf=None, safety=None, gated=[]) == "possible"


def test_headline_b_no_pincode_ignores_confidence_safety_and_gate() -> None:
    sr = safety_result(top=None, none_survived=True)
    assert headline(code="NO_PINCODE_ON_PAN", band="780+", final_conf="low", safety=sr, gated=[]) == "possible"


def test_headline_b_no_pincode_with_unknown_band_is_the_defensive_under_review() -> None:
    assert headline(code="NO_PINCODE_ON_PAN", band=None, final_conf=None) == "under_review"


# ---- (e) OPENAI_FAILURE -> band-derived, BEFORE the generic low rows (K7)


@pytest.mark.parametrize(("band", "expected"), [("NTC", "unlikely"), ("<650", "unlikely"), ("650-720", "possible"),
                                                ("720-780", "possible"), ("780+", "possible")])
def test_headline_e_openai_failure_is_band_derived_never_under_review(band: Band, expected: Headline) -> None:
    got = headline(code="OPENAI_FAILURE", band=band, final_conf="low", safety=None, gated=[gl("L1"), gl("L2")])
    assert got == expected
    assert got != "under_review"


def test_headline_e_openai_failure_beats_the_generic_low_rows() -> None:
    # final low + non-empty gate + (would-be) mixed/none demotions: generic rows would say under_review
    sr = safety_result(top=None, none_survived=True, demoted=[breach("foir_limit")])
    assert headline(code="OPENAI_FAILURE", band="780+", final_conf="low", safety=sr) == "possible"
    # and a band_floor-only demotion (row c would say not_fundable_alone) is also overridden
    sr2 = safety_result(top=None, none_survived=True, demoted=[breach("band_floor")])
    assert headline(code="OPENAI_FAILURE", band="720-780", final_conf="low", safety=sr2) == "possible"


# ---- (c) final low AND (gated empty OR every demotion band_floor/no-headroom) -> not_fundable_alone


def test_headline_c_low_with_an_empty_gate_is_not_fundable_alone() -> None:
    assert headline(final_conf="low", gated=[], safety=safety_result(top=None, none_survived=True)) == "not_fundable_alone"


def test_headline_c_low_with_a_none_gate_is_treated_as_empty() -> None:
    assert compute_headline(
        status="ok", band="780+", final_conf="low", safety=None, gated=None, code=None, model_output=None
    ) == "not_fundable_alone"


def test_headline_c_every_demotion_band_floor_is_not_fundable_alone() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[breach("band_floor"), breach("band_floor", lender_id="L2")])
    assert headline(final_conf="low", safety=sr) == "not_fundable_alone"


def test_headline_c_every_demotion_no_headroom_is_not_fundable_alone() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[NO_HEADROOM, breach("min_loan_amount", NO_HEADROOM_DETAIL, "L2")])
    assert headline(final_conf="low", safety=sr) == "not_fundable_alone"


def test_headline_c_band_floor_plus_no_headroom_mix_is_not_fundable_alone() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[breach("band_floor"), NO_HEADROOM])
    assert headline(final_conf="low", safety=sr) == "not_fundable_alone"


# ---- (d) final low otherwise -> under_review


@pytest.mark.parametrize("rule", ["foir_limit", "age_bounds", "not_in_gate", "duplicate"])
def test_headline_d_low_with_any_other_demotion_rule_is_under_review(rule: Rule) -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[breach("band_floor"), breach(rule)])
    assert headline(final_conf="low", safety=sr) == "under_review"


def test_headline_d_below_min_loan_is_not_the_no_headroom_pattern() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[breach("min_loan_amount", "ceiling below minimum loan")])
    assert headline(final_conf="low", safety=sr) == "under_review"


def test_headline_d_low_when_the_model_returned_nothing_is_under_review_not_vacuous_not_fundable() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[])
    assert headline(final_conf="low", safety=sr, gated=[gl("L1")]) == "under_review"


def test_headline_d_low_with_no_safety_result_and_a_gated_list_is_under_review() -> None:
    assert compute_headline(
        status="ok", band="780+", final_conf="low", safety=None, gated=[gl("L1")], code=None, model_output=None
    ) == "under_review"


def test_headline_d_low_with_a_survivor_and_a_gated_list_is_under_review() -> None:
    # low can come from the model/code confidence even when a pick survived: no co-applicant forcing, just review
    assert headline(final_conf="low", safety=safety_result(top="L1"), gated=[gl("L1")]) == "under_review"


# ---- F3: the demotion-driven rows (c's demotion clause, d) need that NOTHING survived

def test_headline_f3_a_surviving_top_with_a_band_floor_demoted_alternate_is_not_not_fundable_alone() -> None:
    sr = safety_result(top="L1", clears=True, demoted=[breach("band_floor", lender_id="L2")])
    got = headline(band="720-780", final_conf="low", safety=sr, gated=[gl("L1"), gl("L2")])
    assert got != "not_fundable_alone"  # an applicant WITH a surviving lender is not 'not fundable alone'
    assert got == "under_review"  # falls through to row i: no g/h row exists for final 'low'


def test_headline_f3_a_surviving_top_with_a_no_headroom_demoted_alternate_is_not_not_fundable_alone() -> None:
    sr = safety_result(top="L1", clears=True, demoted=[breach("min_loan_amount", NO_HEADROOM_DETAIL, "L2")])
    got = headline(band="780+", final_conf="low", safety=sr, gated=[gl("L1"), gl("L2")])
    assert got != "not_fundable_alone"


def test_headline_f3_the_same_demotions_with_no_survivor_still_fire_not_fundable_alone() -> None:
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[breach("band_floor", lender_id="L2")])
    assert headline(band="720-780", final_conf="low", safety=sr, gated=[gl("L2")]) == "not_fundable_alone"


def test_headline_f3_ntc_with_a_surviving_top_and_a_demoted_alternate_falls_to_the_ntc_row_not_row_d() -> None:
    # before F3 row d fired here ('under_review'); with a survivor the NTC-survivor row f decides
    sr = safety_result(top="L1", clears=None, demoted=[breach("foir_limit", lender_id="L2")])
    assert headline(band="NTC", final_conf="low", safety=sr, gated=[gl("L1"), gl("L2")]) == "unlikely"


@pytest.mark.parametrize(("conf", "expected"), [("high", "fundable"), ("medium", "possible")])
def test_headline_f3_a_surviving_top_with_demoted_alternates_follows_the_confidence_rows(
    conf: Confidence, expected: Headline
) -> None:
    sr = safety_result(top="L1", clears=True, demoted=[breach("band_floor", lender_id="L2"), breach("foir_limit", lender_id="L3")])
    assert headline(band="720-780", final_conf=conf, safety=sr, gated=[gl("L1"), gl("L2"), gl("L3")]) == expected


def test_headline_f3_gated_empty_is_not_demotion_driven_and_needs_no_survivor_flag() -> None:
    # row c's "gate empty" clause is independent of the survivor test
    assert headline(final_conf="low", gated=[], safety=None) == "not_fundable_alone"


# ---- (f) NTC survivor -> unlikely


@pytest.mark.parametrize("conf", ["medium", "high"])
def test_headline_f_ntc_with_a_surviving_pick_is_unlikely(conf: Confidence) -> None:
    sr = safety_result(top="L1", clears=None)
    assert headline(band="NTC", final_conf=conf, safety=sr) == "unlikely"


def test_headline_f_ntc_override_beats_the_possible_and_fundable_rows() -> None:
    sr = safety_result(top="L1", clears=None)
    assert headline(band="NTC", final_conf="medium", safety=sr) != "possible"
    assert headline(band="NTC", final_conf="high", safety=sr) != "fundable"


def test_headline_f_ntc_without_a_survivor_at_medium_is_the_defensive_under_review() -> None:
    sr = safety_result(top=None, none_survived=False, clears=None)
    assert headline(band="NTC", final_conf="medium", safety=sr) == "under_review"


def test_headline_f_ntc_low_is_a_low_row_not_the_override() -> None:
    # NTC with every pick demoted (lender does not take NTC) -> band_floor-only -> not_fundable_alone
    sr = safety_result(top=None, none_survived=True, clears=None, demoted=[breach("band_floor")])
    assert headline(band="NTC", final_conf="low", safety=sr) == "not_fundable_alone"


# ---- (g) high + top -> fundable, (h) medium + top -> possible, (i) defensive


def test_headline_g_high_with_a_top_is_fundable() -> None:
    assert headline(band="780+", final_conf="high", safety=safety_result(top="L1")) == "fundable"


def test_headline_h_medium_with_a_top_is_possible() -> None:
    assert headline(band="720-780", final_conf="medium", safety=safety_result(top="L1")) == "possible"


@pytest.mark.parametrize("conf", ["high", "medium"])
def test_headline_i_no_top_at_medium_or_high_is_the_defensive_under_review(conf: Confidence) -> None:
    sr = safety_result(top=None, none_survived=False, clears=None)
    assert headline(band="780+", final_conf=conf, safety=sr) == "under_review"


def test_headline_i_final_confidence_none_on_an_ok_status_is_under_review() -> None:
    assert compute_headline(
        status="ok", band="780+", final_conf=None, safety=None, gated=[gl("L1")], code=None, model_output=None
    ) == "under_review"


@pytest.mark.parametrize("band", ["<650", "650-720", "720-780", "780+"])
def test_headline_g_h_do_not_depend_on_the_scored_band(band: Band) -> None:
    assert headline(band=band, final_conf="high", safety=safety_result(top="L1")) == "fundable"
    assert headline(band=band, final_conf="medium", safety=safety_result(top="L1")) == "possible"


def test_headline_is_pure_and_deterministic() -> None:
    sr = safety_result(top="L1")
    out = mo("L1")
    a = compute_headline(status="ok", band="780+", final_conf="high", safety=sr, gated=[gl("L1")], code=None, model_output=out)
    b = compute_headline(status="ok", band="780+", final_conf="high", safety=sr, gated=[gl("L1")], code=None, model_output=out)
    assert a == b == "fundable"


def test_headline_row_order_is_exhaustive_over_the_status_ok_cross_product() -> None:
    """Every (code, band, final, survivor, demotion-kind) combination lands on a defined headline: no exceptions and
    never None for status ok."""
    bands: tuple[Band, ...] = ("NTC", "<650", "650-720", "720-780", "780+")
    confs: tuple[Confidence | None, ...] = ("high", "medium", "low", None)
    codes: tuple[Code | None, ...] = (None, "NO_PINCODE_ON_PAN", "OPENAI_FAILURE")
    safeties: tuple[SafetyResult | None, ...] = (
        None,
        safety_result(top="L1"),
        safety_result(top=None, none_survived=True, demoted=[]),
        safety_result(top=None, none_survived=True, demoted=[breach("band_floor")]),
        safety_result(top=None, none_survived=True, demoted=[breach("foir_limit")]),
    )
    valid = {"fundable", "possible", "unlikely", "not_fundable_alone", "under_review"}
    for band in bands:
        for conf in confs:
            for code in codes:
                for sr in safeties:
                    for gated in ([], [gl("L1")]):
                        got = compute_headline(
                            status="ok", band=band, final_conf=conf, safety=sr, gated=gated, code=code, model_output=None
                        )
                        assert got in valid


def test_headline_status_cross_product_is_none_exactly_off_the_ok_path_and_never_none_on_it() -> None:
    statuses: tuple[LcStatus, ...] = ("ok", *NON_OK_STATUSES)
    codes: tuple[Code | None, ...] = (None, "NO_PINCODE_ON_PAN", "OPENAI_FAILURE", "SUREPASS_FAILURE")
    for status in statuses:
        for code in codes:
            for band in ("NTC", "<650", "780+"):
                got = compute_headline(
                    status=status, band=band, final_conf="low", safety=None, gated=[gl("L1")], code=code, model_output=None
                )
                on_ok_path = status == "ok" or (status, code) == K7_PAIR
                assert (got is not None) is on_ok_path, (status, code, band, got)


# --------------------------------------------------------------------------- module-level: purity + registry


def test_safety_module_imports_only_stdlib_state_range_and_gate() -> None:
    tree = ast.parse(Path(inspect.getfile(safety_module)).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module if node.module.startswith("eligibility_agent") else node.module.split(".")[0])
    stdlib = {"__future__", "collections", "typing"}
    sanctioned = {"eligibility_agent.graph.state", "eligibility_agent.derive.range", "eligibility_agent.lenders.gate"}
    assert imported <= stdlib | sanctioned, imported - stdlib - sanctioned


def test_safety_never_reads_a_clock_or_touches_a_repo() -> None:
    source = Path(inspect.getfile(safety_module)).read_text(encoding="utf-8")
    assert not re.search(r"\.now\(|utcnow|time\.time|\.today\(|random|uuid", source)
    names = {n for n in dir(safety_module) if not n.startswith("__")}
    assert not names & {"Repo", "MemoryRepo", "SupabaseRepo", "RawStore", "MemoryRawStore"}


def test_headline_is_homed_in_safety_under_the_compute_name() -> None:
    assert not hasattr(safety_module, "derive_headline")  # K13: named compute_headline
    assert compute_headline.__module__ == "eligibility_agent.lenders.safety"
