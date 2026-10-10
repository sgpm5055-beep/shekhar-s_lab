"""Task 9 (part 2): output/lc_view.py, the four LC-view builders (the ONLY exit boundary).

BINDING pins exercised below:
  * K13: exactly four builders (``build_lc_view`` / ``build_refusal_lc_view`` / ``build_failure_lc_view`` /
    ``build_error_lc_view``); the headline comes from ``lenders.safety.compute_headline`` and is passed the TRUE wire
    status (``service_failure`` + ``OPENAI_FAILURE`` on the K7 path, never a synthetic ``'ok'``).
  * K1: ``pending_verification`` carries ``code=None``; ``MSG_PENDING_VERIFICATION`` rides in ``label``.
  * K6: LenderView numbers are ``LenderNumbers`` display values verbatim (no re-rounding in the view layer).
  * K7: the OPENAI_FAILURE view (confidence low, band-derived headline, no top, at most two code-built, number-free
    alternates whose floor the band clears, ordered by id).
  * K15: the band-read (NO_PINCODE_ON_PAN) and pan_not_verified paths have ``confidence=None`` and never reach
    ``final_confidence``.
  * K20: ``label`` is the one human line, selected inside the builders.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator
from datetime import date
from typing import Any, Literal, cast

import pytest

from eligibility_agent.derive.range import lender_numbers
from eligibility_agent.graph.state import (
    Band,
    CheckState,
    Code,
    Confidence,
    Derived,
    EmiDerivation,
    EmiItem,
    GatedLender,
    Headline,
    Identity,
    Kind,
    LcView,
    LenderCriteria,
    LenderNumbers,
    ModelOutput,
    Pick,
    ReportFacts,
    SafetyBreach,
    SafetyResult,
    ScoreFacts,
)
from eligibility_agent.lenders import safety as safety_module
from eligibility_agent.lenders.safety import check_pick, compute_headline, run_safety
from eligibility_agent.output import lc_view as lc_view_module
from eligibility_agent.output.lc_view import (
    AADHAAR_WARNING,
    GLOBAL_LEAK_PATTERNS,
    LABEL_FULL_FMT,
    LABEL_QUICK,
    LABEL_QUICK_NO_EMI,
    LLM_LEAK_PATTERNS,
    MSG_BUREAU_SERVICE_FAILURE,
    MSG_INTERNAL,
    MSG_NO_PINCODE,
    MSG_OPENAI_FAILURE,
    MSG_PAN_NOT_VERIFIED,
    MSG_PAN_SERVICE_FAILURE,
    MSG_PENDING_VERIFICATION,
    RATIONALE_RANKING_UNAVAILABLE,
    REFUSAL_MESSAGES,
    RedactionContext,
    build_error_lc_view,
    build_failure_lc_view,
    build_lc_view,
    build_refusal_lc_view,
    line_leaks,
)

# --------------------------------------------------------------------------- builders (synthetic only)

ALL_BANDS: tuple[Band, ...] = ("NTC", "<650", "650-720", "720-780", "780+")


def crit(**overrides: Any) -> LenderCriteria:
    """Seed lender L1: FOIR 0.55, 11-14%, 12-48m, loans 50,000-1,500,000, floor 650."""
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


SEED_CRITERIA: dict[str, dict[str, Any]] = {
    "L1": {},
    "L2": {
        "ntc_accepted": True,
        "foir_limit": 0.45,
        "min_income": 20000.0,
        "max_age_at_maturity_years": 60,
        "rate_min_pct": 14.0,
        "rate_max_pct": 18.0,
        "tenure_max_months": 36,
        "loan_min": 25000.0,
        "loan_max": 500000.0,
    },
    "L3": {
        "score_floor": 720,
        "foir_limit": 0.60,
        "min_income": 50000.0,
        "min_age_years": 23,
        "max_age_at_maturity_years": 55,
        "rate_min_pct": 10.5,
        "rate_max_pct": 12.5,
        "tenure_min_months": 24,
        "tenure_max_months": 60,
        "loan_min": 100000.0,
        "loan_max": 2500000.0,
    },
}
SEED_NAMES = {"L1": "Alpha Education Finance", "L2": "Bharat Upskill Credit", "L3": "Crest Capital NBFC"}
SEED_RATE_BANDS = {"L1": "11 to 14%", "L2": "14 to 18%", "L3": "10.5 to 12.5%"}
SEED_TENURES = {"L1": "12 to 48 months", "L2": "12 to 36 months", "L3": "24 to 60 months"}


def gl(lender_id: str = "L1", **overrides: Any) -> GatedLender:
    seed = {**SEED_CRITERIA.get(lender_id, {}), **overrides.pop("criteria", {})}
    fields: dict[str, Any] = {
        "id": lender_id,
        "name": SEED_NAMES.get(lender_id, f"Lender {lender_id}"),
        "criteria": crit(**seed),
        "matched_pincodes": ["560001"],
        "rate_band_label": SEED_RATE_BANDS.get(lender_id, "11 to 14%"),
        "tenure_label": SEED_TENURES.get(lender_id, "12 to 48 months"),
    }
    fields.update(overrides)
    return GatedLender(**fields)


def pick(lender_id: str, rank: int = 1, *, rationale: str | None = None, conditions: list[str] | None = None) -> Pick:
    return Pick(
        lender_id=lender_id,
        rank=rank,
        rationale=rationale if rationale is not None else f"why {lender_id}",
        conditions=conditions if conditions is not None else [f"condition for {lender_id}"],
    )


def mo(
    recommendation: str | None,
    alternates: tuple[str, ...] = (),
    *,
    confidence: Confidence = "medium",
    co_applicant_suggested: bool = False,
    co_applicant_reason: str | None = None,
) -> ModelOutput:
    return ModelOutput(
        recommendation=pick(recommendation, 1) if recommendation is not None else None,
        alternates=[pick(a, i + 2) for i, a in enumerate(alternates)],
        confidence=confidence,
        co_applicant_suggested=co_applicant_suggested,
        co_applicant_reason=co_applicant_reason,
        review_flags=[],
    )


def identity(*, aadhaar_linked: bool | None = True, pan_valid: bool = True) -> Identity:
    return Identity(
        pan_valid=pan_valid,
        status="valid" if pan_valid else "invalid",
        category="person",
        full_name="Ravi Testkumar",
        dob=date(1998, 4, 12),
        aadhaar_linked=aadhaar_linked,
        pan_pincode="560001",
        surepass_client_id="synthetic_client",
    )


def emi_item(confirm_item: str | None, key: str = "acct_0") -> EmiItem:
    return EmiItem(
        account_key=key,
        kind="stale_excluded" if confirm_item else "reported",
        monthly_amount=0.0,
        method=None,
        past_tenure=False,
        confirm_item=confirm_item,
    )


def derived(band: Band = "720-780", *, confirm_items: tuple[str | None, ...] = ()) -> Derived:
    items = [emi_item(c, f"acct_{i}") for i, c in enumerate(confirm_items)]
    return Derived(
        band=band,
        emi=EmiDerivation(
            reported_total=0.0,
            estimated_total=0.0,
            derived_total=0.0,
            declared=None,
            used=0.0,
            used_source="declared_zero_assumed",
            emi_floor_flag=False,
            items=items,
        ),
        current_foir=0.0,
        age_at_application_years=28,
        age_months_at_application=341,
        candidate_pincodes=["560001"],
    )


def score_facts(band: Band = "720-780", pulled_on: date = date(2026, 10, 10)) -> ScoreFacts:
    return ScoreFacts(band=band, score=None if band == "NTC" else 745, is_ntc=band == "NTC", pulled_on=pulled_on, source="score")


def report_facts(band: Band = "720-780", report_date: date | None = date(2026, 10, 1)) -> ReportFacts:
    return ReportFacts(
        band=band,
        score=None if band == "NTC" else 745,
        is_ntc=band == "NTC",
        report_date=report_date,
        open_accounts=[],
        past_due_total=0.0,
        written_off_or_settled=0,
        enquiries_30d=0,
        enquiries_90d=0,
        enquiries_180d=0,
        bureau_pincodes_12m=["560001"],
    )


def numbers_for(
    gated: list[GatedLender], *, income: float | None = 60000.0, existing_emi: float = 0.0
) -> dict[str, LenderNumbers]:
    return {
        g.id: lender_numbers(g.criteria, lender_id=g.id, income=income, existing_emi=existing_emi, age_months=341)
        for g in gated
    }


_UNSET: Any = object()


def make_state(**overrides: Any) -> CheckState:
    """A realistic quick-check state: real lender_numbers + real run_safety; every key overridable.

    ``numbers=`` / ``safety=`` default to the REAL computation; pass ``None`` explicitly to leave the key as None.
    """
    band: Band = overrides.pop("band", "720-780")
    gated: list[GatedLender] = overrides.pop("gated", [gl("L1"), gl("L2")])
    income: float | None = overrides.pop("income", 60000.0)
    nums = overrides.pop("numbers", _UNSET)
    if nums is _UNSET:
        nums = numbers_for(gated, income=income)
    model: ModelOutput | None = overrides.pop("model", mo("L1", ("L2",)))
    confirm_items: tuple[str | None, ...] = overrides.pop("confirm_items", ())
    safety = overrides.pop("safety", _UNSET)
    if safety is _UNSET:
        safety = run_safety(model, gated, nums or {}, band=band, age_app_years=28, age_known=True)
    base: dict[str, Any] = {
        "check_id": "chk-1",
        "kind": "quick",
        "monthly_income": income,
        "existing_emi_declared": 0.0,
        "identity": identity(),
        "score": score_facts(band),
        "derived": derived(band, confirm_items=confirm_items),
        "gated_lenders": gated,
        "lender_numbers": nums,
        "code_confidence": "medium",
        "model_output": model,
        "safety": safety,
    }
    base.update(overrides)
    return cast("CheckState", base)


def ctx(
    *,
    left: int | None = 4,
    checked_on: date | None = date(2026, 10, 10),
    from_cache: bool = False,
    refresh: date | None = date(2026, 11, 9),
) -> RedactionContext:
    return RedactionContext(
        quick_checks_left_today=left, checked_on=checked_on, from_cache=from_cache, refresh_available_on=refresh
    )


# --------------------------------------------------------------------------- scan helpers


def leaves(node: object, key: str = "") -> Iterator[tuple[str, str]]:
    """Every string leaf of a dumped view EXCEPT the two key fields: ``check_id`` (a uuid4 can hold a standalone 3-digit
    run such as '550e8400') and the lender ``id`` (an admin DB key); both are enum/key values, not prose."""
    if isinstance(node, dict):
        for k, v in cast("dict[str, object]", node).items():
            if k in ("check_id", "id"):
                continue
            yield from leaves(v, k)
    elif isinstance(node, list):
        for i, v in enumerate(cast("list[object]", node)):
            yield from leaves(v, f"{key}[{i}]")
    elif isinstance(node, str):
        yield key, node


PROSE_FIELDS = ("rationale", "conditions", "items_to_confirm", "coapplicant_reason")


def assert_clean(view: LcView) -> None:
    """Global tier on every string leaf; the (stricter) LLM tier on the free-text fields; wire round-trip."""
    for key, text in leaves(view.model_dump(mode="json")):
        assert line_leaks(text, GLOBAL_LEAK_PATTERNS) is False, (key, text)
        if key.split("[")[0] in PROSE_FIELDS:
            assert line_leaks(text, LLM_LEAK_PATTERNS) is False, (key, text)
    assert LcView.model_validate(view.model_dump(mode="json")) == view


# --------------------------------------------------------------------------- RedactionContext


def test_redaction_context_is_a_frozen_slotted_dataclass_with_the_four_fields() -> None:
    c = ctx()
    assert [f.name for f in dataclasses.fields(c)] == [
        "quick_checks_left_today",
        "checked_on",
        "from_cache",
        "refresh_available_on",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.from_cache = True  # pyright: ignore[reportAttributeAccessIssue]
    assert not hasattr(c, "__dict__")


# --------------------------------------------------------------------------- build_lc_view: the happy (ok) view


def test_quick_happy_view_numbers_are_the_lender_numbers_display_ints() -> None:
    state = make_state()
    view = build_lc_view(state, ctx())
    assert view.status == "ok" and view.code is None and view.kind == "quick" and view.check_id == "chk-1"
    assert view.headline == "possible" and view.confidence == "medium"
    assert view.label == LABEL_QUICK  # income AND declared EMI both given
    top = view.top_lender
    assert top is not None
    assert (top.id, top.name, top.rate_band, top.tenure_range) == ("L1", "Alpha Education Finance", "11 to 14%", "12 to 48 months")
    assert top.ceiling == 1277000
    assert top.range is not None and (top.range.lower, top.range.upper) == (368000, 1277000)
    assert (top.rationale, top.conditions) == ("why L1", ["condition for L1"])
    assert len(view.alternates) == 1
    alt = view.alternates[0]
    assert (alt.id, alt.name) == ("L2", "Bharat Upskill Credit")
    assert alt.ceiling == 500000  # the lender's loan cap clipped the annuity ceiling upstream
    assert alt.range is not None and (alt.range.lower, alt.range.upper) == (295000, 500000)
    assert view.coapplicant_suggested is False and view.coapplicant_reason is None
    assert view.pan_aadhaar_link_warning is None and view.items_to_confirm == []
    assert (view.checked_on, view.from_cache, view.refresh_available_on, view.quick_checks_left_today) == (
        "2026-10-10",
        False,
        "2026-11-09",
        4,
    )
    assert isinstance(top.ceiling, int) and not isinstance(top.ceiling, bool)
    assert_clean(view)


def test_view_numbers_are_verbatim_never_re_rounded_here() -> None:
    # K6: rounding has ONE home (derive.range, applied upstream). The view layer passes display values through
    # untouched, even ones a real LenderNumbers would never hold (that is what proves nothing rounds here).
    odd = LenderNumbers(
        lender_id="L1",
        available_emi=1.0,
        loan_max_lower=123456.0,
        loan_max_upper=451234.0,
        loan_max_lower_raw=123456.7,
        loan_max_upper_raw=451234.9,
        ceiling=451234.0,
        displayed_foir=0.5,
        age_at_maturity_years=30,
        not_fundable_alone=False,
        clipped="none",
    )
    view = build_lc_view(make_state(gated=[gl("L1")], model=mo("L1"), numbers={"L1": odd}), ctx())
    top = view.top_lender
    assert top is not None and top.ceiling == 451234
    assert top.range is not None and (top.range.lower, top.range.upper) == (123456, 451234)


def test_a_smuggled_three_digit_ceiling_is_killed_upstream_not_here() -> None:
    # The Task-6 helper turns a raw 745-rupee bound into 1,000 BEFORE it reaches LenderNumbers; the view layer shows
    # that display value as-is (a second rounding here would be a K6 violation).
    lender = gl("L1", criteria={"loan_min": 0.0})
    nums = lender_numbers(lender.criteria, lender_id="L1", income=35.0, existing_emi=0.0, age_months=341)
    assert nums.loan_max_upper_raw is not None and 700 < nums.loan_max_upper_raw < 800
    assert nums.loan_max_upper == 1000.0 and nums.ceiling == 1000.0
    view = build_lc_view(make_state(gated=[lender], model=mo("L1"), numbers={"L1": nums}, income=35.0), ctx())
    assert view.top_lender is not None and view.top_lender.ceiling == 1000


def test_no_income_gives_none_ceiling_and_range_on_every_lender_view() -> None:
    state = make_state(income=None, existing_emi_declared=None)
    view = build_lc_view(state, ctx())
    assert view.label == LABEL_QUICK
    shown = [v for v in [view.top_lender, *view.alternates] if v is not None]
    assert [v.id for v in shown] == ["L1", "L2"]
    assert all(v.ceiling is None and v.range is None for v in shown)
    assert_clean(view)


def test_range_is_none_when_either_display_bound_is_missing() -> None:
    half = LenderNumbers(
        lender_id="L1",
        available_emi=1.0,
        loan_max_lower=None,
        loan_max_upper=900000.0,
        loan_max_lower_raw=None,
        loan_max_upper_raw=900000.0,
        ceiling=900000.0,
        displayed_foir=0.5,
        age_at_maturity_years=30,
        not_fundable_alone=False,
        clipped="none",
    )
    view = build_lc_view(make_state(gated=[gl("L1")], model=mo("L1"), numbers={"L1": half}), ctx())
    top = view.top_lender
    assert top is not None and top.ceiling == 900000 and top.range is None


def test_non_finite_display_numbers_become_none_not_a_crash() -> None:
    bad = LenderNumbers(
        lender_id="L1",
        available_emi=1.0,
        loan_max_lower=float("nan"),
        loan_max_upper=float("inf"),
        loan_max_lower_raw=None,
        loan_max_upper_raw=None,
        ceiling=float("inf"),
        displayed_foir=0.5,
        age_at_maturity_years=30,
        not_fundable_alone=False,
        clipped="none",
    )
    view = build_lc_view(make_state(gated=[gl("L1")], model=mo("L1"), numbers={"L1": bad}), ctx())
    top = view.top_lender
    assert top is not None and top.ceiling is None and top.range is None


# --------------------------------------------------------------------------- label selection matrix


@pytest.mark.parametrize(
    ("income", "declared_emi", "expected"),
    [
        (60000.0, None, LABEL_QUICK_NO_EMI),  # income given, EMI not
        (60000.0, 0.0, LABEL_QUICK),  # an explicit 0 IS a given EMI
        (60000.0, 15000.0, LABEL_QUICK),
        (None, None, LABEL_QUICK),
        (None, 5000.0, LABEL_QUICK),
    ],
)
def test_quick_label_matrix(income: float | None, declared_emi: float | None, expected: str) -> None:
    view = build_lc_view(make_state(income=income, existing_emi_declared=declared_emi), ctx())
    assert view.label == expected


@pytest.mark.parametrize("declared_emi", [None, 0.0, 25000.0])
def test_full_label_is_the_provenance_label_with_the_report_date(declared_emi: float | None) -> None:
    state = make_state(kind="full", report=report_facts(report_date=date(2026, 10, 1)), existing_emi_declared=declared_emi)
    view = build_lc_view(state, ctx(left=None))
    assert view.label == "bureau report on 2026-10-01, income as declared"
    assert view.label == LABEL_FULL_FMT.format(report_date="2026-10-01")
    assert_clean(view)


def test_full_label_falls_back_to_the_pull_date_when_the_report_has_no_date() -> None:
    state = make_state(kind="full", report=report_facts(report_date=None))
    assert build_lc_view(state, ctx(checked_on=date(2026, 10, 3))).label == "bureau report on 2026-10-03, income as declared"


def test_full_label_without_any_date_is_a_programming_error() -> None:
    state = make_state(kind="full", report=report_facts(report_date=None))
    with pytest.raises(ValueError, match="report date"):
        build_lc_view(state, ctx(checked_on=None))
    with pytest.raises(ValueError, match="report date"):
        build_lc_view(make_state(kind="full"), ctx(checked_on=None))  # no report facts at all


def test_failure_and_band_read_labels_are_the_fixed_messages() -> None:
    no_pin = build_lc_view(make_state(code="NO_PINCODE_ON_PAN", model=None, safety=None), ctx())
    assert no_pin.label == MSG_NO_PINCODE
    k7 = build_lc_view(make_state(code="OPENAI_FAILURE", model=None, safety=None), ctx())
    assert k7.label == MSG_OPENAI_FAILURE
    pnv = build_lc_view(make_state(status="pan_not_verified"), ctx())
    assert pnv.label == MSG_PAN_NOT_VERIFIED


# --------------------------------------------------------------------------- confidence / headline / low-confidence nulls


def test_high_confidence_full_check_is_fundable_and_shows_the_picks() -> None:
    state = make_state(
        kind="full",
        report=report_facts(),
        code_confidence="high",
        model=mo("L1", ("L2",), confidence="high"),
        existing_emi_declared=None,
    )
    view = build_lc_view(state, ctx(left=3))
    assert (view.headline, view.confidence) == ("fundable", "high")
    assert view.top_lender is not None and view.top_lender.id == "L1"
    assert view.quick_checks_left_today is None  # full never carries a quota counter, whatever the context says
    assert_clean(view)


def test_a_model_that_raises_confidence_changes_nothing() -> None:
    state = make_state(code_confidence="medium", model=mo("L1", ("L2",), confidence="high"))
    view = build_lc_view(state, ctx())
    assert (view.confidence, view.headline) == ("medium", "possible")


def test_a_model_that_lowers_confidence_to_low_hides_the_picks() -> None:
    state = make_state(code_confidence="medium", model=mo("L1", ("L2",), confidence="low"))
    view = build_lc_view(state, ctx())
    assert view.confidence == "low"
    assert view.top_lender is None and view.alternates == []
    assert view.headline == "under_review"  # row i: a surviving top at final low on a scored band
    assert_clean(view)


def test_gate_empty_is_not_fundable_alone_with_a_forced_coapplicant_and_no_lenders() -> None:
    state = make_state(gated=[], numbers={}, model=None, code_confidence="low")
    view = build_lc_view(state, ctx())
    assert (view.status, view.code) == ("ok", None)
    assert (view.headline, view.confidence) == ("not_fundable_alone", "low")
    assert view.top_lender is None and view.alternates == []
    assert view.coapplicant_suggested is True and view.coapplicant_reason is None
    assert view.label == LABEL_QUICK
    assert_clean(view)


def test_every_pick_demoted_for_the_band_floor_is_not_fundable_alone() -> None:
    state = make_state(band="650-720", gated=[gl("L3")], model=mo("L3"))  # L3's floor bucket (720-780) is above the band
    view = build_lc_view(state, ctx())
    assert (view.headline, view.confidence) == ("not_fundable_alone", "low")
    assert view.top_lender is None and view.coapplicant_suggested is True


def test_picks_outside_the_gate_are_under_review_and_do_not_force_a_coapplicant() -> None:
    state = make_state(model=mo("L_FAKE", ("L_GHOST",), co_applicant_suggested=False))
    view = build_lc_view(state, ctx())
    assert (view.headline, view.confidence) == ("under_review", "low")
    assert view.top_lender is None and view.alternates == []
    assert view.coapplicant_suggested is False


def test_a_demoted_recommendation_promotes_the_first_surviving_alternate_with_its_own_prose() -> None:
    state = make_state(
        band="650-720",
        gated=[gl("L1"), gl("L2"), gl("L3")],
        model=mo("L3", ("L1", "L2")),
    )
    view = build_lc_view(state, ctx())
    top = view.top_lender
    assert top is not None and top.id == "L1" and top.rationale == "why L1"
    assert [a.id for a in view.alternates] == ["L2"]
    assert (view.headline, view.confidence) == ("possible", "medium")


def test_ntc_with_a_surviving_ntc_lender_is_unlikely_with_a_coapplicant_and_picks_shown() -> None:
    state = make_state(
        band="NTC",
        income=30000.0,
        gated=[gl("L1"), gl("L2")],
        model=mo(
            "L2",
            ("L1",),
            co_applicant_suggested=True,
            co_applicant_reason="A co-applicant with a credit history would strengthen this case",
        ),
    )
    view = build_lc_view(state, ctx())
    assert (view.headline, view.confidence) == ("unlikely", "medium")
    assert view.coapplicant_suggested is True
    assert view.coapplicant_reason == "A co-applicant with a credit history would strengthen this case"
    top = view.top_lender
    assert top is not None and top.id == "L2"
    assert top.ceiling == 395000 and top.range is not None and (top.range.lower, top.range.upper) == (147000, 395000)
    assert view.alternates == []  # L1 does not accept NTC: demoted by safety, never shown
    assert_clean(view)


def test_unlikely_forces_the_coapplicant_flag_even_when_the_model_said_no() -> None:
    state = make_state(band="NTC", income=30000.0, model=mo("L2", co_applicant_suggested=False))
    view = build_lc_view(state, ctx())
    assert view.headline == "unlikely"
    assert view.coapplicant_suggested is True and view.coapplicant_reason is None


def test_a_model_coapplicant_suggestion_passes_through_when_no_headline_forces_it() -> None:
    state = make_state(model=mo("L1", co_applicant_suggested=True, co_applicant_reason="Consider a co-applicant"))
    view = build_lc_view(state, ctx())
    assert view.headline == "possible"
    assert view.coapplicant_suggested is True and view.coapplicant_reason == "Consider a co-applicant"


def test_the_coapplicant_reason_is_only_shown_when_a_coapplicant_is_suggested() -> None:
    state = make_state(model=mo("L1", co_applicant_suggested=False, co_applicant_reason="Consider a co-applicant"))
    view = build_lc_view(state, ctx())
    assert view.coapplicant_suggested is False and view.coapplicant_reason is None


def test_a_leaky_coapplicant_reason_is_dropped_whole() -> None:
    for leaky in ("Experian shows a thin file", "Score is 736 so add a co-applicant", "a/c no: SYNTH0001 is open"):
        state = make_state(model=mo("L1", co_applicant_suggested=True, co_applicant_reason=leaky))
        view = build_lc_view(state, ctx())
        assert view.coapplicant_suggested is True and view.coapplicant_reason is None


# --------------------------------------------------------------------------- LLM-prose scrub on the picks


def test_rationale_that_leaks_becomes_empty_and_conditions_drop_only_the_leaking_lines() -> None:
    gated = [gl("L1"), gl("L2")]
    model = ModelOutput(
        recommendation=Pick(
            lender_id="L1",
            rank=1,
            rationale="Score is 736 so this lender fits",
            conditions=["Confirm income proof", "Bureaus differ on this learner", "Keep EMI under 3,500", "account number 123456789012"],
        ),
        alternates=[Pick(lender_id="L2", rank=2, rationale="Experian says fine", conditions=[])],
        confidence="medium",
        co_applicant_suggested=False,
        co_applicant_reason=None,
        review_flags=[],
    )
    view = build_lc_view(make_state(gated=gated, model=model), ctx())
    top = view.top_lender
    assert top is not None
    assert top.rationale == ""
    assert top.conditions == ["Confirm income proof", "Keep EMI under 3,500"]
    assert view.alternates[0].rationale == "" and view.alternates[0].conditions == []
    assert_clean(view)


def test_clean_prose_is_passed_through_verbatim() -> None:
    text = "Fits the learner's declared income and the lender's published floor"
    model = ModelOutput(
        recommendation=Pick(lender_id="L1", rank=1, rationale=text, conditions=["Confirm declared income, 45,000 per month"]),
        alternates=[],
        confidence="medium",
        co_applicant_suggested=False,
        co_applicant_reason=None,
        review_flags=[],
    )
    view = build_lc_view(make_state(model=model), ctx())
    top = view.top_lender
    assert top is not None and top.rationale == text
    assert top.conditions == ["Confirm declared income, 45,000 per month"]


def test_pick_rank_and_review_flags_never_reach_the_view() -> None:
    model = ModelOutput(
        recommendation=pick("L1", 1),
        alternates=[],
        confidence="medium",
        co_applicant_suggested=False,
        co_applicant_reason=None,
        review_flags=["internal note: raise for review"],
    )
    dumped = build_lc_view(make_state(model=model), ctx()).model_dump(mode="json")
    assert "rank" not in dumped["top_lender"]
    assert "review_flags" not in dumped
    assert all("internal note" not in text for _, text in leaves(dumped))


# --------------------------------------------------------------------------- admin strings: leak-guarded (global tier)


@pytest.mark.parametrize(
    "name",
    ["Experian Finance Ltd", "CIBIL Credit", "equifax capital", "CRIF Lending", "Alpha 736 Capital", "Lender a/c no: X1"],
)
def test_a_leaky_lender_name_falls_back_to_the_lender_id(name: str) -> None:
    state = make_state(gated=[gl("L1", name=name), gl("L2")])
    view = build_lc_view(state, ctx())
    assert view.top_lender is not None
    assert view.top_lender.name == "L1"
    assert view.alternates[0].name == "Bharat Upskill Credit"
    assert_clean(view)


def test_the_generic_word_bureau_is_allowed_in_an_admin_name_on_the_names_only_tier() -> None:
    view = build_lc_view(make_state(gated=[gl("L1", name="Bureau Credit Ltd"), gl("L2")]), ctx())
    assert view.top_lender is not None and view.top_lender.name == "Bureau Credit Ltd"


@pytest.mark.parametrize("label", ["736% special", "Experian rate", "300 to 500%", "11 to 14% a/c no: 1"])
def test_a_leaky_rate_band_becomes_empty_string(label: str) -> None:
    view = build_lc_view(make_state(gated=[gl("L1", rate_band_label=label), gl("L2")]), ctx())
    assert view.top_lender is not None and view.top_lender.rate_band == ""
    assert view.alternates[0].rate_band == "14 to 18%"


def test_a_leaky_tenure_range_becomes_empty_string() -> None:
    view = build_lc_view(make_state(gated=[gl("L1", tenure_label="12 to 360 months"), gl("L2")]), ctx())
    assert view.top_lender is not None and view.top_lender.tenure_range == ""
    assert view.alternates[0].tenure_range == "12 to 36 months"


def test_clean_admin_strings_pass_through_unchanged() -> None:
    view = build_lc_view(make_state(), ctx())
    assert view.top_lender is not None
    assert view.top_lender.rate_band == SEED_RATE_BANDS["L1"]
    assert view.top_lender.tenure_range == SEED_TENURES["L1"]
    assert view.top_lender.name == SEED_NAMES["L1"]


def test_a_survivor_that_is_not_in_the_gate_is_skipped_not_shown() -> None:
    survivor = SafetyResult(
        top=pick("L9"),
        alternates=[pick("L2", 2)],
        demoted=[],
        none_survived=False,
        review=False,
        top_clears_floor_by_one=True,
    )
    view = build_lc_view(make_state(safety=survivor), ctx())
    assert view.top_lender is None
    assert [a.id for a in view.alternates] == ["L2"]


def test_alternates_are_capped_at_two_even_for_an_unvalidated_safety_result() -> None:
    gated = [gl("L1"), gl("L2"), gl("L3")]
    wide = SafetyResult(
        top=pick("L1"),
        alternates=[pick("L2", 2), pick("L3", 3), pick("L1", 4)],
        demoted=[],
        none_survived=False,
        review=False,
        top_clears_floor_by_one=True,
    )
    view = build_lc_view(make_state(gated=gated, safety=wide, band="780+"), ctx())
    assert [a.id for a in view.alternates] == ["L2", "L3"]


# --------------------------------------------------------------------------- items_to_confirm


def test_items_to_confirm_keep_the_mandated_stale_account_copy_and_drop_leaking_lines() -> None:
    state = make_state(
        confirm_items=(
            "An account last updated 2026-03-12",
            "past tenure, confirm with learner",
            "Experian shows 736 on one account",
            None,
            "confirm with learner: an account last updated 2026-05-20",
            "account number 123456789012",
        )
    )
    view = build_lc_view(state, ctx())
    assert view.items_to_confirm == [
        "An account last updated 2026-03-12",
        "past tenure, confirm with learner",
        "confirm with learner: an account last updated 2026-05-20",
    ]
    assert_clean(view)


def test_items_to_confirm_are_empty_on_every_non_ok_branch() -> None:
    items = ("An account last updated 2026-03-12",)
    for over in (
        {"code": "NO_PINCODE_ON_PAN", "model": None, "safety": None},
        {"code": "OPENAI_FAILURE", "model": None, "safety": None},
    ):
        assert build_lc_view(make_state(confirm_items=items, **over), ctx()).items_to_confirm == []
    assert build_lc_view(make_state(confirm_items=items, status="pan_not_verified"), ctx()).items_to_confirm == []


def test_items_to_confirm_are_still_shown_when_confidence_is_low() -> None:
    state = make_state(
        gated=[], numbers={}, model=None, code_confidence="low", confirm_items=("An account last updated 2026-03-12",)
    )
    assert build_lc_view(state, ctx()).items_to_confirm == ["An account last updated 2026-03-12"]


# --------------------------------------------------------------------------- aadhaar warning truth table


@pytest.mark.parametrize(
    ("linked", "expected"),
    [(False, "PAN not linked to Aadhaar; lenders will ask for this"), (True, None), (None, None)],
)
def test_aadhaar_warning_truth_table(linked: bool | None, expected: str | None) -> None:
    view = build_lc_view(make_state(identity=identity(aadhaar_linked=linked)), ctx())
    assert view.pan_aadhaar_link_warning == expected
    if expected is not None:
        assert view.pan_aadhaar_link_warning == AADHAAR_WARNING


def test_aadhaar_warning_is_none_without_an_identity() -> None:
    assert build_lc_view(make_state(identity=None), ctx()).pan_aadhaar_link_warning is None


def test_aadhaar_warning_rides_on_the_band_read_and_k7_views_too() -> None:
    not_linked = identity(aadhaar_linked=False)
    for over in ({"code": "NO_PINCODE_ON_PAN"}, {"code": "OPENAI_FAILURE"}):
        view = build_lc_view(make_state(identity=not_linked, model=None, safety=None, **over), ctx())
        assert view.pan_aadhaar_link_warning == AADHAAR_WARNING


def test_an_unverified_pan_never_carries_the_aadhaar_warning() -> None:
    view = build_lc_view(make_state(identity=identity(aadhaar_linked=False, pan_valid=False)), ctx())
    assert view.status == "pan_not_verified" and view.pan_aadhaar_link_warning is None


# --------------------------------------------------------------------------- context pass-through and quota counter


def test_context_dates_and_cache_flag_pass_through_as_iso_strings() -> None:
    view = build_lc_view(make_state(), ctx(checked_on=date(2026, 9, 1), from_cache=True, refresh=date(2026, 10, 1)))
    assert (view.checked_on, view.from_cache, view.refresh_available_on) == ("2026-09-01", True, "2026-10-01")
    bare = build_lc_view(make_state(), ctx(checked_on=None, refresh=None))
    assert (bare.checked_on, bare.refresh_available_on) == (None, None)


def test_quick_checks_left_only_rides_on_quick_views() -> None:
    assert build_lc_view(make_state(), ctx(left=2)).quick_checks_left_today == 2
    assert build_lc_view(make_state(), ctx(left=0)).quick_checks_left_today == 0
    assert build_lc_view(make_state(), ctx(left=None)).quick_checks_left_today is None
    full = make_state(kind="full", report=report_facts())
    assert build_lc_view(full, ctx(left=2)).quick_checks_left_today is None


# --------------------------------------------------------------------------- band-read view (NO_PINCODE_ON_PAN)


@pytest.mark.parametrize(
    ("band", "headline", "coapplicant"),
    [
        ("NTC", "unlikely", True),
        ("<650", "unlikely", True),
        ("650-720", "possible", False),
        ("720-780", "possible", False),
        ("780+", "possible", False),
    ],
)
def test_no_pincode_view_is_a_band_read_with_null_confidence(band: Band, headline: Headline, coapplicant: bool) -> None:
    state = make_state(band=band, code="NO_PINCODE_ON_PAN", gated=[], numbers={}, model=None, safety=None)
    view = build_lc_view(state, ctx())
    assert (view.status, view.code) == ("ok", "NO_PINCODE_ON_PAN")
    assert view.label == MSG_NO_PINCODE
    assert view.headline == headline and view.coapplicant_suggested is coapplicant
    assert view.confidence is None  # K15
    assert view.top_lender is None and view.alternates == []
    assert_clean(view)


def test_no_pincode_ignores_any_model_output_or_safety_in_the_state() -> None:
    state = make_state(code="NO_PINCODE_ON_PAN")  # base state still carries a model output + a safety result
    view = build_lc_view(state, ctx())
    assert view.top_lender is None and view.alternates == [] and view.confidence is None
    assert view.headline == "possible"
    assert view.coapplicant_reason is None


def test_no_pincode_band_falls_back_to_the_score_facts_when_derive_did_not_run() -> None:
    state = make_state(code="NO_PINCODE_ON_PAN", model=None, safety=None, derived=None, score=score_facts("<650"))
    assert build_lc_view(state, ctx()).headline == "unlikely"


def test_band_fallback_prefers_the_newer_of_score_and_report() -> None:
    newer_report = make_state(
        code="NO_PINCODE_ON_PAN",
        kind="full",
        model=None,
        safety=None,
        derived=None,
        score=score_facts("720-780", pulled_on=date(2026, 9, 1)),
        report=report_facts("<650", report_date=date(2026, 10, 1)),
    )
    assert build_lc_view(newer_report, ctx(left=None)).headline == "unlikely"
    newer_score = make_state(
        code="NO_PINCODE_ON_PAN",
        kind="full",
        model=None,
        safety=None,
        derived=None,
        score=score_facts("720-780", pulled_on=date(2026, 10, 5)),
        report=report_facts("<650", report_date=date(2026, 10, 1)),
    )
    assert build_lc_view(newer_score, ctx(left=None)).headline == "possible"
    only_report = make_state(
        code="NO_PINCODE_ON_PAN", kind="full", model=None, safety=None, derived=None, score=None, report=report_facts("NTC")
    )
    assert build_lc_view(only_report, ctx(left=None)).headline == "unlikely"


def test_no_band_anywhere_on_a_band_read_is_under_review_not_a_guess() -> None:
    state = make_state(code="NO_PINCODE_ON_PAN", model=None, safety=None, derived=None, score=None)
    assert build_lc_view(state, ctx()).headline == "under_review"


# --------------------------------------------------------------------------- K7: the OPENAI_FAILURE view


def k7_state(band: Band = "720-780", gated: list[GatedLender] | None = None, **over: Any) -> CheckState:
    lenders = gated if gated is not None else [gl("L1"), gl("L2"), gl("L3")]
    return make_state(
        band=band, gated=lenders, code="OPENAI_FAILURE", model=None, safety=None, code_confidence="medium", **over
    )


def test_k7_view_shape() -> None:
    view = build_lc_view(k7_state(), ctx())
    assert (view.status, view.code) == ("service_failure", "OPENAI_FAILURE")
    assert view.label == MSG_OPENAI_FAILURE
    assert view.confidence == "low"
    assert view.headline == "possible"  # band-derived: never under_review on this path
    assert view.top_lender is None
    assert view.coapplicant_suggested is False
    assert (view.checked_on, view.from_cache, view.refresh_available_on) == ("2026-10-10", False, "2026-11-09")
    assert view.quick_checks_left_today == 4
    assert_clean(view)


def test_k7_alternates_are_code_built_and_number_free() -> None:
    view = build_lc_view(k7_state(), ctx())
    assert [a.id for a in view.alternates] == ["L1", "L2"]  # L3's floor (720) clears at 720-780, but id order caps at two
    for alt in view.alternates:
        assert alt.ceiling is None and alt.range is None
        assert alt.rationale == "Automatic ranking unavailable; this lender serves the learner's area"
        assert alt.rationale == RATIONALE_RANKING_UNAVAILABLE
        assert alt.conditions == []
    first = view.alternates[0]
    assert (first.name, first.rate_band, first.tenure_range) == ("Alpha Education Finance", "11 to 14%", "12 to 48 months")
    assert all(not any(ch.isdigit() for ch in a.rationale) for a in view.alternates)


@pytest.mark.parametrize(
    ("band", "expected"),
    [
        ("NTC", ["L2"]),  # only the NTC-accepting lender
        ("<650", []),  # every seed floor (650 / 650 / 720) is above this band
        ("650-720", ["L1", "L2"]),  # L3's floor bucket (720-780) is not cleared
        ("720-780", ["L1", "L2"]),  # all three clear, capped at two by id
        ("780+", ["L1", "L2"]),
    ],
)
def test_k7_alternates_are_the_gate_survivors_whose_floor_the_band_clears(band: Band, expected: list[str]) -> None:
    view = build_lc_view(k7_state(band), ctx())
    assert [a.id for a in view.alternates] == expected


def test_k7_alternates_are_ordered_by_id_whatever_the_gate_order_and_capped_at_two() -> None:
    gated = [gl("L3"), gl("L2"), gl("L1"), gl("L0", criteria={"score_floor": 300})]
    view = build_lc_view(k7_state("780+", gated), ctx())
    assert [a.id for a in view.alternates] == ["L0", "L1"]


def test_k7_a_lender_with_an_unknown_floor_clears_a_scored_band() -> None:
    gated = [gl("L1", criteria={"score_floor": None, "ntc_accepted": False})]
    assert [a.id for a in build_lc_view(k7_state("<650", gated), ctx()).alternates] == ["L1"]
    assert build_lc_view(k7_state("NTC", gated), ctx()).alternates == []  # NTC needs an explicit accept


@pytest.mark.parametrize("band", ALL_BANDS)
@pytest.mark.parametrize("floor", [None, 300, 600, 649, 650, 719, 720, 779, 780, 850])
@pytest.mark.parametrize("ntc_accepted", [True, False, None])
def test_k7_floor_predicate_matches_safety_rule_one(band: Band, floor: int | None, ntc_accepted: bool | None) -> None:
    lender = gl("L1", criteria={"score_floor": floor, "ntc_accepted": ntc_accepted})
    nums = numbers_for([lender])["L1"]
    breach = check_pick(pick("L1"), lender, nums, band=band, age_known=False, age_app_years=None)
    breached_on_floor = breach is not None and breach.rule == "band_floor"
    shown = [a.id for a in build_lc_view(k7_state(band, [lender]), ctx()).alternates]
    assert shown == ([] if breached_on_floor else ["L1"])


@pytest.mark.parametrize(("band", "headline"), [("NTC", "unlikely"), ("<650", "unlikely"), ("650-720", "possible"), ("720-780", "possible"), ("780+", "possible")])
def test_k7_headline_is_band_derived_for_every_band(band: Band, headline: Headline) -> None:
    view = build_lc_view(k7_state(band), ctx())
    assert view.headline == headline
    assert view.coapplicant_suggested is (headline == "unlikely")


def test_k7_status_in_state_may_be_unset_ok_or_service_failure() -> None:
    def view_with(status: str | None) -> LcView:
        over: dict[str, Any] = {} if status is None else {"status": status}
        return build_lc_view(k7_state(**over), ctx())

    views = [view_with(status) for status in (None, "ok", "service_failure")]
    assert views[0] == views[1] == views[2]
    assert views[0].status == "service_failure"


def test_k7_never_shows_model_prose_even_if_the_state_holds_a_model_output() -> None:
    state = k7_state()
    state["model_output"] = mo("L1")  # the rank node leaves None on failure; prove a stray value is still ignored
    view = build_lc_view(state, ctx())
    assert view.top_lender is None
    assert all(a.rationale == RATIONALE_RANKING_UNAVAILABLE for a in view.alternates)


def test_k7_leaky_admin_strings_fall_back_like_the_main_path() -> None:
    gated = [gl("L1", name="Experian Finance Ltd", rate_band_label="736% special"), gl("L2")]
    view = build_lc_view(k7_state(gated=gated), ctx())
    first = view.alternates[0]
    assert (first.name, first.rate_band) == ("L1", "")
    assert_clean(view)


def test_k7_full_kind_carries_no_quota_counter() -> None:
    state = k7_state(kind="full", report=report_facts())
    assert build_lc_view(state, ctx(left=3)).quick_checks_left_today is None


def test_k7_passes_the_true_wire_status_to_compute_headline(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> Headline | None:
        seen.append(kwargs)
        return compute_headline(**kwargs)

    monkeypatch.setattr(lc_view_module, "compute_headline", spy)
    build_lc_view(k7_state(), ctx())
    assert len(seen) == 1
    assert (seen[0]["status"], seen[0]["code"]) == ("service_failure", "OPENAI_FAILURE")


def test_the_headline_is_the_one_imported_from_safety_and_is_the_only_source(monkeypatch: pytest.MonkeyPatch) -> None:
    assert lc_view_module.compute_headline is safety_module.compute_headline
    def forced(**_: object) -> Headline:
        return "under_review"

    monkeypatch.setattr(lc_view_module, "compute_headline", forced)
    assert build_lc_view(make_state(), ctx()).headline == "under_review"


# --------------------------------------------------------------------------- K15: branches before final_confidence


def test_band_read_and_pan_not_verified_never_compute_final_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> Confidence | None:
        raise AssertionError("final_confidence must not run on this path")

    monkeypatch.setattr(lc_view_module, "final_confidence", boom)
    assert build_lc_view(make_state(code="NO_PINCODE_ON_PAN", model=None, safety=None), ctx()).confidence is None
    assert build_lc_view(make_state(status="pan_not_verified"), ctx()).confidence is None
    assert build_lc_view(make_state(identity=identity(pan_valid=False)), ctx()).confidence is None
    assert build_lc_view(k7_state(), ctx()).confidence == "low"


# --------------------------------------------------------------------------- pan_not_verified


@pytest.mark.parametrize("how", ["status", "identity", "both"])
def test_pan_not_verified_view(how: str) -> None:
    over: dict[str, Any] = {}
    if how in ("status", "both"):
        over["status"] = "pan_not_verified"
    if how in ("identity", "both"):
        over["identity"] = identity(pan_valid=False)
    view = build_lc_view(make_state(**over), ctx(left=5, from_cache=True))
    assert (view.status, view.code) == ("pan_not_verified", None)
    assert view.label == MSG_PAN_NOT_VERIFIED
    assert view.headline is None and view.confidence is None
    assert view.top_lender is None and view.alternates == []
    assert view.coapplicant_suggested is False and view.coapplicant_reason is None
    assert view.items_to_confirm == []
    assert (view.checked_on, view.from_cache, view.refresh_available_on) == (None, False, None)  # nothing was cached
    assert view.quick_checks_left_today == 5  # not counted: the context value is unchanged
    assert_clean(view)


def test_pan_not_verified_full_view_has_no_quota_counter() -> None:
    state = make_state(kind="full", status="pan_not_verified")
    assert build_lc_view(state, ctx(left=5)).quick_checks_left_today is None


# --------------------------------------------------------------------------- unhandled states fail closed


@pytest.mark.parametrize(
    "over",
    [
        {"status": "refused", "code": "QUOTA_EXHAUSTED"},
        {"status": "pending_verification"},
        {"status": "service_failure", "code": "SUREPASS_FAILURE"},
        {"status": "service_failure"},
        {"status": "ok", "code": "WARN_NTC_RECENT"},
        {"status": "ok", "code": "SUREPASS_FAILURE"},
        {"status": "pan_not_verified", "code": "NO_PINCODE_ON_PAN"},
    ],
)
def test_a_state_this_builder_does_not_own_becomes_the_internal_error_view(over: dict[str, Any]) -> None:
    view = build_lc_view(make_state(**over), ctx())
    assert view == build_error_lc_view(check_id="chk-1", kind="quick")


@pytest.mark.parametrize("missing", ["check_id", "kind"])
def test_a_state_without_check_id_or_kind_cannot_be_built(missing: str) -> None:
    state = make_state()
    cast("dict[str, Any]", state).pop(missing)
    with pytest.raises(ValueError, match=missing):
        build_lc_view(state, ctx())


def test_build_lc_view_does_not_mutate_the_state() -> None:
    state = make_state(confirm_items=("An account last updated 2026-03-12",))
    before = copy.deepcopy(state)
    build_lc_view(state, ctx())
    assert state == before


# --------------------------------------------------------------------------- build_refusal_lc_view

REFUSAL_CODES: tuple[Code, ...] = (
    "CONSENT_MISSING",
    "QUOTA_EXHAUSTED",
    "PAN_DAILY_CAP",
    "STAGE_TOO_EARLY",
    "REPORT_WINDOW_OPEN",
    "IN_FLIGHT",
    "PAN_LOCKED",
    "WARN_NTC_RECENT",
    "PAN_CHANGE_NEEDS_MANAGER",
    "LIVE_CALLS_DISABLED",
)


@pytest.mark.parametrize("kind", ["quick", "full"])
@pytest.mark.parametrize("code", REFUSAL_CODES)
def test_refusal_view_per_code(code: Code, kind: Kind) -> None:
    view = build_refusal_lc_view(
        check_id="chk-9",
        kind=kind,
        code=code,
        ctx=ctx(left=0, checked_on=None, refresh=None),
        warn_date=date(2026, 10, 10) if code == "WARN_NTC_RECENT" else None,
    )
    expected = REFUSAL_MESSAGES[code]
    if code == "WARN_NTC_RECENT":
        expected = expected.format(date="2026-10-10")
    assert (view.check_id, view.kind, view.status, view.code) == ("chk-9", kind, "refused", code)
    assert view.label == expected
    assert view.headline is None and view.confidence is None
    assert view.top_lender is None and view.alternates == []
    assert view.coapplicant_suggested is False and view.coapplicant_reason is None
    assert view.pan_aadhaar_link_warning is None and view.items_to_confirm == []
    assert view.from_cache is False
    assert view.quick_checks_left_today == (0 if kind == "quick" else None)
    assert_clean(view)


def test_ntc_warning_label_is_the_formatted_prd_string() -> None:
    view = build_refusal_lc_view(
        check_id="c", kind="full", code="WARN_NTC_RECENT", ctx=ctx(left=None), warn_date=date(2026, 10, 10)
    )
    assert view.label == "No bureau file was found on 2026-10-10; a full report will most likely return nothing. Proceed?"


def test_ntc_warning_falls_back_to_the_context_pull_date_then_fails_loudly() -> None:
    from_ctx = build_refusal_lc_view(
        check_id="c", kind="full", code="WARN_NTC_RECENT", ctx=ctx(left=None, checked_on=date(2026, 10, 4))
    )
    assert "2026-10-04" in from_ctx.label
    with pytest.raises(ValueError, match="WARN_NTC_RECENT"):
        build_refusal_lc_view(check_id="c", kind="full", code="WARN_NTC_RECENT", ctx=ctx(left=None, checked_on=None))


@pytest.mark.parametrize("code", ["SUREPASS_FAILURE", "OPENAI_FAILURE", "NO_PINCODE_ON_PAN"])
def test_a_non_refusal_code_cannot_build_a_refusal_view(code: Code) -> None:
    with pytest.raises(ValueError, match=code):
        build_refusal_lc_view(check_id="c", kind="quick", code=code, ctx=ctx())


def test_refusal_carries_refresh_available_on_only_where_the_context_has_it() -> None:
    with_cache = build_refusal_lc_view(
        check_id="c",
        kind="full",
        code="REPORT_WINDOW_OPEN",
        ctx=ctx(left=None, checked_on=date(2026, 10, 10), refresh=date(2026, 11, 9)),
    )
    assert (with_cache.checked_on, with_cache.refresh_available_on) == ("2026-10-10", "2026-11-09")
    no_cache = build_refusal_lc_view(
        check_id="c", kind="quick", code="QUOTA_EXHAUSTED", ctx=ctx(left=0, checked_on=None, refresh=None)
    )
    assert (no_cache.checked_on, no_cache.refresh_available_on) == (None, None)


def test_refusal_quick_checks_left_only_on_quick() -> None:
    quick = build_refusal_lc_view(check_id="c", kind="quick", code="QUOTA_EXHAUSTED", ctx=ctx(left=0))
    full = build_refusal_lc_view(check_id="c", kind="full", code="STAGE_TOO_EARLY", ctx=ctx(left=3))
    assert quick.quick_checks_left_today == 0
    assert full.quick_checks_left_today is None
    assert build_refusal_lc_view(check_id="c", kind="quick", code="IN_FLIGHT", ctx=ctx(left=None)).quick_checks_left_today is None


def test_a_refusal_is_never_served_from_cache_even_if_the_context_says_so() -> None:
    view = build_refusal_lc_view(check_id="c", kind="full", code="REPORT_WINDOW_OPEN", ctx=ctx(left=None, from_cache=True))
    assert view.from_cache is False


# --------------------------------------------------------------------------- build_failure_lc_view


def test_pending_verification_view_carries_code_none_and_the_pending_label() -> None:
    view = build_failure_lc_view(
        check_id="chk-2",
        kind="quick",
        status="pending_verification",
        code=None,
        label=MSG_PENDING_VERIFICATION,
        quick_checks_left_today=3,
    )
    assert (view.status, view.code) == ("pending_verification", None)  # K1
    assert view.label == MSG_PENDING_VERIFICATION
    assert view.headline is None and view.confidence is None and view.top_lender is None and view.alternates == []
    assert view.quick_checks_left_today == 3
    assert (view.checked_on, view.from_cache, view.refresh_available_on) == (None, False, None)
    assert_clean(view)


def test_pending_verification_forces_code_none_even_if_a_stale_caller_passes_one() -> None:
    view = build_failure_lc_view(
        check_id="c",
        kind="quick",
        status="pending_verification",
        code="SUREPASS_FAILURE",
        label=MSG_PENDING_VERIFICATION,
        quick_checks_left_today=None,
    )
    assert view.code is None


@pytest.mark.parametrize("label", [MSG_PAN_SERVICE_FAILURE, MSG_BUREAU_SERVICE_FAILURE])
def test_surepass_service_failure_view(label: str) -> None:
    view = build_failure_lc_view(
        check_id="c", kind="full", status="service_failure", code="SUREPASS_FAILURE", label=label, quick_checks_left_today=2
    )
    assert (view.status, view.code, view.label) == ("service_failure", "SUREPASS_FAILURE", label)
    assert view.headline is None and view.confidence is None
    assert view.quick_checks_left_today is None  # full: never a counter
    assert_clean(view)


def test_service_failure_view_may_carry_no_code() -> None:
    view = build_failure_lc_view(
        check_id="c", kind="quick", status="service_failure", code=None, label=MSG_INTERNAL, quick_checks_left_today=None
    )
    assert (view.status, view.code, view.label) == ("service_failure", None, MSG_INTERNAL)


def test_failure_view_headline_is_none_even_for_openai_failure() -> None:
    view = build_failure_lc_view(
        check_id="c",
        kind="quick",
        status="service_failure",
        code="OPENAI_FAILURE",
        label=MSG_OPENAI_FAILURE,
        quick_checks_left_today=1,
    )
    assert view.headline is None and view.code == "OPENAI_FAILURE"


@pytest.mark.parametrize("label", ["Experian is down", "Try 736 again", "a/c no: 1 failed"])
def test_a_leaky_failure_label_is_replaced_by_the_internal_message(label: str) -> None:
    view = build_failure_lc_view(
        check_id="c", kind="quick", status="service_failure", code=None, label=label, quick_checks_left_today=None
    )
    assert view.label == MSG_INTERNAL


def test_the_generic_bureau_word_is_allowed_in_a_failure_label_on_the_global_tier() -> None:
    view = build_failure_lc_view(
        check_id="c", kind="quick", status="service_failure", code=None, label="bureau check failed", quick_checks_left_today=None
    )
    assert view.label == "bureau check failed"


# --------------------------------------------------------------------------- build_error_lc_view


@pytest.mark.parametrize("kind", ["quick", "full"])
def test_error_view(kind: Literal["quick", "full"]) -> None:
    view = build_error_lc_view(check_id="chk-3", kind=kind)
    assert (view.check_id, view.kind, view.status, view.code) == ("chk-3", kind, "service_failure", None)
    assert view.label == MSG_INTERNAL
    assert view.headline is None and view.confidence is None
    assert view.top_lender is None and view.alternates == []
    assert view.coapplicant_suggested is False and view.coapplicant_reason is None
    assert view.pan_aadhaar_link_warning is None and view.items_to_confirm == []
    assert (view.checked_on, view.from_cache, view.refresh_available_on, view.quick_checks_left_today) == (
        None,
        False,
        None,
        None,
    )
    assert_clean(view)


# --------------------------------------------------------------------------- end to end: every view survives the scan


def test_every_view_shape_passes_the_scan_and_round_trips() -> None:
    views: list[LcView] = [
        build_lc_view(make_state(), ctx()),
        build_lc_view(make_state(kind="full", report=report_facts(), code_confidence="high", model=mo("L1", confidence="high")), ctx(left=None)),
        build_lc_view(make_state(gated=[], numbers={}, model=None, code_confidence="low"), ctx()),
        build_lc_view(make_state(code="NO_PINCODE_ON_PAN", model=None, safety=None), ctx()),
        build_lc_view(k7_state(), ctx()),
        build_lc_view(make_state(status="pan_not_verified"), ctx()),
        build_error_lc_view(check_id="c", kind="quick"),
    ]
    views += [
        build_refusal_lc_view(
            check_id="c", kind="full", code=code, ctx=ctx(left=None), warn_date=date(2026, 10, 10)
        )
        for code in REFUSAL_CODES
    ]
    for view in views:
        assert_clean(view)


def test_safety_breach_detail_numbers_never_appear_in_any_view() -> None:
    # SafetyBreach.detail is internal-only and may carry numbers; nothing in the state's safety.demoted can reach a view.
    breach = SafetyBreach(lender_id="L3", rule="band_floor", detail="band 650-720 is below the lender floor 720 (score 745)")
    safety = SafetyResult(
        top=pick("L1"), alternates=[], demoted=[breach], none_survived=False, review=False, top_clears_floor_by_one=True
    )
    view = build_lc_view(make_state(safety=safety), ctx())
    dumped = str(view.model_dump(mode="json"))
    assert "745" not in dumped and "below the lender floor" not in dumped
