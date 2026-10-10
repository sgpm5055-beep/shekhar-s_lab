"""Task 1: graph/state.py is the Literal + wire-model registry (pure leaf module)."""

from __future__ import annotations

import ast
import dataclasses
import operator
import re
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

import pytest
from pydantic import BaseModel, ValidationError

from eligibility_agent.graph import state as state_module
from eligibility_agent.graph.state import (
    BAND_ORDER,
    CONFIDENCE_ORDER,
    PINCODE_RE_STR,
    AdmissionDecision,
    AdmissionMode,
    Band,
    CheckState,
    Code,
    ConsentCovers,
    ConsentInfo,
    Derived,
    EmiDerivation,
    EmiItem,
    ErrorKind,
    FieldStatus,
    FullAdmissionInputs,
    GatedLender,
    Headline,
    Identity,
    Kind,
    LcStatus,
    LcView,
    LenderCriteria,
    LenderNumbers,
    LenderView,
    LoanRange,
    LockKind,
    ModelOutput,
    OpenAccount,
    PanEventKind,
    Pick,
    PullPlan,
    ReportFacts,
    ReviewKind,
    SafetyBreach,
    SafetyResult,
    ScoreFacts,
    ScoreSource,
    StateUpdate,
    SurePassCallAudit,
)

# --------------------------------------------------------------------------- builders


def _consent() -> ConsentInfo:
    return ConsentInfo(given=True, wording_version="v1", call_ref="call-1", ip="10.0.0.1")


def _identity() -> Identity:
    return Identity(
        pan_valid=True,
        status="valid",
        category="person",
        full_name="Asha Rao",
        dob=date(1990, 5, 17),
        aadhaar_linked=True,
        pan_pincode="560001",
        surepass_client_id="pan_comprehensive_abc",
    )


def _score() -> ScoreFacts:
    return ScoreFacts(
        band="720-780",
        score=745,
        is_ntc=False,
        pulled_on=date(2026, 10, 10),
        source="score",
    )


def _open_account() -> OpenAccount:
    return OpenAccount(
        account_key="acct_0",
        portfolio_type="I",
        account_type="05",
        scheduled_emi=12500.0,
        current_balance=250000.0,
        tenure_months=36,
        rate_of_interest=11.5,
        open_date=date(2025, 1, 1),
        date_reported=date(2026, 9, 30),
        amount_past_due=0.0,
        pincodes=["560001"],
        is_stale=False,
    )


def _report() -> ReportFacts:
    return ReportFacts(
        band="780+",
        score=790,
        is_ntc=False,
        report_date=date(2026, 10, 10),
        open_accounts=[_open_account()],
        past_due_total=0.0,
        written_off_or_settled=0,
        enquiries_30d=1,
        enquiries_90d=2,
        enquiries_180d=3,
        bureau_pincodes_12m=["560001"],
    )


def _emi_item() -> EmiItem:
    return EmiItem(
        account_key="acct_0",
        kind="reported",
        monthly_amount=12500.0,
        method="reported",
        past_tenure=False,
        confirm_item=None,
    )


def _emi() -> EmiDerivation:
    return EmiDerivation(
        reported_total=12500.0,
        estimated_total=0.0,
        derived_total=12500.0,
        declared=None,
        used=12500.0,
        used_source="derived",
        emi_floor_flag=False,
        items=[_emi_item()],
    )


def _criteria() -> LenderCriteria:
    return LenderCriteria(
        score_floor=700,
        ntc_accepted=False,
        foir_limit=0.55,
        min_income=25000.0,
        min_age_years=21,
        max_age_at_maturity_years=65,
        rate_min_pct=11.0,
        rate_max_pct=14.0,
        tenure_min_months=12,
        tenure_max_months=60,
        loan_min=100000.0,
        loan_max=500000.0,
        policy_notes=None,
        field_status={"score_floor": "known", "min_income": "missing", "foir_limit": "uncertain"},
    )


def _gated() -> GatedLender:
    return GatedLender(
        id="L1",
        name="Lender One",
        criteria=_criteria(),
        matched_pincodes=["560001"],
        rate_band_label="11 to 14%",
        tenure_label="12 to 60 months",
    )


def _derived() -> Derived:
    return Derived(
        band="720-780",
        emi=_emi(),
        current_foir=0.15625,
        age_at_application_years=36,
        age_months_at_application=436,
        candidate_pincodes=["560001"],
    )


def _numbers() -> LenderNumbers:
    return LenderNumbers(
        lender_id="L1",
        available_emi=20000.0,
        loan_max_lower=222000.0,
        loan_max_upper=500000.0,
        loan_max_lower_raw=222749.10,
        loan_max_upper_raw=919860.68,
        ceiling=500000.0,
        displayed_foir=0.3606,
        age_at_maturity_years=41,
        not_fundable_alone=False,
        clipped="upper",
    )


def _pick(rank: int = 1, lender_id: str = "L1") -> Pick:
    return Pick(
        lender_id=lender_id,
        rank=rank,
        rationale="Clears the band floor.",
        conditions=["confirm income"],
    )


def _model_output_payload() -> dict[str, Any]:
    return {
        "recommendation": _pick().model_dump(mode="json"),
        "alternates": [_pick(2, "L2").model_dump(mode="json")],
        "confidence": "medium",
        "co_applicant_suggested": False,
        "co_applicant_reason": None,
        "review_flags": [],
    }


def _model_output() -> ModelOutput:
    return ModelOutput.model_validate(_model_output_payload())


def _breach() -> SafetyBreach:
    return SafetyBreach(lender_id="L2", rule="band_floor", detail="band 650-720 < floor 700")


def _safety() -> SafetyResult:
    return SafetyResult(
        top=_pick(),
        alternates=[_pick(2, "L3")],
        demoted=[_breach()],
        none_survived=False,
        review=False,
        top_clears_floor_by_one=True,
    )


def _audit() -> SurePassCallAudit:
    return SurePassCallAudit(
        endpoint="/api/v1/pan/pan-comprehensive",
        http_status=200,
        success=True,
        message_code=None,
        client_id="pan_comprehensive_abc",
        billed=True,
    )


def _loan_range() -> LoanRange:
    return LoanRange(lower=368000, upper=1277000)


def _lender_view() -> LenderView:
    return LenderView(
        id="L1",
        name="Lender One",
        rate_band="11 to 14%",
        tenure_range="12 to 60 months",
        ceiling=1277000,
        range=_loan_range(),
        rationale="Fits the band.",
        conditions=["confirm income"],
    )


def _lc_view() -> LcView:
    return LcView(
        check_id="chk-1",
        kind="quick",
        status="ok",
        code=None,
        headline="possible",
        label="Indicative, income as declared",
        confidence="medium",
        top_lender=_lender_view(),
        alternates=[_lender_view()],
        coapplicant_suggested=False,
        coapplicant_reason=None,
        pan_aadhaar_link_warning=None,
        items_to_confirm=["income"],
        checked_on="2026-10-10",
        from_cache=False,
        refresh_available_on="2026-11-09",
        quick_checks_left_today=4,
    )


def _admission_decision() -> AdmissionDecision:
    return AdmissionDecision()


# --------------------------------------------------------------------------- 1. round-trips

_BUILDERS: list[tuple[str, Any]] = [
    ("ConsentInfo", _consent),
    ("Identity", _identity),
    ("ScoreFacts", _score),
    ("OpenAccount", _open_account),
    ("ReportFacts", _report),
    ("EmiItem", _emi_item),
    ("EmiDerivation", _emi),
    ("LenderCriteria", _criteria),
    ("GatedLender", _gated),
    ("Derived", _derived),
    ("LenderNumbers", _numbers),
    ("Pick", _pick),
    ("ModelOutput", _model_output),
    ("SafetyBreach", _breach),
    ("SafetyResult", _safety),
    ("SurePassCallAudit", _audit),
    ("LoanRange", _loan_range),
    ("LenderView", _lender_view),
    ("LcView", _lc_view),
]


@pytest.mark.parametrize(("name", "builder"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
def test_model_json_round_trip(name: str, builder: Any) -> None:
    instance: BaseModel = builder()
    assert type(instance).__name__ == name
    restored = type(instance).model_validate(instance.model_dump(mode="json"))
    assert restored == instance


def test_frozen_fact_models_are_immutable() -> None:
    with pytest.raises(ValidationError):
        setattr(_score(), "band", "NTC")


def test_fact_models_forbid_unknown_fields() -> None:
    payload = _score().model_dump(mode="json")
    payload["unexpected"] = 1
    with pytest.raises(ValidationError):
        ScoreFacts.model_validate(payload)


# --------------------------------------------------------------------------- 2. ModelOutput / Pick


def test_model_output_rejects_injected_ceiling() -> None:
    payload = _model_output_payload()
    payload["ceiling"] = 450000
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(payload)


def test_model_output_rejects_more_than_two_alternates() -> None:
    payload = _model_output_payload()
    payload["alternates"] = [_pick(i, f"L{i}").model_dump(mode="json") for i in (2, 3, 4)]
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(payload)


def test_model_output_accepts_two_alternates_and_null_recommendation() -> None:
    payload = _model_output_payload()
    payload["recommendation"] = None
    payload["alternates"] = [_pick(i, f"L{i}").model_dump(mode="json") for i in (2, 3)]
    assert len(ModelOutput.model_validate(payload).alternates) == 2


def test_pick_rejects_extra_fields() -> None:
    payload = _pick().model_dump(mode="json")
    payload["ceiling"] = 450000
    with pytest.raises(ValidationError):
        Pick.model_validate(payload)


def _numeric_fields(model: type[BaseModel]) -> set[tuple[str, str, type]]:
    """Every ``(owning model, field, numeric type)`` reachable from ``model``: int / float / Decimal found anywhere in a
    field annotation, through unions, containers, ``Annotated`` and nested models. ``Literal`` values and ``bool`` are
    not numeric types (``bool`` is matched by identity, never by ``issubclass``)."""
    found: set[tuple[str, str, type]] = set()

    def walk(owner: str, field: str, annotation: object) -> None:
        origin = get_origin(annotation)
        if origin is Literal:
            return
        if origin is Annotated:
            walk(owner, field, get_args(annotation)[0])
            return
        if origin is not None:
            for arg in get_args(annotation):
                walk(owner, field, arg)
            return
        if not isinstance(annotation, type):
            return
        if issubclass(annotation, BaseModel):
            for name, info in annotation.model_fields.items():
                walk(annotation.__name__, name, info.annotation)
        elif annotation in (int, float, Decimal):
            found.add((owner, field, annotation))

    for name, info in model.model_fields.items():
        walk(model.__name__, name, info.annotation)
    return found


def test_the_only_numeric_field_anywhere_in_pick_or_model_output_is_rank() -> None:
    # Type-based: no money / score / amount field can creep in under ANY name. ``rank`` is ordering metadata.
    assert _numeric_fields(Pick) == {("Pick", "rank", int)}
    assert _numeric_fields(ModelOutput) == {("Pick", "rank", int)}  # recommendation + alternates both nest Pick


def test_the_numeric_field_walker_is_not_vacuous() -> None:
    class _Planted(BaseModel):
        ok_flag: bool
        nested: list[Pick]
        ceiling: Annotated[float, "money"] | None
        amounts: dict[str, Decimal]

    assert _numeric_fields(_Planted) == {
        ("Pick", "rank", int),
        ("_Planted", "ceiling", float),
        ("_Planted", "amounts", Decimal),
    }


def test_wire_model_field_names_are_the_frozen_client_contract() -> None:
    # PRD v1.1 section 6: these names ARE the client contract. Any rename / addition / removal must fail here first.
    assert set(LoanRange.model_fields.keys()) == {"lower", "upper"}
    assert set(LenderView.model_fields.keys()) == {
        "id",
        "name",
        "rate_band",
        "tenure_range",
        "ceiling",
        "range",
        "rationale",
        "conditions",
    }
    assert set(LcView.model_fields.keys()) == {
        "check_id",
        "kind",
        "status",
        "code",
        "headline",
        "label",
        "confidence",
        "top_lender",
        "alternates",
        "coapplicant_suggested",
        "coapplicant_reason",
        "pan_aadhaar_link_warning",
        "items_to_confirm",
        "checked_on",
        "from_cache",
        "refresh_available_on",
        "quick_checks_left_today",
    }
    assert set(Pick.model_fields.keys()) == {"lender_id", "rank", "rationale", "conditions"}
    assert set(ModelOutput.model_fields.keys()) == {
        "recommendation",
        "alternates",
        "confidence",
        "co_applicant_suggested",
        "co_applicant_reason",
        "review_flags",
    }


# --------------------------------------------------------------------------- 3. consent / identity / LcView


def test_consent_info_given_must_be_literally_true() -> None:
    payload = _consent().model_dump(mode="json")
    # Literal[True] alone would admit 1 and 1.0 (they equal True); the before-validator demands the bool itself.
    for bad in (False, 1, 1.0, 0, 0.0, "true", "True", "1", None):
        with pytest.raises(ValidationError):
            ConsentInfo.model_validate({**payload, "given": bad})
    assert ConsentInfo.model_validate({**payload, "given": True}).given is True


def test_identity_has_no_gender() -> None:
    assert not hasattr(Identity, "gender")
    assert not hasattr(_identity(), "gender")
    assert "gender" not in Identity.model_fields
    with pytest.raises(ValidationError):
        Identity.model_validate({**_identity().model_dump(mode="json"), "gender": "F"})


def test_lc_view_rejects_injected_score_field() -> None:
    payload = _lc_view().model_dump(mode="json")
    payload["score"] = 745
    with pytest.raises(ValidationError):
        LcView.model_validate(payload)


def test_lc_view_rejects_bureau_style_fields() -> None:
    payload = _lc_view().model_dump(mode="json")
    for key in ("score", "band", "enquiries_30d", "open_accounts", "raw_report"):
        with pytest.raises(ValidationError):
            LcView.model_validate({**payload, key: 1})


def test_lc_view_pan_aadhaar_link_warning_accepts_str_or_none() -> None:
    payload = _lc_view().model_dump(mode="json")
    warned = LcView.model_validate({**payload, "pan_aadhaar_link_warning": "PAN not linked to Aadhaar"})
    assert warned.pan_aadhaar_link_warning == "PAN not linked to Aadhaar"
    assert LcView.model_validate({**payload, "pan_aadhaar_link_warning": None}).pan_aadhaar_link_warning is None
    with pytest.raises(ValidationError):
        LcView.model_validate({**payload, "pan_aadhaar_link_warning": False})


def test_lc_view_defaults_and_alternates_cap() -> None:
    minimal = LcView(check_id="c", kind="full", status="refused", label="msg")
    assert minimal.code is None
    assert minimal.headline is None
    assert minimal.confidence is None
    assert minimal.top_lender is None
    assert minimal.alternates == []
    assert minimal.coapplicant_suggested is False
    assert minimal.coapplicant_reason is None
    assert minimal.pan_aadhaar_link_warning is None
    assert minimal.items_to_confirm == []
    assert minimal.checked_on is None
    assert minimal.from_cache is False
    assert minimal.refresh_available_on is None
    assert minimal.quick_checks_left_today is None
    payload = _lc_view().model_dump(mode="json")
    payload["alternates"] = [_lender_view().model_dump(mode="json")] * 3
    with pytest.raises(ValidationError):
        LcView.model_validate(payload)


def test_lc_view_status_is_lc_status_wire_enum() -> None:
    payload = _lc_view().model_dump(mode="json")
    with pytest.raises(ValidationError):
        LcView.model_validate({**payload, "status": "running"})
    with pytest.raises(ValidationError):
        LcView.model_validate({**payload, "status": "error"})


def test_lender_view_ceiling_and_range_may_be_none() -> None:
    view = LenderView(
        id="L1",
        name="Lender One",
        rate_band="",
        tenure_range="",
        ceiling=None,
        range=None,
        rationale="",
    )
    assert view.conditions == []
    assert LenderView.model_validate(view.model_dump(mode="json")) == view


# --------------------------------------------------------------------------- 4. AdmissionDecision / FullAdmissionInputs


def test_admission_decision_constructible_with_defaults() -> None:
    decision = _admission_decision()
    assert decision.admission is None
    assert decision.code is None
    assert decision.pan_plan == "reuse"
    assert decision.bureau_plan == "reuse"
    assert decision.counts_against_quota is False
    assert decision.is_refresh is False
    assert decision.quota_used_today is None
    assert decision.quota_limit is None
    assert decision.create_daily_cap_lock is False
    assert decision.review_reason is None
    assert decision.record_pan_correction is False
    assert decision.ntc_found_on is None
    assert decision.checked_on is None
    assert decision.refresh_available_on is None
    assert decision.original_consent_id is None


def test_admission_decision_is_frozen_slots_dataclass() -> None:
    decision = _admission_decision()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(decision, "admission", "new")
    assert not hasattr(decision, "__dict__")
    assert dataclasses.is_dataclass(decision)


def test_admission_decision_field_order_matches_contract() -> None:
    assert [f.name for f in dataclasses.fields(AdmissionDecision)] == [
        "admission",
        "code",
        "pan_plan",
        "bureau_plan",
        "counts_against_quota",
        "is_refresh",
        "quota_used_today",
        "quota_limit",
        "create_daily_cap_lock",
        "review_reason",
        "record_pan_correction",
        "ntc_found_on",
        "checked_on",
        "refresh_available_on",
        "original_consent_id",
    ]


def test_admission_decision_round_trips_via_asdict() -> None:
    decision = AdmissionDecision(
        admission="refresh",
        code=None,
        pan_plan="call",
        bureau_plan="call",
        counts_against_quota=False,
        is_refresh=True,
        quota_used_today=3,
        quota_limit=5,
        create_daily_cap_lock=False,
        review_reason="pan_daily_cap",
        record_pan_correction=True,
        ntc_found_on=date(2026, 10, 1),
        checked_on=date(2026, 9, 1),
        refresh_available_on=date(2026, 10, 1),
        original_consent_id="consent-1",
    )
    assert AdmissionDecision(**dataclasses.asdict(decision)) == decision
    assert AdmissionDecision(**dataclasses.asdict(_admission_decision())) == _admission_decision()


def test_full_admission_inputs_is_frozen_slots_dataclass() -> None:
    inputs = FullAdmissionInputs(refresh=False, acknowledge_ntc_warning=True)
    assert dataclasses.asdict(inputs) == {"refresh": False, "acknowledge_ntc_warning": True}
    assert FullAdmissionInputs(**dataclasses.asdict(inputs)) == inputs
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(inputs, "refresh", True)
    assert not hasattr(inputs, "__dict__")


# --------------------------------------------------------------------------- 5. literals


def test_code_has_exactly_the_thirteen_prd_values() -> None:
    assert set(get_args(Code)) == {
        "CONSENT_MISSING",
        "QUOTA_EXHAUSTED",
        "PAN_DAILY_CAP",
        "STAGE_TOO_EARLY",
        "REPORT_WINDOW_OPEN",
        "IN_FLIGHT",
        "PAN_LOCKED",
        "WARN_NTC_RECENT",
        "PAN_CHANGE_NEEDS_MANAGER",
        "NO_PINCODE_ON_PAN",
        "LIVE_CALLS_DISABLED",
        "SUREPASS_FAILURE",
        "OPENAI_FAILURE",
    }
    assert len(get_args(Code)) == 13


def test_lc_status_is_the_five_value_wire_enum() -> None:
    assert set(get_args(LcStatus)) == {
        "ok",
        "pan_not_verified",
        "refused",
        "pending_verification",
        "service_failure",
    }
    assert not hasattr(state_module, "CheckStatus")
    assert not hasattr(state_module, "Status")


def test_review_kind_values() -> None:
    values = set(get_args(ReviewKind))
    assert values == {
        "pan_daily_cap",
        "timeout_reconciliation",
        "openai_failure",
        "safety_demoted_all",
        "not_fundable_pattern",
        "pan_change_manager",
        "model_review_flags",
        "surepass_auth",
    }
    assert "surepass_auth" in values


def test_small_literals() -> None:
    assert set(get_args(ScoreSource)) == {"score", "report"}
    assert set(get_args(ConsentCovers)) == {"score", "report"}
    assert set(get_args(LockKind)) == {"in_flight", "timeout", "daily_cap"}
    assert set(get_args(ErrorKind)) == {
        "timeout",
        "transport",
        "auth",
        "rate_limited",
        "upstream",
        "protocol",
        "validation",
    }
    assert set(get_args(PanEventKind)) == {"initial", "correction", "manager_change"}
    assert set(get_args(PullPlan)) == {"reuse", "call"}
    assert set(get_args(AdmissionMode)) == {"cache_hit", "new", "refresh"}
    assert set(get_args(FieldStatus)) == {"known", "missing", "uncertain"}
    assert set(get_args(Kind)) == {"quick", "full"}
    assert set(get_args(Band)) == {"<650", "650-720", "720-780", "780+", "NTC"}
    assert set(get_args(Headline)) == {
        "fundable",
        "possible",
        "unlikely",
        "not_fundable_alone",
        "under_review",
    }


def test_band_and_confidence_order_are_exact() -> None:
    assert BAND_ORDER == {"NTC": 0, "<650": 1, "650-720": 2, "720-780": 3, "780+": 4}
    assert CONFIDENCE_ORDER == {"low": 0, "medium": 1, "high": 2}


def test_pincode_re_str() -> None:
    pattern = re.compile(PINCODE_RE_STR)
    assert pattern.fullmatch("560001")
    assert not pattern.fullmatch("060001")
    assert not pattern.fullmatch("20")
    assert not pattern.fullmatch("5600011")


def test_state_update_alias_is_a_dict() -> None:
    update: StateUpdate = {"transcript": ["admission"]}
    assert update["transcript"] == ["admission"]


# --------------------------------------------------------------------------- CheckState (typing-only asserts)


def test_check_state_is_total_false_with_exactly_the_registered_keys() -> None:
    # Set-equality (Task 1 deferral, closed in Task 11): a key cannot be added or dropped without this list changing.
    assert CheckState.__total__ is False
    hints = get_type_hints(CheckState, include_extras=True)
    expected = {
        "kind",
        "check_id",
        "lc_id",
        "lead_id",
        "opportunity_id",
        "opportunity_stage",
        "pan",
        "name",
        "mobile",
        "pan_hash",
        "mobile_hash",
        "pan_last4",
        "mobile_last4",
        "monthly_income",
        "existing_emi_declared",
        "refresh",
        "acknowledge_ntc_warning",
        "consent",
        "consent_id",
        "is_recompute",
        "admission",
        "identity",
        "score",
        "report",
        "derived",
        "candidate_pincodes",
        "gated_lenders",
        "gate_warnings",
        "lender_numbers",
        "code_confidence",
        "model_output",
        "rank_attempts",
        "rank_error",
        "safety",
        "status",
        "code",
        "headline",
        "final_confidence",
        "lc_view",
        "surepass_calls_made",
        "bureau_called",
        "pan_called",
        "billed",
        "counted_against_quota",
        "langsmith_run_id",
        "errors",
        "transcript",
    }
    assert set(hints) == expected
    assert "gender" not in hints


def test_check_state_amended_key_types() -> None:
    hints = get_type_hints(CheckState)
    assert hints["pan_last4"] is str
    assert hints["mobile_last4"] == (str | None)
    assert hints["lc_view"] == (LcView | None)
    assert hints["status"] == (LcStatus | None)
    assert hints["admission"] == (AdmissionDecision | None)
    assert hints["consent"] == (ConsentInfo | None)
    assert hints["rank_attempts"] is int
    assert hints["rank_error"] == (str | None)
    assert hints["consent_id"] == (str | None)  # this run's consent row (None on CONSENT_MISSING / recompute)
    assert hints["billed"] is bool  # written by persist / persist_refusal (policy.compute_billed)
    assert hints["counted_against_quota"] is bool  # written by persist (policy.compute_counted_against_quota)


def test_check_state_reducers_use_operator_add() -> None:
    hints = get_type_hints(CheckState, include_extras=True)
    for key in ("surepass_calls_made", "errors", "transcript"):
        assert operator.add in hints[key].__metadata__, key


def test_check_state_accepts_typed_partial_dict() -> None:
    state: CheckState = {
        "kind": "quick",
        "check_id": "chk-1",
        "pan_last4": "234F",
        "mobile_last4": None,
        "consent": _consent(),
        "admission": _admission_decision(),
        "lc_view": _lc_view(),
        "status": "ok",
        "surepass_calls_made": [_audit()],
        "errors": [],
        "transcript": ["admission"],
    }
    assert state["pan_last4"] == "234F"


# --------------------------------------------------------------------------- purity of the leaf


def test_state_module_is_a_pure_leaf() -> None:
    source = Path(state_module.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in leaf module"
            assert node.module is not None
            roots.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"pydantic"}
    assert roots <= allowed, roots - allowed
    assert "eligibility_agent" not in roots
