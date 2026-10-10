"""Task 10 (part 2): testing/fake_llm.py (FakeRankModel + scripted misbehaviour) and the structural NEVER-list.

The fake implements the SAME protocol as the production seam (``RankModel.__call__(ctx, *, config=None)``), records every
context and config it receives, and can be scripted to misbehave in exactly the ways the node and the safety fence must
survive: a pick outside the gate, a pick below the lender's floor, a RAISED confidence (ignored by ``min``), and invalid
output on both attempts (the node maps it to OPENAI_FAILURE after exactly two calls).

The second half proves the NEVER-list (raw PAN, mobile, learner name, pincodes, score number, bureau name, account
identifiers, raw envelopes, storage keys) is structurally impossible in ``RankContext``: no field exists to carry it, and
a context built from a poisoned state is independent of every poisoned value.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from datetime import date
from typing import Any, Final, cast, get_args

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from rank_builders import BUREAU_NAMES, POISON, gated_of, gl, make_state, num, numbers_of, open_account
from rank_builders import identity as make_identity

from eligibility_agent.graph.llm import LenderContext, RankContext, RankModel, build_rank_context
from eligibility_agent.graph.state import (
    AdmissionDecision,
    Band,
    CheckState,
    Confidence,
    ConsentInfo,
    Kind,
    ModelOutput,
    ScoreFacts,
    SurePassCallAudit,
)
from eligibility_agent.lenders.safety import final_confidence, run_safety
from eligibility_agent.output.lc_view import scrub_lines
from eligibility_agent.testing.fake_llm import (
    FakeRankModel,
    RankCall,
    below_floor_pick,
    default_ranking,
    invalid_output_error,
    lowered_confidence,
    out_of_gate_pick,
    raised_confidence,
    ranked_output,
)

# --------------------------------------------------------------------------- worlds


def ctx_for(**kwargs: Any) -> RankContext:
    return build_rank_context(make_state(**kwargs))


def mixed_world(band: Band = "720-780") -> RankContext:
    """L1/L4/L5/L6 clear the floor with headroom; L2's floor (800) is above the band; L3 has no EMI headroom."""
    lenders = [gl("L1"), gl("L2", score_floor=800), gl("L3"), gl("L4"), gl("L5"), gl("L6")]
    numbers = {g.id: num(g.id) for g in lenders}
    numbers["L3"] = num("L3", available_emi=0.0, not_fundable_alone=True, ceiling=None, loan_max_lower=None, loan_max_upper=None)
    return ctx_for(band=band, lenders=lenders, numbers=numbers)


# --------------------------------------------------------------------------- protocol parity


def _shape(fn: Callable[..., Any]) -> list[tuple[str, Any, Any, Any]]:
    sig = inspect.signature(fn, eval_str=True)
    return [(p.name, p.kind, p.default, p.annotation) for p in sig.parameters.values()]


def test_fake_signature_matches_the_protocol_including_the_keyword_only_config() -> None:
    proto = inspect.signature(RankModel.__call__, eval_str=True)
    fake = inspect.signature(FakeRankModel.__call__, eval_str=True)
    assert _shape(RankModel.__call__) == _shape(FakeRankModel.__call__)
    assert proto.return_annotation == fake.return_annotation is ModelOutput
    config = fake.parameters["config"]
    assert config.kind is inspect.Parameter.KEYWORD_ONLY
    assert config.default is None
    assert list(fake.parameters) == ["self", "ctx", "config"]


def test_fake_is_assignable_to_the_protocol() -> None:
    model: RankModel = FakeRankModel()  # the assignment itself is the pyright check
    assert isinstance(model(ctx_for()), ModelOutput)


# --------------------------------------------------------------------------- default behaviour


def test_default_ranks_lenders_that_clear_the_floor_and_have_headroom_in_id_order() -> None:
    out = FakeRankModel()(mixed_world())
    assert out.recommendation is not None
    assert out.recommendation.lender_id == "L1"
    assert [p.lender_id for p in out.alternates] == ["L4", "L5"]  # L2 below floor, L3 no headroom, L6 beyond two alternates
    assert [out.recommendation.rank, *(p.rank for p in out.alternates)] == [1, 2, 3]
    assert out.review_flags == []


@pytest.mark.parametrize("code_confidence", ["high", "medium", "low"])
def test_default_never_raises_confidence_it_repeats_the_given_value(code_confidence: Confidence) -> None:
    assert FakeRankModel()(ctx_for(code_confidence=code_confidence)).confidence == code_confidence


def test_default_with_nothing_eligible_recommends_nothing_and_suggests_a_co_applicant() -> None:
    ctx = ctx_for(lenders=[gl("L2", score_floor=800)])
    out = FakeRankModel()(ctx)
    assert out.recommendation is None
    assert out.alternates == []
    assert out.co_applicant_suggested is True
    assert out.co_applicant_reason is not None


def test_default_with_no_lenders_at_all_is_total() -> None:
    out = FakeRankModel()(ctx_for(lenders=[], numbers={}))
    assert out.recommendation is None and out.alternates == []


def test_default_suggests_a_co_applicant_for_thin_bands_only() -> None:
    thin = FakeRankModel()(ctx_for(band="NTC", lenders=[gl("L1", ntc_accepted=True)]))
    assert thin.recommendation is not None and thin.recommendation.lender_id == "L1"
    assert thin.co_applicant_suggested is True and thin.co_applicant_reason is not None
    strong = FakeRankModel()(ctx_for(band="780+"))
    assert strong.co_applicant_suggested is False and strong.co_applicant_reason is None


def test_default_prose_is_number_free_and_survives_the_output_boundary_unchanged() -> None:
    for band in ("NTC", "<650", "650-720", "720-780", "780+"):
        out = FakeRankModel()(ctx_for(band=band, lenders=[gl("L1", ntc_accepted=True, score_floor=None)]))
        lines = [p.rationale for p in (out.recommendation, *out.alternates) if p is not None]
        lines += [c for p in (out.recommendation, *out.alternates) if p is not None for c in p.conditions]
        if out.co_applicant_reason is not None:
            lines.append(out.co_applicant_reason)
        assert lines, band
        assert scrub_lines(lines) == lines, f"default prose would be scrubbed at band {band}"
        assert not any(ch.isdigit() for line in lines for ch in line)


def test_default_survives_the_safety_fence_on_a_healthy_world() -> None:
    state = make_state(lenders=[gl("L1"), gl("L2")])
    ctx = build_rank_context(state)
    out = FakeRankModel()(ctx)
    result = run_safety(out, gated_of(state), numbers_of(state), band="720-780", age_app_years=32, age_known=True)
    assert result.top is not None and result.top.lender_id == "L1"
    assert result.demoted == []
    assert not result.review


# --------------------------------------------------------------------------- recording and scripting


def test_fake_records_every_context_and_config_it_receives() -> None:
    fake = FakeRankModel()
    first, second = ctx_for(), ctx_for(code_confidence="low")
    cfg: RunnableConfig = {"tags": ["tier:quick"], "metadata": {"check_id": "c-1"}}
    fake(first, config=cfg)
    fake(second)
    assert fake.call_count == 2
    assert fake.contexts == [first, second]
    assert fake.configs[0] is cfg
    assert fake.configs[1] is None
    assert fake.calls[0] == RankCall(ctx=first, config=cfg)


def test_script_steps_are_consumed_in_order_and_the_last_one_repeats() -> None:
    a, b = ranked_output("L1"), ranked_output("L2")
    fake = FakeRankModel(script=[a, b])
    ctx = ctx_for()
    assert [fake(ctx) for _ in range(4)] == [a, b, b, b]


def test_script_exceptions_are_raised_and_the_call_is_still_recorded() -> None:
    boom = TimeoutError("simulated 8s timeout")
    fake = FakeRankModel(script=[boom, ranked_output("L1")])
    ctx = ctx_for()
    with pytest.raises(TimeoutError, match="simulated"):
        fake(ctx)
    assert fake.call_count == 1 and fake.contexts == [ctx]
    assert fake(ctx).recommendation is not None  # the retry succeeds
    assert fake.call_count == 2


def test_script_callables_receive_the_context() -> None:
    seen: list[RankContext] = []

    def responder(ctx: RankContext) -> ModelOutput:
        seen.append(ctx)
        return ranked_output(ctx.lenders[-1].lender_id)

    ctx = ctx_for(lenders=[gl("L1"), gl("L2"), gl("L7")])
    out = FakeRankModel(script=[responder])(ctx)
    assert seen == [ctx]
    assert out.recommendation is not None and out.recommendation.lender_id == "L7"


def test_ranked_output_builds_a_valid_model_output() -> None:
    out = ranked_output("L1", "L2", "L3", confidence="low", co_applicant_suggested=True, co_applicant_reason="why", review_flags=["f"])
    assert out.recommendation is not None and out.recommendation.lender_id == "L1"
    assert [p.lender_id for p in out.alternates] == ["L2", "L3"]
    assert out.confidence == "low" and out.co_applicant_suggested and out.co_applicant_reason == "why"
    assert out.review_flags == ["f"]
    empty = ranked_output()
    assert empty.recommendation is None and empty.alternates == []
    with pytest.raises(ValidationError):
        ranked_output("L1", "L2", "L3", "L4")  # the schema allows at most two alternates


# --------------------------------------------------------------------------- scripted misbehaviour


def test_out_of_gate_pick_is_demoted_by_safety_and_none_survive() -> None:
    state = make_state()
    out = FakeRankModel(script=[out_of_gate_pick("L_FAKE", "L3")])(build_rank_context(state))
    assert out.recommendation is not None and out.recommendation.lender_id == "L_FAKE"
    assert [p.lender_id for p in out.alternates] == ["L3"]
    result = run_safety(out, gated_of(state), numbers_of(state), band="720-780", age_app_years=32, age_known=True)
    assert result.none_survived and result.review
    assert [(d.lender_id, d.rule) for d in result.demoted] == [("L_FAKE", "not_in_gate"), ("L3", "not_in_gate")]


def test_out_of_gate_pick_defaults_to_an_invented_lender() -> None:
    out = out_of_gate_pick()
    assert out.recommendation is not None and out.recommendation.lender_id == "L_NOT_IN_GATE"


@pytest.mark.parametrize("band", ["NTC", "<650", "650-720"])
def test_below_floor_pick_picks_only_lenders_whose_floor_the_band_does_not_clear(band: Band) -> None:
    lenders = [gl("L1", score_floor=720, ntc_accepted=False), gl("L2", score_floor=900, ntc_accepted=False)]
    state = make_state(band=band, lenders=lenders)
    ctx = build_rank_context(state)
    assert all(lender.band_clears_floor is False for lender in ctx.lenders)
    out = FakeRankModel(script=[below_floor_pick()])(ctx)
    picked = [p.lender_id for p in (out.recommendation, *out.alternates) if p is not None]
    assert picked == ["L1", "L2"]
    result = run_safety(out, gated_of(state), numbers_of(state), band=band, age_app_years=32, age_known=True)
    assert result.none_survived and result.review
    assert {d.rule for d in result.demoted} == {"band_floor"}


def test_below_floor_pick_skips_lenders_that_do_clear_the_floor() -> None:
    ctx = mixed_world()
    out = below_floor_pick()(ctx)
    assert [p.lender_id for p in (out.recommendation, *out.alternates) if p is not None] == ["L2"]


FAKE_LLM_PY: Final[Path] = Path(__file__).resolve().parent.parent / "eligibility_agent" / "testing" / "fake_llm.py"


def test_the_anti_vacuity_guards_are_raises_not_asserts_so_they_survive_python_dash_o() -> None:
    # ``python -O`` strips ``assert`` statements: a guard written as one would silently stop guarding.
    tree = ast.parse(FAKE_LLM_PY.read_text())
    assert [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)] == []


def test_below_floor_pick_without_a_below_floor_lender_is_a_setup_error() -> None:
    with pytest.raises(AssertionError, match="below"):
        below_floor_pick()(ctx_for())


@pytest.mark.parametrize("code_confidence", ["low", "medium"])
def test_raised_confidence_returns_a_higher_value_and_final_confidence_ignores_it(code_confidence: Confidence) -> None:
    ctx = ctx_for(code_confidence=code_confidence)
    out = FakeRankModel(script=[raised_confidence()])(ctx)
    assert out.confidence == "high"
    assert out.recommendation is not None  # otherwise the default ranking: only the confidence misbehaves
    assert final_confidence(code_confidence, out, safety=None, code=None) == code_confidence


def test_raised_confidence_cannot_raise_above_high() -> None:
    with pytest.raises(AssertionError, match="raise"):
        raised_confidence()(ctx_for(code_confidence="high"))


def test_raised_confidence_honours_an_explicit_level() -> None:
    assert raised_confidence("medium")(ctx_for(code_confidence="low")).confidence == "medium"
    with pytest.raises(AssertionError):
        raised_confidence("low")(ctx_for(code_confidence="low"))


def test_lowered_confidence_is_applied_by_min() -> None:
    ctx = ctx_for(code_confidence="medium")
    out = lowered_confidence()(ctx)
    assert out.confidence == "low"
    assert final_confidence("medium", out, safety=None, code=None) == "low"
    with pytest.raises(AssertionError, match="lower"):
        lowered_confidence("high")(ctx)


def test_invalid_output_error_is_a_real_pydantic_validation_error() -> None:
    err = invalid_output_error()
    assert isinstance(err, ValidationError)
    assert "ModelOutput" in str(err)


def test_invalid_twice_raises_a_validation_error_on_every_call() -> None:
    fake = FakeRankModel.invalid_twice()
    ctx = ctx_for()
    for _ in range(2):
        with pytest.raises(ValidationError):
            fake(ctx)
    assert fake.call_count == 2
    with pytest.raises(ValidationError):  # a (buggy) third call is still invalid, and still counted
        fake(ctx)
    assert fake.call_count == 3


def test_a_retry_that_succeeds_is_scriptable() -> None:
    fake = FakeRankModel(script=[invalid_output_error(), default_ranking])
    ctx = ctx_for()
    with pytest.raises(ValidationError):
        fake(ctx)
    assert fake(ctx).recommendation is not None


# --------------------------------------------------------------------------- the NEVER-list is structural

_RANK_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "kind",
        "band",
        "code_confidence",
        "foir",
        "emi_floor_flag",
        "existing_emi_reported",
        "existing_emi_estimated",
        "emi_past_tenure_accounts",
        "emi_stale_accounts",
        "declared_income",
        "declared_emi",
        "aadhaar_linked",
        "recent_enquiries_180d",
        "past_due_flag",
        "stale_account_flag",
        "lenders",
        "attempt",
        "prior_error",
    }
)
_LENDER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "lender_id",
        "ceiling",
        "range_lower",
        "range_upper",
        "not_fundable_alone",
        "age_at_application",
        "age_at_maturity",
        "band_clears_floor",
        "ntc_accepted",
        "foir_limit",
        "min_income",
        "tenure_label",
        "rate_band_label",
        "policy_notes",
        "uncertain_fields",
    }
)
# The ONLY fields that can hold arbitrary text, with what feeds each (none of it is learner data):
#   lender_id (the lender's id), tenure_label / rate_band_label (built from numeric criteria), policy_notes (the admin's
#   published note), uncertain_fields (criteria FIELD NAMES), prior_error (the model call's own exception text).
_TEXT_FIELDS: Final[frozenset[str]] = frozenset(
    {"lender_id", "tenure_label", "rate_band_label", "policy_notes", "uncertain_fields", "prior_error"}
)
_NEVER_TOKENS: Final[frozenset[str]] = frozenset(
    {"pan", "mobile", "phone", "name", "pincode", "pincodes", "pin", "zip", "dob", "score", "bureau", "raw", "envelope", "storage", "hash", "key", "ip"}
)


def _flatten(annotation: Any) -> set[Any]:
    """Every leaf of an annotation: ``str | None`` -> {str, NoneType}; ``list[str]`` -> {str}; ``Literal['a']`` -> {'a'}."""
    args = get_args(annotation)
    if not args:
        return {annotation}
    leaves: set[Any] = set()
    for arg in args:
        leaves |= _flatten(arg)
    return leaves


def test_context_field_names_are_exactly_the_9_2_payload() -> None:
    assert set(RankContext.model_fields) == _RANK_FIELDS
    assert set(LenderContext.model_fields) == _LENDER_FIELDS


def test_no_field_name_can_carry_the_never_list() -> None:
    for model in (RankContext, LenderContext):
        for name in model.model_fields:
            assert not (set(name.split("_")) & _NEVER_TOKENS), f"{model.__name__}.{name} names a NEVER-list item"


def test_only_six_fields_can_hold_free_text_and_everything_else_is_a_literal_or_a_number() -> None:
    allowed_non_text = {int, float, bool, type(None), LenderContext}
    carriers: set[str] = set()
    for model in (RankContext, LenderContext):
        for name, info in model.model_fields.items():
            leaves = _flatten(info.annotation)
            if str in leaves:
                carriers.add(name)
                continue
            # Literal values are str INSTANCES (never the str type); every other leaf is a numeric / bool / model type.
            assert all(leaf in allowed_non_text or isinstance(leaf, str) for leaf in leaves), f"{model.__name__}.{name}"
    assert carriers == _TEXT_FIELDS


def test_contexts_are_frozen_and_reject_undeclared_fields() -> None:
    for model in (RankContext, LenderContext):
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("extra") == "forbid"
    ctx = ctx_for()
    with pytest.raises(ValidationError):
        setattr(ctx, "band", "NTC")
    payload = ctx.model_dump()
    for smuggled in ("pan", "name", "mobile", "pincodes", "score", "raw_report"):
        with pytest.raises(ValidationError):
            RankContext.model_validate({**payload, smuggled: "x"})
    lender_payload = ctx.lenders[0].model_dump()
    for smuggled in ("name", "matched_pincodes", "score_floor"):
        with pytest.raises(ValidationError):
            LenderContext.model_validate({**lender_payload, smuggled: "x"})


@pytest.mark.parametrize(
    ("rank_attempts", "rank_error"), [(0, None), (1, "Request timed out.")], ids=["first-attempt", "retry-attempt"]
)
def test_a_context_built_from_a_poisoned_state_carries_no_poison_anywhere(rank_attempts: int, rank_error: str | None) -> None:
    state = make_state(rank_attempts=rank_attempts, rank_error=rank_error)
    identity_json = make_identity().model_dump_json()  # the date of birth is a date object in repr(), a string in JSON
    for literal in POISON:  # the fixture really is poisoned: each literal sits in the state the builder reads from
        assert literal in repr(state) or literal in identity_json, literal
    lowered_state = repr(state).lower()
    for name in BUREAU_NAMES:  # every bureau name sits in a state field the builder must not read
        assert name in lowered_state, name
    for key in ("gate_warnings", "surepass_calls_made", "admission", "errors"):
        assert state.get(key), key
    ctx = build_rank_context(state)
    dumped = ctx.model_dump_json()
    for literal in POISON:
        assert literal not in dumped, literal
        assert literal not in repr(ctx), literal
        assert literal not in str(ctx), literal
    for name in BUREAU_NAMES:  # case-insensitive
        assert name not in dumped.lower() and name not in repr(ctx).lower(), name
    # and the allowed facts did arrive
    assert ctx.band == "720-780" and ctx.declared_income == 60000.0 and [lender.lender_id for lender in ctx.lenders] == ["L1", "L2"]


def _replace(state: CheckState, **changes: Any) -> CheckState:
    return cast(CheckState, {**state, **changes})


@pytest.mark.parametrize("kind", ["quick", "full"])
def test_the_context_is_independent_of_every_poisoned_state_field(kind: Kind) -> None:
    state = make_state(kind=kind)
    baseline = build_rank_context(state)
    other = "zzzz-other"
    poisoned_variants: dict[str, CheckState] = {
        "request ids and hashes": _replace(
            state,
            check_id=other, lc_id=other, lead_id=other, opportunity_id=other,
            pan_hash=other, mobile_hash=other, pan_last4="0000", mobile_last4="1111",
        ),
        "raw pan / name / mobile": _replace(state, pan="AAAAA0000A", name="Somebody Else", mobile="9111111111"),
        "identity pii": _replace(
            state,
            identity=make_identity(full_name="Somebody Else", dob=date(1980, 1, 2), pan_pincode="682301", surepass_client_id=other),
        ),
        "score number": _replace(
            state, score=ScoreFacts(band="720-780", score=811, is_ntc=False, pulled_on=date(2026, 9, 1), source="report")
        ),
        "candidate pincodes": _replace(state, candidate_pincodes=["682301", "682302"]),
        "consent": _replace(state, consent=ConsentInfo(given=True, wording_version="v9", call_ref="other", ip="198.51.100.9")),
        "lender names and pincodes": _replace(
            state,
            gated_lenders=[g.model_copy(update={"name": "Another Name Ltd", "matched_pincodes": ["682301"]}) for g in gated_of(state)],
        ),
        "bookkeeping": _replace(state, errors=["x"], transcript=["admission", "derive"], bureau_called=True, pan_called=True),
        "gate warnings": _replace(state, gate_warnings=["lender other excluded: invalid criteria loan_min"]),
        "surepass call audit": _replace(
            state,
            surepass_calls_made=[
                SurePassCallAudit(endpoint="/api/v1/pan/pan-comprehensive", http_status=500, success=False, message_code="err", client_id="other", billed=False)
            ],
        ),
        "admission decision": _replace(state, admission=AdmissionDecision(admission="cache_hit", original_consent_id="other-consent")),
    }
    if kind == "full":
        report = state.get("report")
        assert report is not None
        poisoned_variants["report score, accounts and pincodes"] = _replace(
            state,
            report=report.model_copy(
                update={
                    "score": 811,
                    "bureau_pincodes_12m": ["682301"],
                    "open_accounts": [open_account(7, pincodes=["682302"], scheduled_emi=999.0)],
                }
            ),
        )
    for label, variant in poisoned_variants.items():
        assert build_rank_context(variant) == baseline, f"the context depends on: {label}"
