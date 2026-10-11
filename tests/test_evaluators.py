"""Task 15: the seven evals (PRD 12) over HAND-BUILT outputs dicts, plus a ``run_one`` smoke on two goldens.

Each deterministic evaluator is a plain function ``(inputs, outputs, reference_outputs) -> {"key", "score", "comment"}``
that judges the OUTPUTS contract only (the run_one dict: ``lc_view``, ``billed``, ``counted_against_quota``,
``from_cache``, ``admission``, ``surepass_calls`` [{endpoint, billable}], ``gated_lender_ids``, ``safety_survivor_ids``,
``band``, ``code_confidence``, ``derived``). The outputs below are written by hand here, never produced by the product,
so a mutant evaluator cannot pass by agreeing with the code under test. Expected money values are literals.

Binding rulings pinned here: the leak patterns are the IMPORTED ``output/lc_view`` ones (``check_id`` and lender ``id``
are skipped: uuid hex trips the score / account patterns structurally); ``billing_consistency`` is billable-based
(``billed == any(billable)``) and never depends on the ORDER of ``surepass_calls``; the numbers evaluator compares view
numbers with the code-derived table AND the golden's literal table; the LLM judge is gated on OPENAI_API_KEY.
"""

from __future__ import annotations

import copy
import itertools
import warnings
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pytest
from langsmith.schemas import Example, Run

from eligibility_agent.evals import evaluators
from eligibility_agent.evals.datasets import full_examples, inputs_of, quick_examples
from eligibility_agent.evals.evaluators import (
    DETERMINISTIC_EVALUATORS,
    JudgeVerdict,
    billing_consistency,
    confidence_not_raised,
    judge_evaluators,
    lenders_gated,
    make_llm_judge,
    no_leak,
    numbers_from_code,
    outcome_matches_band,
)
from eligibility_agent.evals.run import run_one
from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT

Dict = dict[str, Any]
Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

# A uuid4 whose hex trips the account / score patterns when scanned as prose ("9013" + "4562"-style digit runs).
UUID_LIKE: Final[str] = "736f0a12-9b84-4c17-8d4e-123456789012"
INPUTS: Final[Dict] = {
    "request": {"pan": "ABCPE1234F", "mobile": "9876543210", "name": "Ravindra K Typedname"},
}


def lender(lender_id: str = "L1", *, ceiling: int | None = 1277000, lower: int | None = 368000, **over: Any) -> Dict:
    out: Dict = {
        "id": lender_id,
        "name": "Alpha Education Finance",
        "rate_band": "11 to 14%",
        "tenure_range": "12 to 48 months",
        "ceiling": ceiling,
        "range": None if lower is None or ceiling is None else {"lower": lower, "upper": ceiling},
        "rationale": "Serves the learner's area and fits the lender's criteria for a strong credit profile.",
        "conditions": [],
    }
    out.update(over)
    return out


def view(**over: Any) -> Dict:
    out: Dict = {
        "check_id": UUID_LIKE,
        "kind": "quick",
        "status": "ok",
        "code": None,
        "headline": "possible",
        "label": "Indicative, based on what the learner told us",
        "confidence": "medium",
        "top_lender": lender("L1"),
        "alternates": [lender("L2", ceiling=500000, lower=295000)],
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-10",
        "from_cache": False,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": 4,
    }
    out.update(over)
    return out


def outputs_of(**over: Any) -> Dict:
    """The happy quick run, hand-built: L1 + L2 gated and surviving, derived numbers equal to the view's."""
    out: Dict = {
        "lc_view": view(),
        "billed": True,
        "counted_against_quota": True,
        "from_cache": False,
        "admission": "new",
        "surepass_calls": [
            {"endpoint": PAN_ENDPOINT, "billable": True},
            {"endpoint": SCORE_ENDPOINT, "billable": True},
        ],
        "gated_lender_ids": ["L1", "L2"],
        "safety_survivor_ids": ["L1", "L2"],
        "band": "720-780",
        "code_confidence": "medium",
        "derived": {
            "L1": {"ceiling": 1277000, "lower": 368000, "upper": 1277000},
            "L2": {"ceiling": 500000, "lower": 295000, "upper": 500000},
        },
    }
    out.update(over)
    return out


def run_eval(evaluator: Evaluator, outputs: Mapping[str, Any], reference: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    result = evaluator(INPUTS, outputs, reference or {})
    assert set(result) == {"key", "score", "comment"}, result
    return result


def passes(evaluator: Evaluator, outputs: Mapping[str, Any], reference: Mapping[str, Any] | None = None) -> bool:
    return bool(run_eval(evaluator, outputs, reference)["score"])


# --------------------------------------------------------------------------- the registry


def test_registry_holds_the_six_deterministic_evaluators_with_distinct_keys() -> None:
    assert [e.__name__ for e in DETERMINISTIC_EVALUATORS] == [
        "no_leak",
        "lenders_gated",
        "numbers_from_code",
        "confidence_not_raised",
        "billing_consistency",
        "outcome_matches_band",
    ]
    for evaluator in DETERMINISTIC_EVALUATORS:
        result = run_eval(evaluator, outputs_of())
        assert result["key"] == evaluator.__name__
        assert result["score"] is True, result


@pytest.mark.parametrize("evaluator", DETERMINISTIC_EVALUATORS, ids=lambda e: e.__name__)
def test_every_evaluator_is_accepted_by_langsmiths_own_wrapper(evaluator: Evaluator) -> None:
    """The name-matched kwargs form (inputs / outputs / reference_outputs) is what ``evaluate()`` introspects: a signature
    drift would surface as a ValueError from the real wrapper, so run the REAL wrapper over a Run / Example pair."""
    with warnings.catch_warnings():  # langsmith's own import trips an ast.Str DeprecationWarning (3.12); not ours
        warnings.simplefilter("ignore", DeprecationWarning)
        from langsmith.evaluation.evaluator import DynamicRunEvaluator
    run = Run(  # pyright: ignore[reportCallIssue]
        id="00000000-0000-4000-8000-000000000001",
        name="target",
        start_time=datetime(2026, 10, 10),
        run_type="chain",
        trace_id="00000000-0000-4000-8000-000000000001",
        dotted_order="x",
        inputs=INPUTS,
        outputs=outputs_of(),
    )
    example = Example(  # pyright: ignore[reportCallIssue]
        id="00000000-0000-4000-8000-000000000002",
        dataset_id="00000000-0000-4000-8000-000000000003",
        inputs=INPUTS,
        outputs={},
    )
    result = DynamicRunEvaluator(evaluator).evaluate_run(run, example)  # pyright: ignore[reportArgumentType]
    assert result.key == evaluator.__name__  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    assert result.score is True  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]


# --------------------------------------------------------------------------- no_leak (C11 import rule)


def test_no_leak_passes_a_clean_view_including_the_mandated_bureau_label() -> None:
    """The PRD-mandated label 'bureau report on <date>, income as declared' names no bureau: the global tier is
    names-only, so the generic word is legal there (and ONLY there)."""
    label = "bureau report on 2026-10-01, income as declared"
    assert passes(no_leak, outputs_of(lc_view=view(label=label, kind="full")))


@pytest.mark.parametrize(
    "plant",
    [
        pytest.param("Please note your account number 1234567890 appears here.", id="account-number-label"),
        pytest.param("The a/c no 98765 is overdue.", id="ac-no"),
        pytest.param("Linked to loan LN00123456789 on file.", id="bare-nine-digit-run"),
        pytest.param("Card ending xxxx1234 is active.", id="masked-id"),
    ],
)
def test_no_leak_catches_an_account_number_plant_in_any_prose_field(plant: str) -> None:
    for place in ("rationale", "conditions", "items_to_confirm", "coapplicant_reason"):
        carrier = view()
        if place == "rationale":
            carrier["top_lender"]["rationale"] = plant
        elif place == "conditions":
            carrier["alternates"][0]["conditions"] = ["ok", plant]
        elif place == "items_to_confirm":
            carrier["items_to_confirm"] = [plant]
        else:
            carrier["coapplicant_reason"] = plant
        result = run_eval(no_leak, outputs_of(lc_view=carrier))
        assert result["score"] is False, (place, result)
        assert place in str(result["comment"]) or "conditions" in str(result["comment"]), result


def test_no_leak_catches_bureau_names_everywhere_and_the_generic_term_only_in_llm_prose() -> None:
    name_in_label = view(label="Experian report on 2026-10-01, income as declared")
    assert not passes(no_leak, outputs_of(lc_view=name_in_label))  # a bureau NAME: the global tier
    assert not passes(no_leak, outputs_of(lc_view=view(label="CIBIL says no")))

    generic_in_rationale = view()
    generic_in_rationale["top_lender"]["rationale"] = "Your bureau history looks strong."
    assert not passes(no_leak, outputs_of(lc_view=generic_in_rationale))  # the generic word: the LLM tier

    generic_in_label = view(label="bureau report on 2026-10-01, income as declared")
    assert passes(no_leak, outputs_of(lc_view=generic_in_label))  # label-with-'bureau' PASS case (mandated copy)

    generic_in_reason = view(coapplicant_reason="A stronger bureau profile would help.")
    assert not passes(no_leak, outputs_of(lc_view=generic_in_reason))


@pytest.mark.parametrize("score_text", ["Indicative score 736 here", "Rs.736 floor"])
def test_no_leak_catches_a_standalone_score_anywhere(score_text: str) -> None:
    assert not passes(no_leak, outputs_of(lc_view=view(label=score_text)))
    carrier = view()
    carrier["alternates"][0]["conditions"] = [score_text]
    assert not passes(no_leak, outputs_of(lc_view=carrier))


def test_no_leak_skips_check_id_and_lender_ids_but_scans_every_other_leaf() -> None:
    """uuid4 hex (and a 9+ digit id) trips the score / account patterns structurally: those two fields are exempt, the
    same text anywhere else is not."""
    hostile = "lender-7360000001234"
    carrier = view(check_id="736f0a12-9b84-4c17-8d4e-123456789012")
    carrier["top_lender"]["id"] = hostile
    carrier["alternates"][0]["id"] = hostile
    assert passes(no_leak, outputs_of(lc_view=carrier))

    elsewhere = view()
    elsewhere["top_lender"]["name"] = hostile  # not an id field
    assert not passes(no_leak, outputs_of(lc_view=elsewhere))
    in_label = view(label=hostile)
    assert not passes(no_leak, outputs_of(lc_view=in_label))


def test_no_leak_catches_the_requests_own_identifiers_in_any_string() -> None:
    """The run's request PAN / mobile / typed name never appear in an LC-facing string (inputs carry the request)."""
    for literal in ("ABCPE1234F", "abcpe1234f", "9876543210", "Ravindra K Typedname"):
        carrier = view(label=f"Indicative for {literal}")
        assert not passes(no_leak, outputs_of(lc_view=carrier)), literal


def test_no_leak_comment_names_the_path_not_the_leaked_text() -> None:
    carrier = view()
    carrier["alternates"][0]["conditions"] = ["account number 1234567890"]
    result = run_eval(no_leak, outputs_of(lc_view=carrier))
    assert "alternates.0.conditions.0" in str(result["comment"])
    assert "1234567890" not in str(result["comment"])


def test_no_leak_imports_its_patterns_from_the_output_boundary_never_copies_them() -> None:
    from eligibility_agent.output import lc_view as boundary

    assert evaluators.GLOBAL_LEAK_PATTERNS is boundary.GLOBAL_LEAK_PATTERNS
    assert evaluators.LLM_LEAK_PATTERNS is boundary.LLM_LEAK_PATTERNS
    assert evaluators.line_leaks is boundary.line_leaks
    source = Path(evaluators.__file__).read_text()
    assert "re.compile" not in source  # no local regex copy of the scrub patterns


# --------------------------------------------------------------------------- lenders_gated


def test_lenders_gated_requires_named_within_survivors_within_gated() -> None:
    assert passes(lenders_gated, outputs_of())
    # a named lender that did not survive safety
    assert not passes(lenders_gated, outputs_of(safety_survivor_ids=["L1"]))
    # a survivor that never passed the pincode gate
    assert not passes(lenders_gated, outputs_of(gated_lender_ids=["L1"]))
    # a lender named in the view that was never in the gate at all
    carrier = view(alternates=[lender("L9", ceiling=1000, lower=1000)])
    assert not passes(lenders_gated, outputs_of(lc_view=carrier))


def test_lenders_gated_allows_a_view_that_names_nobody() -> None:
    nobody = view(top_lender=None, alternates=[], headline="not_fundable_alone", coapplicant_suggested=True)
    assert passes(lenders_gated, outputs_of(lc_view=nobody, safety_survivor_ids=[], gated_lender_ids=["L1", "L2"]))
    assert passes(lenders_gated, outputs_of(lc_view=nobody, safety_survivor_ids=[], gated_lender_ids=[]))


def test_lenders_gated_rejects_a_lender_named_twice() -> None:
    carrier = view(alternates=[lender("L1")])  # top is L1 too
    assert not passes(lenders_gated, outputs_of(lc_view=carrier))


def test_lenders_gated_compares_the_gate_to_the_golden_as_a_set() -> None:
    """The pincode-union golden: L3 (serves only the stale 400001) must NOT be gated; order is irrelevant."""
    assert passes(lenders_gated, outputs_of(gated_lender_ids=["L2", "L1"]), {"gated_lender_ids": ["L1", "L2"]})
    wrong = outputs_of(gated_lender_ids=["L1", "L2", "L3"])
    assert not passes(lenders_gated, wrong, {"gated_lender_ids": ["L1", "L2"]})


def test_lenders_gated_fails_closed_on_a_missing_chain() -> None:
    broken = outputs_of()
    del broken["safety_survivor_ids"]
    assert not passes(lenders_gated, broken)


# --------------------------------------------------------------------------- numbers_from_code


def test_numbers_from_code_passes_when_the_view_is_the_derived_table() -> None:
    assert passes(numbers_from_code, outputs_of())


def test_numbers_from_code_catches_a_ceiling_that_differs_from_the_derived_value() -> None:
    """450,000 shown vs 451,000 derived: the model (or a builder) altered a code number."""
    carrier = view(top_lender=lender("L1", ceiling=450000, lower=368000))
    derived = {
        "L1": {"ceiling": 451000, "lower": 368000, "upper": 451000},
        "L2": {"ceiling": 500000, "lower": 295000, "upper": 500000},
    }
    result = run_eval(numbers_from_code, outputs_of(lc_view=carrier, derived=derived))
    assert result["score"] is False
    assert "L1" in str(result["comment"])


def test_numbers_from_code_catches_a_range_that_differs_from_the_derived_pair() -> None:
    carrier = view(top_lender=lender("L1", ceiling=1277000, lower=370000))
    assert not passes(numbers_from_code, outputs_of(lc_view=carrier))


def test_numbers_from_code_no_income_means_no_numbers_anywhere() -> None:
    no_income = view(
        top_lender=lender("L1", ceiling=None, lower=None), alternates=[lender("L2", ceiling=None, lower=None)]
    )
    derived = {
        "L1": {"ceiling": None, "lower": None, "upper": None},
        "L2": {"ceiling": None, "lower": None, "upper": None},
    }
    assert passes(numbers_from_code, outputs_of(lc_view=no_income, derived=derived))
    # a number smuggled in where the code derived none
    smuggled = view(top_lender=lender("L1"), alternates=[lender("L2", ceiling=None, lower=None)])
    assert not passes(numbers_from_code, outputs_of(lc_view=smuggled, derived=derived))


def test_numbers_from_code_fails_a_lender_with_no_derived_row() -> None:
    assert not passes(numbers_from_code, outputs_of(derived={"L1": {"ceiling": 1277000, "lower": 368000, "upper": 1277000}}))


@pytest.mark.parametrize(
    "prose",
    [
        "You could borrow about 450000 at most.",
        "Aim for 1,277,000",  # grouped is still a figure: three digits in a row
        "Roughly 600 a month",
        "Keep the EMI under 12000.",
    ],
)
def test_numbers_from_code_flags_any_three_digit_figure_in_llm_prose(prose: str) -> None:
    for place in ("rationale", "conditions", "coapplicant_reason"):
        carrier = view()
        if place == "rationale":
            carrier["top_lender"]["rationale"] = prose
        elif place == "conditions":
            carrier["alternates"][0]["conditions"] = [prose]
        else:
            carrier["coapplicant_reason"] = prose
        assert not passes(numbers_from_code, outputs_of(lc_view=carrier)), (place, prose)


def test_numbers_from_code_allows_small_counts_in_prose() -> None:
    carrier = view()
    carrier["top_lender"]["rationale"] = "Repay within 5 years, ideally in 2 instalments of equal size."
    assert passes(numbers_from_code, outputs_of(lc_view=carrier))


def test_numbers_from_code_dates_in_code_built_confirm_items_are_not_prose() -> None:
    carrier = view(items_to_confirm=["confirm with learner: an account last updated 2026-06-01"])
    assert passes(numbers_from_code, outputs_of(lc_view=carrier))


GOLDEN_NUMBERS: Final[Dict] = {
    "lender_numbers": {
        "L1": {"ceiling": 1277000, "range": {"lower": 368000, "upper": 1277000}},
        "L2": {"ceiling": 500000, "range": {"lower": 295000, "upper": 500000}},
    }
}


def test_numbers_from_code_compares_the_view_with_the_goldens_literal_table() -> None:
    """The code-derived table agrees with the view here, but the golden's hand-written literal says 1,276,000: the
    evaluator compares against the golden, never only against numbers recomputed by the code under test."""
    assert passes(numbers_from_code, outputs_of(), GOLDEN_NUMBERS)
    drifted = copy.deepcopy(GOLDEN_NUMBERS)
    drifted["lender_numbers"]["L1"]["ceiling"] = 1276000
    drifted["lender_numbers"]["L1"]["range"]["upper"] = 1276000
    result = run_eval(numbers_from_code, outputs_of(), drifted)
    assert result["score"] is False
    assert "golden" in str(result["comment"])


def test_numbers_from_code_fails_a_named_lender_the_golden_does_not_allow() -> None:
    only_l1 = {"lender_numbers": {"L1": GOLDEN_NUMBERS["lender_numbers"]["L1"]}}
    assert not passes(numbers_from_code, outputs_of(), only_l1)  # L2 is named but absent from the golden's table


def test_numbers_from_code_k7_alternates_must_be_number_free() -> None:
    """K7: a model-failed view names up to two gate survivors with NO numbers (ceiling / range None) although the
    code derived numbers for them; any number there is a leak of a code-derived figure the model never validated."""
    k7 = view(
        status="service_failure",
        code="OPENAI_FAILURE",
        confidence="low",
        top_lender=None,
        alternates=[
            lender("L1", ceiling=None, lower=None, rationale="Automatic ranking unavailable; this lender serves the learner's area"),
            lender("L2", ceiling=None, lower=None, rationale="Automatic ranking unavailable; this lender serves the learner's area"),
        ],
    )
    assert passes(numbers_from_code, outputs_of(lc_view=k7))
    numbered = copy.deepcopy(k7)
    numbered["alternates"][0] = lender("L1")
    numbered["alternates"][0]["rationale"] = k7["alternates"][1]["rationale"]
    assert not passes(numbers_from_code, outputs_of(lc_view=numbered))
    digits = copy.deepcopy(k7)
    digits["alternates"][1]["rationale"] = "Automatic ranking unavailable (code 5000)"
    assert not passes(numbers_from_code, outputs_of(lc_view=digits))


# --------------------------------------------------------------------------- confidence_not_raised

ORDER: Final[tuple[str, ...]] = ("low", "medium", "high")


@pytest.mark.parametrize(("shown", "code"), list(itertools.product((None, *ORDER), ORDER)))
def test_confidence_not_raised_truth_table(shown: str | None, code: str) -> None:
    expected = shown is None or ORDER.index(shown) <= ORDER.index(code)
    carrier = view(confidence=shown)
    result = run_eval(confidence_not_raised, outputs_of(lc_view=carrier, code_confidence=code))
    assert result["score"] is expected, (shown, code, result)


def test_confidence_not_raised_catches_high_over_medium() -> None:
    result = run_eval(confidence_not_raised, outputs_of(lc_view=view(confidence="high"), code_confidence="medium"))
    assert result["score"] is False
    assert "high" in str(result["comment"]) and "medium" in str(result["comment"])


def test_confidence_not_raised_a_confidence_with_no_code_value_fails_closed() -> None:
    assert not passes(confidence_not_raised, outputs_of(lc_view=view(confidence="low"), code_confidence=None))
    assert passes(confidence_not_raised, outputs_of(lc_view=view(confidence=None), code_confidence=None))  # refusals


# --------------------------------------------------------------------------- billing_consistency (K17)

PAN_CALL: Final[Dict] = {"endpoint": PAN_ENDPOINT, "billable": True}
SCORE_CALL: Final[Dict] = {"endpoint": SCORE_ENDPOINT, "billable": True}
REPORT_CALL: Final[Dict] = {"endpoint": REPORT_ENDPOINT, "billable": True}


def _call(call: Dict, *, billable: bool) -> Dict:
    return {**call, "billable": billable}


@pytest.mark.parametrize(
    ("billables", "billed", "ok"),
    [
        pytest.param([True, True], True, True, id="both-billable"),
        pytest.param([True, False], True, True, id="q05-pan-billed-score-not"),
        pytest.param([False, True], True, True, id="order-is-irrelevant"),
        pytest.param([False, False], False, True, id="nothing-billable"),
        pytest.param([False, False], True, False, id="billed-with-no-billable-call"),
        pytest.param([True, False], False, False, id="billable-call-but-not-billed"),
        pytest.param([], True, False, id="billed-with-no-calls"),
        pytest.param([], False, True, id="no-calls-not-billed"),
    ],
)
def test_billed_equals_any_billable_truth_table(billables: list[bool], billed: bool, ok: bool) -> None:
    calls = [_call(PAN_CALL if i == 0 else SCORE_CALL, billable=b) for i, b in enumerate(billables)]
    outputs = outputs_of(
        surepass_calls=calls, billed=billed, counted_against_quota=False, admission="new", lc_view=view(from_cache=False)
    )
    assert passes(billing_consistency, outputs) is ok


def test_billing_consistency_is_independent_of_call_order() -> None:
    """T13 ruling: call order under a frozen clock is uuid-random; permutations must not change a verdict."""
    calls = [PAN_CALL, _call(SCORE_CALL, billable=False), REPORT_CALL]
    verdicts = {
        passes(billing_consistency, outputs_of(surepass_calls=list(p), counted_against_quota=False))
        for p in itertools.permutations(calls)
    }
    assert verdicts == {True}
    wrong = {
        passes(billing_consistency, outputs_of(surepass_calls=list(p), billed=False, counted_against_quota=False))
        for p in itertools.permutations(calls)
    }
    assert wrong == {False}


def test_a_cache_hit_may_not_call_a_bureau_endpoint() -> None:
    hit = outputs_of(
        surepass_calls=[], billed=False, counted_against_quota=False, from_cache=True, admission="cache_hit",
        lc_view=view(from_cache=True),
    )
    assert passes(billing_consistency, hit)
    for call in (SCORE_CALL, REPORT_CALL):
        bad = {**hit, "surepass_calls": [call], "billed": True}
        assert not passes(billing_consistency, bad), call
    # the PAN record can expire before the cached Score does: a billed PAN re-verify on a hit is legal
    pan_only = {**hit, "surepass_calls": [PAN_CALL], "billed": True}
    assert passes(billing_consistency, pan_only)


def test_a_refusal_makes_no_calls_and_bills_nothing() -> None:
    refused = view(status="refused", code="QUOTA_EXHAUSTED", headline=None, confidence=None, top_lender=None, alternates=[])
    clean = outputs_of(
        lc_view=refused, surepass_calls=[], billed=False, counted_against_quota=False, admission="refused"
    )
    assert passes(billing_consistency, clean)
    called = {**clean, "surepass_calls": [PAN_CALL], "billed": True}
    assert not passes(billing_consistency, called)


@pytest.mark.parametrize(
    ("admission", "counted", "billed", "ok"),
    [
        pytest.param("new", True, True, True, id="new-counted-billed"),
        pytest.param("new", False, True, True, id="new-not-counted"),
        pytest.param("refresh", False, True, True, id="refresh-not-counted"),
        pytest.param("refresh", True, True, False, id="refresh-counted"),
        pytest.param("cache_hit", True, False, False, id="hit-counted"),
        pytest.param("new", True, False, False, id="counted-but-not-billed"),
    ],
)
def test_counting_requires_a_billed_new_admission(admission: str, counted: bool, billed: bool, ok: bool) -> None:
    calls = [PAN_CALL, SCORE_CALL] if billed else []
    outputs = outputs_of(
        admission=admission, counted_against_quota=counted, billed=billed, surepass_calls=calls,
        from_cache=False, lc_view=view(from_cache=False),
    )
    assert passes(billing_consistency, outputs) is ok


def test_the_outputs_from_cache_must_agree_with_the_view() -> None:
    assert not passes(billing_consistency, outputs_of(from_cache=True, lc_view=view(from_cache=False)))


def test_billing_consistency_matches_the_goldens_flags_when_present() -> None:
    reference = {"billed": True, "counted_against_quota": True, "from_cache": False}
    assert passes(billing_consistency, outputs_of(), reference)
    assert not passes(billing_consistency, outputs_of(), {**reference, "billed": False})
    assert not passes(billing_consistency, outputs_of(), {**reference, "counted_against_quota": False})
    assert not passes(billing_consistency, outputs_of(), {**reference, "from_cache": True})


def test_billing_consistency_compares_endpoints_as_a_multiset() -> None:
    reference = {"surepass_endpoints": [SCORE_ENDPOINT, PAN_ENDPOINT]}  # a different order than the run's
    assert passes(billing_consistency, outputs_of(), reference)
    assert not passes(billing_consistency, outputs_of(), {"surepass_endpoints": [PAN_ENDPOINT]})
    assert not passes(billing_consistency, outputs_of(), {"surepass_endpoints": [PAN_ENDPOINT, PAN_ENDPOINT, SCORE_ENDPOINT]})
    assert not passes(billing_consistency, outputs_of(), {"surepass_endpoints": []})


def test_billing_consistency_matches_the_goldens_admission_when_present() -> None:
    assert passes(billing_consistency, outputs_of(), {"admission": "new"})
    assert not passes(billing_consistency, outputs_of(), {"admission": "cache_hit"})


# --------------------------------------------------------------------------- outcome_matches_band


@pytest.mark.parametrize("band", ["NTC", "<650"])
def test_thin_bands_need_no_top_lender_or_a_coapplicant_and_are_never_fundable(band: str) -> None:
    nobody = view(top_lender=None, alternates=[], headline="unlikely", coapplicant_suggested=False)
    assert passes(outcome_matches_band, outputs_of(band=band, lc_view=nobody))  # top None is enough
    with_top = view(headline="unlikely", coapplicant_suggested=False)  # a lender shown, no co-applicant nudge
    assert not passes(outcome_matches_band, outputs_of(band=band, lc_view=with_top))
    nudged = view(headline="unlikely", coapplicant_suggested=True)
    assert passes(outcome_matches_band, outputs_of(band=band, lc_view=nudged))
    fundable = view(headline="fundable", coapplicant_suggested=True)
    assert not passes(outcome_matches_band, outputs_of(band=band, lc_view=fundable))


@pytest.mark.parametrize("band", ["650-720", "720-780", "780+"])
def test_scored_bands_are_unconstrained_by_the_thin_band_rule(band: str) -> None:
    assert passes(outcome_matches_band, outputs_of(band=band, lc_view=view(headline="fundable")))


def test_not_fundable_alone_always_suggests_a_coapplicant() -> None:
    bad = view(top_lender=None, alternates=[], headline="not_fundable_alone", coapplicant_suggested=False)
    assert not passes(outcome_matches_band, outputs_of(band="<650", lc_view=bad))
    ok = view(top_lender=None, alternates=[], headline="not_fundable_alone", coapplicant_suggested=True)
    assert passes(outcome_matches_band, outputs_of(band="<650", lc_view=ok))
    assert not passes(outcome_matches_band, outputs_of(band="720-780", lc_view=bad))


def test_an_unknown_band_skips_the_band_rules() -> None:
    refused = view(status="refused", code="QUOTA_EXHAUSTED", headline=None, confidence=None, top_lender=None, alternates=[])
    assert passes(outcome_matches_band, outputs_of(band=None, lc_view=refused))


def test_outcome_matches_the_goldens_pinned_fields() -> None:
    reference: Dict = {
        "status": "ok",
        "code": None,
        "headline": "possible",
        "confidence": "medium",
        "coapplicant_suggested": False,
        "top_lender_id": "L1",
        "alternate_ids": ["L2"],
        "label": "Indicative, based on what the learner told us",
        "items_to_confirm": [],
        "pan_aadhaar_link_warning": None,
        "quick_checks_left_today": 4,
        "checked_on": "2026-10-10",
        "refresh_available_on": "2026-11-09",
        "band": "720-780",
    }
    assert passes(outcome_matches_band, outputs_of(), reference)
    for key, wrong in (
        ("status", "refused"),
        ("code", "OPENAI_FAILURE"),
        ("headline", "fundable"),
        ("confidence", "high"),
        ("coapplicant_suggested", True),
        ("top_lender_id", "L2"),
        ("alternate_ids", []),
        ("label", "Something else"),
        ("items_to_confirm", ["x"]),
        ("pan_aadhaar_link_warning", "PAN not linked to Aadhaar; lenders will ask for this"),
        ("quick_checks_left_today", 5),
        ("checked_on", "2026-10-09"),
        ("refresh_available_on", "2026-11-08"),
        ("band", "<650"),
    ):
        result = run_eval(outcome_matches_band, outputs_of(), {**reference, key: wrong})
        assert result["score"] is False, key
        assert key in str(result["comment"]), (key, result)


def test_headline_in_pins_a_set_of_legal_headlines() -> None:
    """A model-dependent headline (PRD 9.4 low confidence: 'Under review' or 'Not fundable alone') is pinned as a SET."""
    low = view(top_lender=None, alternates=[], headline="under_review", confidence="low", coapplicant_suggested=True)
    legal = {"headline_in": ["not_fundable_alone", "under_review"]}
    assert passes(outcome_matches_band, outputs_of(band="<650", lc_view=low, code_confidence="medium"), legal)
    possible = view(top_lender=None, alternates=[], headline="possible", coapplicant_suggested=True)
    result = run_eval(outcome_matches_band, outputs_of(band="<650", lc_view=possible), legal)
    assert result["score"] is False and "headline" in str(result["comment"])


def test_the_v11_ntc_reversal_pin_a_success_false_score_must_be_a_service_failure() -> None:
    """q05, eval level: the golden says service_failure + SUREPASS_FAILURE. A run that treated success:false as NTC (the
    v1.0 reading: ok / unlikely / counted) fails this evaluator even though every other invariant still holds."""
    reference: Dict = {"status": "service_failure", "code": "SUREPASS_FAILURE", "headline": None, "confidence": None}
    ntc_reading = view(headline="unlikely", coapplicant_suggested=True)
    result = run_eval(outcome_matches_band, outputs_of(band="NTC", lc_view=ntc_reading), reference)
    assert result["score"] is False
    failure = view(
        status="service_failure", code="SUREPASS_FAILURE", headline=None, confidence=None, top_lender=None, alternates=[],
        label="Credit check couldn't complete, retry",
    )
    assert passes(outcome_matches_band, outputs_of(band=None, lc_view=failure), reference)


def test_an_explicit_none_in_the_golden_is_a_pin_not_a_wildcard() -> None:
    assert not passes(outcome_matches_band, outputs_of(), {"top_lender_id": None})
    nobody = view(top_lender=None, alternates=[])
    assert passes(outcome_matches_band, outputs_of(lc_view=nobody), {"top_lender_id": None, "alternate_ids": []})


# --------------------------------------------------------------------------- the LLM judge (7th evaluator, gated)


class _StubJudge:
    def __init__(self, verdict: JudgeVerdict) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> JudgeVerdict:
        self.prompts.append(prompt)
        return self.verdict


def test_judge_scores_the_llm_prose_with_the_rubric() -> None:
    stub = _StubJudge(JudgeVerdict(score=1, reasons="plain, actionable"))
    judge = make_llm_judge(stub)
    result = run_eval(judge, outputs_of())
    assert result["score"] is True and result["key"] == "plain_language_judge"
    assert len(stub.prompts) == 1
    prompt = stub.prompts[0]
    assert evaluators.JUDGE_RUBRIC in prompt
    assert "Serves the learner's area" in prompt  # the rationale under judgement
    assert "L1" not in prompt and UUID_LIKE not in prompt  # ids are not prose

    failing = make_llm_judge(_StubJudge(JudgeVerdict(score=0, reasons="promises approval")))
    result = run_eval(failing, outputs_of())
    assert result["score"] is False and "promises approval" in str(result["comment"])


def test_judge_skips_views_with_no_model_prose() -> None:
    stub = _StubJudge(JudgeVerdict(score=0, reasons="never asked"))
    judge = make_llm_judge(stub)
    refused = view(status="refused", code="QUOTA_EXHAUSTED", headline=None, confidence=None, top_lender=None, alternates=[])
    assert passes(judge, outputs_of(lc_view=refused))
    k7 = view(
        status="service_failure", code="OPENAI_FAILURE", confidence="low", top_lender=None,
        alternates=[lender("L1", ceiling=None, lower=None, rationale="Automatic ranking unavailable; this lender serves the learner's area")],
    )
    assert passes(judge, outputs_of(lc_view=k7))  # code-built prose is not the model's
    assert stub.prompts == []


def test_judge_is_gated_on_the_openai_key() -> None:
    assert judge_evaluators({}) == []
    assert judge_evaluators({"OPENAI_API_KEY": ""}) == []
    assert judge_evaluators({"OPENAI_API_KEY": "   "}) == []


def test_judge_factory_builds_the_pinned_chat_model_only_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Dict] = []
    structured: list[Dict] = []

    class FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        def with_structured_output(self, schema: Any, **kwargs: Any) -> "FakeChat":
            structured.append({"schema": schema, **kwargs})
            return self

        def invoke(self, prompt: str) -> JudgeVerdict:
            return JudgeVerdict(score=1, reasons=f"saw {len(prompt)} chars")

    monkeypatch.setattr(evaluators, "ChatOpenAI", FakeChat)

    (judge,) = judge_evaluators({"OPENAI_API_KEY": "sk-synthetic"})

    assert len(built) == 1
    kwargs = built[0]
    assert (kwargs["model"], kwargs["temperature"], kwargs["timeout"], kwargs["max_retries"]) == ("gpt-4.1", 0, 8.0, 0)
    assert kwargs["api_key"].get_secret_value() == "sk-synthetic"
    assert structured == [{"schema": JudgeVerdict, "method": "function_calling"}]
    assert passes(judge, outputs_of())  # the judge runs through the stubbed chat model
    assert judge.__name__ == "plain_language_judge"


# --------------------------------------------------------------------------- run_one smoke: two goldens, the real graph


@pytest.mark.parametrize("example_id", ["q01_happy_745", "q05_score_success_false"])
def test_every_deterministic_evaluator_passes_on_real_run_one_outputs_of_two_goldens(example_id: str) -> None:
    """q01 (the happy quick check) and q05 (the v1.1 NTC reversal: success:false is a service failure) through the real
    graph with the fake model and fake SurePass: the evaluators accept the run and the golden's literal expectations."""
    example = next(e for e in [*quick_examples(), *full_examples()] if e["example_id"] == example_id)
    inputs = inputs_of(example)

    outputs = run_one(inputs)

    for evaluator in DETERMINISTIC_EVALUATORS:
        result = evaluator(dict(inputs), outputs, example["expected"])
        assert result["score"] is True, (example_id, result)
