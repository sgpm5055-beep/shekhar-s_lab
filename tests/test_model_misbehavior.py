"""PRD 12 items 15-17: the single LLM node misbehaves and the code fence holds.

* (15) The model picks lenders outside the gate (an invented id, a real lender gated out by pincode) or below the band's
  floor: safety demotes them (never retried: a business violation is not an invalid output). Nothing surviving goes to
  review: final confidence low, headline only. Review rows per K11: ``safety_demoted_all`` whenever the model's TOP pick
  was demoted (also when alternates survive and get promoted), ``not_fundable_pattern`` when every demotion was the band
  floor (headline ``not_fundable_alone``); an alternate-only demotion is logged, not reviewed.
* (16) A raised confidence is ignored (final = min(code, model)); a lowered one is honoured. Raising uses the T10 helper,
  which refuses to build a vacuous "raise".
* (17) Invalid output twice: EXACTLY two calls (one retry, never a third), then the K7 view: ``service_failure`` +
  ``OPENAI_FAILURE``, the band-derived headline, confidence low, no top lender, and up to two CODE-BUILT gate survivors
  whose floor the band clears (id order; id / name / rate band / tenure only; no money, a fixed number-free rationale,
  no conditions). Billing and quota are status-independent (K8): the Score was a billed 200 + success pull, so the quick
  run is billed AND counted. Review row ``openai_failure``.

World: the K17 seed lenders; happy_quick_scenario (zip 560001 -> L1 + L2 gated, L3 serves 400001 only; score 745 ->
band 720-780), income 60,000, EMI 0. Money pins as in the happy path (dossier): L1 1,277,000 / 368,000; L2 clipped
500,000 / 295,000.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

import pytest

from eligibility_agent.output.lc_view import LABEL_QUICK, MSG_OPENAI_FAILURE, RATIONALE_RANKING_UNAVAILABLE
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.fake_llm import (
    FakeRankModel,
    below_floor_pick,
    default_ranking,
    invalid_output_error,
    lowered_confidence,
    out_of_gate_pick,
    raised_confidence,
    ranked_output,
)
from eligibility_agent.testing.scenarios import (
    full_emi_four_branch_scenario,
    full_happy_reported_emi_scenario,
    happy_quick_scenario,
    ntc_scenario,
)
from tests.helpers import (
    LC_A,
    OMIT,
    PAN_NTC,
    PAN_OK,
    Api,
    ApiFactory,
    SurePassFactory,
    assert_no_leak,
    criteria_json,
    internal_of,
    seed_lender,
)

RANKING_UNAVAILABLE: Final[str] = "Automatic ranking unavailable; this lender serves the learner's area"
K7_L1: Final[dict[str, Any]] = {
    "id": "L1",
    "name": "Alpha Education Finance",
    "rate_band": "11 to 14%",
    "tenure_range": "12 to 48 months",
    "ceiling": None,
    "range": None,
    "rationale": RANKING_UNAVAILABLE,
    "conditions": [],
}
K7_L2: Final[dict[str, Any]] = {
    "id": "L2",
    "name": "Bharat Upskill Credit",
    "rate_band": "14 to 18%",
    "tenure_range": "12 to 36 months",
    "ceiling": None,
    "range": None,
    "rationale": RANKING_UNAVAILABLE,
    "conditions": [],
}
L1_MONEY: Final[tuple[str, int, dict[str, int]]] = ("L1", 1277000, {"lower": 368000, "upper": 1277000})
L2_MONEY: Final[tuple[str, int, dict[str, int]]] = ("L2", 500000, {"lower": 295000, "upper": 500000})


def _happy(make_api: ApiFactory, rank: FakeRankModel) -> Api:
    return make_api(lambda sink: happy_quick_scenario(sink=sink), rank=rank)


def _l3_only_below_floor() -> SurePassFactory:
    """PAN zip 400001 (only L3, floor 720, serves it) and score 700 -> band 650-720: the one gated lender's floor is NOT
    cleared."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, zip_code="400001")},
            score={PAN_OK: env.experian_score_envelope(credit_score="700")},
            sink=sink,
        )

    return factory


def _money(view: dict[str, Any]) -> tuple[str, int | None, dict[str, int] | None]:
    return view["id"], view["ceiling"], view["range"]


def _review_kinds(api: Api, check_id: str) -> list[str]:
    return [r.kind for r in api.repo.list_reviews() if r.check_id == check_id]


def _demotions(api: Api, check_id: str) -> list[tuple[str, str]]:
    safety = internal_of(api.row(check_id))["safety"]
    return [(d["lender_id"], d["rule"]) for d in safety["demoted"]]


# --------------------------------------------------------------------------- (15) picks outside the gate / below floor


def test_out_of_gate_and_invented_picks_are_demoted_and_the_run_goes_to_review(make_api: ApiFactory) -> None:
    api = _happy(make_api, FakeRankModel([out_of_gate_pick("L3", "L_NOT_IN_GATE")]))

    body = api.quick()

    assert {k: v for k, v in body.items() if k != "check_id"} == {
        "kind": "quick",
        "status": "ok",
        "code": None,
        "headline": "under_review",
        "label": LABEL_QUICK,
        "confidence": "low",
        "top_lender": None,  # low confidence: headline only
        "alternates": [],
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-10",
        "from_cache": False,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": 4,
    }
    check_id = body["check_id"]
    assert api.rank.call_count == 1  # a business violation is never retried
    assert _demotions(api, check_id) == [("L3", "not_in_gate"), ("L_NOT_IN_GATE", "not_in_gate")]
    safety = internal_of(api.row(check_id))["safety"]
    assert (safety["top"], safety["alternates"], safety["none_survived"], safety["review"]) == (None, [], True, True)
    assert all(d["detail"] for d in safety["demoted"])  # each demotion carries its (internal) reason
    assert _review_kinds(api, check_id) == ["safety_demoted_all"]
    row = api.row(check_id)
    assert (row.status, row.billed, row.counted_against_quota) == ("ok", True, True)
    text = json.dumps(body)
    assert "L3" not in text and "Crest Capital" not in text and "L_NOT_IN_GATE" not in text
    assert_no_leak(body)


def test_a_below_floor_pick_is_demoted_and_none_surviving_is_not_fundable_alone(make_api: ApiFactory) -> None:
    api = make_api(_l3_only_below_floor(), rank=FakeRankModel([below_floor_pick()]))

    body = api.quick()

    assert (body["status"], body["code"], body["confidence"], body["headline"]) == (
        "ok",
        None,
        "low",
        "not_fundable_alone",
    )
    assert (body["top_lender"], body["alternates"], body["coapplicant_suggested"]) == (None, [], True)
    check_id = body["check_id"]
    assert internal_of(api.row(check_id))["gated_lender_ids"] == ["L3"]
    assert _demotions(api, check_id) == [("L3", "band_floor")]
    assert _review_kinds(api, check_id) == ["not_fundable_pattern"]
    assert api.rank.call_count == 1
    assert_no_leak(body)


def _band_610(make_api: ApiFactory, rank: FakeRankModel | None = None) -> Api:
    """Score 610 -> band '<650'; PAN zip 560001 gates L1 and L2, both with a 650 floor: no gated lender's floor is met."""
    return make_api(lambda sink: happy_quick_scenario(score="610", sink=sink), rank=rank)


def test_a_compliant_model_that_names_nobody_below_every_floor_is_not_fundable_alone(make_api: ApiFactory) -> None:
    """PRD 9.4: 'No lender surviving the gate OR THE BAND is one path: low, Not fundable alone, co_applicant_suggested true,
    review queue'. The default (compliant) fake sees every floor fail and recommends nobody, so safety has NOTHING to
    demote; the band alone must decide (it used to read 'under_review')."""
    api = _band_610(make_api)

    body = api.quick()

    assert (body["status"], body["code"], body["confidence"], body["headline"]) == (
        "ok",
        None,
        "low",
        "not_fundable_alone",
    )
    assert (body["top_lender"], body["alternates"], body["coapplicant_suggested"]) == (None, [], True)
    assert body["coapplicant_reason"]  # the model's scrubbed reason rides along
    check_id = body["check_id"]
    internal = internal_of(api.row(check_id))
    assert (internal["derived"]["band"], internal["gated_lender_ids"]) == ("<650", ["L1", "L2"])
    assert internal["model_output"]["recommendation"] is None and internal["model_output"]["alternates"] == []
    assert _demotions(api, check_id) == []  # nothing was demoted: the model declined every lender itself
    assert internal["safety"]["none_survived"] is True
    assert _review_kinds(api, check_id) == ["not_fundable_pattern"]  # PRD 9.4: the review queue sees the pattern
    assert api.rank.call_count == 1
    row = api.row(check_id)
    assert (row.status, row.billed, row.counted_against_quota) == ("ok", True, True)
    assert_no_leak(body)


def test_an_uncertain_floor_on_a_gated_lender_keeps_the_review_headline(make_api: ApiFactory) -> None:
    """The admin marked L1's floor 'uncertain': the band's failure of it is not KNOWN, so 'not fundable alone' would be a
    claim the criteria cannot support. Nobody named + an unsure floor stays 'under_review' (still low, still reviewed)."""
    api = _band_610(make_api)
    l1 = seed_lender("L1")
    criteria = criteria_json(l1.criteria)
    criteria["min_score_floor"] = {"value": 650, "status": "uncertain"}
    api.repo.set_lender_published("L1", 1, False)
    api.repo.seed_lender("L1", 2, l1.name, True, criteria, l1.policy_notes, l1.pincodes)

    body = api.quick()

    assert (body["confidence"], body["headline"], body["top_lender"]) == ("low", "under_review", None)
    assert body["coapplicant_suggested"] is True  # the fake's own thin-band nudge; the headline rule is what differs
    assert internal_of(api.row(body["check_id"]))["gated_lender_ids"] == ["L1", "L2"]
    assert_no_leak(body)


def test_the_k7_view_on_a_failed_band_keeps_its_band_derived_headline(make_api: ApiFactory) -> None:
    """The model failing twice is NOT the 'no lender fits' path even when every floor fails: K7 stays band-derived
    ('unlikely' for band '<650'), never 'not_fundable_alone'."""
    api = _band_610(make_api, FakeRankModel.invalid_twice())

    body = api.quick()

    assert (body["status"], body["code"], body["headline"], body["confidence"]) == (
        "service_failure",
        "OPENAI_FAILURE",
        "unlikely",
        "low",
    )
    assert _review_kinds(api, body["check_id"]) == ["openai_failure"]
    assert_no_leak(body)


def test_a_demoted_top_promotes_the_surviving_alternates_and_still_goes_to_review(make_api: ApiFactory) -> None:
    """K11: safety_demoted_all fires whenever the model's TOP pick did not survive, even with survivors to show."""
    api = _happy(make_api, FakeRankModel([ranked_output("L_NOT_IN_GATE", "L1", "L2")]))

    body = api.quick()

    assert (body["status"], body["confidence"], body["headline"]) == ("ok", "medium", "possible")
    assert _money(body["top_lender"]) == L1_MONEY
    assert [_money(a) for a in body["alternates"]] == [L2_MONEY]
    check_id = body["check_id"]
    assert _demotions(api, check_id) == [("L_NOT_IN_GATE", "not_in_gate")]
    safety = internal_of(api.row(check_id))["safety"]
    assert (safety["top"]["lender_id"], safety["none_survived"], safety["review"]) == ("L1", False, True)
    assert _review_kinds(api, check_id) == ["safety_demoted_all"]
    assert "L_NOT_IN_GATE" not in json.dumps(body)
    assert_no_leak(body)


def test_an_alternate_only_demotion_is_logged_not_reviewed(make_api: ApiFactory) -> None:
    api = _happy(make_api, FakeRankModel([ranked_output("L1", "L3")]))

    body = api.quick()

    assert (body["status"], body["confidence"], body["headline"]) == ("ok", "medium", "possible")
    assert _money(body["top_lender"]) == L1_MONEY
    assert body["alternates"] == []
    check_id = body["check_id"]
    assert _demotions(api, check_id) == [("L3", "not_in_gate")]
    assert internal_of(api.row(check_id))["safety"]["review"] is False
    assert _review_kinds(api, check_id) == []
    assert_no_leak(body)


# --------------------------------------------------------------------------- (16) confidence: raise ignored, lower honoured


def test_a_raised_confidence_is_ignored_on_a_quick_check(make_api: ApiFactory) -> None:
    api = _happy(make_api, FakeRankModel([raised_confidence("high")]))

    body = api.quick()

    assert (body["confidence"], body["headline"]) == ("medium", "possible")  # never 'high' / 'fundable'
    assert _money(body["top_lender"]) == L1_MONEY
    internal = internal_of(api.row(body["check_id"]))
    assert (internal["code_confidence"], internal["model_output"]["confidence"], internal["final_confidence"]) == (
        "medium",
        "high",
        "medium",
    )
    assert _review_kinds(api, body["check_id"]) == []
    assert_no_leak(body)


def test_a_raised_confidence_is_ignored_on_a_medium_full_check(make_api: ApiFactory) -> None:
    """The four-branch report carries estimates: code medium. The model claims high; the view stays medium."""
    api = make_api(lambda sink: full_emi_four_branch_scenario(sink=sink), rank=FakeRankModel([raised_confidence("high")]))

    body = api.full(monthly_income=80000, existing_emi=OMIT)

    assert (body["confidence"], body["headline"]) == ("medium", "possible")
    internal = internal_of(api.row(body["check_id"]))
    assert (internal["code_confidence"], internal["model_output"]["confidence"]) == ("medium", "high")
    assert_no_leak(body)


def test_a_lowered_confidence_is_honoured_and_low_is_headline_only(make_api: ApiFactory) -> None:
    api = _happy(make_api, FakeRankModel([lowered_confidence("low")]))

    body = api.quick()

    assert (body["confidence"], body["headline"]) == ("low", "under_review")  # a surviving top at low: defensive row
    assert (body["top_lender"], body["alternates"]) == (None, [])
    safety = internal_of(api.row(body["check_id"]))["safety"]
    assert (safety["top"]["lender_id"], safety["demoted"]) == ("L1", [])  # nothing was demoted: the model said low
    assert _review_kinds(api, body["check_id"]) == ["model_review_flags"]
    assert_no_leak(body)


def test_a_lowered_confidence_on_a_high_full_check_drops_to_medium(make_api: ApiFactory) -> None:
    api = make_api(
        lambda sink: full_happy_reported_emi_scenario(sink=sink), rank=FakeRankModel([lowered_confidence("medium")])
    )

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    assert (body["confidence"], body["headline"]) == ("medium", "possible")
    assert internal_of(api.row(body["check_id"]))["code_confidence"] == "high"
    assert_no_leak(body)


# --------------------------------------------------------------------------- (17) invalid twice -> the K7 view


def _assert_k7_number_free(body: dict[str, Any]) -> None:
    """No alternate carries a money field (ceiling / range) or a digit in its code-built prose; the whole view is
    leak-free."""
    for alternate in body["alternates"]:
        assert (alternate["ceiling"], alternate["range"], alternate["conditions"]) == (None, None, [])
        assert alternate["rationale"] == RANKING_UNAVAILABLE and not re.search(r"\d", alternate["rationale"])
        assert set(alternate) == {"id", "name", "rate_band", "tenure_range", "ceiling", "range", "rationale", "conditions"}
    assert body["top_lender"] is None
    assert_no_leak(body)


def test_invalid_output_twice_is_openai_failure_with_the_band_read_and_code_built_alternates(
    make_api: ApiFactory,
) -> None:
    api = _happy(make_api, FakeRankModel.invalid_twice())

    body = api.quick()

    assert RANKING_UNAVAILABLE == RATIONALE_RANKING_UNAVAILABLE
    assert {k: v for k, v in body.items() if k != "check_id"} == {
        "kind": "quick",
        "status": "service_failure",
        "code": "OPENAI_FAILURE",
        "headline": "possible",  # band-derived (720-780), never under_review on this path
        "label": MSG_OPENAI_FAILURE,
        "confidence": "low",
        "top_lender": None,
        "alternates": [K7_L1, K7_L2],  # gate survivors whose floor the band clears, id order
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-10",
        "from_cache": False,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": 4,
    }
    _assert_k7_number_free(body)

    assert api.rank.call_count == 2  # one retry, never a third call
    assert [ctx.attempt for ctx in api.rank.contexts] == [1, 2]
    assert api.rank.contexts[0].prior_error is None and api.rank.contexts[1].prior_error
    check_id = body["check_id"]
    row = api.row(check_id)
    assert (row.status, row.code, row.billed, row.counted_against_quota) == (
        "service_failure",
        "OPENAI_FAILURE",
        True,
        True,  # K8: the billed Score pull counts whatever the model did
    )
    assert row.billed == any(api.billable(check_id))
    assert api.repo.quick_checks_used_today(LC_A) == 1
    internal = internal_of(row)
    assert (internal["gated_lender_ids"], internal["rank_attempts"], internal["model_output"]) == (["L1", "L2"], 2, None)
    assert internal["safety"] is None
    assert _review_kinds(api, check_id) == ["openai_failure"]
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]


def test_a_raising_model_twice_is_the_same_k7_outcome(make_api: ApiFactory) -> None:
    """The timeout analogue on the LLM seam: any raised exception is a failed attempt."""
    api = _happy(make_api, FakeRankModel([TimeoutError("model timed out")]))

    body = api.quick()

    assert (body["status"], body["code"], body["headline"]) == ("service_failure", "OPENAI_FAILURE", "possible")
    assert body["alternates"] == [K7_L1, K7_L2]
    assert api.rank.call_count == 2
    assert _review_kinds(api, body["check_id"]) == ["openai_failure"]
    _assert_k7_number_free(body)


def test_one_invalid_output_then_a_valid_one_is_a_normal_run(make_api: ApiFactory) -> None:
    api = _happy(make_api, FakeRankModel([invalid_output_error(), default_ranking]))

    body = api.quick()

    assert (body["status"], body["code"], body["confidence"], body["headline"]) == ("ok", None, "medium", "possible")
    assert _money(body["top_lender"]) == L1_MONEY
    assert api.rank.call_count == 2 and api.rank.contexts[1].prior_error
    assert _review_kinds(api, body["check_id"]) == []
    assert_no_leak(body)


def test_k7_on_an_ntc_band_is_unlikely_with_only_the_ntc_accepting_survivor(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: ntc_scenario("empty", sink=sink), rank=FakeRankModel.invalid_twice())

    body = api.quick(pan=PAN_NTC, monthly_income=30000)

    assert (body["status"], body["code"], body["confidence"]) == ("service_failure", "OPENAI_FAILURE", "low")
    assert (body["headline"], body["coapplicant_suggested"], body["coapplicant_reason"]) == ("unlikely", True, None)
    assert body["alternates"] == [K7_L2]  # L1 does not take NTC: its floor is not cleared
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, True)
    _assert_k7_number_free(body)


def test_k7_shows_at_most_two_survivors_in_id_order(make_api: ApiFactory) -> None:
    """Three gated lenders clear the floor (L3 republished to also serve 560001; 745 clears its 720 floor): the K7 view
    lists the first two by id, never a third."""
    api = _happy(make_api, FakeRankModel.invalid_twice())
    l3 = seed_lender("L3")
    api.repo.set_lender_published("L3", 1, False)
    api.repo.seed_lender("L3", 2, l3.name, True, criteria_json(l3.criteria), l3.policy_notes, ("400001", "560001"))

    body = api.quick()

    assert internal_of(api.row(body["check_id"]))["gated_lender_ids"] == ["L1", "L2", "L3"]
    assert body["alternates"] == [K7_L1, K7_L2]
    _assert_k7_number_free(body)


def test_k7_never_lists_a_gated_lender_whose_floor_the_band_misses(make_api: ApiFactory) -> None:
    api = make_api(_l3_only_below_floor(), rank=FakeRankModel.invalid_twice())

    body = api.quick()

    assert (body["status"], body["code"], body["headline"]) == ("service_failure", "OPENAI_FAILURE", "possible")
    assert internal_of(api.row(body["check_id"]))["gated_lender_ids"] == ["L3"]
    assert body["alternates"] == []  # L3 is gated, but band 650-720 is below its 720 floor
    assert api.rank.call_count == 2
    _assert_k7_number_free(body)


def test_k7_on_a_full_check_is_billed_but_never_counted(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink), rank=FakeRankModel.invalid_twice())

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert (body["kind"], body["status"], body["code"], body["label"]) == (
        "full",
        "service_failure",
        "OPENAI_FAILURE",
        MSG_OPENAI_FAILURE,
    )
    assert (body["headline"], body["confidence"], body["quick_checks_left_today"]) == ("possible", "low", None)
    assert body["alternates"] == [K7_L1, K7_L2]
    assert body["items_to_confirm"] == []
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, False)
    assert _review_kinds(api, body["check_id"]) == ["openai_failure"]
    assert api.rank.call_count == 2
    _assert_k7_number_free(body)


def test_k7_keeps_the_code_built_confirm_items(make_api: ApiFactory) -> None:
    """Fix round 1 ruling (PRD 8.3): the stale / past-tenure confirm items are built by code before the model runs, so a
    failed ranking does not hide them (K7 constrains only the lender fields)."""
    api = make_api(lambda sink: full_emi_four_branch_scenario(sink=sink), rank=FakeRankModel.invalid_twice())

    body = api.full(monthly_income=80000, existing_emi=OMIT)

    assert (body["status"], body["code"], body["headline"]) == ("service_failure", "OPENAI_FAILURE", "possible")
    assert body["items_to_confirm"] == [
        "past tenure, confirm with learner",
        "confirm with learner: an account last updated 2026-06-01",
    ]
    assert body["alternates"] == [K7_L1, K7_L2]
    _assert_k7_number_free(body)


@pytest.mark.parametrize("pick", ["L3", "L_NOT_IN_GATE"])
def test_a_misbehaving_pick_never_reaches_the_view_at_any_confidence(make_api: ApiFactory, pick: str) -> None:
    """Even when the model claims high confidence for an out-of-gate top, the view shows only safety survivors."""
    api = _happy(make_api, FakeRankModel([ranked_output(pick, "L2", confidence="high")]))

    body = api.quick()

    assert body["top_lender"] is not None and body["top_lender"]["id"] == "L2"
    assert body["alternates"] == []
    assert pick not in json.dumps(body)
    assert_no_leak(body)
