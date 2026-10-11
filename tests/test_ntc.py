"""PRD 12 item 5 + PRD 8.1 (FIXED v1.1 rule): NTC is HTTP 200 + ``success: true`` + a ``credit_score`` that is NOT a
three-digit string ('', 'NH', key missing, '0745'). It is counted (a no-hit consumes quota), cached 30 days as NTC, and
shown 'unlikely' with a co-applicant suggested plus the NTC-accepting lenders in the pincode (L2 only of the seed set).
A ``success: false`` Score envelope is NEVER NTC: it is a service failure that counts nothing and caches nothing.

Money pin (dossier): income 30,000 -> L2 available 0.45 x 30,000 = 13,500 -> raw 394,995.21 / 147,251.32 -> half-up
395,000 / 147,000.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from eligibility_agent.output.lc_view import LABEL_QUICK, MSG_BUREAU_SERVICE_FAILURE, line_leaks
from eligibility_agent.surepass.client import PAN_ENDPOINT, SCORE_ENDPOINT
from eligibility_agent.testing.fake_llm import FakeRankModel, lowered_confidence
from eligibility_agent.testing.scenarios import NtcVariant, ntc_scenario, score_failure_scenario
from tests.helpers import LC_A, LC_B, PAN_NTC, T0, ApiFactory, assert_no_leak, internal_of, pan_hash_of

L2_AT_30K: dict[str, Any] = {
    "id": "L2",
    "name": "Bharat Upskill Credit",
    "rate_band": "14 to 18%",
    "tenure_range": "12 to 36 months",
    "ceiling": 395000,
    "range": {"lower": 147000, "upper": 395000},
    "conditions": [],
}


@pytest.mark.parametrize("variant", ["empty", "text", "missing", "four_digit"])
def test_ntc_score_is_counted_cached_and_unlikely_with_a_coapplicant(make_api: ApiFactory, variant: NtcVariant) -> None:
    api = make_api(lambda sink: ntc_scenario(variant, sink=sink))

    body = api.quick(pan=PAN_NTC, monthly_income=30000)

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert (body["status"], body["code"], body["headline"], body["label"]) == ("ok", None, "unlikely", LABEL_QUICK)
    assert body["confidence"] == "medium"  # quick checks cap at medium (K16)
    assert body["coapplicant_suggested"] is True
    reason = body["coapplicant_reason"]
    assert isinstance(reason, str) and reason and not line_leaks(reason)
    top = body["top_lender"]
    assert {k: v for k, v in top.items() if k != "rationale"} == L2_AT_30K  # the only NTC-accepting lender
    assert body["alternates"] == []  # L1 (and L3, outside the pincode) do not take NTC
    assert body["quick_checks_left_today"] == 4

    row = api.row(body["check_id"])
    assert (row.status, row.billed, row.counted_against_quota) == ("ok", True, True)  # a no-hit consumes quota
    assert row.billed == any(api.billable(row.id))
    score = api.repo.get_score_cache(pan_hash_of(PAN_NTC))
    assert score is not None
    assert (score.is_ntc, score.source, score.expires_at) == (True, "score", T0 + timedelta(days=30))
    assert internal_of(row)["derived"]["band"] == "NTC"
    assert_no_leak(body)

    # a second run inside 30 days (another LC, five days on): a free hit with the same outcome
    api.clock.advance(timedelta(days=5))
    again = api.quick(pan=PAN_NTC, monthly_income=30000, lc_id=LC_B)

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]  # zero new calls
    assert (again["from_cache"], again["headline"], again["coapplicant_suggested"]) == (True, "unlikely", True)
    assert {k: v for k, v in again["top_lender"].items() if k != "rationale"} == L2_AT_30K
    assert again["checked_on"] == "2026-10-10"
    hit = api.row(again["check_id"])
    assert (hit.billed, hit.counted_against_quota) == (False, False)
    assert_no_leak(again)


def test_success_false_score_is_a_service_failure_never_ntc(make_api: ApiFactory) -> None:
    """The v1.1 regression pin: (200, success:false, no_record_found) is SUREPASS_FAILURE, not counted, nothing cached."""
    api = make_api(lambda sink: score_failure_scenario("success_false", pan=PAN_NTC, sink=sink))

    body = api.quick(pan=PAN_NTC, monthly_income=30000)

    assert (body["status"], body["code"], body["label"]) == ("service_failure", "SUREPASS_FAILURE", MSG_BUREAU_SERVICE_FAILURE)
    assert (body["headline"], body["top_lender"], body["coapplicant_suggested"]) == (None, None, False)
    assert body["quick_checks_left_today"] == 5
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    row = api.row(body["check_id"])
    assert (row.status, row.code, row.counted_against_quota) == ("service_failure", "SUREPASS_FAILURE", False)
    assert api.billable(row.id) == [True, False]  # the PAN call billed (Rs 1, result kept); the Score did not
    assert row.billed is True
    assert api.repo.get_score_cache(pan_hash_of(PAN_NTC)) is None  # NOTHING cached as NTC
    assert api.repo.get_pan_record(pan_hash_of(PAN_NTC)) is not None  # PRD 10: "PAN result kept"
    assert api.repo.quick_checks_used_today(LC_A) == 0
    assert api.rank.call_count == 0
    assert_no_leak(body)


def test_ntc_with_a_surviving_lender_at_low_confidence_is_still_unlikely(make_api: ApiFactory) -> None:
    """T8 deferred vector: NTC, an NTC-accepting survivor, zero demotions, the model lowers confidence to low. Low means
    headline only (no lender shown), and the headline stays 'unlikely' (row f), never 'under_review'."""
    api = make_api(lambda sink: ntc_scenario("empty", sink=sink), rank=FakeRankModel([lowered_confidence("low")]))

    body = api.quick(pan=PAN_NTC, monthly_income=30000)

    assert (body["status"], body["headline"], body["confidence"]) == ("ok", "unlikely", "low")
    assert (body["top_lender"], body["alternates"], body["coapplicant_suggested"]) == (None, [], True)
    safety = internal_of(api.row(body["check_id"]))["safety"]
    assert (safety["top"]["lender_id"], safety["demoted"], safety["none_survived"]) == ("L2", [], False)
    assert_no_leak(body)
