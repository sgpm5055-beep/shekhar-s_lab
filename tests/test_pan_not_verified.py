"""PRD 12 item 4 + PRD 10 PAN row: a PAN that is valid-negative (200, success:true, status != valid or category != person)
is ``pan_not_verified``: the Rs 1 PAN call only, billed, never counted, nothing cached, nothing marked checked; the LC
may correct the PAN and retry. A ``success: false`` PAN envelope is NOT that: under v1.1 it is a SERVICE failure
(``service_failure`` + SUREPASS_FAILURE, "Couldn't verify PAN, retry"), the v1.0 reading reversed (T5 ruling).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest

from eligibility_agent.output.lc_view import MSG_PAN_NOT_VERIFIED, MSG_PAN_SERVICE_FAILURE
from eligibility_agent.surepass.client import PAN_ENDPOINT, SCORE_ENDPOINT, CallLogSink, FakeSurePassClient
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.scenarios import pan_invalid_scenario
from tests.helpers import (
    LC_A,
    PAN_COMPANY,
    PAN_INVALID,
    PAN_OK,
    ApiFactory,
    assert_no_leak,
    one_check_row,
    pan_hash_of,
)

NegativeKind = Literal["status_invalid", "company"]
BAD_PAN: dict[NegativeKind, str] = {"status_invalid": PAN_INVALID, "company": PAN_COMPANY}


def _negative_then_corrected(kind: NegativeKind) -> Callable[[CallLogSink], FakeSurePassClient]:
    """The bad PAN answers valid-negative; the corrected PAN answers like the happy world (PAN + Score)."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        bad = BAD_PAN[kind]
        negative = (
            env.pan_comprehensive_envelope(pan=bad, status="invalid")
            if kind == "status_invalid"
            else env.pan_comprehensive_envelope(pan=bad, category="company")
        )
        return FakeSurePassClient(
            pan={bad: negative, PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK)},
            score={PAN_OK: env.experian_score_envelope(credit_score="745")},
            sink=sink,
        )

    return factory


@pytest.mark.parametrize("kind", ["status_invalid", "company"])
def test_valid_negative_pan_is_pan_not_verified_then_the_corrected_pan_runs(
    make_api: ApiFactory, kind: NegativeKind
) -> None:
    api = make_api(_negative_then_corrected(kind))
    bad = BAD_PAN[kind]

    body = api.quick(pan=bad)

    assert {k: v for k, v in body.items() if k != "check_id"} == {
        "kind": "quick",
        "status": "pan_not_verified",
        "code": None,
        "headline": None,
        "label": MSG_PAN_NOT_VERIFIED,
        "confidence": None,
        "top_lender": None,
        "alternates": [],
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": None,
        "from_cache": False,
        "refresh_available_on": None,
        "quick_checks_left_today": 5,  # no quota was used
    }
    assert api.endpoints() == [PAN_ENDPOINT]  # the Rs 1 call only: no bureau call
    row = one_check_row(api.repo)
    assert (row.status, row.code) == ("pan_not_verified", None)
    assert (row.billed, row.counted_against_quota) == (True, False)
    assert row.billed == any(api.billable(row.id))
    # nothing marked checked: no PAN record, no score row, so a corrected retry starts clean
    assert api.repo.get_pan_record(pan_hash_of(bad)) is None
    assert api.repo.get_score_cache(pan_hash_of(bad)) is None
    assert api.rank.call_count == 0
    assert_no_leak(body)

    corrected = api.quick(pan=PAN_OK)

    assert api.endpoints() == [PAN_ENDPOINT, PAN_ENDPOINT, SCORE_ENDPOINT]
    assert (corrected["status"], corrected["quick_checks_left_today"]) == ("ok", 4)
    retry = api.row(corrected["check_id"])
    assert (retry.billed, retry.counted_against_quota) == (True, True)
    assert api.repo.quick_checks_used_today(LC_A) == 1


def test_success_false_pan_envelope_is_a_service_failure_not_pan_not_verified(make_api: ApiFactory) -> None:
    """(422, success:false) on PAN Comprehensive: 'Couldn't verify PAN, retry'; no bureau call, no quota, no spend."""
    api = make_api(lambda sink: pan_invalid_scenario("failure_envelope", pan=PAN_OK, sink=sink))

    body = api.quick()

    assert (body["status"], body["code"], body["label"]) == ("service_failure", "SUREPASS_FAILURE", MSG_PAN_SERVICE_FAILURE)
    assert (body["headline"], body["confidence"], body["top_lender"]) == (None, None, None)
    assert body["quick_checks_left_today"] == 5
    assert api.endpoints() == [PAN_ENDPOINT]
    row = one_check_row(api.repo)
    assert (row.status, row.code, row.counted_against_quota) == ("service_failure", "SUREPASS_FAILURE", False)
    assert api.billable(row.id) == [False]  # a success:false envelope is not a billable call
    assert row.billed is False
    assert api.repo.get_pan_record(pan_hash_of(PAN_OK)) is None
    assert api.rank.call_count == 0
    assert_no_leak(body)
