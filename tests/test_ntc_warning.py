"""PRD 12 item 11 + PRD 5.14: a full check on a PAN whose stored Score said NTC inside the last 30 days is refused
``WARN_NTC_RECENT`` ("No bureau file was found on <date>; ... Proceed?", the date formatted as the IST date of that
pull) unless the LC acknowledges the warning; the acknowledged re-check pulls the (billed) Report. Once the NTC score is
30 days old the warning is gone. The refusal itself is free and never marks the opportunity's PAN.

World: PAN_NTC (zip 560001), a Score with ``credit_score ''`` (NTC) and an account-less NTC Report, income 30,000.
Money pin (dossier, the NTC quick): L2 available 0.45 x 30,000 = 13,500 -> 395,000 / 147,000.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Final

import pytest

from eligibility_agent.output.lc_view import REFUSAL_MESSAGES
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.scenarios import REPORT_DATE
from tests.helpers import (
    OPP,
    PAN_NTC,
    T0,
    Api,
    ApiFactory,
    assert_no_leak,
    ist,
    pan_hash_of,
    seed_billable_bureau_calls,
)

WARNING_ON_T0: Final[str] = (
    "No bureau file was found on 2026-10-10; a full report will most likely return nothing. Proceed?"
)
L2_AT_30K: Final[tuple[str, int, dict[str, int]]] = ("L2", 395000, {"lower": 147000, "upper": 395000})


def _ntc_world(sink: CallLogSink) -> FakeSurePassClient:
    return FakeSurePassClient(
        pan={PAN_NTC: env.pan_comprehensive_envelope(pan=PAN_NTC)},
        score={PAN_NTC: env.experian_score_envelope(credit_score="")},
        report={PAN_NTC: env.experian_report_envelope(accounts=[], credit_score="", report_date=REPORT_DATE)},
        sink=sink,
    )


def _ntc_quick(api: Api) -> None:
    body = api.quick(pan=PAN_NTC, monthly_income=30000)
    assert (body["status"], body["headline"]) == ("ok", "unlikely")
    score = api.repo.get_score_cache(pan_hash_of(PAN_NTC))
    assert score is not None and score.is_ntc is True


def _full(api: Api, **over: Any) -> dict[str, Any]:
    return api.full(pan=PAN_NTC, monthly_income=30000, **over)


def test_full_check_after_a_recent_ntc_warns_then_the_acknowledged_recheck_pulls_the_report(make_api: ApiFactory) -> None:
    api = make_api(_ntc_world)
    _ntc_quick(api)
    api.clock.advance(timedelta(days=5))

    warned = _full(api)

    assert {k: v for k, v in warned.items() if k != "check_id"} == {
        "kind": "full",
        "status": "refused",
        "code": "WARN_NTC_RECENT",
        "headline": None,
        "label": WARNING_ON_T0,  # the NTC pull's date, not today's (2026-10-15)
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
        "quick_checks_left_today": None,
    }
    assert warned["label"] == REFUSAL_MESSAGES["WARN_NTC_RECENT"].format(date="2026-10-10")
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]  # zero report calls
    row = api.row(warned["check_id"])
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("refused", "WARN_NTC_RECENT", False, False)
    assert api.calls(row.id) == []
    assert api.repo.pan_events_for_opportunity(OPP) == []  # a refused attempt never marks the opportunity's PAN
    assert_no_leak(warned)  # 'bureau' is mandated copy here: the label is scanned on the global tier only

    acknowledged = _full(api, acknowledge_ntc_warning=True)

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT]  # PAN reused (< 180 days)
    assert (acknowledged["status"], acknowledged["code"], acknowledged["from_cache"]) == ("ok", None, False)
    assert (acknowledged["headline"], acknowledged["coapplicant_suggested"]) == ("unlikely", True)
    top = acknowledged["top_lender"]
    assert (top["id"], top["ceiling"], top["range"]) == L2_AT_30K
    assert acknowledged["checked_on"] == "2026-10-15"
    ack_row = api.row(acknowledged["check_id"])
    assert (ack_row.billed, ack_row.counted_against_quota) == (True, False)
    assert ack_row.billed == any(api.billable(ack_row.id))
    assert [c.endpoint for c in api.calls(ack_row.id)] == [REPORT_ENDPOINT]
    assert [e.event for e in api.repo.pan_events_for_opportunity(OPP)] == ["initial"]
    assert_no_leak(acknowledged)


def test_the_warning_date_is_the_ist_date_of_the_ntc_pull(make_api: ApiFactory) -> None:
    """02:00 IST on 11 Oct is 20:30 UTC on 10 Oct: the LC sees the IST date (a UTC date would be a day early)."""
    api = make_api(_ntc_world)
    api.clock.set_now(ist(2026, 10, 11, 2, 0))
    _ntc_quick(api)
    api.clock.advance(timedelta(days=1))

    warned = _full(api)

    assert warned["code"] == "WARN_NTC_RECENT"
    assert warned["label"] == (
        "No bureau file was found on 2026-10-11; a full report will most likely return nothing. Proceed?"
    )
    assert_no_leak(warned)


@pytest.mark.parametrize(
    ("age", "warns"),
    [
        pytest.param(timedelta(days=30) - timedelta(minutes=1), True, id="30d-minus-1min"),
        pytest.param(timedelta(days=30), False, id="30d-exactly"),
        pytest.param(timedelta(days=31), False, id="31d"),
    ],
)
def test_the_warning_lasts_exactly_as_long_as_the_ntc_score_is_fresh(
    make_api: ApiFactory, age: timedelta, warns: bool
) -> None:
    api = make_api(_ntc_world)
    _ntc_quick(api)
    api.clock.set_now(T0 + age)

    body = _full(api)

    if warns:
        assert (body["status"], body["code"], body["label"]) == ("refused", "WARN_NTC_RECENT", WARNING_ON_T0)
        assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    else:
        assert (body["status"], body["code"]) == ("ok", None)
        assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT]
        assert api.row(body["check_id"]).billed is True
    assert_no_leak(body)


def test_the_per_pan_daily_cap_is_checked_before_the_ntc_warning(make_api: ApiFactory) -> None:
    """Admission precedence (FL5 before FL7): a capped PAN is PAN_DAILY_CAP even with a recent NTC on file."""
    api = make_api(_ntc_world)
    _ntc_quick(api)  # bureau call 1 today
    seed_billable_bureau_calls(api.repo, pan=PAN_NTC, n=2, now=api.clock.now())

    body = _full(api)

    assert (body["status"], body["code"]) == ("refused", "PAN_DAILY_CAP")
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert len(api.reviews(PAN_NTC, "pan_daily_cap")) == 1
    assert_no_leak(body)
