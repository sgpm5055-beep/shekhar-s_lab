"""PRD 12 items 8 and 9: the full-check stage gate, and the full check at Offer (PAN reused under 180 days, Report pulled
once, the Score cache refreshed by it, one report per PAN per 30 days shared across opportunities, refresh only after
the window).

World: full_happy_reported_emi_scenario (ONE open account with a reported EMI of 3,500, ReportDate 2026-10-01, score 745,
zip 560001). Money pins from the dossier: income 60,000 -> L1 available 0.55 x 60,000 - 3,500 = 29,500 -> raw
1,141,396.92 / 328,554.93 -> 1,141,000 / 329,000; L2 0.45 x 60,000 - 3,500 = 23,500 -> upper clipped to 5,00,000, raw
lower 256,326.37 -> 256,000. Income 80,000 (second opportunity) is hand-derived by linearity from the dossier's L1@33,000
and L2@27,000 floats: L1 40,500 -> upper 1,567,002.56 clipped to 15,00,000, lower 451,067.84 -> 451,000; L2 32,500 ->
upper clipped to 5,00,000, lower 354,494.07 -> 354,000.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Final

import pytest

from eligibility_agent.output.lc_view import LABEL_QUICK, REFUSAL_MESSAGES
from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT
from eligibility_agent.testing.scenarios import full_happy_reported_emi_scenario
from tests.helpers import (
    MOBILE_OK,
    OPP,
    OPP_TWO,
    PAN_FULL_NAME,
    PAN_OK,
    T0,
    Api,
    ApiFactory,
    assert_no_leak,
    full_payload,
    internal_of,
    pan_hash_of,
    quick_and_full_client,
)

FULL_LABEL: Final[str] = "bureau report on 2026-10-01, income as declared"  # ReportDate of the synthetic report


def _full_world(make_api: ApiFactory) -> Api:
    return make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))


def _money(view: dict[str, Any]) -> tuple[str, int | None, dict[str, int] | None]:
    return view["id"], view["ceiling"], view["range"]


# --------------------------------------------------------------------------- item 8: the stage gate


@pytest.mark.parametrize("stage", ["new", "contacted", "qualified", "counselling"])
def test_full_check_before_offer_is_refused_stage_too_early(make_api: ApiFactory, stage: str) -> None:
    api = _full_world(make_api)

    body = api.full(opportunity_stage=stage)

    assert (body["status"], body["code"], body["label"]) == (
        "refused",
        "STAGE_TOO_EARLY",
        REFUSAL_MESSAGES["STAGE_TOO_EARLY"],
    )
    assert (body["kind"], body["quick_checks_left_today"], body["headline"]) == ("full", None, None)
    assert api.endpoints() == []
    row = api.row(body["check_id"])
    assert (row.status, row.billed, row.counted_against_quota, row.opportunity_stage) == ("refused", False, False, stage)
    assert api.repo.pan_events_for_opportunity(OPP) == []  # a refused attempt never marks the opportunity's PAN
    assert_no_leak(body)


@pytest.mark.parametrize("stage", ["offer", "registration", "enrolled", "won"])
def test_full_check_at_offer_or_beyond_is_admitted(make_api: ApiFactory, stage: str) -> None:
    api = _full_world(make_api)

    body = api.full(opportunity_stage=stage)

    assert (body["status"], body["code"], body["label"]) == ("ok", None, FULL_LABEL)
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota, row.opportunity_stage) == (True, False, stage)


@pytest.mark.parametrize("stage", ["prospect", "Offer", ""])
def test_unknown_stage_is_a_schema_error_and_records_nothing(make_api: ApiFactory, stage: str) -> None:
    api = _full_world(make_api)

    response = api.post_full(full_payload(opportunity_stage=stage))

    assert response.status_code == 422
    assert api.repo.list_checks() == [] and api.repo.list_consents() == [] and api.endpoints() == []


# --------------------------------------------------------------------------- item 9: the full check at Offer


def test_full_at_offer_reuses_the_pan_verified_10_days_ago_and_pulls_the_report_once(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: quick_and_full_client(sink))
    api.quick()  # verifies the PAN (and pulls a score) on 2026-10-10
    api.clock.advance(timedelta(days=10))

    body = api.full()

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT]  # PAN reused (< 180 days)
    assert {k: v for k, v in body.items() if k not in ("check_id", "top_lender", "alternates")} == {
        "kind": "full",
        "status": "ok",
        "code": None,
        "headline": "fundable",
        "label": FULL_LABEL,
        "confidence": "high",
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-20",
        "from_cache": False,
        "refresh_available_on": "2026-11-19",
        "quick_checks_left_today": None,  # full checks never touch the quick quota
    }
    assert _money(body["top_lender"]) == ("L1", 1141000, {"lower": 329000, "upper": 1141000})
    assert [_money(a) for a in body["alternates"]] == [("L2", 500000, {"lower": 256000, "upper": 500000})]

    row = api.row(body["check_id"])
    assert [c.endpoint for c in api.calls(row.id)] == [REPORT_ENDPOINT]
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (True, False, False)
    assert row.billed == any(api.billable(row.id))
    # the Report request carries the STORED PAN record's full_name (reuse path), never the LC-typed name
    report_call = api.surepass.calls[-1]
    assert (report_call.endpoint, report_call.body) == (
        REPORT_ENDPOINT,
        {"name": PAN_FULL_NAME, "mobile": MOBILE_OK, "pan": PAN_OK, "consent": "Y"},
    )
    consent = api.consent(row.id)
    assert (consent.covers, consent.wording_version, consent.opportunity_id) == ("report", "v2-full", OPP)

    pan_hash = pan_hash_of(PAN_OK)
    report = api.repo.get_report_cache(pan_hash)
    assert report is not None
    assert (report.report_date, report.is_ntc) == (date(2026, 10, 1), False)
    assert report.expires_at == T0 + timedelta(days=40)
    score = api.repo.get_score_cache(pan_hash)
    assert score is not None
    assert (score.source, score.storage_key, score.fetched_at) == ("report", report.storage_key, T0 + timedelta(days=10))
    assert_no_leak(body)


def test_report_pull_refreshes_the_score_cache_so_the_next_quick_is_free(make_api: ApiFactory) -> None:
    """PRD 3 / 5.3: any Report pull refreshes the Score cache; the quick check is then served from the report's raw
    (score_cache source='report'). The world has NO Score endpoint: a Score call would fail the run."""
    api = _full_world(make_api)
    api.full()

    quick = api.quick()

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]  # zero calls for the quick check
    assert (quick["status"], quick["from_cache"], quick["label"]) == ("ok", True, LABEL_QUICK)
    assert (quick["checked_on"], quick["refresh_available_on"], quick["quick_checks_left_today"]) == (
        "2026-10-10",
        "2026-11-09",
        5,
    )
    # quick money runs on the DECLARED figures only (income 60,000, EMI 0), never the report's accounts
    assert _money(quick["top_lender"]) == ("L1", 1277000, {"lower": 368000, "upper": 1277000})
    row = api.row(quick["check_id"])
    assert (row.billed, row.counted_against_quota) == (False, False)
    assert internal_of(row)["score"]["source"] == "report"
    assert_no_leak(quick)


def test_second_opportunity_on_the_same_pan_inside_30_days_is_a_free_hit(make_api: ApiFactory) -> None:
    """One report per PAN per 30 days, NOT per opportunity (PRD 5.9): opportunity 2 shares it and re-derives on its own
    declared income."""
    api = _full_world(make_api)
    first = api.full()
    api.clock.advance(timedelta(days=5))

    second = api.full(opportunity_id=OPP_TWO, monthly_income=80000)

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert (second["status"], second["from_cache"], second["label"]) == ("ok", True, FULL_LABEL)
    assert (second["checked_on"], second["refresh_available_on"]) == ("2026-10-10", "2026-11-09")
    row = api.row(second["check_id"])
    assert (row.billed, row.counted_against_quota, row.opportunity_id) == (False, False, OPP_TWO)
    assert api.calls(row.id) == []
    assert _money(second["top_lender"]) == ("L1", 1500000, {"lower": 451000, "upper": 1500000})
    assert [_money(a) for a in second["alternates"]] == [("L2", 500000, {"lower": 354000, "upper": 500000})]
    available = [internal_of(api.row(c["check_id"]))["lender_numbers"]["L1"]["available_emi"] for c in (first, second)]
    assert available == [pytest.approx(29500.0), pytest.approx(40500.0)]
    assert [e.event for e in api.repo.pan_events_for_opportunity(OPP_TWO)] == ["initial"]
    assert_no_leak(second)


def test_refresh_inside_the_report_window_is_refused_then_billed_after_31_days(make_api: ApiFactory) -> None:
    api = _full_world(make_api)
    api.full()
    api.clock.advance(timedelta(days=10))

    reuse = api.full()  # same opportunity, no refresh: the T11 report-reuse cache hit, end to end
    assert (reuse["from_cache"], reuse["checked_on"]) == (True, "2026-10-10")

    refused = api.full(refresh=True)

    assert (refused["status"], refused["code"], refused["label"]) == (
        "refused",
        "REPORT_WINDOW_OPEN",
        REFUSAL_MESSAGES["REPORT_WINDOW_OPEN"],
    )
    assert refused["refresh_available_on"] == "2026-11-09"  # the stored report's pull date + 30 days
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert (api.row(refused["check_id"]).billed, api.row(refused["check_id"]).is_refresh) == (False, False)

    api.clock.advance(timedelta(days=21))  # 31 days after the pull
    refreshed = api.full(refresh=True)

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT, REPORT_ENDPOINT]  # PAN still reused
    assert (refreshed["status"], refreshed["from_cache"], refreshed["checked_on"]) == ("ok", False, "2026-11-10")
    row = api.row(refreshed["check_id"])
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (True, False, True)
    assert row.billed == any(api.billable(row.id))
    assert api.consent(row.id).covers == "report"  # a consent record for THIS pull
    assert api.surepass.calls[-1].body["name"] == PAN_FULL_NAME
    assert_no_leak(refreshed)


def test_pan_record_is_reused_under_180_days_and_reverified_after(make_api: ApiFactory) -> None:
    """PRD 5.3: PAN window 180 days. Day 179: the expired report is re-pulled (billed refresh, PAN reused). Day 181: the
    report is fresh again but the PAN record is not, so the hit re-verifies the PAN (a Rs 1 call) and pulls no report."""
    api = _full_world(make_api)
    api.full()

    api.clock.set_now(T0 + timedelta(days=179))
    refresh = api.full()
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT, REPORT_ENDPOINT]
    assert api.row(refresh["check_id"]).is_refresh is True

    api.clock.set_now(T0 + timedelta(days=181))
    hit = api.full()

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT, REPORT_ENDPOINT, PAN_ENDPOINT]
    assert (hit["status"], hit["from_cache"]) == ("ok", True)
    row = api.row(hit["check_id"])
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (True, False, False)  # the PAN call is billed
    pan_record = api.repo.get_pan_record(pan_hash_of(PAN_OK))
    assert pan_record is not None and pan_record.fetched_at == T0 + timedelta(days=181)
