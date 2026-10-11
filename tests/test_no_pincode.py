"""PRD 12 item 14 + PRD 9.1 / K15: a PAN record without a pincode gives the band read, ``NO_PINCODE_ON_PAN``.

The gate has no candidate pincode, so the check stops after derive: no lender is matched, the ranking model is NEVER
called and safety never runs (rank and safety are skipped, K15), confidence is None (not computed at all), and the
headline is band-derived: NTC / below 650 -> 'unlikely' with the co-applicant nudge, 650 and up -> 'possible'. The
Score pull itself happened, so the quick run is billed AND counted (K8). A full check with no pincode on the PAN and
none on the bureau side is the same band read (K22); with a bureau-side pincode the full check matches normally.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from eligibility_agent.output.lc_view import MSG_NO_PINCODE
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.scenarios import REPORT_DATE, no_pincode_scenario
from tests.helpers import (
    LC_A,
    LC_B,
    OMIT,
    PAN_OK,
    Api,
    ApiFactory,
    SurePassFactory,
    assert_no_leak,
    internal_of,
)


def _no_pin_world(credit_score: str) -> SurePassFactory:
    """PAN zip '' + a Score with ``credit_score``: the quick band-read world for any band."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, zip_code="")},
            score={PAN_OK: env.experian_score_envelope(credit_score=credit_score)},
            sink=sink,
        )

    return factory


def _assert_band_read_internal(api: Api, body: dict[str, Any]) -> None:
    """Rank and safety never ran: no model call, no gate survivors, no numbers, no confidence anywhere."""
    assert api.rank.calls == []
    internal = internal_of(api.row(body["check_id"]))
    assert (internal["gated_lender_ids"], internal["lender_numbers"]) == ([], {})
    assert (internal["code_confidence"], internal["model_output"], internal["safety"]) == (None, None, None)
    assert (internal["final_confidence"], internal["rank_attempts"]) == (None, 0)
    assert internal["derived"]["candidate_pincodes"] == []
    assert "rank" not in internal["transcript"] and "safety" not in internal["transcript"]
    assert api.repo.list_reviews() == []  # a band read is not a review case


def test_quick_check_without_a_pan_pincode_is_the_band_read(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: no_pincode_scenario(sink=sink))

    body = api.quick()

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert {k: v for k, v in body.items() if k != "check_id"} == {
        "kind": "quick",
        "status": "ok",
        "code": "NO_PINCODE_ON_PAN",
        "headline": "possible",  # band 720-780
        "label": MSG_NO_PINCODE,
        "confidence": None,
        "top_lender": None,
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
    row = api.row(body["check_id"])
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("ok", "NO_PINCODE_ON_PAN", True, True)
    assert row.billed == any(api.billable(row.id))
    assert api.repo.quick_checks_used_today(LC_A) == 1
    _assert_band_read_internal(api, body)
    assert_no_leak(body)


@pytest.mark.parametrize(
    ("credit_score", "band", "headline", "nudge"),
    [
        pytest.param("745", "720-780", "possible", False, id="745"),
        pytest.param("650", "650-720", "possible", False, id="650-edge"),
        pytest.param("649", "<650", "unlikely", True, id="649-edge"),
        pytest.param("", "NTC", "unlikely", True, id="ntc"),
    ],
)
def test_the_band_read_headline_follows_the_band(
    make_api: ApiFactory, credit_score: str, band: str, headline: str, nudge: bool
) -> None:
    api = make_api(_no_pin_world(credit_score))

    body = api.quick()

    assert (body["code"], body["confidence"], body["headline"]) == ("NO_PINCODE_ON_PAN", None, headline)
    assert (body["coapplicant_suggested"], body["coapplicant_reason"]) == (nudge, None)  # no model, so no reason
    assert (body["top_lender"], body["alternates"]) == (None, [])
    assert internal_of(api.row(body["check_id"]))["derived"]["band"] == band
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, True)  # an NTC no-hit counts too
    _assert_band_read_internal(api, body)
    assert_no_leak(body)


def test_a_cached_no_pincode_pan_is_still_a_band_read_and_free(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: no_pincode_scenario(sink=sink))
    api.quick(lc_id=LC_A)
    api.clock.advance(timedelta(days=2))

    body = api.quick(lc_id=LC_B)

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]  # zero new calls
    assert (body["code"], body["from_cache"], body["checked_on"]) == ("NO_PINCODE_ON_PAN", True, "2026-10-10")
    assert body["quick_checks_left_today"] == 5
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (False, False)
    _assert_band_read_internal(api, body)
    assert_no_leak(body)


def test_full_check_with_no_pincode_on_either_side_is_the_band_read(make_api: ApiFactory) -> None:
    """K22: the report's holder addresses carry no pincode either, so the full check is the same band read."""
    api = make_api(lambda sink: no_pincode_scenario(sink=sink))

    body = api.full()

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert (body["kind"], body["status"], body["code"], body["label"]) == ("full", "ok", "NO_PINCODE_ON_PAN", MSG_NO_PINCODE)
    assert (body["headline"], body["confidence"], body["top_lender"], body["alternates"]) == ("possible", None, None, [])
    assert body["quick_checks_left_today"] is None
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, False)  # full checks never count
    _assert_band_read_internal(api, body)
    assert_no_leak(body)


def test_a_full_band_read_keeps_the_stale_account_confirm_item(make_api: ApiFactory) -> None:
    """Fix round 1 ruling (PRD 8.3): confirm items are code-built before the gate, so the band read shows them. The
    report's accounts carry no pincode (a fresh reported one, and a stale one last updated 2026-06-01)."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        accounts = [
            env.cais_account(
                scheduled_emi="3500", current_balance="150000", date_reported="20260920", holder_zip="",
                account_number="SYNTHACC0001",
            ),
            env.cais_account(
                scheduled_emi="2000", current_balance="70000", date_reported="20260601", holder_zip="",
                account_number="SYNTHACC0002",
            ),
        ]
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, zip_code="")},
            report={PAN_OK: env.experian_report_envelope(accounts=accounts, report_date=REPORT_DATE)},
            sink=sink,
        )

    api = make_api(factory)

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    assert (body["status"], body["code"], body["confidence"], body["headline"]) == (
        "ok",
        "NO_PINCODE_ON_PAN",
        None,
        "possible",
    )
    assert body["items_to_confirm"] == ["confirm with learner: an account last updated 2026-06-01"]
    assert (body["top_lender"], body["alternates"]) == (None, [])
    _assert_band_read_internal(api, body)
    assert_no_leak(body)


def test_a_bureau_side_pincode_lets_the_full_check_match_where_the_quick_could_not(make_api: ApiFactory) -> None:
    """PRD 8.6: quick uses the PAN pincode only; full uses PAN + bureau addresses (12 months). The PAN has none, the
    report's one account (EMI 3,500) carries 560001: the full check gates L1 / L2 and prices them normally (dossier money:
    L1 1,141,000 / 329,000 at income 60,000)."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        account = env.cais_account(
            scheduled_emi="3500", current_balance="150000", date_reported="20260920", holder_zip="560001",
            account_number="SYNTHACC0001",
        )
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, zip_code="")},
            score={PAN_OK: env.experian_score_envelope(credit_score="745")},
            report={PAN_OK: env.experian_report_envelope(accounts=[account], report_date=REPORT_DATE)},
            sink=sink,
        )

    api = make_api(factory)
    quick = api.quick()
    assert quick["code"] == "NO_PINCODE_ON_PAN"
    assert api.rank.call_count == 0

    full = api.full(monthly_income=60000, existing_emi=OMIT)

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT]
    assert (full["status"], full["code"], full["confidence"], full["headline"]) == ("ok", None, "high", "fundable")
    top = full["top_lender"]
    assert (top["id"], top["ceiling"], top["range"]) == ("L1", 1141000, {"lower": 329000, "upper": 1141000})
    assert [a["id"] for a in full["alternates"]] == ["L2"]
    assert internal_of(api.row(full["check_id"]))["derived"]["candidate_pincodes"] == ["560001"]
    assert api.rank.call_count == 1
    assert_no_leak(quick)
    assert_no_leak(full)
