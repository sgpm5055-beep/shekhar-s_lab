"""PRD 12 item 1: quick happy path (PAN then Score once, billed, counted, no leak), end to end through the API.

World: the K17 seed lenders (L1 + L2 serve 560001, L3 serves 400001 only), happy_quick_scenario (person PAN, zip 560001,
score 745 -> band 720-780), income 60,000, EMI 0, T0 = 2026-10-10 10:00 IST.

Money pins are hand-written from the dossier tables (never recomputed from the code): L1 at available EMI
0.55 x 60,000 = 33,000 -> raw 367,536.02 / 1,276,816.90 -> half-up 368,000 / 1,277,000; L2 at 0.45 x 60,000 = 27,000 ->
raw upper 789,990.42 clipped to its 5,00,000 cap, raw lower 294,502.64 -> 295,000.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Final

import pytest

from eligibility_agent.output.lc_view import LABEL_QUICK, LABEL_QUICK_NO_EMI, MSG_INTERNAL, line_leaks
from eligibility_agent.surepass.client import PAN_ENDPOINT, SCORE_ENDPOINT, FakeSurePassClient
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.scenarios import happy_quick_scenario
from tests.helpers import (
    LC_A,
    LC_TYPED_NAME,
    MOBILE_OK,
    OMIT,
    PAN_FULL_NAME,
    PAN_OK,
    T0,
    Api,
    ApiFactory,
    assert_no_leak,
    internal_of,
    leak_findings,
    mobile_hash_of,
    one_check_row,
    pan_hash_of,
    quick_payload,
    seed_cached_quick_world,
)

L1_VIEW: Final[dict[str, Any]] = {
    "id": "L1",
    "name": "Alpha Education Finance",
    "rate_band": "11 to 14%",
    "tenure_range": "12 to 48 months",
    "ceiling": 1277000,
    "range": {"lower": 368000, "upper": 1277000},
    "conditions": [],
}
L2_VIEW: Final[dict[str, Any]] = {
    "id": "L2",
    "name": "Bharat Upskill Credit",
    "rate_band": "14 to 18%",
    "tenure_range": "12 to 36 months",
    "ceiling": 500000,
    "range": {"lower": 295000, "upper": 500000},
    "conditions": [],
}


def _happy(make_api: ApiFactory) -> Api:
    return make_api(lambda sink: happy_quick_scenario(sink=sink))


def _without_rationale(view: dict[str, Any]) -> dict[str, Any]:
    rationale = view["rationale"]
    assert isinstance(rationale, str) and rationale, "a ranked pick carries the model's scrubbed rationale"
    assert not line_leaks(rationale), rationale
    return {k: v for k, v in view.items() if k != "rationale"}


def test_quick_happy_path_view(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    body = api.quick()

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert {k: v for k, v in body.items() if k not in ("check_id", "top_lender", "alternates")} == {
        "kind": "quick",
        "status": "ok",
        "code": None,
        "headline": "possible",  # quick checks cap at medium (K16) -> 'possible'
        "label": LABEL_QUICK,
        "confidence": "medium",
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-10",
        "from_cache": False,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": 4,
    }
    assert _without_rationale(body["top_lender"]) == L1_VIEW
    assert [_without_rationale(a) for a in body["alternates"]] == [L2_VIEW]
    assert_no_leak(body)
    assert api.rank.call_count == 1


def test_bureau_call_sends_pan_full_name_not_the_lc_typed_name(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    api.quick()

    pan_call, score_call = api.surepass.calls
    assert pan_call.body == {
        "id_number": PAN_OK,
        "get_address": True,
        "get_contact": False,
        "masked_aadhaar_variant": "v1",
    }
    assert score_call.body == {"name": PAN_FULL_NAME, "mobile": MOBILE_OK, "pan": PAN_OK, "consent": "Y"}
    assert PAN_FULL_NAME.casefold() != LC_TYPED_NAME.casefold()  # the pin above would be vacuous otherwise
    assert LC_TYPED_NAME not in str([call.body for call in api.surepass.calls])


@pytest.mark.parametrize("full_name", ["", "   "])
@pytest.mark.parametrize("path", ["stored_pan_record", "fresh_pan_call"])
def test_bureau_call_never_falls_back_to_the_lc_typed_name(make_api: ApiFactory, path: str, full_name: str) -> None:
    """PRD 6: the LC-typed name lives ONLY in the consent record. A PAN record without a usable full_name (stored or
    just pulled) ends the check in the error view BEFORE any bureau call; the typed name reaches no request body."""
    if path == "stored_pan_record":
        api = _happy(make_api)
        seed_cached_quick_world(api.repo, pan=PAN_OK, fetched_at=T0 - timedelta(days=2), with_score=False,
                                full_name=full_name)
    else:
        api = make_api(
            lambda sink: FakeSurePassClient(
                pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, full_name=full_name)},
                score={PAN_OK: env.experian_score_envelope(credit_score="745")},
                sink=sink,
            )
        )

    body = api.quick()

    assert (body["status"], body["code"], body["label"]) == ("service_failure", None, MSG_INTERNAL)
    assert SCORE_ENDPOINT not in api.endpoints()  # ZERO bureau calls
    assert api.endpoints() == ([] if path == "stored_pan_record" else [PAN_ENDPOINT])
    assert all(LC_TYPED_NAME not in str(call.body) for call in api.surepass.calls)
    row = one_check_row(api.repo)
    assert (row.status, row.counted_against_quota) == ("error", False)
    assert row.billed == any(api.billable(row.id))  # the PAN call's Rs 1 on the fresh path, nothing on the stored one
    assert api.active_lock_kinds(PAN_OK) == []  # the boundary released the in-flight lock
    assert api.repo.get_score_cache(pan_hash_of(PAN_OK)) is None


def test_quick_happy_path_persistence(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    body = api.quick()

    row = one_check_row(api.repo)
    calls = api.calls(row.id)
    assert [c.endpoint for c in calls] == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert [c.billable for c in calls] == [True, True]
    assert (row.id, row.kind, row.status, row.code) == (body["check_id"], "quick", "ok", None)
    assert row.billed is True and row.billed == any(c.billable for c in calls)
    assert (row.counted_against_quota, row.is_refresh) == (True, False)
    assert row.lc_view == body
    assert api.repo.quick_checks_used_today(LC_A) == 1

    pan_hash = pan_hash_of(PAN_OK)
    score = api.repo.get_score_cache(pan_hash)
    assert score is not None
    assert (score.source, score.is_ntc, score.expires_at) == ("score", False, T0 + timedelta(days=30))
    assert score.consent_id == row.consent_id
    pan_record = api.repo.get_pan_record(pan_hash)
    assert pan_record is not None and pan_record.expires_at == T0 + timedelta(days=180)

    (consent,) = api.repo.list_consents()
    assert consent.id == row.consent_id
    assert (consent.covers, consent.wording_version, consent.call_ref, consent.ip) == (
        "score",
        "v2-quick",
        "sc_call_123",
        "10.0.0.1",
    )
    assert (consent.pan_hash, consent.mobile_hash, consent.lc_id) == (pan_hash, mobile_hash_of(MOBILE_OK), LC_A)
    assert consent.lc_typed_name == LC_TYPED_NAME  # the typed name lives ONLY in the consent record
    assert api.active_lock_kinds(PAN_OK) == []  # the in-flight lock was released
    # (called_at, id) order: under the frozen clock both rows share called_at, so compare order-insensitively
    recorded = {c["endpoint"]: c for c in internal_of(row)["surepass_calls"]}
    assert len(internal_of(row)["surepass_calls"]) == 2 and recorded == {
        PAN_ENDPOINT: {"endpoint": PAN_ENDPOINT, "billable": True, "error_kind": None},
        SCORE_ENDPOINT: {"endpoint": SCORE_ENDPOINT, "billable": True, "error_kind": None},
    }


def test_quick_without_income_shows_no_ceiling_or_range_anywhere(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    body = api.quick(monthly_income=OMIT, existing_emi=OMIT)

    assert (body["status"], body["code"], body["label"]) == ("ok", None, LABEL_QUICK)
    views = [body["top_lender"], *body["alternates"]]
    assert [v["id"] for v in views] == ["L1", "L2"]  # the picks are still shown, just without amounts
    assert all(v["ceiling"] is None and v["range"] is None for v in views)
    internal = internal_of(one_check_row(api.repo))
    assert internal["derived"]["current_foir"] is None
    assert all(n["displayed_foir"] is None and n["ceiling"] is None for n in internal["lender_numbers"].values())
    assert_no_leak(body)


def test_quick_income_without_emi_assumes_no_other_emis(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    body = api.quick(existing_emi=OMIT)

    assert body["label"] == LABEL_QUICK_NO_EMI
    assert _without_rationale(body["top_lender"]) == L1_VIEW  # EMI 0 used: the same money as the happy path
    assert [_without_rationale(a) for a in body["alternates"]] == [L2_VIEW]
    emi = internal_of(one_check_row(api.repo))["derived"]["emi"]
    assert (emi["used"], emi["used_source"]) == (0.0, "declared_zero_assumed")


@pytest.mark.parametrize("income", [0, -1])
def test_non_positive_income_is_a_schema_error_and_records_nothing(make_api: ApiFactory, income: int) -> None:
    """Declared income is strictly positive at the API (T12): the no-income variant OMITS the field instead."""
    api = _happy(make_api)
    response = api.post_quick(quick_payload(monthly_income=income))

    assert response.status_code == 422
    assert api.repo.list_checks() == [] and api.repo.list_consents() == [] and api.endpoints() == []


def test_leak_scanner_catches_plants_and_skips_only_the_id_fields() -> None:
    """Positive control for ``assert_no_leak``: a scanner that finds nothing on planted leaks proves nothing."""
    planted: dict[str, Any] = {
        "check_id": "7450b1c2-0000-4000-8000-123456789012",  # uuid hex trips the patterns structurally: exempt
        "label": "Experian says hello",
        "items_to_confirm": ["a/c no: SYNTH0001"],
        "coapplicant_reason": "the bureau file is thin",  # generic term: banned in LLM prose only
        "top_lender": {"id": "745", "name": "fine", "rationale": "score 745 is good", "conditions": ["call 9876543210"]},
        "alternates": [{"id": "L2", "name": "Ravi Testkumar Finance"}],
    }
    findings = leak_findings(planted)
    assert set(findings) == {
        "label: global tier",
        "items_to_confirm.0: global tier",
        "items_to_confirm.0: llm tier",
        "coapplicant_reason: llm tier",
        "top_lender.rationale: global tier",
        "top_lender.rationale: llm tier",
        "top_lender.conditions.0: global tier",
        "top_lender.conditions.0: llm tier",
        "top_lender.conditions.0: identity literal",
        "alternates.0.name: identity literal",
    }
