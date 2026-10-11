"""PRD 12 item 20 + PRD 6 / K9: recompute and attach, end to end through the API.

RECOMPUTE (``POST /checks/{id}/recompute``): a NEW checks row (new check_id, so a fresh checkpointer thread) with the
first-class ``recompute_of`` column, re-derived from the STORED raw objects on the new declared figures and the current
published lenders; ``billed`` False, ``counted_against_quota`` False, ``from_cache`` True, the ORIGINAL consent_id and the
original pull's display dates. Zero SurePass calls is structural: the recompute graph has no pull node, and here the
SurePass seam is switched to a fake with NO scenarios before the recompute (any call would raise and end the run in the
error view). The original row is left byte-for-byte untouched. Allowed only from ``ok`` or ``service_failure`` +
``OPENAI_FAILURE``; 404 ``check_not_found``; 409 ``not_recomputable`` / ``raw_unavailable``; a full-check recompute needs
an income (422 without one, T12).

ATTACH (``POST /checks/{id}/attach``), the A-table: quick checks only (409 ``attach_quick_only``), idempotent for the same
contact (a padded twin is the same contact), 409 ``already_attached`` for a different one, 404 for an unknown id; it sets
``checks.contact_id`` AND the check's ``consents.contact_id``.

Money oracles (hand-derived from the dossier floats by linearity; the annuity is linear in the available EMI):
* the original full check, income 80,000, reported EMI 3,500: L1 40,500 -> 15,00,000 cap / 451,000; L2 32,500 ->
  5,00,000 cap / 354,000 (as test_full_check's second opportunity);
* recompute income 50,000 + declared EMI 21,000 (higher than the reported 3,500, so used): L1 0.55 x 50,000 - 21,000 =
  6,500 -> raw 251,494.24 / 72,393.46 -> 251,000 / 72,000; L2 0.45 x 50,000 - 21,000 = 1,500 -> raw upper 43,888.36 ->
  44,000, raw lower 16,361.26 raised to L2's 25,000 minimum loan;
* quick recompute income 90,000, no EMI (test_cache_sharing's oracles): L1 15,00,000 cap / 551,000; L2 5,00,000 cap /
  442,000.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Final

import pytest

from eligibility_agent.output.lc_view import LABEL_QUICK_NO_EMI
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.fake_llm import FakeRankModel, default_ranking, invalid_output_error
from eligibility_agent.testing.scenarios import (
    full_happy_reported_emi_scenario,
    happy_quick_scenario,
    score_failure_scenario,
)
from scripts.purge_raw import run_purge
from tests.helpers import (
    LC_A,
    PAN_INVALID,
    PAN_OK,
    T0,
    Api,
    ApiFactory,
    SurePassFactory,
    SwitchableSurePass,
    assert_no_leak,
    internal_of,
    pan_hash_of,
)

FULL_LABEL: Final[str] = "bureau report on 2026-10-01, income as declared"
UNKNOWN_ID: Final[str] = "00000000-0000-4000-8000-000000000000"
NOT_FOUND: Final[dict[str, str]] = {"detail": "check_not_found"}


def _money(view: dict[str, Any]) -> tuple[str, int | None, dict[str, int] | None]:
    return view["id"], view["ceiling"], view["range"]


class _Seam:
    """A SwitchableSurePass the test can reach after the app is built (to cut SurePass off before a recompute)."""

    def __init__(self, first: SurePassFactory) -> None:
        self._first = first
        self.switchable: SwitchableSurePass | None = None
        self.sink: CallLogSink | None = None

    def __call__(self, sink: CallLogSink) -> SwitchableSurePass:
        first = self._first(sink)
        assert isinstance(first, FakeSurePassClient)
        self.sink, self.switchable = sink, SwitchableSurePass(first)
        return self.switchable

    def cut_off(self) -> None:
        """From now on any SurePass call raises AssertionError ('unexpected SurePass call'): no scenario is configured."""
        assert self.switchable is not None and self.sink is not None
        self.switchable.switch_to(FakeSurePassClient(sink=self.sink))


def _recompute(api: Api, check_id: str, body: dict[str, Any]) -> Any:
    return api.client.post(f"/checks/{check_id}/recompute", json=body)


def _attach(api: Api, check_id: str, contact_id: str) -> Any:
    return api.client.post(f"/checks/{check_id}/attach", json={"contact_id": contact_id})


# --------------------------------------------------------------------------- recompute (K9)


def test_recompute_of_a_full_check_is_a_new_row_rederived_with_zero_surepass_calls(make_api: ApiFactory) -> None:
    seam = _Seam(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    api = make_api(seam)
    original = api.full(monthly_income=80000)
    assert (original["confidence"], original["headline"]) == ("high", "fundable")
    assert _money(original["top_lender"]) == ("L1", 1500000, {"lower": 451000, "upper": 1500000})
    assert [_money(a) for a in original["alternates"]] == [("L2", 500000, {"lower": 354000, "upper": 500000})]
    original_id = original["check_id"]
    before = api.row(original_id)
    calls_before = api.repo.list_calls()
    consents_before = api.repo.list_consents()
    seam.cut_off()
    api.clock.advance(timedelta(days=2))

    response = _recompute(api, original_id, {"monthly_income": 50000, "existing_emi": 21000})

    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    assert body["check_id"] != original_id  # a NEW check (and so a fresh checkpointer thread)
    assert {k: v for k, v in body.items() if k not in ("check_id", "top_lender", "alternates")} == {
        "kind": "full",
        "status": "ok",
        "code": None,
        "headline": "possible",  # a higher declared EMI is in use: medium
        "label": FULL_LABEL,
        "confidence": "medium",
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-10",  # the ORIGINAL pull, not today (2026-10-12)
        "from_cache": True,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": None,
    }
    assert _money(body["top_lender"]) == ("L1", 251000, {"lower": 72000, "upper": 251000})
    assert [_money(a) for a in body["alternates"]] == [("L2", 44000, {"lower": 25000, "upper": 44000})]
    assert_no_leak(body)

    # zero SurePass calls: nothing reached the (scenario-less) seam and no audit row was written
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert api.repo.list_calls() == calls_before
    assert api.calls(body["check_id"]) == []

    new = api.row(body["check_id"])
    assert (new.recompute_of, new.consent_id, new.kind, new.status, new.code) == (
        original_id,
        before.consent_id,
        "full",
        "ok",
        None,
    )
    assert (new.billed, new.counted_against_quota, new.is_refresh) == (False, False, False)
    assert (new.lc_id, new.lead_id, new.opportunity_id, new.pan_hash) == (
        before.lc_id,
        before.lead_id,
        before.opportunity_id,
        before.pan_hash,
    )
    assert new.lc_view == body
    internal = internal_of(new)
    assert internal["is_recompute"] is True
    assert (internal["admission"]["admission"], internal["admission"]["pan_plan"], internal["admission"]["bureau_plan"]) == (
        "cache_hit",
        "reuse",
        "reuse",
    )
    assert internal["original_consent_id"] == before.consent_id
    assert internal["surepass_calls"] == []
    assert internal["transcript"][0] == "derive"  # no admission, PAN or bureau node ran
    emi = internal["derived"]["emi"]
    assert (emi["used"], emi["used_source"]) == (pytest.approx(21000.0), "declared")
    assert internal["lender_numbers"]["L1"]["available_emi"] == pytest.approx(6500.0)

    # the original row is untouched, and nothing else was written for the recompute
    assert api.row(original_id) == before
    assert api.repo.list_consents() == consents_before  # no new consent: the original one rides along
    assert api.active_lock_kinds(PAN_OK) == []
    assert api.repo.list_reviews() == []
    assert api.rank.call_count == 2  # the ranking re-ran on the re-derived facts


def test_recompute_of_a_quick_check_never_touches_the_quota(make_api: ApiFactory) -> None:
    seam = _Seam(lambda sink: happy_quick_scenario(sink=sink))
    api = make_api(seam)
    original = api.quick()
    before = api.row(original["check_id"])
    seam.cut_off()

    response = _recompute(api, original["check_id"], {"monthly_income": 90000})

    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    assert (body["kind"], body["status"], body["from_cache"], body["label"]) == ("quick", "ok", True, LABEL_QUICK_NO_EMI)
    assert body["quick_checks_left_today"] is None  # a recompute view carries no quota counter
    assert _money(body["top_lender"]) == ("L1", 1500000, {"lower": 551000, "upper": 1500000})
    assert [_money(a) for a in body["alternates"]] == [("L2", 500000, {"lower": 442000, "upper": 500000})]
    new = api.row(body["check_id"])
    assert (new.recompute_of, new.consent_id, new.billed, new.counted_against_quota) == (
        original["check_id"],
        before.consent_id,
        False,
        False,
    )
    assert api.repo.quick_checks_used_today(LC_A) == 1  # only the original pull counted
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert api.row(original["check_id"]) == before
    assert_no_leak(body)


def test_recompute_is_allowed_after_an_openai_failure(make_api: ApiFactory) -> None:
    api = make_api(
        lambda sink: happy_quick_scenario(sink=sink),
        rank=FakeRankModel([invalid_output_error(), invalid_output_error(), default_ranking]),
    )
    failed = api.quick()
    assert (failed["status"], failed["code"]) == ("service_failure", "OPENAI_FAILURE")
    before = api.row(failed["check_id"])

    response = _recompute(api, failed["check_id"], {"monthly_income": 60000, "existing_emi": 0})

    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    assert (body["status"], body["code"], body["confidence"], body["headline"]) == ("ok", None, "medium", "possible")
    assert _money(body["top_lender"]) == ("L1", 1277000, {"lower": 368000, "upper": 1277000})
    assert api.rank.call_count == 3
    assert api.row(body["check_id"]).recompute_of == failed["check_id"]
    assert api.row(failed["check_id"]) == before  # still the OPENAI_FAILURE row
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert_no_leak(body)


def test_recompute_of_an_unknown_check_is_404(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: happy_quick_scenario(sink=sink))

    for check_id in (UNKNOWN_ID, "not-a-uuid"):
        response = _recompute(api, check_id, {"monthly_income": 60000})
        assert (response.status_code, response.json()) == (404, NOT_FOUND)
    assert api.repo.list_checks() == []


def _refused_full(make_api: ApiFactory) -> tuple[Api, str]:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    return api, api.full(opportunity_stage="qualified")["check_id"]  # STAGE_TOO_EARLY


def _pan_not_verified(make_api: ApiFactory) -> tuple[Api, str]:
    api = make_api(
        lambda sink: FakeSurePassClient(
            pan={PAN_INVALID: env.pan_comprehensive_envelope(pan=PAN_INVALID, status="invalid")}, sink=sink
        )
    )
    return api, api.quick(pan=PAN_INVALID)["check_id"]


def _surepass_failure(make_api: ApiFactory) -> tuple[Api, str]:
    api = make_api(lambda sink: score_failure_scenario("success_false", sink=sink))
    return api, api.quick()["check_id"]


def _pending_verification(make_api: ApiFactory) -> tuple[Api, str]:
    api = make_api(lambda sink: score_failure_scenario("timeout", sink=sink))
    return api, api.quick()["check_id"]


def _internal_error(make_api: ApiFactory) -> tuple[Api, str]:
    api = make_api(
        lambda sink: FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, full_name="")},
            score={PAN_OK: env.experian_score_envelope()},
            sink=sink,
        )
    )
    return api, api.quick()["check_id"]


@pytest.mark.parametrize(
    ("build", "status"),
    [
        pytest.param(_refused_full, "refused", id="refused"),
        pytest.param(_pan_not_verified, "pan_not_verified", id="pan_not_verified"),
        pytest.param(_surepass_failure, "service_failure", id="surepass_failure"),
        pytest.param(_pending_verification, "pending_verification", id="pending_verification"),
        pytest.param(_internal_error, "error", id="error"),
    ],
)
def test_only_ok_and_openai_failure_checks_are_recomputable(make_api: ApiFactory, build: Any, status: str) -> None:
    api, check_id = build(make_api)
    original = api.row(check_id)
    assert original.status == status
    checks_before, calls_before = api.repo.list_checks(), api.repo.list_calls()

    response = _recompute(api, check_id, {"monthly_income": 60000})

    assert (response.status_code, response.json()) == (409, {"detail": "not_recomputable"})
    assert api.repo.list_checks() == checks_before  # no recompute row
    assert api.repo.list_calls() == calls_before


def test_recompute_after_the_raw_object_is_purged_is_raw_unavailable(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    original = api.full()
    report = api.repo.get_report_cache(pan_hash_of(PAN_OK))
    assert report is not None
    api.repo.raw_store.delete_raw([report.storage_key])  # the object is gone, the cache row still points at it
    checks_before = api.repo.list_checks()

    response = _recompute(api, original["check_id"], {"monthly_income": 60000})

    assert (response.status_code, response.json()) == (409, {"detail": "raw_unavailable"})
    assert api.repo.list_checks() == checks_before


def test_recompute_after_the_retention_purge_is_raw_unavailable(make_api: ApiFactory) -> None:
    """The real purge job at day 181 deletes the raw objects AND the expired cache rows (PRD 11, 180-day retention)."""
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    original = api.full()
    api.clock.set_now(T0 + timedelta(days=40))  # an expired report is still recomputable: freshness is not required
    still_raw = _recompute(api, original["check_id"], {"monthly_income": 60000})
    assert still_raw.status_code == 200, still_raw.text
    assert_no_leak(still_raw.json())

    api.clock.set_now(T0 + timedelta(days=181))
    report = run_purge(api.repo, api.repo.raw_store, now=api.clock.now())
    assert report.objects >= 2 and report.cache_rows >= 2
    checks_before = api.repo.list_checks()

    response = _recompute(api, original["check_id"], {"monthly_income": 60000})

    assert (response.status_code, response.json()) == (409, {"detail": "raw_unavailable"})
    assert api.repo.list_checks() == checks_before


def test_an_expired_report_recomputes_with_its_original_dates(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    original = api.full()
    api.clock.set_now(T0 + timedelta(days=40))

    response = _recompute(api, original["check_id"], {"monthly_income": 60000})

    body: dict[str, Any] = response.json()
    assert (body["status"], body["from_cache"]) == ("ok", True)
    assert (body["checked_on"], body["refresh_available_on"]) == ("2026-10-10", "2026-11-09")
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert_no_leak(body)


def test_a_full_recompute_needs_an_income_and_a_quick_one_does_not(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink, with_score=True))
    full = api.full()
    quick = api.quick()
    checks_before = len(api.repo.list_checks())

    missing = _recompute(api, full["check_id"], {"existing_emi": 1000})
    assert missing.status_code == 422
    assert [error["loc"] for error in missing.json()["detail"]] == [["body", "monthly_income"]]
    assert len(api.repo.list_checks()) == checks_before

    no_income = _recompute(api, quick["check_id"], {})
    assert no_income.status_code == 200, no_income.text
    body: dict[str, Any] = no_income.json()
    views = [body["top_lender"], *body["alternates"]]
    assert views and all(v["ceiling"] is None and v["range"] is None for v in views)
    assert_no_leak(body)


# --------------------------------------------------------------------------- attach (the A-table)


def test_attach_matrix_on_a_quick_check(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: happy_quick_scenario(sink=sink))
    first = api.quick()
    other = api.quick()  # a cache hit with its own consent row: attach must not touch it
    check_id = first["check_id"]

    attached = _attach(api, check_id, "c_1")

    assert (attached.status_code, attached.json()) == (200, {"check_id": check_id, "contact_id": "c_1", "attached": True})
    assert api.row(check_id).contact_id == "c_1"
    assert api.consent(check_id).contact_id == "c_1"  # dual-table update

    for contact in ("c_1", "  c_1 "):  # idempotent; the padded twin is the same contact
        again = _attach(api, check_id, contact)
        assert (again.status_code, again.json()) == (200, {"check_id": check_id, "contact_id": "c_1", "attached": True})

    conflict = _attach(api, check_id, "c_2")

    assert (conflict.status_code, conflict.json()) == (409, {"detail": "already_attached"})
    assert (api.row(check_id).contact_id, api.consent(check_id).contact_id) == ("c_1", "c_1")
    assert (api.row(other["check_id"]).contact_id, api.consent(other["check_id"]).contact_id) == (None, None)

    blank = _attach(api, other["check_id"], "   ")
    assert blank.status_code == 422
    assert api.row(other["check_id"]).contact_id is None


def test_attach_is_quick_only_and_404_for_an_unknown_check(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    full = api.full()

    refused = _attach(api, full["check_id"], "c_1")

    assert (refused.status_code, refused.json()) == (409, {"detail": "attach_quick_only"})
    assert (api.row(full["check_id"]).contact_id, api.consent(full["check_id"]).contact_id) == (None, None)
    for check_id in (UNKNOWN_ID, "not-a-uuid"):
        response = _attach(api, check_id, "c_1")
        assert (response.status_code, response.json()) == (404, NOT_FOUND)
