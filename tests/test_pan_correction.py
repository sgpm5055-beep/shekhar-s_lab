"""PRD 12 item 10 + PRD 5.13: the PAN on an opportunity may be corrected ONCE, before any report exists, without a
manager; a change after a report, or a second correction, is refused ``PAN_CHANGE_NEEDS_MANAGER`` and flagged for review
(review row ``pan_change_manager``). The once-per-30-days report window is per PAN hash, so it follows the CORRECTED PAN.
A report "exists" on the opportunity once a full check on it ended ``ok``, ``pending_verification`` or ``service_failure``
+ ``OPENAI_FAILURE`` (the report was pulled before the ranking failed: T14 fix, ``db.repo.report_check_on_record``).

Events (``opportunity_pan_events``) are recorded only by ADMITTED full checks (T11 ruling): a refused attempt neither
marks the first PAN nor spends the one correction. The policy reads EVERY recorded event (any 'correction' spends the
allowance), not only the latest one; a manager's approved change (``manager_change``, written by the manager tooling,
seeded here) makes that PAN the current one without giving the correction back. These vectors kill the two T7 mutants:
(a) ignoring recorded events (a second free correction) and (b) reading only the latest event's kind.

World: valid person PANs PAN_OK / PAN_TWO / PAN_THREE (zip 560001; each with the reported-EMI report), and two
valid-negative PANs, PAN_INVALID (status invalid) and PAN_COMPANY (category company), which end ``pan_not_verified``
after the Rs 1 PAN call, so no report ever exists for them.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Final

from eligibility_agent.db.repo import OpportunityPanEvent, new_id
from eligibility_agent.graph.state import PanEventKind
from eligibility_agent.output.lc_view import MSG_PAN_NOT_VERIFIED, REFUSAL_MESSAGES
from eligibility_agent.policy.guards import pan_last4
from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, CallLogSink, FakeSurePassClient
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.fake_llm import FakeRankModel
from eligibility_agent.testing.scenarios import REPORT_DATE
from tests.helpers import (
    OPP,
    OPP_TWO,
    PAN_COMPANY,
    PAN_INVALID,
    PAN_OK,
    PAN_THREE,
    PAN_TWO,
    Api,
    ApiFactory,
    assert_no_leak,
    internal_of,
    pan_hash_of,
    seed_billable_bureau_calls,
)

VALID_PANS: Final[tuple[str, ...]] = (PAN_OK, PAN_TWO, PAN_THREE)
NAMES: Final[dict[str, str]] = {
    pan_hash_of(pan): name
    for pan, name in (
        (PAN_OK, "OK"),
        (PAN_TWO, "TWO"),
        (PAN_THREE, "THREE"),
        (PAN_INVALID, "INVALID"),
        (PAN_COMPANY, "COMPANY"),
    )
}
FULL_LABEL: Final[str] = "bureau report on 2026-10-01, income as declared"
# True for BOTH triggers (a report exists, or the one correction is spent): the LC is never told a report exists when
# none does (T14 copy fix).
NEEDS_MANAGER_COPY: Final[str] = (
    "Changing the PAN on this opportunity needs a manager: a report already exists or the one allowed correction has "
    "been used. This has been flagged for review."
)


def _world(sink: CallLogSink) -> FakeSurePassClient:
    account = env.cais_account(
        scheduled_emi="3500", current_balance="150000", date_reported="20260920", account_number="SYNTHACC0001"
    )
    pans = {pan: env.pan_comprehensive_envelope(pan=pan) for pan in VALID_PANS}
    pans[PAN_INVALID] = env.pan_comprehensive_envelope(pan=PAN_INVALID, status="invalid")
    pans[PAN_COMPANY] = env.pan_comprehensive_envelope(pan=PAN_COMPANY, category="company")
    return FakeSurePassClient(
        pan=pans,
        report={pan: env.experian_report_envelope(accounts=[account], report_date=REPORT_DATE) for pan in VALID_PANS},
        sink=sink,
    )


def _events(api: Api, opportunity_id: str = OPP) -> list[tuple[str, str]]:
    return [(e.event, NAMES[e.pan_hash]) for e in api.repo.pan_events_for_opportunity(opportunity_id)]


def _manager_change(api: Api, pan: str) -> None:
    """What the manager tooling writes when it approves a PAN change (no API route exists for it in phase 1)."""
    api.clock.advance(timedelta(minutes=1))
    event: PanEventKind = "manager_change"
    api.repo.insert_pan_event(
        OpportunityPanEvent(
            id=new_id(),
            opportunity_id=OPP,
            pan_hash=pan_hash_of(pan),
            pan_last4=pan_last4(pan),
            event=event,
            actor_lc_id="manager-1",
            check_id=None,
            created_at=api.clock.now(),
        )
    )
    api.clock.advance(timedelta(minutes=1))


def _not_verified(api: Api, pan: str) -> dict[str, Any]:
    body = api.full(pan=pan)
    assert (body["status"], body["label"]) == ("pan_not_verified", MSG_PAN_NOT_VERIFIED)
    assert_no_leak(body)
    return body


def _assert_needs_manager(api: Api, body: dict[str, Any], pan: str, *, calls_before: int) -> None:
    code = "PAN_CHANGE_NEEDS_MANAGER"
    assert (body["status"], body["code"], body["label"]) == ("refused", code, NEEDS_MANAGER_COPY)
    assert REFUSAL_MESSAGES[code] == NEEDS_MANAGER_COPY
    assert (body["headline"], body["top_lender"], body["quick_checks_left_today"]) == (None, None, None)
    assert len(api.endpoints()) == calls_before  # zero SurePass calls
    row = api.row(body["check_id"])
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("refused", code, False, False)
    assert row.pan_hash == pan_hash_of(pan)
    (review,) = [r for r in api.reviews(pan, "pan_change_manager") if r.check_id == body["check_id"]]
    assert review.resolved_at is None
    assert api.active_lock_kinds(pan) == []  # the in-flight lock was released
    assert_no_leak(body)


# --------------------------------------------------------------------------- the one free correction


def test_one_correction_before_any_report_is_allowed_and_logged(make_api: ApiFactory) -> None:
    api = make_api(_world)
    first = _not_verified(api, PAN_INVALID)
    assert _events(api) == [("initial", "INVALID")]
    assert api.repo.report_exists_for_opportunity(OPP) is False

    corrected = api.full(pan=PAN_OK)

    assert api.endpoints() == [PAN_ENDPOINT, PAN_ENDPOINT, REPORT_ENDPOINT]  # the corrected PAN is re-verified
    assert (corrected["status"], corrected["code"], corrected["label"]) == ("ok", None, FULL_LABEL)
    assert _events(api) == [("initial", "INVALID"), ("correction", "OK")]
    events = api.repo.pan_events_for_opportunity(OPP)
    assert [e.check_id for e in events] == [first["check_id"], corrected["check_id"]]
    row = api.row(corrected["check_id"])
    assert internal_of(row)["admission"]["record_pan_correction"] is True  # the correction is logged on the check
    assert (row.billed, row.counted_against_quota) == (True, False)
    assert api.repo.list_reviews() == []  # no manager flag
    assert_no_leak(corrected)

    api.clock.advance(timedelta(days=2))
    again = api.full(pan=PAN_OK)  # the corrected PAN is now the opportunity's PAN: a plain report-reuse hit

    assert (again["status"], again["from_cache"]) == ("ok", True)
    assert len(api.endpoints()) == 3
    assert _events(api) == [("initial", "INVALID"), ("correction", "OK")]


def test_a_pan_change_after_a_report_needs_a_manager(make_api: ApiFactory) -> None:
    api = make_api(_world)
    assert api.full(pan=PAN_OK)["status"] == "ok"
    assert api.repo.report_exists_for_opportunity(OPP) is True

    body = api.full(pan=PAN_TWO)

    _assert_needs_manager(api, body, PAN_TWO, calls_before=2)
    assert _events(api) == [("initial", "OK")]  # the correction is unspent, but a report exists

    other = api.full(pan=PAN_TWO, opportunity_id=OPP_TWO)  # the rule is per opportunity
    assert (other["status"], other["code"]) == ("ok", None)
    assert _events(api, OPP_TWO) == [("initial", "TWO")]


def test_a_report_pulled_by_a_run_whose_ranking_failed_also_needs_a_manager(make_api: ApiFactory) -> None:
    """The OPENAI_FAILURE run pulled (and billed) the report: a report exists on the opportunity even though the check
    ended ``service_failure``, so a PAN change needs a manager (otherwise it would buy a second report)."""
    api = make_api(_world, rank=FakeRankModel.invalid_twice())
    failed = api.full(pan=PAN_OK)
    assert (failed["status"], failed["code"]) == ("service_failure", "OPENAI_FAILURE")
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]  # the report WAS pulled
    assert api.repo.report_exists_for_opportunity(OPP) is True

    body = api.full(pan=PAN_TWO)

    _assert_needs_manager(api, body, PAN_TWO, calls_before=2)
    assert _events(api) == [("initial", "OK")]


def test_a_second_correction_before_any_report_needs_a_manager(make_api: ApiFactory) -> None:
    """Mutant (a): a policy that ignores the recorded events would grant a second free correction here."""
    api = make_api(_world)
    _not_verified(api, PAN_INVALID)
    _not_verified(api, PAN_COMPANY)  # the one correction, admitted (and spent) although the PAN failed verification
    assert _events(api) == [("initial", "INVALID"), ("correction", "COMPANY")]
    assert api.repo.report_exists_for_opportunity(OPP) is False

    body = api.full(pan=PAN_OK)

    _assert_needs_manager(api, body, PAN_OK, calls_before=2)
    assert api.endpoints() == [PAN_ENDPOINT, PAN_ENDPOINT]
    assert _events(api) == [("initial", "INVALID"), ("correction", "COMPANY")]


def test_a_refused_attempt_never_spends_the_correction(make_api: ApiFactory) -> None:
    """The policy said 'record_correction' but admission refused (the per-PAN cap): no event is written, so the
    next PAN still gets the one free correction. A node that recorded the event before the decision would refuse it."""
    api = make_api(_world)
    _not_verified(api, PAN_INVALID)
    seed_billable_bureau_calls(api.repo, pan=PAN_OK, n=3, now=api.clock.now())

    capped = api.full(pan=PAN_OK)

    assert (capped["status"], capped["code"]) == ("refused", "PAN_DAILY_CAP")
    assert _events(api) == [("initial", "INVALID")]

    corrected = api.full(pan=PAN_TWO)

    assert (corrected["status"], corrected["code"]) == ("ok", None)
    assert _events(api) == [("initial", "INVALID"), ("correction", "TWO")]
    assert api.reviews(PAN_TWO, "pan_change_manager") == []
    assert_no_leak(corrected)


# --------------------------------------------------------------------------- manager changes (T7 deferred vectors)


def test_a_manager_change_does_not_give_the_spent_correction_back(make_api: ApiFactory) -> None:
    """Mutant (b): the LATEST event is the manager's change, not a correction; a policy reading only the latest event
    would treat the next change as the free correction. Any recorded correction spends it."""
    api = make_api(_world)
    _not_verified(api, PAN_INVALID)
    _not_verified(api, PAN_COMPANY)
    _manager_change(api, PAN_OK)
    assert _events(api) == [("initial", "INVALID"), ("correction", "COMPANY"), ("manager_change", "OK")]
    assert api.repo.report_exists_for_opportunity(OPP) is False

    body = api.full(pan=PAN_TWO)

    _assert_needs_manager(api, body, PAN_TWO, calls_before=2)

    approved = api.full(pan=PAN_OK)  # the manager-approved PAN is the opportunity's current PAN

    assert (approved["status"], approved["code"], approved["from_cache"]) == ("ok", None, False)
    assert api.endpoints() == [PAN_ENDPOINT, PAN_ENDPOINT, PAN_ENDPOINT, REPORT_ENDPOINT]
    assert api.reviews(PAN_OK, "pan_change_manager") == []
    assert _events(api) == [("initial", "INVALID"), ("correction", "COMPANY"), ("manager_change", "OK")]
    assert_no_leak(approved)


def test_after_a_manager_change_the_new_pan_gets_its_own_report_window(make_api: ApiFactory) -> None:
    """The window is per PAN hash: PAN_OK's fresh report does not block the manager-approved PAN_TWO (a new billed pull),
    and from then on PAN_TWO's own window governs (a forced refresh inside it is REPORT_WINDOW_OPEN)."""
    api = make_api(_world)
    _not_verified(api, PAN_INVALID)
    assert api.full(pan=PAN_OK)["status"] == "ok"  # the correction, with a report
    refused = api.full(pan=PAN_TWO)
    _assert_needs_manager(api, refused, PAN_TWO, calls_before=3)

    (review,) = api.reviews(PAN_TWO, "pan_change_manager")
    api.repo.resolve_review(review.id, resolved_by="manager-1")
    _manager_change(api, PAN_TWO)

    changed = api.full(pan=PAN_TWO)

    assert (changed["status"], changed["from_cache"]) == ("ok", False)
    assert api.endpoints()[3:] == [PAN_ENDPOINT, REPORT_ENDPOINT]  # PAN_TWO verified and its own report pulled
    row = api.row(changed["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, False)
    assert len(api.reviews(PAN_TWO, "pan_change_manager")) == 1  # no new flag

    refresh = api.full(pan=PAN_TWO, refresh=True)

    assert (refresh["status"], refresh["code"]) == ("refused", "REPORT_WINDOW_OPEN")
    assert refresh["refresh_available_on"] == "2026-11-09"
    assert len(api.endpoints()) == 5
    assert_no_leak(changed)
    assert_no_leak(refresh)


def test_the_report_window_follows_the_corrected_pan(make_api: ApiFactory) -> None:
    """PAN_OK already has a fresh report (opportunity 2, yesterday). Correcting opportunity 1's PAN to PAN_OK is served
    from that report: one report per PAN per 30 days, shared across opportunities, so no second pull and no spend."""
    api = make_api(_world)
    assert api.full(pan=PAN_OK, opportunity_id=OPP_TWO)["status"] == "ok"
    api.clock.advance(timedelta(days=1))
    _not_verified(api, PAN_INVALID)

    corrected = api.full(pan=PAN_OK)

    assert (corrected["status"], corrected["from_cache"], corrected["checked_on"]) == ("ok", True, "2026-10-10")
    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT, PAN_ENDPOINT]  # only the wrong PAN's Rs 1 call since
    row = api.row(corrected["check_id"])
    assert (row.billed, row.counted_against_quota) == (False, False)
    assert _events(api) == [("initial", "INVALID"), ("correction", "OK")]

    refresh = api.full(pan=PAN_OK, refresh=True)

    assert (refresh["status"], refresh["code"], refresh["refresh_available_on"]) == (
        "refused",
        "REPORT_WINDOW_OPEN",
        "2026-11-09",
    )
    assert len(api.endpoints()) == 3
    assert_no_leak(corrected)
    assert_no_leak(refresh)
