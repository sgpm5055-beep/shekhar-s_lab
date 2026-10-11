"""PRD 12 items 3, 6, 7 + the IST day boundary: the quick quota (5 per LC per IST day, per-LC override), what counts
against it (PRD 5.6, K8: status-independent), the expired-score refresh (billed, not counted) and the per-PAN hard cap
(3 billed bureau calls per IST day, score and report together -> PAN_DAILY_CAP + day lock + review row).

Billing assertions use the per-call ``billable`` flags (K17: ``billed == any(billable)``); lock assertions use
``lock_until``. Quota / cap days are IST calendar days computed on UTC instants (Review Focus 2): the boundary tests
put 23:59 IST and 00:01 IST on the SAME UTC date.
"""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any

from eligibility_agent.graph.state import Code
from eligibility_agent.output.lc_view import MSG_OPENAI_FAILURE, REFUSAL_MESSAGES
from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT
from eligibility_agent.testing.fake_llm import FakeRankModel
from eligibility_agent.testing.scenarios import happy_quick_scenario, score_failure_scenario
from tests.helpers import (
    LC_A,
    LC_B,
    MOBILE_OK,
    PAN_FULL_NAME,
    PAN_OK,
    IST,
    PAN_TWO,
    T0,
    Api,
    ApiFactory,
    age_score_cache,
    assert_no_leak,
    ist,
    pan_hash_of,
    quick_and_full_client,
    seed_billable_bureau_calls,
    seed_counted_quick_checks,
)


def _happy(make_api: ApiFactory, pan: str = PAN_OK, *, rank: FakeRankModel | None = None) -> Api:
    return make_api(lambda sink: happy_quick_scenario(pan=pan, sink=sink), rank=rank)


def _assert_refused(api: Api, body: dict[str, Any], code: Code, *, left: int | None) -> None:
    assert (body["status"], body["code"], body["label"]) == ("refused", code, REFUSAL_MESSAGES[code])
    assert (body["headline"], body["confidence"], body["top_lender"], body["alternates"]) == (None, None, None, [])
    assert body["quick_checks_left_today"] == left
    row = api.row(str(body["check_id"]))
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("refused", code, False, False)
    assert api.calls(row.id) == []
    assert_no_leak(body)


# --------------------------------------------------------------------------- item 3: the quota


def test_sixth_billed_quick_check_is_refused_quota_exhausted(make_api: ApiFactory) -> None:
    api = _happy(make_api, PAN_TWO)  # would answer if called: the empty call log below is the refusal's doing
    seed_counted_quick_checks(api.repo, lc_id=LC_A, created_at=T0 - timedelta(hours=1), n=5)

    body = api.quick(pan=PAN_TWO)

    _assert_refused(api, body, "QUOTA_EXHAUSTED", left=0)
    assert api.endpoints() == []
    assert api.rank.call_count == 0
    assert api.repo.quick_checks_used_today(LC_A) == 5  # the refusal itself never counts


def test_lc_at_its_quota_can_still_recheck_a_cached_pan(make_api: ApiFactory) -> None:
    """Cache hits are free and outside the quota (PRD 5.4): only the counting path exhausts (QK1 before QK4)."""
    api = _happy(make_api)
    api.quick(lc_id=LC_B)  # LC_B's pull caches the PAN
    seed_counted_quick_checks(api.repo, lc_id=LC_A, created_at=T0, n=5)

    body = api.quick(lc_id=LC_A)

    assert (body["status"], body["from_cache"], body["quick_checks_left_today"]) == ("ok", True, 0)
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]


def test_quota_override_of_seven_admits_the_sixth_check(make_api: ApiFactory) -> None:
    api = _happy(make_api, PAN_TWO)
    api.repo.set_quota_override(LC_A, 7)
    seed_counted_quick_checks(api.repo, lc_id=LC_A, created_at=T0 - timedelta(hours=1), n=5)

    body = api.quick(pan=PAN_TWO)

    assert (body["status"], body["quick_checks_left_today"]) == ("ok", 1)  # 7 - 5 - this one
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, True)
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert api.repo.quick_checks_used_today(LC_A) == 6


def test_openai_failure_quick_run_still_counts_against_the_quota(make_api: ApiFactory) -> None:
    """K8: counting is status-independent. The Score was billed 200+success on a PAN with no stored score, so the run
    counts even though the ranking model failed twice."""
    api = _happy(make_api, rank=FakeRankModel.invalid_twice())

    body = api.quick()

    assert (body["status"], body["code"], body["label"]) == ("service_failure", "OPENAI_FAILURE", MSG_OPENAI_FAILURE)
    assert body["quick_checks_left_today"] == 4
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, True)
    assert row.billed == any(api.billable(row.id))
    assert api.repo.quick_checks_used_today(LC_A) == 1
    assert api.rank.call_count == 2  # one retry, never a third call
    assert_no_leak(body)


# --------------------------------------------------------------------------- item 6: expired-score refresh


def test_expired_score_refresh_is_billed_but_not_counted(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    api.quick()
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]

    api.clock.advance(timedelta(days=31))
    body = api.quick()  # no refresh flag exists on quick: the expired score re-pulls automatically

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]  # PAN reused (< 180 days)
    assert (body["status"], body["from_cache"]) == ("ok", False)
    assert (body["checked_on"], body["refresh_available_on"]) == ("2026-11-10", "2026-12-10")
    assert body["quick_checks_left_today"] == 5
    row = api.row(body["check_id"])
    assert [c.endpoint for c in api.calls(row.id)] == [SCORE_ENDPOINT]
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (True, False, True)
    assert row.billed == any(api.billable(row.id))
    assert api.consent(row.id).covers == "score"  # a consent record for THIS pull
    # the refresh re-adapts the STORED PAN record: its full_name, never the LC-typed name, goes to the bureau
    assert api.surepass.calls[-1].body == {"name": PAN_FULL_NAME, "mobile": MOBILE_OK, "pan": PAN_OK, "consent": "Y"}
    score = api.repo.get_score_cache(pan_hash_of(PAN_OK))
    assert score is not None and score.fetched_at == T0 + timedelta(days=31)
    assert api.repo.quick_checks_used_today(LC_A) == 0


def test_lc_at_its_quota_can_still_refresh_an_expired_score(make_api: ApiFactory) -> None:
    """PRD 5.6 letter: a refresh of an expired stored score does not count, so the quota never blocks it (QK5)."""
    api = _happy(make_api)
    api.quick(lc_id=LC_B)
    api.clock.advance(timedelta(days=31))
    seed_counted_quick_checks(api.repo, lc_id=LC_A, created_at=api.clock.now(), n=5)

    body = api.quick(lc_id=LC_A)

    assert (body["status"], body["quick_checks_left_today"]) == ("ok", 0)
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (True, False, True)
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]


def test_score_window_is_thirty_days_to_the_minute(make_api: ApiFactory) -> None:
    """A row is fresh strictly before its expiry instant: T0 + 30 d - 1 min is a free hit, T0 + 30 d is a refresh."""
    api = _happy(make_api)
    api.quick()

    api.clock.set_now(T0 + timedelta(days=30) - timedelta(minutes=1))
    assert api.quick()["from_cache"] is True
    api.clock.set_now(T0 + timedelta(days=30))
    refreshed = api.quick()

    assert refreshed["from_cache"] is False
    assert api.row(refreshed["check_id"]).is_refresh is True
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]


# --------------------------------------------------------------------------- item 7: the per-PAN hard cap


def test_three_billed_bureau_calls_cap_the_pan_for_the_ist_day(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: quick_and_full_client(sink))
    pan_hash = pan_hash_of(PAN_OK)

    api.quick()  # bureau call 1: Score (new, counted)
    api.full()  # bureau call 2: Report (PAN reused)
    age_score_cache(api.repo, PAN_OK)
    third = api.quick()  # bureau call 3: an expired-score refresh
    assert api.row(third["check_id"]).is_refresh is True
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT]
    assert api.repo.bureau_calls_today(pan_hash) == 3

    age_score_cache(api.repo, PAN_OK)  # the 4th attempt again needs a billed pull
    fourth = api.quick()

    _assert_refused(api, fourth, "PAN_DAILY_CAP", left=4)  # quota untouched: only call 1 counted
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT]
    assert api.repo.bureau_calls_today(pan_hash) == 3
    (lock,) = api.locks(PAN_OK, "daily_cap")
    assert (lock.lock_until, lock.check_id) == (ist(2026, 10, 11), fourth["check_id"])  # next IST midnight
    (review,) = api.reviews(PAN_OK, "pan_daily_cap")
    assert (review.check_id, review.resolved_at) == (fourth["check_id"], None)

    # the day lock pre-empts even a would-be cache hit (the report is fresh)
    hit = api.full()
    _assert_refused(api, hit, "PAN_DAILY_CAP", left=None)
    assert len(api.endpoints()) == 4
    assert len(api.reviews(PAN_OK, "pan_daily_cap")) == 1  # the lock gate refuses; no second review row

    # next IST day: lock expired, count reset
    api.clock.set_now(ist(2026, 10, 11, 0, 1))
    assert api.repo.bureau_calls_today(pan_hash) == 0
    served = api.full()
    assert (served["status"], served["from_cache"]) == ("ok", True)
    refreshed = api.quick()
    assert (refreshed["status"], refreshed["from_cache"]) == ("ok", False)
    assert api.endpoints()[4:] == [SCORE_ENDPOINT]


def test_three_resolved_score_timeouts_also_cap_the_pan(make_api: ApiFactory) -> None:
    """The natural cap path: timed-out Score calls are billable (spend unknown, K17), so they count toward the 3-per-PAN
    cap. Each timeout is reconciled (review resolved) and retried after its 10-minute lock: 10:00, 10:11 and 10:22 IST.
    The fourth attempt, with every lock expired and every review resolved, is refused by the cap alone."""
    api = make_api(lambda sink: score_failure_scenario("timeout", sink=sink))
    pan_hash = pan_hash_of(PAN_OK)

    for minute in (0, 11, 22):
        api.clock.set_now(ist(2026, 10, 10, 10, minute))
        for review in api.reviews(PAN_OK, "timeout_reconciliation"):
            if review.resolved_at is None:
                api.repo.resolve_review(review.id, resolved_by="ops")
        pending = api.quick()
        assert (pending["status"], pending["code"]) == ("pending_verification", None)
        assert api.row(pending["check_id"]).counted_against_quota is False

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]  # PAN reused after call 1
    score_rows = [c for c in api.repo.list_calls() if c.endpoint == SCORE_ENDPOINT]
    assert [(c.error_kind, c.http_status, c.billable) for c in score_rows] == [("timeout", None, True)] * 3
    assert api.repo.bureau_calls_today(pan_hash) == 3

    api.clock.set_now(ist(2026, 10, 10, 10, 33))
    (last,) = [r for r in api.reviews(PAN_OK, "timeout_reconciliation") if r.resolved_at is None]
    api.repo.resolve_review(last.id, resolved_by="ops")
    assert api.active_lock_kinds(PAN_OK) == []  # nothing else stands in the way
    fourth = api.quick()

    _assert_refused(api, fourth, "PAN_DAILY_CAP", left=5)  # no timeout ever counted
    assert len(api.endpoints()) == 4
    (lock,) = api.locks(PAN_OK, "daily_cap")
    assert lock.lock_until == ist(2026, 10, 11)
    assert len(api.reviews(PAN_OK, "pan_daily_cap")) == 1


# --------------------------------------------------------------------------- the IST day boundary (quota + caps)


def test_quota_resets_at_ist_midnight_not_utc_midnight(make_api: ApiFactory) -> None:
    api = _happy(make_api, PAN_TWO)
    api.clock.set_now(ist(2026, 10, 10, 23, 0))
    seed_counted_quick_checks(api.repo, lc_id=LC_A, created_at=api.clock.now(), n=5)

    before, after = ist(2026, 10, 10, 23, 59), ist(2026, 10, 11, 0, 1)
    assert before.astimezone(IST).date() != after.astimezone(IST).date()
    assert before.astimezone(timezone.utc).date() == after.astimezone(timezone.utc).date()  # both 2026-10-10 in UTC

    api.clock.set_now(before)
    _assert_refused(api, api.quick(pan=PAN_TWO), "QUOTA_EXHAUSTED", left=0)

    api.clock.set_now(after)
    body = api.quick(pan=PAN_TWO)
    assert (body["status"], body["quick_checks_left_today"]) == ("ok", 4)
    assert api.row(body["check_id"]).counted_against_quota is True
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]


def test_per_pan_cap_resets_at_ist_midnight_not_utc_midnight(make_api: ApiFactory) -> None:
    api = _happy(make_api, PAN_TWO)
    api.clock.set_now(ist(2026, 10, 10, 23, 0))
    seed_billable_bureau_calls(api.repo, pan=PAN_TWO, n=3, now=api.clock.now())

    api.clock.set_now(ist(2026, 10, 10, 23, 59))
    refused = api.quick(pan=PAN_TWO)
    _assert_refused(api, refused, "PAN_DAILY_CAP", left=5)
    (lock,) = api.locks(PAN_TWO, "daily_cap")
    assert lock.lock_until == ist(2026, 10, 11)
    assert len(api.reviews(PAN_TWO, "pan_daily_cap")) == 1

    api.clock.set_now(ist(2026, 10, 11, 0, 1))  # same UTC date (2026-10-10T18:31Z), new IST day
    assert api.repo.bureau_calls_today(pan_hash_of(PAN_TWO)) == 0
    body = api.quick(pan=PAN_TWO)
    assert (body["status"], body["quick_checks_left_today"]) == ("ok", 4)
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]


def test_checked_on_and_refresh_dates_are_ist_dates_of_the_pull(make_api: ApiFactory) -> None:
    """05:00 IST on 10 Oct is 23:30 UTC on 9 Oct: the LC-facing dates are the IST ones (a UTC date would be a day early)."""
    api = _happy(make_api)
    api.clock.set_now(ist(2026, 10, 10, 5, 0))

    body = api.quick()

    assert (body["checked_on"], body["refresh_available_on"]) == ("2026-10-10", "2026-11-09")
    assert api.repo.quick_checks_used_today(LC_A) == 1  # counted on the IST day it happened
