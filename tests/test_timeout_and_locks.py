"""PRD 12 items 18 and 19 (PRD 5.11 / 5.12, K1 / K2): a bureau timeout is spend-unknown, answered ``pending_verification``
with code None, locks the PAN for 10 minutes AND opens a manual reconciliation review that alone gates every later
billed pull (lock expiry never re-admits: +11 minutes is still PAN_LOCKED until the review is resolved). A request while
another check on the same PAN is in flight is refused IN_FLIGHT immediately, never queued. A non-timeout transport
failure is a plain service failure with no lock: the timeout / transport taxonomy split, pinned through both the fake
and the REAL httpx client (MockTransport, no socket).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eligibility_agent.db.repo import MemoryRepo
from eligibility_agent.graph.prompts import local_prompts
from eligibility_agent.main import create_app
from eligibility_agent.output.lc_view import (
    MSG_BUREAU_SERVICE_FAILURE,
    MSG_PAN_SERVICE_FAILURE,
    MSG_PENDING_VERIFICATION,
    REFUSAL_MESSAGES,
)
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
    LiveSurePassClient,
    SurePassResult,
    SurePassTimeout,
    SurePassTransportError,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.fake_llm import FakeRankModel
from eligibility_agent.testing.scenarios import happy_quick_scenario, score_failure_scenario
from tests.helpers import (
    AUTH,
    LC_B,
    PAN_OK,
    T0,
    Api,
    ApiFactory,
    SwitchableSurePass,
    api_settings,
    assert_no_leak,
    internal_of,
    pan_hash_of,
    quick_payload,
    seed_running_check,
)


def _refused_pan_locked(api: Api, body: dict[str, object]) -> None:
    assert (body["status"], body["code"], body["label"]) == ("refused", "PAN_LOCKED", REFUSAL_MESSAGES["PAN_LOCKED"])
    assert api.calls(str(body["check_id"])) == []


# --------------------------------------------------------------------------- item 18: the bureau timeout (K2 shape)


def test_score_timeout_is_pending_verification_then_gated_until_the_review_is_resolved(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: SwitchableSurePass(score_failure_scenario("timeout", sink=sink)))
    switchable = api.surepass
    assert isinstance(switchable, SwitchableSurePass)

    body = api.quick()

    assert {k: v for k, v in body.items() if k != "check_id"} == {
        "kind": "quick",
        "status": "pending_verification",
        "code": None,  # K1: pending carries no code; PAN_LOCKED is for the requests that follow
        "headline": None,
        "label": MSG_PENDING_VERIFICATION,
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
        "quick_checks_left_today": 5,  # spend unknown never counts
    }
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    row = api.row(body["check_id"])
    assert (row.status, row.code, row.counted_against_quota) == ("pending_verification", None, False)
    score_call = api.calls(row.id)[-1]
    assert (score_call.endpoint, score_call.error_kind, score_call.http_status, score_call.billable) == (
        SCORE_ENDPOINT,
        "timeout",
        None,
        True,  # conservatively billable until reconciled (K17)
    )
    assert row.billed is True and row.billed == any(api.billable(row.id))
    assert internal_of(row)["spend_unknown"] is True
    (lock,) = api.locks(PAN_OK, "timeout")
    assert (lock.lock_until, lock.check_id) == (T0 + timedelta(minutes=10), row.id)
    (review,) = api.reviews(PAN_OK, "timeout_reconciliation")
    assert (review.check_id, review.resolved_at) == (row.id, None)
    assert api.repo.get_score_cache(pan_hash_of(PAN_OK)) is None
    assert_no_leak(body)

    _refused_pan_locked(api, api.quick())  # immediately: the 10-minute lock
    api.clock.advance(timedelta(minutes=11))
    assert "timeout" not in api.active_lock_kinds(PAN_OK)  # the lock itself has expired ...
    _refused_pan_locked(api, api.quick())  # ... but the unresolved reconciliation still gates the pull
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]

    api.repo.resolve_review(review.id, resolved_by="ops")
    switchable.switch_to(happy_quick_scenario(sink=api.repo))  # the Score answers this time
    admitted = api.quick()

    assert (admitted["status"], admitted["quick_checks_left_today"]) == ("ok", 4)
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]  # PAN record reused
    retry = api.row(admitted["check_id"])
    assert (retry.billed, retry.counted_against_quota) == (True, True)  # still no stored score: this one counts
    assert api.consent(retry.id).covers == "score"  # its own consent record for the re-pull


def test_report_timeout_locks_even_cache_hits_then_gates_only_billed_pulls(make_api: ApiFactory) -> None:
    """The K2 shape on the Report endpoint (20 s). While the 10-minute lock is active it pre-empts EVERYTHING, a would-be
    cache hit included (K22, node-level lock gate). Once the lock has expired the unresolved reconciliation gates only
    BILLED pulls: the cached quick check is served, a report pull is still PAN_LOCKED."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK)},
            score={PAN_OK: env.experian_score_envelope(credit_score="745")},
            report={PAN_OK: SurePassTimeout(REPORT_ENDPOINT, "timeout: ReadTimeout")},
            sink=sink,
        )

    api = make_api(factory)
    api.quick()  # caches the PAN record and the score
    body = api.full()

    assert (body["status"], body["code"], body["label"]) == ("pending_verification", None, MSG_PENDING_VERIFICATION)
    assert body["quick_checks_left_today"] is None
    row = api.row(body["check_id"])
    assert (row.status, row.counted_against_quota, row.billed) == ("pending_verification", False, True)
    assert [(c.endpoint, c.error_kind, c.billable) for c in api.calls(row.id)] == [(REPORT_ENDPOINT, "timeout", True)]
    (lock,) = api.locks(PAN_OK, "timeout")
    assert lock.lock_until == T0 + timedelta(minutes=10)
    assert len(api.reviews(PAN_OK, "timeout_reconciliation")) == 1

    _refused_pan_locked(api, api.quick())  # a would-be cache hit, refused while the lock is active
    api.clock.advance(timedelta(minutes=11))
    served = api.quick()
    assert (served["status"], served["from_cache"]) == ("ok", True)  # cache hits make no billed call
    _refused_pan_locked(api, api.full())  # a report pull is still gated by the open reconciliation
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT]


def test_resolving_the_review_does_not_lift_a_still_active_timeout_lock(make_api: ApiFactory) -> None:
    """Both gates must be clear: the lock (10 minutes) AND the reconciliation review."""
    api = make_api(lambda sink: SwitchableSurePass(score_failure_scenario("timeout", sink=sink)))
    switchable = api.surepass
    assert isinstance(switchable, SwitchableSurePass)
    api.quick()
    (review,) = api.reviews(PAN_OK, "timeout_reconciliation")

    api.clock.advance(timedelta(minutes=5))
    api.repo.resolve_review(review.id)
    switchable.switch_to(happy_quick_scenario(sink=api.repo))
    _refused_pan_locked(api, api.quick())

    api.clock.advance(timedelta(minutes=5))  # exactly at lock_until: expired (active means lock_until > now)
    assert api.quick()["status"] == "ok"
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]


def test_pan_endpoint_timeout_is_a_plain_service_failure_without_a_lock(make_api: ApiFactory) -> None:
    """K2: only a BUREAU timeout locks. A PAN Comprehensive timeout is 'Couldn't verify PAN, retry', retryable now."""
    api = make_api(
        lambda sink: FakeSurePassClient(pan={PAN_OK: SurePassTimeout(PAN_ENDPOINT, "timeout: ReadTimeout")}, sink=sink)
    )

    body = api.quick()

    assert (body["status"], body["code"], body["label"]) == ("service_failure", "SUREPASS_FAILURE", MSG_PAN_SERVICE_FAILURE)
    row = api.row(body["check_id"])
    assert (row.counted_against_quota, internal_of(row)["spend_unknown"]) == (False, False)
    assert api.locks(PAN_OK, "timeout") == [] and api.reviews(PAN_OK) == []
    again = api.quick()
    assert again["code"] == "SUREPASS_FAILURE"  # admitted again (not PAN_LOCKED): the PAN call was retried
    assert api.endpoints() == [PAN_ENDPOINT, PAN_ENDPOINT]


# --------------------------------------------------------------------------- item 19: in flight


def test_a_check_while_another_holds_the_pan_is_refused_in_flight_never_queued(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: happy_quick_scenario(sink=sink))
    pan_hash = pan_hash_of(PAN_OK)
    holder = seed_running_check(api.repo, pan=PAN_OK, now=api.clock.now())
    assert api.repo.try_acquire_pan_lock(pan_hash, holder, api.clock.now() + timedelta(seconds=180))

    body = api.quick()

    assert (body["status"], body["code"], body["label"]) == ("refused", "IN_FLIGHT", REFUSAL_MESSAGES["IN_FLIGHT"])
    assert api.endpoints() == []
    row = api.row(body["check_id"])
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("refused", "IN_FLIGHT", False, False)
    (held,) = api.locks(PAN_OK, "in_flight")
    assert (held.check_id, held.released_at) == (holder, None)  # a refusal never releases someone else's lock
    assert_no_leak(body)

    api.repo.release_pan_lock(pan_hash, holder)
    admitted = api.quick()

    assert admitted["status"] == "ok"
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert api.active_lock_kinds(PAN_OK) == []  # and the admitted run released its own


class _BlockingScore(SwitchableSurePass):
    """The happy world, except the Score call parks until the test lets it go (a check genuinely in flight)."""

    def __init__(self, first: FakeSurePassClient) -> None:
        super().__init__(first)
        self.entered = threading.Event()
        self.release = threading.Event()

    def experian_score(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise AssertionError("the parked Score call was never released")
        return super().experian_score(check_id, pan_hash, name=name, mobile=mobile, pan=pan)


def test_a_genuinely_concurrent_request_is_refused_in_flight_while_the_first_runs(make_api: ApiFactory) -> None:
    """PRD 5.11 end to end: request 1 is parked inside its Score call (holding the PAN's in-flight lock); request 2 for
    the same PAN, from another LC, is answered IN_FLIGHT at once instead of waiting; request 1 then completes normally."""
    api = make_api(lambda sink: _BlockingScore(happy_quick_scenario(sink=sink)))
    blocking = api.surepass
    assert isinstance(blocking, _BlockingScore)
    first: dict[str, Any] = {}

    def run_first() -> None:
        first.update(api.quick())

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        assert blocking.entered.wait(timeout=10), "request 1 never reached its Score call"
        assert api.active_lock_kinds(PAN_OK) == ["in_flight"]
        second = api.quick(lc_id=LC_B)
    finally:
        blocking.release.set()
        worker.join(timeout=10)

    assert (second["status"], second["code"]) == ("refused", "IN_FLIGHT")
    assert (first["status"], first["quick_checks_left_today"]) == ("ok", 4)
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]  # request 2 made no call
    assert api.active_lock_kinds(PAN_OK) == []


def test_an_in_flight_lock_past_its_ttl_is_taken_over(make_api: ApiFactory) -> None:
    """A crashed holder never wedges the PAN: once its 180 s TTL has passed the lock is stolen atomically."""
    api = make_api(lambda sink: happy_quick_scenario(sink=sink))
    holder = seed_running_check(api.repo, pan=PAN_OK, now=api.clock.now())
    assert api.repo.try_acquire_pan_lock(pan_hash_of(PAN_OK), holder, api.clock.now() + timedelta(seconds=180))

    assert api.quick()["code"] == "IN_FLIGHT"
    api.clock.advance(timedelta(seconds=181))
    assert api.quick()["status"] == "ok"
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]


# --------------------------------------------------------------------------- the timeout / transport taxonomy split


def test_non_timeout_transport_failure_is_a_service_failure_with_no_lock(make_api: ApiFactory) -> None:
    def factory(sink: CallLogSink) -> FakeSurePassClient:
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK)},
            score={PAN_OK: SurePassTransportError(SCORE_ENDPOINT, "transport: ConnectError")},
            sink=sink,
        )

    api = make_api(factory)
    body = api.quick()

    assert (body["status"], body["code"], body["label"]) == ("service_failure", "SUREPASS_FAILURE", MSG_BUREAU_SERVICE_FAILURE)
    assert body["quick_checks_left_today"] == 5
    row = api.row(body["check_id"])
    assert (row.status, row.counted_against_quota) == ("service_failure", False)
    assert [(c.endpoint, c.error_kind, c.billable) for c in api.calls(row.id)] == [
        (PAN_ENDPOINT, None, True),
        (SCORE_ENDPOINT, "transport", False),  # spend known-failed: not billable, unlike a timeout
    ]
    assert api.locks(PAN_OK, "timeout") == [] and api.reviews(PAN_OK) == []

    again = api.quick()  # no lock, no reconciliation gate: the retry is admitted and calls the Score again
    assert again["status"] == "service_failure"
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT, SCORE_ENDPOINT]


@contextmanager
def _live_app(
    repo: MemoryRepo, clock: FrozenClock, raise_on_score: Callable[[httpx.Request], httpx.HTTPError]
) -> Generator[TestClient]:
    """The REAL LiveSurePassClient over an httpx MockTransport (no socket): PAN answers 200, the Score raises."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PAN_ENDPOINT:
            status, body = env.pan_comprehensive_envelope(pan=PAN_OK)
            return httpx.Response(status, json=body)
        raise raise_on_score(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        live = LiveSurePassClient(token="synthetic-token", live=True, sink=repo, http=http)
        app = create_app(
            api_settings(),
            repo=repo,
            surepass=live,
            rank_model=FakeRankModel(),
            prompts=local_prompts(),
            now=clock.now,
        )
        with TestClient(app, headers=AUTH) as client:
            yield client


def _read_timeout(request: httpx.Request) -> httpx.HTTPError:
    return httpx.ReadTimeout("synthetic read timeout", request=request)


def _connect_error(request: httpx.Request) -> httpx.HTTPError:
    return httpx.ConnectError("synthetic connect failure", request=request)


@pytest.mark.parametrize(
    ("raise_on_score", "status", "error_kind", "locked"),
    [
        (_read_timeout, "pending_verification", "timeout", True),
        (_connect_error, "service_failure", "transport", False),
    ],
    ids=["read-timeout", "connect-error"],
)
def test_real_client_taxonomy_split_timeout_locks_transport_does_not(
    repo: MemoryRepo,
    clock: FrozenClock,
    raise_on_score: Callable[[httpx.Request], httpx.HTTPError],
    status: str,
    error_kind: str,
    locked: bool,
) -> None:
    with _live_app(repo, clock, raise_on_score) as client:
        response = client.post("/checks/quick", json=quick_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == status
    assert body["code"] == (None if locked else "SUREPASS_FAILURE")
    assert body["label"] == (MSG_PENDING_VERIFICATION if locked else MSG_BUREAU_SERVICE_FAILURE)
    calls = repo.list_calls(body["check_id"])
    assert [(c.endpoint, c.error_kind) for c in calls] == [(PAN_ENDPOINT, None), (SCORE_ENDPOINT, error_kind)]
    timeout_locks = [lock for lock in repo.active_locks_snapshot() if lock.kind == "timeout"]
    assert [lock.lock_until for lock in timeout_locks] == ([T0 + timedelta(minutes=10)] if locked else [])
    assert repo.unresolved_review_exists(pan_hash_of(PAN_OK), "timeout_reconciliation") is locked
