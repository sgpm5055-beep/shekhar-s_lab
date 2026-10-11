"""PRD 12 item 21 + PRD 5.17 / acceptance: with ``SUREPASS_LIVE`` anything but the exact lowercase literal ``"true"``,
every SurePass call raises ``LiveCallsDisabled`` BEFORE any I/O (no socket, no sink row), the service still boots and
serves ``/health``, a check that would need a billed call is refused ``LIVE_CALLS_DISABLED``, and a check the caches can
answer is still served (the kill switch is the LAST admission rule; QK1 needs BOTH the PAN record and the score fresh).
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from eligibility_agent.config import Settings
from eligibility_agent.db.repo import MemoryRepo
from eligibility_agent.graph.prompts import local_prompts
from eligibility_agent.main import create_app, runtime_of
from eligibility_agent.output.lc_view import REFUSAL_MESSAGES
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    FakeSurePassClient,
    LiveCallsDisabled,
    LiveSurePassClient,
)
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.fake_llm import FakeRankModel
from eligibility_agent.testing.scenarios import full_happy_reported_emi_scenario
from tests.helpers import (
    AUTH,
    MOBILE_OK,
    OPP_TWO,
    PAN_FULL_NAME,
    PAN_OK,
    PAN_TWO,
    T0,
    ApiFactory,
    api_settings,
    assert_no_leak,
    full_payload,
    make_settings,
    pan_hash_of,
    quick_payload,
    seed_cached_quick_world,
)

NOT_LIVE = ["false", "TRUE", "True", "1", "yes", "", " true"]


def _fail_if_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"network I/O attempted with the kill switch off: {request.url.path}")


def test_live_client_with_the_switch_off_raises_before_any_io_on_all_three_endpoints(repo: MemoryRepo) -> None:
    with httpx.Client(transport=httpx.MockTransport(_fail_if_called)) as http:
        client = LiveSurePassClient(token="synthetic-token", live=False, sink=repo, http=http)
        assert client.calls_enabled is False
        pan_hash = pan_hash_of(PAN_OK)
        with pytest.raises(LiveCallsDisabled):
            client.pan_comprehensive("chk-none", pan_hash, PAN_OK)
        with pytest.raises(LiveCallsDisabled):
            client.experian_score("chk-none", pan_hash, name=PAN_FULL_NAME, mobile=MOBILE_OK, pan=PAN_OK)
        with pytest.raises(LiveCallsDisabled):
            client.experian_report("chk-none", pan_hash, name=PAN_FULL_NAME, mobile=MOBILE_OK, pan=PAN_OK)
    assert repo.list_calls() == []  # pre-I/O: not even an audit row


def test_settings_helper_refuses_a_misspelt_env_name() -> None:
    """T3 note: a misspelt alias would be silently ignored by pydantic-settings, so every negative kill-switch case above
    would pass vacuously; the helper refuses names that are not Settings aliases."""
    with pytest.raises(AssertionError, match="SUREPASS_LIVEE"):
        make_settings(SUREPASS_LIVEE="true")
    assert make_settings(SUREPASS_LIVE="true", SUREPASS_TOKEN="t").surepass_live is True


@contextmanager
def _app_without_fakes(settings: Settings, repo: MemoryRepo, clock: FrozenClock) -> Generator[TestClient]:
    """No SurePass injected: create_app builds the REAL LiveSurePassClient from the settings' kill switch."""
    app = create_app(settings, repo=repo, rank_model=FakeRankModel(), prompts=local_prompts(), now=clock.now)
    with TestClient(app, headers=AUTH) as client:
        yield client


@pytest.mark.parametrize("value", NOT_LIVE)
def test_anything_but_exact_lowercase_true_keeps_every_call_off(
    repo: MemoryRepo, clock: FrozenClock, value: str
) -> None:
    settings = api_settings(SUREPASS_LIVE=value, SUREPASS_TOKEN="synthetic-token")
    assert settings.surepass_live is False
    with _app_without_fakes(settings, repo, clock) as client:
        assert client.get("/health").json() == {"ok": True, "surepass_live": False}
        body = client.post("/checks/quick", json=quick_payload()).json()

    assert (body["status"], body["code"], body["label"]) == (
        "refused",
        "LIVE_CALLS_DISABLED",
        REFUSAL_MESSAGES["LIVE_CALLS_DISABLED"],
    )
    assert repo.list_calls() == []
    assert_no_leak(body)


def test_exact_lowercase_true_is_the_only_live_value(repo: MemoryRepo, clock: FrozenClock) -> None:
    """Positive control for the battery above (health only: a POST here would try the real network)."""
    settings = api_settings(SUREPASS_LIVE="true", SUREPASS_TOKEN="synthetic-token")
    with _app_without_fakes(settings, repo, clock) as client:
        assert client.get("/health").json() == {"ok": True, "surepass_live": True}


def test_degraded_boot_on_an_empty_environment_serves_health_and_refuses_billed_checks() -> None:
    """Acceptance: no fakes, no secrets, nothing injected. The conftest scrubbed every Settings env name except the test
    HMAC key, so this is the empty-environment boot (in-memory repo, the real live client switched off, auth off)."""
    settings = make_settings()  # Settings(_env_file=None): the process environment only
    assert (settings.surepass_live, settings.auth_enabled, settings.startup_errors()) == (False, False, [])
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"ok": True, "surepass_live": False}
        quick = client.post("/checks/quick", json=quick_payload()).json()
        full = client.post("/checks/full", json=full_payload()).json()

    for body in (quick, full):
        assert (body["status"], body["code"]) == ("refused", "LIVE_CALLS_DISABLED")
        assert_no_leak(body)
    rows = runtime_of(app).deps.repo
    assert isinstance(rows, MemoryRepo)
    assert [(r.status, r.code, r.billed) for r in rows.list_checks()] == [("refused", "LIVE_CALLS_DISABLED", False)] * 2
    assert rows.list_calls() == []


def test_cache_still_serves_with_live_off_when_pan_record_and_score_are_both_fresh(
    repo: MemoryRepo, clock: FrozenClock
) -> None:
    """K17: pre-seed BOTH pan_records and score_cache (a pull two days ago); with the real client switched off the quick
    check is a free hit. A PAN with nothing cached, or with a score but no PAN record (QK2: re-verifying the PAN is a
    live call), is refused LIVE_CALLS_DISABLED: the switch refuses only what would need a billed call."""
    seed_cached_quick_world(repo, pan=PAN_OK, fetched_at=T0 - timedelta(days=2))
    seed_cached_quick_world(repo, pan=PAN_TWO, fetched_at=T0 - timedelta(days=2), with_pan_record=False)

    with _app_without_fakes(api_settings(), repo, clock) as client:
        served = client.post("/checks/quick", json=quick_payload()).json()
        score_only = client.post("/checks/quick", json=quick_payload(pan=PAN_TWO)).json()
        nothing = client.post("/checks/quick", json=quick_payload(pan="QRSPT2468W")).json()

    assert (served["status"], served["from_cache"]) == ("ok", True)
    assert (served["checked_on"], served["refresh_available_on"]) == ("2026-10-08", "2026-11-07")
    assert (served["top_lender"]["id"], served["top_lender"]["ceiling"]) == ("L1", 1277000)
    row = repo.get_check(served["check_id"])
    assert row is not None and (row.billed, row.counted_against_quota) == (False, False)
    assert_no_leak(served)
    for refused in (score_only, nothing):
        assert (refused["status"], refused["code"]) == ("refused", "LIVE_CALLS_DISABLED")
    assert repo.list_calls() == []


def test_report_cache_serves_full_and_quick_checks_with_live_off(make_api: ApiFactory) -> None:
    """A report pulled while live (first app) serves a second opportunity's full check AND a quick check (score_cache
    source='report') from a second app over the same database with SurePass switched off."""
    live = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    live.full()
    off = make_api(lambda sink: FakeSurePassClient(sink=sink, calls_enabled=False))
    off.clock.advance(timedelta(days=1))

    full = off.full(opportunity_id=OPP_TWO)
    quick = off.quick()

    assert (full["status"], full["from_cache"], quick["status"], quick["from_cache"]) == ("ok", True, "ok", True)
    assert off.endpoints() == []
    assert live.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    # a new PAN on a new opportunity: nothing cached, so a billed pull would be needed
    refused = off.full(pan=PAN_TWO, opportunity_id="opp-3")
    assert (refused["status"], refused["code"]) == ("refused", "LIVE_CALLS_DISABLED")
