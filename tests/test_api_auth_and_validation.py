"""Task 12: main.py — the app factory, the five routes, bearer auth, request validation and the error boundary.

Every test builds its app explicitly (no conftest yet): ``create_app(Settings(_env_file=None, ...), repo=<MemoryRepo on a
frozen clock>, surepass=<FakeSurePassClient logging into that repo>, rank_model=FakeRankModel(), ...)`` driven through
``TestClient`` as a context manager (the lifespan runs). No network; every identity, amount and pincode is synthetic.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client
from langsmith import utils as ls_utils

from eligibility_agent import main
from eligibility_agent.config import ConfigError, Settings
from eligibility_agent.db.repo import CheckRow, MemoryRepo
from eligibility_agent.graph import nodes, tracing
from eligibility_agent.graph.build import CheckpointerConfigError, GraphDeps, make_checkpoint_serde
from eligibility_agent.graph.llm import RankContext, RankModel
from eligibility_agent.graph.prompts import PromptBundle, local_prompts
from eligibility_agent.graph.state import CheckState, LcView, ModelOutput
from eligibility_agent.output.lc_view import REFUSAL_MESSAGES, build_error_lc_view
from eligibility_agent.policy.guards import hash_mobile, hash_pan
from eligibility_agent.surepass.client import CallLogSink, FakeSurePassClient, LiveSurePassClient
from eligibility_agent.testing.fake_llm import FakeRankModel, default_ranking
from eligibility_agent.testing.scenarios import (
    PAN_OK,
    full_happy_reported_emi_scenario,
    happy_quick_scenario,
    pan_invalid_scenario,
    score_failure_scenario,
)
from tests.graph_builders import HMAC_KEY, LC_ID, LC_TYPED_NAME, MOBILE, SpyRepo, World, make_world

TOKEN: Final[str] = "test-token-0123"
AUTH: Final[dict[str, str]] = {"Authorization": f"Bearer {TOKEN}"}
LEAD: Final[str] = "lead-api-1"
OPP: Final[str] = "opp-api-1"
UNKNOWN_ID: Final[str] = "00000000-0000-4000-8000-000000000001"

# Every env name the app (or langchain / langsmith) reads; scrubbed so a developer's shell cannot leak in, and RESTORED
# afterwards even when create_app writes one (setenv records the original state before delenv removes it).
_ENV_NAMES: Final[tuple[str, ...]] = (
    "SUREPASS_LIVE", "SUREPASS_TOKEN", "SUREPASS_BASE_URL", "SERVICE_API_TOKEN", "IDENTITY_HMAC_KEY", "REPO_BACKEND",
    "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "RAW_BUCKET", "DATABASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "PROMPT_SOURCE", "PROMPT_TAG",
    "GRAPH_TIMEOUT_SECONDS", "LOG_LEVEL", "LANGGRAPH_STRICT_MSGPACK", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    # langsmith memoises its env reads (get_env_var is lru_cached): clear on both sides so no test sees a stale switch.
    clear_langsmith_env_cache()
    package_logger = logging.getLogger("eligibility_agent")
    level = package_logger.level  # create_app sets it from LOG_LEVEL
    yield
    monkeypatch.undo()
    clear_langsmith_env_cache()
    package_logger.setLevel(level)


def clear_langsmith_env_cache() -> None:
    getattr(ls_utils, "get_env_var").cache_clear()


# --------------------------------------------------------------------------- builders


def api_settings(**over: Any) -> Settings:
    values: dict[str, Any] = {"SERVICE_API_TOKEN": TOKEN, "IDENTITY_HMAC_KEY": HMAC_KEY, **over}
    return cast(Any, Settings)(_env_file=None, **values)


def happy_quick(sink: CallLogSink) -> FakeSurePassClient:
    return happy_quick_scenario(sink=sink)


def happy_full(sink: CallLogSink) -> FakeSurePassClient:
    return full_happy_reported_emi_scenario(sink=sink)


def live_off(sink: CallLogSink) -> FakeSurePassClient:
    return FakeSurePassClient(sink=sink, calls_enabled=False)


def pan_status_invalid(sink: CallLogSink) -> FakeSurePassClient:
    return pan_invalid_scenario("status_invalid", pan=PAN_OK, sink=sink)


def score_timeout(sink: CallLogSink) -> FakeSurePassClient:
    return score_failure_scenario("timeout", sink=sink)


def score_http_500(sink: CallLogSink) -> FakeSurePassClient:
    return score_failure_scenario("http_500", sink=sink)


@dataclass
class Api:
    world: World
    app: FastAPI

    @property
    def repo(self) -> SpyRepo:
        return self.world.repo

    @property
    def runtime(self) -> main.AppRuntime:
        return main.runtime_of(self.app)

    def row(self, check_id: str) -> CheckRow:
        got = self.repo.get_check(check_id)
        assert got is not None, check_id
        return got

    def nothing_recorded(self) -> bool:
        return (
            self.repo.list_checks() == []
            and self.repo.list_consents() == []
            and self.repo.list_calls() == []
            and self.world.surepass.calls == []
        )

    def in_flight_held(self, pan_hash: str) -> bool:
        return any(lock.kind == "in_flight" for lock in self.repo.get_active_locks(pan_hash))


def make_api(
    factory: Callable[[CallLogSink], FakeSurePassClient] = happy_quick,
    *,
    rank: FakeRankModel | None = None,
    **settings_over: Any,
) -> Api:
    world = make_world(factory, rank=rank)
    app = main.create_app(
        api_settings(**settings_over),
        repo=world.repo,
        surepass=world.surepass,
        rank_model=world.rank,
        prompts=local_prompts(),
        now=world.clock.now,
    )
    return Api(world=world, app=app)


@contextmanager
def client_of(app: FastAPI, *, auth: bool = True) -> Generator[TestClient]:
    with TestClient(app, headers=AUTH if auth else None) as client:
        yield client


def consent_obj(**over: Any) -> dict[str, Any]:
    return {"given": True, "wording_version": "v2-quick", "call_ref": "sc-call-1", "ip": "10.0.0.1", **over}


def quick_payload(**over: Any) -> dict[str, Any]:
    return {
        "lc_id": LC_ID,
        "lead_id": LEAD,
        "opportunity_id": None,
        "name": LC_TYPED_NAME,
        "mobile": MOBILE,
        "pan": PAN_OK,
        "consent": consent_obj(),
        "monthly_income": 60000,
        "existing_emi": 0,
        **over,
    }


def full_payload(**over: Any) -> dict[str, Any]:
    base = quick_payload(consent=consent_obj(wording_version="v2-full"))
    return {**base, "opportunity_id": OPP, "opportunity_stage": "offer", **over}


def without(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in keys}


Mutation = Callable[[dict[str, Any]], dict[str, Any]]


def raw_json(payload: dict[str, Any], field: str, literal: str) -> str:
    """The payload as JSON text with ``field`` set to a raw JSON literal (NaN / Infinity are not valid JSON numbers but
    Python's decoder, and so the server's, accepts them)."""
    body = json.dumps(without(payload, field))
    return body[:-1] + f', "{field}": {literal}}}'


def pan_hash_of(pan: str = PAN_OK) -> str:
    return hash_pan(pan, key=HMAC_KEY)


def spy_fail_check(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Wrap the route's ``fail_check`` (still the real one underneath); returns the recorded keyword arguments."""
    calls: list[dict[str, Any]] = []
    real = nodes.fail_check

    def spy(deps: GraphDeps, **kwargs: Any) -> LcView:
        calls.append(kwargs)
        return real(deps, **kwargs)

    monkeypatch.setattr(main, "fail_check", spy)
    return calls


def quick_route_ok(api: Api, client: TestClient, **over: Any) -> dict[str, Any]:
    resp = client.post("/checks/quick", json=quick_payload(**over))
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


# =========================================================================== /health


def test_health_is_unauthenticated_and_exactly_two_keys() -> None:
    api = make_api()
    with client_of(api.app, auth=False) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "surepass_live": False}


def test_health_reports_the_exact_kill_switch_literal() -> None:
    api = make_api(SUREPASS_LIVE="true", SUREPASS_TOKEN="sp-token")
    with client_of(api.app, auth=False) as client:
        assert client.get("/health").json() == {"ok": True, "surepass_live": True}
    api = make_api(SUREPASS_LIVE="True", SUREPASS_TOKEN="sp-token")
    with client_of(api.app, auth=False) as client:
        assert client.get("/health").json() == {"ok": True, "surepass_live": False}


# =========================================================================== degraded boot


def test_degraded_boot_empty_env_serves_health_and_refuses_live_calls_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(**kwargs: Any) -> RankModel:
        raise AssertionError("make_rank_model must not be called without an OpenAI key")

    monkeypatch.setattr(main, "make_rank_model", never)
    app = main.create_app(cast(Any, Settings)(_env_file=None))
    runtime = main.runtime_of(app)
    assert isinstance(runtime.deps.repo, MemoryRepo)
    assert isinstance(runtime.deps.surepass, LiveSurePassClient) and runtime.deps.surepass.calls_enabled is False
    with client_of(app, auth=False) as client:  # auth is disabled: no token configured
        assert client.get("/health").json() == {"ok": True, "surepass_live": False}
        resp = client.post("/checks/quick", json=quick_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert (body["status"], body["code"]) == ("refused", "LIVE_CALLS_DISABLED")
    assert body["label"] == REFUSAL_MESSAGES["LIVE_CALLS_DISABLED"]
    repo = runtime.deps.repo
    assert isinstance(repo, MemoryRepo)
    [row] = repo.list_checks()
    assert (row.id, row.status, row.code, row.billed) == (body["check_id"], "refused", "LIVE_CALLS_DISABLED", False)
    assert repo.list_calls() == []


def test_the_owned_surepass_client_is_closed_at_shutdown() -> None:
    app = main.create_app(cast(Any, Settings)(_env_file=None))
    surepass = main.runtime_of(app).deps.surepass
    assert isinstance(surepass, LiveSurePassClient)
    with client_of(app, auth=False):
        assert surepass._http.is_closed is False
    assert surepass._http.is_closed is True


class _StubPool:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("pool.close")


def test_a_database_url_builds_the_k21_checkpointer_in_the_lifespan_and_closes_its_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    saver = InMemorySaver(serde=make_checkpoint_serde())

    def factory(db_url: str) -> tuple[InMemorySaver, _StubPool]:
        events.append(db_url)
        return saver, _StubPool(events)

    monkeypatch.setattr(main, "make_postgres_checkpointer", factory)
    dsn = "postgresql://user:pw@db.invalid:5432/app"
    api = make_api(DATABASE_URL=dsn)
    assert events == []  # nothing connects at construction
    with client_of(api.app) as client:
        assert events == [dsn]
        body = quick_route_ok(api, client)
        checkpoint = saver.get_tuple({"configurable": {"thread_id": body["check_id"], "checkpoint_ns": ""}})
        assert checkpoint is not None  # the graphs were recompiled onto the K21 saver, keyed by check_id
    assert events == [dsn, "pool.close"]


def test_a_failing_checkpointer_build_fails_the_boot_and_still_closes_the_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(db_url: str) -> tuple[InMemorySaver, _StubPool]:
        raise CheckpointerConfigError("LANGGRAPH_STRICT_MSGPACK=true is required")

    monkeypatch.setattr(main, "make_postgres_checkpointer", refuse)
    app = main.create_app(cast(Any, Settings)(_env_file=None, DATABASE_URL="postgresql://u:p@db.invalid:5432/app"))
    surepass = main.runtime_of(app).deps.surepass
    assert isinstance(surepass, LiveSurePassClient)
    with pytest.raises(CheckpointerConfigError), TestClient(app):
        pass
    assert surepass._http.is_closed is True


def test_the_supabase_backend_is_built_from_settings_with_the_raw_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, str, str]] = []
    repo = MemoryRepo()

    def factory(url: str, key: str, *, bucket: str) -> tuple[MemoryRepo, object]:
        built.append((url, key, bucket))
        return repo, object()

    monkeypatch.setattr(main, "create_supabase_backend", factory)
    app = main.create_app(
        api_settings(
            REPO_BACKEND="supabase",
            SUPABASE_URL="https://project.invalid",
            SUPABASE_SERVICE_ROLE_KEY="service-key",
            IDENTITY_HMAC_KEY="prod-like-key-0123456789",
            RAW_BUCKET="bucket-x",
        )
    )
    assert built == [("https://project.invalid", "service-key", "bucket-x")]
    assert main.runtime_of(app).deps.repo is repo


def test_degraded_boot_rank_model_is_an_unconfigured_stub_that_raises() -> None:
    app = main.create_app(cast(Any, Settings)(_env_file=None))
    rank = main.runtime_of(app).deps.rank_model
    ctx = cast(RankContext, object())
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        rank(ctx)


def test_the_real_rank_model_is_built_with_the_settings_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> RankModel:
        built.append(kwargs)
        return FakeRankModel()

    monkeypatch.setattr(main, "make_rank_model", spy)
    prompts = local_prompts()
    app = main.create_app(api_settings(OPENAI_API_KEY="sk-test-not-real", OPENAI_MODEL="gpt-test"), prompts=prompts)
    assert built == [{"model": "gpt-test", "prompts": prompts, "api_key": "sk-test-not-real"}]
    assert isinstance(main.runtime_of(app).deps.rank_model, FakeRankModel)


def test_startup_errors_raise_config_error() -> None:
    with pytest.raises(ConfigError, match="SUPABASE_URL"):
        main.create_app(api_settings(REPO_BACKEND="supabase"))
    with pytest.raises(ConfigError, match="IDENTITY_HMAC_KEY"):
        main.create_app(api_settings(IDENTITY_HMAC_KEY=""))


def test_app_factory_builds_from_get_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = api_settings()
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    app = main.app_factory()
    assert main.runtime_of(app).settings is settings


def test_create_app_hardens_the_langsmith_env_against_payload_export() -> None:
    import os

    main.create_app(api_settings())
    assert (os.environ.get("LANGSMITH_HIDE_INPUTS"), os.environ.get("LANGSMITH_HIDE_OUTPUTS")) == ("true", "true")
    env: dict[str, str] = {"LANGSMITH_HIDE_INPUTS": "false"}
    main.harden_tracing_env(env)
    assert env == {"LANGSMITH_HIDE_INPUTS": "true", "LANGSMITH_HIDE_OUTPUTS": "true"}


def test_create_app_clears_langsmiths_memoised_env_reads_so_the_hide_switch_takes_effect() -> None:
    # langsmith's get_env_var is lru_cached: a read made BEFORE create_app (here: env unset -> None) would otherwise keep
    # the stale value for the life of the process. No cache_clear in this test between the read and the assertions.
    read_env = getattr(ls_utils, "get_env_var")
    assert (read_env("HIDE_INPUTS"), read_env("HIDE_OUTPUTS")) == (None, None)
    main.create_app(api_settings())
    assert (read_env("HIDE_INPUTS"), read_env("HIDE_OUTPUTS")) == ("true", "true")
    default_client = Client(api_key="k", api_url="http://langsmith.invalid", auto_batch_tracing=False)
    assert (getattr(default_client, "_hide_inputs"), getattr(default_client, "_hide_outputs")) == (True, True)


# =========================================================================== auth (K21)

_ROUTES: Final[list[tuple[str, dict[str, Any]]]] = [
    ("/checks/quick", quick_payload()),
    ("/checks/full", full_payload()),
    (f"/checks/{UNKNOWN_ID}/recompute", {"monthly_income": 50000}),
    (f"/checks/{UNKNOWN_ID}/attach", {"contact_id": "c_1"}),
]
_BAD_HEADERS: Final[dict[str, dict[str, str]]] = {
    "missing": {},
    "wrong_token": {"Authorization": "Bearer not-the-token"},
    "token_prefix": {"Authorization": f"Bearer {TOKEN}x"},
    "basic_scheme": {"Authorization": "Basic dGVzdDp0ZXN0"},
    "token_scheme": {"Authorization": f"Token {TOKEN}"},
    "bearer_no_token": {"Authorization": "Bearer"},
    "bare_token": {"Authorization": TOKEN},
    "empty": {"Authorization": ""},
}


@pytest.mark.parametrize("headers", list(_BAD_HEADERS.values()), ids=list(_BAD_HEADERS))
@pytest.mark.parametrize(("path", "payload"), _ROUTES, ids=["quick", "full", "recompute", "attach"])
def test_every_checks_route_rejects_a_missing_wrong_or_malformed_bearer(
    path: str, payload: dict[str, Any], headers: dict[str, str]
) -> None:
    api = make_api()
    with client_of(api.app, auth=False) as client:
        resp = client.post(path, json=payload, headers=headers)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid_or_missing_token"}
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    assert api.nothing_recorded()


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_no_unauthenticated_route_exists_besides_health(path: str) -> None:
    # PRD 6: bearer auth on ALL routes except /health; FastAPI's schema/docs routes would publish the request schema.
    api = make_api()
    with client_of(api.app, auth=False) as client:
        assert client.get(path).status_code == 404
    assert (api.app.docs_url, api.app.redoc_url, api.app.openapi_url) == (None, None, None)


@pytest.mark.parametrize(("configured", "expected"), [("chatty", logging.INFO), ("debug", logging.DEBUG), ("ERROR", logging.ERROR)])
def test_an_unknown_log_level_never_stops_the_boot(
    monkeypatch: pytest.MonkeyPatch, configured: str, expected: int
) -> None:
    # Under pytest the root logger already has handlers, so a real basicConfig would silently skip its level check;
    # the spy sees what uvicorn's bare root would receive (an unknown NAME there raises ValueError).
    levels: list[object] = []

    def basic_config(**kwargs: object) -> None:
        levels.append(kwargs.get("level"))

    monkeypatch.setattr(logging, "basicConfig", basic_config)
    settings = api_settings(LOG_LEVEL=configured)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    assert main.runtime_of(main.app_factory()).settings is settings
    assert levels == [expected]
    assert logging.getLogger("eligibility_agent").level == expected


def test_auth_is_checked_before_the_body_so_an_anonymous_caller_learns_nothing_about_the_schema() -> None:
    api = make_api()
    with client_of(api.app, auth=False) as client:
        resp = client.post("/checks/quick", json={"pan": "nonsense"})
    assert resp.status_code == 401


def test_the_right_token_passes_and_auth_is_open_only_when_no_token_is_configured() -> None:
    api = make_api()
    with client_of(api.app) as client:
        assert client.post("/checks/quick", json=quick_payload()).status_code == 200
    open_api = make_api(SERVICE_API_TOKEN=None)
    with client_of(open_api.app, auth=False) as client:
        assert client.post("/checks/quick", json=quick_payload()).status_code == 200


def test_the_token_comparison_is_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = main.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(main.hmac, "compare_digest", spy)
    api = make_api()
    with client_of(api.app) as client:
        client.post("/checks/quick", json=quick_payload())
    assert seen == [(TOKEN.encode(), TOKEN.encode())]


# =========================================================================== consent (owner ruling: Literal[True])

_CONSENT_CASES: Final[dict[str, Mutation]] = {
    "absent": lambda p: without(p, "consent"),
    "null": lambda p: {**p, "consent": None},
    "a_bare_true": lambda p: {**p, "consent": True},
    "given_false": lambda p: {**p, "consent": consent_obj(given=False)},
    "given_string_true": lambda p: {**p, "consent": consent_obj(given="true")},
    "given_one": lambda p: {**p, "consent": consent_obj(given=1)},
    "given_one_float": lambda p: {**p, "consent": consent_obj(given=1.0)},
    "given_null": lambda p: {**p, "consent": consent_obj(given=None)},
    "given_missing": lambda p: {**p, "consent": without(consent_obj(), "given")},
    "wording_missing": lambda p: {**p, "consent": without(consent_obj(), "wording_version")},
    "wording_blank": lambda p: {**p, "consent": consent_obj(wording_version="   ")},
    "wording_too_long": lambda p: {**p, "consent": consent_obj(wording_version="v" * 33)},
    "extra_key": lambda p: {**p, "consent": consent_obj(recorded_by="lc")},
}


@pytest.mark.parametrize("mutate", list(_CONSENT_CASES.values()), ids=list(_CONSENT_CASES))
@pytest.mark.parametrize("kind", ["quick", "full"])
def test_consent_must_be_the_object_with_given_literally_true_or_nothing_is_recorded(
    kind: str, mutate: Mutation
) -> None:
    api = make_api(happy_quick if kind == "quick" else happy_full)
    payload = mutate(quick_payload() if kind == "quick" else full_payload())
    with client_of(api.app) as client:
        resp = client.post(f"/checks/{kind}", json=payload)
    assert resp.status_code == 422, resp.text
    assert api.nothing_recorded()  # no consents row, no checks row, no SurePass call


def test_a_valid_consent_object_is_recorded_verbatim_on_the_consent_row() -> None:
    api = make_api()
    with client_of(api.app) as client:
        body = quick_route_ok(api, client, name="  Typed Learnername  ")
    [consent] = api.repo.list_consents()
    assert (consent.wording_version, consent.call_ref, consent.ip, consent.covers) == (
        "v2-quick", "sc-call-1", "10.0.0.1", "score"
    )
    assert consent.lc_typed_name == LC_TYPED_NAME  # stripped
    assert api.row(body["check_id"]).consent_id == consent.id


def test_optional_consent_fields_may_be_omitted() -> None:
    api = make_api()
    with client_of(api.app) as client:
        quick_route_ok(api, client, consent={"given": True, "wording_version": "v2-quick"})
    [consent] = api.repo.list_consents()
    assert (consent.call_ref, consent.ip) == (None, None)


# =========================================================================== request bodies


@pytest.mark.parametrize("refresh", [True, False])
def test_a_quick_body_has_no_refresh_field(refresh: bool) -> None:
    api = make_api()
    with client_of(api.app) as client:
        resp = client.post("/checks/quick", json=quick_payload(refresh=refresh))
    assert resp.status_code == 422
    assert api.nothing_recorded()


_QUICK_SHAPE_CASES: Final[dict[str, Mutation]] = {
    "unknown_field": lambda p: {**p, "monthly_incom": 60000},
    "no_lc_id": lambda p: without(p, "lc_id"),
    "no_lead_id": lambda p: without(p, "lead_id"),
    "no_name": lambda p: without(p, "name"),
    "blank_name": lambda p: {**p, "name": "   "},
    "empty_lc_id": lambda p: {**p, "lc_id": ""},
    "long_lc_id": lambda p: {**p, "lc_id": "x" * 65},
    "empty_opp": lambda p: {**p, "opportunity_id": ""},
}


@pytest.mark.parametrize("mutate", list(_QUICK_SHAPE_CASES.values()), ids=list(_QUICK_SHAPE_CASES))
def test_quick_body_shape_failures_are_422(mutate: Mutation) -> None:
    api = make_api()
    with client_of(api.app) as client:
        resp = client.post("/checks/quick", json=mutate(quick_payload()))
    assert resp.status_code == 422
    assert api.nothing_recorded()


@pytest.mark.parametrize(
    "pan", ["ABCDE12345", "ABCD1234F", "", "ABCPE1234FF", "ABCPE1234ı", "ABCPE 1234F", 1234567890, None]
)
def test_a_malformed_pan_is_422(pan: object) -> None:
    api = make_api()
    with client_of(api.app) as client:
        resp = client.post("/checks/quick", json=quick_payload(pan=pan))
    assert resp.status_code == 422
    assert api.nothing_recorded()


def test_the_pan_is_normalised_through_policy_and_hashed_with_the_identity_key() -> None:
    api = make_api()
    with client_of(api.app) as client:
        body = quick_route_ok(api, client, pan="  abcpe1234f ")
    row = api.row(body["check_id"])
    assert (row.pan_hash, row.pan_last4) == (pan_hash_of(PAN_OK), "234F")
    assert [call.pan for call in api.world.surepass.calls] == [PAN_OK, PAN_OK]  # SurePass gets the normalised PAN


@pytest.mark.parametrize("mobile", ["5876543210", "98765", "+1 9876543210", "98765432101", "", "98765abcde", 9876543210])
def test_a_malformed_mobile_is_422(mobile: object) -> None:
    api = make_api()
    with client_of(api.app) as client:
        resp = client.post("/checks/quick", json=quick_payload(mobile=mobile))
    assert resp.status_code == 422
    assert api.nothing_recorded()


@pytest.mark.parametrize("mobile", ["+91 98765-43210", "919876543210", "09876543210", "(98765) 43210", MOBILE])
def test_the_mobile_is_normalised_through_policy(mobile: str) -> None:
    api = make_api()
    with client_of(api.app) as client:
        body = quick_route_ok(api, client, mobile=mobile)
    row = api.row(body["check_id"])
    assert (row.mobile_hash, row.mobile_last4) == (hash_mobile(MOBILE, key=HMAC_KEY), MOBILE[-4:])
    score_call = api.world.surepass.calls[-1]
    assert score_call.body["mobile"] == MOBILE


_MONEY_CASES: Final[list[tuple[str, str]]] = [
    ("monthly_income", "NaN"),
    ("monthly_income", "Infinity"),
    ("monthly_income", "-Infinity"),
    ("monthly_income", "-1"),
    ("monthly_income", "0"),
    ("monthly_income", "100000001"),
    ("monthly_income", "true"),
    ("monthly_income", '"60000"'),
    ("existing_emi", "NaN"),
    ("existing_emi", "Infinity"),
    ("existing_emi", "-0.01"),
    ("existing_emi", "100000001"),
    ("existing_emi", "false"),
    ("existing_emi", '"0"'),
]


@pytest.mark.parametrize(("field", "literal"), _MONEY_CASES, ids=[f"{f}={v}" for f, v in _MONEY_CASES])
@pytest.mark.parametrize("kind", ["quick", "full"])
def test_declared_figures_must_be_finite_non_negative_json_numbers(kind: str, field: str, literal: str) -> None:
    api = make_api(happy_quick if kind == "quick" else happy_full)
    payload = quick_payload() if kind == "quick" else full_payload()
    with client_of(api.app) as client:
        resp = client.post(
            f"/checks/{kind}", content=raw_json(payload, field, literal), headers={"Content-Type": "application/json"}
        )
    assert resp.status_code == 422, resp.text
    assert api.nothing_recorded()


def test_a_zero_declared_emi_and_absent_quick_figures_are_accepted() -> None:
    api = make_api()
    with client_of(api.app) as client:
        quick_route_ok(api, client, existing_emi=0)
        quick_route_ok(api, client, monthly_income=None, existing_emi=None)
        quick_route_ok(api, client, **{"monthly_income": 60000.5})


def test_422_bodies_carry_loc_msg_type_only_and_never_echo_the_request() -> None:
    api = make_api()
    secret_pan, secret_mobile, secret_name = "ABCPE1234", "5876543210", "Echo Testname"
    payload = quick_payload(
        pan=secret_pan,
        mobile=secret_mobile,
        name=secret_name,
        consent=consent_obj(given="true", call_ref=f"ref-{secret_pan}"),
        monthly_income=-5,
        extra_field=PAN_OK,
    )
    with client_of(api.app) as client:
        resp = client.post("/checks/quick", json=payload)
        broken = client.post(
            "/checks/quick", content=json.dumps(quick_payload())[:40], headers={"Content-Type": "application/json"}
        )
    for response in (resp, broken):
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail and all(set(item) == {"loc", "msg", "type"} for item in detail)
        for secret in (secret_pan, secret_mobile, secret_name, PAN_OK, MOBILE, LC_TYPED_NAME):
            assert secret not in response.text
    assert api.nothing_recorded()


def test_an_unknown_key_is_never_echoed_even_through_the_error_loc() -> None:
    # A caller-chosen KEY shaped like a PAN would otherwise come back verbatim as the last ``loc`` segment.
    api = make_api()
    key = "ZZZPE9999Q"
    with client_of(api.app) as client:
        top = client.post("/checks/quick", json={**quick_payload(), key: 1})
        nested = client.post("/checks/quick", json=quick_payload(consent=consent_obj(**{key: True})))
        attach = client.post(f"/checks/{UNKNOWN_ID}/attach", json={"contact_id": "c_1", key: "x"})
    for response, loc in (
        (top, ["body", main.EXTRA_FIELD_LOC]),
        (nested, ["body", "consent", main.EXTRA_FIELD_LOC]),
        (attach, ["body", main.EXTRA_FIELD_LOC]),
    ):
        assert response.status_code == 422
        assert key not in response.text
        assert response.json()["detail"] == [
            {"loc": loc, "msg": "Extra inputs are not permitted", "type": "extra_forbidden"}
        ]
    assert api.nothing_recorded()


_BLANK_ID_CASES: Final[dict[str, tuple[str, Mutation]]] = {
    "blank_lc_id": ("quick", lambda p: {**p, "lc_id": "   "}),
    "blank_lead_id": ("quick", lambda p: {**p, "lead_id": "\t"}),
    "blank_quick_opp": ("quick", lambda p: {**p, "opportunity_id": "  "}),
    "blank_full_opp": ("full", lambda p: {**p, "opportunity_id": " "}),
    "blank_full_lc_id": ("full", lambda p: {**p, "lc_id": "  "}),
}


@pytest.mark.parametrize(("kind", "mutate"), list(_BLANK_ID_CASES.values()), ids=list(_BLANK_ID_CASES))
def test_whitespace_only_identifiers_are_422(kind: str, mutate: Mutation) -> None:
    api = make_api(happy_quick if kind == "quick" else happy_full)
    with client_of(api.app) as client:
        resp = client.post(f"/checks/{kind}", json=mutate(quick_payload() if kind == "quick" else full_payload()))
    assert resp.status_code == 422
    assert api.nothing_recorded()


def test_padded_identifiers_are_stripped_so_a_padded_lc_id_shares_its_twins_quota_bucket() -> None:
    api = make_api()
    with client_of(api.app) as client:
        body = quick_route_ok(api, client, lc_id=f" {LC_ID} ", lead_id=f"{LEAD}\t", opportunity_id=" opp-pad-1 ")
    row = api.row(body["check_id"])
    assert (row.lc_id, row.lead_id, row.opportunity_id) == (LC_ID, LEAD, "opp-pad-1")
    [consent] = api.repo.list_consents()
    assert (consent.lc_id, consent.lead_id, consent.opportunity_id) == (LC_ID, LEAD, "opp-pad-1")
    assert row.counted_against_quota is True
    assert api.repo.quick_checks_used_today(LC_ID) == 1  # the SAME bucket as the unpadded id
    assert body["quick_checks_left_today"] == 4


def test_a_padded_full_opportunity_id_is_stored_stripped() -> None:
    api = make_api(happy_full)
    with client_of(api.app) as client:
        body = client.post("/checks/full", json=full_payload(opportunity_id=f"  {OPP} ")).json()
    assert api.row(body["check_id"]).opportunity_id == OPP


# =========================================================================== full checks + stage gate (K19)


_FULL_SHAPE_CASES: Final[dict[str, Mutation]] = {
    "no_opp": lambda p: without(p, "opportunity_id"),
    "null_opp": lambda p: {**p, "opportunity_id": None},
    "no_stage": lambda p: without(p, "opportunity_stage"),
    "null_stage": lambda p: {**p, "opportunity_stage": None},
    "no_income": lambda p: without(p, "monthly_income"),
    "null_income": lambda p: {**p, "monthly_income": None},
    "string_refresh": lambda p: {**p, "refresh": "true"},
    "int_ack": lambda p: {**p, "acknowledge_ntc_warning": 1},
}


@pytest.mark.parametrize("mutate", list(_FULL_SHAPE_CASES.values()), ids=list(_FULL_SHAPE_CASES))
def test_full_check_required_fields_and_strict_flags(mutate: Mutation) -> None:
    api = make_api(happy_full)
    with client_of(api.app) as client:
        resp = client.post("/checks/full", json=mutate(full_payload()))
    assert resp.status_code == 422
    assert api.nothing_recorded()


@pytest.mark.parametrize("stage", ["prospect", "Offer", "OFFER", " offer", "", 5])
def test_an_unknown_stage_is_a_422_enum_membership_failure(stage: object) -> None:
    api = make_api(happy_full)
    with client_of(api.app) as client:
        resp = client.post("/checks/full", json=full_payload(opportunity_stage=stage))
    assert resp.status_code == 422
    assert api.nothing_recorded()


def test_a_known_early_stage_is_a_persisted_stage_too_early_refusal_with_the_stage_stored_verbatim() -> None:
    api = make_api(happy_full)
    with client_of(api.app) as client:
        resp = client.post("/checks/full", json=full_payload(opportunity_stage="qualified"))
    body = resp.json()
    assert (resp.status_code, body["status"], body["code"]) == (200, "refused", "STAGE_TOO_EARLY")
    row = api.row(body["check_id"])
    assert (row.opportunity_stage, row.opportunity_id) == ("qualified", OPP)
    assert api.world.surepass.calls == []


def test_an_admitted_full_check_stores_its_stage_and_flags() -> None:
    api = make_api(happy_full)
    with client_of(api.app) as client:
        resp = client.post(
            "/checks/full", json=full_payload(opportunity_stage="registration", refresh=True, acknowledge_ntc_warning=True)
        )
    body = resp.json()
    assert (resp.status_code, body["status"], body["kind"]) == (200, "ok", "full")
    assert api.row(body["check_id"]).opportunity_stage == "registration"


# =========================================================================== route order + wire shape


def test_literal_routes_are_never_captured_by_the_check_id_routes() -> None:
    api = make_api()
    with client_of(api.app) as client:
        quick = client.post("/checks/quick", json=quick_payload())
        assert quick.status_code == 200
        LcView.model_validate(quick.json())
        assert client.post("/checks/quick/recompute", json={}).json() == {"detail": "check_not_found"}
        assert client.post("/checks/full/attach", json={"contact_id": "c_1"}).json() == {"detail": "check_not_found"}
        assert client.get("/checks/quick").status_code == 405


def test_check_ids_are_app_side_uuid4() -> None:
    api = make_api()
    with client_of(api.app) as client:
        body = quick_route_ok(api, client)
    assert uuid.UUID(body["check_id"]).version == 4


_STATUS_WORLDS: Final[dict[str, Callable[[CallLogSink], FakeSurePassClient]]] = {
    "ok": happy_quick,
    "refused": live_off,
    "pan_not_verified": pan_status_invalid,
    "pending_verification": score_timeout,
    "service_failure": score_http_500,
}


@pytest.mark.parametrize("status", list(_STATUS_WORLDS), ids=list(_STATUS_WORLDS))
def test_every_status_round_trips_through_lc_view_with_explicit_nulls(status: str) -> None:
    api = make_api(_STATUS_WORLDS[status])
    with client_of(api.app) as client:
        body = quick_route_ok(api, client)
    assert body["status"] == status
    assert LcView.model_validate(body).model_dump(mode="json") == body
    assert set(body) == set(LcView.model_fields)  # every key present, nulls explicit


# =========================================================================== the error boundary


def test_a_graph_exception_returns_the_error_view_and_fail_check_records_and_releases(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    api = make_api()
    api.repo.fail_lenders = RuntimeError(f"lender lookup for {PAN_OK} failed")
    failed = spy_fail_check(monkeypatch)
    with caplog.at_level(logging.DEBUG), client_of(api.app) as client:
        body = quick_route_ok(api, client)
    check_id = body["check_id"]
    assert body == build_error_lc_view(check_id=check_id, kind="quick").model_dump(mode="json")
    assert [f["check_id"] for f in failed] == [check_id]
    assert (failed[0]["pan_hash"], failed[0]["pan_last4"], failed[0]["lc_id"]) == (pan_hash_of(), "234F", LC_ID)
    row = api.row(check_id)
    assert (row.status, row.code, row.billed) == ("error", None, True)  # the Score call was billed before the raise
    assert not api.in_flight_held(pan_hash_of())
    for secret in (PAN_OK, MOBILE, LC_TYPED_NAME):
        assert secret not in caplog.text
    assert "RuntimeError" in caplog.text and check_id in caplog.text


_BAD_FINALS: Final[dict[str, Callable[[str], Mapping[str, Any]]]] = {
    "empty": lambda check_id: {},
    "null_view": lambda check_id: {"lc_view": None},
    "garbage_view": lambda check_id: {"lc_view": "not a view"},
    "another_checks_view": lambda check_id: {"lc_view": build_error_lc_view(check_id=UNKNOWN_ID, kind="quick")},
}


@pytest.mark.parametrize("bad", list(_BAD_FINALS.values()), ids=list(_BAD_FINALS))
def test_a_graph_that_returns_without_a_usable_view_goes_through_fail_check(
    monkeypatch: pytest.MonkeyPatch, bad: Callable[[str], Mapping[str, Any]]
) -> None:
    api = make_api()
    failed = spy_fail_check(monkeypatch)

    def broken_graph(state: CheckState, config: RunnableConfig) -> Mapping[str, Any]:
        return bad(state.get("check_id") or "")

    api.runtime.graphs["quick"] = broken_graph
    with client_of(api.app) as client:
        body = quick_route_ok(api, client)
    assert [f["check_id"] for f in failed] == [body["check_id"]]
    assert body == build_error_lc_view(check_id=body["check_id"], kind="quick").model_dump(mode="json")
    row = api.row(body["check_id"])  # the graph never ran, so the guarded error row is inserted
    assert (row.status, row.pan_hash, row.pan_last4, row.lc_id) == ("error", pan_hash_of(), "234F", LC_ID)


def test_a_graph_raising_timeout_error_itself_is_an_exception_not_a_route_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    api = make_api()
    failed = spy_fail_check(monkeypatch)

    def raising(state: CheckState, config: RunnableConfig) -> Mapping[str, Any]:
        raise TimeoutError("a library timed out inside the graph")

    api.runtime.graphs["quick"] = raising
    with client_of(api.app) as client:
        body = quick_route_ok(api, client)
    assert [f["check_id"] for f in failed] == [body["check_id"]]


def test_a_route_timeout_records_the_error_row_directly_and_leaves_the_in_flight_lock_to_its_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    entered = threading.Event()

    def slow(ctx: RankContext) -> ModelOutput:
        entered.set()
        release.wait(10)
        return default_ranking(ctx)

    api = make_api(rank=FakeRankModel(script=[slow]), GRAPH_TIMEOUT_SECONDS=0.3)
    failed = spy_fail_check(monkeypatch)
    try:
        with client_of(api.app) as client:
            body = quick_route_ok(api, client)
            check_id = body["check_id"]
            assert entered.is_set()
            assert body == build_error_lc_view(check_id=check_id, kind="quick").model_dump(mode="json")
            row = api.row(check_id)
            assert (row.status, row.billed) == ("error", True)  # Score already billed when the budget ran out
            assert row.internal == {"error": main.ROUTE_TIMEOUT_MESSAGE}
            assert failed == []  # never fail_check on a route timeout ...
            assert api.in_flight_held(pan_hash_of())  # ... so the live graph thread keeps its lock
            release.set()
            deadline = time.monotonic() + 5
            while api.in_flight_held(pan_hash_of()) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not api.in_flight_held(pan_hash_of())  # persist's finally released it once the graph finished
            assert api.row(check_id).status == "error"  # persist's terminal update was a guarded no-op
    finally:
        release.set()


def test_a_graph_that_raises_after_its_route_timed_out_logs_the_exception_type_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release = threading.Event()

    def late_raise(state: CheckState, config: RunnableConfig) -> Mapping[str, Any]:
        release.wait(10)
        raise RuntimeError(f"late failure for {PAN_OK} / {MOBILE}")

    api = make_api(GRAPH_TIMEOUT_SECONDS=0.2)
    api.runtime.graphs["quick"] = late_raise
    try:
        with caplog.at_level(logging.WARNING, logger="eligibility_agent"), client_of(api.app) as client:
            body = quick_route_ok(api, client)
            check_id = body["check_id"]
            assert body == build_error_lc_view(check_id=check_id, kind="quick").model_dump(mode="json")
            assert not [r for r in caplog.records if "after its route timed out" in r.getMessage()]
            release.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not [
                r for r in caplog.records if "after its route timed out" in r.getMessage()
            ]:
                time.sleep(0.01)
    finally:
        release.set()
    [record] = [r for r in caplog.records if "after its route timed out" in r.getMessage()]
    text = record.getMessage()
    assert check_id in text and "RuntimeError" in text
    assert PAN_OK not in caplog.text and MOBILE not in caplog.text and "late failure" not in caplog.text


def _raising_graph(state: CheckState, config: RunnableConfig) -> Mapping[str, Any]:
    raise ValueError(f"boom {PAN_OK}")


def _ok_graph(state: CheckState, config: RunnableConfig) -> Mapping[str, Any]:
    return {"lc_view": None}


def test_the_late_failure_is_logged_exactly_once_whichever_side_finishes_first(caplog: pytest.LogCaptureFixture) -> None:
    state: CheckState = {"check_id": "chk-late"}
    config: RunnableConfig = {}
    with caplog.at_level(logging.WARNING, logger="eligibility_agent"):
        # Graph finished first (the wait_for cancel raced a ready result): the route's abandon() logs it.
        early = main._GraphRun(check_id="chk-race")
        outcome = early.call(_raising_graph, state, config)
        assert isinstance(outcome.error, ValueError)
        assert not caplog.records
        early.abandon()
        # Route abandoned first: the graph thread logs it when it finishes.
        late = main._GraphRun(check_id="chk-late")
        late.abandon()
        late.call(_raising_graph, state, config)
        # A late SUCCESS and a run that was never abandoned log nothing.
        quiet = main._GraphRun(check_id="chk-ok")
        quiet.abandon()
        quiet.call(_ok_graph, state, config)
        main._GraphRun(check_id="chk-normal").call(_raising_graph, state, config)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "chk-race" in messages[0] and "ValueError" in messages[0]
    assert "chk-late" in messages[1] and "ValueError" in messages[1]
    assert PAN_OK not in caplog.text and "boom" not in caplog.text


# =========================================================================== runnable config + tracing (K21 / T11)


@dataclass
class Invocation:
    state: CheckState
    config: RunnableConfig


def spy_graphs(api: Api, table: str = "graphs") -> list[Invocation]:
    seen: list[Invocation] = []
    graphs = cast("dict[str, main.GraphInvoker]", getattr(api.runtime, table))
    for kind, real in list(graphs.items()):

        def wrapped(state: CheckState, config: RunnableConfig, _real: main.GraphInvoker = real) -> Mapping[str, Any]:
            seen.append(Invocation(copy.copy(state), config))
            return _real(state, config)

        graphs[kind] = wrapped
    return seen


def test_runnable_config_threads_check_id_and_carries_only_the_pan_hash() -> None:
    api = make_api()
    seen = spy_graphs(api)
    with client_of(api.app) as client:
        body = quick_route_ok(api, client)
    [call] = seen
    check_id = body["check_id"]
    assert call.config.get("configurable") == {"thread_id": check_id}
    assert call.config.get("metadata") == {"check_id": check_id, "pan_hash": pan_hash_of(), "tier": "quick"}
    assert call.config.get("tags") == ["tier:quick", f"lc:{LC_ID}"]
    assert "run_id" not in call.config and "callbacks" not in call.config  # tracing off
    text = repr(call.config)
    for secret in (PAN_OK, MOBILE, LC_TYPED_NAME):
        assert secret not in text
    assert (call.state.get("check_id"), call.state.get("pan_hash"), call.state.get("kind")) == (check_id, pan_hash_of(), "quick")
    assert call.state.get("mobile_hash") == hash_mobile(MOBILE, key=HMAC_KEY)
    assert call.state.get("langsmith_run_id") is None


class RecordingClient(Client):
    """A real langsmith Client (its hide/anonymize logic runs) whose send hooks record instead of doing I/O."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(auto_batch_tracing=False, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.exported: list[dict[str, Any]] = []

    def _create_run(self, run_create: dict[str, Any], **kwargs: Any) -> None:
        self.exported.append(copy.deepcopy(run_create))

    def _update_run(self, run_update: dict[str, Any], **kwargs: Any) -> None:
        self.exported.append(copy.deepcopy(run_update))


@dataclass
class TraceSpy:
    clients: list[Client]
    client_kwargs: list[dict[str, Any]]
    tracers: list[tracing.PayloadFreeTracer]


def spy_tracing(monkeypatch: pytest.MonkeyPatch) -> TraceSpy:
    spy = TraceSpy(clients=[], client_kwargs=[], tracers=[])
    real_client, real_tracer = tracing.make_langsmith_client, tracing.make_tracer

    def make_client(**kwargs: Any) -> Client:
        spy.client_kwargs.append(kwargs)
        client = real_client(api_key="test-key", api_url="http://langsmith.invalid", client_cls=RecordingClient)
        spy.clients.append(client)
        return client

    def make_tracer(client: Client, *, project_name: str) -> tracing.PayloadFreeTracer:
        tracer = real_tracer(client, project_name=project_name)
        spy.tracers.append(tracer)
        return tracer

    monkeypatch.setattr(tracing, "make_langsmith_client", make_client)
    monkeypatch.setattr(tracing, "make_tracer", make_tracer)
    return spy


def metadata_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    extra: object = payload.get("extra")
    metadata: object = cast("dict[str, Any]", extra).get("metadata") if isinstance(extra, dict) else None
    return cast("dict[str, Any]", metadata) if isinstance(metadata, dict) else {}


_STANDALONE_SCORE = re.compile(r"(?<![0-9A-Za-z])745(?![0-9A-Za-z])")


def test_traced_checks_attach_our_payload_free_tracer_and_finish_on_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = spy_tracing(monkeypatch)
    api = make_api(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="ls-key", LANGSMITH_PROJECT="proj-test")
    seen = spy_graphs(api)
    with client_of(api.app) as client:
        body = quick_route_ok(api, client)
    [client_obj] = spy.clients
    assert spy.client_kwargs == [{"api_key": "ls-key"}]
    assert isinstance(client_obj, RecordingClient)
    assert getattr(client_obj, "_hide_inputs") is True and getattr(client_obj, "_anonymizer") is tracing.scrub_trace_payload
    [tracer] = spy.tracers
    assert tracer.client is client_obj and tracer.project_name == "proj-test"
    [call] = seen
    assert call.config.get("callbacks") == [tracer]  # OUR tracer, so langchain adds no default one
    run_id = call.config.get("run_id")
    assert isinstance(run_id, uuid.UUID)
    assert call.state.get("langsmith_run_id") == str(run_id)
    assert api.row(body["check_id"]).langsmith_run_id == str(run_id)
    client_obj.flush()

    exported = client_obj.exported
    assert exported
    for payload in exported:
        assert payload.get("inputs") in (None, {}) and payload.get("outputs") in (None, {})
        visible = repr({k: payload.get(k) for k in ("inputs", "outputs", "extra", "tags", "name", "events", "error")})
        for secret in (PAN_OK, MOBILE, LC_TYPED_NAME, "RAVI TESTKUMAR", "720-780"):
            assert secret not in visible, (secret, payload.get("name"))
        assert _STANDALONE_SCORE.search(visible) is None
    finished = [p for p in exported if str(p.get("id")) == str(run_id) and metadata_of(p).get("admission") == "new"]
    assert finished, "finish_trace must patch the root run through the SAME client"
    metadata = metadata_of(finished[-1])
    assert (metadata["billed"], metadata["counted_against_quota"], metadata["pan_hash"]) == (True, True, pan_hash_of())


def test_an_env_enabled_langchain_tracer_is_replaced_by_ours(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = spy_tracing(monkeypatch)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")  # langchain would otherwise add its own, payload-carrying tracer
    clear_langsmith_env_cache()
    api = make_api()
    seen = spy_graphs(api)
    with client_of(api.app) as client:
        quick_route_ok(api, client)
    [tracer] = spy.tracers
    assert seen[0].config.get("callbacks") == [tracer]


def test_without_tracing_no_langsmith_client_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = spy_tracing(monkeypatch)
    api = make_api()
    with client_of(api.app) as client:
        quick_route_ok(api, client)
    assert spy.clients == [] and spy.tracers == []
    assert api.runtime.trace_client is None


# =========================================================================== recompute (K9)


def test_recompute_returns_a_new_view_on_a_new_row_with_zero_surepass_calls() -> None:
    api = make_api()
    with client_of(api.app) as client:
        first = quick_route_ok(api, client)
        original = api.row(first["check_id"])
        calls_before = len(api.world.surepass.calls)
        resp = client.post(f"/checks/{first['check_id'].upper()}/recompute", json={"monthly_income": 90000, "existing_emi": 5000})
    assert resp.status_code == 200, resp.text
    second = resp.json()
    view = LcView.model_validate(second)
    assert view.check_id != first["check_id"] and view.status == "ok" and view.from_cache is True
    assert len(api.world.surepass.calls) == calls_before
    new_row = api.row(view.check_id)
    assert (new_row.recompute_of, new_row.billed, new_row.counted_against_quota) == (first["check_id"], False, False)
    assert api.row(first["check_id"]) == original  # the original row is untouched
    assert second["top_lender"]["ceiling"] != first["top_lender"]["ceiling"]  # re-derived on the new figures


def test_recompute_runs_through_the_recompute_graph_with_a_fresh_thread() -> None:
    api = make_api()
    with client_of(api.app) as client:
        first = quick_route_ok(api, client)
        seen = spy_graphs(api, "recompute_graphs")
        second = client.post(f"/checks/{first['check_id']}/recompute", json={"monthly_income": 70000}).json()
    [call] = seen
    assert call.config.get("configurable") == {"thread_id": second["check_id"]}
    assert call.config.get("metadata") == {"check_id": second["check_id"], "pan_hash": pan_hash_of(), "tier": "quick"}
    assert (call.state.get("monthly_income"), call.state.get("existing_emi_declared")) == (70000.0, None)


def _insert_row(api: Api, *, kind: str = "quick", status: str = "refused", code: str | None = "QUOTA_EXHAUSTED") -> str:
    check_id = str(uuid.uuid4())
    api.repo.insert_check(
        CheckRow(
            id=check_id, kind=cast(Any, kind), lc_id=LC_ID, lead_id=None, opportunity_id=None, opportunity_stage=None,
            contact_id=None, pan_hash=pan_hash_of(), pan_last4="234F", mobile_hash=None, mobile_last4=None,
            status=cast(Any, status), code=cast(Any, code), billed=False, counted_against_quota=False, is_refresh=False,
            recompute_of=None, consent_id=None, lc_view=None, internal=None, langsmith_run_id=None,
            created_at=api.world.clock.now(),
        )
    )
    return check_id


def test_recompute_refusals_are_404_and_409_details_outside_the_lc_view_shape() -> None:
    api = make_api()
    refused = _insert_row(api)
    with client_of(api.app) as client:
        unknown = client.post(f"/checks/{UNKNOWN_ID}/recompute", json={})
        not_a_uuid = client.post("/checks/not-a-uuid/recompute", json={})
        not_recomputable = client.post(f"/checks/{refused}/recompute", json={})
        ok = quick_route_ok(api, client)
        api.repo.raw_store.delete_raw(api.repo.raw_store.keys())  # purged raw (180-day job)
        purged = client.post(f"/checks/{ok['check_id']}/recompute", json={})
    assert (unknown.status_code, unknown.json()) == (404, {"detail": "check_not_found"})
    assert (not_a_uuid.status_code, not_a_uuid.json()) == (404, {"detail": "check_not_found"})
    assert (not_recomputable.status_code, not_recomputable.json()) == (409, {"detail": "not_recomputable"})
    assert (purged.status_code, purged.json()) == (409, {"detail": "raw_unavailable"})
    assert len(api.repo.list_checks()) == 2  # the refused seed + the one real check; no recompute row was created


@pytest.mark.parametrize(
    "body",
    [{"monthly_income": 1, "extra": 1}, {"monthly_income": -1}, {"existing_emi": -1}, {"monthly_income": "60000"}],
    ids=["extra_key", "negative_income", "negative_emi", "string_income"],
)
def test_recompute_body_validation(body: dict[str, Any]) -> None:
    api = make_api()
    with client_of(api.app) as client:
        first = quick_route_ok(api, client)
        resp = client.post(f"/checks/{first['check_id']}/recompute", json=body)
        nan = client.post(
            f"/checks/{first['check_id']}/recompute",
            content='{"monthly_income": NaN}',
            headers={"Content-Type": "application/json"},
        )
    assert (resp.status_code, nan.status_code) == (422, 422)
    assert len(api.repo.list_checks()) == 1


def test_a_full_check_recompute_needs_a_declared_income() -> None:
    api = make_api(happy_full)
    with client_of(api.app) as client:
        first = client.post("/checks/full", json=full_payload()).json()
        missing = client.post(f"/checks/{first['check_id']}/recompute", json={"existing_emi": 1000})
        ok = client.post(f"/checks/{first['check_id']}/recompute", json={"monthly_income": 80000})
    assert missing.status_code == 422
    assert all(set(item) == {"loc", "msg", "type"} for item in missing.json()["detail"])
    assert ok.status_code == 200 and ok.json()["kind"] == "full"
    assert len(api.repo.list_checks()) == 2


def test_a_recompute_seeding_exception_goes_through_fail_check_on_the_new_id(monkeypatch: pytest.MonkeyPatch) -> None:
    api = make_api()
    failed = spy_fail_check(monkeypatch)
    with client_of(api.app) as client:
        first = quick_route_ok(api, client)

        def storage_down(storage_key: str) -> dict[str, Any]:
            raise ConnectionError("storage unreachable")

        monkeypatch.setattr(api.repo, "fetch_raw", storage_down)
        resp = client.post(f"/checks/{first['check_id']}/recompute", json={"monthly_income": 70000})
    body = resp.json()
    assert resp.status_code == 200 and body["check_id"] != first["check_id"]
    assert body == build_error_lc_view(check_id=body["check_id"], kind="quick").model_dump(mode="json")
    assert [f["check_id"] for f in failed] == [body["check_id"]]
    row = api.row(body["check_id"])
    assert (row.status, row.kind, row.lc_id, row.pan_hash) == ("error", "quick", LC_ID, pan_hash_of())


def test_a_recompute_graph_exception_goes_through_fail_check_on_the_new_row() -> None:
    api = make_api()
    with client_of(api.app) as client:
        first = quick_route_ok(api, client)
        api.repo.fail_lenders = RuntimeError("lenders down")
        resp = client.post(f"/checks/{first['check_id']}/recompute", json={"monthly_income": 70000})
    body = resp.json()
    assert resp.status_code == 200 and body["check_id"] != first["check_id"]
    assert body == build_error_lc_view(check_id=body["check_id"], kind="quick").model_dump(mode="json")
    assert api.row(body["check_id"]).status == "error"
    assert api.row(first["check_id"]).status == "ok"


# =========================================================================== attach (K9 A-table)


def test_attach_matrix() -> None:
    api = make_api()
    full_id = _insert_row(api, kind="full", status="ok", code=None)
    with client_of(api.app) as client:
        check_id = quick_route_ok(api, client)["check_id"]
        first = client.post(f"/checks/{check_id}/attach", json={"contact_id": "c_1"})
        again = client.post(f"/checks/{check_id}/attach", json={"contact_id": "c_1"})
        other = client.post(f"/checks/{check_id}/attach", json={"contact_id": "c_2"})
        full = client.post(f"/checks/{full_id}/attach", json={"contact_id": "c_1"})
        unknown = client.post(f"/checks/{UNKNOWN_ID}/attach", json={"contact_id": "c_1"})
        not_a_uuid = client.post("/checks/nope/attach", json={"contact_id": "c_1"})
    expected = {"check_id": check_id, "contact_id": "c_1", "attached": True}
    assert (first.status_code, first.json()) == (200, expected)
    assert (again.status_code, again.json()) == (200, expected)  # idempotent
    assert (other.status_code, other.json()) == (409, {"detail": "already_attached"})
    assert (full.status_code, full.json()) == (409, {"detail": "attach_quick_only"})
    assert (unknown.status_code, unknown.json()) == (404, {"detail": "check_not_found"})
    assert (not_a_uuid.status_code, not_a_uuid.json()) == (404, {"detail": "check_not_found"})
    row = api.row(check_id)
    assert row.contact_id == "c_1"
    [consent] = api.repo.list_consents()
    assert consent.contact_id == "c_1"  # checks.contact_id AND consents.contact_id
    assert api.row(full_id).contact_id is None


@pytest.mark.parametrize(
    "body",
    [{}, {"contact_id": ""}, {"contact_id": "   "}, {"contact_id": "\t"}, {"contact_id": "x" * 65},
     {"contact_id": "c", "x": 1}, {"contact_id": 5}],
    ids=["empty_body", "empty", "spaces", "tab", "too_long", "extra_key", "not_a_string"],
)
def test_attach_body_validation(body: dict[str, Any]) -> None:
    api = make_api()
    with client_of(api.app) as client:
        check_id = quick_route_ok(api, client)["check_id"]
        resp = client.post(f"/checks/{check_id}/attach", json=body)
    assert resp.status_code == 422
    assert api.row(check_id).contact_id is None


def test_a_padded_contact_id_is_stripped_and_stays_idempotent_with_its_twin() -> None:
    api = make_api()
    with client_of(api.app) as client:
        check_id = quick_route_ok(api, client)["check_id"]
        padded = client.post(f"/checks/{check_id}/attach", json={"contact_id": " c_1 "})
        twin = client.post(f"/checks/{check_id}/attach", json={"contact_id": "c_1"})
    expected = {"check_id": check_id, "contact_id": "c_1", "attached": True}
    assert (padded.status_code, padded.json()) == (200, expected)
    assert (twin.status_code, twin.json()) == (200, expected)  # same contact, not a 409 conflict
    assert api.row(check_id).contact_id == "c_1"


# =========================================================================== typing seams


def test_the_unconfigured_rank_model_and_graph_invokers_satisfy_their_protocols() -> None:
    rank: RankModel = main.UnconfiguredRankModel()
    assert callable(rank)
    prompts: PromptBundle = local_prompts()
    assert prompts.source == "local"
