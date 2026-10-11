"""Session hardening + the shared scenario fixtures (PRD v1.1 section 12).

Hermetic environment (BINDING notes): at configure time every env name that could switch on a real dependency or leak
a developer's shell into the suite is removed (``SUREPASS_*``, ``OPENAI_API_KEY``, ``LANGSMITH_*``, ``LANGCHAIN_*``,
``SUPABASE_*``, ``DATABASE_URL``, ``RAW_BUCKET``, ``PROMPT_SOURCE``, ``PROMPT_TAG``, plus every other ``Settings`` alias),
tracing flags are forced off, and ``IDENTITY_HMAC_KEY`` is the test key. Matching is case-insensitive because
``Settings`` reads env names case-insensitively. An autouse guard re-checks the kill switch around every test. The
primary no-network guarantee stays pytest-socket (``--disable-socket`` in addopts; deliberately NO filterwarnings line).

Fixtures: ``clock`` (FrozenClock at T0 = 2026-10-10 10:00 IST), ``repo`` (MemoryRepo on that clock, the K17 seed lenders
L1/L2/L3 published), ``fake_rank`` (FakeRankModel happy path) and ``make_api``: a factory that wires one SurePass fake
(logging into the repo sink), the FakeRankModel, local prompts and ``now=clock.now`` into ``create_app``, exposes the app's
RUNTIME ``GraphDeps`` (``runtime_of(app).deps``) on the harness and returns a started TestClient (bearer attached).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Final

import pytest
from fastapi.testclient import TestClient

from eligibility_agent.config import Settings
from eligibility_agent.db.repo import MemoryRepo
from eligibility_agent.graph.prompts import local_prompts
from eligibility_agent.main import create_app, runtime_of
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.fake_llm import FakeRankModel
from tests.helpers import (
    AUTH,
    HMAC_KEY,
    T0,
    Api,
    ApiFactory,
    SurePassFactory,
    api_settings,
    seed_default_lenders,
    settings_aliases,
)

_SCRUB_PREFIXES: Final[tuple[str, ...]] = ("SUREPASS_", "LANGSMITH_", "LANGCHAIN_", "SUPABASE_")
_SCRUB_EXACT: Final[frozenset[str]] = frozenset(
    {"OPENAI_API_KEY", "DATABASE_URL", "RAW_BUCKET", "PROMPT_SOURCE", "PROMPT_TAG", "LANGGRAPH_STRICT_MSGPACK"}
)
FORCED_ENV: Final[dict[str, str]] = {
    "IDENTITY_HMAC_KEY": HMAC_KEY,
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
}


def _scrubbed(name: str) -> bool:
    upper = name.upper()
    return upper.startswith(_SCRUB_PREFIXES) or upper in _SCRUB_EXACT or upper in settings_aliases()


def pytest_configure(config: pytest.Config) -> None:
    del config
    for name in [n for n in os.environ if _scrubbed(n)]:
        del os.environ[name]
    os.environ.update(FORCED_ENV)


@pytest.fixture(autouse=True)
def _kill_switch_stays_off() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """No test may run, or leave the process, with the live switch on."""
    assert not _live_switch_on(), "SUREPASS_LIVE is 'true' in the test environment"
    yield
    assert not _live_switch_on(), "a test left SUREPASS_LIVE='true' behind"


def _live_switch_on() -> bool:
    return any(name.upper() == "SUREPASS_LIVE" and value == "true" for name, value in os.environ.items())


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(T0)


@pytest.fixture
def repo(clock: FrozenClock) -> MemoryRepo:
    seeded = MemoryRepo(clock.now)
    seed_default_lenders(seeded)
    return seeded


@pytest.fixture
def fake_rank() -> FakeRankModel:
    return FakeRankModel()


@pytest.fixture
def make_api(clock: FrozenClock, repo: MemoryRepo, fake_rank: FakeRankModel) -> Iterator[ApiFactory]:
    """``make_api(surepass_factory, *, rank=None, settings=None) -> Api``. The factory receives the repo as the audit
    sink. Every TestClient opened here is closed (lifespan shutdown) at teardown."""
    with ExitStack() as stack:

        def factory(
            surepass: SurePassFactory, *, rank: FakeRankModel | None = None, settings: Settings | None = None
        ) -> Api:
            fake = surepass(repo)
            model = rank if rank is not None else fake_rank
            app = create_app(
                settings if settings is not None else api_settings(),
                repo=repo,
                surepass=fake,
                rank_model=model,
                prompts=local_prompts(),
                now=clock.now,
            )
            deps = runtime_of(app).deps  # the GraphDeps the app's graphs actually run on, not a parallel copy
            assert (deps.repo, deps.surepass, deps.rank_model, deps.now) == (repo, fake, model, clock.now)
            client = stack.enter_context(TestClient(app, headers=AUTH))
            return Api(clock=clock, repo=repo, surepass=fake, rank=model, deps=deps, app=app, client=client)

        yield factory
