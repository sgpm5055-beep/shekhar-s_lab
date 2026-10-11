"""FastAPI composition root (PRD v1.1 sections 6, 10, 11): the app factory, five routes, bearer auth, request validation
and the error boundary. The service runs as ``uvicorn 'eligibility_agent.main:app_factory' --factory``; nothing is built
at import time (an import-time app would read a developer's ``.env`` during test collection).

Routes (literal paths are declared before the ``/{check_id}/...`` ones)::

    GET  /health                      {"ok": true, "surepass_live": <bool>}   no auth, degraded-boot safe
    POST /checks/quick                -> LcView
    POST /checks/full                 -> LcView
    POST /checks/{check_id}/recompute -> LcView | 404 check_not_found | 409 not_recomputable / raw_unavailable
    POST /checks/{check_id}/attach    -> {check_id, contact_id, attached} | 404 check_not_found
                                         | 409 attach_quick_only / already_attached

Every learner outcome (ok, refused, pan_not_verified, pending_verification, service_failure) is an HTTP 200 ``LcView``
persisted as a checks row; the only non-200s are 401 (bearer), 422 (schema) and the K9 404/409 transport details, which
sit OUTSIDE the LcView shape. A 5xx would invite blind retries that can double-bill SurePass.

Validation (schema = 422, nothing recorded): ``consent.given`` must be the JSON boolean ``true`` (owner ruling), the PAN
and mobile are normalised by ``policy.guards`` (the only normalisers / hashers), the caller identifiers (lc / lead /
opportunity / contact ids) are stripped and must be non-blank, declared money is a finite JSON number (income strictly
positive, EMI non-negative), the quick body has NO ``refresh`` field (``extra='forbid'``), and ``opportunity_stage`` must
be an ``OpportunityStage`` member (stored verbatim; the ordering gate is a STAGE_TOO_EARLY refusal). The 422 handler
returns ``loc`` / ``msg`` / ``type`` only: FastAPI's ``input`` / ``ctx`` echo would reflect the PAN, mobile and name, and
an unknown key's NAME is masked in ``loc`` (a caller-chosen key can be PAN-shaped).

Error boundary (T11 rulings, binding): ``graph.invoke`` runs in the threadpool under ``asyncio.wait_for`` with
``GRAPH_TIMEOUT_SECONDS``. A raise, or a final state without a usable view, goes through ``nodes.fail_check`` (billed
patch while running, guarded error row, holder-only in-flight release) and answers 200 with ``build_error_lc_view``. A
ROUTE TIMEOUT never calls ``fail_check`` and never releases the in-flight lock (the graph thread may still be mid-call;
the lock self-heals at its 180 s TTL, or persist releases it when the thread finishes); it records the error row through
the repo directly, and persist's later terminal update is a guarded no-op.

Tracing (T11 ruling, binding): when tracing is on, the ONE client comes from ``tracing.make_langsmith_client`` (payloads
hidden, every exported string scrubbed), each run attaches ``tracing.make_tracer(client)`` through
``invoke_config(tracer=...)``, and ``finish_trace`` patches the root run through that same client. Tracing counts as on
when langsmith's own env switches would turn it on too, so langchain never falls back to its default tracer (which hides
nothing). Logs carry ``check_id`` / ``pan_hash`` and exception TYPES only, never a raw identifier or exception text.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Final, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langsmith import Client
from langsmith import utils as ls_utils
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from starlette.concurrency import run_in_threadpool

from eligibility_agent.config import ConfigError, Settings, get_settings, utcnow
from eligibility_agent.db.repo import MemoryRepo, Repo, create_supabase_backend
from eligibility_agent.graph import tracing
from eligibility_agent.graph.build import (
    CompiledGraph,
    GraphDeps,
    build_graph,
    build_recompute_graph,
    make_postgres_checkpointer,
    new_check_state,
    seed_recompute,
)
from eligibility_agent.graph.llm import RankContext, RankModel, make_rank_model
from eligibility_agent.graph.nodes import fail_check
from eligibility_agent.graph.prompts import PromptBundle, load_prompts
from eligibility_agent.graph.state import CheckState, ConsentInfo, Kind, LcView, ModelOutput
from eligibility_agent.output.lc_view import build_error_lc_view
from eligibility_agent.policy.guards import (
    OpportunityStage,
    admit_attach,
    compute_billed,
    hash_mobile,
    hash_pan,
    normalize_mobile,
    normalize_pan,
    pan_last4,
)
from eligibility_agent.surepass.client import LiveSurePassClient, SurePassClient

__all__ = [
    "AUTH_ERROR_DETAIL",
    "CHECK_NOT_FOUND",
    "EXTRA_FIELD_LOC",
    "MAX_DECLARED_AMOUNT",
    "ROUTE_TIMEOUT_MESSAGE",
    "AppRuntime",
    "AttachRequest",
    "AttachResponse",
    "ConsentIn",
    "FullCheckRequest",
    "GraphInvoker",
    "QuickCheckRequest",
    "RecomputeRequest",
    "UnconfiguredRankModel",
    "app_factory",
    "create_app",
    "harden_tracing_env",
    "runtime_of",
]

logger = logging.getLogger(__name__)

KINDS: Final[tuple[Kind, ...]] = ("quick", "full")
MAX_DECLARED_AMOUNT: Final[float] = 100_000_000.0  # INR per month; a sanity ceiling, never a policy figure
AUTH_ERROR_DETAIL: Final[str] = "invalid_or_missing_token"
CHECK_NOT_FOUND: Final[str] = "check_not_found"
ROUTE_TIMEOUT_MESSAGE: Final[str] = "route: graph timeout"
EXTRA_FIELD_LOC: Final[str] = "<extra field>"  # replaces an unknown key's name in a 422 ``loc``: never echo caller keys
# K9 transport details (outside the LcView shape).
_RECOMPUTE_REFUSALS: Final[dict[str, tuple[int, str]]] = {
    "not_found": (404, CHECK_NOT_FOUND),
    "not_recomputable": (409, "not_recomputable"),
    "raw_unavailable": (409, "raw_unavailable"),
}
_ATTACH_REFUSALS: Final[dict[str, tuple[int, str]]] = {
    "not_found": (404, CHECK_NOT_FOUND),
    "not_quick": (409, "attach_quick_only"),
    "conflict": (409, "already_attached"),  # attached to a DIFFERENT contact (policy A4)
}
# Defence in depth for any langsmith client this process builds without ours (the default client honours both).
TRACE_HIDE_ENV: Final[dict[str, str]] = {"LANGSMITH_HIDE_INPUTS": "true", "LANGSMITH_HIDE_OUTPUTS": "true"}

GraphInvoker = Callable[[CheckState, RunnableConfig], Mapping[str, Any]]


# --------------------------------------------------------------------------- request / response models


def _strip_non_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be blank")
    return stripped


def _strip_optional(value: str | None) -> str | None:
    return None if value is None else _strip_non_blank(value)


class ConsentIn(BaseModel):
    """PRD 6 consent object. ``given`` must be the JSON boolean ``true``: absent, false, ``"true"``, ``1`` -> 422."""

    model_config = ConfigDict(extra="forbid")

    given: Literal[True]
    wording_version: str = Field(min_length=1, max_length=32)
    call_ref: str | None = Field(default=None, min_length=1, max_length=128)
    ip: str | None = Field(default=None, max_length=64)

    @field_validator("given", mode="before")
    @classmethod
    def _given_is_the_bool_true(cls, value: object) -> object:
        # pydantic's Literal[True] also admits 1 and 1.0 (they compare equal to True).
        if value is not True:
            raise ValueError("consent.given must be the boolean true")
        return value

    @field_validator("wording_version")
    @classmethod
    def _wording_not_blank(cls, value: str) -> str:
        return _strip_non_blank(value)

    def info(self) -> ConsentInfo:
        return ConsentInfo(given=True, wording_version=self.wording_version, call_ref=self.call_ref, ip=self.ip)


# Money: strict (a JSON number; never a bool or a string), finite, bounded. Income > 0, EMI >= 0.
_Income = Annotated[float, Field(strict=True, gt=0, le=MAX_DECLARED_AMOUNT, allow_inf_nan=False)]
_Emi = Annotated[float, Field(strict=True, ge=0, le=MAX_DECLARED_AMOUNT, allow_inf_nan=False)]


class _CheckBody(BaseModel):
    """Fields shared by both check bodies. PAN / mobile run through the policy normalisers (the single home): a value
    they refuse is a 422 whose message never echoes the input."""

    model_config = ConfigDict(extra="forbid")

    lc_id: str = Field(min_length=1, max_length=64)
    lead_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)  # lives ONLY in the consent record
    mobile: str = Field(max_length=32)
    pan: str = Field(max_length=32)
    consent: ConsentIn
    existing_emi: _Emi | None = None

    @field_validator("pan")
    @classmethod
    def _pan(cls, value: str) -> str:
        return normalize_pan(value)

    @field_validator("mobile")
    @classmethod
    def _mobile(cls, value: str) -> str:
        return normalize_mobile(value)

    @field_validator("name", "lc_id", "lead_id")
    @classmethod
    def _stripped(cls, value: str) -> str:
        # Identifiers key the quota bucket, the PAN-correction rule and attach: " lc-1" must be "lc-1", "   " nothing.
        return _strip_non_blank(value)


class QuickCheckRequest(_CheckBody):
    """PRD 6 quick body. NO ``refresh`` field: an expired stored score re-pulls automatically (billed, not counted)."""

    opportunity_id: str | None = Field(default=None, min_length=1, max_length=64)
    monthly_income: _Income | None = None

    @field_validator("opportunity_id")
    @classmethod
    def _opportunity(cls, value: str | None) -> str | None:
        return _strip_optional(value)


class FullCheckRequest(_CheckBody):
    """PRD 6 full body: opportunity, stage and income required."""

    opportunity_id: str = Field(min_length=1, max_length=64)
    opportunity_stage: OpportunityStage  # enum membership = 422; the ordering gate is a STAGE_TOO_EARLY refusal
    monthly_income: _Income
    acknowledge_ntc_warning: StrictBool = False
    refresh: StrictBool = False

    @field_validator("opportunity_id")
    @classmethod
    def _opportunity(cls, value: str) -> str:
        return _strip_non_blank(value)


class RecomputeRequest(BaseModel):
    """The figures REPLACE the declared ones outright (absent or null clears). A full check needs an income."""

    model_config = ConfigDict(extra="forbid")

    monthly_income: _Income | None = None
    existing_emi: _Emi | None = None


class AttachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_id: str = Field(min_length=1, max_length=64)

    @field_validator("contact_id")
    @classmethod
    def _contact(cls, value: str) -> str:
        return _strip_non_blank(value)


class AttachResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    contact_id: str
    attached: bool


# --------------------------------------------------------------------------- runtime


class UnconfiguredRankModel:
    """The rank seam when OPENAI_API_KEY is unset (degraded boot). Raising is safe: the rank node maps two failed
    attempts to OPENAI_FAILURE, and on an empty degraded boot admission refuses before any model call anyway."""

    def __call__(self, ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput:
        raise RuntimeError("OPENAI_API_KEY is not set: the ranking model is unavailable")


@dataclass
class AppRuntime:
    """Everything the routes need, held on ``app.state.runtime`` (built once per app)."""

    settings: Settings
    deps: GraphDeps
    trace_client: Client | None
    graphs: dict[Kind, GraphInvoker] = field(default_factory=lambda: dict[Kind, GraphInvoker]())
    recompute_graphs: dict[Kind, GraphInvoker] = field(default_factory=lambda: dict[Kind, GraphInvoker]())


def runtime_of(app: FastAPI) -> AppRuntime:
    runtime: object = getattr(app.state, "runtime", None)
    if not isinstance(runtime, AppRuntime):
        raise RuntimeError("app.state.runtime is missing: build the app with create_app()")
    return runtime


def _invoker(graph: CompiledGraph) -> GraphInvoker:
    def invoke(state: CheckState, config: RunnableConfig) -> Mapping[str, Any]:
        final: object = graph.invoke(state, config)  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(final, Mapping):
            raise TypeError(f"graph returned {type(final).__name__}, expected a state mapping")
        return cast("Mapping[str, Any]", final)

    return invoke


def _compile(runtime: AppRuntime, checkpointer: BaseCheckpointSaver[Any] | None) -> None:
    deps = runtime.deps
    runtime.graphs = {kind: _invoker(build_graph(kind, deps, checkpointer=checkpointer)) for kind in KINDS}
    runtime.recompute_graphs = {
        kind: _invoker(build_recompute_graph(kind, deps, checkpointer=checkpointer)) for kind in KINDS
    }


def harden_tracing_env(environ: MutableMapping[str, str]) -> None:
    """Force payload hiding for any langsmith client built from the environment (ours hides explicitly anyway)."""
    environ.update(TRACE_HIDE_ENV)


def _forget_langsmith_env_reads() -> None:
    """langsmith memoises its env reads (``utils.get_env_var`` is lru_cached), so a read made before ``create_app`` would
    keep the old HIDE / TRACING values for the life of the process. Clearing makes the hardened env take effect."""
    getattr(ls_utils, "get_env_var").cache_clear()


def _tracing_wanted(settings: Settings) -> bool:
    """Our flag, OR langsmith's own env switches (LANGCHAIN_TRACING_V2 & co.): whenever langchain would trace, it traces
    through our payload-free tracer instead of adding its default one."""
    env_on: object = ls_utils.tracing_is_enabled()  # pyright: ignore[reportUnknownMemberType]
    return settings.langsmith_tracing or bool(env_on)


def _make_repo(settings: Settings) -> Repo:
    if settings.repo_backend == "supabase":
        url, key = settings.supabase_url, settings.supabase_service_role_key
        if not url or not key:  # unreachable after startup_errors(); keeps the types honest
            raise ConfigError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when REPO_BACKEND=supabase")
        repo, _ = create_supabase_backend(url, key, bucket=settings.raw_bucket)
        return repo
    return MemoryRepo()


def _log_level(settings: Settings) -> int:
    """LOG_LEVEL by name; an unknown name falls back to INFO (it must never stop the boot)."""
    return logging.getLevelNamesMapping().get(settings.log_level.strip().upper(), logging.INFO)


def _make_rank_model(settings: Settings, prompts: PromptBundle) -> RankModel:
    """The real model ONLY when a key is configured: ``make_rank_model`` raises at construction without one, and
    ChatOpenAI reads os.environ only, so the Settings key (possibly from .env) is passed explicitly."""
    if not settings.openai_api_key:
        return UnconfiguredRankModel()
    return make_rank_model(model=settings.openai_model, prompts=prompts, api_key=settings.openai_api_key)


# --------------------------------------------------------------------------- app factory


def create_app(
    settings: Settings | None = None,
    *,
    repo: Repo | None = None,
    surepass: SurePassClient | None = None,
    rank_model: RankModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    prompts: PromptBundle | None = None,
    now: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Compose the service. Every collaborator is injectable (tests pass MemoryRepo / FakeSurePassClient /
    FakeRankModel); anything not injected is built from ``settings`` with an in-process fallback, so an EMPTY environment
    boots (memory repo, stub SurePass that refuses every live call, unconfigured model, auth off with a warning)."""
    settings = settings if settings is not None else get_settings()
    errors = settings.startup_errors()
    if not settings.identity_hmac_key:  # only reachable through an init kwarg (blank env values are ignored)
        errors.append("IDENTITY_HMAC_KEY must be non-empty")
    if errors:
        raise ConfigError("; ".join(errors))
    logging.getLogger("eligibility_agent").setLevel(_log_level(settings))
    if not settings.auth_enabled:
        logger.warning("SERVICE_API_TOKEN unset: API auth DISABLED (dev mode only)")
    harden_tracing_env(os.environ)
    _forget_langsmith_env_reads()

    repo_ = repo if repo is not None else _make_repo(settings)
    owned_surepass: LiveSurePassClient | None = None
    if surepass is None:
        owned_surepass = LiveSurePassClient(
            token=settings.surepass_token or "",
            live=settings.surepass_live,
            sink=repo_,
            base_url=settings.surepass_base_url,
        )
    prompts_ = prompts if prompts is not None else load_prompts(settings)
    deps = GraphDeps(
        repo=repo_,
        surepass=surepass if surepass is not None else cast(SurePassClient, owned_surepass),
        rank_model=rank_model if rank_model is not None else _make_rank_model(settings, prompts_),
        prompts=prompts_,
        now=now if now is not None else utcnow,
    )
    trace_client = tracing.make_langsmith_client(api_key=settings.langsmith_api_key) if _tracing_wanted(settings) else None
    runtime = AppRuntime(settings=settings, deps=deps, trace_client=trace_client)
    _compile(runtime, checkpointer)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        pool: Any = None
        try:
            if checkpointer is None and settings.database_url:
                # K21 checkpointer (manual pool, session pooler, dedicated schema, strict msgpack); recompile onto it.
                saver, pool = await run_in_threadpool(make_postgres_checkpointer, settings.database_url)
                _compile(runtime, cast("BaseCheckpointSaver[Any]", saver))
            yield
        finally:
            if pool is not None:
                pool.close()
            if owned_surepass is not None:
                owned_surepass.close()

    # PRD 6: bearer auth on every route except /health, so FastAPI's unauthenticated docs/schema routes are not mounted.
    app = FastAPI(title="eligibility-agent", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runtime = runtime
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.include_router(health_router)
    app.include_router(checks_router)
    return app


def app_factory() -> FastAPI:
    """``uvicorn 'eligibility_agent.main:app_factory' --factory``."""
    settings = get_settings()
    logging.basicConfig(level=_log_level(settings))
    return create_app(settings)


# --------------------------------------------------------------------------- 422 + auth


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """``loc`` / ``msg`` / ``type`` only: FastAPI's default body echoes ``input`` (and ``ctx``), i.e. the PAN."""
    errors: list[Any] = list(exc.errors()) if isinstance(exc, RequestValidationError) else []
    detail = [
        {"loc": _safe_loc(error), "msg": str(error.get("msg", "")), "type": str(error.get("type", ""))}
        for error in cast("list[Mapping[str, Any]]", errors)
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def _safe_loc(error: Mapping[str, Any]) -> list[Any]:
    """The error's ``loc``, with an unknown key's NAME (caller-chosen, possibly PAN-shaped) masked."""
    loc = list(cast("tuple[Any, ...]", error.get("loc", ())))
    if error.get("type") == "extra_forbidden" and loc:
        loc[-1] = EXTRA_FIELD_LOC
    return loc


_bearer = HTTPBearer(auto_error=False)


async def require_bearer(
    request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> None:
    """K21: ``Authorization: Bearer <SERVICE_API_TOKEN>``, compared in constant time. Open only when no token is
    configured (degraded / dev boot; the supabase backend refuses to boot without one)."""
    expected = runtime_of(request.app).settings.service_api_token
    if not expected:
        return
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail=AUTH_ERROR_DETAIL, headers={"WWW-Authenticate": "Bearer"})


# --------------------------------------------------------------------------- the error boundary


@dataclass(frozen=True, slots=True)
class _RunContext:
    """What the boundary needs to answer for a check that never produced a usable view."""

    kind: Kind
    check_id: str
    lc_id: str
    pan_hash: str
    pan_last4: str
    run_id: UUID | None


@dataclass(frozen=True, slots=True)
class _Outcome:
    final: Mapping[str, Any] | None = None
    error: Exception | None = None


class _GraphRun:
    """One graph invocation shared between the route and its threadpool worker.

    ``call`` (worker) boxes a graph exception, so a TimeoutError raised INSIDE the graph is never mistaken for the route
    budget. ``abandon`` (route, on its timeout) marks the run as orphaned. An exception that ends an abandoned run reaches
    no one, so it is logged here, exactly once whichever side finishes first, as ``check_id`` + exception TYPE only (the
    message could carry PII)."""

    def __init__(self, *, check_id: str) -> None:
        self._check_id = check_id
        self._lock = threading.Lock()
        self._abandoned = False
        self._outcome: _Outcome | None = None

    def call(self, graph: GraphInvoker, state: CheckState, config: RunnableConfig) -> _Outcome:
        try:
            outcome = _Outcome(final=graph(state, config))
        except Exception as exc:
            outcome = _Outcome(error=exc)
        with self._lock:
            self._outcome = outcome
            abandoned = self._abandoned
        if abandoned:
            self._log_orphaned(outcome)
        return outcome

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            outcome = self._outcome  # set only when the worker already finished (wait_for raced a ready result)
        if outcome is not None:
            self._log_orphaned(outcome)

    def _log_orphaned(self, outcome: _Outcome) -> None:
        if outcome.error is not None:
            logger.error(
                "graph raised after its route timed out check_id=%s (%s)", self._check_id, type(outcome.error).__name__
            )


def _view_of(final: Mapping[str, Any] | None, check_id: str) -> LcView:
    raw: object = None if final is None else final.get("lc_view")
    view = raw if isinstance(raw, LcView) else LcView.model_validate(raw)
    if view.check_id != check_id:
        raise ValueError("the graph returned a view for another check")
    return view


def _fail(deps: GraphDeps, ctx: _RunContext, exc: Exception) -> LcView:
    return fail_check(
        deps,
        check_id=ctx.check_id,
        kind=ctx.kind,
        lc_id=ctx.lc_id,
        pan_hash=ctx.pan_hash,
        pan_last4=ctx.pan_last4,
        message=f"route: {type(exc).__name__}",
    )


def _record_route_timeout(repo: Repo, ctx: _RunContext) -> None:
    """The timeout error row, straight through the repo: billed patched from the call log while the row is still running
    (K8), then the guarded ``record_check_error``. NO lock release: the graph thread may still hold a live call."""
    try:
        repo.update_check(ctx.check_id, {"billed": compute_billed(repo.surepass_calls_for_check(ctx.check_id))})
    except Exception as exc:
        logger.error("billing patch on route timeout failed for check_id=%s (%s)", ctx.check_id, type(exc).__name__)
    try:
        repo.record_check_error(
            check_id=ctx.check_id,
            kind=ctx.kind,
            lc_id=ctx.lc_id,
            pan_hash=ctx.pan_hash,
            pan_last4=ctx.pan_last4,
            message=ROUTE_TIMEOUT_MESSAGE,
        )
    except Exception as exc:
        logger.error("record_check_error on route timeout failed for check_id=%s (%s)", ctx.check_id, type(exc).__name__)


async def _run_graph(
    runtime: AppRuntime, graph: GraphInvoker, state: CheckState, config: RunnableConfig, ctx: _RunContext
) -> LcView:
    run = _GraphRun(check_id=ctx.check_id)
    try:
        outcome = await asyncio.wait_for(
            run_in_threadpool(run.call, graph, state, config), timeout=runtime.settings.graph_timeout_seconds
        )
    except TimeoutError:
        run.abandon()
        logger.error("check timed out check_id=%s kind=%s pan_hash=%s", ctx.check_id, ctx.kind, ctx.pan_hash)
        await run_in_threadpool(_record_route_timeout, runtime.deps.repo, ctx)
        return build_error_lc_view(check_id=ctx.check_id, kind=ctx.kind)

    error = outcome.error
    view: LcView | None = None
    if error is None:
        try:
            view = _view_of(outcome.final, ctx.check_id)
        except Exception as exc:
            error = exc
    if view is None or error is not None:
        failure = error if error is not None else RuntimeError("no view")
        logger.error(
            "check failed check_id=%s kind=%s pan_hash=%s (%s)",
            ctx.check_id,
            ctx.kind,
            ctx.pan_hash,
            type(failure).__name__,
        )
        return await run_in_threadpool(_fail, runtime.deps, ctx, failure)

    await run_in_threadpool(
        tracing.finish_trace,
        runtime.trace_client,
        run_id=ctx.run_id,
        kind=ctx.kind,
        lc_id=ctx.lc_id,
        check_id=ctx.check_id,
        pan_hash=ctx.pan_hash,
        final_state=outcome.final or {},
    )
    return view


def _config(runtime: AppRuntime, ctx: _RunContext) -> RunnableConfig:
    client = runtime.trace_client
    tracer = (
        tracing.make_tracer(client, project_name=runtime.settings.langsmith_project) if client is not None else None
    )
    return tracing.invoke_config(
        kind=ctx.kind, check_id=ctx.check_id, lc_id=ctx.lc_id, pan_hash=ctx.pan_hash, run_id=ctx.run_id, tracer=tracer
    )


def _canonical_check_id(raw: str) -> str | None:
    """Ids are app-side uuid4s; anything else is unknown (404), never handed to a uuid-typed column."""
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


# --------------------------------------------------------------------------- routes

health_router = APIRouter()
checks_router = APIRouter(prefix="/checks", dependencies=[Depends(require_bearer)])


@health_router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Liveness only: EXACTLY ``{"ok": true, "surepass_live": <bool>}``, no auth, no dependency probes."""
    return JSONResponse({"ok": True, "surepass_live": runtime_of(request.app).settings.surepass_live})


async def _run_check(runtime: AppRuntime, kind: Kind, body: QuickCheckRequest | FullCheckRequest) -> LcView:
    settings = runtime.settings
    key = settings.identity_hmac_key
    check_id = str(uuid.uuid4())
    pan_hash = hash_pan(body.pan, key=key)
    run_id = tracing.new_run_id(runtime.trace_client is not None)
    full = body if isinstance(body, FullCheckRequest) else None
    state = new_check_state(
        kind=kind,
        check_id=check_id,
        lc_id=body.lc_id,
        lead_id=body.lead_id,
        opportunity_id=body.opportunity_id,
        opportunity_stage=full.opportunity_stage.value if full is not None else None,
        pan=body.pan,
        name=body.name,
        mobile=body.mobile,
        pan_hash=pan_hash,
        mobile_hash=hash_mobile(body.mobile, key=key),
        monthly_income=body.monthly_income,
        existing_emi_declared=body.existing_emi,
        refresh=full.refresh if full is not None else False,
        acknowledge_ntc_warning=full.acknowledge_ntc_warning if full is not None else False,
        consent=body.consent.info(),
        langsmith_run_id=str(run_id) if run_id is not None else None,
    )
    ctx = _RunContext(
        kind=kind, check_id=check_id, lc_id=body.lc_id, pan_hash=pan_hash, pan_last4=pan_last4(body.pan), run_id=run_id
    )
    return await _run_graph(runtime, runtime.graphs[kind], state, _config(runtime, ctx), ctx)


@checks_router.post("/quick", response_model=LcView)
async def post_quick(body: QuickCheckRequest, request: Request) -> LcView:
    return await _run_check(runtime_of(request.app), "quick", body)


@checks_router.post("/full", response_model=LcView)
async def post_full(body: FullCheckRequest, request: Request) -> LcView:
    return await _run_check(runtime_of(request.app), "full", body)


@checks_router.post("/{check_id}/recompute", response_model=LcView)
async def post_recompute(check_id: str, body: RecomputeRequest, request: Request) -> LcView:
    """K9: a NEW checks row re-derived from the stored raw on the new declared figures; zero SurePass calls, no quota."""
    runtime = runtime_of(request.app)
    original_id = _canonical_check_id(check_id)
    original = None if original_id is None else await run_in_threadpool(runtime.deps.repo.get_check, original_id)
    if original is None or original_id is None:
        raise HTTPException(status_code=404, detail=CHECK_NOT_FOUND)
    if original.kind == "full" and body.monthly_income is None:
        raise RequestValidationError(
            [{"loc": ("body", "monthly_income"), "msg": "Field required for a full check", "type": "missing"}]
        )
    run_id = tracing.new_run_id(runtime.trace_client is not None)
    ctx = _RunContext(
        kind=original.kind,
        check_id=str(uuid.uuid4()),
        lc_id=original.lc_id,
        pan_hash=original.pan_hash,
        pan_last4=original.pan_last4,
        run_id=run_id,
    )
    try:
        outcome, state = await run_in_threadpool(
            seed_recompute,
            runtime.deps,
            original_check_id=original_id,
            new_check_id=ctx.check_id,
            monthly_income=body.monthly_income,
            existing_emi_declared=body.existing_emi,
            langsmith_run_id=str(run_id) if run_id is not None else None,
        )
    except Exception as exc:
        logger.error("recompute seeding failed check_id=%s (%s)", ctx.check_id, type(exc).__name__)
        return await run_in_threadpool(_fail, runtime.deps, ctx, exc)
    if state is None:
        status, detail = _RECOMPUTE_REFUSALS.get(outcome, (409, "not_recomputable"))
        raise HTTPException(status_code=status, detail=detail)
    return await _run_graph(runtime, runtime.recompute_graphs[original.kind], state, _config(runtime, ctx), ctx)


@checks_router.post("/{check_id}/attach", response_model=AttachResponse)
async def post_attach(check_id: str, body: AttachRequest, request: Request) -> AttachResponse:
    """K9 / policy A-table: quick checks only; idempotent for the same contact; updates checks.contact_id AND the
    check's consents.contact_id (the contact_id-only patch is exempt from the terminal guard)."""
    repo = runtime_of(request.app).deps.repo
    canonical = _canonical_check_id(check_id)
    row = None if canonical is None else await run_in_threadpool(repo.get_check, canonical)
    outcome = admit_attach(
        row.kind if row is not None else None, row.contact_id if row is not None else None, body.contact_id
    )
    if outcome in _ATTACH_REFUSALS or row is None:
        status, detail = _ATTACH_REFUSALS.get(outcome, (404, CHECK_NOT_FOUND))
        raise HTTPException(status_code=status, detail=detail)
    if outcome == "ok":
        await run_in_threadpool(repo.attach_contact, row.id, body.contact_id)
    return AttachResponse(check_id=row.id, contact_id=body.contact_id, attached=True)
