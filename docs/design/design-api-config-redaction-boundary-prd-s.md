# API, config, redaction boundary (PRD section 9; main.py, config.py, output/lc_view.py)

## Summary
main.py is the composition root: an app factory create_app(settings, repo, surepass, rank_llm, finalize_llm, checkpointer) wires injected dependencies into the two compiled graphs inside a FastAPI lifespan, so tests build a fully-faked app with zero env. Auth is a router-level FastAPI dependency (HTTPBearer, constant-time compare) on /checks only; /health lives on an unauthenticated router and returns exactly {"ok": bool, "surepass_live": bool}. Pydantic v2 request models enforce PAN ^[A-Z]{5}[0-9]{4}[A-Z]$ and mobile ^[6-9][0-9]{9}$ after light normalization, consent as StrictBool (absent/non-bool -> 422; false -> graph -> CONSENT_MISSING refusal row), and the full-check extras (required opportunity_id, stage enum marked ASSUMPTION, monthly_income > 0); every schema failure is a 422 with input echo stripped, every policy failure is an HTTP-200 LcView with status="refused" persisted as a checks row. The route builds CheckState (uuid4 check_id, salted-SHA256 pan_hash via the single canonical hash_pan in config.py), invokes the pre-compiled quick/full graph via ainvoke with RunnableConfig {run_id, thread_id=check_id, tags [tier:*, lc:*], metadata {check_id, pan_hash, langsmith_run_id}} under an asyncio timeout, and returns final_state["lc_view"]; any unexpected node exception yields a deterministic review-outcome LcView (HTTP 200) plus a best-effort checks.status='error' upsert, because blind retries on a live call are the expensive failure mode. config.py (pydantic-settings) declares every env var optional-with-default so Settings() constructs from an empty environment: the SUREPASS_LIVE kill switch is honored only on the exact lowercase literal "true", repo defaults to memory, checkpointer to MemorySaver, LLMs to raising stubs that are unreachable because admission refuses LIVE_CALLS_DISABLED first — that chain is the PRD-13 degraded-boot guarantee. Hard config validation fires only when an external backend is explicitly selected (supabase backend requires URL+key+real salt+API token; live SurePass requires a token). output/lc_view.py is the redaction boundary: pure builders for ok / pan_invalid / refused / error views, lender id->name resolution from state.gated_lenders with a leak-checked name fallback, loan range rounded to whole-thousand ints, pan_aadhaar_link_issue typed Literal[True]|None and set only when identity.aadhaar_linked is False, and the two compiled leak regexes — bureau terms \b(?:cibil|experian|equifax|crif|bureaus?)\b (IGNORECASE) and standalone 300-899 (?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d) — verified against 736 / Rs.736 / (736) / 650 (caught) and 3,500 / 45,000 / 3,50,000 / 12,34,567 / 0.736 / 736.50 / 36 months / 14 percent (passed), with whole-line drop and a fixed fallback line when scrubbing empties talking_points. All six refusal codes get exact persisted-row contents and exact one-line LC strings, each verified clean against the scrubber. The evaluators section should import LEAK_PATTERNS from this module so no_leak and the runtime scrubber can never drift.

## Files
### eligibility_agent/config.py
_Env loading, kill switch, canonical PAN hasher, degraded-boot guarantee. Single source of truth for every env var._

ENV VAR TABLE (all read via pydantic-settings; every field optional-with-default so `Settings()` constructs with NOTHING set):
| env var | field | type | default | semantics |
|---|---|---|---|---|
| SUREPASS_LIVE | surepass_live_flag | str | "" | live iff value == "true" EXACT lowercase literal (FIXED kill switch; no bool coercion) |
| SUREPASS_TOKEN | surepass_token | str\|None | None | bearer token; required only when live |
| SUREPASS_BASE_URL | surepass_base_url | str | "https://kyc-api.surepass.app" | no sandbox exists; override for fakes only |
| SERVICE_API_TOKEN | service_api_token | str\|None | None | API bearer token; None => auth disabled with startup WARNING (degraded boot) |
| PAN_HASH_SALT | pan_hash_salt | str | DEFAULT_PAN_HASH_SALT ("dev-only-salt-do-not-use-in-prod") | salt for hash_pan; default forbidden with supabase backend |
| REPO_BACKEND | repo_backend | Literal["memory","supabase"] | "memory" | memory default is what makes empty-env boot work |
| SUPABASE_URL | supabase_url | str\|None | None | |
| SUPABASE_SERVICE_ROLE_KEY | supabase_service_role_key | str\|None | None | |
| DATABASE_URL | database_url | str\|None | None | LangGraph Postgres checkpointer; None => MemorySaver |
| OPENAI_API_KEY | openai_api_key | str\|None | None | None => raising stub LLMs (unreachable in degraded flow) |
| OPENAI_MODEL | openai_model | str | "gpt-4.1" | |
| OPENAI_TEMPERATURE | openai_temperature | float | 0.0 | |
| LANGSMITH_TRACING | langsmith_tracing_flag | str | "" | on iff == "true" (same literal convention; tests silent by default) |
| LANGSMITH_API_KEY | langsmith_api_key | str\|None | None | |
| LANGSMITH_PROJECT | langsmith_project | str | "eligibility-agent" | |
| PROMPT_SOURCE | prompt_source | Literal["local","hub"] | "local" | "local" = repo-pinned prompt copies; tests never pull Hub |
| PROMPT_RANK_REF | prompt_rank_ref | str | "hero/eligibility-rank" | Hub ref |
| PROMPT_FINALIZE_REF | prompt_finalize_ref | str | "hero/eligibility-finalize" | Hub ref |
| PROMPT_RANK_TAG | prompt_rank_tag | str\|None | None | release tag pin |
| PROMPT_FINALIZE_TAG | prompt_finalize_tag | str\|None | None | release tag pin |
| GRAPH_TIMEOUT_SECONDS | graph_timeout_seconds | float | 90.0 | route-level asyncio.wait_for budget |
| LOG_LEVEL | log_level | str | "INFO" | |

FULL CODE:
```python
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from functools import lru_cache
from typing import Final, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised by create_app when an explicitly selected backend lacks its config."""


DEFAULT_PAN_HASH_SALT: Final[str] = "dev-only-salt-do-not-use-in-prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False,
    )

    surepass_live_flag: str = Field(default="", validation_alias="SUREPASS_LIVE")
    surepass_token: str | None = Field(default=None, validation_alias="SUREPASS_TOKEN")
    surepass_base_url: str = Field(default="https://kyc-api.surepass.app", validation_alias="SUREPASS_BASE_URL")
    service_api_token: str | None = Field(default=None, validation_alias="SERVICE_API_TOKEN")
    pan_hash_salt: str = Field(default=DEFAULT_PAN_HASH_SALT, validation_alias="PAN_HASH_SALT")
    repo_backend: Literal["memory", "supabase"] = Field(default="memory", validation_alias="REPO_BACKEND")
    supabase_url: str | None = Field(default=None, validation_alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(default=None, validation_alias="SUPABASE_SERVICE_ROLE_KEY")
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1", validation_alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, validation_alias="OPENAI_TEMPERATURE")
    langsmith_tracing_flag: str = Field(default="", validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="eligibility-agent", validation_alias="LANGSMITH_PROJECT")
    prompt_source: Literal["local", "hub"] = Field(default="local", validation_alias="PROMPT_SOURCE")
    prompt_rank_ref: str = Field(default="hero/eligibility-rank", validation_alias="PROMPT_RANK_REF")
    prompt_finalize_ref: str = Field(default="hero/eligibility-finalize", validation_alias="PROMPT_FINALIZE_REF")
    prompt_rank_tag: str | None = Field(default=None, validation_alias="PROMPT_RANK_TAG")
    prompt_finalize_tag: str | None = Field(default=None, validation_alias="PROMPT_FINALIZE_TAG")
    graph_timeout_seconds: float = Field(default=90.0, validation_alias="GRAPH_TIMEOUT_SECONDS")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def surepass_live(self) -> bool:
        return self.surepass_live_flag == "true"  # FIXED: exact, case-sensitive literal

    @property
    def langsmith_tracing(self) -> bool:
        return self.langsmith_tracing_flag == "true"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.service_api_token)

    def startup_errors(self) -> list[str]:
        """Hard errors ONLY when an external backend was explicitly selected.
        Empty env -> [] -> degraded boot always succeeds (PRD 13)."""
        errs: list[str] = []
        if self.repo_backend == "supabase":
            if not self.supabase_url or not self.supabase_service_role_key:
                errs.append("REPO_BACKEND=supabase requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
            if self.pan_hash_salt == DEFAULT_PAN_HASH_SALT:
                errs.append("REPO_BACKEND=supabase requires a non-default PAN_HASH_SALT")
            if not self.service_api_token:
                errs.append("REPO_BACKEND=supabase requires SERVICE_API_TOKEN")
        if self.surepass_live and not self.surepass_token:
            errs.append("SUREPASS_LIVE=true requires SUREPASS_TOKEN")
        return errs


def hash_pan(pan: str, *, salt: str) -> str:
    """THE canonical salted PAN hash (cache keys, consents, checks, LangSmith metadata).
    Every other section must import this function, never re-implement.
    hex sha256 of '<salt>:<PAN uppercased+stripped>' -> 64 lowercase hex chars."""
    canonical = pan.strip().upper()
    return hashlib.sha256(f"{salt}:{canonical}".encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(UTC)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```
Degraded-boot chain (PRD 13, spelled out): empty env => Settings() valid; surepass_live False; auth disabled (warned); repo=MemoryRepo; checkpointer=MemorySaver; LLMs=raising stubs. POST /checks/quick then flows admission -> cache(empty) miss -> bureau call needed -> surepass_live False -> refusal LIVE_CALLS_DISABLED -> persist_refusal(MemoryRepo) -> HTTP 200 LcView(status=refused). LLM/SurePass/Supabase never touched. /health returns {"ok": true, "surepass_live": false}.
TEST NOTE: tests construct hermetic settings as Settings(_env_file=None, SUREPASS_LIVE="true", ...) — alias names are the constructor kwargs; _env_file=None ignores any local .env. Policy constants QUICK_DAILY_CAP=5 and CACHE_TTL_DAYS=30 are deliberately NOT env vars (FIXED values; they live as Final constants in policy/guards.py).

### eligibility_agent/main.py
_FastAPI assembly: auth dependency, request models with exact validation, route flow request->CheckState->graph->LcView, error boundary, lifespan composition of graphs/checkpointer._

```python
from __future__ import annotations

import asyncio, hmac, logging, os, uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, AsyncIterator, Literal, NoReturn, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from eligibility_agent.config import ConfigError, Settings, get_settings, hash_pan
from eligibility_agent.db.repo import MemoryRepo, Repo, SupabaseRepo               # db section
from eligibility_agent.graph.build import FinalizeLLM, RankLLM, build_graph, make_openai_llms  # graph section
from eligibility_agent.graph.state import CheckState, LcView                        # graph section
from eligibility_agent.output.lc_view import build_error_lc_view
from eligibility_agent.policy.guards import OpportunityStage                        # policy section (ASSUMPTION enum)
from eligibility_agent.tools.surepass import HttpSurePassClient, SurePassClient     # tools section

logger = logging.getLogger("eligibility_agent")

PAN_PATTERN: str = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
MOBILE_PATTERN: str = r"^[6-9][0-9]{9}$"
```

REQUEST MODELS (Pydantic v2; `extra="forbid"`; normalizers are mode="before" so Field pattern constraints run on the normalized value):
```python
class QuickCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lc_id: str = Field(min_length=1, max_length=64)
    lead_id: str = Field(min_length=1, max_length=64)
    opportunity_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    mobile: str = Field(pattern=MOBILE_PATTERN)
    pan: str = Field(pattern=PAN_PATTERN)
    consent: StrictBool                    # JSON boolean only; "true"/1 -> 422; false -> CONSENT_MISSING refusal downstream
    monthly_income: float | None = Field(default=None, gt=0, le=100_000_000)
    existing_emi: float | None = Field(default=None, ge=0, le=100_000_000)
    refresh: bool = False

    @field_validator("pan", mode="before")
    @classmethod
    def _norm_pan(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("mobile", mode="before")
    @classmethod
    def _norm_mobile(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        s = v.strip().replace(" ", "").replace("-", "")
        return s.removeprefix("+91")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("name must not be blank")
        return s

class FullCheckRequest(QuickCheckRequest):
    opportunity_id: str = Field(min_length=1, max_length=64)        # override -> required
    opportunity_stage: OpportunityStage                              # ASSUMPTION enum; bad value -> 422
    monthly_income: float = Field(gt=0, le=100_000_000)              # override -> required
```

422 vs REFUSAL-ROW TABLE (state clearly; refusals are HTTP 200 LcView persisted as checks rows, schema failures are 422 and never become a checks row because no check_id/state ever existed):
| condition | HTTP | persisted? |
|---|---|---|
| PAN fails ^[A-Z]{5}[0-9]{4}[A-Z]$ after normalization | 422 | no |
| mobile fails ^[6-9][0-9]{9}$ after normalization | 422 | no |
| name blank; lc_id/lead_id missing; unknown extra key | 422 | no |
| consent ABSENT or not a JSON boolean ("true", 1, null) | 422 | no |
| consent == false (valid bool) | 200 | YES: checks row status=refused, refusal_code=CONSENT_MISSING |
| monthly_income <= 0 or non-numeric; existing_emi < 0 | 422 | no |
| full: opportunity_id/opportunity_stage/monthly_income missing | 422 | no |
| full: stage value not in the 8-member enum | 422 | no |
| full: stage valid but earlier than offer | 200 | YES: STAGE_TOO_EARLY refusal row |
| quick: 6th billed check today | 200 | YES: DAILY_CAP_REACHED |
| refresh=true before TTL | 200 | YES: CACHE_FRESH_NO_REFRESH |
| full: fresh full result already on this opportunity | 200 | YES: OPPORTUNITY_ALREADY_CHECKED |
| bureau call needed and SUREPASS_LIVE != "true" | 200 | YES: LIVE_CALLS_DISABLED |
| missing/wrong bearer token (non-/health) | 401 (WWW-Authenticate: Bearer, detail="invalid_or_missing_token") | no |
| PAN Comprehensive says invalid/non-person | 200 | YES: checks row status=pan_invalid (not counted) |
| unexpected node exception / timeout | 200 review-outcome LcView | best-effort checks row status='error' |

AUTH (dependency, not middleware — justification in decisions):
```python
_bearer = HTTPBearer(auto_error=False)

async def require_bearer(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    settings: Settings = request.app.state.settings
    expected = settings.service_api_token
    if expected is None:
        return  # degraded/dev boot: warned once at startup; kill switch + memory repo bound the blast radius
    if creds is None or not hmac.compare_digest(creds.credentials.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="invalid_or_missing_token",
                            headers={"WWW-Authenticate": "Bearer"})
```

422 HANDLER (strip FastAPI's "input"/"ctx" echo so the PAN/mobile/name are never reflected into responses or client logs):
```python
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})
```

APP FACTORY (composition root; everything injectable for tests):
```python
def create_app(
    settings: Settings | None = None,
    *,
    repo: Repo | None = None,
    surepass: SurePassClient | None = None,
    rank_llm: RankLLM | None = None,
    finalize_llm: FinalizeLLM | None = None,
    checkpointer: object | None = None,   # BaseCheckpointSaver; typed via graph section's export
) -> FastAPI:
```
Body, in order:
1. `settings = settings or get_settings()`; `errs = settings.startup_errors()`; if errs: `raise ConfigError("; ".join(errs))` (empty env => no errors => degraded boot proceeds).
2. `logging.basicConfig(level=settings.log_level)`; if not settings.auth_enabled: `logger.warning("SERVICE_API_TOKEN unset: API auth DISABLED (dev mode)")`.
3. `_apply_langsmith_env(settings)`: ONLY when `settings.langsmith_tracing and settings.langsmith_api_key`, `os.environ.setdefault` each of LANGSMITH_TRACING="true", LANGCHAIN_TRACING_V2="true", LANGSMITH_API_KEY, LANGCHAIN_API_KEY, LANGSMITH_PROJECT, LANGCHAIN_PROJECT. Tests never set the flag => tracing is a no-op, no env mutation.
4. Resolve repo: injected, else `SupabaseRepo(settings.supabase_url, settings.supabase_service_role_key)` when backend=="supabase" (values proven non-None by startup_errors) else `MemoryRepo()`.
5. Resolve surepass: injected, else `HttpSurePassClient(token=settings.surepass_token or "", base_url=settings.surepass_base_url, live=settings.surepass_live)` — the client itself raises LiveCallsDisabled when live is False (kill switch enforced in tools layer, double-checked by admission).
6. Resolve LLM callables: injected, else `make_openai_llms(model=settings.openai_model, temperature=settings.openai_temperature, api_key=key, prompt_source=settings.prompt_source, rank_ref=..., finalize_ref=..., rank_tag=..., finalize_tag=...)` when `settings.openai_api_key` else `cast(RankLLM, _UnconfiguredLLM("rank")), cast(FinalizeLLM, _UnconfiguredLLM("finalize"))` where `_UnconfiguredLLM.__call__(*a, **k) -> NoReturn` raises RuntimeError("OPENAI_API_KEY not set"); unreachable in degraded flow because admission refuses before any LLM node.
7. Lifespan:
```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        cp = checkpointer
        if cp is None:
            if settings.database_url:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                cp = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(settings.database_url))
                await cp.setup()
            else:
                from langgraph.checkpoint.memory import MemorySaver
                cp = MemorySaver()
        common = dict(repo=repo, surepass=surepass, rank_llm=rank_llm, finalize_llm=finalize_llm, checkpointer=cp)
        app.state.quick_graph = build_graph("quick", **common)
        app.state.full_graph = build_graph("full", **common)
        yield
```
8. `app = FastAPI(title="eligibility-agent", lifespan=lifespan)`; `app.state.settings = settings`; `app.state.repo = repo`; register validation_handler; mount routers.
9. Routers: `health_router = APIRouter()` (no deps) with GET /health; `checks_router = APIRouter(prefix="/checks", dependencies=[Depends(require_bearer)])` with POST /quick and POST /full, both `response_model=LcView` (default serialization keeps explicit nulls, matching the PRD 9.3 sample).
10. NO module-level `app = create_app()`. Export `def app_factory() -> FastAPI: return create_app()`; run with `uvicorn 'eligibility_agent.main:app_factory' --factory`. (Import-time app construction would read the developer's real .env during test collection.)

ROUTES + FLOW:
```python
@health_router.get("/health")
async def health(request: Request) -> JSONResponse:
    s: Settings = request.app.state.settings
    return JSONResponse({"ok": True, "surepass_live": s.surepass_live})   # EXACT two keys, 200 always, no auth, no dependency probes (liveness only — must succeed on degraded boot)

@checks_router.post("/quick", response_model=LcView)
async def post_quick(req: QuickCheckRequest, request: Request) -> LcView:
    return await _run_check(request.app, "quick", req, stage=None)

@checks_router.post("/full", response_model=LcView)
async def post_full(req: FullCheckRequest, request: Request) -> LcView:
    return await _run_check(request.app, "full", req, stage=req.opportunity_stage)

async def _run_check(app: FastAPI, kind: Literal["quick", "full"], req: QuickCheckRequest,
                     stage: OpportunityStage | None) -> LcView:
    settings: Settings = app.state.settings
    check_id = str(uuid.uuid4())
    run_id = uuid.uuid4()                      # pre-generated so the LangSmith root run id is known before invoke
    pan_hash = hash_pan(req.pan, salt=settings.pan_hash_salt)
    state: CheckState = {
        "kind": kind, "check_id": check_id,
        "lc_id": req.lc_id, "lead_id": req.lead_id,
        "opportunity_id": req.opportunity_id,
        "opportunity_stage": stage.value if stage is not None else None,
        "name": req.name, "mobile": req.mobile, "pan": req.pan, "pan_hash": pan_hash,
        "monthly_income": req.monthly_income, "existing_emi_declared": req.existing_emi,
        "refresh": req.refresh, "consent": req.consent,   # consent is a CheckState extension (see interfaces_consumed)
        "errors": [],
    }
    config: RunnableConfig = {
        "run_id": run_id,
        "run_name": f"eligibility-{kind}",
        "configurable": {"thread_id": check_id},          # checkpointer keyed by check_id (PRD 7.3)
        "tags": [f"tier:{kind}", f"lc:{req.lc_id}"],      # admission node appends its own 'admission:<mode>' tag
        "metadata": {"check_id": check_id, "pan_hash": pan_hash, "lc_id": req.lc_id,
                      "lead_id": req.lead_id, "opportunity_id": req.opportunity_id or "",
                      "langsmith_run_id": str(run_id)},    # persist node copies this onto the checks row
    }
    graph = app.state.quick_graph if kind == "quick" else app.state.full_graph
    try:
        final = await asyncio.wait_for(graph.ainvoke(state, config), timeout=settings.graph_timeout_seconds)
        lc_view = final.get("lc_view")
        if lc_view is None:
            raise RuntimeError("graph finished without lc_view")
        return LcView.model_validate(lc_view)              # tolerant of model-or-dict in state
    except Exception:
        logger.exception("check failed check_id=%s kind=%s pan_hash=%s", check_id, kind, pan_hash)  # NEVER log pan/name/mobile
        try:
            app.state.repo.record_check_error(check_id=check_id, kind=kind, lc_id=req.lc_id,
                                              lead_id=req.lead_id, opportunity_id=req.opportunity_id,
                                              pan_hash=pan_hash, message="internal_error")
        except Exception:
            logger.exception("could not persist error row check_id=%s", check_id)
        return build_error_lc_view(check_id=check_id, kind=kind)   # HTTP 200, outcome=review (see decisions)
```
Note: metadata/tags never carry the PAN, name or mobile (PRD 11: pan_hash only).

### eligibility_agent/output/lc_view.py
_The redaction boundary: exact leak regexes, whole-line scrubbing, LcView builders for ok / pan_invalid / refused / error paths, all six refusal strings, lender name resolution, cache/cap display fields._

CONSTANTS AND REGEXES (exact, compiled at import):
```python
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Literal, Sequence

from eligibility_agent.graph.state import CheckState, LcView, LenderRef, LoanRange

# PRD 7.3 pattern plus optional plural 's' on bureau (strict superset of the FIXED rule).
BUREAU_TERM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:cibil|experian|equifax|crif|bureaus?)\b", re.IGNORECASE)

# Standalone integer 300-899. Guards (all fixed-width, legal in Python re):
#   (?<!\d)      not preceded by a digit            -> kills 1736 / 45000
#   (?<!\d[.,])  not preceded by digit+sep          -> kills 3,500 / 12,34,567 / 0.736 (but NOT Rs.736, 's' is not a digit)
#   (?!\d)       not followed by a digit            -> kills 7360 / 500000
#   (?![.,]\d)   not followed by sep+digit          -> kills 736,000 / 736.50 (decimal => amount; scores are integers)
STANDALONE_SCORE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d)")

LEAK_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (BUREAU_TERM_RE, STANDALONE_SCORE_RE)

SCRUB_FALLBACK_LINE: Final[str] = "Recommendation ready. Review the suggested lenders with the learner."
MANUAL_REVIEW_LINE: Final[str] = "The check could not be completed automatically. Please raise it for manual review."
PAN_INVALID_LINE: Final[str] = ("This PAN could not be verified as a valid individual PAN. "
                                 "Please correct it with the learner and run the check again; today's quota was not used.")
PAN_INVALID_CONFIRM: Final[str] = "The PAN exactly as printed on the card"

REFUSAL_MESSAGES: Final[dict[str, str]] = {
    "CONSENT_MISSING": "Consent was not recorded. Please take the learner's verbal consent for an eligibility check, tick the consent box and run the check again.",
    "DAILY_CAP_REACHED": "You have used all five quick checks for today. The limit resets at midnight; cached results stay available in the meantime.",
    "STAGE_TOO_EARLY": "A full check unlocks once this opportunity reaches the offer stage. Until then, use the quick check from SuperConnect.",
    "OPPORTUNITY_ALREADY_CHECKED": "A full check has already been run for this opportunity and is still current. Open the saved result; a re-run unlocks once it expires.",
    "CACHE_FRESH_NO_REFRESH": "This learner's last result is still current, so a refresh is not available yet. Open the saved result; Refresh unlocks once it expires.",
    "LIVE_CALLS_DISABLED": "Eligibility checks are temporarily switched off on the server. Please try again later or contact the platform team.",
}
# Unit test asserts: every constant above passes line_leaks() == False.
```

SCRUBBER:
```python
def line_leaks(line: str) -> bool:
    return any(p.search(line) for p in LEAK_PATTERNS)

def scrub_lines(lines: Sequence[str]) -> list[str]:
    """Whole-line drop: an offending line is removed, never partially edited."""
    return [ln for ln in lines if not line_leaks(ln)]
```
REGEX DECISION TABLE (must be the unit-test vectors):
| input line | result | why |
|---|---|---|
| "736" / "Score is 736" / "(736)" / "736." / "736-750" | DROP | bare standalone 300-899 |
| "Rs.736 looks strong" | DROP | '.' preceded by 's' not a digit -> currency-abbrev, not decimal |
| "band below 650" | DROP | 650 in range; band-threshold numerals are leaks by PRD design |
| "CIBIL/Experian/Equifax/CRIF says...", "per the bureaus" | DROP | bureau terms, any case, plural |
| "EMI of 3,500 fits" | KEEP | 500 preceded by digit+comma |
| "income 45,000 per month" / "45,500" | KEEP | grouped amount |
| "loan up to 3,50,000" / "12,34,567" | KEEP | Indian grouping; no bare 3-digit run in range |
| "tenure 36 months at 14 percent" | KEEP | 2-digit numbers |
| "736.50 rupees" | KEEP | decimal => amount (scores are integers) |
| "paid 1736" / "500000" / "0.736" | KEEP | digit-adjacent |
| "900" / "299" / "2026" | KEEP | outside 300-899 |
| "₹736" | DROP | currency-symbol-prefixed bare 3-digit: scrubbed in the safe direction (real INR talking-point amounts are >= 4 digits or grouped) |

BUILDERS:
```python
@dataclass(frozen=True, slots=True)
class RedactionContext:
    """Supplied by the redact node (graph section) — lc_view.py stays a pure function."""
    quick_checks_remaining_today: int | None = None   # quick kind only; formula in interfaces_consumed
    cache_fetched_at: datetime | None = None          # cache row fetched_at (fresh pull => now)
    cache_expires_at: datetime | None = None          # cache row expires_at (fetched_at + 30d)

def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt is not None else None

def round_loan_to_thousand(v: float) -> int:
    return int(round(v / 1000.0)) * 1000

def _safe_lender_name(lender_id: str, name: str) -> str:
    return name if not line_leaks(name) else lender_id   # DB names are admin-controlled; belt-and-braces

def build_lc_view(state: CheckState, ctx: RedactionContext) -> LcView: ...
def build_error_lc_view(*, check_id: str, kind: Literal["quick", "full"]) -> LcView: ...
```
build_lc_view branch order (pseudocode):
```
rc = state.get("refusal_code")
if rc is not None:                                    -> REFUSED view
elif (ident := state.get("identity")) is not None and not ident.pan_valid:  -> PAN_INVALID view
elif (final := state.get("final")) is not None:       -> OK view
else:                                                 -> defensive: same shape as build_error_lc_view
```
REFUSED view: status="refused", refusal_code=rc, outcome/confidence/range/top/alternates(=[])/coapplicant=None, talking_points=[REFUSAL_MESSAGES.get(rc, MANUAL_REVIEW_LINE)], confirm_with_learner=[], pan_aadhaar_link_issue=None, from_cache=False, cached_at=_iso(ctx.cache_fetched_at), refresh_available_at=_iso(ctx.cache_expires_at) (non-null exactly for CACHE_FRESH_NO_REFRESH / OPPORTUNITY_ALREADY_CHECKED where a live cache row motivated the refusal), quick_checks_remaining_today=ctx value when kind=="quick" else None (0 for DAILY_CAP_REACHED).
PAN_INVALID view: status="pan_invalid", refusal_code=None, talking_points=[PAN_INVALID_LINE], confirm_with_learner=[PAN_INVALID_CONFIRM], all recommendation fields None/[], from_cache=False, cached_at=None, refresh_available_at=None (nothing cached — policy rule 6), quick_checks_remaining_today=ctx value (unchanged: not counted), pan_aadhaar_link_issue=None.
OK view:
```
names = {gl.id: gl.name for gl in state.get("gated_lenders", [])}
top   = LenderRef(id=t, name=_safe_lender_name(t, names[t])) if (t := final.top_lender_id) in names else None
alts  = [LenderRef(id=a, name=_safe_lender_name(a, names[a]))
         for a in final.alternate_lender_ids[:2] if a in names and a != final.top_lender_id]
if final.indicative_loan_min is not None and final.indicative_loan_max is not None:
    mn, mx = max(0, round_loan_to_thousand(final.indicative_loan_min)), round_loan_to_thousand(final.indicative_loan_max)
    rng = LoanRange(min=mn, max=mx) if mx > 0 and mn <= mx else None
else: rng = None
tp = scrub_lines(final.lc_talking_points)[:4] or [SCRUB_FALLBACK_LINE]
cw = scrub_lines(final.confirm_with_learner)[:3]
link_issue: Literal[True] | None = True if (ident is not None and ident.aadhaar_linked is False) else None
return LcView(check_id=state["check_id"], kind=state["kind"], status="ok", refusal_code=None,
    outcome=final.outcome, confidence=final.confidence, indicative_loan_range=rng,
    top_lender=top, alternates=alts, coapplicant_suggested=final.coapplicant_suggested,
    talking_points=tp, confirm_with_learner=cw, pan_aadhaar_link_issue=link_issue,
    from_cache=state.get("admission") == "cache_hit",
    cached_at=_iso(ctx.cache_fetched_at), refresh_available_at=_iso(ctx.cache_expires_at),
    quick_checks_remaining_today=ctx.quick_checks_remaining_today)
```
build_error_lc_view: LcView(check_id, kind, status="ok", refusal_code=None, outcome="review", confidence="low", indicative_loan_range=None, top_lender=None, alternates=[], coapplicant_suggested=None, talking_points=[MANUAL_REVIEW_LINE], confirm_with_learner=[], pan_aadhaar_link_issue=None, from_cache=False, cached_at=None, refresh_available_at=None, quick_checks_remaining_today=None) — deterministic, mirrors the in-graph fallback_finalize wording.

LCVIEW FIELD-BY-FIELD LEAK AUDIT (why only three fields need scrubbing):
| field | type | leak-capable? |
|---|---|---|
| check_id | uuid str (code-generated) | no |
| kind, status, refusal_code, outcome, confidence | closed Literal enums | no |
| indicative_loan_range.min/max | ints rounded to nearest 1,000 by deterministic code (a smuggled '736' becomes 1000) | no |
| top_lender.id / alternates[].id | DB primary keys, membership-verified against gated+accepted upstream | no |
| top_lender.name / alternates[].name | admin-entered DB strings -> passed through line_leaks; on match display falls back to id | guarded |
| coapplicant_suggested, from_cache | bool (code-set) | no |
| talking_points[] | LLM prose -> scrub_lines, whole-line drop | SCRUBBED |
| confirm_with_learner[] | LLM prose -> scrub_lines | SCRUBBED |
| pan_aadhaar_link_issue | Literal[True]\|None, code-set only from identity.aadhaar_linked is False | no |
| cached_at, refresh_available_at | ISO strings from code-held datetimes | no |
| quick_checks_remaining_today | int 0..5 from deterministic cap math | no |
LcView uses extra="forbid": no undeclared field can ride along; score/bureau/raw-report/masked-Aadhaar have no field to live in.

REFUSAL TABLE (persisted checks row content per code; written by persist_refusal using repo; all rows share id=check_id, kind, lc_id, lead_id, opportunity_id, pan_hash, status='refused', billed=false, counted_against_cap=false, is_refresh=<request.refresh>, lc_view=<refusal LcView JSON above>, langsmith_run_id, created_at):
| refusal_code | consent_id | internal jsonb | LC talking_points string (exact) |
|---|---|---|---|
| CONSENT_MISSING | NULL (no consents row is inserted when the flag is false — see prd_gaps) | {"reason": "consent flag false"} | "Consent was not recorded. Please take the learner's verbal consent for an eligibility check, tick the consent box and run the check again." |
| DAILY_CAP_REACHED | set | {"used_today": 5, "cap": 5} | "You have used all five quick checks for today. The limit resets at midnight; cached results stay available in the meantime." |
| STAGE_TOO_EARLY | set | {"stage": "<given>", "required": "offer"} | "A full check unlocks once this opportunity reaches the offer stage. Until then, use the quick check from SuperConnect." |
| OPPORTUNITY_ALREADY_CHECKED | set | {"prior_check_id": "...", "cache_expires_at": "<iso>"} | "A full check has already been run for this opportunity and is still current. Open the saved result; a re-run unlocks once it expires." |
| CACHE_FRESH_NO_REFRESH | set | {"cache_fetched_at": "<iso>", "cache_expires_at": "<iso>"} | "This learner's last result is still current, so a refresh is not available yet. Open the saved result; Refresh unlocks once it expires." |
| LIVE_CALLS_DISABLED | set | {"surepass_live": false} | "Eligibility checks are temporarily switched off on the server. Please try again later or contact the platform team." |

### eligibility_agent/output/__init__.py
_Package marker; re-export the public redaction API._

from eligibility_agent.output.lc_view import (LEAK_PATTERNS, RedactionContext, build_error_lc_view, build_lc_view, line_leaks, scrub_lines)
__all__ = ["LEAK_PATTERNS", "RedactionContext", "build_error_lc_view", "build_lc_view", "line_leaks", "scrub_lines"]

### .env.example
_Documented env template matching the config.py table exactly; everything commented out so the file also demonstrates degraded boot._

# --- SurePass (PRODUCTION CREDITS - kill switch) ---
# SUREPASS_LIVE=true            # live iff EXACTLY the string: true (anything else = disabled)
# SUREPASS_TOKEN=
# SUREPASS_BASE_URL=https://kyc-api.surepass.app
# --- Service auth ---
# SERVICE_API_TOKEN=            # unset => auth disabled (dev only; required when REPO_BACKEND=supabase)
# --- PAN hashing ---
# PAN_HASH_SALT=                # required (non-default) when REPO_BACKEND=supabase; rotating it invalidates the 30-day cache
# --- Persistence ---
# REPO_BACKEND=memory           # memory | supabase
# SUPABASE_URL=
# SUPABASE_SERVICE_ROLE_KEY=
# DATABASE_URL=                 # Postgres for the LangGraph checkpointer; unset => in-memory checkpoints
# --- LLM ---
# OPENAI_API_KEY=
# OPENAI_MODEL=gpt-4.1
# OPENAI_TEMPERATURE=0
# --- LangSmith ---
# LANGSMITH_TRACING=true        # iff EXACTLY: true
# LANGSMITH_API_KEY=
# LANGSMITH_PROJECT=eligibility-agent
# PROMPT_SOURCE=local           # local (repo-pinned prompts) | hub
# PROMPT_RANK_REF=hero/eligibility-rank
# PROMPT_FINALIZE_REF=hero/eligibility-finalize
# PROMPT_RANK_TAG=
# PROMPT_FINALIZE_TAG=
# --- Service behaviour ---
# GRAPH_TIMEOUT_SECONDS=90
# LOG_LEVEL=INFO

### tests/test_config.py
_TDD cases for config.py._

All tests build Settings(_env_file=None, ...) for hermeticity. Cases:
1. test_empty_env_constructs: Settings(_env_file=None) -> surepass_live False, repo_backend 'memory', auth_enabled False, startup_errors() == [].
2. test_kill_switch_literal: SUREPASS_LIVE in {"true"} -> live True; {"True","TRUE","1","yes",""} -> False each.
3. test_startup_errors_supabase: REPO_BACKEND=supabase alone -> three errors (url/key, salt, api token); fully set with real salt+token -> [].
4. test_live_requires_token: SUREPASS_LIVE=true without SUREPASS_TOKEN -> one error.
5. test_hash_pan_canonical: hash_pan(" abcde1234f ", salt="s") == hash_pan("ABCDE1234F", salt="s") == hashlib.sha256(b"s:ABCDE1234F").hexdigest(); different salt -> different digest; output is 64 lowercase hex chars.
6. test_langsmith_literal: LANGSMITH_TRACING="true" -> property True; "false"/unset -> False.

### tests/test_api_auth_and_validation.py
_TDD cases for main.py (auth, validation split, route flow, error boundary, health). Uses fastapi.testclient.TestClient as a context manager (runs lifespan) over create_app(settings=Settings(_env_file=None, SERVICE_API_TOKEN='t'), repo=MemoryRepo(), surepass=FakeSurePassClient(...), rank_llm=FakeLLM..., finalize_llm=FakeLLM...)._

Cases:
1. test_health_open_and_exact_shape: GET /health with no auth -> 200, body == {"ok": True, "surepass_live": False}.
2. test_auth_required: POST /checks/quick without header -> 401 + WWW-Authenticate: Bearer; wrong token -> 401; right token -> not 401.
3. test_auth_open_when_unset: create_app with SERVICE_API_TOKEN=None -> request without header reaches the graph (gets a 200 LcView refusal, not 401).
4. test_pan_422: pan "ABCDE12345" -> 422; "abcde1234f" (lowercase) -> accepted (normalized); response for the 422 contains loc/msg/type but NO "input" key anywhere (PAN never echoed).
5. test_mobile_422_and_normalization: "5876543210" -> 422; "+91 98765-43210" -> accepted as "9876543210".
6. test_consent_matrix: consent absent -> 422; consent "true" (string) -> 422 (StrictBool); consent false -> 200, status "refused", refusal_code "CONSENT_MISSING", and MemoryRepo holds a checks row with billed=False, counted_against_cap=False, consent_id=None.
7. test_full_required_fields: /checks/full missing opportunity_id / opportunity_stage / monthly_income -> 422 each; opportunity_stage="prospect" -> 422; stage "qualified" -> 200 refused STAGE_TOO_EARLY (persisted row, no SurePass call in fake log).
8. test_unknown_field_forbidden: extra key "monthly_incom" -> 422.
9. test_degraded_boot_refusal: create_app(settings=Settings(_env_file=None)) with NO injected fakes -> POST /checks/quick (valid body, consent true) -> 200 LcView status refused, refusal_code LIVE_CALLS_DISABLED; /health ok. Proves PRD 13 end-to-end with zero config.
10. test_error_boundary: inject a rank_llm that raises RuntimeError -> if graph's fallback chain is bypassed (monkeypatch a node to raise), route returns 200 with outcome "review", confidence "low", talking_points == [MANUAL_REVIEW_LINE]; repo has a status='error' row for the check_id.
11. test_response_keeps_nulls: refusal response JSON contains explicit null for refusal-irrelevant keys (outcome, top_lender, ...), matching PRD 9.3 sample.
12. test_runnable_config: with a spy graph, assert configurable.thread_id == body check_id, tags == ["tier:quick", "lc:lc_123"], metadata has pan_hash (64 hex) and NO raw pan/name/mobile anywhere in config.

### tests/test_lc_view_redaction.py
_TDD cases for output/lc_view.py: regex vector table, builder paths, aadhaar truth table, refusal strings self-clean._

Cases:
1. test_scrub_vectors: parametrize the full DROP/KEEP table from lc_view.py contents (736, Score is 736, (736), 736., 736-750, Rs.736, below 650, ₹736, Experian/CIBIL/Equifax/CRIF/bureaus => DROP; 3,500 / 45,000 / 45,500 / 3,50,000 / 12,34,567 / 36 months / 14 percent / 736.50 / 1736 / 0.736 / 500000 / 900 / 299 / 2026 => KEEP).
2. test_refusal_messages_clean: every value in REFUSAL_MESSAGES plus PAN_INVALID_LINE, PAN_INVALID_CONFIRM, MANUAL_REVIEW_LINE, SCRUB_FALLBACK_LINE has line_leaks(...) is False.
3. test_ok_view: full state (final with top L1, alternates [L2, L3, L4], gated L1..L3) -> alternates truncated to 2 and only gated ids kept; names resolved; loan 2_00_000.4/5_00_400.0 -> {min: 200000, max: 500000}.
4. test_ok_view_scrub_empties_fallback: talking points all leaky -> talking_points == [SCRUB_FALLBACK_LINE].
5. test_aadhaar_truth_table: aadhaar_linked False -> pan_aadhaar_link_issue True; True -> None; None -> None; identity None (refusal path) -> None.
6. test_pan_invalid_view: identity.pan_valid False -> status pan_invalid, cached_at None, refresh_available_at None, quick_checks_remaining_today passed through unchanged.
7. test_refused_view_cache_fields: CACHE_FRESH_NO_REFRESH with ctx cache timestamps -> refresh_available_at set; CONSENT_MISSING with empty ctx -> both None.
8. test_loan_rounding_kills_smuggling: final loan min=max=736.0 -> rounds to 1000/1000 -> range {min:1000,max:1000}; min=max=400.0 -> rounds to 0 -> range None.
9. test_from_cache_flag: admission == "cache_hit" -> from_cache True else False.
10. test_lender_name_leak_fallback: gated lender name "Experian Finance Ltd" -> displayed name falls back to the lender id.

## Interfaces provided
- **name**: config.hash_pan | **definition**: def hash_pan(pan: str, *, salt: str) -> str — sha256(f"{salt}:{pan.strip().upper()}").hexdigest(), 64 lowercase hex chars. THE single canonical PAN hasher; policy (cache keys, consents), db and evals must import this exact function. Salt comes from Settings.pan_hash_salt.
- **name**: config.Settings / get_settings / ConfigError / utcnow | **definition**: class Settings(BaseSettings) with the full env table (see config.py file entry); properties surepass_live: bool (== "true" exact), langsmith_tracing: bool, auth_enabled: bool; startup_errors() -> list[str]. @lru_cache get_settings() -> Settings. Tests: Settings(_env_file=None, <ALIAS>=value). utcnow() -> timezone-aware datetime (UTC).
- **name**: main.create_app | **definition**: def create_app(settings: Settings | None = None, *, repo: Repo | None = None, surepass: SurePassClient | None = None, rank_llm: RankLLM | None = None, finalize_llm: FinalizeLLM | None = None, checkpointer: BaseCheckpointSaver | None = None) -> FastAPI. Tests build a fully-faked app and run it under TestClient as context manager (lifespan compiles graphs). app.state carries settings, repo, quick_graph, full_graph. Also exported: app_factory() -> FastAPI for uvicorn --factory.
- **name**: HTTP contract | **definition**: GET /health -> 200 {"ok": <bool>, "surepass_live": <bool>} exactly, no auth. POST /checks/quick and /checks/full -> 200 LcView JSON with explicit nulls (PRD 9.3 shape) for ok/pan_invalid/refused/error-review; 422 {"detail": [{loc, msg, type}]} with no input echo; 401 {"detail": "invalid_or_missing_token"} + WWW-Authenticate: Bearer. Refusals are always HTTP 200.
- **name**: output.lc_view scrubber API | **definition**: BUREAU_TERM_RE = re.compile(r"\b(?:cibil|experian|equifax|crif|bureaus?)\b", re.IGNORECASE); STANDALONE_SCORE_RE = re.compile(r"(?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d)"); LEAK_PATTERNS: tuple[re.Pattern[str], ...]; line_leaks(line: str) -> bool; scrub_lines(lines: Sequence[str]) -> list[str] (whole-line drop). The evals section's no_leak evaluator MUST import LEAK_PATTERNS/line_leaks from here so runtime scrubbing and evaluation can never drift.
- **name**: output.lc_view builders | **definition**: @dataclass(frozen=True, slots=True) RedactionContext(quick_checks_remaining_today: int | None = None, cache_fetched_at: datetime | None = None, cache_expires_at: datetime | None = None); build_lc_view(state: CheckState, ctx: RedactionContext) -> LcView (branches refused / pan_invalid / ok / defensive-error); build_error_lc_view(*, check_id: str, kind: Literal["quick","full"]) -> LcView; round_loan_to_thousand(v: float) -> int.
- **name**: output.lc_view fixed LC strings | **definition**: REFUSAL_MESSAGES: dict[str, str] with the six exact strings (CONSENT_MISSING, DAILY_CAP_REACHED, STAGE_TOO_EARLY, OPPORTUNITY_ALREADY_CHECKED, CACHE_FRESH_NO_REFRESH, LIVE_CALLS_DISABLED — texts in the lc_view.py file entry); PAN_INVALID_LINE; PAN_INVALID_CONFIRM; MANUAL_REVIEW_LINE ("The check could not be completed automatically. Please raise it for manual review." — must equal fallback_finalize's message in the graph section); SCRUB_FALLBACK_LINE. All guaranteed line_leaks-clean by unit test.

## Interfaces consumed
- **name**: graph/state.py — CheckState + consent extension | **expectation**: CheckState TypedDict exactly per PRD 7.2 PLUS one extension key `consent: bool` (the route writes it; admission reads it to decide CONSENT_MISSING; PRD's state list omits it but admission cannot gate without it). The route populates: kind, check_id, lc_id, lead_id, opportunity_id (None for quick without one), opportunity_stage (stage.value str or None), name, mobile, pan, pan_hash, monthly_income, existing_emi_declared, refresh, consent, errors=[]. mypy --strict: TypedDict total=False with these keys declared.
- **name**: graph/state.py — LcView / LenderRef / LoanRange (wire contract, graph section owns the file, must adopt verbatim) | **expectation**: class LenderRef(BaseModel): id: str; name: str. class LoanRange(BaseModel): min: int; max: int. class LcView(BaseModel): model_config = ConfigDict(extra="forbid"); check_id: str; kind: Literal["quick","full"]; status: Literal["ok","pan_invalid","refused"]; refusal_code: Literal["CONSENT_MISSING","DAILY_CAP_REACHED","STAGE_TOO_EARLY","OPPORTUNITY_ALREADY_CHECKED","CACHE_FRESH_NO_REFRESH","LIVE_CALLS_DISABLED"] | None = None; outcome: Literal["fundable","fundable_with_coapplicant","not_fundable","review"] | None = None; confidence: Literal["high","medium","low"] | None = None; indicative_loan_range: LoanRange | None = None; top_lender: LenderRef | None = None; alternates: list[LenderRef] = []; coapplicant_suggested: bool | None = None; talking_points: list[str] = []; confirm_with_learner: list[str] = []; pan_aadhaar_link_issue: Literal[True] | None = None; from_cache: bool = False; cached_at: str | None = None; refresh_available_at: str | None = None; quick_checks_remaining_today: int | None = None.
- **name**: graph/state.py — Identity / GatedLender / FinalRecommendation fields my builder reads | **expectation**: Identity.pan_valid: bool and Identity.aadhaar_linked: bool | None (adapter: pan_valid = success and status=='valid' and category=='person'; aadhaar_linked is a verified bool per field map). GatedLender.id: str and .name: str. FinalRecommendation per PRD 7.3 with outcome/confidence Literals, indicative_loan_min/max: float | None, top_lender_id: str | None, alternate_lender_ids: list[str] (max 2), coapplicant_suggested: bool, lc_talking_points: list[str] (max 4, each <= 200 chars), confirm_with_learner: list[str] (max 3). state['admission'] uses the literal "cache_hit" for cache hits.
- **name**: graph/build.py — build_graph + LLM seam | **expectation**: build_graph(kind: Literal["quick","full"], *, repo: Repo, surepass: SurePassClient, rank_llm: RankLLM, finalize_llm: FinalizeLLM, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph supporting `await graph.ainvoke(state, config)` returning the final CheckState dict with state['lc_view'] set on EVERY terminal path (ok, pan_invalid, refused). Exports RankLLM / FinalizeLLM as Protocol types (callables) and make_openai_llms(model: str, temperature: float, api_key: str, prompt_source: Literal["local","hub"], rank_ref: str, finalize_ref: str, rank_tag: str | None, finalize_tag: str | None) -> tuple[RankLLM, FinalizeLLM]. fallback_finalize's LC message must be exactly output.lc_view.MANUAL_REVIEW_LINE.
- **name**: graph nodes — redact/persist/admission responsibilities toward my boundary | **expectation**: redact node calls output.lc_view.build_lc_view(state, RedactionContext(...)) and sources: quick_checks_remaining_today = max(0, QUICK_DAILY_CAP - used_before_this_run - (1 if (counts_against_cap and bureau_called and not pan_invalid) else 0)) for kind==quick else None (current check not yet persisted when redact runs — persist comes after); cache_fetched_at/cache_expires_at from the pan_cache row the admission/pull nodes touched (fresh pull: fetched_at=now, expires_at=now+30d; cache hit: the row's stored values; refusal: the row that motivated CACHE_FRESH_NO_REFRESH/OPPORTUNITY_ALREADY_CHECKED, else None; pan_invalid: both None). persist/persist_refusal store lc_view JSON on the checks row, copy config metadata['langsmith_run_id'] into checks.langsmith_run_id, and update LangSmith run metadata with billed/counted_against_cap. admission node appends tag f"admission:{mode}" to its span. Admission precedence must be: consent -> stage (full) -> cache-hit admit (beats everything below so cached results work with live off and at cap) -> CACHE_FRESH_NO_REFRESH -> OPPORTUNITY_ALREADY_CHECKED -> DAILY_CAP_REACHED -> LIVE_CALLS_DISABLED last (only when a billed call would be needed).
- **name**: policy/guards.py — stage enum + cap constants | **expectation**: class OpportunityStage(enum.StrEnum): NEW="new"; CONTACTED="contacted"; QUALIFIED="qualified"; COUNSELLING="counselling"; OFFER="offer"; REGISTRATION="registration"; ENROLLED="enrolled"; WON="won" (ASSUMPTION per PRD 9.2, single swap point for the real FulCRM enum); STAGE_ORDER: dict[OpportunityStage, int] in that order; MIN_FULL_CHECK_STAGE = OpportunityStage.OFFER; QUICK_DAILY_CAP: Final[int] = 5 and CACHE_TTL_DAYS: Final[int] = 30 as code constants (NOT env vars). main.py imports OpportunityStage only; no policy import from main (no cycle).
- **name**: db/repo.py — Repo protocol pieces main.py touches | **expectation**: MemoryRepo() constructible with no args; SupabaseRepo(url: str, service_role_key: str). Protocol includes record_check_error(check_id: str, kind: str, lc_id: str, lead_id: str, opportunity_id: str | None, pan_hash: str, message: str) -> None with UPSERT semantics (insert a checks row status='error' if admission never inserted one, else update status to 'error' and set internal.error). checks table status domain must therefore admit 'error' in addition to started/ok/pan_invalid/refused; LcView.status never carries 'error' (wire enum stays three values).
- **name**: tools/surepass.py — client construction + kill switch | **expectation**: HttpSurePassClient(token: str, base_url: str, live: bool); every call raises LiveCallsDisabled when live is False (client-internal enforcement; main passes live=settings.surepass_live). FakeSurePassClient for tests exposes a call log the API tests assert against (e.g. zero calls on refusals).
- **name**: evals section — regex reuse | **expectation**: no_leak evaluator imports LEAK_PATTERNS / line_leaks from eligibility_agent.output.lc_view (never re-implements the regexes) and applies them to every prose string in LcView: talking_points, confirm_with_learner, lender names.
- **name**: toolchain/pyproject (scaffold section) | **expectation**: Dependencies available: fastapi, uvicorn, pydantic>=2, pydantic-settings, langchain-core, langchain-openai, langgraph, langgraph-checkpoint-postgres, supabase, httpx (TestClient). mypy --strict configured with plugins = ["pydantic.mypy"]. pytest runs with no network; tests construct Settings(_env_file=None, ...) so a developer's local .env never bleeds in.

## Decisions
- **topic**: Auth mechanism | **decision**: Bearer auth is a FastAPI dependency (HTTPBearer(auto_error=False) + hmac.compare_digest) attached at the /checks router level; /health sits on a separate router with no dependency. | **rationale**: Dependencies are overridable per-test (app.dependency_overrides or injected Settings), self-document in OpenAPI, and make the /health exemption structural rather than a path-string match in middleware; constant-time compare avoids token timing leaks.
- **topic**: HTTP status for policy outcomes | **decision**: Refusals and pan_invalid return HTTP 200 with LcView.status carrying the semantics; only schema/transport problems are 4xx (422/401). | **rationale**: SuperConnect/FulCRM render one object uniformly; 4xx would trigger client retry/alerting on normal business outcomes, and the PRD requires refusals to be persisted checks rows, i.e. first-class results.
- **topic**: consent validation split | **decision**: consent is StrictBool: absent or non-boolean JSON -> 422; false -> graph runs and admission produces a persisted CONSENT_MISSING refusal row. CheckState gains a consent: bool extension to carry it. | **rationale**: Implements the task directive resolving PRD 9.1's 'literally true': a false flag is a real LC interaction worth auditing (a checks row), while a malformed request never became a check; StrictBool stops pydantic's lax 'true'/1 coercion from counterfeiting consent.
- **topic**: Unexpected node exception / timeout | **decision**: Route catches Exception around graph.ainvoke (asyncio.wait_for, GRAPH_TIMEOUT_SECONDS=90 default): returns HTTP 200 with a deterministic review-outcome LcView (outcome=review, confidence=low, MANUAL_REVIEW_LINE) and best-effort upserts checks.status='error' with internal error info; never a 500 to the LC. | **rationale**: Mirrors the PRD's own fallback_finalize philosophy; a 500 mid-call invites blind retries that can double-bill SurePass before the cache row exists, while 'raise for manual review' routes to a human; the error row + logs + LangSmith trace preserve ops visibility.
- **topic**: Degraded boot | **decision**: Every env var is optional with an in-process fallback: REPO_BACKEND defaults to memory, no DATABASE_URL -> MemorySaver, no OPENAI_API_KEY -> raising stub LLM callables, no SERVICE_API_TOKEN -> auth disabled with a startup warning. startup_errors() raises ConfigError only when supabase backend or live SurePass is explicitly selected without its prerequisites. | **rationale**: PRD 13 requires: with nothing set the service starts, serves /health, and returns LIVE_CALLS_DISABLED refusals. The stub LLMs are provably unreachable because admission refuses before any pull/LLM node on an empty cache.
- **topic**: Kill-switch parsing | **decision**: SUREPASS_LIVE is read as a raw string and compared to the exact lowercase literal "true" in a Settings property; no pydantic bool coercion. LANGSMITH_TRACING follows the same convention. | **rationale**: PRD FIXED rule says live 'unless env SUREPASS_LIVE == "true"'; bool coercion would make "1", "True", "yes" enable production spend. The client additionally enforces the switch internally (defense in depth, tools section).
- **topic**: Auth when SERVICE_API_TOKEN unset | **decision**: Requests are allowed through with a prominent startup warning; selecting REPO_BACKEND=supabase without SERVICE_API_TOKEN is a hard ConfigError. | **rationale**: The degraded-boot acceptance criterion requires /checks to answer (with refusals) under empty env, which is impossible if auth hard-fails; the blast radius is bounded (kill switch off, memory repo), and the prod backend cannot boot open.
- **topic**: Input normalization before validation | **decision**: PAN: strip+uppercase then ^[A-Z]{5}[0-9]{4}[A-Z]$. Mobile: strip spaces/hyphens, remove a literal '+91' prefix only, then ^[6-9][0-9]{9}$. Name: strip, non-blank. | **rationale**: LCs type these live on a call; normalization kills avoidable 422s without weakening the FIXED patterns (constraints run on the normalized value). Bare '91XXXXXXXXXX' is NOT stripped because a leading 9 is ambiguous with a real subscriber number.
- **topic**: Request model strictness | **decision**: extra="forbid" on both request models; custom RequestValidationError handler returns only loc/msg/type, stripping FastAPI's 'input' and 'ctx' echo. | **rationale**: Forbid catches client contract drift (misspelled field silently dropped = wrong recommendation); stripping the echo keeps PAN/mobile/name out of response bodies and intermediary logs.
- **topic**: Standalone 300-899 regex | **decision**: (?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d) — two fixed-width negative lookbehinds and two lookaheads; bureau terms \b(?:cibil|experian|equifax|crif|bureaus?)\b with re.IGNORECASE (plural 's' is a strict superset of the PRD's FIXED pattern). | **rationale**: The digit+separator lookbehind distinguishes decimals/groupings (0.736, 3,500, 12,34,567 -> keep) from currency abbreviations (Rs.736 -> drop); the separator lookahead treats 736.50/736,000 as amounts; verified by hand against every required vector including bare 736, (736), 736-750 and 650.
- **topic**: Currency-prefixed bare 3-digit numbers | **decision**: "₹736" is still dropped (the symbol is not a digit, so the lookbehind passes). | **rationale**: Safe direction: genuine INR amounts in talking points are >= 4 digits or comma-grouped; a ₹-prefixed 300-899 integer is a plausible disguised score and the no_leak evaluator would flag it anyway.
- **topic**: Scrub granularity and empty-result fallback | **decision**: Whole-line drop (never partial masking) applied to talking_points and confirm_with_learner; if scrubbing empties talking_points on an ok path, substitute the fixed SCRUB_FALLBACK_LINE. | **rationale**: PRD mandates dropping the offending line; partial masking can leave reconstructable context. An empty talking-points array would strand the LC mid-call, so a neutral deterministic line is substituted.
- **topic**: Loan range hygiene | **decision**: indicative_loan_range min/max are rounded to nearest ₹1,000 integers at the redaction boundary; a non-positive max (or missing endpoint) nulls the range. | **rationale**: Matches the 8.4 ceiling rounding, and deterministically destroys any 3-digit score smuggled through the numeric channel (736 -> 1000), closing the only non-prose leak path.
- **topic**: Lender name resolution | **decision**: Names resolve from state.gated_lenders (loaded from DB by gate_lenders); ids missing from the gated map are silently dropped; a DB name failing line_leaks displays the lender id instead. | **rationale**: Keeps build_lc_view a pure function of state (no repo call in the redaction boundary); upstream finalize validation already guarantees membership, so the map lookup is a cheap second fence.
- **topic**: pan_aadhaar_link_issue typing | **decision**: Field type Literal[True] | None; set True iff identity is present and identity.aadhaar_linked is False; None for true/unknown/absent identity. | **rationale**: Encodes the PRD rule ('shown only when it is a problem') in the type system so a 'False' value is unrepresentable; field map confirms aadhaar_linked is a bool but a false sample was never observed, so the negative case is fixture-only.
- **topic**: quick_checks_remaining_today scope | **decision**: Populated for kind=quick on ok, pan_invalid and refused views (0 for DAILY_CAP_REACHED); always null for kind=full. | **rationale**: PRD only shows it on the quick sample; full checks neither consume nor display the quick cap, and returning the live number on refusals is what lets SuperConnect grey out the button.
- **topic**: App construction | **decision**: No module-level app; uvicorn runs the factory ('eligibility_agent.main:app_factory' --factory); create_app accepts settings/repo/surepass/LLMs/checkpointer injections; graphs compile once in lifespan (AsyncPostgresSaver.from_conn_string held open via AsyncExitStack + setup(), MemorySaver when DATABASE_URL unset). | **rationale**: Import-time app construction would read the developer's real .env during pytest collection and could raise ConfigError; lifespan-scoped compilation gives one compiled graph per process and a clean checkpointer lifecycle (pattern verified against current langgraph-checkpoint-postgres usage).
- **topic**: Stage enum ownership (ASSUMPTION) | **decision**: OpportunityStage StrEnum (new, contacted, qualified, counselling, offer, registration, enrolled, won) lives in policy/guards.py and is imported by main.py for 422 membership validation; the >= offer ordering gate stays in policy (STAGE_TOO_EARLY refusal). | **rationale**: Membership is schema (bad string = malformed request), ordering is policy (valid stage, wrong time = auditable refusal); placing the enum in policy avoids a main<->policy import cycle and keeps the FulCRM-enum swap (PRD open item 1) in one file.
- **topic**: Observability identifiers | **decision**: Route pre-generates run_id (uuid4), passes it in RunnableConfig and duplicates it as metadata.langsmith_run_id; thread_id=check_id; tags [tier:<kind>, lc:<id>]; metadata carries pan_hash and never the PAN/name/mobile; logs use check_id/pan_hash only. | **rationale**: Pre-generating the root run id lets the persist node write langsmith_run_id onto the checks row without querying LangSmith; satisfies PRD 11's hash-only constraint everywhere.

## Risks
- Auth-open-when-unset is a deliberate footgun for the degraded-boot guarantee: a prod deploy that forgets SERVICE_API_TOKEN serves unauthenticated (mitigations: startup warning, supabase backend hard-fails without it, kill switch keeps spend impossible; README must call it out).
- Returning a 200 review-LcView on crashes can mask infra outages from client dashboards; ops must alert on checks.status='error' row rate and the error logs instead.
- The 300-899 rule is intentionally blunt: legitimate lines like 'band below 650' or any 300-899 figure (e.g. '365 days') are dropped whole; SCRUB_FALLBACK_LINE keeps the LC view non-empty but wording quality degrades when the LLM uses numerals — the prompt section must keep steering to band language.
- StrictBool consent will 422 clients that serialize booleans as strings; SuperConnect/FulCRM integration notes must specify JSON booleans.
- AsyncPostgresSaver.from_conn_string lifecycle (async context manager held via AsyncExitStack + await setup()) is version-sensitive in langgraph-checkpoint-postgres; pattern verified against current 2.x usage but the graph section should own a pinned version and may prefer to own checkpointer construction entirely.
- If the graph stores lc_view as a dict rather than the LcView model, LcView.model_validate in the route absorbs it, but extra="forbid" means any drift in field names fails loudly at the boundary (intended, but will surface as 200->review errors until fixed).
- PAN_HASH_SALT rotation silently invalidates every pan_cache row and resets dedup/caps keyed on pan_hash; document in README before anyone rotates it.
- Pydantic-settings alias behavior: constructor overrides in tests must use alias names (SUREPASS_LIVE=...), and _env_file=None is mandatory for hermetic tests — easy to get wrong and have a developer's .env leak into CI.

## PRD gaps
- CheckState (PRD 7.2) has no consent field, yet admission must emit CONSENT_MISSING; also no cache-timestamp or status fields for redact to build from_cache/cached_at/refresh_available_at. Resolved via a consent: bool state extension plus RedactionContext args populated by the redact node, with pan_invalid derived from identity.pan_valid — orchestrator should ratify these three seams.
- PRD 7.3 says admission 'Insert a consents row' before gating, but inserting one when the consent flag is false would fabricate a consent record. Designed: consents row only when consent is true; the CONSENT_MISSING checks row carries consent_id NULL. Needs owner confirmation.
- PRD 9.1 'consent must be literally true' vs the section directive 'consent false -> CONSENT_MISSING refusal row': implemented the directive (StrictBool; absent/non-bool -> 422; false -> persisted refusal).
- HTTP status codes for refused/pan_invalid responses are unspecified in the PRD; chose 200 with LcView.status carrying the outcome.
- quick_checks_remaining_today is undefined for full checks and for refusals; chose: populated for quick (including refusals, 0 at cap), always null for full.
- Which cache metadata appears on refusal LcViews is unspecified; chose refresh_available_at populated for CACHE_FRESH_NO_REFRESH and OPPORTUNITY_ALREADY_CHECKED (the row that motivated the refusal), null elsewhere.
- The checks.status column needs an 'error' value for the crash-boundary row, which PRD 10 does not list; wire-level LcView.status remains the fixed three-value enum.
- Stage enum remains ASSUMPTION (PRD open item 1); isolated in policy/guards.py as the single swap point.
- aadhaar_linked == false has never been observed in probe data (field map R3), so the pan_aadhaar_link_issue=True path is exercised only by synthetic fixtures.
