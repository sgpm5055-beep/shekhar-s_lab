# Graph, state, prompts, LangSmith wiring (PRD sections 7 and 11)

## Summary
The graph section delivers two compiled LangGraph StateGraphs (quick/full) that share every node except the pull node and the tier addendum. All policy, money and redaction logic stays in deterministic nodes; the LLM appears only as two injected structured-output callables (rank -> RankingProposal, finalize -> FinalRecommendation) satisfying a single generic StructuredLLM Protocol, so tests swap in a FakeLLM with zero monkeypatching. Dependencies (Repo, SurePassClient, the two LLM callables, PromptBundle, a now() clock) ride in a frozen GraphDeps dataclass bound to node functions via functools.partial at build time. CheckState keeps every PRD 7.2 key and adds attempt counters, rank/finalize error carriers, pan_called/bureau_called flags, cache metadata, consent flag/id, raw_payloads (stripped before persist) and an operator.add transcript. Routers implement the four conditional branches: refusal -> persist_refusal, pan-invalid -> redact, rank retry (max 2 total attempts), finalize retry (max 3 total attempts) then fallback_finalize; cache hits flow through the same linear path with nodes loading from pan_cache.derived instead of calling SurePass. The checkpointer is PostgresSaver over a manually built psycopg ConnectionPool (autocommit=True, dict_row, prepare_threshold=None) pointed at a dedicated non-PostgREST-exposed `checkpointer` schema on Supabase session pooler; tests compile with checkpointer=None (InMemorySaver only for resume tests). Tracing pre-generates the root run_id (verified: a root RunnableConfig run_id becomes the trace id), passes tags tier:/lc: and metadata check_id/pan_hash at invoke, and patches admission:<mode>, billed and counted_against_cap onto the run via Client.update_run after the run; the run id is stored on the checks row from state. prompts.py ships the full literal shared-core system prompt (~270 words), both tier addenda, deterministic user-message renderers with retry-feedback blocks, and a LangSmith Hub pull-by-tag loader that falls back to the pinned repo-local literals and is never used in tests (PROMPTS_SOURCE=local default). Critical files: eligibility_agent/graph/state.py, eligibility_agent/graph/nodes.py, eligibility_agent/graph/build.py, eligibility_agent/graph/prompts.py, eligibility_agent/graph/tracing.py.

## Files
### eligibility_agent/graph/state.py
_Single source of truth for CheckState and every Pydantic fact/output model shared across sections. No imports from any other eligibility_agent module (leaf module; policy/, tools/, output/, db/ all import it without cycles)._

from __future__ import annotations
import operator
from datetime import date, datetime
from typing import Annotated, Any, Final, Literal, TypeAlias, TypedDict
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Kind: TypeAlias = Literal["quick", "full"]
Band: TypeAlias = Literal["<650", "650-720", "720-780", "780+", "NTC"]
BAND_ORDER: Final[dict[Band, int]] = {"NTC": 0, "<650": 1, "650-720": 2, "720-780": 3, "780+": 4}  # shared by tools/lenders.check_hard_cutoffs for min_band comparison
RefusalCode: TypeAlias = Literal["CONSENT_MISSING", "DAILY_CAP_REACHED", "STAGE_TOO_EARLY", "OPPORTUNITY_ALREADY_CHECKED", "CACHE_FRESH_NO_REFRESH", "LIVE_CALLS_DISABLED"]
AdmissionMode: TypeAlias = Literal["cache_hit", "new", "refresh"]
Outcome: TypeAlias = Literal["fundable", "fundable_with_coapplicant", "not_fundable", "review"]
Confidence: TypeAlias = Literal["high", "medium", "low"]
Fit: TypeAlias = Literal["strong", "possible", "weak"]
Status: TypeAlias = Literal["ok", "pan_invalid", "refused"]
StateUpdate: TypeAlias = dict[str, Any]  # node return type
PINCODE_RE_STR: Final[str] = r"^[1-9][0-9]{5}$"  # adapters + gate filter

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class Identity(_Frozen):
    pan_valid: bool                       # success and status=='valid' and category=='person'
    status: str | None                    # raw data.status
    category: str | None                  # raw data.category
    full_name: str | None                 # data.full_name; sent as `name` on bureau calls
    gender: Literal["male", "female"] | None   # M->male, F->female, else None
    dob: date | None                      # from data.dob YYYY-MM-DD; NEVER from bureau holder rows
    aadhaar_linked: bool | None           # data.aadhaar_linked
    pan_pincode: str | None               # data.address.zip iff re.fullmatch(PINCODE_RE_STR)
    surepass_client_id: str | None        # data.client_id (audit). NOTE: masked_aadhaar is deliberately NOT modeled.

class BureauScore(_Frozen):
    band: Band
    score: int | None                     # internal only; never serialized into any prompt or LcView
    is_ntc: bool                          # ASSUMPTION branch: success==false or empty/unparseable credit_score
    report_date: date | None              # CreditProfileHeader.ReportDate YYYYMMDD

class UnreportedEmiAccount(_Frozen):
    balance: float                        # Current_Balance
    tenure_months: int | None             # Repayment_Tenure or Terms_Duration ('0'/''/zero-padded '024' handled by adapter); None = unknown
    rate_annual_pct: float | None         # Rate_of_Interest ('34.900' or '') -> float | None
    account_type: str                     # raw Account_Type code string ('05', '10', ...)

class BureauReport(_Frozen):
    band: Band
    score: int | None
    is_ntc: bool
    report_date: date | None
    existing_emi_reported: float          # sum Scheduled_Monthly_Payment_Amount over open accts where non-empty (0.0 on the real probe file)
    existing_emi_floor: bool              # any open account with empty EMI
    open_accounts_missing_emi: list[UnreportedEmiAccount]
    total_outstanding: float              # sum Current_Balance over open
    past_due_total: float                 # sum Amount_Past_Due over ALL accounts
    max_dpd_24m: int                      # max CAIS_Account_History[].Days_Past_Due within last 24 months
    written_off_or_settled: int           # count of non-empty Written_off_Settled_Status
    enquiries_180d: int                   # TotalCAPS_Summary.TotalCAPSLast180Days
    bureau_pincodes: list[str]            # ZIP_Postal_Code_non_normalized, Date_Reported within 365d, filtered to PINCODE_RE_STR, deduped+sorted

class HardCutoffs(_Frozen):
    min_band: Band | None = None
    max_foir: float | None = None
    min_income: float | None = None
    min_age: int | None = None
    max_age_at_maturity: int | None = None
    max_tenure_months: int | None = None
    annual_rate_pct: float | None = None

class GatedLender(_Frozen):
    id: str
    name: str
    criteria: dict[str, Any]              # free-form lenders.criteria jsonb; serialized into the rank prompt
    hard_cutoffs: HardCutoffs
    matched_pincodes: list[str]

class Applicant(_Frozen):
    band: Band
    monthly_income: float | None
    existing_emi: float | None            # the figure FOIR uses: full tier = max(declared, reported+estimated); quick = declared or None
    existing_emi_floor: bool
    age_years: int | None                 # from Identity.dob only
    foir: float | None                    # existing_emi / monthly_income when both known and income > 0
    income_declared: bool

class LenderAssessment(_Frozen):
    lender_id: str
    cutoffs_passed: bool
    failed_cutoffs: list[str]             # cutoff field names, e.g. ['max_foir']
    unknown_facts: list[str]              # facts unknown so cutoff not applied (lowers confidence, never fails)
    loan_ceiling: float | None            # rounded to nearest 1000; None when income unknown

class Affordability(_Frozen):
    emi_reported: float | None            # bureau-reported portion (full tier) else None
    emi_estimated: float                  # sum of 8.3 estimates over open_accounts_missing_emi (0.0 on quick)
    emi_declared: float | None            # LC-declared
    existing_emi_total: float | None      # == Applicant.existing_emi
    emi_floor: bool
    foir: float | None
    assessments: list[LenderAssessment]

class RankedLender(_Frozen):
    lender_id: str
    fit: Fit
    why: Annotated[str, StringConstraints(max_length=300)]

class RankingProposal(BaseModel):         # LLM structured-output schema for rank_lenders (PRD calls this 'Ranking(ranked=...)')
    model_config = ConfigDict(extra="forbid")
    ranked: list[RankedLender]

class DroppedLender(_Frozen):
    lender_id: str
    reason: str                           # 'not in gated lender list' | 'failed hard cutoffs: <names>' | 'duplicate' | 'llm_error: <msg>'

class Ranking(_Frozen):                   # post-code-verification state model (PRD 7.2 'accepted + dropped')
    accepted: list[RankedLender]
    dropped: list[DroppedLender]

TalkingPoint: TypeAlias = Annotated[str, StringConstraints(max_length=200)]
class FinalRecommendation(BaseModel):     # LLM structured-output schema for finalize — EXACTLY PRD 7.3
    model_config = ConfigDict(extra="forbid")
    outcome: Outcome
    confidence: Confidence
    indicative_loan_min: float | None
    indicative_loan_max: float | None
    top_lender_id: str | None
    alternate_lender_ids: list[str] = Field(default_factory=list, max_length=2)
    coapplicant_suggested: bool
    lc_talking_points: list[TalkingPoint] = Field(default_factory=list, max_length=4)
    confirm_with_learner: list[str] = Field(default_factory=list, max_length=3)
    reasoning_internal: str = Field(default="", max_length=2000)

class LoanRange(_Frozen):
    min: float
    max: float
class LenderRef(_Frozen):
    id: str
    name: str
class LcView(BaseModel):                  # EXACTLY PRD 9.3; the only object the API returns
    model_config = ConfigDict(extra="forbid")
    check_id: str
    kind: Kind
    status: Status
    refusal_code: RefusalCode | None = None
    outcome: Outcome | None = None
    confidence: Confidence | None = None
    indicative_loan_range: LoanRange | None = None
    top_lender: LenderRef | None = None
    alternates: list[LenderRef] = Field(default_factory=list, max_length=2)
    coapplicant_suggested: bool | None = None
    talking_points: list[str] = Field(default_factory=list)
    confirm_with_learner: list[str] = Field(default_factory=list)
    pan_aadhaar_link_issue: bool | None = None   # True ONLY when Identity.aadhaar_linked is False, else None
    from_cache: bool = False
    cached_at: datetime | None = None
    refresh_available_at: datetime | None = None
    quick_checks_remaining_today: int | None = None  # None for kind='full'

class CheckState(TypedDict, total=False):
    # request (set by main.py before invoke)
    kind: Kind
    check_id: str
    lc_id: str
    lead_id: str
    opportunity_id: str | None
    opportunity_stage: str | None
    name: str
    mobile: str
    pan: str
    pan_hash: str
    monthly_income: float | None
    existing_emi_declared: float | None
    refresh: bool
    consent: bool                          # ADDED: admission's CONSENT_MISSING guard input
    # admission
    admission: AdmissionMode
    counts_against_cap: bool
    billed: bool                           # expected at admission; recomputed in persist as pan_called or bureau_called
    counted_against_cap: bool              # ADDED: final value computed in persist (needed by finish_trace metadata)
    refusal_code: RefusalCode | None
    consent_id: str | None                 # ADDED
    quick_checks_used_today: int | None    # ADDED: from AdmissionDecision; redact computes remaining
    cache_fetched_at: datetime | None      # ADDED: LcView.cached_at
    cache_expires_at: datetime | None      # ADDED: LcView.refresh_available_at
    # facts (internal only)
    identity: Identity | None
    score: BureauScore | None
    report: BureauReport | None
    gated_lenders: list[GatedLender]
    applicant: Applicant | None
    affordability: Affordability | None
    ranking: Ranking | None
    final: FinalRecommendation | None
    raw_payloads: dict[str, Any]           # ADDED: {'pan': <raw envelope>, 'score'|'report': <raw envelope>}; written to pan_cache.payload by the pull node; STRIPPED at persist; never rendered into any prompt
    # retry machinery (ADDED)
    rank_attempts: int                     # LLM calls made in rank_lenders
    finalize_attempts: int                 # LLM calls made in finalize
    rank_errors: list[str]                 # replaced each attempt; fed back into the next rank user message
    finalize_errors: list[str]             # replaced each attempt; fed back into the next finalize user message
    pan_called: bool                       # ADDED: a live PAN Comprehensive call happened
    bureau_called: bool                    # ADDED: a live Experian call happened
    # output
    lc_view: LcView | None
    errors: Annotated[list[str], operator.add]
    transcript: Annotated[list[str], operator.add]   # ADDED: node names in execution order -> checks.internal
    langsmith_run_id: str | None           # ADDED: str(root run_id) when tracing enabled, else None


### eligibility_agent/graph/build.py
_Dependency-injection seam (GraphDeps, StructuredLLM protocol, make_structured_llm), build_graph(kind) with every add_node/add_edge/add_conditional_edges call and router, retry-budget constants, and the Postgres checkpointer factory with Supabase caveats._

from __future__ import annotations
import functools
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Final, Literal, Protocol, TypeAlias, TypeVar
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from eligibility_agent.db.repo import Repo                      # Protocol (db section)
from eligibility_agent.tools.surepass import SurePassClientProtocol  # Protocol (tools section)
from eligibility_agent.graph import nodes
from eligibility_agent.graph.prompts import PromptBundle
from eligibility_agent.graph.state import CheckState, FinalRecommendation, Kind, RankingProposal

# ---- retry budgets (PRD 7.3) ----
RANK_MAX_ATTEMPTS: Final[int] = 2      # 1 initial + 1 retry ('retry once' in the rank node contract)
FINALIZE_MAX_ATTEMPTS: Final[int] = 3  # 1 initial + 2 retries ('after 2 retries go to fallback_finalize')
RECURSION_LIMIT: Final[int] = 40       # worst path is 12 node executions; 40 gives headroom
NODE_NAMES: Final[tuple[str, ...]] = ("admission", "persist_refusal", "validate_pan", "pull_score", "pull_report", "gate_lenders", "affordability", "rank_lenders", "finalize", "fallback_finalize", "redact", "persist")

# ---- the LLM seam: ONE generic protocol, two aliases; FakeLLM in tests implements the same __call__ ----
T_co = TypeVar("T_co", bound=BaseModel, covariant=True)
class StructuredLLM(Protocol[T_co]):
    def __call__(self, messages: Sequence[BaseMessage], *, config: RunnableConfig | None = None) -> T_co: ...
RankLLM: TypeAlias = StructuredLLM[RankingProposal]
FinalizeLLM: TypeAlias = StructuredLLM[FinalRecommendation]

TModel = TypeVar("TModel", bound=BaseModel)
def make_structured_llm(schema: type[TModel], *, model: str, timeout: float = 30.0) -> StructuredLLM[TModel]:
    """Production factory. method='function_calling' deliberately (NOT 'json_schema'): OpenAI strict json_schema
    rejects maxLength/maxItems keywords that FinalRecommendation carries; with function_calling the constraints are
    enforced by Pydantic at parse time and violations surface as exceptions that the node retry loop absorbs."""
    chat = ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=1)
    runnable = chat.with_structured_output(schema, method="function_calling")
    def _call(messages: Sequence[BaseMessage], *, config: RunnableConfig | None = None) -> TModel:
        out = runnable.invoke(list(messages), config=config)  # config threads the trace context -> LLM child run nests under the node run
        if not isinstance(out, schema):
            raise TypeError(f"structured output returned {type(out).__name__}, expected {schema.__name__}")
        return out
    return _call

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

@dataclass(frozen=True, slots=True)
class GraphDeps:
    """Everything a node may touch beyond state. Built once in main.py lifespan (prod) or per-test (fakes)."""
    repo: Repo
    surepass: SurePassClientProtocol
    rank_llm: RankLLM
    finalize_llm: FinalizeLLM
    prompts: PromptBundle
    now: Callable[[], datetime] = field(default=_utcnow)   # MUST return tz-aware UTC; tests inject a fixed clock

# ---- routers (pure functions over state; no side effects, no deps) ----
def route_admission(state: CheckState) -> Literal["refused", "admitted"]:
    return "refused" if state.get("refusal_code") else "admitted"

def route_pan(state: CheckState) -> Literal["invalid", "valid"]:
    ident = state.get("identity")
    return "valid" if ident is not None and ident.pan_valid else "invalid"

def route_rank(state: CheckState) -> Literal["retry", "proceed"]:
    gated = state.get("gated_lenders") or []
    if not gated:
        return "proceed"                                  # nothing to rank; finalize handles the empty case
    ranking = state.get("ranking")
    invalid = ranking is None or len(ranking.accepted) == 0
    if invalid and state.get("rank_attempts", 0) < RANK_MAX_ATTEMPTS:
        return "retry"
    return "proceed"

def route_finalize(state: CheckState) -> Literal["ok", "retry", "fallback"]:
    if state.get("final") is not None:
        return "ok"
    if state.get("finalize_attempts", 0) < FINALIZE_MAX_ATTEMPTS:
        return "retry"
    return "fallback"

# ---- build ----
GraphT: TypeAlias = CompiledStateGraph[CheckState, Any, CheckState, CheckState]  # adjust type args to the pinned langgraph 1.x minor if its generic arity differs

def build_graph(kind: Kind, deps: GraphDeps, *, checkpointer: Any | None = None) -> GraphT:
    """checkpointer: BaseCheckpointSaver | None. None in unit tests; InMemorySaver for resume tests; PostgresSaver in prod."""
    bind = lambda fn: functools.partial(fn, deps=deps)  # noqa: E731 — nodes are (state, config, *, deps); partial leaves (state, config), which LangGraph introspects correctly
    g: StateGraph[CheckState] = StateGraph(CheckState)
    g.add_node("admission", bind(nodes.admission))
    g.add_node("persist_refusal", bind(nodes.persist_refusal))
    g.add_node("validate_pan", bind(nodes.validate_pan))
    pull_name = "pull_score" if kind == "quick" else "pull_report"
    g.add_node(pull_name, bind(nodes.pull_score) if kind == "quick" else bind(nodes.pull_report))
    g.add_node("gate_lenders", bind(nodes.gate_lenders))
    g.add_node("affordability", bind(nodes.affordability))
    g.add_node("rank_lenders", bind(nodes.rank_lenders))
    g.add_node("finalize", bind(nodes.finalize))
    g.add_node("fallback_finalize", bind(nodes.fallback_finalize))
    g.add_node("redact", bind(nodes.redact))
    g.add_node("persist", bind(nodes.persist))

    g.add_edge(START, "admission")
    g.add_conditional_edges("admission", route_admission, {"refused": "persist_refusal", "admitted": "validate_pan"})
    g.add_edge("persist_refusal", END)
    g.add_conditional_edges("validate_pan", route_pan, {"invalid": "redact", "valid": pull_name})  # pan_invalid: redact -> persist -> END, nothing consumed against the cap
    g.add_edge(pull_name, "gate_lenders")
    g.add_edge("gate_lenders", "affordability")
    g.add_edge("affordability", "rank_lenders")
    g.add_conditional_edges("rank_lenders", route_rank, {"retry": "rank_lenders", "proceed": "finalize"})
    g.add_conditional_edges("finalize", route_finalize, {"ok": "redact", "retry": "finalize", "fallback": "fallback_finalize"})
    g.add_edge("fallback_finalize", "redact")
    g.add_edge("redact", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=checkpointer)

# ---- checkpointer factory (prod only; never called from tests) ----
# from langgraph.checkpoint.postgres import PostgresSaver; from psycopg_pool import ConnectionPool; from psycopg.rows import dict_row
PG_POOL_KWARGS: Final[dict[str, Any]] = {"autocommit": True, "prepare_threshold": None, "row_factory": None}  # row_factory set to dict_row at runtime (import locally to keep psycopg optional in tests)

def make_postgres_checkpointer(db_url: str) -> tuple[Any, Any]:
    """Returns (PostgresSaver, ConnectionPool). Caller (main.py lifespan) owns pool.close().
    def make_postgres_checkpointer(db_url):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
        pool = ConnectionPool(conninfo=db_url, min_size=1, max_size=5, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row})
        saver = PostgresSaver(pool)
        saver.setup()   # idempotent migrations; REQUIRES autocommit=True
        return saver, pool
    Supabase caveats (verified against langgraph-checkpoint-postgres README + issue #6705):
    - Build the pool manually. PostgresSaver.from_conn_string() hardcodes prepare_threshold=0, which breaks behind
      poolers ('prepared statement _pg3_n does not exist'); prepare_threshold=None disables prepared statements.
    - Use the Supabase SESSION pooler URI (postgresql://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require)
      or the direct connection (db.<ref>.supabase.co:5432 — IPv6-only without the IPv4 add-on). NEVER the transaction
      pooler on :6543 — setup() migrations and pipeline mode do not survive transaction pooling.
    - Isolate the checkpoint tables from PostgREST: checkpoint blobs contain the raw PAN and raw_payloads via CheckState.
      README instructs: CREATE SCHEMA IF NOT EXISTS checkpointer; and append options=-csearch_path%3Dcheckpointer to
      CHECKPOINTER_DB_URL so checkpoints/checkpoint_blobs/checkpoint_writes/checkpoint_migrations land outside the
      PostgREST-exposed 'public' schema.
    - thread_id == check_id: config={'configurable': {'thread_id': check_id}} (set by tracing.invoke_config). Each check
      is a fresh uuid4 thread, so an interrupted run is inspectable in LangSmith and resumable with graph.invoke(None, same_config).
    """
    raise NotImplementedError  # body exactly as the docstring shows

# main.py lifespan contract (API section implements):
#   deps = GraphDeps(repo=SupabaseRepo(...), surepass=SurePassClient(settings) , 
#                    rank_llm=make_structured_llm(RankingProposal, model=settings.openai_model),
#                    finalize_llm=make_structured_llm(FinalRecommendation, model=settings.openai_model),
#                    prompts=load_prompts(settings))
#   saver, pool = make_postgres_checkpointer(settings.checkpointer_db_url) if settings.checkpointer_db_url else (None, None)
#   app.state.graph_quick = build_graph("quick", deps, checkpointer=saver)
#   app.state.graph_full  = build_graph("full", deps, checkpointer=saver)
#   yield; pool.close() if pool else None
# Routes run graph.invoke in a threadpool (all deps are sync).


### eligibility_agent/graph/nodes.py
_All eleven node functions. Uniform signature def node(state: CheckState, config: RunnableConfig, *, deps: GraphDeps) -> StateUpdate; every node also returns {'transcript': ['<node_name>']}. Contains validate_final (pure) and the rank verification logic._

UNIFORM SIGNATURE (mypy --strict):
  def admission(state: CheckState, config: RunnableConfig, *, deps: GraphDeps) -> StateUpdate: ...
  (same shape for every node; config is passed so LLM calls nest in the trace; deps bound by functools.partial in build_graph)

--- admission --------------------------------------------------------------
reads: kind, consent, lc_id, lead_id, opportunity_id, opportunity_stage, pan_hash, refresh
side effects: (1) if not state.get('consent'): return refusal CONSENT_MISSING WITHOUT inserting a consent row (defensive; API schema already requires consent==True). (2) consent_id = deps.repo.insert_consent(lc_id=..., lead_id=..., opportunity_id=..., pan_hash=..., channel='verbal', recorded_at=deps.now()). (3) decision = policy.guards.admit_quick_check(repo=deps.repo, lc_id=..., pan_hash=..., refresh=state['refresh'], now=deps.now(), live_enabled=deps.surepass.live_enabled) for kind=='quick', else policy.guards.admit_full_check(repo=..., opportunity_id=state['opportunity_id'] (non-null, API-validated), opportunity_stage=state['opportunity_stage'], pan_hash=..., refresh=..., now=..., live_enabled=...).
writes: {'consent_id': cid, 'admission': decision.admission, 'billed': decision.billed_expected, 'counts_against_cap': decision.counts_against_cap, 'refusal_code': decision.refusal_code, 'quick_checks_used_today': decision.quick_checks_used_today, 'cache_fetched_at': decision.cache_fetched_at, 'cache_expires_at': decision.cache_expires_at, 'transcript': ['admission']}
notes: cache_hit => billed_expected False, counts False. new/refresh => billed_expected True; counts True only for new quick. refusal_code in RefusalCode. LIVE_CALLS_DISABLED comes from the guard when a billed call would be needed and live_enabled is False (the client's LiveCallsDisabled exception is belt-and-braces).

--- persist_refusal --------------------------------------------------------
reads: check_id, kind, lc_id, lead_id, opportunity_id, pan_hash, refusal_code, consent_id, quick_checks_used_today, langsmith_run_id, transcript, refresh
logic: remaining = None if kind=='full' else max(0, DAILY_QUICK_CAP - (quick_checks_used_today or 0)); view = output.lc_view.build_refusal_view(check_id=..., kind=..., refusal_code=..., quick_checks_remaining_today=remaining)  # one-line talking_points from REFUSAL_MESSAGES[code]
side effects: deps.repo.save_check(CheckRecord(id=check_id, kind=..., lc_id=..., lead_id=..., opportunity_id=..., pan_hash=..., status='refused', refusal_code=..., billed=False, counted_against_cap=False, is_refresh=bool(state.get('refresh')), consent_id=..., lc_view=view.model_dump(mode='json'), internal=_internal_json(state), langsmith_run_id=state.get('langsmith_run_id'), created_at=deps.now()))
writes: {'lc_view': view, 'counted_against_cap': False, 'billed': False, 'transcript': ['persist_refusal']}

--- validate_pan -----------------------------------------------------------
reads: admission, pan, pan_hash, kind, check_id
branch cache_hit: row = deps.repo.get_cache(pan_hash=..., kind=state['kind']); if row is None: raise RuntimeError('cache row vanished between admission and validate_pan') (rare race; run fails, API maps to 502). identity = Identity.model_validate(row.derived['identity']); writes {'identity': identity, 'cache_fetched_at': row.fetched_at, 'cache_expires_at': row.expires_at, 'transcript': [...]}
branch new/refresh: resp = deps.surepass.pan_comprehensive(state['pan'])  # client posts {id_number, get_address: true, get_contact: false}; raises LiveCallsDisabled/SurePassError (propagate — API maps to 502 and marks the check failed)
  deps.repo.log_surepass_call(check_id=state['check_id'], endpoint='/api/v1/pan/pan-comprehensive', surepass_client_id=resp.client_id, http_status=resp.http_status, success=resp.success, message_code=resp.message_code, called_at=deps.now())
  identity = tools.adapters.adapt_pan_comprehensive(resp)   # gate: success and data.status=='valid' and data.category=='person'
  writes {'identity': identity, 'pan_called': True, 'raw_payloads': {'pan': resp.raw}, 'transcript': ['validate_pan']}
router route_pan then sends pan_valid==False to redact (status pan_invalid; nothing counted).

--- pull_score (quick graph only) -------------------------------------------
reads: admission, identity, name, mobile, pan, pan_hash, raw_payloads, check_id
branch cache_hit: row = deps.repo.get_cache(pan_hash, kind='quick'); score = BureauScore.model_validate(row.derived['score']); writes {'score': score, 'transcript': ['pull_score']}
branch new/refresh:
  resp = deps.surepass.experian_score(name=(identity.full_name or state['name']), mobile=state['mobile'], pan=state['pan'])  # client body adds consent:'Y'
  deps.repo.log_surepass_call(check_id=..., endpoint='/api/v1/credit-report-experian/score', surepass_client_id=resp.client_id, http_status=resp.http_status, success=resp.success, message_code=resp.message_code, called_at=deps.now())
  score = tools.adapters.adapt_experian_score(resp)         # NTC ASSUMPTION branch lives in the adapter
  now = deps.now(); expires = now + timedelta(days=policy.guards.CACHE_TTL_DAYS)
  deps.repo.upsert_cache(CacheRow(pan_hash=state['pan_hash'], kind='quick', payload={'pan': state['raw_payloads']['pan'], 'score': resp.raw}, derived={'identity': identity.model_dump(mode='json'), 'score': score.model_dump(mode='json')}, is_ntc=score.is_ntc, fetched_at=now, expires_at=expires))
  writes {'score': score, 'bureau_called': True, 'raw_payloads': {**state['raw_payloads'], 'score': resp.raw}, 'cache_fetched_at': now, 'cache_expires_at': expires, 'transcript': ['pull_score']}

--- pull_report (full graph only) -------------------------------------------
identical shape with deps.surepass.experian_report(...), endpoint '/api/v1/credit-report-experian/fetch-report', adapter adapt_experian_report, derived={'identity':..., 'report': report.model_dump(mode='json')}, is_ntc=report.is_ntc, state key 'report'.

--- gate_lenders -------------------------------------------------------------
reads: kind, identity, report
logic: pins: set[str] = set(); if identity.pan_pincode: pins.add(identity.pan_pincode); if kind=='full' and report: pins |= set(report.bureau_pincodes)   # junk already filtered by the adapter to ^[1-9][0-9]{5}$
  rows = deps.repo.get_active_lenders_for_pincodes(sorted(pins)) if pins else []
  gated = [GatedLender(id=r.id, name=r.name, criteria=r.criteria, hard_cutoffs=HardCutoffs.model_validate(r.hard_cutoffs), matched_pincodes=r.matched_pincodes) for r in rows]
writes: {'gated_lenders': gated, 'transcript': ['gate_lenders']}   # empty list allowed -> lowers confidence downstream

--- affordability ------------------------------------------------------------
reads: kind, identity, score, report, monthly_income, existing_emi_declared, gated_lenders
pseudocode:
  band = report.band if kind=='full' else score.band
  declared = state.get('existing_emi_declared')
  if kind == 'full':
      est = sum(tools.lenders.estimate_monthly_emi(balance=a.balance, tenure_months=a.tenure_months, annual_rate_pct=a.rate_annual_pct) for a in report.open_accounts_missing_emi)   # 8.3: reducing-balance; defaults 36 months / 14% p.a. live in tools
      bureau_emi = report.existing_emi_reported + est
      existing_emi = max(declared, bureau_emi) if declared is not None else bureau_emi   # higher of declared vs bureau-derived
      floor = report.existing_emi_floor or report.is_ntc                                  # 'nothing is known' covers NTC
      emi_reported: float | None = report.existing_emi_reported
  else:
      est = 0.0; existing_emi = declared; floor = declared is None; emi_reported = None
  income = state.get('monthly_income'); age = _age_years(identity.dob, deps.now().date()) if identity.dob else None
  foir = (existing_emi / income) if (existing_emi is not None and income and income > 0) else None
  applicant = Applicant(band=band, monthly_income=income, existing_emi=existing_emi, existing_emi_floor=floor, age_years=age, foir=foir, income_declared=income is not None)
  assessments = [LenderAssessment(lender_id=L.id, cutoffs_passed=res.passed, failed_cutoffs=res.failed, unknown_facts=res.unknown, loan_ceiling=tools.lenders.loan_ceiling(applicant, L.hard_cutoffs)) for L in gated_lenders for res in [tools.lenders.check_hard_cutoffs(L.hard_cutoffs, applicant)]]
writes: {'applicant': applicant, 'affordability': Affordability(emi_reported=emi_reported, emi_estimated=est, emi_declared=declared, existing_emi_total=existing_emi, emi_floor=floor, foir=foir, assessments=assessments), 'transcript': ['affordability']}
_age_years(dob, today) = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

--- rank_lenders (LLM node 1 of 2) --------------------------------------------
reads: kind, applicant, affordability, gated_lenders, rank_attempts, rank_errors
logic:
  gated = state['gated_lenders']
  if not gated: return {'ranking': Ranking(accepted=[], dropped=[]), 'transcript': ['rank_lenders']}   # skip the LLM entirely
  system = deps.prompts.rank_system(state['kind']); user = prompts.render_rank_user(applicant=..., affordability=..., gated_lenders=gated, prior_errors=state.get('rank_errors', []))
  attempts = state.get('rank_attempts', 0) + 1
  try: proposal = deps.rank_llm([SystemMessage(content=system), HumanMessage(content=user)], config=config)
  except Exception as e:  # LLM/network/parse failure counts as a failed attempt; router retries then proceeds
      if attempts >= RANK_MAX_ATTEMPTS:
          return {'ranking': Ranking(accepted=[], dropped=[DroppedLender(lender_id=L.id, reason=f'llm_error: {e}') for L in gated]), 'rank_attempts': attempts, 'rank_errors': [str(e)], 'errors': [f'rank_lenders: {e}'], 'transcript': ['rank_lenders']}
      return {'ranking': None, 'rank_attempts': attempts, 'rank_errors': [str(e)], 'errors': [f'rank_lenders: {e}'], 'transcript': ['rank_lenders']}
  # deterministic verification (the code, not the LLM, decides):
  gated_ids = {L.id for L in gated}
  failed_by_id = {a.lender_id: a.failed_cutoffs for a in state['affordability'].assessments if not a.cutoffs_passed}
  accepted: list[RankedLender] = []; dropped: list[DroppedLender] = []; seen: set[str] = set()
  for entry in proposal.ranked:
      if entry.lender_id in seen: dropped.append(DroppedLender(lender_id=entry.lender_id, reason='duplicate')); continue
      seen.add(entry.lender_id)
      if entry.lender_id not in gated_ids: dropped.append(DroppedLender(lender_id=entry.lender_id, reason='not in gated lender list')); continue
      if entry.lender_id in failed_by_id: dropped.append(DroppedLender(lender_id=entry.lender_id, reason='failed hard cutoffs: ' + ', '.join(failed_by_id[entry.lender_id]))); continue
      accepted.append(entry)
  errs = [f"{d.lender_id}: {d.reason}" for d in dropped] if (not accepted and gated) else []
  return {'ranking': Ranking(accepted=accepted, dropped=dropped), 'rank_attempts': attempts, 'rank_errors': errs, 'transcript': ['rank_lenders']}
(route_rank retries while accepted empty, gated non-empty and rank_attempts < RANK_MAX_ATTEMPTS; drop reasons are appended to the next user message via rank_errors.)

--- finalize (LLM node 2 of 2) -------------------------------------------------
reads: kind, applicant, affordability, ranking, gated_lenders, finalize_attempts, finalize_errors
logic:
  system = deps.prompts.finalize_system(state['kind']); user = prompts.render_finalize_user(kind=..., applicant=..., affordability=..., ranking=state.get('ranking') or Ranking(accepted=[], dropped=[]), gated_lenders=..., prior_errors=state.get('finalize_errors', []))
  attempts = state.get('finalize_attempts', 0) + 1
  try: cand = deps.finalize_llm([SystemMessage(system), HumanMessage(user)], config=config)
  except Exception as e: return {'final': None, 'finalize_attempts': attempts, 'finalize_errors': [str(e)], 'errors': [f'finalize: {e}'], 'transcript': ['finalize']}
  errs = validate_final(cand, ranking=..., applicant=state['applicant'], affordability=state['affordability'])
  if errs: return {'final': None, 'finalize_attempts': attempts, 'finalize_errors': errs, 'transcript': ['finalize']}
  return {'final': cand, 'finalize_attempts': attempts, 'finalize_errors': [], 'transcript': ['finalize']}

validate_final(cand: FinalRecommendation, *, ranking: Ranking, applicant: Applicant, affordability: Affordability) -> list[str]  (pure, unit-testable):
  accepted_ids = [r.lender_id for r in ranking.accepted]
  1. cand.top_lender_id not in accepted_ids (when not None) -> "top_lender_id '<id>' is not in the accepted ranking"
  2. any alt not in accepted_ids -> "alternate_lender_id '<id>' is not in the accepted ranking"; alt == top -> "alternate_lender_id '<id>' duplicates top_lender_id"; len(set(alts)) != len(alts) -> 'duplicate alternate_lender_ids'
  3. min/max both set and min > max -> 'indicative_loan_min (<v>) > indicative_loan_max (<v>)'; either negative -> 'loan amounts must be >= 0'
  4. cand.outcome in ('fundable','fundable_with_coapplicant') and cand.top_lender_id is None and not cand.coapplicant_suggested -> "outcome '<o>' requires top_lender_id unless coapplicant_suggested"
  5. applicant.band in ('NTC','<650') and cand.outcome == 'fundable' and not cand.coapplicant_suggested -> "band '<band>' cannot be plain 'fundable' without coapplicant_suggested"   # additive code enforcement of the outcome_matches_band invariant
  6. ceiling = loan_ceiling of top lender from affordability.assessments; if ceiling is not None and cand.indicative_loan_max is not None and cand.indicative_loan_max > ceiling -> 'indicative_loan_max (<v>) exceeds top lender ceiling (<c>)'   # additive learner protection

--- fallback_finalize ----------------------------------------------------------
reads: finalize_attempts, finalize_errors
writes: {'final': FinalRecommendation(outcome='review', confidence='low', indicative_loan_min=None, indicative_loan_max=None, top_lender_id=None, alternate_lender_ids=[], coapplicant_suggested=False, lc_talking_points=[prompts.FALLBACK_LC_MESSAGE], confirm_with_learner=[], reasoning_internal=f"fallback_finalize after {attempts} invalid attempts; last errors: {'; '.join(errs)[:1500]}"), 'transcript': ['fallback_finalize']}
No side effects, no LLM.

--- redact ----------------------------------------------------------------------
reads: check_id, kind, identity, final, gated_lenders, admission, counts_against_cap, bureau_called, quick_checks_used_today, cache_fetched_at, cache_expires_at
logic:
  status: Status = 'pan_invalid' if (state.get('identity') and not state['identity'].pan_valid) else 'ok'
  will_count = bool(state.get('counts_against_cap')) and bool(state.get('bureau_called')) and status != 'pan_invalid'
  remaining = None if state['kind']=='full' else max(0, DAILY_QUICK_CAP - (state.get('quick_checks_used_today') or 0) - (1 if will_count else 0))
  view = output.lc_view.build_lc_view(check_id=..., kind=..., status=status, refusal_code=None, final=state.get('final'), gated_lenders=state.get('gated_lenders', []), identity=state.get('identity'), from_cache=state.get('admission')=='cache_hit', cached_at=state.get('cache_fetched_at'), refresh_available_at=state.get('cache_expires_at'), quick_checks_remaining_today=remaining)
  # build_lc_view (output section) maps lender ids -> LenderRef via gated_lenders, builds indicative_loan_range, scrubs every LC-facing line with the (?i)\b(cibil|experian|equifax|crif|bureau)\b regex and the standalone-300..899 rule, and sets pan_aadhaar_link_issue=True ONLY when identity.aadhaar_linked is False
writes: {'lc_view': view, 'transcript': ['redact']}

--- persist ----------------------------------------------------------------------
reads: everything except raw request PII it must drop
logic:
  billed = bool(state.get('pan_called')) or bool(state.get('bureau_called'))        # 'any SurePass call made'
  status = state['lc_view'].status
  counted = bool(state.get('counts_against_cap')) and bool(state.get('bureau_called')) and status != 'pan_invalid'
  internal = _internal_json(state)  # keys: identity, score, report, applicant, affordability, ranking, final (all model_dump(mode='json') or None), admission, errors, transcript, rank_attempts, finalize_attempts, langsmith_run_id. EXCLUDES: pan, mobile, name, raw_payloads (code-review acceptance: no raw path to internal).
side effects: deps.repo.save_check(CheckRecord(id=check_id, kind=..., lc_id=..., lead_id=..., opportunity_id=..., pan_hash=..., status=status, refusal_code=None, billed=billed, counted_against_cap=counted, is_refresh=(state.get('admission')=='refresh'), consent_id=state.get('consent_id'), lc_view=state['lc_view'].model_dump(mode='json'), internal=internal, langsmith_run_id=state.get('langsmith_run_id'), created_at=deps.now()))
writes: {'billed': billed, 'counted_against_cap': counted, 'raw_payloads': {}, 'transcript': ['persist']}   # raw payloads cleared from the final state snapshot


### eligibility_agent/graph/prompts.py
_Pinned literal prompts (source of truth), tier addenda, deterministic user-message renderers with retry feedback, PromptBundle, and the LangSmith Hub pull-by-tag loader with repo-local fallback. Tests always take the local path._

PROMPTS_LOCAL_VERSION: Final[str] = '2026-10-08-a'   # bump on any text change; mirrored as the Hub commit tag
HUB_RANK: Final[str] = 'hero/eligibility-rank'; HUB_FINALIZE: Final[str] = 'hero/eligibility-finalize'

SYSTEM_CORE: Final[str] = (
"You are the financing eligibility agent for Hero Vired's sales team. A Learning Consultant (LC) is on a live call with a prospective learner and needs to know whether the learner can plausibly be financed for a programme, and through which lender. Your output is a recommendation only; no loan is applied for and no approval is promised.\n\n"
"Hard rules:\n"
"1. Rank only the lenders given in the input, by their lender_id. Never invent, rename, merge or add a lender. Never suggest overriding or working around the pincode gate. If no lender is listed, say options will be reviewed manually; do not propose one.\n"
"2. The LC must never learn the credit score, any score number, the credit bureau's name, or any raw report detail. The words Experian, CIBIL, Equifax, CRIF and bureau, and any three-digit score, must never appear in lc_talking_points or confirm_with_learner. Those fields use plain, band-driven language a salesperson can say aloud: 'strong credit profile', 'existing EMIs look high next to income', 'limited credit history'. Exact numbers, account facts and bureau references belong only in reasoning_internal.\n"
"3. Indian context: amounts are INR, income is monthly. FOIR means existing monthly EMI divided by monthly income; lower is better.\n"
"4. Prefer outcome fundable_with_coapplicant over not_fundable when the credit band is NTC or below 650, or when FOIR headroom is thin but a working parent or spouse would plausibly clear it.\n"
"5. Confidence: high when band, income and existing EMI are all known and at least one lender passes cleanly; medium when the EMI figure is a floor or estimate, or income is undeclared; low when no lender passed the gate or the facts conflict. At low confidence, ask the LC to confirm the missing or conflicting fact via confirm_with_learner.\n"
"6. Stay within the per-lender loan ceilings you are given; never quote an amount above them.")
# ~280 words; contains NO curly braces (Hub ChatPromptTemplate stores it variable-free; user messages are built in code, so criteria JSON braces never pass through template formatting).

RANK_TASK: Final[str] = ("Task: rank the listed lenders for this applicant. Return structured output with ranked = a list of {lender_id, fit, why}, ordered best first. fit is strong, possible or weak. why is one short sentence grounded in the lender's criteria text and the hard-cutoff results; it may mention the band in words but no score numbers and no bureau names. Do not include a lender whose hard cutoffs failed.")  # written brace-free in the real file: 'a list of entries with lender_id, fit and why'

FINALIZE_TASK: Final[str] = ("Task: write the FinalRecommendation object for the LC. Choose top_lender_id and up to two alternate_lender_ids only from the accepted ranking you are given, never from the dropped list. Set indicative_loan_min and indicative_loan_max as a round-number INR range within the top lender's loan ceiling; leave both null when no ceiling is known. lc_talking_points: at most 4 short sentences the LC can say aloud. confirm_with_learner: at most 3 short items. Put every number taken from the credit file, and all delinquency detail, only in reasoning_internal.")

TIER1_ADDENDUM: Final[str] = ("Quick check (Tier 1). Declared monthly income may be missing. If income was not given, give a band-driven view only: no FOIR claims and no loan range, and add 'Declared monthly income' to confirm_with_learner.")

TIER2_ADDENDUM: Final[str] = ("Full check (Tier 2). If the PAN pincode and the bureau pincodes differ, both were kept in the lender gate; add 'Confirm current pincode' to confirm_with_learner. Flag delinquency signals (amounts past due, DPD above 30 in the last 24 months, written-off or settled accounts) in reasoning_internal only, and let them lower confidence or push the outcome toward a co-applicant.")

FALLBACK_LC_MESSAGE: Final[str] = "The check could not be completed automatically. Please raise it for manual review."

@dataclass(frozen=True)
class PromptBundle:
    rank_system_quick: str; rank_system_full: str; finalize_system_quick: str; finalize_system_full: str
    version: str; source: Literal['local', 'hub']
    def rank_system(self, kind: Kind) -> str: return self.rank_system_quick if kind == 'quick' else self.rank_system_full
    def finalize_system(self, kind: Kind) -> str: return self.finalize_system_quick if kind == 'quick' else self.finalize_system_full

def _compose(core: str, task: str, kind: Kind) -> str:
    return core + '\n\n' + task + '\n\n' + (TIER1_ADDENDUM if kind == 'quick' else TIER2_ADDENDUM)

def local_prompts() -> PromptBundle:
    return PromptBundle(rank_system_quick=_compose(SYSTEM_CORE, RANK_TASK, 'quick'), rank_system_full=_compose(SYSTEM_CORE, RANK_TASK, 'full'), finalize_system_quick=_compose(SYSTEM_CORE, FINALIZE_TASK, 'quick'), finalize_system_full=_compose(SYSTEM_CORE, FINALIZE_TASK, 'full'), version=PROMPTS_LOCAL_VERSION, source='local')

def load_prompts(settings: Settings) -> PromptBundle:
    """PROMPTS_SOURCE=local (default, and always in tests/CI) -> local_prompts(), zero network.
    PROMPTS_SOURCE=hub -> pull the two SYSTEM prompts by tag and compose with the LOCAL task/addendum text:
        from langsmith import Client
        c = Client()
        rank_core = _extract_system_text(c.pull_prompt(f'{HUB_RANK}:{settings.prompts_tag}'))        # 'hero/eligibility-rank:prod' — verified pull-by-tag syntax
        fin_core  = _extract_system_text(c.pull_prompt(f'{HUB_FINALIZE}:{settings.prompts_tag}'))
        return PromptBundle(_compose2(rank_core,'quick'), _compose2(rank_core,'full'), _compose2(fin_core,'quick'), _compose2(fin_core,'full'), version=settings.prompts_tag, source='hub')
    where _compose2(core, kind) = core + addendum (the Hub commit stores core+task as one system message) and
    _extract_system_text(t: ChatPromptTemplate) -> str = t.messages[0].prompt.template (asserted to have no input_variables).
    ANY exception (network, auth, missing tag, unexpected shape) -> logger.warning(...); return local_prompts().  The repo pins
    settings.prompts_tag per release; the local literals are the committed snapshot of the same tag."""

def render_rank_user(*, applicant: Applicant, affordability: Affordability, gated_lenders: Sequence[GatedLender], prior_errors: Sequence[str]) -> str:
    """Deterministic; pure; unit-tested. Serializes ONLY: band, age, income presence (yes/no — never the amount), FOIR to 2dp,
    floor flag; per lender: id, name, criteria json.dumps(sort_keys=True, ensure_ascii=False), PASSED/FAILED + failed cutoff names,
    unknown facts, loan ceiling. Never the score, never pincodes, never raw report fields.
    Layout:
      Applicant profile:
      - credit band: {band}
      - age (years): {age_years | 'unknown'}
      - monthly income declared: {'yes' | 'no'}
      - FOIR (existing EMI / income): {f'{foir:.2f}' | 'unknown'}
      - existing EMI figure is a floor (some EMIs missing or estimated): {'yes' | 'no'}
      \n Lenders that passed the pincode gate - rank ONLY these lender_ids:
      {for L, assessment in zip-by-id:}
      - lender_id: {L.id} | name: {L.name}
        criteria: {json.dumps(L.criteria, sort_keys=True, ensure_ascii=False)}
        hard cutoffs: {'PASSED' | 'FAILED: ' + ', '.join(failed_cutoffs)}
        facts unknown for cutoffs: {', '.join(unknown_facts) | 'none'}
        loan ceiling (INR): {int(ceiling) | 'unknown'}
      {if prior_errors:}
      \n Your previous ranking was rejected:
      - {err}  (one line per entry)
      Return a corrected ranking using only lender_ids listed above whose hard cutoffs did not fail."""

def render_finalize_user(*, kind: Kind, applicant: Applicant, affordability: Affordability, ranking: Ranking, gated_lenders: Sequence[GatedLender], prior_errors: Sequence[str]) -> str:
    """Same applicant block as rank. Then:
      Accepted lender ranking (choose top_lender_id and up to 2 alternate_lender_ids ONLY from these):
      - {lender_id} '{name}' | fit: {fit} | why: {why} | loan ceiling (INR): {int(ceiling) | 'unknown'}
      Dropped during verification (do NOT name these anywhere in LC-facing text):
      - {lender_id}: {reason}          (omit section when empty)
      {if no gated lenders}: 'No lender passed the location gate. A co-applicant cannot change location eligibility: use outcome review or not_fundable, confidence low, top_lender_id null.'
      {elif accepted empty}: 'No lender survived verification. If failures were FOIR- or band-driven, consider fundable_with_coapplicant; otherwise review. confidence low, top_lender_id null.'
      Reminders: indicative range within the top lender ceiling or null; NTC or below-650 band is never plain fundable without coapplicant_suggested.
      {if prior_errors:}
      \n Your previous FinalRecommendation failed validation:
      - {err} (one line per entry)
      Return the full corrected object; change only what is needed to fix these errors."""


### eligibility_agent/graph/tracing.py
_LangSmith wiring: root run id generation, invoke-time RunnableConfig (thread_id, tags, metadata), and the post-run patch that adds admission:<mode>, billed and counted_against_cap. No-ops entirely when tracing is disabled (all tests)._

from __future__ import annotations
import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4
from langchain_core.runnables import RunnableConfig
from eligibility_agent.graph.state import Kind
logger = logging.getLogger(__name__)

# Env contract (config.py): settings.langsmith_enabled = (os.environ.get('LANGSMITH_TRACING','').lower() == 'true').
# When enabled the process also carries LANGSMITH_API_KEY and LANGSMITH_PROJECT=eligibility-agent; langchain-core picks
# these up automatically and traces every graph/node/LLM child run. Tests never set LANGSMITH_TRACING -> full no-op.

def new_run_id(enabled: bool) -> UUID | None:
    """Pre-generate the root run id. Verified: a run_id placed in the ROOT RunnableConfig becomes the LangSmith trace_id,
    so no post-hoc capture (collect_runs) is needed; main.py stores str(run_id) into state['langsmith_run_id'] and persist
    writes it onto the checks row even though the trace uploads asynchronously."""
    return uuid4() if enabled else None

def base_tags(kind: Kind, lc_id: str) -> list[str]:
    return [f"tier:{kind}", f"lc:{lc_id}"]

def invoke_config(*, kind: Kind, check_id: str, lc_id: str, pan_hash: str, run_id: UUID | None) -> RunnableConfig:
    """Used for EVERY graph invocation (traced or not): thread_id keys the Postgres checkpointer by check_id.
    Metadata deliberately excludes the raw PAN, name and mobile; pan_hash only (PRD 11)."""
    cfg: RunnableConfig = {
        "configurable": {"thread_id": check_id},
        "tags": base_tags(kind, lc_id),
        "metadata": {"check_id": check_id, "pan_hash": pan_hash, "tier": kind},
        "recursion_limit": 40,
    }
    if run_id is not None:
        cfg["run_id"] = run_id
    return cfg

def finish_trace(client: Any | None, *, run_id: UUID | None, kind: Kind, lc_id: str, check_id: str, pan_hash: str, final_state: Mapping[str, Any]) -> None:
    """client: langsmith.Client | None (None whenever tracing is off -> no-op). Called by main.py AFTER graph.invoke
    returns (success or refusal). billed / counted_against_cap / admission only exist post-run, so they are patched on:
        admission = final_state.get('admission') or 'refused'
        client.update_run(run_id,
            tags=base_tags(kind, lc_id) + [f'admission:{admission}'],          # update_run replaces tags -> resend base tags
            extra={'metadata': {'check_id': check_id, 'pan_hash': pan_hash, 'tier': kind,
                                'billed': bool(final_state.get('billed')),
                                'counted_against_cap': bool(final_state.get('counted_against_cap'))}})
    Wrapped in try/except Exception: logger.warning('langsmith update_run failed: %s', e) — tracing must never fail a check.
    (Client.update_run(run_id, *, tags=..., extra=...) signature verified against the LangSmith reference.)"""

# main.py usage (API section):
#   run_id = new_run_id(settings.langsmith_enabled)
#   cfg = invoke_config(kind=kind, check_id=check_id, lc_id=req.lc_id, pan_hash=pan_hash, run_id=run_id)
#   final_state = await run_in_threadpool(graph.invoke, initial_state, cfg)
#   finish_trace(app.state.langsmith_client, run_id=run_id, kind=kind, lc_id=req.lc_id, check_id=check_id, pan_hash=pan_hash, final_state=final_state)
#   return final_state['lc_view']


## Interfaces provided
- **name**: eligibility_agent/graph/state.py — all shared models | **definition**: Kind, Band, BAND_ORDER, RefusalCode, AdmissionMode, Outcome, Confidence, Fit, Status, StateUpdate, PINCODE_RE_STR; Pydantic models Identity, BureauScore, UnreportedEmiAccount, BureauReport, HardCutoffs, GatedLender, Applicant, LenderAssessment, Affordability, RankedLender, RankingProposal, DroppedLender, Ranking, FinalRecommendation, LoanRange, LenderRef, LcView; TypedDict CheckState — exact fields in the state.py file entry. Leaf module with no intra-package imports: policy/, tools/, output/, db/ and evals/ all import their types from here.
- **name**: build_graph | **definition**: def build_graph(kind: Kind, deps: GraphDeps, *, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph[CheckState, Any, CheckState, CheckState] — compiles the quick or full graph; only differences between kinds: which pull node is wired and which tier addendum the PromptBundle serves.
- **name**: GraphDeps + StructuredLLM protocol (the DI seam) | **definition**: @dataclass(frozen=True, slots=True) class GraphDeps: repo: Repo; surepass: SurePassClientProtocol; rank_llm: StructuredLLM[RankingProposal]; finalize_llm: StructuredLLM[FinalRecommendation]; prompts: PromptBundle; now: Callable[[], datetime] = _utcnow. class StructuredLLM(Protocol[T_co]): def __call__(self, messages: Sequence[BaseMessage], *, config: RunnableConfig | None = None) -> T_co. Factory: def make_structured_llm(schema: type[TModel], *, model: str, timeout: float = 30.0) -> StructuredLLM[TModel]. Tests build GraphDeps(repo=MemoryRepo(), surepass=FakeSurePassClient(...), rank_llm=FakeRankLLM(...), finalize_llm=FakeFinalizeLLM(...), prompts=local_prompts(), now=fixed_clock) — no monkeypatching anywhere.
- **name**: Retry/graph constants | **definition**: RANK_MAX_ATTEMPTS: Final[int] = 2; FINALIZE_MAX_ATTEMPTS: Final[int] = 3; RECURSION_LIMIT: Final[int] = 40; NODE_NAMES: tuple[str, ...] (for tests/evals asserting transcripts).
- **name**: pan_cache.derived jsonb contract (written and read by pull/validate nodes) | **definition**: kind='quick': {'identity': Identity.model_dump(mode='json'), 'score': BureauScore.model_dump(mode='json')}; kind='full': {'identity': ..., 'report': BureauReport.model_dump(mode='json')}. pan_cache.payload = {'pan': <raw envelope>, 'score'|'report': <raw envelope>}. Cache-hit reads: Identity.model_validate(derived['identity']) etc. DB section stores these opaquely.
- **name**: checks.internal jsonb contract (written by persist/persist_refusal) | **definition**: {'identity','score','report','applicant','affordability','ranking','final': model_dump(mode='json') or None, 'admission': str|None, 'errors': list[str], 'transcript': list[str], 'rank_attempts': int, 'finalize_attempts': int, 'langsmith_run_id': str|None}. Never contains pan, mobile, name or raw_payloads.
- **name**: graph/prompts.py API | **definition**: PromptBundle (frozen dataclass with rank_system(kind)/finalize_system(kind)); local_prompts() -> PromptBundle; load_prompts(settings: Settings) -> PromptBundle; render_rank_user(*, applicant, affordability, gated_lenders, prior_errors) -> str; render_finalize_user(*, kind, applicant, affordability, ranking, gated_lenders, prior_errors) -> str; FALLBACK_LC_MESSAGE: str; PROMPTS_LOCAL_VERSION: str; HUB_RANK='hero/eligibility-rank'; HUB_FINALIZE='hero/eligibility-finalize'.
- **name**: graph/tracing.py API | **definition**: new_run_id(enabled: bool) -> UUID | None; base_tags(kind, lc_id) -> list[str]; invoke_config(*, kind: Kind, check_id: str, lc_id: str, pan_hash: str, run_id: UUID | None) -> RunnableConfig (sets configurable.thread_id=check_id, tags ['tier:<kind>','lc:<lc_id>'], metadata {check_id, pan_hash, tier}, recursion_limit=40, run_id when tracing); finish_trace(client, *, run_id, kind, lc_id, check_id, pan_hash, final_state) -> None (patches admission tag + billed/counted metadata via Client.update_run, swallows all exceptions).
- **name**: make_postgres_checkpointer | **definition**: def make_postgres_checkpointer(db_url: str) -> tuple[PostgresSaver, ConnectionPool] — manual pool with autocommit=True, row_factory=dict_row, prepare_threshold=None; saver.setup() called inside; caller closes the pool at shutdown.
- **name**: Initial-state contract for main.py | **definition**: main.py must invoke with: {'kind','check_id','lc_id','lead_id','opportunity_id','opportunity_stage' (full),'name','mobile','pan' (upper),'pan_hash','monthly_income','existing_emi_declared','refresh','consent': True, 'rank_attempts':0,'finalize_attempts':0,'rank_errors':[],'finalize_errors':[],'pan_called':False,'bureau_called':False,'errors':[],'transcript':[],'langsmith_run_id': str(run_id) | None} and read final_state['lc_view'] (never any other key) for the HTTP response.

## Interfaces consumed
- **name**: db/repo.py — Repo protocol (DB section) | **expectation**: class Repo(Protocol) with EXACTLY these methods used by my nodes (mypy --strict): insert_consent(*, lc_id: str, lead_id: str, opportunity_id: str | None, pan_hash: str, channel: str, recorded_at: datetime) -> str; get_cache(*, pan_hash: str, kind: Literal['quick','full']) -> CacheRow | None; upsert_cache(row: CacheRow) -> None; log_surepass_call(*, check_id: str, endpoint: str, surepass_client_id: str | None, http_status: int, success: bool | None, message_code: str | None, called_at: datetime) -> None; save_check(record: CheckRecord) -> None (UPSERT by id — the API layer pre-inserts a pending row); quick_checks_used_today(*, lc_id: str) -> int (IST calendar day, counted_against_cap only); get_active_lenders_for_pincodes(pincodes: Sequence[str]) -> list[LenderRow] (DISTINCT on lender, active only, with matched_pincodes aggregated; empty input -> []). Dataclasses: CacheRow(pan_hash: str, kind: Literal['quick','full'], payload: dict[str, Any], derived: dict[str, Any], is_ntc: bool, fetched_at: datetime, expires_at: datetime); LenderRow(id: str, name: str, criteria: dict[str, Any], hard_cutoffs: dict[str, Any], matched_pincodes: list[str]); CheckRecord(id, kind, lc_id, lead_id, opportunity_id, pan_hash, status: str, refusal_code: str | None, billed: bool, counted_against_cap: bool, is_refresh: bool, consent_id: str | None, lc_view: dict[str, Any], internal: dict[str, Any], langsmith_run_id: str | None, created_at: datetime). MemoryRepo must implement all of it for tests.
- **name**: policy/guards.py — admission guards (policy section) | **expectation**: DAILY_QUICK_CAP: Final[int] = 5; CACHE_TTL_DAYS: Final[int] = 30; @dataclass(frozen=True) class AdmissionDecision(admission: Literal['cache_hit','new','refresh'] | None, refusal_code: RefusalCode | None, counts_against_cap: bool, billed_expected: bool, quick_checks_used_today: int | None, cache_fetched_at: datetime | None, cache_expires_at: datetime | None); def admit_quick_check(*, repo: Repo, lc_id: str, pan_hash: str, refresh: bool, now: datetime, live_enabled: bool) -> AdmissionDecision; def admit_full_check(*, repo: Repo, opportunity_id: str, opportunity_stage: str, pan_hash: str, refresh: bool, now: datetime, live_enabled: bool) -> AdmissionDecision. Must return refusal LIVE_CALLS_DISABLED when a billed call would be needed and live_enabled is False; CACHE_FRESH_NO_REFRESH when refresh=True before TTL; cache_hit (never a refusal) when cache fresh and refresh=False; refresh admissions get counts_against_cap=False. Decision must NOT contain the cache payload — metadata only (my nodes re-read the row via get_cache).
- **name**: tools/surepass.py — client protocol (tools section) | **expectation**: class SurePassResponse(endpoint: str, http_status: int, success: bool | None, message_code: str | None, client_id: str | None (data.client_id), data: dict[str, Any], raw: dict[str, Any] (full envelope)); exceptions SurePassError(Exception), LiveCallsDisabled(SurePassError); class SurePassClientProtocol(Protocol): live_enabled: bool (property); pan_comprehensive(pan: str) -> SurePassResponse (posts {id_number, get_address: true, get_contact: false}); experian_score(*, name: str, mobile: str, pan: str) -> SurePassResponse (adds consent 'Y'); experian_report(*, name: str, mobile: str, pan: str) -> SurePassResponse. The client does NOT log to the DB (nodes do, because only they know check_id) and must handle the {errors, message} validation envelope (success=None in that case). FakeSurePassClient implements the same protocol and records a call log that the policy_consistency evaluator reads.
- **name**: tools/adapters.py (tools section) | **expectation**: adapt_pan_comprehensive(resp: SurePassResponse) -> Identity (gate success+status=='valid'+category=='person'; gender M/F mapping; zip filtered by PINCODE_RE_STR; NEVER models masked_aadhaar); adapt_experian_score(resp: SurePassResponse) -> BureauScore (NTC ASSUMPTION branch: success false or empty/unparseable credit_score; ReportDate YYYYMMDD); adapt_experian_report(resp: SurePassResponse) -> BureauReport (open predicate Date_Closed=='' or Account_Status=='11'; EMI sum/floor; tenure from Repayment_Tenure ('0'=unknown) else zero-padded Terms_Duration; Rate_of_Interest ''->None; bureau_pincodes Date_Reported within 365d filtered to PINCODE_RE_STR; DOB never read from holder rows); band_from_score(score: int | None) -> Band using FIXED buckets <650 / 650-720 / 720-780 / 780+ / NTC with BAND_ORDER from graph.state.
- **name**: tools/lenders.py (tools section) | **expectation**: @dataclass(frozen=True) class CutoffResult(passed: bool, failed: list[str], unknown: list[str]); check_hard_cutoffs(cutoffs: HardCutoffs, applicant: Applicant) -> CutoffResult (a cutoff fails ONLY when the fact is known and violates; unknown facts -> unknown list); loan_ceiling(applicant: Applicant, cutoffs: HardCutoffs) -> float | None (headroom = max_foir*income - existing_emi; <=0 -> 0.0; else principal whose reducing-balance EMI equals headroom at annual_rate_pct default 14 over max_tenure_months default 36, rounded to nearest 1000; None when income unknown); estimate_monthly_emi(*, balance: float, tenure_months: int | None, annual_rate_pct: float | None) -> float (8.3 defaults 36 months / 14% p.a.).
- **name**: output/lc_view.py — redaction boundary (output section) | **expectation**: build_lc_view(*, check_id: str, kind: Kind, status: Status, refusal_code: RefusalCode | None, final: FinalRecommendation | None, gated_lenders: Sequence[GatedLender], identity: Identity | None, from_cache: bool, cached_at: datetime | None, refresh_available_at: datetime | None, quick_checks_remaining_today: int | None) -> LcView — maps top/alternate lender ids to LenderRef names via gated_lenders, builds indicative_loan_range only when both bounds set, scrubs every LC-facing line (drop lines matching (?i)\b(cibil|experian|equifax|crif|bureau)\b or containing a standalone 300-899 3-digit number, e.g. (?<![0-9.])[3-8][0-9]{2}(?![0-9.])), sets pan_aadhaar_link_issue True ONLY when identity.aadhaar_linked is False, and for status='pan_invalid' emits the fixed PAN_INVALID_MESSAGE talking point. Also: build_refusal_view(*, check_id: str, kind: Kind, refusal_code: RefusalCode, quick_checks_remaining_today: int | None) -> LcView using REFUSAL_MESSAGES: dict[RefusalCode, str] (one-line LC message per code).
- **name**: config.py — Settings (config section) | **expectation**: Settings with at least: surepass_live: bool (env SUREPASS_LIVE == 'true'); openai_model: str = 'gpt-4.1'; langsmith_enabled: bool (env LANGSMITH_TRACING == 'true'); langsmith_project: str = 'eligibility-agent'; prompts_source: Literal['local','hub'] = 'local'; prompts_tag: str = 'prod'; checkpointer_db_url: str | None = None; pan_salt: str; service_api_token: str. Plus hash_pan(pan: str, salt: str) -> str (salted SHA-256 hex) available to main.py before invoke.
- **name**: main.py — route-side obligations (API section) | **expectation**: Per request: generate check_id=str(uuid4()); compute pan_hash; INSERT the pending checks row (status 'pending') BEFORE graph.invoke so surepass_calls.check_id FK holds mid-run; build the initial CheckState exactly per my initial-state contract; run_id=new_run_id(settings.langsmith_enabled); cfg=invoke_config(...); final_state = await run_in_threadpool(graph.invoke, initial, cfg) (all deps are sync); finish_trace(langsmith_client_or_None, ...); respond with final_state['lc_view']. On SurePassError/RuntimeError escaping the graph: mark the check failed (status outside the LcView enum, e.g. 'error') and return HTTP 502. Lifespan builds GraphDeps once, optionally make_postgres_checkpointer, and compiles graph_quick/graph_full once.
- **name**: tests — FakeLLM parity (tests section) | **expectation**: FakeRankLLM and FakeFinalizeLLM implement StructuredLLM.__call__(messages, *, config=None) returning RankingProposal / FinalRecommendation deterministically (e.g. scripted queues or functions of the rendered user message); to exercise retries they return business-rule-violating objects (unknown lender_id) or raise — both paths are covered by the node contracts. No monkeypatching: fakes enter only through GraphDeps.

## Decisions
- **topic**: Retry budgets for the two LLM loops | **decision**: rank_lenders: RANK_MAX_ATTEMPTS=2 total LLM calls (1 initial + 1 retry). finalize: FINALIZE_MAX_ATTEMPTS=3 total LLM calls (1 initial + 2 retries) then fallback_finalize. Counters count LLM calls made, incremented inside the node; routers compare against the constants. | **rationale**: PRD 7.3 node text says rank 'retry once' but the diagram says 'retries < 2' for both loops, while finalize text says 'after 2 retries go to fallback_finalize'. I honored each node's explicit prose; the discrepancy is surfaced in prd_gaps.
- **topic**: Two models where the PRD reuses the name 'Ranking' | **decision**: RankingProposal(ranked=[RankedLender]) is the LLM structured-output schema; Ranking(accepted, dropped) is the verified state model stored in CheckState.ranking. | **rationale**: PRD 7.3 uses 'Ranking' for both the LLM output shape and the post-verification accepted+dropped shape; one Pydantic class cannot be both without optional mush that breaks mypy --strict and the OpenAI tool schema.
- **topic**: Dependency-injection seam | **decision**: Frozen GraphDeps dataclass (repo, surepass, rank_llm, finalize_llm, prompts, now) bound via functools.partial at build_graph time; one generic StructuredLLM Protocol with __call__(messages, *, config) covers both LLM callables; deps are NOT smuggled through config['configurable']. | **rationale**: partial keeps nodes plain typed functions testable in isolation, leaves the (state, config) signature LangGraph introspects, and avoids stringly-typed config lookups; passing config into the LLM invoke guarantees child runs nest under the node's trace span.
- **topic**: LangSmith run id capture | **decision**: Pre-generate run_id = uuid4() in the route when tracing is enabled, pass it in the root RunnableConfig, seed state['langsmith_run_id']=str(run_id) before invoke; persist writes it to the checks row. admission:<mode> tag plus billed/counted_against_cap metadata are patched after the run via Client.update_run (resending base tags because update_run replaces them), inside try/except. | **rationale**: Verified: a root-config run_id becomes the trace_id, which is deterministic and avoids collect_runs post-hoc capture; admission/billed/counted only exist after the run, so they cannot be invoke-time metadata.
- **topic**: Raw SurePass payload transport | **decision**: Raw envelopes ride in state['raw_payloads'] ({'pan': ..., 'score'|'report': ...}) solely so the pull node can write pan_cache.payload; persist clears the key and _internal_json excludes it, along with pan/mobile/name. They are never rendered into any prompt template. | **rationale**: The cache row (payload = both raw responses) is written once, after the bureau call, avoiding partial cache rows on pan_invalid or mid-run failure; the only alternative (re-calling PAN in the pull node) spends money.
- **topic**: Checkpointer configuration and tests | **decision**: PostgresSaver over a manually built psycopg_pool.ConnectionPool with kwargs autocommit=True, row_factory=dict_row, prepare_threshold=None; setup() at FastAPI lifespan startup; Supabase session pooler (or direct) URI with options=-csearch_path=checkpointer pointing at a dedicated non-PostgREST-exposed schema; thread_id = check_id. Tests compile with checkpointer=None; InMemorySaver only for an optional resume test. Checkpointer is also optional in prod (CHECKPOINTER_DB_URL unset -> None) so /health works with no DB. | **rationale**: from_conn_string hardcodes prepare_threshold=0 which breaks behind poolers (verified via langgraph issue #6705 and the package README); checkpoint blobs contain the raw PAN and raw_payloads, and Supabase exposes RLS-less public-schema tables through PostgREST, so schema isolation is a data-leak control, not a nicety.
- **topic**: State keys added beyond PRD 7.2 | **decision**: Added consent, consent_id, counted_against_cap, quick_checks_used_today, cache_fetched_at, cache_expires_at, rank_attempts, finalize_attempts, rank_errors, finalize_errors, pan_called, bureau_called, raw_payloads, transcript (Annotated operator.add, like errors), langsmith_run_id. All PRD keys kept verbatim. | **rationale**: PRD explicitly allows retry counters; the rest are required to compute billed ('any SurePass call made'), counted_against_cap (= counts_against_cap and bureau_called and not pan_invalid), LcView cache fields, quick_checks_remaining_today without a second DB query, and the 'node transcript' the persist contract stores in internal.
- **topic**: rank_lenders with zero gated lenders | **decision**: Skip the LLM entirely and write Ranking(accepted=[], dropped=[]); finalize still runs and its user template carries an explicit 'no lender passed the location gate' instruction (co-applicant cannot fix pincode eligibility). | **rationale**: PRD allows an empty gate result and says it lowers confidence downstream; calling the LLM with nothing to rank wastes tokens and invites invented lenders.
- **topic**: Additive code validations in finalize | **decision**: Beyond PRD 7.3's three checks, validate_final also rejects: plain 'fundable' when band is NTC or <650 without coapplicant_suggested, and indicative_loan_max above the top lender's code-computed ceiling (when known). After exhaustion these fall through to fallback review. | **rationale**: Both mirror stated invariants (the outcome_matches_band evaluator and the 8.4 ceiling); enforcing them deterministically strengthens learner protection and never weakens a FIXED rule. Flagged because it is additive.
- **topic**: Error handling split | **decision**: LLM-call exceptions (network, parse, Pydantic constraint violations under method='function_calling') are caught inside rank/finalize nodes, recorded as a failed attempt, and absorbed by the retry/fallback machinery. SurePass SurePassError/LiveCallsDisabled and repo errors propagate out of the graph; main.py maps them to HTTP 502 and marks the check failed. | **rationale**: An LLM hiccup must not kill a run that already spent a bureau call (fallback still yields a persisted review LcView); a bureau/db failure has no honest LcView to return, and the checkpoint keyed by check_id preserves the partial run for inspection/resume.
- **topic**: Structured output method | **decision**: ChatOpenAI.with_structured_output(schema, method='function_calling'), temperature 0, max_retries=1, 30s timeout, for both callables. | **rationale**: OpenAI strict json_schema mode rejects maxLength/maxItems constraints that FinalRecommendation carries (max 2 alternates, 4 talking points of <=200 chars, 2000-char reasoning); with function_calling Pydantic enforces them at parse time and violations route into the existing retry loop.
- **topic**: Prompt sourcing | **decision**: Local literals in prompts.py are the pinned source of truth (PROMPTS_LOCAL_VERSION); PROMPTS_SOURCE=hub pulls hero/eligibility-rank:<tag> and hero/eligibility-finalize:<tag> via langsmith Client.pull_prompt and falls back to local on any exception; default is local so tests/CI never touch the network. System prompts are brace-free; user messages are rendered by plain Python functions, so lender-criteria JSON never passes through ChatPromptTemplate formatting. | **rationale**: Satisfies 'versioned in LangSmith Hub, pulled by tag, repo pins a tag' and the local-context requirement that tests never pull from the Hub; brace-free storage sidesteps template-variable mangling of JSON.
- **topic**: billed/counted recomputed at persist | **decision**: admission writes the expected billed flag; persist recomputes billed = pan_called or bureau_called and counted_against_cap = counts_against_cap and bureau_called and status != 'pan_invalid', and writes both to state for finish_trace. | **rationale**: Matches the persist contract verbatim and makes pan_invalid behave per open item 3: the PAN call is billed by SurePass (billed=True) but never counts against the LC's five.
- **topic**: EMI floor semantics per tier | **decision**: Full: floor = report.existing_emi_floor or report.is_ntc; existing_emi = max(declared, reported+estimated) when declared given. Quick: existing_emi = declared or None; floor = (declared is None). | **rationale**: Implements 'true when any open account lacks a reported EMI, or when nothing is known' — an NTC file knows nothing, a declared quick-check EMI is known; the probe file shows the estimation path is the norm (0 of 5 open accounts reported an EMI).
- **topic**: quick_checks_remaining_today without a second query | **decision**: AdmissionDecision carries quick_checks_used_today; redact computes remaining = max(0, 5 - used - (1 if this run will count else 0)); None for kind='full'. | **rationale**: The current check's row is only updated in persist (after redact), so the DB count cannot include it; carrying the admission-time count plus the deterministic will-count flag gives an exact value with zero extra queries.

## Risks
- Checkpoint blobs duplicate sensitive data (raw PAN in request keys, raw_payloads mid-run). Mitigated by a dedicated 'checkpointer' Postgres schema excluded from PostgREST and service-role-only access, but there is no TTL/cleanup job for old threads yet — recommend a cron DELETE on checkpoints older than 30 days.
- OpenAI structured-output strictness: method='json_schema' would reject FinalRecommendation's maxLength/maxItems constraints; design pins method='function_calling'. If the implementer switches methods, constraint keywords must be stripped and enforced only in validate_final.
- LangGraph is pinned conceptually to 1.x; CompiledStateGraph's generic arity (StateT, ContextT, InputT, OutputT) has shifted across minors — the GraphT alias is the single place to adjust. Pin langgraph>=1.0,<2 and langgraph-checkpoint-postgres accordingly in pyproject.
- finish_trace patches tags/metadata via Client.update_run after the trace uploads asynchronously; under rare flush delays the patch can race the initial upload. It is wrapped in try/except and purely additive — a lost patch loses only the admission tag/billed metadata on the trace, never on the checks row.
- Supabase direct connections are IPv6-only without the IPv4 add-on; the design prescribes the session pooler URI. If ops later switches to the transaction pooler (:6543), setup() will fail and prepared-statement errors will appear — the PG_POOL_KWARGS comment documents this, but it is an operational foot-gun.
- graph.invoke is synchronous end-to-end (supabase-py, psycopg pool, ChatOpenAI sync); routes must use run_in_threadpool or sync def endpoints, otherwise the event loop blocks for the full bureau+LLM latency.
- Experian NTC/no-hit envelope shape is unverified (field map R10); the adapter owns that ASSUMPTION branch, but my pull_score cache write trusts BureauScore.is_ntc — if the real no-hit arrives as an HTTP error status rather than a parseable envelope, the SurePassClient must still return a SurePassResponse (success=False) rather than raise, or NTC caching (policy rule 7) breaks.

## PRD gaps
- PRD 7.3 rank retry count conflicts with itself: the node contract says 'retry once' while the diagram condition is 'retries < 2' (which would allow two retries). Implemented rank=2 total attempts, finalize=3 total attempts; owner should confirm.
- PRD uses the name 'Ranking' for both the LLM output schema (ranked=[...]) and the verified state model (accepted+dropped); split into RankingProposal and Ranking — rename needs owner sign-off if the evaluators reference the class name.
- LcView.quick_checks_remaining_today semantics for kind='full' are unspecified; set to null. Also unspecified whether a refused quick check should still report remaining checks — I report max(0, 5-used).
- No checks.status value is defined for a hard mid-run failure (SurePass 5xx after admission); LcView only allows ok|pan_invalid|refused. Proposed: checks row status 'error' written by the API exception handler, HTTP 502, no LcView. Needs owner confirmation and a schema enum decision.
- CONSENT_MISSING is unreachable through the API (request schema requires consent literally true); kept as a defensive guard in admission. Confirm whether a consent=false request should instead be a 422 with no checks row.
- Exact trigger for OPPORTUNITY_ALREADY_CHECKED vs cache_hit on a re-run at offer stage (test 8 expects the second full check on the same opportunity to be a cache hit) is left to the policy section; my admission node only consumes AdmissionDecision.
- NTC full-report EMI floor: implemented existing_emi_floor=True for NTC files ('nothing is known'); if the owner prefers known-zero for empty files, flip one condition in the affordability node.
