# Graph, derivations, matching, the single LLM node (PRD v1.1 sections 7, 8, 9)

## Summary
Redesign of the LangGraph pipeline for PRD v1.1: one linear deterministic graph (admission -> pan_comprehensive -> bureau_pull -> derive -> pincode_gate -> range -> code_confidence -> rank -> safety -> redact -> persist, plus persist_refusal and a recompute sub-graph entered at derive) with exactly ONE LLM node (rank) emitting the PRD-verbatim ModelOutput/Pick schema; the model never produces a number and its confidence is only ever min()'d down. CheckState v2 drops gender everywhere, splits EMI into reported/estimated per-account items with stale (>90d) and past-tenure flags, carries per-lender range/ceiling/displayed-FOIR and age-at-maturity, and candidate pincodes. NTC semantics REVERSE v1.0: NTC only on HTTP 200 + success:true + credit_score not a 3-digit string; every success:false envelope now raises (service_failure), so the client taxonomy, adapters and fixtures change; a new SurePassTimeout drives the 10-minute PAN lock + pending_verification terminal path. Money math is pinned with machine-verified examples: EMI algorithm all four branches (12,500 reported; 160000/8=20,000; past-tenure 60000/12=5,000; revolving 0.05*80000=4,000; fallback 90000/36=2,500; stale account excluded), and annuity range math incl. a clipping case (available_emi 20,000, 11-14%, 12-60m, loan cap 500k: upper raw 919,860.68 clipped to 500,000, lower 222,749.10 -> 222,000) and the displayed-FOIR identity resolution (unclipped displayed FOIR == foir_limit exactly; 0.3606 in the clipped example). Reuses verbatim from the committed v1.0 dossier: checkpointer factory (manual psycopg pool, session pooler, dedicated schema, LANGGRAPH_STRICT_MSGPACK), graph/tracing.py (new_run_id/invoke_config/finish_trace), typed-closure node binding, kill-switch exact literal, envelope parsing, FakeSurePassClient scenario machinery, parsing helpers (parse_amount/parse_int_sentinel/parse_yyyymmdd/as_list/dig), band literals and BAND_ORDER, redaction regex single-home. Critical files for implementation: eligibility_agent/graph/state.py, eligibility_agent/graph/nodes.py, eligibility_agent/derive/emi.py, eligibility_agent/derive/range.py, eligibility_agent/lenders/safety.py.

## Files
### eligibility_agent/graph/state.py
_Single leaf module: CheckState v2 TypedDict and every shared Pydantic fact/output model (identity with NO gender, score, report, EMI derivation, per-lender numbers, safety, ModelOutput/Pick, confidence/headline literals). pyright strict; imports stdlib+pydantic only._

from __future__ import annotations
import operator
from datetime import date, datetime
from typing import Annotated, Any, Final, Literal, TypeAlias, TypedDict
from pydantic import BaseModel, ConfigDict, Field

Kind: TypeAlias = Literal["quick", "full"]
Band: TypeAlias = Literal["<650", "650-720", "720-780", "780+", "NTC"]   # SURVIVES v1.0; internal only, never in LC strings
BAND_ORDER: Final[dict[Band, int]] = {"NTC": 0, "<650": 1, "650-720": 2, "720-780": 3, "780+": 4}
Headline: TypeAlias = Literal["fundable", "possible", "unlikely", "not_fundable_alone", "under_review"]
CheckStatus: TypeAlias = Literal["ok", "pan_not_verified", "refused", "pending_verification", "service_failure"]
Code: TypeAlias = Literal["CONSENT_MISSING", "QUOTA_EXHAUSTED", "PAN_DAILY_CAP", "STAGE_TOO_EARLY", "REPORT_WINDOW_OPEN", "IN_FLIGHT", "PAN_LOCKED", "WARN_NTC_RECENT", "PAN_CHANGE_NEEDS_MANAGER", "NO_PINCODE_ON_PAN", "LIVE_CALLS_DISABLED", "SUREPASS_FAILURE", "OPENAI_FAILURE"]
Confidence: TypeAlias = Literal["high", "medium", "low"]
CONFIDENCE_ORDER: Final[dict[Confidence, int]] = {"low": 0, "medium": 1, "high": 2}
AdmissionMode: TypeAlias = Literal["reuse", "call"]
PINCODE_RE_STR: Final[str] = r"^[1-9][0-9]{5}$"
StateUpdate: TypeAlias = dict[str, Any]
FieldStatus: TypeAlias = Literal["known", "missing", "uncertain"]

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class ConsentInfo(_Frozen):
    given: Literal[True]
    wording_version: str
    call_ref: str | None
    ip: str | None

class Identity(_Frozen):            # NO gender field anywhere (Experian takes none; do not derive or send one)
    pan_valid: bool                 # client only returns success:true now; gate = status=='valid' and category=='person'
    status: str | None
    category: str | None
    full_name: str | None           # sent as `name` on bureau calls; LC-typed name lives only in the consent record
    dob: date | None                # data.dob YYYY-MM-DD; NEVER from bureau holder rows
    aadhaar_linked: bool | None     # strict bool; link warning fires ONLY on literal False
    pan_pincode: str | None         # data.address.zip iff fullmatch(PINCODE_RE_STR)
    surepass_client_id: str | None  # masked_aadhaar deliberately NOT modeled (raw-payload only)

class ScoreFacts(_Frozen):
    band: Band
    score: int | None               # internal only; never serialized to any prompt or LcView
    is_ntc: bool                    # ONLY 200 + success:true + credit_score not ^\d{3}$
    pulled_on: date                 # fetched_at date of the cache row / this pull
    source: Literal["score_endpoint", "report_endpoint"]  # Report pull refreshes the score cache

class OpenAccount(_Frozen):         # one row per OPEN account; the 8.3 algorithm's input
    account_key: str                # 'acct_<i>' synthetic; Account_Number never carried
    portfolio_type: Literal["I", "R"] | None
    account_type: str               # raw code string ('05','10',...)
    scheduled_emi: float | None     # parse_amount(Scheduled_Monthly_Payment_Amount); None = not reported
    current_balance: float          # parse_amount or 0.0; negatives clamped to 0.0
    tenure_months: int | None       # Repayment_Tenure if >0 else Terms_Duration if >0 (zero_is_unknown)
    rate_of_interest: float | None  # context for the model ONLY; never used in the estimate
    open_date: date | None          # YYYYMMDD
    date_reported: date | None
    amount_past_due: float
    pincodes: list[str]             # this account's ZIP_Postal_Code_non_normalized values passing PINCODE_RE_STR
    is_stale: bool                  # (report_date - date_reported).days > 90 (or date_reported unparseable)

class ReportFacts(_Frozen):
    band: Band
    score: int | None
    is_ntc: bool                    # no-file: stored as NTC for 30 days
    report_date: date | None        # CreditProfileHeader.ReportDate
    open_accounts: list[OpenAccount]
    past_due_total: float           # sum Amount_Past_Due over ALL accounts
    written_off_or_settled: int     # count non-empty Written_off_Settled_Status
    enquiries_30d: int
    enquiries_90d: int
    enquiries_180d: int             # TotalCAPS_Summary.TotalCAPSLast{30,90,180}Days
    bureau_pincodes_12m: list[str]  # 8.6: open accounts with Date_Reported within 365d of report_date; sorted, deduped

class EmiItem(_Frozen):
    account_key: str
    kind: Literal["reported", "estimated", "stale_excluded"]
    monthly_amount: float           # 0.0 for stale_excluded
    method: Literal["reported", "tenure_remaining", "past_tenure", "revolving_5pct", "balance_36"] | None
    past_tenure: bool
    confirm_item: str | None        # e.g. 'confirm with learner: an account last updated 2026-05-20'

class EmiDerivation(_Frozen):
    reported_total: float
    estimated_total: float
    derived_total: float            # reported + estimated over NON-stale open accounts
    declared: float | None
    used: float                     # full: max(declared, derived); quick: declared or 0.0
    used_source: Literal["derived", "declared", "declared_zero_assumed"]
    emi_floor_flag: bool            # any open account estimated OR flagged (stale / past-tenure)
    items: list[EmiItem]

class LenderCriteria(_Frozen):      # normalised PUBLISHED admin criteria (section 11)
    score_floor: int | None         # entered as an Experian score (open item: lenders quote CIBIL)
    ntc_accepted: bool | None
    foir_limit: float | None
    min_income: float | None
    min_age_years: int | None
    max_age_at_maturity_years: int | None
    rate_min_pct: float | None
    rate_max_pct: float | None
    tenure_min_months: int | None
    tenure_max_months: int | None
    loan_min: float | None
    loan_max: float | None
    policy_notes: str | None
    field_status: dict[str, FieldStatus]   # per-field known/missing/uncertain from the admin UI

class GatedLender(_Frozen):
    id: str
    name: str                       # resolved to LcView only in redact; NEVER in the model payload
    criteria: LenderCriteria
    matched_pincodes: list[str]
    rate_band_label: str            # '11 to 14%'  (for LcView top_lender.rate_band)
    tenure_label: str               # '12 to 60 months'

class Derived(_Frozen):             # written by the derive node
    band: Band                      # newer of ScoreFacts/ReportFacts (by pulled_on vs report_date)
    emi: EmiDerivation
    current_foir: float | None      # emi.used / income; None when no income (quick w/o income)
    age_at_application_years: int | None   # from Identity.dob only; None -> age check skipped, confidence capped medium
    age_months_at_application: int | None  # integer-month basis for maturity math
    candidate_pincodes: list[str]

class LenderNumbers(_Frozen):       # written by the range node, one per gated lender
    lender_id: str
    available_emi: float | None     # None when no income; 0.0 -> not fundable alone with this lender
    loan_max_lower: float | None    # display value, floored to 1000
    loan_max_upper: float | None    # display value, floored to 1000; == ceiling
    loan_max_lower_raw: float | None  # unrounded, internal + safety
    loan_max_upper_raw: float | None
    ceiling: float | None
    displayed_foir: float | None    # (emi.used + emi(P_wc at r_max, n_max)) / income; see derive/range.py
    age_at_maturity_years: int | None  # (age_months + tenure_max)//12
    not_fundable_alone: bool        # available_emi == 0.0
    clipped: Literal["none", "upper", "lower", "both"]

class Pick(BaseModel):              # PRD 9.2 VERBATIM — the LLM output schema
    lender_id: str
    rank: int
    rationale: str
    conditions: list[str]

class ModelOutput(BaseModel):       # PRD 9.2 VERBATIM — the model never produces a number (no numeric money/score fields exist)
    recommendation: Pick | None
    alternates: list[Pick] = Field(max_length=2)
    confidence: Literal["high", "medium", "low"]   # may only LOWER code_confidence (enforced by min())
    co_applicant_suggested: bool
    co_applicant_reason: str | None
    review_flags: list[str]

class SafetyBreach(_Frozen):
    lender_id: str
    rule: Literal["not_in_gate", "band_floor", "foir_limit", "age_bounds", "min_loan_amount", "duplicate"]
    detail: str                     # internal only; may contain numbers; never reaches LcView

class SafetyResult(_Frozen):
    top: Pick | None
    alternates: list[Pick]          # max 2 surviving, order preserved
    demoted: list[SafetyBreach]
    none_survived: bool
    review: bool                    # -> review_queue row at persist
    top_clears_floor_by_one: bool | None  # feeds the 9.4 'high' cap

class AdmissionFacts(_Frozen):      # produced by policy/guards (other section); consumed verbatim here
    mode_pan: AdmissionMode         # reuse when pan_records row < 180d
    mode_bureau: AdmissionMode      # reuse when score/report cache < 30d
    will_bill_bureau: bool
    counts_against_quota: bool      # quick, new billed Score on a PAN with no stored score
    quota_used_today: int | None    # quick only
    quota_limit: int | None         # 5 default, per-LC override
    refusal_code: Code | None
    consent_id: str | None
    pan_cached_on: date | None
    score_cached_on: date | None
    report_cached_on: date | None
    refresh_available_on: date | None
    is_refresh: bool

class SurePassCallAudit(_Frozen):
    endpoint: str
    http_status: int | None
    success: bool | None
    message_code: str | None
    client_id: str | None
    billed: bool

class CheckState(TypedDict, total=False):
    # request (seeded by main.py / recompute route)
    kind: Kind
    check_id: str
    lc_id: str
    lead_id: str
    opportunity_id: str | None
    opportunity_stage: str | None
    pan: str                        # raw, request scope only; stripped at persist
    name: str                       # LC-typed; consent record only
    mobile: str
    pan_hash: str                   # HMAC-SHA256 (config/policy own the function)
    mobile_hash: str
    monthly_income: float | None
    existing_emi_declared: float | None
    refresh: bool
    acknowledge_ntc_warning: bool
    consent: ConsentInfo | None
    is_recompute: bool
    # admission
    admission: AdmissionFacts | None
    # facts (internal; raw payloads NEVER live in state — v1.1 section 7)
    identity: Identity | None
    score: ScoreFacts | None
    report: ReportFacts | None
    # derivations / matching
    derived: Derived | None
    candidate_pincodes: list[str]
    gated_lenders: list[GatedLender]
    gate_warnings: list[str]
    lender_numbers: dict[str, LenderNumbers]
    code_confidence: Confidence | None
    model_output: ModelOutput | None
    rank_attempts: int              # LLM calls made; max 2 (1 + ONE retry)
    rank_error: str | None          # fed into the retry user message
    safety: SafetyResult | None
    # outcome
    status: CheckStatus | None      # set by terminal branches; redact defaults 'ok'
    code: Code | None
    headline: Headline | None
    final_confidence: Confidence | None
    lc_view: Any | None             # output.lc_view.LcView (owned by the API/redaction section)
    # audit
    surepass_calls_made: Annotated[list[SurePassCallAudit], operator.add]
    bureau_called: bool
    pan_called: bool
    langsmith_run_id: str | None
    errors: Annotated[list[str], operator.add]
    transcript: Annotated[list[str], operator.add]   # node names in order; -> checks.internal


### eligibility_agent/graph/build.py
_GraphDeps (ONE LLM seam), typed-closure node binding (survives v1.0 C7), build_graph(kind) and build_recompute_graph(kind) wiring exactly per v1.1 section 7, routers, constants. Checkpointer factory make_postgres_checkpointer SURVIVES v1.0 verbatim (manual psycopg ConnectionPool, autocommit=True, prepare_threshold=None, dict_row, Supabase SESSION pooler :5432, dedicated 'checkpointer' schema, LANGGRAPH_STRICT_MSGPACK + allowed_msgpack_modules incl. eligibility_agent.graph.state, post-setup to_regclass assertion) — see docs/design/design-graph-state-prompts-langsmith-wirin.md and audit-technical-feasibility; do not re-derive._

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Literal, TypeAlias
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from eligibility_agent.db.repo import Repo
from eligibility_agent.surepass.client import SurePassClient
from eligibility_agent.graph import nodes
from eligibility_agent.graph.llm import RankModel
from eligibility_agent.graph.prompts import PromptBundle
from eligibility_agent.graph.state import CheckState, Kind, StateUpdate

RANK_MAX_ATTEMPTS: Final[int] = 2          # 1 initial + ONE retry on invalid output (v1.1 FIXED)
RECURSION_LIMIT: Final[int] = 40
NODE_NAMES: Final[tuple[str, ...]] = ("admission", "persist_refusal", "pan_comprehensive", "bureau_pull", "derive", "pincode_gate", "range", "code_confidence", "rank", "safety", "redact", "persist")
RECOMPUTE_ENTRY: Final[str] = "derive"     # recompute sub-graph: derive -> ... -> persist on stored raw

def _utcnow() -> datetime: return datetime.now(timezone.utc)

@dataclass(frozen=True, slots=True)
class GraphDeps:
    repo: Repo
    surepass: SurePassClient
    rank_model: RankModel          # the ONE LLM seam; FakeRankModel implements the same protocol
    prompts: PromptBundle
    now: Callable[[], datetime] = field(default=_utcnow)   # tz-aware UTC; tests inject FrozenClock

# typed closure bind — SURVIVES v1.0 (audit fix over functools.partial; pyright-strict clean)
NodeFn: TypeAlias = Callable[[CheckState, RunnableConfig], StateUpdate]
def bind(fn: Callable[[CheckState, RunnableConfig, GraphDeps], StateUpdate], deps: GraphDeps) -> NodeFn:
    def node(state: CheckState, config: RunnableConfig) -> StateUpdate:   # param MUST be named 'config' (langgraph injects by name)
        return fn(state, config, deps)
    node.__name__ = getattr(fn, "__name__", "node")
    return node

# ---- routers (pure; no deps) ----
def route_admission(state: CheckState) -> Literal["refused", "proceed"]:
    adm = state.get("admission")
    return "refused" if (adm is not None and adm.refusal_code is not None) else "proceed"

def route_pan(state: CheckState) -> Literal["invalid", "failed", "valid"]:
    if state.get("status") == "service_failure": return "failed"          # PAN service failure -> persist (no bureau call, no quota)
    ident = state.get("identity")
    return "valid" if (ident is not None and ident.pan_valid) else "invalid"  # invalid -> redact (status pan_not_verified)

def route_bureau(state: CheckState) -> Literal["terminal", "ok"]:
    return "terminal" if state.get("status") in ("pending_verification", "service_failure") else "ok"

def route_gate(state: CheckState) -> Literal["band_read", "proceed"]:
    return "band_read" if state.get("code") == "NO_PINCODE_ON_PAN" else "proceed"   # skip range/rank/safety -> redact

def route_rank(state: CheckState) -> Literal["retry", "failed", "safety"]:
    if state.get("model_output") is not None: return "safety"
    if state.get("code") == "OPENAI_FAILURE": return "failed"              # 2nd failure -> redact (band read still shown), review queue
    if not state.get("gated_lenders"): return "safety"                     # nothing to rank: LLM skipped, safety handles none_survived
    if state.get("rank_attempts", 0) < RANK_MAX_ATTEMPTS: return "retry"
    return "failed"

# ---- build ----
def build_graph(kind: Kind, deps: GraphDeps, *, checkpointer: Any | None = None) -> Any:
    g: StateGraph[CheckState] = StateGraph(CheckState)
    for name in NODE_NAMES:
        g.add_node(name, bind(getattr(nodes, name), deps))
    g.add_edge(START, "admission")
    g.add_conditional_edges("admission", route_admission, {"refused": "persist_refusal", "proceed": "pan_comprehensive"})
    g.add_edge("persist_refusal", END)
    g.add_conditional_edges("pan_comprehensive", route_pan, {"invalid": "redact", "failed": "persist", "valid": "bureau_pull"})
    g.add_conditional_edges("bureau_pull", route_bureau, {"terminal": "persist", "ok": "derive"})   # timeout/service failure -> persist -> END
    g.add_edge("derive", "pincode_gate")
    g.add_conditional_edges("pincode_gate", route_gate, {"band_read": "redact", "proceed": "range"})
    g.add_edge("range", "code_confidence")
    g.add_edge("code_confidence", "rank")
    g.add_conditional_edges("rank", route_rank, {"retry": "rank", "failed": "redact", "safety": "safety"})
    g.add_edge("safety", "redact")
    g.add_edge("redact", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=checkpointer)   # thread_id = check_id via tracing.invoke_config (SURVIVES v1.0)

def build_recompute_graph(kind: Kind, deps: GraphDeps, *, checkpointer: Any | None = None) -> Any:
    """POST /checks/{id}/recompute: derive -> pincode_gate -> range -> code_confidence -> rank -> safety -> redact -> persist.
    The route re-adapts STORED raw payloads (bucket) into identity/score/report facts, seeds them + the new
    monthly_income/existing_emi into state with is_recompute=True, and invokes this graph. No SurePass call, no quota;
    persist updates the existing checks row with the new lc_view. Compiled once at startup alongside quick/full."""
    g: StateGraph[CheckState] = StateGraph(CheckState)
    for name in ("derive", "pincode_gate", "range", "code_confidence", "rank", "safety", "redact", "persist"):
        g.add_node(name, bind(getattr(nodes, name), deps))
    g.add_edge(START, "derive")
    g.add_edge("derive", "pincode_gate")
    g.add_conditional_edges("pincode_gate", route_gate, {"band_read": "redact", "proceed": "range"})
    g.add_edge("range", "code_confidence")
    g.add_edge("code_confidence", "rank")
    g.add_conditional_edges("rank", route_rank, {"retry": "rank", "failed": "redact", "safety": "safety"})
    g.add_edge("safety", "redact")
    g.add_edge("redact", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=checkpointer)

# make_postgres_checkpointer(db_url) -> (PostgresSaver, ConnectionPool): SURVIVES v1.0 VERBATIM
# (manual pool kwargs {autocommit: True, prepare_threshold: None, row_factory: dict_row}; saver.setup();
#  session pooler :5432 or direct, NEVER :6543; CREATE SCHEMA checkpointer + role search_path; assert
#  to_regclass('public.checkpoints') IS NULL; LANGGRAPH_STRICT_MSGPACK=true + allowed_msgpack_modules
#  ['eligibility_agent.graph.state']). Reference: docs/design/design-graph-state-prompts-langsmith-wirin.md.


### eligibility_agent/graph/nodes.py
_One function per node, uniform signature def node(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate. All policy calls go through policy/guards; all money math through derive/*; the LLM only in rank._

UNIFORM SIGNATURE (pyright strict): def admission(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate
Every node also returns {'transcript': ['<node_name>']}.

--- admission ---
reads: kind, consent, lc_id, pan_hash, mobile_hash, opportunity_id, opportunity_stage, refresh, acknowledge_ntc_warning
logic: facts = policy.guards.admit(kind=state['kind'], repo=deps.repo, lc_id=..., pan_hash=..., mobile_hash=...,
        opportunity_id=..., opportunity_stage=..., refresh=..., acknowledge_ntc_warning=...,
        consent=state['consent'], live_enabled=deps.surepass.calls_enabled, now=deps.now())
  The guard (other section) does ALL of: write consent row (one per pull it will cover), acquire the per-PAN-hash
  in-flight lock (else refusal IN_FLIGHT), honour an active 10-min timeout lock (PAN_LOCKED), stage gate (full,
  STAGE_TOO_EARLY), evaluate windows (PAN 180d / score 30d / report 30d), quota (QUOTA_EXHAUSTED), per-PAN hard cap
  3/day (PAN_DAILY_CAP), report window (REPORT_WINDOW_OPEN unless refresh & expired), NTC warning (WARN_NTC_RECENT
  unless acknowledge_ntc_warning), PAN-change-needs-manager, kill switch (LIVE_CALLS_DISABLED when a billed call
  would be needed and calls are disabled; cache hits pass).
writes: {'admission': facts, 'status': ('refused' if facts.refusal_code else None), 'code': facts.refusal_code, 'transcript': ['admission']}

--- persist_refusal ---
view = output.lc_view.build_refusal_view(check_id, kind, code=state['code'], quota_left=_quota_left(state), refresh_available_on=facts.refresh_available_on)
deps.repo.finish_check(check_id, status='refused', code=..., billed=False, counted=False, lc_view=view, internal=_internal_json(state), langsmith_run_id=...)
deps.repo.release_pan_lock(pan_hash) only if the guard acquired it (IN_FLIGHT refusals never acquired it)
writes: {'lc_view': view, 'transcript': ['persist_refusal']}

--- pan_comprehensive ---
branch reuse (admission.mode_pan == 'reuse', stored < 180d):
  rec = deps.repo.get_pan_record(pan_hash); identity = Identity.model_validate(rec.derived)  # re-adaptable from raw if schema bumped
  writes {'identity': identity, 'transcript': [...]}
branch call:
  try: result = deps.surepass.pan_comprehensive(check_id, state['pan'])   # body {id_number, get_address: true, get_contact: false, masked_aadhaar_variant: 'v1'}; 8s timeout; client logs the sink row
  except SurePassError as e:   # v1.1 §10: ANY failure (timeout, 5xx, success:false, validation, auth) -> service failure; no bureau call, no quota; PAN timeout does NOT lock the PAN (rule 12 covers Score/Report only)
      view-less terminal: writes {'status': 'service_failure', 'code': 'SUREPASS_FAILURE',
        'lc_view': output.lc_view.build_failure_view(check_id, kind, status='service_failure', code='SUREPASS_FAILURE', message="Couldn't verify PAN, retry"),
        'errors': [f'pan: {e.detail}'], 'pan_called': True, 'transcript': [...]}  -> route_pan 'failed' -> persist
  identity = surepass.adapters.pan_to_identity(result)
  if identity.pan_valid: deps.repo.put_pan_record(pan_hash, raw_ref=deps.repo.store_raw('pan', pan_hash, opportunity_id, result.raw), derived=identity.model_dump(mode='json'), fetched_at=deps.now())
  # NOT valid / not person: do NOT cache, nothing marked checked, no quota (Rs 1 spent) -> route_pan 'invalid' -> redact (status pan_not_verified)
  writes {'identity': identity, 'pan_called': True, 'surepass_calls_made': [audit_row], 'transcript': [...]}

--- bureau_pull ---
quick + mode_bureau 'reuse': sf = deps.repo.get_score_cache(pan_hash) -> ScoreFacts; writes {'score': sf}
full + 'reuse': rf = deps.repo.get_report_cache(pan_hash) -> ReportFacts (re-adapt stored raw via adapters so derivations always run on current adapter code); writes {'report': rf}
branch call (name = identity.full_name or state['name']):
  try:
    result = deps.surepass.experian_score(check_id, name=..., mobile=..., pan=...)   # 12s
       or  deps.surepass.experian_report(check_id, name=..., mobile=..., pan=...)    # 20s; consent:'Y' sent only because a consent row exists (guard guaranteed it)
  except SurePassTimeout:   # spend unknown
    deps.repo.lock_pan(pan_hash, minutes=10, reason='bureau_timeout')
    writes {'status': 'pending_verification', 'lc_view': build_failure_view(..., status='pending_verification', code=None), 'bureau_called': True, 'transcript': [...]} -> persist -> END
  except SurePassError as e:   # 5xx, success:false, auth (ip_not_whitelisted/invalid_token -> alert flag), transport
    writes {'status': 'service_failure', 'code': 'SUREPASS_FAILURE', 'lc_view': build_failure_view(...), 'errors': [...], 'bureau_called': True, 'transcript': [...]}  # quota untouched (persist counts nothing on non-ok)
  quick: facts = adapters.score_to_score_facts(result, pulled_on=deps.now().date()); deps.repo.put_score_cache(pan_hash, raw_ref=store_raw(...), facts, fetched_at=now)   # NTC cached 30d too; counted
  full:  rf = adapters.report_to_report_facts(result, today=deps.now().date()); deps.repo.put_report_cache(...);
         deps.repo.put_score_cache(pan_hash, facts=ScoreFacts(band=rf.band, score=rf.score, is_ntc=rf.is_ntc, pulled_on=now.date(), source='report_endpoint'), ...)   # Report pull ALSO refreshes the score cache (FIXED rule 3)
  writes {'score'|'report': facts, 'bureau_called': True, 'surepass_calls_made': [audit_row], 'transcript': [...]}

--- derive ---  [all code, 8.x]
band: if report and score: use report.band if (report.report_date or date.min) >= score.pulled_on else score.band; elif report: report.band; else score.band
emi = derive.emi.derive_existing_emi(report.open_accounts if (kind=='full' and report) else [], declared=state.get('existing_emi_declared'), kind=kind)
foir = emi.used / income if (income := state.get('monthly_income')) and income > 0 else None
age_y, age_m = derive.range.age_years_and_months(identity.dob, deps.now().date()) if identity.dob else (None, None)
pins = derive.range.candidate_pincodes(identity, state.get('report'), kind=kind)
writes {'derived': Derived(band=..., emi=..., current_foir=foir, age_at_application_years=age_y, age_months_at_application=age_m, candidate_pincodes=pins), 'candidate_pincodes': pins, 'transcript': ['derive']}

--- pincode_gate ---  [code]
if not state['candidate_pincodes']:
    # quick with a valid PAN and no candidate pincode (also full with zero candidates — ASSUMPTION, same treatment):
    writes {'code': 'NO_PINCODE_ON_PAN', 'gated_lenders': [], 'transcript': [...]}  -> route_gate 'band_read' -> redact (band-read view, rank AND safety skipped)
rows = deps.repo.published_lenders_for_pincodes(state['candidate_pincodes'])   # PUBLISHED version only (FIXED)
gated, warnings = lenders.gate.gate_lenders(rows, state['candidate_pincodes'])  # drops lenders missing a required field
writes {'gated_lenders': gated, 'gate_warnings': warnings, 'errors': warnings, 'transcript': [...]}   # empty list allowed -> 9.4 one path (low / not_fundable_alone)

--- range ---  [code, 8.4/8.5]
nums = {L.id: derive.range.lender_numbers(L.criteria, lender_id=L.id, income=state.get('monthly_income'), existing_emi=state['derived'].emi.used, age_months=state['derived'].age_months_at_application) for L in state['gated_lenders']}
writes {'lender_numbers': nums, 'transcript': ['range']}

--- code_confidence ---  [code, 9.4]
cc = lenders.safety.compute_code_confidence(kind=kind, identity=..., derived=..., gated=..., numbers=..., openai_failed=False)
writes {'code_confidence': cc, 'transcript': [...]}

--- rank (THE one LLM node) ---
if not state['gated_lenders']: writes {'model_output': None, 'transcript': ['rank']}  # LLM skipped entirely; safety -> none_survived path
ctx = graph.llm.build_rank_context(state)   # 9.2 payload ONLY — see graph/llm.py; NEVER pan/name/mobile/aadhaar/account numbers/raw report/score number
attempt = state.get('rank_attempts', 0) + 1
try: out = deps.rank_model(ctx, config=config)   # ChatOpenAI temp 0, 8s timeout, max_retries=0, with_structured_output(ModelOutput, method='function_calling')
except Exception as e:
    if attempt >= RANK_MAX_ATTEMPTS:
        writes {'code': 'OPENAI_FAILURE', 'rank_attempts': attempt, 'rank_error': str(e)[:500], 'errors': [...], 'transcript': [...]}  -> route_rank 'failed' -> redact (band read + gated list fallback view, review queue at persist)
    writes {'rank_attempts': attempt, 'rank_error': str(e)[:500], 'transcript': [...]}  -> route_rank 'retry' (prior error rendered into the retry user message)
writes {'model_output': out, 'rank_attempts': attempt, 'rank_error': None, 'transcript': ['rank']}
# NOTE: out-of-gate / below-floor picks are NOT retried — safety demotes them (pinned by the PRD test
# 'model picks a lender outside the gate or below floor (demoted; none survive goes to review)').
# A raised confidence is NOT an error — final = min() ignores it (test 'model raises confidence (ignored)').

--- safety ---  [code, 9.3]
res = lenders.safety.run_safety(model_output=state['model_output'], gated=state['gated_lenders'], numbers=state['lender_numbers'], band=state['derived'].band, age_app_years=state['derived'].age_at_application_years, age_known=state['derived'].age_at_application_years is not None)
writes {'safety': res, 'transcript': ['safety']}

--- redact ---  [code; the ONLY LcView factory call]
final_conf = lenders.safety.final_confidence(state['code_confidence'], state.get('model_output'), safety=state.get('safety'), code=state.get('code'))
headline = lenders.safety.derive_headline(status='ok', band=state['derived'].band if state.get('derived') else None, final_conf=final_conf, safety=state.get('safety'), gated=state.get('gated_lenders'), code=state.get('code'), model_output=state.get('model_output'))
view = output.lc_view.build_lc_view(  # redaction boundary owned by the API section; imports its regexes, drops leaky lines
    check_id, kind, status=state.get('status') or ('pan_not_verified' if identity and not identity.pan_valid else 'ok'),
    code=state.get('code'), headline=headline, confidence=final_conf, label=_label(kind, income_given, emi_given, report_date),
    top=safety.top, alternates=safety.alternates, gated=gated_lenders, numbers=lender_numbers,  # names/rate_band/tenure labels resolved here
    coapplicant=(model_output.co_applicant_suggested if model_output else _code_coapplicant(band, safety)), coapplicant_reason=...,
    aadhaar_linked=identity.aadhaar_linked, items_to_confirm=_items(state),  # declared income/EMI items + per-account 'confirm with learner: an account last updated <date>'
    checked_on=..., from_cache=admission.mode_bureau=='reuse', refresh_available_on=admission.refresh_available_on,
    quick_checks_left_today=_quota_left(state))
# OPENAI_FAILURE special case (v1.1 §7/§10 'band read + gated list still shown'): top=None; alternates = up to 2
# band-floor-clearing gated lenders as code-built entries with rationale 'Automatic ranking unavailable; this lender
# serves the learner's area' and NO numbers; confidence low. Decision D-OPENAI below.
writes {'lc_view': view, 'final_confidence': final_conf, 'headline': headline, 'status': <as computed>, 'transcript': ['redact']}

--- persist ---
billed = any(c.billed for c in state['surepass_calls_made'])    # ground truth = call log (survives v1.0 C12 pattern)
counted = admission.counts_against_quota and state.get('bureau_called') and state.get('status') in (None, 'ok') and kind=='quick'
review = (state.get('safety') and state['safety'].review) or state.get('code') in ('OPENAI_FAILURE', 'PAN_DAILY_CAP') or (final_confidence=='low' and status=='ok')
deps.repo.finish_check(check_id, status=..., code=..., billed=..., counted_against_quota=counted, is_refresh=admission.is_refresh, consent_id=admission.consent_id, lc_view=..., internal=_internal_json(state), langsmith_run_id=...)
if review: deps.repo.insert_review_queue(check_id, pan_hash, reason=_review_reason(state))
deps.repo.release_pan_lock(pan_hash)        # in-flight lock only; the 10-min timeout lock stays (manual reconciliation)
if state.get('is_recompute'): finish_check updates the SAME row's lc_view/internal (no billing fields touched)
writes {'transcript': ['persist']}
_internal_json(state): identity/score/report/derived/lender_numbers/code_confidence/model_output/safety dumps + admission + errors + transcript + rank_attempts + langsmith_run_id. EXCLUDES pan, name, mobile, raw anything.


### eligibility_agent/derive/emi.py
_The exact v1.1 8.3 existing-EMI algorithm with per-account items, stale (>90d) exclusion, past-tenure flag, declared-vs-derived resolution. Pure, unit-tested, fully deterministic._

from __future__ import annotations
from datetime import date
from typing import Final, Sequence
from eligibility_agent.graph.state import EmiDerivation, EmiItem, Kind, OpenAccount

REVOLVING_EMI_PCT: Final[float] = 0.05
FALLBACK_TENURE_MONTHS: Final[int] = 36
PAST_TENURE_DIVISOR: Final[int] = 12

def months_since(open_date: date, anchor: date) -> int:
    """Calendar-month difference, day-agnostic (ASSUMPTION: v1.1 gives no day rule), clamped >= 0."""
    return max(0, (anchor.year - open_date.year) * 12 + (anchor.month - open_date.month))

def estimate_account_emi(a: OpenAccount, *, anchor: date) -> EmiItem:
    """Branches EXACTLY per 8.3 (stale handled by the caller; this never sees stale rows):
    1. scheduled_emi is a positive number -> reported.
    2. elif tenure_months (Repayment_Tenure if positive else Terms_Duration if positive — resolved by the adapter)
       and current_balance > 0:
         remaining = tenure_months - months_since(open_date or anchor, anchor)
         if remaining >= 1: estimate = current_balance / remaining            (method 'tenure_remaining')
         else:              estimate = current_balance / 12; past_tenure=True (method 'past_tenure',
                            confirm_item 'past tenure, confirm with learner')
    3. elif portfolio_type == 'R': estimate = 0.05 * current_balance          (method 'revolving_5pct')
    4. else: estimate = current_balance / 36                                  (method 'balance_36')
    rate_of_interest is NEVER used here (model context only)."""

def derive_existing_emi(accounts: Sequence[OpenAccount], *, declared: float | None, kind: Kind, anchor: date | None = None) -> EmiDerivation:
    """QUICK: accounts==[] always. used = declared if declared is not None else 0.0;
       used_source = 'declared' | 'declared_zero_assumed' (LcView label gains 'assuming no other EMIs');
       emi_floor_flag = declared is None (nothing known).
    FULL:
       anchor = report_date (passed by the derive node) or today.
       for a in accounts:
           if a.is_stale:  items.append(EmiItem(kind='stale_excluded', monthly_amount=0.0, method=None,
               confirm_item=f'confirm with learner: an account last updated {a.date_reported.isoformat()}')); continue
           item = estimate_account_emi(a, anchor=anchor)
           reported_total += item if reported else estimated_total += item
       derived_total = reported_total + estimated_total
       used = max(declared, derived_total) if declared is not None else derived_total   # if declared is higher, use declared; BOTH go to the model and safety
       used_source = 'declared' if (declared is not None and declared > derived_total) else 'derived'
       emi_floor_flag = any(item.kind in ('estimated','stale_excluded') or item.past_tenure for item in items)
       NTC report (no accounts): derived_total=0.0, emi_floor_flag=True (nothing known)."""

# WORKED EXAMPLE (full check; ReportDate anchor = 2026-10-01; all numbers exact):
# acct A: Scheduled_Monthly_Payment_Amount '12500', Date_Reported 2026-09-15 (16d, fresh)
#         -> branch 1 reported: +12,500.00 reported
# acct B: EMI '', Repayment_Tenure '0', Terms_Duration '024'->24, Open_Date 2025-06-01,
#         Current_Balance '160000', fresh -> months_since = (2026-2025)*12 + (10-6) = 16;
#         remaining = 24-16 = 8 >= 1 -> 160000/8 = +20,000.00 estimated (tenure_remaining)
# acct C: EMI '', Repayment_Tenure '12', Open_Date 2024-09-01, Balance '60000', fresh
#         -> months_since = 25; remaining = 12-25 = -13 < 1 -> 60000/12 = +5,000.00 estimated,
#         past_tenure=True, confirm_item 'past tenure, confirm with learner'
# acct D: EMI '', no tenure, Portfolio_Type 'R', Balance '80000', fresh -> 0.05*80000 = +4,000.00 (revolving_5pct)
# acct E: EMI '', no tenure, Portfolio_Type 'I', Balance '90000', fresh -> 90000/36 = +2,500.00 (balance_36)
# acct F: EMI '9000', Date_Reported 2026-05-20 -> (2026-10-01 - 2026-05-20) = 134 days > 90 -> STALE:
#         excluded from every sum; item 'confirm with learner: an account last updated 2026-05-20'
# => reported_total = 12,500.00; estimated_total = 31,500.00; derived_total = 44,000.00; emi_floor_flag = True
# declared 50,000 -> used = 50,000.00, used_source='declared' (both figures shown to model+safety)
# declared 30,000 -> used = 44,000.00, used_source='derived'
# quick, income given, no declared EMI -> used = 0.00, used_source='declared_zero_assumed', label 'assuming no other EMIs'


### eligibility_agent/derive/range.py
_v1.1 8.1/8.4/8.5/8.6 in one module per the PRD layout: band bucketing + NTC predicate, annuity range/ceiling/displayed-FOIR per lender, age at application and maturity, candidate pincodes. Pure functions, machine-verified worked examples._

from __future__ import annotations
import re
from datetime import date
from typing import Final
from eligibility_agent.graph.state import Band, Identity, Kind, LenderCriteria, LenderNumbers, ReportFacts, PINCODE_RE_STR

THREE_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{3}$")
PIN_RE: Final[re.Pattern[str]] = re.compile(PINCODE_RE_STR)
FOIR_EPS: Final[float] = 1e-9

# ---- 8.1 band ----
def is_ntc_score_value(credit_score: object) -> bool:
    """NTC ONLY when the (HTTP 200, success:true) response's credit_score is not a 3-digit string.
    REVERSES v1.0: success:false is a SERVICE FAILURE (client raises), never NTC."""
    return THREE_DIGIT_RE.fullmatch(str(credit_score).strip() if credit_score is not None else "") is None

def band_from_score(score: int | None) -> Band:
    # None -> 'NTC'; half-open buckets [300,650) [650,720) [720,780) [780,900]:
    # score < 650 -> '<650'; 650 <= s < 720 -> '650-720'; 720 <= s < 780 -> '720-780'; s >= 780 -> '780+'
    # Defensive (ASSUMPTION, surfaced): a 3-digit value < 300 maps to '<650', > 900 maps to '780+';
    # the adapter appends an internal 'score_out_of_range' error so the run is reviewable.

# ---- 8.5 age ----
def age_years_and_months(dob: date, on: date) -> tuple[int, int]:
    months = (on.year - dob.year) * 12 + (on.month - dob.month) - (1 if on.day < dob.day else 0)
    months = max(0, months)
    return months // 12, months

def age_at_maturity_years(age_months_at_application: int, tenure_max_months: int) -> int:
    return (age_months_at_application + tenure_max_months) // 12   # integer-month math, no float fuzz
# Example: dob 1998-04-12, today 2026-10-10 -> months = 28*12 + 6 - 1 = 341 -> age 28.
# Maturity at tenure 60: 341+60 = 401 months -> 33 years; max_age_at_maturity 35 (420 months) passes.
# No dob on the stored PAN record: age check skipped entirely; code_confidence capped at medium (8.5).

# ---- 8.6 candidate pincodes ----
def candidate_pincodes(identity: Identity, report: ReportFacts | None, *, kind: Kind) -> list[str]:
    """quick: [identity.pan_pincode] if set. full: pan zip UNION report.bureau_pincodes_12m
    (adapter already restricted those to OPEN accounts with Date_Reported within 365 days of
    CreditProfileHeader.ReportDate — closed or older accounts excluded — and to ^[1-9][0-9]{5}$).
    NOTE an account stale for EMI (>90d) still contributes pincodes if within 12 months.
    Returns sorted deduped list. The LC can never enter or override a pincode."""

# ---- 8.4 FOIR, range, ceiling ----
def available_emi(foir_limit: float, income: float, existing_emi: float) -> float:
    return max(0.0, foir_limit * income - existing_emi)   # 0.0 -> 'not fundable alone' with this lender

def annuity_loan_max(a_emi: float, annual_rate_pct: float, n_months: int) -> float:
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0.0: return a_emi * n_months
    f = (1.0 + r) ** n_months
    return a_emi * (f - 1.0) / (r * f)        # loan_max = A * ((1+r)^n - 1) / (r (1+r)^n)

def annuity_emi(principal: float, annual_rate_pct: float, n_months: int) -> float:
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0.0: return principal / n_months
    f = (1.0 + r) ** n_months
    return principal * r * f / (f - 1.0)

def floor_to_1000(v: float) -> float:
    return float(int(v // 1000.0) * 1000)      # display rounding only; never overstates (DECISION; v1.0 half-up NOT binding)

def lender_numbers(c: LenderCriteria, *, lender_id: str, income: float | None, existing_emi: float, age_months: int | None) -> LenderNumbers:
    """No income (quick without income): every money field None (no FOIR, no range, no ceiling) — headline+lender-list view only.
    Else (required criteria guaranteed known by the gate):
      A = available_emi(c.foir_limit, income, existing_emi)
      if A == 0.0: not_fundable_alone=True; bounds None; displayed_foir=None
      upper_raw = annuity_loan_max(A, c.rate_min_pct, c.tenure_max_months)   # lender MIN rate + MAX tenure
      lower_raw = annuity_loan_max(A, c.rate_max_pct, c.tenure_min_months)   # lender MAX rate + MIN tenure
      CLIP (one-sided; never raise a bound to loan_min — safety catches under-min ceilings):
        upper_c = min(upper_raw, c.loan_max); lower_c = min(max(lower_raw, c.loan_min), upper_c)
      ceiling = floor_to_1000(upper_c); lower = floor_to_1000(lower_c)
      displayed FOIR (DECISION resolving the 8.4 wording — the inverse-identity reading):
        P_wc = min(annuity_loan_max(A, c.rate_max_pct, c.tenure_max_months), c.loan_max)   # 'loan_max at max rate, max tenure'
        displayed_foir = (existing_emi + annuity_emi(P_wc, c.rate_max_pct, c.tenure_max_months)) / income
        == foir_limit EXACTLY when unclipped (annuity_emi(annuity_loan_max(A,r,n),r,n) == A, machine-verified),
        < foir_limit when the lender's loan_max clips. The rejected reading (re-price the min-rate ceiling at the
        max rate) makes displayed_foir > foir_limit for EVERY lender with a rate band, so 9.3's 'FOIR at the
        ceiling at or below the limit' would demote every pick — flagged as a v1.1 contradiction, see prd_gaps.
      age_at_maturity_years = (age_months + c.tenure_max_months)//12 when age known else None"""

# WORKED EXAMPLES (machine-verified, exact):
# L1: foir_limit 0.50, rates 11-14%, tenure 12-60m, loan range [50,000, 500,000]; income 60,000; existing_emi 10,000
#   A = 0.50*60,000 - 10,000 = 20,000.00
#   upper_raw = 20,000*((1+0.11/12)^60 - 1)/((0.11/12)(1+0.11/12)^60) = 919,860.6767 -> CLIPPED to 500,000 (clipped='upper')
#   lower_raw = 20,000*AF(14%,12) = 222,749.1040 -> >= 50,000, <= upper -> floor -> 222,000
#   ceiling = 500,000; range [222,000, 500,000]
#   displayed FOIR: P_wc = min(annuity_loan_max(20,000, 14, 60) = 859,540.3294, 500,000) = 500,000;
#     annuity_emi(500,000, 14, 60) = 11,634.1254; displayed = (10,000 + 11,634.1254)/60,000 = 0.360569
# L2: foir 0.55, rates 12.5-16%, tenure 24-48m, loans [100,000, 2,000,000]; income 60,000; emi 10,000
#   A = 23,000.00; upper_raw = 865,312.3009 -> 865,000; lower_raw = 469,741.3983 -> 469,000 (no clip)
#   displayed FOIR = (10,000 + 23,000)/60,000 = 0.550000 exactly (identity; passes 9.3 with FOIR_EPS)
# Zero headroom: income 30,000, emi 16,000, foir 0.50 -> A = max(0, 15,000-16,000) = 0.0 -> not fundable alone (this lender)
# Quick with income 50,000, no EMI (used=0, 'assuming no other EMIs'), L1 criteria:
#   A = 25,000; upper_raw = 1,149,825.85 -> clip 500,000; lower_raw = 278,436.38 -> 278,000
# r=0 degenerate (admin enters 0%): loan_max = A*n (branch pinned).


### eligibility_agent/lenders/gate.py
_Pincode gate (9.1) over PUBLISHED lender criteria; excludes lenders missing a required field until fixed. Repo does the pincode join; gate re-filters and validates._

from __future__ import annotations
from typing import Final, Sequence, TypedDict, Any
from eligibility_agent.graph.state import GatedLender, LenderCriteria
from eligibility_agent.derive.range import PIN_RE

class PublishedLenderRow(TypedDict):   # what Repo.published_lenders_for_pincodes returns (db section)
    id: str
    name: str
    criteria: dict[str, Any]           # normalised jsonb incl. field_status
    matched_pincodes: list[str]

# Required for matching: both range/safety formulas need these. min_income / age limits are OPTIONAL
# (checks skipped when absent). 'floor' is satisfied by score_floor known OR ntc_accepted known.
REQUIRED_FOR_MATCHING: Final[tuple[str, ...]] = (
    "foir_limit", "rate_min_pct", "rate_max_pct", "tenure_min_months", "tenure_max_months", "loan_min", "loan_max")

def is_matchable(c: LenderCriteria) -> tuple[bool, list[str]]:
    """missing = [f for f in REQUIRED_FOR_MATCHING if c value is None or field_status[f] == 'missing']
    + ['score_floor'] when score_floor is None AND ntc_accepted is None.
    'uncertain' required fields stay IN (ASSUMPTION): usable value, but the field names are passed to the
    model as uncertain_fields and cap that lender's contribution to confidence at medium via code_confidence.
    Returns (len(missing)==0, missing)."""

def gate_lenders(rows: Sequence[PublishedLenderRow], pincodes: Sequence[str]) -> tuple[list[GatedLender], list[str]]:
    """pins re-filtered defensively through PIN_RE (SURVIVES v1.0). A lender stays in iff >=1 candidate pincode
    is in its published approved list (repo join already guarantees; re-asserted). Rows failing
    LenderCriteria.model_validate or is_matchable are SKIPPED with a warning string
    'lender <id> excluded: missing <fields>' (-> state.gate_warnings/errors; never aborts the check).
    Builds rate_band_label f'{rate_min:g} to {rate_max:g}%' and tenure_label f'{tmin} to {tmax} months'.
    Output sorted by lender id (deterministic). Never silently skip the gate (FIXED 9.1)."""


### eligibility_agent/lenders/safety.py
_Safety check (9.3), code confidence (9.4) decision table, final = min(code, model), and the headline decision table. All code; the LLM never sees this logic._

from __future__ import annotations
from typing import Final, Sequence
from eligibility_agent.graph.state import (Band, BAND_ORDER, Confidence, CONFIDENCE_ORDER, Code, Derived,
    GatedLender, Headline, Identity, Kind, LenderNumbers, ModelOutput, Pick, SafetyBreach, SafetyResult)
from eligibility_agent.derive.range import FOIR_EPS

# ---- 9.3 safety ----
def check_pick(pick: Pick, lender: GatedLender, nums: LenderNumbers, *, band: Band, age_known: bool, age_app_years: int | None) -> SafetyBreach | None:
    """Rules in order; first breach wins (detail strings are internal-only, may contain numbers):
    0. not_in_gate: pick.lender_id has no GatedLender (model invented / outside gate) -> demote.
    1. band_floor: band == 'NTC' -> breach unless criteria.ntc_accepted is True.
       else breach when criteria.score_floor is not None and BAND_ORDER[band] < BAND_ORDER[band_of(score_floor)]
       (floor converted to its bucket via band_from_score; 'at or above the floor' = band bucket >= floor bucket).
    2. foir_limit: nums.displayed_foir is not None and nums.displayed_foir > criteria.foir_limit + FOIR_EPS
       (holds by construction post-range; catches recompute drift and declared-figure changes). Skipped when
       displayed_foir is None (no income) — confidence already capped.
    3. age_bounds (ONLY when dob known): age_app_years < min_age_years (if set) OR
       nums.age_at_maturity_years > max_age_at_maturity_years (if set) -> breach. dob unknown -> skipped.
    4. min_loan_amount: nums.ceiling is not None and criteria.loan_min is not None and nums.ceiling < criteria.loan_min -> breach.
       Also nums.not_fundable_alone (available_emi == 0.0) -> breach via rule 4 detail 'no EMI headroom'."""

def run_safety(model_output: ModelOutput | None, gated: Sequence[GatedLender], numbers: dict[str, LenderNumbers], *, band: Band, age_app_years: int | None, age_known: bool) -> SafetyResult:
    """candidates = ([model_output.recommendation] if set else []) + model_output.alternates (model_output None ->
    no candidates -> none_survived=True, review=True). Dedupe by lender_id ('duplicate' breach). Each candidate
    checked; breaches -> demoted list; survivors keep model order: first survivor = top, next <=2 = alternates.
    none_survived = no survivor; review = none_survived or any demotion occurred... review=True ONLY when
    none_survived OR the original recommendation was demoted (alternate-only demotions logged, not review).
    top_clears_floor_by_one: for the surviving top, BAND_ORDER[band] - BAND_ORDER[floor_bucket] >= 1
    (None when floor unknown / NTC-accepted path)."""

# ---- 9.4 code confidence (decision table; computed BEFORE rank, re-capped after safety) ----
def compute_code_confidence(*, kind: Kind, identity: Identity, derived: Derived, gated: Sequence[GatedLender], numbers: dict[str, LenderNumbers], openai_failed: bool) -> Confidence:
    """| # | condition                                                                 | result |
    | L1 | no lender survives the gate (gated == [])                                  | low    |
    | L2 | openai_failed (model invalid twice) — applied at redact                    | low    |
    | L3 | safety demoted every pick (none_survived) — applied at redact              | low    |
    | H  | ALL of: identity.pan_valid; kind == 'full'; EMI fully reported             | high   |
    |    |   (emi.estimated_total == 0 and no stale_excluded and no past_tenure item);         |
    |    | no declared-only figure governs the EMI (emi.used_source == 'derived')     |        |
    |    |   [ASSUMPTION: income is always declared in this product, so 'declared-only'|        |
    |    |    is read as the EMI figure — flagged];                                   |        |
    |    | dob known (age check ran);                                                 |        |
    |    | no 'uncertain' required criteria field on any gated lender;                |        |
    |    | at least one gated lender whose floor the band clears by >= 1 bucket       |        |
    |    |   (BAND_ORDER[band] - BAND_ORDER[floor_bucket] >= 1)                       |        |
    | M  | anything else (quick checks ALWAYS land here or lower; unknown age;        | medium |
    |    |  estimated/stale EMI; declared EMI override; floor cleared by exactly 0)   |        |
    Post-safety cap: if the SURVIVING top pick's floor is cleared by 0 buckets (top_clears_floor_by_one is False),
    high is capped to medium inside final_confidence."""

def final_confidence(code_conf: Confidence, model_output: ModelOutput | None, *, safety: SafetyResult | None, code: Code | None) -> Confidence:
    """low when code == 'OPENAI_FAILURE' or (safety and safety.none_survived) or code_conf == 'low'.
    else base = code_conf, capped to medium when safety.top_clears_floor_by_one is False.
    final = min(base, model_output.confidence) by CONFIDENCE_ORDER when model_output present —
    the model may only LOWER; a raised value is ignored by min() (test-pinned)."""

# ---- headline (code; v1.1 §3/§9.4 display matrix) ----
def derive_headline(*, status: str, band: Band | None, final_conf: Confidence | None, safety: SafetyResult | None, gated: Sequence[GatedLender] | None, code: Code | None, model_output: ModelOutput | None) -> Headline | None:
    """| condition (first match wins)                                            | headline            |
    | status != 'ok' (refused / pan_not_verified / pending / service_failure) | None                |
    | code == 'NO_PINCODE_ON_PAN' (band-read): band NTC or '<650'             | 'unlikely' (+coapp) |
    |   band >= '650-720'                                                     | 'possible'          |
    | final low AND (gated == [] OR every demotion was band_floor/no-headroom)| 'not_fundable_alone'|
    |   (coapplicant_suggested forced True; review queue — 9.4 'one path')    |                     |
    | final low otherwise (OPENAI_FAILURE, mixed demotions)                   | 'under_review'      |
    | band == 'NTC' and a pick survived (NTC-accepting lender)                | 'unlikely' (+coapp, |
    |   (§10: 'Unlikely alone, co-applicant suggested' + NTC lenders)         |  picks still shown) |
    | final high and safety.top is not None                                   | 'fundable'          |
    | final medium and safety.top is not None                                 | 'possible'          |
    | fallback (top None at medium/high — unreachable, defensive)             | 'under_review'      |
    Display matrix (9.4, enforced by redact/build_lc_view): high -> lender + ceiling shown ('Recommended');
    medium -> lender + ceiling + items_to_confirm ('Possible, confirm with learner'); low -> headline ONLY,
    top_lender/alternates null/empty, run in the review queue — EXCEPT code OPENAI_FAILURE (see decision D-OPENAI:
    gated-list fallback entries without numbers) and the NTC-survivor row above (ASSUMPTION, flagged)."""


### eligibility_agent/graph/llm.py
_The ONE LLM seam: RankContext (exact 9.2 payload), RankModel protocol (ctx + config -> ModelOutput), production factory with method pin / 8s timeout / zero SDK retries, context builder. FakeRankModel in testing/ implements the same protocol._

from __future__ import annotations
from typing import Final, Protocol
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict
from eligibility_agent.graph.state import Band, CheckState, Confidence, Kind, ModelOutput
from eligibility_agent.graph.prompts import PromptBundle, render_rank_user

OPENAI_TIMEOUT_S: Final[float] = 8.0        # v1.1 FIXED per-call timeout

class LenderContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lender_id: str                           # id only — names resolved in redact, never sent to the model
    ceiling: float | None                    # code-derived; the model repeats NO numbers
    range_lower: float | None
    range_upper: float | None
    not_fundable_alone: bool
    age_at_application: int | None
    age_at_maturity: int | None
    band_clears_floor: bool | None           # pre-computed; floor VALUE never sent (it is a score number)
    ntc_accepted: bool | None
    foir_limit: float | None
    min_income: float | None
    tenure_label: str                        # '12 to 60 months'
    rate_band_label: str                     # '11 to 14%'
    policy_notes: str | None
    uncertain_fields: list[str]

class RankContext(BaseModel):
    """EXACTLY the 9.2 payload. NEVER present: PAN, name, mobile, Aadhaar, account numbers, raw report,
    raw score number, bureau name, pincodes."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Kind
    band: Band                               # bucket label only, never the raw score
    code_confidence: Confidence              # the given value the model may only lower
    foir: float | None                       # current FOIR (used EMI / income)
    emi_floor_flag: bool
    existing_emi_reported: float | None      # split reported vs estimated, with flags
    existing_emi_estimated: float | None
    emi_past_tenure_accounts: int
    emi_stale_accounts: int
    declared_income: float | None
    declared_emi: float | None               # shown alongside derived when both exist
    aadhaar_linked: bool | None
    recent_enquiries_180d: int | None
    past_due_flag: bool | None               # report.past_due_total > 0
    stale_account_flag: bool | None
    lenders: list[LenderContext]             # surviving gate only
    attempt: int
    prior_error: str | None                  # retry feedback (exception text, truncated 500 chars)

class RankModel(Protocol):                   # the DI seam; FakeRankModel implements the same __call__
    def __call__(self, ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput: ...

def build_rank_context(state: CheckState) -> RankContext: ...   # pure mapping from state; unit-tested to assert the NEVER-list is structurally impossible (RankContext has no such fields)

def make_rank_model(*, model: str, prompts: PromptBundle, timeout: float = OPENAI_TIMEOUT_S) -> RankModel:
    """chat = ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=0)   # NO automatic retries anywhere (v1.1 FIXED); the ONE retry on invalid output lives in the rank node
    runnable = chat.with_structured_output(ModelOutput, method='function_calling')     # explicit method pin SURVIVES v1.0 C6 (library default moved to json_schema; function_calling keeps constraint violations as Pydantic parse exceptions the node retry absorbs)
    def _call(ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput:
        msgs = [SystemMessage(content=prompts.rank_system(ctx.kind)), HumanMessage(content=render_rank_user(ctx))]
        out = runnable.invoke(msgs, config=config)      # config threads the trace -> LLM child run nests under the rank node
        if not isinstance(out, ModelOutput): raise TypeError('structured output returned wrong type')
        return out
    return _call"""
# Retry semantics (node contract): retry ONLY on raised exceptions (timeout, network, schema-invalid parse,
# wrong type). Business violations (pick outside gate / below floor) are NOT retried — safety demotes them.
# Raised confidence is NOT an error — final_confidence min() ignores it. 2nd failure -> code OPENAI_FAILURE,
# review queue, band read + gated-list fallback still returned (decision D-OPENAI).


### eligibility_agent/graph/prompts.py
_Pinned literal system prompt for the single rank node (LangSmith Hub 'hero/eligibility-rank' pulled by tag with repo-local fallback — Hub loader pattern SURVIVES v1.0), tier addenda, deterministic user renderer. Brace-free._

PROMPTS_LOCAL_VERSION: Final[str] = '2026-10-10-a'
HUB_RANK: Final[str] = 'hero/eligibility-rank'   # pinned by tag per release; PROMPT_SOURCE=local default (tests/CI never pull)

RANK_SYSTEM_CORE: Final[str] = (
"You advise Hero Vired Learning Consultants (LCs) on education-loan fundability. You receive code-computed facts "
"about one applicant and a list of lenders that already passed a location gate and policy checks. Your job is to "
"rank those lenders, explain the choice in plain spoken language, and suggest a co-applicant when that is the "
"honest path.\n\n"
"Hard rules:\n"
"1. Rank only the lenders given, by lender_id. Never invent, rename, merge or add a lender. Never suggest "
"overriding or working around the location gate or any policy check.\n"
"2. You never produce a number. Every amount, ceiling, range, rate and score is computed by code and shown "
"separately. rationale and conditions must contain no digits taken from the credit file, no credit score, no "
"score range, and no credit agency's name. Use band-driven plain words: strong repayment record, existing EMIs "
"look high next to income, limited credit history.\n"
"3. Prefer co_applicant_suggested true over an empty recommendation when the credit history is new or thin, when "
"the band sits below every lender's comfort, or when EMI headroom is thin. A co-applicant cannot fix location "
"eligibility: with no lenders listed, return recommendation null and say options will be reviewed.\n"
"4. confidence: you are given code_confidence. You may return the same value or a lower one, never a higher one. "
"Lower it when facts are estimated, declared-only, stale or conflicting, and name why in review_flags.\n"
"5. conditions are short actionable items an LC can say aloud (for example: PAN to Aadhaar link needed; confirm "
"declared income with a payslip). rationale is one or two short sentences.\n"
"6. Indian context: amounts are INR, income is monthly, FOIR is existing monthly EMI divided by monthly income; "
"lower is better.")
# brace-free; stored as the single Hub system message, composed locally with one tier addendum:
QUICK_ADDENDUM: Final[str] = ("Quick check: figures are declared by the learner, not verified. If income was not "
"given, no amounts exist; rank on band and criteria fit only.")
FULL_ADDENDUM: Final[str] = ("Full check: facts come from a fresh credit file plus declared income. Estimated or "
"stale EMI rows and past-tenure accounts are flagged; treat them as items to confirm, and lower confidence when "
"they drive the decision.")

@dataclass(frozen=True)
class PromptBundle:
    rank_system_quick: str; rank_system_full: str; version: str; source: Literal['local','hub']
    def rank_system(self, kind: Kind) -> str: ...

def local_prompts() -> PromptBundle: ...   # compose(CORE, addendum)
def load_prompts(settings) -> PromptBundle: ...   # SURVIVES v1.0: Client.pull_prompt(f'{HUB_RANK}:{tag}'), any exception -> local_prompts(); never in tests

def render_rank_user(ctx: RankContext) -> str:
    """Deterministic, pure, unit-tested. Layout:
    Applicant:
    - credit band: {band}            (bucket label; internal literal never copied into LC text by the model per rule 2)
    - code confidence given: {code_confidence}
    - monthly income declared: {amount | 'no'}; declared existing EMI: {amount | 'none'}
    - existing EMI from file: reported {r} + estimated {e}; floor flag {yes/no}; past-tenure accounts {n}; stale accounts {n}
    - FOIR: {foir:.2f | 'unknown'}; age now {n|'unknown'}; PAN-Aadhaar linked: {yes/no/unknown}
    - recent credit enquiries (180d): {n|'unknown'}; overdue amounts on file: {yes/no}; 
    Lenders that passed the gate — rank ONLY these lender_ids:
    - lender_id {id} | ceiling INR {int|'none'} | range {lo}-{hi|'none'} | not fundable alone: {y/n} |
      band clears floor: {yes/no/ntc-accepted} | FOIR limit {x} | min income {x|'-'} | tenure {label} | rate {label} |
      age at maturity {n|'unknown'} | notes: {policy_notes|'-'} | uncertain fields: {csv|'none'}
    {if attempt > 1}: Your previous output was invalid: {prior_error}. Return a corrected ModelOutput.
    NOTE: numbers appear in the INPUT for grounding; the OUTPUT schema has no numeric fields except rank —
    'the model never produces a number' is structural, and redact scrubs every outbound string anyway."""


### eligibility_agent/surepass/client.py
_DELTAS to the v1.0 tools/surepass.py design (which otherwise SURVIVES: envelope parsing both shapes, kill-switch exact literal 'true' pre-I/O, CallLogSink one-row-per-attempt, FakeSurePassClient scenario machinery, endpoint constants live HERE, zero retries)._

MOVED: tools/surepass.py -> surepass/client.py (v1.1 layout). Everything in docs/design/design-surepass-client-adapters-lender-mat.md §surepass.py survives EXCEPT:

1. PER-ENDPOINT TIMEOUTS (v1.1 FIXED, replaces 5s/60s):
   PAN_TIMEOUT_S: Final[float] = 8.0; SCORE_TIMEOUT_S: Final[float] = 12.0; REPORT_TIMEOUT_S: Final[float] = 20.0
   TIMEOUT_BY_ENDPOINT: Final[dict[str, float]] = {EP_PAN_COMPREHENSIVE: 8.0, EP_EXPERIAN_SCORE: 12.0, EP_EXPERIAN_REPORT: 20.0}
   _post passes timeout=httpx.Timeout(t, connect=min(5.0, t)) per request (httpx timeouts are per-phase, not
   wall-clock — documented limitation, risk noted).

2. NEW EXCEPTION: class SurePassTimeout(SurePassTransportError) raised on httpx.TimeoutException (connect/read/
   write/pool); other httpx.HTTPError stays SurePassTransportError. The graph needs the distinction: a BUREAU
   timeout locks the PAN 10 minutes + pending_verification; other failures are plain service_failure.
   One sink row either way (http_status=None).

3. TAXONOMY REVERSAL (v1.1 §8.1/§10 FIXED — REVERSES the v1.0 ASSUMPTION): interpret_response now RAISES
   SurePassUpstreamError(endpoint, f'success_false:{message_code}') on ANY parsed envelope with success == false,
   on ALL THREE endpoints (v1.1 failure table lists 'success: false' under service failure for PAN, Score and
   Report alike). success:false is NEVER NTC and NEVER pan_invalid. Consequences:
   - NON_BUREAU_FAILURE_CODES hook DELETED (superseded — every success:false already raises).
   - Adapters only ever see success:true results; their success-false branches are removed.
   - Fixtures: score_ntc_envelope variants 'success_false' is no longer an NTC vector — it becomes a
     service-failure vector; NTC vectors are 200+success:true with credit_score '' / null / '-1' / 'NH' /
     missing key (any non-3-digit string).
   - SurePassValidationError ({errors,message} shape) still raises (unchanged); 401/403/429/5xx unchanged.

4. consent body field: experian_score/experian_report send consent:'Y' ONLY because the admission guard already
   wrote a consent record for this pull (section 5 rule 1); the client API is unchanged, the comment is updated.

5. FakeSurePassClient: constructor/scenario semantics survive (dict scenarios resolve through the SAME
   interpret_response); add scenario type 'SurePassTimeout' (an exception instance works already) and update
   canned scenarios for the reversed NTC rule. FakeCall(endpoint, check_id, body) unchanged.


### eligibility_agent/surepass/adapters.py
_DELTAS to the v1.0 adapters design (parsing helpers parse_amount/parse_int_sentinel/parse_rate/parse_yyyymmdd/parse_iso_date/as_list/dig/resolve_tenure_months all SURVIVE verbatim). New NTC predicate, no gender, per-account OpenAccount rows for the 8.3 engine, open-account 12-month pincodes, raw models in passthrough mode._

MOVED: tools/adapters.py -> surepass/adapters.py. Deltas from the committed v1.0 design:

1. NO GENDER ANYWHERE: _map_gender deleted; Identity has no gender field; Experian takes none — never derive or send one.

2. pan_to_identity(result) -> Identity: unchanged gate (status=='valid' and category=='person'; success is
   guaranteed true — client raises otherwise, so pan_valid = status/category gate only). Strict-bool aadhaar_linked,
   zip via PIN_RE, dob ISO, masked_aadhaar never modeled — all survive.

3. RAW PASSTHROUGH MODELS (v1.1 8.2): class RawAccount(BaseModel): model_config = ConfigDict(extra='allow');
   every used field declared as str (every number arrives as a string, '' = not reported, dates YYYYMMDD).
   Same for RawReport/RawScore envelopes' data. Adapters validate raw -> convert to typed facts.

4. score_to_score_facts(result, *, pulled_on: date) -> ScoreFacts:
   raw = str(dig(result.data, 'credit_score') or '')
   is_ntc = not re.fullmatch(r'[0-9]{3}', raw.strip())      # v1.1 8.1 VERBATIM; the FCIREXScore fallback is DROPPED
   score = int(raw) if not is_ntc else None; band = band_from_score(score)
   # 3-digit out-of-range defensive mapping per derive/range.band_from_score (ASSUMPTION, internal error noted)
   source='score_endpoint'. (bureau_pull builds the report-sourced ScoreFacts itself.)

5. report_to_report_facts(result, *, today: date | None = None) -> ReportFacts:
   anchor = parse_yyyymmdd(CreditProfileHeader.ReportDate) or today or date.today()
   is_ntc = credit_score not a 3-digit string (same predicate; 'no file' -> NTC ReportFacts with empty accounts,
   stored/cached as NTC for 30 days). Open predicate SURVIVES: Date_Closed=='' OR Account_Status=='11'.
   For each OPEN account build OpenAccount:
     account_key=f'acct_{i}' (Account_Number never carried); portfolio_type 'I'/'R' else None;
     scheduled_emi=parse_amount(Scheduled_Monthly_Payment_Amount) with values <= 0 treated as None
     ('positive number' per 8.3; '0' is NOT a reported EMI — DELTA from v1.0 which counted 0.0 as reported);
     current_balance=max(0.0, parse_amount(Current_Balance) or 0.0);
     tenure_months=resolve_tenure_months (Repayment_Tenure if positive else Terms_Duration if positive;
       zero_is_unknown; Terms_Frequency not in ('', 'M') -> None — survives);
     rate_of_interest=parse_rate (context only); open_date/date_reported=parse_yyyymmdd;
     amount_past_due=parse_amount or 0.0;
     pincodes=[z for addr rows if PIN_RE.fullmatch(z)];
     is_stale=(date_reported is None) or (anchor - date_reported).days > 90   # 8.2 staleness guard
   past_due_total: sum over ALL accounts (open+closed). written_off_or_settled: count non-empty
   Written_off_Settled_Status. enquiries_30/90/180 from TotalCAPS_Summary (parse_int_sentinel or 0).
   bureau_pincodes_12m (v1.1 8.6 DELTA from v1.0: OPEN accounts only, was all accounts):
     sorted({z for a in OPEN accounts if a.date_reported and (anchor - a.date_reported).days <= 365 for z in a.pincodes})
   CAIS_Holder_Details DOB: NEVER read (survives; contradictory-DOB fixture trap kept).
   DROPPED from facts (not consumed by v1.1 nodes): max_dpd_24m, total_outstanding, existing_emi_* aggregates
   (the 8.3 engine now works from per-account rows); Payment_History_Profile retained only in raw.


### eligibility_agent/graph/tracing.py
_SURVIVES v1.0 VERBATIM — new_run_id / base_tags / invoke_config (thread_id=check_id, tags tier:/lc:, metadata {check_id, pan_hash, tier} never PII, recursion_limit, run_id) / finish_trace (patches admission mode, billed, counted via Client.update_run, resending base tags, try/except swallow). v1.1 additions: metadata gains 'admission' mode and 'billed' flag naming per §11 ('tier, admission mode, billed flag'); tags gain 'admission:<reuse|call>'. Reference docs/design/design-graph-state-prompts-langsmith-wirin.md §tracing.py._

Unchanged API: new_run_id(enabled: bool) -> UUID | None; invoke_config(*, kind, check_id, lc_id, pan_hash, run_id) -> RunnableConfig; finish_trace(client, *, run_id, kind, lc_id, check_id, pan_hash, final_state) -> None. finish_trace metadata now {check_id, pan_hash, tier, admission_pan, admission_bureau, billed, counted_against_quota} from AdmissionFacts + persist outputs. Everything else verbatim from the committed v1.0 design.

## Interfaces provided
- **name**: graph/state.py — all shared models (leaf module) | **definition**: Kind, Band('<650'|'650-720'|'720-780'|'780+'|'NTC'), BAND_ORDER, Headline, CheckStatus('ok'|'pan_not_verified'|'refused'|'pending_verification'|'service_failure'), Code (13 values), Confidence+CONFIDENCE_ORDER, PINCODE_RE_STR, StateUpdate; models ConsentInfo, Identity (NO gender), ScoreFacts, OpenAccount, ReportFacts, EmiItem, EmiDerivation, LenderCriteria(+field_status), GatedLender, Derived, LenderNumbers, Pick, ModelOutput (PRD-verbatim), SafetyBreach, SafetyResult, AdmissionFacts, SurePassCallAudit; TypedDict CheckState (total=False). Imports stdlib+pydantic only; every other package imports from here.
- **name**: build_graph / build_recompute_graph / GraphDeps | **definition**: GraphDeps(repo: Repo, surepass: SurePassClient, rank_model: RankModel, prompts: PromptBundle, now: Callable[[], datetime]) frozen; build_graph(kind, deps, *, checkpointer=None) wires admission->pan_comprehensive->bureau_pull->derive->pincode_gate->range->code_confidence->rank->safety->redact->persist (+persist_refusal, terminal persist paths); build_recompute_graph(kind, deps, *, checkpointer=None) enters at derive. Constants RANK_MAX_ATTEMPTS=2, RECURSION_LIMIT=40, NODE_NAMES. Routers route_admission/route_pan/route_bureau/route_gate/route_rank are pure exported functions.
- **name**: graph/llm.py — the ONE LLM seam | **definition**: class RankModel(Protocol): __call__(ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput. RankContext/LenderContext frozen pydantic (exact 9.2 payload; structurally cannot carry PAN/name/mobile/aadhaar/account numbers/raw report/score number/pincodes/lender names). make_rank_model(*, model, prompts, timeout=8.0) -> RankModel (temperature 0, max_retries=0, with_structured_output(ModelOutput, method='function_calling')). build_rank_context(state) -> RankContext. OPENAI_TIMEOUT_S=8.0.
- **name**: derive/emi.py | **definition**: months_since(open_date, anchor) -> int; estimate_account_emi(a: OpenAccount, *, anchor) -> EmiItem; derive_existing_emi(accounts, *, declared, kind, anchor=None) -> EmiDerivation. Constants REVOLVING_EMI_PCT=0.05, FALLBACK_TENURE_MONTHS=36, PAST_TENURE_DIVISOR=12. Worked-example constants pinned for tests: reported 12,500 / estimates 20,000+5,000+4,000+2,500 / derived_total 44,000 / stale account excluded with its confirm_item.
- **name**: derive/range.py | **definition**: is_ntc_score_value(v) -> bool; band_from_score(score|None) -> Band; age_years_and_months(dob, on) -> tuple[int,int]; age_at_maturity_years(age_months, tenure_max) -> int; candidate_pincodes(identity, report|None, *, kind) -> list[str]; available_emi(foir_limit, income, emi) -> float; annuity_loan_max(A, rate_pct, n) -> float; annuity_emi(P, rate_pct, n) -> float; floor_to_1000(v) -> float; lender_numbers(criteria, *, lender_id, income, existing_emi, age_months) -> LenderNumbers; FOIR_EPS=1e-9; PIN_RE. Pinned test constants: 919,860.6767 / 222,749.1040 / 865,312.3009 / 469,741.3983 / 859,540.3294 / EMI(500000,14,60)=11,634.1254 / displayed FOIR 0.360569 and 0.550000.
- **name**: lenders/gate.py | **definition**: PublishedLenderRow(TypedDict: id, name, criteria jsonb, matched_pincodes); REQUIRED_FOR_MATCHING tuple; is_matchable(c: LenderCriteria) -> tuple[bool, list[str]]; gate_lenders(rows, pincodes) -> tuple[list[GatedLender], list[str]] (sorted by id, warnings for excluded lenders, published-version rows only — repo guarantees).
- **name**: lenders/safety.py | **definition**: check_pick(pick, lender, nums, *, band, age_known, age_app_years) -> SafetyBreach|None; run_safety(model_output|None, gated, numbers, *, band, age_app_years, age_known) -> SafetyResult; compute_code_confidence(*, kind, identity, derived, gated, numbers, openai_failed) -> Confidence (9.4 table); final_confidence(code_conf, model_output|None, *, safety, code) -> Confidence (= min, model can only lower); derive_headline(*, status, band, final_conf, safety, gated, code, model_output) -> Headline|None (pinned decision table).
- **name**: surepass/client.py (deltas over surviving v1.0 contract) | **definition**: SurePassClient protocol unchanged (calls_enabled; pan_comprehensive(check_id, pan); experian_score/report(check_id, *, name, mobile, pan)); endpoint constants EP_* live here (binding). NEW: SurePassTimeout(SurePassTransportError); TIMEOUT_BY_ENDPOINT {PAN:8.0, SCORE:12.0, REPORT:20.0}; interpret_response raises SurePassUpstreamError on every success:false envelope (NON_BUREAU_FAILURE_CODES removed). FakeSurePassClient/FakeCall/CallLogSink survive.
- **name**: surepass/adapters.py | **definition**: pan_to_identity(result) -> Identity (no gender); score_to_score_facts(result, *, pulled_on) -> ScoreFacts; report_to_report_facts(result, *, today=None) -> ReportFacts (OpenAccount rows with is_stale >90d, positive-EMI-only reported, 12-month open-account bureau_pincodes_12m, enquiries 30/90/180, holder DOB never read); RawAccount/RawReport pydantic passthrough models (extra='allow', all-string leaves); surviving helpers re-exported.
- **name**: graph/prompts.py | **definition**: PromptBundle(rank_system_quick, rank_system_full, version, source) with rank_system(kind); RANK_SYSTEM_CORE/QUICK_ADDENDUM/FULL_ADDENDUM literals (brace-free); HUB_RANK='hero/eligibility-rank' pulled by tag with local fallback; render_rank_user(ctx: RankContext) -> str deterministic.
- **name**: Initial-state contract for main.py | **definition**: Routes seed: kind, check_id, lc_id, lead_id, opportunity_id, opportunity_stage, pan (upper), name, mobile, pan_hash, mobile_hash, monthly_income, existing_emi_declared, refresh, acknowledge_ntc_warning, consent: ConsentInfo, is_recompute=False, rank_attempts=0, bureau_called=False, pan_called=False, surepass_calls_made=[], errors=[], transcript=[], langsmith_run_id=str(run_id)|None; invoke with tracing.invoke_config (thread_id=check_id); read ONLY final_state['lc_view']. Recompute route additionally seeds identity/score/report facts re-adapted from stored raw plus the new income/EMI.

## Interfaces consumed
- **name**: policy/guards.py (policy section) | **expectation**: admit(kind, repo, *, lc_id, pan_hash, mobile_hash, opportunity_id, opportunity_stage, refresh, acknowledge_ntc_warning, consent: ConsentInfo, live_enabled: bool, now: datetime) -> AdmissionFacts (the state.py model, or a structurally identical frozen type). The guard — not graph nodes — owns: consent-row insert per pull, per-PAN-hash in-flight lock acquire (IN_FLIGHT), 10-minute timeout-lock honoring (PAN_LOCKED), stage gate (STAGE_TOO_EARLY), windows PAN 180d / score 30d / report 30d, quick quota 5/LC/IST-day with per-LC override (QUOTA_EXHAUSTED), per-PAN hard cap 3 bureau calls/day (PAN_DAILY_CAP, locks PAN + review flag), report window (REPORT_WINDOW_OPEN), NTC warning (WARN_NTC_RECENT unless acknowledged, from a quick NTC result <30d), PAN-change rules (PAN_CHANGE_NEEDS_MANAGER), kill switch (LIVE_CALLS_DISABLED only when a billed call would be needed; cache hits pass). Also: lock_pan(pan_hash, minutes=10, reason) and release_pan_lock(pan_hash) callable from nodes; constants PAN_WINDOW_DAYS=180, SCORE_WINDOW_DAYS=30, REPORT_WINDOW_DAYS=30, HARD_CAP_PER_PAN_PER_DAY=3, QUICK_QUOTA_DEFAULT=5; HMAC-SHA256 hashing of PAN and mobile with the server-side key lives in config/policy, single definition, called before graph invoke.
- **name**: db/repo.py Repo protocol (db section) | **expectation**: get_pan_record(pan_hash) / put_pan_record(pan_hash, raw_ref, derived, fetched_at) [180d]; get_score_cache(pan_hash) -> ScoreFacts-shaped row / put_score_cache(...) [30d]; get_report_cache(pan_hash) -> raw ref + derived / put_report_cache(...) [30d] — report cache keyed per PAN hash, shared across opportunities; store_raw(endpoint_kind, pan_hash, opportunity_id, raw) -> object key into the private encrypted Supabase Storage bucket (raw NEVER returns into CheckState on the hot path; recompute route reads it back explicitly); published_lenders_for_pincodes(pincodes) -> list[PublishedLenderRow] (published version only, >=1 pincode match, matched_pincodes aggregated, ordered by id, [] on empty input without querying); insert_surepass_call(*, check_id, endpoint, surepass_client_id, http_status, success, message_code) (the CallLogSink — client-owned logging, one row per HTTP attempt, nullable fields); finish_check(check_id, *, status, code, billed, counted_against_quota, is_refresh, consent_id, lc_view, internal, langsmith_run_id) with terminal-guard (only updates a running row); insert_review_queue(check_id, pan_hash, reason); MemoryRepo implements everything for the no-network suite.
- **name**: output/lc_view.py (API/redaction section) | **expectation**: Single redaction home (binding): BUREAU_TERM_RE (incl. bureaus?) and STANDALONE_SCORE_RE with the digit/separator lookarounds, imported by tests+evals. build_lc_view(...) accepting my redact-node outputs (status, code, headline, confidence, label, top: Pick|None, alternates, gated lenders for id->name/rate_band/tenure resolution, numbers for ceiling/range, coapplicant fields, aadhaar_linked, items_to_confirm, checked_on, from_cache, refresh_available_on, quick_checks_left_today) -> LcView exactly per v1.1 §6 (top_lender/alternates null when confidence low EXCEPT the D-OPENAI fallback; range/ceiling only when income entered; 'assuming no other EMIs' label variant; pan_aadhaar_link_warning only on literal False). build_refusal_view(check_id, kind, code, quota_left, refresh_available_on) and build_failure_view(check_id, kind, status, code, message) for the terminal paths. NOTE: the §3 full-check label literal 'bureau report on <date>' collides with BUREAU_TERM_RE — see prd_gaps; suggest 'credit report on <date>, income as declared'.
- **name**: config.py (config section) | **expectation**: Settings.openai_model default 'gpt-4.1'; surepass_live exact-literal 'true' property (binding); PROMPT_SOURCE/PROMPT_TAG; hash_pan_hmac(pan, *, key) and hash_mobile_hmac(mobile, *, key) HMAC-SHA256 with a server-side key held outside the database, last-4 stored separately — single definition, used by routes before invoke; per-endpoint timeout constants may live with the client (my choice: client module owns TIMEOUT_BY_ENDPOINT).
- **name**: main.py routes (API section) | **expectation**: POST /checks/quick and /checks/full seed the initial-state contract and invoke the matching compiled graph via run_in_threadpool with tracing.invoke_config; POST /checks/{id}/recompute loads the stored raw payloads from the bucket, re-adapts them through surepass.adapters, seeds facts + new income/EMI + is_recompute=True, and invokes build_recompute_graph(kind) — no SurePass call, no quota; POST /checks/{id}/attach touches no graph; GET /health -> {ok, surepass_live}. Error boundary, auth, 422 handling per the API section.
- **name**: evals/tests sections | **expectation**: no_leak / lenders_gated / numbers_from_code / confidence_not_raised / billing_consistency / outcome_matches_band evaluators consume: NODE_NAMES transcripts, LenderNumbers exact values (my pinned constants), SafetyResult subsets, final = min() invariant, FakeSurePassClient.calls vs checks-row flags. FakeRankModel implements RankModel(ctx, *, config) and must be able to return out-of-gate picks, below-floor picks, raised confidence, and schema-invalid outputs (raise) to drive the pinned misbehavior tests. Fixtures: NTC vectors are 200+success:true with non-3-digit credit_score; success:false vectors assert service_failure.
- **name**: langchain-openai 1.6.x / langgraph 1.2.x (pinned, verified in audit-technical-feasibility) | **expectation**: ChatOpenAI(model, temperature=0, timeout: float, max_retries=0) and .with_structured_output(schema, method='function_calling') keep current semantics; StateGraph(CheckState) add_node/add_conditional_edges/compile(checkpointer=...), config injection by param name 'config', Annotated operator.add reducers; PostgresSaver(pool) manual-pool pattern; LANGGRAPH_STRICT_MSGPACK allowlist includes eligibility_agent.graph.state.

## Decisions
- **topic**: Displayed FOIR interpretation (8.4) | **decision**: displayed_foir = (existing_emi + annuity_emi(P_wc, rate_max, tenure_max)) / income where P_wc = min(annuity_loan_max(available_emi, rate_max, tenure_max), lender.loan_max). By the inverse identity this equals foir_limit exactly when unclipped and is below it when clipped. | **rationale**: The naive reading (re-price the min-rate-derived ceiling at the max rate) yields displayed_foir > foir_limit for EVERY lender with a non-degenerate rate band (machine-verified: 0.5752 vs limit 0.55), so 9.3's 'FOIR at the ceiling at or below the limit' would demote every pick — the two sections would be mutually unsatisfiable. The chosen reading matches the words 'emi(loan_max at max rate, max tenure)' (loan_max evaluated AT max rate) and keeps 9.3 satisfiable; flagged in prd_gaps for owner confirmation.
- **topic**: NTC predicate and client taxonomy reversal | **decision**: is_ntc = (HTTP 200 and success:true and credit_score does not fullmatch ^[0-9]{3}$). interpret_response raises SurePassUpstreamError on ANY success:false envelope on all three endpoints; NON_BUREAU_FAILURE_CODES hook deleted; FCIREXScore fallback dropped; fixtures' success_false NTC variant becomes a service-failure vector. | **rationale**: v1.1 8.1 and the §10 failure table are explicit and FIXED ('anything else is a service failure, not NTC'; 'success: false' listed under service failure for PAN, Score and Report) — this reverses the v1.0 dossier ASSUMPTION and removes the 30-day NTC mis-cache risk the v1.0 hook existed for.
- **topic**: Scheduled_Monthly_Payment_Amount '0' handling | **decision**: A parsed EMI value <= 0 is treated as NOT reported (falls to the estimation branches), unlike v1.0 which counted '0' as a reported 0.0. | **rationale**: v1.1 8.3 says 'if Scheduled_Monthly_Payment_Amount is a positive number, use it' — positive is explicit; a zero EMI on an open account with a balance is a data artifact, and estimating is the conservative (higher-EMI) path.
- **topic**: months_since definition | **decision**: months_since(open_date, anchor) = max(0, (anchor.year-open.year)*12 + (anchor.month-open.month)), day-agnostic; anchor = CreditProfileHeader.ReportDate (fallback: today passed by the node). | **rationale**: 8.3 names no day rule; calendar-month difference is deterministic, reproducible on cache re-derivation (the ReportDate anchor survives v1.0's verified design), and at worst shifts one estimate by one month's denominator.
- **topic**: Display rounding | **decision**: Ceiling and both range bounds are floored to the nearest 1,000 for display (floor_to_1000); raw floats kept internally for safety checks with FOIR_EPS. | **rationale**: v1.1 specifies no rounding; flooring never overstates what a learner could borrow (learner protection), and the v1.0 half-up helper was not in the binding-rulings list. numbers_from_code evaluator compares against these exact code-derived values.
- **topic**: Clip semantics | **decision**: upper = min(upper_raw, loan_max); lower = min(max(lower_raw, loan_min), upper). A bound is never raised above what the applicant can afford; a ceiling below the lender's loan_min is left as-is and demoted by safety rule 4. | **rationale**: Clamping a bound UP to loan_min would fabricate affordability; 9.3's explicit 'ceiling at or above the lender's minimum loan amount' check only makes sense if ceilings below the minimum can exist.
- **topic**: OPENAI_FAILURE view (D-OPENAI) | **decision**: On the second invalid model output: code=OPENAI_FAILURE, confidence low, headline from band, top_lender null, and alternates filled with up to two code-selected gated lenders whose floor the band clears (no ceilings/ranges, rationale 'Automatic ranking unavailable; this lender serves the learner's area'), review-queue row written. | **rationale**: v1.1 §7/§10 require 'band read + gated list still shown' while §9.4 says low confidence shows headline only — they conflict, and LcView has no other lender-list field. Alternates-without-numbers honors the explicit OPENAI_FAILURE clause as the narrower, later rule; flagged in prd_gaps.
- **topic**: Band-read headline table (NO_PINCODE_ON_PAN and quick-without-income) | **decision**: Band-read: NTC or <650 -> 'unlikely' (+co-applicant nudge for NTC); 650-720/720-780/780+ -> 'possible'. Full pipeline: high->'fundable', medium->'possible', NTC-with-surviving-NTC-lender -> 'unlikely'+coapplicant with picks still shown, low -> 'not_fundable_alone' (gate-empty or all band/headroom demotions, coapplicant forced true) else 'under_review'. | **rationale**: v1.1 names the five headline values but no mapping; never saying 'fundable' without a gated+safety-passed lender is the conservative completion; §10's NTC row ('Unlikely alone, co-applicant suggested' plus NTC-accepting lenders) pins the NTC override. ASSUMPTION, surfaced.
- **topic**: Full check with zero candidate pincodes | **decision**: Treated identically to the quick-check rule: code NO_PINCODE_ON_PAN, band-read view, rank and safety skipped, review-queue row. | **rationale**: v1.1 defines the behavior only for quick; a full check where neither PAN nor any open recent account yields a valid pincode cannot run the gate, and 9.1 forbids silently skipping it. ASSUMPTION, surfaced.
- **topic**: 'No declared-only figure' in the 9.4 high row | **decision**: Read as governing the EMI figure: high requires emi.used_source=='derived' with zero estimated/stale/past-tenure items. Declared income does not by itself block high (it is the only income source the product has). | **rationale**: Income is declared and mandatory on every full check, so the literal reading makes 'high' unreachable; the EMI reading preserves a reachable high tier consistent with 'EMI fully reported' appearing in the same sentence. ASSUMPTION, surfaced in prd_gaps.
- **topic**: Floor-clearance recheck after safety | **decision**: code_confidence computes 'high' pre-rank using the best-clearing gated lender; final_confidence caps to medium when the SURVIVING top pick clears its floor by zero buckets (safety records top_clears_floor_by_one). | **rationale**: 9.4's 'band clears the floor by at least one bucket' is lender-relative but code_confidence runs before the pick exists (node order is FIXED); the two-phase check makes the rule hold for the lender actually recommended without reordering nodes.
- **topic**: Rank retry trigger | **decision**: The single retry fires only on raised exceptions (timeout, network, schema-invalid structured output). Picks outside the gate or below floors are NOT retried — safety demotes them; a raised confidence is ignored by min(), not retried. | **rationale**: v1.1's own test list pins both behaviors ('model picks a lender outside the gate or below floor (demoted; none survive goes to review)'; 'model raises confidence (ignored)'), and §9.2 reserves the retry for invalid output.
- **topic**: LLM construction | **decision**: ChatOpenAI(model=settings.openai_model default 'gpt-4.1', temperature=0, timeout=8.0, max_retries=0).with_structured_output(ModelOutput, method='function_calling'); ONE compiled seam (RankModel protocol: ctx + config -> ModelOutput); fake implements the same protocol. | **rationale**: v1.1 FIXED: 8s OpenAI timeout and 'no automatic retries anywhere' forces max_retries=0 (delta from v1.0's 1); the explicit method pin survives v1.0 C6 (library default moved to json_schema; function_calling keeps constraint violations as parse exceptions the node retry absorbs). Verified in the feasibility audit against langchain-openai 1.6.x.
- **topic**: Recompute as a second compiled graph | **decision**: build_recompute_graph(kind) compiles derive->pincode_gate->range->code_confidence->rank->safety->redact->persist; the route re-adapts STORED raw payloads into facts and seeds them with the new income/EMI and is_recompute=True; persist updates the same checks row and touches no billing fields. | **rationale**: v1.1 §7 names the sub-graph explicitly; re-adapting stored raw (rather than reusing stored derived) guarantees 'derivations always re-run on current inputs and current published criteria' including adapter fixes, with zero SurePass spend.
- **topic**: Uncertain criteria fields | **decision**: A required field marked 'missing' excludes the lender from matching; 'uncertain' keeps it in but the field names ride to the model as uncertain_fields and block the 9.4 high tier. | **rationale**: §11 only mandates exclusion for missing required fields; excluding on 'uncertain' would let one hedged admin entry silently remove a lender, while confidence-capping surfaces the doubt. ASSUMPTION, surfaced.
- **topic**: Band on dual-source facts | **decision**: derive uses report.band when report_date >= score.pulled_on, else score.band ('whichever is newer' per 8.1); a Report pull writes a report-sourced ScoreFacts row so later quick checks are free and consistent. | **rationale**: Directly implements 8.1 and FIXED rule 3 (Report refreshes the score cache); comparing the two dates is the only deterministic 'newer' available.
- **topic**: Out-of-range 3-digit scores | **decision**: A 3-digit credit_score below 300 maps to '<650', above 900 to '780+', with an internal 'score_out_of_range' error recorded (never shown). | **rationale**: The NTC predicate is FIXED on 'not a 3-digit string', so 299 or 950 cannot be NTC; mapping to the nearest bucket keeps the run serviceable while the error makes it auditable. ASSUMPTION, surfaced.
- **topic**: PAN-endpoint timeout does not lock the PAN | **decision**: Only Score/Report timeouts trigger the 10-minute PAN lock + pending_verification; a PAN Comprehensive timeout is a plain service_failure ('Couldn't verify PAN, retry'). | **rationale**: §5 rule 12 names 'Timeout on Score or Report' for the lock, and §10's PAN row prescribes the retryable message with no lock; the Rs-1 PAN call carries negligible double-spend risk.
- **topic**: Reused v1.0 dossier assets (named) | **decision**: Reused verbatim: make_postgres_checkpointer (manual pool + session pooler + checkpointer schema + strict msgpack), graph/tracing.py API, typed-closure bind, kill-switch exact literal + pre-I/O raise, parse_surepass_body/interpret_response envelope handling (minus the success:false return), FakeSurePassClient scenario machinery, parsing helpers (parse_amount/parse_int_sentinel/parse_rate/parse_yyyymmdd/parse_iso_date/as_list/dig/resolve_tenure_months), Band literals + BAND_ORDER + PINCODE_RE_STR, open-account predicate, holder-DOB prohibition + fixture trap, redaction regexes single-home in output/lc_view.py, thread_id=check_id, prompt Hub pull-by-tag with local fallback. | **rationale**: These were verified in the v1.0 audits against live library sources and are unaffected by the v1.1 deltas; re-deriving them would only risk drift from the binding rulings.

## Risks
- The displayed-FOIR safety check passes by construction after range (equality within FOIR_EPS); its real value is only on recompute drift and future refactors — if an implementer 'optimizes' range or safety independently the invariant can silently invert. The pinned worked examples (0.550000 exact, 0.360569 clipped) must be test-locked.
- httpx timeouts are per-phase, not wall-clock: a pathological slow-drip response could exceed the 8/12/20s budget without raising. Acceptable for Phase 1; p95 SLOs would surface it. Document in the client.
- Band literal strings ('650-720', '720-780') contain 3-digit score-range numbers; they are internal-only, but any accidental copy of a Band value into an LC-facing string must be caught — STANDALONE_SCORE_RE does catch them (test vector 'below 650' drops), so every LC string must flow through the scrubber, including items_to_confirm built from EMI items.
- Recompute re-adapts STORED raw payloads: an adapter bug fix changes historical check results on recompute. That is the intended 'current derivations' semantics, but ops should know a recompute is not a replay.
- The reversed success:false taxonomy means an Experian 'no-hit' that actually arrives as success:false (shape still unverified — NTC probe is an open item) would surface as SUREPASS_FAILURE instead of NTC until the probe lands. This is PRD-mandated but will look like an outage for genuinely thin files if SurePass uses success:false for no-hits.
- OpenAccount rows are built only for open accounts; if the §12 code tables later reveal Account_Status values beyond '11'/'13' that mean open, the open predicate (Date_Closed=='' OR status=='11') may misclassify — mitigated by the OR and raw retention for re-adaptation.
- Pick.rank is a model-produced int (allowed: it is ordering metadata, not a money/score number); the no_leak evaluator must not flag it, and redact must never render rank into LC strings.
- Float money math at INR magnitudes is exact enough in double precision; all cross-section comparisons must use the floored display values or pytest.approx on raws — mixing the two in tests will flake.

## PRD gaps
- 8.4 vs 9.3 contradiction: 'displayed FOIR = (existing_emi + emi(loan_max at max rate, max tenure))/income' read as re-pricing the min-rate ceiling at the max rate ALWAYS exceeds foir_limit (machine-verified 0.5752 vs 0.55), so 9.3's 'FOIR at the ceiling at or below the limit' would demote every pick. Implemented the inverse-identity reading (displayed FOIR == foir_limit when unclipped); owner should confirm.
- §7/§10 'OPENAI_FAILURE: band read + gated list still shown' conflicts with §9.4 'low confidence: headline only, no lender named' and with the LcView schema having no plain lender-list field. Implemented code-built alternates without numbers for this code only (D-OPENAI); needs sign-off.
- §3 quick-check output 'lenders serving the pincode whose floor the band clears' (without income) has no LcView home other than top_lender/alternates with null ceilings; implemented that way — confirm the LcView shape intends it.
- §3/§6 full-check label literal 'bureau report on <date>, income as declared' contains the word 'bureau', which the binding redaction regex (BUREAU_TERM_RE incl. bureaus?) drops — the PRD's own label fails its own no_leak evaluator. Proposed label 'credit report on <date>, income as declared'; needs owner choice.
- §9.4 'no declared-only figure' makes 'high' unreachable if it covers income (income is always declared in this product; FulCRM holds none). Read as governing the EMI figure only; confirm.
- Headline mapping (fundable/possible/unlikely/not_fundable_alone/under_review) is never defined as a function of band/confidence/safety; the decision table in lenders/safety.py is an ASSUMPTION needing product sign-off, especially the band-read rows for NO_PINCODE_ON_PAN.
- Full check with zero candidate pincodes is unspecified (8.6 defines sources, 9.1 only the quick behavior); implemented the quick NO_PINCODE_ON_PAN treatment — confirm.
- A 3-digit credit_score outside [300,900] (e.g. '299', '950') is neither NTC (it IS a 3-digit string) nor inside any bucket; mapped to the nearest bucket with an internal error — confirm, ties into the open NTC-probe item.
- months_since(Open_Date) has no day-precision rule; calendar-month difference pinned.
- Lender floors are entered as Experian scores but lenders quote CIBIL floors (v1.1 §12 open item, Credit to resolve); the safety band_floor check converts the stored floor via band_from_score and will inherit whatever number Credit enters — no code mitigation possible.
- §9.3 'demotes the pick' does not say whether alternates promote into the top slot; implemented promotion in model order — confirm.
- 'EMI fully reported' high-tier condition vs the NTC report (no accounts): implemented emi_floor_flag=True for NTC files ('nothing is known', carried over from v1.0 decision) — confirm.
