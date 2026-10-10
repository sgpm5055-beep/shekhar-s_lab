"""Graph wiring (PRD v1.1 section 7): ``GraphDeps``, the typed-closure ``bind``, pure routers, ``build_graph(kind)``, the
recompute sub-graph and its seeding, the initial-state seed, and the Postgres checkpointer factory (K21).

Wiring (one linear deterministic graph per kind; the kinds share every node, which branch on ``state['kind']``)::

    START -> admission --refused--> persist_refusal -> END
                       --proceed--> pan_comprehensive --invalid--> redact
                                                      --failed---> persist                      (PAN service failure)
                                                      --valid----> bureau_pull --terminal--> persist (timeout / failure)
                                                                               --ok-------> derive
    derive -> pincode_gate --band_read--> redact                                     (NO_PINCODE_ON_PAN: K15)
                           --proceed----> range -> code_confidence --skip_rank--> redact   (zero gated lenders)
                                                                   --rank-------> rank
    rank --retry--> rank (ONE retry) | --failed--> redact (OPENAI_FAILURE, K7) | --safety--> safety -> redact
    redact -> persist -> END

Zero gated lenders OR a band read skip rank AND safety (T10 ruling). No node carries a retry policy: there are no
automatic retries anywhere (PRD 5.12); the rank node's single business retry is an explicit edge. The recompute
sub-graph (K9) enters at ``derive`` on facts that ``seed_recompute`` re-adapted from the stored raw objects.

Checkpointer (K21, v1.0 verbatim, re-pathed): a manually built psycopg pool (``autocommit=True``,
``prepare_threshold=None``, ``dict_row``), the Supabase SESSION pooler (:5432, never the transaction pooler :6543), a
dedicated ``checkpointer`` schema reached through the role's ``search_path``, a post-setup assertion that
``to_regclass('public.checkpoints') IS NULL``, and strict msgpack: ``LANGGRAPH_STRICT_MSGPACK=true`` in the environment
AND an explicit allowlist of the ``eligibility_agent.graph.state`` model classes on the saver's serializer. Tests never
connect; ``thread_id = check_id`` comes from ``tracing.invoke_config``.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel

from eligibility_agent.db.repo import CheckRow, RawUnavailable, Repo
from eligibility_agent.graph import nodes
from eligibility_agent.graph import state as state_module
from eligibility_agent.graph.llm import RankModel
from eligibility_agent.graph.nodes import RANK_MAX_ATTEMPTS
from eligibility_agent.graph.prompts import PromptBundle
from eligibility_agent.graph.state import (
    AdmissionDecision,
    CheckState,
    ConsentInfo,
    Kind,
    ReportFacts,
    ScoreFacts,
    StateUpdate,
)
from eligibility_agent.graph.tracing import RECURSION_LIMIT
from eligibility_agent.policy.guards import RecomputeOutcome, admit_recompute, ist_date
from eligibility_agent.surepass.adapters import (
    identity_from_envelope,
    report_facts_from_envelope,
    score_facts_from_envelope,
)
from eligibility_agent.surepass.client import SurePassClient

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

__all__ = [
    "NODE_NAMES",
    "RANK_MAX_ATTEMPTS",
    "RECOMPUTE_NODE_NAMES",
    "RECURSION_LIMIT",
    "STATE_MODULE",
    "CheckpointerConfigError",
    "CompiledGraph",
    "GraphDeps",
    "assert_checkpointer_isolated",
    "bind",
    "build_graph",
    "build_recompute_graph",
    "check_checkpointer_dsn",
    "make_checkpoint_serde",
    "make_postgres_checkpointer",
    "new_check_state",
    "route_admission",
    "route_bureau",
    "route_confidence",
    "route_gate",
    "route_pan",
    "route_rank",
    "seed_recompute",
    "state_msgpack_allowlist",
]

NODE_NAMES: Final[tuple[str, ...]] = (
    "admission",
    "persist_refusal",
    "pan_comprehensive",
    "bureau_pull",
    "derive",
    "pincode_gate",
    "range",
    "code_confidence",
    "rank",
    "safety",
    "redact",
    "persist",
)
RECOMPUTE_NODE_NAMES: Final[tuple[str, ...]] = (
    "derive",
    "pincode_gate",
    "range",
    "code_confidence",
    "rank",
    "safety",
    "redact",
    "persist",
)
_KINDS: Final[frozenset[str]] = frozenset({"quick", "full"})

CompiledGraph: TypeAlias = CompiledStateGraph[CheckState, None, CheckState, CheckState]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class GraphDeps:
    """Everything a node may touch beyond state (K18). Built once at startup (prod) or per test (fakes); never smuggled
    through ``config['configurable']``."""

    repo: Repo
    surepass: SurePassClient
    rank_model: RankModel  # the ONE LLM seam; FakeRankModel implements the same protocol
    prompts: PromptBundle  # the version is recorded in checks.internal; the rank model already holds the text
    now: Callable[[], datetime] = field(default=_utcnow)  # tz-aware UTC; tests inject FrozenClock.now


# --------------------------------------------------------------------------- typed-closure bind (v1.0 C7, verbatim)

NodeImpl: TypeAlias = Callable[[CheckState, RunnableConfig, GraphDeps], StateUpdate]


class NodeFn(Protocol):
    """A bound node as LangGraph sees it: NAMED ``state`` and ``config`` parameters (a positional-only ``Callable``
    alias would not match LangGraph's ``_NodeWithConfig`` protocol)."""

    def __call__(self, state: CheckState, config: RunnableConfig) -> StateUpdate: ...


def bind(fn: NodeImpl, deps: GraphDeps) -> NodeFn:
    """Close a node over its deps. The second parameter MUST be named ``config``: LangGraph injects it by name."""

    def node(state: CheckState, config: RunnableConfig) -> StateUpdate:
        return fn(state, config, deps)

    node.__name__ = getattr(fn, "__name__", "node")
    return node


_NODES: Final[Mapping[str, NodeImpl]] = {
    "admission": nodes.admission,
    "persist_refusal": nodes.persist_refusal,
    "pan_comprehensive": nodes.pan_comprehensive,
    "bureau_pull": nodes.bureau_pull,
    "derive": nodes.derive,
    "pincode_gate": nodes.pincode_gate,
    "range": nodes.range_,
    "code_confidence": nodes.code_confidence,
    "rank": nodes.rank,
    "safety": nodes.safety,
    "redact": nodes.redact,
    "persist": nodes.persist,
}


# --------------------------------------------------------------------------- routers (pure; no deps)


def route_admission(state: CheckState) -> Literal["refused", "proceed"]:
    """Keys off the decision's code (refusal decisions carry 'reuse' plans; plans are never consulted here). No decision
    is never a licence to call out."""
    decision = state.get("admission")
    if decision is None or decision.code is not None or decision.admission is None:
        return "refused"
    return "proceed"


def route_pan(state: CheckState) -> Literal["invalid", "failed", "valid"]:
    if state.get("status") == "service_failure":
        return "failed"  # PAN service failure -> persist (no bureau call, no quota)
    identity = state.get("identity")
    return "valid" if identity is not None and identity.pan_valid else "invalid"  # invalid -> redact


def route_bureau(state: CheckState) -> Literal["terminal", "ok"]:
    return "terminal" if state.get("status") in ("pending_verification", "service_failure") else "ok"


def route_gate(state: CheckState) -> Literal["band_read", "proceed"]:
    return "band_read" if state.get("code") == "NO_PINCODE_ON_PAN" else "proceed"


def route_confidence(state: CheckState) -> Literal["skip_rank", "rank"]:
    """Zero gated lenders: nothing to rank, so the LLM AND safety are skipped (low / not_fundable_alone path)."""
    return "rank" if state.get("gated_lenders") else "skip_rank"


def route_rank(state: CheckState) -> Literal["retry", "failed", "safety"]:
    if state.get("model_output") is not None:
        return "safety"
    if state.get("code") == "OPENAI_FAILURE":
        return "failed"
    if state.get("rank_attempts", 0) < RANK_MAX_ATTEMPTS:
        return "retry"
    return "failed"  # defensive: never a third call


# --------------------------------------------------------------------------- build


def _wire_from_derive(graph: StateGraph[CheckState, None, CheckState, CheckState]) -> None:
    graph.add_edge("derive", "pincode_gate")
    graph.add_conditional_edges("pincode_gate", route_gate, {"band_read": "redact", "proceed": "range"})
    graph.add_edge("range", "code_confidence")
    graph.add_conditional_edges("code_confidence", route_confidence, {"skip_rank": "redact", "rank": "rank"})
    graph.add_conditional_edges("rank", route_rank, {"retry": "rank", "failed": "redact", "safety": "safety"})
    graph.add_edge("safety", "redact")
    graph.add_edge("redact", "persist")
    graph.add_edge("persist", END)


def _require_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"unknown check kind {kind!r}")


def build_graph(kind: Kind, deps: GraphDeps, *, checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledGraph:
    """Compile the quick or full graph. ``checkpointer`` None in unit tests; the K21 PostgresSaver in production."""
    _require_kind(kind)
    graph: StateGraph[CheckState, None, CheckState, CheckState] = StateGraph(CheckState)
    for name in NODE_NAMES:
        graph.add_node(name, bind(_NODES[name], deps))  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge(START, "admission")
    graph.add_conditional_edges("admission", route_admission, {"refused": "persist_refusal", "proceed": "pan_comprehensive"})
    graph.add_edge("persist_refusal", END)
    graph.add_conditional_edges(
        "pan_comprehensive", route_pan, {"invalid": "redact", "failed": "persist", "valid": "bureau_pull"}
    )
    graph.add_conditional_edges("bureau_pull", route_bureau, {"terminal": "persist", "ok": "derive"})
    _wire_from_derive(graph)
    # compile's checkpointer parameter is typed with an unparameterised BaseCheckpointSaver (Unknown) upstream.
    return graph.compile(checkpointer=checkpointer, name=f"eligibility-{kind}")  # pyright: ignore[reportUnknownMemberType]


def build_recompute_graph(
    kind: Kind, deps: GraphDeps, *, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledGraph:
    """The recompute sub-graph: derive -> ... -> persist on facts seeded by ``seed_recompute`` (no admission, no
    SurePass node exists in it, so zero calls is structural)."""
    _require_kind(kind)
    graph: StateGraph[CheckState, None, CheckState, CheckState] = StateGraph(CheckState)
    for name in RECOMPUTE_NODE_NAMES:
        graph.add_node(name, bind(_NODES[name], deps))  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge(START, "derive")
    _wire_from_derive(graph)
    return graph.compile(  # pyright: ignore[reportUnknownMemberType]  (see build_graph)
        checkpointer=checkpointer, name=f"eligibility-{kind}-recompute"
    )


# --------------------------------------------------------------------------- initial state + recompute seeding


def new_check_state(
    *,
    kind: Kind,
    check_id: str,
    lc_id: str,
    lead_id: str | None,
    opportunity_id: str | None,
    opportunity_stage: str | None,
    pan: str,
    name: str,
    mobile: str,
    pan_hash: str,
    mobile_hash: str,
    monthly_income: float | None,
    existing_emi_declared: float | None,
    refresh: bool,
    acknowledge_ntc_warning: bool,
    consent: ConsentInfo | None,
    langsmith_run_id: str | None,
) -> CheckState:
    """The initial-state contract for a quick / full run (the route computes the hashes via ``policy.guards`` first).
    Counters start at zero; nothing derived is seeded (admission computes the last-4 values itself, K15)."""
    state: CheckState = {
        "kind": kind,
        "check_id": check_id,
        "lc_id": lc_id,
        "opportunity_id": opportunity_id,
        "opportunity_stage": opportunity_stage,
        "pan": pan,
        "name": name,
        "mobile": mobile,
        "pan_hash": pan_hash,
        "mobile_hash": mobile_hash,
        "monthly_income": monthly_income,
        "existing_emi_declared": existing_emi_declared,
        "refresh": refresh,
        "acknowledge_ntc_warning": acknowledge_ntc_warning,
        "consent": consent,
        "is_recompute": False,
        "rank_attempts": 0,
        "rank_error": None,
        "bureau_called": False,
        "pan_called": False,
        "surepass_calls_made": [],
        "errors": [],
        "transcript": [],
        "langsmith_run_id": langsmith_run_id,
    }
    if lead_id is not None:
        state["lead_id"] = lead_id
    return state


def seed_recompute(
    deps: GraphDeps,
    *,
    original_check_id: str,
    new_check_id: str,
    monthly_income: float | None,
    existing_emi_declared: float | None,
    langsmith_run_id: str | None = None,
) -> tuple[RecomputeOutcome, CheckState | None]:
    """K9: admit a recompute (``policy.admit_recompute`` R1-R5) and, when admitted, insert the NEW checks row and return
    the sub-graph's initial state. ``(outcome, None)`` for every refusal (the route maps them to 404 / 409).

    Facts are re-adapted on CURRENT adapter code from the stored raw objects the CURRENT cache rows point at (K5). The
    synthetic admission (pan / bureau plans 'reuse', counts False, is_refresh False, the cache row's display dates, the
    original consent) means redact and persist never branch on a missing decision. The new row: ``recompute_of`` =
    the original, the original's consent_id, billed / counted False; NO consent row, NO locks, NO quota, NO SurePass.
    """
    repo = deps.repo
    original = repo.get_check(original_check_id)
    if original is None:
        return admit_recompute(None, None, None, False, False), None
    kind = original.kind
    pan_row = repo.get_pan_record(original.pan_hash)
    score_row = repo.get_score_cache(original.pan_hash) if kind == "quick" else None
    report_row = repo.get_report_cache(original.pan_hash) if kind == "full" else None
    bureau_present = score_row is not None if kind == "quick" else report_row is not None
    outcome = admit_recompute(original.status, original.code, kind, pan_row is not None, bureau_present)
    if outcome != "ok" or pan_row is None:
        return outcome, None
    score: ScoreFacts | None = None
    report: ReportFacts | None = None
    try:
        identity = identity_from_envelope(repo.fetch_raw(pan_row.storage_key))
        if score_row is not None:
            score = score_facts_from_envelope(
                repo.fetch_raw(score_row.storage_key), pulled_on=ist_date(score_row.fetched_at), source=score_row.source
            )
            fetched_at, expires_at = score_row.fetched_at, score_row.expires_at
        elif report_row is not None:
            report = report_facts_from_envelope(
                repo.fetch_raw(report_row.storage_key), today=ist_date(report_row.fetched_at)
            )
            fetched_at, expires_at = report_row.fetched_at, report_row.expires_at
        else:  # unreachable: admit_recompute already required the bureau row
            return "raw_unavailable", None
    except RawUnavailable:
        return "raw_unavailable", None

    now = deps.now()
    repo.insert_check(
        CheckRow(
            id=new_check_id,
            kind=kind,
            lc_id=original.lc_id,
            lead_id=original.lead_id,
            opportunity_id=original.opportunity_id,
            opportunity_stage=original.opportunity_stage,
            contact_id=original.contact_id,
            pan_hash=original.pan_hash,
            pan_last4=original.pan_last4,
            mobile_hash=original.mobile_hash,
            mobile_last4=original.mobile_last4,
            status="running",
            code=None,
            billed=False,
            counted_against_quota=False,
            is_refresh=False,
            recompute_of=original.id,
            consent_id=original.consent_id,
            lc_view=None,
            internal=None,
            langsmith_run_id=langsmith_run_id,
            created_at=now,
        )
    )
    decision = AdmissionDecision(
        admission="cache_hit",
        code=None,
        pan_plan="reuse",
        bureau_plan="reuse",
        counts_against_quota=False,
        is_refresh=False,
        checked_on=ist_date(fetched_at),
        refresh_available_on=ist_date(expires_at),
        original_consent_id=original.consent_id,
    )
    state: CheckState = {
        "kind": kind,
        "check_id": new_check_id,
        "lc_id": original.lc_id,
        "opportunity_id": original.opportunity_id,
        "opportunity_stage": original.opportunity_stage,
        "pan_hash": original.pan_hash,
        "pan_last4": original.pan_last4,
        "mobile_last4": original.mobile_last4,
        "monthly_income": monthly_income,
        "existing_emi_declared": existing_emi_declared,
        "is_recompute": True,
        "consent_id": None,  # no new consent: the original's id rides on the row and on the synthetic decision
        "admission": decision,
        "identity": identity,
        "rank_attempts": 0,
        "rank_error": None,
        "bureau_called": False,
        "pan_called": False,
        "surepass_calls_made": [],
        "errors": [],
        "transcript": [],
        "langsmith_run_id": langsmith_run_id,
    }
    if score is not None:
        state["score"] = score
    if report is not None:
        state["report"] = report
    if original.lead_id is not None:
        state["lead_id"] = original.lead_id
    if original.mobile_hash is not None:
        state["mobile_hash"] = original.mobile_hash
    return "ok", state


# --------------------------------------------------------------------------- checkpointer (K21)

STATE_MODULE: Final[str] = "eligibility_agent.graph.state"
STRICT_MSGPACK_ENV: Final[str] = "LANGGRAPH_STRICT_MSGPACK"
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes"})  # langgraph's own reading of the flag
TRANSACTION_POOLER_PORT: Final[str] = "6543"
ISOLATION_SQL: Final[str] = (
    "SELECT to_regclass('public.checkpoints') IS NULL AS public_clear, current_schema() AS current_schema"
)
CHECKPOINTER_SCHEMA: Final[str] = "checkpointer"


class CheckpointerConfigError(RuntimeError):
    """The checkpointer would be unsafe (non-strict serde, transaction pooler, or tables visible to PostgREST)."""


def state_msgpack_allowlist() -> frozenset[tuple[str, str]]:
    """``(module, name)`` for every pydantic model and dataclass defined in ``eligibility_agent.graph.state``: the
    module-scoped allowlist (langgraph matches exact ``(module, name)`` pairs, not module prefixes)."""
    allow: set[tuple[str, str]] = set()
    members: Mapping[str, object] = vars(state_module)
    for obj in members.values():
        if not isinstance(obj, type):
            continue
        cls = cast("type[object]", obj)
        if cls.__module__ != STATE_MODULE or cls.__name__.startswith("_"):  # private bases are never instantiated
            continue
        if issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls):
            allow.add((STATE_MODULE, cls.__name__))
    return frozenset(allow)


def make_checkpoint_serde() -> JsonPlusSerializer:
    """Strict msgpack: langgraph's built-in safe types plus the state module's models, nothing else, whatever the
    process environment says (an explicit allowlist is strict by construction)."""
    return JsonPlusSerializer(allowed_msgpack_modules=state_msgpack_allowlist())


def check_checkpointer_dsn(db_url: str) -> None:
    """Refuse the Supabase TRANSACTION pooler (:6543): ``setup()`` migrations and pipeline mode do not survive it.
    Accepts URI and key=value DSNs."""
    from psycopg.conninfo import conninfo_to_dict

    ports = str(conninfo_to_dict(db_url).get("port") or "")
    if TRANSACTION_POOLER_PORT in {p.strip() for p in ports.split(",")}:
        raise CheckpointerConfigError(
            f"DATABASE_URL points at port {TRANSACTION_POOLER_PORT} (transaction pooler); use the session pooler on 5432"
        )


class _Cursor(Protocol):
    def fetchone(self) -> Mapping[str, Any] | None: ...


class _SqlConnection(Protocol):
    def execute(self, query: str) -> _Cursor: ...


def assert_checkpointer_isolated(conn: _SqlConnection) -> None:
    """Post-setup assertion: the checkpoint tables must NOT be visible in ``public`` (PostgREST-exposed; blobs carry the
    raw PAN) and the session's schema must be ``checkpointer`` (the role's search_path took effect)."""
    row = conn.execute(ISOLATION_SQL).fetchone()
    if row is None or row.get("public_clear") is not True:
        raise CheckpointerConfigError("checkpoint tables are visible in the public schema (PostgREST-exposed)")
    if row.get("current_schema") != CHECKPOINTER_SCHEMA:
        raise CheckpointerConfigError(
            f"checkpointer role search_path is not '{CHECKPOINTER_SCHEMA}' (got {row.get('current_schema')!r})"
        )


def make_postgres_checkpointer(
    db_url: str, *, min_size: int = 1, max_size: int = 5
) -> tuple[PostgresSaver, ConnectionPool[Any]]:
    """The production saver over a manually built pool (``from_conn_string`` hard-codes ``prepare_threshold=0``, which
    breaks behind poolers). Every precondition is checked BEFORE any connection is made; the caller owns
    ``pool.close()`` at shutdown."""
    if os.environ.get(STRICT_MSGPACK_ENV, "").strip().lower() not in _TRUTHY:
        raise CheckpointerConfigError(f"{STRICT_MSGPACK_ENV}=true is required before a checkpointer is built")
    check_checkpointer_dsn(db_url)

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool: ConnectionPool[Any] = ConnectionPool(
        conninfo=db_url,
        min_size=min_size,
        max_size=max_size,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
    )
    try:
        saver = PostgresSaver(cast(Any, pool), serde=make_checkpoint_serde())
        saver.setup()  # idempotent migrations; requires autocommit
        with pool.connection() as conn:
            assert_checkpointer_isolated(cast(_SqlConnection, conn))
    except BaseException:
        pool.close()
        raise
    return saver, pool
