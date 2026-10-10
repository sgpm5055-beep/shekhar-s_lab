"""LangSmith run plumbing (PRD v1.1 section 11; v1.0 dossier ``graph/tracing.py`` reused verbatim, re-pathed per K21).

* ``new_run_id``: the root run id is pre-generated (a ``run_id`` in the ROOT ``RunnableConfig`` becomes the LangSmith trace
  id), so the route seeds ``state['langsmith_run_id'] = str(run_id)`` and persist writes it onto the checks row; no
  post-hoc capture.
* ``invoke_config``: used for EVERY graph invocation, traced or not: ``configurable.thread_id = check_id`` keys the
  Postgres checkpointer (each check, recompute included, is a fresh thread), ``recursion_limit`` bounds the run.
* ``finish_trace``: billed / counted / the admission outcome exist only after the run, so they are patched onto the root
  run with ``Client.update_run`` (which REPLACES tags, hence the base tags are re-sent). Tracing must never fail a check:
  every exception is swallowed with a warning that names the exception TYPE only.

PII rule (PRD 11, brief ruling 13): trace metadata carries ``check_id``, ``pan_hash``, the tier, the admission outcome and
plans, and the billed / counted flags, NOTHING else; never the raw PAN, the learner's name or the mobile number. The key
sets are frozen below and pinned by tests.

Run PAYLOADS never leave the process (fix round 1, binding). Every node run's inputs / outputs are the graph state (raw
PAN, mobile, names, score facts, prompts), so ``make_langsmith_client`` builds the client with ``hide_inputs`` /
``hide_outputs`` callables that drop the payload entirely, and ``make_tracer`` binds a ``LangChainTracer`` to it.
``invoke_config(tracer=...)`` attaches that tracer explicitly; with a ``LangChainTracer`` already among the callbacks,
langchain-core adds no env-driven default tracer (whose default client would not hide anything). Metadata and tags are
kept: they are the only thing a trace is for here. Residual (flagged): a child run's ``error`` string is not dropped by
these hooks; the nodes' own exception texts are PII-free by construction.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final, Protocol
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from eligibility_agent.graph.state import AdmissionDecision, Kind

__all__ = [
    "FINISH_METADATA_KEYS",
    "INVOKE_METADATA_KEYS",
    "RECURSION_LIMIT",
    "RunUpdater",
    "base_tags",
    "drop_payload",
    "finish_trace",
    "invoke_config",
    "make_langsmith_client",
    "make_tracer",
    "new_run_id",
    "trace_metadata",
]

logger = logging.getLogger(__name__)

# Worst path is 12 node executions (admission, pan, bureau, derive, gate, range, confidence, rank x2, safety, redact,
# persist); 40 leaves headroom without letting a wiring loop spin.
RECURSION_LIMIT: Final[int] = 40

INVOKE_METADATA_KEYS: Final[frozenset[str]] = frozenset({"check_id", "pan_hash", "tier"})
FINISH_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {"check_id", "pan_hash", "tier", "admission", "admission_pan", "admission_bureau", "billed", "counted_against_quota"}
)


class RunUpdater(Protocol):
    """The slice of ``langsmith.Client`` this module uses (structurally satisfied by the real client)."""

    def update_run(self, run_id: UUID, *, tags: list[str] | None = ..., extra: dict[str, Any] | None = ...) -> None: ...


def drop_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The ``hide_inputs`` / ``hide_outputs`` hook: whatever a run carried, nothing of it is exported."""
    del payload
    return {}


def make_langsmith_client(
    *, api_key: str | None = None, api_url: str | None = None, client_cls: type[Client] = Client
) -> Client:
    """The ONE LangSmith client the service traces with: run inputs and outputs are dropped client-side, before any
    upload; metadata and tags pass. ``api_key`` / ``api_url`` default to the LANGSMITH_* environment. ``client_cls``
    exists for tests (a recording subclass keeps the real hiding logic and replaces only the send hooks)."""
    return client_cls(api_key=api_key, api_url=api_url, hide_inputs=drop_payload, hide_outputs=drop_payload)


def make_tracer(client: Client, *, project_name: str) -> LangChainTracer:
    """The tracer every graph invocation attaches through ``invoke_config(tracer=...)`` when tracing is on."""
    return LangChainTracer(client=client, project_name=project_name)


def new_run_id(enabled: bool) -> UUID | None:
    """Pre-generate the root run id when tracing is on (``Settings.langsmith_tracing``); None when it is off."""
    return uuid4() if enabled else None


def base_tags(kind: Kind, lc_id: str) -> list[str]:
    return [f"tier:{kind}", f"lc:{lc_id}"]


def invoke_config(
    *,
    kind: Kind,
    check_id: str,
    lc_id: str,
    pan_hash: str,
    run_id: UUID | None,
    tracer: BaseCallbackHandler | None = None,
) -> RunnableConfig:
    """The ``RunnableConfig`` for one graph invocation. Metadata is ``{check_id, pan_hash, tier}`` and nothing else.
    ``tracer`` (from ``make_tracer``) is attached as the run's callback when tracing is on."""
    config: RunnableConfig = {
        "configurable": {"thread_id": check_id},
        "tags": base_tags(kind, lc_id),
        "metadata": {"check_id": check_id, "pan_hash": pan_hash, "tier": kind},
        "recursion_limit": RECURSION_LIMIT,
    }
    if run_id is not None:
        config["run_id"] = run_id
    if tracer is not None:
        config["callbacks"] = [tracer]
    return config


def _admission_outcome(admission: object) -> tuple[str, str | None, str | None]:
    """``(outcome, pan_plan, bureau_plan)``: outcome is the admission mode, ``'refused'``, or ``'none'`` (no decision)."""
    if not isinstance(admission, AdmissionDecision):
        return "none", None, None
    if admission.code is not None:
        return "refused", None, None
    return admission.admission or "none", admission.pan_plan, admission.bureau_plan


def trace_metadata(*, kind: Kind, check_id: str, pan_hash: str, final_state: Mapping[str, Any]) -> dict[str, Any]:
    """The post-run metadata patch: exactly ``FINISH_METADATA_KEYS``, read from the final graph state."""
    outcome, pan_plan, bureau_plan = _admission_outcome(final_state.get("admission"))
    return {
        "check_id": check_id,
        "pan_hash": pan_hash,
        "tier": kind,
        "admission": outcome,
        "admission_pan": pan_plan,
        "admission_bureau": bureau_plan,
        "billed": final_state.get("billed") is True,
        "counted_against_quota": final_state.get("counted_against_quota") is True,
    }


def finish_trace(
    client: RunUpdater | None,
    *,
    run_id: UUID | None,
    kind: Kind,
    lc_id: str,
    check_id: str,
    pan_hash: str,
    final_state: Mapping[str, Any],
) -> None:
    """Patch the admission outcome and the billed / counted flags onto the root run. No-op without a client or run id
    (tracing off); never raises."""
    if client is None or run_id is None:
        return
    try:
        metadata = trace_metadata(kind=kind, check_id=check_id, pan_hash=pan_hash, final_state=final_state)
        client.update_run(
            run_id,
            tags=[*base_tags(kind, lc_id), f"admission:{metadata['admission']}"],  # update_run replaces tags
            extra={"metadata": metadata},
        )
    except Exception as exc:  # tracing must never fail a check
        logger.warning("langsmith update_run failed (%s)", type(exc).__name__)
