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
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final, Protocol
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig

from eligibility_agent.graph.state import AdmissionDecision, Kind

__all__ = [
    "FINISH_METADATA_KEYS",
    "INVOKE_METADATA_KEYS",
    "RECURSION_LIMIT",
    "RunUpdater",
    "base_tags",
    "finish_trace",
    "invoke_config",
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


def new_run_id(enabled: bool) -> UUID | None:
    """Pre-generate the root run id when tracing is on (``Settings.langsmith_tracing``); None when it is off."""
    return uuid4() if enabled else None


def base_tags(kind: Kind, lc_id: str) -> list[str]:
    return [f"tier:{kind}", f"lc:{lc_id}"]


def invoke_config(*, kind: Kind, check_id: str, lc_id: str, pan_hash: str, run_id: UUID | None) -> RunnableConfig:
    """The ``RunnableConfig`` for one graph invocation. Metadata is ``{check_id, pan_hash, tier}`` and nothing else."""
    config: RunnableConfig = {
        "configurable": {"thread_id": check_id},
        "tags": base_tags(kind, lc_id),
        "metadata": {"check_id": check_id, "pan_hash": pan_hash, "tier": kind},
        "recursion_limit": RECURSION_LIMIT,
    }
    if run_id is not None:
        config["run_id"] = run_id
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
