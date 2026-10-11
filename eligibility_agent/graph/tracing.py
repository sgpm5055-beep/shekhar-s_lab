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

Run PAYLOADS never leave the process, and every string that does is scrubbed (fix rounds 1-2, binding):

* ``make_langsmith_client``: ``hide_inputs=True`` / ``hide_outputs=True`` (every node run's inputs / outputs are the
  graph state: raw PAN, mobile, names, score facts, prompts) plus ``anonymizer=scrub_trace_payload``, which langsmith
  routes the run ``error`` text and every metadata value through.
* ``PayloadFreeTracer`` (``make_tracer``): run ``name`` and ``tags`` never reach langsmith's anonymizer, so the tracer
  scrubs them on the run itself, at start and again before the end patch.
* The scrub is the output boundary's GLOBAL tier (``GLOBAL_LEAK_PATTERNS`` + ``line_leaks``, imported, never copied:
  bureau names, standalone 300-899 numbers incl. band-label numbers, account ids incl. any 9+ digit run such as a
  mobile) plus a raw-PAN search pattern derived from ``policy.guards.PAN_RE`` (the global tier alone does not match a
  PAN). Matched spans are replaced with ``TRACE_MASK``; fail-closed: a string whose normalised form still leaks, or one
  longer than ``MAX_TRACE_STRING_CHARS``, is replaced whole. Identifier-shaped values under ``pan_hash`` (64 hex) and
  ``check_id`` / ``thread_id`` (uuid) are kept verbatim: their hex can trip the score pattern structurally.
* ``invoke_config(tracer=...)`` attaches the tracer explicitly; with a ``LangChainTracer`` already among the callbacks,
  langchain-core adds no env-driven default tracer (whose default client would hide nothing).

Residual: a learner's NAME in free text has no pattern; names only reach traced strings through inputs / outputs,
which are dropped whole.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, Final, Protocol, cast
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run
from langsmith import Client

from eligibility_agent.graph.state import AdmissionDecision, Kind
from eligibility_agent.output.lc_view import GLOBAL_LEAK_PATTERNS, line_leaks
from eligibility_agent.policy.guards import PAN_RE

__all__ = [
    "FINISH_METADATA_KEYS",
    "INVOKE_METADATA_KEYS",
    "MAX_TRACE_STRING_CHARS",
    "PAN_SEARCH_RE",
    "RECURSION_LIMIT",
    "TRACE_MASK",
    "TRACE_SCRUB_PATTERNS",
    "PayloadFreeTracer",
    "RunUpdater",
    "base_tags",
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


# --------------------------------------------------------------------------- the trace scrub

TRACE_MASK: Final[str] = "[redacted]"
MAX_TRACE_STRING_CHARS: Final[int] = 32_000  # longer strings are masked whole (fail closed; bounds normalisation cost)
# A raw PAN anywhere in a string: the policy's own PAN pattern, unanchored, any case, not inside a longer token.
PAN_SEARCH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])" + PAN_RE.pattern.removeprefix("^").removesuffix("$") + r"(?![A-Za-z0-9])", re.IGNORECASE
)
TRACE_SCRUB_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (*GLOBAL_LEAK_PATTERNS, PAN_SEARCH_RE)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_UUID: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Identifier values kept verbatim, by key AND shape (anything else under these keys is scrubbed like any string).
_ID_SHAPES: Final[dict[str, re.Pattern[str]]] = {"pan_hash": _HEX64, "check_id": _UUID, "thread_id": _UUID}


def scrub_trace_text(text: str) -> str:
    """Replace every span the trace patterns match with ``TRACE_MASK``. Fail closed: a string whose NORMALISED form
    still leaks (fullwidth digits, invisible separators) or that exceeds ``MAX_TRACE_STRING_CHARS`` is masked whole."""
    if len(text) > MAX_TRACE_STRING_CHARS:
        return TRACE_MASK
    scrubbed = text
    for pattern in TRACE_SCRUB_PATTERNS:
        scrubbed = pattern.sub(TRACE_MASK, scrubbed)
    return TRACE_MASK if line_leaks(scrubbed, TRACE_SCRUB_PATTERNS) else scrubbed


def _scrub_value(key: str | None, value: object) -> object:
    if isinstance(value, str):
        shape = _ID_SHAPES.get(key) if key is not None else None
        return value if shape is not None and shape.fullmatch(value) else scrub_trace_text(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {k: _scrub_value(str(k), v) for k, v in items}
    if isinstance(value, (list, tuple)):
        return [_scrub_value(key, v) for v in cast("list[object] | tuple[object, ...]", value)]
    return value


def scrub_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The langsmith ``anonymizer``: every string at any depth goes through ``scrub_trace_text`` (it receives the run's
    metadata, and its error text wrapped as ``{"error": ...}``). Total; on any surprise every value is masked."""
    try:
        return {key: _scrub_value(key, value) for key, value in payload.items()}
    except Exception:
        return {key: TRACE_MASK for key in payload}


class PayloadFreeTracer(LangChainTracer):
    """A ``LangChainTracer`` that also scrubs each run's ``name`` and ``tags`` (langsmith's anonymizer never sees them)."""

    def _start_trace(self, run: Run) -> None:
        super()._start_trace(run)
        _scrub_labels(run)

    def _end_trace(self, run: Run) -> None:
        _scrub_labels(run)  # before the end patch is built
        super()._end_trace(run)


def _scrub_labels(run: Run) -> None:
    if run.name:
        run.name = scrub_trace_text(run.name)
    if run.tags:
        run.tags = [scrub_trace_text(tag) for tag in run.tags]


def make_langsmith_client(
    *, api_key: str | None = None, api_url: str | None = None, client_cls: type[Client] = Client
) -> Client:
    """The ONE LangSmith client the service traces with: run inputs and outputs are dropped and every other exported
    string (error text, metadata values) is scrubbed client-side, before any upload. ``api_key`` / ``api_url`` default
    to the LANGSMITH_* environment. ``client_cls`` exists for tests (a recording subclass keeps the real hiding and
    anonymizing logic and replaces only the send hooks)."""
    return client_cls(
        api_key=api_key, api_url=api_url, hide_inputs=True, hide_outputs=True, anonymizer=scrub_trace_payload
    )


def make_tracer(client: Client, *, project_name: str) -> PayloadFreeTracer:
    """The tracer every graph invocation attaches through ``invoke_config(tracer=...)`` when tracing is on."""
    return PayloadFreeTracer(client=client, project_name=project_name)


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
        tags = [*base_tags(kind, lc_id), f"admission:{metadata['admission']}"]  # update_run replaces tags
        client.update_run(
            run_id,
            # This patch bypasses PayloadFreeTracer, so its strings are scrubbed here (fix round 3); the metadata also
            # goes through the scrub so a non-anonymizing client cannot export a raw string either.
            tags=[scrub_trace_text(tag) for tag in tags],
            extra={"metadata": scrub_trace_payload(metadata)},
        )
    except Exception as exc:  # tracing must never fail a check
        logger.warning("langsmith update_run failed (%s)", type(exc).__name__)
