"""Every graph node (PRD v1.1 section 7): one function per node, uniform signature ``(state, config, deps) -> update``.

This is where every prior module meets. Nodes hold NO business rules of their own: admission tables, windows and the
persist formulas are ``policy.guards``; money maths is ``derive``; the gate / safety / confidence are ``lenders``; the
only LcView factories are the four ``output.lc_view`` builders; the only LLM call is ``deps.rank_model`` in ``rank``.
Every node returns ``{'transcript': [<node name>]}`` (an ``operator.add`` reducer accumulates the path).

Load-bearing orderings and owners (controller rulings, each pinned in tests/test_graph_wiring.py):

* ADMISSION follows the 8-step write order in the ``policy/guards.py`` module docstring: consent row (1) ->
  ``insert_check(status='running')`` (1.5, BEFORE any lock read: ``pan_locks.check_id`` and ``surepass_calls.check_id``
  are FKs to ``checks``; MemoryRepo does not enforce the lock FK, so this order is pinned by a call-order test) -> the
  node-level lock gate (2: active timeout / daily_cap locks refuse PAN_LOCKED / PAN_DAILY_CAP before the pure tables,
  even for would-be cache hits) -> atomic in-flight acquire (3: IN_FLIGHT) -> full-only stage gate and PAN-correction
  policy (5) -> ``AdmissionFacts`` snapshot (6) -> ``admit_quick`` / ``admit_full`` (7) -> directives (8).
* Nodes key off ``code`` / ``admission``, never off a refusal's plans (refusals carry ``'reuse'`` plans).
* Cache reuse (K5): ``repo.fetch_raw(row.storage_key)`` + the adapter on CURRENT code, every time. No derived columns.
  A Report pull upserts BOTH ``report_cache`` and ``score_cache(source='report')`` pointing at the same raw object.
* Bureau TIMEOUT (K1/K2): the client already logged the billable sink row; this node writes the unresolved
  ``timeout_reconciliation`` review row FIRST (it alone gates re-pulls) and then the 10-minute timeout lock ->
  ``pending_verification`` with code None -> persist. A PAN-endpoint timeout is a plain service failure (no lock).
* Review-row owners (K11): admission -> ``pan_daily_cap`` / ``pan_change_manager``; bureau node ->
  ``timeout_reconciliation``; the node that sees a SurePass auth failure -> ``surepass_auth`` + ``logger.critical``;
  persist -> ``openai_failure`` / ``safety_demoted_all`` / ``not_fundable_pattern`` / ``model_review_flags``.
* Rank (T10): ``build_rank_context(state, gated)`` with ``gated`` passed explicitly; ONE retry on a raised / invalid
  output (attempt 2 carries ``prior_error``); after two failures ``OPENAI_FAILURE``. Business violations are safety's
  job (never retried); a raised confidence is ignored downstream by ``final_confidence``'s ``min()``.
* Redact (T9): ``build_lc_view(state, RedactionContext)`` owns confidence and headline; the node reads them back from
  the view. A builder ``ValueError`` becomes the internal-error view plus a guarded error row, so ``lc_view`` is ALWAYS
  set.
* Persist / persist_refusal: ONE guarded terminal ``update_check``, always the LAST write (persist's review rows go
  before it); billed / counted come from ``policy.compute_billed`` / ``policy.compute_counted_against_quota``; ANY
  exception (not just ValueError) becomes a guarded error row and the error view; the in-flight lock is released
  holder-only in ``finally`` on every terminal. Error rows keep the true billed / counted flags (K8): they are patched
  while the row is still running, before ``record_check_error``. Both terminals blank the raw ``pan`` / ``name`` /
  ``mobile`` in the final state (request scope only). An exception raised BEFORE a terminal node escapes
  ``graph.invoke``; ``fail_check`` is the route boundary's helper for that case.

PII: no node logs or stores in ``internal`` the raw PAN, the mobile or a name; log lines carry ``check_id`` and
exception TYPES only. Run payloads DO carry them (node inputs / outputs are the state), so tracing exports metadata
only: ``graph/tracing.py``'s tracer runs on a client whose ``hide_inputs`` / ``hide_outputs`` drop every payload.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from langchain_core.runnables import RunnableConfig

from eligibility_agent.db.repo import (
    CheckPatch,
    CheckRow,
    ConsentRow,
    LenderRow,
    OpportunityPanEvent,
    PanRecordRow,
    ReportCacheRow,
    Repo,
    ReviewRow,
    ScoreCacheRow,
    SurePassCallRow,
    new_id,
)
from eligibility_agent.derive.emi import derive_existing_emi
from eligibility_agent.derive.range import age_years_and_months, candidate_pincodes, lender_numbers
from eligibility_agent.graph.llm import build_rank_context
from eligibility_agent.graph.prompts import PRIOR_ERROR_MAX_CHARS
from eligibility_agent.graph.state import (
    AdmissionDecision,
    Band,
    CheckState,
    Code,
    Derived,
    FullAdmissionInputs,
    GatedLender,
    Identity,
    Kind,
    LcView,
    ModelOutput,
    ReportFacts,
    ReviewKind,
    ScoreFacts,
    StateUpdate,
    SurePassCallAudit,
)
from eligibility_agent.lenders.gate import PublishedLenderRow, gate_lenders
from eligibility_agent.lenders.safety import compute_code_confidence, run_safety
from eligibility_agent.output.lc_view import (
    MSG_BUREAU_SERVICE_FAILURE,
    MSG_INTERNAL,
    MSG_PAN_SERVICE_FAILURE,
    MSG_PENDING_VERIFICATION,
    RedactionContext,
    build_error_lc_view,
    build_failure_lc_view,
    build_lc_view,
    build_refusal_lc_view,
)
from eligibility_agent.policy.guards import (
    IN_FLIGHT_LOCK_TTL,
    RAW_RETENTION,
    TIMEOUT_LOCK_TTL,
    AdmissionFacts,
    PanChangeOutcome,
    admit_full,
    admit_quick,
    compute_billed,
    compute_counted_against_quota,
    evaluate_pan_change,
    ist_date,
    mobile_last4,
    next_ist_midnight,
    pan_expiry,
    pan_last4,
    quick_checks_left,
    report_expiry,
    score_expiry,
    stage_allows_full_check,
)
from eligibility_agent.surepass.adapters import (
    identity_from_envelope,
    report_facts_from_envelope,
    score_facts_from_envelope,
)
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    LiveCallsDisabled,
    SurePassAuthError,
    SurePassError,
    SurePassTimeout,
)

if TYPE_CHECKING:  # build.py imports this module at runtime; the deps type is only needed for annotations
    from eligibility_agent.graph.build import GraphDeps

__all__ = [
    "CONSENT_RETENTION_YEARS",
    "RANK_MAX_ATTEMPTS",
    "admission",
    "bureau_pull",
    "code_confidence",
    "derive",
    "fail_check",
    "pan_comprehensive",
    "persist",
    "persist_refusal",
    "pincode_gate",
    "published_lender_row",
    "range_",
    "rank",
    "redact",
    "safety",
]

logger = logging.getLogger(__name__)

RANK_MAX_ATTEMPTS: Final[int] = 2  # 1 call + ONE retry on a raised / invalid output (PRD 9.2, FIXED)
CONSENT_RETENTION_YEARS: Final[int] = 5  # consents.purge_after = given_at + interval '5 years' (PRD 11)
SCORE_RANGE: Final[tuple[int, int]] = (300, 900)  # a three-digit score outside it is bucketed + recorded as an error
# The raw identifiers are request-scope only (state.py): both terminal nodes blank them so the FINAL state (returned to
# the route, patched onto the trace, held by the last checkpoint) never carries them. last-4 values stay.
_STRIPPED_IDENTIFIERS: Final[dict[str, str]] = {"pan": "", "name": "", "mobile": ""}


# --------------------------------------------------------------------------- small helpers


def _kind(state: CheckState) -> Kind:
    kind = state.get("kind")
    if kind is None:
        raise ValueError("state has no kind")
    return kind


def _check_id(state: CheckState) -> str:
    check_id = state.get("check_id")
    if not check_id:
        raise ValueError("state has no check_id")
    return check_id


def _pan_hash(state: CheckState) -> str:
    pan_hash = state.get("pan_hash")
    if not pan_hash:
        raise ValueError("state has no pan_hash")
    return pan_hash


def _admission(state: CheckState) -> AdmissionDecision:
    decision = state.get("admission")
    if decision is None:
        raise ValueError("state has no admission decision")
    return decision


def _identity(state: CheckState) -> Identity:
    identity = state.get("identity")
    if identity is None:
        raise ValueError("state has no identity facts")
    return identity


def _derived(state: CheckState) -> Derived:
    derived = state.get("derived")
    if derived is None:
        raise ValueError("state has no derived facts")
    return derived


def _add_years(moment: datetime, years: int) -> datetime:
    """Calendar years, Postgres ``interval 'N years'`` semantics: 29 Feb lands on 28 Feb."""
    try:
        return moment.replace(year=moment.year + years)
    except ValueError:
        return moment.replace(year=moment.year + years, day=28)


def _insert_review(repo: Repo, *, now: datetime, check_id: str, pan_hash: str, kind: ReviewKind, reason: str) -> None:
    repo.insert_review_item(
        ReviewRow(
            id=new_id(),
            check_id=check_id,
            pan_hash=pan_hash,
            kind=kind,
            reason=reason,
            created_at=now,
            resolved_at=None,
            resolved_by=None,
        )
    )


def _calls_for(repo: Repo, check_id: str, endpoint: str) -> list[SurePassCallRow]:
    return [call for call in repo.surepass_calls_for_check(check_id) if call.endpoint == endpoint]


def _audit(call: SurePassCallRow) -> SurePassCallAudit:
    return SurePassCallAudit(
        endpoint=call.endpoint,
        http_status=call.http_status,
        success=call.success,
        message_code=call.message_code,
        client_id=call.client_id,
        billed=call.billable,
    )


def _success_storage_key(calls: Sequence[SurePassCallRow], endpoint: str) -> str:
    """The raw object of the successful call just made (the client's sink logged it before returning).

    A missing object means the client is not wired to the repo sink: caching a pointer-less row would break K5, so this
    fails loudly instead.
    """
    for call in reversed(calls):
        if call.http_status == 200 and call.success is True and call.storage_key is not None:
            return call.storage_key
    raise RuntimeError(f"no stored raw object for the successful {endpoint} call (is the client wired to the sink?)")


def _quick_left(deps: GraphDeps, state: CheckState) -> int | None:
    """``quick_checks_left_today`` for a quick view (None on full and on recompute). ``used_before`` is read now, before
    persist flips this run's flag; whether THIS run counts comes from the policy formula over its call rows."""
    if state.get("kind") != "quick" or state.get("is_recompute"):
        return None
    repo = deps.repo
    decision = state.get("admission")
    lc_id = state.get("lc_id") or ""
    counted = compute_counted_against_quota(
        kind="quick",
        counts_against_quota=decision.counts_against_quota if decision is not None else False,
        calls=repo.surepass_calls_for_check(_check_id(state)),
    )
    limit = decision.quota_limit if decision is not None and decision.quota_limit is not None else repo.quota_limit(lc_id)
    return quick_checks_left(limit, repo.quick_checks_used_today(lc_id), counted)


def _is_error_view(view: LcView) -> bool:
    return view.status == "service_failure" and view.code is None and view.label == MSG_INTERNAL


def _release_in_flight(deps: GraphDeps, pan_hash: str | None, check_id: str | None) -> None:
    """Holder-only, idempotent release; a failure to release is logged (the lock self-heals at its TTL), never raised
    over the result the LC is about to receive."""
    if not pan_hash or not check_id:
        return
    try:
        deps.repo.release_pan_lock(pan_hash, check_id)
    except Exception as exc:
        logger.error("in-flight lock release failed for check_id=%s (%s)", check_id, type(exc).__name__)


def _record_error(deps: GraphDeps, state: CheckState, message: str) -> tuple[LcView, bool, bool]:
    """Guarded error row + the internal-error view; returns ``(view, billed, counted)`` as written on the row.

    K8 (fix round 1): spend is status-independent, so WHILE the row is still running it is first patched with
    ``billed = policy.compute_billed(calls)`` and, when the admission decision is in state (every node-level path),
    ``counted = policy.compute_counted_against_quota(...)``; only then does ``record_check_error`` (update-if-running,
    else insert ``error``) flip it, keeping both flags. Without a decision (the route-level ``fail_check``) counted
    stays False. Best effort: a failing repo is logged, never raised; the flags returned are the ones the row holds.
    """
    check_id = state.get("check_id") or ""
    kind: Kind = state.get("kind") or "quick"
    billed = counted = False
    if check_id:
        try:
            calls = deps.repo.surepass_calls_for_check(check_id)
            spent = compute_billed(calls)
            decision = state.get("admission")
            would_count = decision is not None and compute_counted_against_quota(
                kind=kind, counts_against_quota=decision.counts_against_quota, calls=calls
            )
            deps.repo.update_check(check_id, {"billed": spent, "counted_against_quota": would_count})
            billed, counted = spent, would_count
        except Exception as exc:
            logger.error("billing patch before the error row failed for check_id=%s (%s)", check_id, type(exc).__name__)
    try:
        deps.repo.record_check_error(
            check_id=check_id,
            kind=kind,
            lc_id=state.get("lc_id") or "",
            pan_hash=state.get("pan_hash") or "",
            pan_last4=state.get("pan_last4") or "",
            message=message,
        )
    except Exception as exc:
        logger.error("record_check_error failed for check_id=%s (%s)", check_id, type(exc).__name__)
    return build_error_lc_view(check_id=check_id, kind=kind), billed, counted


def fail_check(
    deps: GraphDeps, *, check_id: str, kind: Kind, lc_id: str, pan_hash: str, pan_last4: str, message: str
) -> LcView:
    """The route boundary's helper for an exception that ESCAPED ``graph.invoke`` before a terminal node ran.

    Use it only once ``graph.invoke`` has actually returned or raised. NEVER on a route timeout: the graph thread may
    still be mid-call (a SurePass or model request in flight), so its in-flight lock must be left to expire at its TTL
    rather than be released under a live run.

    Patches ``billed`` from the call log while the row is still running, writes the guarded error row, and releases this
    check's in-flight lock (holder-only, so safe whether or not the run ever acquired it); returns the internal-error
    view. Never raises. RESIDUAL: the route has no admission decision, so ``counted_against_quota`` stays False here even
    if a counting Score call happened (the call log keeps the spend; node-level error paths do count).
    """
    state: CheckState = {"check_id": check_id, "kind": kind, "lc_id": lc_id, "pan_hash": pan_hash, "pan_last4": pan_last4}
    try:
        view, _, _ = _record_error(deps, state, message)
        return view
    finally:
        _release_in_flight(deps, pan_hash, check_id)


# --------------------------------------------------------------------------- admission (the 8-step write order)


def _refusal(code: Code, base: StateUpdate) -> StateUpdate:
    # Refusal decisions carry 'reuse' plans; downstream keys off code / admission.code only.
    return {**base, "admission": AdmissionDecision(code=code), "status": "refused", "code": code}


def admission(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """The 8-step order of ``policy/guards.py`` (see this module's docstring). Writes the decision into state."""
    del config
    repo = deps.repo
    now = deps.now()
    kind = _kind(state)
    check_id = _check_id(state)
    pan_hash = _pan_hash(state)
    lc_id = state.get("lc_id")
    mobile_hash = state.get("mobile_hash")
    pan = state.get("pan")
    if not lc_id or not mobile_hash or not pan:
        raise ValueError("admission needs lc_id, mobile_hash and pan in the initial state")
    opportunity_id = state.get("opportunity_id")
    if kind == "full" and not opportunity_id:
        raise ValueError("a full check needs opportunity_id")
    last4 = pan_last4(pan)  # K15: computed here via the policy helpers
    mobile = state.get("mobile")
    m_last4 = mobile_last4(mobile) if mobile else None

    # 1. consent: a row per pull it covers; absent consent is a CONSENT_MISSING refusal (no row is fabricated).
    consent = state.get("consent")
    consent_id: str | None = None
    if consent is not None and consent.given is True:
        consent_id = new_id()
        repo.insert_consent(
            ConsentRow(
                id=consent_id,
                lc_id=lc_id,
                lead_id=state.get("lead_id"),
                opportunity_id=opportunity_id,
                contact_id=None,
                pan_hash=pan_hash,
                mobile_hash=mobile_hash,
                lc_typed_name=state.get("name") or "",
                wording_version=consent.wording_version,
                call_ref=consent.call_ref,
                ip=consent.ip,
                covers="score" if kind == "quick" else "report",
                given_at=now,
                purge_after=_add_years(now, CONSENT_RETENTION_YEARS),
            )
        )

    # 1.5. insert the running row for EVERY outcome, before any lock read / acquire and before any SurePass call.
    repo.insert_check(
        CheckRow(
            id=check_id,
            kind=kind,
            lc_id=lc_id,
            lead_id=state.get("lead_id"),
            opportunity_id=opportunity_id,
            opportunity_stage=state.get("opportunity_stage"),
            contact_id=None,
            pan_hash=pan_hash,
            pan_last4=last4,
            mobile_hash=mobile_hash,
            mobile_last4=m_last4,
            status="running",
            code=None,
            billed=False,
            counted_against_quota=False,
            is_refresh=False,  # the decision is not known yet; persist carries it on the terminal patch
            recompute_of=None,
            consent_id=consent_id,
            lc_view=None,
            internal=None,
            langsmith_run_id=state.get("langsmith_run_id"),
            created_at=now,
        )
    )
    base: StateUpdate = {
        "pan_last4": last4,
        "mobile_last4": m_last4,
        "consent_id": consent_id,
        "transcript": ["admission"],
    }
    if consent_id is None:
        return _refusal("CONSENT_MISSING", base)

    # 2. the node-level lock gate: BEFORE the pure tables, so it pre-empts even a would-be cache hit (K22 / T7 ruling).
    active = {lock.kind for lock in repo.get_active_locks(pan_hash)}
    if "timeout" in active:
        return _refusal("PAN_LOCKED", base)
    if "daily_cap" in active:
        return _refusal("PAN_DAILY_CAP", base)

    # 3. atomic in-flight acquire: refuse, never queue (PRD 5.11).
    if not repo.try_acquire_pan_lock(pan_hash, check_id, now + IN_FLIGHT_LOCK_TTL):
        return _refusal("IN_FLIGHT", base)

    # 5. full only: stage gate, then the PAN-correction policy.
    pan_change: PanChangeOutcome | None = None
    if kind == "full" and opportunity_id:
        if not stage_allows_full_check(state.get("opportunity_stage")):
            return _refusal("STAGE_TOO_EARLY", base)
        pan_change = evaluate_pan_change(
            repo.pan_events_for_opportunity(opportunity_id), repo.report_exists_for_opportunity(opportunity_id), pan_hash
        )
        if pan_change == "needs_manager":
            _insert_review(
                repo,
                now=now,
                check_id=check_id,
                pan_hash=pan_hash,
                kind="pan_change_manager",
                reason="PAN change on an opportunity after a report or after the one allowed correction",
            )
            return _refusal("PAN_CHANGE_NEEDS_MANAGER", base)

    # 6. the facts snapshot (quota_used on EVERY quick path), 7. the pure decision table.
    facts = AdmissionFacts(
        now=now,
        pan_row=repo.get_pan_record(pan_hash),
        score_row=repo.get_score_cache(pan_hash),
        report_row=repo.get_report_cache(pan_hash) if kind == "full" else None,
        quota_used=repo.quick_checks_used_today(lc_id) if kind == "quick" else 0,
        quota_limit=repo.quota_limit(lc_id) if kind == "quick" else 0,
        bureau_calls_today=repo.bureau_calls_today(pan_hash),
        reconciliation_pending=repo.unresolved_review_exists(pan_hash, "timeout_reconciliation"),
        surepass_live=deps.surepass.calls_enabled,
    )
    decision = (
        admit_quick(facts)
        if kind == "quick"
        else admit_full(
            facts,
            FullAdmissionInputs(
                refresh=state.get("refresh") is True,
                acknowledge_ntc_warning=state.get("acknowledge_ntc_warning") is True,
            ),
        )
    )

    # 8. directives.
    if decision.create_daily_cap_lock:
        # Review row first, then the lock (same order as the timeout path): the count-based cap keeps refusing on its
        # own until IST midnight, so a failed lock write still leaves both the gate and the review signal.
        _insert_review(
            repo,
            now=now,
            check_id=check_id,
            pan_hash=pan_hash,
            kind=decision.review_reason or "pan_daily_cap",
            reason="per-PAN daily cap of billed bureau calls reached; PAN locked until the next IST midnight",
        )
        repo.create_pan_lock(pan_hash, "daily_cap", check_id, next_ist_midnight(now))
    if decision.code is None and opportunity_id and pan_change in ("first_pan", "record_correction"):
        # Events are recorded only for an ADMITTED full check: a refused attempt made no pull, so it neither marks the
        # first PAN nor spends the one correction (PRD 5.13).
        repo.insert_pan_event(
            OpportunityPanEvent(
                id=new_id(),
                opportunity_id=opportunity_id,
                pan_hash=pan_hash,
                pan_last4=last4,
                event="initial" if pan_change == "first_pan" else "correction",
                actor_lc_id=lc_id,
                check_id=check_id,
                created_at=now,
            )
        )
        if pan_change == "record_correction":
            decision = dataclasses.replace(decision, record_pan_correction=True)
    return {
        **base,
        "admission": decision,
        "status": "refused" if decision.code is not None else None,
        "code": decision.code,
    }


# --------------------------------------------------------------------------- persist_refusal


def persist_refusal(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """The refusal terminal: the refusal view + ONE guarded terminal update; the in-flight lock released in finally
    (holder-only, so IN_FLIGHT / pre-acquire refusals that never held it are a no-op)."""
    del config
    check_id = state.get("check_id")
    pan_hash = state.get("pan_hash")
    billed = counted = False
    try:
        kind = _kind(state)
        decision = _admission(state)
        code = decision.code
        if code is None:
            raise ValueError("persist_refusal reached without a refusal code")
        calls = deps.repo.surepass_calls_for_check(_check_id(state))
        billed = compute_billed(calls)
        counted = compute_counted_against_quota(
            kind=kind, counts_against_quota=decision.counts_against_quota, calls=calls
        )
        ctx = RedactionContext(
            quick_checks_left_today=_quick_left(deps, state),
            checked_on=None,
            from_cache=False,
            refresh_available_on=decision.refresh_available_on,
        )
        view = build_refusal_lc_view(
            check_id=_check_id(state), kind=kind, code=code, ctx=ctx, warn_date=decision.ntc_found_on
        )
        patch: CheckPatch = {
            "status": "refused",
            "code": code,
            "billed": billed,
            "counted_against_quota": counted,
            "lc_view": view.model_dump(mode="json"),
            "internal": _internal(deps, state, view, calls, node="persist_refusal"),
            "langsmith_run_id": state.get("langsmith_run_id"),
        }
        deps.repo.update_check(_check_id(state), patch)
    except Exception as exc:  # ANY exception, not just ValueError (T2 note): guarded error row + the error view
        logger.error("persist_refusal failed for check_id=%s (%s)", check_id, type(exc).__name__)
        view, billed, counted = _record_error(deps, state, f"persist_refusal: {type(exc).__name__}")
    finally:
        _release_in_flight(deps, pan_hash, check_id)
    return {
        "lc_view": view,
        "status": view.status,
        "code": view.code,
        "billed": billed,
        "counted_against_quota": counted,
        **_STRIPPED_IDENTIFIERS,
        "transcript": ["persist_refusal"],
    }


# --------------------------------------------------------------------------- SurePass failures (PAN and bureau)


def _surepass_failure(
    deps: GraphDeps,
    state: CheckState,
    exc: SurePassError,
    *,
    node: Literal["pan_comprehensive", "bureau_pull"],
    endpoint: str,
    label: str,
) -> StateUpdate:
    """A non-timeout SurePass failure (or a PAN timeout): ``service_failure`` + SUREPASS_FAILURE, nothing cached, quota
    untouched. An auth failure additionally alerts engineering (critical log + a ``surepass_auth`` review row)."""
    check_id = _check_id(state)
    pan_hash = _pan_hash(state)
    if isinstance(exc, SurePassAuthError):
        logger.critical(
            "surepass_auth_failure check_id=%s endpoint=%s http_status=%s", check_id, exc.endpoint, exc.http_status
        )
        _insert_review(
            deps.repo,
            now=deps.now(),
            check_id=check_id,
            pan_hash=pan_hash,
            kind="surepass_auth",
            reason=f"SurePass auth failure on {exc.endpoint} (http {exc.http_status}); check the egress IP whitelist "
            "and the token",
        )
    view = build_failure_lc_view(
        check_id=check_id,
        kind=_kind(state),
        status="service_failure",
        code="SUREPASS_FAILURE",
        label=label,
        quick_checks_left_today=_quick_left(deps, state),
    )
    called = not isinstance(exc, LiveCallsDisabled)  # the kill switch raises before any I/O and logs no row
    return {
        "status": "service_failure",
        "code": "SUREPASS_FAILURE",
        "lc_view": view,
        "pan_called" if node == "pan_comprehensive" else "bureau_called": called,
        "surepass_calls_made": [_audit(c) for c in _calls_for(deps.repo, check_id, endpoint)],
        "errors": [f"{node}: {type(exc).__name__}: {exc.detail}"],
        "transcript": [node],
    }


# --------------------------------------------------------------------------- pan_comprehensive


def pan_comprehensive(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """Stored < 180 days: re-adapt the stored raw (no call). Else call (8 s); cache ONLY valid + person identities.

    Not valid / not a person: ``pan_not_verified`` (nothing cached, nothing marked checked, no quota) -> redact.
    Any SurePass failure, timeout included: ``service_failure`` -> persist (a PAN timeout never locks the PAN).
    """
    del config
    repo = deps.repo
    decision = _admission(state)
    check_id = _check_id(state)
    pan_hash = _pan_hash(state)

    if decision.pan_plan == "reuse":
        row = repo.get_pan_record(pan_hash)
        if row is None:
            raise RuntimeError("admission planned a PAN reuse but no pan_records row exists")
        identity = identity_from_envelope(repo.fetch_raw(row.storage_key))
        update: StateUpdate = {"identity": identity, "pan_called": False, "transcript": ["pan_comprehensive"]}
        if not identity.pan_valid:
            update["status"] = "pan_not_verified"
        return update

    pan = state.get("pan")
    if not pan:
        raise ValueError("a PAN call needs the raw PAN in state")
    now = deps.now()
    try:
        result = deps.surepass.pan_comprehensive(check_id, pan_hash, pan)
    except SurePassError as exc:
        return _surepass_failure(
            deps, state, exc, node="pan_comprehensive", endpoint=PAN_ENDPOINT, label=MSG_PAN_SERVICE_FAILURE
        )
    identity = identity_from_envelope(result.raw)
    calls = _calls_for(repo, check_id, PAN_ENDPOINT)
    if identity.pan_valid:
        repo.put_pan_record(
            PanRecordRow(
                pan_hash=pan_hash,
                pan_last4=state.get("pan_last4") or pan_last4(pan),
                storage_key=_success_storage_key(calls, PAN_ENDPOINT),
                client_id=result.client_id,
                fetched_at=now,
                expires_at=pan_expiry(now),
                purge_after=now + RAW_RETENTION,
            )
        )
    update = {
        "identity": identity,
        "pan_called": True,
        "surepass_calls_made": [_audit(c) for c in calls],
        "transcript": ["pan_comprehensive"],
    }
    if not identity.pan_valid:
        update["status"] = "pan_not_verified"
    return update


# --------------------------------------------------------------------------- bureau_pull


def bureau_pull(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """Quick: Score (12 s) unless stored < 30 d. Full: Report (20 s) unless stored < 30 d; a Report pull also writes the
    score cache (same raw object, ``source='report'``). Reuse re-adapts the stored raw on current code (K5)."""
    del config
    repo = deps.repo
    decision = _admission(state)
    kind = _kind(state)
    check_id = _check_id(state)
    pan_hash = _pan_hash(state)

    if decision.bureau_plan == "reuse":
        if kind == "quick":
            score_row = repo.get_score_cache(pan_hash)
            if score_row is None:
                raise RuntimeError("admission planned a score reuse but no score_cache row exists")
            score = score_facts_from_envelope(
                repo.fetch_raw(score_row.storage_key), pulled_on=ist_date(score_row.fetched_at), source=score_row.source
            )
            return {"score": score, "bureau_called": False, "transcript": ["bureau_pull"]}
        report_row = repo.get_report_cache(pan_hash)
        if report_row is None:
            raise RuntimeError("admission planned a report reuse but no report_cache row exists")
        report = report_facts_from_envelope(repo.fetch_raw(report_row.storage_key), today=ist_date(report_row.fetched_at))
        return {"report": report, "bureau_called": False, "transcript": ["bureau_pull"]}

    identity = _identity(state)
    pan = state.get("pan")
    mobile = state.get("mobile")
    if not pan or not mobile:
        raise ValueError("a bureau call needs the raw PAN and mobile in state")
    # Bureau calls send PAN Comprehensive's full_name; the LC-typed name lives in the consent record (PRD 6).
    name = identity.full_name or state.get("name") or ""
    endpoint = SCORE_ENDPOINT if kind == "quick" else REPORT_ENDPOINT
    now = deps.now()
    try:
        if kind == "quick":
            result = deps.surepass.experian_score(check_id, pan_hash, name=name, mobile=mobile, pan=pan)
        else:
            result = deps.surepass.experian_report(check_id, pan_hash, name=name, mobile=mobile, pan=pan)
    except SurePassTimeout as exc:
        # K2: spend unknown. The client already wrote the billable sink row; open the reconciliation gate (lock expiry
        # alone never re-admits a billed pull) and lock the PAN for 10 minutes -> pending_verification, code None (K1).
        # The unresolved review row goes FIRST (fix round 1: it alone gates every re-pull, QK7 / FL6, should the lock
        # write fail) and the lock is written in ``finally`` (fix round 3: should the REVIEW write fail, the 10-minute
        # lock still stops an immediate second pull on a spend-unknown PAN). Either failure then escapes to the route.
        try:
            _insert_review(
                repo,
                now=now,
                check_id=check_id,
                pan_hash=pan_hash,
                kind="timeout_reconciliation",
                reason=f"{exc.endpoint} timed out (spend unknown); reconcile against the SurePass console before any "
                "re-pull",
            )
        finally:
            repo.create_pan_lock(pan_hash, "timeout", check_id, now + TIMEOUT_LOCK_TTL)
        view = build_failure_lc_view(
            check_id=check_id,
            kind=kind,
            status="pending_verification",
            code=None,
            label=MSG_PENDING_VERIFICATION,
            quick_checks_left_today=_quick_left(deps, state),
        )
        return {
            "status": "pending_verification",
            "code": None,
            "lc_view": view,
            "bureau_called": True,
            "surepass_calls_made": [_audit(c) for c in _calls_for(repo, check_id, endpoint)],
            "errors": [f"bureau_pull: SurePassTimeout: {exc.detail}"],
            "transcript": ["bureau_pull"],
        }
    except SurePassError as exc:
        return _surepass_failure(
            deps, state, exc, node="bureau_pull", endpoint=endpoint, label=MSG_BUREAU_SERVICE_FAILURE
        )

    calls = _calls_for(repo, check_id, endpoint)
    storage_key = _success_storage_key(calls, endpoint)
    consent_id = state.get("consent_id")
    audit = [_audit(c) for c in calls]
    if kind == "quick":
        score = score_facts_from_envelope(result.raw, pulled_on=ist_date(now), source="score")
        repo.put_score_cache(
            ScoreCacheRow(
                pan_hash=pan_hash,
                storage_key=storage_key,
                source="score",
                is_ntc=score.is_ntc,
                client_id=result.client_id,
                consent_id=consent_id,
                fetched_at=now,
                expires_at=score_expiry(now),
                purge_after=now + RAW_RETENTION,
            )
        )
        return {"score": score, "bureau_called": True, "surepass_calls_made": audit, "transcript": ["bureau_pull"]}

    report = report_facts_from_envelope(result.raw, today=ist_date(now))
    repo.put_report_cache(
        ReportCacheRow(
            pan_hash=pan_hash,
            storage_key=storage_key,
            is_ntc=report.is_ntc,
            report_date=report.report_date,
            client_id=result.client_id,
            consent_id=consent_id,
            fetched_at=now,
            expires_at=report_expiry(now),
            purge_after=now + RAW_RETENTION,
        )
    )
    # PRD 5.3: a Report pull ALSO refreshes the score cache, pointing at the SAME report object (score facts are read
    # from the report envelope's data.credit_score on a later hit).
    repo.put_score_cache(
        ScoreCacheRow(
            pan_hash=pan_hash,
            storage_key=storage_key,
            source="report",
            is_ntc=report.is_ntc,
            client_id=result.client_id,
            consent_id=consent_id,
            fetched_at=now,
            expires_at=score_expiry(now),
            purge_after=now + RAW_RETENTION,
        )
    )
    return {"report": report, "bureau_called": True, "surepass_calls_made": audit, "transcript": ["bureau_pull"]}


# --------------------------------------------------------------------------- derive


def _band(score: ScoreFacts | None, report: ReportFacts | None) -> Band:
    """PRD 8.1: the band of the Score or the Report, whichever is newer (report date vs score pull date)."""
    if report is not None and score is not None:
        return report.band if (report.report_date or date.min) >= score.pulled_on else score.band
    if report is not None:
        return report.band
    if score is not None:
        return score.band
    raise ValueError("derive needs score or report facts")


def derive(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """Band, existing EMI (8.3), FOIR, age (8.5) and candidate pincodes (8.6). All code."""
    del config
    kind = _kind(state)
    identity = _identity(state)
    score = state.get("score")
    report = state.get("report")
    today = ist_date(deps.now())  # the IST calendar date, like the adapters' anchors and pulled_on (fix round 1)
    band = _band(score, report)
    # T6 ruling: the EMI tenure maths runs from the report date, or today when the report carries none.
    anchor = report.report_date if report is not None and report.report_date is not None else today
    emi = derive_existing_emi(
        report.open_accounts if kind == "full" and report is not None else [],
        declared=state.get("existing_emi_declared"),
        kind=kind,
        anchor=anchor,
    )
    income = state.get("monthly_income")
    foir = emi.used / income if income is not None and income > 0 else None
    age_years, age_months = age_years_and_months(identity.dob, today) if identity.dob is not None else (None, None)
    pins = candidate_pincodes(identity, report, kind=kind)
    low, high = SCORE_RANGE
    scores = [s for s in (score.score if score else None, report.score if report else None) if s is not None]
    errors = ["derive: score_out_of_range"] if any(not low <= s <= high for s in scores) else []
    return {
        "derived": Derived(
            band=band,
            emi=emi,
            current_foir=foir,
            age_at_application_years=age_years,
            age_months_at_application=age_months,
            candidate_pincodes=pins,
        ),
        "candidate_pincodes": pins,
        "errors": errors,
        "transcript": ["derive"],
    }


# --------------------------------------------------------------------------- pincode_gate


def published_lender_row(row: LenderRow) -> PublishedLenderRow:
    """The gate's input from a db ``LenderRow`` (T8 seam): ``policy_notes`` mapped explicitly, pincodes tuple -> list."""
    return {
        "id": row.id,
        "name": row.name,
        "criteria": row.criteria,
        "matched_pincodes": list(row.matched_pincodes),
        "policy_notes": row.policy_notes,
    }


def pincode_gate(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """PRD 9.1: lenders whose PUBLISHED pincode list holds a candidate pincode. Zero candidates (quick, or a full check
    with none either: K22) -> NO_PINCODE_ON_PAN band read; the gate itself is never silently skipped."""
    del config
    pins = state.get("candidate_pincodes") or []
    if not pins:
        return {"code": "NO_PINCODE_ON_PAN", "gated_lenders": [], "gate_warnings": [], "transcript": ["pincode_gate"]}
    rows = [published_lender_row(row) for row in deps.repo.lenders_for_pincodes(pins) if row.published]
    gated, warnings = gate_lenders(rows, pins)
    return {"gated_lenders": gated, "gate_warnings": warnings, "transcript": ["pincode_gate"]}


# --------------------------------------------------------------------------- range / code_confidence


def range_(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """PRD 8.4 per gated lender: range, ceiling, displayed FOIR, age at maturity (display rounding applied once there)."""
    del config, deps
    derived = _derived(state)
    gated: list[GatedLender] = state.get("gated_lenders") or []
    numbers = {
        lender.id: lender_numbers(
            lender.criteria,
            lender_id=lender.id,
            income=state.get("monthly_income"),
            existing_emi=derived.emi.used,
            age_months=derived.age_months_at_application,
        )
        for lender in gated
    }
    return {"lender_numbers": numbers, "transcript": ["range"]}


def code_confidence(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """PRD 9.4 code confidence, before the model runs (re-capped after safety inside the view builder)."""
    del config, deps
    confidence = compute_code_confidence(
        kind=_kind(state),
        identity=_identity(state),
        derived=_derived(state),
        gated=state.get("gated_lenders") or [],
        numbers=state.get("lender_numbers") or {},
        openai_failed=False,
    )
    return {"code_confidence": confidence, "transcript": ["code_confidence"]}


# --------------------------------------------------------------------------- rank (the ONE LLM node)


def rank(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """One model call per visit. A raised exception or a non-ModelOutput is a failed attempt: the first is retried
    (the router loops back, attempt 2 carries ``prior_error``); the second maps to OPENAI_FAILURE. Never retried:
    business violations (safety demotes them) and a raised confidence (``min()`` ignores it)."""
    gated: list[GatedLender] = state.get("gated_lenders") or []
    attempt = state.get("rank_attempts", 0) + 1
    context = build_rank_context(state, gated)  # gated passed EXPLICITLY (T10 ruling); a wiring bug raises here
    try:
        output = deps.rank_model(context, config=config)
        if not isinstance(output, ModelOutput):  # pyright: ignore[reportUnnecessaryIsInstance]  (runtime guard)
            raise TypeError(f"rank model returned {type(output).__name__}, expected ModelOutput")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:PRIOR_ERROR_MAX_CHARS]
        update: StateUpdate = {
            "rank_attempts": attempt,
            "rank_error": error,
            "errors": [f"rank attempt {attempt}: {type(exc).__name__}"],
            "transcript": ["rank"],
        }
        if attempt >= RANK_MAX_ATTEMPTS:
            update.update({"status": "service_failure", "code": "OPENAI_FAILURE"})
        return update
    return {"model_output": output, "rank_attempts": attempt, "rank_error": None, "transcript": ["rank"]}


# --------------------------------------------------------------------------- safety


def safety(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """PRD 9.3: demote picks that breach the gate, floors, FOIR, age or minimum loan; survivors keep model order."""
    del config, deps
    derived = _derived(state)
    result = run_safety(
        state.get("model_output"),
        state.get("gated_lenders") or [],
        state.get("lender_numbers") or {},
        band=derived.band,
        age_app_years=derived.age_at_application_years,
        age_known=derived.age_at_application_years is not None,
    )
    return {"safety": result, "transcript": ["safety"]}


# --------------------------------------------------------------------------- redact


def _cache_dates(deps: GraphDeps, state: CheckState) -> tuple[date | None, date | None]:
    """``(checked_on, refresh_available_on)``: the admission's (cache hit / recompute: the ORIGINAL pull), else the
    governing cache row this run's pull just wrote. None when no bureau facts exist (e.g. an unverified PAN)."""
    decision = state.get("admission")
    if decision is not None and decision.checked_on is not None:
        return decision.checked_on, decision.refresh_available_on
    pan_hash = state.get("pan_hash")
    if not pan_hash:
        return None, None
    if state.get("kind") == "full" and state.get("report") is not None:
        row = deps.repo.get_report_cache(pan_hash)
    elif state.get("kind") == "quick" and state.get("score") is not None:
        row = deps.repo.get_score_cache(pan_hash)
    else:
        return None, None
    if row is None:
        return None, None
    return ist_date(row.fetched_at), ist_date(row.expires_at)


def redact(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """The view for every non-refusal, non-SurePass-failure path, built ONLY by ``build_lc_view`` (which owns confidence
    and headline; this node reads them back). ``lc_view`` is always set: a builder ``ValueError`` becomes the
    internal-error view, and an error view always leaves a guarded error row behind."""
    del config
    checked_on, refresh_on = _cache_dates(deps, state)
    decision = state.get("admission")
    ctx = RedactionContext(
        quick_checks_left_today=_quick_left(deps, state),
        checked_on=checked_on,
        from_cache=decision is not None and decision.bureau_plan == "reuse",
        refresh_available_on=refresh_on,
    )
    errors: list[str] = []
    try:
        view = build_lc_view(state, ctx)
    except ValueError as exc:
        errors.append(f"redact: {type(exc).__name__}: {exc}")
        view = build_error_lc_view(check_id=state.get("check_id") or "", kind=state.get("kind") or "quick")
    if _is_error_view(view):
        view, _, _ = _record_error(deps, state, errors[0] if errors else "redact: unhandled view state")
    return {
        "lc_view": view,
        "status": view.status,
        "code": view.code,
        "headline": view.headline,
        "final_confidence": view.confidence,
        "errors": errors,
        "transcript": ["redact"],
    }


# --------------------------------------------------------------------------- persist


def _review_items(state: CheckState, view: LcView) -> list[tuple[ReviewKind, str]]:
    """The review rows persist owns (K11). Admission / bureau / auth rows are written where they happen."""
    if view.code == "OPENAI_FAILURE":
        return [("openai_failure", "ranking model failed twice; band read and gated lenders shown")]
    if view.status != "ok" or view.code is not None:
        return []  # band read, unverified PAN, failures: nothing for persist to flag
    items: list[tuple[ReviewKind, str]] = []
    result = state.get("safety")
    model = state.get("model_output")
    if view.headline == "not_fundable_alone":
        items.append(("not_fundable_pattern", "no lender survived the gate or the band; co-applicant suggested"))
    elif result is not None and result.review:
        items.append(("safety_demoted_all", f"safety demoted {len(result.demoted)} pick(s); top pick did not survive"))
    if model is not None and model.review_flags:
        items.append(("model_review_flags", f"model raised {len(model.review_flags)} review flag(s)"))
    elif view.confidence == "low" and not items:
        items.append(("model_review_flags", "model lowered confidence to low"))
    return items


def _admission_json(decision: AdmissionDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    out: dict[str, Any] = {}
    for f in dataclasses.fields(decision):
        value = getattr(decision, f.name)
        out[f.name] = value.isoformat() if isinstance(value, date) else value
    return out


def _internal(
    deps: GraphDeps, state: CheckState, view: LcView, calls: Sequence[SurePassCallRow], *, node: str
) -> dict[str, Any]:
    """``checks.internal``: facts minus raw payloads. NEVER the raw PAN, the mobile, a name, the PAN date of birth or a
    storage key (raw objects are reachable relationally through surepass_calls)."""
    decision = state.get("admission")
    identity = state.get("identity")
    score = state.get("score")
    report = state.get("report")
    derived = state.get("derived")
    model = state.get("model_output")
    result = state.get("safety")
    return {
        "admission": _admission_json(decision),
        "original_consent_id": decision.original_consent_id if decision is not None else None,
        "is_recompute": state.get("is_recompute") is True,
        "identity": identity.model_dump(mode="json", exclude={"full_name", "dob"}) if identity is not None else None,
        "score": score.model_dump(mode="json") if score is not None else None,
        "report": report.model_dump(mode="json") if report is not None else None,
        "derived": derived.model_dump(mode="json") if derived is not None else None,
        "gated_lender_ids": [g.id for g in state.get("gated_lenders") or []],
        "gate_warnings": list(state.get("gate_warnings") or []),
        "lender_numbers": {k: v.model_dump(mode="json") for k, v in (state.get("lender_numbers") or {}).items()},
        "code_confidence": state.get("code_confidence"),
        "model_output": model.model_dump(mode="json") if model is not None else None,
        "safety": result.model_dump(mode="json") if result is not None else None,
        "final_confidence": view.confidence,
        "headline": view.headline,
        "rank_attempts": state.get("rank_attempts", 0),
        "rank_error": state.get("rank_error"),
        "surepass_calls": [{"endpoint": c.endpoint, "billable": c.billable, "error_kind": c.error_kind} for c in calls],
        "spend_unknown": view.status == "pending_verification",
        "prompts_version": deps.prompts.version,
        "errors": list(state.get("errors") or []),
        "transcript": [*(state.get("transcript") or []), node],
        "langsmith_run_id": state.get("langsmith_run_id"),
    }


def persist(state: CheckState, config: RunnableConfig, deps: GraphDeps) -> StateUpdate:
    """The terminal for every admitted run: persist's review rows, THEN the ONE guarded ``update_check`` (status / code
    from the view; billed / counted from the policy formulas, status-independent per K8; is_refresh from the decision).

    Write order (fix round 1): the terminal update is the LAST write, so any earlier failure (a review insert included)
    leaves the row running for the error path to flip, and the LC's view always matches the final row. ANY exception
    becomes a guarded error row carrying the true billed / counted flags + the error view; the in-flight lock is
    released in ``finally``."""
    del config
    check_id = state.get("check_id")
    pan_hash = state.get("pan_hash")
    billed = counted = False
    view = state.get("lc_view")
    try:
        if view is None:
            raise ValueError("persist reached without an lc_view")
        repo = deps.repo
        if _is_error_view(view):
            # The row was already flipped to 'error' (with its billed / counted flags) where the view was produced;
            # report what it holds so the trace agrees with the row.
            row = repo.get_check(_check_id(state))
            billed, counted = (row.billed, row.counted_against_quota) if row is not None else (False, False)
        else:
            decision = _admission(state)
            kind = _kind(state)
            calls = repo.surepass_calls_for_check(_check_id(state))
            patch: CheckPatch = {
                "status": view.status,
                "code": view.code,
                "billed": compute_billed(calls),
                "counted_against_quota": compute_counted_against_quota(
                    kind=kind, counts_against_quota=decision.counts_against_quota, calls=calls
                ),
                "is_refresh": decision.is_refresh,
                "lc_view": view.model_dump(mode="json"),
                "internal": _internal(deps, state, view, calls, node="persist"),
                "langsmith_run_id": state.get("langsmith_run_id"),
            }
            now = deps.now()
            for review_kind, reason in _review_items(state, view):
                _insert_review(repo, now=now, check_id=_check_id(state), pan_hash=_pan_hash(state), kind=review_kind,
                               reason=reason)
            repo.update_check(_check_id(state), patch)
            billed, counted = patch["billed"], patch["counted_against_quota"]
    except Exception as exc:  # ANY exception, not just ValueError (T2 note)
        logger.error("persist failed for check_id=%s (%s)", check_id, type(exc).__name__)
        view, billed, counted = _record_error(deps, state, f"persist: {type(exc).__name__}")
    finally:
        _release_in_flight(deps, pan_hash, check_id)
    return {
        "lc_view": view,
        "status": view.status,
        "code": view.code,
        "billed": billed,
        "counted_against_quota": counted,
        **_STRIPPED_IDENTIFIERS,
        "transcript": ["persist"],
    }
