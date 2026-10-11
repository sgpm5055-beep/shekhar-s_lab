"""Task 11: graph wiring transcripts, node side effects, persist/lock guarantees, recompute entry, serde, checkpointer.

Every test builds its world explicitly (no conftest yet): MemoryRepo on a frozen clock, a FakeSurePassClient logging into
that repo's sink, FakeRankModel, GraphDeps. Fixtures are synthetic (tests/graph_builders.py).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langsmith import Client

from eligibility_agent.db.repo import CheckRow, LenderRow, SurePassCallRow
from eligibility_agent.graph import build, nodes, tracing
from eligibility_agent.graph import state as state_module
from eligibility_agent.graph.build import (
    NODE_NAMES,
    RANK_MAX_ATTEMPTS,
    RECOMPUTE_NODE_NAMES,
    CheckpointerConfigError,
    assert_checkpointer_isolated,
    assert_checkpointer_schema,
    build_graph,
    check_checkpointer_dsn,
    make_checkpoint_serde,
    make_postgres_checkpointer,
    route_admission,
    route_bureau,
    route_confidence,
    route_gate,
    route_pan,
    route_rank,
    seed_recompute,
    state_msgpack_allowlist,
)
from eligibility_agent.graph.prompts import local_prompts
from eligibility_agent.graph.state import (
    AdmissionDecision,
    CheckState,
    ConsentInfo,
    EmiDerivation,
    Identity,
    LcView,
    ReportFacts,
)
from eligibility_agent.output.lc_view import (
    MSG_BUREAU_SERVICE_FAILURE,
    MSG_INTERNAL,
    MSG_OPENAI_FAILURE,
    MSG_PAN_NOT_VERIFIED,
    MSG_PAN_SERVICE_FAILURE,
    MSG_PENDING_VERIFICATION,
    RATIONALE_RANKING_UNAVAILABLE,
)
from eligibility_agent.policy.guards import IN_FLIGHT_LOCK_TTL, TIMEOUT_LOCK_TTL, next_ist_midnight
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
    SurePassTimeout,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.fake_llm import (
    FakeRankModel,
    default_ranking,
    invalid_output_error,
    out_of_gate_pick,
    raised_confidence,
    ranked_output,
)
from eligibility_agent.testing.scenarios import (
    PAN_INVALID,
    PAN_NTC,
    PAN_OK,
    full_happy_reported_emi_scenario,
    happy_quick_scenario,
    no_pincode_scenario,
    ntc_scenario,
    pan_invalid_scenario,
    score_failure_scenario,
)
from tests.graph_builders import (
    L1_NOTES,
    LC_ID,
    LC_TYPED_NAME,
    OTHER_PIN,
    T0,
    SurePassFactory,
    World,
    check_state,
    invoke,
    make_world,
    nothing_configured,
    pan_hash_of,
    run,
    run_recompute,
)

FULL_PATH = [
    "admission",
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
]


def happy_quick(sink: CallLogSink) -> FakeSurePassClient:
    return happy_quick_scenario(sink=sink)


def happy_full(sink: CallLogSink) -> FakeSurePassClient:
    return full_happy_reported_emi_scenario(sink=sink)


def view_of(final: dict[str, Any]) -> LcView:
    view = final["lc_view"]
    assert isinstance(view, LcView)
    return view


def endpoints(world: World, check_id: str) -> list[str]:
    return [c.endpoint for c in world.calls(check_id)]


def review_kinds(world: World) -> list[str]:
    return sorted(r.kind for r in world.reviews())


def seed_other_check(world: World, check_id: str, pan_hash: str) -> None:
    """A running checks row owned by someone else (the FK parent for locks / calls the test plants)."""
    world.repo.insert_check(
        CheckRow(
            id=check_id, kind="quick", lc_id="lc-other", lead_id=None, opportunity_id=None, opportunity_stage=None,
            contact_id=None, pan_hash=pan_hash, pan_last4="234F", mobile_hash=None, mobile_last4=None, status="running",
            code=None, billed=False, counted_against_quota=False, is_refresh=False, recompute_of=None, consent_id=None,
            lc_view=None, internal=None, langsmith_run_id=None, created_at=world.clock.now(),
        )
    )


# =========================================================================== routers (pure)


def test_routers_key_off_admission_code_status_and_counters() -> None:
    refused: CheckState = {"admission": AdmissionDecision(code="IN_FLIGHT")}
    admitted: CheckState = {"admission": AdmissionDecision(admission="new", pan_plan="call", bureau_plan="call")}
    assert route_admission(refused) == "refused"
    assert route_admission(admitted) == "proceed"
    assert route_admission({}) == "refused"  # no decision is never a licence to call out

    valid = Identity(pan_valid=True, status="valid", category="person", full_name=None, dob=None, aadhaar_linked=None,
                     pan_pincode=None, surepass_client_id=None)
    assert route_pan({"status": "service_failure"}) == "failed"
    assert route_pan({"identity": valid.model_copy(update={"pan_valid": False})}) == "invalid"
    assert route_pan({}) == "invalid"
    assert route_pan({"identity": valid}) == "valid"

    assert route_bureau({"status": "pending_verification"}) == "terminal"
    assert route_bureau({"status": "service_failure"}) == "terminal"
    assert route_bureau({}) == "ok"

    assert route_gate({"code": "NO_PINCODE_ON_PAN"}) == "band_read"
    assert route_gate({}) == "proceed"

    assert route_confidence({"gated_lenders": []}) == "skip_rank"
    assert route_confidence({}) == "skip_rank"

    assert route_rank({"model_output": default_ranking_output()}) == "safety"
    assert route_rank({"code": "OPENAI_FAILURE"}) == "failed"
    assert route_rank({"rank_attempts": 1}) == "retry"
    assert route_rank({"rank_attempts": RANK_MAX_ATTEMPTS}) == "failed"  # defensive: never a third call
    assert RANK_MAX_ATTEMPTS == 2


def default_ranking_output() -> Any:
    return ranked_output("L1")


def test_node_name_registries() -> None:
    assert NODE_NAMES == (
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
    assert RECOMPUTE_NODE_NAMES == (
        "derive", "pincode_gate", "range", "code_confidence", "rank", "safety", "redact", "persist",
    )


# =========================================================================== happy paths


def test_quick_happy_path_transcript_and_side_effects() -> None:
    world = make_world(happy_quick)
    final = run(world, check_state("quick", "chk-q1"))
    view = view_of(final)

    assert final["transcript"] == FULL_PATH
    assert endpoints(world, "chk-q1") == [PAN_ENDPOINT, SCORE_ENDPOINT]
    assert world.rank.call_count == 1
    assert view.status == "ok" and view.code is None
    assert view.headline == "possible" and view.confidence == "medium"  # quick caps at medium (K16)
    assert view.top_lender is not None and view.top_lender.id == "L1"
    assert view.from_cache is False and view.checked_on == "2026-10-10" and view.refresh_available_on == "2026-11-09"
    assert view.quick_checks_left_today == 4  # 5 - 0 used - this counted run

    row = world.row("chk-q1")
    assert (row.status, row.code, row.billed, row.counted_against_quota, row.is_refresh) == ("ok", None, True, True, False)
    assert row.lc_view == view.model_dump(mode="json")
    assert final["billed"] is True and final["counted_against_quota"] is True

    consents = world.repo.list_consents()
    assert len(consents) == 1 and consents[0].covers == "score" and consents[0].lc_typed_name == LC_TYPED_NAME
    assert row.consent_id == consents[0].id == final["consent_id"]

    pan_hash = pan_hash_of()
    pan_row = world.repo.get_pan_record(pan_hash)
    score_row = world.repo.get_score_cache(pan_hash)
    assert pan_row is not None and score_row is not None
    calls = {c.endpoint: c for c in world.calls("chk-q1")}
    assert pan_row.storage_key == calls[PAN_ENDPOINT].storage_key  # caches point at the sink's raw objects (K5)
    assert score_row.storage_key == calls[SCORE_ENDPOINT].storage_key
    assert (score_row.source, score_row.is_ntc, score_row.consent_id) == ("score", False, consents[0].id)
    assert not world.in_flight_held(pan_hash)
    assert world.reviews() == []


def test_full_happy_path_report_pull_writes_both_caches() -> None:
    world = make_world(happy_full)
    final = run(world, check_state("full", "chk-f1"))
    view = view_of(final)

    assert final["transcript"] == FULL_PATH
    assert endpoints(world, "chk-f1") == [PAN_ENDPOINT, REPORT_ENDPOINT]  # a full check never calls Score
    assert view.status == "ok" and view.headline == "fundable" and view.confidence == "high"
    assert view.label == "bureau report on 2026-10-01, income as declared"
    assert view.quick_checks_left_today is None

    pan_hash = pan_hash_of()
    report_row = world.repo.get_report_cache(pan_hash)
    score_row = world.repo.get_score_cache(pan_hash)
    report_call = world.calls("chk-f1")[-1]
    assert report_row is not None and score_row is not None
    assert report_row.storage_key == score_row.storage_key == report_call.storage_key  # ONE raw object, two pointers
    assert score_row.source == "report" and report_row.report_date == date(2026, 10, 1)
    assert world.repo.list_consents()[0].covers == "report"

    row = world.row("chk-f1")
    assert (row.status, row.billed, row.counted_against_quota) == ("ok", True, False)  # reports sit outside any quota
    assert [e.event for e in world.repo.pan_events_for_opportunity("opp-graph-1")] == ["initial"]


def test_quick_ntc_is_counted_and_cached_and_unlikely() -> None:
    world = make_world(lambda sink: ntc_scenario("empty", sink=sink))
    final = run(world, check_state("quick", "chk-ntc", pan=PAN_NTC))
    view = view_of(final)
    assert view.status == "ok" and view.headline == "unlikely" and view.coapplicant_suggested is True
    assert view.top_lender is not None and view.top_lender.id == "L2"  # the only NTC-accepting lender
    row = world.row("chk-ntc")
    assert (row.billed, row.counted_against_quota) == (True, True)
    score_row = world.repo.get_score_cache(pan_hash_of(PAN_NTC))
    assert score_row is not None and score_row.is_ntc is True


# =========================================================================== admission: order, refusals, directives


def test_admission_write_order_consent_then_check_then_locks_then_acquire() -> None:
    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-order"))
    log = world.repo.log
    order = [log.index(name) for name in ("insert_consent", "insert_check", "get_active_locks", "try_acquire_pan_lock")]
    assert order == sorted(order)  # K10: insert_check (1.5) BEFORE any lock read or acquire (FK on pan_locks.check_id)


def test_consent_missing_refuses_with_a_row_and_no_consent_or_lock() -> None:
    world = make_world(nothing_configured)
    final = run(world, check_state("quick", "chk-noconsent", with_consent=False))
    view = view_of(final)
    assert final["transcript"] == ["admission", "persist_refusal"]
    assert (view.status, view.code) == ("refused", "CONSENT_MISSING")
    assert world.repo.list_consents() == []
    assert "try_acquire_pan_lock" not in world.repo.log
    row = world.row("chk-noconsent")
    assert (row.status, row.code, row.consent_id, row.billed) == ("refused", "CONSENT_MISSING", None, False)


def test_in_flight_refusal_leaves_a_row_and_never_releases_the_holders_lock() -> None:
    world = make_world(nothing_configured)
    pan_hash = pan_hash_of()
    seed_other_check(world, "chk-holder", pan_hash)
    assert world.repo.try_acquire_pan_lock(pan_hash, "chk-holder", T0 + IN_FLIGHT_LOCK_TTL)

    final = run(world, check_state("quick", "chk-second"))
    view = view_of(final)
    assert final["transcript"] == ["admission", "persist_refusal"]
    assert (view.status, view.code) == ("refused", "IN_FLIGHT")
    row = world.row("chk-second")  # the refusal is auditable: insert-early (K10 1.5)
    assert (row.status, row.code) == ("refused", "IN_FLIGHT")
    held = [lock for lock in world.repo.get_active_locks(pan_hash) if lock.kind == "in_flight"]
    assert [lock.check_id for lock in held] == ["chk-holder"]  # holder-only release


def test_active_timeout_lock_refuses_pan_locked_even_for_a_would_be_cache_hit() -> None:
    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-first"))  # score + PAN now cached: the next run would be a QK1 cache hit
    pan_hash = pan_hash_of()
    world.repo.create_pan_lock(pan_hash, "timeout", "chk-first", T0 + TIMEOUT_LOCK_TTL)
    world.repo.log.clear()

    final = run(world.with_surepass(nothing_configured), check_state("quick", "chk-locked"))
    assert (view_of(final).status, view_of(final).code) == ("refused", "PAN_LOCKED")
    assert "try_acquire_pan_lock" not in world.repo.log  # the lock gate runs before the acquire and the tables
    assert "fetch_raw" not in world.repo.log  # the would-be cache hit is never served


def test_active_daily_cap_lock_refuses_pan_daily_cap() -> None:
    world = make_world(nothing_configured)
    pan_hash = pan_hash_of()
    seed_other_check(world, "chk-earlier", pan_hash)
    world.repo.create_pan_lock(pan_hash, "daily_cap", "chk-earlier", next_ist_midnight(T0))
    final = run(world, check_state("quick", "chk-capped"))
    assert view_of(final).code == "PAN_DAILY_CAP"


def test_hard_cap_refusal_writes_the_daily_cap_lock_and_its_review_row() -> None:
    world = make_world(nothing_configured)
    pan_hash = pan_hash_of()
    seed_other_check(world, "chk-billed", pan_hash)
    for _ in range(3):
        world.repo.log_surepass_call(
            check_id="chk-billed", pan_hash=pan_hash, endpoint=SCORE_ENDPOINT, client_id=None, http_status=200,
            success=True, message_code="success", latency_ms=1, error_kind=None, raw_envelope=None,
        )
    final = run(world, check_state("quick", "chk-cap"))
    assert view_of(final).code == "PAN_DAILY_CAP"
    locks = {lock.kind: lock for lock in world.repo.get_active_locks(pan_hash)}
    assert locks["daily_cap"].lock_until == next_ist_midnight(T0) and locks["daily_cap"].check_id == "chk-cap"
    assert "in_flight" not in locks  # released by persist_refusal
    assert review_kinds(world) == ["pan_daily_cap"]  # admission owns this kind (K11)


def test_live_calls_disabled_refusal_after_acquire_releases_the_lock() -> None:
    world = make_world(lambda sink: FakeSurePassClient(sink=sink, calls_enabled=False))
    final = run(world, check_state("quick", "chk-off"))
    assert view_of(final).code == "LIVE_CALLS_DISABLED"
    assert "try_acquire_pan_lock" in world.repo.log and "release_pan_lock" in world.repo.log
    assert not world.in_flight_held(pan_hash_of())


def test_full_stage_gate_refuses_stage_too_early() -> None:
    world = make_world(nothing_configured)
    final = run(world, check_state("full", "chk-stage", stage="counselling"))
    assert view_of(final).code == "STAGE_TOO_EARLY"


def test_full_warn_ntc_recent_carries_the_found_date() -> None:
    world = make_world(lambda sink: ntc_scenario("text", sink=sink))
    run(world, check_state("quick", "chk-ntc-quick", pan=PAN_NTC))
    world.clock.advance(timedelta(days=2))
    final = run(world.with_surepass(nothing_configured), check_state("full", "chk-ntc-full", pan=PAN_NTC))
    view = view_of(final)
    assert view.code == "WARN_NTC_RECENT" and "2026-10-10" in view.label


def test_pan_change_after_a_report_needs_a_manager_and_writes_its_review_row() -> None:
    world = make_world(happy_full)
    run(world, check_state("full", "chk-report"))
    other_pan = "ABCPX5678K"
    final = run(world.with_surepass(nothing_configured), check_state("full", "chk-newpan", pan=other_pan))
    assert view_of(final).code == "PAN_CHANGE_NEEDS_MANAGER"
    assert review_kinds(world) == ["pan_change_manager"]


def test_one_pan_correction_before_any_report_is_recorded_on_the_admitted_check() -> None:
    wrong_pan = PAN_INVALID

    def correction_world(sink: CallLogSink) -> FakeSurePassClient:
        account = env.cais_account(scheduled_emi="3500", date_reported="20260920")
        return FakeSurePassClient(
            pan={
                wrong_pan: env.pan_comprehensive_envelope(pan=wrong_pan, status="invalid"),
                PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK),
            },
            report={PAN_OK: env.experian_report_envelope(accounts=[account])},
            sink=sink,
        )

    world = make_world(correction_world)
    first = run(world, check_state("full", "chk-wrong", pan=wrong_pan))
    assert view_of(first).status == "pan_not_verified"  # admitted, so the first PAN is recorded; no report exists
    final = run(world, check_state("full", "chk-corrected"))
    assert view_of(final).status == "ok"
    assert [e.event for e in world.repo.pan_events_for_opportunity("opp-graph-1")] == ["initial", "correction"]
    decision = final["admission"]
    assert isinstance(decision, AdmissionDecision) and decision.record_pan_correction is True


def test_a_refused_full_check_records_no_pan_event() -> None:
    world = make_world(live_off)
    final = run(world, check_state("full", "chk-off-full"))
    assert view_of(final).code == "LIVE_CALLS_DISABLED"
    assert world.repo.pan_events_for_opportunity("opp-graph-1") == []  # a refused attempt never spends the correction


def test_expired_score_refresh_is_billed_not_counted_and_marks_is_refresh() -> None:
    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-old"))
    world.clock.advance(timedelta(days=31))  # score window (30d) expired, PAN window (180d) still fresh
    later = world.with_surepass(happy_quick)
    final = run(later, check_state("quick", "chk-refresh"))
    assert endpoints(later, "chk-refresh") == [SCORE_ENDPOINT]  # PAN reused
    row = later.row("chk-refresh")
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (True, False, True)
    assert view_of(final).quick_checks_left_today == 5


# =========================================================================== PAN node


@pytest.mark.parametrize("kind", ["status_invalid", "company"])
def test_pan_not_verified_stops_before_the_bureau(kind: str) -> None:
    world = make_world(lambda sink: pan_invalid_scenario(cast(Any, kind), pan=PAN_OK, sink=sink))
    final = run(world, check_state("quick", "chk-pnv"))
    view = view_of(final)
    assert final["transcript"] == ["admission", "pan_comprehensive", "redact", "persist"]
    assert (view.status, view.code, view.confidence, view.label) == ("pan_not_verified", None, None, MSG_PAN_NOT_VERIFIED)
    assert endpoints(world, "chk-pnv") == [PAN_ENDPOINT]
    row = world.row("chk-pnv")
    assert (row.status, row.billed, row.counted_against_quota) == ("pan_not_verified", True, False)  # Rs 1, no quota
    assert world.repo.get_pan_record(pan_hash_of()) is None  # nothing cached, nothing marked checked
    assert not world.in_flight_held(pan_hash_of())


def test_pan_failure_envelope_is_a_service_failure_never_pan_not_verified() -> None:
    world = make_world(lambda sink: pan_invalid_scenario("failure_envelope", pan=PAN_OK, sink=sink))
    final = run(world, check_state("quick", "chk-pfail"))
    view = view_of(final)
    assert final["transcript"] == ["admission", "pan_comprehensive", "persist"]
    assert (view.status, view.code, view.label) == ("service_failure", "SUREPASS_FAILURE", MSG_PAN_SERVICE_FAILURE)
    row = world.row("chk-pfail")
    assert (row.status, row.code, row.counted_against_quota) == ("service_failure", "SUREPASS_FAILURE", False)
    assert view.quick_checks_left_today == 5
    assert world.repo.get_active_locks(pan_hash_of()) == []  # a PAN-endpoint failure never locks the PAN


def test_pan_timeout_is_a_plain_service_failure_and_never_locks_the_pan() -> None:
    def pan_timeout(sink: CallLogSink) -> FakeSurePassClient:
        return FakeSurePassClient(pan={PAN_OK: SurePassTimeout(PAN_ENDPOINT, "timeout: ReadTimeout")}, sink=sink)

    world = make_world(pan_timeout)
    final = run(world, check_state("quick", "chk-pantimeout"))
    view = view_of(final)
    assert (view.status, view.code, view.label) == ("service_failure", "SUREPASS_FAILURE", MSG_PAN_SERVICE_FAILURE)
    assert world.repo.get_active_locks(pan_hash_of()) == []  # PRD 5.12 locks on Score / Report timeouts only
    assert world.reviews() == []
    assert world.row("chk-pantimeout").billed is True  # the timeout row is conservatively billable (Rs 1)


# =========================================================================== bureau node


def test_bureau_timeout_is_pending_with_no_code_a_timeout_lock_and_a_reconciliation_row() -> None:
    world = make_world(lambda sink: score_failure_scenario("timeout", sink=sink))
    final = run(world, check_state("quick", "chk-timeout"))
    view = view_of(final)
    assert final["transcript"] == ["admission", "pan_comprehensive", "bureau_pull", "persist"]
    assert (view.status, view.code, view.label) == ("pending_verification", None, MSG_PENDING_VERIFICATION)  # K1
    pan_hash = pan_hash_of()
    locks = {lock.kind: lock for lock in world.repo.get_active_locks(pan_hash)}
    assert set(locks) == {"timeout"} and locks["timeout"].lock_until == T0 + TIMEOUT_LOCK_TTL
    reviews = world.reviews()
    assert [(r.kind, r.resolved_at, r.check_id) for r in reviews] == [("timeout_reconciliation", None, "chk-timeout")]
    row = world.row("chk-timeout")
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("pending_verification", None, True, False)
    assert row.internal is not None and row.internal["spend_unknown"] is True
    assert world.repo.get_score_cache(pan_hash) is None


@pytest.mark.parametrize("variant", ["success_false", "http_500"])
def test_bureau_service_failure_keeps_quota_and_caches_nothing(variant: str) -> None:
    world = make_world(lambda sink: score_failure_scenario(cast(Any, variant), sink=sink))
    final = run(world, check_state("quick", "chk-bfail"))
    view = view_of(final)
    assert final["transcript"] == ["admission", "pan_comprehensive", "bureau_pull", "persist"]
    assert (view.status, view.code, view.label) == ("service_failure", "SUREPASS_FAILURE", MSG_BUREAU_SERVICE_FAILURE)
    pan_hash = pan_hash_of()
    assert world.repo.get_score_cache(pan_hash) is None
    assert world.repo.get_pan_record(pan_hash) is not None  # PAN result kept (PRD 10)
    row = world.row("chk-bfail")
    assert (row.billed, row.counted_against_quota) == (True, False)
    assert world.repo.get_active_locks(pan_hash) == []
    assert world.reviews() == []


def test_surepass_auth_failure_alerts_and_writes_a_surepass_auth_row(caplog: pytest.LogCaptureFixture) -> None:
    world = make_world(lambda sink: score_failure_scenario("auth_403", sink=sink))
    with caplog.at_level(logging.CRITICAL, logger="eligibility_agent.graph.nodes"):
        final = run(world, check_state("quick", "chk-auth"))
    assert view_of(final).code == "SUREPASS_FAILURE"
    assert review_kinds(world) == ["surepass_auth"]
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical and all(PAN_OK not in r.getMessage() for r in critical)


# =========================================================================== cache hit (K5)


def test_cache_hit_makes_no_surepass_call_but_refetches_raw_and_reruns_adapters_and_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-pull"))
    world.clock.advance(timedelta(days=3))

    adapted: list[str] = []
    real_identity, real_score = nodes.identity_from_envelope, nodes.score_facts_from_envelope

    def spy_identity(raw: dict[str, Any]) -> Identity:
        adapted.append("identity")
        return real_identity(raw)

    def spy_score(raw: dict[str, Any], **kwargs: Any) -> Any:
        adapted.append("score")
        return real_score(raw, **kwargs)

    monkeypatch.setattr(nodes, "identity_from_envelope", spy_identity)
    monkeypatch.setattr(nodes, "score_facts_from_envelope", spy_score)
    hit = world.with_surepass(nothing_configured)  # ANY SurePass call would raise
    world.repo.log.clear()
    final = run(hit, check_state("quick", "chk-hit", income=45000.0))
    view = view_of(final)

    assert final["transcript"] == FULL_PATH
    assert hit.surepass.calls == [] and hit.calls("chk-hit") == []
    assert world.repo.log.count("fetch_raw") == 2  # PAN object + score object
    assert adapted == ["identity", "score"]  # current adapter code re-runs on the stored raw
    assert world.rank.call_count == 2  # rank re-runs on the current inputs
    assert view.from_cache is True and view.checked_on == "2026-10-10"  # the ORIGINAL pull date
    row = world.row("chk-hit")
    assert (row.billed, row.counted_against_quota) == (False, False)
    internal = row.internal
    assert internal is not None and internal["original_consent_id"] == world.row("chk-pull").consent_id


# =========================================================================== band read / zero gated / rank


def test_no_pincode_band_read_skips_range_rank_and_safety() -> None:
    world = make_world(lambda sink: no_pincode_scenario(sink=sink))
    final = run(world, check_state("quick", "chk-nopin"))
    view = view_of(final)
    assert final["transcript"] == ["admission", "pan_comprehensive", "bureau_pull", "derive", "pincode_gate", "redact", "persist"]
    assert world.rank.call_count == 0
    assert (view.status, view.code, view.confidence) == ("ok", "NO_PINCODE_ON_PAN", None)  # K15
    assert "lenders_for_pincodes" not in world.repo.log
    row = world.row("chk-nopin")
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("ok", "NO_PINCODE_ON_PAN", True, True)


def test_zero_gated_lenders_skips_the_llm_and_safety() -> None:
    world = make_world(happy_quick, lender_pincodes=(OTHER_PIN,))
    final = run(world, check_state("quick", "chk-nogate"))
    view = view_of(final)
    assert final["transcript"] == [
        "admission", "pan_comprehensive", "bureau_pull", "derive", "pincode_gate", "range", "code_confidence", "redact",
        "persist",
    ]
    assert world.rank.call_count == 0
    assert (view.headline, view.confidence, view.coapplicant_suggested) == ("not_fundable_alone", "low", True)
    assert view.top_lender is None
    assert review_kinds(world) == ["not_fundable_pattern"]


def test_openai_failure_after_exactly_two_rank_calls() -> None:
    world = make_world(happy_quick, rank=FakeRankModel.invalid_twice())
    final = run(world, check_state("quick", "chk-oai"))
    view = view_of(final)
    assert world.rank.call_count == 2
    assert final["transcript"] == FULL_PATH[:8] + ["rank", "redact", "persist"]
    second = world.rank.contexts[1]
    assert second.attempt == 2 and second.prior_error is not None
    assert world.rank.contexts[0].attempt == 1 and world.rank.contexts[0].prior_error is None
    assert (view.status, view.code, view.label, view.confidence) == ("service_failure", "OPENAI_FAILURE", MSG_OPENAI_FAILURE, "low")
    assert view.headline == "possible" and view.top_lender is None
    assert [a.rationale for a in view.alternates] == [RATIONALE_RANKING_UNAVAILABLE] * len(view.alternates) != []
    row = world.row("chk-oai")
    assert (row.status, row.code, row.billed, row.counted_against_quota) == ("service_failure", "OPENAI_FAILURE", True, True)
    assert review_kinds(world) == ["openai_failure"]


def test_one_retry_on_invalid_output_then_success() -> None:
    world = make_world(happy_quick, rank=FakeRankModel(script=[invalid_output_error(), default_ranking]))
    final = run(world, check_state("quick", "chk-retry"))
    assert world.rank.call_count == 2 and world.rank.contexts[1].prior_error is not None
    assert view_of(final).status == "ok" and final["transcript"].count("rank") == 2


def test_rank_passes_the_graph_config_through_to_the_model() -> None:
    world = make_world(happy_quick)
    config = tracing.invoke_config(kind="quick", check_id="chk-cfg", lc_id=LC_ID, pan_hash=pan_hash_of(), run_id=None)
    invoke(build_graph("quick", world.deps), check_state("quick", "chk-cfg"), config)
    seen = world.rank.configs[0]
    assert seen is not None and seen.get("metadata", {}).get("check_id") == "chk-cfg"


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ([out_of_gate_pick()], ["safety_demoted_all"]),
        ([ranked_output("L1", confidence="medium", review_flags=["declared income unverified"])], ["model_review_flags"]),
    ],
)
def test_persist_owns_the_model_and_safety_review_kinds(script: list[Any], expected: list[str]) -> None:
    world = make_world(happy_quick, rank=FakeRankModel(script=script))
    run(world, check_state("quick", "chk-review"))
    assert review_kinds(world) == expected
    assert world.rank.call_count == 1  # a business violation is safety's job: never retried


def test_a_raised_confidence_is_ignored_and_never_retried() -> None:
    world = make_world(happy_quick, rank=FakeRankModel(script=[raised_confidence("high")]))
    final = run(world, check_state("quick", "chk-raised"))
    assert world.rank.call_count == 1
    assert view_of(final).confidence == "medium"  # min(code medium, model high)


def test_no_node_carries_an_automatic_retry_policy() -> None:
    graph = build_graph("full", make_world(nothing_configured).deps)
    specs = graph.builder.nodes
    assert set(specs) == set(NODE_NAMES)
    assert all(not spec.retry_policy for spec in specs.values())  # PRD 5.12: no automatic retries anywhere


# =========================================================================== derive / gate seams


def test_derive_passes_the_report_date_as_the_emi_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    anchors: list[date | None] = []
    real = nodes.derive_existing_emi

    def spy(accounts: Any, *, declared: float | None, kind: Any, anchor: date | None = None) -> EmiDerivation:
        anchors.append(anchor)
        return real(accounts, declared=declared, kind=kind, anchor=anchor)

    monkeypatch.setattr(nodes, "derive_existing_emi", spy)
    run(make_world(happy_full), check_state("full", "chk-anchor"))
    assert anchors == [date(2026, 10, 1)]


def test_derive_falls_back_to_today_when_the_report_has_no_date(monkeypatch: pytest.MonkeyPatch) -> None:
    anchors: list[date | None] = []
    real = nodes.derive_existing_emi

    def spy(accounts: Any, *, declared: float | None, kind: Any, anchor: date | None = None) -> EmiDerivation:
        anchors.append(anchor)
        return real(accounts, declared=declared, kind=kind, anchor=anchor)

    monkeypatch.setattr(nodes, "derive_existing_emi", spy)
    # 2026-10-10T20:00Z is 2026-10-11 01:30 IST: the UTC and IST calendar dates differ here.
    world = make_world(nothing_configured, start=datetime(2026, 10, 10, 20, 0, tzinfo=T0.tzinfo))
    report = ReportFacts(
        band="720-780", score=745, is_ntc=False, report_date=None, open_accounts=[], past_due_total=0.0,
        written_off_or_settled=0, enquiries_30d=0, enquiries_90d=0, enquiries_180d=0, bureau_pincodes_12m=[],
    )
    identity = Identity(pan_valid=True, status="valid", category="person", full_name=None, dob=date(2000, 10, 11),
                        aadhaar_linked=None, pan_pincode="560001", surepass_client_id=None)
    state: CheckState = {"kind": "full", "check_id": "chk-x", "identity": identity, "report": report, "monthly_income": 1.0}
    update = nodes.derive(state, cast(RunnableConfig, {}), world.deps)
    assert anchors == [date(2026, 10, 11)]  # the IST date, like the adapters' anchors and pulled_on
    assert update["derived"].band == "720-780"
    assert update["derived"].age_at_application_years == 26  # IST birthday reached (UTC would still say 25)


def test_gate_input_is_built_from_lender_rows_with_policy_notes_and_list_pincodes() -> None:
    row = LenderRow(id="L9", version=2, name="Nine", published=True, criteria={"x": 1}, policy_notes="note 9",
                    matched_pincodes=("560001", "560002"))
    built = nodes.published_lender_row(row)
    assert built == {"id": "L9", "name": "Nine", "criteria": {"x": 1}, "matched_pincodes": ["560001", "560002"],
                     "policy_notes": "note 9"}
    assert isinstance(built["matched_pincodes"], list)

    world = make_world(happy_quick)
    final = run(world, check_state("quick", "chk-notes"))
    gated = {g.id: g for g in final["gated_lenders"]}
    assert gated["L1"].criteria.policy_notes == L1_NOTES  # passthrough from the db row
    assert gated["L2"].criteria.policy_notes is None  # '' -> None
    assert gated["L1"].matched_pincodes == ["560001"]


# =========================================================================== persist: formulas, locks, failures


def test_persist_formulas_are_delegated_to_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, bool, int]] = []

    def fake_billed(calls: list[SurePassCallRow]) -> bool:
        seen.append(("billed", False, len(calls)))
        return True

    def fake_counted(*, kind: str, counts_against_quota: bool, calls: list[SurePassCallRow]) -> bool:
        seen.append((kind, counts_against_quota, len(calls)))
        return True

    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-seed"))
    monkeypatch.setattr(nodes, "compute_billed", fake_billed)
    monkeypatch.setattr(nodes, "compute_counted_against_quota", fake_counted)
    run(world.with_surepass(nothing_configured), check_state("quick", "chk-delegated"))  # a cache hit: no calls
    row = world.row("chk-delegated")
    assert (row.billed, row.counted_against_quota) == (True, True)  # whatever policy says, persist writes
    assert ("billed", False, 0) in seen and ("quick", False, 0) in seen


def pan_status_invalid(sink: CallLogSink) -> FakeSurePassClient:
    return pan_invalid_scenario("status_invalid", pan=PAN_OK, sink=sink)


def pan_failure_envelope(sink: CallLogSink) -> FakeSurePassClient:
    return pan_invalid_scenario("failure_envelope", pan=PAN_OK, sink=sink)


def score_timeout(sink: CallLogSink) -> FakeSurePassClient:
    return score_failure_scenario("timeout", sink=sink)


def score_http_500(sink: CallLogSink) -> FakeSurePassClient:
    return score_failure_scenario("http_500", sink=sink)


def no_pincode(sink: CallLogSink) -> FakeSurePassClient:
    return no_pincode_scenario(sink=sink)


def live_off(sink: CallLogSink) -> FakeSurePassClient:
    return FakeSurePassClient(sink=sink, calls_enabled=False)


TERMINAL_WORLDS: list[tuple[str, SurePassFactory, FakeRankModel | None]] = [
    ("ok", happy_quick, None),
    ("pan_not_verified", pan_status_invalid, None),
    ("pan_service_failure", pan_failure_envelope, None),
    ("pending_verification", score_timeout, None),
    ("bureau_service_failure", score_http_500, None),
    ("band_read", no_pincode, None),
    ("openai_failure", happy_quick, FakeRankModel.invalid_twice()),
    ("refused_after_acquire", live_off, None),
]


@pytest.mark.parametrize(("name", "factory", "rank"), TERMINAL_WORLDS, ids=[t[0] for t in TERMINAL_WORLDS])
def test_in_flight_lock_is_released_on_every_terminal(
    name: str, factory: SurePassFactory, rank: FakeRankModel | None
) -> None:
    world = make_world(factory, rank=rank)
    run(world, check_state("quick", f"chk-{name}"))
    assert "try_acquire_pan_lock" in world.repo.log and world.repo.log[-1] == "release_pan_lock"
    assert not world.in_flight_held(pan_hash_of())
    assert world.row(f"chk-{name}").status != "running"


@pytest.mark.parametrize("failure", [RuntimeError("db down"), ValueError("illegal write")])
def test_persist_catches_any_exception_records_an_error_row_and_still_releases(failure: Exception) -> None:
    world = make_world(happy_quick)
    world.repo.fail_update_check = failure
    final = run(world, check_state("quick", "chk-dbdown"))
    view = view_of(final)
    assert (view.status, view.code, view.label) == ("service_failure", None, MSG_INTERNAL)
    assert world.row("chk-dbdown").status == "error"
    assert not world.in_flight_held(pan_hash_of())


def test_persist_refusal_catches_any_exception_and_still_releases() -> None:
    world = make_world(lambda sink: FakeSurePassClient(sink=sink, calls_enabled=False))
    world.repo.fail_update_check = RuntimeError("db down")
    final = run(world, check_state("quick", "chk-refdown"))
    assert view_of(final).label == MSG_INTERNAL
    assert world.row("chk-refdown").status == "error"
    assert not world.in_flight_held(pan_hash_of())


def test_a_mid_graph_raise_escapes_and_fail_check_releases_the_lock() -> None:
    world = make_world(happy_quick)
    world.repo.fail_lenders = RuntimeError("lenders table unavailable")
    with pytest.raises(RuntimeError):
        run(world, check_state("quick", "chk-crash"))
    pan_hash = pan_hash_of()
    assert world.in_flight_held(pan_hash)  # the graph did not reach persist ...
    view = nodes.fail_check(
        world.deps, check_id="chk-crash", kind="quick", lc_id=LC_ID, pan_hash=pan_hash, pan_last4="234F", message="crash"
    )
    assert (view.status, view.label) == ("service_failure", MSG_INTERNAL)
    assert not world.in_flight_held(pan_hash)  # ... so the error boundary helper releases it
    row = world.row("chk-crash")
    # route level: billed from the call log; counted is the documented residual (no admission context at the route)
    assert (row.status, row.billed, row.counted_against_quota) == ("error", True, False)


def test_redact_catches_builder_value_errors_and_records_a_guarded_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(state: CheckState, ctx: Any) -> LcView:
        raise ValueError("a full ok view needs a report date")

    monkeypatch.setattr(nodes, "build_lc_view", boom)
    world = make_world(happy_quick)
    final = run(world, check_state("quick", "chk-redact"))
    view = view_of(final)
    assert view.label == MSG_INTERNAL and final["lc_view"] is not None
    row = world.row("chk-redact")
    # K8: the billed 200 Score call still bills and still counts, on the error row too (node-level: admission known).
    assert (row.status, row.billed, row.counted_against_quota) == ("error", True, True)
    assert world.repo.quick_checks_used_today(LC_ID) == 1
    assert (final["billed"], final["counted_against_quota"]) == (True, True)  # the trace matches the row
    assert not world.in_flight_held(pan_hash_of())


def test_persist_writes_its_review_rows_before_the_terminal_update() -> None:
    world = make_world(happy_quick, rank=FakeRankModel.invalid_twice())
    run(world, check_state("quick", "chk-order-review"))
    log = world.repo.log
    assert log.index("insert_review_item") < log.index("update_check")


def test_a_failing_review_insert_leaves_the_view_and_the_row_in_agreement() -> None:
    world = make_world(happy_quick, rank=FakeRankModel.invalid_twice())  # persist owns an openai_failure row here
    world.repo.fail_review_insert = RuntimeError("review_queue unavailable")
    final = run(world, check_state("quick", "chk-review-down"))
    view = view_of(final)
    row = world.row("chk-review-down")
    assert (view.status, view.code, view.label) == ("service_failure", None, MSG_INTERNAL)
    assert (row.status, row.code) == ("error", None)  # never a service_failure/OPENAI_FAILURE row the LC never saw
    assert (row.billed, row.counted_against_quota) == (True, True) == (final["billed"], final["counted_against_quota"])
    assert world.reviews() == []
    assert not world.in_flight_held(pan_hash_of())


def test_bureau_timeout_writes_the_reconciliation_row_before_the_lock() -> None:
    world = make_world(score_timeout)
    run(world, check_state("quick", "chk-timeout-order"))
    log = world.repo.log
    assert log.index("insert_review_item") < log.index("create_pan_lock")


def test_a_failing_timeout_lock_write_still_leaves_the_reconciliation_gate() -> None:
    world = make_world(score_timeout)
    world.repo.fail_create_lock = RuntimeError("pan_locks unavailable")
    with pytest.raises(RuntimeError):
        run(world, check_state("quick", "chk-lockdown"))
    pan_hash = pan_hash_of()
    assert [(r.kind, r.resolved_at) for r in world.reviews()] == [("timeout_reconciliation", None)]
    view = nodes.fail_check(  # the route boundary
        world.deps, check_id="chk-lockdown", kind="quick", lc_id=LC_ID, pan_hash=pan_hash, pan_last4="234F",
        message="crash",
    )
    row = world.row("chk-lockdown")
    assert (view.status, view.code, view.label) == ("service_failure", None, MSG_INTERNAL)
    assert (row.status, row.billed) == ("error", True)  # the timeout row is conservatively billable
    assert world.repo.get_active_locks(pan_hash) == []  # no timeout lock, no leaked in-flight lock ...
    world.clock.advance(timedelta(minutes=1))
    retry = run(world.with_surepass(nothing_configured), check_state("quick", "chk-after-lockdown"))
    assert view_of(retry).code == "PAN_LOCKED"  # ... yet the unresolved review alone still gates the re-pull (K2)


def test_a_failing_reconciliation_insert_still_leaves_the_timeout_lock() -> None:
    world = make_world(score_timeout)
    world.repo.fail_review_insert = RuntimeError("review_queue unavailable")
    with pytest.raises(RuntimeError):
        run(world, check_state("quick", "chk-reviewdown"))
    pan_hash = pan_hash_of()
    locks = {lock.kind: lock for lock in world.repo.get_active_locks(pan_hash)}
    assert locks["timeout"].lock_until == T0 + TIMEOUT_LOCK_TTL  # written in the finally, despite the review failure
    nodes.fail_check(  # the route boundary releases the in-flight lock
        world.deps, check_id="chk-reviewdown", kind="quick", lc_id=LC_ID, pan_hash=pan_hash, pan_last4="234F",
        message="crash",
    )
    world.repo.fail_review_insert = None
    world.clock.advance(timedelta(minutes=1))
    retry_world = world.with_surepass(nothing_configured)
    retry = run(retry_world, check_state("quick", "chk-after-reviewdown"))
    assert view_of(retry).code == "PAN_LOCKED"  # no second pull on a spend-unknown PAN
    assert retry_world.surepass.calls == [] and retry_world.calls("chk-after-reviewdown") == []


def test_full_check_report_reuse_cache_hit() -> None:
    world = make_world(happy_full)
    run(world, check_state("full", "chk-report-pull"))
    world.clock.advance(timedelta(days=4))
    hit = world.with_surepass(nothing_configured)
    world.repo.log.clear()
    final = run(hit, check_state("full", "chk-report-hit", opportunity_id="opp-graph-2"))
    view = view_of(final)
    assert final["transcript"] == FULL_PATH and hit.calls("chk-report-hit") == []
    assert world.repo.log.count("fetch_raw") == 2  # PAN object + report object
    assert view.status == "ok" and view.from_cache is True and view.checked_on == "2026-10-10"
    assert view.label == "bureau report on 2026-10-01, income as declared"
    row = world.row("chk-report-hit")
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (False, False, False)


def test_a_cache_hit_regates_against_the_currently_published_lenders() -> None:
    world = make_world(happy_quick)
    first_top = view_of(run(world, check_state("quick", "chk-gate-1"))).top_lender
    assert first_top is not None and first_top.id == "L1"
    key = ("L1", 1)
    world.repo._lenders[key] = dataclasses.replace(world.repo._lenders[key], published=False)  # admin unpublishes L1
    final = run(world.with_surepass(nothing_configured), check_state("quick", "chk-gate-2"))
    assert view_of(final).from_cache is True
    assert [g.id for g in final["gated_lenders"]] == ["L2", "L3"]
    top = view_of(final).top_lender
    assert top is not None and top.id != "L1"


def test_a_quick_check_is_served_from_the_report_sourced_score_row() -> None:
    world = make_world(happy_full)
    run(world, check_state("full", "chk-report-first"))
    score_row = world.repo.get_score_cache(pan_hash_of())
    assert score_row is not None and score_row.source == "report"
    final = run(world.with_surepass(nothing_configured), check_state("quick", "chk-quick-from-report"))
    view = view_of(final)
    assert final["score"].source == "report" and final["score"].band == "720-780"  # read from the report envelope
    assert view.status == "ok" and view.from_cache is True and view.checked_on == "2026-10-10"
    row = world.row("chk-quick-from-report")
    assert (row.billed, row.counted_against_quota) == (False, False)


def test_internal_carries_no_raw_pan_mobile_or_names() -> None:
    world = make_world(happy_full)
    run(world, check_state("full", "chk-pii"))
    internal = world.row("chk-pii").internal
    assert internal is not None
    text = repr(internal)
    for literal in (PAN_OK, "9876543210", LC_TYPED_NAME, "RAVI TESTKUMAR", "1998-04-12", "SYNTHACC0001"):
        assert literal not in text, literal
    assert "storage_key" not in text and f"pan/{pan_hash_of()}/" not in text  # no storage refs (keys are pan/<hash>/..)


# =========================================================================== recompute (K9)


def test_recompute_creates_a_new_row_and_enters_at_derive_with_zero_surepass_calls() -> None:
    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-orig"))
    original = world.row("chk-orig")
    world.clock.advance(timedelta(days=2))
    calls_before = list(world.surepass.calls)

    outcome, state = seed_recompute(
        world.deps, original_check_id="chk-orig", new_check_id="chk-recomp", monthly_income=30000.0,
        existing_emi_declared=5000.0,
    )
    assert outcome == "ok" and state is not None
    admission = state.get("admission")
    assert admission == AdmissionDecision(
        admission="cache_hit", pan_plan="reuse", bureau_plan="reuse", counts_against_quota=False, is_refresh=False,
        checked_on=date(2026, 10, 10), refresh_available_on=date(2026, 11, 9), original_consent_id=original.consent_id,
    )
    final = run_recompute(world, state)
    view = view_of(final)

    assert final["transcript"] == list(RECOMPUTE_NODE_NAMES)
    assert world.surepass.calls == calls_before and world.calls("chk-recomp") == []
    row = world.row("chk-recomp")
    assert (row.recompute_of, row.consent_id, row.billed, row.counted_against_quota) == (
        "chk-orig", original.consent_id, False, False,
    )
    assert view.status == "ok" and view.from_cache is True and view.checked_on == "2026-10-10"
    assert view.quick_checks_left_today is None  # null on recompute views
    assert world.row("chk-orig") == original  # the original row is untouched


def test_recompute_of_a_full_check_reads_the_stored_report() -> None:
    world = make_world(happy_full)
    run(world, check_state("full", "chk-full-orig"))
    outcome, state = seed_recompute(world.deps, original_check_id="chk-full-orig", new_check_id="chk-full-re",
                                    monthly_income=90000.0, existing_emi_declared=None)
    assert outcome == "ok" and state is not None and state.get("report") is not None and "score" not in state
    view = view_of(run_recompute(world, state))
    assert view.status == "ok" and view.label == "bureau report on 2026-10-01, income as declared"
    assert world.row("chk-full-re").recompute_of == "chk-full-orig"


def test_recompute_is_allowed_after_an_openai_failure() -> None:
    world = make_world(happy_quick, rank=FakeRankModel.invalid_twice())
    run(world, check_state("quick", "chk-oai-orig"))
    healthy = world.with_rank(FakeRankModel())
    outcome, state = seed_recompute(healthy.deps, original_check_id="chk-oai-orig", new_check_id="chk-oai-re",
                                    monthly_income=60000.0, existing_emi_declared=None)
    assert outcome == "ok" and state is not None
    view = view_of(run_recompute(healthy, state))
    assert (view.status, view.code) == ("ok", None) and healthy.rank.call_count == 1


def test_persist_strips_the_raw_identifiers_from_the_final_state() -> None:
    worlds: list[tuple[SurePassFactory, bool]] = [(happy_quick, True), (nothing_configured, False)]  # persist, refusal
    for factory, consent_given in worlds:
        world = make_world(factory)
        final = run(world, check_state("quick", f"chk-strip-{consent_given}", with_consent=consent_given))
        assert (final["pan"], final["name"], final["mobile"]) == ("", "", "")  # request scope only (state.py)
        assert final["pan_last4"] == "234F"


def test_recompute_outcomes_not_found_not_recomputable_raw_unavailable() -> None:
    world = make_world(happy_quick)
    assert seed_recompute(world.deps, original_check_id="nope", new_check_id="n1", monthly_income=None,
                          existing_emi_declared=None) == ("not_found", None)

    run(world.with_surepass(nothing_configured), check_state("quick", "chk-refused", with_consent=False))
    assert seed_recompute(world.deps, original_check_id="chk-refused", new_check_id="n2", monthly_income=None,
                          existing_emi_declared=None)[0] == "not_recomputable"

    run(world, check_state("quick", "chk-ok"))
    world.repo.raw_store.delete_raw(world.repo.raw_store.keys())
    assert seed_recompute(world.deps, original_check_id="chk-ok", new_check_id="n3", monthly_income=None,
                          existing_emi_declared=None) == ("raw_unavailable", None)
    assert world.repo.get_check("n3") is None  # no row for a refused recompute


def test_recompute_never_answers_ok_without_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    world = make_world(happy_quick)
    run(world, check_state("quick", "chk-norows"))
    world.repo.delete_expired_cache_rows(T0 + timedelta(days=400))  # drop every cache row
    def wrongly_admitted(*args: object) -> str:
        return "ok"

    monkeypatch.setattr(build, "admit_recompute", wrongly_admitted)  # a policy that wrongly admitted
    assert seed_recompute(world.deps, original_check_id="chk-norows", new_check_id="n4", monthly_income=None,
                          existing_emi_declared=None) == ("raw_unavailable", None)  # never a contradictory ("ok", None)


# =========================================================================== serde + checkpointer (K21)


def _populated_state() -> CheckState:
    world = make_world(happy_full)
    final = run(world, check_state("full", "chk-serde"))
    assert isinstance(final["lc_view"], LcView) and isinstance(final["admission"], AdmissionDecision)
    assert isinstance(final["consent"], ConsentInfo)
    return cast(CheckState, final)


def test_strict_msgpack_round_trips_a_populated_check_state() -> None:
    populated = _populated_state()
    serde = make_checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(populated))
    assert restored == populated
    assert isinstance(restored["lc_view"], LcView) and isinstance(restored["admission"], AdmissionDecision)


def test_without_the_state_allowlist_strict_mode_does_not_restore_the_models() -> None:
    populated = _populated_state()
    strict_safe_only = JsonPlusSerializer(allowed_msgpack_modules=None)  # strict: SAFE_MSGPACK_TYPES only
    restored = strict_safe_only.loads_typed(strict_safe_only.dumps_typed(populated))
    assert not isinstance(restored["lc_view"], LcView)  # the allowlist is load-bearing, not decorative


def test_allowlist_is_exactly_the_state_modules_model_classes() -> None:
    allow = state_msgpack_allowlist()
    assert all(module == "eligibility_agent.graph.state" for module, _ in allow)
    for name in ("LcView", "LenderView", "LoanRange", "AdmissionDecision", "FullAdmissionInputs", "ConsentInfo",
                 "Identity", "ScoreFacts", "ReportFacts", "OpenAccount", "Derived", "EmiDerivation", "EmiItem",
                 "GatedLender", "LenderCriteria", "LenderNumbers", "ModelOutput", "Pick", "SafetyResult",
                 "SafetyBreach", "SurePassCallAudit"):
        assert ("eligibility_agent.graph.state", name) in allow, name
        assert hasattr(state_module, name)


def test_graph_runs_with_a_checkpointer_keyed_by_check_id() -> None:
    world = make_world(happy_quick)
    saver = InMemorySaver(serde=make_checkpoint_serde())
    graph = build_graph("quick", world.deps, checkpointer=saver)
    config = tracing.invoke_config(kind="quick", check_id="chk-ckpt", lc_id=LC_ID, pan_hash=pan_hash_of(), run_id=None)
    invoke(graph, check_state("quick", "chk-ckpt"), config)
    snapshot = graph.get_state(config)
    values = cast(dict[str, Any], snapshot.values)
    assert isinstance(values["lc_view"], LcView) and isinstance(values["admission"], AdmissionDecision)
    assert snapshot.config.get("configurable", {}).get("thread_id") == "chk-ckpt"


class _StubCursor:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _StubConnection:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row
        self.sql: list[str] = []

    def execute(self, query: str) -> _StubCursor:
        self.sql.append(query)
        return _StubCursor(self.row)


def test_pre_setup_schema_assertion_requires_the_checkpointer_schema() -> None:
    good = _StubConnection({"current_schema": "checkpointer"})
    assert_checkpointer_schema(good)
    assert "current_schema()" in good.sql[0]
    for row in ({"current_schema": "public"}, None):
        with pytest.raises(CheckpointerConfigError):
            assert_checkpointer_schema(_StubConnection(row))


class _StubPool:
    def __init__(self, row: dict[str, Any], events: list[str], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.events = events
        self.row = row

    def connection(self) -> Any:
        pool = self

        class _Ctx:
            def __enter__(self) -> _StubConnection:
                pool.events.append("connection")
                return _StubConnection(pool.row)

            def __exit__(self, *exc: object) -> None:
                return None

        return _Ctx()

    def close(self) -> None:
        self.events.append("close")


def _stub_factory(monkeypatch: pytest.MonkeyPatch, row: dict[str, Any]) -> list[str]:
    import langgraph.checkpoint.postgres as lg_postgres
    import psycopg_pool

    events: list[str] = []

    class _StubSaver:
        def __init__(self, conn: Any, serde: Any = None) -> None:
            events.append("saver")

        def setup(self) -> None:
            events.append("setup")

    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    def stub_pool(**kwargs: Any) -> _StubPool:
        return _StubPool(row, events, **kwargs)

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", stub_pool)
    monkeypatch.setattr(lg_postgres, "PostgresSaver", _StubSaver)
    return events


def test_checkpointer_factory_checks_the_schema_before_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _stub_factory(monkeypatch, {"current_schema": "checkpointer", "public_clear": True})
    make_postgres_checkpointer("postgresql://svc:pw@localhost:5432/postgres")
    assert events == ["connection", "saver", "setup", "connection"]  # schema check -> setup -> isolation check


def test_checkpointer_factory_never_runs_setup_on_the_wrong_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _stub_factory(monkeypatch, {"current_schema": "public", "public_clear": True})
    with pytest.raises(CheckpointerConfigError):
        make_postgres_checkpointer("postgresql://svc:pw@localhost:5432/postgres")
    assert "setup" not in events and events[-1] == "close"  # public.checkpoints is never created, even transiently


def test_checkpointer_isolation_assertion_passes_only_on_the_checkpointer_schema() -> None:
    good = _StubConnection({"public_clear": True, "current_schema": "checkpointer"})
    assert_checkpointer_isolated(good)
    assert "to_regclass('public.checkpoints') IS NULL" in good.sql[0]
    for row in (
        {"public_clear": False, "current_schema": "checkpointer"},  # tables leaked into PostgREST-exposed public
        {"public_clear": True, "current_schema": "public"},  # role search_path not applied
        None,
    ):
        with pytest.raises(CheckpointerConfigError):
            assert_checkpointer_isolated(_StubConnection(row))


@pytest.mark.parametrize(
    ("dsn", "ok"),
    [
        ("postgresql://svc:pw@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require", True),
        ("postgresql://svc:pw@aws-0-ap-south-1.pooler.supabase.com:6543/postgres", False),
        ("host=db.example port=6543 dbname=postgres user=svc", False),
        ("host=db.example port=5432 dbname=postgres user=svc", True),
    ],
)
def test_checkpointer_dsn_refuses_the_transaction_pooler(dsn: str, ok: bool) -> None:
    if ok:
        check_checkpointer_dsn(dsn)
    else:
        with pytest.raises(CheckpointerConfigError):
            check_checkpointer_dsn(dsn)


def test_checkpointer_factory_requires_strict_msgpack_before_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGGRAPH_STRICT_MSGPACK", raising=False)
    with pytest.raises(CheckpointerConfigError, match="LANGGRAPH_STRICT_MSGPACK"):
        make_postgres_checkpointer("postgresql://svc:pw@localhost:5432/postgres")
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    with pytest.raises(CheckpointerConfigError, match="6543"):
        make_postgres_checkpointer("postgresql://svc:pw@localhost:6543/postgres")


# =========================================================================== tracing (K21)


class _SpyClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[UUID, list[str] | None, dict[str, Any] | None]] = []

    def update_run(self, run_id: UUID, *, tags: list[str] | None = None, extra: dict[str, Any] | None = None) -> None:
        if self.fail:
            raise RuntimeError("langsmith down")
        self.calls.append((run_id, tags, extra))


def test_invoke_config_threads_check_id_and_carries_no_pii() -> None:
    run_id = UUID(int=7)
    config = tracing.invoke_config(kind="full", check_id="chk-t", lc_id=LC_ID, pan_hash="h" * 64, run_id=run_id)
    assert config.get("configurable") == {"thread_id": "chk-t"}
    assert config.get("tags") == ["tier:full", f"lc:{LC_ID}"]
    assert config.get("metadata") == {"check_id": "chk-t", "pan_hash": "h" * 64, "tier": "full"}
    assert config.get("recursion_limit") == tracing.RECURSION_LIMIT == build.RECURSION_LIMIT
    assert config.get("run_id") == run_id
    assert "run_id" not in tracing.invoke_config(kind="quick", check_id="c", lc_id="l", pan_hash="h", run_id=None)
    assert tracing.new_run_id(False) is None and isinstance(tracing.new_run_id(True), UUID)


def test_finish_trace_patches_admission_billed_counted_and_nothing_personal() -> None:
    world = make_world(happy_quick)
    final = run(world, check_state("quick", "chk-trace"))
    client = _SpyClient()
    run_id = UUID(int=9)
    tracing.finish_trace(client, run_id=run_id, kind="quick", lc_id=LC_ID, check_id="chk-trace",
                         pan_hash=pan_hash_of(), final_state=final)
    [(got_id, tags, extra)] = client.calls
    assert got_id == run_id and tags == ["tier:quick", f"lc:{LC_ID}", "admission:new"]
    assert extra is not None
    metadata = extra["metadata"]
    assert set(metadata) == tracing.FINISH_METADATA_KEYS
    assert metadata == {
        "check_id": "chk-trace", "pan_hash": pan_hash_of(), "tier": "quick", "admission": "new", "admission_pan": "call",
        "admission_bureau": "call", "billed": True, "counted_against_quota": True,
    }
    assert all(literal not in repr(extra) for literal in (PAN_OK, "9876543210", LC_TYPED_NAME, "RAVI TESTKUMAR"))


def test_finish_trace_is_a_no_op_without_a_client_and_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    tracing.finish_trace(None, run_id=UUID(int=1), kind="quick", lc_id="l", check_id="c", pan_hash="h", final_state={})
    tracing.finish_trace(_SpyClient(), run_id=None, kind="quick", lc_id="l", check_id="c", pan_hash="h", final_state={})
    with caplog.at_level(logging.WARNING):
        tracing.finish_trace(_SpyClient(fail=True), run_id=UUID(int=1), kind="quick", lc_id="l", check_id="c",
                             pan_hash="h", final_state={})
    assert "RuntimeError" in caplog.text and "langsmith down" not in caplog.text
    refused = tracing.trace_metadata(kind="quick", check_id="c", pan_hash="h",
                                     final_state={"admission": AdmissionDecision(code="IN_FLIGHT")})
    assert (refused["admission"], refused["billed"]) == ("refused", False)


class RecordingClient(Client):
    """A real langsmith Client (so its hide_inputs / hide_outputs logic runs) whose two send hooks record the
    post-hiding payload instead of doing I/O: exactly what would leave the process."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(auto_batch_tracing=False, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self.exported: list[dict[str, Any]] = []

    def _create_run(self, run_create: dict[str, Any], **kwargs: Any) -> None:
        self.exported.append(copy.deepcopy(run_create))

    def _update_run(self, run_update: dict[str, Any], **kwargs: Any) -> None:
        self.exported.append(copy.deepcopy(run_update))


_STANDALONE_SCORE = re.compile(r"(?<![0-9A-Za-z])745(?![0-9A-Za-z])")


def test_traced_runs_export_metadata_but_never_payloads() -> None:
    world = make_world(happy_quick)
    client = tracing.make_langsmith_client(
        api_key="test-key", api_url="http://langsmith.invalid", client_cls=RecordingClient
    )
    assert isinstance(client, RecordingClient)
    tracer = tracing.make_tracer(client, project_name="eligibility-agent-test")
    run_id = tracing.new_run_id(True)
    config = tracing.invoke_config(kind="quick", check_id="chk-traced", lc_id=LC_ID, pan_hash=pan_hash_of(),
                                   run_id=run_id, tracer=tracer)
    final = invoke(build_graph("quick", world.deps), check_state("quick", "chk-traced"), config)
    tracer.wait_for_futures()
    tracing.finish_trace(client, run_id=run_id, kind="quick", lc_id=LC_ID, check_id="chk-traced",
                         pan_hash=pan_hash_of(), final_state=final)
    assert view_of(final).status == "ok"

    exported = client.exported
    assert len(exported) >= 2 * len(FULL_PATH)  # the root, every node and every router: posted and patched
    forbidden = (PAN_OK, "9876543210", LC_TYPED_NAME, "RAVI TESTKUMAR", "1998-04-12", "720-780", "650-720",
                 "780+", "<650")
    for payload in exported:
        assert payload.get("inputs") in (None, {}) and payload.get("outputs") in (None, {})
        visible = repr({k: payload.get(k) for k in ("inputs", "outputs", "extra", "tags", "name", "events")})
        for literal in forbidden:
            assert literal not in visible, (literal, payload.get("name"))
        assert _STANDALONE_SCORE.search(visible) is None, payload.get("name")
    root = [p for p in exported if p.get("id") == run_id and p.get("extra")]
    assert root and all(p["extra"]["metadata"]["pan_hash"] == pan_hash_of() for p in root)


_STANDALONE_720 = re.compile(r"(?<![0-9A-Za-z])720(?![0-9A-Za-z])")


def test_a_poisoned_error_name_and_tag_export_scrubbed_while_pan_hash_survives() -> None:
    world = make_world(happy_quick)
    world.repo.fail_lenders = RuntimeError(f"lender lookup for {PAN_OK} failed at Experian with score 720")
    client = tracing.make_langsmith_client(
        api_key="test-key", api_url="http://langsmith.invalid", client_cls=RecordingClient
    )
    assert isinstance(client, RecordingClient)
    tracer = tracing.make_tracer(client, project_name="eligibility-agent-test")
    run_id = tracing.new_run_id(True)
    config = tracing.invoke_config(kind="quick", check_id="chk-poison", lc_id="lc-720", pan_hash=pan_hash_of(),
                                   run_id=run_id, tracer=tracer)
    config["run_name"] = f"check for {PAN_OK}"  # a poisoned run name
    with pytest.raises(RuntimeError):
        invoke(build_graph("quick", world.deps), check_state("quick", "chk-poison"), config)
    tracer.wait_for_futures()

    exported = client.exported
    errors = [str(p["error"]) for p in exported if p.get("error")]
    assert errors  # the failing node and the root both carry the error
    for text in errors:
        assert PAN_OK not in text and "experian" not in text.lower() and _STANDALONE_720.search(text) is None
    assert any("lender lookup for" in t and tracing.TRACE_MASK in t for t in errors)  # spans replaced, rest kept
    for payload in exported:
        visible = repr({k: payload.get(k) for k in ("name", "tags", "extra")})
        assert PAN_OK not in visible and _STANDALONE_720.search(visible) is None, payload.get("name")
    root = [p for p in exported if p.get("id") == run_id and p.get("extra")]
    assert root and all(p["extra"]["metadata"]["pan_hash"] == pan_hash_of() for p in root)  # intact, not scrubbed
    assert all(p["extra"]["metadata"]["check_id"] == "chk-poison" for p in root)

    # finish_trace patches the root through client.update_run, bypassing the tracer: its tags are scrubbed too.
    before = len(client.exported)
    tracing.finish_trace(client, run_id=run_id, kind="quick", lc_id=f"lc-{PAN_OK}-720", check_id="chk-poison",
                         pan_hash=pan_hash_of(), final_state={})
    [patch] = client.exported[before:]
    tags = cast("list[str]", patch["tags"])
    assert tags == ["tier:quick", f"lc:lc-{tracing.TRACE_MASK}-{tracing.TRACE_MASK}", "admission:none"]
    assert patch["extra"]["metadata"]["pan_hash"] == pan_hash_of()


def test_trace_scrub_replaces_spans_and_fails_closed() -> None:
    scrub = tracing.scrub_trace_text
    mask = tracing.TRACE_MASK
    assert scrub(f"pan {PAN_OK.lower()} at CIBIL scored 745") == f"pan {mask} at {mask} scored {mask}"
    assert scrub("band 650-720 on file") == f"band {mask}-{mask} on file"  # band labels are score numbers
    assert scrub("mobile 9876543210") == f"mobile {mask}"
    assert scrub("bureau report on 2026-10-01") == "bureau report on 2026-10-01"  # global tier: names only, dates pass
    assert scrub("score \uff17\uff14\uff15 (fullwidth)") == mask  # only the normalised form leaks: whole string
    assert scrub("x" * (tracing.MAX_TRACE_STRING_CHARS + 1)) == mask  # oversized: fail closed

    hex_hash = pan_hash_of()
    uuid_text = "1a2b745c-0000-4000-8000-000000000745"
    payload = {
        "pan_hash": hex_hash,
        "check_id": uuid_text,
        "thread_id": uuid_text,
        "tier": "quick",
        "nested": {"pan_hash": f"not a hash {PAN_OK}", "items": ["fine", "cibil", 720, None]},
        "error": f"failed for {PAN_OK}",
    }
    out = tracing.scrub_trace_payload(payload)
    assert (out["pan_hash"], out["check_id"], out["thread_id"]) == (hex_hash, uuid_text, uuid_text)  # id shapes kept
    assert out["nested"] == {"pan_hash": f"not a hash {mask}", "items": ["fine", mask, 720, None]}
    assert out["error"] == f"failed for {mask}" and out["tier"] == "quick"


def test_the_trace_client_hides_payloads_and_anonymizes_the_rest() -> None:
    client = tracing.make_langsmith_client(api_key="k", api_url="http://langsmith.invalid", client_cls=RecordingClient)
    assert getattr(client, "_hide_inputs") is True and getattr(client, "_hide_outputs") is True
    assert getattr(client, "_anonymizer") is tracing.scrub_trace_payload
    assert isinstance(tracing.make_tracer(client, project_name="p"), tracing.PayloadFreeTracer)


def test_new_check_state_seeds_counters_and_no_derived_keys() -> None:
    state = check_state("quick", "chk-seed-shape", langsmith_run_id="run-1")
    assert state.get("rank_attempts") == 0 and state.get("transcript") == [] and state.get("errors") == []
    assert state.get("surepass_calls_made") == [] and state.get("is_recompute") is False
    assert state.get("langsmith_run_id") == "run-1"
    assert "admission" not in state and "lc_view" not in state


def test_build_graph_rejects_an_unknown_kind() -> None:
    world = make_world(nothing_configured)
    with pytest.raises(ValueError):
        build_graph(cast(Any, "medium"), world.deps)


def test_graph_deps_default_clock_is_aware_utc() -> None:
    deps = build.GraphDeps(repo=make_world(nothing_configured).repo, surepass=FakeSurePassClient(),
                           rank_model=FakeRankModel(), prompts=local_prompts())
    now = deps.now()
    assert isinstance(now, datetime) and now.utcoffset() == timedelta(0)


def test_envelope_builders_stay_synthetic() -> None:
    # The fixtures this module uses come from testing/envelopes (probe-literal guarded there); spot-check one here.
    assert env.pan_comprehensive_envelope()[1]["data"]["address"]["zip"] == "560001"
