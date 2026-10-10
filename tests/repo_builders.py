"""Shared row builders and a frozen clock for the db-layer tests (no conftest: Task 3 owns that)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from eligibility_agent.db.repo import (
    CheckRow,
    ConsentRow,
    MemoryRepo,
    OpportunityPanEvent,
    PanRecordRow,
    ReportCacheRow,
    ReviewRow,
    ScoreCacheRow,
)

UTC = timezone.utc
# 2026-10-10 08:00Z == 13:30 IST on 2026-10-10 (mid-day IST, well away from any IST midnight).
T0 = datetime(2026, 10, 10, 8, 0, 0, tzinfo=UTC)

PAN_A = "a" * 64
PAN_B = "b" * 64

# Local literals: surepass/client.py (Task 4) owns the canonical endpoint constants.
PAN_EP = "/api/v1/pan/pan-comprehensive"
SCORE_EP = "/api/v1/credit-report-experian/score"
REPORT_EP = "/api/v1/credit-report-experian/fetch-report"

ENVELOPE: dict[str, Any] = {"status_code": 200, "success": True, "data": {"credit_score": "745"}}


class FrozenClock:
    """Callable clock the tests move by hand."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> datetime:
        self.now += timedelta(**kwargs)
        return self.now


def make_check(**overrides: Any) -> CheckRow:
    fields: dict[str, Any] = {
        "id": "chk-1",
        "kind": "quick",
        "lc_id": "lc-1",
        "lead_id": "lead-1",
        "opportunity_id": None,
        "opportunity_stage": None,
        "contact_id": None,
        "pan_hash": PAN_A,
        "pan_last4": "234F",
        "mobile_hash": "m" * 64,
        "mobile_last4": "3210",
        "status": "running",
        "code": None,
        "billed": False,
        "counted_against_quota": False,
        "is_refresh": False,
        "recompute_of": None,
        "consent_id": None,
        "lc_view": None,
        "internal": None,
        "langsmith_run_id": None,
        "created_at": T0,
    }
    fields.update(overrides)
    return CheckRow(**fields)


def make_consent(**overrides: Any) -> ConsentRow:
    fields: dict[str, Any] = {
        "id": "con-1",
        "lc_id": "lc-1",
        "lead_id": "lead-1",
        "opportunity_id": None,
        "contact_id": None,
        "pan_hash": PAN_A,
        "mobile_hash": "m" * 64,
        "lc_typed_name": "Asha Verma",
        "wording_version": "v2-quick",
        "call_ref": None,
        "ip": None,
        "covers": "score",
        "given_at": T0,
        "purge_after": T0 + timedelta(days=5 * 365),
    }
    fields.update(overrides)
    return ConsentRow(**fields)


def make_pan_record(**overrides: Any) -> PanRecordRow:
    fields: dict[str, Any] = {
        "pan_hash": PAN_A,
        "pan_last4": "234F",
        "storage_key": "pan/aaa/pan-comprehensive/k1.json",
        "client_id": "cid-1",
        "fetched_at": T0,
        "expires_at": T0 + timedelta(days=180),
        "purge_after": T0 + timedelta(days=180),
    }
    fields.update(overrides)
    return PanRecordRow(**fields)


def make_score_cache(**overrides: Any) -> ScoreCacheRow:
    fields: dict[str, Any] = {
        "pan_hash": PAN_A,
        "storage_key": "pan/aaa/experian-score/k2.json",
        "source": "score",
        "is_ntc": False,
        "client_id": "cid-1",
        "consent_id": None,
        "fetched_at": T0,
        "expires_at": T0 + timedelta(days=30),
        "purge_after": T0 + timedelta(days=180),
    }
    fields.update(overrides)
    return ScoreCacheRow(**fields)


def make_report_cache(**overrides: Any) -> ReportCacheRow:
    fields: dict[str, Any] = {
        "pan_hash": PAN_A,
        "storage_key": "pan/aaa/experian-report/k3.json",
        "is_ntc": False,
        "report_date": None,
        "client_id": "cid-1",
        "consent_id": None,
        "fetched_at": T0,
        "expires_at": T0 + timedelta(days=30),
        "purge_after": T0 + timedelta(days=180),
    }
    fields.update(overrides)
    return ReportCacheRow(**fields)


def make_review(**overrides: Any) -> ReviewRow:
    fields: dict[str, Any] = {
        "id": "rev-1",
        "check_id": None,
        "pan_hash": PAN_A,
        "kind": "timeout_reconciliation",
        "reason": "score call timed out",
        "created_at": T0,
        "resolved_at": None,
        "resolved_by": None,
    }
    fields.update(overrides)
    return ReviewRow(**fields)


def make_pan_event(**overrides: Any) -> OpportunityPanEvent:
    fields: dict[str, Any] = {
        "id": "ev-1",
        "opportunity_id": "opp-1",
        "pan_hash": PAN_A,
        "pan_last4": "234F",
        "event": "initial",
        "actor_lc_id": "lc-1",
        "check_id": None,
        "created_at": T0,
    }
    fields.update(overrides)
    return OpportunityPanEvent(**fields)


def log_call(
    repo: MemoryRepo,
    check_id: str,
    *,
    endpoint: str = SCORE_EP,
    pan_hash: str = PAN_A,
    http_status: int | None = 200,
    success: bool | None = True,
    error_kind: Any = None,
    envelope: dict[str, Any] | None = ENVELOPE,
) -> str | None:
    """Drive the v2 sink the way the SurePass client will."""
    return repo.log_surepass_call(
        check_id=check_id,
        pan_hash=pan_hash,
        endpoint=endpoint,
        client_id="cid-1",
        http_status=http_status,
        success=success,
        message_code=None,
        latency_ms=120,
        error_kind=error_kind,
        raw_envelope=envelope,
    )
