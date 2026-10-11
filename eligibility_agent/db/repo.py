# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Persistence layer (PRD v1.1 sections 5, 6, 11): row types, the Repo protocol, raw storage, two backends.

Layering: this module imports the shared Literals from ``graph/state.py`` (K14) and defines only the 7-value
row ``CheckStatus`` itself. ``policy/`` imports row types and a few SQL-mirror constants from here, never the
reverse. It is the ONLY module that touches supabase / storage3 / postgrest (hence the file-level pyright
pragma above: those packages ship untyped surfaces, and no ``Any`` leaves this file; every public method
returns typed rows). ``supabase.Client`` is imported under TYPE_CHECKING only; ``create_supabase_backend``
imports ``create_client`` lazily, so importing this module never pulls the supabase stack in.

Two backends satisfy the same protocol: ``MemoryRepo`` + ``MemoryRawStore`` (the only backends tests use) and
``SupabaseRepo`` + ``SupabaseRawStore``. The protocol carries NO business rules: admission, quota math and
window arithmetic live in ``policy/``; the IST-day and billable predicates here are strict mirrors of the SQL
functions / the audit sink the design assigns to the repo.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import uuid
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, TypedDict, TypeVar, get_args
from zoneinfo import ZoneInfo

from eligibility_agent.graph.state import (
    PINCODE_RE_STR,
    Code,
    ConsentCovers,
    ErrorKind,
    Kind,
    LockKind,
    PanEventKind,
    ReviewKind,
    ScoreSource,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from supabase import Client

# --------------------------------------------------------------------------- literals, clock, ids

# The DB row status. 'running' and 'error' are internal; LcView never shows them (its 5-value LcStatus is in state).
CheckStatus: TypeAlias = Literal[
    "running",
    "ok",
    "pan_not_verified",
    "refused",
    "pending_verification",
    "service_failure",
    "error",
]

Clock: TypeAlias = Callable[[], datetime]

DEFAULT_RAW_BUCKET: Final[str] = "raw-responses"
# SQL / sink mirrors that policy/guards re-exports rather than redefines (repo cannot import policy).
QUICK_DAILY_QUOTA_DEFAULT: Final[int] = 5  # quota_limit() when lc_quota_overrides has no row
RAW_RETENTION: Final[timedelta] = timedelta(days=180)  # surepass_calls.purge_after = called_at + this

_IST: Final = ZoneInfo("Asia/Kolkata")  # no DST: a day is exactly 24h
_PAN_ENDPOINT: Final = "/api/v1/pan/pan-comprehensive"
_SCORE_ENDPOINT: Final = "/api/v1/credit-report-experian/score"
_REPORT_ENDPOINT: Final = "/api/v1/credit-report-experian/fetch-report"
_ENDPOINT_SLUGS: Final[dict[str, str]] = {
    _PAN_ENDPOINT: "pan-comprehensive",
    _SCORE_ENDPOINT: "experian-score",
    _REPORT_ENDPOINT: "experian-report",
}
_BUREAU_ENDPOINTS: Final[frozenset[str]] = frozenset({_SCORE_ENDPOINT, _REPORT_ENDPOINT})
_CACHE_TABLES: Final[tuple[str, ...]] = ("pan_records", "score_cache", "report_cache")
_PINCODE_RE: Final = re.compile(PINCODE_RE_STR)
_PAGE_SIZE: Final[int] = 1000  # PostgREST's default max-rows


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    """ALL ids are app-generated uuid4 strings."""
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- status x code legality

# The 10 admission refusals = the 13 codes minus the mid-run codes (single source: graph/state.Code).
_REFUSAL_CODES: Final[frozenset[str]] = frozenset(get_args(Code)) - {
    "NO_PINCODE_ON_PAN",
    "SUREPASS_FAILURE",
    "OPENAI_FAILURE",
}


def code_legal(status: CheckStatus, code: Code | None) -> bool:
    """Mirror of the SQL ``code_legality`` CHECK on checks.

    refused -> code in the 10 refusal codes (required); service_failure -> SUREPASS_FAILURE | OPENAI_FAILURE
    (required); ok -> None or NO_PINCODE_ON_PAN; running | pan_not_verified | pending_verification | error
    -> None (K1: pending_verification carries no code).
    """
    if status == "refused":
        return code in _REFUSAL_CODES
    if status == "service_failure":
        return code in ("SUREPASS_FAILURE", "OPENAI_FAILURE")
    if status == "ok":
        return code is None or code == "NO_PINCODE_ON_PAN"
    return code is None


def _require_code_legal(status: CheckStatus, code: Code | None) -> None:
    if not code_legal(status, code):
        raise ValueError(f"code_legality violation: status={status!r} cannot carry code={code!r}")


# --------------------------------------------------------------------------- frozen rows


@dataclass(frozen=True, slots=True)
class ConsentRow:
    id: str
    lc_id: str
    lead_id: str | None
    opportunity_id: str | None
    contact_id: str | None
    pan_hash: str
    mobile_hash: str
    lc_typed_name: str  # the LC-typed name lives ONLY here (PRD 6)
    wording_version: str
    call_ref: str | None
    ip: str | None
    covers: ConsentCovers
    given_at: datetime
    purge_after: datetime  # given_at + 5 years


@dataclass(frozen=True, slots=True)
class PanRecordRow:
    """180-day PAN Comprehensive cache; ONLY valid+person responses are cached."""

    pan_hash: str
    pan_last4: str
    storage_key: str
    client_id: str | None
    fetched_at: datetime
    expires_at: datetime  # fetched_at + 180d
    purge_after: datetime  # fetched_at + 180d


@dataclass(frozen=True, slots=True)
class ScoreCacheRow:
    """30-day Experian Score cache; ALSO refreshed by every successful Report pull (source='report')."""

    pan_hash: str
    storage_key: str
    source: ScoreSource
    is_ntc: bool
    client_id: str | None
    consent_id: str | None  # consent of the ORIGINAL pull; cache hits surface this reference (PRD 5.1)
    fetched_at: datetime
    expires_at: datetime  # fetched + 30d
    purge_after: datetime  # fetched + 180d (raw retention outlives the window)


@dataclass(frozen=True, slots=True)
class ReportCacheRow:
    """30-day Experian Report cache, per PAN hash, shared across opportunities."""

    pan_hash: str
    storage_key: str
    is_ntc: bool
    report_date: date | None  # CreditProfileHeader.ReportDate
    client_id: str | None
    consent_id: str | None
    fetched_at: datetime
    expires_at: datetime
    purge_after: datetime


@dataclass(frozen=True, slots=True)
class CheckRow:
    id: str
    kind: Kind
    lc_id: str
    lead_id: str | None
    opportunity_id: str | None
    opportunity_stage: str | None  # caller-supplied, trusted, LOGGED verbatim (PRD 5.8)
    contact_id: str | None
    pan_hash: str
    pan_last4: str
    mobile_hash: str | None
    mobile_last4: str | None
    status: CheckStatus
    code: Code | None
    billed: bool
    counted_against_quota: bool
    is_refresh: bool
    recompute_of: str | None
    consent_id: str | None
    lc_view: dict[str, Any] | None
    internal: dict[str, Any] | None
    langsmith_run_id: str | None
    created_at: datetime


class CheckPatch(TypedDict, total=False):
    status: CheckStatus
    code: Code | None
    billed: bool
    counted_against_quota: bool
    # K4 says is_refresh is set at insert_check, but K10 inserts the row at admission step 1.5, BEFORE the decision that
    # knows it; so persist carries it on the ONE guarded terminal patch (Task 11). consent_id stays insert-only.
    is_refresh: bool
    contact_id: str | None
    lc_view: dict[str, Any] | None
    internal: dict[str, Any] | None
    langsmith_run_id: str | None


@dataclass(frozen=True, slots=True)
class SurePassCallRow:
    id: str
    check_id: str
    pan_hash: str  # denormalised for bureau_calls_today()
    endpoint: str
    client_id: str | None
    http_status: int | None
    success: bool | None
    message_code: str | None
    latency_ms: int | None  # PRD 11 log fields
    billable: bool  # (200 AND success) OR error_kind == 'timeout'
    error_kind: ErrorKind | None
    storage_key: str | None  # raw envelope object; None when no body arrived
    called_at: datetime
    purge_after: datetime  # called_at + RAW_RETENTION
    purged_at: datetime | None


@dataclass(frozen=True, slots=True)
class PanLockRow:
    """active := released_at is None and lock_until > now."""

    pan_hash: str
    kind: LockKind
    check_id: str | None
    locked_at: datetime
    lock_until: datetime
    released_at: datetime | None


@dataclass(frozen=True, slots=True)
class ReviewRow:
    id: str
    check_id: str | None
    pan_hash: str
    kind: ReviewKind
    reason: str
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None


@dataclass(frozen=True, slots=True)
class OpportunityPanEvent:
    id: str
    opportunity_id: str
    pan_hash: str
    pan_last4: str
    event: PanEventKind
    actor_lc_id: str
    check_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class LenderRow:
    """Always the PUBLISHED version (matching reads published only, PRD 11)."""

    id: str
    version: int
    name: str
    published: bool
    criteria: dict[str, Any]  # each field {'value': ..., 'status': 'known' | 'missing' | 'uncertain'}
    policy_notes: str
    matched_pincodes: tuple[str, ...]  # populated by lenders_for_pincodes, () from published_lenders


# --------------------------------------------------------------------------- audit-sink helpers


REPORT_ON_RECORD_STATUSES: Final[tuple[CheckStatus, ...]] = ("ok", "pending_verification", "service_failure")


def report_check_on_record(status: str | None, code: str | None) -> bool:
    """Whether a FULL check with this terminal ``(status, code)`` means a report exists for its opportunity (PRD 5.13).

    ``ok`` (a fresh pull or a report-reuse hit, band reads included); ``pending_verification`` (a timed-out pull: spend
    unknown, so the report may exist); and ``service_failure`` + ``OPENAI_FAILURE``: the report WAS pulled and billed, only
    the ranking failed afterwards (the same facts K9 lets a recompute re-derive). Every other failure, a refusal, an
    unverified PAN, a running row and an ``error`` row is not. RESIDUAL: an ``error`` row written after a report pull
    (e.g. a persist failure) does not count; the surepass_calls log keeps that spend.
    """
    if status == "service_failure":
        return code == "OPENAI_FAILURE"
    return status in ("ok", "pending_verification")


def call_is_billable(http_status: int | None, success: bool | None, error_kind: ErrorKind | None) -> bool:
    """(HTTP 200 and success:true) or a timeout (spend unknown, so conservatively billable)."""
    return (http_status == 200 and success is True) or error_kind == "timeout"


def storage_key_for(pan_hash: str, endpoint: str, called_at: datetime, call_id: str) -> str:
    """``pan/{pan_hash}/{slug}/{called_at:%Y%m%dT%H%M%SZ}-{call_id}.json``.

    Slug mapping: pan-comprehensive | experian-score | experian-report. One object per HTTP attempt that
    returned a body (every raw response is kept verbatim); opportunity linkage is relational through
    surepass_calls.check_id -> checks.opportunity_id, never part of the key.
    """
    slug = _ENDPOINT_SLUGS.get(endpoint)
    if slug is None:
        raise ValueError(f"no raw-storage slug for endpoint {endpoint!r}")
    stamp = called_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pan/{pan_hash}/{slug}/{stamp}-{call_id}.json"


def _ist_day_bounds(now: datetime) -> tuple[datetime, datetime]:
    """[start, end) of the IST calendar day containing ``now``, exactly as the SQL functions compute it."""
    start = now.astimezone(_IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


# --------------------------------------------------------------------------- raw storage


class RawUnavailable(Exception):
    """The storage object is missing (never written, or purged)."""


class RawStore(Protocol):
    def put_raw(self, key: str, envelope: dict[str, Any]) -> None: ...

    def get_raw(self, key: str) -> dict[str, Any]: ...  # raises RawUnavailable

    def delete_raw(self, keys: Sequence[str]) -> None: ...  # missing keys are not an error


class MemoryRawStore:
    """In-memory bucket. Stores the same compact JSON text the real bucket would, so a non-JSON envelope
    fails here too and get_raw always returns a fresh deep copy."""

    def __init__(self) -> None:
        self._objects: dict[str, str] = {}

    def put_raw(self, key: str, envelope: dict[str, Any]) -> None:
        self._objects[key] = json.dumps(envelope, separators=(",", ":"))  # upsert, like upsert=true

    def get_raw(self, key: str) -> dict[str, Any]:
        try:
            text = self._objects[key]
        except KeyError:
            raise RawUnavailable(key) from None
        loaded: dict[str, Any] = json.loads(text)
        return loaded

    def delete_raw(self, keys: Sequence[str]) -> None:
        for key in keys:
            self._objects.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._objects)


class SupabaseRawStore:
    """Private Supabase Storage bucket (AES-256 at rest). Bucket creation is an ops step (see schema.sql)."""

    def __init__(self, client: Client, bucket: str = DEFAULT_RAW_BUCKET) -> None:
        self._client = client
        self._bucket = bucket

    def put_raw(self, key: str, envelope: dict[str, Any]) -> None:
        self._client.storage.from_(self._bucket).upload(
            path=key,
            file=json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
            file_options={"content-type": "application/json", "upsert": "true"},
        )

    def get_raw(self, key: str) -> dict[str, Any]:
        from storage3.exceptions import StorageApiError

        try:
            data: bytes = self._client.storage.from_(self._bucket).download(key)
        except StorageApiError as exc:
            raise RawUnavailable(key) from exc
        loaded: dict[str, Any] = json.loads(data)
        return loaded

    def delete_raw(self, keys: Sequence[str]) -> None:
        if keys:
            self._client.storage.from_(self._bucket).remove(list(keys))


# --------------------------------------------------------------------------- the protocols


class Repo(Protocol):
    """Repo protocol v2: 29 methods. Both backends satisfy it structurally."""

    # ---- checks (C1 lifecycle: insert-early, ONE guarded terminal update)
    def insert_check(self, row: CheckRow) -> None: ...  # ValueError on duplicate id

    def update_check(self, check_id: str, patch: CheckPatch) -> None: ...
    # Terminal patches apply ONLY while the row's status is 'running' (a no-op is silent, never an error);
    # a patch containing ONLY contact_id is exempt.

    def get_check(self, check_id: str) -> CheckRow | None: ...

    def record_check_error(
        self, *, check_id: str, kind: Kind, lc_id: str, pan_hash: str, pan_last4: str, message: str
    ) -> None: ...  # update-if-running, else insert status='error'

    def attach_contact(self, check_id: str, contact_id: str) -> None: ...  # checks.contact_id + its consent's

    # ---- consents
    def insert_consent(self, row: ConsentRow) -> None: ...

    def get_consent(self, consent_id: str) -> ConsentRow | None: ...  # attach reads the consent's contact (A2b)

    # ---- caches: plain upserts; stale rows are returned too (freshness is policy's call)
    def get_pan_record(self, pan_hash: str) -> PanRecordRow | None: ...

    def put_pan_record(self, row: PanRecordRow) -> None: ...

    def get_score_cache(self, pan_hash: str) -> ScoreCacheRow | None: ...

    def put_score_cache(self, row: ScoreCacheRow) -> None: ...  # also called by the Report pull (source='report')

    def get_report_cache(self, pan_hash: str) -> ReportCacheRow | None: ...

    def put_report_cache(self, row: ReportCacheRow) -> None: ...

    def fetch_raw(self, storage_key: str) -> dict[str, Any]: ...  # RawUnavailable when purged

    # ---- quota / caps (IST day)
    def quick_checks_used_today(self, lc_id: str) -> int: ...

    def quota_limit(self, lc_id: str) -> int: ...

    def bureau_calls_today(self, pan_hash: str) -> int: ...

    # ---- locks
    def try_acquire_pan_lock(self, pan_hash: str, check_id: str, until: datetime) -> bool: ...

    def release_pan_lock(self, pan_hash: str, check_id: str) -> None: ...  # holder-only, in_flight-only, idempotent

    # timeout/daily_cap upsert only: in_flight is written solely by try_acquire_pan_lock (atomic RPC), so ``kind``
    # excludes it and pyright rejects any caller that would overwrite a live in-flight lock.
    def create_pan_lock(
        self, pan_hash: str, kind: Literal["timeout", "daily_cap"], check_id: str | None, until: datetime
    ) -> None: ...

    def get_active_locks(self, pan_hash: str) -> list[PanLockRow]: ...

    # ---- review queue
    def insert_review_item(self, row: ReviewRow) -> None: ...

    def unresolved_review_exists(self, pan_hash: str, kind: ReviewKind) -> bool: ...

    # ---- PAN correction (PRD 5.13)
    def pan_events_for_opportunity(self, opportunity_id: str) -> list[OpportunityPanEvent]: ...  # created_at asc

    def insert_pan_event(self, row: OpportunityPanEvent) -> None: ...

    def report_exists_for_opportunity(self, opportunity_id: str) -> bool: ...  # see report_check_on_record

    # ---- SurePass audit sink (the client calls this once per HTTP attempt)
    def log_surepass_call(
        self,
        *,
        check_id: str,
        pan_hash: str,
        endpoint: str,
        client_id: str | None,
        http_status: int | None,
        success: bool | None,
        message_code: str | None,
        latency_ms: int | None,
        error_kind: ErrorKind | None,
        raw_envelope: dict[str, Any] | None,
    ) -> str | None: ...

    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]: ...  # called_at asc, id asc

    # ---- lenders (published version only)
    def published_lenders(self) -> list[LenderRow]: ...  # ordered by id

    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]: ...


class PurgeRepo(Protocol):
    """What scripts/purge_raw.py needs on top of RawStore. Kept off ``Repo`` so the hot protocol stays at 29."""

    def expired_surepass_calls(self, now: datetime, *, limit: int | None) -> list[SurePassCallRow]: ...
    # purge_after < now and purged_at is None, ordered (called_at, id); limit=None returns all of them

    def mark_calls_purged(self, call_ids: Sequence[str], purged_at: datetime) -> None: ...

    def expired_cache_rows(self, now: datetime) -> list[str]: ...  # 'table:pan_hash', purge_after < now

    def delete_expired_cache_rows(self, now: datetime) -> int: ...

    def expired_consent_ids(self, now: datetime) -> list[str]: ...  # purge_after < now

    def delete_expired_consents(self, now: datetime) -> int: ...  # referencing rows are unlinked (ON DELETE SET NULL)


class LenderWriter(Protocol):
    """What scripts/seed_lenders.py needs. Kept off ``Repo``: the service never writes lenders (the criteria admin page
    does, against the same tables); this is the bootstrap path for a fresh database."""

    def lender_versions(self, lender_id: str) -> list[tuple[int, bool]]: ...  # (version, published), ascending

    def create_published_lender(
        self,
        lender_id: str,
        version: int,
        name: str,
        criteria: dict[str, Any],
        policy_notes: str,
        pincodes: Collection[str],
    ) -> None: ...
    # one lenders row (published) + its lender_pincodes rows; ValueError on a duplicate (id, version), a second
    # published version of the same lender, or a malformed pincode. All-or-nothing as far as the backend allows.


# --------------------------------------------------------------------------- MemoryRepo


class MemoryRepo:
    """In-memory Repo + PurgeRepo with an injected clock. Deep-copies on put AND get; NOT thread-safe.

    IST-day logic mirrors the SQL functions; lock semantics mirror acquire_pan_lock (steal only when the held
    lock is released or expired); the status/code CHECK, the terminal guard, the one-published-version index
    and the surepass_calls -> checks FK are enforced. Also a ``LenderWriter``. Non-protocol helpers for tests and
    evals: seed_lender, set_lender_published, set_quota_override, resolve_review, list_checks, list_consents,
    list_reviews, list_calls, active_locks_snapshot.
    """

    def __init__(self, clock: Clock = utc_now, raw_store: MemoryRawStore | None = None) -> None:
        self._clock = clock
        self.raw_store: MemoryRawStore = raw_store if raw_store is not None else MemoryRawStore()
        self._checks: dict[str, CheckRow] = {}
        self._consents: dict[str, ConsentRow] = {}
        self._pan_records: dict[str, PanRecordRow] = {}
        self._score_cache: dict[str, ScoreCacheRow] = {}
        self._report_cache: dict[str, ReportCacheRow] = {}
        self._calls: list[SurePassCallRow] = []
        self._locks: dict[tuple[str, LockKind], PanLockRow] = {}
        self._reviews: dict[str, ReviewRow] = {}
        self._pan_events: dict[str, OpportunityPanEvent] = {}
        self._quota_overrides: dict[str, int] = {}
        self._lenders: dict[tuple[str, int], LenderRow] = {}
        self._lender_pincodes: dict[tuple[str, int], frozenset[str]] = {}

    # ---- checks

    def insert_check(self, row: CheckRow) -> None:
        if row.id in self._checks:
            raise ValueError(f"duplicate check id {row.id!r}")
        _require_code_legal(row.status, row.code)
        self._checks[row.id] = copy.deepcopy(row)

    def update_check(self, check_id: str, patch: CheckPatch) -> None:
        current = self._checks.get(check_id)
        if current is None:
            return  # no row matched: silent, like a filtered UPDATE
        if set(patch) != {"contact_id"} and current.status != "running":
            return  # terminal guard: only a running row accepts a terminal patch
        merged = replace(current, **copy.deepcopy(dict(patch)))
        _require_code_legal(merged.status, merged.code)
        self._checks[check_id] = merged

    def get_check(self, check_id: str) -> CheckRow | None:
        row = self._checks.get(check_id)
        return copy.deepcopy(row) if row is not None else None

    def record_check_error(
        self, *, check_id: str, kind: Kind, lc_id: str, pan_hash: str, pan_last4: str, message: str
    ) -> None:
        current = self._checks.get(check_id)
        if current is None:
            self._checks[check_id] = CheckRow(
                id=check_id,
                kind=kind,
                lc_id=lc_id,
                lead_id=None,
                opportunity_id=None,
                opportunity_stage=None,
                contact_id=None,
                pan_hash=pan_hash,
                pan_last4=pan_last4,
                mobile_hash=None,
                mobile_last4=None,
                status="error",
                code=None,
                billed=False,
                counted_against_quota=False,
                is_refresh=False,
                recompute_of=None,
                consent_id=None,
                lc_view=None,
                internal={"error": message},
                langsmith_run_id=None,
                created_at=self._clock(),
            )
        elif current.status == "running":
            self._checks[check_id] = replace(current, status="error", code=None, internal={"error": message})
        # else: already terminal; never overwrite a persisted result and never raise from the error boundary

    def attach_contact(self, check_id: str, contact_id: str) -> None:
        current = self._checks.get(check_id)
        if current is None:
            return
        self._checks[check_id] = replace(current, contact_id=contact_id)
        if current.consent_id is not None and current.consent_id in self._consents:
            self._consents[current.consent_id] = replace(self._consents[current.consent_id], contact_id=contact_id)

    # ---- consents

    def insert_consent(self, row: ConsentRow) -> None:
        if row.id in self._consents:
            raise ValueError(f"duplicate consent id {row.id!r}")
        self._consents[row.id] = copy.deepcopy(row)

    def get_consent(self, consent_id: str) -> ConsentRow | None:
        return copy.deepcopy(self._consents.get(consent_id))

    # ---- caches

    def get_pan_record(self, pan_hash: str) -> PanRecordRow | None:
        return copy.deepcopy(self._pan_records.get(pan_hash))

    def put_pan_record(self, row: PanRecordRow) -> None:
        self._pan_records[row.pan_hash] = copy.deepcopy(row)

    def get_score_cache(self, pan_hash: str) -> ScoreCacheRow | None:
        return copy.deepcopy(self._score_cache.get(pan_hash))

    def put_score_cache(self, row: ScoreCacheRow) -> None:
        self._score_cache[row.pan_hash] = copy.deepcopy(row)

    def get_report_cache(self, pan_hash: str) -> ReportCacheRow | None:
        return copy.deepcopy(self._report_cache.get(pan_hash))

    def put_report_cache(self, row: ReportCacheRow) -> None:
        self._report_cache[row.pan_hash] = copy.deepcopy(row)

    def fetch_raw(self, storage_key: str) -> dict[str, Any]:
        return self.raw_store.get_raw(storage_key)

    # ---- quota / caps

    def quick_checks_used_today(self, lc_id: str) -> int:
        start, end = _ist_day_bounds(self._clock())
        return sum(
            1
            for c in self._checks.values()
            if c.lc_id == lc_id and c.kind == "quick" and c.counted_against_quota and start <= c.created_at < end
        )

    def quota_limit(self, lc_id: str) -> int:
        return self._quota_overrides.get(lc_id, QUICK_DAILY_QUOTA_DEFAULT)

    def bureau_calls_today(self, pan_hash: str) -> int:
        start, end = _ist_day_bounds(self._clock())
        return sum(
            1
            for c in self._calls
            if c.pan_hash == pan_hash and c.billable and c.endpoint in _BUREAU_ENDPOINTS and start <= c.called_at < end
        )

    # ---- locks

    def try_acquire_pan_lock(self, pan_hash: str, check_id: str, until: datetime) -> bool:
        now = self._clock()
        held = self._locks.get((pan_hash, "in_flight"))
        if held is not None and held.released_at is None and held.lock_until > now:
            return False
        self._locks[(pan_hash, "in_flight")] = PanLockRow(pan_hash, "in_flight", check_id, now, until, None)
        return True

    def release_pan_lock(self, pan_hash: str, check_id: str) -> None:
        held = self._locks.get((pan_hash, "in_flight"))
        if held is not None and held.check_id == check_id and held.released_at is None:
            self._locks[(pan_hash, "in_flight")] = replace(held, released_at=self._clock())

    def create_pan_lock(
        self, pan_hash: str, kind: Literal["timeout", "daily_cap"], check_id: str | None, until: datetime
    ) -> None:
        self._locks[(pan_hash, kind)] = PanLockRow(pan_hash, kind, check_id, self._clock(), until, None)

    def get_active_locks(self, pan_hash: str) -> list[PanLockRow]:
        now = self._clock()
        active = [
            lock
            for (pan, _), lock in self._locks.items()
            if pan == pan_hash and lock.released_at is None and lock.lock_until > now
        ]
        return sorted(active, key=lambda lock: lock.kind)

    # ---- review queue

    def insert_review_item(self, row: ReviewRow) -> None:
        if row.id in self._reviews:
            raise ValueError(f"duplicate review id {row.id!r}")
        self._reviews[row.id] = copy.deepcopy(row)

    def unresolved_review_exists(self, pan_hash: str, kind: ReviewKind) -> bool:
        return any(r.pan_hash == pan_hash and r.kind == kind and r.resolved_at is None for r in self._reviews.values())

    # ---- PAN correction

    def pan_events_for_opportunity(self, opportunity_id: str) -> list[OpportunityPanEvent]:
        events = [e for e in self._pan_events.values() if e.opportunity_id == opportunity_id]
        return copy.deepcopy(sorted(events, key=lambda e: e.created_at))  # stable: ties keep insertion order

    def insert_pan_event(self, row: OpportunityPanEvent) -> None:
        if row.id in self._pan_events:
            raise ValueError(f"duplicate pan event id {row.id!r}")
        self._pan_events[row.id] = copy.deepcopy(row)

    def report_exists_for_opportunity(self, opportunity_id: str) -> bool:
        return any(
            c.opportunity_id == opportunity_id and c.kind == "full" and report_check_on_record(c.status, c.code)
            for c in self._checks.values()
        )

    # ---- audit sink

    def log_surepass_call(
        self,
        *,
        check_id: str,
        pan_hash: str,
        endpoint: str,
        client_id: str | None,
        http_status: int | None,
        success: bool | None,
        message_code: str | None,
        latency_ms: int | None,
        error_kind: ErrorKind | None,
        raw_envelope: dict[str, Any] | None,
    ) -> str | None:
        if check_id not in self._checks:
            # FK parity: surepass_calls.check_id references checks(id); the admission node inserts the row first.
            raise AssertionError(f"FK violation: no checks row {check_id!r} for this SurePass call")
        call_id = new_id()
        called_at = self._clock()
        storage_key: str | None = None
        if raw_envelope is not None:
            storage_key = storage_key_for(pan_hash, endpoint, called_at, call_id)
            self.raw_store.put_raw(storage_key, raw_envelope)  # raw FIRST: no row may point at a missing object
        self._calls.append(
            SurePassCallRow(
                id=call_id,
                check_id=check_id,
                pan_hash=pan_hash,
                endpoint=endpoint,
                client_id=client_id,
                http_status=http_status,
                success=success,
                message_code=message_code,
                latency_ms=latency_ms,
                billable=call_is_billable(http_status, success, error_kind),
                error_kind=error_kind,
                storage_key=storage_key,
                called_at=called_at,
                purge_after=called_at + RAW_RETENTION,
                purged_at=None,
            )
        )
        return storage_key

    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]:
        return sorted((c for c in self._calls if c.check_id == check_id), key=lambda c: (c.called_at, c.id))

    # ---- lenders

    def published_lenders(self) -> list[LenderRow]:
        rows = [r for r in self._lenders.values() if r.published]
        return copy.deepcopy(sorted(rows, key=lambda r: r.id))

    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]:
        wanted = set(pincodes)
        if not wanted:
            return []
        out: list[LenderRow] = []
        for key, row in sorted(self._lenders.items(), key=lambda item: item[0][0]):
            if not row.published:
                continue
            matched = sorted(wanted & self._lender_pincodes[key])  # the SAME (id, version) only
            if matched:
                out.append(copy.deepcopy(replace(row, matched_pincodes=tuple(matched))))
        return out

    # ---- purge operations

    def expired_surepass_calls(self, now: datetime, *, limit: int | None) -> list[SurePassCallRow]:
        due = sorted(
            (c for c in self._calls if c.purge_after < now and c.purged_at is None),
            key=lambda c: (c.called_at, c.id),
        )
        return due if limit is None else due[:limit]

    def mark_calls_purged(self, call_ids: Sequence[str], purged_at: datetime) -> None:
        wanted = set(call_ids)
        self._calls = [replace(c, purged_at=purged_at) if c.id in wanted else c for c in self._calls]

    def expired_cache_rows(self, now: datetime) -> list[str]:
        return [
            f"{table}:{pan_hash}"
            for table, rows in self._cache_tables()
            for pan_hash, row in sorted(rows.items())
            if row.purge_after < now
        ]

    def delete_expired_cache_rows(self, now: datetime) -> int:
        deleted = 0
        for _, rows in self._cache_tables():
            for pan_hash in [h for h, row in rows.items() if row.purge_after < now]:
                del rows[pan_hash]
                deleted += 1
        return deleted

    def expired_consent_ids(self, now: datetime) -> list[str]:
        return sorted(c.id for c in self._consents.values() if c.purge_after < now)

    def delete_expired_consents(self, now: datetime) -> int:
        gone = set(self.expired_consent_ids(now))
        for consent_id in gone:
            del self._consents[consent_id]
        # ON DELETE SET NULL: rows that referenced a purged consent survive, unlinked
        self._checks = {k: replace(c, consent_id=None) if c.consent_id in gone else c for k, c in self._checks.items()}
        self._score_cache = {
            k: replace(c, consent_id=None) if c.consent_id in gone else c for k, c in self._score_cache.items()
        }
        self._report_cache = {
            k: replace(c, consent_id=None) if c.consent_id in gone else c for k, c in self._report_cache.items()
        }
        return len(gone)

    def _cache_tables(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("pan_records", self._pan_records),
            ("score_cache", self._score_cache),
            ("report_cache", self._report_cache),
        ]

    # ---- LenderWriter (scripts/seed_lenders.py)

    def lender_versions(self, lender_id: str) -> list[tuple[int, bool]]:
        return sorted((version, row.published) for (row_id, version), row in self._lenders.items() if row_id == lender_id)

    def create_published_lender(
        self,
        lender_id: str,
        version: int,
        name: str,
        criteria: dict[str, Any],
        policy_notes: str,
        pincodes: Collection[str],
    ) -> None:
        self.seed_lender(lender_id, version, name, True, criteria, policy_notes, pincodes)

    # ---- test / eval helpers (not part of the protocol)

    def seed_lender(
        self,
        lender_id: str,
        version: int,
        name: str,
        published: bool,
        criteria: dict[str, Any],
        policy_notes: str,
        pincodes: Collection[str],
    ) -> None:
        if (lender_id, version) in self._lenders:
            raise ValueError(f"duplicate lender version ({lender_id!r}, {version})")
        if published and any(r.id == lender_id and r.published for r in self._lenders.values()):
            raise ValueError(f"lender {lender_id!r} already has a published version (idx_lenders_one_published)")
        bad = [p for p in pincodes if not _PINCODE_RE.fullmatch(p)]
        if bad:
            raise ValueError(f"invalid pincode(s) {bad!r}")
        self._lenders[(lender_id, version)] = LenderRow(
            id=lender_id,
            version=version,
            name=name,
            published=published,
            criteria=copy.deepcopy(criteria),
            policy_notes=policy_notes,
            matched_pincodes=(),
        )
        self._lender_pincodes[(lender_id, version)] = frozenset(pincodes)

    def set_lender_published(self, lender_id: str, version: int, published: bool) -> None:
        """The admin's publish / unpublish flip. KeyError for an unknown (id, version); ValueError when publishing while
        ANOTHER version of the lender is published (idx_lenders_one_published). Idempotent."""
        key = (lender_id, version)
        row = self._lenders[key]  # KeyError on an unknown version
        if published and any(r.id == lender_id and r.published and k != key for k, r in self._lenders.items()):
            raise ValueError(f"lender {lender_id!r} already has a published version (idx_lenders_one_published)")
        self._lenders[key] = replace(row, published=published)

    def set_quota_override(self, lc_id: str, daily_quota: int) -> None:
        if daily_quota < 0:
            raise ValueError("daily_quota must be >= 0")
        self._quota_overrides[lc_id] = daily_quota

    def resolve_review(self, review_id: str, resolved_by: str = "test") -> None:
        row = self._reviews[review_id]  # KeyError on an unknown id
        self._reviews[review_id] = replace(row, resolved_at=self._clock(), resolved_by=resolved_by)

    def list_checks(self) -> list[CheckRow]:
        return copy.deepcopy(list(self._checks.values()))

    def list_consents(self) -> list[ConsentRow]:
        return copy.deepcopy(list(self._consents.values()))

    def list_reviews(self) -> list[ReviewRow]:
        return copy.deepcopy(list(self._reviews.values()))

    def list_calls(self, check_id: str | None = None) -> list[SurePassCallRow]:
        return [c for c in self._calls if check_id is None or c.check_id == check_id]

    def active_locks_snapshot(self) -> list[PanLockRow]:
        """Every lock row held in memory, released or expired ones included."""
        return list(self._locks.values())


# --------------------------------------------------------------------------- Supabase mapping helpers

_R = TypeVar("_R", bound="DataclassInstance")


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("naive datetime: every timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _encode_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_encode_value(v) for v in value]
    return value


def _encode(row: DataclassInstance) -> dict[str, Any]:
    return {f.name: _encode_value(getattr(row, f.name)) for f in dataclasses.fields(row)}


def _decode(cls: type[_R], data: dict[str, Any], *, ts: tuple[str, ...] = (), dates: tuple[str, ...] = ()) -> _R:
    values: dict[str, Any] = {f.name: data.get(f.name) for f in dataclasses.fields(cls)}
    for name in ts:
        if values[name] is not None:
            values[name] = _ts(values[name])
    for name in dates:
        if values[name] is not None:
            values[name] = date.fromisoformat(values[name])
    return cls(**values)


def _rows(response: Any) -> list[dict[str, Any]]:
    data = response.data
    return list(data) if data else []


def _scalar(response: Any) -> Any:
    """An rpc call returning a scalar yields that bare JSON value in ``.data``."""
    return response.data


def _decode_call(data: dict[str, Any]) -> SurePassCallRow:
    return _decode(SurePassCallRow, data, ts=("called_at", "purge_after", "purged_at"))


def _decode_lender(data: dict[str, Any], matched: Sequence[str] = ()) -> LenderRow:
    return LenderRow(
        id=data["id"],
        version=int(data["version"]),
        name=data["name"],
        published=bool(data["published"]),
        criteria=data.get("criteria") or {},
        policy_notes=data.get("policy_notes") or "",
        matched_pincodes=tuple(matched),
    )


# --------------------------------------------------------------------------- SupabaseRepo


class SupabaseRepo:
    """Repo + PurgeRepo over supabase-py v2 (sync client, SERVICE ROLE key; RLS on, no policies).

    Mappings: row inserts/updates/upserts field-for-field with ISO timestamps; the terminal update_check adds
    ``.eq('status', 'running')``; the quota / cap / lock-acquire methods call the SQL functions through rpc.
    The sink uploads the raw object first, then inserts the call row.
    """

    def __init__(self, client: Client, raw_store: RawStore, clock: Clock = utc_now) -> None:
        self._client = client
        self._raw_store = raw_store
        self._clock = clock

    # ---- small helpers

    def _insert(self, table: str, row: DataclassInstance) -> None:
        try:
            self._client.table(table).insert(_encode(row)).execute()
        except Exception as exc:
            if getattr(exc, "code", None) == "23505":  # unique_violation (primary key)
                raise ValueError(f"duplicate primary key inserting into {table}") from exc
            raise

    def _select_one(self, table: str, column: str, value: str) -> dict[str, Any] | None:
        rows = _rows(self._client.table(table).select("*").eq(column, value).limit(1).execute())
        return rows[0] if rows else None

    # ---- checks

    def insert_check(self, row: CheckRow) -> None:
        self._insert("checks", row)

    def update_check(self, check_id: str, patch: CheckPatch) -> None:
        if not patch:
            return
        payload: dict[str, Any] = dict(patch)
        query = self._client.table("checks").update(payload).eq("id", check_id)
        if set(patch) != {"contact_id"}:
            query = query.eq("status", "running")  # the C1 terminal guard
        query.execute()

    def get_check(self, check_id: str) -> CheckRow | None:
        data = self._select_one("checks", "id", check_id)
        return _decode(CheckRow, data, ts=("created_at",)) if data is not None else None

    def record_check_error(
        self, *, check_id: str, kind: Kind, lc_id: str, pan_hash: str, pan_last4: str, message: str
    ) -> None:
        patch: dict[str, Any] = {"status": "error", "code": None, "internal": {"error": message}}
        updated = self._client.table("checks").update(patch).eq("id", check_id).eq("status", "running").execute()
        if _rows(updated):
            return
        row = CheckRow(
            id=check_id,
            kind=kind,
            lc_id=lc_id,
            lead_id=None,
            opportunity_id=None,
            opportunity_stage=None,
            contact_id=None,
            pan_hash=pan_hash,
            pan_last4=pan_last4,
            mobile_hash=None,
            mobile_last4=None,
            status="error",
            code=None,
            billed=False,
            counted_against_quota=False,
            is_refresh=False,
            recompute_of=None,
            consent_id=None,
            lc_view=None,
            internal={"error": message},
            langsmith_run_id=None,
            created_at=self._clock(),
        )
        # Absent -> inserted; already terminal -> the conflict is ignored (never overwrite a persisted result).
        self._client.table("checks").upsert(_encode(row), on_conflict="id", ignore_duplicates=True).execute()

    def attach_contact(self, check_id: str, contact_id: str) -> None:
        data = self._select_one("checks", "id", check_id)
        if data is None:
            return
        self._client.table("checks").update({"contact_id": contact_id}).eq("id", check_id).execute()
        consent_id = data.get("consent_id")
        if consent_id:
            self._client.table("consents").update({"contact_id": contact_id}).eq("id", consent_id).execute()

    # ---- consents

    def insert_consent(self, row: ConsentRow) -> None:
        self._insert("consents", row)

    def get_consent(self, consent_id: str) -> ConsentRow | None:
        data = self._select_one("consents", "id", consent_id)
        return _decode(ConsentRow, data, ts=("given_at", "purge_after")) if data is not None else None

    # ---- caches

    def get_pan_record(self, pan_hash: str) -> PanRecordRow | None:
        data = self._select_one("pan_records", "pan_hash", pan_hash)
        return _decode(PanRecordRow, data, ts=("fetched_at", "expires_at", "purge_after")) if data else None

    def put_pan_record(self, row: PanRecordRow) -> None:
        self._client.table("pan_records").upsert(_encode(row), on_conflict="pan_hash").execute()

    def get_score_cache(self, pan_hash: str) -> ScoreCacheRow | None:
        data = self._select_one("score_cache", "pan_hash", pan_hash)
        return _decode(ScoreCacheRow, data, ts=("fetched_at", "expires_at", "purge_after")) if data else None

    def put_score_cache(self, row: ScoreCacheRow) -> None:
        self._client.table("score_cache").upsert(_encode(row), on_conflict="pan_hash").execute()

    def get_report_cache(self, pan_hash: str) -> ReportCacheRow | None:
        data = self._select_one("report_cache", "pan_hash", pan_hash)
        if not data:
            return None
        return _decode(ReportCacheRow, data, ts=("fetched_at", "expires_at", "purge_after"), dates=("report_date",))

    def put_report_cache(self, row: ReportCacheRow) -> None:
        self._client.table("report_cache").upsert(_encode(row), on_conflict="pan_hash").execute()

    def fetch_raw(self, storage_key: str) -> dict[str, Any]:
        return self._raw_store.get_raw(storage_key)

    # ---- quota / caps

    def quick_checks_used_today(self, lc_id: str) -> int:
        response = self._client.rpc("quick_checks_used_today", {"p_lc_id": lc_id}).execute()
        return int(_scalar(response) or 0)

    def quota_limit(self, lc_id: str) -> int:
        rows = _rows(self._client.table("lc_quota_overrides").select("daily_quota").eq("lc_id", lc_id).limit(1).execute())
        return int(rows[0]["daily_quota"]) if rows else QUICK_DAILY_QUOTA_DEFAULT  # 0 is a real override

    def bureau_calls_today(self, pan_hash: str) -> int:
        response = self._client.rpc("bureau_calls_today", {"p_pan_hash": pan_hash}).execute()
        return int(_scalar(response) or 0)

    # ---- locks

    def try_acquire_pan_lock(self, pan_hash: str, check_id: str, until: datetime) -> bool:
        response = self._client.rpc(
            "acquire_pan_lock", {"p_pan_hash": pan_hash, "p_check_id": check_id, "p_until": _iso(until)}
        ).execute()
        return bool(_scalar(response))

    def release_pan_lock(self, pan_hash: str, check_id: str) -> None:
        (
            self._client.table("pan_locks")
            .update({"released_at": _iso(self._clock())})
            .eq("pan_hash", pan_hash)
            .eq("kind", "in_flight")
            .eq("check_id", check_id)
            .is_("released_at", "null")
            .execute()
        )

    def create_pan_lock(
        self, pan_hash: str, kind: Literal["timeout", "daily_cap"], check_id: str | None, until: datetime
    ) -> None:
        row = PanLockRow(pan_hash, kind, check_id, self._clock(), until, None)
        self._client.table("pan_locks").upsert(_encode(row), on_conflict="pan_hash,kind").execute()

    def get_active_locks(self, pan_hash: str) -> list[PanLockRow]:
        response = (
            self._client.table("pan_locks")
            .select("*")
            .eq("pan_hash", pan_hash)
            .is_("released_at", "null")
            .gt("lock_until", _iso(self._clock()))
            .order("kind")
            .execute()
        )
        return [_decode(PanLockRow, r, ts=("locked_at", "lock_until", "released_at")) for r in _rows(response)]

    # ---- review queue

    def insert_review_item(self, row: ReviewRow) -> None:
        self._insert("review_queue", row)

    def unresolved_review_exists(self, pan_hash: str, kind: ReviewKind) -> bool:
        response = (
            self._client.table("review_queue")
            .select("id")
            .eq("pan_hash", pan_hash)
            .eq("kind", kind)
            .is_("resolved_at", "null")
            .limit(1)
            .execute()
        )
        return bool(_rows(response))

    # ---- PAN correction

    def pan_events_for_opportunity(self, opportunity_id: str) -> list[OpportunityPanEvent]:
        response = (
            self._client.table("opportunity_pan_events")
            .select("*")
            .eq("opportunity_id", opportunity_id)
            .order("created_at")
            .execute()
        )
        return [_decode(OpportunityPanEvent, r, ts=("created_at",)) for r in _rows(response)]

    def insert_pan_event(self, row: OpportunityPanEvent) -> None:
        self._insert("opportunity_pan_events", row)

    def report_exists_for_opportunity(self, opportunity_id: str) -> bool:
        # One opportunity holds a handful of full checks: fetch the candidate statuses, apply the shared predicate here.
        response = (
            self._client.table("checks")
            .select("status,code")
            .eq("opportunity_id", opportunity_id)
            .eq("kind", "full")
            .in_("status", list(REPORT_ON_RECORD_STATUSES))
            .execute()
        )
        return any(report_check_on_record(row.get("status"), row.get("code")) for row in _rows(response))

    # ---- audit sink

    def log_surepass_call(
        self,
        *,
        check_id: str,
        pan_hash: str,
        endpoint: str,
        client_id: str | None,
        http_status: int | None,
        success: bool | None,
        message_code: str | None,
        latency_ms: int | None,
        error_kind: ErrorKind | None,
        raw_envelope: dict[str, Any] | None,
    ) -> str | None:
        call_id = new_id()
        called_at = self._clock()
        storage_key: str | None = None
        if raw_envelope is not None:
            storage_key = storage_key_for(pan_hash, endpoint, called_at, call_id)
            # Upload FIRST: a crash between the two writes orphans an object (harmless, swept later) instead of
            # leaving a row whose storage_key points at nothing.
            self._raw_store.put_raw(storage_key, raw_envelope)
        row = SurePassCallRow(
            id=call_id,
            check_id=check_id,
            pan_hash=pan_hash,
            endpoint=endpoint,
            client_id=client_id,
            http_status=http_status,
            success=success,
            message_code=message_code,
            latency_ms=latency_ms,
            billable=call_is_billable(http_status, success, error_kind),
            error_kind=error_kind,
            storage_key=storage_key,
            called_at=called_at,
            purge_after=called_at + RAW_RETENTION,
            purged_at=None,
        )
        self._insert("surepass_calls", row)
        return storage_key

    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]:
        response = (
            self._client.table("surepass_calls")
            .select("*")
            .eq("check_id", check_id)
            .order("called_at")
            .order("id")
            .execute()
        )
        return [_decode_call(r) for r in _rows(response)]

    # ---- lenders

    def published_lenders(self) -> list[LenderRow]:
        response = self._client.table("lenders").select("*").eq("published", True).order("id").execute()
        return [_decode_lender(r) for r in _rows(response)]

    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]:
        wanted = sorted(set(pincodes))
        if not wanted:
            return []  # no query on empty input
        hits = _rows(
            self._client.table("lender_pincodes").select("lender_id,version,pincode").in_("pincode", wanted).execute()
        )
        matched: dict[tuple[str, int], set[str]] = {}
        for hit in hits:
            matched.setdefault((hit["lender_id"], int(hit["version"])), set()).add(hit["pincode"])
        if not matched:
            return []
        lender_ids = sorted({lender_id for lender_id, _ in matched})
        lenders = _rows(
            self._client.table("lenders").select("*").in_("id", lender_ids).eq("published", True).order("id").execute()
        )
        # Join on (id, version) app-side: a pincode attached to an unpublished draft must not match the published row.
        return [
            _decode_lender(r, sorted(matched[(r["id"], int(r["version"]))]))
            for r in lenders
            if (r["id"], int(r["version"])) in matched
        ]

    # ---- LenderWriter (scripts/seed_lenders.py)

    def lender_versions(self, lender_id: str) -> list[tuple[int, bool]]:
        response = (
            self._client.table("lenders").select("version,published").eq("id", lender_id).order("version").execute()
        )
        return [(int(r["version"]), bool(r["published"])) for r in _rows(response)]

    def create_published_lender(
        self,
        lender_id: str,
        version: int,
        name: str,
        criteria: dict[str, Any],
        policy_notes: str,
        pincodes: Collection[str],
    ) -> None:
        """Insert UNPUBLISHED, add the pincodes, then flip ``published``: a failure part-way never leaves a published
        lender without its pincodes. The half-written draft is deleted again (the pincode rows cascade) so a rerun starts
        clean; if that delete also fails, the leftover draft is visible (unpublished) and the seed script skips it."""
        wanted = sorted(set(pincodes))
        bad = [p for p in wanted if not _PINCODE_RE.fullmatch(p)]
        if bad:
            raise ValueError(f"invalid pincode(s) {bad!r}")
        try:
            self._client.table("lenders").insert(
                {
                    "id": lender_id,
                    "version": version,
                    "name": name,
                    "published": False,
                    "criteria": copy.deepcopy(criteria),
                    "policy_notes": policy_notes,
                    "created_by": "seed_lenders",
                }
            ).execute()
        except Exception as exc:
            if getattr(exc, "code", None) == "23505":  # unique_violation (primary key)
                raise ValueError(f"duplicate lender version ({lender_id!r}, {version})") from exc
            raise
        try:
            if wanted:
                self._client.table("lender_pincodes").insert(
                    [{"lender_id": lender_id, "version": version, "pincode": p} for p in wanted]
                ).execute()
            try:
                self._client.table("lenders").update({"published": True}).eq("id", lender_id).eq(
                    "version", version
                ).execute()
            except Exception as exc:
                if getattr(exc, "code", None) == "23505":  # idx_lenders_one_published
                    raise ValueError(f"lender {lender_id!r} already has a published version") from exc
                raise
        except Exception:
            try:
                self._client.table("lenders").delete().eq("id", lender_id).eq("version", version).execute()
            except Exception:  # best effort: the original failure is the one worth raising
                pass
            raise

    # ---- purge operations

    def expired_surepass_calls(self, now: datetime, *, limit: int | None) -> list[SurePassCallRow]:
        def due() -> Any:
            return (
                self._client.table("surepass_calls")
                .select("*")
                .lt("purge_after", _iso(now))
                .is_("purged_at", "null")
                .order("called_at")
                .order("id")
            )

        if limit is not None:
            return [_decode_call(r) for r in _rows(due().limit(limit).execute())]
        out: list[SurePassCallRow] = []
        start = 0
        while True:  # unbounded (dry-run): page past PostgREST's max-rows cap
            page = _rows(due().range(start, start + _PAGE_SIZE - 1).execute())
            out.extend(_decode_call(r) for r in page)
            if len(page) < _PAGE_SIZE:
                return out
            start += _PAGE_SIZE

    def mark_calls_purged(self, call_ids: Sequence[str], purged_at: datetime) -> None:
        ids = list(call_ids)
        for start in range(0, len(ids), 100):  # keep the in.() filter's URL short
            self._client.table("surepass_calls").update({"purged_at": _iso(purged_at)}).in_(
                "id", ids[start : start + 100]
            ).execute()

    def expired_cache_rows(self, now: datetime) -> list[str]:
        out: list[str] = []
        for table in _CACHE_TABLES:
            response = self._client.table(table).select("pan_hash").lt("purge_after", _iso(now)).order("pan_hash").execute()
            out.extend(f"{table}:{r['pan_hash']}" for r in _rows(response))
        return out

    def delete_expired_cache_rows(self, now: datetime) -> int:
        deleted = 0
        for table in _CACHE_TABLES:
            deleted += len(_rows(self._client.table(table).delete().lt("purge_after", _iso(now)).execute()))
        return deleted

    def expired_consent_ids(self, now: datetime) -> list[str]:
        response = self._client.table("consents").select("id").lt("purge_after", _iso(now)).order("id").execute()
        return [r["id"] for r in _rows(response)]

    def delete_expired_consents(self, now: datetime) -> int:
        return len(_rows(self._client.table("consents").delete().lt("purge_after", _iso(now)).execute()))


def create_supabase_backend(
    url: str, service_role_key: str, *, bucket: str = DEFAULT_RAW_BUCKET, clock: Clock = utc_now
) -> tuple[SupabaseRepo, SupabaseRawStore]:
    """Build the production backend pair from credentials (the only place ``create_client`` is called)."""
    from supabase import create_client

    client = create_client(url, service_role_key)
    raw_store = SupabaseRawStore(client, bucket)
    return SupabaseRepo(client, raw_store, clock), raw_store
