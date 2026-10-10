# Policy guards and persistence (PRD sections 6 and 10): policy/guards.py, db/repo.py, db/schema.sql

## Summary
This section makes every money-spending and learner-protecting rule deterministic and auditable. policy/guards.py exposes two pure admission functions (admit_quick_check, admit_full_check) that evaluate, in a fixed order, consent -> stage gate -> opportunity-already-checked -> cache lookup -> refresh semantics -> daily cap -> SUREPASS_LIVE kill switch, returning a frozen AdmissionDecision that maps every path to exactly one of the six refusal codes or one of three admission modes (literal decision tables below); it also owns the salted PAN hash, the IST cap constants, the cache-TTL constant, the LC-facing refusal one-liners, and the persist-time formulas for billed / counted_against_cap / quick_checks_remaining / refresh_available_at. db/repo.py defines a Repo Protocol (11 fully typed methods over frozen dataclass rows), a SupabaseRepo mapping each method to supabase-py v2 calls (RPC quick_checks_used_today for the cap; upsert on_conflict="pan_hash,kind" for the cache; a two-query join for lenders_for_pincodes), and a MemoryRepo with an injectable clock and zoneinfo("Asia/Kolkata") day computation that exactly mirrors the SQL. db/schema.sql is the full DDL for the six PRD-10 tables with PKs, FKs, CHECK constraints, a partial index supporting the cap count, RLS enabled with no policies (service-role only), and a sargable quick_checks_used_today(p_lc_id) SQL function that converts to the Asia/Kolkata calendar day. pan_cache.derived is specified literally per kind (identity+score for quick, identity+report for full, with schema_version) so a hit skips SurePass entirely while gate/affordability/rank/finalize re-run per request; payload holds raw envelopes for audit only. The flag lifecycle is specified end to end: admission inserts the checks row early with status='running' and billed/counted false, every logged SurePass call flips billed=True immediately, and persist computes ground-truth billed from surepass_calls, counted_against_cap = counts_against_cap AND bureau_called AND NOT pan_invalid, and the response's remaining/refresh fields, in one final update. All edge cases demanded by the task (refresh with no cache row, expired cache without refresh, cache hit with live calls off, prior-full-check fresh vs stale, same-day refresh/cap interplay) are resolved and justified in decisions.

## Files
### eligibility_agent/policy/guards.py
_Deterministic policy layer (PRD section 6): admission guards with the fixed evaluation order and decision tables, salted PAN hashing, constants (cap, TTL, stage order, endpoint classification), LC-facing refusal messages, and the persist-time flag formulas. Pure read-only functions: guards read via Repo but never write; graph nodes do the writes._

"""Deterministic policy guards. No LLM, no network, no writes.
Layering: policy/ imports ONLY eligibility_agent.db.repo and stdlib. Tools/graph may import policy; never the reverse.

ADMISSION WRITE PATTERN (implemented by the graph admission node, in this order):
  1. consent_id = new_id(); repo.insert_consent(ConsentRow(...))   ONLY when request.consent is True
  2. decision = admit_quick_check(...) | admit_full_check(...)      (read-only, this module)
  3. repo.insert_check(CheckRow(id=check_id, status="running", refusal_code=None, billed=False,
       counted_against_cap=False, is_refresh=(decision.admission == "refresh"), consent_id=consent_id, created_at=now, ...))
     The row is inserted for EVERY admission outcome, refusals included (PRD 9.3: refusals are persisted too),
     and before any SurePass call (surepass_calls.check_id FK requires it).
"""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal
from zoneinfo import ZoneInfo
from eligibility_agent.db.repo import CacheRow, CheckKind, RefusalCode, Repo, SurePassCallRow

# ---- constants (single source of truth; other modules import these) ----
QUICK_DAILY_CAP: Final[int] = 5                       # PRD 6.3 (FIXED)
CACHE_TTL: Final[timedelta] = timedelta(days=30)      # PRD 6.4 (FIXED)
IST: Final[ZoneInfo] = ZoneInfo("Asia/Kolkata")
DERIVED_SCHEMA_VERSION: Final[int] = 1                # bump invalidates cache (treated as stale)
STAGE_ORDER: Final[tuple[str, ...]] = ("new", "contacted", "qualified", "counselling", "offer", "registration", "enrolled", "won")  # ASSUMPTION (PRD 9.2 / open item 1)
FULL_CHECK_MIN_STAGE: Final[str] = "offer"
PAN_ENDPOINT: Final[str] = "/api/v1/pan/pan-comprehensive"
SCORE_ENDPOINT: Final[str] = "/api/v1/credit-report-experian/score"
REPORT_ENDPOINT: Final[str] = "/api/v1/credit-report-experian/fetch-report"
BUREAU_ENDPOINTS: Final[frozenset[str]] = frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT})
PAN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

AdmissionMode = Literal["cache_hit", "new", "refresh"]

# LC-facing one-liners for refusal LcViews (policy-owned so wording passes the no-leak scrubber;
# no bureau words, no standalone 300-899 numbers). output/lc_view.py consumes this map verbatim.
REFUSAL_MESSAGES: Final[dict[RefusalCode, str]] = {
    "CONSENT_MISSING": "Record the learner's verbal consent before running a financing check.",
    "DAILY_CAP_REACHED": "Daily quick-check limit reached for today. Please try again tomorrow.",
    "STAGE_TOO_EARLY": "Full checks unlock once the opportunity reaches the offer stage.",
    "OPPORTUNITY_ALREADY_CHECKED": "A full check already exists for this opportunity. It can be re-run after the current result expires.",
    "CACHE_FRESH_NO_REFRESH": "This learner was checked recently and the result is still fresh. Refresh unlocks after it expires.",
    "LIVE_CALLS_DISABLED": "Financing checks are temporarily unavailable. Please try again shortly or raise it with the platform team.",
}

# ---- PAN hashing (called in the API layer, main.py, right after request validation;
#      the raw PAN lives only in request scope / CheckState.pan for SurePass calls, never in any table column) ----
def normalize_pan(pan: str) -> str:
    """strip() then upper(); raises ValueError if the result does not match PAN_RE."""
    p = pan.strip().upper()
    if PAN_RE.fullmatch(p) is None:
        raise ValueError("invalid PAN format")
    return p

def hash_pan(pan: str, salt: str) -> str:
    """sha256 over UTF-8 of (salt + normalize_pan(pan)); returns 64-char lowercase hex.
    salt = Settings.pan_hash_salt from env PAN_HASH_SALT (required, non-empty; config.py fails fast).
    Raises ValueError on empty salt or invalid PAN. Deterministic across processes -> cache sharing works."""
    if not salt:
        raise ValueError("PAN_HASH_SALT must be non-empty")
    return hashlib.sha256((salt + normalize_pan(pan)).encode("utf-8")).hexdigest()

# ---- stage gate ----
def stage_allows_full_check(stage: str | None) -> bool:
    """True iff normalized stage is a known stage at or beyond FULL_CHECK_MIN_STAGE.
    None/unknown stages -> False (defensive: refuse STAGE_TOO_EARLY; API enum should prevent)."""
    if stage is None:
        return False
    s = stage.strip().lower()
    if s not in STAGE_ORDER:
        return False
    return STAGE_ORDER.index(s) >= STAGE_ORDER.index(FULL_CHECK_MIN_STAGE)

# ---- cache freshness ----
def is_cache_fresh(row: CacheRow | None, now: datetime) -> bool:
    """Fresh iff row exists AND row.expires_at > now AND derived.schema_version == DERIVED_SCHEMA_VERSION.
    A version mismatch is treated as stale (forces a billed re-pull after a derived-shape change)."""
    if row is None:
        return False
    if row.expires_at <= now:
        return False
    ver = row.derived.get("schema_version")
    return isinstance(ver, int) and ver == DERIVED_SCHEMA_VERSION

# ---- admission decision ----
@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Exactly one of (admission, refusal_code) is non-None."""
    admission: AdmissionMode | None
    refusal_code: RefusalCode | None
    billed_expected: bool            # a billed SurePass call will be needed (new/refresh); ground-truth `billed` is recomputed at persist
    counts_against_cap: bool         # quick+new only; copied to CheckState.counts_against_cap
    cache_row: CacheRow | None       # row for (pan_hash, kind) regardless of freshness; on cache_hit the nodes reuse row.derived without re-query
    cache_fresh: bool
    quick_checks_used_today: int | None   # populated on quick paths that evaluated the cap (incl. refusals); None for full
    refresh_available_at: datetime | None # cache_row.expires_at when a row exists, else None

def _refused(code: RefusalCode, *, cache_row: CacheRow | None, refresh_available_at: datetime | None,
             cache_fresh: bool = False, used: int | None = None) -> AdmissionDecision:
    return AdmissionDecision(admission=None, refusal_code=code, billed_expected=False, counts_against_cap=False,
                             cache_row=cache_row, cache_fresh=cache_fresh,
                             quick_checks_used_today=used, refresh_available_at=refresh_available_at)

# ---- QUICK admission (PRD 6.1, 6.3, 6.4, 6.5; evaluation order is NORMATIVE, first match wins) ----
# | # | consent | cache(pan,'quick') | refresh | used<5 | live  | admission | refusal_code            | billed_expected | counts_against_cap |
# |Q1 | False   | any                | any     |  -     |  -    |    -      | CONSENT_MISSING         | False           | False |
# |Q2 | True    | fresh              | False   |  -     |  -    | cache_hit |            -            | False           | False |   <- no cap check, no live check: serves even when SUREPASS_LIVE=false
# |Q3 | True    | fresh              | True    |  -     |  -    |    -      | CACHE_FRESH_NO_REFRESH  | False           | False |   <- PRD 6.5: refresh before TTL refused
# |Q4 | True    | stale              | True    |  -     | True  | refresh   |            -            | True            | False |   <- billed but cap-exempt (PRD 6.5)
# |Q5 | True    | stale              | True    |  -     | False |    -      | LIVE_CALLS_DISABLED     | False           | False |
# |Q6 | True    | stale              | False   | True   | True  | new       |            -            | True            | True  |   <- expired cache without refresh flag = ordinary new check
# |Q7 | True    | stale              | False   | False  |  -    |    -      | DAILY_CAP_REACHED       | False           | False |
# |Q8 | True    | stale              | False   | True   | False |    -      | LIVE_CALLS_DISABLED     | False           | False |
# |Q9 | True    | none               | any     | True   | True  | new       |            -            | True            | True  |   <- refresh=true with no cache row is treated as new (cap applies)
# |Q10| True    | none               | any     | False  |  -    |    -      | DAILY_CAP_REACHED       | False           | False |
# |Q11| True    | none               | any     | True   | False |    -      | LIVE_CALLS_DISABLED     | False           | False |
# used<5 means repo.quick_checks_used_today(lc_id) < QUICK_DAILY_CAP; cap is evaluated BEFORE the kill switch,
# so a capped LC sees DAILY_CAP_REACHED even when live calls are off.
def admit_quick_check(repo: Repo, *, lc_id: str, pan_hash: str, consent: bool, refresh: bool,
                      surepass_live: bool, now: datetime) -> AdmissionDecision:
    row = repo.get_cache(pan_hash, "quick")
    avail = row.expires_at if row is not None else None
    if consent is not True:                                   # 1. consent
        return _refused("CONSENT_MISSING", cache_row=row, refresh_available_at=avail)
    fresh = is_cache_fresh(row, now)                          # 2. cache lookup
    if fresh and not refresh:
        return AdmissionDecision("cache_hit", None, False, False, row, True, None, avail)
    if fresh and refresh:                                     # 3. refresh semantics
        return _refused("CACHE_FRESH_NO_REFRESH", cache_row=row, refresh_available_at=avail, cache_fresh=True)
    stale = row is not None                                   # here: row absent or expired/version-stale
    mode: Literal["new", "refresh"] = "refresh" if (stale and refresh) else "new"
    counts = mode == "new"
    used = repo.quick_checks_used_today(lc_id)
    if counts and used >= QUICK_DAILY_CAP:                    # 4. daily cap (new only; refresh exempt)
        return _refused("DAILY_CAP_REACHED", cache_row=row, refresh_available_at=avail, used=used)
    if not surepass_live:                                     # 5. kill switch (only when a billed call is needed)
        return _refused("LIVE_CALLS_DISABLED", cache_row=row, refresh_available_at=avail, used=used)
    return AdmissionDecision(mode, None, True, counts, row, False, used, avail)

# ---- FULL admission (PRD 6.1, 6.4, 6.5, 6.8; order NORMATIVE, first match wins) ----
# prior        = latest checks row with kind='full' AND status='ok' for this opportunity (any pan_hash)
# prior_fresh  = is_cache_fresh(pan_cache(prior.pan_hash,'full'))     (missing row => stale => re-runnable)
# cur          = pan_cache(request.pan_hash,'full')
# | # | consent | stage>=offer | prior | prior_fresh | cur     | refresh | live  | admission | refusal_code                | counts_against_cap |
# |F1 | False   | any          |  -    |  -          |  -      |  -      |  -    |    -      | CONSENT_MISSING             | False |
# |F2 | True    | False/unknown|  -    |  -          |  -      |  -      |  -    |    -      | STAGE_TOO_EARLY             | False |
# |F3 | True    | True         | yes   | True        | not fresh| any    |  -    |    -      | OPPORTUNITY_ALREADY_CHECKED | False |  <- only when a BILLED pull would be needed
# |F4 | True    | True         | any*  | any*        | fresh   | False   |  -    | cache_hit |            -                | False |  <- test 8: 2nd call on same opportunity = cache hit (free)
# |F5 | True    | True         | any*  | any*        | fresh   | True    |  -    |    -      | CACHE_FRESH_NO_REFRESH      | False |
# |F6 | True    | True         | no or stale | -     | stale   | True    | True  | refresh   |            -                | False |
# |F7 | True    | True         | no or stale | -     | stale   | True    | False |    -      | LIVE_CALLS_DISABLED         | False |
# |F8 | True    | True         | no or stale | -     | stale/none | False (or none+True) | True  | new |      -           | False |  <- refresh with no row = new; 6.8 re-run allowed when stale
# |F9 | True    | True         | no or stale | -     | stale/none | False (or none+True) | False |  -  | LIVE_CALLS_DISABLED | False |
# (* F4/F5 are reachable with prior present because F3 requires cur NOT fresh; when the PANs are equal,
#    prior_fresh == cur_fresh (same row), so F3 can never fire for the same PAN.)
# billed_expected is True exactly on F6/F8; counts_against_cap is False on EVERY full row (cap is quick-only, PRD 6.3).
def admit_full_check(repo: Repo, *, lc_id: str, pan_hash: str, opportunity_id: str, opportunity_stage: str | None,
                     consent: bool, refresh: bool, surepass_live: bool, now: datetime) -> AdmissionDecision:
    row = repo.get_cache(pan_hash, "full")
    avail = row.expires_at if row is not None else None
    if consent is not True:                                   # 1. consent
        return _refused("CONSENT_MISSING", cache_row=row, refresh_available_at=avail)
    if not stage_allows_full_check(opportunity_stage):        # 2. stage gate
        return _refused("STAGE_TOO_EARLY", cache_row=row, refresh_available_at=avail)
    fresh = is_cache_fresh(row, now)
    if not fresh:                                             # 3. opportunity-already-checked (only if billing would be needed)
        prior = repo.latest_full_check_for_opportunity(opportunity_id)
        if prior is not None:
            prior_row = row if prior.pan_hash == pan_hash else repo.get_cache(prior.pan_hash, "full")
            if is_cache_fresh(prior_row, now):
                return _refused("OPPORTUNITY_ALREADY_CHECKED", cache_row=row, refresh_available_at=avail)
    if fresh and not refresh:                                 # 4. cache lookup
        return AdmissionDecision("cache_hit", None, False, False, row, True, None, avail)
    if fresh and refresh:                                     # 5. refresh semantics
        return _refused("CACHE_FRESH_NO_REFRESH", cache_row=row, refresh_available_at=avail, cache_fresh=True)
    stale = row is not None
    mode: Literal["new", "refresh"] = "refresh" if (stale and refresh) else "new"
    if not surepass_live:                                     # 6. kill switch (no cap step for full)
        return _refused("LIVE_CALLS_DISABLED", cache_row=row, refresh_available_at=avail)
    return AdmissionDecision(mode, None, True, False, row, False, None, avail)

# ---- persist-time formulas (policy-owned; graph persist node calls these) ----
def compute_billed(calls: list[SurePassCallRow]) -> bool:
    """Ground truth from the audit log: any SurePass call attempt made for this check (PRD 7.3)."""
    return len(calls) > 0

def compute_bureau_called(calls: list[SurePassCallRow]) -> bool:
    return any(c.endpoint in BUREAU_ENDPOINTS for c in calls)

def compute_counted_against_cap(*, counts_against_cap: bool, bureau_called: bool, pan_invalid: bool) -> bool:
    """PRD 7.3 persist contract, verbatim: counts_against_cap AND bureau_called AND NOT pan_invalid."""
    return counts_against_cap and bureau_called and not pan_invalid

def quick_checks_remaining(used_before_this_check: int, counted_this_check: bool) -> int:
    """For LcView.quick_checks_remaining_today. Call repo.quick_checks_used_today BEFORE the final
    update_check flips this run's counted_against_cap, then add this run explicitly; clamp at 0."""
    return max(0, QUICK_DAILY_CAP - used_before_this_check - (1 if counted_this_check else 0))

def cache_expiry(fetched_at: datetime) -> datetime:
    """expires_at for new cache rows: fetched_at + CACHE_TTL (pull nodes use this when calling put_cache)."""
    return fetched_at + CACHE_TTL

def refresh_available_at(cache_row: CacheRow | None) -> datetime | None:
    """LcView.refresh_available_at = expires_at of the (possibly just-written) cache row; persist re-reads
    repo.get_cache(pan_hash, kind) after the run so new/refresh runs report the NEW expiry."""
    return cache_row.expires_at if cache_row is not None else None


### eligibility_agent/db/repo.py
_Lowest layer: shared Literal types, frozen dataclass rows, the Repo Protocol (structural typing so SupabaseRepo and MemoryRepo both satisfy it under mypy --strict), SupabaseRepo mapped to supabase-py v2, and MemoryRepo with injectable clock and exact IST-day cap semantics. db/ imports nothing from the rest of the package._

from __future__ import annotations
import copy, uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast
from zoneinfo import ZoneInfo
if TYPE_CHECKING:
    from supabase import Client  # constructed in config/main; repo never builds it

CheckKind = Literal["quick", "full"]
# 'running' (inserted at admission) and 'error' (set by the API-layer exception handler) are INTERNAL
# statuses; LcView.status only ever shows ok | pan_invalid | refused (PRD 9.3).
CheckStatus = Literal["running", "ok", "pan_invalid", "refused", "error"]
RefusalCode = Literal["CONSENT_MISSING", "DAILY_CAP_REACHED", "STAGE_TOO_EARLY",
                      "OPPORTUNITY_ALREADY_CHECKED", "CACHE_FRESH_NO_REFRESH", "LIVE_CALLS_DISABLED"]
_IST = ZoneInfo("Asia/Kolkata")

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def new_id() -> str:
    return str(uuid.uuid4())

@dataclass(frozen=True, slots=True)
class ConsentRow:
    id: str; lc_id: str; lead_id: str; opportunity_id: str | None
    pan_hash: str; channel: str; recorded_at: datetime        # channel == "verbal" in v1; recorded_at aware UTC

@dataclass(frozen=True, slots=True)
class CacheRow:
    pan_hash: str; kind: CheckKind
    payload: dict[str, Any]   # raw SurePass envelopes, audit only: NEVER into LcView, internal, or any LLM message
    derived: dict[str, Any]   # see 'pan_cache.derived contract' in schema.sql design; includes schema_version
    is_ntc: bool; fetched_at: datetime; expires_at: datetime

@dataclass(frozen=True, slots=True)
class CheckRow:
    id: str; kind: CheckKind; lc_id: str; lead_id: str; opportunity_id: str | None; pan_hash: str
    status: CheckStatus; refusal_code: RefusalCode | None
    billed: bool; counted_against_cap: bool; is_refresh: bool
    consent_id: str | None; lc_view: dict[str, Any] | None; internal: dict[str, Any] | None
    langsmith_run_id: str | None; created_at: datetime

class CheckPatch(TypedDict, total=False):
    """Partial update: a key present (even with value None for nullable columns) is written; absent keys untouched."""
    status: CheckStatus
    refusal_code: RefusalCode | None
    billed: bool
    counted_against_cap: bool
    lc_view: dict[str, Any] | None
    internal: dict[str, Any] | None
    langsmith_run_id: str | None

@dataclass(frozen=True, slots=True)
class SurePassCallRow:
    id: str; check_id: str
    endpoint: str                   # EXACT path constant from policy.guards (PAN_ENDPOINT / SCORE_ENDPOINT / REPORT_ENDPOINT)
    surepass_client_id: str | None  # data.client_id when the envelope carried one
    http_status: int | None         # None when transport failed before any HTTP response
    success: bool | None            # envelope `success`; None for the {errors, message} validation envelope
    message_code: str | None; called_at: datetime

@dataclass(frozen=True, slots=True)
class LenderRow:
    id: str; name: str; active: bool
    criteria: dict[str, Any]; hard_cutoffs: dict[str, Any]
    matched_pincodes: tuple[str, ...]   # sorted+deduped intersection of the lender's pincodes with the query input

class Repo(Protocol):
    def insert_consent(self, row: ConsentRow) -> None: ...
    def insert_check(self, row: CheckRow) -> None: ...                       # raises ValueError on duplicate id
    def update_check(self, check_id: str, patch: CheckPatch) -> None: ...    # raises ValueError if row missing
    def get_check(self, check_id: str) -> CheckRow | None: ...
    def latest_full_check_for_opportunity(self, opportunity_id: str) -> CheckRow | None: ...
        # newest by created_at (tie-break id desc) among kind='full' AND status='ok'; refused/pan_invalid/running/error rows never gate
    def quick_checks_used_today(self, lc_id: str) -> int: ...
        # COUNT of checks with kind='quick' AND counted_against_cap AND created_at on the CURRENT Asia/Kolkata calendar date
    def get_cache(self, pan_hash: str, kind: CheckKind) -> CacheRow | None: ...   # returns stale rows too; freshness judged by policy.guards.is_cache_fresh
    def put_cache(self, row: CacheRow) -> None: ...                               # UPSERT on (pan_hash, kind): refresh overwrites payload/derived/is_ntc/fetched_at/expires_at
    def log_surepass_call(self, row: SurePassCallRow) -> None: ...                # called for EVERY attempt, success or failure
    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]: ...  # ordered by called_at asc, id asc
    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]: ...
        # active lenders with >=1 matching lender_pincodes row; empty input -> [] WITHOUT querying; result ordered by lender id asc

class MemoryRepo:
    """Test/dev backend. Exact semantics: deep-copies on put AND get (no aliasing); injectable clock for the
    IST cap; NOT thread-safe (pytest is single-threaded). Extra non-Protocol helper: seed_lender."""
    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock
        self._consents: dict[str, ConsentRow] = {}
        self._checks: dict[str, CheckRow] = {}
        self._cache: dict[tuple[str, CheckKind], CacheRow] = {}
        self._calls: list[SurePassCallRow] = []
        self._lenders: dict[str, LenderRow] = {}          # matched_pincodes field unused here
        self._lender_pins: dict[str, frozenset[str]] = {}
    def seed_lender(self, *, id: str, name: str, active: bool, criteria: dict[str, Any],
                    hard_cutoffs: dict[str, Any], pincodes: Collection[str]) -> None:
        self._lenders[id] = LenderRow(id, name, active, dict(criteria), dict(hard_cutoffs), ())
        self._lender_pins[id] = frozenset(pincodes)
    def insert_consent(self, row: ConsentRow) -> None:
        if row.id in self._consents: raise ValueError(f"duplicate consent id {row.id}")
        self._consents[row.id] = row
    def insert_check(self, row: CheckRow) -> None:
        if row.id in self._checks: raise ValueError(f"duplicate check id {row.id}")
        self._checks[row.id] = copy.deepcopy(row)
    def update_check(self, check_id: str, patch: CheckPatch) -> None:
        cur = self._checks.get(check_id)
        if cur is None: raise ValueError(f"unknown check id {check_id}")
        self._checks[check_id] = replace(cur, **copy.deepcopy(dict(patch)))
    def get_check(self, check_id: str) -> CheckRow | None:
        row = self._checks.get(check_id)
        return copy.deepcopy(row) if row is not None else None
    def latest_full_check_for_opportunity(self, opportunity_id: str) -> CheckRow | None:
        cands = [c for c in self._checks.values()
                 if c.kind == "full" and c.status == "ok" and c.opportunity_id == opportunity_id]
        if not cands: return None
        return copy.deepcopy(max(cands, key=lambda c: (c.created_at, c.id)))
    def quick_checks_used_today(self, lc_id: str) -> int:
        today = self._clock().astimezone(_IST).date()
        return sum(1 for c in self._checks.values()
                   if c.kind == "quick" and c.lc_id == lc_id and c.counted_against_cap
                   and c.created_at.astimezone(_IST).date() == today)
    def get_cache(self, pan_hash: str, kind: CheckKind) -> CacheRow | None:
        row = self._cache.get((pan_hash, kind))
        return copy.deepcopy(row) if row is not None else None
    def put_cache(self, row: CacheRow) -> None:
        self._cache[(row.pan_hash, row.kind)] = copy.deepcopy(row)
    def log_surepass_call(self, row: SurePassCallRow) -> None:
        self._calls.append(row)
    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]:
        return sorted((c for c in self._calls if c.check_id == check_id), key=lambda c: (c.called_at, c.id))
    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]:
        pset = set(pincodes)
        if not pset: return []
        out: list[LenderRow] = []
        for lid in sorted(self._lenders):
            l = self._lenders[lid]
            matched = tuple(sorted(self._lender_pins.get(lid, frozenset()) & pset))
            if l.active and matched:
                out.append(replace(copy.deepcopy(l), matched_pincodes=matched))
        return out

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _ts(s: str) -> datetime:
    d = datetime.fromisoformat(s)                 # py3.12 parses offsets and 'Z'
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)

class SupabaseRepo:
    """Maps the Protocol 1:1 onto supabase-py v2 (sync client built with the SERVICE ROLE key; RLS is
    enabled with no policies, so only this key can touch the tables). All datetimes serialized with _iso."""
    def __init__(self, client: "Client") -> None:
        self._c = client
    def insert_consent(self, row: ConsentRow) -> None:
        self._c.table("consents").insert({"id": row.id, "lc_id": row.lc_id, "lead_id": row.lead_id,
            "opportunity_id": row.opportunity_id, "pan_hash": row.pan_hash, "channel": row.channel,
            "recorded_at": _iso(row.recorded_at)}).execute()
    def insert_check(self, row: CheckRow) -> None:
        self._c.table("checks").insert(_check_to_json(row)).execute()
    def update_check(self, check_id: str, patch: CheckPatch) -> None:
        self._c.table("checks").update(cast(dict[str, Any], dict(patch))).eq("id", check_id).execute()
    def get_check(self, check_id: str) -> CheckRow | None:
        res = self._c.table("checks").select("*").eq("id", check_id).limit(1).execute()
        return _check_from_json(res.data[0]) if res.data else None
    def latest_full_check_for_opportunity(self, opportunity_id: str) -> CheckRow | None:
        res = (self._c.table("checks").select("*").eq("opportunity_id", opportunity_id)
               .eq("kind", "full").eq("status", "ok").order("created_at", desc=True).limit(1).execute())
        return _check_from_json(res.data[0]) if res.data else None
    def quick_checks_used_today(self, lc_id: str) -> int:
        res = self._c.rpc("quick_checks_used_today", {"p_lc_id": lc_id}).execute()
        return int(cast(int, res.data))           # scalar-returning SQL function -> bare integer
    def get_cache(self, pan_hash: str, kind: CheckKind) -> CacheRow | None:
        res = (self._c.table("pan_cache").select("*").eq("pan_hash", pan_hash)
               .eq("kind", kind).limit(1).execute())
        if not res.data: return None
        r = res.data[0]
        return CacheRow(r["pan_hash"], cast(CheckKind, r["kind"]), r["payload"], r["derived"],
                        bool(r["is_ntc"]), _ts(r["fetched_at"]), _ts(r["expires_at"]))
    def put_cache(self, row: CacheRow) -> None:
        self._c.table("pan_cache").upsert({"pan_hash": row.pan_hash, "kind": row.kind,
            "payload": row.payload, "derived": row.derived, "is_ntc": row.is_ntc,
            "fetched_at": _iso(row.fetched_at), "expires_at": _iso(row.expires_at)},
            on_conflict="pan_hash,kind").execute()
    def log_surepass_call(self, row: SurePassCallRow) -> None:
        self._c.table("surepass_calls").insert({"id": row.id, "check_id": row.check_id,
            "endpoint": row.endpoint, "surepass_client_id": row.surepass_client_id,
            "http_status": row.http_status, "success": row.success, "message_code": row.message_code,
            "called_at": _iso(row.called_at)}).execute()
    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]:
        res = (self._c.table("surepass_calls").select("*").eq("check_id", check_id)
               .order("called_at", desc=False).order("id", desc=False).execute())
        return [SurePassCallRow(r["id"], r["check_id"], r["endpoint"], r["surepass_client_id"],
                r["http_status"], r["success"], r["message_code"], _ts(r["called_at"])) for r in res.data]
    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]:
        pins = sorted(set(pincodes))
        if not pins: return []
        lp = self._c.table("lender_pincodes").select("lender_id,pincode").in_("pincode", pins).execute()
        by_lender: dict[str, set[str]] = {}
        for r in lp.data: by_lender.setdefault(r["lender_id"], set()).add(r["pincode"])
        if not by_lender: return []
        ls = (self._c.table("lenders").select("*").in_("id", sorted(by_lender)).eq("active", True).execute())
        rows = [LenderRow(r["id"], r["name"], bool(r["active"]), r["criteria"], r["hard_cutoffs"],
                          tuple(sorted(by_lender[r["id"]]))) for r in ls.data]
        return sorted(rows, key=lambda l: l.id)

# module-private (de)serializers _check_to_json / _check_from_json: field-for-field dict mapping with
# _iso/_ts on created_at; jsonb columns pass through as dicts. Keep them pure so adapter tests cover them.


### eligibility_agent/db/schema.sql
_Full DDL for the six PRD section 10 tables, constraints, indexes (including the partial index behind the cap count), the quick_checks_used_today IST function, RLS lockdown, the pan_cache.derived/payload contract, a synthetic seed example, and the LangGraph checkpointer note._

-- eligibility_agent/db/schema.sql  (idempotent; run via Supabase SQL editor or psql "$SUPABASE_DB_URL" -f ...)
-- All ids are generated app-side (uuid4 strings) so MemoryRepo and SupabaseRepo behave identically.

create table if not exists lenders (
  id            text primary key,              -- e.g. 'L1'; admin-chosen slug
  name          text not null,
  active        boolean not null default true,
  criteria      jsonb  not null default '{}'::jsonb,   -- free-form text/fields the LLM reads at rank time
  hard_cutoffs  jsonb  not null default '{}'::jsonb    -- {min_band,max_foir,min_income,min_age,max_age_at_maturity,max_tenure_months,annual_rate_pct}; enforced in code only
);

create table if not exists lender_pincodes (
  lender_id text not null references lenders(id) on delete cascade,
  pincode   text not null check (pincode ~ '^[1-9][0-9]{5}$'),
  primary key (lender_id, pincode)
);
create index if not exists idx_lender_pincodes_pincode on lender_pincodes (pincode);  -- gate queries by pincode set

create table if not exists consents (
  id             uuid primary key,
  lc_id          text not null,
  lead_id        text not null,
  opportunity_id text,
  pan_hash       text not null,                -- salted sha256; the raw PAN is never stored in any column
  channel        text not null default 'verbal',
  recorded_at    timestamptz not null default now()
);
create index if not exists idx_consents_pan_hash on consents (pan_hash, recorded_at desc);

create table if not exists pan_cache (
  pan_hash   text not null,
  kind       text not null check (kind in ('quick','full')),
  payload    jsonb not null,   -- RAW SurePass envelopes, audit/reprocessing only; never returned by any API, never sent to the LLM
                               --   quick: {"pan_comprehensive": <envelope>, "experian_score": <envelope>}
                               --   full:  {"pan_comprehensive": <envelope>, "experian_report": <envelope>}
  derived    jsonb not null,   -- DERIVED CONTRACT (schema_version 1): enough to skip SurePass entirely on a hit;
                               -- gate/affordability/rank/finalize always RE-RUN per request (declared income/EMI vary; LcView is NOT cached).
                               --   quick: {"schema_version":1,
                               --           "identity":{"pan_valid":bool,"full_name":str,"gender":"male"|"female"|null,"dob":"YYYY-MM-DD"|null,
                               --                        "aadhaar_linked":bool|null,"pan_pincode":str|null,"surepass_client_id":str|null},
                               --           "score":{"score":int|null,"band":"<650"|"650-720"|"720-780"|"780+"|"NTC","report_date":str|null,"is_ntc":bool}}
                               --   full:  {"schema_version":1, "identity":{...same...},
                               --           "report":{"band":...,"score":int|null,"report_date":str|null,
                               --                      "existing_emi_reported":float,"existing_emi_floor":bool,
                               --                      "open_accounts_missing_emi":[{"balance":float,"tenure_months":int|null,"rate":float|null,"account_type":str}],
                               --                      "total_outstanding":float,"past_due_total":float,"max_dpd_24m":int,
                               --                      "written_off_or_settled":int,"enquiries_180d":int,
                               --                      "bureau_pincodes":[str],"is_ntc":bool}}
                               -- (identity/score/report are exactly Identity/BureauScore/BureauReport.model_dump(mode="json") from graph/state.py)
  is_ntc     boolean not null default false,   -- PRD 6.7: bureau no-hit on a valid PAN cached as NTC for 30 days
  fetched_at timestamptz not null,
  expires_at timestamptz not null,             -- fetched_at + 30 days (policy.guards.CACHE_TTL); freshness means expires_at > now()
  primary key (pan_hash, kind)                 -- quick and full cached separately (PRD 6.4)
);

create table if not exists checks (
  id                  uuid primary key,
  kind                text not null check (kind in ('quick','full')),
  lc_id               text not null,
  lead_id             text not null,
  opportunity_id      text,
  pan_hash            text not null,
  status              text not null check (status in ('running','ok','pan_invalid','refused','error')),
  refusal_code        text check (refusal_code in ('CONSENT_MISSING','DAILY_CAP_REACHED','STAGE_TOO_EARLY',
                        'OPPORTUNITY_ALREADY_CHECKED','CACHE_FRESH_NO_REFRESH','LIVE_CALLS_DISABLED')),
  billed              boolean not null default false,  -- ground truth at persist: any surepass_calls row exists for this check
  counted_against_cap boolean not null default false,  -- persist: counts_against_cap AND bureau_called AND NOT pan_invalid
  is_refresh          boolean not null default false,  -- admission == 'refresh' only (stale cache + refresh flag)
  consent_id          uuid references consents(id),
  lc_view             jsonb,                           -- exactly what the API returned; no score/bureau/raw fragments
  internal            jsonb,                           -- facts minus raw payload + node transcript; NEVER raw envelopes
  langsmith_run_id    text,
  created_at          timestamptz not null,            -- app-supplied (guard's `now`) so MemoryRepo and tests agree
  constraint refusal_code_iff_refused check ((refusal_code is not null) = (status = 'refused'))
);
-- Supports quick_checks_used_today: partial index over exactly the counted rows, range-scanned by day bounds.
create index if not exists idx_checks_cap on checks (lc_id, created_at) where kind = 'quick' and counted_against_cap;
-- Supports latest_full_check_for_opportunity (PRD 6.8 once-per-opportunity gate).
create index if not exists idx_checks_opp_full_ok on checks (opportunity_id, created_at desc) where kind = 'full' and status = 'ok';
create index if not exists idx_checks_pan_hash on checks (pan_hash, created_at desc);

create table if not exists surepass_calls (
  id                 uuid primary key,
  check_id           uuid not null references checks(id),   -- requires the checks row to be inserted at admission, BEFORE any call
  endpoint           text not null,                          -- '/api/v1/pan/pan-comprehensive' | '/api/v1/credit-report-experian/score' | '/api/v1/credit-report-experian/fetch-report'
  surepass_client_id text,
  http_status        integer,                                -- null if transport failed pre-response
  success            boolean,                                -- null for the {errors,message} validation envelope
  message_code       text,
  called_at          timestamptz not null default now()
);
create index if not exists idx_surepass_calls_check on surepass_calls (check_id, called_at);

-- 5 billed quick checks per LC per IST calendar day (PRD 6.3). Sargable: converts the IST day start back to a
-- timestamptz bound so idx_checks_cap is used. IST has no DST, but the round-trip is correct regardless.
create or replace function quick_checks_used_today(p_lc_id text)
returns integer
language sql
stable
as $$
  with ist_day as (
    select (date_trunc('day', now() at time zone 'Asia/Kolkata') at time zone 'Asia/Kolkata') as day_start
  )
  select count(*)::integer
  from checks c, ist_day d
  where c.lc_id = p_lc_id
    and c.kind = 'quick'
    and c.counted_against_cap
    and c.created_at >= d.day_start
    and c.created_at <  d.day_start + interval '1 day';
$$;

-- Service-role only (PRD 10): enable RLS with NO policies -> anon/authenticated see nothing; the
-- service-role key (used by SupabaseRepo) bypasses RLS.
alter table lenders         enable row level security;
alter table lender_pincodes enable row level security;
alter table consents        enable row level security;
alter table pan_cache       enable row level security;
alter table checks          enable row level security;
alter table surepass_calls  enable row level security;

-- LangGraph checkpointer tables (checkpoints, checkpoint_writes, checkpoint_blobs, migrations) are created by
-- langgraph-checkpoint-postgres via PostgresSaver.setup() against the direct Postgres DSN (env SUPABASE_DB_URL).
-- They are intentionally NOT defined here; do not hand-create or migrate them.

-- Synthetic seed example (documented in README; real criteria come from the admin UI writing to these tables):
-- insert into lenders (id, name, active, criteria, hard_cutoffs) values
--   ('L1', 'Lender One', true,
--    '{"summary": "Prefers salaried applicants; funds tech and management programmes; co-applicant accepted."}',
--    '{"min_band": "650-720", "max_foir": 0.55, "min_income": 25000, "min_age": 21,
--      "max_age_at_maturity": 60, "max_tenure_months": 48, "annual_rate_pct": 14}');
-- insert into lender_pincodes (lender_id, pincode) values ('L1', '201301'), ('L1', '560071');


## Interfaces provided
- **name**: policy.guards.admit_quick_check / admit_full_check | **definition**: def admit_quick_check(repo: Repo, *, lc_id: str, pan_hash: str, consent: bool, refresh: bool, surepass_live: bool, now: datetime) -> AdmissionDecision; def admit_full_check(repo: Repo, *, lc_id: str, pan_hash: str, opportunity_id: str, opportunity_stage: str | None, consent: bool, refresh: bool, surepass_live: bool, now: datetime) -> AdmissionDecision. Read-only (repo reads only); `now` is aware UTC captured once per run by the admission node; surepass_live comes from Settings. AdmissionDecision (frozen dataclass): admission: Literal['cache_hit','new','refresh'] | None; refusal_code: RefusalCode | None (exactly one of the two is set); billed_expected: bool; counts_against_cap: bool; cache_row: CacheRow | None (reuse derived on cache_hit, no re-query); cache_fresh: bool; quick_checks_used_today: int | None; refresh_available_at: datetime | None. Decision tables Q1-Q11 and F1-F9 in the file are normative.
- **name**: policy.guards constants and refusal messages | **definition**: QUICK_DAILY_CAP=5; CACHE_TTL=timedelta(days=30); IST=ZoneInfo('Asia/Kolkata'); DERIVED_SCHEMA_VERSION=1; STAGE_ORDER=('new','contacted','qualified','counselling','offer','registration','enrolled','won') with FULL_CHECK_MIN_STAGE='offer' (ASSUMPTION); PAN_ENDPOINT='/api/v1/pan/pan-comprehensive'; SCORE_ENDPOINT='/api/v1/credit-report-experian/score'; REPORT_ENDPOINT='/api/v1/credit-report-experian/fetch-report'; BUREAU_ENDPOINTS=frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT}); PAN_RE=re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$'); REFUSAL_MESSAGES: dict[RefusalCode, str] with the six LC-safe one-liners (output/lc_view.py uses them verbatim as the single talking_points entry on refusals).
- **name**: policy.guards.hash_pan | **definition**: def normalize_pan(pan: str) -> str (strip+upper, ValueError if not PAN_RE); def hash_pan(pan: str, salt: str) -> str = sha256(utf8(salt + normalized)).hexdigest(), 64 lowercase hex chars; main.py calls it with Settings.pan_hash_salt immediately after request validation.
- **name**: policy.guards persist-time formulas | **definition**: compute_billed(calls: list[SurePassCallRow]) -> bool (any logged attempt); compute_bureau_called(calls) -> bool (endpoint in BUREAU_ENDPOINTS); compute_counted_against_cap(*, counts_against_cap: bool, bureau_called: bool, pan_invalid: bool) -> bool (verbatim PRD 7.3 formula); quick_checks_remaining(used_before_this_check: int, counted_this_check: bool) -> int = max(0, 5 - used - (1 if counted else 0)); cache_expiry(fetched_at: datetime) -> datetime = fetched_at + CACHE_TTL (pull nodes MUST use this for put_cache); refresh_available_at(cache_row) -> datetime | None; is_cache_fresh(row, now) -> bool (expires_at > now AND derived.schema_version == 1); stage_allows_full_check(stage: str | None) -> bool.
- **name**: db.repo.Repo protocol + row types | **definition**: Literals: CheckKind, CheckStatus ('running','ok','pan_invalid','refused','error'), RefusalCode (6 codes). Frozen dataclasses: ConsentRow, CacheRow(pan_hash, kind, payload: dict, derived: dict, is_ntc, fetched_at, expires_at), CheckRow(16 fields), CheckPatch (TypedDict, total=False: status, refusal_code, billed, counted_against_cap, lc_view, internal, langsmith_run_id), SurePassCallRow, LenderRow(id, name, active, criteria: dict, hard_cutoffs: dict, matched_pincodes: tuple[str, ...]). Protocol methods: insert_consent(ConsentRow) -> None; insert_check(CheckRow) -> None; update_check(check_id: str, patch: CheckPatch) -> None; get_check(check_id) -> CheckRow | None; latest_full_check_for_opportunity(opportunity_id: str) -> CheckRow | None (kind='full' AND status='ok', newest created_at); quick_checks_used_today(lc_id: str) -> int (counted_against_cap rows on current IST date); get_cache(pan_hash: str, kind: CheckKind) -> CacheRow | None (stale rows returned too); put_cache(CacheRow) -> None (upsert on (pan_hash, kind)); log_surepass_call(SurePassCallRow) -> None; surepass_calls_for_check(check_id) -> list[SurePassCallRow]; lenders_for_pincodes(pincodes: Collection[str]) -> list[LenderRow] (active, >=1 match, ordered by id, [] on empty input). Helpers: utc_now() -> datetime, new_id() -> str. MemoryRepo(clock=utc_now) with seed_lender(...) test helper; SupabaseRepo(client) built with the service-role key.
- **name**: db/schema.sql artifacts | **definition**: Tables lenders, lender_pincodes (PK (lender_id, pincode), pincode CHECK ^[1-9][0-9]{5}$), consents, pan_cache (PK (pan_hash, kind)), checks (CHECKed status/refusal_code enums, refusal_code_iff_refused constraint), surepass_calls (FK check_id -> checks.id, so the checks row MUST be inserted at admission before any call). Indexes: idx_checks_cap partial (lc_id, created_at) WHERE kind='quick' AND counted_against_cap; idx_checks_opp_full_ok partial (opportunity_id, created_at desc) WHERE kind='full' AND status='ok'; idx_lender_pincodes_pincode; idx_checks_pan_hash; idx_surepass_calls_check. SQL function quick_checks_used_today(p_lc_id text) returns integer, STABLE, sargable via IST day-start bounds; invoked by SupabaseRepo as client.rpc('quick_checks_used_today', {'p_lc_id': lc_id}). RLS enabled on all six tables with no policies (service-role only). LangGraph checkpointer tables are created by PostgresSaver.setup() against SUPABASE_DB_URL and are NOT in this file.
- **name**: pan_cache derived/payload contract per kind | **definition**: derived.quick = {schema_version: 1, identity: Identity.model_dump(mode='json'), score: BureauScore.model_dump(mode='json')}; derived.full = {schema_version: 1, identity: ..., report: BureauReport.model_dump(mode='json')}. On cache_hit: validate_pan does Identity.model_validate(row.derived['identity']); pull_score/pull_report do BureauScore/BureauReport.model_validate(row.derived['score'|'report']); NO SurePass call, NO put_cache. On new/refresh: the pull node calls put_cache exactly once after successful adaptation with payload = {'pan_comprehensive': <raw envelope>, 'experian_score'|'experian_report': <raw envelope>}, derived as above, is_ntc = (band == 'NTC'), fetched_at = now, expires_at = cache_expiry(now). Declared income/EMI, gated lenders, affordability, ranking, final and lc_view are NEVER cached; payload is NEVER read back into state, LcView, internal, or any LLM message.
- **name**: Flag lifecycle contract (admission -> persist) | **definition**: admission node: (1) if consent is True: repo.insert_consent(ConsentRow(id=new_id(), ..., channel='verbal', recorded_at=now)); (2) decision = admit_*(); (3) repo.insert_check(CheckRow(id=check_id, status='running', refusal_code=None, billed=False, counted_against_cap=False, is_refresh=(decision.admission=='refresh'), consent_id, created_at=now)); state.admission/billed(=billed_expected)/counts_against_cap/refusal_code from decision. Tool wrapper: after EVERY SurePass attempt -> repo.log_surepass_call(row) then repo.update_check(check_id, {'billed': True}). persist_refusal: update_check(status='refused', refusal_code=state.refusal_code, lc_view=refusal view using REFUSAL_MESSAGES + quick_checks_remaining(used, False) + refresh_available_at). persist: calls = repo.surepass_calls_for_check(check_id); billed = compute_billed(calls); counted = compute_counted_against_cap(counts_against_cap=state.counts_against_cap, bureau_called=compute_bureau_called(calls), pan_invalid=(status=='pan_invalid')); used_before = repo.quick_checks_used_today(lc_id) BEFORE the final update; remaining = quick_checks_remaining(used_before, counted); row = repo.get_cache(pan_hash, kind) for from_cache(=admission=='cache_hit')/cached_at(=fetched_at)/refresh_available_at(=expires_at); single update_check(status, billed, counted_against_cap, lc_view, internal, langsmith_run_id).

## Interfaces consumed
- **name**: eligibility_agent/config.py (config section) | **expectation**: Settings with pan_hash_salt: str (env PAN_HASH_SALT, required non-empty, fail fast at startup), surepass_live: bool (True iff env SUREPASS_LIVE == 'true', exact string), supabase_url/supabase_service_role_key (service-role Client factory for SupabaseRepo), and SUPABASE_DB_URL (direct Postgres DSN for the LangGraph checkpointer and for applying schema.sql). A repo factory (config or main) chooses SupabaseRepo when Supabase env is present; tests construct MemoryRepo directly.
- **name**: graph/state.py models (graph section) | **expectation**: Identity, BureauScore, BureauReport are Pydantic models whose model_dump(mode='json') emits exactly the keys in my derived contract: Identity{pan_valid, full_name, gender, dob, aadhaar_linked, pan_pincode, surepass_client_id}; BureauScore{score, band, report_date, is_ntc}; BureauReport{band, score, report_date, existing_emi_reported, existing_emi_floor, open_accounts_missing_emi[], total_outstanding, past_due_total, max_dpd_24m, written_off_or_settled, enquiries_180d, bureau_pincodes, is_ntc}; band values exactly '<650'|'650-720'|'720-780'|'780+'|'NTC'. model_validate must round-trip these dicts (cache-hit load path). CheckState carries kind, check_id, lc_id, pan_hash, admission, billed, counts_against_cap, refusal_code as in PRD 7.2.
- **name**: graph/nodes.py wiring (graph section) | **expectation**: Nodes receive repo/surepass/LLM callables via the DI seam the graph section owns. The admission node implements my 3-step write pattern verbatim (consent insert iff consent True, then guard, then insert_check status='running' BEFORE any SurePass call - the surepass_calls FK depends on it). pull_score/pull_report call put_cache exactly once with both raw envelopes; the raw PAN Comprehensive envelope must travel from validate_pan to the pull node in an internal-only state slot that redact strips and that is never placed in an LLM message. persist/persist_refusal implement my flag-lifecycle contract and capture `now` once at admission for guard evaluation.
- **name**: tools/surepass.py (tools section) | **expectation**: The client logs EVERY call attempt (success, envelope failure, validation envelope, transport error) via repo.log_surepass_call with endpoint equal to my exact path constants (import PAN_ENDPOINT/SCORE_ENDPOINT/REPORT_ENDPOINT from policy.guards or match them byte-for-byte), surepass_client_id from data.client_id when present, http_status/success/message_code from the envelope (success=None for the {errors, message} validation shape), followed immediately by update_check({'billed': True}). It raises LiveCallsDisabled unless SUREPASS_LIVE is true - defense in depth behind my LIVE_CALLS_DISABLED guard, which must fire first.
- **name**: main.py API layer (API section) | **expectation**: Routes validate PAN (^[A-Z]{5}[0-9]{4}[A-Z]$) and mobile (^[6-9][0-9]{9}$), accept consent: bool (NOT Literal[True] - consent False must reach the graph so the CONSENT_MISSING refusal is persisted as a checks row per PRD 9.3), generate check_id = new_id(), compute pan_hash = hash_pan(pan, settings.pan_hash_salt) before invoking the graph, and never log the raw PAN. Full-check route validates opportunity_stage against STAGE_ORDER (ASSUMPTION enum). supabase-py is sync: run graph+repo work off the event loop (threadpool) - API section's call.
- **name**: output/lc_view.py (output section) | **expectation**: Refusal LcViews use REFUSAL_MESSAGES[code] as the single talking_points line; response fields quick_checks_remaining_today, from_cache, cached_at, refresh_available_at are taken from the persist-computed values per my lifecycle contract (not recomputed elsewhere); lc_view/internal dicts handed to update_check contain no raw payload fragments.
- **name**: evals/tests (evals + test sections) | **expectation**: policy_consistency evaluator and tests 1-9/12 read ground truth via repo.get_check and repo.surepass_calls_for_check on MemoryRepo, drive time with MemoryRepo(clock=...) plus the guards' now parameter (cap day-boundary and TTL tests need no monkeypatching), and seed lenders via MemoryRepo.seed_lender.

## Decisions
- **topic**: Admission evaluation order (normative) | **decision**: consent -> stage gate (full) -> opportunity-already-checked (full) -> cache lookup -> refresh semantics -> daily cap (quick, new-mode only) -> SUREPASS_LIVE kill switch; first match wins, each refusal maps to exactly one code; cache hits never reach the cap or kill-switch steps. | **rationale**: This is the order mandated by the section brief; putting the cap before the kill switch makes a capped LC see DAILY_CAP_REACHED deterministically even when live calls are off, and putting the kill switch last guarantees LIVE_CALLS_DISABLED fires only when a billed call would actually be needed (acceptance criterion: service serves cache hits and refusals with SUREPASS_LIVE unset).
- **topic**: Cache hit while SUREPASS_LIVE is false | **decision**: Served normally (admission=cache_hit, billed=false); the kill switch is only evaluated on paths that need a billed call. | **rationale**: PRD section 3 says live calls must be impossible by default, not that the service is unusable; a cache hit spends nothing, and acceptance criteria require the service to run with SUREPASS_LIVE unset.
- **topic**: refresh=true with no cache row | **decision**: Treated as a plain NEW check: admission='new', counts_against_cap=true (quick), is_refresh=false; the refresh flag is honored only when a stale cache row exists. | **rationale**: There is nothing to refresh, refusing would block a legitimate first check, and honoring cap-exemption would let a client bypass the 5/day cap by always sending refresh=true. Cap-safe and surfaced here for review.
- **topic**: Expired cache without refresh flag | **decision**: Plain NEW check: billed, counts against the quick cap, is_refresh=false; the stale row is overwritten by put_cache at pull time. | **rationale**: PRD 6.5's cap exemption is tied to the explicit Refresh action the UI unlocks after TTL; silently upgrading ordinary clicks to cap-exempt refreshes would hollow out the 6.3 cap. The LC who wants the exemption presses Refresh (sends refresh=true).
- **topic**: Refresh before TTL | **decision**: fresh row + refresh=true -> refuse CACHE_FRESH_NO_REFRESH (never silently serve the cache), with refresh_available_at = expires_at in the response. | **rationale**: PRD 6.5 FIXED: 'Refresh before TTL is refused'; returning the cache instead would make the code unreachable and hide the policy from the UI.
- **topic**: Once-per-opportunity gate formulation (PRD 6.8 vs test 8) | **decision**: Refuse OPPORTUNITY_ALREADY_CHECKED only when (a) a prior status='ok' full check exists for the opportunity, (b) the pan_cache row for THAT check's pan_hash+'full' is fresh, and (c) the current request's own full cache row is NOT fresh (i.e. a billed pull would be needed). Same-PAN repeats therefore fall through to cache_hit/CACHE_FRESH_NO_REFRESH; prior-stale always re-runs; a missing prior cache row counts as stale. | **rationale**: Reconciles PRD 6.8 ('once per opportunity, re-runnable only when the cached result is stale') with required test 8 ('second call on the same opportunity is a cache hit'): the invariant enforced is at most one BILLED full pull per opportunity per freshness window, while free cache hits are always allowed. Consequence (surfaced as a gap): a DIFFERENT PAN on the same opportunity within the prior result's TTL is refused in v1.
- **topic**: Quick cap vs same-day refresh interplay | **decision**: admission='refresh' sets counts_against_cap=false and skips the cap step entirely: an LC at 5/5 can still run a post-TTL refresh (billed), and refreshes never free or consume slots; an expired-cache check WITHOUT the flag at 5/5 is refused DAILY_CAP_REACHED. | **rationale**: PRD 6.5 FIXED: 'A refresh is billed but does not deduct from the LC's 5 daily checks'; the asymmetry is exactly what makes the Refresh button meaningful.
- **topic**: Flag lifecycle and ground truth for billed | **decision**: Admission inserts the checks row early (status='running', billed=false, counted_against_cap=false, is_refresh fixed at insert); every log_surepass_call is immediately followed by update_check({'billed': True}); persist recomputes billed = any surepass_calls row, bureau_called = any call to a credit-report-experian endpoint, counted_against_cap = counts_against_cap AND bureau_called AND NOT pan_invalid, then writes status/flags/lc_view/internal/langsmith_run_id in one final update. persist_refusal updates the same row to status='refused' + refusal_code with billed=false. | **rationale**: PRD 7.3 says persist 'updates' the row (so it must exist earlier), surepass_calls.check_id FK needs the parent row before the first call, 'billed (any SurePass call made)' demands the call log be the source of truth (so an invalid-PAN run is billed=true but counted=false, matching 6.6 and open item 3), and the immediate billed=true write keeps billing truthful if a run crashes mid-graph.
- **topic**: quick_checks_remaining_today and refresh_available_at computation | **decision**: persist computes used_before = repo.quick_checks_used_today(lc_id) BEFORE the final update (the current row is still counted_against_cap=false), then remaining = max(0, 5 - used_before - (1 if counted_this_check else 0)); refresh_available_at/cached_at/from_cache come from repo.get_cache(pan_hash, kind) re-read at persist time (so new/refresh runs report the NEW expiry) and from admission=='cache_hit'. Computed for full checks too (it is the LC's quick budget, unchanged by full runs). | **rationale**: One repo query, no double counting, correct on cache hits (test 2: remaining unchanged) and on refusals (DAILY_CAP_REACHED shows 0); re-reading the cache row avoids plumbing expiry through state.
- **topic**: Cap counting race | **decision**: Optimistic: in-flight runs are invisible to the cap (counted flips only at persist); two truly concurrent checks by one LC could both pass at 4/5. Accepted and documented; no pessimistic pre-count. | **rationale**: Pre-counting would transiently consume slots for runs that end pan_invalid (violating 6.6 'nothing is consumed') and corrupt quick_checks_remaining_today; one LC double-submitting within seconds is negligible at this traffic, and surepass_calls remains the audit trail.
- **topic**: Consent row semantics | **decision**: insert_consent runs only when request.consent is literally True, BEFORE the guard (PRD admission contract order); channel='verbal'; the row is kept even when the check is subsequently refused (cap/stage/etc.); CONSENT_MISSING inserts no consent row; checks.consent_id is nullable for that case. | **rationale**: PRD 6.1: our consent row is the only gate and must be recorded before any pull; recording consent that WAS given is correct audit even for refused checks, while fabricating a row for consent=false would be false evidence.
- **topic**: PAN hashing | **decision**: hash_pan(pan, salt) = sha256(utf8(salt + upper(strip(pan)))).hexdigest(), validated against ^[A-Z]{5}[0-9]{4}[A-Z]$ before hashing; salt from required env PAN_HASH_SALT; hashing happens in main.py route handlers right after Pydantic validation, so consents/checks/pan_cache only ever see pan_hash and the raw PAN exists only in request scope + CheckState.pan for SurePass bodies. Note: pan_cache.payload (raw envelopes) still contains the PAN echoed by SurePass - acceptable because payload is audit-only, service-role-only, and never surfaced. | **rationale**: Salted SHA-256 prevents rainbow-table reversal of the 10-char PAN space while staying deterministic across LCs and processes, which 6.4's shared PAN-level cache requires; normalization guarantees 'abcde1234f ' and 'ABCDE1234F' share one cache row.
- **topic**: No cross-kind cache reuse; invalid PANs not cached | **decision**: A full check never reuses a quick row's identity (and vice versa); PAN-invalid results are not cached at all. | **rationale**: PRD 6.4 caches quick and full separately and 7.3 says a non-cache-hit run calls PAN Comprehensive; an invalid PAN is corrected and resubmitted as a different pan_hash, so caching the failure buys nothing (re-billing a PAN call on an identical retry is accepted and flagged as a risk).
- **topic**: checks.status internal values | **decision**: Column allows 'running' (inserted at admission) and 'error' (set by the API exception handler on graph crashes) in addition to the PRD's ok|pan_invalid|refused; LcView.status never exposes the extra two; CHECK constraint ties refusal_code to status='refused' exactly. | **rationale**: Insert-early needs a pre-terminal state and crashed runs need a terminal marker for ops queries; the LcView contract (PRD 9.3) is unaffected.
- **topic**: derived schema_version | **decision**: derived carries schema_version=1; is_cache_fresh treats any version mismatch as stale, forcing a billed re-pull after an adapter shape change. | **rationale**: Safer than attempting to read old derived shapes with new code paths that drive lending decisions; the cost is a bounded re-billing wave only when the version is deliberately bumped.
- **topic**: Repo as Protocol with app-generated ids and injected clock | **decision**: Repo is a typing.Protocol; all ids are uuid4 strings generated app-side (new_id()); created_at/recorded_at/called_at are app-supplied aware-UTC datetimes; MemoryRepo takes clock: Callable[[], datetime] and computes the IST day with zoneinfo exactly like the SQL function; SupabaseRepo delegates the cap count to RPC quick_checks_used_today(p_lc_id). | **rationale**: Structural typing keeps db/ import-free of graph/policy (no cycles) and lets fakes satisfy mypy --strict; app-side ids and clocks make the required cap/TTL tests (2, 3, 6) deterministic without monkeypatching the database.

## Risks
- Cap race: two concurrent quick checks by the same LC can both pass admission at 4/5 because counted_against_cap flips only at persist; accepted for v1 traffic, documented in guards.py - a Postgres advisory lock per lc_id is the escalation path if it ever matters.
- Crashed runs leave checks rows in status='running'; billed stays truthful via the per-call update_check({'billed': True}), but counted_against_cap may undercount a run that made a bureau call then crashed before persist - surepass_calls is the reconciliation source; an ops sweeper is future work.
- Rotating PAN_HASH_SALT silently orphans pan_cache/consents/checks correlation (every PAN re-hashes): caches self-heal in 30 days but the once-per-opportunity gate and cap history weaken during the window; treat the salt as a long-lived secret, never rotate casually.
- pan_cache.payload necessarily contains the PAN echoed inside raw SurePass envelopes; it is service-role-only and never surfaced, but the section-13 code review must verify no path from payload (or the transient raw-envelope state slot) to LcView/internal/LLM messages.
- zoneinfo('Asia/Kolkata') needs tz data: fine on macOS/Linux, but add the tzdata wheel to pyproject for slim containers or CI images without a system database.
- supabase-py RPC returns the scalar as parsed JSON; the int(...) cast is defensive - if postgrest ever wraps it, quick_checks_used_today breaks loudly, which the MemoryRepo-parity unit test should catch in CI against a real instance if one is ever added.
- DERIVED_SCHEMA_VERSION bumps deliberately invalidate all cache rows, causing a one-time billed re-pull wave; coordinate bumps with the owner because they spend SurePass credits.
- The refusal_code_iff_refused CHECK constraint means update_check must set refusal_code and status='refused' in the SAME patch (my persist_refusal contract does); a future caller splitting them across two updates would hit a constraint violation.

## PRD gaps
- PRD 6.8 vs test 8 tension resolved by interpretation: 'once per opportunity' is enforced as 'at most one BILLED full pull per opportunity per freshness window' (cache hits always allowed). Corollary needing owner sign-off: a DIFFERENT PAN submitted for the same opportunity while the prior result is fresh is refused OPPORTUNITY_ALREADY_CHECKED in v1 - there is no PRD rule for correcting the PAN on an opportunity inside the 30-day window.
- The PRD never says whether an expired cache WITHOUT refresh=true should auto-upgrade to a cap-exempt refresh; I made it a cap-counted new check (cap-safe). If the owner intends 'any post-TTL re-check is cap-exempt', flip Q6 to counts_against_cap=false - one line in the decision table.
- HTTP status for refusals is unspecified (PRD 9.3 only defines the LcView body); I assume the API section returns 200 with status='refused' since refusals are persisted, successful policy outcomes - needs confirmation.
- LcView shows quick_checks_remaining_today in the quick example only; I compute it for full checks too (it is the LC's quick budget, unchanged by full runs) - confirm whether FulCRM wants it omitted on full responses.
- checks.status needs values beyond the PRD's ok|pan_invalid|refused for insert-early ('running') and crash handling ('error'); LcView never exposes them, but the column CHECK constraint extends the PRD's set.
- Whether a consents row should be written when the check is subsequently refused (cap, stage, kill switch): I record it whenever consent was actually given, since PRD 6.1 makes our row the audit of verbal consent - confirm this reading.
- Open item 3 implemented as stated (invalid-PAN run: billed=true from the PAN call, counted_against_cap=false) - the owner should still confirm SurePass's actual billing of PAN Comprehensive on invalid PANs.
- Stage enum and 'offer' unlock point are ASSUMPTION per PRD 9.2 / open item 1; STAGE_ORDER is a single constant to swap when FulCRM's enum arrives.
