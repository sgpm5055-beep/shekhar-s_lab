# Policy, caches, locks, quota, consent, persistence (PRD v1.1 sections 5, 6-partial, 11): policy/guards.py, db/repo.py, db/schema.sql, raw-response storage

## Summary
Redesigns the v1.0 policy/persistence dossier for PRD v1.1. Hashing moves from salted SHA-256 to HMAC-SHA256 (env IDENTITY_HMAC_KEY, domain-separated 'pan:'/'mobile:' messages, last-4 display columns, pinned test vectors). The single pan_cache(kind) table is replaced by THREE caches — pan_records 180d, score_cache 30d, report_cache 30d — that store NO derived facts: each row references the verbatim raw SurePass envelope in a private Supabase Storage bucket ('raw-responses', object key pan/{pan_hash}/{endpoint-slug}/{ts}-{call_id}.json, one object per call attempt, owned by the surepass_calls row), and every hit/recompute re-downloads raw and re-runs adapters+derivations+gate+rank+safety on current inputs and the current published lender version (v1.1 5.4; the v1.0 'derived jsonb' contract and DERIVED_SCHEMA_VERSION are VOID). A Report pull upserts BOTH report_cache and score_cache (source='report'). Admission is split into pure first-match-wins decision tables (quick QK1–QK9, full FL1–FL13, recompute R1–R5, attach A1–A5) evaluated over a pre-fetched AdmissionFacts snapshot, with the admission node executing writes in the PRD §7 order: consent row (object consent, covers score|report, per pull; cache hits surface the original consent reference) → PAN lock acquisition (pan_locks with kinds in_flight/timeout/daily_cap; atomic acquire via RPC; second request IN_FLIGHT, never queued) → stage gate → PAN-correction policy (opportunity_pan_events: one correction before a report exists, PAN_CHANGE_NEEDS_MANAGER + review row after) → windows → quota (5/LC/IST-day, lc_quota_overrides; counts ONLY a billed 200+success:true Score call on a PAN with no stored score row; expired-score refresh exempt) → 3-per-PAN-per-day hard cap (PAN_DAILY_CAP + day lock + review row) → timeout-reconciliation gate → WARN_NTC_RECENT → kill switch last. schema.sql is rewritten in full: versioned lenders with one-published partial unique index and known/missing/uncertain criteria fields, lender_pincodes per version, consents (5y TTL, lc_typed_name lives here only), the three caches, checks with the 7-status/13-code legality CHECK, surepass_calls (pan_hash + billable + latency_ms + storage_key + 180d purge columns), pan_locks, review_queue, lc_quota_overrides, opportunity_pan_events, IST SQL functions quick_checks_used_today/bureau_calls_today, acquire_pan_lock RPC, RLS lockdown, and the surviving checkpointer schema-isolation block. Repo protocol v2 (27 typed methods incl. RawStore, lock acquire/release, quota override, review queue, pan events, attach) with MemoryRepo+MemoryRawStore mirrors and SupabaseRepo/storage3 mappings (upload file_options {'content-type':'application/json','upsert':'true'}, download → bytes). counted_against_quota lifecycle: insert running/false/false at admission, one guarded terminal update at persist computing billed=any billable call, counted=quick AND no-stored-score AND score call 200+success:true. C1 (insert-early + terminal-guard), C15 (checkpointer), C16, kill-switch literal and no-network rulings survive and are referenced, not re-derived.

## Files
### eligibility_agent/policy/guards.py
_Deterministic policy layer for PRD v1.1 section 5: HMAC hashing + normalization, window/quota/cap constants, pure admission decision tables (quick/full/recompute/attach) over an AdmissionFacts snapshot, PAN-correction evaluation, persist-time billing/quota formulas, and the LcView date fields (checked_on, refresh_available_on, quick_checks_left_today). Pure: no network, no writes; the graph admission node executes the returned directives._

"""Layering: policy/ imports ONLY stdlib + type/row definitions from eligibility_agent.db.repo.
Graph nodes call these; the LLM never sees or influences them (PRD 5 preamble).
ADMISSION NODE WRITE ORDER (PRD section 7, executed by graph/nodes.py admission, NOT here):
  1. if consent.given: consent_id = new_id(); repo.insert_consent(ConsentRow(covers='score'|'report'))
     else -> refuse CONSENT_MISSING (no consent row fabricated).
  2. locks = repo.get_active_locks(pan_hash, now):
       any kind='timeout'  active -> refuse PAN_LOCKED
       any kind='daily_cap' active -> refuse PAN_DAILY_CAP
  3. acquired = repo.try_acquire_pan_lock(pan_hash, check_id, until=now+IN_FLIGHT_LOCK_TTL)
       not acquired -> refuse IN_FLIGHT (never queued, PRD 5.11)
  4. repo.insert_check(CheckRow(status='running', billed=False, counted_against_quota=False, ...))
     -- for EVERY outcome incl. refusals, BEFORE any SurePass call (surepass_calls FK; C1 SURVIVES)
  5. full only: stage gate, then PAN-correction policy via evaluate_pan_change(...)
  6. facts = AdmissionFacts(...) fetched via repo; decision = admit_quick(facts) | admit_full(facts, full_inputs)
  7. execute directives: decision.create_daily_cap_lock -> repo.create_pan_lock(kind='daily_cap',
     until=next IST midnight) + repo.insert_review_item(kind='pan_daily_cap');
     decision.record_pan_correction -> repo.insert_pan_event(event='correction').
  8. refusal -> persist_refusal -> END. persist/persist_refusal ALWAYS release the in-flight lock (finally).
"""
from __future__ import annotations
import hmac, hashlib, re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Literal
from zoneinfo import ZoneInfo
from eligibility_agent.db.repo import (Code, CheckKind, ConsentCovers, LockKind, PanRecordRow,
    ReportCacheRow, ReviewKind, ScoreCacheRow, OpportunityPanEvent)

# ---- constants (single source of truth; v1.1 FIXED) ----
PAN_WINDOW: Final[timedelta] = timedelta(days=180)        # PRD 5.3
SCORE_WINDOW: Final[timedelta] = timedelta(days=30)       # PRD 5.3
REPORT_WINDOW: Final[timedelta] = timedelta(days=30)      # PRD 5.3 / 5.9
QUICK_DAILY_QUOTA_DEFAULT: Final[int] = 5                 # PRD 5.5; per-LC override via lc_quota_overrides
PAN_DAILY_BUREAU_CAP: Final[int] = 3                      # PRD 5.7 (score+report together)
TIMEOUT_LOCK_TTL: Final[timedelta] = timedelta(minutes=10)  # PRD 5.12
IN_FLIGHT_LOCK_TTL: Final[timedelta] = timedelta(seconds=180)  # ASSUMPTION: 2x GRAPH_TIMEOUT_SECONDS(90) so a crashed run self-heals
RAW_RETENTION: Final[timedelta] = timedelta(days=180)     # PRD 11 provisional retention (raw)
CONSENT_RETENTION_SQL: Final[str] = "interval '5 years'"  # PRD 11 (consents); SQL-side only
IST: Final[ZoneInfo] = ZoneInfo("Asia/Kolkata")
PAN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
MOBILE_RE: Final[re.Pattern[str]] = re.compile(r"^[6-9][0-9]{9}$")
IDENTITY_HMAC_KEY_ENV: Final[str] = "IDENTITY_HMAC_KEY"   # config.py reads it; the key NEVER appears in any table/bucket/log
# Per-call timeouts (PRD 5.12) live in config.py as PAN_TIMEOUT_S=8.0, SCORE_TIMEOUT_S=12.0,
# REPORT_TIMEOUT_S=20.0, OPENAI_TIMEOUT_S=8.0 -- consumed by surepass/client.py and graph/llm; not here.

# STAGE GATE: OpportunityStage StrEnum + STAGE_ORDER + FULL_CHECK_MIN_STAGE='offer' and
# stage_allows_full_check(stage: str | None) -> bool SURVIVE VERBATIM from v1.0
# (docs/design/design-policy-guards-and-persistence-prd-s.md lines 34-35, 72-81; ruling C18).
# Unknown/None stage fails closed -> STAGE_TOO_EARLY. v1.1 adds: the caller-supplied stage string is
# stored on checks.opportunity_stage verbatim (trusted and logged, PRD 5.8).

# ---- normalization + HMAC-SHA256 hashing (PRD 5.2; replaces v1.0 salted-SHA256 / ruling C3) ----
def normalize_pan(pan: str) -> str:
    """strip() then upper(); ValueError unless PAN_RE matches."""

def normalize_mobile(mobile: str) -> str:
    """Remove spaces/hyphens/parens; strip a leading '+91'; strip leading '91' when 12 digits remain;
    strip leading '0' when 11 digits remain; ValueError unless MOBILE_RE matches the 10-digit result."""

def _hmac_hex(key: str, msg: bytes) -> str:
    if not key: raise ValueError("IDENTITY_HMAC_KEY must be non-empty")
    return hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()

def hash_pan(pan: str, *, key: str) -> str:
    """HMAC-SHA256 over b'pan:' + normalize_pan(pan); 64 lowercase hex. Domain-separated so one key
    serves both identifiers without cross-type collisions (strengthening, never weakening, of 5.2).
    PINNED TEST VECTOR: hash_pan(' abcde1234f ', key='test-hmac-key')
      == '7c0b91b00e0094dad4090227778b2996da1ec7c7e9c45d5b4ac4b0883e55edc0'"""
    return _hmac_hex(key, b"pan:" + normalize_pan(pan).encode("ascii"))

def hash_mobile(mobile: str, *, key: str) -> str:
    """PINNED: hash_mobile('+91 98765-43210', key='test-hmac-key')
      == '160381e0c64be1a856ab25167a3a22468ccce58e2641993fd4407183cd2c858a' (normalizes to '9876543210')"""
    return _hmac_hex(key, b"mobile:" + normalize_mobile(mobile).encode("ascii"))

def pan_last4(pan: str) -> str:   # 'ABCDE1234F' -> '234F' ; stored separately for display (PRD 5.2)
    return normalize_pan(pan)[-4:]
def mobile_last4(mobile: str) -> str:  # '9876543210' -> '3210'
    return normalize_mobile(mobile)[-4:]

# ---- freshness ----
def is_fresh(expires_at: datetime, now: datetime) -> bool:
    return expires_at > now
def pan_expiry(fetched_at: datetime) -> datetime:    return fetched_at + PAN_WINDOW
def score_expiry(fetched_at: datetime) -> datetime:  return fetched_at + SCORE_WINDOW
def report_expiry(fetched_at: datetime) -> datetime: return fetched_at + REPORT_WINDOW
def ist_date(dt: datetime) -> date:                  return dt.astimezone(IST).date()
# WORKED EXAMPLE (pins the PRD section 6 sample): report fetched 2026-10-10T14:00:00Z ->
#   expires_at = 2026-11-09T14:00:00Z; LcView.checked_on = '2026-10-10' (IST date of fetched_at);
#   LcView.refresh_available_on = '2026-11-09' (IST date of expires_at). PAN record fetched the same
#   moment expires 2027-04-08T14:00:00Z.

# ---- types ----
AdmissionMode = Literal["cache_hit", "new", "refresh"]
PullPlan = Literal["reuse", "call"]
REFUSAL_CODES: Final[frozenset[Code]] = frozenset({
    "CONSENT_MISSING", "QUOTA_EXHAUSTED", "PAN_DAILY_CAP", "STAGE_TOO_EARLY", "REPORT_WINDOW_OPEN",
    "IN_FLIGHT", "PAN_LOCKED", "WARN_NTC_RECENT", "PAN_CHANGE_NEEDS_MANAGER", "LIVE_CALLS_DISABLED"})
# Mid-run codes (never produced by admission): NO_PINCODE_ON_PAN (pincode_gate), SUREPASS_FAILURE
# (bureau_pull/pan node), OPENAI_FAILURE (rank node, after the single retry).

@dataclass(frozen=True, slots=True)
class AdmissionFacts:
    """Snapshot fetched by the admission node via Repo BEFORE calling the pure guard."""
    now: datetime                      # aware UTC, captured once (deps.now())
    pan_row: PanRecordRow | None       # repo.get_pan_record(pan_hash)
    score_row: ScoreCacheRow | None    # repo.get_score_cache(pan_hash)
    report_row: ReportCacheRow | None  # repo.get_report_cache(pan_hash); None supplied for quick
    quota_used: int                    # repo.quick_checks_used_today(lc_id)  (quick; 0 for full)
    quota_limit: int                   # repo.quota_limit(lc_id) = override or QUICK_DAILY_QUOTA_DEFAULT
    bureau_calls_today: int            # repo.bureau_calls_today(pan_hash)   (billable, IST day)
    reconciliation_pending: bool       # repo.unresolved_review_exists(pan_hash, 'timeout_reconciliation')
    surepass_live: bool                # settings.surepass_live (exact-literal 'true'; ruling survives)

@dataclass(frozen=True, slots=True)
class FullAdmissionInputs:
    refresh: bool
    acknowledge_ntc_warning: bool

@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Exactly one of (admission, code) is non-None. Directive fields are executed by the node."""
    admission: AdmissionMode | None
    code: Code | None                      # member of REFUSAL_CODES when set
    pan_plan: PullPlan                     # 'call' iff pan_row is None or expired (180d)
    bureau_plan: PullPlan                  # 'call' iff admitted mode is new/refresh
    counts_against_quota: bool             # quick AND bureau_plan=='call' AND score_row is None (PRD 5.6)
    is_refresh: bool                       # a non-fresh cache row existed and a pull was admitted
    quota_used_today: int | None           # echoed on every quick path (incl. refusals); None on full
    quota_limit: int | None
    create_daily_cap_lock: bool            # directive: daily_cap lock until next IST midnight + review row
    review_reason: ReviewKind | None       # 'pan_daily_cap' when PAN_DAILY_CAP fired
    ntc_found_on: date | None              # IST date of the NTC score row (WARN_NTC_RECENT message / label)
    checked_on: date | None                # cache-hit display: ist_date(bureau row.fetched_at)
    refresh_available_on: date | None      # ist_date(bureau row.expires_at) when a row exists
    original_consent_id: str | None        # cache hit: consent reference of the ORIGINAL pull (PRD 5.1)

# ---- QUICK admission (pure; consent/locks/in-flight already handled by the node per module docstring).
# NORMATIVE TABLE, first match wins. sf=score_row fresh, se=score_row expired, sn=score_row None,
# pf=pan_row fresh. Order: windows -> quota -> hard cap -> reconciliation -> kill switch
# (kill switch LAST so cache hits serve with SUREPASS_LIVE unset -- v1.0 decision SURVIVES).
# | #  | score      | pan | quota                | cap  | recon | live  | outcome |
# |QK1 | sf         | pf  |  -                   |  -   |  -    |  -    | cache_hit; pan_plan=reuse; bureau_plan=reuse; counts=F; checked_on/refresh_available_on/original_consent_id from score_row |
# |QK2 | sf         | !pf |  -                   |  -   |  -    | False | LIVE_CALLS_DISABLED  (identity re-verify needs a live Rs-1 PAN call) |
# |QK3 | sf         | !pf |  -                   |  -   |  -    | True  | cache_hit; pan_plan=call; bureau_plan=reuse; counts=F |
# |QK4 | sn         | any | used>=limit          |  -   |  -    |  -    | QUOTA_EXHAUSTED  (only the counting path can exhaust; PRD 5.5/5.6) |
# |QK5 | se         | any | (exempt: refresh)    |  -   |  -    |  -    | fall through (expired-score re-pull never counts, PRD 5.6) |
# |QK6 | sn|se      | any | pass                 | >=3  |  -    |  -    | PAN_DAILY_CAP; create_daily_cap_lock=T; review_reason='pan_daily_cap' (PRD 5.7) |
# |QK7 | sn|se      | any | pass                 | <3   | True  |  -    | PAN_LOCKED  (manual reconciliation required before any re-pull, PRD 5.12) |
# |QK8 | sn|se      | any | pass                 | <3   | False | False | LIVE_CALLS_DISABLED |
# |QK9 | sn|se      | any | pass                 | <3   | False | True  | admitted: mode='new' if sn else 'refresh'; bureau_plan=call; pan_plan=call if !pf else reuse; counts=(sn); is_refresh=(se) |
# NOTE: v1.1 has NO refresh flag on quick -- an expired stored score is auto-refreshed (billed, never counted).
def admit_quick(facts: AdmissionFacts) -> AdmissionDecision: ...

# ---- FULL admission. Node pre-steps before this guard: stage gate (STAGE_TOO_EARLY), PAN-correction
# policy (below). NORMATIVE TABLE: rf=report_row fresh, re=expired, rn=None; ntc_recent = score_row is
# not None AND score_row.is_ntc AND score_row.expires_at > now (PRD 5.14).
# | #   | report | refresh | pan | cap  | recon | ntc_recent & !ack | live  | outcome |
# |FL1  | rf     | False   | pf  |  -   |  -    |  -                |  -    | cache_hit; reuse both; counts=F; checked_on=ist_date(report_row.fetched_at) |
# |FL2  | rf     | True    | any |  -   |  -    |  -                |  -    | REPORT_WINDOW_OPEN; refresh_available_on=ist_date(report_row.expires_at)  (forced refresh inside a window: none, for anyone -- PRD 5.10) |
# |FL3  | rf     | False   | !pf |  -   |  -    |  -                | False | LIVE_CALLS_DISABLED |
# |FL4  | rf     | False   | !pf |  -   |  -    |  -                | True  | cache_hit; pan_plan=call; bureau_plan=reuse |
# |FL5  | re|rn  | any     | any | >=3  |  -    |  -                |  -    | PAN_DAILY_CAP + lock + review (PRD order: cap before NTC warning) |
# |FL6  | re|rn  | any     | any | <3   | True  |  -                |  -    | PAN_LOCKED (timeout reconciliation outstanding) |
# |FL7  | re|rn  | any     | any | <3   | False | True              |  -    | WARN_NTC_RECENT; ntc_found_on=ist_date(score_row.fetched_at) (PRD 5.14) |
# |FL8  | re|rn  | any     | any | <3   | False | False/acked       | False | LIVE_CALLS_DISABLED |
# |FL9  | re|rn  | any     | any | <3   | False | False/acked       | True  | admitted: mode='refresh' if re else 'new'; is_refresh=(re); bureau_plan=call; pan_plan per pan_row; counts=F ALWAYS (report pulls are outside any quota, PRD 5.9) |
# refresh=True with rn (nothing to refresh) falls into FL9 as mode='new' -- v1.0 decision SURVIVES.
def admit_full(facts: AdmissionFacts, full: FullAdmissionInputs) -> AdmissionDecision: ...

# ---- PAN correction policy (PRD 5.13; full check only; evaluated by the node BEFORE admit_full) ----
PanChangeOutcome = Literal["first_pan", "same_pan", "record_correction", "needs_manager"]
def evaluate_pan_change(events: list[OpportunityPanEvent], report_exists: bool,
                        request_pan_hash: str) -> PanChangeOutcome:
    """TABLE (first match wins):
    | events empty                                              -> 'first_pan'   (node records event='initial')
    | request_pan_hash == latest event's pan_hash               -> 'same_pan'
    | report_exists (any full check on this opportunity with status in ('ok','pending_verification'))
                                                               -> 'needs_manager' (refuse PAN_CHANGE_NEEDS_MANAGER + review row kind='pan_change_manager')
    | any event with event=='correction' already exists         -> 'needs_manager' (the ONE correction is spent)
    | else                                                      -> 'record_correction' (node writes event='correction'; re-verification is automatic because the new pan_hash has no pan_record; once-per-window is per pan_hash so the corrected PAN is a new pull -- nothing extra to do)
    report_exists definition is an ASSUMPTION ('pending_verification' counts: spend unknown means a report may exist)."""

# ---- recompute admission (POST /checks/{check_id}/recompute; PRD section 6) ----
RecomputeOutcome = Literal["ok", "not_found", "not_recomputable", "raw_unavailable"]
# | R1 | repo.get_check(check_id) is None                                   -> 'not_found' (API: 404) |
# | R2 | check.status not in {'ok'} and not (status=='service_failure' and code=='OPENAI_FAILURE')
#                                                                           -> 'not_recomputable' (API: 409; only runs that produced facts can re-derive) |
# | R3 | kind=='quick': pan_records row or score_cache row missing for check.pan_hash -> 'raw_unavailable' (API: 409; purged) |
# | R4 | kind=='full' : pan_records row or report_cache row missing          -> 'raw_unavailable' |
# | R5 | else 'ok': NO consent row, NO locks, NO quota, NO SurePass; freshness NOT required (stored raw,
#        labelled checked_on = original fetched_at); new checks row with recompute_of=check_id,
#        consent_id=original.consent_id, billed=False, counted_against_quota=False, from_cache=True;
#        reruns derive->gate->range->code_confidence->rank->safety->redact->persist on CURRENT published lenders. |
def admit_recompute(check_status: str, check_code: str | None, kind: CheckKind,
                    pan_row_present: bool, bureau_row_present: bool) -> RecomputeOutcome: ...

# ---- attach admission (POST /checks/{check_id}/attach; quick only) ----
AttachOutcome = Literal["ok", "not_found", "not_quick", "already_attached", "conflict"]
# | A1 | check missing                      -> 'not_found' (404) |
# | A2 | check.kind != 'quick'              -> 'not_quick' (409) |
# | A3 | check.contact_id == contact_id     -> 'already_attached' (idempotent 200) |
# | A4 | check.contact_id set and different -> 'conflict' (409) |
# | A5 | else 'ok': repo.attach_contact(check_id, contact_id) updates checks.contact_id AND
#        consents.contact_id for check.consent_id ('contact id when attached', PRD 11). |
def admit_attach(kind: CheckKind, existing_contact_id: str | None, contact_id: str) -> AttachOutcome: ...

# ---- persist-time formulas (graph persist node calls these; C1 single guarded terminal update SURVIVES) ----
def compute_billed(calls: "list[SurePassCallRow]") -> bool:
    """True iff any call row has billable=True (PAN Rs-1 calls included). billable is computed by the
    Repo sink: (http_status == 200 AND success is True) OR error_kind == 'timeout' (spend unknown ->
    conservatively billed; manual reconciliation corrects, PRD 5.12 / zero-double-billing metric)."""

def compute_counted_against_quota(*, kind: CheckKind, counts_against_quota: bool,
                                  calls: "list[SurePassCallRow]") -> bool:
    """PRD 5.6 verbatim: kind=='quick' AND counts_against_quota (admission: no stored score row existed)
    AND any call row with endpoint==SCORE_ENDPOINT and http_status==200 and success is True.
    Consequences pinned by tests: cache hit F; PAN invalid (bureau never called) F; service failure /
    timeout F (no 200+success row); NTC T (200+success with non-3-digit score); expired-score refresh F
    (admission already set counts_against_quota=False)."""

def quick_checks_left(limit: int, used_before: int, counted_this_check: bool) -> int:
    """max(0, limit - used_before - (1 if counted_this_check else 0)). used_before read BEFORE the final
    update flips this run's flag. WORKED: limit 5, used 3, counting run -> 1 ('1 of 5 checks left today');
    override limit 8, used 5, cache hit -> 3. Quick only; None on full (v1.0 C20 stance kept, flagged)."""

def next_ist_midnight(now: datetime) -> datetime:
    """daily_cap lock_until: (ist_date(now) + 1 day) at 00:00 IST converted to UTC (quota resets 00:00 IST)."""
# Endpoint constants PAN_ENDPOINT/SCORE_ENDPOINT/REPORT_ENDPOINT stay owned by surepass/client.py
# (binding ruling) and are imported here; BUREAU_ENDPOINTS = frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT}).


### eligibility_agent/db/repo.py
_Single registry of status/code/kind Literals, frozen row dataclasses, the RawStore protocol (Supabase Storage + in-memory fake), the Repo Protocol v2 (every method typed, pyright strict), MemoryRepo with injected clock and exact IST/lock semantics, and SupabaseRepo mappings including storage upload/download and the three RPCs._

from __future__ import annotations
# imports: copy, json, uuid, dataclasses, datetime, typing (Protocol, Literal, TypedDict, Any), zoneinfo
# supabase Client under TYPE_CHECKING only (single boundary; ruling survives).

# ---- Literals (db/repo.py is the registry; policy re-imports) ----
CheckKind = Literal["quick", "full"]
CheckStatus = Literal["running", "ok", "pan_not_verified", "refused", "pending_verification",
                      "service_failure", "error"]   # 'running'+'error' internal; LcView never shows them
Code = Literal["CONSENT_MISSING", "QUOTA_EXHAUSTED", "PAN_DAILY_CAP", "STAGE_TOO_EARLY",
               "REPORT_WINDOW_OPEN", "IN_FLIGHT", "PAN_LOCKED", "WARN_NTC_RECENT",
               "PAN_CHANGE_NEEDS_MANAGER", "NO_PINCODE_ON_PAN", "LIVE_CALLS_DISABLED",
               "SUREPASS_FAILURE", "OPENAI_FAILURE"]
ConsentCovers = Literal["score", "report"]
LockKind = Literal["in_flight", "timeout", "daily_cap"]
ScoreSource = Literal["score", "report"]
ErrorKind = Literal["timeout", "transport", "auth", "rate_limited", "upstream", "protocol", "validation"]
ReviewKind = Literal["pan_daily_cap", "timeout_reconciliation", "openai_failure",
                     "safety_demoted_all", "not_fundable_pattern", "pan_change_manager",
                     "model_review_flags"]
PanEventKind = Literal["initial", "correction", "manager_change"]
Clock = Callable[[], datetime]
def utc_now() -> datetime: ...
def new_id() -> str: ...   # uuid4 string; ALL ids app-generated (v1.0 decision survives)

# ---- STATUS x CODE LEGALITY (enforced by SQL CHECK + a unit-tested predicate here) ----
def code_legal(status: CheckStatus, code: Code | None) -> bool:
    """refused            -> code in the 10 REFUSAL codes (required)
       service_failure    -> code in {SUREPASS_FAILURE, OPENAI_FAILURE} (required)
       ok                 -> code is None or code == 'NO_PINCODE_ON_PAN' (band-read-only view)
       running|pan_not_verified|pending_verification|error -> code is None
       (OPENAI_FAILURE rows still carry an LcView with headline + band read + gated list, confidence low,
        review row kind='openai_failure' -- PRD section 10.)"""

# ---- frozen dataclass rows (slots=True on all) ----
@dataclass(frozen=True, slots=True)
class ConsentRow:
    id: str; lc_id: str; lead_id: str | None; opportunity_id: str | None; contact_id: str | None
    pan_hash: str; mobile_hash: str; lc_typed_name: str   # the LC-typed name lives ONLY here (PRD 6)
    wording_version: str; call_ref: str | None; ip: str | None
    covers: ConsentCovers; given_at: datetime; purge_after: datetime   # given_at + 5 years

@dataclass(frozen=True, slots=True)
class PanRecordRow:      # 180-day PAN Comprehensive cache; ONLY valid+person responses are cached
    pan_hash: str; pan_last4: str; storage_key: str; client_id: str | None
    fetched_at: datetime; expires_at: datetime; purge_after: datetime  # both = fetched_at + 180d

@dataclass(frozen=True, slots=True)
class ScoreCacheRow:     # 30-day Experian Score cache; ALSO refreshed by every successful Report pull
    pan_hash: str; storage_key: str; source: ScoreSource; is_ntc: bool; client_id: str | None
    consent_id: str | None   # consent of the ORIGINAL pull; cache hits surface this reference (PRD 5.1)
    fetched_at: datetime; expires_at: datetime  # fetched + 30d
    purge_after: datetime                       # fetched + 180d (raw retention outlives the window)

@dataclass(frozen=True, slots=True)
class ReportCacheRow:    # 30-day Experian Report cache, per PAN hash, shared across opportunities
    pan_hash: str; storage_key: str; is_ntc: bool; report_date: date | None  # CreditProfileHeader.ReportDate
    client_id: str | None; consent_id: str | None
    fetched_at: datetime; expires_at: datetime; purge_after: datetime

@dataclass(frozen=True, slots=True)
class CheckRow:          # 23 fields
    id: str; kind: CheckKind; lc_id: str; lead_id: str | None; opportunity_id: str | None
    opportunity_stage: str | None   # caller-supplied, trusted, LOGGED verbatim (PRD 5.8)
    contact_id: str | None
    pan_hash: str; pan_last4: str; mobile_hash: str | None; mobile_last4: str | None
    status: CheckStatus; code: Code | None
    billed: bool; counted_against_quota: bool; is_refresh: bool
    recompute_of: str | None; consent_id: str | None
    lc_view: dict[str, Any] | None; internal: dict[str, Any] | None
    langsmith_run_id: str | None; created_at: datetime

class CheckPatch(TypedDict, total=False):
    status: CheckStatus; code: Code | None; billed: bool; counted_against_quota: bool
    contact_id: str | None; lc_view: dict[str, Any] | None; internal: dict[str, Any] | None
    langsmith_run_id: str | None

@dataclass(frozen=True, slots=True)
class SurePassCallRow:
    id: str; check_id: str; pan_hash: str          # pan_hash denormalized for bureau_calls_today()
    endpoint: str; client_id: str | None
    http_status: int | None; success: bool | None; message_code: str | None
    latency_ms: int | None                          # PRD 11 log fields
    billable: bool                                  # (200 AND success) OR error_kind=='timeout'
    error_kind: ErrorKind | None
    storage_key: str | None                         # raw envelope object; None when no body arrived
    called_at: datetime; purge_after: datetime; purged_at: datetime | None

@dataclass(frozen=True, slots=True)
class PanLockRow:
    pan_hash: str; kind: LockKind; check_id: str | None
    locked_at: datetime; lock_until: datetime; released_at: datetime | None
    # active := released_at is None and lock_until > now

@dataclass(frozen=True, slots=True)
class ReviewRow:
    id: str; check_id: str | None; pan_hash: str; kind: ReviewKind; reason: str
    created_at: datetime; resolved_at: datetime | None; resolved_by: str | None

@dataclass(frozen=True, slots=True)
class OpportunityPanEvent:
    id: str; opportunity_id: str; pan_hash: str; pan_last4: str
    event: PanEventKind; actor_lc_id: str; check_id: str | None; created_at: datetime

@dataclass(frozen=True, slots=True)
class LenderRow:         # always the PUBLISHED version (matching reads published only, PRD 11)
    id: str; version: int; name: str; published: bool
    criteria: dict[str, Any]       # each field {'value':..., 'status':'known'|'missing'|'uncertain'}
    policy_notes: str
    matched_pincodes: tuple[str, ...]  # populated by lenders_for_pincodes, () from published_lenders

class RawUnavailable(Exception): ...   # storage object missing/purged

class RawStore(Protocol):
    def put_raw(self, key: str, envelope: dict[str, Any]) -> None: ...
    def get_raw(self, key: str) -> dict[str, Any]: ...        # raises RawUnavailable
    def delete_raw(self, keys: Sequence[str]) -> None: ...

def storage_key_for(pan_hash: str, endpoint: str, called_at: datetime, call_id: str) -> str:
    """'pan/{pan_hash}/{slug}/{called_at:%Y%m%dT%H%M%SZ}-{call_id}.json' with slug mapping
    pan-comprehensive | experian-score | experian-report. One object per HTTP attempt that returned a
    body: EVERY raw response is stored verbatim (PRD goal 4 / 11); opportunity linkage is relational via
    surepass_calls.check_id -> checks.opportunity_id."""

class MemoryRawStore:    # dict[str, dict]; deep-copies on put/get; .keys() helper for purge tests
class SupabaseRawStore:
    """SupabaseRawStore(client: Client, bucket: str = 'raw-responses') -- bucket is PRIVATE; Supabase
    Storage is AES-256 encrypted at rest (satisfies 'private encrypted bucket', ASSUMPTION surfaced).
    put_raw: client.storage.from_(bucket).upload(path=key,
        file=json.dumps(envelope, separators=(',', ':')).encode('utf-8'),
        file_options={'content-type': 'application/json', 'upsert': 'true'})   # verified supabase-py v2 API
    get_raw: data: bytes = client.storage.from_(bucket).download(key); json.loads(data)
             (StorageApiError / not-found -> raise RawUnavailable(key))
    delete_raw: client.storage.from_(bucket).remove(list(keys))
    Bucket creation is an ops step documented in README: client.storage.create_bucket('raw-responses',
    options={'public': False})."""

# ---- Repo Protocol v2 (structural; MemoryRepo and SupabaseRepo both satisfy it) ----
class Repo(Protocol):
    # checks (C1 lifecycle survives: insert-early, ONE guarded terminal update)
    def insert_check(self, row: CheckRow) -> None: ...                     # ValueError on duplicate id
    def update_check(self, check_id: str, patch: CheckPatch) -> None: ... # terminal patches apply ONLY when current status=='running' (SupabaseRepo adds .eq('status','running'); MemoryRepo mirrors); contact_id-only patches are exempt from the guard
    def get_check(self, check_id: str) -> CheckRow | None: ...
    def record_check_error(self, *, check_id: str, kind: CheckKind, lc_id: str, pan_hash: str,
                           pan_last4: str, message: str) -> None: ...     # update-if-running else insert status='error' (C1)
    def attach_contact(self, check_id: str, contact_id: str) -> None: ... # updates checks.contact_id + consents.contact_id of the check's consent_id
    # consents
    def insert_consent(self, row: ConsentRow) -> None: ...
    # caches (plain upserts; storage_key must equal an existing surepass_calls.storage_key)
    def get_pan_record(self, pan_hash: str) -> PanRecordRow | None: ...   # stale rows returned too; freshness judged by policy
    def put_pan_record(self, row: PanRecordRow) -> None: ...              # upsert on pan_hash
    def get_score_cache(self, pan_hash: str) -> ScoreCacheRow | None: ...
    def put_score_cache(self, row: ScoreCacheRow) -> None: ...            # upsert; called by pull_score AND by pull_report (source='report')
    def get_report_cache(self, pan_hash: str) -> ReportCacheRow | None: ...
    def put_report_cache(self, row: ReportCacheRow) -> None: ...
    def fetch_raw(self, storage_key: str) -> dict[str, Any]: ...          # delegate to RawStore; cache hits + recompute re-run adapters on this (PRD 5.4)
    # quota / caps (IST day)
    def quick_checks_used_today(self, lc_id: str) -> int: ...             # RPC quick_checks_used_today
    def quota_limit(self, lc_id: str) -> int: ...                         # lc_quota_overrides row or QUICK_DAILY_QUOTA_DEFAULT
    def bureau_calls_today(self, pan_hash: str) -> int: ...               # RPC bureau_calls_today (billable, score+report)
    # locks
    def try_acquire_pan_lock(self, pan_hash: str, check_id: str, until: datetime) -> bool: ...  # RPC acquire_pan_lock; atomic; False -> IN_FLIGHT
    def release_pan_lock(self, pan_hash: str, check_id: str) -> None: ... # only the holder releases; idempotent
    def create_pan_lock(self, pan_hash: str, kind: LockKind, check_id: str | None, until: datetime) -> None: ...  # timeout/daily_cap upsert
    def get_active_locks(self, pan_hash: str) -> list[PanLockRow]: ...    # released_at IS NULL AND lock_until > now
    # review queue
    def insert_review_item(self, row: ReviewRow) -> None: ...
    def unresolved_review_exists(self, pan_hash: str, kind: ReviewKind) -> bool: ...
    # PAN correction (PRD 5.13)
    def pan_events_for_opportunity(self, opportunity_id: str) -> list[OpportunityPanEvent]: ...  # created_at asc
    def insert_pan_event(self, row: OpportunityPanEvent) -> None: ...
    def report_exists_for_opportunity(self, opportunity_id: str) -> bool: ...  # any kind='full' check, status in ('ok','pending_verification')
    # surepass audit sink (THE CallLogSink; the surepass client calls this, one row per HTTP attempt)
    def log_surepass_call(self, *, check_id: str, pan_hash: str, endpoint: str, client_id: str | None,
                          http_status: int | None, success: bool | None, message_code: str | None,
                          latency_ms: int | None, error_kind: ErrorKind | None,
                          raw_envelope: dict[str, Any] | None) -> str | None: ...
        # repo: id=new_id(); called_at=clock(); billable=(http_status==200 and success is True) or
        # error_kind=='timeout'; storage_key=storage_key_for(...) and RawStore.put_raw when
        # raw_envelope is not None (EVERY body persisted verbatim, success or failure);
        # purge_after=called_at+180d; insert row; return storage_key. Zero rows for LiveCallsDisabled
        # (no I/O happened -- C5 survives).
    def surepass_calls_for_check(self, check_id: str) -> list[SurePassCallRow]: ...  # called_at asc, id asc
    # lenders (published version only)
    def published_lenders(self) -> list[LenderRow]: ...                   # ordered by id
    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]: ...
        # published-only, >=1 matching pincode on the SAME (id,version); matched_pincodes sorted; [] on empty input without querying

class MemoryRepo:
    """MemoryRepo(clock: Clock = utc_now, raw_store: MemoryRawStore | None = None). Deep-copies on
    put AND get; IST day via zoneinfo exactly mirroring the SQL functions; NOT thread-safe.
    Lock semantics mirror the RPC: try_acquire inserts/steals (pan_hash,'in_flight') iff absent,
    released, or lock_until <= clock(); release sets released_at only when check_id matches.
    quota_limit: dict override or 5. bureau_calls_today: count of billable bureau-endpoint call rows
    whose called_at falls on the current IST date. unresolved_review_exists: any row with matching
    pan_hash+kind and resolved_at None. Extra non-Protocol helpers: seed_lender(id, version, name,
    published, criteria, policy_notes, pincodes), set_quota_override(lc_id, n), resolve_review(id),
    list_checks(), list_calls(check_id=None), active_locks_snapshot() -- tests and evals use these."""

class SupabaseRepo:
    """SupabaseRepo(client: Client, raw_store: SupabaseRawStore, clock: Clock = utc_now). Sync client
    with the SERVICE ROLE key (RLS enabled, no policies). Mappings: table inserts/updates/upserts
    field-for-field with _iso/_ts datetime (de)serializers (survive from v1.0); terminal update_check
    adds .eq('status','running'); quick_checks_used_today -> client.rpc('quick_checks_used_today',
    {'p_lc_id': lc_id}); bureau_calls_today -> rpc('bureau_calls_today', {'p_pan_hash': pan_hash});
    try_acquire_pan_lock -> rpc('acquire_pan_lock', {'p_pan_hash':..., 'p_check_id':...,
    'p_until': _iso(until)}) -> bool; quota_limit -> select lc_quota_overrides limit 1 else default;
    lenders_for_pincodes -> two queries: lender_pincodes .in_('pincode', pins) then lenders
    .in_ on (id) .eq('published', True), joining versions app-side on (lender_id, version);
    published_lenders -> .eq('published', True).order('id')."""


### eligibility_agent/db/schema.sql
_Full idempotent DDL for PRD v1.1 section 11: versioned lenders + pincodes, consents (5y TTL), the three caches, checks (7-status/13-code legality CHECK), surepass_calls (raw TTL columns), pan_locks, review_queue, lc_quota_overrides, opportunity_pan_events, IST SQL functions, the atomic lock RPC, RLS lockdown, and the surviving checkpointer schema-isolation block._

-- eligibility_agent/db/schema.sql (idempotent; psql "$SUPABASE_DB_URL" -f ... or Supabase SQL editor)
-- All ids app-generated uuid4 strings. All timestamps timestamptz (aware UTC).

create table if not exists lenders (
  id           text not null,                 -- stable slug, e.g. 'L1'
  version      integer not null,
  name         text not null,
  published    boolean not null default false,
  criteria     jsonb not null default '{}'::jsonb,
  -- criteria contract: every field is {"value": <typed>, "status": "known"|"missing"|"uncertain"}.
  -- REQUIRED keys (lender excluded from matching unless ALL have status='known' -- PRD 11):
  --   min_score_floor (int, entered as an Experian score; open item), ntc_accepted (bool),
  --   foir_limit (number 0..1), min_income (int), min_age (int), max_age_at_maturity (int),
  --   rate_min_pct (number), rate_max_pct (number), tenure_min_months (int), tenure_max_months (int),
  --   loan_min (int), loan_max (int).
  -- OPTIONAL: min_income_by_employment (object), state_list (text[]).
  policy_notes text not null default '',
  created_by   text,
  created_at   timestamptz not null default now(),
  primary key (id, version)
);
create unique index if not exists idx_lenders_one_published on lenders (id) where published;  -- at most ONE published version per lender

create table if not exists lender_pincodes (
  lender_id text not null,
  version   integer not null,
  pincode   text not null check (pincode ~ '^[1-9][0-9]{5}$'),
  primary key (lender_id, version, pincode),
  foreign key (lender_id, version) references lenders(id, version) on delete cascade
);
create index if not exists idx_lender_pincodes_pincode on lender_pincodes (pincode);

create table if not exists consents (                     -- one row PER PULL (PRD 5.1 / 11); 5-year retention
  id              uuid primary key,
  lc_id           text not null,
  lead_id         text,
  opportunity_id  text,
  contact_id      text,                                   -- set by /attach
  pan_hash        text not null,
  mobile_hash     text not null,
  lc_typed_name   text not null,                          -- the LC-typed name lives ONLY here (PRD 6)
  wording_version text not null,                          -- 'v2-quick' | 'v2-full' ...
  call_ref        text,                                   -- SuperConnect call reference when applicable
  ip              text,
  covers          text not null check (covers in ('score','report')),
  given_at        timestamptz not null,
  purge_after     timestamptz not null                    -- given_at + interval '5 years'
);
create index if not exists idx_consents_pan on consents (pan_hash, given_at desc);

create table if not exists pan_records (                  -- PAN Comprehensive cache, 180 d; valid+person only
  pan_hash    text primary key,
  pan_last4   text not null,
  storage_key text not null,                              -- raw envelope object in the 'raw-responses' bucket
  client_id   text,
  fetched_at  timestamptz not null,
  expires_at  timestamptz not null,                       -- fetched_at + interval '180 days'
  purge_after timestamptz not null                        -- fetched_at + interval '180 days'
);

create table if not exists score_cache (                  -- Experian Score cache, 30 d; refreshed by Report pulls too
  pan_hash    text primary key,
  storage_key text not null,                              -- score envelope OR report envelope (source='report')
  source      text not null check (source in ('score','report')),
  is_ntc      boolean not null default false,             -- 200 + success:true + credit_score not a 3-digit string (PRD 8.1)
  client_id   text,
  consent_id  uuid references consents(id),               -- ORIGINAL pull's consent; cache hits surface it
  fetched_at  timestamptz not null,
  expires_at  timestamptz not null,                       -- fetched_at + interval '30 days'
  purge_after timestamptz not null                        -- fetched_at + interval '180 days' (raw retention)
);

create table if not exists report_cache (                 -- Experian Report cache, 30 d, per PAN, shared across opportunities
  pan_hash    text primary key,
  storage_key text not null,
  is_ntc      boolean not null default false,             -- 'no file' stored as NTC for 30 days (PRD 10)
  report_date date,                                       -- CreditProfileHeader.ReportDate; null on NTC
  client_id   text,
  consent_id  uuid references consents(id),
  fetched_at  timestamptz not null,
  expires_at  timestamptz not null,
  purge_after timestamptz not null
);

create table if not exists checks (
  id                    uuid primary key,
  kind                  text not null check (kind in ('quick','full')),
  lc_id                 text not null,
  lead_id               text,
  opportunity_id        text,
  opportunity_stage     text,                             -- caller-supplied, trusted, logged (PRD 5.8)
  contact_id            text,                             -- /attach (quick) or from the opportunity (full)
  pan_hash              text not null,
  pan_last4             text not null,
  mobile_hash           text,
  mobile_last4          text,
  status                text not null check (status in ('running','ok','pan_not_verified','refused','pending_verification','service_failure','error')),
  code                  text check (code in ('CONSENT_MISSING','QUOTA_EXHAUSTED','PAN_DAILY_CAP','STAGE_TOO_EARLY','REPORT_WINDOW_OPEN','IN_FLIGHT','PAN_LOCKED','WARN_NTC_RECENT','PAN_CHANGE_NEEDS_MANAGER','NO_PINCODE_ON_PAN','LIVE_CALLS_DISABLED','SUREPASS_FAILURE','OPENAI_FAILURE')),
  billed                boolean not null default false,   -- persist ground truth: any billable surepass_calls row
  counted_against_quota boolean not null default false,   -- persist: quick AND no-stored-score AND 200+success Score call
  is_refresh            boolean not null default false,
  recompute_of          uuid references checks(id),       -- /recompute lineage
  consent_id            uuid references consents(id),
  lc_view               jsonb,                            -- exactly what the API returned; no score/bureau/raw fragments
  internal              jsonb,                            -- facts minus raw payloads; carries original_consent_id on hits
  langsmith_run_id      text,
  created_at            timestamptz not null,
  constraint code_legality check (
       (status = 'refused'         and code in ('CONSENT_MISSING','QUOTA_EXHAUSTED','PAN_DAILY_CAP','STAGE_TOO_EARLY','REPORT_WINDOW_OPEN','IN_FLIGHT','PAN_LOCKED','WARN_NTC_RECENT','PAN_CHANGE_NEEDS_MANAGER','LIVE_CALLS_DISABLED'))
    or (status = 'service_failure' and code in ('SUREPASS_FAILURE','OPENAI_FAILURE'))
    or (status = 'ok'              and (code is null or code = 'NO_PINCODE_ON_PAN'))
    or (status in ('running','pan_not_verified','pending_verification','error') and code is null))
);
create index if not exists idx_checks_quota on checks (lc_id, created_at) where kind = 'quick' and counted_against_quota;
create index if not exists idx_checks_pan on checks (pan_hash, created_at desc);
create index if not exists idx_checks_opp_full on checks (opportunity_id, created_at desc) where kind = 'full';

create table if not exists surepass_calls (               -- one row per HTTP attempt; client-owned logging via Repo sink
  id          uuid primary key,
  check_id    uuid not null references checks(id),        -- checks row inserted at admission BEFORE any call (C1)
  pan_hash    text not null,                              -- denormalized for bureau_calls_today()
  endpoint    text not null,                              -- '/api/v1/pan/pan-comprehensive' | '/api/v1/credit-report-experian/score' | '/api/v1/credit-report-experian/fetch-report'
  client_id   text,
  http_status integer,                                    -- null when transport failed / timed out pre-response
  success     boolean,                                    -- null for the {errors,message} validation envelope
  message_code text,
  latency_ms  integer,                                    -- PRD 11 log fields: client_id, endpoint, status, message_code, latency
  billable    boolean not null default false,             -- (200 AND success) OR error_kind='timeout' (spend unknown, conservative)
  error_kind  text check (error_kind in ('timeout','transport','auth','rate_limited','upstream','protocol','validation')),
  storage_key text,                                       -- raw envelope object; null when no body arrived; THE object owner for purge
  called_at   timestamptz not null,
  purge_after timestamptz not null,                       -- called_at + interval '180 days'
  purged_at   timestamptz                                 -- set by the purge job after object deletion
);
create index if not exists idx_calls_check on surepass_calls (check_id, called_at);
create index if not exists idx_calls_pan_billable on surepass_calls (pan_hash, called_at) where billable;
create index if not exists idx_calls_purge on surepass_calls (purge_after) where purged_at is null;

create table if not exists pan_locks (                    -- PRD 5.7 / 5.11 / 5.12
  pan_hash    text not null,
  kind        text not null check (kind in ('in_flight','timeout','daily_cap')),
  check_id    uuid references checks(id),
  locked_at   timestamptz not null default now(),
  lock_until  timestamptz not null,                       -- in_flight: now+180s; timeout: now+10min; daily_cap: next IST midnight
  released_at timestamptz,
  primary key (pan_hash, kind)                            -- active := released_at is null AND lock_until > now()
);

create table if not exists review_queue (
  id          uuid primary key,
  check_id    uuid references checks(id),
  pan_hash    text not null,
  kind        text not null check (kind in ('pan_daily_cap','timeout_reconciliation','openai_failure','safety_demoted_all','not_fundable_pattern','pan_change_manager','model_review_flags')),
  reason      text not null,
  created_at  timestamptz not null,
  resolved_at timestamptz,                                -- unresolved 'timeout_reconciliation' blocks billed re-pulls (PRD 5.12)
  resolved_by text
);
create index if not exists idx_review_open on review_queue (pan_hash, kind) where resolved_at is null;

create table if not exists lc_quota_overrides (           -- PRD 5.5: manager-configurable per LC
  lc_id       text primary key,
  daily_quota integer not null check (daily_quota >= 0),
  set_by      text not null,
  updated_at  timestamptz not null default now()
);

create table if not exists opportunity_pan_events (       -- PRD 5.13 PAN-correction audit
  id             uuid primary key,
  opportunity_id text not null,
  pan_hash       text not null,
  pan_last4      text not null,
  event          text not null check (event in ('initial','correction','manager_change')),
  actor_lc_id    text not null,
  check_id       uuid references checks(id),
  created_at     timestamptz not null
);
create index if not exists idx_opp_pan on opportunity_pan_events (opportunity_id, created_at);

-- ---- SQL functions (both IST-day; sargable via day-start bounds; IST has no DST) ----
create or replace function quick_checks_used_today(p_lc_id text) returns integer
language sql stable as $$
  with d as (select (date_trunc('day', now() at time zone 'Asia/Kolkata') at time zone 'Asia/Kolkata') as s)
  select count(*)::integer from checks c, d
  where c.lc_id = p_lc_id and c.kind = 'quick' and c.counted_against_quota
    and c.created_at >= d.s and c.created_at < d.s + interval '1 day';
$$;

create or replace function bureau_calls_today(p_pan_hash text) returns integer
language sql stable as $$
  with d as (select (date_trunc('day', now() at time zone 'Asia/Kolkata') at time zone 'Asia/Kolkata') as s)
  select count(*)::integer from surepass_calls sc, d
  where sc.pan_hash = p_pan_hash and sc.billable
    and sc.endpoint in ('/api/v1/credit-report-experian/score','/api/v1/credit-report-experian/fetch-report')
    and sc.called_at >= d.s and sc.called_at < d.s + interval '1 day';
$$;

create or replace function acquire_pan_lock(p_pan_hash text, p_check_id uuid, p_until timestamptz)
returns boolean language plpgsql as $$
declare n integer;
begin
  insert into pan_locks (pan_hash, kind, check_id, locked_at, lock_until, released_at)
  values (p_pan_hash, 'in_flight', p_check_id, now(), p_until, null)
  on conflict (pan_hash, kind) do update
    set check_id = excluded.check_id, locked_at = now(), lock_until = excluded.lock_until, released_at = null
    where pan_locks.released_at is not null or pan_locks.lock_until <= now();
  get diagnostics n = row_count;
  return n = 1;          -- false -> the lock is genuinely held -> IN_FLIGHT, never queued (PRD 5.11)
end $$;

-- ---- RLS lockdown: service-role only (no policies) ----
alter table lenders enable row level security;            alter table lender_pincodes enable row level security;
alter table consents enable row level security;           alter table pan_records enable row level security;
alter table score_cache enable row level security;        alter table report_cache enable row level security;
alter table checks enable row level security;             alter table surepass_calls enable row level security;
alter table pan_locks enable row level security;          alter table review_queue enable row level security;
alter table lc_quota_overrides enable row level security; alter table opportunity_pan_events enable row level security;

-- ---- LangGraph checkpointer isolation (SURVIVES VERBATIM from the v1.0 dossier / ruling C15 +
-- audit-technical-feasibility fix: server-side, pooler-agnostic) ----
create schema if not exists checkpointer;
-- Ops (documented, run once with a real password): create role checkpointer_svc login password '...';
-- DATABASE_URL (the checkpointer DSN) authenticates as checkpointer_svc via the SESSION pooler (:5432), never :6543.
-- alter role checkpointer_svc in database postgres set search_path = checkpointer;
-- grant usage, create on schema checkpointer to checkpointer_svc;
-- Checkpointer tables are created by PostgresSaver.setup(); after setup the app asserts
-- to_regclass('public.checkpoints') IS NULL and fails fast (checkpoint blobs carry raw PAN/payloads;
-- public schema is PostgREST-exposed).

-- Storage (ops, documented in README): private bucket 'raw-responses' (Supabase Storage, encrypted at
-- rest); created via storage API create_bucket('raw-responses', {'public': false}). Purge job
-- (scripts/purge_raw.py, day one, cron): (1) select id, storage_key from surepass_calls where
-- purge_after < now() and purged_at is null -> RawStore.delete_raw(batch of keys) -> set purged_at;
-- (2) delete cache rows (pan_records/score_cache/report_cache) where purge_after < now();
-- (3) delete from consents where purge_after < now().  Cache rows always point at keys owned by a
-- surepass_calls row, so object deletion is driven solely by surepass_calls.


### scripts/purge_raw.py
_Day-one TTL purge job (PRD 11: schema carries a TTL column and a purge job from day one). Runs against Repo+RawStore; idempotent; dry-run mode._

"""usage: python -m scripts.purge_raw [--dry-run]
Sequence (matches the schema.sql comment):
  1. rows = select surepass_calls where purge_after < now() and purged_at is null (batches of 100)
     keys = [r.storage_key for r in rows if r.storage_key]; raw_store.delete_raw(keys);
     update purged_at = now() (the call-log METADATA row is kept forever -- only the object dies).
  2. delete pan_records / score_cache / report_cache rows where purge_after < now()
     (their storage keys were owned by step-1 rows; RawUnavailable on a later stale read is expected
      and handled by admit_recompute -> 'raw_unavailable').
  3. delete consents where purge_after < now()  (5-year retention).
--dry-run prints counts and the first 10 keys per step without deleting. Exit 0 on success.
Typed against the Repo protocol so MemoryRepo-based unit tests cover all three steps."""


### eligibility_agent/graph/nodes.py (admission/persist contract only -- node internals owned by the graph section)
_Records the exact write-pattern my section requires from the admission, bureau_pull and persist nodes so the flag lifecycle, locks and caches stay correct. Not my file; the contract is binding on the graph section._

ADMISSION NODE: executes the 8-step order in policy/guards.py module docstring (consent row -> lock
checks -> atomic in-flight acquire -> insert_check(status='running') -> stage gate -> PAN-change policy ->
AdmissionFacts fetch -> admit_quick/admit_full -> directives). Refusals carry decision fields into state
for persist_refusal.

PAN NODE (pan_comprehensive): pan_plan=='reuse' -> raw = repo.fetch_raw(pan_row.storage_key); adapter
re-parses (PRD 5.4). pan_plan=='call' -> client call (8 s timeout); on 200+success+status=='valid'+
category=='person' -> repo.put_pan_record(PanRecordRow(storage_key=key returned by the sink,
fetched_at=now, expires_at=pan_expiry(now), purge_after=now+RAW_RETENTION)). Invalid/not-person -> NOT
cached, status='pan_not_verified', nothing marked checked, quota untouched (PRD section 7).

BUREAU NODE (bureau_pull): bureau_plan=='reuse' -> fetch_raw from score_row/report_row; adapters re-run.
'call' quick -> Score (12 s): on 200+success -> put_score_cache(source='score', is_ntc per 8.1,
consent_id=this run's consent, expires=score_expiry(now)). 'call' full -> Report (20 s): on 200+success
-> put_report_cache(...) AND put_score_cache(source='report', storage_key=SAME report object,
is_ntc=report NTC) -- the Report pull refreshes the score cache (PRD 5.3). TIMEOUT -> repo.create_pan_lock
(kind='timeout', until=now+TIMEOUT_LOCK_TTL) + insert_review_item(kind='timeout_reconciliation') ->
status='pending_verification' (code null) -> persist -> END. Other service failure ->
status='service_failure', code='SUREPASS_FAILURE', quota untouched.

PERSIST / PERSIST_REFUSAL: ONE guarded terminal update_check (C1 survives): status, code,
billed=compute_billed(calls), counted_against_quota=compute_counted_against_quota(...), lc_view,
internal (carries original_consent_id, admission mode, storage refs NEVER), langsmith_run_id. Reads
used_before=repo.quick_checks_used_today BEFORE the update; LcView.quick_checks_left_today =
quick_checks_left(limit, used_before, counted). ALWAYS (finally) repo.release_pan_lock(pan_hash,
check_id) -- including refusal, pan_not_verified, pending_verification, service_failure and the API
error boundary. RECOMPUTE entry: skips consent/locks/quota entirely per admit_recompute; new checks row
recompute_of=original.

## Interfaces provided
- **name**: policy.guards hashing + normalization | **definition**: normalize_pan(str)->str; normalize_mobile(str)->str; hash_pan(pan,*,key)->str (HMAC-SHA256 of b'pan:'+normalized, 64 hex); hash_mobile(mobile,*,key)->str (b'mobile:' prefix); pan_last4/mobile_last4. Pinned vectors with key 'test-hmac-key': ' abcde1234f ' -> 7c0b91b00e0094dad4090227778b2996da1ec7c7e9c45d5b4ac4b0883e55edc0; '+91 98765-43210' -> 160381e0c64be1a856ab25167a3a22468ccce58e2641993fd4407183cd2c858a. main.py computes both hashes + last4 right after request validation; raw PAN lives only in request scope/CheckState for SurePass bodies.
- **name**: policy.guards constants | **definition**: PAN_WINDOW=180d, SCORE_WINDOW=30d, REPORT_WINDOW=30d, QUICK_DAILY_QUOTA_DEFAULT=5, PAN_DAILY_BUREAU_CAP=3, TIMEOUT_LOCK_TTL=10min, IN_FLIGHT_LOCK_TTL=180s (ASSUMPTION), RAW_RETENTION=180d, IST, PAN_RE, MOBILE_RE, REFUSAL_CODES (10 admission codes), BUREAU_ENDPOINTS. Per-call timeouts (8/12/20/8 s) are config.py's to own.
- **name**: policy.guards admission API | **definition**: AdmissionFacts / FullAdmissionInputs / AdmissionDecision frozen dataclasses; admit_quick(facts)->AdmissionDecision; admit_full(facts, full)->AdmissionDecision; evaluate_pan_change(events, report_exists, request_pan_hash)->PanChangeOutcome; admit_recompute(...)->RecomputeOutcome; admit_attach(...)->AttachOutcome; stage_allows_full_check(stage)->bool. Decision tables QK1-QK9, FL1-FL9, R1-R5, A1-A5 are NORMATIVE (every row becomes a parametrized test). Decision carries pan_plan/bureau_plan ('reuse'|'call'), counts_against_quota, is_refresh, directives (create_daily_cap_lock, review_reason, record_pan_correction), display dates (checked_on, refresh_available_on, ntc_found_on) and original_consent_id.
- **name**: policy.guards persist formulas | **definition**: compute_billed(calls) = any(c.billable); compute_counted_against_quota(*, kind, counts_against_quota, calls) = quick AND admission-flag AND any Score call with 200+success:true; quick_checks_left(limit, used_before, counted) = max(0, limit-used_before-(1 if counted else 0)); next_ist_midnight(now); ist_date(dt); pan_expiry/score_expiry/report_expiry. Worked: report fetched 2026-10-10T14:00Z -> checked_on 2026-10-10, refresh_available_on 2026-11-09 (matches the PRD section 6 sample).
- **name**: db.repo Repo protocol v2 + rows | **definition**: Literals CheckKind/CheckStatus(7)/Code(13)/ConsentCovers/LockKind/ScoreSource/ErrorKind/ReviewKind/PanEventKind; frozen rows ConsentRow(14f incl. lc_typed_name, covers, purge_after), PanRecordRow, ScoreCacheRow(source, is_ntc, consent_id), ReportCacheRow(report_date), CheckRow(23f incl. opportunity_stage, pan_last4, mobile_hash/last4, recompute_of), CheckPatch, SurePassCallRow(15f incl. pan_hash, latency_ms, billable, error_kind, storage_key, purge columns), PanLockRow, ReviewRow, OpportunityPanEvent, LenderRow(version, published, criteria known/missing/uncertain, matched_pincodes); 27 protocol methods as listed in db/repo.py contents (checks lifecycle incl. guarded terminal update + record_check_error + attach_contact; three cache get/put pairs + fetch_raw; quota_limit/quick_checks_used_today/bureau_calls_today; try_acquire/release/create locks + get_active_locks; review insert/unresolved_review_exists; pan events; log_surepass_call sink returning storage_key; surepass_calls_for_check; published_lenders/lenders_for_pincodes); code_legal(status, code) predicate.
- **name**: CallLogSink v2 (consumed by surepass/client.py, implemented by Repo) | **definition**: log_surepass_call(*, check_id, pan_hash, endpoint, client_id, http_status, success, message_code, latency_ms, error_kind, raw_envelope: dict|None) -> str|None. Repo stamps id/called_at, computes billable=(200 AND success) OR error_kind=='timeout', uploads raw_envelope verbatim to the bucket (key=storage_key_for(...)), returns the key. One row per HTTP attempt; zero rows for LiveCallsDisabled. The client passes the parsed envelope dict (or None when no body) and measured latency.
- **name**: db/schema.sql artifacts | **definition**: 12 app tables + code_legality CHECK + quota/opp/pan/purge indexes; SQL functions quick_checks_used_today(p_lc_id) and bureau_calls_today(p_pan_hash) (IST day-start bounds, sargable) and acquire_pan_lock(p_pan_hash, p_check_id, p_until)->boolean (atomic upsert-steal); RLS enabled no-policies on all 12; checkpointer schema isolation block (schema + role + search_path + to_regclass assertion note). Raw bucket 'raw-responses' private; object key pan/{pan_hash}/{slug}/{ts}-{call_id}.json.
- **name**: LcView persistence-derived fields (consumed by output/lc_view.py) | **definition**: from_cache = (bureau_plan=='reuse'); checked_on = ist_date(bureau fetched_at); refresh_available_on = ist_date(bureau fetched_at + window); quick_checks_left_today = quick_checks_left(...) on quick, null on full/recompute; WARN_NTC_RECENT refusal message date = decision.ntc_found_on; pan/mobile last4 available on the checks row for any display need. Refusal one-liners for the 10 admission codes live in output/lc_view.py (redaction-boundary single home ruling survives).

## Interfaces consumed
- **name**: config.py (API/config section) | **expectation**: Settings exposes identity_hmac_key (env IDENTITY_HMAC_KEY; required non-empty + non-default when REPO_BACKEND=supabase), raw_bucket (env RAW_BUCKET, default 'raw-responses'), surepass_live via the exact-literal 'true' property (binding ruling), DATABASE_URL for the checkpointer DSN, SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY for SupabaseRepo+SupabaseRawStore construction in main, and the per-call timeout constants PAN_TIMEOUT_S=8.0 / SCORE_TIMEOUT_S=12.0 / REPORT_TIMEOUT_S=20.0 / OPENAI_TIMEOUT_S=8.0. Degraded boot: empty env constructs Settings with MemoryRepo and serves /health + LIVE_CALLS_DISABLED.
- **name**: surepass/client.py (SurePass section) | **expectation**: Client owns audit logging through the v2 sink exactly once per HTTP attempt, passing pan_hash (received per call or at construction), measured latency_ms, error_kind ('timeout' on per-endpoint timeout expiry; 8/12/20 s from config), and the verbatim parsed envelope dict for storage; raises on timeout AFTER logging so the node can create the timeout lock; endpoint constants PAN_ENDPOINT/SCORE_ENDPOINT/REPORT_ENDPOINT stay defined there (I import them); FakeSurePassClient logs through the same sink so MemoryRepo call rows + MemoryRawStore objects are test ground truth; zero automatic retries.
- **name**: surepass/adapters.py (SurePass section) | **expectation**: Adapters are pure functions over raw envelope dicts (fetched from storage on hits/recompute, or fresh from the client): pan facts (status/category gate, full_name, dob, address.zip, aadhaar_linked, masked_aadhaar stored-only, client_id), score facts readable from EITHER a Score envelope OR a Report envelope via data.credit_score (needed because score_cache.source='report' points at the report object), NTC iff 200+success:true+credit_score not a 3-digit string, report facts per 8.2 with Pydantic extra='allow'. Adapters never receive storage keys or repo handles.
- **name**: graph/nodes.py + build.py (graph section) | **expectation**: Admission node executes my 8-step write order verbatim (consent row -> lock checks -> atomic acquire -> insert_check running -> stage gate -> PAN-change -> facts -> guard -> directives); pan/bureau nodes honor pan_plan/bureau_plan, write cache rows exactly as specified (Report pull writes BOTH caches), create the timeout lock + reconciliation review row on timeout; persist does ONE guarded terminal update and releases the in-flight lock in a finally that also covers the API error boundary; recompute entry runs the sub-graph with no consent/locks/quota; thread_id=check_id and the C15 checkpointer factory are reused unchanged.
- **name**: main.py routes (API section) | **expectation**: Routes validate consent as the v1.1 object ({given, wording_version, call_ref, ip}, given must be true at the schema), normalize+hash PAN/mobile via policy.guards before invoking, pass opportunity_stage through verbatim for logging, map RecomputeOutcome/AttachOutcome to 404/409/200, return refusals as HTTP-200 LcViews, and never log/echo raw PAN, mobile or name (422 handler strips input echo -- v1.0 pattern survives).
- **name**: output/lc_view.py (API/redaction section) | **expectation**: Owns the LC-facing one-liner strings for all 10 refusal codes (each passing the surviving BUREAU_TERM_RE/STANDALONE_SCORE_RE scrubbers; the WARN_NTC_RECENT line interpolates decision.ntc_found_on as an ISO date -- dates are exempt from the score regex by its lookarounds); consumes my from_cache/checked_on/refresh_available_on/quick_checks_left_today values without recomputing; never reads cache payloads or storage keys.
- **name**: tests/evals (tests section) | **expectation**: MemoryRepo(clock=frozen) + MemoryRawStore are the only backends in tests; billing_consistency reads checks rows + surepass_calls rows + MemoryRawStore keys as ground truth; every QK/FL/R/A table row, the HMAC pinned vectors, the lock acquire/steal/release semantics, IST boundary (23:59 vs 00:01 IST same UTC date) for BOTH SQL-function mirrors, quota-override path, purge job three steps, and the status x code legality predicate get parametrized unit tests; pytest-socket no-network and probe-literal-guard rulings apply unchanged.

## Decisions
- **topic**: HMAC single home and domain separation | **decision**: normalize_pan/normalize_mobile/hash_pan/hash_mobile/pan_last4/mobile_last4 live ONLY in policy/guards.py; config.py owns only the env key IDENTITY_HMAC_KEY (required non-empty with REPO_BACKEND=supabase, dev default allowed for memory backend). One key serves PAN and mobile with domain-separated messages b'pan:'+PAN and b'mobile:'+MOBILE. v1.0 ruling C3 (hash in config.py, salted SHA-256, PAN_HASH_SALT) is VOID. | **rationale**: v1.1 5.2 fixes HMAC-SHA256 with a server-side key held outside the database; domain separation prevents cross-identifier collisions with one secret and is a strengthening, never a weakening; single-home preserves the C3 intent (no second definition anywhere) while moving the code into the policy module the task assigns it to.
- **topic**: Caches store pointers to raw, not derived facts | **decision**: pan_records/score_cache/report_cache rows carry (storage_key, client_id, consent_id, is_ntc, fetched_at, expires_at, purge_after) and NOTHING derived; hits and recompute call repo.fetch_raw(storage_key) and re-run adapters, derivations, gate, rank and safety on current inputs and the current published lender version. v1.0's derived jsonb contract, DERIVED_SCHEMA_VERSION and is_cache_fresh schema-version check are VOID. | **rationale**: v1.1 5.4 is explicit: 'raw response reused; derivations, gate, ranking and safety check always re-run'; storing derived facts would freeze them against adapter fixes and lender-criteria updates, exactly what 5.4 forbids.
- **topic**: Raw object ownership and bucket layout | **decision**: One object per HTTP attempt that returned a body, written by the Repo sink inside log_surepass_call, key pan/{pan_hash}/{endpoint-slug}/{ts}-{call_id}.json in the private 'raw-responses' bucket; cache rows reference the same key as the owning surepass_calls row; purge is driven solely by surepass_calls.purge_after (called_at+180d), with cache rows deleted on their own purge_after. | **rationale**: PRD goal 4 and section 11 require EVERY raw response stored verbatim keyed to PAN (opportunity linkage stays relational via check_id); single ownership makes the 180-day purge job unambiguous and keeps upload atomic with the audit row the client already writes (C5's client-owned logging survives with a widened sink).
- **topic**: Report pull refreshes score_cache by pointing at the report object | **decision**: A successful Report pull upserts score_cache with source='report', storage_key = the report envelope's object, is_ntc from the report; a later quick check cache-hits and the score adapter reads data.credit_score from the report envelope (PRD 8.1: 'Experian Score or Experian Report, whichever is newer'). | **rationale**: Avoids duplicating raw objects or synthesizing fake score envelopes; 8.1 already requires the band derivation to accept both shapes, so the adapters section must expose score-facts-from-either-envelope.
- **topic**: Quota counting (PRD 5.6 implemented literally) | **decision**: counts_against_quota is set at admission ONLY when kind=quick AND no score_cache row exists at all (any age row present => exempt refresh); finalized at persist only if a Score call row with http 200 + success:true exists. Cache hits, PAN-invalid runs, service failures, timeouts and expired-score refreshes never count. QUOTA_EXHAUSTED can only fire on the counting path, so an LC at 5/5 can still re-check a previously-checked PAN (billed, uncounted). | **rationale**: Rule 6 is verbatim: counts only 'a billed Score call that returned HTTP 200 with success:true on a PAN with no stored score'; 'refresh of an expired stored score' is in the does-not-count list, which forces the row-exists (not row-fresh) reading; the 3-per-PAN daily cap bounds the resulting uncounted spend.
- **topic**: PAN lock model | **decision**: pan_locks PK (pan_hash, kind) with kinds in_flight (TTL 180 s, ASSUMPTION, atomic acquire via RPC acquire_pan_lock, holder-only release in persist's finally), timeout (10 min, created by the bureau node on timeout, paired with an UNRESOLVED review_queue row kind='timeout_reconciliation' that keeps blocking billed bureau pulls on that pan_hash even after the 10 minutes until manually resolved), daily_cap (until next IST midnight, created when the hard cap refuses). Active timeout/daily_cap locks refuse ALL runs on that pan_hash (PAN_LOCKED / PAN_DAILY_CAP), including would-be cache hits. | **rationale**: 5.11 demands refuse-not-queue (atomic upsert-steal gives exactly-one winner and crashed-run self-healing); 5.12 says both 'lock the PAN for 10 minutes' AND 'require manual reconciliation before any re-pull' -- the lock satisfies the first, the unresolved-review gate the second; 5.7/5.12 say 'the PAN is locked', so blocking even cache hits follows the letter rather than weakening a FIXED rule (surfaced for owner review as possibly stricter than intent).
- **topic**: Admission evaluation order | **decision**: consent -> existing timeout/daily-cap locks -> in-flight acquire -> insert checks row (running) -> stage gate (full) -> PAN-correction policy (full) -> windows -> quota (quick, counting path only) -> 3/PAN hard cap -> reconciliation gate -> WARN_NTC_RECENT (full, pull paths only) -> kill switch LAST; first match wins; guards are pure over an AdmissionFacts snapshot and return directives (daily-cap lock + review row, pan-correction event) that the node executes. | **rationale**: Mirrors the PRD section 7 admission node text ('consent row written; PAN lock acquired (else IN_FLIGHT); stage gate; windows; quota + per-PAN hard cap; NTC warning'); kill-switch-last preserves the v1.0 acceptance behavior that cache hits and refusals serve with SUREPASS_LIVE unset; pure-guard + node-executed writes keeps TOCTOU-sensitive lock acquisition atomic while the tables stay unit-testable without a repo.
- **topic**: Hard cap mechanics and billable predicate | **decision**: bureau_calls_today counts surepass_calls rows with billable=true on bureau endpoints on the IST date, where billable := (http_status==200 AND success) OR error_kind=='timeout'. At >=3 when a pull is needed: refuse PAN_DAILY_CAP, write the daily_cap lock and a review row. Timeouts count toward the cap and set checks.billed=true (spend unknown treated as spent until reconciled). | **rationale**: 5.7 caps billed bureau calls; a timed-out call's spend is unknown (5.12), so under-counting it would allow >3 real charges -- conservative counting can only err toward spending less, which is the direction the FIXED rules protect.
- **topic**: checks.status x code legality and the OPENAI_FAILURE shape | **decision**: Seven statuses (running/ok/pan_not_verified/refused/pending_verification/service_failure/error; running+error internal only). code required and refusal-valued on refused; required in {SUREPASS_FAILURE, OPENAI_FAILURE} on service_failure; NO_PINCODE_ON_PAN is the only code legal with ok; all other statuses carry null. OPENAI_FAILURE rows are status='service_failure' but their LcView still carries the headline band read and gated lender list (confidence low, review row). | **rationale**: Encoding legality as a CHECK constraint plus a shared code_legal() predicate makes illegal writes fail loudly in both backends; section 10's 'band read and gated list still shown' fixes the LcView content while 'Couldn't generate a recommendation, retry' fixes the failure status.
- **topic**: pending_verification lifecycle | **decision**: Bureau timeout -> timeout lock (10 min) + unresolved timeout_reconciliation review row -> persist status='pending_verification', code null, billed=true (conservative, from the billable timeout row), counted_against_quota=false -> END. Re-pulls stay blocked (PAN_LOCKED) until the review row is resolved by Credit against the SurePass console; resolution flips nothing automatically (manual reconciliation may correct billed). | **rationale**: Implements 5.12 end-to-end with an auditable gate instead of an untracked procedural rule; the 10-minute figure alone cannot carry 'require manual reconciliation before any re-pull'.
- **topic**: Consent rows on cache hits and refusals | **decision**: A consent row is written whenever request.consent.given is true (covers='score' for quick, 'report' for full), BEFORE lock acquisition, even when the run then cache-hits or is refused; cache rows store the ORIGINAL pull's consent_id, which persist copies into internal.original_consent_id as 'the original reference'. consent.given false (or absent) refuses CONSENT_MISSING with no fabricated row. v1.0 C21 (consent: Literal[True] at the schema) is superseded by the v1.1 consent object; the API section should require given==true at the schema and keep CONSENT_MISSING as the defensive guard. | **rationale**: 5.1 makes our record the only gate and the graph text says 'consent row written' first; recording a tick that WAS given is correct audit even for refused runs, while 'a cache hit needs no new consent and shows the original reference' is satisfied by surfacing the cached consent_id rather than by skipping the gate.
- **topic**: PAN correction tracked in opportunity_pan_events | **decision**: Dedicated audit table (initial/correction/manager_change events per opportunity). First PAN seen -> 'initial'; one 'correction' allowed while NO report exists on the opportunity (report_exists := any full check on the opportunity with status in ('ok','pending_verification') -- ASSUMPTION); any further change or any change after a report -> PAN_CHANGE_NEEDS_MANAGER + review row kind='pan_change_manager'. Re-verification and new-window behavior are automatic because all caches key on the new pan_hash. | **rationale**: 5.13's once-only correction cannot be derived reliably from checks rows (refused attempts also carry PANs); an explicit event log is the 'logged' audit the rule demands and gives managers a single place to record a manager_change later.
- **topic**: REPORT_WINDOW_OPEN and refresh semantics on full | **decision**: report fresh + refresh=true -> REPORT_WINDOW_OPEN with refresh_available_on; report fresh + no refresh -> free cache hit (shared across opportunities); report expired -> billed pull regardless of the flag, is_refresh := (expired row existed), never counted against any quota; refresh=true with no stored report -> plain new pull. | **rationale**: 5.9/5.10: one billed report per PAN per 30d NOT per opportunity, refresh only after 30d, refresh billed outside any quota, forced refresh inside a window: none for anyone; the v1.0 'refresh with no row = new' decision survives because there is nothing to refresh.
- **topic**: Recompute and attach as first-class admission tables | **decision**: recompute: allowed only from status 'ok' or ('service_failure','OPENAI_FAILURE'); no consent/locks/quota/SurePass; freshness NOT required (stored raw, original checked_on label); creates a NEW checks row with recompute_of + original consent_id; raw purged -> 'raw_unavailable' (HTTP 409, flagged gap). attach: quick-only, idempotent on same contact, conflict on different contact; updates checks.contact_id AND consents.contact_id. | **rationale**: Section 6 fixes 'no SurePass call, no quota' and 'returns a new LcView' (a new auditable row keeps every recommendation ever shown); PRD 11 lists 'contact id when attached' on the consent record, so attach must touch both tables; allowing recompute after OPENAI_FAILURE gives the rank node a free second chance on stored facts.
- **topic**: Score-fresh-but-PAN-expired edge (QK2/QK3, FL3/FL4) | **decision**: A bureau cache hit with an expired/absent pan_record still admits as cache_hit but with pan_plan='call' (fresh Rs-1 PAN Comprehensive, no quota, no consent implications); with SUREPASS_LIVE unset this path refuses LIVE_CALLS_DISABLED instead of serving. | **rationale**: Derivations need dob/zip from PAN facts and 5.3 fixes the 180-day window; reusing a >180d identity would silently extend a FIXED window, and the raw object may already be purged at 180d.
- **topic**: quick_checks_left_today presence | **decision**: Computed for quick checks only (max(0, limit - used_before - counted_this)); null on full-check and recompute LcViews. The persist node reads used_before before flipping this run's flag. | **rationale**: The PRD example shows it on the shared LcView but 5.5 scopes the quota to quick checks; v1.0's C20 register already flagged this exact choice for the owner -- kept and re-surfaced.
- **topic**: Versioned lenders with one published version | **decision**: lenders PK (id, version), partial unique index on (id) WHERE published; lender_pincodes keyed per (lender_id, version); matching reads published rows only; criteria jsonb stores every field as {value, status: known|missing|uncertain} and the gate excludes any lender whose REQUIRED_CRITERIA are not all 'known'. | **rationale**: PRD 11: 'versioned, published flag, normalised criteria with each field marked known / missing / uncertain' and 'a lender missing a required field is excluded from matching until fixed'; the admin UI writes drafts as new versions and publishing is an atomic flag flip.
- **topic**: Surviving v1.0 material referenced, not re-derived | **decision**: The following survive verbatim and are only re-pathed: C1 insert-early checks row + single guarded terminal update (.eq('status','running')) + record_check_error; OpportunityStage StrEnum/STAGE_ORDER/fail-closed stage gate (C18); endpoint constants owned by surepass/client.py; app-generated uuid4 ids + injected Clock + deep-copy MemoryRepo semantics; _iso/_ts serializers; IST SQL-function pattern; RLS-no-policies lockdown; checkpointer schema/role/search_path isolation block + to_regclass assertion (C15); kill-switch exact literal; thread_id=check_id. VOID from v1.0: pan_cache table, derived jsonb + DERIVED_SCHEMA_VERSION, salted-SHA256 hash_pan, 6-code refusal enum, DAILY_CAP_REACHED/OPPORTUNITY_ALREADY_CHECKED/CACHE_FRESH_NO_REFRESH codes, latest_full_check_for_opportunity gate, per-call update_check({'billed': True}) flip (billed now derives from billable call rows at persist). | **rationale**: The task mandates referencing surviving dossier pieces instead of re-deriving; the void list prevents TDD agents from resurrecting v1.0 semantics that v1.1 supersedes.

## Risks
- IN_FLIGHT_LOCK_TTL=180s is an assumption: if a graph run legitimately exceeds it (e.g., slow Report + slow OpenAI retry), a concurrent request could steal the lock and double-pull; mitigation is the hard cap + billing audit, escalation is raising the TTL or heartbeat renewal.
- acquire_pan_lock's upsert-steal races only through the RPC; any code path inserting pan_locks('in_flight') directly would break mutual exclusion -- the Repo must be the sole writer and a MemoryRepo/SQL parity test should pin steal-only-when-expired semantics.
- Storage upload and the surepass_calls insert are not transactional: an upload that succeeds followed by an insert crash orphans an object (harmless, purged by key-prefix sweep later), while insert-before-upload could leave a dangling storage_key -- the sink uploads FIRST, inserts second, and fetch_raw treats missing objects as RawUnavailable.
- Supabase Storage download on every cache hit adds latency to the p95 budget (25s quick / 35s full); envelopes are <1 MB JSON and one object per hit, but a region mismatch between the service and the bucket could hurt -- co-locate and measure.
- Rotating IDENTITY_HMAC_KEY orphans every cache row, consent linkage, lock, quota history and PAN-correction trail at once (worse than v1.0's salt rotation because three caches + locks now key on the hash); treat it as a long-lived secret with an explicit, owner-approved rotation runbook.
- Timeouts counting as billable means a flaky SurePass day can daily-cap PANs that were never charged; reconciliation un-blocks them only manually -- ops must watch review_queue kind='timeout_reconciliation' volume.
- The purge job deletes raw at 180d while score/report cache rows become unreadable at the same moment; recompute on old checks then 409s. If Phase 3 needs longer raw retention, only purge_after defaults change, but compliance sign-off is pending (open item).
- Composite-FK versioned lender_pincodes means republishing a lender requires re-inserting its pincode CSV for the new version; the admin UI (built separately) must copy-forward pincodes atomically or the gate silently loses coverage on publish.

## PRD gaps
- Rule 5.1 says a cache hit needs no new consent, but the admission order evaluates consent first and both tiers' UIs mandate the tick; I refuse CONSENT_MISSING on given=false even when a fresh cache could have served. Fail-closed reading -- confirm.
- 5.12's 'lock the PAN for 10 minutes' and 5.7's 'locked for the day' are implemented literally: active timeout/daily-cap locks refuse even would-be cache hits. If the intent was to block only billed pulls, relax QK/FL lock rows to pull-paths only (one-line change).
- The 'manual reconciliation before any re-pull' gate is modeled as an unresolved review_queue row (kind='timeout_reconciliation') blocking billed bureau pulls on that pan_hash indefinitely until resolved; the PRD names no resolution mechanism or UI -- Credit needs a resolve action (resolved_at/resolved_by columns exist).
- Rule 6's 'a PAN with no stored score' is read as 'no score_cache row exists at all' (an expired row makes the pull a non-counting refresh). Consequence: an LC at 5/5 can still trigger billed pulls on previously-checked PANs, bounded only by the 3/PAN/day cap -- confirm the cost model accepts this.
- The hard-cap lock is created only when a 4th pull is attempted; a cache-hit run on a PAN already at 3 calls (no lock yet) still serves. Asymmetric with the lock-blocks-everything rule above -- flagged, not resolved.
- 'Before any report exists on this opportunity' (5.13) is interpreted as any full check on the opportunity with status in ('ok','pending_verification') -- pending counts because spend/report existence is unknown. ASSUMPTION.
- recompute on purged raw (>180d) has no fitting code among the 13; proposed HTTP 409 problem response outside the LcView shape. Same for attach conflicts (409). Needs owner sign-off on the wire shape.
- timeout of a Score/Report call sets checks.billed=true conservatively (billable includes timeouts) while section 12's 'zero double-billed pulls' metric may want these reported separately; surepass_calls.error_kind='timeout' + pending_verification status give ops the distinction.
- The LcView sample shows quick_checks_left_today on what section 3 calls a shared view; I emit it on quick only (null on full/recompute), carrying over the v1.0 C20 flag -- confirm FulCRM tolerates null.
- PRD 11 says raw responses are 'keyed to PAN and opportunity'; object keys carry pan_hash only, with opportunity reachable relationally via surepass_calls.check_id -> checks.opportunity_id (a report is shared across opportunities, so a path-level opportunity key would be wrong for cache reuse). ASSUMPTION.
- 'Encrypted' bucket is satisfied by Supabase Storage's default AES-256 at-rest encryption on a private bucket; no application-layer envelope encryption is designed. Compliance should confirm this meets the retention sign-off (open item).
- v1.1 names no code for a refused run caused by a PAN lock created by the daily cap -- I return PAN_DAILY_CAP (not PAN_LOCKED) whenever the daily-cap condition or its lock blocks, reserving PAN_LOCKED for timeout/reconciliation locks; confirm the SuperConnect copy distinguishes them.
