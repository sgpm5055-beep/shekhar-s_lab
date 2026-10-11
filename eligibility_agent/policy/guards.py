"""Deterministic policy layer (PRD v1.1 section 5): HMAC identity hashing, windows, the pure admission decision
tables, PAN-correction / recompute / attach admission and the persist-time billing and quota formulas.

Layering: policy/ imports ONLY the standard library plus type and row definitions from ``db.repo``, the shared
shapes from ``graph.state`` (``AdmissionDecision`` / ``FullAdmissionInputs`` live there and are NOT redefined) and
the endpoint constants owned by ``surepass.client``. Every function here is pure: no network, no repo writes and
no clock reads (``now`` always arrives as ``AdmissionFacts.now`` or as an argument). The graph nodes call these;
the LLM never sees or influences them (PRD 5 preamble).

ADMISSION NODE WRITE ORDER (PRD section 7; executed by the graph admission node, NOT here). K10 amendment: the
check row is inserted at step 1.5, right after the consent row and BEFORE any lock check or acquire, because
``pan_locks.check_id`` is a foreign key to ``checks(id)`` and so is ``surepass_calls.check_id``.
  1.   consent given -> consent_id = new_id(); repo.insert_consent(ConsentRow(covers='score'|'report')).
       consent absent -> the outcome is a CONSENT_MISSING refusal (no consent row is fabricated); the flow still
       passes step 1.5 so the refusal has a check row to terminate.
  1.5. repo.insert_check(CheckRow(status='running', billed=False, counted_against_quota=False, ...)) -- EVERY
       outcome incl. refusals, BEFORE the lock checks and BEFORE any SurePass call (C1 insert-early; terminal
       update is guarded on status='running').
  2.   locks = repo.get_active_locks(pan_hash):   (node-level gate: runs BEFORE the tables, see LOCK GATE below)
         any kind='timeout'   active -> refuse PAN_LOCKED
         any kind='daily_cap' active -> refuse PAN_DAILY_CAP
  3.   acquired = repo.try_acquire_pan_lock(pan_hash, check_id, until=now + IN_FLIGHT_LOCK_TTL)
         not acquired -> refuse IN_FLIGHT (never queued, PRD 5.11)
  4.   (no write: the insert_check that used to sit here moved up to 1.5 -- K10)
  5.   full only: stage gate (stage_allows_full_check), then the PAN-correction policy via evaluate_pan_change(...).
  6.   facts = AdmissionFacts(...) fetched via repo; decision = admit_quick(facts) | admit_full(facts, full_inputs).
  7.   execute directives: decision.create_daily_cap_lock -> repo.create_pan_lock(kind='daily_cap',
       until=next_ist_midnight(now)) + repo.insert_review_item(kind=decision.review_reason);
       a 'record_correction' outcome from step 5 -> repo.insert_pan_event(event='correction') (the node sets
       ``record_pan_correction`` itself; the pure tables never do).
  8.   refusal -> persist_refusal -> END. persist / persist_refusal ALWAYS release the in-flight lock (finally).

LOCK GATE vs PURE TABLES. The step-2 refusals (active timeout / daily_cap lock) run in the NODE, BEFORE the pure
tables, and therefore preempt EVERYTHING the tables would decide, table cache hits (QK1 / QK3 / FL1 / FL4)
included: the tables never see a locked state. That is the node's lock gate, not a table row.

K2 RECONCILIATION GATE (billed pulls only). Once the 10-minute timeout lock has expired, the node's step-2 gate no
longer fires; the tables' own gate is ``AdmissionFacts.reconciliation_pending`` -- an unresolved
'timeout_reconciliation' review row -- which refuses PAN_LOCKED on the PULL paths only (QK7 / FL6), independently
of any lock row or of the clock. Lock expiry alone never re-admits a billed pull: after TIMEOUT_LOCK_TTL has
elapsed, QK7 / FL6 keep refusing until the review row is resolved. It does NOT gate cache hits: with the lock
expired, QK1 / QK3 / FL1 / FL4 are served under ``reconciliation_pending`` (a cache hit makes no billed call).

Mid-run codes (never produced by admission): NO_PINCODE_ON_PAN (pincode gate), SUREPASS_FAILURE (bureau / PAN
node) and OPENAI_FAILURE (rank node, after its single retry). Per-call timeouts live in config.py.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Final, Literal
from zoneinfo import ZoneInfo

from eligibility_agent.db.repo import QUICK_DAILY_QUOTA_DEFAULT as QUICK_DAILY_QUOTA_DEFAULT  # re-export (Task 2)
from eligibility_agent.db.repo import RAW_RETENTION as RAW_RETENTION  # re-export (Task 2)
from eligibility_agent.db.repo import (
    OpportunityPanEvent,
    PanRecordRow,
    ReportCacheRow,
    ScoreCacheRow,
    SurePassCallRow,
)
from eligibility_agent.graph.state import AdmissionDecision, Code, FullAdmissionInputs, Kind
from eligibility_agent.surepass.client import REPORT_ENDPOINT, SCORE_ENDPOINT

# --------------------------------------------------------------------------- constants (single source; v1.1 FIXED)

PAN_WINDOW: Final[timedelta] = timedelta(days=180)  # PRD 5.3
SCORE_WINDOW: Final[timedelta] = timedelta(days=30)  # PRD 5.3
REPORT_WINDOW: Final[timedelta] = timedelta(days=30)  # PRD 5.3 / 5.9
PAN_DAILY_BUREAU_CAP: Final[int] = 3  # PRD 5.7 (score + report together)
TIMEOUT_LOCK_TTL: Final[timedelta] = timedelta(minutes=10)  # PRD 5.12
IN_FLIGHT_LOCK_TTL: Final[timedelta] = timedelta(seconds=180)  # ASSUMPTION: 2x GRAPH_TIMEOUT_SECONDS(90), self-heals
CONSENT_RETENTION_SQL: Final[str] = "interval '5 years'"  # PRD 11 (consents); SQL-side only
IST: Final[ZoneInfo] = ZoneInfo("Asia/Kolkata")  # no DST: an IST day is exactly 24h
PAN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")  # runs on the NORMALIZED PAN
MOBILE_RE: Final[re.Pattern[str]] = re.compile(r"^[6-9][0-9]{9}$")  # runs on the NORMALIZED 10-digit mobile
IDENTITY_HMAC_KEY_ENV: Final[str] = "IDENTITY_HMAC_KEY"  # config.py reads it; the key NEVER reaches a table or log
# QUICK_DAILY_QUOTA_DEFAULT (5, PRD 5.5; per-LC override via lc_quota_overrides) and RAW_RETENTION (180d, PRD 11)
# are RE-EXPORTED from db.repo above: the SQL-side mirrors live there, so they are imported, never redefined.
BUREAU_ENDPOINTS: Final[frozenset[str]] = frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT})  # the billed bureau calls

# The 10 admission refusals. Anything else in ``Code`` is a mid-run code admission can never produce.
REFUSAL_CODES: Final[frozenset[Code]] = frozenset(
    {
        "CONSENT_MISSING",
        "QUOTA_EXHAUSTED",
        "PAN_DAILY_CAP",
        "STAGE_TOO_EARLY",
        "REPORT_WINDOW_OPEN",
        "IN_FLIGHT",
        "PAN_LOCKED",
        "WARN_NTC_RECENT",
        "PAN_CHANGE_NEEDS_MANAGER",
        "LIVE_CALLS_DISABLED",
    }
)

# --------------------------------------------------------------------------- stage gate (K19 / C18)


class OpportunityStage(StrEnum):
    """FulCRM opportunity stages in pipeline order (ASSUMPTION: PRD 9.2 / open item 1)."""

    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    COUNSELLING = "counselling"
    OFFER = "offer"
    REGISTRATION = "registration"
    ENROLLED = "enrolled"
    WON = "won"


STAGE_ORDER: Final[tuple[OpportunityStage, ...]] = tuple(OpportunityStage)
FULL_CHECK_MIN_STAGE: Final[OpportunityStage] = OpportunityStage.OFFER


def stage_allows_full_check(stage: str | None) -> bool:
    """True iff the trimmed, case-folded stage is a known stage at or beyond FULL_CHECK_MIN_STAGE.

    Fails closed: None and unknown stages are False (the node then refuses STAGE_TOO_EARLY). The caller-supplied
    stage string is stored verbatim on checks.opportunity_stage; only this gate normalizes it (PRD 5.8).
    """
    if stage is None:
        return False
    try:
        known = OpportunityStage(stage.strip().lower())
    except ValueError:
        return False
    return STAGE_ORDER.index(known) >= STAGE_ORDER.index(FULL_CHECK_MIN_STAGE)


# --------------------------------------------------------------------------- normalization + HMAC hashing (K3)


def normalize_pan(pan: str) -> str:
    """strip() then upper(); ValueError unless the result matches PAN_RE.

    Non-ASCII input is refused BEFORE upper-casing: Unicode case folding ('ß' -> 'SS', dotless 'ı' -> 'I') would
    otherwise map a non-PAN string onto a valid-looking PAN and so onto a real identity hash. The message never
    echoes the input (PII).
    """
    candidate = pan.strip()
    if not candidate.isascii():
        raise ValueError("invalid PAN format")
    candidate = candidate.upper()
    if PAN_RE.fullmatch(candidate) is None:
        raise ValueError("invalid PAN format")
    return candidate


_MOBILE_NOISE_RE: Final[re.Pattern[str]] = re.compile(r"[\s\-()]")


def normalize_mobile(mobile: str) -> str:
    """Remove spaces / hyphens / parentheses; strip a leading '+91'; strip a leading '91' when 12 digits remain;
    strip a leading '0' when 11 digits remain; ValueError unless MOBILE_RE matches the 10-digit result."""
    digits = _MOBILE_NOISE_RE.sub("", mobile)
    if digits.startswith("+91"):
        digits = digits[3:]
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if MOBILE_RE.fullmatch(digits) is None:
        raise ValueError("invalid mobile number")
    return digits


def _hmac_hex(key: str, message: bytes) -> str:
    if not key:
        raise ValueError(f"{IDENTITY_HMAC_KEY_ENV} must be non-empty")
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def hash_pan(pan: str, *, key: str) -> str:
    """HMAC-SHA256 over ``b'pan:' + normalize_pan(pan)``; 64 lowercase hex. Domain-separated so one key serves both
    identifiers without cross-type collisions.

    PINNED: hash_pan(' abcde1234f ', key='test-hmac-key')
      == '7c0b91b00e0094dad4090227778b2996da1ec7c7e9c45d5b4ac4b0883e55edc0'
    """
    return _hmac_hex(key, b"pan:" + normalize_pan(pan).encode("ascii"))


def hash_mobile(mobile: str, *, key: str) -> str:
    """HMAC-SHA256 over ``b'mobile:' + normalize_mobile(mobile)``; 64 lowercase hex.

    PINNED: hash_mobile('+91 98765-43210', key='test-hmac-key')
      == '160381e0c64be1a856ab25167a3a22468ccce58e2641993fd4407183cd2c858a' (normalizes to '9876543210')
    """
    return _hmac_hex(key, b"mobile:" + normalize_mobile(mobile).encode("ascii"))


def pan_last4(pan: str) -> str:
    """'ABCDE1234F' -> '234F'; stored separately for display (PRD 5.2)."""
    return normalize_pan(pan)[-4:]


def mobile_last4(mobile: str) -> str:
    """'9876543210' -> '3210'."""
    return normalize_mobile(mobile)[-4:]


# --------------------------------------------------------------------------- freshness and IST dates


def is_fresh(expires_at: datetime, now: datetime) -> bool:
    """A row is fresh strictly before its expiry instant (expires_at == now is expired)."""
    return expires_at > now


def pan_expiry(fetched_at: datetime) -> datetime:
    return fetched_at + PAN_WINDOW


def score_expiry(fetched_at: datetime) -> datetime:
    return fetched_at + SCORE_WINDOW


def report_expiry(fetched_at: datetime) -> datetime:
    return fetched_at + REPORT_WINDOW


def ist_date(dt: datetime) -> date:
    """The IST calendar date of an aware instant (a naive datetime would silently use the host zone: refused)."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("an aware datetime is required")
    return dt.astimezone(IST).date()


def next_ist_midnight(now: datetime) -> datetime:
    """daily_cap lock_until: (ist_date(now) + 1 day) at 00:00 IST, returned in UTC (quota resets 00:00 IST).

    IST has no DST, so the result is always 18:30 UTC and within the next 24 hours.
    """
    tomorrow = ist_date(now) + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=IST).astimezone(timezone.utc)


# WORKED EXAMPLE (pins the PRD section 6 sample): a report fetched 2026-10-10T14:00:00Z expires
# 2026-11-09T14:00:00Z, so LcView.checked_on = '2026-10-10' (IST date of fetched_at) and
# LcView.refresh_available_on = '2026-11-09' (IST date of expires_at). A PAN record fetched at the same moment
# expires 2027-04-08T14:00:00Z.


# --------------------------------------------------------------------------- admission facts


@dataclass(frozen=True, slots=True)
class AdmissionFacts:
    """Snapshot the admission node fetches via the Repo BEFORE calling a pure guard."""

    now: datetime  # aware UTC, captured once by the node
    pan_row: PanRecordRow | None  # repo.get_pan_record(pan_hash); stale rows included
    score_row: ScoreCacheRow | None  # repo.get_score_cache(pan_hash)
    report_row: ReportCacheRow | None  # repo.get_report_cache(pan_hash); None supplied for quick
    quota_used: int  # repo.quick_checks_used_today(lc_id)  (quick; 0 for full)
    quota_limit: int  # repo.quota_limit(lc_id) = override or QUICK_DAILY_QUOTA_DEFAULT
    bureau_calls_today: int  # repo.bureau_calls_today(pan_hash)  (billable, IST day)
    reconciliation_pending: bool  # repo.unresolved_review_exists(pan_hash, 'timeout_reconciliation')
    surepass_live: bool  # settings.surepass_live (exact-literal 'true')


def _row_is_fresh(row: PanRecordRow | ScoreCacheRow | ReportCacheRow | None, now: datetime) -> bool:
    return row is not None and is_fresh(row.expires_at, now)


# --------------------------------------------------------------------------- decision builders
# Exactly one of (admission, code) is non-None on every decision: ``_refuse`` sets only ``code``; ``_cache_hit`` and
# ``_pull`` set only ``admission``. Every table row funnels through these three builders, so the invariant is
# structural rather than a convention. Refusals never plan a call (pan_plan / bureau_plan 'reuse', never counted,
# never a refresh).


def _refuse(
    code: Code,
    *,
    quota_used_today: int | None = None,
    quota_limit: int | None = None,
    daily_cap: bool = False,
    ntc_found_on: date | None = None,
    refresh_available_on: date | None = None,
) -> AdmissionDecision:
    return AdmissionDecision(
        admission=None,
        code=code,
        pan_plan="reuse",
        bureau_plan="reuse",
        counts_against_quota=False,
        is_refresh=False,
        quota_used_today=quota_used_today,
        quota_limit=quota_limit,
        create_daily_cap_lock=daily_cap,
        review_reason="pan_daily_cap" if daily_cap else None,
        ntc_found_on=ntc_found_on,
        refresh_available_on=refresh_available_on,
    )


def _cache_hit(
    row: ScoreCacheRow | ReportCacheRow,
    *,
    pan_fresh: bool,
    quota_used_today: int | None = None,
    quota_limit: int | None = None,
) -> AdmissionDecision:
    """Served from the governing cache row; display fields come from that row (the ORIGINAL pull)."""
    return AdmissionDecision(
        admission="cache_hit",
        code=None,
        pan_plan="reuse" if pan_fresh else "call",
        bureau_plan="reuse",
        counts_against_quota=False,
        is_refresh=False,
        quota_used_today=quota_used_today,
        quota_limit=quota_limit,
        checked_on=ist_date(row.fetched_at),
        refresh_available_on=ist_date(row.expires_at),
        original_consent_id=row.consent_id,
    )


def _pull(
    mode: Literal["new", "refresh"],
    *,
    pan_fresh: bool,
    counts: bool,
    quota_used_today: int | None = None,
    quota_limit: int | None = None,
) -> AdmissionDecision:
    """A billed bureau pull. Display dates are derived after the pull from the freshly written cache row."""
    return AdmissionDecision(
        admission=mode,
        code=None,
        pan_plan="reuse" if pan_fresh else "call",
        bureau_plan="call",
        counts_against_quota=counts,
        is_refresh=mode == "refresh",
        quota_used_today=quota_used_today,
        quota_limit=quota_limit,
    )


# --------------------------------------------------------------------------- QUICK admission


def admit_quick(facts: AdmissionFacts) -> AdmissionDecision:
    """Pure quick-check admission. Consent, locks and in-flight were handled by the node (module docstring).

    NORMATIVE TABLE, first match wins. sf = score_row fresh, se = score_row expired, sn = score_row None,
    pf = pan_row fresh. Order: windows -> quota -> hard cap -> reconciliation -> kill switch (kill switch LAST, so
    cache hits are served with SUREPASS_LIVE unset). quota_used_today / quota_limit are echoed on EVERY path.

    | #   | score | pan | quota             | cap | recon | live  | outcome |
    | QK1 | sf    | pf  | -                 | -   | -     | -     | cache_hit; pan reuse; bureau reuse; counts F |
    | QK2 | sf    | !pf | -                 | -   | -     | False | LIVE_CALLS_DISABLED (PAN re-verify is a live call) |
    | QK3 | sf    | !pf | -                 | -   | -     | True  | cache_hit; pan call; bureau reuse; counts F |
    | QK4 | sn    | any | used >= limit     | -   | -     | -     | QUOTA_EXHAUSTED (only the counting path exhausts) |
    | QK5 | se    | any | (exempt: refresh) | -   | -     | -     | fall through (an expired-score re-pull never counts) |
    | QK6 | sn/se | any | pass              | >=3 | -     | -     | PAN_DAILY_CAP; daily-cap lock + review 'pan_daily_cap' |
    | QK7 | sn/se | any | pass              | <3  | True  | -     | PAN_LOCKED (manual reconciliation before any re-pull) |
    | QK8 | sn/se | any | pass              | <3  | False | False | LIVE_CALLS_DISABLED |
    | QK9 | sn/se | any | pass              | <3  | False | True  | admitted: 'new' if sn else 'refresh'; bureau call;
    |     |       |     |                   |     |       |       | pan call iff !pf; counts = sn; is_refresh = se |

    v1.1 has NO refresh flag on quick: an expired stored score is auto-refreshed (billed, never counted).
    """
    now = facts.now
    score = facts.score_row
    used, limit = facts.quota_used, facts.quota_limit
    pan_fresh = _row_is_fresh(facts.pan_row, now)

    if score is not None and is_fresh(score.expires_at, now):
        if pan_fresh:  # QK1
            return _cache_hit(score, pan_fresh=True, quota_used_today=used, quota_limit=limit)
        if not facts.surepass_live:  # QK2
            return _refuse("LIVE_CALLS_DISABLED", quota_used_today=used, quota_limit=limit)
        return _cache_hit(score, pan_fresh=False, quota_used_today=used, quota_limit=limit)  # QK3

    if score is None and used >= limit:  # QK4; QK5: an expired row skips this check entirely
        return _refuse("QUOTA_EXHAUSTED", quota_used_today=used, quota_limit=limit)
    if facts.bureau_calls_today >= PAN_DAILY_BUREAU_CAP:  # QK6
        return _refuse("PAN_DAILY_CAP", quota_used_today=used, quota_limit=limit, daily_cap=True)
    if facts.reconciliation_pending:  # QK7
        return _refuse("PAN_LOCKED", quota_used_today=used, quota_limit=limit)
    if not facts.surepass_live:  # QK8
        return _refuse("LIVE_CALLS_DISABLED", quota_used_today=used, quota_limit=limit)
    return _pull(  # QK9
        "new" if score is None else "refresh",
        pan_fresh=pan_fresh,
        counts=score is None,
        quota_used_today=used,
        quota_limit=limit,
    )


# --------------------------------------------------------------------------- FULL admission


def admit_full(facts: AdmissionFacts, full: FullAdmissionInputs) -> AdmissionDecision:
    """Pure full-check admission. Node pre-steps: stage gate (STAGE_TOO_EARLY) and the PAN-correction policy.

    NORMATIVE TABLE, first match wins. rf = report_row fresh, re = expired, rn = None;
    ntc_recent = score_row exists AND score_row.is_ntc AND score_row.expires_at > now (PRD 5.14).
    Quota fields stay None: report pulls sit outside any quick quota (PRD 5.9), so counts is always False.

    | #   | report | refresh | pan | cap | recon | ntc_recent & !ack | live  | outcome |
    | FL1 | rf     | False   | pf  | -   | -     | -                 | -     | cache_hit; reuse both |
    | FL2 | rf     | True    | any | -   | -     | -                 | -     | REPORT_WINDOW_OPEN; refresh_available_on |
    | FL3 | rf     | False   | !pf | -   | -     | -                 | False | LIVE_CALLS_DISABLED |
    | FL4 | rf     | False   | !pf | -   | -     | -                 | True  | cache_hit; pan call; bureau reuse |
    | FL5 | re/rn  | any     | any | >=3 | -     | -                 | -     | PAN_DAILY_CAP + lock + review (cap before NTC) |
    | FL6 | re/rn  | any     | any | <3  | True  | -                 | -     | PAN_LOCKED (timeout reconciliation outstanding) |
    | FL7 | re/rn  | any     | any | <3  | False | True              | -     | WARN_NTC_RECENT; ntc_found_on |
    | FL8 | re/rn  | any     | any | <3  | False | False / acked     | False | LIVE_CALLS_DISABLED |
    | FL9 | re/rn  | any     | any | <3  | False | False / acked     | True  | admitted: 'refresh' if re else 'new'; bureau call;
    |     |        |         |     |     |       |                   |       | pan call iff !pf; is_refresh = re; counts F |

    refresh=True with rn (nothing to refresh) falls into FL9 as mode 'new'.
    """
    now = facts.now
    report = facts.report_row
    pan_fresh = _row_is_fresh(facts.pan_row, now)

    if report is not None and is_fresh(report.expires_at, now):
        if full.refresh:  # FL2: a forced refresh inside the window is refused for everyone (PRD 5.10)
            return _refuse("REPORT_WINDOW_OPEN", refresh_available_on=ist_date(report.expires_at))
        if pan_fresh:  # FL1
            return _cache_hit(report, pan_fresh=True)
        if not facts.surepass_live:  # FL3
            return _refuse("LIVE_CALLS_DISABLED")
        return _cache_hit(report, pan_fresh=False)  # FL4

    if facts.bureau_calls_today >= PAN_DAILY_BUREAU_CAP:  # FL5
        return _refuse("PAN_DAILY_CAP", daily_cap=True)
    if facts.reconciliation_pending:  # FL6
        return _refuse("PAN_LOCKED")
    score = facts.score_row
    if score is not None and score.is_ntc and is_fresh(score.expires_at, now) and not full.acknowledge_ntc_warning:
        return _refuse("WARN_NTC_RECENT", ntc_found_on=ist_date(score.fetched_at))  # FL7
    if not facts.surepass_live:  # FL8
        return _refuse("LIVE_CALLS_DISABLED")
    return _pull("new" if report is None else "refresh", pan_fresh=pan_fresh, counts=False)  # FL9


# --------------------------------------------------------------------------- PAN correction policy (PRD 5.13)

PanChangeOutcome = Literal["first_pan", "same_pan", "record_correction", "needs_manager"]


def evaluate_pan_change(
    events: Sequence[OpportunityPanEvent], report_exists: bool, request_pan_hash: str
) -> PanChangeOutcome:
    """Full checks only; evaluated by the node BEFORE admit_full. TABLE, first match wins:

    | events empty                                        -> 'first_pan' (node records event='initial')
    | request_pan_hash == latest event's pan_hash         -> 'same_pan'
    | report_exists                                       -> 'needs_manager' (refuse PAN_CHANGE_NEEDS_MANAGER +
    |                                                        review row kind='pan_change_manager')
    | any event with event == 'correction' already exists -> 'needs_manager' (the ONE correction is spent)
    | else                                                -> 'record_correction' (node writes event='correction';
    |   the corrected PAN has no pan_record so it is re-verified automatically, and the once-per-window rule is
    |   per pan_hash so it is a new pull)

    'Latest' is by created_at (ties keep list order), so the order the repo returns events in is not relied on.
    report_exists is an ASSUMPTION: any full check on this opportunity with status in ('ok',
    'pending_verification') ('pending_verification' counts: unknown spend means a report may exist), or
    'service_failure' + 'OPENAI_FAILURE' (the report was pulled and billed; only the ranking failed). The predicate's
    single home is ``db.repo.report_check_on_record``.
    """
    if not events:
        return "first_pan"
    latest = sorted(events, key=lambda event: event.created_at)[-1]
    if request_pan_hash == latest.pan_hash:
        return "same_pan"
    if report_exists:
        return "needs_manager"
    if any(event.event == "correction" for event in events):
        return "needs_manager"
    return "record_correction"


# --------------------------------------------------------------------------- recompute admission (PRD section 6)

RecomputeOutcome = Literal["ok", "not_found", "not_recomputable", "raw_unavailable"]


def admit_recompute(
    check_status: str | None,
    check_code: str | None,
    kind: Kind | None,
    pan_row_present: bool,
    bureau_row_present: bool,
) -> RecomputeOutcome:
    """POST /checks/{check_id}/recompute. ``check_status`` None means ``repo.get_check`` found nothing (and ``kind`` is
    then None too: a missing check has no kind, so R1 never needs a dummy one; Task 11 widening, ruling 12).

    | R1 | check missing (check_status is None)                                  -> 'not_found' (API 404) |
    | R2 | status != 'ok' and not (status == 'service_failure' and code == 'OPENAI_FAILURE')
    |    |                                                  -> 'not_recomputable' (API 409; only runs that
    |    |                                                     produced facts can re-derive) |
    | R3 | quick: pan_records or score_cache row missing for check.pan_hash     -> 'raw_unavailable' (409; purged) |
    | R4 | full:  pan_records or report_cache row missing                       -> 'raw_unavailable' |
    | R5 | else 'ok': NO consent row, NO locks, NO quota, NO SurePass; freshness NOT required (stored raw, labelled
    |    | checked_on = the original fetched_at); a new checks row with recompute_of = check_id, the original
    |    | consent_id, billed False, counted_against_quota False, from_cache True; re-runs derive -> gate -> range
    |    | -> confidence -> rank -> safety -> redact -> persist on the CURRENT published lenders. |

    ``bureau_row_present`` is the presence of the bureau row matching ``kind`` (score for quick, report for full),
    so R3 and R4 share one predicate and ``kind`` is accepted only to keep the table's shape.
    """
    del kind  # R3 / R4 differ only in WHICH bureau row the caller checked for presence
    if check_status is None:
        return "not_found"
    recomputable = check_status == "ok" or (check_status == "service_failure" and check_code == "OPENAI_FAILURE")
    if not recomputable:
        return "not_recomputable"
    if not (pan_row_present and bureau_row_present):
        return "raw_unavailable"
    return "ok"


# --------------------------------------------------------------------------- attach admission (quick only)

AttachOutcome = Literal["ok", "not_found", "not_quick", "already_attached", "conflict"]


def admit_attach(
    kind: Kind | None, existing_contact_id: str | None, contact_id: str, *, consent_contact_id: str | None
) -> AttachOutcome:
    """POST /checks/{check_id}/attach. ``kind`` None means the check does not exist. ``consent_contact_id`` is the
    contact already on the check's CONSENT row (None when unset or when the check has no consent row).

    | A1  | check missing                                   -> 'not_found' (404) |
    | A2  | check.kind != 'quick'                           -> 'not_quick' (409) |
    | A2b | the consent row carries a DIFFERENT contact     -> 'conflict' (409) |
    | A3  | check.contact_id == contact_id                  -> 'already_attached' (idempotent 200) |
    | A4  | check.contact_id set and different              -> 'conflict' (409) |
    | A5  | else 'ok': repo.attach_contact updates checks.contact_id AND consents.contact_id (PRD 11). |

    A2b (T14 fix round 1): a recompute row SHARES its original's consent row (K9), so attaching one of them sets the
    contact on a consent the other also points at; attaching the other to a different contact would silently rewrite
    the compliance record. The consent's contact therefore conflicts like the check's own (same contact: idempotent).
    """
    if kind is None:
        return "not_found"
    if kind != "quick":
        return "not_quick"
    if consent_contact_id is not None and consent_contact_id != contact_id:
        return "conflict"
    if existing_contact_id == contact_id:
        return "already_attached"
    if existing_contact_id is not None:
        return "conflict"
    return "ok"


# --------------------------------------------------------------------------- persist-time formulas (K8)


def compute_billed(calls: Sequence[SurePassCallRow]) -> bool:
    """True iff any call row is billable (PAN Rs-1 calls included). ``billable`` is computed by the Repo sink:
    (http_status == 200 and success is True) or error_kind == 'timeout' (spend unknown -> conservatively billed;
    manual reconciliation corrects it, PRD 5.12)."""
    return any(call.billable for call in calls)


def compute_counted_against_quota(*, kind: Kind, counts_against_quota: bool, calls: Sequence[SurePassCallRow]) -> bool:
    """PRD 5.6: kind == 'quick' AND counts_against_quota (admission: no stored score row of ANY age existed) AND
    any call row with endpoint == SCORE_ENDPOINT, http_status == 200 and success is True.

    Status-independent by construction (it takes no check status): a quick run that later fails in the rank node
    (OPENAI_FAILURE) still spent a billed 200+success Score call and still counts. Consequences pinned by tests:
    cache hit F; PAN invalid (bureau never called) F; service failure / timeout F (no 200+success row); NTC T
    (200+success with a non-3-digit score); expired-score refresh F (admission already set the flag False).
    """
    return (
        kind == "quick"
        and counts_against_quota
        and any(
            call.endpoint == SCORE_ENDPOINT and call.http_status == 200 and call.success is True for call in calls
        )
    )


def quick_checks_left(limit: int, used_before: int, counted_this_check: bool) -> int:
    """max(0, limit - used_before - (1 if counted_this_check else 0)); used_before is read BEFORE the final update
    flips this run's flag. WORKED: limit 5, used 3, counting run -> 1 ('1 of 5 checks left today'); override limit
    8, used 5, cache hit -> 3. Quick only; the LcView field is None on full."""
    return max(0, limit - used_before - (1 if counted_this_check else 0))
