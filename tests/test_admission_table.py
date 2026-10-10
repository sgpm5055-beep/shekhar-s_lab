"""Task 7: the deterministic policy layer (PRD 5): admission decision tables, PAN-correction policy, recompute and
attach admission, persist-time billing/quota formulas, windows and the stage gate.

Every decision-table row is pinned on the WHOLE ``AdmissionDecision`` (equality on the frozen dataclass), so a stray
directive, a wrong plan or a missing quota echo fails the row that owns it. Precedence is pinned separately with
facts that satisfy several rows at once. The exactly-one-of-(admission, code) invariant (T1 note) is swept over the
full cartesian product of the facts that can change an outcome.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast, get_args

import pytest
from repo_builders import (
    PAN_A,
    PAN_B,
    T0,
    UTC,
    make_pan_event,
    make_pan_record,
    make_report_cache,
    make_score_cache,
)

from eligibility_agent.db import repo as repo_module
from eligibility_agent.db.repo import SurePassCallRow, call_is_billable
from eligibility_agent.graph import state as state_module
from eligibility_agent.graph.state import AdmissionDecision, Code, FullAdmissionInputs, Kind
from eligibility_agent.policy import guards
from eligibility_agent.policy.guards import (
    BUREAU_ENDPOINTS,
    FULL_CHECK_MIN_STAGE,
    IN_FLIGHT_LOCK_TTL,
    IST,
    PAN_DAILY_BUREAU_CAP,
    PAN_WINDOW,
    REFUSAL_CODES,
    REPORT_WINDOW,
    SCORE_WINDOW,
    STAGE_ORDER,
    TIMEOUT_LOCK_TTL,
    AdmissionFacts,
    OpportunityStage,
    admit_attach,
    admit_full,
    admit_quick,
    admit_recompute,
    compute_billed,
    compute_counted_against_quota,
    evaluate_pan_change,
    is_fresh,
    ist_date,
    next_ist_midnight,
    pan_expiry,
    quick_checks_left,
    report_expiry,
    score_expiry,
    stage_allows_full_check,
)
from eligibility_agent.surepass import client as client_module
from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT

DAY = timedelta(days=1)

# --------------------------------------------------------------------------- fixtures: rows with chosen freshness
# Dates are chosen so the IST date differs from the UTC date: a UTC-date implementation fails the row.

SCORE_FRESH_AT = datetime(2026, 9, 20, 20, 0, tzinfo=UTC)  # IST 2026-09-21 01:30; expires 2026-10-20T20:00Z (IST 10-21)
SCORE_CHECKED_ON = date(2026, 9, 21)
SCORE_REFRESH_ON = date(2026, 10, 21)
SCORE_EXPIRED_AT = T0 - 40 * DAY  # expired 10 days before T0
NTC_FRESH_AT = datetime(2026, 10, 1, 19, 0, tzinfo=UTC)  # IST 2026-10-02 00:30; expires 2026-10-31T19:00Z
NTC_FOUND_ON = date(2026, 10, 2)
REPORT_FRESH_AT = datetime(2026, 10, 5, 21, 0, tzinfo=UTC)  # IST 2026-10-06 02:30; expires 2026-11-04T21:00Z
REPORT_CHECKED_ON = date(2026, 10, 6)
REPORT_REFRESH_ON = date(2026, 11, 5)
REPORT_EXPIRED_AT = T0 - 31 * DAY
PAN_FRESH_AT = T0 - 100 * DAY
PAN_EXPIRED_AT = T0 - 181 * DAY  # expired one day before T0


def _score(fetched: datetime, *, is_ntc: bool = False, consent_id: str | None = "con-score") -> Any:
    return make_score_cache(
        fetched_at=fetched,
        expires_at=fetched + SCORE_WINDOW,
        purge_after=fetched + timedelta(days=180),
        is_ntc=is_ntc,
        consent_id=consent_id,
    )


def _report(fetched: datetime, *, consent_id: str | None = "con-report") -> Any:
    return make_report_cache(
        fetched_at=fetched,
        expires_at=fetched + REPORT_WINDOW,
        purge_after=fetched + timedelta(days=180),
        consent_id=consent_id,
    )


def _pan(fetched: datetime) -> Any:
    return make_pan_record(fetched_at=fetched, expires_at=fetched + PAN_WINDOW, purge_after=fetched + PAN_WINDOW)


SCORE_FRESH = _score(SCORE_FRESH_AT)
SCORE_EXPIRED = _score(SCORE_EXPIRED_AT)
SCORE_NTC = _score(NTC_FRESH_AT, is_ntc=True)
SCORE_NTC_EXPIRED = _score(SCORE_EXPIRED_AT, is_ntc=True)
REPORT_FRESH = _report(REPORT_FRESH_AT)
REPORT_EXPIRED = _report(REPORT_EXPIRED_AT)
PAN_FRESH = _pan(PAN_FRESH_AT)
PAN_EXPIRED = _pan(PAN_EXPIRED_AT)


def make_facts(**overrides: Any) -> AdmissionFacts:
    fields: dict[str, Any] = {
        "now": T0,
        "pan_row": None,
        "score_row": None,
        "report_row": None,
        "quota_used": 2,
        "quota_limit": 5,
        "bureau_calls_today": 0,
        "reconciliation_pending": False,
        "surepass_live": True,
    }
    fields.update(overrides)
    return AdmissionFacts(**fields)


def full_facts(**overrides: Any) -> AdmissionFacts:
    """Full checks carry quota_used=0 (the node does not read the quick quota for them)."""
    return make_facts(quota_used=0, **overrides)


def D(**fields: Any) -> AdmissionDecision:
    """The whole expected decision: anything not named keeps the neutral AdmissionDecision default."""
    return AdmissionDecision(**fields)


def echo(used: int = 2, limit: int = 5) -> dict[str, int]:
    return {"quota_used_today": used, "quota_limit": limit}


CAP_DIRECTIVES: dict[str, Any] = {"create_daily_cap_lock": True, "review_reason": "pan_daily_cap"}

# --------------------------------------------------------------------------- constants and re-exports


def test_fixed_window_and_cap_constants() -> None:
    assert PAN_WINDOW == timedelta(days=180)
    assert SCORE_WINDOW == timedelta(days=30)
    assert REPORT_WINDOW == timedelta(days=30)
    assert PAN_DAILY_BUREAU_CAP == 3
    assert TIMEOUT_LOCK_TTL == timedelta(minutes=10)
    assert IN_FLIGHT_LOCK_TTL == timedelta(seconds=180)
    assert IST.key == "Asia/Kolkata"
    assert guards.IDENTITY_HMAC_KEY_ENV == "IDENTITY_HMAC_KEY"


def test_repo_mirrors_are_reexported_not_redefined() -> None:
    assert guards.QUICK_DAILY_QUOTA_DEFAULT == 5
    assert guards.QUICK_DAILY_QUOTA_DEFAULT is repo_module.QUICK_DAILY_QUOTA_DEFAULT
    assert guards.RAW_RETENTION == timedelta(days=180)
    assert guards.RAW_RETENTION is repo_module.RAW_RETENTION


def test_decision_shapes_are_imported_from_state_not_redefined() -> None:
    assert guards.AdmissionDecision is state_module.AdmissionDecision
    assert guards.FullAdmissionInputs is state_module.FullAdmissionInputs


def test_refusal_codes_are_exactly_the_ten_admission_refusals() -> None:
    assert REFUSAL_CODES == frozenset(
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
    assert len(REFUSAL_CODES) == 10
    mid_run = {"NO_PINCODE_ON_PAN", "SUREPASS_FAILURE", "OPENAI_FAILURE"}
    assert REFUSAL_CODES == frozenset(get_args(Code)) - mid_run  # no drift from the registry
    assert REFUSAL_CODES == repo_module._REFUSAL_CODES  # pyright: ignore[reportPrivateUsage]  # and none from repo


def test_bureau_endpoints_are_built_from_the_client_constants() -> None:
    assert BUREAU_ENDPOINTS == frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT})
    assert BUREAU_ENDPOINTS == client_module.BUREAU_ENDPOINTS
    assert PAN_ENDPOINT not in BUREAU_ENDPOINTS
    assert guards.SCORE_ENDPOINT is client_module.SCORE_ENDPOINT


def test_pure_layering_imports_only_stdlib_and_the_three_sanctioned_modules() -> None:
    tree = ast.parse(Path(inspect.getfile(guards)).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module if node.module.startswith("eligibility_agent") else node.module.split(".")[0])
    stdlib = {"__future__", "collections", "dataclasses", "datetime", "enum", "hashlib", "hmac", "re", "typing", "zoneinfo"}
    sanctioned = {"eligibility_agent.db.repo", "eligibility_agent.graph.state", "eligibility_agent.surepass.client"}
    assert imported <= stdlib | sanctioned, imported - stdlib - sanctioned


def test_guards_never_read_a_clock() -> None:
    """Pure: ``now`` is always a parameter (facts.now / function arg)."""
    source = Path(inspect.getfile(guards)).read_text()
    assert not re.search(r"\.now\(|utcnow|time\.time|utc_now|\.today\(", source)


def test_guards_have_no_repo_access() -> None:
    names = {n for n in dir(guards) if not n.startswith("__")}
    assert not names & {"Repo", "MemoryRepo", "SupabaseRepo", "RawStore", "MemoryRawStore"}


def test_module_docstring_documents_the_eight_step_write_order_with_insert_check_at_one_point_five() -> None:
    doc = guards.__doc__
    assert doc is not None
    lines = doc.splitlines()

    def index_of(pattern: str) -> int:
        return next(i for i, line in enumerate(lines) if re.search(pattern, line))

    consent = index_of(r"^\s*1\.\s.*insert_consent")
    insert_check = index_of(r"^\s*1\.5\.?\s.*insert_check")
    locks = index_of(r"^\s*2\.\s.*get_active_locks")
    acquire = index_of(r"^\s*3\.\s.*try_acquire_pan_lock")
    assert consent < insert_check < locks < acquire  # the check row precedes the lock FK (K10)
    assert "status='running'" in lines[insert_check] or "status='running'" in lines[insert_check + 1]
    steps = {1.0, 1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0}
    found = {float(m.group(1)) for line in lines if (m := re.match(r"^\s*(\d(?:\.5)?)\.\s", line))}
    assert steps <= found
    flat = " ".join(doc.split()).lower()  # the prose wraps across lines
    assert "reconciliation_pending" in flat and "lock expiry alone never re-admits" in flat


# --------------------------------------------------------------------------- freshness, windows, IST, worked dates


def test_is_fresh_is_strictly_greater_than_now() -> None:
    assert is_fresh(T0 + timedelta(seconds=1), T0) is True
    assert is_fresh(T0, T0) is False
    assert is_fresh(T0 - timedelta(seconds=1), T0) is False


def test_expiry_helpers_add_their_window() -> None:
    assert pan_expiry(T0) == T0 + timedelta(days=180)
    assert score_expiry(T0) == T0 + timedelta(days=30)
    assert report_expiry(T0) == T0 + timedelta(days=30)


def test_section_six_worked_dates() -> None:
    fetched = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)
    assert report_expiry(fetched) == datetime(2026, 11, 9, 14, 0, tzinfo=UTC)
    assert score_expiry(fetched) == datetime(2026, 11, 9, 14, 0, tzinfo=UTC)
    assert pan_expiry(fetched) == datetime(2027, 4, 8, 14, 0, tzinfo=UTC)
    assert ist_date(fetched).isoformat() == "2026-10-10"  # LcView.checked_on
    assert ist_date(report_expiry(fetched)).isoformat() == "2026-11-09"  # LcView.refresh_available_on


def test_worked_dates_flow_through_a_full_check_cache_hit() -> None:
    fetched = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)
    decision = admit_full(
        full_facts(now=fetched + 5 * DAY, report_row=_report(fetched), pan_row=PAN_FRESH),
        FullAdmissionInputs(refresh=False, acknowledge_ntc_warning=False),
    )
    assert (decision.checked_on, decision.refresh_available_on) == (date(2026, 10, 10), date(2026, 11, 9))


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        (datetime(2026, 10, 10, 18, 29, 59, tzinfo=UTC), date(2026, 10, 10)),  # 23:59:59 IST
        (datetime(2026, 10, 10, 18, 30, 0, tzinfo=UTC), date(2026, 10, 11)),  # 00:00:00 IST
        (datetime(2026, 10, 10, 0, 0, 0, tzinfo=UTC), date(2026, 10, 10)),
        (datetime(2026, 12, 31, 18, 30, tzinfo=UTC), date(2027, 1, 1)),
        (datetime(2026, 10, 11, 1, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))), date(2026, 10, 11)),
    ],
)
def test_ist_date(instant: datetime, expected: date) -> None:
    assert ist_date(instant) == expected


def test_ist_date_refuses_a_naive_datetime() -> None:
    with pytest.raises(ValueError):
        ist_date(datetime(2026, 10, 10, 12, 0))


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (T0, datetime(2026, 10, 10, 18, 30, tzinfo=UTC)),  # 13:30 IST -> that night's 00:00 IST
        (datetime(2026, 10, 10, 18, 29, 59, tzinfo=UTC), datetime(2026, 10, 10, 18, 30, tzinfo=UTC)),
        (datetime(2026, 10, 10, 18, 30, 0, tzinfo=UTC), datetime(2026, 10, 11, 18, 30, tzinfo=UTC)),  # exactly midnight
        (datetime(2026, 10, 10, 18, 30, 1, tzinfo=UTC), datetime(2026, 10, 11, 18, 30, tzinfo=UTC)),
        (datetime(2026, 12, 31, 20, 0, tzinfo=UTC), datetime(2027, 1, 1, 18, 30, tzinfo=UTC)),  # year boundary
        (datetime(2027, 2, 28, 10, 0, tzinfo=UTC), datetime(2027, 2, 28, 18, 30, tzinfo=UTC)),
        (datetime(2028, 2, 28, 10, 0, tzinfo=UTC), datetime(2028, 2, 28, 18, 30, tzinfo=UTC)),  # leap year: Feb 29 00:00
        (datetime(2028, 2, 28, 19, 0, tzinfo=UTC), datetime(2028, 2, 29, 18, 30, tzinfo=UTC)),
        (datetime(2026, 3, 29, 3, 0, tzinfo=UTC), datetime(2026, 3, 29, 18, 30, tzinfo=UTC)),  # no DST anywhere in IST
        (datetime(2026, 10, 25, 3, 0, tzinfo=UTC), datetime(2026, 10, 25, 18, 30, tzinfo=UTC)),
    ],
)
def test_next_ist_midnight_vectors(now: datetime, expected: datetime) -> None:
    result = next_ist_midnight(now)
    assert result == expected
    assert result.utcoffset() == timedelta(0)  # returned in UTC


def test_next_ist_midnight_is_always_18_30_utc_within_the_next_24h() -> None:
    start = datetime(2026, 12, 30, 0, 0, tzinfo=UTC)
    for minutes in range(0, 3 * 24 * 60, 37):
        now = start + timedelta(minutes=minutes)
        result = next_ist_midnight(now)
        assert (result.hour, result.minute, result.second) == (18, 30, 0)
        assert now < result <= now + DAY


def test_next_ist_midnight_accepts_any_aware_zone_and_refuses_naive() -> None:
    ist_aware = datetime(2026, 10, 10, 13, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert next_ist_midnight(ist_aware) == datetime(2026, 10, 10, 18, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        next_ist_midnight(datetime(2026, 10, 10, 13, 30))


# --------------------------------------------------------------------------- stage gate (K19 / C18)


def test_opportunity_stage_enum_and_order() -> None:
    assert [s.value for s in OpportunityStage] == [
        "new",
        "contacted",
        "qualified",
        "counselling",
        "offer",
        "registration",
        "enrolled",
        "won",
    ]
    assert len(OpportunityStage) == 8
    assert STAGE_ORDER == tuple(OpportunityStage)
    assert FULL_CHECK_MIN_STAGE == "offer"
    assert isinstance(OpportunityStage.OFFER, str)


@pytest.mark.parametrize(
    ("stage", "allowed"),
    [
        ("new", False),
        ("contacted", False),
        ("qualified", False),
        ("counselling", False),
        ("offer", True),
        ("registration", True),
        ("enrolled", True),
        ("won", True),
        ("OFFER", True),  # case-insensitive
        (" Offer ", True),  # trimmed
        ("Counselling", False),
        (None, False),  # fail closed
        ("", False),
        ("lost", False),  # unknown fails closed
        ("offered", False),
        ("offer stage", False),
    ],
)
def test_stage_allows_full_check(stage: str | None, allowed: bool) -> None:
    assert stage_allows_full_check(stage) is allowed


# --------------------------------------------------------------------------- AdmissionFacts


def test_admission_facts_shape_is_frozen_and_slotted() -> None:
    names = [f.name for f in dataclasses.fields(AdmissionFacts)]
    assert names == [
        "now",
        "pan_row",
        "score_row",
        "report_row",
        "quota_used",
        "quota_limit",
        "bureau_calls_today",
        "reconciliation_pending",
        "surepass_live",
    ]
    facts = make_facts()
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.quota_used = 9  # pyright: ignore[reportAttributeAccessIssue]
    assert not hasattr(facts, "__dict__")  # slots=True


def test_facts_carry_no_lock_field_so_reconciliation_is_the_only_post_expiry_gate() -> None:
    assert not [f.name for f in dataclasses.fields(AdmissionFacts) if "lock" in f.name]


# --------------------------------------------------------------------------- QUICK admission: QK1 - QK9
# score: sn=None / sf=fresh / se=expired;  pan: pf=fresh, else None or expired.

QUICK_ROWS = [
    pytest.param(
        make_facts(score_row=SCORE_FRESH, pan_row=PAN_FRESH),
        D(
            admission="cache_hit",
            pan_plan="reuse",
            bureau_plan="reuse",
            counts_against_quota=False,
            is_refresh=False,
            **echo(),
            checked_on=SCORE_CHECKED_ON,
            refresh_available_on=SCORE_REFRESH_ON,
            original_consent_id="con-score",
        ),
        id="QK1-cache-hit",
    ),
    pytest.param(
        make_facts(score_row=_score(SCORE_FRESH_AT, consent_id=None), pan_row=PAN_FRESH),
        D(
            admission="cache_hit",
            **echo(),
            checked_on=SCORE_CHECKED_ON,
            refresh_available_on=SCORE_REFRESH_ON,
            original_consent_id=None,
        ),
        id="QK1-cache-hit-consent-unknown",
    ),
    pytest.param(
        make_facts(score_row=SCORE_FRESH, pan_row=None, surepass_live=False),
        D(code="LIVE_CALLS_DISABLED", **echo()),
        id="QK2-no-pan-live-off",
    ),
    pytest.param(
        make_facts(score_row=SCORE_FRESH, pan_row=PAN_EXPIRED, surepass_live=False),
        D(code="LIVE_CALLS_DISABLED", **echo()),
        id="QK2-expired-pan-live-off",
    ),
    pytest.param(
        make_facts(score_row=SCORE_FRESH, pan_row=None),
        D(
            admission="cache_hit",
            pan_plan="call",
            bureau_plan="reuse",
            counts_against_quota=False,
            **echo(),
            checked_on=SCORE_CHECKED_ON,
            refresh_available_on=SCORE_REFRESH_ON,
            original_consent_id="con-score",
        ),
        id="QK3-cache-hit-pan-missing",
    ),
    pytest.param(
        make_facts(score_row=SCORE_FRESH, pan_row=PAN_EXPIRED),
        D(
            admission="cache_hit",
            pan_plan="call",
            bureau_plan="reuse",
            **echo(),
            checked_on=SCORE_CHECKED_ON,
            refresh_available_on=SCORE_REFRESH_ON,
            original_consent_id="con-score",
        ),
        id="QK3-cache-hit-pan-expired",
    ),
    pytest.param(
        make_facts(quota_used=5, quota_limit=5),
        D(code="QUOTA_EXHAUSTED", **echo(5, 5)),
        id="QK4-at-limit",
    ),
    pytest.param(
        make_facts(quota_used=7, quota_limit=5, pan_row=PAN_FRESH),
        D(code="QUOTA_EXHAUSTED", **echo(7, 5)),
        id="QK4-over-limit-pan-fresh",
    ),
    pytest.param(
        make_facts(quota_used=8, quota_limit=8),
        D(code="QUOTA_EXHAUSTED", **echo(8, 8)),
        id="QK4-override-limit",
    ),
    pytest.param(
        make_facts(quota_used=0, quota_limit=0),
        D(code="QUOTA_EXHAUSTED", **echo(0, 0)),
        id="QK4-limit-zero",
    ),
    pytest.param(
        make_facts(score_row=SCORE_EXPIRED, pan_row=PAN_FRESH, quota_used=5, quota_limit=5),
        D(
            admission="refresh",
            pan_plan="reuse",
            bureau_plan="call",
            counts_against_quota=False,
            is_refresh=True,
            **echo(5, 5),
        ),
        id="QK5-expired-score-exempt-from-exhausted-quota",
    ),
    pytest.param(
        make_facts(score_row=SCORE_EXPIRED, pan_row=None, quota_used=9, quota_limit=5),
        D(
            admission="refresh",
            pan_plan="call",
            bureau_plan="call",
            counts_against_quota=False,
            is_refresh=True,
            **echo(9, 5),
        ),
        id="QK5-expired-score-exempt-over-limit",
    ),
    pytest.param(
        make_facts(bureau_calls_today=3),
        D(code="PAN_DAILY_CAP", **echo(), **CAP_DIRECTIVES),
        id="QK6-cap-reached-no-score",
    ),
    pytest.param(
        make_facts(bureau_calls_today=4, score_row=SCORE_EXPIRED, pan_row=PAN_FRESH),
        D(code="PAN_DAILY_CAP", **echo(), **CAP_DIRECTIVES),
        id="QK6-cap-exceeded-expired-score",
    ),
    pytest.param(
        make_facts(bureau_calls_today=2, reconciliation_pending=True),
        D(code="PAN_LOCKED", **echo()),
        id="QK7-reconciliation-no-score",
    ),
    pytest.param(
        make_facts(reconciliation_pending=True, score_row=SCORE_EXPIRED, pan_row=PAN_FRESH),
        D(code="PAN_LOCKED", **echo()),
        id="QK7-reconciliation-expired-score",
    ),
    pytest.param(
        make_facts(surepass_live=False),
        D(code="LIVE_CALLS_DISABLED", **echo()),
        id="QK8-kill-switch-no-score",
    ),
    pytest.param(
        make_facts(surepass_live=False, score_row=SCORE_EXPIRED, pan_row=PAN_FRESH),
        D(code="LIVE_CALLS_DISABLED", **echo()),
        id="QK8-kill-switch-expired-score",
    ),
    pytest.param(
        make_facts(),
        D(
            admission="new",
            pan_plan="call",
            bureau_plan="call",
            counts_against_quota=True,
            is_refresh=False,
            **echo(),
        ),
        id="QK9-new-no-pan",
    ),
    pytest.param(
        make_facts(pan_row=PAN_FRESH),
        D(admission="new", pan_plan="reuse", bureau_plan="call", counts_against_quota=True, **echo()),
        id="QK9-new-pan-fresh",
    ),
    pytest.param(
        make_facts(pan_row=PAN_EXPIRED),
        D(admission="new", pan_plan="call", bureau_plan="call", counts_against_quota=True, **echo()),
        id="QK9-new-pan-expired",
    ),
    pytest.param(
        make_facts(quota_used=4, quota_limit=5, bureau_calls_today=2),
        D(admission="new", pan_plan="call", bureau_plan="call", counts_against_quota=True, **echo(4, 5)),
        id="QK9-new-last-slot",
    ),
    pytest.param(
        make_facts(score_row=SCORE_EXPIRED, pan_row=PAN_FRESH),
        D(
            admission="refresh",
            pan_plan="reuse",
            bureau_plan="call",
            counts_against_quota=False,
            is_refresh=True,
            **echo(),
        ),
        id="QK9-refresh-pan-fresh",
    ),
    pytest.param(
        make_facts(score_row=SCORE_EXPIRED, pan_row=PAN_EXPIRED),
        D(admission="refresh", pan_plan="call", bureau_plan="call", is_refresh=True, **echo()),
        id="QK9-refresh-pan-expired",
    ),
    pytest.param(
        make_facts(score_row=SCORE_EXPIRED, pan_row=None),
        D(admission="refresh", pan_plan="call", bureau_plan="call", is_refresh=True, **echo()),
        id="QK9-refresh-no-pan",
    ),
]


@pytest.mark.parametrize(("facts", "expected"), QUICK_ROWS)
def test_admit_quick_table(facts: AdmissionFacts, expected: AdmissionDecision) -> None:
    assert admit_quick(facts) == expected


def test_quick_report_row_is_ignored() -> None:
    """The node supplies report_row=None for quick; a stray row must not change the outcome."""
    base = make_facts(score_row=SCORE_FRESH, pan_row=PAN_FRESH)
    with_report = make_facts(score_row=SCORE_FRESH, pan_row=PAN_FRESH, report_row=REPORT_FRESH)
    assert admit_quick(with_report) == admit_quick(base)


def test_quick_counts_only_when_there_is_no_score_row_of_any_age() -> None:
    for pan in (None, PAN_FRESH, PAN_EXPIRED):
        assert admit_quick(make_facts(pan_row=pan)).counts_against_quota is True
        for score in (SCORE_FRESH, SCORE_EXPIRED, SCORE_NTC, SCORE_NTC_EXPIRED):
            assert admit_quick(make_facts(pan_row=pan, score_row=score)).counts_against_quota is False


def test_quick_score_row_expiring_exactly_now_is_expired() -> None:
    expiring = make_score_cache(fetched_at=T0 - SCORE_WINDOW, expires_at=T0)
    decision = admit_quick(make_facts(score_row=expiring, pan_row=PAN_FRESH))
    assert (decision.admission, decision.is_refresh, decision.counts_against_quota) == ("refresh", True, False)
    one_second_left = make_score_cache(fetched_at=T0 - SCORE_WINDOW, expires_at=T0 + timedelta(seconds=1))
    assert admit_quick(make_facts(score_row=one_second_left, pan_row=PAN_FRESH)).admission == "cache_hit"


def test_quick_pan_row_expiring_exactly_now_is_not_fresh() -> None:
    expiring = make_pan_record(fetched_at=T0 - PAN_WINDOW, expires_at=T0)
    decision = admit_quick(make_facts(pan_row=expiring))
    assert decision.pan_plan == "call"


# --------------------------------------------------------------------------- FULL admission: FL1 - FL9

NO_ACK = FullAdmissionInputs(refresh=False, acknowledge_ntc_warning=False)
ACK = FullAdmissionInputs(refresh=False, acknowledge_ntc_warning=True)
REFRESH = FullAdmissionInputs(refresh=True, acknowledge_ntc_warning=False)
REFRESH_ACK = FullAdmissionInputs(refresh=True, acknowledge_ntc_warning=True)

REPORT_HIT_DISPLAY: dict[str, Any] = {
    "checked_on": REPORT_CHECKED_ON,
    "refresh_available_on": REPORT_REFRESH_ON,
    "original_consent_id": "con-report",
}

FULL_ROWS = [
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=PAN_FRESH),
        NO_ACK,
        D(
            admission="cache_hit",
            pan_plan="reuse",
            bureau_plan="reuse",
            counts_against_quota=False,
            is_refresh=False,
            **REPORT_HIT_DISPLAY,
        ),
        id="FL1-cache-hit",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=PAN_FRESH, bureau_calls_today=3, reconciliation_pending=True),
        ACK,
        D(admission="cache_hit", **REPORT_HIT_DISPLAY),
        id="FL1-cache-hit-beats-cap-and-reconciliation",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=PAN_FRESH),
        REFRESH,
        D(code="REPORT_WINDOW_OPEN", refresh_available_on=REPORT_REFRESH_ON),
        id="FL2-forced-refresh-in-window-pan-fresh",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=None, surepass_live=False),
        REFRESH_ACK,
        D(code="REPORT_WINDOW_OPEN", refresh_available_on=REPORT_REFRESH_ON),
        id="FL2-forced-refresh-beats-kill-switch",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=PAN_EXPIRED, bureau_calls_today=3),
        REFRESH,
        D(code="REPORT_WINDOW_OPEN", refresh_available_on=REPORT_REFRESH_ON),
        id="FL2-forced-refresh-beats-cap",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=None, surepass_live=False),
        NO_ACK,
        D(code="LIVE_CALLS_DISABLED"),
        id="FL3-no-pan-live-off",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=PAN_EXPIRED, surepass_live=False),
        NO_ACK,
        D(code="LIVE_CALLS_DISABLED"),
        id="FL3-expired-pan-live-off",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=None),
        NO_ACK,
        D(admission="cache_hit", pan_plan="call", bureau_plan="reuse", **REPORT_HIT_DISPLAY),
        id="FL4-cache-hit-pan-missing",
    ),
    pytest.param(
        full_facts(report_row=REPORT_FRESH, pan_row=PAN_EXPIRED),
        NO_ACK,
        D(admission="cache_hit", pan_plan="call", bureau_plan="reuse", **REPORT_HIT_DISPLAY),
        id="FL4-cache-hit-pan-expired",
    ),
    pytest.param(
        full_facts(bureau_calls_today=3),
        NO_ACK,
        D(code="PAN_DAILY_CAP", **CAP_DIRECTIVES),
        id="FL5-cap-no-report",
    ),
    pytest.param(
        full_facts(report_row=REPORT_EXPIRED, pan_row=PAN_FRESH, bureau_calls_today=5),
        REFRESH,
        D(code="PAN_DAILY_CAP", **CAP_DIRECTIVES),
        id="FL5-cap-expired-report",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC, bureau_calls_today=3),
        NO_ACK,
        D(code="PAN_DAILY_CAP", **CAP_DIRECTIVES),
        id="FL5-cap-before-ntc-warning",
    ),
    pytest.param(
        full_facts(reconciliation_pending=True),
        NO_ACK,
        D(code="PAN_LOCKED"),
        id="FL6-reconciliation-no-report",
    ),
    pytest.param(
        full_facts(report_row=REPORT_EXPIRED, reconciliation_pending=True, bureau_calls_today=2),
        ACK,
        D(code="PAN_LOCKED"),
        id="FL6-reconciliation-expired-report",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC, reconciliation_pending=True),
        NO_ACK,
        D(code="PAN_LOCKED"),
        id="FL6-reconciliation-before-ntc-warning",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC),
        NO_ACK,
        D(code="WARN_NTC_RECENT", ntc_found_on=NTC_FOUND_ON),
        id="FL7-ntc-warning-no-report",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC, report_row=REPORT_EXPIRED, pan_row=PAN_FRESH),
        REFRESH,
        D(code="WARN_NTC_RECENT", ntc_found_on=NTC_FOUND_ON),
        id="FL7-ntc-warning-expired-report",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC, surepass_live=False),
        NO_ACK,
        D(code="WARN_NTC_RECENT", ntc_found_on=NTC_FOUND_ON),
        id="FL7-ntc-warning-before-kill-switch",
    ),
    pytest.param(
        full_facts(surepass_live=False),
        NO_ACK,
        D(code="LIVE_CALLS_DISABLED"),
        id="FL8-kill-switch-no-report",
    ),
    pytest.param(
        full_facts(report_row=REPORT_EXPIRED, pan_row=PAN_FRESH, surepass_live=False),
        NO_ACK,
        D(code="LIVE_CALLS_DISABLED"),
        id="FL8-kill-switch-expired-report",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC, surepass_live=False),
        ACK,
        D(code="LIVE_CALLS_DISABLED"),
        id="FL8-kill-switch-after-ntc-acknowledged",
    ),
    pytest.param(
        full_facts(),
        NO_ACK,
        D(admission="new", pan_plan="call", bureau_plan="call", counts_against_quota=False, is_refresh=False),
        id="FL9-new-no-report-no-pan",
    ),
    pytest.param(
        full_facts(pan_row=PAN_FRESH),
        NO_ACK,
        D(admission="new", pan_plan="reuse", bureau_plan="call"),
        id="FL9-new-pan-fresh",
    ),
    pytest.param(
        full_facts(report_row=REPORT_EXPIRED, pan_row=PAN_FRESH),
        NO_ACK,
        D(admission="refresh", pan_plan="reuse", bureau_plan="call", counts_against_quota=False, is_refresh=True),
        id="FL9-refresh-expired-report",
    ),
    pytest.param(
        full_facts(report_row=REPORT_EXPIRED, pan_row=PAN_EXPIRED),
        REFRESH,
        D(admission="refresh", pan_plan="call", bureau_plan="call", is_refresh=True),
        id="FL9-refresh-flag-true-with-expired-report",
    ),
    pytest.param(
        full_facts(pan_row=PAN_FRESH),
        REFRESH,
        D(admission="new", pan_plan="reuse", bureau_plan="call", is_refresh=False),
        id="FL9-refresh-flag-true-with-nothing-to-refresh-is-new",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC, pan_row=PAN_FRESH),
        ACK,
        D(admission="new", pan_plan="reuse", bureau_plan="call"),
        id="FL9-ntc-warning-acknowledged",
    ),
    pytest.param(
        full_facts(score_row=SCORE_NTC_EXPIRED),
        NO_ACK,
        D(admission="new", pan_plan="call", bureau_plan="call"),
        id="FL9-expired-ntc-score-gives-no-warning",
    ),
    pytest.param(
        full_facts(score_row=SCORE_FRESH),
        NO_ACK,
        D(admission="new", pan_plan="call", bureau_plan="call"),
        id="FL9-fresh-non-ntc-score-gives-no-warning",
    ),
    pytest.param(
        make_facts(quota_used=9, quota_limit=5, bureau_calls_today=2),
        NO_ACK,
        D(admission="new", pan_plan="call", bureau_plan="call", counts_against_quota=False),
        id="FL9-report-pulls-are-outside-the-quick-quota",
    ),
]


@pytest.mark.parametrize(("facts", "full", "expected"), FULL_ROWS)
def test_admit_full_table(facts: AdmissionFacts, full: FullAdmissionInputs, expected: AdmissionDecision) -> None:
    assert admit_full(facts, full) == expected


def test_full_decisions_never_echo_quota() -> None:
    for facts, full in [(full_facts(), NO_ACK), (full_facts(bureau_calls_today=3), NO_ACK)]:
        decision = admit_full(facts, full)
        assert decision.quota_used_today is None and decision.quota_limit is None


def test_full_report_pulls_never_count_against_quota() -> None:
    for pan, report in itertools.product((None, PAN_FRESH, PAN_EXPIRED), (None, REPORT_FRESH, REPORT_EXPIRED)):
        for full in (NO_ACK, ACK, REFRESH):
            assert admit_full(full_facts(pan_row=pan, report_row=report), full).counts_against_quota is False


def test_ntc_recent_requires_a_not_yet_expired_ntc_score() -> None:
    expiring_now = make_score_cache(fetched_at=T0 - SCORE_WINDOW, expires_at=T0, is_ntc=True)
    decision = admit_full(full_facts(score_row=expiring_now), NO_ACK)
    assert decision.code is None and decision.admission == "new"  # expires_at == now is expired
    one_second = make_score_cache(fetched_at=T0 - SCORE_WINDOW, expires_at=T0 + timedelta(seconds=1), is_ntc=True)
    assert admit_full(full_facts(score_row=one_second), NO_ACK).code == "WARN_NTC_RECENT"


def test_ntc_found_on_is_the_ist_date_of_the_score_pull() -> None:
    late_utc = _score(datetime(2026, 10, 3, 19, 0, tzinfo=UTC), is_ntc=True)  # 2026-10-04 00:30 IST
    decision = admit_full(full_facts(score_row=late_utc), NO_ACK)
    assert decision.ntc_found_on == date(2026, 10, 4)


# --------------------------------------------------------------------------- precedence pins

WORST_CASE: dict[str, Any] = {
    "quota_used": 5,
    "quota_limit": 5,
    "bureau_calls_today": 3,
    "reconciliation_pending": True,
    "surepass_live": False,
}


def test_quick_cache_hit_beats_quota_cap_reconciliation_and_kill_switch() -> None:
    """QK1: a fresh score + fresh PAN is free to serve however exhausted/capped/locked the PAN or LC is."""
    decision = admit_quick(make_facts(score_row=SCORE_FRESH, pan_row=PAN_FRESH, **WORST_CASE))
    assert (decision.admission, decision.code, decision.bureau_plan) == ("cache_hit", None, "reuse")
    assert decision.create_daily_cap_lock is False  # a cache hit never triggers the cap directive


def test_quick_cache_hit_needing_a_pan_call_beats_everything_billed_but_not_the_kill_switch() -> None:
    worst_live = {**WORST_CASE, "surepass_live": True}
    decision = admit_quick(make_facts(score_row=SCORE_FRESH, pan_row=None, **worst_live))
    assert (decision.admission, decision.pan_plan, decision.bureau_plan) == ("cache_hit", "call", "reuse")
    killed = admit_quick(make_facts(score_row=SCORE_FRESH, pan_row=None, **WORST_CASE))
    assert killed.code == "LIVE_CALLS_DISABLED"  # QK2: the PAN re-verify is a live call


def test_full_cache_hit_beats_cap_and_reconciliation() -> None:
    decision = admit_full(make_facts(report_row=REPORT_FRESH, pan_row=PAN_FRESH, **WORST_CASE), ACK)
    assert decision.admission == "cache_hit"


def test_quick_quota_is_checked_before_the_cap_and_cap_directives_do_not_fire() -> None:
    decision = admit_quick(make_facts(quota_used=5, quota_limit=5, bureau_calls_today=3, reconciliation_pending=True))
    assert decision.code == "QUOTA_EXHAUSTED"
    assert decision.create_daily_cap_lock is False and decision.review_reason is None


def test_cap_comes_before_reconciliation_and_the_kill_switch() -> None:
    quick = admit_quick(make_facts(bureau_calls_today=3, reconciliation_pending=True, surepass_live=False))
    assert quick.code == "PAN_DAILY_CAP" and quick.create_daily_cap_lock is True
    full = admit_full(full_facts(bureau_calls_today=3, reconciliation_pending=True, surepass_live=False), NO_ACK)
    assert full.code == "PAN_DAILY_CAP" and full.create_daily_cap_lock is True


def test_reconciliation_comes_before_the_kill_switch() -> None:
    assert admit_quick(make_facts(reconciliation_pending=True, surepass_live=False)).code == "PAN_LOCKED"
    assert admit_full(full_facts(reconciliation_pending=True, surepass_live=False), NO_ACK).code == "PAN_LOCKED"


def test_kill_switch_is_last_and_fires_only_when_everything_else_would_admit() -> None:
    assert admit_quick(make_facts(surepass_live=False)).code == "LIVE_CALLS_DISABLED"
    assert admit_quick(make_facts(surepass_live=True)).admission == "new"
    assert admit_full(full_facts(surepass_live=False), NO_ACK).code == "LIVE_CALLS_DISABLED"
    assert admit_full(full_facts(surepass_live=True), NO_ACK).admission == "new"


def test_full_cap_comes_before_the_ntc_warning_in_the_overlap_case() -> None:
    """FL5 vs FL7: capped PAN with a recent NTC score and no acknowledgement -> PAN_DAILY_CAP, not the warning."""
    decision = admit_full(full_facts(score_row=SCORE_NTC, bureau_calls_today=3), NO_ACK)
    assert decision.code == "PAN_DAILY_CAP"
    assert decision.ntc_found_on is None
    assert (decision.create_daily_cap_lock, decision.review_reason) == (True, "pan_daily_cap")


def test_cap_boundary_is_three_billed_calls() -> None:
    assert admit_quick(make_facts(bureau_calls_today=2)).admission == "new"
    assert admit_quick(make_facts(bureau_calls_today=PAN_DAILY_BUREAU_CAP)).code == "PAN_DAILY_CAP"
    assert admit_full(full_facts(bureau_calls_today=2), NO_ACK).admission == "new"
    assert admit_full(full_facts(bureau_calls_today=PAN_DAILY_BUREAU_CAP), NO_ACK).code == "PAN_DAILY_CAP"


@pytest.mark.parametrize("later", [timedelta(minutes=10), timedelta(minutes=11), timedelta(hours=6), timedelta(days=3)])
def test_reconciliation_blocks_after_the_timeout_lock_has_expired(later: timedelta) -> None:
    """K2: the 10-minute lock expiring never re-admits. The gate is the unresolved review row, independent of time."""
    assert later >= TIMEOUT_LOCK_TTL
    quick = admit_quick(make_facts(now=T0 + later, reconciliation_pending=True))
    full = admit_full(full_facts(now=T0 + later, reconciliation_pending=True), NO_ACK)
    assert quick.code == "PAN_LOCKED" and full.code == "PAN_LOCKED"


def test_a_resolved_reconciliation_readmits_regardless_of_elapsed_time() -> None:
    assert admit_quick(make_facts(now=T0 + timedelta(minutes=11), reconciliation_pending=False)).admission == "new"
    assert admit_full(full_facts(reconciliation_pending=False), NO_ACK).admission == "new"


def test_decisions_are_pure_functions_of_their_inputs() -> None:
    facts = make_facts(score_row=SCORE_EXPIRED, pan_row=PAN_FRESH)
    first = admit_quick(facts)
    assert admit_quick(facts) == first
    assert admit_quick(dataclasses.replace(facts)) == first
    shifted = dataclasses.replace(facts, now=T0 + 90 * DAY)  # `now` is the only clock: the PAN row (80d left) has expired
    assert admit_quick(shifted).pan_plan == "call" and first.pan_plan == "reuse"


# --------------------------------------------------------------------------- exactly-one-of invariant (T1 note) sweep

SCORE_STATES = [None, SCORE_FRESH, SCORE_EXPIRED, SCORE_NTC]
PAN_STATES = [None, PAN_FRESH, PAN_EXPIRED]
REPORT_STATES = [None, REPORT_FRESH, REPORT_EXPIRED]


def _decision_invariants(d: AdmissionDecision, facts: AdmissionFacts, *, quick: bool) -> None:
    # exactly one of (admission, code)
    assert (d.admission is None) != (d.code is None), d
    if d.code is not None:
        assert d.code in REFUSAL_CODES  # never a mid-run code
        assert (d.pan_plan, d.bureau_plan, d.counts_against_quota, d.is_refresh) == ("reuse", "reuse", False, False)
        assert d.checked_on is None and d.original_consent_id is None
        assert (d.refresh_available_on is not None) == (d.code == "REPORT_WINDOW_OPEN")
    else:
        assert d.admission in ("cache_hit", "new", "refresh")
        assert d.ntc_found_on is None and not d.create_daily_cap_lock and d.review_reason is None
        assert (d.bureau_plan == "call") == (d.admission in ("new", "refresh"))
        assert d.is_refresh == (d.admission == "refresh")
        if d.admission == "cache_hit":
            assert d.counts_against_quota is False and d.checked_on is not None and d.refresh_available_on is not None
        else:
            assert d.checked_on is None and d.refresh_available_on is None and d.original_consent_id is None
        # pan_plan == 'call' exactly when the PAN record is missing/expired
        pan_fresh = facts.pan_row is not None and is_fresh(facts.pan_row.expires_at, facts.now)
        assert (d.pan_plan == "call") == (not pan_fresh)
        # anything that calls out requires the kill switch off
        if d.pan_plan == "call" or d.bureau_plan == "call":
            assert facts.surepass_live is True
    # directive coupling
    assert d.create_daily_cap_lock == (d.code == "PAN_DAILY_CAP")
    assert (d.review_reason == "pan_daily_cap") == (d.code == "PAN_DAILY_CAP")
    assert (d.ntc_found_on is not None) == (d.code == "WARN_NTC_RECENT")
    # the quota echo is quick-only and present on EVERY quick path
    if quick:
        assert (d.quota_used_today, d.quota_limit) == (facts.quota_used, facts.quota_limit)
        assert d.counts_against_quota == (d.admission in ("new", "refresh") and facts.score_row is None)
    else:
        assert d.quota_used_today is None and d.quota_limit is None and d.counts_against_quota is False


def test_every_quick_outcome_sets_exactly_one_of_admission_or_code() -> None:
    combos = itertools.product(SCORE_STATES, PAN_STATES, [0, 5], [0, 3], [False, True], [False, True])
    count = 0
    for score, pan, used, bureau, recon, live in combos:
        facts = make_facts(
            score_row=score,
            pan_row=pan,
            quota_used=used,
            quota_limit=5,
            bureau_calls_today=bureau,
            reconciliation_pending=recon,
            surepass_live=live,
        )
        _decision_invariants(admit_quick(facts), facts, quick=True)
        count += 1
    assert count == 4 * 3 * 2 * 2 * 2 * 2


def test_every_full_outcome_sets_exactly_one_of_admission_or_code() -> None:
    combos = itertools.product(
        SCORE_STATES, PAN_STATES, REPORT_STATES, [0, 3], [False, True], [False, True], [False, True], [False, True]
    )
    count = 0
    for score, pan, report, bureau, recon, live, refresh, ack in combos:
        facts = full_facts(
            score_row=score,
            pan_row=pan,
            report_row=report,
            bureau_calls_today=bureau,
            reconciliation_pending=recon,
            surepass_live=live,
        )
        decision = admit_full(facts, FullAdmissionInputs(refresh=refresh, acknowledge_ntc_warning=ack))
        _decision_invariants(decision, facts, quick=False)
        count += 1
    assert count == 4 * 3 * 3 * 2 * 2 * 2 * 2 * 2


def test_every_expected_row_in_this_file_satisfies_the_invariant_too() -> None:
    for param in QUICK_ROWS + FULL_ROWS:
        expected = cast(AdmissionDecision, param.values[-1])
        assert (expected.admission is None) != (expected.code is None), param.id


def test_each_quick_and_full_row_id_is_covered() -> None:
    covered = {str(p.id).split("-")[0] for p in QUICK_ROWS + FULL_ROWS}
    assert covered >= {f"QK{i}" for i in range(1, 10)} | {f"FL{i}" for i in range(1, 10)}


# --------------------------------------------------------------------------- PAN-correction policy (PRD 5.13)


def _events(*specs: tuple[str, str]) -> list[Any]:
    return [
        make_pan_event(id=f"ev-{i}", pan_hash=pan, event=event, created_at=T0 + timedelta(minutes=i))
        for i, (event, pan) in enumerate(specs)
    ]


@pytest.mark.parametrize(
    ("events", "report_exists", "request_pan", "expected"),
    [
        pytest.param([], False, PAN_A, "first_pan", id="empty-events"),
        pytest.param([], True, PAN_A, "first_pan", id="empty-events-report-flag-irrelevant"),
        pytest.param(_events(("initial", PAN_A)), False, PAN_A, "same_pan", id="same-pan"),
        pytest.param(_events(("initial", PAN_A)), True, PAN_A, "same_pan", id="same-pan-beats-report-exists"),
        pytest.param(_events(("initial", PAN_A)), True, PAN_B, "needs_manager", id="report-exists-pan-differs"),
        pytest.param(
            _events(("initial", PAN_A), ("correction", PAN_B)), False, "c" * 64, "needs_manager", id="correction-spent"
        ),
        pytest.param(_events(("initial", PAN_A)), False, PAN_B, "record_correction", id="first-correction"),
        pytest.param(
            _events(("initial", PAN_A), ("correction", PAN_B)), False, PAN_B, "same_pan", id="latest-event-is-the-pan"
        ),
        pytest.param(
            _events(("initial", PAN_A), ("correction", PAN_B)),
            False,
            PAN_A,
            "needs_manager",
            id="reverting-to-the-original-pan-is-a-second-change",
        ),
        pytest.param(
            _events(("initial", PAN_A), ("correction", PAN_B)), True, PAN_B, "same_pan", id="same-pan-after-correction"
        ),
    ],
)
def test_evaluate_pan_change_table(events: list[Any], report_exists: bool, request_pan: str, expected: str) -> None:
    assert evaluate_pan_change(events, report_exists, request_pan) == expected


def test_evaluate_pan_change_uses_the_latest_event_by_created_at_not_list_order() -> None:
    older = make_pan_event(id="ev-a", pan_hash=PAN_A, event="initial", created_at=T0)
    newer = make_pan_event(id="ev-b", pan_hash=PAN_B, event="correction", created_at=T0 + timedelta(minutes=1))
    assert evaluate_pan_change([newer, older], False, PAN_B) == "same_pan"
    assert evaluate_pan_change([newer, older], False, PAN_A) == "needs_manager"


def test_evaluate_pan_change_does_not_mutate_its_input() -> None:
    events = _events(("initial", PAN_A), ("correction", PAN_B))
    snapshot = list(events)
    evaluate_pan_change(events, False, PAN_A)
    assert events == snapshot


# --------------------------------------------------------------------------- recompute admission R1 - R5


@pytest.mark.parametrize(
    ("status", "code", "kind", "pan_row", "bureau_row", "expected"),
    [
        pytest.param(None, None, "quick", True, True, "not_found", id="R1-missing-check"),
        pytest.param(None, None, "full", False, False, "not_found", id="R1-missing-check-beats-everything"),
        # T11 (ruling 12): a missing check has no kind, so R1 takes None rather than a dummy kind.
        pytest.param(None, None, None, False, False, "not_found", id="R1-missing-check-no-kind"),
        pytest.param("ok", None, "quick", True, True, "ok", id="R5-ok-quick"),
        pytest.param("ok", None, "full", True, True, "ok", id="R5-ok-full"),
        pytest.param("ok", "NO_PINCODE_ON_PAN", "quick", True, True, "ok", id="R5-ok-no-pincode-code"),
        pytest.param("service_failure", "OPENAI_FAILURE", "quick", True, True, "ok", id="R5-openai-failure-quick"),
        pytest.param("service_failure", "OPENAI_FAILURE", "full", True, True, "ok", id="R5-openai-failure-full"),
        pytest.param("service_failure", "SUREPASS_FAILURE", "quick", True, True, "not_recomputable", id="R2-surepass"),
        pytest.param("service_failure", None, "quick", True, True, "not_recomputable", id="R2-service-failure-no-code"),
        pytest.param("refused", "QUOTA_EXHAUSTED", "quick", True, True, "not_recomputable", id="R2-refused"),
        pytest.param("pan_not_verified", None, "quick", True, True, "not_recomputable", id="R2-pan-not-verified"),
        pytest.param("pending_verification", None, "full", True, True, "not_recomputable", id="R2-pending"),
        pytest.param("running", None, "quick", True, True, "not_recomputable", id="R2-running"),
        pytest.param("error", None, "quick", True, True, "not_recomputable", id="R2-error"),
        pytest.param("refused", "PAN_LOCKED", "full", False, False, "not_recomputable", id="R2-beats-raw-purged"),
        pytest.param("ok", None, "quick", False, True, "raw_unavailable", id="R3-quick-pan-record-missing"),
        pytest.param("ok", None, "quick", True, False, "raw_unavailable", id="R3-quick-score-cache-missing"),
        pytest.param("ok", None, "quick", False, False, "raw_unavailable", id="R3-quick-both-missing"),
        pytest.param("ok", None, "full", False, True, "raw_unavailable", id="R4-full-pan-record-missing"),
        pytest.param("ok", None, "full", True, False, "raw_unavailable", id="R4-full-report-cache-missing"),
        pytest.param(
            "service_failure", "OPENAI_FAILURE", "quick", True, False, "raw_unavailable", id="R3-openai-failure-purged"
        ),
    ],
)
def test_admit_recompute_table(
    status: str | None, code: str | None, kind: Kind | None, pan_row: bool, bureau_row: bool, expected: str
) -> None:
    assert admit_recompute(status, code, kind, pan_row, bureau_row) == expected


def test_recompute_does_not_require_freshness_inputs() -> None:
    """R5: stored raw is re-derived whatever its age; the function takes presence flags only, never a clock."""
    params = list(inspect.signature(admit_recompute).parameters)
    assert params == ["check_status", "check_code", "kind", "pan_row_present", "bureau_row_present"]


# --------------------------------------------------------------------------- attach admission A1 - A5


@pytest.mark.parametrize(
    ("kind", "existing", "contact_id", "expected"),
    [
        pytest.param(None, None, "c-1", "not_found", id="A1-missing-check"),
        pytest.param(None, "c-1", "c-1", "not_found", id="A1-beats-everything"),
        pytest.param("full", None, "c-1", "not_quick", id="A2-full"),
        pytest.param("full", "c-1", "c-1", "not_quick", id="A2-beats-already-attached"),
        pytest.param("quick", "c-1", "c-1", "already_attached", id="A3-idempotent"),
        pytest.param("quick", "c-1", "c-2", "conflict", id="A4-different-contact"),
        pytest.param("quick", None, "c-1", "ok", id="A5-unattached"),
    ],
)
def test_admit_attach_table(kind: Kind | None, existing: str | None, contact_id: str, expected: str) -> None:
    assert admit_attach(kind, existing, contact_id) == expected


# --------------------------------------------------------------------------- persist-time formulas (K8)


def _call(
    endpoint: str = SCORE_ENDPOINT,
    *,
    http_status: int | None = 200,
    success: bool | None = True,
    error_kind: Any = None,
    call_id: str = "call-1",
) -> SurePassCallRow:
    return SurePassCallRow(
        id=call_id,
        check_id="chk-1",
        pan_hash=PAN_A,
        endpoint=endpoint,
        client_id="cid-1",
        http_status=http_status,
        success=success,
        message_code=None,
        latency_ms=120,
        billable=call_is_billable(http_status, success, error_kind),  # the sink's own predicate
        error_kind=error_kind,
        storage_key=None,
        called_at=T0,
        purge_after=T0 + timedelta(days=180),
        purged_at=None,
    )


PAN_OK = _call(PAN_ENDPOINT, call_id="pan-ok")
PAN_INVALID_STILL_BILLED = _call(PAN_ENDPOINT, http_status=200, success=True, call_id="pan-invalid")
SCORE_OK = _call(SCORE_ENDPOINT, call_id="score-ok")
SCORE_NTC_CALL = _call(SCORE_ENDPOINT, call_id="score-ntc")  # 200 + success:true with a non-3-digit score
SCORE_TIMEOUT = _call(SCORE_ENDPOINT, http_status=None, success=None, error_kind="timeout", call_id="score-timeout")
SCORE_UPSTREAM = _call(SCORE_ENDPOINT, http_status=500, success=None, error_kind="upstream", call_id="score-500")
SCORE_FAILED = _call(SCORE_ENDPOINT, http_status=200, success=False, error_kind="upstream", call_id="score-false")
REPORT_OK = _call(REPORT_ENDPOINT, call_id="report-ok")


def test_compute_billed() -> None:
    assert compute_billed([]) is False
    assert compute_billed([SCORE_UPSTREAM]) is False
    assert compute_billed([SCORE_FAILED]) is False
    assert compute_billed([PAN_OK]) is True  # the Rs 1 PAN call counts
    assert compute_billed([SCORE_OK]) is True
    assert compute_billed([SCORE_UPSTREAM, REPORT_OK]) is True
    assert compute_billed([SCORE_TIMEOUT]) is True  # spend unknown -> conservatively billed
    assert compute_billed(tuple([SCORE_OK])) is True  # any Sequence


def test_counted_against_quota_signature_is_keyword_only_and_status_free() -> None:
    params = inspect.signature(compute_counted_against_quota).parameters
    assert list(params) == ["kind", "counts_against_quota", "calls"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())  # no `status` to branch on


# (kind, admission.counts_against_quota, calls, expected counted, expected billed)
CONSEQUENCES = [
    pytest.param("quick", False, [], False, False, id="cache-hit"),
    pytest.param("quick", True, [PAN_INVALID_STILL_BILLED], False, True, id="pan-invalid-bureau-never-called"),
    pytest.param("quick", True, [PAN_OK, SCORE_OK], True, True, id="happy-path-new"),
    pytest.param("quick", True, [PAN_OK, SCORE_NTC_CALL], True, True, id="ntc-counts"),
    pytest.param("quick", False, [PAN_OK, SCORE_OK], False, True, id="expired-score-refresh-billed-not-counted"),
    pytest.param("quick", True, [PAN_OK, SCORE_TIMEOUT], False, True, id="timeout-not-counted-but-billed"),
    pytest.param("quick", True, [PAN_OK, SCORE_UPSTREAM], False, True, id="service-failure-5xx-not-counted"),
    pytest.param("quick", True, [SCORE_UPSTREAM], False, False, id="service-failure-nothing-billed"),
    pytest.param("quick", True, [PAN_OK, SCORE_FAILED], False, True, id="success-false-not-counted"),
    pytest.param(
        "quick", True, [PAN_OK, SCORE_OK], True, True, id="openai-failure-quick-still-counts-status-independent"
    ),
    pytest.param("quick", True, [SCORE_OK], True, True, id="score-only-reused-pan"),
    pytest.param("quick", True, [PAN_OK, REPORT_OK], False, True, id="a-report-call-is-not-a-score-call"),
    pytest.param("full", True, [PAN_OK, REPORT_OK, SCORE_OK], False, True, id="full-never-counts"),
    pytest.param("full", False, [REPORT_OK], False, True, id="full-report-billed-outside-quota"),
]


@pytest.mark.parametrize(("kind", "counts", "calls", "counted", "billed"), CONSEQUENCES)
def test_persist_formula_consequence_table(
    kind: Kind, counts: bool, calls: list[SurePassCallRow], counted: bool, billed: bool
) -> None:
    assert compute_counted_against_quota(kind=kind, counts_against_quota=counts, calls=calls) is counted
    assert compute_billed(calls) is billed


def test_counted_requires_the_admission_directive_even_with_a_billed_200_score_call() -> None:
    """counts_against_quota False (any stored score row existed) wins over a perfectly good score call."""
    assert compute_counted_against_quota(kind="quick", counts_against_quota=False, calls=[SCORE_OK]) is False


def test_counted_does_not_depend_on_call_row_billable_flag() -> None:
    """The formula reads (endpoint, http_status, success) - not the sink's `billable` column."""
    fudged = dataclasses.replace(SCORE_OK, billable=False)
    assert compute_counted_against_quota(kind="quick", counts_against_quota=True, calls=[fudged]) is True


@pytest.mark.parametrize(
    ("limit", "used_before", "counted", "expected"),
    [
        (5, 3, True, 1),  # "1 of 5 checks left today"
        (5, 3, False, 2),
        (8, 5, False, 3),  # override limit, cache hit
        (8, 5, True, 2),
        (7, 2, True, 4),  # override-limit 7
        (7, 7, False, 0),
        (5, 5, True, 0),  # floor at zero
        (5, 9, True, 0),
        (5, 0, False, 5),
        (0, 0, False, 0),
        (5, 4, True, 0),
    ],
)
def test_quick_checks_left_arithmetic(limit: int, used_before: int, counted: bool, expected: int) -> None:
    assert quick_checks_left(limit, used_before, counted) == expected


def test_quick_checks_left_matches_the_admission_echo_for_a_counting_run() -> None:
    decision = admit_quick(make_facts(quota_used=3, quota_limit=5))
    assert decision.counts_against_quota is True
    counted = compute_counted_against_quota(
        kind="quick", counts_against_quota=decision.counts_against_quota, calls=[PAN_OK, SCORE_OK]
    )
    assert quick_checks_left(decision.quota_limit or 0, decision.quota_used_today or 0, counted) == 1
