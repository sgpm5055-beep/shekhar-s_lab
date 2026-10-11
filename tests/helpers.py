"""Shared synthetic world for the end-to-end scenario suite (PRD v1.1 section 12): identities, the seed lenders, request
payloads, the API harness, repo seeding helpers and the LcView leak scanner.

Everything here is synthetic. The seed lenders below are the SINGLE source (K17): tests, eval datasets and
``scripts/seed_lenders.py`` import ``SEED_LENDERS`` from this module, so the three never drift. Criteria use the
``db/schema.sql`` jsonb key spellings ONLY (``min_score_floor`` / ``min_age`` / ``max_age_at_maturity``: T8 ruling).

The leak scanner IMPORTS its patterns from ``eligibility_agent.output.lc_view`` (no local regex copies) and skips the
two uuid-ish fields (``LcView.check_id`` and every lender ``id``): uuid4 hex can trip the score / account patterns
structurally (T9 ruling).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, cast
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eligibility_agent.config import Settings
from eligibility_agent.db.repo import (
    CheckRow,
    CheckStatus,
    MemoryRepo,
    PanLockRow,
    PanRecordRow,
    ReviewRow,
    ScoreCacheRow,
    SurePassCallRow,
    new_id,
    storage_key_for,
)
from eligibility_agent.graph.build import GraphDeps
from eligibility_agent.graph.state import LcView
from eligibility_agent.output.lc_view import GLOBAL_LEAK_PATTERNS, LLM_LEAK_PATTERNS, line_leaks
from eligibility_agent.policy.guards import (
    PAN_WINDOW,
    RAW_RETENTION,
    SCORE_WINDOW,
    hash_mobile,
    hash_pan,
    pan_last4,
)
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    SCORE_ENDPOINT,
    CallLogSink,
    FakeCall,
    FakeSurePassClient,
    SurePassResult,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing import scenarios
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.fake_llm import FakeRankModel

if TYPE_CHECKING:
    from httpx2 import Response  # starlette's TestClient is built on httpx2 (not the httpx the service uses)

# --------------------------------------------------------------------------- the canonical epoch and keys

IST: Final[ZoneInfo] = ZoneInfo("Asia/Kolkata")
T0: Final[datetime] = datetime(2026, 10, 10, 10, 0, tzinfo=IST)  # 04:30 UTC
HMAC_KEY: Final[str] = "test-hmac-key"  # conftest exports it as IDENTITY_HMAC_KEY too
TOKEN: Final[str] = "test-token"
AUTH: Final[dict[str, str]] = {"Authorization": f"Bearer {TOKEN}"}

# --------------------------------------------------------------------------- synthetic identities

PAN_OK: Final[str] = scenarios.PAN_OK  # "ABCPE1234F": person, valid, zip 560001
PAN_TWO: Final[str] = "FGHPJ5678K"
PAN_NTC: Final[str] = scenarios.PAN_NTC  # "KLMPN4321Q"
PAN_COMPANY: Final[str] = scenarios.PAN_COMPANY  # "AAACA9999Z"
PAN_INVALID: Final[str] = scenarios.PAN_INVALID  # "ZZZPZ0000Z"
MOBILE_OK: Final[str] = "9876543210"
# The PAN Comprehensive full_name every synthetic PAN envelope carries; bureau calls send THIS, never the typed name.
PAN_FULL_NAME: Final[str] = "RAVI TESTKUMAR"
# What the LC typed. Differs from PAN_FULL_NAME even case-folded, so the "bureau body carries full_name" pin bites.
LC_TYPED_NAME: Final[str] = "Ravindra K Typedname"
LC_A: Final[str] = "lc-a"
LC_B: Final[str] = "lc-b"
LEAD: Final[str] = "lead-1"
OPP: Final[str] = "opp-1"
OPP_TWO: Final[str] = "opp-2"

# Literals that must never reach an LcView string (PRD 6: no PAN Comprehensive field, no pincode, no identity).
IDENTITY_LITERALS: Final[tuple[str, ...]] = (
    PAN_OK,
    PAN_TWO,
    PAN_NTC,
    PAN_COMPANY,
    PAN_INVALID,
    MOBILE_OK,
    PAN_FULL_NAME,
    LC_TYPED_NAME,
    "XXXXXXXX0000",  # the envelopes' masked_aadhaar
    "1998-04-12",  # the envelopes' PAN dob
    "560001",
    "560002",
    "110001",
    "400001",
)


def pan_hash_of(pan: str) -> str:
    return hash_pan(pan, key=HMAC_KEY)


def mobile_hash_of(mobile: str = MOBILE_OK) -> str:
    return hash_mobile(mobile, key=HMAC_KEY)


def ist(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


# --------------------------------------------------------------------------- seed lenders (K17 single source)


def known(**values: Any) -> dict[str, dict[str, Any]]:
    """The ``lenders.criteria`` jsonb contract: every field ``{"value": ..., "status": "known"}``."""
    return {key: {"value": value, "status": "known"} for key, value in values.items()}


@dataclass(frozen=True, slots=True)
class SeedLender:
    """One published lender version + its approved pincodes, exactly as the admin tables hold it."""

    id: str
    name: str
    criteria: Mapping[str, Any]  # schema.sql spellings only
    policy_notes: str
    pincodes: tuple[str, ...]
    version: int = 1


SEED_LENDERS: Final[tuple[SeedLender, ...]] = (
    SeedLender(
        id="L1",
        name="Alpha Education Finance",
        criteria=known(
            min_score_floor=650,
            ntc_accepted=False,
            foir_limit=0.55,
            min_income=25000,
            min_age=21,
            max_age_at_maturity=58,
            rate_min_pct=11.0,
            rate_max_pct=14.0,
            tenure_min_months=12,
            tenure_max_months=48,
            loan_min=50000,
            loan_max=1500000,
        ),
        policy_notes="Prefers salaried applicants; collateral-free up to the cap.",
        pincodes=("560001", "560002", "110001"),
    ),
    SeedLender(
        id="L2",
        name="Bharat Upskill Credit",
        criteria=known(
            min_score_floor=650,
            ntc_accepted=True,
            foir_limit=0.45,
            min_income=20000,
            min_age=21,
            max_age_at_maturity=60,
            rate_min_pct=14.0,
            rate_max_pct=18.0,
            tenure_min_months=12,
            tenure_max_months=36,
            loan_min=25000,
            loan_max=500000,
        ),
        policy_notes="",
        pincodes=("560001", "110001"),
    ),
    SeedLender(
        id="L3",
        name="Crest Capital NBFC",
        criteria=known(
            min_score_floor=720,
            ntc_accepted=False,
            foir_limit=0.60,
            min_income=50000,
            min_age=23,
            max_age_at_maturity=55,
            rate_min_pct=10.5,
            rate_max_pct=12.5,
            tenure_min_months=24,
            tenure_max_months=60,
            loan_min=100000,
            loan_max=2500000,
        ),
        policy_notes="",
        pincodes=("400001",),
    ),
)


def seed_lender_row(repo: MemoryRepo, lender: SeedLender, *, published: bool = True) -> None:
    repo.seed_lender(
        lender.id,
        lender.version,
        lender.name,
        published,
        copy.deepcopy(dict(lender.criteria)),
        lender.policy_notes,
        lender.pincodes,
    )


def seed_default_lenders(repo: MemoryRepo) -> None:
    for lender in SEED_LENDERS:
        seed_lender_row(repo, lender)


def seed_lender(lender_id: str) -> SeedLender:
    return next(lender for lender in SEED_LENDERS if lender.id == lender_id)


# --------------------------------------------------------------------------- settings + payloads


def settings_aliases() -> frozenset[str]:
    out: set[str] = set()
    for info in Settings.model_fields.values():
        alias = info.validation_alias
        if not isinstance(alias, str):
            raise AssertionError("every Settings field carries its env name as a plain-str validation_alias")
        out.add(alias)
    return frozenset(out)


def make_settings(**env_values: Any) -> Settings:
    """Hermetic Settings from env-name kwargs (no .env file). Unknown names raise: a misspelt alias would otherwise be
    silently ignored and make a negative test vacuous (T3 note)."""
    unknown = set(env_values) - settings_aliases()
    if unknown:
        raise AssertionError(f"not Settings aliases: {sorted(unknown)}")
    return cast(Any, Settings)(_env_file=None, **env_values)


def api_settings(**over: Any) -> Settings:
    return make_settings(SERVICE_API_TOKEN=TOKEN, IDENTITY_HMAC_KEY=HMAC_KEY, **over)


def consent_obj(**over: Any) -> dict[str, Any]:
    return {"given": True, "wording_version": "v2-quick", "call_ref": "sc_call_123", "ip": "10.0.0.1", **over}


def quick_payload(**over: Any) -> dict[str, Any]:
    """The PRD 6 quick body. Pass ``key=OMIT`` to drop a key entirely (e.g. the no-income variant)."""
    body: dict[str, Any] = {
        "lc_id": LC_A,
        "lead_id": LEAD,
        "opportunity_id": None,
        "name": LC_TYPED_NAME,
        "mobile": MOBILE_OK,
        "pan": PAN_OK,
        "consent": consent_obj(),
        "monthly_income": 60000,
        "existing_emi": 0,
    }
    return _apply(body, over)


def full_payload(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "lc_id": LC_A,
        "lead_id": LEAD,
        "opportunity_id": OPP,
        "opportunity_stage": "offer",
        "name": LC_TYPED_NAME,
        "mobile": MOBILE_OK,
        "pan": PAN_OK,
        "consent": consent_obj(wording_version="v2-full"),
        "monthly_income": 60000,
        "existing_emi": 0,
    }
    return _apply(body, over)


class _Omit:
    """Sentinel: ``quick_payload(existing_emi=OMIT)`` removes the key (not the same as an explicit null)."""

    def __repr__(self) -> str:
        return "OMIT"


OMIT: Final[_Omit] = _Omit()


def _apply(body: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(body)
    for key, value in over.items():
        if value is OMIT:
            out.pop(key, None)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- SurePass fakes


class RecordingSurePass(Protocol):
    """A SurePass seam that also exposes the ordered call log (FakeSurePassClient and SwitchableSurePass)."""

    @property
    def calls_enabled(self) -> bool: ...

    @property
    def calls(self) -> list[FakeCall]: ...

    def endpoints_called(self) -> list[str]: ...

    def pan_comprehensive(self, check_id: str, pan_hash: str, pan: str) -> SurePassResult: ...

    def experian_score(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult: ...

    def experian_report(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult: ...


SurePassFactory = Callable[[CallLogSink], RecordingSurePass]


def endpoints(fake: RecordingSurePass) -> list[str]:
    return fake.endpoints_called()


class SwitchableSurePass:
    """A SurePass seam whose behaviour changes between requests (e.g. a Score that times out, then answers once the
    reconciliation is resolved). Phases are sequential, so the concatenated call log is in call order."""

    def __init__(self, first: FakeSurePassClient) -> None:
        self._phases: list[FakeSurePassClient] = [first]

    def switch_to(self, fake: FakeSurePassClient) -> None:
        self._phases.append(fake)

    @property
    def _current(self) -> FakeSurePassClient:
        return self._phases[-1]

    @property
    def calls_enabled(self) -> bool:
        return self._current.calls_enabled

    @property
    def calls(self) -> list[FakeCall]:
        return [call for phase in self._phases for call in phase.calls]

    def endpoints_called(self) -> list[str]:
        return [call.endpoint for call in self.calls]

    def pan_comprehensive(self, check_id: str, pan_hash: str, pan: str) -> SurePassResult:
        return self._current.pan_comprehensive(check_id, pan_hash, pan)

    def experian_score(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._current.experian_score(check_id, pan_hash, name=name, mobile=mobile, pan=pan)

    def experian_report(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._current.experian_report(check_id, pan_hash, name=name, mobile=mobile, pan=pan)


def happy_report_envelope() -> env.EnvelopePair:
    """The full_happy_reported_emi world's report: ONE open account, reported EMI 3500, ReportDate 20261001, score 745."""
    account = env.cais_account(
        scheduled_emi="3500",
        current_balance="150000",
        date_reported="20260920",
        holder_zip="560001",
        account_number="SYNTHACC0001",
    )
    return env.experian_report_envelope(accounts=[account], credit_score="745", report_date=scenarios.REPORT_DATE)


def quick_and_full_client(sink: CallLogSink, *, pan: str = PAN_OK) -> FakeSurePassClient:
    """PAN + Score + Report for one PAN (the scenario factories wire quick worlds PAN+Score and full worlds PAN+Report
    only; quick-then-full flows need all three, T5 note)."""
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan)},
        score={pan: env.experian_score_envelope(credit_score="745")},
        report={pan: happy_report_envelope()},
        sink=sink,
    )


# --------------------------------------------------------------------------- the API harness


@dataclass
class Api:
    """One app over one MemoryRepo (frozen clock, seeded lenders), one recording SurePass fake and one FakeRankModel,
    driven through a started ``TestClient`` (bearer header pre-attached)."""

    clock: FrozenClock
    repo: MemoryRepo
    surepass: RecordingSurePass
    rank: FakeRankModel
    deps: GraphDeps
    app: FastAPI
    client: TestClient

    # ---- requests

    def post_quick(self, payload: Mapping[str, Any]) -> Response:
        return self.client.post("/checks/quick", json=dict(payload))

    def post_full(self, payload: Mapping[str, Any]) -> Response:
        return self.client.post("/checks/full", json=dict(payload))

    def quick(self, **over: Any) -> dict[str, Any]:
        """POST /checks/quick with ``quick_payload(**over)``; asserts HTTP 200 and returns the LcView JSON."""
        return self._ok(self.post_quick(quick_payload(**over)))

    def full(self, **over: Any) -> dict[str, Any]:
        return self._ok(self.post_full(full_payload(**over)))

    def _ok(self, response: Response) -> dict[str, Any]:
        assert response.status_code == 200, response.text
        return cast("dict[str, Any]", response.json())

    # ---- observations

    def endpoints(self) -> list[str]:
        return self.surepass.endpoints_called()

    def row(self, check_id: str) -> CheckRow:
        got = self.repo.get_check(check_id)
        assert got is not None, check_id
        return got

    def calls(self, check_id: str) -> list[SurePassCallRow]:
        """Insertion order (under a frozen clock the protocol's (called_at, id) order is uuid order)."""
        return self.repo.list_calls(check_id)

    def billable(self, check_id: str) -> list[bool]:
        return [call.billable for call in self.calls(check_id)]

    def locks(self, pan: str, kind: str) -> list[PanLockRow]:
        """Every lock row of ``kind`` for the PAN, released / expired included (assert on ``lock_until``)."""
        pan_hash = pan_hash_of(pan)
        return [lock for lock in self.repo.active_locks_snapshot() if lock.pan_hash == pan_hash and lock.kind == kind]

    def active_lock_kinds(self, pan: str) -> list[str]:
        return [lock.kind for lock in self.repo.get_active_locks(pan_hash_of(pan))]

    def reviews(self, pan: str, kind: str | None = None) -> list[ReviewRow]:
        return [
            r for r in self.repo.list_reviews() if r.pan_hash == pan_hash_of(pan) and (kind is None or r.kind == kind)
        ]


class ApiFactory(Protocol):
    def __call__(
        self,
        surepass: SurePassFactory,
        *,
        rank: FakeRankModel | None = None,
        settings: Settings | None = None,
    ) -> Api: ...


def one_check_row(repo: MemoryRepo) -> CheckRow:
    rows = repo.list_checks()
    assert len(rows) == 1, [r.status for r in rows]
    return rows[0]


def internal_of(row: CheckRow) -> dict[str, Any]:
    assert row.internal is not None, row.id
    return row.internal


# --------------------------------------------------------------------------- repo seeding (world state, not flows)


def _seed_check(
    repo: MemoryRepo,
    *,
    lc_id: str,
    pan_hash: str,
    created_at: datetime,
    counted: bool,
    billed: bool,
    status: CheckStatus = "ok",
) -> str:
    check_id = new_id()
    repo.insert_check(
        CheckRow(
            id=check_id,
            kind="quick",
            lc_id=lc_id,
            lead_id=LEAD,
            opportunity_id=None,
            opportunity_stage=None,
            contact_id=None,
            pan_hash=pan_hash,
            pan_last4="000Z",
            mobile_hash=None,
            mobile_last4=None,
            status=status,
            code=None,
            billed=billed,
            counted_against_quota=counted,
            is_refresh=False,
            recompute_of=None,
            consent_id=None,
            lc_view=None,
            internal=None,
            langsmith_run_id=None,
            created_at=created_at,
        )
    )
    return check_id


def seed_counted_quick_checks(repo: MemoryRepo, *, lc_id: str, created_at: datetime, n: int) -> list[str]:
    """``n`` counted, billed, ok quick checks for ``lc_id`` (each on its own synthetic PAN hash) at ``created_at``."""
    return [
        _seed_check(repo, lc_id=lc_id, pan_hash=f"{index:064x}", created_at=created_at, counted=True, billed=True)
        for index in range(1, n + 1)
    ]


def seed_billable_bureau_calls(
    repo: MemoryRepo, *, pan: str, n: int, now: datetime, endpoint: str = SCORE_ENDPOINT
) -> str:
    """``n`` billable (200 + success) bureau call rows for ``pan``, under one seeded check row (surepass_calls.check_id
    is a FK). ``now`` must be the repo clock's current instant: the sink stamps ``called_at`` from that clock. Returns
    the seeded check id."""
    check_id = _seed_check(
        repo, lc_id="lc-seed", pan_hash=pan_hash_of(pan), created_at=now, counted=False, billed=True
    )
    for _ in range(n):
        repo.log_surepass_call(
            check_id=check_id,
            pan_hash=pan_hash_of(pan),
            endpoint=endpoint,
            client_id="seeded_synthetic_call",
            http_status=200,
            success=True,
            message_code="success",
            latency_ms=0,
            error_kind=None,
            raw_envelope=None,
        )
    return check_id


def seed_running_check(repo: MemoryRepo, *, pan: str, now: datetime, lc_id: str = LC_B) -> str:
    """A running check row on ``pan`` (the holder of a foreign in-flight lock in the IN_FLIGHT scenario)."""
    return _seed_check(
        repo, lc_id=lc_id, pan_hash=pan_hash_of(pan), created_at=now, counted=False, billed=False, status="running"
    )


def age_score_cache(repo: MemoryRepo, pan: str, *, by: timedelta = SCORE_WINDOW + timedelta(days=1)) -> None:
    """Back-date the PAN's score_cache row by ``by`` (default 31 days) so it reads as EXPIRED: the next quick check is an
    expired-score refresh. A world-state trick for driving several billed pulls inside ONE IST day."""
    row = repo.get_score_cache(pan_hash_of(pan))
    assert row is not None, "no score_cache row to age"
    repo.put_score_cache(dataclasses.replace(row, fetched_at=row.fetched_at - by, expires_at=row.expires_at - by))


def seed_cached_quick_world(
    repo: MemoryRepo,
    *,
    pan: str,
    fetched_at: datetime,
    with_pan_record: bool = True,
    with_score: bool = True,
    credit_score: str = "745",
    full_name: str = PAN_FULL_NAME,
) -> None:
    """Pre-seed the raw bucket + ``pan_records`` and/or ``score_cache`` rows exactly as a pull at ``fetched_at`` would
    have left them (raw envelope stored verbatim, rows pointing at it). Used where no pull may happen (live off), or to
    plant a stored PAN record with an unusable ``full_name``."""
    pan_hash = pan_hash_of(pan)
    if with_pan_record:
        key = storage_key_for(pan_hash, PAN_ENDPOINT, fetched_at, new_id())
        repo.raw_store.put_raw(key, env.pan_comprehensive_envelope(pan=pan, full_name=full_name)[1])
        repo.put_pan_record(
            PanRecordRow(
                pan_hash=pan_hash,
                pan_last4=pan_last4(pan),
                storage_key=key,
                client_id="pan_comprehensive_synthetic_0001",
                fetched_at=fetched_at,
                expires_at=fetched_at + PAN_WINDOW,
                purge_after=fetched_at + RAW_RETENTION,
            )
        )
    if with_score:
        _, body = env.experian_score_envelope(credit_score=credit_score)
        key = storage_key_for(pan_hash, SCORE_ENDPOINT, fetched_at, new_id())
        repo.raw_store.put_raw(key, body)
        repo.put_score_cache(
            ScoreCacheRow(
                pan_hash=pan_hash,
                storage_key=key,
                source="score",
                is_ntc=False,
                client_id="experian_score_synthetic_0001",
                consent_id=None,
                fetched_at=fetched_at,
                expires_at=fetched_at + SCORE_WINDOW,
                purge_after=fetched_at + RAW_RETENTION,
            )
        )


# --------------------------------------------------------------------------- the leak scanner (T9 ruling)

# LLM-authored / derived prose: the strict tier (generic 'bureau' term included) applies here on top of the global one.
LLM_AUTHORED_FIELDS: Final[frozenset[str]] = frozenset({"rationale", "conditions", "items_to_confirm", "coapplicant_reason"})
_LENDER_FIELDS: Final[frozenset[str]] = frozenset({"top_lender", "alternates"})


def _string_leaves(node: object, path: tuple[str | int, ...] = ()) -> Iterator[tuple[tuple[str | int, ...], str]]:
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, Mapping):
        for key, value in cast("Mapping[str, object]", node).items():
            yield from _string_leaves(value, (*path, key))
    elif isinstance(node, Sequence):
        for index, value in enumerate(cast("Sequence[object]", node)):
            yield from _string_leaves(value, (*path, index))


def _exempt(path: tuple[str | int, ...]) -> bool:
    """``check_id`` at the root and ``id`` on a lender view (top_lender.id / alternates[i].id): uuid-ish, never prose."""
    if path == ("check_id",):
        return True
    return bool(path) and path[-1] == "id" and path[0] in _LENDER_FIELDS


def leak_findings(body: Mapping[str, Any], *, forbidden: Sequence[str] = IDENTITY_LITERALS) -> list[str]:
    """Every offending string leaf, as ``'<dotted path>: <reason>'``. Empty means clean."""
    findings: list[str] = []
    lowered = [literal.casefold() for literal in forbidden]
    for path, text in _string_leaves(body):
        if _exempt(path):
            continue
        dotted = ".".join(str(p) for p in path)
        if line_leaks(text, GLOBAL_LEAK_PATTERNS):
            findings.append(f"{dotted}: global tier")
        if any(isinstance(p, str) and p in LLM_AUTHORED_FIELDS for p in path) and line_leaks(text, LLM_LEAK_PATTERNS):
            findings.append(f"{dotted}: llm tier")
        folded = text.casefold()
        findings.extend(f"{dotted}: identity literal" for literal in lowered if literal in folded)
    return findings


def assert_no_leak(body: Mapping[str, Any]) -> None:
    """The LcView carries no undeclared field (extra='forbid' round-trip) and no leaking string anywhere."""
    LcView.model_validate(body)
    findings = leak_findings(body)
    assert findings == [], findings
