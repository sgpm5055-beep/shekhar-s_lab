"""Task 2: the db layer's contract, exercised through MemoryRepo + MemoryRawStore (the only backends tests use)."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast, get_args, get_type_hints

import pytest
from repo_builders import (
    ENVELOPE,
    PAN_A,
    PAN_B,
    PAN_EP,
    REPORT_EP,
    SCORE_EP,
    T0,
    UTC,
    FrozenClock,
    log_call,
    make_check,
    make_consent,
    make_pan_event,
    make_pan_record,
    make_report_cache,
    make_review,
    make_score_cache,
)

from eligibility_agent.db import repo as repo_module
from eligibility_agent.db.repo import (
    QUICK_DAILY_QUOTA_DEFAULT,
    RAW_RETENTION,
    CheckPatch,
    CheckRow,
    CheckStatus,
    ConsentRow,
    MemoryRawStore,
    MemoryRepo,
    OpportunityPanEvent,
    PanLockRow,
    PanRecordRow,
    RawUnavailable,
    Repo,
    ReportCacheRow,
    ReviewRow,
    ScoreCacheRow,
    SupabaseRepo,
    SurePassCallRow,
    call_is_billable,
    code_legal,
    new_id,
    storage_key_for,
    utc_now,
)
from eligibility_agent.graph.state import Code, LockKind

REPO_FILE = Path(repo_module.__file__)
PACKAGE_ROOT = REPO_FILE.parents[1]  # eligibility_agent/

# The 29 enumerated Repo protocol methods (the design says "27"; K4 fixes the count at 29).
PROTOCOL_METHODS = (
    "insert_check",
    "update_check",
    "get_check",
    "record_check_error",
    "attach_contact",
    "insert_consent",
    "get_pan_record",
    "put_pan_record",
    "get_score_cache",
    "put_score_cache",
    "get_report_cache",
    "put_report_cache",
    "fetch_raw",
    "quick_checks_used_today",
    "quota_limit",
    "bureau_calls_today",
    "try_acquire_pan_lock",
    "release_pan_lock",
    "create_pan_lock",
    "get_active_locks",
    "insert_review_item",
    "unresolved_review_exists",
    "pan_events_for_opportunity",
    "insert_pan_event",
    "report_exists_for_opportunity",
    "log_surepass_call",
    "surepass_calls_for_check",
    "published_lenders",
    "lenders_for_pincodes",
)

ALL_CODES: tuple[str, ...] = get_args(Code)
REFUSAL_CODES = frozenset(
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
ALL_STATUSES: tuple[str, ...] = get_args(CheckStatus)


def _legal_code_for(status: str) -> str | None:
    """A code that is legal for this status (for building non-running rows)."""
    return {"refused": "PAN_LOCKED", "service_failure": "SUREPASS_FAILURE"}.get(status)


# --------------------------------------------------------------------------- registry + legality


def test_check_status_is_the_seven_value_row_status() -> None:
    assert get_args(CheckStatus) == (
        "running",
        "ok",
        "pan_not_verified",
        "refused",
        "pending_verification",
        "service_failure",
        "error",
    )


def test_repo_defines_no_literal_alias_except_check_status() -> None:
    """K14: every other Literal lives in graph/state.py and is imported."""
    tree = ast.parse(REPO_FILE.read_text())
    literal_aliases: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
        else:
            continue
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == "Literal":
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            assert isinstance(target, ast.Name)
            literal_aliases.append(target.id)
    assert literal_aliases == ["CheckStatus"]


def test_supabase_is_confined_to_db_repo_behind_the_pyright_pragma() -> None:
    roots = {"supabase", "storage3", "postgrest", "gotrue", "realtime", "supafunc"}
    offenders: list[str] = []
    for path in [*PACKAGE_ROOT.rglob("*.py"), *(PACKAGE_ROOT.parent / "scripts").rglob("*.py")]:
        if path == REPO_FILE:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            if any(name.split(".")[0] in roots for name in names):
                offenders.append(str(path))
    assert offenders == []

    pragma = (
        "# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, "
        "reportUnknownVariableType=false, reportUnknownArgumentType=false"
    )
    assert pragma in REPO_FILE.read_text().splitlines()[:5]

    # ...and even inside db/repo.py nothing imports supabase at module import time
    # (Client is TYPE_CHECKING-only; the factory imports create_client lazily).
    for node in ast.parse(REPO_FILE.read_text()).body:
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] in roots for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in roots


@pytest.mark.parametrize("status", ALL_STATUSES)
@pytest.mark.parametrize("code", [*ALL_CODES, None])
def test_code_legal_matrix(status: str, code: str | None) -> None:
    allowed: dict[str, set[str | None]] = {
        "refused": set(REFUSAL_CODES),
        "service_failure": {"SUREPASS_FAILURE", "OPENAI_FAILURE"},
        "ok": {None, "NO_PINCODE_ON_PAN"},
        "running": {None},
        "pan_not_verified": {None},
        "pending_verification": {None},  # K1
        "error": {None},
    }
    assert code_legal(cast(CheckStatus, status), cast(Code | None, code)) is (code in allowed[status])


def test_code_legal_spot_checks() -> None:
    assert code_legal("pending_verification", None)
    assert not code_legal("pending_verification", "PAN_LOCKED")
    assert not code_legal("refused", None)
    assert not code_legal("service_failure", None)
    assert code_legal("ok", "NO_PINCODE_ON_PAN")
    assert not code_legal("ok", "SUREPASS_FAILURE")
    assert not code_legal("refused", "NO_PINCODE_ON_PAN")


# --------------------------------------------------------------------------- row shapes


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(cls))


def test_check_row_has_the_designs_enumerated_fields() -> None:
    # The design prose says "23 fields"; the enumerated list (and the SQL table) has 22.
    assert _field_names(CheckRow) == (
        "id",
        "kind",
        "lc_id",
        "lead_id",
        "opportunity_id",
        "opportunity_stage",
        "contact_id",
        "pan_hash",
        "pan_last4",
        "mobile_hash",
        "mobile_last4",
        "status",
        "code",
        "billed",
        "counted_against_quota",
        "is_refresh",
        "recompute_of",
        "consent_id",
        "lc_view",
        "internal",
        "langsmith_run_id",
        "created_at",
    )


def test_other_row_shapes() -> None:
    assert len(_field_names(ConsentRow)) == 14
    assert "lc_typed_name" in _field_names(ConsentRow)
    assert len(_field_names(SurePassCallRow)) == 15
    assert _field_names(ScoreCacheRow) == (
        "pan_hash",
        "storage_key",
        "source",
        "is_ntc",
        "client_id",
        "consent_id",
        "fetched_at",
        "expires_at",
        "purge_after",
    )
    assert _field_names(ReportCacheRow)[3] == "report_date"
    assert _field_names(PanLockRow) == ("pan_hash", "kind", "check_id", "locked_at", "lock_until", "released_at")


@pytest.mark.parametrize(
    "cls",
    [
        ConsentRow,
        PanRecordRow,
        ScoreCacheRow,
        ReportCacheRow,
        CheckRow,
        SurePassCallRow,
        PanLockRow,
        ReviewRow,
        OpportunityPanEvent,
    ],
)
def test_rows_are_frozen_slots_dataclasses(cls: type) -> None:
    assert dataclasses.is_dataclass(cls)
    assert "__slots__" in vars(cls)
    params = getattr(cls, "__dataclass_params__")
    assert params.frozen


def test_repo_protocol_is_the_29_enumerated_methods_and_both_backends_match_it() -> None:
    names = [n for n, _ in inspect.getmembers(Repo, inspect.isfunction) if not n.startswith("_")]
    assert sorted(names) == sorted(PROTOCOL_METHODS)
    assert len(PROTOCOL_METHODS) == 29
    for backend in (MemoryRepo, SupabaseRepo):
        for name in PROTOCOL_METHODS:
            expected = inspect.signature(getattr(Repo, name))
            actual = inspect.signature(getattr(backend, name))
            assert [(p.name, p.kind) for p in actual.parameters.values()] == [
                (p.name, p.kind) for p in expected.parameters.values()
            ], f"{backend.__name__}.{name} signature differs from the Repo protocol"


def test_memory_repo_satisfies_the_protocol_statically() -> None:
    repo: Repo = MemoryRepo()  # pyright checks the structural conformance
    assert repo.quota_limit("lc-1") == QUICK_DAILY_QUOTA_DEFAULT == 5


def test_utc_now_and_new_id() -> None:
    assert utc_now().tzinfo is not None and utc_now().utcoffset() == timedelta(0)
    first, second = new_id(), new_id()
    assert first != second
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", first)


# --------------------------------------------------------------------------- storage key + MemoryRawStore


def test_storage_key_format_and_slug_mapping() -> None:
    called_at = datetime(2026, 10, 10, 8, 5, 9, tzinfo=UTC)
    assert (
        storage_key_for(PAN_A, PAN_EP, called_at, "call-1")
        == f"pan/{PAN_A}/pan-comprehensive/20261010T080509Z-call-1.json"
    )
    assert (
        storage_key_for(PAN_A, SCORE_EP, called_at, "call-2")
        == f"pan/{PAN_A}/experian-score/20261010T080509Z-call-2.json"
    )
    assert (
        storage_key_for(PAN_B, REPORT_EP, called_at, "call-3")
        == f"pan/{PAN_B}/experian-report/20261010T080509Z-call-3.json"
    )


def test_storage_key_is_stamped_in_utc_whatever_the_input_zone() -> None:
    from zoneinfo import ZoneInfo

    ist = datetime(2026, 10, 10, 13, 35, 9, tzinfo=ZoneInfo("Asia/Kolkata"))  # == 08:05:09Z
    assert storage_key_for(PAN_A, SCORE_EP, ist, "c").endswith("/20261010T080509Z-c.json")


def test_storage_key_rejects_an_unknown_endpoint() -> None:
    with pytest.raises(ValueError):
        storage_key_for(PAN_A, "/api/v1/something/else", T0, "c")


def test_memory_raw_store_round_trips_deep_copies_and_json_normalises() -> None:
    store = MemoryRawStore()
    envelope: dict[str, Any] = {"data": {"scores": [1, 2]}, "success": True}
    store.put_raw("k1", envelope)
    envelope["data"]["scores"].append(3)  # caller mutation after put must not leak in
    got = store.get_raw("k1")
    assert got == {"data": {"scores": [1, 2]}, "success": True}
    got["data"]["scores"].append(99)  # nor must mutation of a returned copy
    assert store.get_raw("k1") == {"data": {"scores": [1, 2]}, "success": True}
    assert store.keys() == ["k1"]


def test_memory_raw_store_matches_storage_semantics() -> None:
    store = MemoryRawStore()
    with pytest.raises(RawUnavailable):
        store.get_raw("missing")
    store.put_raw("a", {"x": 1})
    store.put_raw("a", {"x": 2})  # upsert: the bucket upload uses upsert=true
    assert store.get_raw("a") == {"x": 2}
    store.put_raw("b", {"y": 1})
    store.delete_raw(["a", "never-existed"])  # deleting a missing object is not an error
    assert store.keys() == ["b"]
    with pytest.raises(TypeError):  # the real bucket stores JSON: a non-JSON envelope must fail here too
        store.put_raw("c", {"when": T0})


# --------------------------------------------------------------------------- checks: lifecycle + terminal guard


def test_insert_get_check_round_trip_deep_copy_and_duplicate() -> None:
    repo = MemoryRepo(FrozenClock())
    row = make_check(lc_view={"k": ["v"]})
    repo.insert_check(row)
    assert repo.get_check("chk-1") == row
    assert repo.get_check("nope") is None
    got = repo.get_check("chk-1")
    assert got is not None and got.lc_view is not None
    got.lc_view["k"].append("mutated")  # returned copy must not alias the store
    again = repo.get_check("chk-1")
    assert again is not None and again.lc_view == {"k": ["v"]}
    with pytest.raises(ValueError):
        repo.insert_check(row)
    assert [c.id for c in repo.list_checks()] == ["chk-1"]


def test_insert_check_rejects_an_illegal_status_code_pair() -> None:
    repo = MemoryRepo(FrozenClock())
    with pytest.raises(ValueError):
        repo.insert_check(make_check(status="refused", code=None))
    with pytest.raises(ValueError):
        repo.insert_check(make_check(status="running", code="PAN_LOCKED"))
    assert repo.list_checks() == []


def test_terminal_patch_applies_while_running() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    patch: CheckPatch = {
        "status": "ok",
        "code": None,
        "billed": True,
        "counted_against_quota": True,
        "lc_view": {"headline": "possible"},
        "internal": {"admission": "new"},
        "langsmith_run_id": "run-1",
    }
    repo.update_check("chk-1", patch)
    got = repo.get_check("chk-1")
    assert got is not None
    assert (got.status, got.billed, got.counted_against_quota) == ("ok", True, True)
    assert got.lc_view == {"headline": "possible"}
    assert got.internal == {"admission": "new"}
    assert got.langsmith_run_id == "run-1"
    assert got.pan_hash == PAN_A  # untouched fields survive


def test_terminal_patch_carries_is_refresh() -> None:
    # K4 vs K10 (Task 11): the row is inserted at admission step 1.5, BEFORE the decision that knows is_refresh, so the
    # flag rides on the ONE guarded terminal patch (never a separate update).
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    patch: CheckPatch = {"status": "ok", "code": None, "is_refresh": True}
    repo.update_check("chk-1", patch)
    got = repo.get_check("chk-1")
    assert got is not None
    assert got.is_refresh is True


@pytest.mark.parametrize("status", [s for s in ALL_STATUSES if s != "running"])
def test_terminal_guard_makes_every_non_running_row_immutable_to_terminal_patches(status: str) -> None:
    repo = MemoryRepo(FrozenClock())
    original = make_check(status=status, code=_legal_code_for(status))
    repo.insert_check(original)
    repo.update_check("chk-1", {"status": "ok", "code": None, "billed": True, "lc_view": {"x": 1}})
    assert repo.get_check("chk-1") == original  # silent no-op, not an error


def test_terminal_guard_error_then_ok_noops_and_ok_then_error_noops() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(id="e", status="error"))
    repo.update_check("e", {"status": "ok"})
    assert getattr(repo.get_check("e"), "status") == "error"

    repo.insert_check(make_check(id="o"))
    repo.update_check("o", {"status": "ok"})
    repo.update_check("o", {"status": "error", "internal": {"error": "late"}})
    got = repo.get_check("o")
    assert got is not None and got.status == "ok" and got.internal is None


def test_billed_only_patch_is_also_guarded() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(status="ok"))
    repo.update_check("chk-1", {"billed": True})
    assert getattr(repo.get_check("chk-1"), "billed") is False


def test_contact_id_only_patch_is_exempt_from_the_guard() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(status="ok"))
    repo.update_check("chk-1", {"contact_id": "contact-9"})
    got = repo.get_check("chk-1")
    assert got is not None and got.contact_id == "contact-9" and got.status == "ok"

    repo.insert_check(make_check(id="r"))  # and it also works on a running row
    repo.update_check("r", {"contact_id": "contact-8"})
    assert getattr(repo.get_check("r"), "contact_id") == "contact-8"


def test_contact_id_mixed_with_other_fields_is_not_exempt() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(status="ok"))
    repo.update_check("chk-1", {"contact_id": "contact-9", "billed": True})
    got = repo.get_check("chk-1")
    assert got is not None and got.contact_id is None and got.billed is False


def test_update_of_an_unknown_check_is_silent() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.update_check("ghost", {"status": "ok"})
    repo.update_check("ghost", {"contact_id": "c"})
    assert repo.list_checks() == []


def test_update_rejects_an_illegal_status_code_pair_and_leaves_the_row_alone() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    with pytest.raises(ValueError):
        repo.update_check("chk-1", {"status": "refused"})  # refused needs a refusal code
    with pytest.raises(ValueError):
        repo.update_check("chk-1", {"status": "service_failure", "code": "PAN_LOCKED"})
    assert getattr(repo.get_check("chk-1"), "status") == "running"
    repo.update_check("chk-1", {"status": "service_failure", "code": "OPENAI_FAILURE"})
    got = repo.get_check("chk-1")
    assert got is not None and (got.status, got.code) == ("service_failure", "OPENAI_FAILURE")


def test_pending_verification_carries_no_code() -> None:
    """K1."""
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    with pytest.raises(ValueError):
        repo.update_check("chk-1", {"status": "pending_verification", "code": "PAN_LOCKED"})
    repo.update_check("chk-1", {"status": "pending_verification", "code": None, "billed": True})
    assert getattr(repo.get_check("chk-1"), "status") == "pending_verification"


def test_update_patch_values_are_deep_copied() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    view: dict[str, Any] = {"items": ["a"]}
    repo.update_check("chk-1", {"status": "ok", "lc_view": view})
    view["items"].append("b")
    got = repo.get_check("chk-1")
    assert got is not None and got.lc_view == {"items": ["a"]}


# --------------------------------------------------------------------------- record_check_error


def test_record_check_error_updates_a_running_row() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(billed=True))
    repo.record_check_error(
        check_id="chk-1", kind="quick", lc_id="lc-1", pan_hash=PAN_A, pan_last4="234F", message="boom"
    )
    got = repo.get_check("chk-1")
    assert got is not None
    assert (got.status, got.code, got.internal) == ("error", None, {"error": "boom"})
    assert got.billed is True  # an update, not a replacement
    assert len(repo.list_checks()) == 1


def test_record_check_error_inserts_when_admission_never_did() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.record_check_error(
        check_id="chk-new", kind="full", lc_id="lc-7", pan_hash=PAN_B, pan_last4="9999", message="crash"
    )
    got = repo.get_check("chk-new")
    assert got is not None
    assert (got.kind, got.lc_id, got.pan_hash, got.pan_last4) == ("full", "lc-7", PAN_B, "9999")
    assert (got.status, got.code, got.internal) == ("error", None, {"error": "crash"})
    assert (got.billed, got.counted_against_quota, got.is_refresh) == (False, False, False)
    assert got.created_at == clock.now
    assert got.lc_view is None and got.lead_id is None and got.opportunity_id is None


def test_record_check_error_never_overwrites_a_terminal_row_and_never_raises() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(status="ok", lc_view={"final": True}))
    repo.record_check_error(
        check_id="chk-1", kind="quick", lc_id="lc-1", pan_hash=PAN_A, pan_last4="234F", message="late crash"
    )
    got = repo.get_check("chk-1")
    assert got is not None and got.status == "ok" and got.lc_view == {"final": True} and got.internal is None


# --------------------------------------------------------------------------- consents + attach_contact


def test_insert_consent_stores_a_copy_and_rejects_duplicates() -> None:
    repo = MemoryRepo(FrozenClock())
    consent = make_consent()
    repo.insert_consent(consent)
    assert repo.list_consents() == [consent]
    with pytest.raises(ValueError):
        repo.insert_consent(consent)


def test_attach_contact_updates_the_check_and_the_checks_consent() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_consent(make_consent(id="con-1"))
    repo.insert_consent(make_consent(id="con-2"))  # an unrelated consent must stay untouched
    repo.insert_check(make_check(consent_id="con-1", status="ok"))  # attach works on a finished check
    repo.attach_contact("chk-1", "contact-42")
    assert getattr(repo.get_check("chk-1"), "contact_id") == "contact-42"
    by_id = {c.id: c for c in repo.list_consents()}
    assert by_id["con-1"].contact_id == "contact-42"
    assert by_id["con-2"].contact_id is None


def test_attach_contact_without_a_consent_or_check() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(consent_id=None))
    repo.attach_contact("chk-1", "contact-1")
    assert getattr(repo.get_check("chk-1"), "contact_id") == "contact-1"
    repo.attach_contact("ghost", "contact-1")  # unknown check: no-op, no error
    assert [c.id for c in repo.list_checks()] == ["chk-1"]


# --------------------------------------------------------------------------- the three caches + fetch_raw


def test_pan_record_upsert_and_stale_rows_are_returned() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    assert repo.get_pan_record(PAN_A) is None
    repo.put_pan_record(make_pan_record(storage_key="k-old"))
    repo.put_pan_record(make_pan_record(storage_key="k-new"))  # upsert on pan_hash
    clock.advance(days=400)  # long past expires_at: freshness is policy's call, not the repo's
    got = repo.get_pan_record(PAN_A)
    assert got is not None and got.storage_key == "k-new"
    assert repo.get_pan_record(PAN_B) is None


def test_score_and_report_caches_are_independent_upserts() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.put_score_cache(make_score_cache(storage_key="s1"))
    repo.put_report_cache(make_report_cache(storage_key="r1", report_date=T0.date()))
    repo.put_score_cache(make_score_cache(storage_key="s2"))
    score, report = repo.get_score_cache(PAN_A), repo.get_report_cache(PAN_A)
    assert score is not None and score.storage_key == "s2" and score.source == "score"
    assert report is not None and report.storage_key == "r1" and report.report_date == T0.date()
    assert repo.get_score_cache(PAN_B) is None and repo.get_report_cache(PAN_B) is None


def test_a_report_pull_refreshes_the_score_cache_with_the_same_object() -> None:
    """The write pair the bureau node performs after a successful Report pull (PRD 5.3)."""
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(kind="full"))
    key = log_call(repo, "chk-1", endpoint=REPORT_EP, envelope={"data": {"credit_score": "801"}})
    assert key is not None
    repo.put_report_cache(make_report_cache(storage_key=key))
    repo.put_score_cache(make_score_cache(storage_key=key, source="report"))
    score, report = repo.get_score_cache(PAN_A), repo.get_report_cache(PAN_A)
    assert score is not None and report is not None
    assert score.source == "report" and score.storage_key == report.storage_key == key
    assert repo.fetch_raw(key) == {"data": {"credit_score": "801"}}


def test_fetch_raw_returns_the_stored_envelope_and_raises_when_gone() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    key = log_call(repo, "chk-1")
    assert key is not None
    assert repo.fetch_raw(key) == ENVELOPE
    with pytest.raises(RawUnavailable):
        repo.fetch_raw("pan/none/experian-score/missing.json")
    repo.raw_store.delete_raw([key])  # purged
    with pytest.raises(RawUnavailable):
        repo.fetch_raw(key)


# --------------------------------------------------------------------------- the v2 audit sink


class _SpyRawStore(MemoryRawStore):
    """Records how many surepass_calls rows existed at the instant each object was uploaded."""

    def __init__(self) -> None:
        super().__init__()
        self.repo: MemoryRepo | None = None
        self.rows_at_put: list[int] = []
        self.fail = False

    def put_raw(self, key: str, envelope: dict[str, Any]) -> None:
        assert self.repo is not None
        self.rows_at_put.append(len(self.repo.list_calls()))
        if self.fail:
            raise RuntimeError("storage down")
        super().put_raw(key, envelope)


def test_sink_uploads_raw_first_then_inserts_the_row() -> None:
    spy = _SpyRawStore()
    repo = MemoryRepo(FrozenClock(), raw_store=spy)
    spy.repo = repo
    repo.insert_check(make_check())
    key = log_call(repo, "chk-1")
    assert spy.rows_at_put == [0]  # object landed while no row existed yet
    assert key is not None and spy.keys() == [key]
    (row,) = repo.list_calls()
    assert row.storage_key == key


def test_sink_leaves_no_row_pointing_at_a_missing_object_when_the_upload_fails() -> None:
    spy = _SpyRawStore()
    repo = MemoryRepo(FrozenClock(), raw_store=spy)
    spy.repo = repo
    spy.fail = True
    repo.insert_check(make_check())
    with pytest.raises(RuntimeError):
        log_call(repo, "chk-1")
    assert repo.list_calls() == []


def test_sink_row_fields_key_and_retention() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.insert_check(make_check())
    key = log_call(repo, "chk-1", endpoint=SCORE_EP)
    (row,) = repo.list_calls()
    assert re.fullmatch(rf"pan/{PAN_A}/experian-score/20261010T080000Z-{row.id}\.json", key or "")
    assert row.check_id == "chk-1" and row.pan_hash == PAN_A and row.endpoint == SCORE_EP
    assert (row.client_id, row.http_status, row.success, row.latency_ms) == ("cid-1", 200, True, 120)
    assert row.called_at == clock.now
    assert row.purge_after == clock.now + RAW_RETENTION == clock.now + timedelta(days=180)
    assert row.purged_at is None
    assert repo.fetch_raw(key or "") == ENVELOPE


def test_sink_stores_failure_bodies_verbatim_and_returns_none_without_a_body() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    failure = {"success": False, "message": "No record found", "status_code": 422}
    key = log_call(repo, "chk-1", http_status=422, success=False, envelope=failure)
    assert key is not None and repo.fetch_raw(key) == failure  # every body is kept, success or not

    assert log_call(repo, "chk-1", http_status=None, success=None, error_kind="transport", envelope=None) is None
    rows = repo.list_calls("chk-1")
    assert len(rows) == 2 and rows[1].storage_key is None
    assert len(repo.raw_store.keys()) == 1  # nothing uploaded for the body-less attempt


@pytest.mark.parametrize(
    ("http_status", "success", "error_kind", "billable"),
    [
        (200, True, None, True),
        (200, False, None, False),  # unreachable live (success:false raises) but the predicate is pinned
        (200, None, None, False),  # the {errors, message} validation envelope
        (500, False, "upstream", False),
        (500, None, "upstream", False),
        (401, False, "auth", False),
        (429, False, "rate_limited", False),
        (None, None, "transport", False),
        (None, None, "timeout", True),  # spend unknown: conservatively billable
        (200, True, "timeout", True),
        (None, None, None, False),
    ],
)
def test_billable_truth_table(http_status: int | None, success: bool | None, error_kind: Any, billable: bool) -> None:
    assert call_is_billable(http_status, success, error_kind) is billable
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check())
    log_call(repo, "chk-1", http_status=http_status, success=success, error_kind=error_kind)
    (row,) = repo.list_calls()
    assert row.billable is billable
    assert row.error_kind == error_kind


def test_sink_asserts_the_parent_check_exists_fk_parity() -> None:
    repo = MemoryRepo(FrozenClock())
    with pytest.raises(AssertionError):
        log_call(repo, "no-such-check")
    assert repo.list_calls() == [] and repo.raw_store.keys() == []


def test_surepass_calls_for_check_is_ordered_and_scoped() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.insert_check(make_check(id="c1"))
    repo.insert_check(make_check(id="c2"))
    first = log_call(repo, "c1", endpoint=PAN_EP)
    clock.advance(seconds=5)
    log_call(repo, "c2")
    clock.advance(seconds=5)
    third = log_call(repo, "c1", endpoint=SCORE_EP)
    rows = repo.surepass_calls_for_check("c1")
    assert [r.endpoint for r in rows] == [PAN_EP, SCORE_EP]
    assert [r.storage_key for r in rows] == [first, third]
    assert [r.check_id for r in repo.list_calls()] == ["c1", "c2", "c1"]
    assert repo.surepass_calls_for_check("nope") == []


# --------------------------------------------------------------------------- locks


def _until(clock: FrozenClock, **delta: float) -> datetime:
    return clock.now + timedelta(**delta)


def test_try_acquire_is_exclusive_while_the_lock_is_fresh() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    assert repo.try_acquire_pan_lock(PAN_A, "chk-1", _until(clock, seconds=180)) is True
    assert repo.try_acquire_pan_lock(PAN_A, "chk-2", _until(clock, seconds=180)) is False  # IN_FLIGHT, never queued
    assert repo.try_acquire_pan_lock(PAN_A, "chk-1", _until(clock, seconds=180)) is False  # not re-entrant either
    assert repo.try_acquire_pan_lock(PAN_B, "chk-3", _until(clock, seconds=180)) is True  # other PANs unaffected
    (lock, other) = sorted(repo.active_locks_snapshot(), key=lambda r: r.pan_hash)
    assert (lock.pan_hash, lock.kind, lock.check_id) == (PAN_A, "in_flight", "chk-1")
    assert lock.locked_at == clock.now and lock.lock_until == clock.now + timedelta(seconds=180)
    assert other.pan_hash == PAN_B


def test_try_acquire_steals_only_when_expired() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    assert repo.try_acquire_pan_lock(PAN_A, "chk-1", _until(clock, seconds=180))
    clock.advance(seconds=179)
    assert repo.try_acquire_pan_lock(PAN_A, "chk-2", _until(clock, seconds=180)) is False
    clock.advance(seconds=1)  # lock_until <= now: a crashed run self-heals
    assert repo.try_acquire_pan_lock(PAN_A, "chk-2", _until(clock, seconds=180)) is True
    (lock,) = repo.get_active_locks(PAN_A)
    assert lock.check_id == "chk-2" and lock.released_at is None
    # the displaced holder's release must not free the thief's lock
    repo.release_pan_lock(PAN_A, "chk-1")
    assert len(repo.get_active_locks(PAN_A)) == 1


def test_try_acquire_reacquires_a_released_lock() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    assert repo.try_acquire_pan_lock(PAN_A, "chk-1", _until(clock, seconds=180))
    repo.release_pan_lock(PAN_A, "chk-1")
    assert repo.try_acquire_pan_lock(PAN_A, "chk-2", _until(clock, seconds=180)) is True
    (lock,) = repo.get_active_locks(PAN_A)
    assert lock.check_id == "chk-2" and lock.released_at is None


def test_release_is_holder_only_and_idempotent() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.try_acquire_pan_lock(PAN_A, "chk-1", _until(clock, seconds=180))
    repo.release_pan_lock(PAN_A, "someone-else")  # not the holder: no-op
    assert len(repo.get_active_locks(PAN_A)) == 1
    repo.release_pan_lock(PAN_A, "chk-1")
    assert repo.get_active_locks(PAN_A) == []
    repo.release_pan_lock(PAN_A, "chk-1")  # idempotent
    repo.release_pan_lock(PAN_B, "chk-1")  # nothing there: no error
    (snapshot,) = repo.active_locks_snapshot()  # released row stays as history, released_at stamped
    assert snapshot.released_at == clock.now


def test_release_only_touches_the_in_flight_lock() -> None:
    """Persist releases in a finally; a timeout lock the same check created must survive it."""
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.try_acquire_pan_lock(PAN_A, "chk-1", _until(clock, seconds=180))
    repo.create_pan_lock(PAN_A, "timeout", "chk-1", _until(clock, minutes=10))
    repo.release_pan_lock(PAN_A, "chk-1")
    assert [lock.kind for lock in repo.get_active_locks(PAN_A)] == ["timeout"]


def test_create_pan_lock_upserts_per_kind_and_get_active_locks_honours_expiry_and_release() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.create_pan_lock(PAN_A, "timeout", "chk-1", _until(clock, minutes=10))
    repo.create_pan_lock(PAN_A, "daily_cap", None, _until(clock, hours=10))
    repo.create_pan_lock(PAN_B, "timeout", "chk-9", _until(clock, minutes=10))
    assert [(lock.kind, lock.check_id) for lock in repo.get_active_locks(PAN_A)] == [
        ("daily_cap", None),
        ("timeout", "chk-1"),
    ]
    repo.create_pan_lock(PAN_A, "timeout", "chk-2", _until(clock, minutes=20))  # same (pan, kind): replaced
    timeout = [lock for lock in repo.get_active_locks(PAN_A) if lock.kind == "timeout"]
    assert [lock.check_id for lock in timeout] == ["chk-2"]
    clock.advance(minutes=20)  # lock_until == now: no longer active (strictly greater required)
    assert [lock.kind for lock in repo.get_active_locks(PAN_A)] == ["daily_cap"]
    clock.advance(hours=10)
    assert repo.get_active_locks(PAN_A) == []
    assert [lock.pan_hash for lock in repo.get_active_locks(PAN_B)] == []  # PAN_B's lock expired too


def test_create_pan_lock_kind_is_narrowed_to_timeout_and_daily_cap_on_protocol_and_both_backends() -> None:
    """Task 7: in_flight locks are written ONLY by the try_acquire RPC (the sole-safe-writer invariant), so
    create_pan_lock's ``kind`` is ``Literal['timeout', 'daily_cap']`` on the Protocol, MemoryRepo and SupabaseRepo
    together; pyright (strict, part of the gate) then rejects any in_flight caller. LockKind itself is unchanged."""
    for owner in (Repo, MemoryRepo, SupabaseRepo):
        assert get_type_hints(owner.create_pan_lock)["kind"] == Literal["timeout", "daily_cap"], owner
    assert get_args(LockKind) == ("in_flight", "timeout", "daily_cap")
    assert get_args(get_type_hints(PanLockRow)["kind"]) == ("in_flight", "timeout", "daily_cap")  # rows still hold all 3


def test_in_flight_acquire_does_not_disturb_timeout_or_daily_cap_locks() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    repo.create_pan_lock(PAN_A, "timeout", "chk-1", _until(clock, minutes=10))
    assert repo.try_acquire_pan_lock(PAN_A, "chk-2", _until(clock, seconds=180)) is True
    assert sorted(lock.kind for lock in repo.get_active_locks(PAN_A)) == ["in_flight", "timeout"]


# --------------------------------------------------------------------------- IST-day quota + cap mirrors


# (created_at, now, same IST day?)  IST midnight == 18:30Z; 23:59 IST and 00:01 IST share the UTC date 2026-10-10.
IST_VECTORS = [
    (datetime(2026, 10, 10, 18, 29, 59, 999999, tzinfo=UTC), datetime(2026, 10, 10, 18, 29, tzinfo=UTC), True),
    (datetime(2026, 10, 10, 18, 30, tzinfo=UTC), datetime(2026, 10, 10, 18, 29, tzinfo=UTC), False),
    (datetime(2026, 10, 10, 18, 30, tzinfo=UTC), datetime(2026, 10, 10, 18, 31, tzinfo=UTC), True),
    (datetime(2026, 10, 10, 18, 29, 59, 999999, tzinfo=UTC), datetime(2026, 10, 10, 18, 31, tzinfo=UTC), False),
    (datetime(2026, 10, 9, 18, 30, tzinfo=UTC), datetime(2026, 10, 10, 18, 29, tzinfo=UTC), True),  # IST 00:00:00
    (datetime(2026, 10, 9, 18, 29, 59, tzinfo=UTC), datetime(2026, 10, 10, 1, 0, tzinfo=UTC), False),
    (datetime(2026, 10, 10, 1, 0, tzinfo=UTC), datetime(2026, 10, 10, 18, 31, tzinfo=UTC), False),  # same UTC date!
]


@pytest.mark.parametrize(("created_at", "now", "same_day"), IST_VECTORS)
def test_quick_checks_used_today_counts_the_ist_day(created_at: datetime, now: datetime, same_day: bool) -> None:
    clock = FrozenClock(now)
    repo = MemoryRepo(clock)
    repo.insert_check(make_check(counted_against_quota=True, created_at=created_at))
    assert repo.quick_checks_used_today("lc-1") == (1 if same_day else 0)


@pytest.mark.parametrize(("created_at", "now", "same_day"), IST_VECTORS)
def test_bureau_calls_today_counts_the_ist_day(created_at: datetime, now: datetime, same_day: bool) -> None:
    clock = FrozenClock(created_at)
    repo = MemoryRepo(clock)
    repo.insert_check(make_check(created_at=created_at))
    log_call(repo, "chk-1", endpoint=SCORE_EP)  # called_at == created_at via the clock
    clock.now = now
    assert repo.bureau_calls_today(PAN_A) == (1 if same_day else 0)


def test_quick_checks_used_today_counts_only_counted_quick_rows_of_this_lc() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(id="a", counted_against_quota=True))
    repo.insert_check(make_check(id="b", counted_against_quota=True))
    repo.insert_check(make_check(id="not-counted", counted_against_quota=False))
    repo.insert_check(make_check(id="full", kind="full", counted_against_quota=True))
    repo.insert_check(make_check(id="other-lc", lc_id="lc-2", counted_against_quota=True))
    assert repo.quick_checks_used_today("lc-1") == 2
    assert repo.quick_checks_used_today("lc-2") == 1
    assert repo.quick_checks_used_today("lc-3") == 0


def test_bureau_calls_today_counts_billable_bureau_endpoint_rows_of_this_pan_only() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(id="c1"))
    repo.insert_check(make_check(id="c2", pan_hash=PAN_B))
    log_call(repo, "c1", endpoint=SCORE_EP)  # counts
    log_call(repo, "c1", endpoint=REPORT_EP)  # counts
    log_call(repo, "c1", endpoint=SCORE_EP, http_status=None, success=None, error_kind="timeout")  # timeouts count
    log_call(repo, "c1", endpoint=PAN_EP)  # PAN Comprehensive is not a bureau call
    log_call(repo, "c1", endpoint=SCORE_EP, http_status=500, success=False, error_kind="upstream")  # not billable
    log_call(repo, "c1", endpoint=SCORE_EP, http_status=200, success=False)  # not billable
    log_call(repo, "c2", endpoint=SCORE_EP, pan_hash=PAN_B)  # another PAN
    assert repo.bureau_calls_today(PAN_A) == 3
    assert repo.bureau_calls_today(PAN_B) == 1
    assert repo.bureau_calls_today("c" * 64) == 0


def test_quota_limit_default_and_override() -> None:
    repo = MemoryRepo(FrozenClock())
    assert repo.quota_limit("lc-1") == 5
    repo.set_quota_override("lc-1", 12)
    assert repo.quota_limit("lc-1") == 12
    assert repo.quota_limit("lc-2") == 5
    repo.set_quota_override("lc-1", 0)  # daily_quota >= 0 is legal (blocks the LC)
    assert repo.quota_limit("lc-1") == 0
    with pytest.raises(ValueError):
        repo.set_quota_override("lc-1", -1)  # mirrors the SQL CHECK


# --------------------------------------------------------------------------- review queue


def test_unresolved_review_exists_and_resolve_helper() -> None:
    clock = FrozenClock()
    repo = MemoryRepo(clock)
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is False
    repo.insert_review_item(make_review(id="r1"))
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is True
    assert repo.unresolved_review_exists(PAN_A, "pan_daily_cap") is False  # other kind
    assert repo.unresolved_review_exists(PAN_B, "timeout_reconciliation") is False  # other PAN
    repo.insert_review_item(make_review(id="r2", kind="surepass_auth"))  # K11: the 8th kind
    assert repo.unresolved_review_exists(PAN_A, "surepass_auth") is True
    clock.advance(minutes=30)
    repo.resolve_review("r1", resolved_by="credit-1")
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is False
    (resolved,) = [r for r in repo.list_reviews() if r.id == "r1"]
    assert resolved.resolved_at == clock.now and resolved.resolved_by == "credit-1"
    with pytest.raises(ValueError):
        repo.insert_review_item(make_review(id="r1"))
    with pytest.raises(KeyError):
        repo.resolve_review("ghost")


def test_a_second_unresolved_row_keeps_blocking_after_the_first_is_resolved() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_review_item(make_review(id="r1"))
    repo.insert_review_item(make_review(id="r2"))
    repo.resolve_review("r1")
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is True


# --------------------------------------------------------------------------- PAN-correction events


def test_pan_events_are_per_opportunity_and_created_at_ascending() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_pan_event(make_pan_event(id="e2", event="correction", created_at=T0 + timedelta(hours=2)))
    repo.insert_pan_event(make_pan_event(id="e1", event="initial", created_at=T0 + timedelta(hours=1)))
    repo.insert_pan_event(make_pan_event(id="e3", opportunity_id="opp-2", created_at=T0))
    assert [e.id for e in repo.pan_events_for_opportunity("opp-1")] == ["e1", "e2"]
    assert [e.id for e in repo.pan_events_for_opportunity("opp-2")] == ["e3"]
    assert repo.pan_events_for_opportunity("opp-3") == []
    with pytest.raises(ValueError):
        repo.insert_pan_event(make_pan_event(id="e1"))


@pytest.mark.parametrize(
    ("kind", "status", "code", "expected"),
    [
        ("full", "ok", None, True),
        ("full", "pending_verification", None, True),  # spend/report existence unknown
        ("full", "refused", "PAN_LOCKED", False),
        ("full", "service_failure", "SUREPASS_FAILURE", False),
        ("full", "service_failure", "OPENAI_FAILURE", True),  # the report was pulled; only the ranking failed
        ("full", "ok", "NO_PINCODE_ON_PAN", True),  # a band read still pulled the report
        ("quick", "service_failure", "OPENAI_FAILURE", False),
        ("full", "running", None, False),
        ("full", "error", None, False),
        ("full", "pan_not_verified", None, False),
        ("quick", "ok", None, False),
    ],
)
def test_report_exists_for_opportunity(kind: str, status: str, code: str | None, expected: bool) -> None:
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(id="c", kind=kind, status=status, code=code, opportunity_id="opp-1"))
    assert repo.report_exists_for_opportunity("opp-1") is expected
    assert repo.report_exists_for_opportunity("opp-2") is False


# --------------------------------------------------------------------------- lenders


CRITERIA: dict[str, Any] = {"min_score_floor": {"value": 650, "status": "known"}}


def _seed(repo: MemoryRepo) -> None:
    repo.seed_lender("L2", 1, "Bharat", True, CRITERIA, "notes-2", ["560001", "110001"])
    repo.seed_lender("L1", 1, "Alpha", True, CRITERIA, "notes-1", ["110001", "400001"])
    repo.seed_lender("L1", 2, "Alpha v2 draft", False, CRITERIA, "draft", ["560001", "700001"])
    repo.seed_lender("L3", 1, "Crest", False, CRITERIA, "unpublished", ["110001"])


def test_published_lenders_are_published_only_ordered_by_id_without_pincodes() -> None:
    repo = MemoryRepo(FrozenClock())
    _seed(repo)
    rows = repo.published_lenders()
    assert [(r.id, r.version, r.name, r.published) for r in rows] == [("L1", 1, "Alpha", True), ("L2", 1, "Bharat", True)]
    assert all(r.matched_pincodes == () for r in rows)
    assert rows[0].criteria == CRITERIA and rows[0].policy_notes == "notes-1"
    rows[0].criteria["min_score_floor"]["value"] = 1  # a returned copy
    assert repo.published_lenders()[0].criteria == CRITERIA


def test_lenders_for_pincodes_joins_on_id_and_version_published_only() -> None:
    repo = MemoryRepo(FrozenClock())
    _seed(repo)
    # 560001 belongs to L2 v1 (published) and to L1 v2 (an UNPUBLISHED draft): L1 must not match through its draft.
    assert [(r.id, r.version) for r in repo.lenders_for_pincodes(["560001"])] == [("L2", 1)]
    assert repo.lenders_for_pincodes(["700001"]) == []  # only the draft carries it
    assert repo.lenders_for_pincodes(["110001"]) != []
    assert [r.id for r in repo.lenders_for_pincodes(["110001"])] == ["L1", "L2"]  # L3 unpublished, excluded


def test_lenders_for_pincodes_matched_pincodes_sorted_and_deduped_and_input_agnostic() -> None:
    repo = MemoryRepo(FrozenClock())
    _seed(repo)
    rows = repo.lenders_for_pincodes(("400001", "110001", "110001", "999999"))
    assert [(r.id, r.matched_pincodes) for r in rows] == [("L1", ("110001", "400001")), ("L2", ("110001",))]
    assert [r.id for r in repo.lenders_for_pincodes({"400001"})] == ["L1"]


def test_lenders_for_pincodes_empty_input_returns_empty() -> None:
    repo = MemoryRepo(FrozenClock())
    _seed(repo)
    assert repo.lenders_for_pincodes([]) == []
    assert repo.lenders_for_pincodes(set()) == []


def test_seed_lender_mirrors_the_one_published_version_index() -> None:
    repo = MemoryRepo(FrozenClock())
    repo.seed_lender("L1", 1, "Alpha", True, CRITERIA, "", [])
    with pytest.raises(ValueError):
        repo.seed_lender("L1", 2, "Alpha again", True, CRITERIA, "", [])
    repo.seed_lender("L1", 2, "Alpha draft", False, CRITERIA, "", [])
    with pytest.raises(ValueError):
        repo.seed_lender("L1", 2, "dup", False, CRITERIA, "", [])  # PK (id, version)
    with pytest.raises(ValueError):
        repo.seed_lender("L9", 1, "bad pincode", True, CRITERIA, "", ["012345"])  # mirrors the pincode CHECK


def test_set_lender_published_is_the_admin_publish_flip_under_the_one_published_index() -> None:
    """The admin's publish / unpublish action (test + eval helper): matching sees the change on the next read, and a
    second published version of the same lender is refused (idx_lenders_one_published)."""
    repo = MemoryRepo(FrozenClock())
    _seed(repo)  # L1 v1 published, L1 v2 draft
    with pytest.raises(ValueError):
        repo.set_lender_published("L1", 2, True)  # v1 is still published
    repo.set_lender_published("L1", 1, False)
    assert [r.id for r in repo.published_lenders()] == ["L2"]
    assert [r.id for r in repo.lenders_for_pincodes(["400001"])] == []  # only L1 v1 carried it
    repo.set_lender_published("L1", 2, True)
    assert [(r.id, r.version, r.name) for r in repo.lenders_for_pincodes(["700001"])] == [("L1", 2, "Alpha v2 draft")]
    repo.set_lender_published("L1", 2, True)  # idempotent
    with pytest.raises(KeyError):
        repo.set_lender_published("L9", 1, False)  # no such (id, version)


# --------------------------------------------------------------------------- serialisation sanity


def test_memory_rows_are_json_serialisable_dataclasses() -> None:
    """lc_view / internal are jsonb: the dict fields must survive a json round-trip untouched."""
    repo = MemoryRepo(FrozenClock())
    repo.insert_check(make_check(lc_view={"a": [1, {"b": None}]}, internal={"c": "d"}))
    got = repo.get_check("chk-1")
    assert got is not None
    assert json.loads(json.dumps(got.lc_view)) == got.lc_view
    assert isinstance(got, CheckRow)
