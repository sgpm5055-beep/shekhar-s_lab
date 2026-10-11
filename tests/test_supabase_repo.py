"""Task 2: SupabaseRepo / SupabaseRawStore call shapes, against an in-memory fake of the supabase-py client.

MemoryRepo is the only backend the rest of the suite uses; this file keeps the real backend honest without any
network (pytest-socket stays on): filters, rpc names/params, upload options, row <-> dataclass round trips.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import pytest
from fake_supabase import FakeSupabase
from repo_builders import (
    ENVELOPE,
    PAN_A,
    PAN_B,
    PAN_EP,
    REPORT_EP,
    SCORE_EP,
    T0,
    FrozenClock,
    make_check,
    make_consent,
    make_pan_event,
    make_pan_record,
    make_report_cache,
    make_review,
    make_score_cache,
)

from eligibility_agent.db.repo import (
    PurgeRepo,
    RawStore,
    RawUnavailable,
    Repo,
    SupabaseRawStore,
    SupabaseRepo,
    create_supabase_backend,
)


def _build() -> tuple[SupabaseRepo, FakeSupabase, FrozenClock]:
    fake = FakeSupabase()
    clock = FrozenClock()
    client: Any = fake
    return SupabaseRepo(client, SupabaseRawStore(client), clock), fake, clock


def _log(repo: SupabaseRepo, check_id: str = "chk-1", **kwargs: Any) -> str | None:
    params: dict[str, Any] = {
        "check_id": check_id,
        "pan_hash": PAN_A,
        "endpoint": SCORE_EP,
        "client_id": "cid-1",
        "http_status": 200,
        "success": True,
        "message_code": None,
        "latency_ms": 120,
        "error_kind": None,
        "raw_envelope": ENVELOPE,
    }
    params.update(kwargs)
    return repo.log_surepass_call(**params)


def test_supabase_repo_satisfies_both_protocols_statically() -> None:
    repo, _, _ = _build()
    typed: Repo = repo
    purge: PurgeRepo = repo
    raw: RawStore = SupabaseRawStore(cast(Any, FakeSupabase()))
    assert (typed, purge, raw) is not None


# --------------------------------------------------------------------------- raw store


def test_raw_store_upload_shape_and_round_trip() -> None:
    fake = FakeSupabase()
    store = SupabaseRawStore(cast(Any, fake))  # default bucket
    envelope = {"success": True, "data": {"n": [1, 2]}}
    store.put_raw("pan/h/experian-score/k.json", envelope)
    ((bucket, path, options),) = fake.uploads
    assert (bucket, path) == ("raw-responses", "pan/h/experian-score/k.json")
    assert options == {"content-type": "application/json", "upsert": "true"}
    assert fake.objects[("raw-responses", path)] == b'{"success":true,"data":{"n":[1,2]}}'  # compact separators
    assert store.get_raw(path) == envelope


def test_raw_store_missing_object_is_raw_unavailable_and_custom_bucket_is_used() -> None:
    fake = FakeSupabase()
    store = SupabaseRawStore(cast(Any, fake), bucket="other")
    with pytest.raises(RawUnavailable):
        store.get_raw("gone.json")
    store.put_raw("k.json", {"a": 1})
    assert ("other", "k.json") in fake.objects


def test_raw_store_delete_calls_remove_with_the_keys_and_skips_an_empty_batch() -> None:
    fake = FakeSupabase()
    store = SupabaseRawStore(cast(Any, fake))
    store.put_raw("a.json", {})
    store.put_raw("b.json", {})
    store.delete_raw([])
    assert fake.removals == []
    store.delete_raw(["a.json", "b.json", "never-existed.json"])
    assert fake.removals == [["a.json", "b.json", "never-existed.json"]]
    assert fake.objects == {}


# --------------------------------------------------------------------------- checks


def test_check_round_trips_through_the_table_with_iso_timestamps() -> None:
    repo, fake, _ = _build()
    row = make_check(lc_view={"a": [1]}, internal={"b": None}, status="refused", code="PAN_LOCKED", billed=True)
    repo.insert_check(row)
    stored = fake.tables["checks"][0]
    assert stored["created_at"] == "2026-10-10T08:00:00+00:00"
    assert stored["status"] == "refused" and stored["lc_view"] == {"a": [1]}
    assert repo.get_check("chk-1") == row
    assert repo.get_check("nope") is None
    with pytest.raises(ValueError):
        repo.insert_check(row)  # PK violation (23505) surfaces as the protocol's ValueError


def test_update_check_applies_terminal_patches_only_while_running() -> None:
    repo, fake, _ = _build()
    repo.insert_check(make_check())
    repo.update_check("chk-1", {"status": "ok", "billed": True, "lc_view": {"headline": "possible"}})
    query = fake.queries[-1]
    assert query.op == "update" and query.has_filter("eq", "id", "chk-1") and query.has_filter("eq", "status", "running")
    assert fake.tables["checks"][0]["status"] == "ok" and fake.tables["checks"][0]["billed"] is True

    repo.update_check("chk-1", {"status": "error", "billed": False})  # row no longer running: silent no-op
    assert fake.tables["checks"][0]["status"] == "ok" and fake.tables["checks"][0]["billed"] is True


def test_contact_id_only_patch_is_not_status_guarded_and_empty_patch_sends_nothing() -> None:
    repo, fake, _ = _build()
    repo.insert_check(make_check(status="ok"))
    repo.update_check("chk-1", {"contact_id": "contact-9"})
    query = fake.queries[-1]
    assert query.has_filter("eq", "id", "chk-1") and not query.has_filter("eq", "status")
    assert fake.tables["checks"][0]["contact_id"] == "contact-9"
    sent = len(fake.queries)
    repo.update_check("chk-1", {})
    assert len(fake.queries) == sent
    repo.update_check("chk-1", {"contact_id": "c2", "billed": True})  # mixed: guarded as a whole
    assert fake.queries[-1].has_filter("eq", "status", "running")
    assert fake.tables["checks"][0]["contact_id"] == "contact-9"


def test_record_check_error_updates_inserts_or_leaves_a_terminal_row() -> None:
    repo, fake, clock = _build()
    kwargs: dict[str, Any] = {"kind": "quick", "lc_id": "lc-1", "pan_hash": PAN_A, "pan_last4": "234F"}
    repo.insert_check(make_check())
    repo.record_check_error(check_id="chk-1", message="boom", **kwargs)
    assert fake.tables["checks"][0]["status"] == "error" and fake.tables["checks"][0]["internal"] == {"error": "boom"}
    assert len(fake.tables["checks"]) == 1

    repo.record_check_error(check_id="chk-1", message="again", **kwargs)  # now terminal: untouched, no raise
    assert fake.tables["checks"][0]["internal"] == {"error": "boom"}

    clock.advance(seconds=3)
    repo.record_check_error(check_id="chk-new", message="crash", **kwargs)  # admission never inserted a row
    inserted = repo.get_check("chk-new")
    assert inserted is not None
    assert (inserted.status, inserted.code, inserted.internal) == ("error", None, {"error": "crash"})
    assert (inserted.billed, inserted.counted_against_quota, inserted.is_refresh) == (False, False, False)
    assert inserted.created_at == clock.now and inserted.pan_last4 == "234F"


def test_attach_contact_updates_the_check_and_its_consent() -> None:
    repo, fake, _ = _build()
    repo.insert_consent(make_consent(id="con-1"))
    repo.insert_consent(make_consent(id="con-2"))
    repo.insert_check(make_check(consent_id="con-1", status="ok"))
    repo.attach_contact("chk-1", "contact-42")
    assert fake.tables["checks"][0]["contact_id"] == "contact-42"
    contacts = {r["id"]: r["contact_id"] for r in fake.tables["consents"]}
    assert contacts == {"con-1": "contact-42", "con-2": None}
    repo.attach_contact("ghost", "contact-1")  # unknown check: no-op


# --------------------------------------------------------------------------- consents + caches


def test_consent_round_trip_shape() -> None:
    repo, fake, _ = _build()
    repo.insert_consent(make_consent())
    stored = fake.tables["consents"][0]
    assert stored["given_at"] == "2026-10-10T08:00:00+00:00" and stored["lc_typed_name"] == "Asha Verma"
    assert stored["covers"] == "score"
    with pytest.raises(ValueError):
        repo.insert_consent(make_consent())
    assert repo.get_consent(make_consent().id) == make_consent()  # timestamps decoded back to aware datetimes
    assert repo.get_consent("con-9") is None


def test_cache_rows_round_trip_and_upsert() -> None:
    repo, fake, _ = _build()
    pan, score, report = make_pan_record(), make_score_cache(source="report"), make_report_cache(report_date=T0.date())
    repo.put_pan_record(pan)
    repo.put_score_cache(score)
    repo.put_report_cache(report)
    assert repo.get_pan_record(PAN_A) == pan
    assert repo.get_score_cache(PAN_A) == score
    assert repo.get_report_cache(PAN_A) == report
    assert fake.tables["report_cache"][0]["report_date"] == "2026-10-10"
    repo.put_pan_record(make_pan_record(storage_key="k2"))
    assert len(fake.tables["pan_records"]) == 1 and fake.tables["pan_records"][0]["storage_key"] == "k2"
    assert repo.get_pan_record(PAN_B) is None and repo.get_score_cache(PAN_B) is None


def test_fetch_raw_delegates_to_the_raw_store() -> None:
    repo, _, _ = _build()
    repo.insert_check(make_check())
    key = _log(repo)
    assert key is not None and repo.fetch_raw(key) == ENVELOPE
    with pytest.raises(RawUnavailable):
        repo.fetch_raw("missing.json")


# --------------------------------------------------------------------------- the sink


def test_sink_uploads_first_then_inserts_and_returns_the_key() -> None:
    repo, fake, clock = _build()
    repo.insert_check(make_check())
    fake.events.clear()
    key = _log(repo, endpoint=REPORT_EP)
    (row,) = fake.tables["surepass_calls"]
    assert key == f"pan/{PAN_A}/experian-report/20261010T080000Z-{row['id']}.json"
    assert fake.events == [f"upload:{key}", "insert:surepass_calls"]  # raw object first, row second
    assert row["storage_key"] == key and row["billable"] is True and row["endpoint"] == REPORT_EP
    assert row["called_at"] == "2026-10-10T08:00:00+00:00"
    assert row["purge_after"] == (clock.now + timedelta(days=180)).isoformat()
    assert row["purged_at"] is None and row["pan_hash"] == PAN_A and row["latency_ms"] == 120


def test_sink_without_a_body_uploads_nothing_and_a_timeout_is_billable() -> None:
    repo, fake, _ = _build()
    repo.insert_check(make_check())
    key = _log(repo, http_status=None, success=None, error_kind="timeout", raw_envelope=None)
    assert key is None and fake.uploads == []
    (row,) = fake.tables["surepass_calls"]
    assert row["billable"] is True and row["storage_key"] is None and row["error_kind"] == "timeout"


def test_sink_billable_follows_the_truth_table() -> None:
    repo, fake, _ = _build()
    repo.insert_check(make_check())
    _log(repo, http_status=200, success=False)
    _log(repo, http_status=500, success=False, error_kind="upstream")
    _log(repo, http_status=200, success=True)
    assert [r["billable"] for r in fake.tables["surepass_calls"]] == [False, False, True]


def test_surepass_calls_for_check_orders_by_called_at_then_id() -> None:
    repo, fake, clock = _build()
    repo.insert_check(make_check(id="c1"))
    repo.insert_check(make_check(id="c2"))
    _log(repo, "c1", endpoint=PAN_EP)
    clock.advance(seconds=1)
    _log(repo, "c2")
    clock.advance(seconds=1)
    _log(repo, "c1", endpoint=SCORE_EP)
    assert [r.endpoint for r in repo.surepass_calls_for_check("c1")] == [PAN_EP, SCORE_EP]
    assert fake.queries[-1].orders == [("called_at", False), ("id", False)]
    assert repo.surepass_calls_for_check("nope") == []


# --------------------------------------------------------------------------- quota / caps / locks (rpc mappings)


def test_quota_and_cap_rpc_mappings() -> None:
    repo, fake, _ = _build()
    fake.rpc_handlers["quick_checks_used_today"] = lambda _p: 3
    fake.rpc_handlers["bureau_calls_today"] = lambda _p: 2
    assert repo.quick_checks_used_today("lc-1") == 3
    assert repo.bureau_calls_today(PAN_A) == 2
    assert fake.rpcs == [("quick_checks_used_today", {"p_lc_id": "lc-1"}), ("bureau_calls_today", {"p_pan_hash": PAN_A})]


def test_try_acquire_maps_to_the_atomic_rpc() -> None:
    repo, fake, clock = _build()
    outcomes = iter([True, False])
    fake.rpc_handlers["acquire_pan_lock"] = lambda _p: next(outcomes)
    until = clock.now + timedelta(seconds=180)
    assert repo.try_acquire_pan_lock(PAN_A, "chk-1", until) is True
    assert repo.try_acquire_pan_lock(PAN_A, "chk-2", until) is False
    assert fake.rpcs[0] == (
        "acquire_pan_lock",
        {"p_pan_hash": PAN_A, "p_check_id": "chk-1", "p_until": "2026-10-10T08:03:00+00:00"},
    )


def test_quota_limit_reads_the_override_row_even_when_zero() -> None:
    repo, fake, _ = _build()
    assert repo.quota_limit("lc-1") == 5
    fake.tables["lc_quota_overrides"].append({"lc_id": "lc-1", "daily_quota": 12, "set_by": "mgr", "updated_at": ""})
    fake.tables["lc_quota_overrides"].append({"lc_id": "lc-2", "daily_quota": 0, "set_by": "mgr", "updated_at": ""})
    assert repo.quota_limit("lc-1") == 12
    assert repo.quota_limit("lc-2") == 0  # 0 is an override, not "missing"
    assert repo.quota_limit("lc-3") == 5


def _lock_row(kind: str, check_id: str | None, until: datetime, released: datetime | None = None) -> dict[str, Any]:
    return {
        "pan_hash": PAN_A,
        "kind": kind,
        "check_id": check_id,
        "locked_at": T0.isoformat(),
        "lock_until": until.isoformat(),
        "released_at": released.isoformat() if released else None,
    }


def test_release_is_holder_only_in_flight_only_and_idempotent() -> None:
    repo, fake, clock = _build()
    fake.tables["pan_locks"].append(_lock_row("in_flight", "chk-1", T0 + timedelta(seconds=180)))
    fake.tables["pan_locks"].append(_lock_row("timeout", "chk-1", T0 + timedelta(minutes=10)))
    repo.release_pan_lock(PAN_A, "someone-else")
    assert all(r["released_at"] is None for r in fake.tables["pan_locks"])
    clock.advance(seconds=5)
    repo.release_pan_lock(PAN_A, "chk-1")
    released = {r["kind"]: r["released_at"] for r in fake.tables["pan_locks"]}
    assert released == {"in_flight": "2026-10-10T08:00:05+00:00", "timeout": None}  # the timeout lock survives
    clock.advance(seconds=5)
    repo.release_pan_lock(PAN_A, "chk-1")  # idempotent: the first stamp is kept
    assert fake.tables["pan_locks"][0]["released_at"] == "2026-10-10T08:00:05+00:00"


def test_create_pan_lock_upserts_per_pan_and_kind() -> None:
    repo, fake, clock = _build()
    repo.create_pan_lock(PAN_A, "timeout", "chk-1", clock.now + timedelta(minutes=10))
    repo.create_pan_lock(PAN_A, "timeout", "chk-2", clock.now + timedelta(minutes=20))
    repo.create_pan_lock(PAN_A, "daily_cap", None, clock.now + timedelta(hours=10))
    assert fake.queries[0].op == "upsert" and fake.queries[0].on_conflict == "pan_hash,kind"
    rows = {r["kind"]: r for r in fake.tables["pan_locks"]}
    assert len(fake.tables["pan_locks"]) == 2 and rows["timeout"]["check_id"] == "chk-2"
    assert rows["timeout"]["released_at"] is None and rows["timeout"]["locked_at"] == clock.now.isoformat()


def test_get_active_locks_filters_released_and_expired_against_the_injected_clock() -> None:
    repo, fake, clock = _build()
    fake.tables["pan_locks"].extend(
        [
            _lock_row("daily_cap", None, T0 + timedelta(hours=10)),  # active
            _lock_row("in_flight", "chk-1", T0 + timedelta(seconds=180), released=T0),  # released
            _lock_row("timeout", "chk-2", T0),  # lock_until == now: expired (strictly greater is required)
        ]
    )
    assert [lock.kind for lock in repo.get_active_locks(PAN_A)] == ["daily_cap"]
    assert repo.get_active_locks(PAN_B) == []
    query = fake.queries[0]
    assert query.has_filter("is", "released_at") and query.has_filter("gt", "lock_until", clock.now.isoformat())
    lock = repo.get_active_locks(PAN_A)[0]
    assert lock.lock_until == T0 + timedelta(hours=10) and lock.check_id is None and lock.released_at is None


# --------------------------------------------------------------------------- review / pan events


def test_review_queue_round_trip() -> None:
    repo, fake, _ = _build()
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is False
    repo.insert_review_item(make_review(id="r1"))
    repo.insert_review_item(make_review(id="r2", kind="surepass_auth"))
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is True
    assert repo.unresolved_review_exists(PAN_A, "surepass_auth") is True
    assert repo.unresolved_review_exists(PAN_A, "pan_daily_cap") is False
    assert repo.unresolved_review_exists(PAN_B, "timeout_reconciliation") is False
    fake.tables["review_queue"][0]["resolved_at"] = (T0 + timedelta(hours=1)).isoformat()
    assert repo.unresolved_review_exists(PAN_A, "timeout_reconciliation") is False
    with pytest.raises(ValueError):
        repo.insert_review_item(make_review(id="r2"))


def test_pan_events_ordered_ascending_and_report_exists() -> None:
    repo, _, _ = _build()
    repo.insert_pan_event(make_pan_event(id="e2", event="correction", created_at=T0 + timedelta(hours=2)))
    repo.insert_pan_event(make_pan_event(id="e1", created_at=T0 + timedelta(hours=1)))
    repo.insert_pan_event(make_pan_event(id="e3", opportunity_id="opp-2"))
    events = repo.pan_events_for_opportunity("opp-1")
    assert [e.id for e in events] == ["e1", "e2"] and events[1].event == "correction"
    assert events[0] == make_pan_event(id="e1", created_at=T0 + timedelta(hours=1))
    assert repo.pan_events_for_opportunity("opp-9") == []

    repo.insert_check(make_check(id="a", kind="full", status="refused", code="PAN_LOCKED", opportunity_id="opp-1"))
    assert repo.report_exists_for_opportunity("opp-1") is False
    repo.insert_check(make_check(id="b", kind="quick", status="ok", opportunity_id="opp-1"))
    assert repo.report_exists_for_opportunity("opp-1") is False
    repo.insert_check(make_check(id="c", kind="full", status="pending_verification", opportunity_id="opp-1"))
    assert repo.report_exists_for_opportunity("opp-1") is True
    assert repo.report_exists_for_opportunity("opp-2") is False

    # service_failure counts only with OPENAI_FAILURE (the report was pulled; the ranking failed afterwards)
    failed = {"kind": "full", "status": "service_failure", "opportunity_id": "opp-2"}
    repo.insert_check(make_check(id="d", code="SUREPASS_FAILURE", **failed))
    assert repo.report_exists_for_opportunity("opp-2") is False
    repo.insert_check(make_check(id="e", code="OPENAI_FAILURE", **failed))
    assert repo.report_exists_for_opportunity("opp-2") is True


# --------------------------------------------------------------------------- lenders


def _seed_lenders(fake: FakeSupabase) -> None:
    crit = {"min_score_floor": {"value": 650, "status": "known"}}
    for lender_id, version, name, published in (
        ("L2", 1, "Bharat", True),
        ("L1", 1, "Alpha", True),
        ("L1", 2, "Alpha draft", False),
        ("L3", 1, "Crest", False),
    ):
        fake.tables["lenders"].append(
            {
                "id": lender_id,
                "version": version,
                "name": name,
                "published": published,
                "criteria": crit,
                "policy_notes": f"notes-{lender_id}",
                "created_by": None,
                "created_at": T0.isoformat(),
            }
        )
    for lender_id, version, pincode in (
        ("L2", 1, "560001"),
        ("L2", 1, "110001"),
        ("L1", 1, "110001"),
        ("L1", 1, "400001"),
        ("L1", 2, "560001"),  # the draft's coverage must not leak into the published L1
        ("L3", 1, "110001"),
    ):
        fake.tables["lender_pincodes"].append({"lender_id": lender_id, "version": version, "pincode": pincode})


def test_published_lenders_are_published_only_ordered_by_id() -> None:
    repo, fake, _ = _build()
    _seed_lenders(fake)
    rows = repo.published_lenders()
    assert [(r.id, r.version, r.name, r.published) for r in rows] == [("L1", 1, "Alpha", True), ("L2", 1, "Bharat", True)]
    assert all(r.matched_pincodes == () for r in rows)
    assert rows[0].criteria == {"min_score_floor": {"value": 650, "status": "known"}}
    assert rows[0].policy_notes == "notes-L1"


def test_lenders_for_pincodes_joins_versions_app_side_published_only() -> None:
    repo, fake, _ = _build()
    _seed_lenders(fake)
    assert [(r.id, r.version) for r in repo.lenders_for_pincodes(["560001"])] == [("L2", 1)]  # L1 only via its draft
    rows = repo.lenders_for_pincodes(("400001", "110001", "110001", "999999"))
    assert [(r.id, r.matched_pincodes) for r in rows] == [("L1", ("110001", "400001")), ("L2", ("110001",))]
    assert repo.lenders_for_pincodes(["700001"]) == []


def test_lenders_for_pincodes_empty_input_issues_no_query() -> None:
    repo, fake, _ = _build()
    _seed_lenders(fake)
    assert repo.lenders_for_pincodes([]) == []
    assert repo.lenders_for_pincodes(set()) == []
    assert fake.queries == []


# --------------------------------------------------------------------------- purge operations


def test_purge_operations_round_trip() -> None:
    repo, fake, clock = _build()
    repo.insert_consent(make_consent(id="con-old", purge_after=T0 + timedelta(days=10)))
    repo.insert_consent(make_consent(id="con-new"))
    repo.insert_check(make_check(consent_id="con-old"))
    key = _log(repo)
    clock.advance(seconds=1)
    _log(repo, http_status=None, success=None, error_kind="transport", raw_envelope=None)
    clock.advance(days=100)
    _log(repo)  # not due at day 181
    repo.put_pan_record(make_pan_record())
    repo.put_score_cache(make_score_cache(purge_after=T0 + timedelta(days=400), consent_id="con-old"))
    repo.put_report_cache(make_report_cache(consent_id="con-old"))
    now = T0 + timedelta(days=181)

    due = repo.expired_surepass_calls(now, limit=None)
    assert len(due) == 2 and due[0].storage_key == key
    assert len(repo.expired_surepass_calls(now, limit=1)) == 1
    repo.mark_calls_purged([r.id for r in due], now)
    assert repo.expired_surepass_calls(now, limit=None) == []
    assert len(fake.tables["surepass_calls"]) == 3  # metadata rows are kept

    assert repo.expired_cache_rows(now) == [f"pan_records:{PAN_A}", f"report_cache:{PAN_A}"]
    assert repo.delete_expired_cache_rows(now) == 2
    assert repo.get_pan_record(PAN_A) is None and repo.get_score_cache(PAN_A) is not None

    assert repo.expired_consent_ids(now) == ["con-old"]
    assert repo.delete_expired_consents(now) == 1
    assert [r["id"] for r in fake.tables["consents"]] == ["con-new"]


def test_expired_surepass_calls_pages_through_everything_when_unbounded() -> None:
    repo, fake, _ = _build()
    repo.insert_check(make_check())
    for _ in range(2500):
        fake.tables["surepass_calls"].append(
            {
                "id": f"id-{len(fake.tables['surepass_calls']):05d}",
                "check_id": "chk-1",
                "pan_hash": PAN_A,
                "endpoint": SCORE_EP,
                "client_id": None,
                "http_status": 200,
                "success": True,
                "message_code": None,
                "latency_ms": None,
                "billable": True,
                "error_kind": None,
                "storage_key": None,
                "called_at": T0.isoformat(),
                "purge_after": T0.isoformat(),
                "purged_at": None,
            }
        )
    assert len(repo.expired_surepass_calls(T0 + timedelta(days=1), limit=None)) == 2500  # > PostgREST's 1000-row page


# --------------------------------------------------------------------------- the factory (the only supabase import)


def test_create_supabase_backend_wires_client_bucket_and_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    import supabase

    fake = FakeSupabase()
    seen: list[tuple[str, str]] = []

    def fake_create_client(url: str, key: str) -> FakeSupabase:
        seen.append((url, key))
        return fake

    monkeypatch.setattr(supabase, "create_client", fake_create_client)
    clock = FrozenClock()
    repo, raw = create_supabase_backend("https://x.supabase.co", "svc-key", bucket="b2", clock=clock)
    assert seen == [("https://x.supabase.co", "svc-key")]
    assert isinstance(repo, SupabaseRepo) and isinstance(raw, SupabaseRawStore)
    raw.put_raw("k.json", {"a": 1})
    assert ("b2", "k.json") in fake.objects
    repo.insert_check(make_check())
    assert fake.tables["checks"][0]["id"] == "chk-1"
