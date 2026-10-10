"""Task 2: scripts/purge_raw.py (3 steps + --dry-run) run against MemoryRepo + MemoryRawStore."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import pytest
from repo_builders import (
    PAN_A,
    PAN_B,
    T0,
    FrozenClock,
    log_call,
    make_check,
    make_consent,
    make_pan_record,
    make_report_cache,
    make_score_cache,
)

from eligibility_agent.db.repo import MemoryRawStore, MemoryRepo, PurgeRepo, RawStore, Repo
from scripts.purge_raw import BATCH_SIZE, SAMPLE_SIZE, PurgeReport, main, run_purge

NOW = T0 + timedelta(days=181)  # T0-born rows (purge_after = T0 + 180d) are due; T0+100d-born rows are not.


class _SpyRawStore(MemoryRawStore):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []

    def delete_raw(self, keys: Sequence[str]) -> None:
        self.batches.append(len(keys))
        super().delete_raw(keys)


def _world() -> tuple[MemoryRepo, FrozenClock, dict[str, str | None]]:
    """Three call rows (one due w/ body, one due w/o body, one not yet due), caches + consents of both ages."""
    clock = FrozenClock(T0)
    repo = MemoryRepo(clock)
    repo.insert_consent(make_consent(id="con-old", purge_after=T0 + timedelta(days=10)))
    repo.insert_consent(make_consent(id="con-new"))  # purge_after = T0 + 5y
    repo.insert_check(make_check(id="c1", consent_id="con-old"))
    keys: dict[str, str | None] = {}
    keys["due"] = log_call(repo, "c1")  # called_at T0 -> purge_after T0+180d
    keys["due_no_body"] = log_call(repo, "c1", http_status=None, success=None, error_kind="transport", envelope=None)
    clock.advance(days=100)
    keys["fresh"] = log_call(repo, "c1")  # purge_after T0+280d
    repo.put_pan_record(make_pan_record(storage_key=keys["due"] or ""))  # purge_after T0+180d: due
    repo.put_report_cache(make_report_cache(storage_key=keys["due"] or "", consent_id="con-old"))  # due
    repo.put_score_cache(make_score_cache(storage_key=keys["fresh"] or "", purge_after=T0 + timedelta(days=400)))
    return repo, clock, keys


def test_memory_repo_is_both_a_repo_and_a_purge_repo_statically() -> None:
    repo = MemoryRepo()
    typed_repo: Repo = repo
    purge_repo: PurgeRepo = repo
    raw_store: RawStore = repo.raw_store
    assert (typed_repo, purge_repo, raw_store) is not None


def test_step_1_deletes_due_objects_marks_purged_and_keeps_the_call_log_rows() -> None:
    repo, _, keys = _world()
    report = run_purge(repo, repo.raw_store, now=NOW)
    assert (report.calls, report.objects) == (2, 1)  # the no-body row is marked purged but has no object
    assert repo.raw_store.keys() == [keys["fresh"]]
    calls = {row.storage_key: row for row in repo.list_calls()}
    assert len(repo.list_calls()) == 3  # the call-log METADATA row is kept forever
    due = calls[keys["due"]]
    assert due.purged_at == NOW and due.storage_key == keys["due"]
    no_body = calls[None]
    assert no_body.purged_at == NOW
    assert calls[keys["fresh"]].purged_at is None
    assert report.call_key_sample == (keys["due"],)


def test_a_purged_object_is_unreadable_afterwards() -> None:
    from eligibility_agent.db.repo import RawUnavailable

    repo, _, keys = _world()
    run_purge(repo, repo.raw_store, now=NOW)
    with pytest.raises(RawUnavailable):
        repo.fetch_raw(keys["due"] or "")
    assert repo.fetch_raw(keys["fresh"] or "")["success"] is True


def test_step_2_deletes_due_cache_rows_of_all_three_tables() -> None:
    repo, _, keys = _world()
    repo.put_pan_record(make_pan_record(pan_hash=PAN_B, purge_after=T0 + timedelta(days=500)))  # not due
    report = run_purge(repo, repo.raw_store, now=NOW)
    assert report.cache_rows == 2
    assert repo.get_pan_record(PAN_A) is None  # due
    assert repo.get_report_cache(PAN_A) is None  # due
    assert repo.get_pan_record(PAN_B) is not None  # kept
    score = repo.get_score_cache(PAN_A)
    assert score is not None and score.storage_key == keys["fresh"]  # purge_after T0+400d: kept
    assert report.cache_row_sample == (f"pan_records:{PAN_A}", f"report_cache:{PAN_A}")


def test_step_3_deletes_due_consents_and_nulls_the_references_like_on_delete_set_null() -> None:
    repo, _, _ = _world()
    report = run_purge(repo, repo.raw_store, now=NOW)
    assert report.consents == 1 and report.consent_sample == ("con-old",)
    assert [c.id for c in repo.list_consents()] == ["con-new"]
    check = repo.get_check("c1")
    assert check is not None and check.consent_id is None  # the check row survives, unlinked


def test_purge_after_is_a_strict_cutoff() -> None:
    clock = FrozenClock(T0)
    repo = MemoryRepo(clock)
    repo.insert_check(make_check(id="c1"))
    log_call(repo, "c1")  # purge_after == T0 + 180d
    repo.put_pan_record(make_pan_record())  # purge_after == T0 + 180d
    repo.insert_consent(make_consent(purge_after=T0 + timedelta(days=180)))
    on_the_dot = run_purge(repo, repo.raw_store, now=T0 + timedelta(days=180))
    assert (on_the_dot.calls, on_the_dot.cache_rows, on_the_dot.consents) == (0, 0, 0)
    just_after = run_purge(repo, repo.raw_store, now=T0 + timedelta(days=180, microseconds=1))
    assert (just_after.calls, just_after.cache_rows, just_after.consents) == (1, 1, 1)


def test_second_run_is_a_no_op() -> None:
    repo, _, _ = _world()
    run_purge(repo, repo.raw_store, now=NOW)
    again = run_purge(repo, repo.raw_store, now=NOW + timedelta(hours=1))
    assert (again.calls, again.objects, again.cache_rows, again.consents) == (0, 0, 0, 0)
    assert again.call_key_sample == () and again.cache_row_sample == () and again.consent_sample == ()


def test_step_1_works_in_batches_of_100() -> None:
    clock = FrozenClock(T0)
    spy = _SpyRawStore()
    repo = MemoryRepo(clock, raw_store=spy)
    repo.insert_check(make_check(id="c1"))
    for _ in range(250):
        log_call(repo, "c1")
    assert BATCH_SIZE == 100
    report = run_purge(repo, spy, now=NOW)
    assert spy.batches == [100, 100, 50]
    assert (report.calls, report.objects) == (250, 250)
    assert spy.keys() == []
    assert all(row.purged_at == NOW for row in repo.list_calls())


def test_dry_run_changes_nothing_and_reports_counts_and_the_first_ten_keys() -> None:
    clock = FrozenClock(T0)
    repo = MemoryRepo(clock)
    repo.insert_consent(make_consent(id="con-old", purge_after=T0 + timedelta(days=10)))
    repo.insert_check(make_check(id="c1"))
    for _ in range(12):
        log_call(repo, "c1")
        clock.advance(seconds=1)
    repo.put_pan_record(make_pan_record())
    repo.put_score_cache(make_score_cache())
    before_keys = repo.raw_store.keys()
    report = run_purge(repo, repo.raw_store, now=NOW, dry_run=True)
    assert report.dry_run is True
    assert (report.calls, report.objects, report.cache_rows, report.consents) == (12, 12, 2, 1)
    assert len(report.call_key_sample) == SAMPLE_SIZE == 10
    expected_first_ten = tuple(row.storage_key for row in repo.surepass_calls_for_check("c1")[:10])
    assert report.call_key_sample == expected_first_ten
    assert report.cache_row_sample == (f"pan_records:{PAN_A}", f"score_cache:{PAN_A}")
    assert report.consent_sample == ("con-old",)
    # ...and nothing was touched
    assert repo.raw_store.keys() == before_keys
    assert all(row.purged_at is None for row in repo.list_calls())
    assert repo.get_pan_record(PAN_A) is not None and repo.get_score_cache(PAN_A) is not None
    assert [c.id for c in repo.list_consents()] == ["con-old"]


def test_nothing_due_returns_an_empty_report() -> None:
    repo = MemoryRepo(FrozenClock(T0))
    repo.insert_check(make_check(id="c1"))
    log_call(repo, "c1")
    report = run_purge(repo, repo.raw_store, now=T0 + timedelta(days=1))
    assert report == PurgeReport(
        dry_run=False,
        calls=0,
        objects=0,
        call_key_sample=(),
        cache_rows=0,
        cache_row_sample=(),
        consents=0,
        consent_sample=(),
    )


# --------------------------------------------------------------------------- CLI


def test_main_dry_run_prints_counts_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    repo, _, keys = _world()
    code = main(["--dry-run"], backend=lambda: (repo, repo.raw_store), clock=lambda: NOW)
    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run" in out
    assert "2 call rows" in out and "1 objects" in out  # step 1
    assert "2 cache rows" in out  # step 2
    assert "1 consents" in out  # step 3
    assert (keys["due"] or "") in out  # the first keys are printed
    assert all(row.purged_at is None for row in repo.list_calls())  # nothing deleted


def test_main_real_run_purges_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    repo, _, keys = _world()
    code = main([], backend=lambda: (repo, repo.raw_store), clock=lambda: NOW)
    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run" not in out
    assert repo.raw_store.keys() == [keys["fresh"]]
    assert repo.get_pan_record(PAN_A) is None


def test_main_without_credentials_exits_2_and_never_builds_a_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "RAW_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    assert main([]) == 2
    assert "SUPABASE_URL" in capsys.readouterr().err


def test_main_rejects_unknown_flags() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--nuke"])
    assert excinfo.value.code == 2


def test_main_builds_the_supabase_backend_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_create(url: str, key: str, *, bucket: str, clock: Any) -> tuple[MemoryRepo, MemoryRawStore]:
        seen.update(url=url, key=key, bucket=bucket)
        repo = MemoryRepo(clock)
        return repo, repo.raw_store

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")
    monkeypatch.setenv("RAW_BUCKET", "other-bucket")
    monkeypatch.setattr("scripts.purge_raw.create_supabase_backend", fake_create)
    assert main(["--dry-run"], clock=lambda: datetime(2026, 10, 10, tzinfo=T0.tzinfo)) == 0
    assert seen == {"url": "https://example.supabase.co", "key": "service-role-key", "bucket": "other-bucket"}
