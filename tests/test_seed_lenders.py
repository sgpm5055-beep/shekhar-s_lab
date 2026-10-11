"""Task 16: scripts/seed_lenders.py.

Three layers: the script against MemoryRepo (the contract: the seeded database equals the one the scenario suite builds
from the same single source), the Supabase writer against the in-memory fake client (call order, all-or-nothing cleanup,
created_by), and the CLI (--dry-run connects to nothing, a missing credential exits 2).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from fake_supabase import FakeResponse, FakeSupabase
from postgrest.exceptions import APIError

from eligibility_agent.db.repo import LenderRow, LenderWriter, MemoryRepo, SupabaseRawStore, SupabaseRepo
from eligibility_agent.lenders.gate import PublishedLenderRow, gate_lenders
from scripts import seed_lenders
from scripts.seed_lenders import SeedOutcome, describe, main, run_seed
from tests.helpers import SEED_LENDERS, criteria_json, seed_criteria, seed_default_lenders

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SPELLINGS = {"min_score_floor", "min_age", "max_age_at_maturity"}
FALLBACK_SPELLINGS = {"score_floor", "min_age_years", "max_age_at_maturity_years"}


def _published(repo: MemoryRepo) -> list[LenderRow]:
    return repo.published_lenders()


def _writer(repo: MemoryRepo) -> Callable[[], LenderWriter | None]:
    return lambda: repo


# --------------------------------------------------------------------------- the single source


def test_the_seeded_database_equals_the_one_the_scenario_suite_builds() -> None:
    seeded, reference = MemoryRepo(), MemoryRepo()
    assert main([], backend=_writer(seeded)) == 0
    seed_default_lenders(reference)
    assert _published(seeded) == _published(reference)
    pincodes = sorted({p for lender in SEED_LENDERS for p in lender.pincodes})
    assert seeded.lenders_for_pincodes(pincodes) == reference.lenders_for_pincodes(pincodes)


def test_each_lender_is_one_published_version_one_row_with_its_pincodes() -> None:
    repo = MemoryRepo()
    run_seed(repo)
    rows = {row.id: row for row in _published(repo)}
    assert sorted(rows) == ["L1", "L2", "L3"]
    for lender in SEED_LENDERS:
        row = rows[lender.id]
        assert (row.version, row.published, row.name, row.policy_notes) == (1, True, lender.name, lender.policy_notes)
        assert repo.lender_versions(lender.id) == [(1, True)]
        assert [r.id for r in repo.lenders_for_pincodes(lender.pincodes) if r.id == lender.id] == [lender.id]
        (matched,) = (r for r in repo.lenders_for_pincodes(lender.pincodes) if r.id == lender.id)
        assert set(matched.matched_pincodes) == set(lender.pincodes)


def test_criteria_use_the_schema_spellings_only_and_every_field_is_known() -> None:
    repo = MemoryRepo()
    run_seed(repo)
    for row in _published(repo):
        assert SCHEMA_SPELLINGS <= set(row.criteria)
        assert not FALLBACK_SPELLINGS & set(row.criteria)  # a fallback key could shadow an admin edit
        for field, entry in row.criteria.items():
            assert set(entry) == {"value", "status"}, field
            assert entry["status"] == "known", field


def test_the_seeded_rows_pass_the_pincode_gate_with_no_warnings() -> None:
    repo = MemoryRepo()
    run_seed(repo)

    def gated_ids(pincode: str) -> list[str]:
        rows = [
            PublishedLenderRow(
                id=row.id,
                name=row.name,
                criteria=row.criteria,
                matched_pincodes=list(row.matched_pincodes),
                policy_notes=row.policy_notes,
            )
            for row in repo.lenders_for_pincodes([pincode])
        ]
        gated, warnings = gate_lenders(rows, [pincode])
        assert warnings == []
        return [g.id for g in gated]

    assert gated_ids("560001") == ["L1", "L2"]
    assert gated_ids("400001") == ["L3"]
    assert gated_ids("110001") == ["L1", "L2"]


def test_the_database_copy_is_plain_data_and_the_single_source_stays_read_only() -> None:
    before = [criteria_json(lender.criteria) for lender in SEED_LENDERS]
    repo = MemoryRepo()
    run_seed(repo)
    for row in _published(repo):
        json.dumps(row.criteria)  # the jsonb payload: plain dicts, no MappingProxyType
        row.criteria["foir_limit"]["value"] = 0.99  # a returned copy; mutating it must not reach the source or the repo
    assert [criteria_json(lender.criteria) for lender in SEED_LENDERS] == before
    assert all(row.criteria["foir_limit"]["value"] != 0.99 for row in _published(repo))
    with pytest.raises(TypeError):
        SEED_LENDERS[0].criteria["foir_limit"]["value"] = 0.99  # pyright: ignore[reportIndexIssue]


# --------------------------------------------------------------------------- idempotence and never-overwrite


def test_a_second_run_changes_nothing_and_reports_every_lender_skipped() -> None:
    repo = MemoryRepo()
    first = run_seed(repo)
    snapshot = (_published(repo), repo.lenders_for_pincodes(["560001", "400001", "110001"]))
    second = run_seed(repo)
    assert [o.action for o in first] == ["created"] * 3
    assert [o.action for o in second] == ["skipped"] * 3
    assert all("already present (v1 published)" in o.detail for o in second)
    assert (_published(repo), repo.lenders_for_pincodes(["560001", "400001", "110001"])) == snapshot


def test_an_admin_edited_lender_is_never_touched_and_the_others_are_still_seeded() -> None:
    repo = MemoryRepo()
    repo.seed_lender("L1", 2, "Alpha (admin edit)", True, criteria_json(seed_criteria(foir_limit=0.3)), "kept", ["560001"])
    outcomes = run_seed(repo)
    assert {o.lender_id: o.action for o in outcomes} == {"L1": "skipped", "L2": "created", "L3": "created"}
    l1 = next(row for row in _published(repo) if row.id == "L1")
    assert (l1.version, l1.name, l1.policy_notes) == (2, "Alpha (admin edit)", "kept")
    assert repo.lender_versions("L1") == [(2, True)]  # no v1 was added next to the admin's v2


def test_an_unpublished_draft_blocks_the_seed_instead_of_being_published_over() -> None:
    repo = MemoryRepo()
    repo.seed_lender("L2", 1, "Bharat draft", False, criteria_json(seed_criteria(foir_limit=0.3)), "", ["560001"])
    outcomes = {o.lender_id: o for o in run_seed(repo)}
    assert outcomes["L2"].action == "skipped" and "v1 unpublished" in outcomes["L2"].detail
    assert repo.lender_versions("L2") == [(1, False)]
    assert [row.id for row in _published(repo)] == ["L1", "L3"]


def test_a_writer_failure_is_not_swallowed() -> None:
    class Boom(MemoryRepo):
        def create_published_lender(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        run_seed(Boom())


# --------------------------------------------------------------------------- the CLI


def test_dry_run_prints_every_row_and_never_builds_a_backend(capsys: pytest.CaptureFixture[str]) -> None:
    def forbidden() -> LenderWriter | None:
        raise AssertionError("--dry-run must not connect to anything")

    assert main(["--dry-run"], backend=forbidden) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out and "nothing is written" in out
    for lender in SEED_LENDERS:
        assert f"{lender.id} v{lender.version} {lender.name!r}" in out
        assert json.dumps(criteria_json(lender.criteria), sort_keys=True) in out
        assert ", ".join(sorted(set(lender.pincodes))) in out


def test_dry_run_needs_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert main(["--dry-run"]) == 0


def test_the_real_cli_entry_point_dry_runs_from_the_repo_root() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "scripts.seed_lenders", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "L1 v1 'Alpha Education Finance'" in done.stdout
    assert "L3 v1 'Crest Capital NBFC'" in done.stdout


def test_describe_lists_what_a_real_run_writes() -> None:
    lines = describe(SEED_LENDERS[0])
    assert lines[0] == "L1 v1 'Alpha Education Finance': published, 3 pincodes"
    assert lines[3] == "  pincodes: 110001, 560001, 560002"


def test_a_real_run_prints_one_line_per_lender_and_the_synthetic_warning(capsys: pytest.CaptureFixture[str]) -> None:
    repo = MemoryRepo()
    assert main([], backend=_writer(repo)) == 0
    out = capsys.readouterr().out
    assert "L1 v1: created (published, 3 pincodes)" in out and "L3 v1: created (published, 1 pincode)" in out
    assert "synthetic example lenders" in out
    capsys.readouterr()
    assert main([], backend=_writer(repo)) == 0
    assert "skipped (already present (v1 published); left untouched)" in capsys.readouterr().out


def test_missing_credentials_exit_2_with_a_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert main([]) == 2
    assert "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set" in capsys.readouterr().err
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    assert main([]) == 2  # a URL without the key is still a configuration error


def test_the_env_backend_is_the_supabase_repo_on_the_configured_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSupabase()
    client: Any = fake
    seen: dict[str, Any] = {}

    def fake_backend(url: str, key: str, *, bucket: str) -> tuple[SupabaseRepo, SupabaseRawStore]:
        seen.update(url=url, key=key, bucket=bucket)
        raw = SupabaseRawStore(client, bucket)
        return SupabaseRepo(client, raw), raw

    monkeypatch.setattr(seed_lenders, "create_supabase_backend", fake_backend)
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RAW_BUCKET", "custom-bucket")
    assert main([]) == 0
    assert seen == {"url": "https://example.invalid", "key": "service-key", "bucket": "custom-bucket"}
    assert {row["id"] for row in fake.tables["lenders"]} == {"L1", "L2", "L3"}


# --------------------------------------------------------------------------- the Supabase writer (fake client)


def _supabase() -> tuple[SupabaseRepo, FakeSupabase]:
    fake = FakeSupabase()
    client: Any = fake
    return SupabaseRepo(client, SupabaseRawStore(client)), fake


def test_supabase_rows_are_published_seed_stamped_and_carry_every_pincode() -> None:
    repo, fake = _supabase()
    outcomes = run_seed(repo)
    assert [o.action for o in outcomes] == ["created"] * 3
    rows = {row["id"]: row for row in fake.tables["lenders"]}
    assert sorted(rows) == ["L1", "L2", "L3"]
    for lender in SEED_LENDERS:
        row = rows[lender.id]
        assert row["version"] == lender.version and row["published"] is True
        assert row["created_by"] == "seed_lenders"
        assert row["criteria"] == criteria_json(lender.criteria)
        json.dumps(row["criteria"])
        stored = {p["pincode"] for p in fake.tables["lender_pincodes"] if p["lender_id"] == lender.id}
        assert stored == set(lender.pincodes)
        assert all(p["version"] == lender.version for p in fake.tables["lender_pincodes"] if p["lender_id"] == lender.id)
    assert repo.lender_versions("L1") == [(1, True)]
    assert [r.id for r in repo.published_lenders()] == ["L1", "L2", "L3"]
    assert {r.id for r in repo.lenders_for_pincodes(["400001"])} == {"L3"}


def test_supabase_publishes_last_so_a_failure_never_leaves_a_pincodeless_published_lender() -> None:
    repo, fake = _supabase()
    run_seed(repo, SEED_LENDERS[:1])
    inserts = [e for e in fake.events if e.endswith(":lenders") or e.endswith(":lender_pincodes")]
    assert inserts == ["select:lenders", "insert:lenders", "insert:lender_pincodes", "update:lenders"]
    assert [q.payload for q in fake.queries if q.op == "update"] == [{"published": True}]
    first_insert = next(q for q in fake.queries if q.op == "insert" and q.table == "lenders")
    assert cast("dict[str, Any]", first_insert.payload)["published"] is False


def test_supabase_second_run_skips_everything() -> None:
    repo, fake = _supabase()
    run_seed(repo)
    before = (list(fake.tables["lenders"]), list(fake.tables["lender_pincodes"]))
    again = run_seed(repo)
    assert [o.action for o in again] == ["skipped"] * 3
    assert (list(fake.tables["lenders"]), list(fake.tables["lender_pincodes"])) == before


def _fail_on(fake: FakeSupabase, monkeypatch: pytest.MonkeyPatch, op: str, table: str, code: str | None = None) -> None:
    real_run = fake.run

    def run(query: Any) -> FakeResponse:
        if query.op == op and query.table == table:
            raise APIError({"message": "injected failure", "code": code or "XX000", "hint": None, "details": None})
        return real_run(query)

    monkeypatch.setattr(fake, "run", run)


def test_a_pincode_insert_failure_deletes_the_half_written_lender_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake = _supabase()
    _fail_on(fake, monkeypatch, "insert", "lender_pincodes")
    with pytest.raises(APIError, match="injected failure"):
        run_seed(repo)
    assert fake.tables["lenders"] == []  # no draft left behind: a rerun starts clean
    assert not any(q.op == "update" for q in fake.queries)  # never published


def test_a_publish_failure_deletes_the_draft_too_and_a_unique_violation_is_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake = _supabase()
    _fail_on(fake, monkeypatch, "update", "lenders", code="23505")
    with pytest.raises(ValueError, match="already has a published version"):
        run_seed(repo, SEED_LENDERS[:1])
    assert fake.tables["lenders"] == []


def test_a_duplicate_version_insert_is_a_value_error() -> None:
    repo, fake = _supabase()
    run_seed(repo, SEED_LENDERS[:1])
    with pytest.raises(ValueError, match=r"duplicate lender version \('L1', 1\)"):
        repo.create_published_lender("L1", 1, "again", {}, "", ["560001"])
    assert len(fake.tables["lenders"]) == 1  # the existing row was neither replaced nor deleted by the failed attempt


def test_a_malformed_pincode_is_refused_before_any_write_on_both_backends() -> None:
    sup, fake = _supabase()
    mem = MemoryRepo()
    for writer in (sup, mem):
        with pytest.raises(ValueError, match="invalid pincode"):
            writer.create_published_lender("LX", 1, "X", {}, "", ["056001"])
    assert fake.tables["lenders"] == [] and mem.lender_versions("LX") == []


def test_a_leftover_draft_from_a_failed_cleanup_is_skipped_not_published(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake = _supabase()
    healthy = fake.run
    _fail_on(fake, monkeypatch, "insert", "lender_pincodes")
    failing_pincodes = fake.run

    def run(query: Any) -> FakeResponse:
        if query.op == "delete":
            raise APIError({"message": "cleanup failed", "code": "XX000", "hint": None, "details": None})
        return failing_pincodes(query)

    monkeypatch.setattr(fake, "run", run)
    with pytest.raises(APIError, match="injected failure"):  # the ORIGINAL failure surfaces, not the cleanup's
        run_seed(repo, SEED_LENDERS[:1])
    assert [(r["id"], r["published"]) for r in fake.tables["lenders"]] == [("L1", False)]
    monkeypatch.setattr(fake, "run", healthy)
    (outcome,) = run_seed(repo, SEED_LENDERS[:1])
    assert isinstance(outcome, SeedOutcome) and outcome.action == "skipped" and "v1 unpublished" in outcome.detail
