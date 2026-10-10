"""Task 2: db/schema.sql is idempotent DDL covering 12 tables, 3 functions, RLS, the checkpointer block."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from eligibility_agent.db import repo as repo_module
from eligibility_agent.graph.state import Code, ErrorKind, ReviewKind

SCHEMA = (Path(repo_module.__file__).parent / "schema.sql").read_text()
SQL = re.sub(r"--[^\n]*", "", SCHEMA)  # comments stripped: assertions are about executable DDL only

TABLES = (
    "lenders",
    "lender_pincodes",
    "consents",
    "pan_records",
    "score_cache",
    "report_cache",
    "checks",
    "surepass_calls",
    "pan_locks",
    "review_queue",
    "lc_quota_overrides",
    "opportunity_pan_events",
)
FUNCTIONS = ("quick_checks_used_today", "bureau_calls_today", "acquire_pan_lock")
REFUSAL_CODES = (
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
)


def _quoted(fragment: str) -> list[str]:
    return re.findall(r"'([^']*)'", fragment)


def _table_body(name: str) -> str:
    match = re.search(rf"create table if not exists {name}\s*\((.*?)\n\);", SQL, flags=re.S)
    assert match is not None, f"table {name} not found"
    return match.group(1)


def test_all_twelve_tables_are_created_idempotently() -> None:
    created = re.findall(r"create table (if not exists )?(\w+)", SQL)
    assert [name for _, name in created] == list(TABLES)
    assert all(flag for flag, _ in created)  # every create table is `if not exists`


def test_every_index_is_idempotent() -> None:
    for statement in re.findall(r"create (?:unique )?index [^;]*;", SQL):
        assert "if not exists" in statement, statement


def test_the_three_functions_are_create_or_replace() -> None:
    created = re.findall(r"create or replace function (\w+)\(", SQL)
    assert created == list(FUNCTIONS)
    assert "create function" not in SQL


def test_rls_is_enabled_on_all_twelve_tables_with_no_policies() -> None:
    enabled = re.findall(r"alter table (\w+) enable row level security", SQL)
    assert sorted(enabled) == sorted(TABLES)
    assert "create policy" not in SQL.lower()


def test_checks_status_and_code_lists_match_the_registry() -> None:
    body = _table_body("checks")
    status_check = re.search(r"status\s+text not null check \(status in \(([^)]*)\)\)", body)
    assert status_check is not None
    assert _quoted(status_check.group(1)) == [
        "running",
        "ok",
        "pan_not_verified",
        "refused",
        "pending_verification",
        "service_failure",
        "error",
    ]
    code_check = re.search(r"code\s+text check \(code in \(([^)]*)\)\)", body)
    assert code_check is not None
    assert _quoted(code_check.group(1)) == list(get_args(Code))
    assert len(get_args(Code)) == 13


def test_code_legality_constraint_matches_code_legal() -> None:
    body = _table_body("checks")
    match = re.search(r"constraint code_legality check \((.*)\)\s*$", body.strip(), flags=re.S)
    assert match is not None
    clauses = [c.strip() for c in re.split(r"\n\s*or ", match.group(1).strip())]
    assert len(clauses) == 4
    refused, failed, ok, rest = clauses
    assert "status = 'refused'" in refused and _quoted(refused.split("code in", 1)[1]) == list(REFUSAL_CODES)
    assert "status = 'service_failure'" in failed and _quoted(failed.split("code in", 1)[1]) == [
        "SUREPASS_FAILURE",
        "OPENAI_FAILURE",
    ]
    assert "status = 'ok'" in ok and "code is null or code = 'NO_PINCODE_ON_PAN'" in ok
    # K1: pending_verification is code-less
    assert _quoted(rest.split("code is null")[0]) == ["running", "pan_not_verified", "pending_verification", "error"]
    assert rest.rstrip(") ").endswith("code is null")


def test_review_queue_kind_check_lists_all_eight_kinds() -> None:
    body = _table_body("review_queue")
    match = re.search(r"kind\s+text not null check \(kind in \(([^)]*)\)\)", body)
    assert match is not None
    assert _quoted(match.group(1)) == list(get_args(ReviewKind))
    assert len(get_args(ReviewKind)) == 8 and "surepass_auth" in _quoted(match.group(1))


def test_surepass_calls_error_kind_check_matches_the_registry() -> None:
    body = _table_body("surepass_calls")
    match = re.search(r"error_kind\s+text check \(error_kind in \(([^)]*)\)\)", body)
    assert match is not None
    assert _quoted(match.group(1)) == list(get_args(ErrorKind))
    for column in ("pan_hash", "billable", "latency_ms", "storage_key", "purge_after", "purged_at"):
        assert re.search(rf"\b{column}\b", body), column


def test_checks_columns_are_the_enumerated_row_fields() -> None:
    from dataclasses import fields

    from eligibility_agent.db.repo import CheckRow

    body = _table_body("checks")
    columns = re.findall(r"^\s{2}(\w+)\s+(?:uuid|text|boolean|jsonb|timestamptz)\b", body, flags=re.M)
    assert columns == [f.name for f in fields(CheckRow)]


def test_lenders_are_versioned_with_one_published_version_per_lender() -> None:
    assert "primary key (id, version)" in _table_body("lenders")
    assert re.search(r"create unique index if not exists idx_lenders_one_published on lenders \(id\) where published", SQL)
    pincodes = _table_body("lender_pincodes")
    assert "foreign key (lender_id, version) references lenders(id, version) on delete cascade" in pincodes
    assert r"'^[1-9][0-9]{5}$'" in pincodes


def test_consent_references_clear_on_consent_purge() -> None:
    """The 5-year consent purge must not be blocked by rows that point at the consent."""
    for table in ("checks", "score_cache", "report_cache"):
        assert re.search(r"consent_id\s+uuid references consents\(id\) on delete set null", _table_body(table)), table


def test_pan_locks_primary_key_and_kinds() -> None:
    body = _table_body("pan_locks")
    assert "primary key (pan_hash, kind)" in body
    assert "'in_flight','timeout','daily_cap'" in body


def test_cache_tables_hold_pointers_not_derived_facts() -> None:
    for table in ("pan_records", "score_cache", "report_cache"):
        body = _table_body(table)
        assert "storage_key" in body and "purge_after" in body
        assert "derived" not in body  # K5: the v1.0 derived jsonb column is void


def test_ist_functions_use_the_ist_day_and_only_bureau_endpoints() -> None:
    quick = re.search(r"function quick_checks_used_today.*?\$\$;", SQL, flags=re.S)
    bureau = re.search(r"function bureau_calls_today.*?\$\$;", SQL, flags=re.S)
    assert quick is not None and bureau is not None
    for fn in (quick.group(0), bureau.group(0)):
        assert "'Asia/Kolkata'" in fn and "interval '1 day'" in fn
    assert "c.counted_against_quota" in quick.group(0) and "c.kind = 'quick'" in quick.group(0)
    assert "sc.billable" in bureau.group(0)
    assert "'/api/v1/credit-report-experian/score'" in bureau.group(0)
    assert "'/api/v1/credit-report-experian/fetch-report'" in bureau.group(0)
    assert "pan-comprehensive" not in bureau.group(0)


def test_acquire_pan_lock_is_an_atomic_steal_only_when_expired_upsert() -> None:
    match = re.search(r"function acquire_pan_lock.*?end \$\$;", SQL, flags=re.S)
    assert match is not None
    fn = match.group(0)
    assert "returns boolean" in fn
    assert "on conflict (pan_hash, kind) do update" in fn
    assert "pan_locks.released_at is not null or pan_locks.lock_until <= now()" in fn
    assert "get diagnostics n = row_count" in fn and "return n = 1" in fn


def test_checkpointer_isolation_block_and_ops_comments_survive() -> None:
    assert "create schema if not exists checkpointer;" in SQL
    assert "checkpointer_svc" in SCHEMA and "search_path = checkpointer" in SCHEMA
    assert "to_regclass('public.checkpoints')" in SCHEMA
    assert "raw-responses" in SCHEMA and "create_bucket" in SCHEMA  # bucket ops note
    assert "scripts/purge_raw.py" in SCHEMA  # purge ops note
    assert re.search(r"pooler|:5432", SCHEMA)
