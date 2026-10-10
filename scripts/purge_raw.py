"""Day-one TTL purge job (PRD 11: the schema carries TTL columns and a purge job from day one).

usage: python -m scripts.purge_raw [--dry-run]

Sequence (matches the schema.sql comment):
  1. surepass_calls rows with purge_after < now and purged_at is null, in batches of 100:
     delete their storage objects, then set purged_at = now. The call-log METADATA row is kept forever;
     only the raw object dies.
  2. delete pan_records / score_cache / report_cache rows with purge_after < now (their storage keys were
     owned by step-1 rows; a later stale read raises RawUnavailable, which admit_recompute turns into
     'raw_unavailable').
  3. delete consents with purge_after < now (5-year retention); rows that reference them are unlinked.

--dry-run prints the counts and the first 10 keys/ids per step without deleting anything.
Idempotent: a second run finds nothing due. Exit 0 on success, 2 on a usage / configuration error.
Credentials come from SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and (optional) RAW_BUCKET.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from eligibility_agent.db.repo import (
    DEFAULT_RAW_BUCKET,
    Clock,
    PurgeRepo,
    RawStore,
    create_supabase_backend,
    utc_now,
)

BATCH_SIZE = 100
SAMPLE_SIZE = 10


@dataclass(frozen=True, slots=True)
class PurgeReport:
    dry_run: bool
    calls: int  # step 1: surepass_calls rows purged (or due, in a dry run)
    objects: int  # step 1: of those, rows that owned a storage object
    call_key_sample: tuple[str, ...]  # first SAMPLE_SIZE storage keys
    cache_rows: int  # step 2
    cache_row_sample: tuple[str, ...]  # 'table:pan_hash'
    consents: int  # step 3
    consent_sample: tuple[str, ...]


def run_purge(
    repo: PurgeRepo, raw_store: RawStore, *, now: datetime, dry_run: bool = False, batch_size: int = BATCH_SIZE
) -> PurgeReport:
    # ---- step 1: raw objects owned by surepass_calls
    calls = 0
    keys_seen: list[str] = []
    if dry_run:
        due = repo.expired_surepass_calls(now, limit=None)
        calls = len(due)
        keys_seen = [row.storage_key for row in due if row.storage_key is not None]
    else:
        while True:
            batch = repo.expired_surepass_calls(now, limit=batch_size)
            if not batch:
                break
            keys = [row.storage_key for row in batch if row.storage_key is not None]
            if keys:
                raw_store.delete_raw(keys)  # object first: a failure here leaves rows unmarked, so a rerun retries
            repo.mark_calls_purged([row.id for row in batch], now)
            calls += len(batch)
            keys_seen.extend(keys)

    # ---- step 2: expired cache rows
    cache_rows = repo.expired_cache_rows(now)
    if not dry_run:
        repo.delete_expired_cache_rows(now)

    # ---- step 3: expired consents
    consent_ids = repo.expired_consent_ids(now)
    if not dry_run:
        repo.delete_expired_consents(now)

    return PurgeReport(
        dry_run=dry_run,
        calls=calls,
        objects=len(keys_seen),
        call_key_sample=tuple(keys_seen[:SAMPLE_SIZE]),
        cache_rows=len(cache_rows),
        cache_row_sample=tuple(cache_rows[:SAMPLE_SIZE]),
        consents=len(consent_ids),
        consent_sample=tuple(consent_ids[:SAMPLE_SIZE]),
    )


def _print_report(report: PurgeReport) -> None:
    mode = "dry-run: nothing is deleted" if report.dry_run else "done"
    print(f"purge_raw ({mode})")
    print(f"step 1 raw objects: {report.calls} call rows, {report.objects} objects")
    for key in report.call_key_sample:
        print(f"  {key}")
    print(f"step 2 cache rows: {report.cache_rows} cache rows")
    for ident in report.cache_row_sample:
        print(f"  {ident}")
    print(f"step 3 consents: {report.consents} consents")
    for consent_id in report.consent_sample:
        print(f"  {consent_id}")


def _backend_from_env(clock: Clock) -> tuple[PurgeRepo, RawStore] | None:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("purge_raw: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        return None
    bucket = os.environ.get("RAW_BUCKET") or DEFAULT_RAW_BUCKET
    return create_supabase_backend(url, key, bucket=bucket, clock=clock)


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: Callable[[], tuple[PurgeRepo, RawStore]] | None = None,
    clock: Clock = utc_now,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.purge_raw", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print counts and the first 10 keys per step; delete nothing")
    args = parser.parse_args(argv)
    dry_run: bool = args.dry_run

    pair = backend() if backend is not None else _backend_from_env(clock)
    if pair is None:
        return 2
    repo, raw_store = pair
    _print_report(run_purge(repo, raw_store, now=clock(), dry_run=dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
