"""Seed the three SYNTHETIC example lenders (L1 / L2 / L3) into a fresh database.

usage: python -m scripts.seed_lenders [--dry-run]     (run from the repository root)

The lenders, their criteria and their pincodes come from ``tests.helpers.SEED_LENDERS``, the single source the scenario
suite and the eval goldens share, so a seeded database, the tests and the evals never drift. ``tests/`` is not part of
the installed wheel: like ``scripts.purge_raw`` this runs from a repository checkout.

For each lender the script writes one ``lenders`` row (version 1, ``published = true``, ``created_by = 'seed_lenders'``)
plus its ``lender_pincodes`` rows. The criteria jsonb uses the ``db/schema.sql`` key spellings (``min_score_floor``,
``min_age``, ``max_age_at_maturity``) and every field is ``{"value": ..., "status": "known"}``.

* Never overwrites: a lender id that already has ANY version (published or draft) is skipped and reported, so reruns are
  safe and an admin's edits are never touched. To re-seed one lender, delete its ``lenders`` row first (its
  ``lender_pincodes`` rows cascade).
* ``--dry-run`` prints the rows it would write and exits 0 WITHOUT connecting to anything (no credentials needed), so it
  cannot tell which lenders already exist; a real run skips those.
* Credentials come from SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. Exit 0 on success, 2 on a usage / configuration error.

These are placeholders, not real lenders (the real floors, FOIR limits and pincode lists are loaded through the criteria
admin page before UAT). Do NOT leave them published in production: their pincodes are real postal codes, so a real learner
there would be matched to a fictional lender. Take them down with
``update lenders set published = false where created_by = 'seed_lenders';``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from eligibility_agent.db.repo import DEFAULT_RAW_BUCKET, LenderWriter, create_supabase_backend
from tests.helpers import SEED_LENDERS, SeedLender, criteria_json

Action = Literal["created", "skipped"]


@dataclass(frozen=True, slots=True)
class SeedOutcome:
    lender_id: str
    version: int
    action: Action
    detail: str


def _pincodes_text(lender: SeedLender) -> str:
    count = len(set(lender.pincodes))
    return f"{count} pincode{'' if count == 1 else 's'}"


def _versions_text(versions: Sequence[tuple[int, bool]]) -> str:
    return ", ".join(f"v{version} {'published' if published else 'unpublished'}" for version, published in versions)


def run_seed(writer: LenderWriter, lenders: Sequence[SeedLender] = SEED_LENDERS) -> list[SeedOutcome]:
    """Write each lender that does not exist yet; report the rest as skipped. A failure on one lender propagates (the
    writer has already cleaned up that lender where the backend allows), so a half-seeded run is visible, never silent."""
    outcomes: list[SeedOutcome] = []
    for lender in lenders:
        existing = writer.lender_versions(lender.id)
        if existing:
            outcomes.append(
                SeedOutcome(
                    lender.id,
                    lender.version,
                    "skipped",
                    f"already present ({_versions_text(existing)}); left untouched; "
                    "delete the lender row to re-seed (its pincodes cascade)",
                )
            )
            continue
        writer.create_published_lender(
            lender.id,
            lender.version,
            lender.name,
            criteria_json(lender.criteria),
            lender.policy_notes,
            lender.pincodes,
        )
        outcomes.append(
            SeedOutcome(lender.id, lender.version, "created", f"published, {_pincodes_text(lender)}")
        )
    return outcomes


def describe(lender: SeedLender) -> list[str]:
    """The dry-run rendering of one lender: exactly what a real run would write."""
    return [
        f"{lender.id} v{lender.version} {lender.name!r}: published, {_pincodes_text(lender)}",
        f"  criteria: {json.dumps(criteria_json(lender.criteria), sort_keys=True)}",
        f"  policy_notes: {lender.policy_notes!r}",
        f"  pincodes: {', '.join(sorted(set(lender.pincodes)))}",
    ]


def _backend_from_env() -> LenderWriter | None:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("seed_lenders: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        return None
    repo, _ = create_supabase_backend(url, key, bucket=os.environ.get("RAW_BUCKET") or DEFAULT_RAW_BUCKET)
    return repo


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: Callable[[], LenderWriter | None] | None = None,
    lenders: Sequence[SeedLender] = SEED_LENDERS,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.seed_lenders", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the rows that would be written; connect to nothing")
    args = parser.parse_args(argv)
    dry_run: bool = args.dry_run

    if dry_run:
        print("seed_lenders (dry-run: nothing is written, no connection is made)")
        for lender in lenders:
            print("\n".join(describe(lender)))
        return 0

    writer = backend() if backend is not None else _backend_from_env()
    if writer is None:
        return 2
    print("seed_lenders")
    for outcome in run_seed(writer, lenders):
        print(f"{outcome.lender_id} v{outcome.version}: {outcome.action} ({outcome.detail})")
    print("NOTE: these are synthetic example lenders; do not leave them published in production (see the module docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
