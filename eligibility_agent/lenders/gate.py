"""Pincode gate over PUBLISHED lender criteria (PRD v1.1 9.1).

The gate decides which lenders the model may rank. It is pure: published rows (already joined to the candidate
pincodes by the repo) go in, ``(GatedLenders sorted by id, warning strings)`` come out. No I/O, no clock, no repo.

* Criteria arrive as the ``lenders.criteria`` jsonb contract (Task 2's ``LenderRow.criteria``): every field is
  ``{"value": <typed>, "status": "known" | "missing" | "uncertain"}``. The admin-UI key names ``min_score_floor``,
  ``min_age`` and ``max_age_at_maturity`` are mapped onto ``LenderCriteria.score_floor`` / ``min_age_years`` /
  ``max_age_at_maturity_years`` (the canonical names are accepted too); the per-field status becomes
  ``LenderCriteria.field_status`` for all twelve criteria fields (an absent key is ``'missing'``).
* ``REQUIRED_FOR_MATCHING`` plus the floor rule (``score_floor`` OR ``ntc_accepted`` known) decide matchability. A lender
  that fails is EXCLUDED with a warning string (``lender <id> excluded: missing <fields>``) and never aborts the check.
  ``'uncertain'`` required fields stay IN with their usable value; they ride on ``field_status`` and cap the lender's
  contribution to code confidence at medium (``uncertain_required_fields``).
* ``min_income`` and the age limits are optional: the checks that use them are skipped when absent.
* The pincode filter is applied defensively even though the repo join already guarantees it: candidates and each row's
  matched pincodes are re-checked against ``PIN_RE``, and a lender stays in only while at least one valid candidate
  pincode is in its approved list. The gate is never silently skipped (FIXED 9.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, NotRequired, TypedDict, cast

from pydantic import ValidationError

from eligibility_agent.derive.range import PIN_RE
from eligibility_agent.graph.state import FieldStatus, GatedLender, LenderCriteria

__all__ = [
    "FLOOR_FIELDS",
    "REQUIRED_FOR_MATCHING",
    "PublishedLenderRow",
    "gate_lenders",
    "is_matchable",
    "uncertain_required_fields",
]


class PublishedLenderRow(TypedDict):
    """One published lender as the repo hands it over (``LenderRow`` minus the version / published flags)."""

    id: str
    name: str
    criteria: dict[str, Any]  # jsonb contract: {field: {"value": ..., "status": known|missing|uncertain}}
    matched_pincodes: list[str]
    policy_notes: NotRequired[str | None]  # the table column; '' / absent -> LenderCriteria.policy_notes None


# Required for matching: both the range and the safety formulas need these. min_income and the age limits are OPTIONAL
# (their checks are skipped when absent). The floor is satisfied by score_floor known OR ntc_accepted known.
REQUIRED_FOR_MATCHING: Final[tuple[str, ...]] = (
    "foir_limit",
    "rate_min_pct",
    "rate_max_pct",
    "tenure_min_months",
    "tenure_max_months",
    "loan_min",
    "loan_max",
)

# The two fields that make up the floor rule; an 'uncertain' answer on either undermines the band-clears-floor test.
FLOOR_FIELDS: Final[tuple[str, ...]] = ("score_floor", "ntc_accepted")

# Every criteria field the jsonb contract can carry, in LenderCriteria order (policy_notes is a table column, not jsonb).
_CRITERIA_FIELDS: Final[tuple[str, ...]] = (
    "score_floor",
    "ntc_accepted",
    "foir_limit",
    "min_income",
    "min_age_years",
    "max_age_at_maturity_years",
    "rate_min_pct",
    "rate_max_pct",
    "tenure_min_months",
    "tenure_max_months",
    "loan_min",
    "loan_max",
)

# schema.sql / admin-UI jsonb key -> LenderCriteria field (the canonical names are accepted too, and win when both exist).
_JSONB_KEY_ALIASES: Final[dict[str, str]] = {
    "score_floor": "min_score_floor",
    "min_age_years": "min_age",
    "max_age_at_maturity_years": "max_age_at_maturity",
}

_VALID_STATUSES: Final[frozenset[str]] = frozenset({"known", "missing", "uncertain"})
_FLOAT_FIELDS: Final[tuple[str, ...]] = ("foir_limit", "min_income", "rate_min_pct", "rate_max_pct", "loan_min", "loan_max")


def _status_of(c: LenderCriteria, name: str) -> FieldStatus | None:
    return c.field_status.get(name)


def _is_known(c: LenderCriteria, name: str) -> bool:
    """A usable answer: a non-None value whose status is not 'missing' ('uncertain' values are usable)."""
    return getattr(c, name) is not None and _status_of(c, name) != "missing"


def is_matchable(c: LenderCriteria) -> tuple[bool, list[str]]:
    """``(len(missing) == 0, missing)``.

    ``missing`` is every ``REQUIRED_FOR_MATCHING`` field whose value is None or whose status is ``'missing'`` (registry
    order), plus ``'score_floor'`` when neither ``score_floor`` nor ``ntc_accepted`` is known (``ntc_accepted=False`` IS a
    known answer). ``'uncertain'`` required fields stay in: usable value, flagged through ``field_status``.
    """
    missing = [f for f in REQUIRED_FOR_MATCHING if not _is_known(c, f)]
    if not any(_is_known(c, f) for f in FLOOR_FIELDS):
        missing.append("score_floor")
    return (len(missing) == 0, missing)


def uncertain_required_fields(c: LenderCriteria) -> list[str]:
    """The required criteria fields (``REQUIRED_FOR_MATCHING`` then the floor pair) the admin marked ``'uncertain'``.

    These ride to the model as the lender's ``uncertain_fields`` and block the ``high`` code-confidence row.
    """
    return [f for f in (*REQUIRED_FOR_MATCHING, *FLOOR_FIELDS) if _status_of(c, f) == "uncertain"]


def _parse_criteria(raw: object, policy_notes: str | None) -> tuple[LenderCriteria | None, list[str]]:
    """Normalise the jsonb contract into a ``LenderCriteria``; ``(None, invalid_field_names)`` when it cannot be.

    A missing key is ``'missing'`` (value None); status ``'missing'`` forces the value to None so nothing downstream can
    use it. A bare (non ``{value, status}``) entry, an unknown status, a value of the wrong type, or a non-finite
    number makes the field invalid.
    """
    source: dict[str, Any] = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
    data: dict[str, Any] = {"policy_notes": policy_notes or None}
    statuses: dict[str, FieldStatus] = {}
    invalid: list[str] = []
    for name in _CRITERIA_FIELDS:
        key = name if name in source else _JSONB_KEY_ALIASES.get(name, name)
        if key not in source:
            data[name] = None
            statuses[name] = "missing"
            continue
        entry: Any = source[key]
        entry_dict = cast("dict[str, Any]", entry) if isinstance(entry, dict) else {}
        status: Any = entry_dict.get("status")
        if not isinstance(status, str) or status not in _VALID_STATUSES:
            invalid.append(name)
            data[name] = None
            statuses[name] = "missing"
            continue
        statuses[name] = cast(FieldStatus, status)
        data[name] = None if status == "missing" else entry_dict.get("value")
    data["field_status"] = statuses
    if invalid:
        return None, invalid
    try:
        criteria = LenderCriteria.model_validate(data)
    except ValidationError as exc:
        bad = {str(err["loc"][0]) for err in exc.errors() if err["loc"]}
        return None, [f for f in _CRITERIA_FIELDS if f in bad] or ["criteria"]
    not_finite = [f for f in _FLOAT_FIELDS if (v := getattr(criteria, f)) is not None and not math.isfinite(v)]
    if not_finite:
        return None, not_finite
    return criteria, []


def gate_lenders(
    rows: Sequence[PublishedLenderRow], pincodes: Sequence[str]
) -> tuple[list[GatedLender], list[str]]:
    """Gate the repo's pincode-joined published rows for the applicant's candidate pincodes.

    Returns the lenders the model may rank, sorted by id (deterministic), and warning strings for the rows excluded
    on data quality: ``'lender <id> excluded: missing <fields>'`` (``is_matchable``), ``'lender <id> excluded: invalid
    criteria <fields>'`` (a field that is not a usable ``{value, status}``), ``'lender <id> excluded: duplicate row'``.
    A lender that simply serves none of the candidate pincodes is out without a warning. Never raises on a bad row.

    Pure over its inputs (nothing is mutated): rows are processed in id order, so the same rows in any order, and the
    candidates in any order or repeated, give an identical result (the defensive duplicate-id path keeps the first row
    in input order; the one-published-version index makes it unreachable).
    """
    candidates = {p for p in pincodes if PIN_RE.fullmatch(p) is not None}
    gated: list[GatedLender] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda r: r["id"]):
        lender_id = row["id"]
        matched = sorted({p for p in row["matched_pincodes"] if PIN_RE.fullmatch(p) is not None} & candidates)
        if not matched:
            continue  # serves none of this applicant's pincodes: the gate doing its job, not a data defect
        if lender_id in seen:
            warnings.append(f"lender {lender_id} excluded: duplicate row")
            continue
        seen.add(lender_id)
        criteria, invalid = _parse_criteria(row["criteria"], row.get("policy_notes"))
        if criteria is None:
            warnings.append(f"lender {lender_id} excluded: invalid criteria {', '.join(invalid)}")
            continue
        ok, missing = is_matchable(criteria)
        if not ok:
            warnings.append(f"lender {lender_id} excluded: missing {', '.join(missing)}")
            continue
        gated.append(
            GatedLender(
                id=lender_id,
                name=row["name"],
                criteria=criteria,
                matched_pincodes=matched,
                rate_band_label=f"{criteria.rate_min_pct:g} to {criteria.rate_max_pct:g}%",
                tenure_label=f"{criteria.tenure_min_months} to {criteria.tenure_max_months} months",
            )
        )
    return gated, warnings
