"""Existing-EMI derivation (PRD v1.1 section 8.3): pure, deterministic, total.

Per OPEN bureau account the estimator runs four branches in strict order (``rate_of_interest`` is NEVER used):

1. a POSITIVE reported ``scheduled_emi``                       -> that figure                (``reported``)
2. else a valid tenure AND a positive balance                  -> balance / remaining months (``tenure_remaining``),
   where ``remaining = tenure - months_since(open_date or anchor, anchor)``; once ``remaining < 1`` the account is past
   its tenure: balance / 12, ``past_tenure=True`` and the confirm item ``past tenure, confirm with learner``
3. else portfolio type ``R`` (revolving)                       -> 5% of balance             (``revolving_5pct``)
4. else                                                        -> balance / 36              (``balance_36``)

Stale accounts (``is_stale``: last updated more than 90 days before the report date, or an unparseable date) are
decided by the ADAPTER; here they are excluded from every sum and surfaced as a confirm item, and the per-account
estimator never sees them. ``derive_existing_emi`` then resolves declared vs derived (full) or takes the declared
figure (quick).

Overflow guard (Task-5 deferment): a tenure outside ``1..MAX_TENURE_MONTHS`` (1200 months = 100 years) is treated as
UNKNOWN and falls through the branches, so an absurd ``Terms_Duration`` such as ``10**400`` can never reach
``balance / remaining`` (an ``int`` that large cannot be converted to a float: ``OverflowError``). A balance that is
not a finite non-negative number contributes ``0.0``. Nothing in this module raises on any ``OpenAccount``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Final, Literal, TypeAlias

from eligibility_agent.graph.state import EmiDerivation, EmiItem, Kind, OpenAccount

REVOLVING_EMI_PCT: Final[float] = 0.05
FALLBACK_TENURE_MONTHS: Final[int] = 36
PAST_TENURE_DIVISOR: Final[int] = 12
MAX_TENURE_MONTHS: Final[int] = 1200  # bureau tenures above this are junk: unknown, not a divisor
PAST_TENURE_CONFIRM_ITEM: Final[str] = "past tenure, confirm with learner"
_STALE_CONFIRM_PREFIX: Final[str] = "confirm with learner: an account last updated"
_STALE_UNKNOWN_DATE_ITEM: Final[str] = "confirm with learner: an account with an unknown last update date"

_EstimateMethod: TypeAlias = Literal["tenure_remaining", "past_tenure", "revolving_5pct", "balance_36"]


def months_since(open_date: date, anchor: date) -> int:
    """Calendar-month difference, day-agnostic (v1.1 gives no day rule), clamped to >= 0."""
    return max(0, (anchor.year - open_date.year) * 12 + (anchor.month - open_date.month))


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _clean_balance(balance: float) -> float:
    """Non-finite or negative balances (junk) count as 0.0 so every division stays finite."""
    return balance if math.isfinite(balance) and balance > 0.0 else 0.0


def _estimated(account_key: str, amount: float, method: _EstimateMethod, *, past_tenure: bool = False) -> EmiItem:
    return EmiItem(
        account_key=account_key,
        kind="estimated",
        monthly_amount=amount,
        method=method,
        past_tenure=past_tenure,
        confirm_item=PAST_TENURE_CONFIRM_ITEM if past_tenure else None,
    )


def estimate_account_emi(a: OpenAccount, *, anchor: date) -> EmiItem:
    """The four 8.3 branches for ONE non-stale open account (see the module docstring). Total: never raises."""
    if a.scheduled_emi is not None and _is_positive_finite(a.scheduled_emi):
        return EmiItem(
            account_key=a.account_key,
            kind="reported",
            monthly_amount=a.scheduled_emi,
            method="reported",
            past_tenure=False,
            confirm_item=None,
        )
    balance = _clean_balance(a.current_balance)
    tenure = a.tenure_months
    if tenure is not None and 1 <= tenure <= MAX_TENURE_MONTHS and balance > 0.0:
        remaining = tenure - months_since(a.open_date or anchor, anchor)
        if remaining >= 1:
            return _estimated(a.account_key, balance / remaining, "tenure_remaining")
        return _estimated(a.account_key, balance / PAST_TENURE_DIVISOR, "past_tenure", past_tenure=True)
    if a.portfolio_type == "R":
        return _estimated(a.account_key, REVOLVING_EMI_PCT * balance, "revolving_5pct")
    return _estimated(a.account_key, balance / FALLBACK_TENURE_MONTHS, "balance_36")


def _stale_item(a: OpenAccount) -> EmiItem:
    return EmiItem(
        account_key=a.account_key,
        kind="stale_excluded",
        monthly_amount=0.0,
        method=None,
        past_tenure=False,
        confirm_item=_STALE_UNKNOWN_DATE_ITEM
        if a.date_reported is None
        else f"{_STALE_CONFIRM_PREFIX} {a.date_reported.isoformat()}",
    )


def derive_existing_emi(
    accounts: Sequence[OpenAccount], *, declared: float | None, kind: Kind, anchor: date | None = None
) -> EmiDerivation:
    """Resolve the existing monthly EMI burden.

    QUICK: bureau accounts are never used (any passed in are ignored). ``used`` is the declared figure, or 0.0 when
    none was declared (``used_source='declared_zero_assumed'``, and the LcView label gains 'assuming no other EMIs').
    ``emi_floor_flag`` is True exactly when nothing was declared.

    FULL: stale accounts become ``stale_excluded`` items (amount 0.0, confirm item naming the last-updated date);
    every other account is estimated. ``derived_total = reported_total + estimated_total``. ``used`` is
    ``max(declared, derived_total)`` when a declared figure exists (``used_source='declared'`` only when it is
    STRICTLY higher), else ``derived_total``; both figures stay on the derivation for the model and safety.
    ``emi_floor_flag`` is True when any item is estimated / stale / past-tenure, and for an account-less report (NTC:
    nothing is known, so derived 0.0 is a floor).

    ``anchor`` is the date the tenure maths runs from: the derive node passes the report date. When omitted on a
    full check it falls back to today's date (the one non-pure input here; pass it explicitly to stay deterministic).
    """
    if kind == "quick":
        used = declared if declared is not None else 0.0
        return EmiDerivation(
            reported_total=0.0,
            estimated_total=0.0,
            derived_total=0.0,
            declared=declared,
            used=used,
            used_source="declared" if declared is not None else "declared_zero_assumed",
            emi_floor_flag=declared is None,
            items=[],
        )

    effective_anchor = anchor if anchor is not None else date.today()
    items: list[EmiItem] = []
    reported_total = 0.0
    estimated_total = 0.0
    for account in accounts:
        if account.is_stale:
            items.append(_stale_item(account))
            continue
        item = estimate_account_emi(account, anchor=effective_anchor)
        items.append(item)
        if item.kind == "reported":
            reported_total += item.monthly_amount
        else:
            estimated_total += item.monthly_amount
    derived_total = reported_total + estimated_total
    declared_wins = declared is not None and declared > derived_total
    used = declared if declared is not None and declared_wins else derived_total
    floor = not items or any(i.kind in ("estimated", "stale_excluded") or i.past_tenure for i in items)
    return EmiDerivation(
        reported_total=reported_total,
        estimated_total=estimated_total,
        derived_total=derived_total,
        declared=declared,
        used=used,
        used_source="declared" if declared_wins else "derived",
        emi_floor_flag=floor,
        items=items,
    )
