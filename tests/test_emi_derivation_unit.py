"""Task 6: derive/emi.py, the PRD 8.3 existing-EMI algorithm (pure, deterministic).

Load-bearing pins:

* Four branches in strict order: (1) a POSITIVE reported EMI; (2) tenure + balance -> balance / remaining months
  (``tenure_remaining``) or, once past tenure, balance / 12 with a flag + confirm item (``past_tenure``);
  (3) revolving ``R`` -> 5% of balance; (4) otherwise balance / 36. ``rate_of_interest`` is never used.
* ``months_since`` is day-agnostic and clamped at zero.
* Stale accounts (> 90 days) are excluded from every sum and surfaced as a confirm item; the estimator never sees them.
* Full: ``used = max(declared, derived)`` (``'declared'`` only when strictly higher); quick: declared or 0.0 with
  ``'declared_zero_assumed'`` and a floor flag when nothing was declared. An account-less full check (NTC) is a floor.
* Overflow guard (Task-5 deferment): a tenure outside 1..1200 months is "unknown", and nothing raises.

The six-account oracle below is written out by hand from the fixture's documented numbers (delta-3), NOT by calling
the module under test.
"""

from __future__ import annotations

import ast
import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import eligibility_agent
from eligibility_agent.derive.emi import (
    FALLBACK_TENURE_MONTHS,
    MAX_TENURE_MONTHS,
    PAST_TENURE_CONFIRM_ITEM,
    PAST_TENURE_DIVISOR,
    REVOLVING_EMI_PCT,
    derive_existing_emi,
    estimate_account_emi,
    months_since,
)
from eligibility_agent.graph.state import EmiDerivation, EmiItem, OpenAccount
from eligibility_agent.surepass.adapters import report_to_report_facts
from eligibility_agent.testing import scenarios as scn

ANCHOR = date(2026, 10, 1)  # ReportDate of every fixture world


def acct(**overrides: Any) -> OpenAccount:
    """A fresh, open, unremarkable account: no EMI, no tenure, zero balance. Override what the case needs."""
    fields: dict[str, Any] = {
        "account_key": "acct_0",
        "portfolio_type": "I",
        "account_type": "05",
        "scheduled_emi": None,
        "current_balance": 0.0,
        "tenure_months": None,
        "rate_of_interest": None,
        "open_date": None,
        "date_reported": date(2026, 9, 15),
        "amount_past_due": 0.0,
        "pincodes": [],
        "is_stale": False,
    }
    fields.update(overrides)
    return OpenAccount(**fields)


def keyed(i: int, **overrides: Any) -> OpenAccount:
    return acct(account_key=f"acct_{i}", **overrides)


# --------------------------------------------------------------------------- constants and module surface


def test_constants_are_the_prd_numbers() -> None:
    assert REVOLVING_EMI_PCT == 0.05
    assert FALLBACK_TENURE_MONTHS == 36
    assert PAST_TENURE_DIVISOR == 12
    assert MAX_TENURE_MONTHS == 1200
    assert PAST_TENURE_CONFIRM_ITEM == "past tenure, confirm with learner"


def test_emi_module_import_surface_is_pure() -> None:
    tree = ast.parse((Path(eligibility_agent.__file__).parent / "derive" / "emi.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    allowed = {"__future__", "math", "datetime", "collections.abc", "typing", "eligibility_agent.graph.state"}
    assert found <= allowed, found - allowed


# --------------------------------------------------------------------------- months_since


@pytest.mark.parametrize(
    ("opened", "anchor", "expected"),
    [
        (date(2024, 1, 15), date(2026, 10, 1), 33),  # (2026-2024)*12 + (10-1): the day of month is ignored
        (date(2025, 6, 1), date(2026, 10, 1), 16),
        (date(2024, 9, 1), date(2026, 10, 1), 25),
        (date(2026, 10, 31), date(2026, 10, 1), 0),  # same month, later day: still 0, not -1
        (date(2026, 10, 1), date(2026, 10, 31), 0),
        (date(2026, 9, 30), date(2026, 10, 1), 1),  # one calendar month apart although only 1 day elapsed
        (date(2025, 12, 31), date(2026, 1, 1), 1),
        (date(2027, 3, 1), date(2026, 10, 1), 0),  # opened in the future (junk date): clamped to 0
        (date(2026, 10, 1), date(2026, 10, 1), 0),
    ],
)
def test_months_since_is_day_agnostic_and_clamped(opened: date, anchor: date, expected: int) -> None:
    assert months_since(opened, anchor) == expected


# --------------------------------------------------------------------------- estimate_account_emi: the four branches


def test_branch_1_reported_emi_wins() -> None:
    item = estimate_account_emi(acct(scheduled_emi=12500.0, current_balance=150000.0, tenure_months=24), anchor=ANCHOR)
    assert item == EmiItem(
        account_key="acct_0", kind="reported", monthly_amount=12500.0, method="reported", past_tenure=False,
        confirm_item=None,
    )


@pytest.mark.parametrize("not_positive", [0.0, -1.0, None, math.nan, math.inf, -math.inf])
def test_branch_1_needs_a_positive_finite_reported_emi(not_positive: float | None) -> None:
    # '0' means "not reported" on Experian: the account falls through to estimation instead of contributing 0
    item = estimate_account_emi(acct(scheduled_emi=not_positive, current_balance=90000.0), anchor=ANCHOR)
    assert item.kind == "estimated" and item.method == "balance_36" and item.monthly_amount == 2500.0


def test_branch_2_tenure_remaining_design_example() -> None:
    # delta-2 acct B: balance 160000, Terms 24, opened 2025-06-01, anchor 2026-10-01 -> months_since 16, remaining 8
    item = estimate_account_emi(
        acct(current_balance=160000.0, tenure_months=24, open_date=date(2025, 6, 1)), anchor=ANCHOR
    )
    assert item == EmiItem(
        account_key="acct_0", kind="estimated", monthly_amount=20000.0, method="tenure_remaining", past_tenure=False,
        confirm_item=None,
    )


def test_branch_2_six_account_fixture_a2() -> None:
    # delta-3 a2: balance 120000, tenure 48, opened 2024-01-15 -> months_since 33, remaining 15 -> 8000
    item = estimate_account_emi(
        acct(current_balance=120000.0, tenure_months=48, open_date=date(2024, 1, 15)), anchor=ANCHOR
    )
    assert item.monthly_amount == 8000.0 and item.method == "tenure_remaining" and item.past_tenure is False


def test_branch_2_past_tenure_flags_and_confirm_item() -> None:
    # delta-2 acct C: balance 60000, tenure 12, opened 2024-09-01 -> months_since 25, remaining -13 -> 60000 / 12
    item = estimate_account_emi(
        acct(current_balance=60000.0, tenure_months=12, open_date=date(2024, 9, 1)), anchor=ANCHOR
    )
    assert item == EmiItem(
        account_key="acct_0", kind="estimated", monthly_amount=5000.0, method="past_tenure", past_tenure=True,
        confirm_item="past tenure, confirm with learner",
    )


def test_branch_2_boundary_remaining_one_is_still_tenure_remaining_remaining_zero_is_past() -> None:
    # tenure 12 opened 2025-11 -> months_since 11 -> remaining 1 (>= 1): balance / 1
    last = estimate_account_emi(
        acct(current_balance=7000.0, tenure_months=12, open_date=date(2025, 11, 20)), anchor=ANCHOR
    )
    assert last.method == "tenure_remaining" and last.monthly_amount == 7000.0 and last.past_tenure is False
    # opened 2025-10 -> months_since 12 -> remaining 0 (< 1): balance / 12 + flag
    over = estimate_account_emi(
        acct(current_balance=7200.0, tenure_months=12, open_date=date(2025, 10, 5)), anchor=ANCHOR
    )
    assert over.method == "past_tenure" and over.monthly_amount == 600.0 and over.past_tenure is True


def test_branch_2_missing_open_date_falls_back_to_the_anchor() -> None:
    # open_date None -> months_since(anchor, anchor) == 0 -> remaining == tenure
    item = estimate_account_emi(acct(current_balance=48000.0, tenure_months=24, open_date=None), anchor=ANCHOR)
    assert item.method == "tenure_remaining" and item.monthly_amount == 2000.0


def test_branch_2_needs_a_positive_balance_else_falls_through() -> None:
    item = estimate_account_emi(acct(current_balance=0.0, tenure_months=24, open_date=date(2025, 6, 1)), anchor=ANCHOR)
    assert item.method == "balance_36" and item.monthly_amount == 0.0 and item.kind == "estimated"
    revolving = estimate_account_emi(
        acct(portfolio_type="R", current_balance=0.0, tenure_months=24), anchor=ANCHOR
    )
    assert revolving.method == "revolving_5pct" and revolving.monthly_amount == 0.0


def test_branch_3_revolving_is_five_percent_of_balance() -> None:
    item = estimate_account_emi(acct(portfolio_type="R", current_balance=80000.0), anchor=ANCHOR)
    assert item == EmiItem(
        account_key="acct_0", kind="estimated", monthly_amount=4000.0, method="revolving_5pct", past_tenure=False,
        confirm_item=None,
    )


def test_branch_3_an_r_account_with_tenure_still_takes_the_tenure_branch_first() -> None:
    # branch order: tenure + balance beats the revolving rule
    item = estimate_account_emi(
        acct(portfolio_type="R", current_balance=24000.0, tenure_months=24, open_date=date(2025, 10, 1)), anchor=ANCHOR
    )
    assert item.method == "tenure_remaining" and item.monthly_amount == 24000.0 / 12


@pytest.mark.parametrize("portfolio", ["I", None])
def test_branch_4_fallback_is_balance_over_36(portfolio: str | None) -> None:
    item = estimate_account_emi(acct(portfolio_type=portfolio, current_balance=90000.0), anchor=ANCHOR)
    assert item == EmiItem(
        account_key="acct_0", kind="estimated", monthly_amount=2500.0, method="balance_36", past_tenure=False,
        confirm_item=None,
    )


def test_rate_of_interest_is_never_used() -> None:
    base = dict(current_balance=120000.0, tenure_months=48, open_date=date(2024, 1, 15))
    plain = estimate_account_emi(acct(**base), anchor=ANCHOR)
    priced = estimate_account_emi(acct(rate_of_interest=13.5, **base), anchor=ANCHOR)
    assert priced == plain
    no_tenure = dict(current_balance=90000.0)
    assert estimate_account_emi(acct(rate_of_interest=29.9, **no_tenure), anchor=ANCHOR) == estimate_account_emi(
        acct(**no_tenure), anchor=ANCHOR
    )


def test_account_key_is_carried_onto_the_item() -> None:
    assert estimate_account_emi(keyed(7, scheduled_emi=100.0), anchor=ANCHOR).account_key == "acct_7"


# --------------------------------------------------------------------------- overflow guard (Task-5 deferment)


@pytest.mark.parametrize("tenure", [0, -1, -240, 1201, 5000, 10**9, 10**400])
def test_tenure_outside_1_to_1200_is_unknown_and_never_raises(tenure: int) -> None:
    # 10**400 cannot even be converted to a float: unguarded, balance / remaining raises OverflowError
    item = estimate_account_emi(
        acct(current_balance=90000.0, tenure_months=tenure, open_date=date(2024, 1, 15)), anchor=ANCHOR
    )
    assert item.method == "balance_36" and item.monthly_amount == 2500.0 and item.past_tenure is False
    revolving = estimate_account_emi(
        acct(portfolio_type="R", current_balance=80000.0, tenure_months=tenure), anchor=ANCHOR
    )
    assert revolving.method == "revolving_5pct" and revolving.monthly_amount == 4000.0


@pytest.mark.parametrize("tenure", [1, 1200])
def test_tenure_bounds_1_and_1200_are_still_valid(tenure: int) -> None:
    item = estimate_account_emi(
        acct(current_balance=120000.0, tenure_months=tenure, open_date=ANCHOR), anchor=ANCHOR
    )
    assert item.method == "tenure_remaining" and item.monthly_amount == 120000.0 / tenure


@pytest.mark.parametrize("balance", [math.nan, math.inf, -math.inf, -5.0])
def test_pathological_balances_cannot_raise_and_contribute_zero(balance: float) -> None:
    for kw in ({}, {"tenure_months": 24}, {"portfolio_type": "R"}):
        item = estimate_account_emi(acct(current_balance=balance, **kw), anchor=ANCHOR)
        assert item.kind == "estimated" and item.monthly_amount == 0.0 and math.isfinite(item.monthly_amount)


def test_extreme_dates_do_not_raise() -> None:
    item = estimate_account_emi(
        acct(current_balance=1000.0, tenure_months=1200, open_date=date(1, 1, 1)), anchor=date(9999, 12, 31)
    )
    assert item.method == "past_tenure" and item.monthly_amount == 1000.0 / 12  # ~120k months since open
    future = estimate_account_emi(
        acct(current_balance=1000.0, tenure_months=1200, open_date=date(9999, 12, 31)), anchor=date(1, 1, 1)
    )
    assert future.method == "tenure_remaining" and future.monthly_amount == 1000.0 / 1200  # clamped to 0 months


# --------------------------------------------------------------------------- derive_existing_emi: the six-account table


def six_accounts() -> list[OpenAccount]:
    """The fixture world through the REAL adapters: a1..a6 as acct_0..acct_5 (a6 is stale, 122 days)."""
    client = scn.full_emi_four_branch_scenario()
    result = client.experian_report("chk", "h" * 64, name="Asha Verma", mobile="9876543210", pan=scn.PAN_OK)
    facts = report_to_report_facts(result)
    assert facts.report_date == ANCHOR
    return facts.open_accounts


def test_six_account_table_totals_and_items() -> None:
    d = derive_existing_emi(six_accounts(), declared=None, kind="full", anchor=ANCHOR)
    # hand oracle (delta-3): a1 reported 3500; a2 120000/(48-33)=8000; a3 past tenure 60000/12=5000;
    # a4 revolving 0.05*40000=2000; a5 90000/36=2500; a6 (EMI 9999, 122 days) stale: excluded.
    assert d.reported_total == 3500.0
    assert d.estimated_total == 8000.0 + 5000.0 + 2000.0 + 2500.0 == 17500.0
    assert d.derived_total == 21000.0
    assert d.declared is None and d.used == 21000.0 and d.used_source == "derived"
    assert d.emi_floor_flag is True
    assert [i.account_key for i in d.items] == [f"acct_{n}" for n in range(6)]
    assert [(i.kind, i.method, i.monthly_amount, i.past_tenure) for i in d.items] == [
        ("reported", "reported", 3500.0, False),
        ("estimated", "tenure_remaining", 8000.0, False),
        ("estimated", "past_tenure", 5000.0, True),
        ("estimated", "revolving_5pct", 2000.0, False),
        ("estimated", "balance_36", 2500.0, False),
        ("stale_excluded", None, 0.0, False),
    ]
    assert [i.confirm_item for i in d.items] == [
        None,
        None,
        "past tenure, confirm with learner",
        None,
        None,
        "confirm with learner: an account last updated 2026-06-01",
    ]


def test_stale_account_is_excluded_from_every_sum() -> None:
    # (2026-10-01 - 2026-06-01) is 122 days > 90: the 9999 must appear nowhere
    accounts = six_accounts()
    assert accounts[5].is_stale is True and accounts[5].scheduled_emi == 9999.0
    d = derive_existing_emi(accounts, declared=None, kind="full", anchor=ANCHOR)
    assert d.reported_total == 3500.0 and d.derived_total == 21000.0
    assert not any(i.monthly_amount == 9999.0 for i in d.items)
    assert sum(i.monthly_amount for i in d.items) == 21000.0


def test_stale_flag_is_the_adapters_not_recomputed_here() -> None:
    # estimate_account_emi never sees stale rows; derive_existing_emi trusts is_stale from the adapter
    stale_but_recent_date = keyed(0, scheduled_emi=4000.0, is_stale=True, date_reported=date(2026, 9, 30))
    d = derive_existing_emi([stale_but_recent_date], declared=None, kind="full", anchor=ANCHOR)
    assert d.derived_total == 0.0 and d.items[0].kind == "stale_excluded"
    assert d.items[0].confirm_item == "confirm with learner: an account last updated 2026-09-30"


def test_stale_account_without_a_reported_date_does_not_raise() -> None:
    # adapters mark an unparseable Date_Reported as stale, so date_reported can be None
    stale = keyed(0, scheduled_emi=4000.0, is_stale=True, date_reported=None)
    d = derive_existing_emi([stale], declared=None, kind="full", anchor=ANCHOR)
    (item,) = d.items
    assert item.kind == "stale_excluded" and item.monthly_amount == 0.0 and item.method is None
    assert item.confirm_item is not None and item.confirm_item.startswith("confirm with learner:")
    assert "None" not in item.confirm_item


def test_design_two_worked_example_44000() -> None:
    # delta-2's independent worked example (all numbers by hand): 12500 + (20000 + 5000 + 4000 + 2500) = 44000
    accounts = [
        keyed(0, scheduled_emi=12500.0, current_balance=300000.0, date_reported=date(2026, 9, 15)),
        keyed(1, current_balance=160000.0, tenure_months=24, open_date=date(2025, 6, 1)),
        keyed(2, current_balance=60000.0, tenure_months=12, open_date=date(2024, 9, 1)),
        keyed(3, portfolio_type="R", current_balance=80000.0),
        keyed(4, portfolio_type="I", current_balance=90000.0),
        keyed(5, scheduled_emi=9000.0, date_reported=date(2026, 5, 20), is_stale=True),
    ]
    d = derive_existing_emi(accounts, declared=None, kind="full", anchor=ANCHOR)
    assert (d.reported_total, d.estimated_total, d.derived_total) == (12500.0, 31500.0, 44000.0)
    assert d.emi_floor_flag is True
    assert d.items[5].confirm_item == "confirm with learner: an account last updated 2026-05-20"
    higher = derive_existing_emi(accounts, declared=50000.0, kind="full", anchor=ANCHOR)
    assert higher.used == 50000.0 and higher.used_source == "declared" and higher.derived_total == 44000.0
    lower = derive_existing_emi(accounts, declared=30000.0, kind="full", anchor=ANCHOR)
    assert lower.used == 44000.0 and lower.used_source == "derived"


# --------------------------------------------------------------------------- declared-vs-derived resolution (full)


def test_declared_higher_wins_and_lower_loses() -> None:
    accounts = six_accounts()
    high = derive_existing_emi(accounts, declared=25000.0, kind="full", anchor=ANCHOR)
    assert high.used == 25000.0 and high.used_source == "declared"
    assert high.declared == 25000.0 and high.derived_total == 21000.0  # both figures survive for the model + safety
    low = derive_existing_emi(accounts, declared=2000.0, kind="full", anchor=ANCHOR)
    assert low.used == 21000.0 and low.used_source == "derived" and low.declared == 2000.0


def test_declared_equal_to_derived_is_attributed_to_derived() -> None:
    d = derive_existing_emi(six_accounts(), declared=21000.0, kind="full", anchor=ANCHOR)
    assert d.used == 21000.0 and d.used_source == "derived"  # 'declared' only when STRICTLY higher


def test_declared_zero_in_full_check_uses_derived() -> None:
    d = derive_existing_emi(six_accounts(), declared=0.0, kind="full", anchor=ANCHOR)
    assert d.used == 21000.0 and d.used_source == "derived" and d.declared == 0.0


def test_declared_never_changes_the_derived_split() -> None:
    a = derive_existing_emi(six_accounts(), declared=None, kind="full", anchor=ANCHOR)
    b = derive_existing_emi(six_accounts(), declared=99999.0, kind="full", anchor=ANCHOR)
    assert (a.reported_total, a.estimated_total, a.derived_total, a.items) == (
        b.reported_total, b.estimated_total, b.derived_total, b.items,
    )
    assert a.emi_floor_flag is b.emi_floor_flag is True


# --------------------------------------------------------------------------- emi_floor_flag


def test_floor_flag_is_false_only_when_every_account_is_fresh_and_reported() -> None:
    reported_only = derive_existing_emi(
        [keyed(0, scheduled_emi=3500.0), keyed(1, scheduled_emi=1500.0)], declared=None, kind="full", anchor=ANCHOR
    )
    assert reported_only.emi_floor_flag is False
    assert (reported_only.reported_total, reported_only.estimated_total, reported_only.used) == (5000.0, 0.0, 5000.0)


def test_one_estimated_account_raises_the_floor_flag() -> None:
    d = derive_existing_emi(
        [keyed(0, scheduled_emi=3500.0), keyed(1, current_balance=90000.0)], declared=None, kind="full", anchor=ANCHOR
    )
    assert d.emi_floor_flag is True


def test_one_stale_account_raises_the_floor_flag_even_beside_reported_ones() -> None:
    d = derive_existing_emi(
        [keyed(0, scheduled_emi=3500.0), keyed(1, scheduled_emi=800.0, is_stale=True)],
        declared=None, kind="full", anchor=ANCHOR,
    )
    assert d.emi_floor_flag is True and d.reported_total == 3500.0


def test_a_declared_figure_does_not_lower_the_floor_flag() -> None:
    d = derive_existing_emi([keyed(0, current_balance=90000.0)], declared=50000.0, kind="full", anchor=ANCHOR)
    assert d.emi_floor_flag is True and d.used == 50000.0


def test_ntc_full_check_has_no_accounts_derived_zero_and_a_floor() -> None:
    d = derive_existing_emi([], declared=None, kind="full", anchor=ANCHOR)
    assert d == EmiDerivation(
        reported_total=0.0, estimated_total=0.0, derived_total=0.0, declared=None, used=0.0,
        used_source="derived", emi_floor_flag=True, items=[],
    )


def test_ntc_full_check_with_a_declared_emi() -> None:
    d = derive_existing_emi([], declared=5000.0, kind="full", anchor=ANCHOR)
    assert d.used == 5000.0 and d.used_source == "declared" and d.derived_total == 0.0 and d.emi_floor_flag is True
    zero = derive_existing_emi([], declared=0.0, kind="full", anchor=ANCHOR)
    assert zero.used == 0.0 and zero.used_source == "derived" and zero.emi_floor_flag is True


# --------------------------------------------------------------------------- quick


def test_quick_without_a_declared_emi_assumes_zero_and_flags_the_floor() -> None:
    d = derive_existing_emi([], declared=None, kind="quick")
    assert d == EmiDerivation(
        reported_total=0.0, estimated_total=0.0, derived_total=0.0, declared=None, used=0.0,
        used_source="declared_zero_assumed", emi_floor_flag=True, items=[],
    )


def test_quick_with_a_declared_emi_uses_it() -> None:
    d = derive_existing_emi([], declared=12000.0, kind="quick")
    assert d == EmiDerivation(
        reported_total=0.0, estimated_total=0.0, derived_total=0.0, declared=12000.0, used=12000.0,
        used_source="declared", emi_floor_flag=False, items=[],
    )


def test_quick_with_an_explicit_zero_declared_is_declared_not_assumed() -> None:
    d = derive_existing_emi([], declared=0.0, kind="quick")
    assert d.used == 0.0 and d.used_source == "declared" and d.emi_floor_flag is False and d.declared == 0.0


def test_quick_never_uses_bureau_accounts_even_if_handed_some() -> None:
    d = derive_existing_emi(six_accounts(), declared=None, kind="quick")
    assert d.items == [] and d.derived_total == 0.0 and d.used == 0.0 and d.used_source == "declared_zero_assumed"


def test_quick_ignores_the_anchor() -> None:
    assert derive_existing_emi([], declared=None, kind="quick") == derive_existing_emi(
        [], declared=None, kind="quick", anchor=ANCHOR
    )


# --------------------------------------------------------------------------- purity / determinism / totality


def test_derivation_is_deterministic_and_does_not_mutate_inputs() -> None:
    accounts = six_accounts()
    before = [a.model_copy() for a in accounts]
    one = derive_existing_emi(accounts, declared=25000.0, kind="full", anchor=ANCHOR)
    two = derive_existing_emi(accounts, declared=25000.0, kind="full", anchor=ANCHOR)
    assert one == two and accounts == before


def test_derive_is_total_over_pathological_accounts() -> None:
    nasty = [
        keyed(0, scheduled_emi=math.inf, current_balance=math.nan, tenure_months=10**400, open_date=date(1, 1, 1)),
        keyed(1, scheduled_emi=-5.0, current_balance=math.inf, tenure_months=-3),
        keyed(2, portfolio_type=None, current_balance=-1.0, tenure_months=0),
        keyed(3, scheduled_emi=None, current_balance=1e308, tenure_months=1200, open_date=ANCHOR),
    ]
    d = derive_existing_emi(nasty, declared=None, kind="full", anchor=ANCHOR)
    assert math.isfinite(d.used) and math.isfinite(d.derived_total)
    assert len(d.items) == 4


def test_account_order_is_preserved_in_items() -> None:
    accounts = [keyed(2, scheduled_emi=1.0), keyed(0, scheduled_emi=2.0), keyed(1, scheduled_emi=3.0)]
    d = derive_existing_emi(accounts, declared=None, kind="full", anchor=ANCHOR)
    assert [i.account_key for i in d.items] == ["acct_2", "acct_0", "acct_1"]
