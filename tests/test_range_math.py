"""Task 6: derive/range.py: half-up display rounding (K6), annuity range / ceiling / displayed FOIR (K22), age, pincodes.

Every pinned number here was recomputed independently with 60-digit ``Decimal`` arithmetic (NOT the module under
test, and NOT copied from the design dossier: delta-2's floor examples are void under K6). Raw floats are compared
with ``pytest.approx(rel=1e-9)`` (the float ``(1+r)**n`` form drifts ~1e-14 from the exact value); display values are
compared EXACTLY, and every raw -> display pair is listed in ``DISPLAY_TABLE`` so a rounding regression names itself.

Seed lenders (delta-3 SEED LENDERS v2):
  L1: FOIR 0.55, 11-14%, 12-48m, loans 50,000-1,500,000      L2: FOIR 0.45, 14-18%, 12-36m, loans 25,000-500,000
"""

from __future__ import annotations

import ast
import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import eligibility_agent
from eligibility_agent.derive import range as range_mod
from eligibility_agent.derive.range import (
    FOIR_EPS,
    MAX_CRITERIA_LOAN_AMOUNT,
    MAX_CRITERIA_TENURE_MONTHS,
    age_at_maturity_years,
    age_years_and_months,
    annuity_emi,
    annuity_loan_max,
    available_emi,
    candidate_pincodes,
    lender_numbers,
    round_to_nearest_1000,
)
from eligibility_agent.graph.state import Identity, LenderCriteria, LenderNumbers, ReportFacts
from eligibility_agent.surepass.adapters import pan_to_identity, report_to_report_facts
from eligibility_agent.testing import scenarios as scn


def crit(**overrides: Any) -> LenderCriteria:
    """Seed lender L1 unless overridden."""
    fields: dict[str, Any] = {
        "score_floor": 650,
        "ntc_accepted": False,
        "foir_limit": 0.55,
        "min_income": 25000.0,
        "min_age_years": 21,
        "max_age_at_maturity_years": 58,
        "rate_min_pct": 11.0,
        "rate_max_pct": 14.0,
        "tenure_min_months": 12,
        "tenure_max_months": 48,
        "loan_min": 50000.0,
        "loan_max": 1500000.0,
        "policy_notes": None,
        "field_status": {},
    }
    fields.update(overrides)
    return LenderCriteria(**fields)


L1 = crit()
L2 = crit(
    score_floor=650, ntc_accepted=True, foir_limit=0.45, min_income=20000.0, max_age_at_maturity_years=60,
    rate_min_pct=14.0, rate_max_pct=18.0, tenure_min_months=12, tenure_max_months=36, loan_min=25000.0,
    loan_max=500000.0,
)
# delta-2's worked lender (its 'L1'): FOIR 0.50, 11-14%, 12-60m, loans 50,000-500,000
L1_D2 = crit(foir_limit=0.50, tenure_max_months=60, loan_max=500000.0)


def nums(c: LenderCriteria, income: float | None, existing: float, age_months: int | None = 341) -> LenderNumbers:
    return lender_numbers(c, lender_id="L", income=income, existing_emi=existing, age_months=age_months)


# --------------------------------------------------------------------------- K6: the ONE half-up rounding home


@pytest.mark.parametrize(
    ("raw", "display"),
    [
        (731500.0, 732000),  # exactly half -> up
        (367536.02, 368000),
        (222749.104, 223000),  # delta-2 floored this to 222000: VOID
        (469741.3983, 470000),  # 741.4 >= 500 rounds UP (delta-2's 469000 is void)
        (499.9, 0),
        (500.0, 1000),  # half-up at the smallest scale
        (499.999, 0),
        (999.5, 1000),
        (0.0, 0),
        (1000.0, 1000),
        (1500.0, 2000),
        (2500.0, 3000),  # half-up, NOT banker's rounding (which would give 2000)
        (3500.0, 4000),
        (1141396.9236949403, 1141000),
        (328554.92842160386, 329000),
        (865312.3009392032, 865000),
        (1149825.8458347924, 1150000),
    ],
)
def test_round_to_nearest_1000_is_half_up(raw: float, display: int) -> None:
    assert round_to_nearest_1000(raw) == display


def test_round_to_nearest_1000_returns_a_plain_int() -> None:
    out = round_to_nearest_1000(367536.02)
    assert type(out) is int


def test_round_to_nearest_1000_properties_over_a_grid() -> None:
    previous = -1
    for step in range(0, 3_000_000, 977):  # coprime-ish stride walks every residue mod 1000
        v = step + (step % 7) / 7.0
        out = round_to_nearest_1000(v)
        assert out % 1000 == 0
        assert abs(out - v) <= 500.0 + 1e-6
        assert out >= previous  # monotone non-decreasing
        assert round_to_nearest_1000(float(out)) == out  # idempotent
        previous = out


def test_floor_to_1000_is_gone_and_rounding_has_one_home() -> None:
    assert not hasattr(range_mod, "floor_to_1000")
    src = Path(eligibility_agent.__file__).parent
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if "floor_to_1000" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --------------------------------------------------------------------------- annuity identities


RATES = [0.0, 0.5, 8.5, 11.0, 14.0, 18.0, 24.0, 36.0]
TENURES = [1, 6, 12, 36, 48, 60, 120, 360]
AMOUNTS = [1000.0, 27000.0, 123456.78]


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("n", TENURES)
def test_annuity_round_trip_identities(rate: float, n: int) -> None:
    for a in AMOUNTS:
        principal = annuity_loan_max(a, rate, n)
        assert annuity_emi(principal, rate, n) == pytest.approx(a, rel=1e-9)
        assert annuity_loan_max(annuity_emi(a, rate, n), rate, n) == pytest.approx(a, rel=1e-9)


def test_annuity_known_values_against_the_decimal_oracle() -> None:
    assert annuity_loan_max(20000.0, 11.0, 60) == pytest.approx(919860.6766678339, rel=1e-9)
    assert annuity_loan_max(33000.0, 11.0, 48) == pytest.approx(1276816.897692634, rel=1e-9)
    assert annuity_loan_max(27000.0, 18.0, 36) == pytest.approx(746838.4763929661, rel=1e-9)
    assert annuity_emi(500000.0, 18.0, 36) == pytest.approx(18076.19776795842, rel=1e-9)
    assert annuity_emi(500000.0, 14.0, 60) == pytest.approx(11634.125424693848, rel=1e-9)
    assert annuity_loan_max(1000.0, 12.0, 1) == pytest.approx(1000.0 / 1.01, rel=1e-12)  # one payment: A / (1 + r)
    assert annuity_emi(10000.0, 12.0, 1) == pytest.approx(10100.0, rel=1e-12)  # one payment: P * (1 + r)


def test_float_formula_reproduces_delta_3s_raw_floats_bit_for_bit() -> None:
    # delta-3's pinned raws came from A * (f - 1) / (r * f) with f = (1 + r) ** n. Downstream scenario suites may
    # compare raws with ==, so the formula's FORM is pinned too (the Decimal oracle agrees to ~1e-14).
    assert annuity_loan_max(33000.0, 11.0, 48) == 1276816.8976926452
    assert annuity_loan_max(33000.0, 14.0, 12) == 367536.02162416704
    assert annuity_loan_max(27000.0, 14.0, 36) == 789990.4174501724
    assert annuity_loan_max(27000.0, 18.0, 12) == 294502.6405882871
    assert annuity_loan_max(13500.0, 14.0, 36) == 394995.2087250862
    assert annuity_loan_max(13500.0, 18.0, 12) == 147251.32029414354
    assert annuity_loan_max(23000.0, 11.0, 48) == 889902.6862706315
    assert annuity_loan_max(23000.0, 14.0, 12) == 256161.4696168437
    assert annuity_loan_max(29500.0, 11.0, 48) == 1141396.9236949403
    assert annuity_loan_max(29500.0, 14.0, 12) == 328554.92842160386
    assert annuity_loan_max(23500.0, 14.0, 36) == 687584.25222515
    assert annuity_loan_max(23500.0, 18.0, 12) == 256326.3723638795
    assert annuity_loan_max(1000.0, 18.0, 12) == 10907.505206973596
    assert annuity_loan_max(1000.0, 14.0, 36) == 29258.90435000638
    assert annuity_emi(500000.0, 18.0, 36) == 18076.197767958507


def test_annuity_is_monotone_in_rate_and_tenure() -> None:
    assert annuity_loan_max(10000.0, 11.0, 60) > annuity_loan_max(10000.0, 14.0, 60)  # cheaper money lends more
    assert annuity_loan_max(10000.0, 14.0, 60) > annuity_loan_max(10000.0, 14.0, 12)  # longer lends more
    assert annuity_emi(100000.0, 11.0, 60) < annuity_emi(100000.0, 14.0, 60)
    assert annuity_emi(100000.0, 14.0, 60) < annuity_emi(100000.0, 14.0, 12)


def test_zero_rate_branches() -> None:
    assert annuity_loan_max(20000.0, 0.0, 48) == 960000.0  # A * n, exactly
    assert annuity_emi(960000.0, 0.0, 48) == 20000.0  # P / n, exactly
    assert annuity_loan_max(0.0, 0.0, 12) == 0.0
    assert annuity_loan_max(20000.0, 0.0, 1) == 20000.0 and annuity_emi(20000.0, 0.0, 1) == 20000.0


def test_annuity_branches_are_total() -> None:
    # r > 0 but so tiny that (1 + r) rounds to 1.0: the formula's 0/0; falls back to the r -> 0 limit
    assert annuity_loan_max(1000.0, 1e-18, 12) == pytest.approx(12000.0, rel=1e-9)
    assert annuity_emi(12000.0, 1e-18, 12) == pytest.approx(1000.0, rel=1e-9)
    # (1 + r) ** n overflows a float: the perpetuity limit A / r (and the interest-only emi P * r)
    r = 100000.0 / 100.0 / 12.0
    assert annuity_loan_max(1000.0, 100000.0, 1200) == pytest.approx(1000.0 / r, rel=1e-9)
    assert annuity_loan_max(1000.0, 100000.0, 1200) == pytest.approx(12.0, rel=1e-9)
    assert annuity_emi(1000.0, 100000.0, 1200) == pytest.approx(1000.0 * r, rel=1e-9)
    # no payments: nothing can be lent / the emi is not defined -> 0.0 rather than raising
    assert annuity_loan_max(1000.0, 12.0, 0) == 0.0
    assert annuity_emi(1000.0, 12.0, 0) == 0.0


# --------------------------------------------------------------------------- available_emi


def test_available_emi_formula() -> None:
    assert available_emi(0.55, 60000.0, 0.0) == pytest.approx(33000.0)
    assert available_emi(0.55, 80000.0, 21000.0) == pytest.approx(23000.0)  # delta-3 full-check case
    assert available_emi(0.50, 60000.0, 10000.0) == pytest.approx(20000.0)
    assert available_emi(0.45, 60000.0, 3500.0) == pytest.approx(23500.0)


def test_available_emi_is_floored_at_zero() -> None:
    assert available_emi(0.50, 30000.0, 16000.0) == 0.0  # 15000 - 16000 < 0
    assert available_emi(0.50, 30000.0, 15000.0) == 0.0  # exactly zero headroom
    assert available_emi(0.50, 30000.0, 15000.01) == 0.0


def test_available_emi_snaps_float_fuzz_at_the_boundary_to_exactly_zero() -> None:
    # 0.55 * 10500 is 5775.000000000001 in binary floating point: a declared EMI of exactly 5775 must still be "no headroom"
    assert 0.55 * 10500 - 5775 > 0.0
    assert available_emi(0.55, 10500.0, 5775.0) == 0.0
    # ...without swallowing real (one-paisa) headroom
    assert available_emi(0.55, 10500.0, 5774.99) == pytest.approx(0.01, abs=1e-6)


# --------------------------------------------------------------------------- lender_numbers: pinned cases
# (id, criteria, income, existing_emi, available, upper_raw, lower_raw, upper_display, lower_display, clipped, foir)

DISPLAY_TABLE = [
    # delta-3 quick happy path: L1 @ 60,000 income, no EMI -> unclipped, displayed FOIR == the limit
    ("L1 quick", L1, 60000.0, 0.0, 33000.0, 1276816.897692634, 367536.0216241658, 1277000, 368000, "none", 0.55),
    # L2 @ 60,000: upper 789,990 exceeds the 500,000 loan cap -> clipped; lower 294,502.64 -> 295,000
    ("L2 quick (clip)", L2, 60000.0, 0.0, 27000.0, 789990.4174501697, 294502.6405882887, 500000, 295000, "upper", 0.30126996279930696),
    # NTC: only L2 accepts; income 30,000
    ("L2 NTC", L2, 30000.0, 0.0, 13500.0, 394995.20872508484, 147251.32029414436, 395000, 147000, "none", 0.45),
    # full four-branch table: income 80,000, existing EMI 21,000
    ("L1 full six-account", L1, 80000.0, 21000.0, 23000.0, 889902.6862706237, 256161.46961684283, 890000, 256000, "none", 0.55),
    # full high-confidence: income 60,000, reported EMI 3,500 only
    ("L1 full reported-only", L1, 60000.0, 3500.0, 29500.0, 1141396.9236949305, 328554.92842160276, 1141000, 329000, "none", 0.55),
    ("L2 full reported-only (clip)", L2, 60000.0, 3500.0, 23500.0, 687584.2522251477, 256326.37236388092, 500000, 256000, "upper", 0.3596032961326403),
    # clip-up: available 1,000 -> lower raw 10,907.51 is below the 25,000 loan_min and is raised to it
    ("L2 clip-up lower", L2, 20000.0, 8000.0, 1000.0, 29258.904350006283, 10907.505206973656, 29000, 25000, "lower", 0.45),
    # delta-2's worked lender (FOIR 0.50, 12-60m, cap 500,000) @ income 60,000, EMI 10,000: floor gave 222,000, half-up 223,000
    ("D2-L1 clip", L1_D2, 60000.0, 10000.0, 20000.0, 919860.6766678339, 222749.10401464594, 500000, 223000, "upper", 0.3605687570782308),
    # delta-2 quick example: income 50,000, no EMI
    ("D2-L1 quick", L1_D2, 50000.0, 0.0, 25000.0, 1149825.8458347924, 278436.3800183074, 500000, 278000, "upper", 0.23268250849387695),
    # delta-2's second lender (FOIR 0.55, 12.5-16%, 24-48m, loans 100k-2M): both bounds round UP
    (
        "D2-L2",
        crit(foir_limit=0.55, rate_min_pct=12.5, rate_max_pct=16.0, tenure_min_months=24, tenure_max_months=48, loan_min=100000.0, loan_max=2000000.0),
        60000.0, 10000.0, 23000.0, 865312.3009392032, 469741.39830695256, 865000, 470000, "none", 0.55,
    ),
]


@pytest.mark.parametrize(
    ("label", "c", "income", "existing", "available", "upper_raw", "lower_raw", "upper_disp", "lower_disp", "clipped", "foir"),
    DISPLAY_TABLE,
    ids=[row[0] for row in DISPLAY_TABLE],
)
def test_lender_numbers_pinned_cases(
    label: str, c: LenderCriteria, income: float, existing: float, available: float, upper_raw: float,
    lower_raw: float, upper_disp: int, lower_disp: int, clipped: str, foir: float,
) -> None:
    n = lender_numbers(c, lender_id=label, income=income, existing_emi=existing, age_months=341)
    assert n.lender_id == label
    assert n.available_emi == pytest.approx(available, rel=1e-12)
    # raws are the UNCLIPPED annuity values (internal + safety)
    assert n.loan_max_upper_raw == pytest.approx(upper_raw, rel=1e-9)
    assert n.loan_max_lower_raw == pytest.approx(lower_raw, rel=1e-9)
    # display values: half-up to 1000, computed once, exact
    assert n.loan_max_upper == upper_disp and n.loan_max_lower == lower_disp
    assert n.ceiling == n.loan_max_upper == upper_disp
    assert n.clipped == clipped
    assert n.not_fundable_alone is False
    assert n.displayed_foir == pytest.approx(foir, rel=1e-9)


def test_display_values_are_whole_thousands_stored_as_floats() -> None:
    n = nums(L1, 60000.0, 0.0)
    for v in (n.loan_max_lower, n.loan_max_upper, n.ceiling):
        assert isinstance(v, float) and v % 1000.0 == 0.0
    assert isinstance(n.loan_max_lower_raw, float) and n.loan_max_lower_raw % 1000.0 != 0.0


def test_display_is_the_helper_applied_to_the_clipped_unrounded_bound() -> None:
    # one rounding, one home: the display value IS round_to_nearest_1000(clipped bound), never a re-rounded display
    for _label, c, income, existing, _a, up, lo, *_rest in DISPLAY_TABLE:
        n = lender_numbers(c, lender_id="x", income=income, existing_emi=existing, age_months=None)
        assert n.loan_max_upper_raw is not None and n.loan_max_lower_raw is not None
        assert c.loan_max is not None and c.loan_min is not None
        upper_c = min(n.loan_max_upper_raw, c.loan_max)
        lower_c = min(max(n.loan_max_lower_raw, c.loan_min), upper_c)
        assert n.loan_max_upper == float(round_to_nearest_1000(upper_c))
        assert n.loan_max_lower == float(round_to_nearest_1000(lower_c))
        assert n.loan_max_upper_raw == pytest.approx(up, rel=1e-9) and n.loan_max_lower_raw == pytest.approx(lo, rel=1e-9)


# --------------------------------------------------------------------------- clip semantics


def test_clip_up_lower_goes_to_loan_min_while_upper_stays_at_affordability() -> None:
    n = nums(L2, 20000.0, 8000.0)  # available 1,000
    assert n.loan_max_lower == 25000.0 and n.loan_max_upper == 29000.0 and n.clipped == "lower"
    assert n.loan_max_lower_raw == pytest.approx(10907.505206973656, rel=1e-9)  # the raw stays raw
    assert n.loan_max_upper_raw == pytest.approx(29258.904350006283, rel=1e-9)


def test_a_bound_is_never_raised_above_affordability() -> None:
    # available 100: even the upper raw (2,925.89) is far below loan_min 25,000. The lower bound may be raised to
    # loan_min but is then capped at the upper: it can never exceed what the applicant can afford. The ceiling
    # (3,000) is left below loan_min on purpose for safety rule 4 to demote.
    n = nums(L2, 20000.0, 8900.0)
    assert n.available_emi == pytest.approx(100.0)
    assert n.loan_max_upper == 3000.0 and n.loan_max_lower == 3000.0 and n.ceiling == 3000.0
    assert n.loan_max_upper is not None and L2.loan_min is not None and n.loan_max_upper < L2.loan_min
    assert n.clipped == "lower"  # lower was raised then capped; upper untouched


def test_both_bounds_clipped() -> None:
    # available 100,000: upper (2.9M) is capped at loan_max; lower (1.09M) also exceeds loan_max -> capped to the upper
    n = nums(L2, 250000.0, 12500.0)
    assert n.available_emi == pytest.approx(100000.0)
    assert n.loan_max_upper == 500000.0 and n.loan_max_lower == 500000.0 and n.clipped == "both"
    assert n.loan_max_upper_raw == pytest.approx(2925890.4350006282, rel=1e-9)
    assert n.loan_max_lower_raw == pytest.approx(1090750.5206973655, rel=1e-9)


def test_lower_never_exceeds_upper_even_with_inverted_admin_criteria() -> None:
    # admin typed tenure_min 48 / tenure_max 12: lower raw (731,890) > upper raw (226,291)
    n = nums(crit(foir_limit=0.5, tenure_min_months=48, tenure_max_months=12, loan_max=2000000.0), 40000.0, 0.0)
    assert n.loan_max_lower_raw == pytest.approx(731890.9191832735, rel=1e-9)
    assert n.loan_max_upper_raw == pytest.approx(226291.29544011888, rel=1e-9)
    assert n.loan_max_lower == n.loan_max_upper == 226000.0 and n.clipped == "lower"


def test_loan_min_above_loan_max_is_total() -> None:
    n = nums(crit(loan_min=800000.0, loan_max=500000.0), 60000.0, 0.0)  # admin error
    assert n.loan_max_upper == 500000.0 and n.loan_max_lower == 500000.0  # lower capped at the (clipped) upper


# --------------------------------------------------------------------------- displayed FOIR (K22 inverse-identity reading)


@pytest.mark.parametrize(
    ("c", "income", "existing", "limit"),
    [
        (L1, 60000.0, 0.0, 0.55),
        (L1, 80000.0, 21000.0, 0.55),
        (L1, 60000.0, 3500.0, 0.55),
        (L2, 30000.0, 0.0, 0.45),  # NTC case
        (L2, 20000.0, 8000.0, 0.45),  # lower clipped UP but loan_max side unclipped: still exactly the limit
        (crit(foir_limit=0.5, rate_min_pct=0.0, rate_max_pct=0.0, loan_max=2000000.0), 40000.0, 0.0, 0.5),  # r = 0
        (
            crit(foir_limit=0.6, rate_min_pct=9.0, rate_max_pct=9.0, tenure_min_months=60, tenure_max_months=60, loan_max=5000000.0),
            90000.0, 12345.0, 0.6,
        ),
    ],
)
def test_displayed_foir_equals_the_limit_exactly_when_unclipped(
    c: LenderCriteria, income: float, existing: float, limit: float
) -> None:
    n = nums(c, income, existing)
    assert n.displayed_foir is not None
    assert abs(n.displayed_foir - limit) <= FOIR_EPS  # the safety rule's own tolerance; the identity is exact to rounding


def test_displayed_foir_is_below_the_limit_when_the_loan_cap_clips() -> None:
    n = nums(L2, 60000.0, 0.0)  # clip: P_wc = 500,000
    assert n.displayed_foir is not None
    # (0 + EMI(500000, 18%, 36m)) / 60000, EMI = 18,076.19776795842
    assert n.displayed_foir == pytest.approx(18076.19776795842 / 60000.0, rel=1e-9)
    assert n.displayed_foir == pytest.approx(0.301269963, abs=1e-9)  # delta-3's constant: same to 9 digits
    assert L2.foir_limit is not None and n.displayed_foir < L2.foir_limit - 0.1


def test_displayed_foir_is_the_inverse_identity_not_the_repriced_ceiling() -> None:
    # the REJECTED reading re-prices the MIN-rate ceiling at the max rate and would give > the limit for L1 (0.5815...)
    n = nums(L1, 60000.0, 0.0)
    assert n.ceiling == 1277000.0
    repriced = annuity_emi(1277000.0, 14.0, 48) / 60000.0
    assert repriced > 0.55 + 0.02  # the reading the design rejected
    assert n.displayed_foir is not None and n.displayed_foir <= 0.55 + FOIR_EPS


def test_displayed_foir_includes_the_existing_emi() -> None:
    n = nums(L2, 60000.0, 3500.0)  # clip at 500,000
    assert n.displayed_foir == pytest.approx((3500.0 + 18076.19776795842) / 60000.0, rel=1e-9)


# --------------------------------------------------------------------------- not fundable alone / no income / defensive


def test_zero_headroom_is_not_fundable_alone() -> None:
    n = nums(crit(foir_limit=0.5), 30000.0, 16000.0)  # 15,000 - 16,000 -> 0
    assert n == LenderNumbers(
        lender_id="L", available_emi=0.0, loan_max_lower=None, loan_max_upper=None, loan_max_lower_raw=None,
        loan_max_upper_raw=None, ceiling=None, displayed_foir=None, age_at_maturity_years=(341 + 48) // 12,
        not_fundable_alone=True, clipped="none",
    )


def test_exact_zero_headroom_and_the_fuzz_boundary_are_not_fundable_alone() -> None:
    assert nums(crit(foir_limit=0.5), 30000.0, 15000.0).not_fundable_alone is True
    fuzz = nums(crit(foir_limit=0.55), 10500.0, 5775.0)  # 0.55 * 10500 - 5775 == 9e-13 in floating point
    assert fuzz.not_fundable_alone is True and fuzz.available_emi == 0.0 and fuzz.ceiling is None


def test_recompute_example_is_still_fundable_with_a_smaller_ceiling() -> None:
    # recompute example: income 50,000, existing 21,000, L1: 0.55*50000 - 21000 = 6,500 (still fundable, smaller ceiling)
    n = nums(L1, 50000.0, 21000.0)
    assert n.available_emi == pytest.approx(6500.0) and n.not_fundable_alone is False
    assert n.ceiling is not None and 0.0 < n.ceiling < 890000.0


@pytest.mark.parametrize("existing", [math.nan, math.inf, -math.inf])
def test_a_non_finite_existing_emi_is_unknown_not_an_exception(existing: float) -> None:
    n = nums(L1, 60000.0, existing)
    assert n.available_emi is None and n.ceiling is None and n.displayed_foir is None and n.not_fundable_alone is False


@pytest.mark.parametrize("income", [None, 0.0, -1.0, -60000.0, math.nan, math.inf, -math.inf])
def test_no_usable_income_means_every_money_field_is_none(income: float | None) -> None:
    n = nums(L1, income, 3500.0)
    assert n == LenderNumbers(
        lender_id="L", available_emi=None, loan_max_lower=None, loan_max_upper=None, loan_max_lower_raw=None,
        loan_max_upper_raw=None, ceiling=None, displayed_foir=None, age_at_maturity_years=(341 + 48) // 12,
        not_fundable_alone=False, clipped="none",
    )


NEEDED = ["foir_limit", "rate_min_pct", "rate_max_pct", "tenure_min_months", "tenure_max_months", "loan_min", "loan_max"]


@pytest.mark.parametrize("missing", NEEDED)
def test_any_missing_required_criterion_makes_the_money_fields_none(missing: str) -> None:
    n = nums(crit(**{missing: None}), 60000.0, 0.0)
    assert (n.available_emi, n.loan_max_lower, n.loan_max_upper, n.loan_max_lower_raw, n.loan_max_upper_raw) == (
        None, None, None, None, None,
    )
    assert n.ceiling is None and n.displayed_foir is None and n.not_fundable_alone is False and n.clipped == "none"
    expected_maturity = None if missing == "tenure_max_months" else (341 + 48) // 12
    assert n.age_at_maturity_years == expected_maturity


BAD_CRITERIA: list[tuple[str, dict[str, Any]]] = [
    ("negative rate_min", {"rate_min_pct": -1.0}),
    ("negative rate_max", {"rate_max_pct": -0.5}),
    ("nan rate_min", {"rate_min_pct": math.nan}),
    ("inf rate_max", {"rate_max_pct": math.inf}),
    ("zero tenure_min", {"tenure_min_months": 0}),
    ("zero tenure_max", {"tenure_max_months": 0}),
    ("negative tenure_min", {"tenure_min_months": -12}),
    ("tenure_max above the cap", {"tenure_max_months": MAX_CRITERIA_TENURE_MONTHS + 1}),
    ("absurd tenure_min", {"tenure_min_months": 10**400}),
    ("nan foir_limit", {"foir_limit": math.nan}),
    ("inf foir_limit", {"foir_limit": math.inf}),
    ("nan loan_min", {"loan_min": math.nan}),
    ("negative loan_max", {"loan_max": -1.0}),
    ("absurd loan_max", {"loan_max": 1.7976931348623157e308}),
    ("loan_min above the cap", {"loan_min": MAX_CRITERIA_LOAN_AMOUNT * 2}),
]


@pytest.mark.parametrize(("label", "bad"), BAD_CRITERIA, ids=[b[0] for b in BAD_CRITERIA])
def test_nonsense_criteria_are_unknown_not_an_exception(label: str, bad: dict[str, Any]) -> None:
    n = nums(crit(**bad), 60000.0, 0.0)
    assert n.available_emi is None and n.ceiling is None and n.loan_max_lower is None and n.displayed_foir is None
    assert n.not_fundable_alone is False and n.clipped == "none"


def test_lender_numbers_survives_overflow_prone_criteria() -> None:
    n = nums(crit(rate_min_pct=100000.0, rate_max_pct=200000.0, tenure_min_months=1200, tenure_max_months=1200), 60000.0, 0.0)
    assert n.ceiling is not None and math.isfinite(n.ceiling)
    assert n.displayed_foir is not None and math.isfinite(n.displayed_foir)


def test_zero_foir_limit_is_zero_headroom() -> None:
    n = nums(crit(foir_limit=0.0), 60000.0, 0.0)
    assert n.not_fundable_alone is True and n.available_emi == 0.0


# --------------------------------------------------------------------------- r = 0 branch through lender_numbers


def test_zero_rate_lender_uses_a_times_n() -> None:
    c = crit(foir_limit=0.5, rate_min_pct=0.0, rate_max_pct=0.0, loan_max=2000000.0)
    n = nums(c, 40000.0, 0.0)  # available 20,000
    assert n.loan_max_upper_raw == 960000.0 and n.loan_max_lower_raw == 240000.0  # 20,000 * 48 and * 12, exactly
    assert (n.loan_max_upper, n.loan_max_lower, n.clipped) == (960000.0, 240000.0, "none")
    assert n.displayed_foir == pytest.approx(0.5, abs=FOIR_EPS)


def test_zero_rate_with_clip_displayed_foir() -> None:
    c = crit(foir_limit=0.5, rate_min_pct=0.0, rate_max_pct=0.0, loan_max=500000.0)
    n = nums(c, 40000.0, 0.0)
    assert (n.loan_max_upper, n.loan_max_lower, n.clipped) == (500000.0, 240000.0, "upper")
    assert n.displayed_foir == pytest.approx(500000.0 / 48 / 40000.0, rel=1e-12)  # P / n, no interest


def test_mixed_zero_min_rate_and_positive_max_rate() -> None:
    c = crit(foir_limit=0.5, rate_min_pct=0.0, rate_max_pct=12.0, loan_max=2000000.0)
    n = nums(c, 40000.0, 0.0)
    assert n.loan_max_upper_raw == pytest.approx(960000.0)  # r = 0 at the min rate: A * 48
    assert n.loan_max_lower_raw == pytest.approx(225101.5494696926, rel=1e-9)  # 12% over 12 months
    assert (n.loan_max_upper, n.loan_max_lower) == (960000.0, 225000.0)
    assert n.displayed_foir == pytest.approx(0.5, abs=FOIR_EPS)


# --------------------------------------------------------------------------- age (8.5)


def test_age_example_from_the_prd() -> None:
    assert age_years_and_months(date(1998, 4, 12), date(2026, 10, 10)) == (28, 341)  # 28*12 + 6 - 1 (day 10 < 12)


@pytest.mark.parametrize(
    ("dob", "on", "years", "months"),
    [
        (date(1998, 10, 10), date(2026, 10, 10), 28, 336),  # birthday today: no borrow
        (date(1998, 10, 11), date(2026, 10, 10), 27, 335),  # birthday tomorrow: borrow one month
        (date(1998, 10, 9), date(2026, 10, 10), 28, 336),
        (date(2000, 1, 31), date(2026, 3, 1), 26, 313),  # day 1 < 31: borrow
        (date(2000, 2, 29), date(2026, 2, 28), 25, 311),  # leap-day dob, day before the non-leap anniversary
        (date(2000, 2, 29), date(2026, 3, 1), 26, 312),
        (date(2026, 10, 10), date(2026, 10, 10), 0, 0),
        (date(2005, 10, 11), date(2026, 10, 10), 20, 251),  # 21st birthday tomorrow: still 20
        (date(2005, 10, 10), date(2026, 10, 10), 21, 252),  # 21 today
    ],
)
def test_age_years_and_months(dob: date, on: date, years: int, months: int) -> None:
    assert age_years_and_months(dob, on) == (years, months)


def test_age_months_are_clamped_at_zero_for_a_future_dob() -> None:
    assert age_years_and_months(date(2027, 1, 1), date(2026, 10, 10)) == (0, 0)
    assert age_years_and_months(date(2026, 10, 11), date(2026, 10, 10)) == (0, 0)  # -1 month before the clamp


def test_age_years_agree_with_months_over_a_walk() -> None:
    on = date(2026, 10, 10)
    for year in range(1960, 2027):
        for month in (1, 2, 6, 10, 12):
            for day in (1, 9, 10, 11, 28):
                y, m = age_years_and_months(date(year, month, day), on)
                assert y == m // 12 and m >= 0


@pytest.mark.parametrize(
    ("age_months", "tenure_max", "years"),
    [
        (341, 60, 33),  # PRD example: 401 months
        (341, 48, 32),  # 389 months
        (419, 1, 35),  # 420 months exactly -> 35
        (418, 1, 34),  # 419 months -> 34 (integer math, no rounding up)
        (0, 0, 0),
        (252, 36, 24),
    ],
)
def test_age_at_maturity_is_integer_month_math(age_months: int, tenure_max: int, years: int) -> None:
    assert age_at_maturity_years(age_months, tenure_max) == years


def test_age_at_maturity_reaches_lender_numbers() -> None:
    assert nums(L1, 60000.0, 0.0, age_months=341).age_at_maturity_years == (341 + 48) // 12 == 32
    assert nums(L2, 60000.0, 0.0, age_months=341).age_at_maturity_years == (341 + 36) // 12 == 31
    assert nums(L1_D2, 60000.0, 0.0, age_months=341).age_at_maturity_years == 33  # the PRD's tenure-60 example
    assert nums(L1, 60000.0, 0.0, age_months=None).age_at_maturity_years is None  # dob unknown: age check skipped


# --------------------------------------------------------------------------- candidate pincodes (8.6)


def identity(pan_pincode: str | None) -> Identity:
    return Identity(
        pan_valid=True, status="valid", category="person", full_name="ASHA VERMA", dob=date(1998, 4, 12),
        aadhaar_linked=True, pan_pincode=pan_pincode, surepass_client_id=None,
    )


def report(bureau_pincodes: list[str]) -> ReportFacts:
    return ReportFacts(
        band="720-780", score=745, is_ntc=False, report_date=date(2026, 10, 1), open_accounts=[], past_due_total=0.0,
        written_off_or_settled=0, enquiries_30d=0, enquiries_90d=0, enquiries_180d=0, bureau_pincodes_12m=bureau_pincodes,
    )


def test_quick_pincodes_are_the_pan_pincode_only() -> None:
    assert candidate_pincodes(identity("560001"), None, kind="quick") == ["560001"]
    assert candidate_pincodes(identity(None), None, kind="quick") == []


def test_quick_ignores_a_report_even_if_one_is_handed_in() -> None:
    assert candidate_pincodes(identity("560001"), report(["110001"]), kind="quick") == ["560001"]


def test_full_pincodes_are_the_sorted_deduped_union() -> None:
    assert candidate_pincodes(identity("560002"), report(["110001"]), kind="full") == ["110001", "560002"]
    assert candidate_pincodes(identity("560001"), report(["560001", "400001", "110001"]), kind="full") == [
        "110001", "400001", "560001",
    ]
    assert candidate_pincodes(identity("560001"), report(["400001", "400001"]), kind="full") == ["400001", "560001"]


def test_full_pincodes_without_a_pan_pincode_or_without_a_report() -> None:
    assert candidate_pincodes(identity(None), report(["400001", "110001"]), kind="full") == ["110001", "400001"]
    assert candidate_pincodes(identity("560001"), None, kind="full") == ["560001"]
    assert candidate_pincodes(identity(None), None, kind="full") == []
    assert candidate_pincodes(identity(None), report([]), kind="full") == []


def test_candidate_pincodes_are_filtered_through_the_registry_pattern() -> None:
    assert candidate_pincodes(identity("056001"), report(["20", "5600011", "110001", ""]), kind="full") == ["110001"]
    assert candidate_pincodes(identity("12345"), None, kind="quick") == []


def test_full_pincode_union_on_the_fixture_world() -> None:
    client = scn.full_pincode_union_scenario()
    ident = pan_to_identity(client.pan_comprehensive("chk", "h" * 64, scn.PAN_OK))
    facts = report_to_report_facts(
        client.experian_report("chk", "h" * 64, name="Asha Verma", mobile="9876543210", pan=scn.PAN_OK)
    )
    assert ident.pan_pincode == "560002" and facts.bureau_pincodes_12m == ["110001"]
    assert candidate_pincodes(ident, facts, kind="full") == ["110001", "560002"]  # 400001 was reported > 12 months ago
    assert candidate_pincodes(ident, facts, kind="quick") == ["560002"]


# --------------------------------------------------------------------------- module surface


def test_range_module_keeps_its_seed_functions_and_a_pure_import_surface() -> None:
    tree = ast.parse((Path(eligibility_agent.__file__).parent / "derive" / "range.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    assert found <= {"__future__", "math", "re", "datetime", "typing", "eligibility_agent.graph.state"}, found
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {
        "is_ntc_value", "band_from_score", "round_to_nearest_1000", "age_years_and_months", "age_at_maturity_years",
        "candidate_pincodes", "available_emi", "annuity_loan_max", "annuity_emi", "lender_numbers",
    } <= names


def test_foir_eps_is_the_pinned_tolerance() -> None:
    assert FOIR_EPS == 1e-9
