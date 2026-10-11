"""PRD 12 item 2 + PRD 5.4: the same PAN from a second LC inside 30 days is a free cache hit whose derivations, gate,
ranking and safety re-run on the CURRENT inputs and the CURRENT published lenders.

Money pins (hand-derived from the dossier's raw floats; the annuity is linear in the available EMI): income 90,000 gives
L1 0.55 x 90,000 = 49,500 = 1.5 x 33,000 -> raw upper 1.5 x 1,276,816.90 = 1,915,225.35, clipped to L1's 15,00,000 cap;
raw lower 1.5 x 367,536.02 = 551,304.03 -> 551,000. L2 0.45 x 90,000 = 40,500 = 1.5 x 27,000 -> upper clipped to
5,00,000; raw lower 1.5 x 294,502.64 = 441,753.96 -> 442,000.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from eligibility_agent.surepass.client import PAN_ENDPOINT, SCORE_ENDPOINT
from eligibility_agent.testing.scenarios import happy_quick_scenario
from tests.helpers import (
    LC_A,
    LC_B,
    SeedLender,
    Api,
    ApiFactory,
    assert_no_leak,
    internal_of,
    known,
    seed_lender,
    seed_lender_row,
)


def _happy(make_api: ApiFactory) -> Api:
    return make_api(lambda sink: happy_quick_scenario(sink=sink))


def _money(view: dict[str, Any]) -> tuple[str, int | None, dict[str, int] | None]:
    return view["id"], view["ceiling"], view["range"]


def test_second_lc_same_pan_within_30_days_is_free_and_rederives_on_its_own_inputs(make_api: ApiFactory) -> None:
    api = _happy(make_api)
    first = api.quick(lc_id=LC_A)
    first_row = api.row(first["check_id"])
    assert (first_row.billed, first_row.counted_against_quota, first["quick_checks_left_today"]) == (True, True, 4)

    api.clock.advance(timedelta(days=3))
    second = api.quick(lc_id=LC_B, monthly_income=90000)

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]  # unchanged: zero new SurePass calls
    assert api.calls(second["check_id"]) == []
    assert (second["status"], second["from_cache"]) == ("ok", True)
    assert (second["checked_on"], second["refresh_available_on"]) == ("2026-10-10", "2026-11-09")  # the ORIGINAL pull
    assert second["quick_checks_left_today"] == 5  # LC_B used nothing
    row = api.row(second["check_id"])
    assert (row.billed, row.counted_against_quota, row.is_refresh) == (False, False, False)
    assert row.billed == any(api.billable(second["check_id"]))

    # derivations re-ran on LC_B's income: new money, from the same raw response
    assert _money(second["top_lender"]) == ("L1", 1500000, {"lower": 551000, "upper": 1500000})
    assert [_money(a) for a in second["alternates"]] == [("L2", 500000, {"lower": 442000, "upper": 500000})]
    first_l1 = internal_of(first_row)["lender_numbers"]["L1"]["available_emi"]
    second_l1 = internal_of(row)["lender_numbers"]["L1"]["available_emi"]
    assert (first_l1, second_l1) == (pytest.approx(33000.0), pytest.approx(49500.0))  # 0.55 x 60,000 / 0.55 x 90,000
    assert api.rank.call_count == 2  # ranking re-ran on the hit (PRD 5.4)
    # the hit shows the ORIGINAL consent reference (PRD 5.1)
    assert internal_of(row)["original_consent_id"] == first_row.consent_id
    assert api.repo.quick_checks_used_today(LC_B) == 0
    assert_no_leak(second)


def test_cache_hit_regates_against_the_currently_published_lenders(make_api: ApiFactory) -> None:
    """T11 carried pin, end to end: between two hits the admin unpublishes L1, then publishes an L1 v2 with a lower loan
    cap. Each hit re-reads the published set; the stored raw is the same throughout."""
    api = _happy(make_api)
    first = api.quick()
    assert [first["top_lender"]["id"], *(a["id"] for a in first["alternates"])] == ["L1", "L2"]

    api.repo.set_lender_published("L1", 1, False)
    unpublished = api.quick(lc_id=LC_B)
    assert unpublished["from_cache"] is True
    assert _money(unpublished["top_lender"]) == ("L2", 500000, {"lower": 295000, "upper": 500000})
    assert unpublished["alternates"] == []
    assert "L1" not in internal_of(api.row(unpublished["check_id"]))["gated_lender_ids"]

    l1 = seed_lender("L1")
    seed_lender_row(
        api.repo,
        SeedLender(
            id="L1",
            name=l1.name,
            criteria={**l1.criteria, **known(loan_max=1000000)},
            policy_notes=l1.policy_notes,
            pincodes=l1.pincodes,
            version=2,
        ),
    )
    republished = api.quick(lc_id=LC_B)
    assert republished["from_cache"] is True
    # 1,276,816.90 at 11% / 48 m now clips to v2's 10,00,000 cap; the lower bound is unchanged
    assert _money(republished["top_lender"]) == ("L1", 1000000, {"lower": 368000, "upper": 1000000})
    assert [a["id"] for a in republished["alternates"]] == ["L2"]

    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]  # one pull served all three checks
    assert api.rank.call_count == 3
    assert_no_leak(republished)


def test_same_lc_repeat_on_a_later_day_is_a_free_hit(make_api: ApiFactory) -> None:
    """Same PAN, same LC, three days later: a hit (QK1) that neither bills nor counts; the new IST day's counter is
    untouched."""
    api = _happy(make_api)
    api.quick()
    api.clock.advance(timedelta(days=3))
    again = api.quick()

    assert (again["from_cache"], again["quick_checks_left_today"]) == (True, 5)  # day 3: a new IST day, nothing counted
    assert api.endpoints() == [PAN_ENDPOINT, SCORE_ENDPOINT]
    row = api.row(again["check_id"])
    assert (row.billed, row.counted_against_quota) == (False, False)
