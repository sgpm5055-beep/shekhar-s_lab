"""PRD 12 item 12 at graph level: the EMI algorithm on all four branches plus the stale and past-tenure flags, end to
end through the API (the unit battery is ``test_emi_derivation_unit.py``).

World: full_emi_four_branch_scenario (ReportDate 2026-10-01 = the tenure anchor, PAN zip 560001, score 745 -> band
720-780), income 80,000, stage offer, T0 = 2026-10-10 10:00 IST. The dossier's six-account table (delta-api tests part 2):

| acct | branch                                  | EMI                                       |
| a1   | 1 reported                               | 3,500                                     |
| a2   | 2 tenure remaining (48 - 33 = 15 months) | 1,20,000 / 15 = 8,000                     |
| a3   | 2 past tenure (24 - 33 < 1)              | 60,000 / 12 = 5,000 + the past-tenure item |
| a4   | 3 revolving                              | 5% x 40,000 = 2,000                        |
| a5   | 4 fallback                               | 90,000 / 36 = 2,500                        |
| a6   | STALE (Date_Reported 122 days before)    | excluded from every sum + its confirm item |

Derived 21,000 (reported 3,500 + estimated 17,500); FOIR 21,000 / 80,000 = 0.2625.

Money oracles are hand-written (never recomputed from the code): L1 available 0.55 x 80,000 - 21,000 = 23,000 -> the
dossier's raw 889,902.69 / 256,161.47 -> half-up 890,000 / 256,000. L2 available 0.45 x 80,000 - 21,000 = 15,000,
hand-derived by linearity from the dossier's L2@27,000 floats: raw 438,883.57 / 163,612.58 -> 439,000 / 164,000.
Declared 25,000 (declared higher wins): L1 available 19,000 -> by linearity from L1@33,000: raw 735,137.00 /
211,611.65 -> 735,000 / 212,000.

The confirm-item strings are what ``derive/emi.py`` emits (T9 note): 'past tenure, confirm with learner' and
'confirm with learner: an account last updated <YYYY-MM-DD>'.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT
from eligibility_agent.testing.scenarios import full_emi_four_branch_scenario
from tests.helpers import OMIT, Api, ApiFactory, assert_no_leak, internal_of, string_leaves

FULL_LABEL: Final[str] = "bureau report on 2026-10-01, income as declared"
PAST_TENURE_ITEM: Final[str] = "past tenure, confirm with learner"
STALE_ITEM: Final[str] = "confirm with learner: an account last updated 2026-06-01"

L1_AT_23K: Final[tuple[str, int, dict[str, int]]] = ("L1", 890000, {"lower": 256000, "upper": 890000})
L2_AT_15K: Final[tuple[str, int, dict[str, int]]] = ("L2", 439000, {"lower": 164000, "upper": 439000})
L1_AT_19K: Final[tuple[str, int, dict[str, int]]] = ("L1", 735000, {"lower": 212000, "upper": 735000})

# (kind, method, monthly_amount, past_tenure, confirm_item) per account, in report order.
ITEMS_TABLE: Final[list[tuple[str, str | None, float, bool, str | None]]] = [
    ("reported", "reported", 3500.0, False, None),
    ("estimated", "tenure_remaining", 8000.0, False, None),
    ("estimated", "past_tenure", 5000.0, True, PAST_TENURE_ITEM),
    ("estimated", "revolving_5pct", 2000.0, False, None),
    ("estimated", "balance_36", 2500.0, False, None),
    ("stale_excluded", None, 0.0, False, STALE_ITEM),
]


def _world(make_api: ApiFactory) -> Api:
    return make_api(lambda sink: full_emi_four_branch_scenario(sink=sink))


def _money(view: dict[str, Any]) -> tuple[str, int | None, dict[str, int] | None]:
    return view["id"], view["ceiling"], view["range"]


def _without_rationale(view: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in view.items() if k != "rationale"}


def test_six_account_report_drives_the_full_view(make_api: ApiFactory) -> None:
    api = _world(make_api)

    body = api.full(monthly_income=80000, existing_emi=OMIT)

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert {k: v for k, v in body.items() if k not in ("check_id", "top_lender", "alternates")} == {
        "kind": "full",
        "status": "ok",
        "code": None,
        "headline": "possible",  # estimates present: medium, never 'fundable'
        "label": FULL_LABEL,
        "confidence": "medium",
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [PAST_TENURE_ITEM, STALE_ITEM],
        "checked_on": "2026-10-10",
        "from_cache": False,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": None,
    }
    assert _money(body["top_lender"]) == L1_AT_23K
    assert [_money(a) for a in body["alternates"]] == [L2_AT_15K]
    assert _without_rationale(body["top_lender"]) == {
        "id": "L1",
        "name": "Alpha Education Finance",
        "rate_band": "11 to 14%",
        "tenure_range": "12 to 48 months",
        "ceiling": 890000,
        "range": {"lower": 256000, "upper": 890000},
        "conditions": [],
    }
    row = api.row(body["check_id"])
    assert (row.billed, row.counted_against_quota) == (True, False)
    assert_no_leak(body)


def test_the_derivation_table_is_recorded_in_internal(make_api: ApiFactory) -> None:
    api = _world(make_api)

    body = api.full(monthly_income=80000, existing_emi=OMIT)

    internal = internal_of(api.row(body["check_id"]))
    emi = internal["derived"]["emi"]
    assert (emi["reported_total"], emi["estimated_total"], emi["derived_total"]) == (
        pytest.approx(3500.0),
        pytest.approx(17500.0),
        pytest.approx(21000.0),
    )
    assert (emi["declared"], emi["used"], emi["used_source"], emi["emi_floor_flag"]) == (
        None,
        pytest.approx(21000.0),
        "derived",
        True,
    )
    assert [
        (i["kind"], i["method"], pytest.approx(i["monthly_amount"]), i["past_tenure"], i["confirm_item"])
        for i in emi["items"]
    ] == ITEMS_TABLE
    keys = [i["account_key"] for i in emi["items"]]
    assert len(set(keys)) == 6 and not any("SYNTHACC" in key for key in keys)  # synthetic keys, never Account_Number
    assert internal["derived"]["current_foir"] == pytest.approx(0.2625)
    assert internal["derived"]["band"] == "720-780"
    assert internal["code_confidence"] == "medium"

    l1 = internal["lender_numbers"]["L1"]
    assert l1["available_emi"] == pytest.approx(23000.0)
    assert (l1["ceiling"], l1["loan_max_lower"], l1["loan_max_upper"]) == (890000, 256000, 890000)
    assert (l1["loan_max_upper_raw"], l1["loan_max_lower_raw"]) == (
        pytest.approx(889902.6862706315, rel=1e-12),
        pytest.approx(256161.4696168437, rel=1e-12),
    )
    assert internal["lender_numbers"]["L2"]["available_emi"] == pytest.approx(15000.0)


@pytest.mark.parametrize(
    ("declared", "used", "source", "foir", "l1"),
    [
        pytest.param(0, 21000.0, "derived", 0.2625, L1_AT_23K, id="declared-zero"),
        pytest.param(2000, 21000.0, "derived", 0.2625, L1_AT_23K, id="declared-lower-ignored"),
        pytest.param(21000, 21000.0, "derived", 0.2625, L1_AT_23K, id="declared-equal-is-not-higher"),
        pytest.param(25000, 25000.0, "declared", 0.3125, L1_AT_19K, id="declared-higher-wins"),
    ],
)
def test_declared_emi_is_used_only_when_strictly_higher_than_the_derived_total(
    make_api: ApiFactory,
    declared: int,
    used: float,
    source: str,
    foir: float,
    l1: tuple[str, int, dict[str, int]],
) -> None:
    api = _world(make_api)

    body = api.full(monthly_income=80000, existing_emi=declared)

    emi = internal_of(api.row(body["check_id"]))["derived"]["emi"]
    assert (emi["declared"], emi["derived_total"]) == (pytest.approx(float(declared)), pytest.approx(21000.0))
    assert (emi["used"], emi["used_source"]) == (pytest.approx(used), source)
    assert internal_of(api.row(body["check_id"]))["derived"]["current_foir"] == pytest.approx(foir)
    assert _money(body["top_lender"]) == l1
    # both figures are visible to the confirm prompts either way; the label stays the full-check provenance
    assert (body["label"], body["items_to_confirm"]) == (FULL_LABEL, [PAST_TENURE_ITEM, STALE_ITEM])
    assert body["confidence"] == "medium"
    assert_no_leak(body)


def test_derived_estimates_never_reach_an_lc_facing_string(make_api: ApiFactory) -> None:
    """PRD 7.3 / 8.3: the per-account estimates, the derived total and the stale account's EMI live in ``internal`` only.
    Every string leaf of the view (the uuid-ish ids aside) is free of them, grouped or not."""
    api = _world(make_api)

    body = api.full(monthly_income=80000, existing_emi=OMIT)

    texts = _strings(body)
    assert texts, "the view carries prose (label, items, names): the scan below must not be vacuous"
    for figure in ("21000", "21,000", "17500", "17,500", "8000", "8,000", "3500", "3,500", "2500", "9999", "0.2625"):
        assert not any(figure in text for text in texts), figure


def _strings(body: dict[str, Any]) -> list[str]:
    """Every string leaf of the view except the uuid-ish ``check_id`` and lender ``id`` fields."""
    return [text for path, text in string_leaves(body) if path[-1:] not in (("check_id",), ("id",))]
