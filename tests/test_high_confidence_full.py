"""PRD 9.4 / K16 at graph level: the HIGH-confidence full check, and one contrast per high-tier condition.

High requires ALL of: a full check; every open account's EMI bureau-reported (no estimate, no stale, no past-tenure
account); ``used_source == 'derived'`` (no higher declared EMI); the PAN date of birth known; no 'uncertain' required
criteria field on any gated lender; a gated lender whose floor the band clears by at least one bucket, and the surviving
top clearing its own floor by at least one bucket (the post-safety cap). Breaking exactly one condition lands at medium
('possible'); a model can lower confidence but never raise it back.

World: full_happy_reported_emi_scenario (ONE open account, reported EMI 3,500, ReportDate 2026-10-01, score 745 -> band
720-780, zips 560001, PAN dob 1998-04-12), income 60,000. Money oracles (dossier, hand-written): L1 available
0.55 x 60,000 - 3,500 = 29,500 -> raw 1,141,396.92 / 328,554.93 -> 1,141,000 / 329,000; L2 available
0.45 x 60,000 - 3,500 = 23,500 -> raw upper 687,584.25 clipped to its 5,00,000 cap, raw lower 256,326.37 -> 256,000.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import pytest

from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, CallLogSink, FakeSurePassClient
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing.fake_llm import FakeRankModel, ranked_output
from eligibility_agent.testing.scenarios import REPORT_DATE, full_happy_reported_emi_scenario
from tests.helpers import (
    OMIT,
    PAN_OK,
    ApiFactory,
    SurePassFactory,
    assert_no_leak,
    criteria_json,
    internal_of,
    quick_and_full_client,
    seed_lender,
)

FULL_LABEL: Final[str] = "bureau report on 2026-10-01, income as declared"
L1_VIEW: Final[dict[str, Any]] = {
    "id": "L1",
    "name": "Alpha Education Finance",
    "rate_band": "11 to 14%",
    "tenure_range": "12 to 48 months",
    "ceiling": 1141000,
    "range": {"lower": 329000, "upper": 1141000},
    "conditions": [],
}
L2_VIEW: Final[dict[str, Any]] = {
    "id": "L2",
    "name": "Bharat Upskill Credit",
    "rate_band": "14 to 18%",
    "tenure_range": "12 to 36 months",
    "ceiling": 500000,
    "range": {"lower": 256000, "upper": 500000},
    "conditions": [],
}


def _reported_account() -> dict[str, Any]:
    """The happy world's one account: EMI 3,500 reported 11 days before the report."""
    return env.cais_account(
        scheduled_emi="3500", current_balance="150000", date_reported="20260920", account_number="SYNTHACC0001"
    )


def _report_world(
    *, extra_accounts: Sequence[dict[str, Any]] = (), credit_score: str = "745", dob: str = "1998-04-12"
) -> SurePassFactory:
    """PAN + Report for PAN_OK: the happy world's account plus ``extra_accounts``, one knob changed at a time."""

    def factory(sink: CallLogSink) -> FakeSurePassClient:
        report = env.experian_report_envelope(
            accounts=[_reported_account(), *extra_accounts], credit_score=credit_score, report_date=REPORT_DATE
        )
        return FakeSurePassClient(
            pan={PAN_OK: env.pan_comprehensive_envelope(pan=PAN_OK, dob=dob)}, report={PAN_OK: report}, sink=sink
        )

    return factory


def _without_rationale(view: dict[str, Any]) -> dict[str, Any]:
    rationale = view["rationale"]
    assert isinstance(rationale, str) and rationale
    return {k: v for k, v in view.items() if k != "rationale"}


def _assert_medium(body: dict[str, Any]) -> None:
    assert (body["status"], body["code"], body["confidence"], body["headline"]) == ("ok", None, "medium", "possible")
    assert body["top_lender"] is not None  # medium still shows the picks
    assert_no_leak(body)


# --------------------------------------------------------------------------- the high tier


def test_reported_emi_only_full_check_is_high_and_fundable(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    assert api.endpoints() == [PAN_ENDPOINT, REPORT_ENDPOINT]
    assert {k: v for k, v in body.items() if k not in ("check_id", "top_lender", "alternates")} == {
        "kind": "full",
        "status": "ok",
        "code": None,
        "headline": "fundable",
        "label": FULL_LABEL,
        "confidence": "high",
        "coapplicant_suggested": False,
        "coapplicant_reason": None,
        "pan_aadhaar_link_warning": None,
        "items_to_confirm": [],
        "checked_on": "2026-10-10",
        "from_cache": False,
        "refresh_available_on": "2026-11-09",
        "quick_checks_left_today": None,
    }
    assert _without_rationale(body["top_lender"]) == L1_VIEW
    assert [_without_rationale(a) for a in body["alternates"]] == [L2_VIEW]

    internal = internal_of(api.row(body["check_id"]))
    emi = internal["derived"]["emi"]
    assert (emi["reported_total"], emi["estimated_total"], emi["used"], emi["used_source"], emi["emi_floor_flag"]) == (
        pytest.approx(3500.0),
        pytest.approx(0.0),
        pytest.approx(3500.0),
        "derived",
        False,
    )
    assert internal["derived"]["age_at_application_years"] is not None  # dob known
    assert (internal["code_confidence"], internal["final_confidence"]) == ("high", "high")
    assert internal["safety"]["top_clears_floor_by_one"] is True
    l2 = internal["lender_numbers"]["L2"]
    assert (l2["clipped"], l2["loan_max_upper_raw"]) == ("upper", pytest.approx(687584.25222515, rel=1e-12))
    assert internal["lender_numbers"]["L1"]["loan_max_upper_raw"] == pytest.approx(1141396.9236949403, rel=1e-12)
    assert_no_leak(body)


def test_a_declared_zero_emi_still_leaves_the_derived_figure_in_use(make_api: ApiFactory) -> None:
    """``existing_emi: 0`` is declared but not higher than the reported 3,500: used_source stays 'derived', so high."""
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))

    body = api.full(monthly_income=60000, existing_emi=0)

    assert (body["confidence"], body["headline"]) == ("high", "fundable")
    assert _without_rationale(body["top_lender"]) == L1_VIEW


def test_the_model_may_lower_high_to_medium(make_api: ApiFactory) -> None:
    api = make_api(
        lambda sink: full_happy_reported_emi_scenario(sink=sink),
        rank=FakeRankModel([ranked_output("L1", "L2", confidence="medium")]),
    )

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    _assert_medium(body)
    assert internal_of(api.row(body["check_id"]))["code_confidence"] == "high"


# --------------------------------------------------------------------------- one broken condition each -> medium


def test_quick_check_on_the_same_facts_caps_at_medium(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: quick_and_full_client(sink))
    assert api.full(monthly_income=60000, existing_emi=OMIT)["confidence"] == "high"

    body = api.quick(monthly_income=60000, existing_emi=3500)  # served from the report's score: same band

    assert body["from_cache"] is True
    _assert_medium(body)
    assert internal_of(api.row(body["check_id"]))["code_confidence"] == "medium"


def test_a_higher_declared_emi_is_used_and_caps_at_medium(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))

    body = api.full(monthly_income=60000, existing_emi=5000)

    _assert_medium(body)
    emi = internal_of(api.row(body["check_id"]))["derived"]["emi"]
    assert (emi["used"], emi["used_source"]) == (pytest.approx(5000.0), "declared")


@pytest.mark.parametrize(
    ("extra", "item"),
    [
        pytest.param(
            env.cais_account(
                current_balance="90000", portfolio_type="I", open_date="20250601", date_reported="20260915",
                account_number="SYNTHACC0002",
            ),
            None,
            id="estimated-balance-36",
        ),
        pytest.param(
            env.cais_account(
                current_balance="60000", terms_duration="024", open_date="20240115", date_reported="20260915",
                account_number="SYNTHACC0002",
            ),
            "past tenure, confirm with learner",
            id="past-tenure",
        ),
        pytest.param(
            env.cais_account(
                scheduled_emi="2000", current_balance="70000", date_reported="20260601", account_number="SYNTHACC0002"
            ),
            "confirm with learner: an account last updated 2026-06-01",
            id="stale",
        ),
    ],
)
def test_any_unreported_emi_account_caps_at_medium(make_api: ApiFactory, extra: dict[str, Any], item: str | None) -> None:
    api = make_api(_report_world(extra_accounts=[extra]))

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    _assert_medium(body)
    assert body["items_to_confirm"] == ([] if item is None else [item])
    assert internal_of(api.row(body["check_id"]))["code_confidence"] == "medium"


def test_an_unknown_pan_date_of_birth_caps_at_medium(make_api: ApiFactory) -> None:
    api = make_api(_report_world(dob=""))

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    _assert_medium(body)
    derived = internal_of(api.row(body["check_id"]))["derived"]
    assert (derived["age_at_application_years"], derived["age_months_at_application"]) == (None, None)


def test_a_band_on_the_floor_bucket_caps_at_medium(make_api: ApiFactory) -> None:
    """Score 700 -> band 650-720: every gated floor (650) sits in the band's own bucket, cleared by zero buckets."""
    api = make_api(_report_world(credit_score="700"))

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    _assert_medium(body)
    internal = internal_of(api.row(body["check_id"]))
    assert (internal["derived"]["band"], internal["code_confidence"]) == ("650-720", "medium")


def test_an_uncertain_required_criteria_field_caps_at_medium(make_api: ApiFactory) -> None:
    api = make_api(lambda sink: full_happy_reported_emi_scenario(sink=sink))
    l1 = seed_lender("L1")
    criteria = criteria_json(l1.criteria)
    criteria["foir_limit"] = {"value": 0.55, "status": "uncertain"}  # same value, the admin is unsure of it
    api.repo.set_lender_published("L1", 1, False)
    api.repo.seed_lender("L1", 2, l1.name, True, criteria, l1.policy_notes, l1.pincodes)

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    _assert_medium(body)
    assert _without_rationale(body["top_lender"]) == L1_VIEW  # same numbers: only the confidence moved


def test_a_surviving_top_on_its_floor_bucket_caps_high_to_medium(make_api: ApiFactory) -> None:
    """Code confidence is high (L1 / L2 clear their floors by one bucket), but the model's top is a lender whose floor
    (720) sits in the band's own bucket: the post-safety cap brings the final confidence down to medium."""
    api = make_api(
        lambda sink: full_happy_reported_emi_scenario(sink=sink),
        rank=FakeRankModel([ranked_output("L3", "L1", "L2", confidence="high")]),
    )
    l3 = seed_lender("L3")
    api.repo.set_lender_published("L3", 1, False)
    api.repo.seed_lender("L3", 2, l3.name, True, criteria_json(l3.criteria), l3.policy_notes, ("400001", "560001"))
    assert l3.criteria["min_score_floor"]["value"] == 720  # bucket 720-780: the band's own bucket

    body = api.full(monthly_income=60000, existing_emi=OMIT)

    _assert_medium(body)
    assert body["top_lender"]["id"] == "L3"
    internal = internal_of(api.row(body["check_id"]))
    assert (internal["code_confidence"], internal["safety"]["top_clears_floor_by_one"]) == ("high", False)
