"""Golden example definitions (pure data, 100% synthetic) and the LangSmith dataset upload.

Two datasets: ``eligibility-quick-golden`` (12 examples) and ``eligibility-full-golden`` (10). Every example is plain JSON so
it survives a LangSmith round trip; the runner (``evals.run``) interprets it:

* ``request``: the API body, built by ``tests.helpers.quick_payload`` / ``full_payload`` (the shared synthetic world:
  PANs, names, mobiles come from those pools, never from a probe file);
* ``lenders``: ids of the K17 seed lenders (``tests.helpers.SEED_LENDERS``, the single source) published for the run;
* ``surepass``: the world, ``{"scenario": name, "kwargs": {...}}`` (``evals.worlds``);
* ``rank``: ``"llm"`` (the model the run is configured for) or ``"invalid_twice"`` (the scripted fake that fails twice:
  the K7 golden must always use it, even in a real-model run);
* ``pre_state``: ordered setup steps, each ``{"op": ...}``: ``seed_counted_quick_checks``, ``seed_cached_quick_world``,
  ``run_check`` (a prior check through the same graph), ``advance_days``;
* ``expected``: the golden's ``reference_outputs``, HAND-WRITTEN LITERALS (money, labels, dates, flags). Nothing here is
  recomputed from the code under test: the dossier's float tables are hand-scaled where the example needs another
  available EMI, and every literal was checked against the run, never copied from it. Derivation notes sit beside them.

Expected-value keys (all optional; a present key is a pin, ``None`` is a pin too): the view fields (``status``, ``code``,
``headline``, ``confidence``, ``coapplicant_suggested``, ``label``, ``items_to_confirm``, ``pan_aadhaar_link_warning``,
``quick_checks_left_today``, ``checked_on``, ``refresh_available_on``), ``top_lender_id`` / ``alternate_ids``, ``band``,
``gated_lender_ids``, ``lender_numbers`` (every lender the view may name -> ``{"ceiling", "range"}``), ``billed``,
``counted_against_quota``, ``from_cache``, ``admission``, ``surepass_endpoints`` (a multiset: order is never pinned).

Timeline: every run starts at EVAL_T0 = 2026-10-10 10:00 IST on a frozen clock. Money: income 60,000 and EMI 0 unless
stated; L1 (FOIR 0.55, 11-14%, 12-48 months, 50,000-15,00,000), L2 (0.45, 14-18%, 12-36 months, 25,000-5,00,000,
accepts NTC), L3 (720 floor, serves 400001 only). Half-up to 1,000 (K6).
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Sequence
from typing import Any, Final, Literal, Protocol, TypeAlias, TypedDict
from uuid import UUID

from langsmith.schemas import ExampleCreate

from eligibility_agent.evals.worlds import scenario
from eligibility_agent.surepass.client import PAN_ENDPOINT, REPORT_ENDPOINT, SCORE_ENDPOINT
from tests.helpers import (
    LC_A,
    OMIT,
    OPP_TWO,
    PAN_COMPANY,
    PAN_NTC,
    PAN_OK,
    PAN_TWO,
    full_payload,
    quick_payload,
)

__all__ = [
    "ALL_LENDERS",
    "FULL_DATASET",
    "QUICK_DATASET",
    "DatasetChoice",
    "DatasetsClient",
    "GoldenExample",
    "GoldenInputs",
    "build_langsmith_datasets",
    "dataset_names",
    "full_examples",
    "inputs_of",
    "quick_examples",
    "selected_examples",
]

QUICK_DATASET: Final[str] = "eligibility-quick-golden"
FULL_DATASET: Final[str] = "eligibility-full-golden"
DatasetChoice: TypeAlias = Literal["quick", "full", "both"]
ALL_LENDERS: Final[tuple[str, ...]] = ("L1", "L2", "L3")  # SEED_LENDERS ids

_EXAMPLE_NAMESPACE: Final[UUID] = uuid.uuid5(uuid.NAMESPACE_DNS, "eligibility-agent.evals.goldens")


class GoldenInputs(TypedDict):
    example_id: str
    kind: Literal["quick", "full"]
    request: dict[str, Any]
    lenders: list[str]
    surepass: dict[str, Any]
    rank: Literal["llm", "invalid_twice"]
    pre_state: list[dict[str, Any]]


class GoldenExample(GoldenInputs):
    expected: dict[str, Any]


# --------------------------------------------------------------------------- hand-written literals
# Copy shared by several goldens (the strings the LC sees; never imported from output/lc_view).
QUICK_LABEL: Final[str] = "Indicative, based on what the learner told us"
FULL_LABEL: Final[str] = "bureau report on 2026-10-01, income as declared"
AADHAAR_WARNING: Final[str] = "PAN not linked to Aadhaar; lenders will ask for this"
NO_PINCODE_LABEL: Final[str] = "No pincode on the PAN record; lender matching will be available after the full check"
PAN_NOT_VERIFIED_LABEL: Final[str] = (
    "PAN not verified. Correct the PAN with the learner and run the check again; no quota was used."
)
BUREAU_FAILURE_LABEL: Final[str] = "Credit check couldn't complete, retry"
OPENAI_FAILURE_LABEL: Final[str] = "Couldn't generate a recommendation, retry"
QUOTA_LABEL: Final[str] = (
    "You have used all your quick checks for today. The limit resets at midnight; saved results stay available."
)
STAGE_LABEL: Final[str] = "A full check unlocks once this opportunity reaches the Offer stage. Until then, use the quick check."
WINDOW_LABEL: Final[str] = "This learner's full report is still current. Refresh unlocks once it is 30 days old."
PAST_TENURE_ITEM: Final[str] = "past tenure, confirm with learner"
STALE_ITEM_2026_06_01: Final[str] = "confirm with learner: an account last updated 2026-06-01"
STALE_ITEM_2025_09_01: Final[str] = "confirm with learner: an account last updated 2025-09-01"


def _money(ceiling: int | None, lower: int | None = None) -> dict[str, Any]:
    """``{"ceiling", "range"}`` for a lender whose range top equals its ceiling (always true: ceiling = upper)."""
    return {"ceiling": ceiling, "range": None if ceiling is None or lower is None else {"lower": lower, "upper": ceiling}}


# Money at income 60,000 / EMI 0 (the quick world): L1 available 0.55 x 60,000 = 33,000 -> raw 367,536.02 / 1,276,816.90;
# L2 available 0.45 x 60,000 = 27,000 -> raw lower 294,502.64, raw upper 789,990.42 clipped to its 5,00,000 cap.
QUICK_L1: Final[dict[str, Any]] = _money(1277000, 368000)
QUICK_L2: Final[dict[str, Any]] = _money(500000, 295000)
NO_NUMBERS: Final[dict[str, Any]] = {"ceiling": None, "range": None}


def _example(
    example_id: str,
    kind: Literal["quick", "full"],
    request: dict[str, Any],
    surepass: dict[str, Any],
    expected: dict[str, Any],
    *,
    rank: Literal["llm", "invalid_twice"] = "llm",
    pre_state: Sequence[dict[str, Any]] = (),
    lenders: Sequence[str] = ALL_LENDERS,
) -> GoldenExample:
    return {
        "example_id": example_id,
        "kind": kind,
        "request": request,
        "lenders": list(lenders),
        "surepass": surepass,
        "rank": rank,
        "pre_state": [dict(step) for step in pre_state],
        "expected": copy.deepcopy(expected),
    }


def _ok_quick(**over: Any) -> dict[str, Any]:
    """The happy quick outcome (q01): medium confidence (quick checks cap at medium, K16), 'possible', L1 then L2."""
    base: dict[str, Any] = {
        "status": "ok",
        "code": None,
        "headline": "possible",
        "confidence": "medium",
        "coapplicant_suggested": False,
        "top_lender_id": "L1",
        "alternate_ids": ["L2"],
        "label": QUICK_LABEL,
        "items_to_confirm": [],
        "pan_aadhaar_link_warning": None,
        "quick_checks_left_today": 4,
        "checked_on": "2026-10-10",
        "refresh_available_on": "2026-11-09",
        "band": "720-780",
        "gated_lender_ids": ["L1", "L2"],
        "lender_numbers": {"L1": QUICK_L1, "L2": QUICK_L2},
        "billed": True,
        "counted_against_quota": True,
        "from_cache": False,
        "admission": "new",
        "surepass_endpoints": [PAN_ENDPOINT, SCORE_ENDPOINT],
    }
    base.update(over)
    return base


def _below_floor_610() -> dict[str, Any]:
    expected = _ok_quick(
        confidence="low",
        coapplicant_suggested=True,
        top_lender_id=None,
        alternate_ids=[],
        band="<650",
    )
    del expected["headline"]
    expected["headline_in"] = ["not_fundable_alone", "under_review"]
    return expected


# --------------------------------------------------------------------------- quick (12)


def quick_examples() -> list[GoldenExample]:
    ntc_30k: dict[str, Any] = {
        # NTC (HTTP 200 + success:true + a score that is not three digits): counted (a no-hit consumes quota), 'unlikely'
        # with a co-applicant; L2 is the only NTC-accepting lender in the pincode. L2 at income 30,000: available
        # 0.45 x 30,000 = 13,500 -> raw 394,995.21 / 147,251.32 -> 395,000 / 147,000.
        "status": "ok",
        "code": None,
        "headline": "unlikely",
        "confidence": "medium",
        "coapplicant_suggested": True,
        "top_lender_id": "L2",
        "alternate_ids": [],
        "label": QUICK_LABEL,
        "items_to_confirm": [],
        "pan_aadhaar_link_warning": None,
        "quick_checks_left_today": 4,
        "checked_on": "2026-10-10",
        "refresh_available_on": "2026-11-09",
        "band": "NTC",
        "gated_lender_ids": ["L1", "L2"],
        "lender_numbers": {"L2": _money(395000, 147000)},
        "billed": True,
        "counted_against_quota": True,
        "from_cache": False,
        "admission": "new",
        "surepass_endpoints": [PAN_ENDPOINT, SCORE_ENDPOINT],
    }
    return [
        _example(
            "q01_happy_745",
            "quick",
            quick_payload(),
            scenario("happy_quick"),
            _ok_quick(),
        ),
        _example(
            "q02_ntc_empty",
            "quick",
            quick_payload(pan=PAN_NTC, monthly_income=30000),
            scenario("ntc", variant="empty"),
            ntc_30k,
        ),
        _example(
            "q03_ntc_text",
            "quick",
            quick_payload(pan=PAN_NTC, monthly_income=30000),
            scenario("ntc", variant="text"),
            ntc_30k,
        ),
        _example(
            # K7 (settled by the spec: the band read and the gated list still show). The model fails twice: service
            # failure + OPENAI_FAILURE, the band-derived headline ('possible' for 720-780, never 'under_review'),
            # confidence low, no top lender, up to two code-built gate survivors (id order, floor cleared) with NO
            # numbers, and the Score pull that happened is billed AND counted (K8: status-independent).
            "q04_openai_invalid_twice",
            "quick",
            quick_payload(),
            scenario("happy_quick"),
            {
                "status": "service_failure",
                "code": "OPENAI_FAILURE",
                "headline": "possible",
                "confidence": "low",
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": ["L1", "L2"],
                "label": OPENAI_FAILURE_LABEL,
                "items_to_confirm": [],
                "pan_aadhaar_link_warning": None,
                "quick_checks_left_today": 4,
                "checked_on": "2026-10-10",
                "refresh_available_on": "2026-11-09",
                "band": "720-780",
                "gated_lender_ids": ["L1", "L2"],
                "lender_numbers": {"L1": NO_NUMBERS, "L2": NO_NUMBERS},
                "billed": True,
                "counted_against_quota": True,
                "from_cache": False,
                "admission": "new",
                "surepass_endpoints": [PAN_ENDPOINT, SCORE_ENDPOINT],
            },
            rank="invalid_twice",
        ),
        _example(
            # THE v1.1 NTC-reversal regression pin: (HTTP 200, success:false, no_record_found) is a SERVICE FAILURE, not
            # NTC. The PAN call (a billed Rs 1) happened; the Score did not bill; nothing counts, nothing is cached.
            "q05_score_success_false",
            "quick",
            quick_payload(pan=PAN_NTC, monthly_income=30000),
            scenario("score_failure", variant="success_false", pan=PAN_NTC),
            {
                "status": "service_failure",
                "code": "SUREPASS_FAILURE",
                "headline": None,
                "confidence": None,
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": [],
                "label": BUREAU_FAILURE_LABEL,
                "items_to_confirm": [],
                "quick_checks_left_today": 5,
                "billed": True,
                "counted_against_quota": False,
                "from_cache": False,
                "admission": "new",
                "surepass_endpoints": [PAN_ENDPOINT, SCORE_ENDPOINT],
            },
        ),
        _example(
            "q06_pan_company",
            "quick",
            quick_payload(pan=PAN_COMPANY),
            scenario("pan_invalid", kind="company", pan=PAN_COMPANY),
            {
                "status": "pan_not_verified",
                "code": None,
                "headline": None,
                "confidence": None,
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": [],
                "label": PAN_NOT_VERIFIED_LABEL,
                "quick_checks_left_today": 5,
                "billed": True,
                "counted_against_quota": False,
                "from_cache": False,
                "admission": "new",
                "surepass_endpoints": [PAN_ENDPOINT],
            },
        ),
        _example(
            # A PAN + Score pulled two days ago (the raw rows are pre-seeded): the check is free, served from cache with
            # the ORIGINAL dates, derivations re-run on today's inputs, and makes no call at all.
            "q07_cache_hit",
            "quick",
            quick_payload(),
            scenario("none"),
            _ok_quick(
                checked_on="2026-10-08",
                refresh_available_on="2026-11-07",
                quick_checks_left_today=5,
                billed=False,
                counted_against_quota=False,
                from_cache=True,
                admission="cache_hit",
                surepass_endpoints=[],
            ),
            pre_state=[{"op": "seed_cached_quick_world", "pan": PAN_OK, "days_before": 2}],
        ),
        _example(
            # Five counted quick checks already today for this LC: the sixth is refused before any call.
            "q08_quota_exhausted",
            "quick",
            quick_payload(pan=PAN_TWO),
            scenario("happy_quick", pan=PAN_TWO),
            {
                "status": "refused",
                "code": "QUOTA_EXHAUSTED",
                "headline": None,
                "confidence": None,
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": [],
                "label": QUOTA_LABEL,
                "items_to_confirm": [],
                "quick_checks_left_today": 0,
                "billed": False,
                "counted_against_quota": False,
                "from_cache": False,
                "admission": "refused",
                "surepass_endpoints": [],
            },
            pre_state=[{"op": "seed_counted_quick_checks", "lc_id": LC_A, "n": 5, "hours_before": 1}],
        ),
        _example(
            # The PAN record has no pincode: the band read only (720-780 -> 'possible'), no lender, no model call; the
            # Score pull happened, so billed and counted.
            "q09_no_pincode",
            "quick",
            quick_payload(),
            scenario("no_pincode"),
            {
                "status": "ok",
                "code": "NO_PINCODE_ON_PAN",
                "headline": "possible",
                "confidence": None,
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": [],
                "label": NO_PINCODE_LABEL,
                "items_to_confirm": [],
                "quick_checks_left_today": 4,
                "band": "720-780",
                "gated_lender_ids": [],
                "lender_numbers": {},
                "billed": True,
                "counted_against_quota": True,
                "from_cache": False,
                "admission": "new",
                "surepass_endpoints": [PAN_ENDPOINT, SCORE_ENDPOINT],
            },
        ),
        _example(
            # No declared income and no EMI: lenders are still named, but there is no number anywhere.
            "q10_no_income",
            "quick",
            quick_payload(monthly_income=OMIT, existing_emi=OMIT),
            scenario("happy_quick"),
            _ok_quick(lender_numbers={"L1": NO_NUMBERS, "L2": NO_NUMBERS}),
        ),
        _example(
            # Score 610 -> band below 650: both pincode lenders (floor 650) are out of reach, so nobody survives: final
            # confidence is low (headline only, no lender, PRD 9.4) with a co-applicant suggested. The headline itself is
            # the MODEL's choice between two legal low-confidence ones (PRD 9.4: 'Under review' or 'Not fundable
            # alone'): a model that names nobody (the fake) reads 'under_review', one whose floor-ignoring picks are all
            # demoted reads 'not_fundable_alone'. Pinned as a set, not a point.
            "q11_band_610",
            "quick",
            quick_payload(),
            scenario("happy_quick", score="610"),
            _below_floor_610(),
        ),
        _example(
            "q12_aadhaar_not_linked",
            "quick",
            quick_payload(),
            scenario("quick_world", aadhaar_linked=False),
            _ok_quick(pan_aadhaar_link_warning=AADHAAR_WARNING),
        ),
    ]


# --------------------------------------------------------------------------- full (10)


def full_examples() -> list[GoldenExample]:
    """Full checks run at the Offer stage (``full_payload`` defaults: stage 'offer', opportunity opp-1, income 60,000)."""

    def ok_full(**over: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "status": "ok",
            "code": None,
            "headline": "possible",
            "confidence": "medium",
            "coapplicant_suggested": False,
            "top_lender_id": "L1",
            "alternate_ids": ["L2"],
            "label": FULL_LABEL,
            "items_to_confirm": [],
            "pan_aadhaar_link_warning": None,
            "quick_checks_left_today": None,  # full checks never touch the quick quota
            "checked_on": "2026-10-10",
            "refresh_available_on": "2026-11-09",
            "band": "720-780",
            "gated_lender_ids": ["L1", "L2"],
            "billed": True,
            "counted_against_quota": False,  # a full check never counts against the quick quota
            "from_cache": False,
            "admission": "new",
            "surepass_endpoints": [PAN_ENDPOINT, REPORT_ENDPOINT],
        }
        base.update(over)
        return base

    # Reported EMI 3,500 only, income 60,000: L1 available 0.55 x 60,000 - 3,500 = 29,500 -> raw 1,141,396.92 /
    # 328,554.93 -> 1,141,000 / 329,000. L2 available 23,500 -> upper clipped to 5,00,000, raw lower 256,326.37.
    reported_l1, reported_l2 = _money(1141000, 329000), _money(500000, 256000)
    # Four-branch world, income 80,000, derived EMI 21,000 (3,500 + 8,000 + 5,000 + 2,000 + 2,500; the stale account
    # excluded): L1 available 0.55 x 80,000 - 21,000 = 23,000 -> raw 889,902.69 / 256,161.47 -> 890,000 / 256,000.
    four_branch = {
        "items_to_confirm": [PAST_TENURE_ITEM, STALE_ITEM_2026_06_01],
        "lender_numbers": {"L1": _money(890000, 256000), "L2": _money(439000, 164000)},
    }
    seeded_full = [{"op": "run_check", "kind": "full", "request": full_payload()}]
    return [
        _example(
            "f01_happy_offer_reported_emi",
            "full",
            full_payload(),
            scenario("full_happy_reported_emi"),
            ok_full(
                headline="fundable",
                confidence="high",
                lender_numbers={"L1": reported_l1, "L2": reported_l2},
            ),
        ),
        _example(
            "f02_stage_too_early",
            "full",
            full_payload(opportunity_stage="qualified"),
            scenario("full_happy_reported_emi"),
            {
                "status": "refused",
                "code": "STAGE_TOO_EARLY",
                "headline": None,
                "confidence": None,
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": [],
                "label": STAGE_LABEL,
                "items_to_confirm": [],
                "quick_checks_left_today": None,
                "billed": False,
                "counted_against_quota": False,
                "from_cache": False,
                "admission": "refused",
                "surepass_endpoints": [],
            },
        ),
        _example(
            # PAN zip 560002; bureau zips 110001 (open account reported 2026-09-01) and junk '20'; 400001 sits on an
            # account reported 2025-09-01, over the 12-month window, so L3 (400001 only) must NOT be gated. The old
            # account is also stale (over 90 days before the report): excluded from the EMI and a confirm item.
            # EMI used = 5,000: L1 available 33,000 - 5,000 = 28,000 -> raw 1,083,359.79 / 311,848.75 -> 1,083,000 /
            # 312,000; L2 available 27,000 - 5,000 = 22,000 -> raw upper 643,695.90 clipped to 5,00,000, raw lower
            # 239,965.11 -> 240,000; all scaled by hand from the dossier's L1@33,000 / L2@27,000 floats.
            "f03_pincode_union",
            "full",
            full_payload(),
            scenario("full_pincode_union"),
            ok_full(
                items_to_confirm=[STALE_ITEM_2025_09_01],
                lender_numbers={"L1": _money(1083000, 312000), "L2": _money(500000, 240000)},
            ),
        ),
        _example(
            "f04_emi_four_branch",
            "full",
            full_payload(monthly_income=80000, existing_emi=OMIT),
            scenario("full_emi_four_branch"),
            ok_full(**four_branch),
        ),
        _example(
            # Reported 3,500 plus one STALE account (EMI 9,999, reported 122 days before the report): excluded from the
            # sum, a confirm item, and confidence falls from high to medium. Money is the f01 money (EMI 3,500 only).
            "f05_stale_account",
            "full",
            full_payload(),
            scenario("full_stale_account"),
            ok_full(
                items_to_confirm=[STALE_ITEM_2026_06_01],
                lender_numbers={"L1": reported_l1, "L2": reported_l2},
            ),
        ),
        _example(
            # One report per PAN per 30 days, NOT per opportunity: opportunity 2, five days later, is a free hit that
            # re-derives on ITS declared income 80,000 (L1 available 0.55 x 80,000 - 3,500 = 40,500 -> raw upper
            # 1,567,002.56 clipped to 15,00,000, raw lower 451,067.84; L2 available 32,500 -> upper clipped to
            # 5,00,000, raw lower 354,494.07).
            "f06_second_opportunity_same_pan",
            "full",
            full_payload(opportunity_id=OPP_TWO, monthly_income=80000),
            scenario("full_happy_reported_emi"),
            ok_full(
                headline="fundable",
                confidence="high",
                lender_numbers={"L1": _money(1500000, 451000), "L2": _money(500000, 354000)},
                billed=False,
                from_cache=True,
                admission="cache_hit",
                surepass_endpoints=[],
            ),
            pre_state=[
                {"op": "run_check", "kind": "full", "request": full_payload()},
                {"op": "advance_days", "days": 5},
            ],
        ),
        _example(
            # PRD 3: a Report pull refreshes the Score cache. The next QUICK check on the PAN calls nothing (the world
            # has no Score endpoint) and is served from the report's raw, on the DECLARED figures only (income 60,000,
            # EMI 0 -> the quick money), free of quota.
            "f07_report_refreshes_score",
            "quick",
            quick_payload(),
            scenario("full_happy_reported_emi"),
            _ok_quick(
                quick_checks_left_today=5,
                billed=False,
                counted_against_quota=False,
                from_cache=True,
                admission="cache_hit",
                surepass_endpoints=[],
            ),
            pre_state=seeded_full,
        ),
        _example(
            # NTC warning handshake: a quick check found no file on 2026-10-10; five days later a full check without the
            # acknowledgement is refused WARN_NTC_RECENT (the pre-state), and the ACKNOWLEDGED re-check pulls the
            # (billed, uncounted) Report: 'unlikely' with a co-applicant, L2 the only NTC-accepting lender (income
            # 30,000 -> 395,000 / 147,000).
            "f08_ntc_full",
            "full",
            full_payload(pan=PAN_NTC, monthly_income=30000, acknowledge_ntc_warning=True),
            scenario("ntc_quick_and_full"),
            ok_full(
                headline="unlikely",
                coapplicant_suggested=True,
                top_lender_id="L2",
                alternate_ids=[],
                checked_on="2026-10-15",
                refresh_available_on="2026-11-14",
                band="NTC",
                lender_numbers={"L2": _money(395000, 147000)},
                surepass_endpoints=[REPORT_ENDPOINT],
            ),
            pre_state=[
                {"op": "run_check", "kind": "quick", "request": quick_payload(pan=PAN_NTC, monthly_income=30000)},
                {"op": "advance_days", "days": 5},
                {"op": "run_check", "kind": "full", "request": full_payload(pan=PAN_NTC, monthly_income=30000)},
            ],
        ),
        _example(
            # A declared EMI of 25,000 is higher than the derived 21,000, so it is the one used: L1 available
            # 0.55 x 80,000 - 25,000 = 19,000 -> raw 735,137.00 / 211,611.65 -> 735,000 / 212,000; L2 available
            # 0.45 x 80,000 - 25,000 = 11,000 -> raw 321,847.95 / 119,982.56 -> 322,000 / 120,000 (scaled from the
            # dossier's L2@27,000 floats).
            "f09_declared_emi_wins",
            "full",
            full_payload(monthly_income=80000, existing_emi=25000),
            scenario("full_emi_four_branch"),
            ok_full(
                items_to_confirm=four_branch["items_to_confirm"],
                lender_numbers={"L1": _money(735000, 212000), "L2": _money(322000, 120000)},
            ),
        ),
        _example(
            # The report is 10 days old: refresh is refused until it is 30 days old, with the date it unlocks.
            "f10_report_window_refresh",
            "full",
            full_payload(refresh=True),
            scenario("full_happy_reported_emi"),
            {
                "status": "refused",
                "code": "REPORT_WINDOW_OPEN",
                "headline": None,
                "confidence": None,
                "coapplicant_suggested": False,
                "top_lender_id": None,
                "alternate_ids": [],
                "label": WINDOW_LABEL,
                "items_to_confirm": [],
                "quick_checks_left_today": None,
                "refresh_available_on": "2026-11-09",
                "billed": False,
                "counted_against_quota": False,
                "from_cache": False,
                "admission": "refused",
                "surepass_endpoints": [],
            },
            pre_state=[*seeded_full, {"op": "advance_days", "days": 10}],
        ),
    ]


# --------------------------------------------------------------------------- selection helpers


def dataset_names(choice: DatasetChoice) -> list[str]:
    return {"quick": [QUICK_DATASET], "full": [FULL_DATASET], "both": [QUICK_DATASET, FULL_DATASET]}[choice]


def selected_examples(dataset_name: str) -> list[GoldenExample]:
    if dataset_name == QUICK_DATASET:
        return quick_examples()
    if dataset_name == FULL_DATASET:
        return full_examples()
    raise KeyError(f"unknown golden dataset {dataset_name!r}")


def inputs_of(example: GoldenExample) -> GoldenInputs:
    """The example minus its expected values: what a LangSmith run receives as ``inputs`` (and what ``run_one`` takes)."""
    return {
        "example_id": example["example_id"],
        "kind": example["kind"],
        "request": copy.deepcopy(example["request"]),
        "lenders": list(example["lenders"]),
        "surepass": copy.deepcopy(example["surepass"]),
        "rank": example["rank"],
        "pre_state": copy.deepcopy(example["pre_state"]),
    }


# --------------------------------------------------------------------------- LangSmith upload (modern API)


class _HasId(Protocol):
    @property
    def id(self) -> UUID: ...


class DatasetsClient(Protocol):
    """The slice of ``langsmith.Client`` the upload uses (the real client conforms structurally; tests pass a recorder)."""

    def has_dataset(self, *, dataset_name: str) -> bool: ...

    def create_dataset(self, dataset_name: str, *, description: str) -> _HasId: ...

    def read_dataset(self, *, dataset_name: str) -> _HasId: ...

    def delete_dataset(self, *, dataset_id: UUID) -> None: ...

    def create_examples(self, *, dataset_id: UUID, examples: Sequence[ExampleCreate]) -> object: ...


def _example_create(dataset_name: str, example: GoldenExample) -> ExampleCreate:
    """One upload: a deterministic id (re-running the build UPSERTS rather than duplicating), the inputs, the golden's
    expected values as the reference outputs, and the id / kind as metadata."""
    return ExampleCreate(
        id=uuid.uuid5(_EXAMPLE_NAMESPACE, f"{dataset_name}:{example['example_id']}"),
        inputs=dict(inputs_of(example)),
        outputs=copy.deepcopy(example["expected"]),
        metadata={"example_id": example["example_id"], "kind": example["kind"]},
    )


def build_langsmith_datasets(client: DatasetsClient, *, overwrite: bool = False) -> dict[str, int]:
    """Create each golden dataset when missing and upload its examples; returns ``{dataset_name: examples uploaded}``.

    Uses the current ``create_examples(dataset_id=..., examples=[ExampleCreate, ...])`` form (no deprecated ``inputs=`` /
    ``outputs=`` kwargs). Example ids are deterministic, so a second run updates in place; ``overwrite=True`` deletes and
    recreates the dataset first. Never called from pytest (network); the runner calls it for ``--build-datasets``.
    """
    uploaded: dict[str, int] = {}
    for name in (QUICK_DATASET, FULL_DATASET):
        examples = selected_examples(name)
        exists = client.has_dataset(dataset_name=name)
        if exists and overwrite:
            client.delete_dataset(dataset_id=client.read_dataset(dataset_name=name).id)
            exists = False
        dataset = (
            client.read_dataset(dataset_name=name)
            if exists
            else client.create_dataset(name, description=f"Synthetic golden examples ({len(examples)}) for the eligibility agent")
        )
        client.create_examples(dataset_id=dataset.id, examples=[_example_create(name, e) for e in examples])
        uploaded[name] = len(examples)
    return uploaded
