"""Shared synthetic worlds for the rank-seam tests (Task 10: graph/prompts.py, graph/llm.py, testing/fake_llm.py).

Every value is synthetic. ``POISON`` is the list of literals that must NEVER reach the model: every state this module
builds carries them in the fields the rank seam must not read (PAN, learner name, mobile, pincodes, the score number,
the date of birth, ids, hashes, the lender's display name, account keys), so a rendered context that is free of them
proves the mapping, not the fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Final, cast

from eligibility_agent.graph.state import (
    AdmissionDecision,
    Band,
    CheckState,
    Confidence,
    ConsentInfo,
    Derived,
    EmiDerivation,
    EmiItem,
    GatedLender,
    Identity,
    Kind,
    LenderCriteria,
    LenderNumbers,
    OpenAccount,
    ReportFacts,
    ScoreFacts,
    SurePassCallAudit,
)

PAN: Final[str] = "QZXPL4826M"
LEARNER_NAME: Final[str] = "Quillon Vandersloot"
MOBILE: Final[str] = "9000047218"
PIN_PAN: Final[str] = "411057"
PIN_BUREAU_A: Final[str] = "700091"
PIN_BUREAU_B: Final[str] = "380015"
SCORE_NUMBER: Final[int] = 767
DOB: Final[date] = date(1994, 7, 21)
LENDER_DISPLAY_NAME: Final[str] = "Zenith Finserv Pvt Ltd"
CLIENT_ID: Final[str] = "kyc_client_zq81tw"
CONSENT_ID: Final[str] = "consent-zq-0042"
WARNING_LENDER: Final[str] = "zq-lender-warn"
BUREAU_ENDPOINT: Final[str] = "/api/v1/credit-report-experian/score"

# Bureau names are never allowed to reach the model (nor an LC). The poisoned state below carries every one of them in
# fields the rank seam must not read (a gate warning, an error line, a SurePass call audit endpoint), and the scans are
# case-insensitive.
BUREAU_NAMES: Final[tuple[str, ...]] = ("experian", "cibil", "equifax", "crif", "transunion", "highmark")
GATE_WARNING: Final[str] = f"lender {WARNING_LENDER} excluded: missing score_floor (CIBIL cutoff, Equifax and CRIF unknown)"
ERROR_LINE: Final[str] = "Highmark and TransUnion lookups skipped; zq-error-line"

# Every literal below is planted in a state field the rank seam must not read. (The score's digits are checked as a
# bare substring on purpose: the fixtures' own numbers are chosen so that none of them contains 767.)
POISON: Final[tuple[str, ...]] = (
    PAN,
    PAN[-4:],
    LEARNER_NAME,
    "Quillon",
    "Vandersloot",
    MOBILE,
    MOBILE[-4:],
    PIN_PAN,
    PIN_BUREAU_A,
    PIN_BUREAU_B,
    str(SCORE_NUMBER),
    DOB.isoformat(),
    "1994",
    CLIENT_ID,
    "203.0.113.77",
    "callref-q7z",
    "lc-zq-0042",
    "lead-zq-0042",
    "opp-zq-0042",
    "chk-zq-0042",
    "9c4e1d7a" * 8,
    "5b2f8e0c" * 8,
    LENDER_DISPLAY_NAME,
    "Zenith",
    "acct_0",
    "acct_1",
    CONSENT_ID,
    WARNING_LENDER,
    "zq-error-line",
    "credit-report-experian",
)


def crit(**overrides: Any) -> LenderCriteria:
    """Seed lender L1: FOIR 0.55, 11-14%, 12-48m, loans 50,000-1,500,000, floor 650 (bucket '650-720')."""
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


def gl(lender_id: str = "L1", **criteria_overrides: Any) -> GatedLender:
    """A gated lender whose DISPLAY NAME and pincodes are poison (neither may reach the model)."""
    return GatedLender(
        id=lender_id,
        name=f"{LENDER_DISPLAY_NAME} {lender_id}",
        criteria=crit(**criteria_overrides),
        matched_pincodes=[PIN_PAN, PIN_BUREAU_A],
        rate_band_label="11 to 14%",
        tenure_label="12 to 48 months",
    )


def num(lender_id: str = "L1", **overrides: Any) -> LenderNumbers:
    """Healthy numbers for the seed lender: nothing breaches unless an override says so."""
    fields: dict[str, Any] = {
        "lender_id": lender_id,
        "available_emi": 20000.0,
        "loan_max_lower": 300000.0,
        "loan_max_upper": 800000.0,
        "loan_max_lower_raw": 300400.0,
        "loan_max_upper_raw": 800100.0,
        "ceiling": 800000.0,
        "displayed_foir": 0.5,
        "age_at_maturity_years": 36,
        "not_fundable_alone": False,
        "clipped": "none",
    }
    fields.update(overrides)
    return LenderNumbers(**fields)


def identity(**overrides: Any) -> Identity:
    fields: dict[str, Any] = {
        "pan_valid": True,
        "status": "valid",
        "category": "person",
        "full_name": LEARNER_NAME,
        "dob": DOB,
        "aadhaar_linked": True,
        "pan_pincode": PIN_PAN,
        "surepass_client_id": CLIENT_ID,
    }
    fields.update(overrides)
    return Identity(**fields)


def open_account(index: int, **overrides: Any) -> OpenAccount:
    fields: dict[str, Any] = {
        "account_key": f"acct_{index}",
        "portfolio_type": "I",
        "account_type": "05",
        "scheduled_emi": 12500.0,
        "current_balance": 150000.0,
        "tenure_months": 24,
        "rate_of_interest": 11.5,
        "open_date": date(2025, 6, 1),
        "date_reported": date(2026, 9, 15),
        "amount_past_due": 0.0,
        "pincodes": [PIN_BUREAU_A],
        "is_stale": False,
    }
    fields.update(overrides)
    return OpenAccount(**fields)


def report_facts(**overrides: Any) -> ReportFacts:
    fields: dict[str, Any] = {
        "band": "720-780",
        "score": SCORE_NUMBER,
        "is_ntc": False,
        "report_date": date(2026, 10, 1),
        "open_accounts": [open_account(0), open_account(1, scheduled_emi=None, is_stale=True)],
        "past_due_total": 0.0,
        "written_off_or_settled": 0,
        "enquiries_30d": 1,
        "enquiries_90d": 2,
        "enquiries_180d": 3,
        "bureau_pincodes_12m": [PIN_BUREAU_A, PIN_BUREAU_B],
    }
    fields.update(overrides)
    return ReportFacts(**fields)


def full_emi(**overrides: Any) -> EmiDerivation:
    """The full-check EMI table shape: 12,500 reported + 31,500 estimated (one past-tenure), one stale account excluded."""
    fields: dict[str, Any] = {
        "reported_total": 12500.0,
        "estimated_total": 31500.0,
        "derived_total": 44000.0,
        "declared": None,
        "used": 44000.0,
        "used_source": "derived",
        "emi_floor_flag": True,
        "items": [
            EmiItem(
                account_key="acct_0",
                kind="reported",
                monthly_amount=12500.0,
                method="reported",
                past_tenure=False,
                confirm_item=None,
            ),
            EmiItem(
                account_key="acct_2",
                kind="estimated",
                monthly_amount=26500.0,
                method="past_tenure",
                past_tenure=True,
                confirm_item="past tenure, confirm with learner",
            ),
            EmiItem(
                account_key="acct_3",
                kind="estimated",
                monthly_amount=5000.0,
                method="balance_36",
                past_tenure=False,
                confirm_item=None,
            ),
            EmiItem(
                account_key="acct_1",
                kind="stale_excluded",
                monthly_amount=0.0,
                method=None,
                past_tenure=False,
                confirm_item="confirm with learner: an account last updated 2026-05-20",
            ),
        ],
    }
    fields.update(overrides)
    return EmiDerivation(**fields)


def quick_emi(declared: float | None = None) -> EmiDerivation:
    """What ``derive_existing_emi(kind='quick')`` produces."""
    return EmiDerivation(
        reported_total=0.0,
        estimated_total=0.0,
        derived_total=0.0,
        declared=declared,
        used=declared if declared is not None else 0.0,
        used_source="declared" if declared is not None else "declared_zero_assumed",
        emi_floor_flag=declared is None,
        items=[],
    )


def derived(*, band: Band = "720-780", emi: EmiDerivation | None = None, **overrides: Any) -> Derived:
    fields: dict[str, Any] = {
        "band": band,
        "emi": emi if emi is not None else full_emi(),
        "current_foir": 0.7333333333,
        "age_at_application_years": 32,
        "age_months_at_application": 387,
        "candidate_pincodes": [PIN_PAN, PIN_BUREAU_A, PIN_BUREAU_B],
    }
    fields.update(overrides)
    return Derived(**fields)


def make_state(
    *,
    kind: Kind = "full",
    band: Band = "720-780",
    code_confidence: Confidence = "medium",
    lenders: Sequence[GatedLender] | None = None,
    numbers: dict[str, LenderNumbers] | None = None,
    **overrides: Any,
) -> CheckState:
    """A realistic state as it stands when the rank node runs (everything up to ``code_confidence`` has written).

    ``full``: a report with two open accounts, EMI split 12,500 reported + 31,500 estimated, income 60,000.
    ``quick``: a score only, declared figures only. Every PII-ish field is populated with POISON.
    """
    gated = list(lenders) if lenders is not None else [gl("L1"), gl("L2")]
    score_number = None if band == "NTC" else SCORE_NUMBER
    nums = numbers if numbers is not None else {g.id: num(g.id) for g in gated}
    state: CheckState = {
        "kind": kind,
        "check_id": "chk-zq-0042",
        "lc_id": "lc-zq-0042",
        "lead_id": "lead-zq-0042",
        "opportunity_id": "opp-zq-0042",
        "opportunity_stage": "Counselling",
        "pan": PAN,
        "name": LEARNER_NAME,
        "mobile": MOBILE,
        "pan_hash": "9c4e1d7a" * 8,
        "mobile_hash": "5b2f8e0c" * 8,
        "pan_last4": PAN[-4:],
        "mobile_last4": MOBILE[-4:],
        "monthly_income": 60000.0,
        "existing_emi_declared": None,
        "consent": ConsentInfo(given=True, wording_version="v1", call_ref="callref-q7z", ip="203.0.113.77"),
        "identity": identity(),
        "score": ScoreFacts(band=band, score=score_number, is_ntc=band == "NTC", pulled_on=date(2026, 10, 1), source="score"),
        "candidate_pincodes": [PIN_PAN, PIN_BUREAU_A, PIN_BUREAU_B],
        "gated_lenders": gated,
        "lender_numbers": nums,
        "code_confidence": code_confidence,
        "rank_attempts": 0,
        "rank_error": None,
        # bookkeeping the rank seam must never read: gate warnings, the SurePass call audit (client id, bureau endpoint),
        # the admission decision (consent id), the accumulated error lines.
        "gate_warnings": [GATE_WARNING],
        "surepass_calls_made": [
            SurePassCallAudit(
                endpoint=BUREAU_ENDPOINT, http_status=200, success=True, message_code="success", client_id=CLIENT_ID, billed=True
            )
        ],
        "admission": AdmissionDecision(
            admission="new",
            pan_plan="call",
            bureau_plan="call",
            counts_against_quota=True,
            quota_used_today=2,
            quota_limit=5,
            original_consent_id=CONSENT_ID,
        ),
        "errors": [ERROR_LINE],
    }
    if kind == "full":
        state["report"] = report_facts(band=band, score=score_number, is_ntc=band == "NTC")
        state["derived"] = derived(band=band)
    else:
        state["report"] = None
        state["derived"] = derived(band=band, emi=quick_emi(), current_foir=0.0, candidate_pincodes=[PIN_PAN])
    return cast(CheckState, {**state, **overrides})


def gated_of(state: CheckState) -> list[GatedLender]:
    return state.get("gated_lenders", [])


def numbers_of(state: CheckState) -> dict[str, LenderNumbers]:
    return state.get("lender_numbers", {})
