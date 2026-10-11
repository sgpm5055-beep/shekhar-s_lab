"""Canned ``FakeSurePassClient`` factories: one function per named world, shared by tests and eval datasets.

Every factory builds FRESH envelopes on each call (no shared mutable state between tests) and keys them by the PAN the
world is about, so a call for any other PAN raises ``KeyError`` (a wiring bug, not a silent default). Envelope dicts
and ``(status, dict)`` pairs resolve through the SAME ``interpret_response`` the live client uses; exception scenarios
are logged to the sink BEFORE they are raised, exactly like live. Pass ``sink=`` (a ``MemoryRepo`` or any
``CallLogSink``) to record the audit rows.

Endpoint wiring is deliberate: quick worlds configure PAN + Score; full worlds configure PAN + Report (a Report pull
also refreshes the score cache, so a full check never calls Score). An unconfigured endpoint raises ``AssertionError``
("unexpected SurePass call"), which is how cache-hit and no-bureau-after-invalid-PAN tests prove a call did not happen.

All PANs, names, zips, balances and dates below are synthetic.
"""

from __future__ import annotations

from typing import Any, Final, Literal, TypeAlias

from eligibility_agent.surepass.client import (
    SCORE_ENDPOINT,
    CallLogSink,
    FakeSurePassClient,
    Scenario,
    SurePassTimeout,
)
from eligibility_agent.testing import envelopes as env

PAN_OK: Final[str] = "ABCPE1234F"
PAN_NTC: Final[str] = "KLMPN4321Q"
PAN_COMPANY: Final[str] = "AAACA9999Z"
PAN_INVALID: Final[str] = "ZZZPZ0000Z"

# The report every full world is anchored on: ReportDate 2026-10-01.
REPORT_DATE: Final[str] = "20261001"

NtcVariant: TypeAlias = Literal["empty", "text", "missing", "four_digit"]
ScoreFailureVariant: TypeAlias = Literal["success_false", "http_500", "auth_403", "timeout"]
PanInvalidKind: TypeAlias = Literal["status_invalid", "company", "failure_envelope"]

_NTC_KWARGS: Final[dict[NtcVariant, dict[str, Any]]] = {
    "empty": {"credit_score": ""},
    "text": {"credit_score": "NH"},
    "missing": {"omit_score": True},
    "four_digit": {"credit_score": "0745"},
}


def happy_quick_scenario(
    pan: str = PAN_OK, score: str = "745", zip_code: str = "560001", *, sink: CallLogSink | None = None
) -> FakeSurePassClient:
    """Valid person PAN + a three-digit score. The quick-check happy world."""
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan, zip_code=zip_code)},
        score={pan: env.experian_score_envelope(credit_score=score)},
        sink=sink,
    )


def ntc_scenario(variant: NtcVariant, pan: str = PAN_NTC, *, sink: CallLogSink | None = None) -> FakeSurePassClient:
    """Valid PAN, then a Score that is HTTP 200 + success:true with a non-three-digit ``credit_score``.

    All four variants pin the ``^[0-9]{3}$`` rule: ``''`` (empty), ``'NH'`` (text), key absent (missing), ``'0745'``."""
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan)},
        score={pan: env.experian_score_envelope(**_NTC_KWARGS[variant])},
        sink=sink,
    )


def score_failure_scenario(
    variant: ScoreFailureVariant, pan: str = PAN_OK, *, sink: CallLogSink | None = None
) -> FakeSurePassClient:
    """Valid PAN, then a Score that FAILS. Every variant is a service failure (SUREPASS_FAILURE), never NTC:

    ``success_false`` -> HTTP 200 + success:false (SurePassUpstreamError); ``http_500`` -> SurePassUpstreamError;
    ``auth_403`` -> SurePassAuthError; ``timeout`` -> a ``SurePassTimeout`` instance (the bureau-timeout lock path).
    """
    failure: Scenario
    if variant == "success_false":
        failure = env.experian_score_envelope(success=False)
    elif variant == "http_500":
        failure = env.failure_envelope(http_status=500, message_code="internal_error", message="Internal Server Error")
    elif variant == "auth_403":
        failure = env.failure_envelope(http_status=403, message_code="ip_not_whitelisted", message="IP not whitelisted")
    else:
        failure = SurePassTimeout(SCORE_ENDPOINT, "timeout: ReadTimeout")
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan)},
        score={pan: failure},
        sink=sink,
    )


def pan_invalid_scenario(
    kind: PanInvalidKind, pan: str | None = None, *, sink: CallLogSink | None = None
) -> FakeSurePassClient:
    """PAN endpoint only: no bureau endpoint is configured, so any bureau call after it raises AssertionError.

    ``status_invalid`` -> success:true with ``status='invalid'``; ``company`` -> success:true with
    ``category='company'`` (both: ``Identity.pan_valid`` False -> ``pan_not_verified``). ``failure_envelope`` ->
    HTTP 422 + success:false, which under v1.1 is a SERVICE FAILURE (SurePassUpstreamError), NOT an invalid PAN; it
    stays here as the regression vector for that reversal.
    """
    key = pan if pan is not None else (PAN_COMPANY if kind == "company" else PAN_INVALID)
    if kind == "status_invalid":
        pair = env.pan_comprehensive_envelope(pan=key, status="invalid")
    elif kind == "company":
        pair = env.pan_comprehensive_envelope(pan=key, category="company")
    else:
        pair = env.failure_envelope(http_status=422, message_code="invalid_id", message="Invalid PAN")
    return FakeSurePassClient(pan={key: pair}, sink=sink)


def _full_world(
    pan: str,
    *,
    pan_zip: str,
    accounts: list[dict[str, Any]],
    score: str = "745",
    sink: CallLogSink | None,
    with_score: bool = False,
) -> FakeSurePassClient:
    report = env.experian_report_envelope(accounts=accounts, credit_score=score, report_date=REPORT_DATE)
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan, zip_code=pan_zip)},
        score={pan: env.experian_score_envelope(credit_score=score, report_date=REPORT_DATE)} if with_score else None,
        report={pan: report},
        sink=sink,
    )


def full_happy_reported_emi_scenario(
    pan: str = PAN_OK, *, sink: CallLogSink | None = None, with_score: bool = False
) -> FakeSurePassClient:
    """ONE open account with a REPORTED EMI ``"3500"`` (Date_Reported 20260920), ReportDate 20261001, zips 560001,
    score ``"745"``. Drives the HIGH-confidence full-check test. ``with_score=True`` also wires the Score endpoint
    (same score, same ReportDate) for quick-then-full flows."""
    account = env.cais_account(
        scheduled_emi="3500",
        current_balance="150000",
        date_reported="20260920",
        holder_zip="560001",
        account_number="SYNTHACC0001",
    )
    return _full_world(pan, pan_zip="560001", accounts=[account], sink=sink, with_score=with_score)


def full_emi_four_branch_scenario(pan: str = PAN_OK, *, sink: CallLogSink | None = None) -> FakeSurePassClient:
    """Six OPEN accounts, ReportDate 20261001 (anchor); PAN zip 560001; score ``"745"``. Per PRD 8.3:

    * a1  EMI ``"3500"``, Date_Reported 20260920                       -> reported 3500
    * a2  balance 120000, Terms_Duration ``"048"``, Open_Date 20240115  -> 120000 / (48 - 33) = 8000 (tenure_remaining)
    * a3  balance 60000,  Terms_Duration ``"024"``, Open_Date 20240115  -> remaining 24 - 33 < 1: 60000 / 12 = 5000 (past tenure)
    * a4  Portfolio_Type ``"R"``, balance 40000, no tenure             -> 5% = 2000 (revolving)
    * a5  Portfolio_Type ``"I"``, balance 90000, no tenure             -> 90000 / 36 = 2500 (balance_36)
    * a6  EMI ``"9999"``, Date_Reported 20260601 (122 days: STALE)     -> excluded from every sum

    reported 3500 + estimated 17500 = derived 21000 over the non-stale rows.
    """
    accounts = [
        env.cais_account(
            account_type="05", scheduled_emi="3500", current_balance="150000", date_reported="20260920",
            account_number="SYNTHACC0001",
        ),
        env.cais_account(
            account_type="05", current_balance="120000", terms_duration="048", open_date="20240115",
            date_reported="20260915", account_number="SYNTHACC0002",
        ),
        env.cais_account(
            account_type="05", current_balance="60000", terms_duration="024", open_date="20240115",
            date_reported="20260915", account_number="SYNTHACC0003",
        ),
        env.cais_account(
            account_type="10", portfolio_type="R", current_balance="40000", open_date="20250601",
            date_reported="20260915", account_number="SYNTHACC0004",
        ),
        env.cais_account(
            account_type="06", portfolio_type="I", current_balance="90000", open_date="20250601",
            date_reported="20260915", account_number="SYNTHACC0005",
        ),
        env.cais_account(
            account_type="05", scheduled_emi="9999", current_balance="70000", date_reported="20260601",
            account_number="SYNTHACC0006",
        ),
    ]
    return _full_world(pan, pan_zip="560001", accounts=accounts, sink=sink)


def full_pincode_union_scenario(pan: str = PAN_OK, *, sink: CallLogSink | None = None) -> FakeSurePassClient:
    """PAN zip 560002. Two OPEN accounts, ReportDate 20261001:

    * a1  Date_Reported 20260901, holder zips ``110001`` (kept) and ``"20"`` (junk, filtered)
    * a2  Date_Reported 20250901 (more than 12 months old), holder zip ``400001`` -> excluded from the 12-month set

    Expected union: PAN 560002 + bureau [110001] -> [110001, 560002].
    """
    accounts = [
        env.cais_account(
            scheduled_emi="5000", current_balance="200000", date_reported="20260901", holder_zips=("110001", "20"),
            account_number="SYNTHACC0001",
        ),
        env.cais_account(
            scheduled_emi="2500", current_balance="80000", date_reported="20250901", holder_zips=("400001",),
            account_number="SYNTHACC0002",
        ),
    ]
    return _full_world(pan, pan_zip="560002", accounts=accounts, sink=sink)


def no_pincode_scenario(pan: str = PAN_OK, *, sink: CallLogSink | None = None) -> FakeSurePassClient:
    """PAN ``address.zip`` is ``""`` -> NO_PINCODE_ON_PAN path. The bureau side carries no pincode either (holder zip
    ``""``), so a FULL check also ends with zero candidate pincodes. Score and Report are both configured."""
    account = env.cais_account(
        scheduled_emi="4000", current_balance="100000", date_reported="20260920", holder_zip="",
        account_number="SYNTHACC0001",
    )
    return _full_world(pan, pan_zip="", accounts=[account], sink=sink, with_score=True)
