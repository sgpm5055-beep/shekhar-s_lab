"""Synthetic SurePass envelope builders: the ONLY source of SurePass-shaped data for tests and eval datasets.

Pure dict builders (no pydantic) so they stay byte-faithful to the wire shape: every bureau leaf is a STRING, dense keys
with ``''`` meaning "not reported", dates ``YYYYMMDD`` (PAN ``dob`` is ``YYYY-MM-DD``), validation failures carry NO
``success`` / ``data`` keys. Every value here is INVENTED: nothing is copied from the real probe files (the
probe-literal guard in ``tests/test_envelope_builders.py`` scans every builder output and this package's source).

NTC (v1.1): NTC is HTTP 200 + ``success: true`` + a ``credit_score`` that is not a three-digit string. Vectors:
``credit_score=""`` / ``"NH"`` / ``"0745"`` and ``omit_score=True``. ``success=False`` builds the SERVICE-FAILURE vector
(HTTP 200, ``success: false``, ``message_code: "no_record_found"``): the client raises on it, it is never NTC.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Final, TypeAlias

EnvelopePair: TypeAlias = tuple[int, dict[str, Any]]  # (http_status, body)
HistoryRow: TypeAlias = tuple[str, str, str]  # (Year, Month, Days_Past_Due)

# The 23 scalar field-map keys every CAIS account row carries (dense: present on every row, '' = not reported) ...
CAIS_SCALAR_KEYS: Final[tuple[str, ...]] = (
    "Account_Number",
    "Account_Type",
    "Account_Status",
    "Portfolio_Type",
    "Date_Closed",
    "Open_Date",
    "Date_Reported",
    "Date_of_Last_Payment",
    "DateOfAddition",
    "Scheduled_Monthly_Payment_Amount",
    "Current_Balance",
    "Amount_Past_Due",
    "Highest_Credit_or_Original_Loan_Amount",
    "Credit_Limit_Amount",
    "Repayment_Tenure",
    "Terms_Duration",
    "Terms_Frequency",
    "Rate_of_Interest",
    "Payment_History_Profile",
    "Payment_Rating",
    "Written_off_Settled_Status",
    "Settlement_Amount",
    "SuitFiled_WilfulDefault",
)
# ... plus these four nested collections.
CAIS_NESTED_KEYS: Final[tuple[str, ...]] = (
    "CAIS_Account_History",
    "CAIS_Holder_Address_Details",
    "CAIS_Holder_Phone_Details",
    "CAIS_Holder_Details",
)

# Deliberately CONTRADICTORY holder DOBs: any test that sees one leak into an age proves a rule violation (the real
# per-account holder rows disagree with each other; age comes from the PAN record only).
_DEFAULT_HOLDER_DOB: Final[str] = "19800101"
_CONTRADICTORY_HOLDER_DOB: Final[str] = "19900101"


# --------------------------------------------------------------------------- envelope wrappers


def _ok_envelope(data: dict[str, Any], message_code: str = "success") -> EnvelopePair:
    # message is deliberately None: business code must never branch on the message text.
    return (200, {"data": data, "status_code": 200, "success": True, "message": None, "message_code": message_code})


def failure_envelope(
    *, http_status: int = 422, message_code: str = "invalid_id", message: str | None = "Invalid PAN"
) -> EnvelopePair:
    """A ``success: false`` envelope. Under v1.1 the client raises on EVERY one of these (service failure)."""
    return (
        http_status,
        {"data": {}, "status_code": http_status, "success": False, "message": message, "message_code": message_code},
    )


def validation_error_envelope(field: str = "id_number", msg: str = "Missing required parameter.") -> EnvelopePair:
    # Field-map rule: validation failures have NO success / data keys.
    return (422, {"errors": {field: msg}, "message": "Input payload validation failed"})


def _no_record_found() -> EnvelopePair:
    return failure_envelope(http_status=200, message_code="no_record_found", message="No record found")


# --------------------------------------------------------------------------- PAN comprehensive


def pan_comprehensive_envelope(
    *,
    pan: str = "ABCPE1234F",
    full_name: str = "RAVI TESTKUMAR",
    status: str = "valid",
    category: str = "person",
    gender: str = "M",
    dob: str = "1998-04-12",
    aadhaar_linked: bool | None = True,
    zip_code: str = "560001",
    client_id: str = "pan_comprehensive_synthetic_0001",
) -> EnvelopePair:
    """``aadhaar_linked=None`` OMITS the key (exercises the strict-bool rule). ``gender`` exists only because real
    payloads carry the key; the service never reads, derives or sends a gender."""
    data: dict[str, Any] = {
        "client_id": client_id,
        "pan_number": pan,
        "full_name": full_name,
        "full_name_split": full_name.split(" "),
        "masked_aadhaar": "XXXXXXXX0000",
        "dob": dob,
        "gender": gender,
        "status": status,
        "category": category,
        "input_dob": None,
        "dob_verified": True,
        "dob_check": False,
        "less_info": False,
        "phone_number": "",
        "email": "",
        "address": {
            "line_1": "12 Synthetic Lane",
            "line_2": "",
            "street_name": "",
            "zip": zip_code,
            "city": "Testpur",
            "state": "Karnataka",
            "country": "",
            "full": "12 Synthetic Lane, Testpur",
        },
    }
    if aadhaar_linked is not None:
        data["aadhaar_linked"] = aadhaar_linked
    return _ok_envelope(data)


# --------------------------------------------------------------------------- Experian score


def _score_block(credit_score: str, omit_score: bool, report_date: str | None) -> dict[str, Any]:
    header: dict[str, Any] = {} if report_date is None else {"ReportDate": report_date}
    score: dict[str, Any] = {"FCIREXScoreConfidLevel": ""}
    if not omit_score:
        score["FCIREXScore"] = credit_score
    return {"CreditProfileHeader": header, "SCORE": score}


def experian_score_envelope(
    *,
    credit_score: str = "745",
    report_date: str | None = "20261001",
    success: bool = True,
    omit_score: bool = False,
    client_id: str = "experian_score_synthetic_0001",
) -> EnvelopePair:
    """``success=False`` -> the service-failure vector (200, success:false, ``no_record_found``): never NTC.
    NTC vectors are success:true with ``credit_score`` ``""`` / ``"NH"`` / ``"0745"`` or ``omit_score=True``."""
    if not success:
        return _no_record_found()
    credit_report = _score_block(credit_score, omit_score, report_date)
    credit_report["Match_result"] = {"Exact_match": "Y"}
    data: dict[str, Any] = {"client_id": client_id, "credit_report": credit_report}
    if not omit_score:
        data["credit_score"] = credit_score
    return _ok_envelope(data)


# --------------------------------------------------------------------------- Experian report


def cais_account(
    *,
    account_type: str = "05",  # numeric code strings; '10' is the card
    account_status: str = "11",  # '11' open, '13' closed, '15' closed
    date_closed: str = "",  # '' = open; YYYYMMDD when closed
    scheduled_emi: str = "",  # '' = not reported: the NORM on Experian
    current_balance: str = "120000",
    amount_past_due: str = "0",
    repayment_tenure: str = "0",  # '0' = not reported
    terms_duration: str = "",  # zero-padded like '024', or ''
    terms_frequency: str = "M",
    rate_of_interest: str = "",  # '13.500'-style string or ''
    open_date: str = "20240115",
    date_reported: str = "20260901",
    date_of_last_payment: str = "20260825",
    written_off_settled_status: str = "",  # '99' when written off / settled
    settlement_amount: str = "",
    payment_rating: str = "0",
    payment_history_profile: str = "000000000000",
    history: Sequence[HistoryRow] = (("2026", "09", "0"),),
    holder_zip: str = "560001",  # junk like '20' is allowed for filter tests
    holder_zips: Sequence[str] | None = None,  # one address row per value; overrides ``holder_zip``
    holder_dob: str = _DEFAULT_HOLDER_DOB,  # trap: the adapters never read it
    highest_credit: str = "200000",
    credit_limit: str = "",
    account_number: str = "SYNTHACC0001",
    portfolio_type: str = "I",
) -> dict[str, Any]:
    """A DENSE account dict: all 23 scalar field-map keys present on every account (``''`` = not reported), plus the
    four nested collections. ``CAIS_Holder_Details`` carries TWO rows with contradictory DOBs on purpose."""
    zips = (holder_zip,) if holder_zips is None else tuple(holder_zips)
    return {
        "Account_Number": account_number,
        "Account_Type": account_type,
        "Account_Status": account_status,
        "Portfolio_Type": portfolio_type,
        "Date_Closed": date_closed,
        "Open_Date": open_date,
        "Date_Reported": date_reported,
        "Date_of_Last_Payment": date_of_last_payment,
        "DateOfAddition": date_reported,
        "Scheduled_Monthly_Payment_Amount": scheduled_emi,
        "Current_Balance": current_balance,
        "Amount_Past_Due": amount_past_due,
        "Highest_Credit_or_Original_Loan_Amount": highest_credit,
        "Credit_Limit_Amount": credit_limit,
        "Repayment_Tenure": repayment_tenure,
        "Terms_Duration": terms_duration,
        "Terms_Frequency": terms_frequency,
        "Rate_of_Interest": rate_of_interest,
        "Payment_History_Profile": payment_history_profile,
        "Payment_Rating": payment_rating,
        "Written_off_Settled_Status": written_off_settled_status,
        "Settlement_Amount": settlement_amount,
        "SuitFiled_WilfulDefault": "",
        "CAIS_Account_History": [
            {"Year": y, "Month": m, "Days_Past_Due": dpd, "Asset_Classification": "?"} for (y, m, dpd) in history
        ],
        "CAIS_Holder_Address_Details": [
            {
                "First_Line_Of_Address_non_normalized": "12 SYNTHETIC LANE",
                "City_non_normalized": "TESTPUR",
                "State_non_normalized": "29",
                "ZIP_Postal_Code_non_normalized": z,
                "Address_indicator_non_normalized": "",
                "Residence_code_non_normalized": "",
            }
            for z in zips
        ],
        "CAIS_Holder_Phone_Details": [{"Telephone_Number": "", "Mobile_Telephone_Number": "", "EMailId": ""}],
        "CAIS_Holder_Details": [
            {"Date_of_birth": holder_dob, "Gender_Code": "1"},
            {"Date_of_birth": _CONTRADICTORY_HOLDER_DOB, "Gender_Code": "1"},
        ],
    }


def experian_report_envelope(
    *,
    accounts: Sequence[dict[str, Any]] = (),
    credit_score: str = "745",
    report_date: str | None = "20261001",
    caps_30: str = "0",
    caps_90: str = "0",
    caps_180: str = "0",
    success: bool = True,
    omit_score: bool = False,
    client_id: str = "experian_report_synthetic_0001",
) -> EnvelopePair:
    """The full report envelope. ``report_date=None`` omits ``CreditProfileHeader.ReportDate`` (anchor-fallback tests).
    The score lives at ``data.credit_score`` exactly as on a Score envelope, so score facts read from either."""
    if not success:
        return _no_record_found()
    rows = [copy.deepcopy(a) for a in accounts]
    credit_report = _score_block(credit_score, omit_score, report_date)
    credit_report.update(
        {
            "Current_Application": {"Current_Application_Details": {"Income": "0", "Current_Applicant_Address_Details": {}}},
            "CAIS_Account": {
                "CAIS_Summary": {"Credit_Account": {"CreditAccountTotal": str(len(rows))}},
                "CAIS_Account_DETAILS": rows,
            },
            "TotalCAPS_Summary": {
                "TotalCAPSLast7Days": "0",
                "TotalCAPSLast30Days": caps_30,
                "TotalCAPSLast90Days": caps_90,
                "TotalCAPSLast180Days": caps_180,
            },
            "Match_result": {"Exact_match": "Y"},
        }
    )
    data: dict[str, Any] = {"client_id": client_id, "credit_report": credit_report}
    if not omit_score:
        data["credit_score"] = credit_score
    return _ok_envelope(data)
