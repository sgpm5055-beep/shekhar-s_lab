"""Pure adapters: RAW SurePass envelope dict -> typed facts (PRD v1.1 sections 8.1-8.3, 8.6; plan K5).

Canonical entry points take the raw envelope dict (what ``Repo.fetch_raw`` returns / ``SurePassResult.raw``), so a
cached object is re-adapted on CURRENT code every time it is reused; the ``*_to_*`` wrappers just hand over ``result.raw``.

* ``identity_from_envelope``      -> ``Identity``    (NO gender, anywhere; strict-bool aadhaar; zip via the pincode regex)
* ``score_facts_from_envelope``   -> ``ScoreFacts``  (reads ``data.credit_score``: works on a Score AND a Report envelope)
* ``report_facts_from_envelope``  -> ``ReportFacts`` (one ``OpenAccount`` per OPEN account; the 8.3 EMI engine's input)

NTC (v1.1): ``is_ntc_value(data.credit_score)``, nothing else. There is NO FCIREXScore fallback and NO NTC-from-failure
branch: the client guarantees ``success: true`` (every ``success: false`` raises there), so an envelope that is not
``success is True`` raises ``ValueError`` here rather than being read as "no file".

TOTALITY: on a success envelope the adapters never raise, whatever the bureau strings contain (junk, NaN, absurd
numbers, wrong types, missing sections). Every helper returns ``None`` / ``0.0`` / ``[]`` instead. Holder DOB and gender
rows (``CAIS_Holder_Details``) are deliberately never read, and ``Account_Number`` / the masked Aadhaar are never carried.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date
from typing import Any, Final, Literal, cast

from eligibility_agent.derive.range import PIN_RE, band_from_score, is_ntc_value
from eligibility_agent.graph.state import Identity, OpenAccount, ReportFacts, ScoreFacts, ScoreSource
from eligibility_agent.surepass.client import REPORT_ENDPOINT, SurePassResult

STALE_AFTER_DAYS: Final[int] = 90  # an account last reported more than this many days before the anchor is stale
RECENT_ADDRESS_WINDOW_DAYS: Final[int] = 365  # bureau pincodes: open accounts reported within this window
RATE_MAX_PCT: Final[float] = 100.0

_YYYYMMDD_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]{8}")
_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ZIP_KEY: Final[str] = "ZIP_Postal_Code_non_normalized"


# --------------------------------------------------------------------------- parsing helpers (total: junk in, None out)


def as_list(value: object) -> list[dict[str, Any]]:
    """A list -> its dict items; a single dict -> ``[dict]`` (XML->JSON single-element collapse); anything else -> ``[]``."""
    if isinstance(value, dict):
        return [cast("dict[str, Any]", value)]
    if isinstance(value, list):
        return [cast("dict[str, Any]", item) for item in cast("list[object]", value) if isinstance(item, dict)]
    return []


def dig(data: Mapping[str, Any] | None, *keys: str) -> Any:
    """Nested ``get``; ``None`` the moment the cursor is not a mapping or a key is missing."""
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cast("Mapping[str, Any]", cursor).get(key)
    return cursor


def parse_amount(value: object) -> float | None:
    """``None`` / bool / junk / non-finite -> ``None``; number -> float; string -> float after stripping and removing
    the digit-grouping commas (``"1,25,000.50"``). The sign is preserved: callers clamp or filter."""
    if value is None or isinstance(value, bool):
        return None
    number: float
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            number = float(text)
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def parse_int_sentinel(value: object, *, zero_is_unknown: bool = False) -> int | None:
    """Base-10 integer (``"024"`` -> 24) or ``None``. With ``zero_is_unknown`` a parsed ``0`` also means "not reported"
    (``Repayment_Tenure "0"``, ``Terms_Duration ""`` / ``"000"``)."""
    if value is None or isinstance(value, bool):
        return None
    number: int
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            number = int(text, 10)
        except ValueError:  # junk, and Python's int-digit-limit guard on absurdly long digit strings
            return None
    if zero_is_unknown and number == 0:
        return None
    return number


def parse_rate(value: object) -> float | None:
    """Annual percentage rate, or ``None`` when absent / non-positive / above 100 ("0" and absurd values mean unknown)."""
    rate = parse_amount(value)
    if rate is None or rate <= 0.0 or rate > RATE_MAX_PCT:
        return None
    return rate


def parse_yyyymmdd(value: object) -> date | None:
    """Bureau ``YYYYMMDD`` (exactly eight ASCII digits forming a real date), else ``None``."""
    text = str(value).strip() if value is not None else ""
    if not _YYYYMMDD_RE.fullmatch(text):
        return None
    try:
        return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def parse_iso_date(value: object) -> date | None:
    """PAN ``dob`` in the exact ``YYYY-MM-DD`` shape, else ``None``."""
    text = value.strip() if isinstance(value, str) else ""
    if not _ISO_DATE_RE.fullmatch(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def resolve_tenure_months(account: Mapping[str, Any]) -> int | None:
    """``Repayment_Tenure`` if positive, else ``Terms_Duration`` if positive (``"0"`` / ``""`` mean unknown).

    ``Terms_Frequency`` other than ``""`` / ``"M"`` means the tenure is not in months: unknown."""
    frequency = str(account.get("Terms_Frequency") or "").strip()
    if frequency not in ("", "M"):
        return None
    for key in ("Repayment_Tenure", "Terms_Duration"):
        months = parse_int_sentinel(account.get(key), zero_is_unknown=True)
        if months is not None and months > 0:
            return months
    return None


def is_open_account(account: Mapping[str, Any]) -> bool:
    """The exact PRD predicate: ``Date_Closed == ""`` OR ``Account_Status == "11"``."""
    closed_on = str(account.get("Date_Closed") or "").strip()
    status = str(account.get("Account_Status") or "").strip()
    return closed_on == "" or status == "11"


# --------------------------------------------------------------------------- envelope plumbing


def _require_success(raw: object) -> None:
    """The ONLY place an adapter raises: the client guarantees ``success: true``, so anything else is a wiring bug."""
    if not isinstance(raw, Mapping) or cast("Mapping[str, Any]", raw).get("success") is not True:
        raise ValueError("adapter received non-success envelope")


def _mapping(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _data(raw: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(raw.get("data"))


def _text(value: object) -> str | None:
    """A non-empty (after strip) string, else ``None``."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


# --------------------------------------------------------------------------- PAN comprehensive -> Identity


def identity_from_envelope(raw: dict[str, Any]) -> Identity:
    """PAN identity. ``pan_valid`` is the status / category gate only (success is guaranteed by the guard above).

    ``aadhaar_linked`` is a STRICT bool (anything else, including a missing key, stays ``None`` so the link warning can
    fire only on a literal ``False``); ``pan_pincode`` passes the pincode regex or is ``None``; ``dob`` is ISO
    ``YYYY-MM-DD``. The masked Aadhaar is never modelled, and there is no gender."""
    _require_success(raw)
    data = _data(raw)
    status = data.get("status")
    category = data.get("category")
    status_text = status if isinstance(status, str) else None
    category_text = category if isinstance(category, str) else None
    linked = data.get("aadhaar_linked")
    pincode = str(dig(data, "address", "zip") or "").strip()
    client_id = data.get("client_id")
    return Identity(
        pan_valid=status_text == "valid" and category_text == "person",
        status=status_text,
        category=category_text,
        full_name=_text(data.get("full_name")),
        dob=parse_iso_date(data.get("dob")),
        aadhaar_linked=linked if isinstance(linked, bool) else None,
        pan_pincode=pincode if PIN_RE.fullmatch(pincode) else None,
        surepass_client_id=client_id if isinstance(client_id, str) and client_id else None,
    )


def pan_to_identity(result: SurePassResult) -> Identity:
    return identity_from_envelope(result.raw)


# --------------------------------------------------------------------------- Experian score / report -> ScoreFacts


def _score_of(data: Mapping[str, Any]) -> tuple[int | None, bool]:
    """``(score, is_ntc)`` from ``data.credit_score``. The int is parsed only when the value is not NTC."""
    credit_score = data.get("credit_score")
    if is_ntc_value(credit_score):
        return None, True
    return int(str(credit_score).strip()), False


def score_facts_from_envelope(raw: dict[str, Any], *, pulled_on: date, source: ScoreSource) -> ScoreFacts:
    """Score facts from a Score envelope OR a Report envelope (both carry ``data.credit_score``): a Report pull
    refreshes the score cache, whose row then points at the report object (``source="report"``)."""
    _require_success(raw)
    score, is_ntc = _score_of(_data(raw))
    return ScoreFacts(band=band_from_score(score), score=score, is_ntc=is_ntc, pulled_on=pulled_on, source=source)


def score_to_score_facts(result: SurePassResult, *, pulled_on: date) -> ScoreFacts:
    """Thin wrapper over ``result.raw``; the source is ``"report"`` when the result came from the Report endpoint."""
    source: Literal["score", "report"] = "report" if result.endpoint == REPORT_ENDPOINT else "score"
    return score_facts_from_envelope(result.raw, pulled_on=pulled_on, source=source)


# --------------------------------------------------------------------------- Experian report -> ReportFacts


def _non_negative_amount(value: object) -> float:
    amount = parse_amount(value)
    return 0.0 if amount is None or amount < 0.0 else amount


def _account_pincodes(account: Mapping[str, Any]) -> list[str]:
    """This account's holder-address zips that pass the pincode regex; input order kept, duplicates dropped."""
    pincodes: list[str] = []
    for address in as_list(account.get("CAIS_Holder_Address_Details")):
        zip_code = str(address.get(_ZIP_KEY) or "").strip()
        if PIN_RE.fullmatch(zip_code) and zip_code not in pincodes:
            pincodes.append(zip_code)
    return pincodes


def _open_account(index: int, account: Mapping[str, Any], anchor: date) -> OpenAccount:
    date_reported = parse_yyyymmdd(account.get("Date_Reported"))
    scheduled = parse_amount(account.get("Scheduled_Monthly_Payment_Amount"))
    portfolio = str(account.get("Portfolio_Type") or "").strip()
    return OpenAccount(
        account_key=f"acct_{index}",  # synthetic; Account_Number is never carried
        portfolio_type="I" if portfolio == "I" else "R" if portfolio == "R" else None,
        account_type=str(account.get("Account_Type") or "").strip(),
        scheduled_emi=scheduled if scheduled is not None and scheduled > 0.0 else None,  # '0' is NOT a reported EMI
        current_balance=_non_negative_amount(account.get("Current_Balance")),
        tenure_months=resolve_tenure_months(account),
        rate_of_interest=parse_rate(account.get("Rate_of_Interest")),  # model context only
        open_date=parse_yyyymmdd(account.get("Open_Date")),
        date_reported=date_reported,
        amount_past_due=_non_negative_amount(account.get("Amount_Past_Due")),
        pincodes=_account_pincodes(account),
        is_stale=date_reported is None or (anchor - date_reported).days > STALE_AFTER_DAYS,
    )


def _enquiry_count(summary: Mapping[str, Any], key: str) -> int:
    count = parse_int_sentinel(summary.get(key))
    return 0 if count is None or count < 0 else count


def report_facts_from_envelope(raw: dict[str, Any], *, today: date | None = None) -> ReportFacts:
    """Report facts. The anchor for staleness and the 12-month pincode window is ``CreditProfileHeader.ReportDate``,
    else ``today``, else ``date.today()``. An NTC report (credit_score not three digits) has no accounts and zeroed
    aggregates. ``past_due_total`` and ``written_off_or_settled`` span ALL accounts; ``open_accounts`` and
    ``bureau_pincodes_12m`` come from OPEN accounts only. Holder DOB rows are never read."""
    _require_success(raw)
    data = _data(raw)
    report = _mapping(data.get("credit_report"))
    report_date = parse_yyyymmdd(dig(report, "CreditProfileHeader", "ReportDate"))
    anchor = report_date or today or date.today()
    score, is_ntc = _score_of(data)
    if is_ntc:
        return ReportFacts(
            band=band_from_score(None),
            score=None,
            is_ntc=True,
            report_date=report_date,
            open_accounts=[],
            past_due_total=0.0,
            written_off_or_settled=0,
            enquiries_30d=0,
            enquiries_90d=0,
            enquiries_180d=0,
            bureau_pincodes_12m=[],
        )
    accounts = as_list(dig(report, "CAIS_Account", "CAIS_Account_DETAILS"))
    open_accounts = [
        _open_account(index, account, anchor)
        for index, account in enumerate(a for a in accounts if is_open_account(a))
    ]
    summary = _mapping(report.get("TotalCAPS_Summary"))
    recent_pincodes = {
        pincode
        for account in open_accounts
        if account.date_reported is not None and (anchor - account.date_reported).days <= RECENT_ADDRESS_WINDOW_DAYS
        for pincode in account.pincodes
    }
    return ReportFacts(
        band=band_from_score(score),
        score=score,
        is_ntc=False,
        report_date=report_date,
        open_accounts=open_accounts,
        past_due_total=sum((_non_negative_amount(a.get("Amount_Past_Due")) for a in accounts), start=0.0),
        written_off_or_settled=sum(1 for a in accounts if str(a.get("Written_off_Settled_Status") or "").strip() != ""),
        enquiries_30d=_enquiry_count(summary, "TotalCAPSLast30Days"),
        enquiries_90d=_enquiry_count(summary, "TotalCAPSLast90Days"),
        enquiries_180d=_enquiry_count(summary, "TotalCAPSLast180Days"),
        bureau_pincodes_12m=sorted(recent_pincodes),
    )


def report_to_report_facts(result: SurePassResult, *, today: date | None = None) -> ReportFacts:
    return report_facts_from_envelope(result.raw, today=today)
