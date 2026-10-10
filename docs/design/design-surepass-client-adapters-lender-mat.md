# SurePass client, adapters, lender math (PRD section 8 — eligibility_agent/tools/)

## Summary
This section designs the three tools modules plus a shared synthetic-fixtures module. tools/surepass.py delivers a sync SurePassClient Protocol with per-call check_id, a LiveSurePassClient over httpx (bearer auth, exact FIXED request bodies, 5s/60s timeouts, zero retries because a timed-out call has unknown spend), a call-time SUREPASS_LIVE kill switch raising LiveCallsDisabled before any I/O, a pure two-shape envelope parser (success envelope vs {errors,message} validation failure) with a status+success+message_code error taxonomy, client-owned audit logging through a narrow CallLogSink protocol (exactly one surepass_calls row per HTTP attempt, zero rows for kill-switch raises), and a scenario-driven FakeSurePassClient that runs injected raw envelopes through the same interpret_response code path and exposes an ordered call log. tools/adapters.py pins every string-convention parser (parse_amount, parse_int_sentinel with zero_is_unknown, parse_rate, parse_score clamped to [300,900], parse_yyyymmdd, as_list, dig) and implements pan_to_identity, score_to_bureau_score (success:false OR empty/unparseable score → NTC, marked ASSUMPTION with a NON_BUREAU_FAILURE_CODES override hook), and report_to_bureau_report implementing every PRD 8.2 rule, including the 24-month DPD window anchored to CreditProfileHeader.ReportDate and the 365-day bureau-pincode recency filter. Band bucketing is left-closed (<650 | [650,720) | [720,780) | ≥780 | NTC, so 650→"650-720", 720→"720-780", 780→"780+") with NTC ordered lowest for min_band cutoffs. tools/lenders.py provides gate_lenders over a LenderSource protocol (active lenders joined to lender_pincodes, deterministic ordering, malformed hard_cutoffs skipped with a warning), check_hard_cutoffs with a literal decision table where a cutoff fails only on a known violating fact, and the closed-form reducing-balance EMI math: EMI = P·r·(1+r)^n/((1+r)^n−1), inverse P = EMI·(1−(1+r)^−n)/r, r=0 branches, defaults 36 months/14% p.a., half-up rounding to the nearest 1,000, with machine-verified worked examples (EMI(300000,14,36)=10,253.29; ceiling for income 60k, EMI 8k, FOIR 0.55 = 731,000). The most critical files are eligibility_agent/tools/surepass.py, eligibility_agent/tools/adapters.py, eligibility_agent/tools/lenders.py, eligibility_agent/tools/fixtures.py, and eligibility_agent/graph/state.py (consumed model contract). All designs were verified against /Users/shekhargupta/HV Downloads/HV Documents/FulCRM Loan Approval Engine/SurePass_Field_Map_and_Reconciliation.md and API Tested Guide.md; no probe values are reproduced anywhere.

## Files
### eligibility_agent/tools/__init__.py
_Package marker for tools/._

Empty file (no re-exports, so mypy --strict sees one canonical import path per symbol).

### eligibility_agent/tools/surepass.py
_SurePass KYC client: constants, error taxonomy, SurePassResult, envelope parsing for both shapes, SUREPASS_LIVE kill switch, audit-log hook, LiveSurePassClient (httpx), FakeSurePassClient with inspectable call log._

from __future__ import annotations; imports: dataclasses, typing (Any, Final, Protocol, Union), httpx.

== Constants (FIXED) ==
SUREPASS_BASE_URL: Final[str] = "https://kyc-api.surepass.app"
EP_PAN_COMPREHENSIVE: Final[str] = "/api/v1/pan/pan-comprehensive"
EP_EXPERIAN_SCORE: Final[str] = "/api/v1/credit-report-experian/score"
EP_EXPERIAN_REPORT: Final[str] = "/api/v1/credit-report-experian/fetch-report"
CONSENT_Y: Final[str] = "Y"
CONNECT_TIMEOUT_S: Final[float] = 5.0
READ_TIMEOUT_S: Final[float] = 60.0   # Experian report bodies are large
# success:false message_codes that mean infra/billing failure, NOT a bureau no-hit.
# Deliberately empty until a real code is observed (field-map R10, PRD open item 4). When a
# code lands in this set, interpret_response raises SurePassUpstreamError instead of returning,
# so the adapter can never cache that run as NTC.
NON_BUREAU_FAILURE_CODES: Final[frozenset[str]] = frozenset()

== Exceptions (taxonomy) ==
class SurePassError(Exception):
    """Base. Messages never contain PAN/mobile/name or body content."""
    def __init__(self, endpoint: str, detail: str, http_status: int | None = None) -> None:
        super().__init__(f"{endpoint}: {detail}"); self.endpoint = endpoint; self.detail = detail; self.http_status = http_status
class LiveCallsDisabled(SurePassError): ...     # kill switch; raised BEFORE any I/O; NOT logged to surepass_calls
class SurePassAuthError(SurePassError): ...     # HTTP 401/403 (observed codes: ip_not_whitelisted, auth_failed, unauthorized) or empty token
class SurePassRateLimited(SurePassError): ...   # HTTP 429
class SurePassUpstreamError(SurePassError): ... # HTTP >= 500, or message_code in NON_BUREAU_FAILURE_CODES
class SurePassTransportError(SurePassError): ...# httpx.HTTPError (DNS/conn/read timeout). Spend status UNKNOWN -> never auto-retried
class SurePassProtocolError(SurePassError): ... # non-JSON / non-dict body (e.g. 404 HTML page), or dict matching neither envelope shape
class SurePassValidationError(SurePassError):   # {"errors": {...}, "message": "Input payload validation failed"} shape (NO success/data keys)
    def __init__(self, endpoint: str, errors: dict[str, str], message: str, http_status: int | None = None) -> None:
        super().__init__(endpoint, "request validation failed", http_status); self.errors = errors; self.message = message

== Typed result ==
@dataclass(frozen=True, slots=True)
class SurePassResult:
    endpoint: str
    http_status: int
    success: bool                      # always a real bool: validation-failure shape raises instead
    envelope_status_code: int | None   # body["status_code"] when int
    message: str | None                # audit only; business code NEVER branches on it (FIXED)
    message_code: str | None
    data: dict[str, Any] | None        # body["data"] when dict, else None
    raw: dict[str, Any]                # exact parsed JSON body -> pan_cache.payload; never to the LLM or LcView
    @property
    def client_id(self) -> str | None:  # data["client_id"] if a non-empty str, else None

== Audit-log hook ==
class CallLogSink(Protocol):           # structurally satisfied by db.repo.Repo (Supabase + Memory)
    def insert_surepass_call(self, *, check_id: str, endpoint: str, surepass_client_id: str | None,
                             http_status: int | None, success: bool | None, message_code: str | None) -> None: ...
# Contract: exactly ONE row per HTTP attempt (returned OR raised-after-response; http_status=None for pure
# transport failure). LiveCallsDisabled / empty-token raises happen before I/O and write NO row. The sink
# receives only the five audit fields above -- never request bodies, names, PANs or response data
# (matches API-guide logging guidance). Sink exceptions propagate (strict audit, fail loudly).

== Client protocol (DI seam; graph nodes depend on this, never on the concrete class) ==
class SurePassClient(Protocol):
    @property
    def calls_enabled(self) -> bool: ...   # admission refuses LIVE_CALLS_DISABLED without attempting a call when False
    def pan_comprehensive(self, check_id: str, pan: str) -> SurePassResult: ...
    def experian_score(self, check_id: str, *, name: str, mobile: str, pan: str) -> SurePassResult: ...
    def experian_report(self, check_id: str, *, name: str, mobile: str, pan: str) -> SurePassResult: ...
# check_id is a per-call argument (NOT constructor state): the client is a process-lifetime singleton
# injected at graph-build time, and explicit args beat ContextVar magic for tests.

== Pure parsing (shared by live client, fake client, and cache reprocessing) ==
def parse_surepass_body(endpoint: str, http_status: int, body: object) -> SurePassResult:
    # 1. not isinstance(body, dict) -> raise SurePassProtocolError(endpoint, "non-dict body", http_status)
    # 2. "success" in body and isinstance(body["success"], bool) -> build SurePassResult:
    #       envelope_status_code = body.get("status_code") if isinstance(int, not bool) else None
    #       message = body.get("message") if isinstance(str) else None; message_code likewise
    #       data = body.get("data") if isinstance(dict) else None; raw = body
    # 3. "errors" in body or ("message" in body and "data" not in body) ->
    #       raise SurePassValidationError(endpoint, {str(k): str(v) for k, v in (body.get("errors") or {}).items()},
    #                                     str(body.get("message") or ""), http_status)   # branch on SHAPE, never message text
    # 4. else raise SurePassProtocolError(endpoint, "unrecognized envelope", http_status)

def interpret_response(endpoint: str, http_status: int, body: object) -> SurePassResult:
    # Status taxonomy evaluated IN ORDER (branching on HTTP status + success + message_code only -- FIXED):
    # | condition                                   | action                                             |
    # | http_status in (401, 403)                   | raise SurePassAuthError                            |
    # | http_status == 429                          | raise SurePassRateLimited                          |
    # | http_status >= 500                          | raise SurePassUpstreamError                        |
    # | otherwise                                   | result = parse_surepass_body(...)                  |
    # | result.message_code in NON_BUREAU_FAILURE_CODES | raise SurePassUpstreamError                    |
    # | else                                        | return result  (covers 200 AND 4xx success:false   |
    # |                                             |  envelopes: PAN-invalid and score-no-hit stay typed)|

== Live client ==
class LiveSurePassClient:
    def __init__(self, *, token: str, live: bool, sink: CallLogSink,
                 base_url: str = SUREPASS_BASE_URL, http: httpx.Client | None = None) -> None:
        # live comes from Settings.surepass_live (os.environ.get("SUREPASS_LIVE") == "true", exact literal).
        # self._http = http or httpx.Client(base_url=base_url, timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S))
        # remember _owns_http for close()
    @property
    def calls_enabled(self) -> bool: return self._live and bool(self._token)
    def close(self) -> None: ...  # closes the httpx.Client only if owned
    def pan_comprehensive(self, check_id: str, pan: str) -> SurePassResult:
        return self._post(check_id, EP_PAN_COMPREHENSIVE, {"id_number": pan, "get_address": True, "get_contact": False})
    def experian_score(self, check_id: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._post(check_id, EP_EXPERIAN_SCORE, {"name": name, "mobile": mobile, "pan": pan, "consent": CONSENT_Y})
    def experian_report(self, check_id: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._post(check_id, EP_EXPERIAN_REPORT, {"name": name, "mobile": mobile, "pan": pan, "consent": CONSENT_Y})
        # consent "Y" is always sent even though Experian does not enforce it (field-map R15); our consents row is the real gate.
    def _post(self, check_id: str, endpoint: str, body: dict[str, Any]) -> SurePassResult:
        # 1. if not self._live: raise LiveCallsDisabled(endpoint, "SUREPASS_LIVE != 'true'")   # strictly before any I/O or client use
        # 2. if not self._token: raise SurePassAuthError(endpoint, "missing SurePass token")   # before I/O; no log row
        # 3. try: resp = self._http.post(endpoint, json=body, headers={"Authorization": f"Bearer {self._token}"})
        #    except httpx.HTTPError as exc:
        #        self._sink.insert_surepass_call(check_id=check_id, endpoint=endpoint, surepass_client_id=None,
        #                                        http_status=None, success=None, message_code=None)
        #        raise SurePassTransportError(endpoint, f"transport: {type(exc).__name__}") from exc   # NO retry: spend unknown
        # 4. status = resp.status_code; parsed: object = None
        #    try: parsed = resp.json()
        #    except ValueError: parsed = None   # 404-with-HTML case from the API guide
        # 5. try: result = interpret_response(endpoint, status, parsed if parsed is not None else resp.text)
        #    except SurePassError as exc:
        #        self._sink.insert_surepass_call(check_id=check_id, endpoint=endpoint,
        #            surepass_client_id=_peek_client_id(parsed), http_status=status,
        #            success=_peek_success(parsed), message_code=_peek_message_code(parsed))
        #        raise
        # 6. self._sink.insert_surepass_call(check_id=check_id, endpoint=endpoint, surepass_client_id=result.client_id,
        #                                    http_status=status, success=result.success, message_code=result.message_code)
        #    return result
    # Module-private peek helpers (total functions, None on any shape mismatch):
    # def _peek_success(parsed: object) -> bool | None; def _peek_message_code(parsed: object) -> str | None;
    # def _peek_client_id(parsed: object) -> str | None   # parsed["data"]["client_id"]

== Fake client (tests + evals; no network, ever) ==
Scenario = Union[SurePassResult, SurePassError, dict[str, Any], tuple[int, dict[str, Any]]]
#  dict            -> interpret_response(endpoint, 200, dict)   (same parsing path as live)
#  (status, dict)  -> interpret_response(endpoint, status, dict) (exercises 401/429/5xx/422 taxonomy)
#  SurePassResult  -> returned as-is
#  SurePassError   -> raised after recording the attempt
@dataclass(frozen=True, slots=True)
class FakeCall:
    endpoint: str
    check_id: str
    body: dict[str, Any]   # in-memory only, for request-shape assertions; never persisted
class FakeSurePassClient:
    def __init__(self, *, pan: Scenario | None = None, score: Scenario | None = None, report: Scenario | None = None,
                 sink: CallLogSink | None = None, calls_enabled: bool = True) -> None: ...
    calls: list[FakeCall]                      # ordered; the eval policy_consistency source of truth
    @property
    def calls_enabled(self) -> bool: ...
    def endpoints_called(self) -> list[str]:   # [c.endpoint for c in self.calls]
    # Each protocol method builds the SAME body dict as the live client, then _respond(endpoint, check_id, body, scenario):
    #   1. if not self._calls_enabled: raise LiveCallsDisabled(endpoint, ...)      # mirrors live: nothing recorded/logged
    #   2. if scenario is None: raise AssertionError(f"unexpected SurePass call: {endpoint}")  # cache-hit/cap tests rely on this
    #   3. append FakeCall; resolve scenario per the table above; log one row to sink (if provided) with the same
    #      fields/ordering rules as the live client (on raised SurePassError: peeked values; on result: result fields); return/raise.

mypy --strict: every def annotated; dict bodies typed dict[str, Any]; no Any leaks in public signatures.

### eligibility_agent/tools/adapters.py
_Pure adapters SurePassResult -> Identity/BureauScore/BureauReport plus every string-convention parsing helper, band bucketing and band ordering._

from __future__ import annotations; imports: re, datetime (date), typing (Any, Final, Literal, Mapping, Sequence), graph.state models (Identity, BureauScore, BureauReport, OpenAccountMissingEmi, Band), tools.surepass (SurePassResult).

== Shared constants ==
PIN_RE: Final[re.Pattern[str]] = re.compile(r"^[1-9][0-9]{5}$")   # single source; lenders.py imports it
BAND_ORDER: Final[dict[Band, int]] = {"NTC": 0, "<650": 1, "650-720": 2, "720-780": 3, "780+": 4}  # NTC sorts LOWEST
SCORE_MIN: Final[int] = 300; SCORE_MAX: Final[int] = 900
RECENT_ADDRESS_WINDOW_DAYS: Final[int] = 365
DPD_WINDOW_MONTHS: Final[int] = 24

== Band bucketing (FIXED buckets; boundary membership pinned here) ==
def band_from_score(score: int | None) -> Band:
    # None -> "NTC"; score < 650 -> "<650"; 650 <= score < 720 -> "650-720";
    # 720 <= score < 780 -> "720-780"; score >= 780 -> "780+"
    # i.e. left-closed/right-open: 650 -> "650-720", 720 -> "720-780", 780 -> "780+".

== Parsing helpers (total functions: junk in, None out; never raise on bureau strings) ==
def as_list(v: object) -> list[dict[str, Any]]:
    # list -> items that are dicts; single dict -> [dict] (defensive: XML->JSON single-element collapse); else []
def dig(d: Mapping[str, Any] | None, *keys: str) -> Any:
    # nested get; returns None the moment the cursor is not a Mapping
def parse_amount(v: object) -> float | None:
    # None -> None; bool -> None; int/float -> float(v); str: strip, remove ","; "" -> None; float() else None
def parse_int_sentinel(v: object, *, zero_is_unknown: bool = False) -> int | None:
    # None -> None; s = str(v).strip(); "" -> None; int(s, 10) (so "024" -> 24; int() has no octal trap) else None;
    # if zero_is_unknown and n == 0 -> None  (Repayment_Tenure "0" and Terms_Duration ""/"000" mean not reported)
def parse_rate(v: object) -> float | None:
    # parse_amount; then None if f is None or f <= 0.0 or f > 100.0  ("34.900" -> 34.9; 0/absurd -> unknown -> default 14%)
def parse_score(v: object) -> int | None:
    # bool -> None; try n = int(str(v).strip()) except -> None; return n if SCORE_MIN <= n <= SCORE_MAX else None
    # ("-1"/"0" no-hit sentinels fall out as None -> NTC, aligned with field-map R10)
def parse_yyyymmdd(v: object) -> date | None:
    # s = str(v).strip(); len 8 and isdigit required; date(y, m, d) guarded by ValueError -> None
def parse_iso_date(v: object) -> date | None:
    # for PAN dob "YYYY-MM-DD": date.fromisoformat guarded -> None
def age_years(dob: date, on: date) -> int:
    # completed years: on.year - dob.year - (1 if (on.month, on.day) < (dob.month, dob.day) else 0); floored at 0
def resolve_tenure_months(a: Mapping[str, Any]) -> int | None:
    # freq = str(a.get("Terms_Frequency") or "").strip(); if freq not in ("", "M"): return None  # non-monthly -> unknown
    # for key in ("Repayment_Tenure", "Terms_Duration"):  # PRD order
    #     v = parse_int_sentinel(a.get(key), zero_is_unknown=True)
    #     if v is not None and v > 0: return v
    # return None
def _map_gender(v: object) -> Literal["male", "female"] | None:
    # "M" -> "male", "F" -> "female", anything else -> None (explicit ifs so mypy narrows the Literal)

== PAN Comprehensive -> Identity ==
def pan_to_identity(result: SurePassResult) -> Identity:
    d = result.data or {}
    status = d.get("status") if isinstance str else None; category likewise
    pan_valid = result.success is True and status == "valid" and category == "person"   # the ONLY gate (policy rule 6)
    full_name = d["full_name"] if non-empty str else None
    gender = _map_gender(d.get("gender"))            # internal only; Experian requests never send gender
    dob = parse_iso_date(d.get("dob"))
    v = d.get("aadhaar_linked"); aadhaar_linked = v if isinstance(v, bool) else None
    #   STRICT bool: missing/None stays None so LcView.pan_aadhaar_link_issue fires ONLY on literal False (policy rule 10)
    z = str(dig(d, "address", "zip") or "").strip(); pan_pincode = z if PIN_RE.fullmatch(z) else None
    surepass_client_id = result.client_id
    return Identity(pan_valid=..., status=status, category=category, full_name=..., gender=..., dob=...,
                    aadhaar_linked=..., pan_pincode=..., surepass_client_id=...)
    # masked_aadhaar is NEVER read into Identity; it survives only inside result.raw (-> pan_cache.payload).
    # Adapter is total: success:false / wrong category still return Identity(pan_valid=False, ...).

== Experian score -> BureauScore ==
def score_to_bureau_score(result: SurePassResult) -> BureauScore:
    # if result.success is False:   # ASSUMPTION(R10): NTC shape unverified; success:false treated as bureau no-hit.
    #     return BureauScore(score=None, is_ntc=True, band="NTC", report_date=None, ntc_reason=result.message_code or "success_false")
    # d = result.data or {}
    # score = parse_score(d.get("credit_score")) or-else parse_score(dig(d, "credit_report", "SCORE", "FCIREXScore"))
    #   (explicit two-step: fallback runs only when the first parse is None)
    # report_date = parse_yyyymmdd(dig(d, "credit_report", "CreditProfileHeader", "ReportDate"))   # YYYYMMDD
    # return BureauScore(score=score, is_ntc=score is None, band=band_from_score(score), report_date=report_date,
    #                    ntc_reason=("score_unparseable" if score is None else None))

== Experian report -> BureauReport (every PRD 8.2 rule) ==
def is_open_account(a: Mapping[str, Any]) -> bool:
    # str(a.get("Date_Closed") or "").strip() == "" or str(a.get("Account_Status") or "").strip() == "11"   # exact PRD OR
def report_to_bureau_report(result: SurePassResult, *, today: date | None = None) -> BureauReport:
    # if result.success is False: return BureauReport(band="NTC", score=None, is_ntc=True, report_date=None,
    #     existing_emi_reported=0.0, existing_emi_floor=False, open_accounts_missing_emi=[], open_account_count=0,
    #     total_outstanding=0.0, past_due_total=0.0, max_dpd_24m=0, written_off_or_settled=0, enquiries_180d=0,
    #     bureau_pincodes=[])   # ASSUMPTION(R10), same marker as the score branch
    # d = result.data or {}; cr = d.get("credit_report") if dict else {}
    # A = as_list(dig(d, "credit_report", "CAIS_Account", "CAIS_Account_DETAILS"))   # dense dicts, "" = not reported
    # score = parse_score(d.get("credit_score")) fallback parse_score(dig(cr, "SCORE", "FCIREXScore"))
    # report_date = parse_yyyymmdd(dig(cr, "CreditProfileHeader", "ReportDate"))
    # anchor: date = report_date or today or date.today()   # tests always pass today= for determinism
    # open_accounts = [a for a in A if is_open_account(a)]
    # emi_reported = 0.0; floor = False; missing: list[OpenAccountMissingEmi] = []
    # for a in open_accounts:
    #     v = parse_amount(a.get("Scheduled_Monthly_Payment_Amount"))
    #     if v is None:   # "" or junk -> not reported (probe: 100% of open accounts)
    #         floor = True
    #         missing.append(OpenAccountMissingEmi(balance=parse_amount(a.get("Current_Balance")) or 0.0,
    #             tenure_months=resolve_tenure_months(a), rate_annual_pct=parse_rate(a.get("Rate_of_Interest")),
    #             account_type=str(a.get("Account_Type") or "")))
    #     else: emi_reported += v   # "0" parses to 0.0 and counts as reported
    # total_outstanding = sum(parse_amount(a.get("Current_Balance")) or 0.0 for a in open_accounts)
    # past_due_total   = sum(parse_amount(a.get("Amount_Past_Due")) or 0.0 for a in A)      # ALL accounts
    # written_off_or_settled = sum(1 for a in A if str(a.get("Written_off_Settled_Status") or "").strip() != "")
    # max_dpd_24m: over ALL accounts (closed delinquency still matters), for h in as_list(a.get("CAIS_Account_History")):
    #     y = parse_int_sentinel(h.get("Year")); m = parse_int_sentinel(h.get("Month"))
    #     if y is None or m is None or not 1 <= m <= 12: continue
    #     months_ago = (anchor.year - y) * 12 + (anchor.month - m)
    #     if not 0 <= months_ago <= DPD_WINDOW_MONTHS - 1: continue   # 24-month window; anchor month = month 0
    #     dpd = parse_int_sentinel(h.get("Days_Past_Due"))            # "" / non-numeric rows skipped, "0" is a real 0
    #     if dpd is not None: candidates.append(dpd)
    #   max_dpd_24m = max(candidates, default=0)
    # bureau_pincodes: pins: set[str] = set(); for a in A:
    #     dr = parse_yyyymmdd(a.get("Date_Reported"))
    #     if dr is None or (anchor - dr).days > RECENT_ADDRESS_WINDOW_DAYS: continue   # unparseable = stale; future-dated kept
    #     for addr in as_list(a.get("CAIS_Holder_Address_Details")):
    #         z = str(addr.get("ZIP_Postal_Code_non_normalized") or "").strip()
    #         if PIN_RE.fullmatch(z): pins.add(z)          # junk like "20" filtered here
    #   bureau_pincodes = sorted(pins)
    # enquiries_180d = parse_int_sentinel(dig(cr, "TotalCAPS_Summary", "TotalCAPSLast180Days")) or 0
    # DOB: deliberately NOT read -- CAIS_Holder_Details rows contradict each other (field-map R17); age comes from Identity.dob only.
    # return BureauReport(band=band_from_score(score), score=score, is_ntc=score is None, report_date=report_date,
    #     existing_emi_reported=emi_reported, existing_emi_floor=floor, open_accounts_missing_emi=missing,
    #     open_account_count=len(open_accounts), total_outstanding=..., past_due_total=..., max_dpd_24m=...,
    #     written_off_or_settled=..., enquiries_180d=..., bureau_pincodes=...)

### eligibility_agent/tools/lenders.py
_Pincode gate over the lender tables, hard-cutoff checks, reducing-balance EMI math (estimator + loan ceiling) with pinned defaults, formulas and worked examples._

from __future__ import annotations; imports: dataclasses, typing (Any, Final, Iterable, Protocol, Sequence, TypedDict), graph.state (Applicant, GatedLender, HardCutoffs, OpenAccountMissingEmi), tools.adapters (BAND_ORDER, PIN_RE).

== Constants (ASSUMPTION per PRD 8.3/8.4; PRD open item 2) ==
DEFAULT_TENURE_MONTHS: Final[int] = 36
DEFAULT_ANNUAL_RATE_PCT: Final[float] = 14.0
FOIR_EPS: Final[float] = 1e-9   # float guard on max_foir comparison

== Core annuity math (shared by estimator and ceiling) ==
def monthly_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    # r = annual_rate_pct / 100.0 / 12.0; n = tenure_months if tenure_months > 0 else DEFAULT_TENURE_MONTHS (defensive)
    # r == 0.0 -> principal / n
    # else f = (1.0 + r) ** n; return principal * r * f / (f - 1.0)
    # Worked example (machine-verified): monthly_emi(300_000, 14.0, 36) = 10_253.2889...  ((1+0.14/12)^36 = 1.5182660)
    # r=0 branch: monthly_emi(300_000, 0.0, 36) = 8_333.33...
def principal_for_emi(emi: float, annual_rate_pct: float, tenure_months: int) -> float:
    # inverse closed form: r = annual_rate_pct/100/12; n as above
    # r == 0.0 -> emi * n          (principal_for_emi(25_000, 0.0, 36) = 900_000.0)
    # else return emi * (1.0 - (1.0 + r) ** (-n)) / r
    # Annuity factor AF(14%, 36m) = (1 - 1.0116667^-36) / 0.0116667 = 29.2589044
    # Round-trip property (test): principal_for_emi(monthly_emi(P, r, n), r, n) == pytest.approx(P)
def round_to_nearest_1000(value: float) -> float:
    # half-up, NOT banker's rounding: float(int(value / 1000.0 + 0.5) * 1000); value is always >= 0 here

== EMI estimation for unreported open accounts (PRD 8.3, ASSUMPTION) ==
def estimate_existing_emi(rows: Sequence[OpenAccountMissingEmi]) -> float:
    # total = 0.0
    # for row in rows:
    #     if row.balance <= 0: continue
    #     rate = row.rate_annual_pct if (row.rate_annual_pct is not None and row.rate_annual_pct > 0) else DEFAULT_ANNUAL_RATE_PCT
    #     tenure = row.tenure_months if (row.tenure_months is not None and row.tenure_months > 0) else DEFAULT_TENURE_MONTHS
    #     total += monthly_emi(row.balance, rate, tenure)
    # return total
    # Worked example: one open account, balance 300_000, tenure unknown, rate unknown -> 10_253.29/month.
    # This is the NORM on real data (probe: EMI empty on 100% of accounts). The affordability node keeps the
    # estimate separate from the reported sum and sets the floor flag so the LC wording says "confirm existing EMIs".

== Pincode gate (policy rule 9; the LC can never override it) ==
class LenderRow(TypedDict):
    id: str
    name: str
    criteria: dict[str, Any]
    hard_cutoffs: dict[str, Any]
    matched_pincodes: list[str]
class LenderSource(Protocol):   # structurally satisfied by db.repo.Repo (Supabase + Memory)
    def lenders_for_pincodes(self, pincodes: Sequence[str]) -> list[LenderRow]: ...
def gate_lenders(source: LenderSource, pincodes: Iterable[str]) -> tuple[list[GatedLender], list[str]]:
    # pins = sorted({str(p).strip() for p in pincodes if PIN_RE.fullmatch(str(p).strip())})   # defensive re-filter
    # if not pins: return ([], [])          # no query; empty gate is allowed and lowers confidence downstream
    # rows = source.lenders_for_pincodes(pins)
    # lenders: list[GatedLender] = []; warnings: list[str] = []
    # for row in rows:
    #     try: hc = HardCutoffs.model_validate(row["hard_cutoffs"])   # extra="ignore" on the model
    #     except ValidationError: warnings.append(f"lender {row['id']} skipped: invalid hard_cutoffs"); continue
    #     lenders.append(GatedLender(id=row["id"], name=row["name"], criteria=row["criteria"],
    #                                hard_cutoffs=hc, matched_pincodes=sorted(set(row["matched_pincodes"]))))
    # return (sorted(lenders, key=lambda l: l.id), warnings)   # deterministic for tests/evals
    # Repo semantics (reference SQL; supabase-py two-step or RPC is the db section's choice):
    #   SELECT l.id, l.name, l.criteria, l.hard_cutoffs, array_agg(DISTINCT lp.pincode) AS matched_pincodes
    #   FROM lenders l JOIN lender_pincodes lp ON lp.lender_id = l.id
    #   WHERE l.active = true AND lp.pincode = ANY(:pins) GROUP BY l.id, l.name, l.criteria, l.hard_cutoffs ORDER BY l.id;

== Hard cutoffs (enforced in code; the LLM never sees this logic, only its results) ==
@dataclass(frozen=True, slots=True)
class CutoffOutcome:
    passed: bool               # True iff failed == []  (unknown facts NEVER fail -- they lower confidence)
    failed: list[str]          # cutoff key names, e.g. ["min_band", "max_foir"]
    unknown: list[str]         # cutoff key names skipped because the applicant fact is unknown
    details: dict[str, str]    # internal-only wording; MAY contain numbers; must never reach LcView
def check_hard_cutoffs(lender: GatedLender, applicant: Applicant) -> CutoffOutcome:
    # c = lender.hard_cutoffs. Decision table (a key absent/None on the lender = no constraint):
    # | key                 | fact            | unknown when                       | FAILS when (fact known)                                      |
    # | min_band            | applicant.band  | never (band always set post-pull;  | BAND_ORDER[band] < BAND_ORDER[c.min_band]                    |
    # |                     |                 | NTC is a KNOWN value, rank 0)      |   (so NTC fails any min_band above "NTC")                    |
    # | max_foir            | applicant.foir  | foir is None (income or EMI unknown)| foir > c.max_foir + FOIR_EPS                                 |
    # | min_income          | monthly_income  | income is None                     | monthly_income < c.min_income                                |
    # | min_age             | age_years       | age_years is None (dob missing)    | age_years < c.min_age                                        |
    # | max_age_at_maturity | age_years       | age_years is None                  | age_years*12 + (c.max_tenure_months or DEFAULT_TENURE_MONTHS)|
    # |                     |                 |                                    |   > c.max_age_at_maturity*12   (integer months, no float fuzz)|
    # | max_tenure_months   | --              | n/a                                | never fails an applicant: ceiling/maturity parameter only    |
    # | annual_rate_pct     | --              | n/a                                | never fails: ceiling parameter only                          |
    # existing_emi_floor does NOT fail max_foir (the computed foir is the known fact; floor-ness lowers confidence via prompts).
    # details example: {"max_foir": "foir 0.62 > cap 0.55"} -- internal only.

== Loan ceiling (PRD 8.4, ASSUMPTION) ==
def loan_ceiling(applicant: Applicant, lender: GatedLender) -> float | None:
    # c = lender.hard_cutoffs
    # if applicant.monthly_income is None or applicant.monthly_income <= 0: return None   # "Null when income is unknown"
    # if c.max_foir is None: return None   # DECISION: no PRD default FOIR exists; refuse to invent one
    # emi = applicant.existing_emi if applicant.existing_emi is not None else 0.0   # unknown EMI -> 0; floor flag carries doubt
    # headroom = c.max_foir * applicant.monthly_income - emi
    # if headroom <= 0: return 0.0
    # rate = c.annual_rate_pct if c.annual_rate_pct is not None else DEFAULT_ANNUAL_RATE_PCT
    # n = c.max_tenure_months if c.max_tenure_months is not None else DEFAULT_TENURE_MONTHS
    # return round_to_nearest_1000(principal_for_emi(headroom, rate, n))
    # Worked example (machine-verified): income 60_000, existing_emi 8_000, max_foir 0.55, rate/tenure absent ->
    #   headroom = 0.55*60_000 - 8_000 = 25_000; P = 25_000 * 29.2589044 = 731_472.61 -> 731_000.0
    # Edge: income 20_000, emi 11_000, max_foir 0.55 -> headroom 0 -> 0.0. r=0 lender: P = headroom * n.

### eligibility_agent/tools/fixtures.py
_Synthetic SurePass envelope builders shared by tests/ and evals/datasets.py. 100% invented values shaped per the field map; NEVER copy anything from surepass-lab/out/bureau._

from __future__ import annotations; imports: typing (Any, Literal, Sequence). Pure dict builders (no pydantic) so they stay byte-faithful to the wire shape: all bureau leaves are strings, dense keys, "" = not reported, dates YYYYMMDD.

def success_envelope(data: dict[str, Any], *, status_code: int = 200, message: str | None = "Success",
                     message_code: str = "success") -> dict[str, Any]:
    # {"data": data, "status_code": status_code, "success": True, "message": message, "message_code": message_code}
def failure_envelope(*, message_code: str, status_code: int = 422, data: dict[str, Any] | None = None,
                     message: str | None = None) -> dict[str, Any]:
    # same keys with "success": False
def validation_failure_body(field_errors: dict[str, str]) -> dict[str, Any]:
    # {"errors": dict(field_errors), "message": "Input payload validation failed"}   # NO success/data keys

def pan_envelope(*, status: str = "valid", category: str = "person", full_name: str = "TEST LEARNER",
                 gender: str = "M", dob: str = "1998-04-12", aadhaar_linked: bool | None = True,
                 zip_code: str = "110001", client_id: str = "pan_comprehensive_test0001",
                 masked_aadhaar: str = "XXXXXXXX9999") -> dict[str, Any]:
    # data mirrors the API-guide sample: pan_number "ABCDE1234F", full_name_split, address{line_1..full, zip},
    # phone_number "98XXXXXX10", email "", input_dob None, dob_verified/dob_check/less_info False, aadhaar_linked
    # (key OMITTED when arg is None -- exercises the strict-bool adapter rule), category, status, masked_aadhaar.
# Convenience wrappers: pan_invalid_envelope() -> pan_envelope(status="invalid"); pan_non_person_envelope() ->
# pan_envelope(category="company"); pan_unlinked_envelope() -> pan_envelope(aadhaar_linked=False).

def score_envelope(*, score: str = "736", report_date: str = "20261001",
                   client_id: str = "credit_report_experian_test0001") -> dict[str, Any]:
    # data = {client_id, credit_score: score, credit_report: {SCORE: {FCIREXScore: score, FCIREXScoreConfidLevel: ""},
    #         CreditProfileHeader: {ReportDate: report_date}, Match_result: {Exact_match: "Y"}}} wrapped in success_envelope
def score_ntc_envelope(variant: Literal["success_false", "empty", "null", "junk"]) -> dict[str, Any]:
    # ASSUMPTION(R10) -- shape unverified; all four variants MUST map to NTC:
    # "success_false" -> failure_envelope(message_code="no_record_found", data={"client_id": ...})
    # "empty" -> score_envelope with credit_score "" and FCIREXScore ""; "null" -> credit_score None, no SCORE key;
    # "junk" -> credit_score "-1"

def synthetic_account(*, account_type: str = "05", account_status: str = "11", date_closed: str = "",
        scheduled_emi: str = "", current_balance: str = "250000", amount_past_due: str = "0",
        repayment_tenure: str = "0", terms_duration: str = "024", terms_frequency: str = "M",
        rate_of_interest: str = "", open_date: str = "20240101", date_reported: str = "20260930",
        written_off_status: str = "", zip_codes: Sequence[str] = ("110001",),
        history: Sequence[tuple[str, str, str]] = ()) -> dict[str, Any]:
    # Returns a DENSE account dict (every key present, exact field-map spellings): Scheduled_Monthly_Payment_Amount,
    # Date_Closed, Account_Status, Account_Type, Portfolio_Type ("R" iff account_type == "10" else ""), Current_Balance,
    # Amount_Past_Due, Highest_Credit_or_Original_Loan_Amount, Credit_Limit_Amount, Repayment_Tenure, Terms_Duration,
    # Terms_Frequency, Rate_of_Interest, Open_Date, Date_Reported, Date_of_Last_Payment, DateOfAddition,
    # Payment_History_Profile ("00" * 18), Payment_Rating, Written_off_Settled_Status, Settlement_Amount,
    # SuitFiled_WilfulDefault, Account_Number ("XXXXACCT" + counter),
    # CAIS_Account_History: [{Year, Month, Days_Past_Due, Asset_Classification: "01"} for (y, m, dpd) in history],
    # CAIS_Holder_Address_Details: [{First_Line_Of_Address_non_normalized: "12 TEST ROAD", City_non_normalized: "TESTVILLE",
    #     State_non_normalized: "07", ZIP_Postal_Code_non_normalized: z, Address_indicator_non_normalized: "",
    #     Residence_code_non_normalized: ""} for z in zip_codes],
    # CAIS_Holder_Phone_Details: [{Telephone_Number: "", Mobile_Telephone_Number: "9800000000", EMailId: ""}],
    # CAIS_Holder_Details: [{Date_of_birth: "19800101", Gender_Code: "1"}, {Date_of_birth: "19900101", Gender_Code: "1"}]
    #   -- two DELIBERATELY contradictory DOBs so any test that sees 1980/1990 leak into age proves a rule violation.

def report_envelope(*, score: str = "736", report_date: str = "20261001",
                    accounts: Sequence[dict[str, Any]] | None = None, caps_180: str = "0",
                    client_id: str = "credit_report_experian_test0002") -> dict[str, Any]:
    # data = {client_id, credit_score: score, credit_report: {
    #   CreditProfileHeader: {ReportDate: report_date}, SCORE: {FCIREXScore: score, FCIREXScoreConfidLevel: ""},
    #   Current_Application: {Current_Application_Details: {Income: "0", Current_Applicant_Address_Details: {}}},
    #   CAIS_Account: {CAIS_Summary: {Credit_Account: {CreditAccountTotal: str(len), CreditAccountActive: ..., CreditAccountDefault: "0",
    #       CreditAccountClosed: ...}, Total_Outstanding_Balance: {Outstanding_Balance_All: "0", Outstanding_Balance_Secured: "0",
    #       Outstanding_Balance_UnSecured: "0"}}, CAIS_Account_DETAILS: list(accounts or [synthetic_account()])},
    #   TotalCAPS_Summary: {TotalCAPSLast7Days: "0", TotalCAPSLast30Days: "0", TotalCAPSLast90Days: "0", TotalCAPSLast180Days: caps_180},
    #   CAPS: {CAPS_Application_Details: []}, Match_result: {Exact_match: "Y"}}} wrapped in success_envelope
# Canned scenarios used by tests and evals/datasets.py:
# report_with_junk_and_stale_pincodes() -> report_envelope(report_date="20261001", accounts=[
#   synthetic_account(zip_codes=("20", "110001"), date_reported="20260901"),      # junk "20" filtered, 110001 kept
#   synthetic_account(zip_codes=("560001",), date_reported="20240101", account_status="13", date_closed="20240601")])
#   -> expected bureau_pincodes == ["110001"]
# report_all_emis_missing(n_open: int = 3) and report_with_delinquency(history=(("2026", "08", "45"), ...)).

### tests/tools/test_surepass_client.py
_Unit tests for kill switch, envelope parsing, error taxonomy, audit logging, fake call log (PRD test 12 + taxonomy coverage). No network anywhere._

Uses FakeSurePassClient + a MemorySink (local list-appending CallLogSink test double, or db.repo.MemoryRepo once available). Cases:
1. test_kill_switch_blocks_every_endpoint: LiveSurePassClient(token="t", live=False, sink=sink, http=httpx.Client(transport=httpx.MockTransport(fail-if-called))) -> all three methods raise LiveCallsDisabled; sink has 0 rows; transport never invoked (PRD test 12).
2. test_live_true_but_empty_token_raises_auth_before_io.
3. test_request_bodies_exact: FakeSurePassClient with scenarios; after pan_comprehensive(check_id, "ABCDE1234F") assert calls[0].body == {"id_number": "ABCDE1234F", "get_address": True, "get_contact": False}; score/report bodies == {"name", "mobile", "pan", "consent": "Y"} exactly.
4. test_parse_success_envelope / test_parse_success_false_envelope_returned_not_raised (fixture failure_envelope -> SurePassResult.success is False).
5. test_validation_failure_shape_raises: parse_surepass_body(EP, 422, validation_failure_body({"id_number": "Missing data..."})) raises SurePassValidationError carrying the errors dict.
6. test_status_taxonomy: interpret_response raises SurePassAuthError on 401/403, SurePassRateLimited on 429, SurePassUpstreamError on 500; SurePassProtocolError on non-dict body (HTML string) and on unrecognized dict.
7. test_success_false_envelope_on_http_422_still_returns_typed_result.
8. test_live_client_logs_one_row_per_attempt: MockTransport returning pan_envelope -> exactly one sink row (endpoint, client_id from data, 200, True, "success"); MockTransport raising httpx.ConnectError -> one row with http_status None then SurePassTransportError.
9. test_fake_unconfigured_endpoint_asserts: calling score with score=None scenario raises AssertionError.
10. test_fake_calls_enabled_false_raises_LiveCallsDisabled_and_records_nothing.
11. test_fake_logs_to_sink_like_live: dict scenario -> sink row fields match the returned SurePassResult.
12. test_non_bureau_failure_codes_hook: monkeypatched frozenset({"insufficient_credits"}) -> interpret_response raises SurePassUpstreamError on a success:false envelope with that message_code.

### tests/tools/test_adapters.py
_Adapter unit tests (PRD test 13): PAN gate, score NTC variants, report derivations (open predicate, EMI floor, pincode filter, DPD window, band boundaries). All inputs from tools/fixtures.py._

All report tests pass a fixed today=date(2026, 10, 8) where the anchor fallback is exercised; normally anchor = ReportDate.
1. test_pan_valid_person -> Identity(pan_valid=True, gender="male", dob=date(1998,4,12), aadhaar_linked True, pan_pincode "110001", surepass_client_id set).
2. test_pan_invalid_status / test_pan_non_person_category -> pan_valid False (fields still parsed).
3. test_pan_zip_junk -> zip "0411"/"20"/"" -> pan_pincode None; test_pan_aadhaar_link_missing_key -> aadhaar_linked None (NOT False); linked False stays False.
4. test_identity_never_contains_masked_aadhaar: model_dump() of Identity from a pan_envelope has no value containing "XXXXXXXX9999".
5. test_score_hit -> score 736, band "720-780", is_ntc False, report_date date(2026,10,1).
6. test_score_ntc_all_variants (parametrized "success_false"/"empty"/"null"/"junk") -> is_ntc True, band "NTC", score None.
7. test_score_fallback_to_fcirex: credit_score "" but FCIREXScore "736" -> 736.
8. test_band_boundaries (parametrized): 649->"<650", 650->"650-720", 719->"650-720", 720->"720-780", 779->"720-780", 780->"780+", 900->"780+", 300->"<650"; parse_score: "-1"->None, "0"->None, "901"->None, ""->None, " 736 "->736.
9. test_open_predicate: Date_Closed "" OR Account_Status "11" is open; status "13"/"15" with Date_Closed set is closed; status "13" with Date_Closed "" still OPEN (OR semantics, exact PRD rule).
10. test_emi_reported_sum_and_floor: mixed accounts (one EMI "8000", one "") -> existing_emi_reported 8000.0, floor True, one missing row with tenure 24 (from "024"), rate None.
11. test_all_emis_empty_probe_norm: report_all_emis_missing(3) -> reported 0.0, floor True, 3 missing rows (field-map R1: the NORM).
12. test_tenure_resolution: Repayment_Tenure "0" + Terms_Duration "024" -> 24; both "0"/"" -> None; Repayment_Tenure "12" wins over Terms_Duration; Terms_Frequency "Q" -> None.
13. test_bureau_pincode_filter: junk "20" dropped, stale (>365d before ReportDate) dropped, dupes deduped, output sorted; unparseable Date_Reported excluded.
14. test_max_dpd_24m_window: history rows ("2026","08","45") inside, ("2023","01","90") outside with ReportDate 20261001 -> 45; "" DPD skipped; no rows -> 0; month 24 months back (months_ago 24) excluded, 23 included.
15. test_past_due_over_all_accounts_and_outstanding_over_open_only; test_written_off_count ("99" -> counted, "" -> not).
16. test_enquiries_180d: "2" -> 2; missing -> 0.
17. test_dob_never_from_holder_rows: synthetic_account's contradictory holder DOBs (1980/1990) appear nowhere in BureauReport output.
18. test_report_success_false_is_ntc (ASSUMPTION branch): zeroed BureauReport with band "NTC".
19. test_single_account_dict_collapse: CAIS_Account_DETAILS given as a single dict (not list) still parses via as_list.

### tests/tools/test_lenders.py
_Unit tests for gate_lenders, check_hard_cutoffs decision table, EMI/ceiling math with the pinned worked examples._

Uses a StubLenderSource(list[LenderRow]) implementing LenderSource.
1. test_emi_formula_pinned: monthly_emi(300_000, 14.0, 36) == pytest.approx(10_253.288927, abs=0.01); monthly_emi(300_000, 0.0, 36) == pytest.approx(8_333.3333).
2. test_principal_inverse_pinned: principal_for_emi(25_000, 14.0, 36) == pytest.approx(731_472.609, abs=0.01); r=0 -> 900_000.0; round-trip property for (P, r, n) grid [(1e5, 10.5, 24), (3e5, 0.0, 36), (12e5, 18.0, 60)].
3. test_round_to_nearest_1000: 731_472.61 -> 731_000.0; 731_500.0 -> 732_000.0 (half-up); 499.9 -> 0.0.
4. test_estimate_existing_emi: defaults applied when tenure/rate None; balance 0 skipped; multi-row sum.
5. test_loan_ceiling_worked_example: Applicant(band="720-780", monthly_income=60_000, existing_emi=8_000, existing_emi_floor=True, age_years=27, foir=8_000/60_000), lender max_foir 0.55, no rate/tenure -> 731_000.0.
6. test_loan_ceiling_edges: income None -> None; income 0 -> None; max_foir None -> None; headroom <= 0 -> 0.0; existing_emi None -> treated as 0; lender rate 0 -> headroom * n.
7. test_check_hard_cutoffs_table (parametrized over the decision table): min_band pass/fail incl. NTC vs min_band "<650" fails and band "650-720" vs min_band "650-720" passes (equality passes); max_foir unknown when foir None; exact-equality foir == max_foir passes (FOIR_EPS); min_income/min_age known-fail vs unknown-skip; max_age_at_maturity: age 27, tenure default 36 -> fails cap 29 (27*12+36 > 348), passes cap 30 (360 <= 360); lender with empty hard_cutoffs {} -> passed=True, no unknowns.
8. test_gate_lenders: pincode defensive filter (junk "20" input dropped); empty pins -> ([], []) and source NOT queried (stub asserts); malformed hard_cutoffs row skipped with warning while good lenders survive; output sorted by lender id; matched_pincodes sorted/deduped.
9. test_unknowns_never_fail: Applicant with income None, emi None, age None vs lender with all seven keys -> passed=True, unknown == ["max_foir", "min_income", "min_age", "max_age_at_maturity"].

## Interfaces provided
- **name**: tools.surepass.SurePassClient (Protocol) + LiveSurePassClient + exceptions | **definition**: class SurePassClient(Protocol): calls_enabled: bool (read-only property); pan_comprehensive(check_id: str, pan: str) -> SurePassResult; experian_score(check_id: str, *, name: str, mobile: str, pan: str) -> SurePassResult; experian_report(check_id: str, *, name: str, mobile: str, pan: str) -> SurePassResult. LiveSurePassClient(*, token: str, live: bool, sink: CallLogSink, base_url: str = "https://kyc-api.surepass.app", http: httpx.Client | None = None); .close(). Exceptions (all subclass SurePassError(endpoint, detail, http_status=None)): LiveCallsDisabled, SurePassAuthError, SurePassRateLimited, SurePassUpstreamError, SurePassTransportError, SurePassProtocolError, SurePassValidationError(errors: dict[str, str], message: str). Constants: SUREPASS_BASE_URL, EP_PAN_COMPREHENSIVE = "/api/v1/pan/pan-comprehensive", EP_EXPERIAN_SCORE = "/api/v1/credit-report-experian/score", EP_EXPERIAN_REPORT = "/api/v1/credit-report-experian/fetch-report", CONSENT_Y = "Y", NON_BUREAU_FAILURE_CODES: frozenset[str].
- **name**: tools.surepass.SurePassResult + parse_surepass_body + interpret_response | **definition**: @dataclass(frozen=True, slots=True) SurePassResult(endpoint: str, http_status: int, success: bool, envelope_status_code: int | None, message: str | None, message_code: str | None, data: dict[str, Any] | None, raw: dict[str, Any]); property client_id -> str | None. parse_surepass_body(endpoint: str, http_status: int, body: object) -> SurePassResult (raises SurePassValidationError / SurePassProtocolError). interpret_response(endpoint: str, http_status: int, body: object) -> SurePassResult adds the 401/403->Auth, 429->RateLimited, >=500->Upstream, NON_BUREAU_FAILURE_CODES->Upstream raises. Graph/db sections: persist result.raw into pan_cache.payload; reprocess a cached payload with parse_surepass_body(endpoint, 200, payload) then the matching adapter.
- **name**: tools.surepass.CallLogSink (Protocol) | **definition**: def insert_surepass_call(self, *, check_id: str, endpoint: str, surepass_client_id: str | None, http_status: int | None, success: bool | None, message_code: str | None) -> None. Exactly one row per HTTP attempt; no row for LiveCallsDisabled/missing-token; db layer stamps called_at. SupabaseRepo and MemoryRepo must satisfy this structurally.
- **name**: tools.surepass.FakeSurePassClient + FakeCall | **definition**: FakeSurePassClient(*, pan: Scenario | None = None, score: Scenario | None = None, report: Scenario | None = None, sink: CallLogSink | None = None, calls_enabled: bool = True) where Scenario = SurePassResult | SurePassError | dict[str, Any] | tuple[int, dict[str, Any]]. Attributes: calls: list[FakeCall(endpoint: str, check_id: str, body: dict[str, Any])]; endpoints_called() -> list[str]. Unconfigured endpoint -> AssertionError; calls_enabled=False -> LiveCallsDisabled with nothing recorded; dict scenarios run through interpret_response (same parsing as live) and are logged to the sink.
- **name**: tools.adapters public API | **definition**: PIN_RE: re.Pattern[str] = ^[1-9][0-9]{5}$; BAND_ORDER: dict[Band, int] = {NTC:0, <650:1, 650-720:2, 720-780:3, 780+:4}; band_from_score(score: int | None) -> Band (left-closed buckets; 650->650-720, 720->720-780, 780->780+); pan_to_identity(result: SurePassResult) -> Identity; score_to_bureau_score(result: SurePassResult) -> BureauScore; report_to_bureau_report(result: SurePassResult, *, today: date | None = None) -> BureauReport; is_open_account(a: Mapping[str, Any]) -> bool; age_years(dob: date, on: date) -> int (completed years); helpers as_list, dig, parse_amount, parse_int_sentinel(v, *, zero_is_unknown: bool = False), parse_rate, parse_score, parse_yyyymmdd, parse_iso_date, resolve_tenure_months. All total functions; adapters never raise on bureau string junk (only on being handed a result the client would have raised for).
- **name**: tools.lenders public API | **definition**: DEFAULT_TENURE_MONTHS = 36; DEFAULT_ANNUAL_RATE_PCT = 14.0; class LenderRow(TypedDict): id, name, criteria: dict[str, Any], hard_cutoffs: dict[str, Any], matched_pincodes: list[str]; class LenderSource(Protocol): lenders_for_pincodes(pincodes: Sequence[str]) -> list[LenderRow]; gate_lenders(source: LenderSource, pincodes: Iterable[str]) -> tuple[list[GatedLender], list[str]] (lenders sorted by id, warnings for skipped malformed rows; empty pins -> no query); @dataclass(frozen=True) CutoffOutcome(passed: bool, failed: list[str], unknown: list[str], details: dict[str, str]); check_hard_cutoffs(lender: GatedLender, applicant: Applicant) -> CutoffOutcome (fails ONLY on known violating facts); monthly_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float; principal_for_emi(emi: float, annual_rate_pct: float, tenure_months: int) -> float; estimate_existing_emi(rows: Sequence[OpenAccountMissingEmi]) -> float; loan_ceiling(applicant: Applicant, lender: GatedLender) -> float | None (nearest-1000 half-up; None when income unknown/<=0 or max_foir absent; 0.0 when headroom <= 0). CutoffOutcome.details is internal-only and may contain numbers: it must NEVER be copied into LcView.
- **name**: tools.fixtures synthetic builders (tests + evals/datasets.py) | **definition**: success_envelope(data, *, status_code=200, message="Success", message_code="success") -> dict; failure_envelope(*, message_code, status_code=422, data=None, message=None) -> dict; validation_failure_body(field_errors) -> dict; pan_envelope(...) + pan_invalid_envelope/pan_non_person_envelope/pan_unlinked_envelope; score_envelope(*, score="736", report_date="20261001", ...); score_ntc_envelope(variant: Literal["success_false","empty","null","junk"]); synthetic_account(... dense Experian account dict with every field-map key ...); report_envelope(*, score, report_date, accounts, caps_180, client_id); canned report_with_junk_and_stale_pincodes(), report_all_emis_missing(n_open=3), report_with_delinquency(history). All values synthetic; nothing sourced from surepass-lab/out/bureau.

## Interfaces consumed
- **name**: graph/state.py Pydantic models (graph section owns the file; these exact shapes are what my adapters emit and my lender math reads) | **expectation**: Band = Literal["<650", "650-720", "720-780", "780+", "NTC"]. class Identity(BaseModel): pan_valid: bool; status: str | None; category: str | None; full_name: str | None; gender: Literal["male","female"] | None; dob: date | None; aadhaar_linked: bool | None; pan_pincode: str | None; surepass_client_id: str | None. class BureauScore(BaseModel): score: int | None; is_ntc: bool; band: Band; report_date: date | None; ntc_reason: str | None. class OpenAccountMissingEmi(BaseModel): balance: float; tenure_months: int | None; rate_annual_pct: float | None; account_type: str. class BureauReport(BaseModel): band: Band; score: int | None; is_ntc: bool; report_date: date | None; existing_emi_reported: float; existing_emi_floor: bool; open_accounts_missing_emi: list[OpenAccountMissingEmi]; open_account_count: int; total_outstanding: float; past_due_total: float; max_dpd_24m: int; written_off_or_settled: int; enquiries_180d: int; bureau_pincodes: list[str]. class HardCutoffs(BaseModel): model_config = ConfigDict(extra="ignore"); min_band: Band | None = None; max_foir: float | None = None; min_income: float | None = None; min_age: int | None = None; max_age_at_maturity: int | None = None; max_tenure_months: int | None = None; annual_rate_pct: float | None = None. class GatedLender(BaseModel): id: str; name: str; criteria: dict[str, Any]; hard_cutoffs: HardCutoffs; matched_pincodes: list[str]. class Applicant(BaseModel): band: Band; monthly_income: float | None; existing_emi: float | None; existing_emi_floor: bool; age_years: int | None; foir: float | None. Identity.score/BureauScore.score/BureauReport fields are internal-only and must never be serialized into LcView.
- **name**: db/repo.py Repo protocol (db section): audit log + lender gate queries | **expectation**: Repo (SupabaseRepo and MemoryRepo) structurally satisfies BOTH tools protocols: (1) insert_surepass_call(*, check_id: str, endpoint: str, surepass_client_id: str | None, http_status: int | None, success: bool | None, message_code: str | None) -> None, stamping called_at; schema.sql must make surepass_calls.surepass_client_id, http_status, success, message_code all NULLable (transport failures log http_status NULL). (2) lenders_for_pincodes(pincodes: Sequence[str]) -> list[LenderRow] returning only active=true lenders having at least one matching pincode, with matched_pincodes aggregated per lender (reference SQL: SELECT l.id, l.name, l.criteria, l.hard_cutoffs, array_agg(DISTINCT lp.pincode) FROM lenders l JOIN lender_pincodes lp ON lp.lender_id = l.id WHERE l.active AND lp.pincode = ANY(:pins) GROUP BY l.id, l.name, l.criteria, l.hard_cutoffs ORDER BY l.id). MemoryRepo must implement both for the no-network test suite.
- **name**: config.py Settings (config/scaffold section) | **expectation**: Settings.surepass_live: bool computed EXACTLY as os.environ.get("SUREPASS_LIVE") == "true" (case-sensitive literal, no strip/lower); Settings.surepass_token: str (default ""); Settings.surepass_base_url: str default "https://kyc-api.surepass.app". main.py wires LiveSurePassClient(token=settings.surepass_token, live=settings.surepass_live, sink=repo, base_url=settings.surepass_base_url). pyproject must declare httpx (>=0.27) and pydantic (>=2.7); no retry/backoff library.
- **name**: graph nodes/build (graph section): DI seam and node usage of tools | **expectation**: Nodes receive SurePassClient and Repo at graph-build time and pass state["check_id"] to every client call. admission: when a billed call would be required (admission in {new, refresh}) and client.calls_enabled is False -> refusal_code LIVE_CALLS_DISABLED without calling. validate_pan: result = client.pan_comprehensive(check_id, pan); identity = pan_to_identity(result); gate on identity.pan_valid; keep result.raw for pan_cache.payload. pull_score/pull_report: name = identity.full_name or request name; score_to_bureau_score / report_to_bureau_report(result, today=IST today); cache derived = model_dump(mode="json"), is_ntc from the model. gate_lenders node: pincodes = {identity.pan_pincode} minus None, union report.bureau_pincodes for Tier 2; lenders, warnings = gate_lenders(repo, pincodes); warnings -> state["errors"]. affordability node: estimated = estimate_existing_emi(report.open_accounts_missing_emi); bureau_emi = report.existing_emi_reported + estimated; existing_emi = max(declared, bureau_emi) if declared is not None else bureau_emi (Tier 1: declared or None); floor = report.existing_emi_floor or (Tier 1 and declared is None); normalize monthly_income <= 0 to None; foir = existing_emi / monthly_income only when both known; age_years = adapters.age_years(identity.dob, ist_today) when dob known; per lender: check_hard_cutoffs(lender, applicant) and loan_ceiling(applicant, lender). Nodes catch SurePassError subclasses -> state["errors"] + their failure routing; SurePassValidationError/ProtocolError are bugs (fail the run as error, do NOT mark pan_invalid or NTC).
- **name**: policy/guards.py (policy section) | **expectation**: Refusal constant LIVE_CALLS_DISABLED exists in the refusal_code set; admit_* functions receive calls_enabled: bool (or the client) so the refusal is decided BEFORE any node attempts a SurePass call; cache-hit admissions bypass the kill switch entirely (no call needed). Cap accounting counts only rows where a bureau endpoint (EP_EXPERIAN_SCORE/EP_EXPERIAN_REPORT) was actually called, which equals 'a surepass_calls row for that endpoint exists for this check_id'.
- **name**: evals section (datasets/evaluators) | **expectation**: evals/datasets.py builds examples exclusively from tools.fixtures builders (never from surepass-lab/out/bureau); policy_consistency evaluator diffs FakeSurePassClient.calls / endpoints_called() against the checks row flags and MemoryRepo surepass_calls rows; no_leak evaluator can rely on the guarantee that CutoffOutcome.details and all SurePassResult.raw content stay out of LcView by construction in redact/persist.

## Decisions
- **topic**: How the client learns the current check_id | **decision**: Every SurePassClient method takes check_id as an explicit first positional parameter; the client itself is a process-lifetime singleton injected at graph-build time. | **rationale**: The DI note fixes client construction at build time, so constructor-held check_id is impossible; a per-call argument is explicit, thread-safe, trivially fakeable, and avoids ContextVar magic that hides the audit trail from tests.
- **topic**: Who writes surepass_calls rows | **decision**: The client owns audit logging through a narrow CallLogSink protocol (insert_surepass_call), writing exactly one row per HTTP attempt -- including attempts that end in a raised SurePassError with http_status=None on pure transport failure -- and zero rows for LiveCallsDisabled/missing-token raises (no I/O happened). Sink exceptions propagate. | **rationale**: PRD 8.1 places logging in the client contract; one-row-per-attempt makes the policy_consistency evaluator and billing audit exact, and kill-switch raises must not fabricate spend records.
- **topic**: Validation-failure envelope ({errors, message}, no success/data keys) | **decision**: parse_surepass_body raises SurePassValidationError carrying the parsed errors dict instead of returning a typed result. | **rationale**: Both envelope shapes are parsed as required, but no PRD policy branch consumes a 422: our API layer regex-validates PAN/mobile first, so this shape means a programming bug; failing loudly prevents mis-caching a malformed request as pan_invalid or NTC.
- **topic**: success:false envelopes | **decision**: Any JSON body with a boolean success key is returned as a typed SurePassResult regardless of HTTP status (200/400/404/422), after the 401/403/429/5xx raises; adapters then map PAN success:false to pan_valid=False and score/report success:false to NTC (marked ASSUMPTION R10 in code). | **rationale**: Policy rules 6 and 7 branch on these outcomes deterministically, and the NTC shape is unverified, so the client must hand them to adapters rather than raise.
- **topic**: Guard against mis-caching infra failures as NTC | **decision**: NON_BUREAU_FAILURE_CODES: frozenset[str] = frozenset() in surepass.py; interpret_response raises SurePassUpstreamError when a success:false message_code is in the set. Ships empty. | **rationale**: A success:false meaning 'credits exhausted' would otherwise be cached as NTC for 30 days; branching on message_code (never message text) is explicitly allowed by the FIXED rule, and the empty set documents the open item until real codes are observed.
- **topic**: Admission's LIVE_CALLS_DISABLED refusal | **decision**: SurePassClient exposes a read-only calls_enabled property (live and token present; Fake: constructor flag, default True) so policy/guards.py can refuse LIVE_CALLS_DISABLED up front when a billed call would be needed; the call-time raise remains as defense in depth, and cache hits still succeed with calls disabled. | **rationale**: Acceptance criteria require LIVE_CALLS_DISABLED refusals (not exceptions) when SUREPASS_LIVE is unset, and cache hits must not be blocked by the kill switch.
- **topic**: Transport, timeouts, retries | **decision**: Synchronous httpx.Client, connect timeout 5s, read timeout 60s, ZERO automatic retries; httpx.HTTPError maps to SurePassTransportError. | **rationale**: The API guide is explicit that a timed-out call has unknown spend status ('check the console before retrying a billed call'); auto-retry risks double billing. Sync keeps nodes plain functions; FastAPI runs sync routes on its threadpool.
- **topic**: Band boundary membership and ordering | **decision**: Left-closed buckets: <650 | [650,720) | [720,780) | >=780; so 650 -> "650-720", 720 -> "720-780", 780 -> "780+". BAND_ORDER = {NTC:0, <650:1, 650-720:2, 720-780:3, 780+:4}; NTC sorts lowest and is a KNOWN fact (min_band applies to it). parse_score clamps to [300,900]; anything else (incl. "-1", "0") -> None -> NTC. | **rationale**: PRD labels imply half-open intervals; NTC-lowest makes min_band='NTC' mean 'accepts new-to-credit'; the [300,900] clamp absorbs plausible no-hit sentinels per R10.
- **topic**: Ceiling rounding | **decision**: round_to_nearest_1000 uses explicit half-up: float(int(value/1000.0 + 0.5) * 1000), not Python round(). | **rationale**: round() does banker's rounding (round(731.5) == 732 but round(730.5) == 730), which would make golden tests flaky at .5 boundaries.
- **topic**: loan_ceiling with lender max_foir absent | **decision**: Return None (alongside the PRD-mandated None when income is unknown). | **rationale**: The ceiling formula is defined in terms of max_foir and the PRD provides defaults only for rate (14) and tenure (36); inventing a default FOIR would be a product decision. Surfaced as a PRD gap.
- **topic**: loan_ceiling with existing_emi unknown | **decision**: Treat existing_emi None as 0.0 in the headroom formula; the floor flag and confidence wording carry the uncertainty. | **rationale**: PRD says ceiling is null only when income is unknown, and Tier 1 with optional EMI must still produce a richer output when income is given; marked ASSUMPTION.
- **topic**: Rate and tenure string conventions | **decision**: parse_rate accepts (0,100] only ("34.900" -> 34.9; 0/negative/>100 -> None -> default 14). resolve_tenure_months tries Repayment_Tenure then Terms_Duration with zero_is_unknown=True (so "0", "", "000" -> unknown; "024" -> 24 via int(s,10)) and returns None when Terms_Frequency is neither "" nor "M". | **rationale**: Matches the verified field map exactly; a non-monthly frequency would silently corrupt month math, so it degrades to the default instead.
- **topic**: 24-month DPD window rule | **decision**: Anchor = CreditProfileHeader.ReportDate (fallback: caller-supplied today, then date.today()); a history row counts when months_ago = (anchor.year - Y)*12 + (anchor.month - M) is in [0, 23]; Days_Past_Due parsed as plain int ("0" is a real 0, ""/non-numeric skipped); max over ALL accounts, default 0. | **rationale**: PRD says 'last 24 months' without an anchor; the report's own date is the only timestamp that makes cached reports reproducible, and closed-account delinquency inside the window still matters for the Tier-2 prompt.
- **topic**: 365-day bureau-address recency | **decision**: An account contributes pincodes when (anchor - Date_Reported).days <= 365 with anchor = ReportDate; unparseable Date_Reported excludes the account; future-dated rows (negative delta) are kept; zips filtered to ^[1-9][0-9]{5}$, output deduped and sorted. | **rationale**: Implements the PRD filter with a deterministic anchor; the probe showed junk zips ('20') and multi-year-stale addresses, and sorting makes gate inputs stable for tests and evals.
- **topic**: Defensive list handling | **decision**: as_list() treats a bare dict as a one-element list and non-lists as empty, applied to CAIS_Account_DETAILS, CAIS_Account_History and CAIS_Holder_Address_Details. | **rationale**: Experian JSON is converted from XML upstream; single-element collapse is a classic failure mode and the guard costs nothing (marked defensive ASSUMPTION).
- **topic**: aadhaar_linked strictness | **decision**: Identity.aadhaar_linked is the raw value only when it is literally a bool, else None; masked_aadhaar is never read into Identity (it survives only in SurePassResult.raw -> pan_cache.payload). | **rationale**: Policy rule 10 shows the link issue only when aadhaar_linked == False; coercing a missing key to False would raise false alarms to LCs, and the masked-Aadhaar negative requirement is FIXED.
- **topic**: Malformed lender rows | **decision**: gate_lenders validates hard_cutoffs via HardCutoffs.model_validate (extra='ignore'); a row failing validation is skipped and reported in a warnings list returned alongside the lenders (node appends to state.errors); it never aborts the check. | **rationale**: The admin UI writes these rows; one typo taking down every eligibility check is worse than excluding one lender with a visible warning.
- **topic**: Fake/live parity | **decision**: FakeSurePassClient accepts raw envelope dicts (optionally with an HTTP status) and resolves them through the same interpret_response/parse_surepass_body code path as the live client, logs to the same CallLogSink, and raises AssertionError on any endpoint without an injected scenario. | **rationale**: Tests then exercise real envelope parsing on every run, the policy_consistency evaluator can diff sink rows against fake.calls, and cache-hit/cap tests fail loudly if a node calls SurePass when it must not.
- **topic**: New module tools/fixtures.py | **decision**: Synthetic envelope builders live in eligibility_agent/tools/fixtures.py (shipped code), used by both tests/ and evals/datasets.py; fixture holder rows deliberately contain two contradictory DOBs. | **rationale**: The PRD layout omits a fixture home but evals need the same synthetic shapes as tests; implementer's choice that weakens no FIXED rule. The DOB trap makes 'DOB never from holder rows' testable. Probe data is never copied (hard rule).
- **topic**: Adapter determinism | **decision**: report_to_bureau_report takes keyword-only today: date | None used only when ReportDate is missing; adapters are pure functions over SurePassResult and are reused on cache reprocessing via parse_surepass_body(endpoint, 200, cached_payload). | **rationale**: No hidden clock reads in tests; cached raw payloads in pan_cache.payload can be re-adapted after adapter fixes without new bureau spend.
- **topic**: EMI estimator semantics | **decision**: estimate_existing_emi applies monthly_emi(balance, rate or 14, tenure or 36) per missing-EMI open account, exactly as PRD 8.3 states, including revolving accounts (Account_Type '10'); rows with balance <= 0 contribute 0. | **rationale**: Implement-as-stated ASSUMPTION; bureau tenure is the original tenure so the figure can understate remaining EMIs for seasoned loans, and revolving credit has no amortizing tenure -- both flagged for owner review (open items 2 and 5) rather than silently redesigned.

## Risks
- No-retry policy means one transient SurePass timeout fails the whole check run (by design: spend status after a timeout is unknown and retrying could double-bill); the graph section must surface SurePassTransportError as a clean retriable error state rather than a refusal.
- If Experian ever returns success:false for non-no-hit reasons before NON_BUREAU_FAILURE_CODES is populated, that PAN is cached as NTC for 30 days; mitigated by ntc_reason being stored on BureauScore and raw payloads kept in pan_cache.payload for reprocessing.
- Envelope drift: a renamed nesting level would degrade to NTC/empty facts (dig returns None) rather than crash; mitigate with the surepass_calls audit trail, LangSmith traces, and adapter unit tests pinned to the field map.
- Experian's known stale-open-tradeline behavior inflates total_outstanding and the EMI estimate (conservative for lending, but can under-recommend ceilings); documented field-map finding, no code workaround in v1.
- Float money math: amounts and the annuity formulas use float; at INR magnitudes (< 1e9) double precision is exact enough, and the only externally visible rounding (nearest 1,000, half-up) is pinned; tests must use pytest.approx for intermediate EMI values.
- tools/lenders.py imports BAND_ORDER/PIN_RE from tools/adapters.py and both import models from graph/state.py; graph/state.py must therefore not import from tools/* or a circular import results -- state.py stays a leaf module.

## PRD gaps
- PRD 8.2 treats success:false on the Experian score as NTC, but success:false could also mean an infra/billing failure (credits exhausted); the NTC shape is unverified (field-map R10, open item 4). I ship the NTC mapping as stated plus an empty NON_BUREAU_FAILURE_CODES frozenset that converts known infra codes into SurePassUpstreamError -- the owner must supply real codes once an NTC PAN and a failure case are observed, or 30-day NTC mis-caches are possible.
- PRD 8.4's ceiling formula requires the lender's max_foir but defines no default for it (defaults exist only for rate and tenure); I return None (no ceiling) when max_foir is absent -- confirm whether a house default FOIR (e.g. 0.55) should apply instead.
- 'Bureau pincodes ... within 365 days' and 'DPD in 24 months' name no anchor date; I anchored both to CreditProfileHeader.ReportDate (fallback: caller-supplied today) so cached reports re-derive identically -- confirm.
- Band boundary membership is not stated in the PRD; I fixed left-closed buckets (650 -> 650-720, 720 -> 720-780, 780 -> 780+) and NTC ordered below <650 for min_band -- confirm, since golden datasets will bake these in.
- The EMI estimator uses bureau tenure, which is the ORIGINAL tenure of the loan, so EMI(current balance, original tenure) understates remaining EMIs on seasoned loans, and revolving accounts (Account_Type '10') get the same term-loan formula; implemented exactly as PRD 8.3 states but both distortions need owner sign-off (open items 2 and 5).
- Whether SurePass bills a request that fails its own payload validation (the {errors, message} 422 shape) is unknown; I raise SurePassValidationError and still log the attempt, and our upstream regex validation should make it unreachable -- confirm billing treatment with SurePass.
- The envelope 'message_code' vocabulary is almost entirely unobserved for the three endpoints (only 'success', 'ip_not_whitelisted', 'auth_failed' are documented); the taxonomy therefore keys on HTTP status + success + shape, with message_code recorded for audit and the NON_BUREAU hook -- revisit once production traffic shows real codes.
