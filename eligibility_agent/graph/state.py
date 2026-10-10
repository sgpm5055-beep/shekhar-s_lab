"""Literal registry, shared fact models and the LcView wire models (v1.1).

This module is the PURE LEAF of the codebase: it imports the standard library and
pydantic ONLY, never another ``eligibility_agent`` module. Every other package imports
its literals and models from here, and the LangGraph checkpointer serialises instances
of the classes below, so nothing in this file may depend on policy, db, derive or
lenders code (those packages own the functions that operate on these types).

Confidence note (K16): quick checks cap at ``medium`` confidence. That rule is enforced
later in ``lenders/safety.py`` / the confidence step; this module only hosts the types.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Final, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- literal registry

Kind: TypeAlias = Literal["quick", "full"]

# Internal only; never appears in LC-facing strings.
Band: TypeAlias = Literal["<650", "650-720", "720-780", "780+", "NTC"]
BAND_ORDER: Final[dict[Band, int]] = {"NTC": 0, "<650": 1, "650-720": 2, "720-780": 3, "780+": 4}

Headline: TypeAlias = Literal["fundable", "possible", "unlikely", "not_fundable_alone", "under_review"]

Confidence: TypeAlias = Literal["high", "medium", "low"]
CONFIDENCE_ORDER: Final[dict[Confidence, int]] = {"low": 0, "medium": 1, "high": 2}

# The 13 PRD codes: 10 admission refusals plus 3 mid-run codes.
Code: TypeAlias = Literal[
    "CONSENT_MISSING",
    "QUOTA_EXHAUSTED",
    "PAN_DAILY_CAP",
    "STAGE_TOO_EARLY",
    "REPORT_WINDOW_OPEN",
    "IN_FLIGHT",
    "PAN_LOCKED",
    "WARN_NTC_RECENT",
    "PAN_CHANGE_NEEDS_MANAGER",
    "NO_PINCODE_ON_PAN",
    "LIVE_CALLS_DISABLED",
    "SUREPASS_FAILURE",
    "OPENAI_FAILURE",
]

# The 5-value LcView wire status. (The 7-value DB row status lives in db/repo.py.)
LcStatus: TypeAlias = Literal["ok", "pan_not_verified", "refused", "pending_verification", "service_failure"]

ScoreSource: TypeAlias = Literal["score", "report"]
ConsentCovers: TypeAlias = Literal["score", "report"]
LockKind: TypeAlias = Literal["in_flight", "timeout", "daily_cap"]
ErrorKind: TypeAlias = Literal["timeout", "transport", "auth", "rate_limited", "upstream", "protocol", "validation"]
ReviewKind: TypeAlias = Literal[
    "pan_daily_cap",
    "timeout_reconciliation",
    "openai_failure",
    "safety_demoted_all",
    "not_fundable_pattern",
    "pan_change_manager",
    "model_review_flags",
    "surepass_auth",
]
PanEventKind: TypeAlias = Literal["initial", "correction", "manager_change"]

# PullPlan: whether a pull is served from cache or billed. AdmissionMode: how admission resolved.
PullPlan: TypeAlias = Literal["reuse", "call"]
AdmissionMode: TypeAlias = Literal["cache_hit", "new", "refresh"]

FieldStatus: TypeAlias = Literal["known", "missing", "uncertain"]

PINCODE_RE_STR: Final[str] = r"^[1-9][0-9]{5}$"

StateUpdate: TypeAlias = dict[str, Any]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- request / identity facts


class ConsentInfo(_Frozen):
    given: Literal[True]
    wording_version: str
    call_ref: str | None
    ip: str | None

    @field_validator("given", mode="before")
    @classmethod
    def _given_is_the_bool_true(cls, value: object) -> object:
        # pydantic's Literal[True] also admits 1 and 1.0 (they equal True); consent must be the bool itself.
        if value is not True:
            raise ValueError("consent.given must be the boolean true")
        return value


class Identity(_Frozen):
    """PAN identity. NO gender field anywhere (Experian takes none; never derive or send one)."""

    pan_valid: bool  # gate = status == 'valid' and category == 'person'
    status: str | None
    category: str | None
    full_name: str | None  # sent as `name` on bureau calls; LC-typed name lives only in the consent record
    dob: date | None  # data.dob YYYY-MM-DD; NEVER from bureau holder rows
    aadhaar_linked: bool | None  # strict bool; the link warning fires ONLY on literal False
    pan_pincode: str | None  # data.address.zip iff it fullmatches PINCODE_RE_STR
    surepass_client_id: str | None  # masked_aadhaar deliberately NOT modelled (raw payload only)


# --------------------------------------------------------------------------- bureau facts


class ScoreFacts(_Frozen):
    band: Band
    score: int | None  # internal only; never serialised to any prompt or LcView
    is_ntc: bool  # ONLY HTTP 200 + success:true + credit_score not ^\d{3}$
    pulled_on: date  # fetched_at date of the cache row / this pull
    source: ScoreSource  # a Report pull refreshes the score cache


class OpenAccount(_Frozen):
    """One row per OPEN account; the section 8.3 EMI algorithm's input."""

    account_key: str  # 'acct_<i>' synthetic; Account_Number is never carried
    portfolio_type: Literal["I", "R"] | None
    account_type: str  # raw code string ('05', '10', ...)
    scheduled_emi: float | None  # parsed Scheduled_Monthly_Payment_Amount; None = not reported
    current_balance: float  # parsed or 0.0; negatives clamped to 0.0
    tenure_months: int | None  # Repayment_Tenure if > 0 else Terms_Duration if > 0
    rate_of_interest: float | None  # context for the model ONLY; never used in the estimate
    open_date: date | None
    date_reported: date | None
    amount_past_due: float
    pincodes: list[str]  # this account's ZIP values passing PINCODE_RE_STR
    is_stale: bool  # (report_date - date_reported).days > 90, or date_reported unparseable


class ReportFacts(_Frozen):
    band: Band
    score: int | None
    is_ntc: bool  # no-file: stored as NTC for 30 days
    report_date: date | None  # CreditProfileHeader.ReportDate
    open_accounts: list[OpenAccount]
    past_due_total: float  # sum of Amount_Past_Due over ALL accounts
    written_off_or_settled: int  # count of non-empty Written_off_Settled_Status
    enquiries_30d: int
    enquiries_90d: int
    enquiries_180d: int
    bureau_pincodes_12m: list[str]  # open accounts reported within 365d of report_date; sorted, deduped


# --------------------------------------------------------------------------- derivations


class EmiItem(_Frozen):
    account_key: str
    kind: Literal["reported", "estimated", "stale_excluded"]
    monthly_amount: float  # 0.0 for stale_excluded
    method: Literal["reported", "tenure_remaining", "past_tenure", "revolving_5pct", "balance_36"] | None
    past_tenure: bool
    confirm_item: str | None  # e.g. 'confirm with learner: an account last updated 2026-05-20'


class EmiDerivation(_Frozen):
    reported_total: float
    estimated_total: float
    derived_total: float  # reported + estimated over NON-stale open accounts
    declared: float | None
    used: float  # full: max(declared, derived); quick: declared or 0.0
    used_source: Literal["derived", "declared", "declared_zero_assumed"]
    emi_floor_flag: bool  # any open account estimated OR flagged (stale / past-tenure)
    items: list[EmiItem]


class LenderCriteria(_Frozen):
    """Normalised PUBLISHED admin criteria (section 11)."""

    score_floor: int | None  # entered as an Experian score (open item: lenders quote CIBIL)
    ntc_accepted: bool | None
    foir_limit: float | None
    min_income: float | None
    min_age_years: int | None
    max_age_at_maturity_years: int | None
    rate_min_pct: float | None
    rate_max_pct: float | None
    tenure_min_months: int | None
    tenure_max_months: int | None
    loan_min: float | None
    loan_max: float | None
    policy_notes: str | None
    field_status: dict[str, FieldStatus]  # per-field known/missing/uncertain from the admin UI


class GatedLender(_Frozen):
    id: str
    name: str  # resolved to LcView only in redact; NEVER in the model payload
    criteria: LenderCriteria
    matched_pincodes: list[str]
    rate_band_label: str  # '11 to 14%' (for LcView top_lender.rate_band)
    tenure_label: str  # '12 to 60 months'


class Derived(_Frozen):
    """Written by the derive node."""

    band: Band  # newer of ScoreFacts/ReportFacts (by pulled_on vs report_date)
    emi: EmiDerivation
    current_foir: float | None  # emi.used / income; None when there is no income
    age_at_application_years: int | None  # from Identity.dob only; None -> age check skipped, confidence capped medium
    age_months_at_application: int | None  # integer-month basis for maturity math
    candidate_pincodes: list[str]


class LenderNumbers(_Frozen):
    """Written by the range node, one per gated lender."""

    lender_id: str
    available_emi: float | None  # None when no income; 0.0 -> not fundable alone with this lender
    loan_max_lower: float | None  # display value, half-up to 1000 (K6: derive.range.round_to_nearest_1000)
    loan_max_upper: float | None  # display value, half-up to 1000 (K6); == ceiling
    loan_max_lower_raw: float | None  # unrounded AND unclipped annuity value, internal + safety
    loan_max_upper_raw: float | None
    ceiling: float | None
    displayed_foir: float | None  # (emi.used + emi(P_wc at r_max, n_max)) / income
    age_at_maturity_years: int | None  # (age_months + tenure_max) // 12
    not_fundable_alone: bool  # available_emi == 0.0
    clipped: Literal["none", "upper", "lower", "both"]


# --------------------------------------------------------------------------- the single LLM node's schema


class Pick(BaseModel):
    """PRD 9.2 verbatim, extra=forbid. `rank` is ordering metadata, not a score or money figure."""

    model_config = ConfigDict(extra="forbid")

    lender_id: str
    rank: int
    rationale: str
    conditions: list[str]


class ModelOutput(BaseModel):
    """PRD 9.2 verbatim, extra=forbid. The model never produces a number: no numeric money/score fields exist."""

    model_config = ConfigDict(extra="forbid")

    recommendation: Pick | None
    alternates: list[Pick] = Field(max_length=2)
    confidence: Confidence  # may only LOWER code_confidence (enforced by min())
    co_applicant_suggested: bool
    co_applicant_reason: str | None
    review_flags: list[str]


class SafetyBreach(_Frozen):
    lender_id: str
    rule: Literal["not_in_gate", "band_floor", "foir_limit", "age_bounds", "min_loan_amount", "duplicate"]
    detail: str  # internal only; may contain numbers; never reaches LcView


class SafetyResult(_Frozen):
    top: Pick | None
    alternates: list[Pick]  # max 2 surviving, order preserved
    demoted: list[SafetyBreach]
    none_survived: bool
    review: bool  # -> review_queue row at persist
    top_clears_floor_by_one: bool | None  # feeds the 9.4 'high' cap


class SurePassCallAudit(_Frozen):
    endpoint: str
    http_status: int | None
    success: bool | None
    message_code: str | None
    client_id: str | None
    billed: bool


# --------------------------------------------------------------------------- admission (pure data)


@dataclass(frozen=True, slots=True)
class FullAdmissionInputs:
    refresh: bool
    acknowledge_ntc_warning: bool


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Outcome of an admission guard (policy/guards owns the functions that build it).

    Pure frozen data, so it is safe for the checkpointer serde allowlist. When produced by
    the guards exactly one of (admission, code) is non-None; the defaults here are a neutral
    "nothing decided" value, not a valid admission.
    """

    admission: AdmissionMode | None = None
    code: Code | None = None  # a refusal code when set
    pan_plan: PullPlan = "reuse"  # 'call' iff the PAN record is missing or expired
    bureau_plan: PullPlan = "reuse"  # 'call' iff the admitted mode is new/refresh
    counts_against_quota: bool = False
    is_refresh: bool = False
    quota_used_today: int | None = None
    quota_limit: int | None = None
    create_daily_cap_lock: bool = False  # directive: daily_cap lock until next IST midnight + review row
    review_reason: ReviewKind | None = None
    record_pan_correction: bool = False  # directive: record a PAN correction event
    ntc_found_on: date | None = None
    checked_on: date | None = None
    refresh_available_on: date | None = None
    original_consent_id: str | None = None


# --------------------------------------------------------------------------- LcView v2 wire models
# The field names below are the frozen client contract (PRD v1.1 section 6). extra="forbid"
# means no undeclared field (score, band, enquiry counts, account lists) can ride along.


class LoanRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower: int
    upper: int


class LenderView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    rate_band: str  # e.g. "11 to 14%": admin string, leak-guarded
    tenure_range: str  # e.g. "12 to 48 months"
    ceiling: int | None  # None on quick without income
    range: LoanRange | None
    rationale: str  # LLM prose, scrubbed; "" when scrubbed away or code-built view
    conditions: list[str] = Field(default_factory=list)  # LLM prose, scrubbed


class LcView(BaseModel):
    model_config = ConfigDict(extra="forbid")  # no undeclared field can ride along

    check_id: str
    kind: Kind
    status: LcStatus
    code: Code | None = None
    headline: Headline | None = None  # None when no band is known (refusals, SUREPASS failures)
    label: str  # the ONE human line: provenance label on ok, section-10 message otherwise
    confidence: Confidence | None = None
    top_lender: LenderView | None = None
    alternates: list[LenderView] = Field(default_factory=list[LenderView], max_length=2)
    coapplicant_suggested: bool = False
    coapplicant_reason: str | None = None
    pan_aadhaar_link_warning: str | None = None  # EXACT string or None, never False
    items_to_confirm: list[str] = Field(default_factory=list)
    checked_on: str | None = None  # "YYYY-MM-DD" IST date of the raw pull (original date on cache hits)
    from_cache: bool = False
    refresh_available_on: str | None = None  # "YYYY-MM-DD" = pull date + 30d
    quick_checks_left_today: int | None = None  # quick only; None on full


# --------------------------------------------------------------------------- graph state


class CheckState(TypedDict, total=False):
    # request (seeded by main.py / recompute route)
    kind: Kind
    check_id: str
    lc_id: str
    lead_id: str
    opportunity_id: str | None
    opportunity_stage: str | None
    pan: str  # raw, request scope only; stripped at persist
    name: str  # LC-typed; consent record only
    mobile: str
    pan_hash: str  # HMAC-SHA256 (config/policy own the function)
    mobile_hash: str
    pan_last4: str
    mobile_last4: str | None
    monthly_income: float | None
    existing_emi_declared: float | None
    refresh: bool
    acknowledge_ntc_warning: bool
    consent: ConsentInfo | None
    is_recompute: bool
    # admission
    admission: AdmissionDecision | None
    # facts (internal; raw payloads NEVER live in state)
    identity: Identity | None
    score: ScoreFacts | None
    report: ReportFacts | None
    # derivations / matching
    derived: Derived | None
    candidate_pincodes: list[str]
    gated_lenders: list[GatedLender]
    gate_warnings: list[str]
    lender_numbers: dict[str, LenderNumbers]
    code_confidence: Confidence | None
    model_output: ModelOutput | None
    rank_attempts: int  # LLM calls made; max 2 (1 + ONE retry)
    rank_error: str | None  # fed into the retry user message
    safety: SafetyResult | None
    # outcome
    status: LcStatus | None  # set by terminal branches; redact defaults 'ok'
    code: Code | None
    headline: Headline | None
    final_confidence: Confidence | None
    lc_view: LcView | None
    # audit
    surepass_calls_made: Annotated[list[SurePassCallAudit], operator.add]
    bureau_called: bool
    pan_called: bool
    langsmith_run_id: str | None
    errors: Annotated[list[str], operator.add]
    transcript: Annotated[list[str], operator.add]  # node names in order; -> checks.internal
