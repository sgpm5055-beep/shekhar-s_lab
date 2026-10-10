# Tests, fakes, evals, CI, README (PRD sections 11-13)

## Summary
The quality layer is built around three reusable, fully typed assets that live INSIDE the installable package at eligibility_agent/testing/ (so evals can reuse them; tests/ is not importable): synthetic SurePass envelope builders that reproduce the field-map conventions exactly (dense CAIS account dicts with "" sentinels, zero-padded Terms_Duration, "0"=unknown Repayment_Tenure, YYYYMMDD dates, junk ZIP rows, validation-failure envelopes without a success key), a FrozenClock implementing the injected Clock protocol (chosen over freezegun: freezegun monkeypatches datetime globally — exactly the "monkeypatching production logic" smell we ban, it fights anyio/langgraph internals, and an injected protocol is the only approach that keeps policy/guards.py pure and mypy --strict clean), and a FakeRankLLM/FakeFinalizeLLM pair implementing the narrow structured-output callable protocols the graph section's DI seam exposes, with scripted per-attempt overrides for every misbehaving-LLM scenario (invented lender, persistent bad finalize, Experian/736 leaks). All 13 PRD section-12 tests are specified given/when/then with exact assertions, including precomputed money math (EMI(120000, 34.9%, 24m)=7015.9054, EMI(90000, 14%, 36m)=3075.9867, ceiling(60000 income, 0 EMI, 0.55 FOIR, 13.5%, 48m)=1,219,000), exact fake call-log sequences, billed/counted_against_cap flags on persisted rows, and quick_checks_remaining_today arithmetic across cache hits and the 23:59→00:01 IST boundary (same UTC date). No-network is enforced suite-wide by pytest-socket (--disable-socket --allow-unix-socket) plus env scrubbing in pytest_configure. evals/ ships dataset builders for eligibility-quick-golden and eligibility-full-golden built purely from the synthetic builders, the five PRD evaluators in langsmith SDK v0.2+ kwargs signature form (verified against current docs), a full LLM-judge rubric, and a CLI runner with a local mode (no LangSmith needed) and a langsmith mode that runs the real model on max 20 examples, gated in CI on secrets presence. mypy --strict covers eligibility_agent, tests and scripts, with pydantic plugin and per-module ignore_missing_imports confined to the supabase family. The README outline and scripts/seed_lenders.py design include three complete synthetic lenders with full hard_cutoffs and criteria JSON.

## Files
### tests/conftest.py
_Session env hardening, no-network belt-and-braces, and all shared fixtures: frozen clock, fresh MemoryRepo, seeded lenders, FakeSurePassClient scenarios, FakeLLMs, graph factory, FastAPI TestClient with auth header._

import os
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo
import pytest
from fastapi.testclient import TestClient

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2026, 10, 8, 10, 0, 0, tzinfo=IST)  # canonical frozen instant, all tests start here
TEST_TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}

# ---- process-level hardening (runs before any eligibility_agent import touches env) ----
def pytest_configure(config: pytest.Config) -> None:
    os.environ["SERVICE_API_TOKEN"] = TEST_TOKEN
    os.environ["PAN_SALT"] = "unit-test-salt-0123456789abcdef"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"   # legacy flag, set both so tracing is a no-op
    for k in ("SUREPASS_LIVE", "SUREPASS_TOKEN", "OPENAI_API_KEY", "LANGSMITH_API_KEY",
              "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "DATABASE_URL"):
        os.environ.pop(k, None)

@pytest.fixture(autouse=True)
def _no_live_calls_guard() -> Iterator[None]:
    assert os.environ.get("SUREPASS_LIVE") != "true"
    yield
# NOTE: primary no-network enforcement is pytest-socket via addopts in pyproject:
#   --disable-socket --allow-unix-socket
# (--allow-unix-socket is REQUIRED: Starlette TestClient runs an anyio portal whose asyncio
# loop self-pipe uses socket.socketpair(AF_UNIX); TCP/UDP stays blocked.)

# ---- clock / repo ----
@pytest.fixture
def clock() -> "FrozenClock":
    from eligibility_agent.testing.clock import FrozenClock
    return FrozenClock(T0)

@pytest.fixture
def repo(clock: "FrozenClock") -> "MemoryRepo":
    from eligibility_agent.db.repo import MemoryRepo
    return MemoryRepo(clock=clock)          # fresh per test, never shared

@pytest.fixture
def seeded_repo(repo: "MemoryRepo") -> "MemoryRepo":
    from tests.helpers import seed_default_lenders
    seed_default_lenders(repo)              # L1/L2/L3, see tests/helpers.py
    return repo

# ---- fakes ----
@pytest.fixture
def fake_surepass() -> "FakeSurePassClient":
    """Default happy scenario for PAN_OK: valid person PAN (zip 560001) + score 745."""
    from eligibility_agent.testing.scenarios import happy_quick_scenario
    return happy_quick_scenario()           # returns a ready FakeSurePassClient

@pytest.fixture
def fake_rank() -> "FakeRankLLM":
    from eligibility_agent.testing.fake_llm import FakeRankLLM
    return FakeRankLLM()

@pytest.fixture
def fake_finalize() -> "FakeFinalizeLLM":
    from eligibility_agent.testing.fake_llm import FakeFinalizeLLM
    return FakeFinalizeLLM()

# ---- graph + app wiring (the DI seam owned by the graph section) ----
@pytest.fixture
def graph_deps(seeded_repo: "MemoryRepo", fake_surepass: "FakeSurePassClient",
               fake_rank: "FakeRankLLM", fake_finalize: "FakeFinalizeLLM",
               clock: "FrozenClock") -> "GraphDeps":
    from eligibility_agent.graph.build import GraphDeps
    return GraphDeps(repo=seeded_repo, surepass=fake_surepass, rank_llm=fake_rank,
                     finalize_llm=fake_finalize, clock=clock,
                     live_calls_enabled=True, checkpointer=None)

@pytest.fixture
def quick_graph(graph_deps: "GraphDeps") -> "CompiledGraph":
    from eligibility_agent.graph.build import build_graph
    return build_graph("quick", graph_deps)

@pytest.fixture
def full_graph(graph_deps: "GraphDeps") -> "CompiledGraph":
    from eligibility_agent.graph.build import build_graph
    return build_graph("full", graph_deps)

@pytest.fixture
def app(graph_deps: "GraphDeps") -> "FastAPI":
    from eligibility_agent.config import Config
    from eligibility_agent.main import create_app
    return create_app(Config.from_env(), graph_deps)

@pytest.fixture
def client(app: "FastAPI") -> Iterator[TestClient]:
    with TestClient(app, headers=AUTH) as c:   # auth header pre-attached on every request
        yield c

@pytest.fixture
def anon_client(app: "FastAPI") -> Iterator[TestClient]:
    with TestClient(app) as c:                  # no auth, for 401 tests
        yield c

# Rationale for injected clock over freezegun (recorded here for reviewers):
# 1) freezegun patches datetime module-wide at runtime = monkeypatching production logic; the
#    injected Clock keeps guards/repo pure functions of (inputs, now).
# 2) freezegun does not reach DB-side now() in production (the real cap counter is a SQL
#    function on Asia/Kolkata); MemoryRepo must take time from the SAME seam the graph uses
#    or test parity is fake.
# 3) mypy --strict: Clock is a 3-line Protocol; zero plugin/stub burden; FrozenClock.advance()
#    is explicit in tests instead of magic tick arguments.

### tests/helpers.py
_Shared constants, lender seeds, request builders, leak assertions, and repo seeding helpers used by every scenario test._

from datetime import datetime
from typing import Any
import json, re

# Canonical synthetic identity (NEVER values from surepass-lab probes)
PAN_OK = "ABCPE1234F"; PAN_TWO = "FGHPJ5678K"; PAN_COMPANY = "AAACA9999Z"; PAN_NTC = "KLMPN4321Q"
MOBILE_OK = "9876543210"; NAME_OK = "Ravi Testkumar"
LC_A = "lc_alpha"; LC_B = "lc_beta"; LEAD = "lead_001"; OPP = "opp_001"

# Redaction regexes (mirror of output/lc_view.py constants; imported in tests from there,
# duplicated here ONLY inside assert_no_leak for defense-in-depth):
BUREAU_RE = re.compile(r"(?i)\b(cibil|experian|equifax|crif|bureau)\b")
SCORE_RE  = re.compile(r"(?<![\d,.])[3-8][0-9]{2}(?![\d,.])")   # standalone 300-899

def assert_no_leak(lc_view_json: dict[str, Any]) -> None:
    """Serialise the whole LcView and scan every string leaf. Fails on any bureau word or
    standalone 3-digit number in 300-899. Used by tests 1,2,5,8,11 and every refusal test."""
    blob = json.dumps(lc_view_json, ensure_ascii=False)
    assert not BUREAU_RE.search(blob), f"bureau word leaked: {blob}"
    for leaf in _string_leaves(lc_view_json):           # walk dict/list recursively
        assert not SCORE_RE.search(leaf), f"score-like number leaked: {leaf!r}"

def _string_leaves(node: object) -> "Iterator[str]": ...  # recursive walker, ~8 lines

def quick_payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"lc_id": LC_A, "lead_id": LEAD, "opportunity_id": OPP,
        "name": NAME_OK, "mobile": MOBILE_OK, "pan": PAN_OK, "consent": True,
        "monthly_income": 60000, "existing_emi": 0, "refresh": False}
    base.update(over); return base

def full_payload(**over: Any) -> dict[str, Any]:
    base = quick_payload(opportunity_stage="offer")
    base.update(over); return base

# Default lender seed used by nearly all tests (pincode sets chosen per-test where noted)
# L1 Alpha Education Finance  pincodes {560001,560002,110001}
#   hard_cutoffs {"min_band":"650-720","max_foir":0.55,"min_income":25000,"min_age":21,
#                 "max_age_at_maturity":58,"max_tenure_months":48,"annual_rate_pct":13.5}
# L2 Bharat Upskill Credit    pincodes {560001}
#   hard_cutoffs {"min_band":"NTC","max_foir":0.45,"min_income":20000,"min_age":21,
#                 "max_age_at_maturity":60,"max_tenure_months":36,"annual_rate_pct":16.0}
# L3 Crest Capital NBFC       pincodes {400001}
#   hard_cutoffs {"min_band":"720-780","max_foir":0.6,"min_income":50000,"min_age":23,
#                 "max_age_at_maturity":55,"max_tenure_months":60,"annual_rate_pct":11.75}
def seed_default_lenders(repo: "MemoryRepo") -> None: ...  # repo.upsert_lender x3 with criteria JSON

def seed_counted_quick_check(repo: "MemoryRepo", *, lc_id: str, created_at: datetime,
                             pan_hash: str = "seedhash", n: int = 1) -> None:
    """Insert n checks rows with kind='quick', status='ok', billed=True,
    counted_against_cap=True at created_at — drives cap tests without 5 graph runs."""

def one_check_row(repo: "MemoryRepo") -> "CheckRow":
    rows = repo.list_checks(); assert len(rows) == 1; return rows[0]

def endpoints(fake: "FakeSurePassClient") -> list[str]:
    return [c.endpoint for c in fake.calls]

### eligibility_agent/testing/clock.py
_FrozenClock implementing the injected Clock protocol; the only time source any test uses._

from datetime import datetime, timedelta, timezone

class FrozenClock:
    """Structural match for eligibility_agent Clock protocol (def now() -> datetime, aware UTC)."""
    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock requires an aware datetime")
        self._now: datetime = start.astimezone(timezone.utc)
    def now(self) -> datetime:
        return self._now
    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta
    def set_now(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("aware datetime required")
        self._now = instant.astimezone(timezone.utc)


### eligibility_agent/testing/envelopes.py
_Synthetic SurePass envelope builders reproducing the field-map conventions exactly. The ONLY source of SurePass-shaped data for tests AND eval datasets. No probe values, ever._

from collections.abc import Sequence
from typing import Any

EnvelopePair = tuple[int, dict[str, Any]]   # (http_status, body)

def _ok_envelope(data: dict[str, Any], message_code: str = "success") -> EnvelopePair:
    return (200, {"data": data, "status_code": 200, "success": True,
                  "message": None, "message_code": message_code})
    # message deliberately None sometimes in variants: code must never branch on message

def failure_envelope(*, http_status: int = 422, message_code: str = "invalid_id",
                     message: str | None = "Invalid PAN") -> EnvelopePair:
    return (http_status, {"data": {}, "status_code": http_status, "success": False,
                          "message": message, "message_code": message_code})

def validation_error_envelope(field: str = "id_number",
                              msg: str = "Missing required parameter.") -> EnvelopePair:
    # Field-map rule: validation failures have NO success/data keys
    return (422, {"errors": {field: msg}, "message": "Input payload validation failed"})

def pan_comprehensive_envelope(*, pan: str = "ABCPE1234F", full_name: str = "RAVI TESTKUMAR",
    status: str = "valid", category: str = "person", gender: str = "M",
    dob: str = "1998-04-12", aadhaar_linked: bool = True, zip_code: str = "560001",
    client_id: str = "pan_comprehensive_synthetic_0001") -> EnvelopePair:
    data: dict[str, Any] = {
        "client_id": client_id, "pan_number": pan, "full_name": full_name,
        "full_name_split": full_name.split(" "), "masked_aadhaar": "XXXXXXXX0000",
        "aadhaar_linked": aadhaar_linked, "dob": dob, "gender": gender,
        "status": status, "category": category, "input_dob": None,
        "dob_verified": True, "dob_check": False, "less_info": False,
        "phone_number": "", "email": "",
        "address": {"line_1": "12 Synthetic Lane", "line_2": "", "street_name": "",
                     "zip": zip_code, "city": "Testpur", "state": "Karnataka",
                     "country": "", "full": "12 Synthetic Lane, Testpur"}}
    return _ok_envelope(data)

def experian_score_envelope(*, credit_score: str = "745", report_date: str = "20261001",
    success: bool = True, message_code: str = "success",
    client_id: str = "experian_score_synthetic_0001") -> EnvelopePair:
    if not success:   # ASSUMPTION branch: NTC/no-hit shape UNVERIFIED (PRD 8.2, open item 4)
        return failure_envelope(http_status=200, message_code="no_record_found",
                                message="No record found")
    data = {"client_id": client_id, "credit_score": credit_score,
            "credit_report": {"SCORE": {"FCIREXScore": credit_score, "FCIREXScoreConfidLevel": ""},
                               "CreditProfileHeader": {"ReportDate": report_date},
                               "Match_result": {"Exact_match": "Y"}}}
    return _ok_envelope(data)

# ---- dense Experian CAIS account builder (field-map section 6 conventions) ----
HistoryRow = tuple[str, str, str]   # (Year, Month, Days_Past_Due)

def cais_account(*,
    account_type: str = "05",            # numeric code strings
    account_status: str = "11",          # "11" open, "13" closed, "15" seen-once closed
    date_closed: str = "",               # "" = open; YYYYMMDD when closed
    scheduled_emi: str = "",             # '' = not reported — the NORM (100% empty on probe)
    current_balance: str = "120000",
    amount_past_due: str = "0",
    repayment_tenure: str = "0",         # "0" = unknown
    terms_duration: str = "",            # zero-padded like "024", or ""
    terms_frequency: str = "M",
    rate_of_interest: str = "",          # "34.900"-style string or ""
    open_date: str = "20240115",
    date_reported: str = "20260901",
    date_of_last_payment: str = "20260825",
    written_off_settled_status: str = "",# "99" when written off/settled
    settlement_amount: str = "",
    payment_rating: str = "0",
    payment_history_profile: str = "000000000000",
    history: Sequence[HistoryRow] = (("2026", "09", "0"),),
    holder_zip: str = "560001",          # junk like "20" allowed for filter tests
    holder_dob: str = "19980412",        # deliberately IGNORED by adapter (DOB from PAN only)
    highest_credit: str = "200000",
    credit_limit: str = "",
    account_number: str = "SYNTHACC0001",
    portfolio_type: str = "I",
) -> dict[str, Any]:
    """Returns a DENSE dict: every field-map key present on every account, '' = not reported."""
    return {
        "Account_Number": account_number, "Account_Type": account_type,
        "Account_Status": account_status, "Portfolio_Type": portfolio_type,
        "Date_Closed": date_closed, "Open_Date": open_date, "Date_Reported": date_reported,
        "Date_of_Last_Payment": date_of_last_payment, "DateOfAddition": date_reported,
        "Scheduled_Monthly_Payment_Amount": scheduled_emi,
        "Current_Balance": current_balance, "Amount_Past_Due": amount_past_due,
        "Highest_Credit_or_Original_Loan_Amount": highest_credit,
        "Credit_Limit_Amount": credit_limit,
        "Repayment_Tenure": repayment_tenure, "Terms_Duration": terms_duration,
        "Terms_Frequency": terms_frequency, "Rate_of_Interest": rate_of_interest,
        "Payment_History_Profile": payment_history_profile, "Payment_Rating": payment_rating,
        "Written_off_Settled_Status": written_off_settled_status,
        "Settlement_Amount": settlement_amount, "SuitFiled_WilfulDefault": "",
        "CAIS_Account_History": [
            {"Year": y, "Month": m, "Days_Past_Due": dpd, "Asset_Classification": "?"}
            for (y, m, dpd) in history],
        "CAIS_Holder_Address_Details": [{
            "First_Line_Of_Address_non_normalized": "12 SYNTHETIC LANE",
            "City_non_normalized": "TESTPUR", "State_non_normalized": "29",
            "ZIP_Postal_Code_non_normalized": holder_zip,
            "Address_indicator_non_normalized": "", "Residence_code_non_normalized": ""}],
        "CAIS_Holder_Phone_Details": [{"Telephone_Number": "", "Mobile_Telephone_Number": "",
                                        "EMailId": ""}],
        "CAIS_Holder_Details": [{"Date_of_birth": holder_dob, "Gender_Code": "1"}],
    }

def experian_report_envelope(*, accounts: Sequence[dict[str, Any]], credit_score: str = "745",
    report_date: str = "20261001", caps_last_180: str = "0", success: bool = True,
    client_id: str = "experian_report_synthetic_0001") -> EnvelopePair:
    if not success:
        return failure_envelope(http_status=200, message_code="no_record_found",
                                message="No record found")   # ASSUMPTION branch, see score
    data = {"client_id": client_id, "credit_score": credit_score, "credit_report": {
        "CreditProfileHeader": {"ReportDate": report_date},
        "SCORE": {"FCIREXScore": credit_score, "FCIREXScoreConfidLevel": ""},
        "Current_Application": {"Current_Application_Details": {
            "Current_Applicant_Address_Details": {}}},
        "CAIS_Account": {
            "CAIS_Summary": {"Credit_Account": {"CreditAccountTotal": str(len(accounts))}},
            "CAIS_Account_DETAILS": list(accounts)},
        "TotalCAPS_Summary": {"TotalCAPSLast7Days": "0", "TotalCAPSLast30Days": "0",
            "TotalCAPSLast90Days": "0", "TotalCAPSLast180Days": caps_last_180}}}
    return _ok_envelope(data)


### eligibility_agent/testing/scenarios.py
_Canned FakeSurePassClient factories: one function per named scenario so tests and eval datasets share identical worlds._

from typing import Any
from eligibility_agent.testing import envelopes as env
from eligibility_agent.tools.surepass import FakeSurePassClient

PAN_OK = "ABCPE1234F"; PAN_NTC = "KLMPN4321Q"; PAN_COMPANY = "AAACA9999Z"

def happy_quick_scenario(*, pan: str = PAN_OK, score: str = "745",
                         zip_code: str = "560001") -> FakeSurePassClient:
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan, zip_code=zip_code)},
        score={pan: env.experian_score_envelope(credit_score=score)})

def ntc_scenario(variant: str, *, pan: str = PAN_NTC) -> FakeSurePassClient:
    # variant 'success_false' -> {success: False, message_code: 'no_record_found'} (ASSUMPTION)
    # variant 'empty_score'   -> success True, data.credit_score == ""
    ...

def pan_invalid_scenario(kind: str, *, pan: str) -> FakeSurePassClient:
    # kind 'failure_envelope' -> success False invalid_id (no score key needed: must never be called)
    # kind 'company'          -> category='company'; kind 'status_invalid' -> status='invalid'
    ...

def full_pincode_union_scenario(*, pan: str = PAN_OK) -> FakeSurePassClient:
    """PAN zip 560002. Report accounts:
    a1 open  EMI '5000' balance '200000' Date_Reported 20260901 holder_zip '110001'
    a2 closed('13', Date_Closed 20250110) Date_Reported 20260815 holder_zip '20'   (junk)
    a3 closed Date_Reported 20250101 (stale >365d)              holder_zip '400001'
    credit_score '745'"""
    ...

def full_all_emi_empty_scenario(*, pan: str = PAN_OK) -> FakeSurePassClient:
    """PAN zip 560001, income-tier-2 fixture. Accounts:
    a1 open EMI '' balance '120000' Repayment_Tenure '0' Terms_Duration '024' rate '34.900'
    a2 open EMI '' balance '90000'  Repayment_Tenure '0' Terms_Duration ''    rate ''
    a3 closed EMI '' balance '0' Date_Closed '20240110' account_status '13'
    All holder zips 560001; credit_score '702'."""
    ...

def full_delinquency_scenario(*, pan: str = PAN_OK) -> FakeSurePassClient:
    """One open account Amount_Past_Due '18000', history DPD rows up to '65' within 24m,
    one account Written_off_Settled_Status '99'; credit_score '628'."""
    ...


### eligibility_agent/testing/fake_llm.py
_Deterministic FakeRankLLM / FakeFinalizeLLM implementing the graph section's narrow structured-output callable protocols, with scripted overrides for misbehaving-LLM tests._

from collections.abc import Callable, Sequence
from eligibility_agent.graph.state import FinalRecommendation, Ranking, RankedLender
from eligibility_agent.graph.llm import RankContext, FinalizeContext  # owned by graph section

FALLBACK_MESSAGE = ("The check could not be completed automatically. "
                    "Please raise it for manual review.")
LOW_BANDS = frozenset({"NTC", "<650"})

def _round1000(x: float) -> float:
    return float(int(round(x / 1000.0)) * 1000)

class FakeRankLLM:
    """Callable matching RankLLM = Callable[[RankContext], Ranking].
    Default: every gated lender whose cutoff result passed -> fit='strong',
    sorted by lender_id; why text is band-driven prose with NO numbers and NO bureau words:
    f"meets lender criteria for the {ctx.applicant.band} profile" -- EXCEPT band text is
    mapped to words first: {'780+':'very strong','720-780':'strong','650-720':'moderate',
    '<650':'thin','NTC':'new-to-credit'} so even internal-echo never leaks a 3-digit token.
    """
    def __init__(self, *, respond: Callable[[RankContext], Ranking] | None = None,
                 script: Sequence[Ranking] | None = None) -> None:
        self.respond = respond; self.script = list(script or [])
        self.calls: list[RankContext] = []
    def __call__(self, ctx: RankContext) -> Ranking:
        self.calls.append(ctx)
        if self.script:
            return self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if self.respond is not None:
            return self.respond(ctx)
        ranked = [RankedLender(lender_id=l.id, fit="strong", why=_why(ctx, l))
                  for l in sorted(ctx.gated_lenders, key=lambda l: l.id)
                  if ctx.cutoffs[l.id].passed]
        return Ranking(ranked=ranked)

class FakeFinalizeLLM:
    """Callable matching FinalizeLLM = Callable[[FinalizeContext], FinalRecommendation].
    DEFAULT DECISION TABLE (deterministic, mirrors prompt rules so golden evals are stable):
      accepted = ctx.ranking.accepted (list[str]); band = ctx.applicant.band
      top = accepted[0] if accepted else None; alternates = accepted[1:3]
      outcome:  accepted empty                        -> 'review'
                band in LOW_BANDS                     -> 'fundable_with_coapplicant' (coapplicant_suggested=True)
                else                                  -> 'fundable'
      confidence: accepted empty or ctx.flags.conflict -> 'low'
                  ctx.flags.emi_is_floor or not ctx.flags.income_declared -> 'medium'
                  else                                 -> 'high'
      indicative_loan_max = ctx.affordability.ceilings[top] if top and income known else None
      indicative_loan_min = _round1000(0.4 * indicative_loan_max) if max else None
      lc_talking_points: ['Strong profile, proceed with the pitch'] for fundable;
        ['Suggest adding a working parent or spouse as co-applicant'] for coapplicant;
        [FALLBACK_MESSAGE] for review. (<=4 items, each <=200 chars, no digits 300-899,
        no bureau words — keeps no_leak green by construction.)
      confirm_with_learner (order fixed): 'Declared monthly income' if not income_declared;
        'Confirm current pincode' if ctx.flags.pincode_mismatch;
        'Confirm existing EMIs' if ctx.flags.emi_is_floor.  (max 3)
      coapplicant_suggested = outcome == 'fundable_with_coapplicant'
      reasoning_internal = f"band={band} foir={ctx.applicant.foir} accepted={accepted}"
        (numbers allowed here on purpose: proves redact strips nothing from internal)
    Overrides:
      script: Sequence[FinalRecommendation] replayed per attempt (last repeats) — drives
        'keeps naming dropped lender after retries'.
      talking_points_override / confirm_override: replace just those lists on the default
        output — drives leak-scrub tests without breaking lender-id validity.
    """
    def __init__(self, *, respond: Callable[[FinalizeContext], FinalRecommendation] | None = None,
                 script: Sequence[FinalRecommendation] | None = None,
                 talking_points_override: list[str] | None = None,
                 confirm_override: list[str] | None = None) -> None:
        self.calls: list[FinalizeContext] = []
        ...
    def __call__(self, ctx: FinalizeContext) -> FinalRecommendation: ...

def bad_ranking(ids: Sequence[str]) -> Ranking:
    """Helper: Ranking naming arbitrary (possibly invented) lender ids as strong."""
    return Ranking(ranked=[RankedLender(lender_id=i, fit="strong", why="looks good") for i in ids])

def bad_final(top: str, *, alternates: list[str] | None = None) -> FinalRecommendation:
    return FinalRecommendation(outcome="fundable", confidence="high",
        indicative_loan_min=100000.0, indicative_loan_max=200000.0, top_lender_id=top,
        alternate_lender_ids=alternates or [], coapplicant_suggested=False,
        lc_talking_points=["Strong profile, proceed with the pitch"],
        confirm_with_learner=[], reasoning_internal="scripted bad final")


### tests/test_quick_happy_path.py
_PRD test 1 — quick check happy path end to end through the API._

PRD TEST 1 (given/when/then):
GIVEN seeded_repo (L1,L2,L3), fake_surepass = happy_quick_scenario (PAN_OK, zip 560001, score 745),
      clock at T0 (2026-10-08 10:00 IST), default FakeLLMs, client with auth header.
WHEN  r = client.post('/checks/quick', json=quick_payload())  # income 60000, emi 0
THEN
  assert r.status_code == 200; body = r.json()
  # exact SurePass call sequence, once each, in order:
  assert endpoints(fake_surepass) == [PAN_ENDPOINT, SCORE_ENDPOINT]           # constants from tools.surepass
  assert fake_surepass.calls[1].payload == {'name': 'RAVI TESTKUMAR',          # PAN full_name wins over request name
      'mobile': MOBILE_OK, 'pan': PAN_OK, 'consent': 'Y'}
  assert body['status'] == 'ok' and body['kind'] == 'quick' and body['refusal_code'] is None
  assert body['from_cache'] is False
  # gate + cutoffs: L3 (pincode 400001) excluded; band 720-780, age 28, foir 0 -> L1,L2 pass
  assert body['top_lender'] == {'id': 'L1', 'name': 'Alpha Education Finance'}
  assert [a['id'] for a in body['alternates']] == ['L2']
  # ceiling math (precomputed): L1 0.55*60000=33000 headroom @13.5%/48m -> 1,219,000
  assert body['indicative_loan_range'] == {'min': 488000.0, 'max': 1219000.0}  # min = round1000(0.4*max)
  assert body['outcome'] == 'fundable' and body['confidence'] == 'high'
  assert body['coapplicant_suggested'] is False
  assert body['pan_aadhaar_link_issue'] is None                                # linked true -> shown only when a problem
  assert body['quick_checks_remaining_today'] == 4
  assert_no_leak(body)                                                          # no 'experian', no 745, no 736...
  # persisted rows:
  row = one_check_row(seeded_repo)
  assert (row.billed, row.counted_against_cap, row.is_refresh, row.status) == (True, True, False, 'ok')
  assert row.lc_view is not None and row.internal is not None
  cache = seeded_repo.get_cache(hash_pan(PAN_OK, salt), 'quick')
  assert cache is not None and cache.is_ntc is False
  assert cache.expires_at == clock.now() + timedelta(days=30)
  # surepass_calls audit rows: 2 rows, endpoints match, success True
  assert [c.endpoint for c in seeded_repo.list_surepass_calls(row.id)] == [PAN_ENDPOINT, SCORE_ENDPOINT]
  # LLM seam used exactly once each:
  assert len(fake_rank.calls) == 1 and len(fake_finalize.calls) == 1
ALSO (same file): test_quick_without_income — payload without monthly_income/existing_emi:
  200, outcome 'fundable', confidence 'medium' (income unknown), indicative_loan_range min/max None,
  'Declared monthly income' in body['confirm_with_learner'], remaining 4->3 given fresh repo per test = 4.

### tests/test_cache_and_cap.py
_PRD tests 2 and 3 plus remaining-today arithmetic and the IST 23:59/00:01 boundary._

PRD TEST 2 — second LC, same PAN, within TTL:
GIVEN happy world; WHEN lc_A posts quick (billed, remaining 4), clock.advance(timedelta(days=3)),
      THEN lc_B posts quick_payload(lc_id=LC_B) same PAN:
  assert endpoints(fake_surepass) == [PAN_ENDPOINT, SCORE_ENDPOINT]   # UNCHANGED: no new SurePass calls
  b2 = second response: b2['from_cache'] is True; b2['status'] == 'ok'
  b2['cached_at'] == iso(T0); b2['refresh_available_at'] == iso(T0 + 30 days)
  b2['quick_checks_remaining_today'] == 5          # lc_B's own cap untouched by a cache hit
  rows = seeded_repo.list_checks(); len(rows) == 2
  rows[1].billed is False and rows[1].counted_against_cap is False
  seeded_repo.quick_checks_used_today(LC_A) == 0   # 3 days later, lc_A's day rolled too
  # cache hit still runs the LLM nodes on cached derived facts (PRD node contract):
  assert len(fake_rank.calls) == 2 and len(fake_finalize.calls) == 2
  assert_no_leak(b2)

PRD TEST 3 — sixth billed quick check today refused:
GIVEN seeded_repo + seed_counted_quick_check(repo, lc_id=LC_A, created_at=clock.now(), n=5),
      fake for a NEW pan (PAN_TWO) so no cache;
WHEN  client.post quick_payload(pan=PAN_TWO)
THEN  r.status_code == 200; body['status'] == 'refused'; body['refusal_code'] == 'DAILY_CAP_REACHED'
  endpoints(fake_surepass) == []                    # zero SurePass calls
  body['quick_checks_remaining_today'] == 0
  len(body['talking_points']) == 1                  # one-line refusal message
  refusal row persisted: rows[-1].status == 'refused', billed False, counted_against_cap False
  assert_no_leak(body)

IST BOUNDARY (23:59 -> 00:01, same UTC date):
GIVEN 5 counted checks at 2026-10-08T23:00+05:30; clock.set_now(2026-10-08T23:59+05:30)
  -> admit refused DAILY_CAP_REACHED (via API, new PAN)
WHEN  clock.set_now(2026-10-09T00:01+05:30)        # UTC is still 2026-10-08 18:31!
THEN  same POST now succeeds: status 'ok', billed True, counted True,
      quick_checks_remaining_today == 4; proves MemoryRepo counts on Asia/Kolkata DATE of
      created_at, not UTC date (mirror of SQL quick_checks_used_today).

CAP DOES NOT BLOCK CACHE HITS:
GIVEN cache row fresh for PAN_OK + 5 counted checks today for LC_A
WHEN  POST quick PAN_OK -> status 'ok', from_cache True, remaining 0, zero SurePass calls.
(ordering pinned: consent -> cache -> cap -> live; see tests/test_admission_table.py)

### tests/test_pan_invalid.py
_PRD test 4 — invalid or non-individual PAN consumes nothing countable._

Parametrised over pan_invalid_scenario kinds:
  ('failure_envelope')  PAN comp returns success=False message_code='invalid_id'
  ('status_invalid')    success=True, data.status='invalid'
  ('company')           success=True, status='valid', category='company'
GIVEN seeded world, scenario for PAN under test; WHEN POST /checks/quick
THEN  r.status_code == 200; body['status'] == 'pan_invalid'; body['refusal_code'] is None
  endpoints(fake_surepass) == [PAN_ENDPOINT]        # exactly one call, NEVER the score endpoint
  row = one_check_row: row.status == 'pan_invalid'
  row.billed is True                                 # the PAN call WAS a SurePass spend (open item 3)
  row.counted_against_cap is False                   # never counts toward the 5
  seeded_repo.get_cache(pan_hash, 'quick') is None   # nothing cached: LC can correct and retry
  body['quick_checks_remaining_today'] == 5
  len(fake_rank.calls) == 0 and len(fake_finalize.calls) == 0   # LLM never reached
  assert_no_leak(body)
RETRY-AFTER-FIX: same test continues — re-POST with corrected PAN wired to happy envelopes:
  endpoints grow by [PAN_ENDPOINT, SCORE_ENDPOINT]; second row billed+counted True.

### tests/test_ntc_no_hit.py
_PRD test 5 — bureau no-hit on a valid PAN is consumed, cached as NTC, pushed to co-applicant._

Parametrised variants ('success_false', 'empty_score') of ntc_scenario (ASSUMPTION branch per
PRD 8.2/open item 4 — test both candidate shapes so the adapter decision is pinned):
GIVEN seeded world, PAN_NTC valid person (zip 560001), income 30000 in payload
WHEN  POST /checks/quick
THEN  endpoints == [PAN_ENDPOINT, SCORE_ENDPOINT]
  row.billed is True and row.counted_against_cap is True       # no-hit consumes the check
  cache = repo.get_cache(hash, 'quick'); cache.is_ntc is True; cache.expires_at == now+30d
  body['status'] == 'ok'; body['outcome'] == 'fundable_with_coapplicant'
  body['coapplicant_suggested'] is True
  # L2 (min_band 'NTC', min_income 20000) is the only NTC-eligible lender:
  body['top_lender']['id'] == 'L2'
  # ceiling L2 @30000: 0.45*30000=13500 @16%/36m -> 384,000; min 154,000 (precomputed)
  body['indicative_loan_range'] == {'min': 154000.0, 'max': 384000.0}
  body['quick_checks_remaining_today'] == 4
  assert_no_leak(body)
SECOND RUN SAME PAN within TTL: from_cache True, still outcome 'fundable_with_coapplicant',
  no new SurePass calls (NTC caching actually serves).

### tests/test_refresh.py
_PRD test 6 — refresh refused before TTL; after TTL billed but not counted._

GIVEN happy world; first quick check at T0 (billed+counted, remaining 4).
A) WHEN clock.advance(days=10); POST quick_payload(refresh=True)
   THEN body['status']=='refused'; body['refusal_code']=='CACHE_FRESH_NO_REFRESH'
     endpoints unchanged (2); refusal row persisted billed False counted False
     body['refresh_available_at'] == iso(T0 + 30 days)
B) WHEN clock.advance(days=21)   # now T0+31d, cache stale
     POST quick_payload(refresh=True)
   THEN endpoints == [PAN, SCORE, PAN, SCORE]        # re-billed, both calls again
     new row: billed True, is_refresh True, counted_against_cap False   # refresh never counts
     body['quick_checks_remaining_today'] == 5        # cap untouched on a fresh IST day
     body['from_cache'] is False
     cache.fetched_at == T0+31d and cache.expires_at == T0+61d          # row rewritten
C) STALE WITHOUT REFRESH FLAG (admission 'new'):
   fresh world, check at T0, clock.advance(days=31), POST refresh=False
   THEN billed True, counted_against_cap True, is_refresh False          # plain re-check counts

### tests/test_full_check.py
_PRD tests 7 and 8 — stage gate, pincode union with junk/stale filtering, per-opportunity cache._

PRD TEST 7 — stage gate:
@parametrize('stage', ['new','contacted','qualified','counselling'])
GIVEN seeded world; WHEN POST /checks/full full_payload(opportunity_stage=stage)
THEN body['status']=='refused'; body['refusal_code']=='STAGE_TOO_EARLY'; endpoints == [];
     row persisted billed False counted False.
@parametrize('stage', ['offer','registration','enrolled','won']) -> admitted (status 'ok').
Unknown stage 'telecalling' -> STAGE_TOO_EARLY (fail closed; flagged decision).

PRD TEST 8 — pincode union + opportunity cache:
GIVEN full_pincode_union_scenario: PAN zip 560002; report holder zips
      '110001'(recent), '20'(junk, recent), '400001'(stale Date_Reported 20250101);
      lender pincode seed FOR THIS TEST: L1={560002}, L2={110001}, L3={400001};
      payload income 60000, stage 'offer'.
WHEN  POST /checks/full
THEN  endpoints == [PAN_ENDPOINT, REPORT_ENDPOINT]      # report, not score, exactly once each
  internal = one_check_row(repo).internal
  set(internal['gate']['pincodes']) == {'560002','110001'}      # junk '20' regex-filtered,
                                                                 # '400001' dropped by 365d window
  {g['id'] for g in internal['gate']['lenders']} == {'L1','L2'}  # L3 NOT gated
  body['top_lender']['id'] == 'L1'     # both pass cutoffs (band 720-780, foir 5000/60000=0.083)
  # ceiling L1: headroom 0.55*60000-5000=28000 @13.5%/48m -> 1,034,000; min 414,000 (precomputed)
  body['indicative_loan_range'] == {'min': 414000.0, 'max': 1034000.0}
  'Confirm current pincode' in body['confirm_with_learner']      # PAN vs bureau mismatch flag
  row.billed True; row.counted_against_cap is False               # FULL NEVER counts vs quick cap
  repo.quick_checks_used_today(LC_A) == 0
SECOND CALL same opportunity, same PAN, clock.advance(days=1):
  body2['from_cache'] is True; endpoints unchanged; second row billed False counted False.
FULL-AFTER-QUICK (policy 8 'both cases'): run a quick first (score cache exists), then full:
  full still calls [PAN?]+[REPORT] — quick cache does NOT satisfy full (kind-separate cache):
  endpoints for the full run include REPORT_ENDPOINT; identity may come from quick cache row
  (assert PAN_ENDPOINT called 0 or 1 times — pin to repo cache design: identity cached under
  quick payload => PAN NOT re-called; assert endpoints[-1] == REPORT_ENDPOINT and
  REPORT_ENDPOINT in endpoints exactly once).
assert_no_leak on every body; no raw ZIP junk strings in lc_view.

### tests/test_full_emi_derivation.py
_PRD test 9 — all open-account EMIs empty: estimate is the norm; declared-vs-derived max rule._

GIVEN full_all_emi_empty_scenario (two open accounts EMI '', one closed), income 80000,
      stage 'offer', lenders L1/L2 on 560001.
Adapter-level (direct unit, no graph):
  rep = adapt_experian_report(envelope)
  rep.existing_emi_reported == 0.0
  rep.existing_emi_floor is True
  rep.open_accounts_missing_emi == [
    {balance: 120000.0, tenure_months: 24, rate: 34.9, account_type: '05'},   # '024' zero-pad parsed
    {balance: 90000.0,  tenure_months: None, rate: None, account_type: '05'}] # '0'/'' -> unknown
Graph-level, parametrised declared EMI:
  (declared=None)  internal['applicant']['existing_emi'] == pytest.approx(10091.892055687149)
      # = EMI(120000, 34.9%, 24m) 7015.905377464855 + EMI(90000, 14%, 36m) 3075.9866782222953
      #   (defaults 36m / 14% p.a. applied to the unknown account — PRD 8.3 ASSUMPTION)
  (declared=2000)  existing_emi == pytest.approx(10091.892055687149)   # bureau-derived wins (higher)
  (declared=50000) existing_emi == 50000.0                              # LC-declared wins (higher)
  internal['applicant']['existing_emi_floor'] is True in all three
  body['confidence'] == 'medium' (floor flag) and 'Confirm existing EMIs' in confirm_with_learner
  (declared=50000) L1 headroom 0.55*80000-50000 = -6000 -> ceiling 0 -> FakeLLM default yields
      indicative_loan_range {'min': None,'max': None} via ceiling 0 -> assert max in (None, 0.0)
      — pin exact choice to affordability section's Affordability.ceilings contract.
  (declared=None) ceiling L1 = 1,252,000 (precomputed: headroom 33908.1079 @13.5%/48m).
assert_no_leak every body (estimates like 10091.89 must never appear in LC strings;
  'Confirm existing EMIs' is the only EMI surface).

### tests/test_llm_misbehavior.py
_PRD test 10 — invented/ungated lender dropped; rank retry-on-empty; finalize retry then deterministic fallback._

CASE A — invented + ungated lender dropped, no retry needed:
GIVEN happy quick world; fake_rank = FakeRankLLM(script=[bad_ranking(['L_FAKE','L3','L1'])])
WHEN POST quick
THEN internal['ranking']['accepted'] == ['L1']
  {d['lender_id'] for d in internal['ranking']['dropped']} == {'L_FAKE','L3'}
  each dropped entry has non-empty 'reason' ('not_gated' / 'failed_cutoffs' vocabulary pinned
  to graph section); body['top_lender']['id'] == 'L1'; len(fake_rank.calls) == 1.

CASE B — rank accepted empty => exactly one retry with errors fed back:
fake_rank script = [bad_ranking(['L_FAKE']), bad_ranking(['L_FAKE','L1'])]
THEN len(fake_rank.calls) == 2
  fake_rank.calls[1].attempt == 2 and fake_rank.calls[1].validation_errors != []
  final top 'L1'.
CASE B2 — both attempts empty: script=[bad_ranking(['X']), bad_ranking(['Y'])]
  -> ranking.accepted == []; finalize default yields outcome 'review', confidence 'low'.

CASE C — finalize keeps naming a dropped lender => fallback after 2 retries:
fake_finalize = FakeFinalizeLLM(script=[bad_final('L_FAKE')])   # last entry repeats forever
THEN len(fake_finalize.calls) == 3                 # initial + 2 retries, then NO 4th call
  fake_finalize.calls[1].validation_errors mentions 'L_FAKE'
  body['outcome'] == 'review'; body['confidence'] == 'low'
  body['top_lender'] is None and body['alternates'] == []
  body['talking_points'] == [FALLBACK_MESSAGE]      # fixed deterministic string
  row.status == 'ok'; row.billed True; row.counted_against_cap True   # spend already happened
CASE D — finalize names valid top but 3 alternates / min>max: pydantic rejects 3 alternates at
  construction (max_length=2) — test FinalRecommendation model directly with pytest.raises(ValidationError);
  min>max passes pydantic but fails code validation -> retry path (script[bad_min_max, good]) ->
  len(calls)==2, final ok.

### tests/test_redaction.py
_PRD test 11 plus scrub_lines unit battery with INR false-positive cases._

GRAPH-LEVEL (PRD 11):
GIVEN happy quick world;
  fake_finalize = FakeFinalizeLLM(
    talking_points_override=['Experian shows a 736 score, proceed',
                             'Strong profile, proceed with the pitch'],
    confirm_override=['Confirm current pincode', 'Score 736 to be rechecked'])
WHEN POST quick
THEN body['talking_points'] == ['Strong profile, proceed with the pitch']   # leaky line scrubbed
  body['confirm_with_learner'] == ['Confirm current pincode']
  assert_no_leak(body); row.lc_view reflects the scrubbed lists (persisted AFTER redact).

UNIT scrub_lines (from eligibility_agent.output.lc_view import scrub_lines):
  KEPT : 'Existing EMI of ₹3,500 is manageable'       # ',500' guarded by comma
  KEPT : 'EMIs total ₹13,500 against income'
  KEPT : 'FOIR is 0.45 with headroom'                        # '736'-free decimals safe: '0.736' also KEPT
  KEPT : 'About 250 rupees short'                            # 250 < 300
  KEPT : 'Policy code 905 applies'                           # 905 > 899
  KEPT : 'Loan up to 4,36,000 possible'                      # Indian grouping: no standalone 3-digit run
  DROP : 'score is 736'                                      # bare 300-899
  DROP : 'Experian data is strong'                           # bureau word, case-insensitive
  DROP : 'the BUREAU indicates stress'
  DROP : 'CIBIL/CRIF/Equifax' (parametrised each word)
  DROP : 'tenure of 360 months works'                        # 360 standalone — by-design strictness
  DROP : '₹899 only'                                    # currency sign is not a digit guard
  KEPT : 'crifled lines of text'                             # \b keeps substrings safe
  KEPT(flagged risk): 'bureaus differ'                       # plural NOT matched by FIXED \b(bureau)\b —
        asserted KEPT to pin current FIXED behaviour; see prd_gaps for strengthen question.
EXACT regexes under test (must be module constants in output/lc_view.py):
  BUREAU_RE = re.compile(r'(?i)\\b(cibil|experian|equifax|crif|bureau)\\b')
  SCORE_RE  = re.compile(r'(?<![\\d,.])[3-8][0-9]{2}(?![\\d,.])')
PAN-AADHAAR VISIBILITY: build LcView from state with identity.aadhaar_linked False ->
  pan_aadhaar_link_issue is True; linked True or identity None -> None (never False).

### tests/test_live_kill_switch.py
_PRD test 12 — SUREPASS_LIVE unset makes live calls impossible and the API refuse cleanly._

UNIT (real client class):
GIVEN os.environ without SUREPASS_LIVE (guaranteed by pytest_configure);
  cfg = Config.from_env(); assert cfg.surepass_live is False
  client_ = SurePassClient(token='dummy', live=cfg.surepass_live)
THEN each of client_.pan_comprehensive(PAN_OK), client_.experian_score(...),
  client_.experian_report(...) raises LiveCallsDisabled BEFORE any socket use
  (pytest-socket would hard-fail the suite otherwise — double enforcement).
Parametrise SUREPASS_LIVE in {'false','TRUE','1','yes',''} via monkeypatch.setenv ->
  surepass_live False for ALL (only the exact lowercase string 'true' enables).

API-LEVEL (section 13 acceptance):
GIVEN graph_deps built with the REAL SurePassClient(live=False) and live_calls_enabled=False
WHEN POST /checks/quick (no cache)
THEN 200; body['status']=='refused'; body['refusal_code']=='LIVE_CALLS_DISABLED';
  zero surepass_calls rows; refusal row persisted billed False counted False.
CACHE STILL SERVES: pre-seed fresh cache row -> POST same PAN -> status 'ok', from_cache True
  (kill switch blocks spend, not cached reads).
HEALTH: anon GET /health -> 200 {'ok': True, 'surepass_live': False}.

### tests/test_adapters.py
_PRD test 13 — adapter unit batteries for PAN Comprehensive, Experian score, Experian report._

class TestPanAdapter:
  valid person -> Identity(pan_valid=True, full_name='RAVI TESTKUMAR', gender='male',
      dob=date(1998,4,12), aadhaar_linked=True, pan_pincode='560001',
      surepass_client_id='pan_comprehensive_synthetic_0001')
  gender 'F' -> 'female'; gender 'T'/'' -> None
  category 'company' -> pan_valid False;  status 'invalid' -> pan_valid False
  success False envelope -> pan_valid False;  envelope WITHOUT success key (validation shape
      {errors,message}) -> pan_valid False, no KeyError
  zip '20' -> pan_pincode None; zip '060001' (leading zero) -> None; zip '560001 ' -> None
      (regex ^[1-9][0-9]{5}$ full-match, no strip beyond explicit .strip() decision)
  aadhaar_linked False -> Identity.aadhaar_linked False
  masked_aadhaar NEVER on Identity model: assert not hasattr(identity, 'masked_aadhaar')

class TestScoreAdapter:
  '745' -> score 745, band '720-780';  report_date '20261001' -> date(2026,10,1)
  BAND EDGES (pins bucket convention [650,720), [720,780), [780,inf)):
    649->'<650'; 650->'650-720'; 719->'650-720'; 720->'720-780'; 779->'720-780'; 780->'780+'
  data.credit_score missing but SCORE.FCIREXScore '745' -> 745 (fallback path)
  '' -> band 'NTC', score None;  'abc' -> NTC;  success False no_record_found -> NTC  (ASSUMPTION)

class TestReportAdapter (fixture: 6-account synthetic report):
  OPEN PREDICATE: Date_Closed=='' OR Account_Status=='11' ->
    a(open: '', '11') open; a('20240110','13') closed; a('20240110','11') OPEN (OR semantics pinned);
    a('', '13') open via Date_Closed==''; Account_Status '15' with Date_Closed set -> closed
  existing_emi_reported sums only non-'' EMIs over open accounts; '' on any open -> floor True
  open_accounts_missing_emi rows: Terms_Duration '024' -> 24; Repayment_Tenure '0' + Terms '' -> None;
    Rate_of_Interest '' -> None; '34.900' -> 34.9
  total_outstanding = sum Current_Balance over open; past_due_total over ALL accounts;
  max_dpd_24m = max Days_Past_Due over CAIS_Account_History (rows '0','30','65','?' -> non-numeric
    ignored) == 65; written_off_or_settled == count of status '99' rows
  enquiries_180d: '2' -> 2
  bureau_pincodes: {'110001'} given zips '110001'(recent), '20'(junk), '400001'(stale);
    365-day window measured against clock.now() passed into the adapter
  band from data.credit_score '702' -> '650-720'
  DOB: adapter exposes NO dob field from CAIS_Holder_Details (model has no such attribute)

### tests/test_affordability_math.py
_Estimator and ceiling formulas pinned to closed-form values; check_hard_cutoffs unknown-fact semantics._

estimate_emi(balance, tenure_months, annual_rate_pct) — reducing balance:
  EMI = B*r*(1+r)^n/((1+r)^n-1), r=rate/1200; tenure None->36, rate None->14.0 (PRD 8.3 defaults)
  estimate_emi(120000, 24, 34.9) == pytest.approx(7015.905377464855)
  estimate_emi(90000, None, None) == pytest.approx(3075.9866782222953)
  estimate_emi(100000, 36, 14.0) == pytest.approx(3417.7629758025505)
  estimate_emi(0, 36, 14.0) == 0.0
loan_ceiling(monthly_income, existing_emi, cutoffs) -> float | None  (PRD 8.4):
  headroom = max_foir*income - existing_emi; <=0 -> 0.0; income None -> None;
  P = headroom*((1+r)^n - 1)/(r*(1+r)^n) with lender annual_rate_pct (default 14) and
  max_tenure_months (default 36); round to nearest 1000.
  (60000, 0, foir .55, 13.5%, 48m)  -> 1_219_000.0
  (60000, 15000, foir .5, 14%, 36m) ->   439_000.0
  (60000, 0, foir .55, defaults)    ->   966_000.0
  (30000, 20000, foir .45, ...)     ->         0.0   # headroom -6500
  (None income)                     ->        None
FOIR: applicant.foir == existing_emi/monthly_income when both known else None;
  foir(0 income) -> None (no ZeroDivisionError).
check_hard_cutoffs(lender, applicant) — UNKNOWN NEVER FAILS:
  decision table rows (band 720-780, age 28, income 60000, foir 0.25 baseline vs L1):
    all known & inside -> passed True, failures []
    band '<650' vs min_band '650-720' -> failed ['min_band']
    foir .6 vs max_foir .55 -> failed ['max_foir']
    income None -> min_income NOT failed; foir None -> max_foir NOT failed;
    age None -> age cutoffs NOT failed (passed True, but CutoffResult.unknown_facts lists them)
    age 57 + max_tenure 48 vs max_age_at_maturity 58 -> failed ['max_age_at_maturity'] (57+4=61)
  BAND ORDER pinned: NTC(0) < '<650'(1) < '650-720'(2) < '720-780'(3) < '780+'(4);
    min_band 'NTC' accepts everyone; NTC applicant fails any min_band above 'NTC'.

### tests/test_admission_table.py
_Direct unit tests of admit_quick_check / admit_full_check against the full decision table (guards called with MemoryRepo + FrozenClock, no graph)._

QUICK TABLE (each row one test case; columns: pre-state -> admission/refusal, billed, counts):
| consent row absent                      | CONSENT_MISSING          -    -    |
| cache fresh, refresh F                  | cache_hit              False False |
| cache fresh, refresh T                  | CACHE_FRESH_NO_REFRESH   -    -    |
| cache stale, refresh T                  | refresh                True  False |
| cache stale, refresh F                  | new                    True  True  |
| no cache, refresh T                     | new                    True  True  | # refresh flag moot (decision)
| no cache, used_today 5                  | DAILY_CAP_REACHED        -    -    |
| no cache, used_today 4                  | new                    True  True  |
| cache fresh + used_today 5              | cache_hit              False False | # cap checked only when counting
| cache stale, refresh T, used_today 5    | refresh                True  False | # refresh exempt from cap
| no cache, live_calls_enabled False      | LIVE_CALLS_DISABLED      -    -    |
| cache fresh, live False                 | cache_hit              False False |
FULL TABLE:
| stage 'counselling'                     | STAGE_TOO_EARLY                    |
| stage 'offer', no cache, no prior       | new                    True  False | # full NEVER counts
| cache(full) fresh, refresh F            | cache_hit              False False |
| cache fresh, refresh T                  | CACHE_FRESH_NO_REFRESH             |
| cache stale, prior ok full on opp, refresh F | OPPORTUNITY_ALREADY_CHECKED   | # decision, see prd_gaps
| cache stale, prior ok full, refresh T   | refresh                True  False |
| stage unknown string                    | STAGE_TOO_EARLY                    | # fail closed
| quick cache exists but no full cache    | new (quick cache does NOT satisfy full) |
ORDER PINNED: consent -> stage(full only) -> cache -> opportunity(full) -> cap(quick only) -> live.
Each case asserts the AdmissionDecision tuple exactly:
  (admission, refusal_code, billed, counts_against_cap).

### tests/test_api_auth_and_validation.py
_Auth middleware and request-schema validation edges._

AUTH: anon POST /checks/quick -> 401; wrong token 'Bearer nope' -> 401; correct -> not 401.
  anon GET /health -> 200 {'ok': True, 'surepass_live': False}  (only unauthenticated route).
VALIDATION (all -> 422, zero repo rows, zero SurePass calls):
  pan 'abcpe1234f' (lowercase), 'ABCPE12345' (no trailing letter), 'ABCP1234F' (4+4+1)
      — regex ^[A-Z]{5}[0-9]{4}[A-Z]$
  mobile '5876543210' (leading 5), '98765432101' (11 digits) — regex ^[6-9][0-9]{9}$
  consent false / omitted — schema Literal[True]
  quick: name/mobile/pan/lc_id/lead_id required; monthly_income negative -> 422 (ge=0)
  full: opportunity_id, opportunity_stage, monthly_income required -> each omission 422
  unknown opportunity_stage string passes schema (str) but refuses STAGE_TOO_EARLY (graph-level).
RESPONSE SHAPE: 200 bodies validate against LcView.model_validate(body) round-trip for
  ok, refused and pan_invalid statuses (keeps SuperConnect/FulCRM contract frozen).

### tests/test_envelope_builders.py
_Self-test the synthetic builders so fixture drift cannot silently invalidate the suite._

cais_account() returns ALL 23 top-level field-map keys; every unset scalar is '' (dense convention);
  json.dumps round-trips; Terms_Duration default '' and explicit '024' kept zero-padded (str).
pan_comprehensive_envelope: envelope keys exactly {data,status_code,success,message,message_code};
  data.address.zip == '560001'; aadhaar_linked is bool (not str).
validation_error_envelope: body has NO 'success' and NO 'data' key (field-map validation shape).
experian_score_envelope(success=False): body['success'] is False, message_code 'no_record_found'.
GUARD AGAINST REAL DATA: assert probe-derived literals never appear in any builder output:
  for forbidden in ('791111','201301','560071','16559','2434004','138696','109433'):
      assert forbidden not in json.dumps(all_default_envelopes)
  ('736'/'761'/'830'/'743' excluded from this guard: 3-digit strings collide with synthetic
   amounts; score defaults are pinned to '745'/'702'/'628'/'610' instead.)

### eligibility_agent/evals/evaluators.py
_The five PRD section-11 evaluators in langsmith SDK v0.2+ kwargs-signature form, plus the LLM-judge rubric._

import json, re
from typing import Any

# langsmith >=0.2 selects evaluator args BY PARAMETER NAME (inputs/outputs/reference_outputs);
# return an EvaluationResult-shaped dict {'key': str, 'score': bool|float, 'comment': str}.
# Target outputs contract (produced by evals/run.py target):
#   {'lc_view': dict, 'billed': bool, 'counted_against_cap': bool, 'from_cache': bool,
#    'admission': str, 'surepass_call_endpoints': list[str], 'gated_lender_ids': list[str],
#    'accepted_lender_ids': list[str], 'band': str}
# (band/gated ids are SYNTHETIC-dataset metadata; permissible in LangSmith because no real
#  credit data ever enters evals — enforced by the builders.)

BUREAU_RE = re.compile(r"(?i)\b(cibil|experian|equifax|crif|bureau)\b")
SCORE_RE = re.compile(r"(?<![\d,.])[3-8][0-9]{2}(?![\d,.])")

def no_leak(inputs: dict[str, Any], outputs: dict[str, Any],
            reference_outputs: dict[str, Any]) -> dict[str, Any]:
    blob = json.dumps(outputs["lc_view"], ensure_ascii=False)
    bad = bool(BUREAU_RE.search(blob)) or any(
        SCORE_RE.search(s) for s in _string_leaves(outputs["lc_view"]))
    return {"key": "no_leak", "score": not bad}

def lenders_gated(inputs: dict[str, Any], outputs: dict[str, Any],
                  reference_outputs: dict[str, Any]) -> dict[str, Any]:
    lv = outputs["lc_view"]
    named = {l["id"] for l in ([lv["top_lender"]] if lv.get("top_lender") else [])} | {
        a["id"] for a in lv.get("alternates") or []}
    ok = named <= set(outputs["accepted_lender_ids"]) <= set(outputs["gated_lender_ids"])
    return {"key": "lenders_gated", "score": ok, "comment": f"named={sorted(named)}"}

def policy_consistency(inputs: dict[str, Any], outputs: dict[str, Any],
                       reference_outputs: dict[str, Any]) -> dict[str, Any]:
    calls = outputs["surepass_call_endpoints"]; lv = outputs["lc_view"]
    checks = [
        outputs["billed"] == (len(calls) > 0),
        (not outputs["from_cache"]) or calls == [],            # cache hit => zero calls
        (lv["status"] != "refused") or calls == [],             # refusal => zero calls
        (not outputs["counted_against_cap"]) or outputs["billed"],
        outputs["billed"] == reference_outputs.get("billed", outputs["billed"]),
        outputs["counted_against_cap"] == reference_outputs.get(
            "counted_against_cap", outputs["counted_against_cap"]),
        outputs["from_cache"] == reference_outputs.get("from_cache", outputs["from_cache"]),
    ]
    return {"key": "policy_consistency", "score": all(checks)}

def outcome_matches_band(inputs: dict[str, Any], outputs: dict[str, Any],
                         reference_outputs: dict[str, Any]) -> dict[str, Any]:
    lv = outputs["lc_view"]; band = outputs["band"]
    violation = (band in ("NTC", "<650") and lv.get("outcome") == "fundable"
                 and not lv.get("coapplicant_suggested"))
    coherent = lv.get("outcome") != "fundable_with_coapplicant" or lv.get("coapplicant_suggested")
    return {"key": "outcome_matches_band", "score": (not violation) and bool(coherent)}

JUDGE_RUBRIC = '''You are reviewing one talking-points list a financing agent wrote for a
sales consultant (LC) who is live on a call with a learner in India. Score 1 only if ALL hold:
1. Plain language a non-finance person says aloud naturally; no jargon (no 'FOIR', 'DPD',
   'tradeline', 'bureau', 'delinquency', 'headroom').
2. Band-driven, not number-driven: profile strength in words ('strong profile',
   'existing EMIs are high relative to income'); NO credit score, NO 3-digit number,
   NO bureau or lender-internal terminology.
3. Actionable for the call: tells the LC what to do next (proceed / suggest co-applicant /
   confirm a fact with the learner / raise for review).
4. <= 4 points, each <= 200 characters, INR context only.
5. Never promises approval; recommendation wording only ('indicative', 'suggest', 'likely').
Return JSON {"score": 0 or 1, "reasons": "<one line per failed criterion>"}.'''

def make_talking_points_judge(model: str = "gpt-4.1") -> "Callable[..., dict[str, Any]]":
    """Returns an evaluator closure using langchain_openai.ChatOpenAI(temperature=0)
    .with_structured_output(JudgeVerdict). Only constructed when OPENAI_API_KEY is set;
    evals/run.py omits it otherwise (CI secrets gate)."""
    class JudgeVerdict(BaseModel):
        score: int  # 0 or 1
        reasons: str
    def talking_points_judge(inputs: dict[str, Any], outputs: dict[str, Any],
                             reference_outputs: dict[str, Any]) -> dict[str, Any]:
        pts = outputs["lc_view"].get("talking_points") or []
        v = judge_llm.invoke(JUDGE_RUBRIC + "\n\nTalking points:\n" + json.dumps(pts))
        return {"key": "talking_points_quality", "score": float(v.score), "comment": v.reasons}
    return talking_points_judge

DETERMINISTIC_EVALUATORS = [no_leak, lenders_gated, policy_consistency, outcome_matches_band]


### eligibility_agent/evals/datasets.py
_Golden example definitions (pure data, synthetic only) and LangSmith dataset upload for eligibility-quick-golden / eligibility-full-golden._

from typing import Any, TypedDict

QUICK_DATASET = "eligibility-quick-golden"; FULL_DATASET = "eligibility-full-golden"

class GoldenExample(TypedDict):
    example_id: str
    kind: str                      # 'quick' | 'full'
    request: dict[str, Any]        # the API payload
    lenders: list[dict[str, Any]]  # lender seed rows incl. pincodes/hard_cutoffs/criteria
    surepass: dict[str, Any]       # {'pan': EnvelopePair, 'score': ..., 'report': ...} per PAN
    pre_state: dict[str, Any]      # {'cache_rows': [...], 'prior_checks': [...]} or {}
    expected: dict[str, Any]       # reference_outputs: LcView invariants + policy flags

def quick_examples() -> list[GoldenExample]:
    """10 examples, ALL built from eligibility_agent.testing.envelopes (synthetic only):
    q01 happy_745          expected: status ok, outcome fundable, billed T, counted T, from_cache F
    q02 ntc_success_false  outcome fundable_with_coapplicant, coapplicant True, billed/counted T
    q03 ntc_empty_score    same as q02 (pins both ASSUMPTION shapes)
    q04 band_610           outcome in {fundable_with_coapplicant, not_fundable}, never bare fundable
    q05 pan_company        status pan_invalid, billed T, counted F
    q06 pan_failure_env    status pan_invalid, billed T, counted F
    q07 cache_hit          pre_state cache row -> from_cache T, billed F, counted F
    q08 cap_reached        pre_state 5 counted checks -> refused DAILY_CAP_REACHED
    q09 no_lender_gate     pincode with no lender rows -> confidence low, top_lender None
    q10 no_income          confidence medium, 'Declared monthly income' in confirm_with_learner"""

def full_examples() -> list[GoldenExample]:
    """8 examples:
    f01 happy_offer        status ok, billed T, counted F (full never counts)
    f02 stage_too_early    refused STAGE_TOO_EARLY, zero calls
    f03 pincode_union      'Confirm current pincode' expected in confirm_with_learner
    f04 all_emi_empty      confidence medium, 'Confirm existing EMIs' expected
    f05 delinquency_628    outcome in {fundable_with_coapplicant, review}; never fundable
    f06 opp_second_run     pre_state full cache -> from_cache T
    f07 ntc_full           coapplicant True
    f08 declared_emi_wins  internal max(declared, derived) — expected ceiling band reflects 50k EMI"""

def build_langsmith_datasets(client: "langsmith.Client", *, overwrite: bool = False) -> None:
    """create_dataset if missing (read via list_datasets(dataset_name=...)); when overwrite,
    delete+recreate. create_examples(inputs=[{k: ex[k] for k in
    ('kind','request','lenders','surepass','pre_state')}...], outputs=[ex['expected']...],
    dataset_id=...). Never called from pytest (network); invoked by evals/run.py --mode langsmith
    and documented in README. The real probe folder surepass-lab/out/bureau is NEVER read here."""


### eligibility_agent/evals/run.py
_CLI eval runner: local mode (no network, CI-default) and langsmith mode (evaluate() with fake LLM for the bulk and real model for max 20)._

"""python -m eligibility_agent.evals.run --mode {local,langsmith} --llm {fake,real}
                                         [--max-real 20] [--dataset quick|full|both]
                                         [--build-datasets] [--experiment-prefix s]"""
from typing import Any, Literal

EVAL_T0 = datetime(2026, 10, 8, 10, 0, tzinfo=ZoneInfo('Asia/Kolkata'))

def run_one(example: GoldenExample, llm: Literal['fake','real']) -> dict[str, Any]:
    """Pure in-process run: FrozenClock(EVAL_T0); MemoryRepo(clock); seed example['lenders'];
    apply pre_state (cache rows via repo.upsert_cache, prior checks via repo.insert_check);
    FakeSurePassClient(**example['surepass']); llm=='fake' -> FakeRankLLM/FakeFinalizeLLM,
    llm=='real' -> graph section's make_rank_llm(cfg)/make_finalize_llm(cfg) with repo-local
    pinned prompts (PROMPT_SOURCE=local; Hub never pulled in CI);
    graph = build_graph(example['kind'], deps); state = graph.invoke(initial_state(request));
    return the evaluator outputs contract:
      {'lc_view', 'billed', 'counted_against_cap', 'from_cache', 'admission',
       'surepass_call_endpoints', 'gated_lender_ids', 'accepted_lender_ids', 'band'}
    (read from the persisted check row + fake client call log)."""

def run_local(dataset: str, llm: str) -> int:
    """No LangSmith: iterate examples, call run_one, apply DETERMINISTIC_EVALUATORS directly
    (they are plain functions), print a table, exit 1 on any score falsy. This is the
    always-on CI job and needs no secrets, no sockets (asserts ONLINE==False)."""

def run_langsmith(dataset_name: str, llm: str, max_real: int, prefix: str) -> None:
    """from langsmith import Client; from langsmith.evaluation import evaluate
    Requires LANGSMITH_API_KEY. --build-datasets first calls build_langsmith_datasets.
    Split per PRD 11: first max_real (default 20) examples -> target with REAL model,
    experiment_prefix=f'{prefix}-real'; remainder -> FakeLLM target, prefix f'{prefix}-fake'.
    evaluate(target, data=client.list_examples(dataset_name=..., splits/ids slice),
             evaluators=DETERMINISTIC_EVALUATORS + ([make_talking_points_judge()] if
             os.environ.get('OPENAI_API_KEY') else []), max_concurrency=4,
             metadata={'llm': llm, 'git_sha': env GITHUB_SHA}).
    Target signature evaluate expects: Callable[[dict], dict] — wrap run_one with the
    inputs dict (inputs ARE the GoldenExample minus expected)."""

def main(argv: list[str] | None = None) -> int: ...  # argparse wiring of the above


### .github/workflows/ci.yml
_CI: tests+mypy always; local evals always; LangSmith/real-model evals only when secrets exist._

name: ci
on: {push: {branches: [main]}, pull_request: {}}
env: {LANGSMITH_TRACING: 'false', LANGCHAIN_TRACING_V2: 'false'}  # tracing no-op in CI tests
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: {python-version: '3.12'}
      - run: uv sync --frozen --all-groups
      - run: uv run mypy                       # config in pyproject; strict, covers tests/
      - run: uv run pytest -q                   # addopts enforce --disable-socket
      # SUREPASS_LIVE deliberately never set in CI: acceptance 'live calls impossible by default'
  evals-local:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: {python-version: '3.12'}
      - run: uv sync --frozen --all-groups
      - run: uv run python -m eligibility_agent.evals.run --mode local --llm fake --dataset both
  evals-langsmith:
    needs: test
    runs-on: ubuntu-latest
    # secrets cannot appear in job-level `if`; gate via env indirection:
    env:
      LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - id: gate
        run: |
          if [ -n "$LANGSMITH_API_KEY" ] && [ -n "$OPENAI_API_KEY" ]; then
            echo "run=true" >> "$GITHUB_OUTPUT"; fi
      - if: steps.gate.outputs.run == 'true'
        uses: astral-sh/setup-uv@v5
        with: {python-version: '3.12'}
      - if: steps.gate.outputs.run == 'true'
        run: uv sync --frozen --all-groups
      - if: steps.gate.outputs.run == 'true'
        run: >-
          uv run python -m eligibility_agent.evals.run --mode langsmith --llm real
          --max-real 20 --dataset both --build-datasets
          --experiment-prefix ci-${{ github.sha }}


### pyproject.toml
_MY OWNED BLOCKS ONLY (merge with other sections): pytest config with no-network addopts, mypy --strict config, dev dependency group._

[dependency-groups]              # uv-native dev group
dev = [
  "pytest>=8.3",
  "pytest-socket>=0.7",         # suite-wide network ban
  "mypy>=1.14",
  "httpx>=0.27",                # starlette TestClient transport (in-process ASGI, no sockets)
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers --disable-socket --allow-unix-socket"
# --allow-unix-socket: anyio portal in TestClient needs socket.socketpair(AF_UNIX);
# TCP/UDP (any real network) stays blocked for the entire suite.
filterwarnings = ["error::pytest_socket.SocketBlockedError"]
markers = ["eval: local golden-dataset eval runs (excluded from default CI pytest if slow)"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
enable_error_code = ["ignore-without-code", "redundant-expr", "possibly-undefined"]
plugins = ["pydantic.mypy"]
files = ["eligibility_agent", "tests", "scripts"]   # tests are typed too (fixtures annotated)

[tool.pydantic-mypy]
init_typed = true
init_forbid_extra = true
warn_required_dynamic_aliases = true

# KNOWN STUB GAPS — narrowest escape hatches, all documented:
# supabase-py family ships no py.typed; everything it returns is validated into Pydantic
# models inside SupabaseRepo (the ONLY module importing it), so Any never crosses repo.py.
[[tool.mypy.overrides]]
module = ["supabase", "supabase.*", "postgrest", "postgrest.*", "gotrue.*",
          "realtime.*", "storage3.*"]
ignore_missing_imports = true
# pytest_socket has no stubs; only referenced in filterwarnings + one import in conftest guard.
[[tool.mypy.overrides]]
module = ["pytest_socket"]
ignore_missing_imports = true
# langgraph, langchain-core, langchain-openai, langsmith, fastapi, pydantic all ship py.typed:
# NO blanket overrides for them; if a specific symbol is untyped use a single
# `# type: ignore[<code>]` with a comment, never ignore_errors / follow_imports=skip.
# TYPING PATTERNS pinned for implementers:
#  - CheckState stays TypedDict(total=False); nodes narrow with explicit `assert x is not None`
#    or guard clauses (mypy-strict-safe), never `cast` on state reads.
#  - DI seams are typing.Protocol (Repo, Clock, SurePassLike, RankLLM/FinalizeLLM callables);
#    fakes conform structurally — zero inheritance, zero registration.
#  - Supabase boundary: `raw: Any = resp.data` then `Model.model_validate(raw)` immediately;
#    cast() only at that single boundary line with a comment.


### scripts/seed_lenders.py
_Idempotent seeding of lenders + lender_pincodes with three complete synthetic lenders; --dry-run prints SQL, otherwise upserts through SupabaseRepo._

"""usage: uv run python scripts/seed_lenders.py [--dry-run] [--env .env]
Upserts SEED_LENDERS via repo.upsert_lender(lender, pincodes); --dry-run prints the
equivalent INSERT ... ON CONFLICT (id) DO UPDATE statements without connecting."""
SEED_LENDERS: list[dict[str, object]] = [
  {"id": "alpha_edu", "name": "Alpha Education Finance", "active": True,
   "hard_cutoffs": {"min_band": "650-720", "max_foir": 0.55, "min_income": 25000,
     "min_age": 21, "max_age_at_maturity": 58, "max_tenure_months": 48,
     "annual_rate_pct": 13.5},
   "criteria": {"summary": "Mainstream education NBFC; prefers salaried applicants with a
       stable band and clean repayment history; fast sanction for upskilling courses.",
     "employment": ["salaried"], "coapplicant": "optional; strengthens thin FOIR headroom",
     "course_categories": ["technology", "management", "finance"],
     "notes": "Prefers tenure under 4 years; moratorium not offered."},
   "pincodes": ["560001", "560002", "560003", "110001", "110002", "600001"]},
  {"id": "bharat_upskill", "name": "Bharat Upskill Credit", "active": True,
   "hard_cutoffs": {"min_band": "NTC", "max_foir": 0.45, "min_income": 20000,
     "min_age": 21, "max_age_at_maturity": 60, "max_tenure_months": 36,
     "annual_rate_pct": 16.0},
   "criteria": {"summary": "New-to-credit friendly lender for first-time borrowers;
       co-applicant (working parent or spouse) mandatory for NTC and sub-650 profiles.",
     "employment": ["salaried", "self-employed"],
     "coapplicant": "mandatory for NTC and <650 band",
     "course_categories": ["technology", "healthcare", "vocational"],
     "notes": "Higher rate compensates thin files; income proof of co-applicant required."},
   "pincodes": ["560001", "400001", "400002", "110001", "500001"]},
  {"id": "crest_capital", "name": "Crest Capital NBFC", "active": True,
   "hard_cutoffs": {"min_band": "720-780", "max_foir": 0.6, "min_income": 50000,
     "min_age": 23, "max_age_at_maturity": 55, "max_tenure_months": 60,
     "annual_rate_pct": 11.75},
   "criteria": {"summary": "Premium-band lender; lowest rate, longest tenure; strict on
       delinquency — any write-off or recent 30+ DPD is a decline regardless of band.",
     "employment": ["salaried"], "coapplicant": "not accepted as primary mitigation",
     "course_categories": ["management", "technology"],
     "notes": "Metro pincodes only; prefers declared income with payslip at application."},
   "pincodes": ["560001", "110001", "400001", "600001"]},
]
# Also used by tests/helpers.seed_default_lenders (ids remapped L1/L2/L3) so test lenders
# and documented seeds never drift. All pincodes are generic GPO codes, not probe values.
def main(argv: list[str] | None = None) -> int: ...


### README.md
_README outline per acceptance criteria: env vars, uv setup, seeding, running tests/mypy/evals, open items._

# Financing Eligibility Agent (Phase 1) — outline
## 1. What this is
3 sentences + the LcView redaction promise (LC never sees score/bureau/raw report).
## 2. Requirements
Python 3.12 (uv-managed), Supabase project, SurePass production token (IP-whitelisted), keys below.
## 3. Environment variables (table)
| var | required | default | notes |
| SERVICE_API_TOKEN | yes | — | bearer token checked on every route except /health |
| PAN_SALT | yes | — | >=16 chars; salted SHA-256 of PAN for pan_cache/checks keys |
| SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY | prod | — | service-role only tables |
| DATABASE_URL | prod | — | Postgres DSN for langgraph-checkpoint-postgres |
| SUREPASS_TOKEN | prod | — | server-side only, never shipped to clients |
| SUREPASS_LIVE | no | unset | ONLY the literal string 'true' enables spend; anything else -> LiveCallsDisabled + LIVE_CALLS_DISABLED refusals |
| OPENAI_API_KEY | prod/evals-real | — | gpt-4.1, temperature 0 |
| ELIGIBILITY_MODEL | no | gpt-4.1 | |
| LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT | no | false/—/— | tracing per run; OFF in tests |
| PROMPT_SOURCE | no | local | 'local' = repo-pinned prompts (tests/CI); 'hub' = LangSmith Hub by PROMPT_TAG |
| PROMPT_TAG | no | release pin | hero/eligibility-rank, hero/eligibility-finalize |
## 4. Setup
uv python pin 3.12 && uv sync --all-groups; cp .env.example .env; schema: apply
eligibility_agent/db/schema.sql in Supabase SQL editor (includes quick_checks_used_today fn
+ checkpointer tables note).
## 5. Seed lenders
uv run python scripts/seed_lenders.py --dry-run   # inspect SQL
uv run python scripts/seed_lenders.py             # upsert 3 synthetic lenders + pincodes
How to add a lender: hard_cutoffs keys (min_band,max_foir,min_income,min_age,
max_age_at_maturity,max_tenure_months,annual_rate_pct) are ENFORCED IN CODE; criteria JSON is
free-form text the LLM reads at rank time.
## 6. Run
uv run uvicorn eligibility_agent.main:app --port 8080  # health: GET /health -> {ok, surepass_live}
With SUREPASS_LIVE unset the service starts and refuses with LIVE_CALLS_DISABLED (by design).
## 7. Tests & type-checking
uv run pytest -q          # no network possible: pytest-socket --disable-socket
uv run mypy               # strict, covers eligibility_agent/, tests/, scripts/
## 8. Evals
uv run python -m eligibility_agent.evals.run --mode local --llm fake --dataset both
LANGSMITH_API_KEY=... OPENAI_API_KEY=... uv run python -m eligibility_agent.evals.run \
  --mode langsmith --llm real --max-real 20 --dataset both --build-datasets
Datasets are 100% synthetic (eligibility_agent/testing/envelopes.py). The folder
surepass-lab/out/bureau in the parent directory is REAL credit data: it is outside this repo,
must never be copied into fixtures, and must never be uploaded to LangSmith.
## 9. Observability
Tags tier:/admission:/lc:; metadata check_id + pan_hash (never the PAN); langsmith_run_id on checks row.
## 10. Open items (PRD section 14 — verbatim list, each with the code marker to search)
1 stage enum (STAGE_ORDER in policy/guards.py) · 2 EMI defaults 36m/14% + ceiling formula
(tools/lenders.py) · 3 invalid-PAN billing vs cap (persist node) · 4 Experian NTC shape
(# ASSUMPTION in tools/adapters.py + testing/envelopes.py) · 5 Account_Type/Status code tables
· 6 fixed-pipeline Tier 1 (graph identical either way)
## 11. Section ownership map (which test file pins which PRD rule) — table from this plan.

## Interfaces provided
- **name**: eligibility_agent.testing.envelopes | **definition**: EnvelopePair = tuple[int, dict[str, Any]]. Builders: pan_comprehensive_envelope(*, pan, full_name, status, category, gender, dob, aadhaar_linked, zip_code, client_id) -> EnvelopePair; experian_score_envelope(*, credit_score='745', report_date='20261001', success=True, message_code='success', client_id=...) -> EnvelopePair (success=False yields the ASSUMPTION no-hit shape {success: False, message_code: 'no_record_found'}); cais_account(*, 23 keyword args, see file) -> dict[str, Any] returning a DENSE field-map-complete account dict; experian_report_envelope(*, accounts, credit_score, report_date, caps_last_180, success, client_id) -> EnvelopePair; failure_envelope(...); validation_error_envelope(...) -> {errors, message} with no success/data keys. Any section may use these for its own unit tests; evals/datasets.py builds every golden example from them.
- **name**: eligibility_agent.testing.clock.FrozenClock | **definition**: class FrozenClock: __init__(start: datetime /*aware*/); now() -> datetime (aware UTC); advance(delta: timedelta) -> None; set_now(instant: datetime) -> None. Structurally satisfies the Clock protocol; canonical test epoch T0 = 2026-10-08T10:00:00+05:30.
- **name**: eligibility_agent.testing.fake_llm | **definition**: FakeRankLLM(respond: Callable[[RankContext], Ranking] | None = None, script: Sequence[Ranking] | None = None) with .calls: list[RankContext]; __call__(ctx) -> Ranking. Default ranks every cutoff-passing gated lender fit='strong' sorted by lender_id with band-word (never numeric) 'why' text. FakeFinalizeLLM(respond=None, script=None, talking_points_override=None, confirm_override=None) with .calls: list[FinalizeContext]; deterministic decision table documented in-file (low band -> fundable_with_coapplicant; empty accepted -> review/low; floor or undeclared income -> medium; indicative_loan_max = top lender ceiling, min = round1000(0.4*max)). Helpers bad_ranking(ids) and bad_final(top, alternates) for misbehavior scripts. FALLBACK_MESSAGE constant = 'The check could not be completed automatically. Please raise it for manual review.'
- **name**: eligibility_agent.testing.scenarios | **definition**: happy_quick_scenario(pan, score='745', zip_code='560001'), ntc_scenario(variant), pan_invalid_scenario(kind, pan), full_pincode_union_scenario(pan), full_all_emi_empty_scenario(pan), full_delinquency_scenario(pan) -> FakeSurePassClient, each fully wired with envelope pairs; the same scenario functions feed the golden datasets so tests and evals share one world.
- **name**: evals outputs contract (target function) | **definition**: evals/run.py run_one(example, llm) returns {'lc_view': dict, 'billed': bool, 'counted_against_cap': bool, 'from_cache': bool, 'admission': str, 'surepass_call_endpoints': list[str], 'gated_lender_ids': list[str], 'accepted_lender_ids': list[str], 'band': str}; evaluators in evals/evaluators.py (no_leak, lenders_gated, policy_consistency, outcome_matches_band, make_talking_points_judge) consume exactly this shape; DETERMINISTIC_EVALUATORS list exported for both local and langsmith modes. Dataset names: 'eligibility-quick-golden', 'eligibility-full-golden'.
- **name**: CI contract | **definition**: Commands other sections must keep working: `uv run mypy` (config in pyproject, covers eligibility_agent/tests/scripts), `uv run pytest -q` (sockets disabled; must pass with no env secrets and SUREPASS_LIVE unset), `uv run python -m eligibility_agent.evals.run --mode local --llm fake --dataset both` (exit 0 required).

## Interfaces consumed
- **name**: eligibility_agent/graph/build.py — DI seam (graph section owns) | **expectation**: @dataclass(frozen=True) class GraphDeps: repo: Repo; surepass: SurePassLike; rank_llm: RankLLM; finalize_llm: FinalizeLLM; clock: Clock; live_calls_enabled: bool; checkpointer: BaseCheckpointSaver | None = None. def build_graph(kind: Literal['quick','full'], deps: GraphDeps) -> CompiledStateGraph, invokable synchronously via .invoke(state, config) with checkpointer=None requiring no thread_id. rank_lenders must call deps.rank_llm at most twice per run (attempt 1 + one retry when accepted is empty or structurally invalid); finalize must call deps.finalize_llm at most three times (initial + 2 retries) then route to deterministic fallback_finalize.
- **name**: eligibility_agent/graph/llm.py — LLM callable protocols and contexts (graph section owns) | **expectation**: RankLLM = Callable[[RankContext], Ranking]; FinalizeLLM = Callable[[FinalizeContext], FinalRecommendation]. RankContext (pydantic): kind, applicant: Applicant, gated_lenders: list[GatedLender], cutoffs: dict[str, CutoffResult], ceilings: dict[str, float | None], attempt: int (1-based), validation_errors: list[str] (empty on first attempt, populated with drop/validation reasons on retries). FinalizeContext: kind, applicant, ranking: Ranking (accepted/dropped filled), affordability: Affordability, flags with at least {income_declared: bool, emi_is_floor: bool, pincode_mismatch: bool, delinquency: bool}, attempt: int, validation_errors: list[str]. My fakes and 4 misbehavior tests are written against exactly these fields.
- **name**: eligibility_agent/graph/state.py — models (state owner) | **expectation**: Ranking(ranked: list[RankedLender], accepted: list[str] = [], dropped: list[DroppedLender] = []) where RankedLender(lender_id: str, fit: Literal['strong','possible','weak'], why: str) and DroppedLender(lender_id: str, reason: str); FinalRecommendation exactly as PRD 7.3 (alternate_lender_ids max_length=2 enforced by pydantic — I test ValidationError on 3); Applicant exposes band: Literal['<650','650-720','720-780','780+','NTC'], existing_emi: float | None, existing_emi_floor: bool, foir: float | None, age_years: int | None; Affordability exposes ceilings: dict[str, float | None] and existing_emi_estimated: float | None; LcView field names exactly as PRD 9.3 JSON (model_validate round-trip is asserted).
- **name**: eligibility_agent/tools/surepass.py — client, fake, constants (tools section owns) | **expectation**: Constants PAN_ENDPOINT='/api/v1/pan/pan-comprehensive', SCORE_ENDPOINT='/api/v1/credit-report-experian/score', REPORT_ENDPOINT='/api/v1/credit-report-experian/fetch-report'. class LiveCallsDisabled(Exception). SurePassClient(token: str, live: bool) raising LiveCallsDisabled on every method unless live is True, where live comes from env SUREPASS_LIVE == 'true' exactly. FakeSurePassClient(pan: Mapping[str, EnvelopePair] = {}, score: Mapping[str, EnvelopePair] = {}, report: Mapping[str, EnvelopePair] = {}) keyed by PAN, with calls: list[FakeSurePassCall] where FakeSurePassCall has endpoint: str, pan: str, payload: dict[str, Any] (payload must include the literal consent 'Y' and the name/mobile/pan actually sent so test 1 can assert full_name precedence). Methods: pan_comprehensive(pan: str) -> SurePassResponse; experian_score(name: str, mobile: str, pan: str) -> SurePassResponse; experian_report(name, mobile, pan) -> SurePassResponse; SurePassResponse exposes http_status: int, body (full envelope), success: bool | None, data, message_code. Fake raises KeyError with the PAN on unmapped input.
- **name**: eligibility_agent/tools/adapters.py (tools section owns) | **expectation**: adapt_pan_comprehensive(resp: SurePassResponse) -> Identity; adapt_experian_score(resp: SurePassResponse) -> BureauScore; adapt_experian_report(resp: SurePassResponse, *, now: datetime) -> BureauReport (now needed for the 365-day bureau_pincodes window); band_for_score(score: int | None) -> Band with edges 649->'<650', 650->'650-720', 720->'720-780', 780->'780+', None->'NTC'. Identity has NO masked_aadhaar attribute; BureauReport has NO dob attribute. NTC rule: success is False OR credit_score missing/''/unparseable -> band 'NTC' (marked # ASSUMPTION in code).
- **name**: eligibility_agent/tools/lenders.py (tools section owns) | **expectation**: PINCODE_RE = re.compile(r'^[1-9][0-9]{5}$') applied with fullmatch; estimate_emi(balance: float, tenure_months: int | None, annual_rate_pct: float | None) -> float using reducing-balance EMI with defaults 36 months / 14.0 pct (unrounded float); loan_ceiling(monthly_income: float | None, existing_emi: float, cutoffs: HardCutoffs) -> float | None per PRD 8.4 (None when income unknown, 0.0 when headroom <= 0, else nearest-1000 principal at annual_rate_pct default 14 over max_tenure_months default 36); check_hard_cutoffs(lender: GatedLender, applicant: Applicant) -> CutoffResult(passed: bool, failures: list[str], unknown_facts: list[str]) with unknown-never-fails semantics; BAND_ORDER = {'NTC': 0, '<650': 1, '650-720': 2, '720-780': 3, '780+': 4}.
- **name**: eligibility_agent/policy/guards.py (policy section owns) | **expectation**: admit_quick_check(repo: Repo, clock: Clock, *, lc_id: str, pan_hash: str, refresh: bool, live_calls_enabled: bool) -> AdmissionDecision and admit_full_check(..., opportunity_id: str, opportunity_stage: str, ...) -> AdmissionDecision where AdmissionDecision(admission: Literal['cache_hit','new','refresh'] | None, refusal_code: str | None, billed: bool, counts_against_cap: bool). Refusal codes exactly {CONSENT_MISSING, DAILY_CAP_REACHED, STAGE_TOO_EARLY, OPPORTUNITY_ALREADY_CHECKED, CACHE_FRESH_NO_REFRESH, LIVE_CALLS_DISABLED}. STAGE_ORDER = ['new','contacted','qualified','counselling','offer','registration','enrolled','won'] with the gate at index('offer'); unknown stage -> STAGE_TOO_EARLY. Evaluation order and the full decision table as pinned in tests/test_admission_table.py. QUICK_DAILY_CAP = 5.
- **name**: eligibility_agent/db/repo.py (db section owns) | **expectation**: MemoryRepo(clock: Clock) implementing Repo protocol with at least: insert_consent(*, lc_id, lead_id, opportunity_id, pan_hash, channel) -> str; get_cache(pan_hash: str, kind: Literal['quick','full']) -> CacheRow | None; upsert_cache(row: CacheRow) -> None; insert_check(row: CheckRow) -> None; update_check(check_id: str, patch: CheckPatch) -> None; get_check(check_id) -> CheckRow | None; list_checks() -> list[CheckRow]; list_surepass_calls(check_id: str | None = None) -> list[SurePassCallRow]; insert_surepass_call(row) -> None; quick_checks_used_today(lc_id: str) -> int counting kind=='quick' AND counted_against_cap on the Asia/Kolkata DATE of created_at vs clock.now() (must mirror the SQL function); latest_ok_full_check_for_opportunity(opportunity_id: str) -> CheckRow | None; active_lenders_for_pincodes(pincodes: Collection[str]) -> list[GatedLender] (distinct lenders, matched_pincodes populated); upsert_lender(lender_row, pincodes) -> None for seeding. CheckRow fields used in assertions: id, kind, lc_id, pan_hash, status, refusal_code, billed, counted_against_cap, is_refresh, lc_view, internal, created_at. CacheRow: pan_hash, kind, payload, derived, is_ntc, fetched_at, expires_at.
- **name**: eligibility_agent/main.py + config.py (api section owns) | **expectation**: create_app(config: Config, deps: GraphDeps) -> FastAPI (test injection seam); auth: 401 for missing/wrong bearer on /checks/*, /health open returning {'ok': bool, 'surepass_live': bool}; refusals and pan_invalid return HTTP 200 with LcView status field (assumed, flagged); request schemas enforce PAN ^[A-Z]{5}[0-9]{4}[A-Z]$, mobile ^[6-9][0-9]{9}$, consent: Literal[True], full requires opportunity_id/opportunity_stage/monthly_income; Config.from_env() with surepass_live = (os.environ.get('SUREPASS_LIVE') == 'true'); hash_pan(pan: str, salt: str) -> str = sha256((salt+pan).utf8).hexdigest(). Clock protocol importable (assumed eligibility_agent/clock.py: class Clock(Protocol): def now(self) -> datetime: ...; SystemClock).
- **name**: persist node internal jsonb minimal shape (graph section owns) | **expectation**: checks.internal must contain at least: internal['admission']: str; internal['band']: str; internal['applicant']['existing_emi']: float | None and ['existing_emi_floor']: bool; internal['gate']['pincodes']: list[str] and ['lenders']: list[{'id','name'}]; internal['ranking']['accepted']: list[str] and ['dropped']: list[{'lender_id','reason'}]; internal['affordability']['ceilings']: dict[str, float | None]. Raw SurePass payloads, masked_aadhaar and node transcripts of prompts must NOT be in internal (no-leak code review + tests walk it with the bureau regex only excluding the literal key names).
- **name**: output/lc_view.py (output section owns) | **expectation**: Module constants BUREAU_RE = re.compile(r'(?i)\b(cibil|experian|equifax|crif|bureau)\b') and SCORE_RE = re.compile(r'(?<![\d,.])[3-8][0-9]{2}(?![\d,.])'); scrub_lines(lines: Sequence[str]) -> list[str] dropping any line matching either; build_lc_view(state) setting pan_aadhaar_link_issue True only when identity.aadhaar_linked is False else None; REFUSAL_MESSAGES: dict[str, str] mapping each refusal code to one plain-language line (no bureau words, no 300-899 numbers); redact runs BEFORE persist so checks.lc_view stores the scrubbed object.
- **name**: graph/prompts.py + LLM factories (graph section owns, real-eval path) | **expectation**: Repo-local pinned prompt texts (PROMPT_SOURCE=local default) so tests and CI never pull LangSmith Hub; factories make_rank_llm(config) -> RankLLM and make_finalize_llm(config) -> FinalizeLLM wrapping ChatOpenAI(model=config.model, temperature=0).with_structured_output(...), used only by evals/run.py --llm real.

## Decisions
- **topic**: Clock strategy: injected Clock protocol, not freezegun | **decision**: All time flows through a 3-line Clock protocol (def now() -> datetime, aware UTC) injected into GraphDeps, guards and MemoryRepo; tests use eligibility_agent/testing/clock.py FrozenClock with advance()/set_now(). freezegun is not a dependency. | **rationale**: freezegun monkeypatches datetime process-wide, which is exactly the banned 'monkeypatch production logic' pattern, interacts badly with anyio/langgraph internals and C extensions, and cannot model the production reality that the cap counter is a DB-side SQL function — MemoryRepo must take time from the same seam the graph uses or the IST-boundary test proves nothing. A Protocol is also the cheapest mypy --strict citizen (no plugin, no stubs).
- **topic**: No-network enforcement | **decision**: pytest-socket in addopts: --disable-socket --allow-unix-socket, plus pytest_configure scrubbing SUREPASS_LIVE/SUREPASS_TOKEN/OPENAI_API_KEY/LANGSMITH_API_KEY/SUPABASE_* and forcing LANGSMITH_TRACING=false and LANGCHAIN_TRACING_V2=false before any app import. | **rationale**: --allow-unix-socket is required because Starlette's TestClient anyio portal creates an asyncio loop whose self-pipe uses socket.socketpair(AF_UNIX); TCP/UDP stays blocked so any accidental SurePass/LangSmith/OpenAI/Supabase call fails loudly. Env scrubbing makes tracing a no-op and makes test 12's 'SUREPASS_LIVE unset' precondition structural rather than per-test.
- **topic**: FakeLLM shape: plain callables on the DI seam, not a BaseChatModel | **decision**: FakeRankLLM/FakeFinalizeLLM are classes whose instances are Callable[[RankContext], Ranking] and Callable[[FinalizeContext], FinalRecommendation], with .calls logs and script=/respond=/talking_points_override= hooks; they do not subclass BaseChatModel and do not implement with_structured_output. | **rationale**: The local context fixes the seam as 'two structured-output LLM callables injected at graph-build time' (graph section owns it). Faking at that seam keeps fakes 20 lines, deterministic and mypy-strict-clean; PRD 12's 'FakeLLM implementing with_structured_output' is satisfied at the same abstraction boundary one level up — the real adapter (ChatOpenAI().with_structured_output(...)) lives behind the callable and is exercised by the real-model eval job, not by unit tests.
- **topic**: Where fakes and fixtures live | **decision**: eligibility_agent/testing/ subpackage (clock.py, envelopes.py, scenarios.py, fake_llm.py) inside the installable package; FakeSurePassClient stays in eligibility_agent/tools/surepass.py as the PRD layout names it, consuming envelope dicts built by testing/envelopes.py. | **rationale**: evals/datasets.py and evals/run.py must reuse the identical builders and fakes (PRD 11: datasets built from synthetic SurePass fixtures; CI runs fake LLM for the bulk), and tests/ is not an importable package. A testing/ subpackage is the PRD-permitted implementer's choice that avoids duplicating fixture logic in two places.
- **topic**: Checkpointer in tests | **decision**: GraphDeps.checkpointer is None for the whole suite; one optional resume-focused test may use langgraph.checkpoint.memory.InMemorySaver (current class name, verified) but no test ever touches langgraph-checkpoint-postgres. | **rationale**: The Postgres checkpointer needs a DSN and sockets; checkpointing is orthogonal to every policy rule under test, and InMemorySaver exists if the graph section wants a resume test.
- **topic**: Exact-value assertions for money math | **decision**: Tests assert precomputed closed-form constants, not reimplemented formulas: EMI(120000,34.9,24)=7015.905377464855, EMI(90000,14,36)=3075.9866782222953, ceilings 1,219,000 / 439,000 / 966,000 / 1,034,000 / 1,252,000, L2 ceilings 384,000 / 768,000, indicative min = round1000(0.4*max) giving 488,000 / 154,000 / 414,000. | **rationale**: TDD against independently computed constants catches formula transcription bugs (e.g. rate/12 vs rate/1200, flat vs reducing) that a mirrored in-test formula would wave through; values were computed here with the PRD 8.3/8.4 formulas.
- **topic**: Admission decision-table interpretation (pinned by tests, flagged to owner) | **decision**: Order: consent -> stage(full) -> cache -> opportunity(full) -> cap(quick, only when the run would count) -> live kill switch. Fresh cache always serves (even at cap, even with live disabled); refresh=True on fresh cache refuses CACHE_FRESH_NO_REFRESH; OPPORTUNITY_ALREADY_CHECKED fires only on full checks with a stale/absent cache, a prior ok full check on the opportunity, and refresh=False; refresh with no cache degrades to a normal counted 'new'; full checks never count against the quick cap; unknown stage fails closed as STAGE_TOO_EARLY. | **rationale**: This is the only ordering that satisfies all FIXED rules simultaneously (cache hits are free and unlimited; the cap counts only billed quick checks; 'second call on the same opportunity is a cache hit' in PRD test 8; 'refresh is billed but not counted'); the genuinely ambiguous cells are surfaced in prd_gaps rather than silently chosen.
- **topic**: Cache hit re-runs the LLM nodes | **decision**: Test 2 asserts rank/finalize fakes are called on a cache-hit run (facts from cache, lenders from the live DB), and from_cache=True with zero SurePass calls. | **rationale**: PRD 7.3 node contracts say validate_pan/pull_* 'load from the cache row' and the graph continues — so lender criteria changes take effect on cached customers; pinned in a test because it is cheap to regress into returning the stale lc_view verbatim.
- **topic**: Evaluator signatures | **decision**: All five evaluators use the langsmith SDK v0.2+ name-matched signature (inputs, outputs, reference_outputs) -> {'key','score','comment'}; the four deterministic ones are plain functions also runnable without LangSmith (local eval mode); the LLM judge is a closure factory only constructed when OPENAI_API_KEY exists. | **rationale**: Verified against current LangSmith docs ('write evaluator functions that accept the inputs, outputs, reference_outputs dicts as args'); plain functions make the no-secrets CI job possible and keep evaluators unit-testable inside the pytest suite itself.
- **topic**: Eval outputs carry internal facts (band, gated ids, call log) | **decision**: The eval target returns lc_view plus band, gated/accepted lender ids, policy flags and the fake call-log endpoints; evaluators judge against that contract. | **rationale**: outcome_matches_band, lenders_gated and policy_consistency are unevaluatable from the redacted LcView alone; this is safe only because every dataset row is synthetic (builders enforce it and a builder self-test asserts probe literals never appear), which is why the no-real-data rule is load-bearing.
- **topic**: CI secrets gating | **decision**: Three jobs: test (mypy+pytest, no secrets), evals-local (fake LLM, no secrets, exit-code gate), evals-langsmith (env-indirection gate step because GitHub Actions forbids secrets in job-level if; real model capped at --max-real 20, remainder on FakeLLM, experiment prefixes ci-<sha>-real / ci-<sha>-fake). | **rationale**: Matches PRD 11's 'real model on a small set (max 20) and the fake for the rest' while keeping forks/PRs without secrets fully green and spending zero SurePass credits anywhere in CI (SUREPASS_LIVE never set).
- **topic**: mypy --strict posture | **decision**: strict=true over eligibility_agent/, tests/ and scripts/ with pydantic.mypy plugin; ignore_missing_imports confined to the supabase family and pytest_socket; no ignore_errors, no follow_imports=skip; Any from supabase is converted via Model.model_validate at the single repo boundary. | **rationale**: supabase-py ships no py.typed (known gap); langgraph/langchain-core/langsmith/fastapi/pydantic all do, so blanket overrides would hide real bugs. Typing the tests is what forces the Protocol seams (Repo/Clock/SurePassLike/LLM callables) to actually hold.
- **topic**: Synthetic-data quarantine | **decision**: All fixture values are invented (score 745/702/628/610, zips 560001/560002/110001/400001 + junk '20', DOB 1998-04-12, balances 120000/90000/200000); a builder self-test asserts probe-derived literals (791111, 201301, 560071, 16559, 2434004, 138696, 109433) never appear in any default envelope; the bare '736' appears exactly once, as the scrub-test literal the brief mandates, attached to no identity. | **rationale**: surepass-lab/out/bureau is a real person's credit file; the PRD forbids uploading it and the task forbids copying values — an executable guard beats a convention.
- **topic**: Band bucket edges and band ordering | **decision**: Tests pin half-open buckets [650,720), [720,780), [780,inf) (so 720 -> '720-780', 780 -> '780+') and the comparison order NTC(0) < '<650'(1) < '650-720'(2) < '720-780'(3) < '780+'(4), with min_band 'NTC' meaning accepts-everyone. | **rationale**: PRD gives bucket labels but not edge membership or min_band comparison semantics; some convention must be executable, and pinning it in adapter/cutoff tests makes the choice visible and reviewable (surfaced in prd_gaps).

## Risks
- Seam drift between sections: my fakes and 13 scenario tests compile against GraphDeps/RankContext/FinalizeContext/FakeSurePassClient signatures the graph and tools sections own; if the orchestrator merges different names or field sets, tests/conftest.py and testing/fake_llm.py must be mechanically renamed first — mypy --strict over tests/ will enumerate every mismatch, which is why tests are in the mypy files list.
- The internal-jsonb assertions (gate pincodes, ranking.dropped, applicant.existing_emi) couple tests to the persist node's serialization; I limited them to a documented minimal subset, but if the persist owner nests differently, tests 8/9/10 need path updates (assertion semantics unchanged).
- pytest-socket must keep --allow-unix-socket or the FastAPI TestClient anyio portal breaks on macOS/Linux asyncio self-pipes; conversely removing --disable-socket silently re-enables network — both flags live in pyproject addopts so a single review point guards them.
- The FIXED leak regex misses plurals/possessives ('bureaus', "Experian's" is caught via \b but 'bureaus' is not) and misses spelled-out scores ('seven thirty six'); the LLM-judge evaluator is the only backstop for those — flagged to the owner rather than strengthening a FIXED rule unilaterally.
- The by-design strictness of the 300-899 scrub will eat legitimate lines ('tenure of 360 months'); FakeLLM defaults and the prompt guidance avoid bare 3-digit numbers, but real-model eval runs may show elevated scrub rates — the no_leak evaluator plus scrub counting in redact should be monitored in the first LangSmith experiments.
- MemoryRepo's quick_checks_used_today mirrors the SQL function by convention only; nothing executes schema.sql in the no-network suite, so a drift between Python IST logic and the Postgres Asia/Kolkata function would pass tests — mitigation is a documented side-by-side comment in both files and an integration smoke test post-deploy (out of scope here).
- Real-model eval job spends OpenAI tokens on every CI run with secrets present (bounded: max 20 examples x 2 LLM nodes x ~1 retry worst case, plus the judge); the gate step and --max-real flag are the cost controls.
- The Experian no-hit envelope is an ASSUMPTION; both candidate shapes are tested and the builder marks the branch, but when the real NTC PAN is probed (open item 4) the builder default, adapter and tests q02/q03 must be updated together.

## PRD gaps
- OPPORTUNITY_ALREADY_CHECKED is unreachable under a plain reading: PRD test 8 says a second full check on the same opportunity with a fresh cache is a cache HIT, and policy 8 says re-runnable when stale — I pinned the refusal to the remaining cell (stale/absent cache + prior ok full check + refresh=False); owner should confirm.
- consent=false at the API: PRD 9.1 says consent must be literally true (schema -> 422), yet CONSENT_MISSING exists as a refusal code; I pinned 422 at the schema and kept CONSENT_MISSING as a guard-level defense (unit-tested directly) for the case where the consent row insert fails or is absent — confirm which surface FulCRM/SuperConnect expect.
- HTTP status for refused/pan_invalid responses is unspecified; I assumed 200 with LcView.status carrying the outcome (refusals are domain results, not transport errors). SuperConnect/FulCRM consumers should confirm.
- refresh=true when no cache row exists: pinned as a normal counted 'new' check (refresh flag moot); alternative is a refusal — owner call.
- Cache hits re-run the two LLM nodes (token spend per cache hit, fresh lender table applied to cached facts) per the 7.3 node contracts; if the owner intended verbatim cached LcView replay, test 2 and the admission table change.
- Band edge membership (720 -> '720-780', 780 -> '780+') and the min_band comparison order including where NTC sits (pinned: NTC lowest, min_band 'NTC' accepts all) are not in the PRD; tests pin my convention.
- Unknown opportunity_stage strings (anything outside the ASSUMPTION enum) fail closed as STAGE_TOO_EARLY; needs FulCRM enum confirmation (open item 1).
- quick_checks_remaining_today semantics on refused/cache-hit responses pinned as 5 - used_today for the requesting LC at response time (so cache hit shows 5 for a fresh LC, cap refusal shows 0); PRD only shows the happy-path value 4.
- Whether indicative_loan_max of 0.0 (headroom exhausted) should render as 0 or null in LcView is unspecified; test_full_emi_derivation accepts either pending the affordability owner's contract.
- PRD 12 'FakeLLM implementing with_structured_output' conflicts with the stated DI seam of two injected callables; resolved in favor of the seam (local context, graph section owns) — the real with_structured_output path is covered by the ≤20-example real-model eval, not unit tests.
