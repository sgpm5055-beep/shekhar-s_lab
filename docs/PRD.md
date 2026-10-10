# PRD: Financing Eligibility Agent (Phase 1) on LangGraph, LangChain and LangSmith

Owner: Shekhar Gupta, Product Manager, AI Sales Technology & Platform, Hero Vired
Version: 1.0, 8 October 2026
Audience: a coding agent (VS Code) building this from scratch as a standalone service

This document is written to be executed by a coding agent. Sections 4 to 10 are the build contract. Where a value is a product decision it is marked FIXED (do not change) or ASSUMPTION (implement as stated, surface for review). Anything not covered here is the implementer's choice, but it must not weaken a FIXED rule.

---

## 1. Problem

Learning Consultants (LCs) at Hero Vired pitch programmes to learners before knowing whether the learner can be financed. Loan rejections after the pitch waste LC effort, hurt conversion and damage the learner experience. The earlier design (three bureau calls per check) was too expensive per check relative to cost per lead, so it was replaced with a two-tier model: a cheap score-only check during the call, and a full credit report only once an opportunity is at offer stage.

## 2. Goal

A server-side service that an LC can trigger from SuperConnect (Tier 1) or from the opportunity profile in FulCRM (Tier 2), which returns a band-driven financing recommendation (indicative loan range, top lender, up to two alternates, co-applicant suggestion) without ever exposing the credit score, the bureau name or the raw report to the LC.

The orchestration is an agent built on LangGraph. Every rule that costs money or protects the learner is enforced in graph nodes and tool wrappers, not in the prompt.

## 3. Non-goals

- No lender integration or loan application submission. Output is a recommendation only.
- No requested loan amount. FulCRM does not hold one; output is a loan ceiling per lender.
- No name matching of the PAN holder against the CRM contact (v1).
- No UI. SuperConnect and FulCRM consume the API; the admin UI for lender criteria writes to the same database but is out of scope here.
- No sandbox. All SurePass calls spend production credits; the service must make live calls impossible by default.

## 4. Stack (FIXED)

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Agent orchestration | LangGraph (StateGraph, typed state, checkpointer) |
| LLM + tools | LangChain (`langchain-openai`, `@tool`, structured output via Pydantic) |
| Observability and evals | LangSmith (tracing on every run, datasets + evaluators for regression) |
| HTTP | FastAPI |
| Persistence | Supabase (Postgres) via `supabase-py`; LangGraph checkpointer on Postgres (`langgraph-checkpoint-postgres`) |
| External data | SurePass KYC API (`https://kyc-api.surepass.app`) |
| Model | `gpt-4.1` default, configurable; temperature 0 |
| Tests | pytest; fakes for SurePass and the LLM; no network in tests |

Project layout:

```
eligibility_agent/
  main.py                 FastAPI app, auth middleware, routes
  config.py               env loading; SUREPASS_LIVE kill switch
  graph/
    state.py              CheckState (TypedDict/Pydantic)
    build.py              build_graph(kind) -> compiled StateGraph
    nodes.py              admission, validate_pan, pull_score, pull_report, gate_lenders, affordability, rank, finalize, redact, persist
    prompts.py            system prompts per tier
  tools/
    surepass.py           SurePassClient (live/fake), envelope handling
    adapters.py           PAN Comprehensive, Experian score, Experian report -> facts
    lenders.py            pincode gate, hard cutoffs, loan ceiling
  policy/
    guards.py             consent, daily cap, cache, refresh, stage gate
  db/
    repo.py               Repo protocol; SupabaseRepo; MemoryRepo
    schema.sql
  output/
    lc_view.py            redaction boundary
  evals/
    datasets.py           LangSmith dataset builders
    evaluators.py         custom evaluators
tests/
```

## 5. Users and entry points

Tier 1, quick check. Actor: LC in SuperConnect, Financing section, during a live call. Inputs: name and mobile (CRM-prefilled), PAN, verbal consent flag, optional monthly income, optional existing EMI. Button label: "Check eligibility".

Tier 2, full check. Actor: LC (or BDM) on the opportunity profile in FulCRM, Financing section. Unlocks only when the opportunity stage is "offer" or beyond (registration done). Inputs: as Tier 1 plus opportunity id, opportunity stage and monthly income (required).

## 6. Policy rules (FIXED)

These are the Oct 2026 decisions. Implement each in `policy/guards.py` and call them from graph nodes; the LLM never sees or controls them.

1. Consent. Verbal consent is recorded by this service before any SurePass pull. Experian via SurePass does not validate consent server-side; our consent row is the only gate.
2. Bureau. Both the score and the report come from Experian via SurePass. CIBIL, Equifax and CRIF are not called. The Aadhaar-PAN link check endpoint is not used (it does not exist at SurePass); link status comes from PAN Comprehensive `aadhaar_linked`.
3. Quick-check cap. 5 billed quick checks per LC per IST calendar day, across all of that LC's opportunities. Only actual billed bureau calls count; cache hits do not.
4. Cache. Results are cached at PAN level (salted SHA-256 of the PAN) for 30 days, separately for `quick` and `full`, because multiple LCs may work the same customer under the opportunity model. A cache hit is returned without any SurePass call and is not billed or counted.
5. Refresh. After the 30-day TTL a Refresh action unlocks. A refresh is billed but does not deduct from the LC's 5 daily checks. Refresh before TTL is refused.
6. PAN invalid. If PAN Comprehensive says the PAN is not valid or not an individual (`category != "person"`), no bureau call is made and nothing is consumed; the LC can correct and retry.
7. Bureau no-hit. A no-hit on a valid PAN does consume the check and is cached as NTC (new to credit) for 30 days.
8. Full check. Once per opportunity, at stage offer or beyond, re-runnable only when the cached result is stale. Both cases are handled: opportunity with a prior quick check and without.
9. Pincode gate. The only deterministic lender filter. In Tier 1 it runs on the PAN-registered pincode. In Tier 2 it re-runs on the union of the PAN pincode and the bureau pincodes. The LC can never override it.
10. LC visibility. The LC never sees the credit score, the bureau name or any raw report content. The LC sees the recommendation only. The PAN-Aadhaar link status is shown only when it is a problem (`aadhaar_linked == false`).
11. Inputs. Tier 1 requires name, mobile, PAN and consent; monthly income and existing EMI are optional and produce richer output when given. Tier 2 requires monthly income.
12. All SurePass calls are server-side. No key ever reaches a client.

## 7. Agent design (LangGraph)

### 7.1 Why a graph and where the LLM sits

The flow is mostly deterministic. The LLM is used in exactly two places: ranking the gated lenders against free-form lender criteria, and writing the LC-facing wording. Everything else is a plain node. This keeps credit spend, caps and redaction out of the model's hands and makes every run reproducible in LangSmith.

### 7.2 State

```python
class CheckState(TypedDict, total=False):
    # request
    kind: Literal["quick", "full"]
    check_id: str
    lc_id: str; lead_id: str; opportunity_id: str | None; opportunity_stage: str | None
    name: str; mobile: str; pan: str; pan_hash: str
    monthly_income: float | None; existing_emi_declared: float | None
    refresh: bool
    # admission
    admission: Literal["cache_hit", "new", "refresh"]
    counts_against_cap: bool; billed: bool
    refusal_code: str | None
    # facts (internal only)
    identity: Identity | None          # from PAN Comprehensive
    score: BureauScore | None          # Tier 1
    report: BureauReport | None        # Tier 2
    gated_lenders: list[GatedLender]
    applicant: Applicant | None        # band, income, existing_emi, floor flag, age
    affordability: Affordability | None
    ranking: Ranking | None            # accepted + dropped after code verification
    final: FinalRecommendation | None
    # output
    lc_view: LcView | None
    errors: list[str]
```

Pydantic models for `Identity`, `BureauScore`, `BureauReport`, `GatedLender`, `Applicant`, `Affordability`, `Ranking`, `FinalRecommendation`, `LcView` live in `graph/state.py`. `FinalRecommendation` is the structured-output schema for the LLM.

### 7.3 Nodes and edges

```
START
 -> admission            (policy: consent row, cache, cap, stage, refresh)      [deterministic]
   -> refused? ----------> persist_refusal -> END
 -> validate_pan         (PAN Comprehensive, or cached identity)                [deterministic, billed call]
   -> pan invalid? ------> redact -> persist -> END   (status=pan_invalid, nothing consumed)
 -> pull_score | pull_report   (Experian score | report, or cache)              [deterministic, billed call]
 -> gate_lenders         (pincode union -> lenders from DB)                     [deterministic]
 -> affordability        (EMI derivation, FOIR, age, hard cutoffs, ceilings)   [deterministic]
 -> rank_lenders         (LLM proposes ranking; code verifies and drops)        [LLM, structured output]
   -> ranking invalid and retries < 2? -> rank_lenders (with validation errors fed back)
 -> finalize             (LLM writes FinalRecommendation; code validates lender ids)  [LLM, structured output]
   -> finalize invalid and retries < 2? -> finalize (errors fed back)
   -> still invalid? -----> fallback_finalize (outcome=review, confidence=low)
 -> redact               (build LcView; scrub)                                  [deterministic]
 -> persist              (checks row, cache row, surepass_calls rows, cap accounting) -> END
```

Build two compiled graphs with `build_graph("quick")` and `build_graph("full")`; they share nodes and differ only in which pull node is wired and in the system prompt. Use a Postgres checkpointer keyed by `check_id` so an interrupted run can be inspected in LangSmith and resumed.

Node contracts:

admission. Insert a `consents` row. Then: Tier 1 calls `admit_quick_check`; Tier 2 calls `admit_full_check`. Sets `admission`, `billed`, `counts_against_cap`, or `refusal_code` from the set `CONSENT_MISSING, DAILY_CAP_REACHED, STAGE_TOO_EARLY, OPPORTUNITY_ALREADY_CHECKED, CACHE_FRESH_NO_REFRESH, LIVE_CALLS_DISABLED`.

validate_pan. If `admission == cache_hit`, load identity from the cache row. Else call PAN Comprehensive with `get_address=true, get_contact=false`, log the call, adapt to `Identity`. Gate: `status == "valid" and category == "person"`.

pull_score / pull_report. If cache hit, load derived facts. Else call Experian score or report with `name` (from PAN Comprehensive `full_name`, fallback to request name), `mobile`, `pan`, `consent: "Y"`; log the call; adapt; write the cache row with `expires_at = now + 30 days` and `is_ntc` set when the band is NTC.

gate_lenders. `pincodes = {identity.pan_pincode} ∪ report.bureau_pincodes` (Tier 2) or `{identity.pan_pincode}` (Tier 1). Query `lender_pincodes` joined to active `lenders`. Store `GatedLender(id, name, criteria, hard_cutoffs, matched_pincodes)`. Empty result is allowed and lowers confidence downstream.

affordability. Build `Applicant`:
- `band` from report (Tier 2) or score (Tier 1).
- `existing_emi`: Tier 2 = reported EMI on open accounts + estimated EMI for open accounts with no reported EMI (see 8.3); if the LC declared an existing EMI, use the higher of declared vs bureau-derived. Tier 1 = declared or null.
- `existing_emi_floor` true when any open account lacks a reported EMI, or when nothing is known.
- `age_years` from `identity.dob` (never from a bureau holder row).
- `foir = existing_emi / monthly_income` when both known.
For each gated lender run `check_hard_cutoffs` and compute `loan_ceiling` (see 8.4). Store `Affordability`.

rank_lenders. LLM call with structured output `Ranking(ranked=[{lender_id, fit: strong|possible|weak, why}])`, given the applicant summary (band, foir, floor flag, age, income presence), the gated lenders with their `criteria` text, and the per-lender hard-cutoff results. Code then drops any lender id not in `gated_lenders` or failing hard cutoffs, records them under `dropped` with the reason, and keeps the rest as `accepted`. If `accepted` is empty but gated lenders exist, retry once with the drop reasons appended.

finalize. LLM call with structured output `FinalRecommendation`:

```python
class FinalRecommendation(BaseModel):
    outcome: Literal["fundable", "fundable_with_coapplicant", "not_fundable", "review"]
    confidence: Literal["high", "medium", "low"]
    indicative_loan_min: float | None
    indicative_loan_max: float | None
    top_lender_id: str | None
    alternate_lender_ids: list[str] = Field(max_length=2)
    coapplicant_suggested: bool
    lc_talking_points: list[str] = Field(max_length=4)      # each <= 200 chars
    confirm_with_learner: list[str] = Field(max_length=3)
    reasoning_internal: str = Field(max_length=2000)
```

Code validation after the call: `top_lender_id` and every alternate must be in `ranking.accepted`; loan min <= max; when `outcome in (fundable, fundable_with_coapplicant)` a top lender is required unless `coapplicant_suggested`. On failure, retry with the validation errors in the message; after 2 retries go to `fallback_finalize`.

fallback_finalize. Deterministic `FinalRecommendation(outcome="review", confidence="low", ...)` with a fixed LC message: "The check could not be completed automatically. Please raise it for manual review."

redact. Build `LcView` (section 9.3). Scrub every LC-facing string: drop any line matching `(?i)\b(cibil|experian|equifax|crif|bureau)\b` or containing a standalone 3-digit number in 300 to 899. Set `pan_aadhaar_link_issue = True` only when `identity.aadhaar_linked is False`.

persist. Update the `checks` row with status, `billed` (any SurePass call made), `counted_against_cap` (= `counts_against_cap and bureau_called and not pan_invalid`), `lc_view`, and `internal` (facts minus raw payload, node transcript, LangSmith run id). Return `lc_view`.

### 7.4 Prompts

System prompt, shared core (store in `graph/prompts.py`, keep under 400 words):

- Role: financing eligibility agent for Hero Vired's sales team; an LC on a live call needs to know whether the learner can be financed and by which lender.
- You only rank the lenders you are given. Never invent lenders. Never suggest overriding the pincode gate.
- The LC must never learn the credit score, the bureau's name or raw report details. LC-facing text is band-driven, plain language ("strong profile", "existing EMIs are high relative to income"). Numbers and bureau facts go only in `reasoning_internal`.
- Indian context: INR, monthly income, FOIR = existing EMI / monthly income.
- Prefer `fundable_with_coapplicant` over `not_fundable` when band is NTC or below 650, or when FOIR headroom is thin but a working parent or spouse would plausibly clear it.
- Confidence: high when band, income and EMI are known and at least one lender passes cleanly; medium when EMI is a floor or income is undeclared; low when no lender passed the gate or facts conflict (ask the LC to confirm).

Tier 1 addendum: if income was not given, give a band-driven view only and put "declared monthly income" under `confirm_with_learner`.

Tier 2 addendum: if PAN pincode and bureau pincodes differ, keep both in the gate and add "confirm current pincode" to `confirm_with_learner`. Flag delinquency (past due, DPD > 30 in 24 months, write-offs) in `reasoning_internal` and let it lower confidence or push toward co-applicant.

Prompts are versioned in LangSmith Hub (`hero/eligibility-rank`, `hero/eligibility-finalize`) and pulled by tag; the repo pins a tag per release.

## 8. External data and derivations

Source of truth for shapes: project doc "SurePass per-API field map and spec reconciliation" (6 to 7 Oct 2026). Every path below was verified against saved probes. Values are strings unless noted.

### 8.1 SurePass client

Host `https://kyc-api.surepass.app`, headers `Authorization: Bearer <token>`, `Content-Type: application/json`. Envelope `{data, status_code, success, message, message_code}`; validation failures use `{errors, message}` with no `success`. Branch on HTTP status + `success` + `message_code`, never on `message`.

Kill switch (FIXED): the client raises `LiveCallsDisabled` on every call unless env `SUREPASS_LIVE == "true"`. Tests and local runs use `FakeSurePassClient`. Log every call to `surepass_calls` with endpoint, `data.client_id`, HTTP status, `success`, `message_code`.

Endpoints:
- `POST /api/v1/pan/pan-comprehensive` body `{id_number, get_address: true, get_contact: false}`.
- `POST /api/v1/credit-report-experian/score` body `{name, mobile, pan, consent: "Y"}`.
- `POST /api/v1/credit-report-experian/fetch-report` body `{name, mobile, pan, consent: "Y"}`.

### 8.2 Adapters

PAN Comprehensive -> `Identity`: `pan_valid = success and data.status == "valid" and data.category == "person"`; `full_name`; `gender` mapped `M -> male`, `F -> female`, else null; `dob` (YYYY-MM-DD); `aadhaar_linked` (bool); `pan_pincode = data.address.zip` if it matches `^[1-9][0-9]{5}$`; `surepass_client_id = data.client_id`. Never store `masked_aadhaar` beyond the raw cache payload.

Experian score -> `BureauScore`: `score = int(data.credit_score or data.credit_report.SCORE.FCIREXScore)`; NTC when `success` is false or the value is empty or unparseable (ASSUMPTION: the NTC shape is unverified; mark the branch and confirm once an NTC PAN is tested). `report_date` from `CreditProfileHeader.ReportDate` (YYYYMMDD).

Experian report -> `BureauReport`. Let `A = data.credit_report.CAIS_Account.CAIS_Account_DETAILS[]` (dense: every key present, `""` = not reported, dates YYYYMMDD):
- open predicate: `Date_Closed == "" or Account_Status == "11"`.
- `existing_emi_reported` = sum of `Scheduled_Monthly_Payment_Amount` over open accounts where non-empty; `existing_emi_floor = True` if any open account is empty. Note: it was empty on 100% of the test file.
- `open_accounts_missing_emi` = for each open account with empty EMI: `{balance: Current_Balance, tenure_months: Repayment_Tenure or Terms_Duration (0 or "" = unknown), rate: Rate_of_Interest or null, account_type: Account_Type}`.
- `total_outstanding` = sum of `Current_Balance` over open accounts; `past_due_total` = sum of `Amount_Past_Due` over all accounts; `max_dpd_24m` = max `CAIS_Account_History[].Days_Past_Due`; `written_off_or_settled` = count of accounts with non-empty `Written_off_Settled_Status`.
- `enquiries_180d` = `TotalCAPS_Summary.TotalCAPSLast180Days`.
- `bureau_pincodes` = set of `CAIS_Holder_Address_Details[].ZIP_Postal_Code_non_normalized` over accounts whose `Date_Reported` is within 365 days, filtered to `^[1-9][0-9]{5}$` (junk like "20" exists).
- `band` from `data.credit_score`.
- DOB is never taken from `CAIS_Holder_Details` (rows disagree with each other); use PAN Comprehensive.

Band buckets (FIXED): `<650`, `650-720`, `720-780`, `780+`, `NTC`.

### 8.3 EMI estimation for unreported open accounts (ASSUMPTION)

Reducing-balance EMI on `balance` over `tenure_months` at `rate`; when tenure is unknown use 36 months, when rate is unknown use 14% p.a. Surface the estimate and the floor flag separately in `Affordability` so the LC wording can say "confirm existing EMIs".

### 8.4 Lender model and ceiling

`lenders.hard_cutoffs` JSON (enforced in code): `min_band, max_foir, min_income, min_age, max_age_at_maturity, max_tenure_months, annual_rate_pct`. `lenders.criteria` JSON: free-form text and fields the LLM reads when ranking (product fit, salaried vs self-employed, co-applicant rules, course categories).

`check_hard_cutoffs(lender, applicant)`: a cutoff fails only when the applicant fact is known and violates it; unknown facts do not fail a cutoff, they lower confidence.

`loan_ceiling(applicant, lender)` (ASSUMPTION): `headroom = max_foir * monthly_income - existing_emi`; if `headroom <= 0` ceiling is 0; else the principal whose EMI equals `headroom` at the lender's `annual_rate_pct` (default 14) over `max_tenure_months` (default 36), rounded to the nearest 1,000. Null when income is unknown.

## 9. API

Auth: `Authorization: Bearer <SERVICE_API_TOKEN>` on every route except `/health`.

### 9.1 `POST /checks/quick`

```json
{ "lc_id": "lc_123", "lead_id": "lead_456", "opportunity_id": "opp_789",
  "name": "Full Name", "mobile": "98XXXXXXXX", "pan": "ABCDE1234F",
  "consent": true, "monthly_income": 60000, "existing_emi": 0, "refresh": false }
```
`consent` must be literally `true`; `opportunity_id`, `monthly_income`, `existing_emi`, `refresh` optional. PAN validated against `^[A-Z]{5}[0-9]{4}[A-Z]$`, mobile against `^[6-9][0-9]{9}$`.

### 9.2 `POST /checks/full`

Quick fields plus required `opportunity_id`, `opportunity_stage`, `monthly_income`. Stage order (ASSUMPTION, replace with FulCRM's enum): `new, contacted, qualified, counselling, offer, registration, enrolled, won`; the gate opens at `offer`.

### 9.3 Response: `LcView` (the only object SuperConnect/FulCRM receive)

```json
{ "check_id": "uuid", "kind": "quick", "status": "ok | pan_invalid | refused",
  "refusal_code": null,
  "outcome": "fundable", "confidence": "high",
  "indicative_loan_range": { "min": 200000, "max": 500000 },
  "top_lender": { "id": "L1", "name": "Lender One" },
  "alternates": [ { "id": "L2", "name": "Lender Two" } ],
  "coapplicant_suggested": false,
  "talking_points": ["Strong profile, proceed with the pitch"],
  "confirm_with_learner": ["Current pincode"],
  "pan_aadhaar_link_issue": null,
  "from_cache": false, "cached_at": "iso", "refresh_available_at": "iso",
  "quick_checks_remaining_today": 4 }
```

No field in this object may carry the score, the bureau name, a raw report fragment or masked Aadhaar. Refusals return `status: "refused"` with `refusal_code` and a one-line `talking_points` message; they are persisted as `checks` rows too.

### 9.4 `GET /health` -> `{ ok, surepass_live }`.

## 10. Data model (Supabase)

Tables, all service-role only: `lenders(id, name, active, criteria jsonb, hard_cutoffs jsonb)`, `lender_pincodes(lender_id, pincode)`, `consents(id, lc_id, lead_id, opportunity_id, pan_hash, channel, recorded_at)`, `pan_cache(pan_hash, kind, payload jsonb, derived jsonb, is_ntc, fetched_at, expires_at, pk(pan_hash, kind))`, `checks(id, kind, lc_id, lead_id, opportunity_id, pan_hash, status, refusal_code, billed, counted_against_cap, is_refresh, consent_id, lc_view jsonb, internal jsonb, langsmith_run_id, created_at)`, `surepass_calls(id, check_id, endpoint, surepass_client_id, http_status, success, message_code, called_at)`, plus the LangGraph checkpointer tables.

SQL function `quick_checks_used_today(lc_id)` counts `checks` where `kind='quick' and counted_against_cap` on the current `Asia/Kolkata` date.

`pan_cache.payload` holds the raw SurePass responses for audit and reprocessing; it is never returned by any API and never sent to the LLM.

## 11. LangSmith

- Tracing on for every graph run with tags `tier:quick|full`, `admission:<mode>`, `lc:<id>`; metadata includes `check_id`, `pan_hash` (never the PAN), `billed`, `counted_against_cap`.
- Store the LangSmith run id on the `checks` row.
- Datasets: `eligibility-quick-golden` and `eligibility-full-golden`, built from synthetic SurePass fixtures (no real credit data; the saved probes under `surepass-lab/out/bureau` are a real file and must not be uploaded). Each example: request + fake SurePass responses + expected `LcView` invariants.
- Evaluators (custom, in `evals/evaluators.py`):
  - `no_leak`: LC-facing strings contain no bureau name and no 300 to 899 number.
  - `lenders_gated`: top and alternates are a subset of gated and accepted lenders.
  - `policy_consistency`: `billed`, `counted_against_cap`, `from_cache` match the admission mode and the fake client's call log.
  - `outcome_matches_band`: NTC or <650 never yields `fundable` without `coapplicant_suggested`.
  - LLM-as-judge on `lc_talking_points` for plain-language, band-driven wording (rubric in the file).
- Run evals in CI against the compiled graphs with the fake LLM swapped for the real model on a small set (max 20 examples) and the fake for the rest.

## 12. Test plan (pytest, no network)

Fakes: `FakeSurePassClient` returning synthetic envelopes modelled on the field map; `FakeLLM` implementing `with_structured_output` deterministically from state.

Required tests:
1. Quick check happy path: calls PAN then score once, bills, counts, returns a lender that passed the gate and cutoffs, response contains no leak.
2. Second LC, same PAN within TTL: cache hit, no SurePass call, not counted, `quick_checks_remaining_today` unchanged.
3. Sixth billed quick check in a day: refused `DAILY_CAP_REACHED`, no SurePass call.
4. PAN invalid or non-individual: PAN call only, status `pan_invalid`, not counted.
5. Bureau no-hit: counted, cached with `is_ntc=True`, outcome `fundable_with_coapplicant`.
6. Refresh before TTL refused; after TTL billed, `is_refresh=True`, not counted.
7. Full check before offer stage: refused `STAGE_TOO_EARLY`, no SurePass call.
8. Full check at offer: PAN then report once; gate used the union of PAN and bureau pincodes with junk pincodes filtered; second call on the same opportunity is a cache hit.
9. Full check with all open-account EMIs empty: `existing_emi_reported == 0`, estimate > 0, floor flag true; LC-declared EMI wins when higher.
10. LLM proposes an ungated or invented lender: it is dropped; if finalize keeps naming it, fallback produces `review`.
11. LLM leaks a score or bureau name in talking points: the line is scrubbed.
12. `SUREPASS_LIVE` unset: every client call raises `LiveCallsDisabled`.
13. Adapter unit tests for PAN Comprehensive, Experian score, Experian report (open predicate, EMI floor, pincode filter, band).

## 13. Acceptance criteria

- All tests in section 12 pass; `mypy --strict` (or `pyright`) clean.
- A full LangSmith trace exists for every API call, with the check id in metadata.
- With `SUREPASS_LIVE` unset the service starts, serves `/health`, and returns `LIVE_CALLS_DISABLED` refusals rather than making calls.
- A code review confirms that no path from `pan_cache.payload` or raw SurePass data reaches `LcView` or an LLM message.
- The README documents env vars, how to seed lenders and pincodes, how to run evals, and the open items in section 14.

## 14. Open items for the owner

1. Confirm the FulCRM opportunity stage enum and the exact stage at which Tier 2 unlocks.
2. Confirm the EMI estimation defaults (36 months, 14%) and the ceiling formula.
3. Confirm that a PAN Comprehensive call on an invalid PAN (billed by SurePass) should not count against the LC's five.
4. Test the Experian no-hit shape with the NTC PAN being sourced and update the adapter.
5. Map Experian `Account_Type` and `Account_Status` code tables beyond 11 = open, 13 = closed.
6. Decide whether the score-only Tier 1 should run as a fixed pipeline with the LLM used only for ranking and wording (cheaper, faster) rather than a full agent loop. This PRD already places the LLM in only those two nodes, so the graph is the same either way.
