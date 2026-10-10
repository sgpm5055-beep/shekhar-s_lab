# PRD: Financing Eligibility Agent (Phase 1) on LangGraph, LangChain and LangSmith

Owner: Shekhar Gupta, Product Manager, AI Sales Technology & Platform, Hero Vired
Version: 1.1, 10 October 2026. Reconciled line by line against Phase1_Financing_Eligibility_Spec_v2 (7 Oct 2026). Where this PRD and the v2 spec disagree, the v2 spec wins and this PRD is wrong.
Audience: a coding agent (VS Code) building this from scratch as a standalone service.

Sections 4 to 12 are the build contract. FIXED = product decision, do not change. ASSUMPTION = implement as stated, surface for review. Anything not covered is the implementer's choice but must not weaken a FIXED rule.

---

## 1. Problem and goals

LCs pitch loans to learners who are then rejected; about 72% of rejections are score-driven (low score or new to credit) and the LC finds out only after the application is filed. Phase 1 gives the LC a fundability read inside SuperConnect before the pitch, at a unit cost that survives a Rs 600 CPL. v1 ran PAN Comprehensive, a link check and the full CIBIL report on every check at Rs 91; v2 is two checks on one bureau (Experian via SurePass) with a PAN-keyed cache, about Rs 11 per lead.

Goals (FIXED):
1. From name, mobile, PAN and a logged verbal consent, return a fundability read in one LC action, p95 under 25 s (quick) and 35 s (full).
2. Never show the score, the bureau name or raw report data to the LC; show a recommendation only.
3. Cost per lead under Rs 30 across both checks, zero double billing.
4. Store every raw response, keyed to PAN and opportunity, as the system of record for audit and Phase 3.

## 2. Non-goals (Phase 1)

Document collection, DigiLocker eKYC, learned lender criteria, auto-submission to lenders, co-applicant processing, standalone Aadhaar KYC, any second bureau, showing the score to anyone outside the review queue, a requested loan amount (FulCRM does not hold one; output is a ceiling per lender), name matching of PAN holder vs CRM contact. No UI in this repo; SuperConnect and FulCRM consume the API. The lender criteria admin page writes to the same database and is specified in section 11 but built separately.

## 3. The two checks

| | Quick check | Full check |
|---|---|---|
| Where | Financing section in the SuperConnect sidebar | Financing section inside the opportunity profile |
| When | Any time; 5 billed checks per LC per IST day across all opportunities | Opportunity stage Offer or beyond, read from FulCRM at click time |
| LC enters | Name, mobile, PAN, consent tick. Optional: monthly income, existing monthly EMI | Consent tick (fresh), monthly income (mandatory). Name, mobile, PAN prefilled from the opportunity. Optional: existing EMI as cross-check |
| SurePass calls | PAN Comprehensive (if none stored in 180 days), then Experian Score | Experian Report; PAN Comprehensive only if none stored in 180 days |
| Stops before bureau call when | PAN status not "valid" or category not "person" (Rs 1 spent, no quota used) | Same |
| Output | Headline Fundable / Possible / Unlikely / Not fundable alone / Under review; lenders serving the pincode whose floor the band clears; co-applicant nudge. With income: indicative ceiling per lender. With income and EMI: range and confidence, labelled "based on declared figures" | Full recommendation: loan ceiling and range per lender, top lender with rate band and tenure, up to two alternates, conditions, labelled "bureau report on <date>, income as declared" |
| Billing | One billed Score pull per PAN per 30 days; cache hits free, no quota | One billed Report pull per PAN per 30 days, shared across opportunities; Refresh only when the stored report is over 30 days old; refresh billed, outside any quota |
| Never shown | Score, bureau name | Score, bureau name, raw report |

Both checks produce the same recommendation view; only inputs, cost and label differ. Any Report pull also refreshes the Score cache for that PAN, so a later quick check is free.

## 4. Stack (FIXED)

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Orchestration | LangGraph `StateGraph`, typed state, Postgres checkpointer (`langgraph-checkpoint-postgres`) |
| LLM | LangChain `langchain-openai`, `with_structured_output` on a Pydantic schema, temperature 0, one retry on invalid output, no automatic retries elsewhere |
| Observability | LangSmith tracing on every run; datasets and evaluators in CI |
| HTTP | FastAPI |
| Persistence | Supabase Postgres via `supabase-py`; raw responses in a private encrypted bucket (Supabase Storage) |
| External | SurePass KYC API `https://kyc-api.surepass.app` |
| Model | `gpt-4.1` default, configurable |
| Tests | pytest, fakes for SurePass and the LLM, fixtures are the saved 6 and 7 Oct responses (synthetic copies, never the real file in the repo). No automated tests ever hit SurePass |

Layout:

```
eligibility_agent/
  main.py              FastAPI, auth, routes
  config.py            env; SUREPASS_LIVE kill switch; timeouts
  graph/state.py       CheckState + Pydantic facts models
  graph/build.py       build_graph(kind)
  graph/nodes.py       one function per node in section 7
  graph/prompts.py     ranking prompt
  surepass/client.py   SurePassClient (live) + envelope handling; FakeSurePassClient
  surepass/adapters.py PAN Comprehensive, Experian Score, Experian Report -> facts
  policy/guards.py     consent, cache windows, quota, hard cap, stage gate, locks
  derive/emi.py        existing EMI algorithm (section 8.3)
  derive/range.py      loan range + ceiling (8.4), age (8.5), band (8.1)
  lenders/gate.py      pincode gate
  lenders/safety.py    safety check + code confidence
  output/lc_view.py    redaction boundary
  db/repo.py, db/schema.sql
  evals/
tests/
```

## 5. Policy rules (FIXED, from v2 sections 2, 4, 8, 9)

Implement in `policy/guards.py`; graph nodes call them; the LLM never sees or influences them.

1. Consent first. No SurePass call without a consent record for that pull (score or report). A cache hit needs no new consent and shows the original reference. SurePass does not validate consent on Experian endpoints; this gate is ours alone. `consent: "Y"` is sent only when a record exists.
2. Cache key: HMAC-SHA256 of the PAN with a server-side key held outside the database; shared across LCs and opportunities. Mobile hashed the same way. Last 4 characters of each stored separately for display.
3. PAN Comprehensive window 180 days. Experian Score window 30 days, also refreshed by any Report pull on that PAN. Experian Report window 30 days per PAN; opportunities sharing a PAN share the report and re-run derivations on their own inputs.
4. Cache hit: raw response reused; derivations, gate, ranking and safety check always re-run on the current inputs and the current published lender criteria; shown as "checked on <date>"; free; no quota.
5. Quick-check quota: 5 billed Score pulls per LC per calendar day, reset 00:00 IST, across all opportunities; configurable per LC by a manager; surfaced as "N of 5 checks left today".
6. Counts against quota: a billed Score call that returned HTTP 200 with `success: true` (a score or a no-hit) on a PAN with no stored score. Does not count: cache hits; PAN invalid or not an individual; a refresh of an expired stored score; any service failure.
7. Hard cap regardless of branch: 3 billed bureau calls per PAN per day (score and report together); beyond that the PAN is locked for the day and the run flagged to the review queue.
8. Full-check gate: opportunity stage Offer or beyond, read from FulCRM at click time (the stage is passed in the request by the caller; the service trusts it and logs it).
9. Full-check limit: one billed report per PAN per 30 days. Not per opportunity. Button disabled until the stored report is 30 days old, then Refresh; refresh billed, outside any quota.
10. Forced refresh inside a window: none, for anyone, at either tier.
11. Concurrency: server-side lock per PAN hash; a second request while one is in flight is refused with `IN_FLIGHT`, not queued.
12. Timeout on Score or Report: spend unknown; lock the PAN for 10 minutes, return "pending verification", and require manual reconciliation against the SurePass console before any re-pull. Per-call timeouts PAN 8 s, Score 12 s, Report 20 s, OpenAI 8 s. No automatic retries anywhere.
13. PAN correction on a full check: before any report exists on this opportunity the LC may correct the PAN once (re-verified, logged). After a report exists, a PAN change requires a manager action and is flagged to the review queue. The once-per-window rule is per PAN hash, so a corrected PAN is a new pull.
14. NTC warning: on a full check, a quick-check NTC result under 30 days old triggers a warning before pulling ("no bureau file found on <date>; a full report will most likely return nothing. Proceed?"). The API returns `WARN_NTC_RECENT` unless the request carries `acknowledge_ntc_warning: true`.
15. The LC cannot enter or override a pincode.
16. All SurePass calls server-side; token in the secret store; one static egress IP whitelisted.
17. Development runs on production credits. The client refuses every outbound call unless `SUREPASS_LIVE=true`.

## 6. Inputs, outputs and API

Auth: `Authorization: Bearer <SERVICE_API_TOKEN>` on all routes except `/health`.

`POST /checks/quick`
```json
{ "lc_id": "lc_1", "lead_id": "lead_1", "opportunity_id": null,
  "name": "As typed by LC", "mobile": "98XXXXXXXX", "pan": "ABCDE1234F",
  "consent": { "given": true, "wording_version": "v2-quick", "call_ref": "sc_call_123", "ip": "..." },
  "monthly_income": 60000, "existing_emi": 0 }
```
The LC-typed name lives only in the consent record; bureau calls send PAN Comprehensive's `full_name`.

`POST /checks/full`
Quick fields plus `opportunity_id` (required), `opportunity_stage` (required), `monthly_income` (required), optional `existing_emi`, optional `acknowledge_ntc_warning`, optional `refresh`.

`POST /checks/{check_id}/recompute`
`{ "monthly_income": ..., "existing_emi": ... }`. Income and EMI are editable after a check without a new pull; this re-runs derivations, gate, ranking and safety on the stored raw data and returns a new `LcView`. No SurePass call, no quota.

`POST /checks/{check_id}/attach`
`{ "contact_id": "..." }`. Quick check only: when the mobile matches exactly one FulCRM contact the caller offers "attach to <contact name>"; on LC confirmation this attaches the check and PAN hash to the contact. Until then the result is stored keyed on PAN hash, unattached. The contact lookup itself is the caller's responsibility; the service accepts the attach.

`GET /health` -> `{ ok, surepass_live }`.

Response `LcView` (the only object SuperConnect/FulCRM receive):
```json
{ "check_id": "uuid", "kind": "quick|full",
  "status": "ok | pan_not_verified | refused | pending_verification | service_failure",
  "code": null,
  "headline": "fundable | possible | unlikely | not_fundable_alone | under_review",
  "label": "Indicative, based on what the learner told us",
  "confidence": "high | medium | low",
  "top_lender": { "id": "L1", "name": "...", "rate_band": "11 to 14%", "tenure_range": "12 to 60 months",
                  "ceiling": 450000, "range": { "lower": 380000, "upper": 450000 },
                  "rationale": "...", "conditions": ["..."] },
  "alternates": [ { "...": "same shape, up to two" } ],
  "coapplicant_suggested": false, "coapplicant_reason": null,
  "pan_aadhaar_link_warning": null,
  "items_to_confirm": ["Declared income", "An account last updated 2026-03-12"],
  "checked_on": "2026-10-10", "from_cache": true,
  "refresh_available_on": "2026-11-09",
  "quick_checks_left_today": 4 }
```
Rules: `top_lender` and `alternates` are null/empty when confidence is low (headline only). `range` and `ceiling` appear on a quick check only when income was entered; when EMI was not entered the label says "assuming no other EMIs". `pan_aadhaar_link_warning` is set only when `aadhaar_linked` is false: "PAN not linked to Aadhaar; lenders will ask for this". Never present in this object: score, band boundaries, bureau name, raw report, enquiry counts, account lists, PAN Comprehensive fields other than the link warning.

`code` values: `CONSENT_MISSING, QUOTA_EXHAUSTED, PAN_DAILY_CAP, STAGE_TOO_EARLY, REPORT_WINDOW_OPEN, IN_FLIGHT, PAN_LOCKED, WARN_NTC_RECENT, PAN_CHANGE_NEEDS_MANAGER, NO_PINCODE_ON_PAN, LIVE_CALLS_DISABLED, SUREPASS_FAILURE, OPENAI_FAILURE`.

## 7. Graph (LangGraph)

The flow is deterministic except for one node. The model ranks and explains; it never produces a number and never raises confidence.

State (`graph/state.py`): request fields; `pan_hash`, `mobile_hash`; `admission` facts (which windows are fresh, what will be billed, what will count); `identity`, `score`, `report` (internal facts, never the raw payload in state); `candidate_pincodes`; `gated_lenders`; `derived` (band, existing EMI with flags, FOIR, age at application and at maturity per lender, range and ceiling per lender); `code_confidence`; `model_output`; `safety`; `lc_view`; `audit` (SurePass call log entries, LangSmith run id); `errors`.

```
START
-> admission            consent row written; PAN lock acquired (else IN_FLIGHT); stage gate (full);
                        windows evaluated (PAN 180d, score 30d, report 30d); quota + per-PAN hard cap;
                        NTC warning (full); refusal -> persist_refusal -> END
-> pan_comprehensive    stored < 180d ? reuse : call (8 s). Not valid / not person -> status pan_not_verified,
                        nothing marked checked, no quota -> redact -> persist -> END
-> bureau_pull          quick: Score (12 s) unless stored < 30d. full: Report (20 s) unless stored < 30d;
                        a Report pull also writes the score cache. Timeout -> lock PAN 10 min, status
                        pending_verification -> persist -> END. Service failure -> status service_failure,
                        quota untouched -> persist -> END
-> derive               band; existing EMI (full); FOIR; age; candidate pincodes          [code, 8.x]
-> pincode_gate         lenders with any candidate pincode in their published list. Quick check with no
                        candidate pincode on a valid PAN -> code NO_PINCODE_ON_PAN, band-read view only,
                        skip rank and safety -> redact                                        [code]
-> range                per surviving lender: loan_max lower/upper, ceiling, displayed FOIR  [code, 8.4]
-> code_confidence      high / medium / low from facts                                      [code, 9.4]
-> rank                 OpenAI structured output over survivors (9.2); one retry on invalid output;
                        second failure -> OPENAI_FAILURE, review queue, band read + gated list still shown
-> safety               demote picks that breach floors; none survive -> review              [code, 9.3]
-> redact               LcView                                                               [code]
-> persist              checks row, cache rows, raw response to bucket, surepass_calls, quota accounting,
                        review-queue row when flagged; release PAN lock -> END
```

`build_graph("quick")` and `build_graph("full")` share nodes; they differ in which bureau pull is wired, whether EMI derivation runs, and the label. The Postgres checkpointer is keyed by `check_id`. The `recompute` route runs the sub-graph `derive -> pincode_gate -> range -> code_confidence -> rank -> safety -> redact -> persist` on stored raw data.

## 8. Derivations (code, unit-tested, run before the model sees anything)

### 8.1 Band
`Number(data.credit_score)` from Experian Score or Experian Report, whichever is newer. Half-open buckets `[300,650)`, `[650,720)`, `[720,780)`, `[780,900]`. NTC only when the response is HTTP 200 with `success: true` and `credit_score` is not a 3-digit string. Anything else is a service failure, not NTC. Lender floors are entered as Experian scores (open item in section 12).

### 8.2 Adapters (shapes verified on the 6 and 7 Oct probes)
PAN Comprehensive request `{id_number, get_address: true, get_contact: false, masked_aadhaar_variant: "v1"}`; use `data.status == "valid"`, `data.category == "person"`, `data.full_name`, `data.dob` (YYYY-MM-DD), `data.address.zip`, `data.aadhaar_linked` (bool), `data.masked_aadhaar` (stored only), `data.client_id`. Experian takes no gender; do not derive or send one.

Experian Score request `{name, mobile, pan, consent: "Y"}`; use `data.credit_score`, `data.client_id`.

Experian Report, `A = data.credit_report.CAIS_Account.CAIS_Account_DETAILS[]`, every key present, `""` = not reported, dates YYYYMMDD: `Date_Closed == ""` is open (agrees with `Account_Status "11"`); `Scheduled_Monthly_Payment_Amount`; `Current_Balance`, `Highest_Credit_or_Original_Loan_Amount`, `Credit_Limit_Amount`; `Repayment_Tenure` ("0" = not reported), `Terms_Duration` ("024"), `Rate_of_Interest`; `Account_Type`, `Portfolio_Type` (I/R); `Date_Reported` (staleness guard: open account reported more than 90 days before `CreditProfileHeader.ReportDate` is unconfirmed); `Amount_Past_Due`, `Payment_History_Profile`, `CAIS_Account_History[].Days_Past_Due`, `Written_off_Settled_Status`, `Payment_Rating` (risk signals); `CAIS_Holder_Address_Details[].ZIP_Postal_Code_non_normalized` (bureau pincodes); `CAIS_Holder_Details[].Date_of_birth` stored only, never used for age; `CAIS_Summary`, `TotalCAPS_Summary.TotalCAPSLast{7,30,90,180}Days`, `CAPS_Application_Details[]`; `Match_result.Exact_match` ("Y" on a hit; no-hit shape untested); `Account_Number` not used. Parse with Pydantic models in passthrough mode (`extra="allow"`); every number arrives as a string.

### 8.3 Existing EMI (full check only)
Over open accounts, skipping any whose `Date_Reported` is older than 90 days (flagged stale, listed to the LC as "confirm with learner: an account last updated <date>"):
1. If `Scheduled_Monthly_Payment_Amount` is a positive number, use it (reported).
2. Else tenure = `Repayment_Tenure` if positive, else `Terms_Duration` if positive. If tenure exists and `Current_Balance` is positive: `remaining_months = tenure - months_since(Open_Date)`. If `remaining_months >= 1`, estimate `Current_Balance / remaining_months`. If `< 1`, estimate `Current_Balance / 12` and flag "past tenure, confirm with learner".
3. Else if `Portfolio_Type == "R"`: estimate 5% of `Current_Balance`.
4. Else: estimate `Current_Balance / 36`.
5. `existing_emi` = sum of reported + estimated; `emi_floor_flag = true` when any open account was estimated or flagged. `Rate_of_Interest` is stored and passed to the model as context, not used in the estimate.

When the LC also declared an existing EMI, show both to the model and the safety check; if declared is higher, use declared. Quick check with income but no EMI: `existing_emi = 0`, labelled "assuming no other EMIs".

### 8.4 FOIR, range and ceiling (per lender, from published admin criteria)
`current_foir = existing_emi / declared_income`; undefined without income (quick check without income carries no FOIR, no range, no ceiling).

`available_emi = max(0, foir_limit * income - existing_emi)`. At 0 the lender is "not fundable alone".
`loan_max = available_emi * ((1+r)^n - 1) / (r (1+r)^n)` with `r = annual_rate / 12`, `n = tenure_months`. Upper bound at the lender's minimum rate and maximum tenure; lower bound at its maximum rate and minimum tenure; both clipped to the lender's loan-amount range. The ceiling is `loan_max_upper`; displayed FOIR = `(existing_emi + emi(loan_max at max rate, max tenure)) / income`. Output "up to loan_max" with range `[lower, upper]`.

### 8.5 Age
From PAN Comprehensive `dob`, at application and at maturity (the lender's maximum tenure ahead). No dob on the stored PAN record: age check skipped, confidence capped at medium. Bureau holder DOB rows are never used.

### 8.6 Candidate pincodes
Quick: PAN `address.zip`. Full: PAN `address.zip` plus `ZIP_Postal_Code_non_normalized` from open accounts with `Date_Reported` in the last 12 months. All must match `^[1-9][0-9]{5}$`. Closed or older accounts excluded.

## 9. Matching

### 9.1 Pincode gate (code)
A lender stays in only if at least one candidate pincode is in its published approved-pincode list (CSV in the admin UI, optional state list). No candidate pincode on a quick check: skip ranking, show the band read with "no pincode on PAN record; lender matching available after the full check". Never silently skip the gate.

### 9.2 Ranking (OpenAI, structured output)
Input to the model: band bucket only (never the raw score), FOIR and its floor flag, existing EMI split reported vs estimated with flags, declared income, loan ceiling and range per lender, age at application and at maturity, Aadhaar link status, recent-enquiry count, past-due flag, stale-account flag, and for each surviving lender its normalised criteria plus policy notes. Never in the payload: PAN, name, mobile, Aadhaar, account numbers, raw report. temperature 0; one retry on invalid output, then review queue.

Output schema (Pydantic, validated before anything is displayed; the model never produces a number):
```python
class Pick(BaseModel):
    lender_id: str; rank: int; rationale: str; conditions: list[str]
class ModelOutput(BaseModel):
    recommendation: Pick | None
    alternates: list[Pick] = Field(max_length=2)
    confidence: Literal["high","medium","low"]    # may only lower code_confidence
    co_applicant_suggested: bool
    co_applicant_reason: str | None
    review_flags: list[str]
```
Prompt stored in LangSmith Hub as `hero/eligibility-rank`, pinned by tag per release. Core instructions: rank only the lenders given, never invent one, never suggest overriding the gate; prefer co-applicant over "not fundable" for NTC or below-floor bands and thin FOIR headroom; rationale and conditions are plain language for an LC, no numbers from the report, no bureau name; confidence may be lowered from the given value, never raised.

### 9.3 Safety check (code)
For the top pick and each alternate: band at or above the lender's floor (or NTC accepted by that lender); FOIR at the ceiling at or below the limit; age inside the lender's bounds when DOB is known; ceiling at or above the lender's minimum loan amount. A breach demotes the pick. No pick survives: run goes to the review queue, headline "Under review" or "Not fundable alone".

### 9.4 Confidence (code first, model may lower)
High only when identity is verified, EMI fully reported, no stale or flagged account, no declared-only figure, and the band clears the floor by at least one bucket. Medium when any of those is missing (including unknown age). Low when the safety check demoted every pick, the model failed twice, or no lender survives the gate. Final = `min(code_confidence, model.confidence)`.

| Confidence | LC sees |
|---|---|
| high | "Recommended" with lender and ceiling |
| medium | "Possible, confirm with learner" with lender and ceiling plus items to confirm |
| low | Headline only ("Under review" or "Not fundable alone, co-applicant suggested"); no lender named; run in the review queue |

No lender surviving the gate or the band is one path: low, "Not fundable alone", `co_applicant_suggested: true`, review queue for Credit to see the pattern.

## 10. Failure handling (one screen message, one stored outcome, no retries)

| Call | Service failure (timeout, 5xx, ip_not_whitelisted, invalid_token, success: false) | Negative but valid (200, success: true) |
|---|---|---|
| PAN Comprehensive | "Couldn't verify PAN, retry"; no bureau call, no quota | status != valid or category != person: "PAN not verified"; stop; LC may correct and retry; nothing marked checked |
| Experian Score | "Credit check couldn't complete, retry"; quota not used; PAN result kept; timeout locks PAN 10 min; per-PAN hard cap still applies | credit_score not a 3-digit string: NTC; quota used; stored 30 days; "Unlikely alone, co-applicant suggested" plus lenders accepting NTC in the pincode |
| Experian Report | Same; opportunity not marked checked; timeout locks PAN 10 min, reconcile before re-pull | No file: stored as NTC for 30 days; same output, labelled with report date |
| OpenAI | "Couldn't generate a recommendation, retry" after one retry; run flagged; band read and gated lender list still shown | n/a |
| ip_not_whitelisted, invalid_token | Alert engineering; no retry; LC sees a service failure | n/a |

## 11. Data, security and the lender criteria admin

Consent record per pull: wording version, timestamp, LC id, opportunity id and contact id when attached, mobile hash, PAN hash, SuperConnect call reference when applicable, IP, which call it covers (score or report).

Storage: every raw response verbatim, encrypted, in a private bucket keyed to PAN hash and opportunity, with client_id, endpoint, status, message_code, timestamp. Provisional retention: raw bureau responses 180 days, consent records 5 years; schema carries a TTL column and a purge job from day one. Only masked Aadhaar is ever received or stored.

Logs: client_id, endpoint, status, message_code, latency. Never request bodies, response data, full PAN or mobile. LangSmith metadata carries `check_id`, `pan_hash`, tier, admission mode, billed flag; never PII.

Access: LCs see the recommendation view for their own opportunities and quick checks they ran. Raw data and the review queue are Credit / Post-sales roles only.

Tables: `lenders` (versioned, published flag, normalised criteria with each field marked known / missing / uncertain), `lender_pincodes`, `consents`, `pan_records` (PAN Comprehensive cache, 180 d), `score_cache` (30 d), `report_cache` (30 d), `checks` (status, code, billed, counted_against_quota, is_refresh, consent_id, contact_id, lc_view, internal, langsmith_run_id), `surepass_calls`, `pan_locks` (in-flight and 10-minute timeout locks), `review_queue`, `lc_quota_overrides`, plus checkpointer tables. SQL functions: `quick_checks_used_today(lc_id)` on the IST date; `bureau_calls_today(pan_hash)`.

Lender criteria admin (separate UI, same tables): minimum score floor as an Experian score plus NTC handling; FOIR limit; minimum income, optionally by employment type; age limits at application and maturity; approved pincodes CSV plus optional state list; interest-rate band; tenure range; loan-amount range; free-text policy notes. A lender missing a required field is excluded from matching until fixed. Matching and the model always read the published version.

## 12. LangSmith, tests, acceptance, open items

LangSmith: trace every run with tags `tier`, `admission`; datasets `eligibility-quick-golden` and `eligibility-full-golden` from synthetic fixtures modelled on the probes (the real file never leaves the lab folder). Evaluators: `no_leak` (no bureau name, no 300 to 899 number, no account list in any LC-facing string), `lenders_gated` (picks are a subset of gate survivors that passed safety), `numbers_from_code` (every amount in the view equals the code-derived value), `confidence_not_raised` (final <= code confidence), `billing_consistency` (billed, counted, from_cache match admission and the fake client's call log), `outcome_matches_band` (NTC or below-floor never yields a lender without co-applicant), and an LLM judge on rationale wording. CI runs evaluators with the fake LLM on the full set and the real model on at most 20 examples.

Required tests (pytest, no network): quick happy path (PAN then Score once, billed, counted, no leak); same PAN second LC within 30 days (free, derivations re-run on new inputs); sixth billed quick check refused; PAN not verified (Rs 1 call only, no quota); NTC on Score (counted, cached 30 d, co-applicant); refresh of an expired score (billed, not counted); 3-per-PAN-per-day hard cap locks the PAN; full check before Offer refused; full check at Offer (PAN reused if under 180 d, Report once, Score cache refreshed, second opportunity on the same PAN inside 30 days is a free hit); PAN correction once before report, manager flag after; NTC warning unless acknowledged; EMI algorithm on all four branches plus the stale and past-tenure flags; range upper/lower and clipping; no-pincode quick check returns band read with `NO_PINCODE_ON_PAN`; model picks a lender outside the gate or below floor (demoted; none survive goes to review); model raises confidence (ignored); model output invalid twice (OPENAI_FAILURE with band read and gated list still returned); timeout locks the PAN 10 minutes and returns pending_verification; concurrent request refused IN_FLIGHT; recompute re-runs without a SurePass call; `SUREPASS_LIVE` unset makes every client call raise; adapters.

Acceptance: all tests pass; `pyright` clean; a LangSmith trace per API call; with `SUREPASS_LIVE` unset the service serves `/health` and returns `LIVE_CALLS_DISABLED` instead of calling; code review confirms no path from raw responses to `LcView` or to any LLM message; README covers env, seeding lenders and pincodes, running evals, and the open items below. Success metrics for the first 30 days (from v2): SurePass success rate at or above 98% excluding invalid-input 4xx; p95 quick under 25 s and full under 35 s; weekly cache hit rate; cost per lead under Rs 30; zero double-billed pulls; 100% of raw responses persisted; zero tokens, full PAN, Aadhaar or report values in logs, browser or OpenAI payloads; model output schema-valid first try at or above 95%; review-queue rate tracked with reasons.

Open items (owner in brackets, from v2 section 10): lender floors are CIBIL scores, band is Experian (Credit); whether SurePass caches on its side (Shekhar with SurePass); soft vs hard pull in writing for both Experian endpoints (Shekhar with SurePass); Experian code tables for Account_Type, Account_Status, Written_off_Settled_Status, Payment_Rating (SurePass); one NTC PAN and one wrong-PAN probe on consenting identities before the NTC branch is coded (Shekhar); `get_contact: false` and `masked_aadhaar_variant: "empty"` acceptance, one Rs 1 call (engineering); two mockups, quick and full (design); at least three lenders loaded with real floors, FOIR limits and pincode CSVs before UAT (Credit); compliance sign-off on retention (compliance); manager UI for per-LC quota override (product).
