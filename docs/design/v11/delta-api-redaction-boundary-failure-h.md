# API, redaction boundary, failure handling, tests, evals, pyright (PRD v1.1 sections 6, 10, 12)

## Summary
Redesigns main.py (5 routes: quick/full/recompute/attach/health; consent as an object with given: Literal[True]; no refresh field on quick), config.py (PII_HMAC_KEY for HMAC-SHA256 PAN+mobile hashing, RAW_BUCKET, FIXED per-call timeouts 8/12/20/8s as code constants, degraded-boot guarantee intact), and output/lc_view.py (LcView v2 with status/code/headline/label/top_lender{ceiling,range{lower,upper}}, two-tier leak scan: binding BUREAU_TERM_RE+STANDALONE_SCORE_RE for LLM prose plus a names-only global tier so the PRD-mandated 'bureau report on <date>' label survives, new ACCOUNT_ID_RE for the no-account-list rule, headline decision table, exact section-10 failure strings carried in `label`). Replaces mypy with pyright strict ([tool.pyright] config, pyright[nodejs] pin, supabase-boundary file pragma, tests executionEnvironment). Specifies the full ~22-scenario pytest suite with precomputed v1.1 annuity-range and four-branch-EMI constants (e.g. L1 quick ceiling 1,277,000 / range 368,000–1,277,000; EMI branches 3500+8000+5000+2000+2500=21,000 with stale 122-day account excluded), and the seven v1.1 evaluators (no_leak+account-list, lenders_gated, numbers_from_code, confidence_not_raised, billing_consistency, outcome_matches_band, LLM judge) with fake-on-full-set + real-model<=20 CI. Reuses v1.0 assets by name: FrozenClock, dense envelope builders, pytest-socket harness (no filterwarnings), probe-literal guard, C11 scrub vectors, C13/C14 config/app-factory patterns. NTC fixtures flip to 200+success:true+non-3-digit only; success:false becomes SUREPASS_FAILURE.

## Files
### eligibility_agent/config.py
_Env registry v2, kill switch, canonical HMAC PII hashers, FIXED timeout constants, degraded-boot guarantee. Single source of truth for every env var; v1.0 C13 pattern survives with renames._

ENV TABLE (pydantic-settings, every field optional-with-default so Settings() constructs from empty env; constructor kwargs use alias names, tests pass _env_file=None):
| env | field | type | default | semantics |
|---|---|---|---|---|
| SUREPASS_LIVE | surepass_live_flag | str | "" | live iff value == "true" EXACT lowercase literal (BINDING, unchanged) |
| SUREPASS_TOKEN | surepass_token | str|None | None | required only when live |
| SUREPASS_BASE_URL | surepass_base_url | str | "https://kyc-api.surepass.app" | |
| SERVICE_API_TOKEN | service_api_token | str|None | None | None => auth disabled + startup WARNING; hard ConfigError when REPO_BACKEND=supabase |
| PII_HMAC_KEY | pii_hmac_key | str | DEFAULT_DEV_HMAC_KEY ("dev-only-hmac-key-do-not-use-in-prod") | HMAC-SHA256 key for PAN+mobile (PRD 5.2 'server-side key held outside the database'); default forbidden with supabase backend |
| REPO_BACKEND | repo_backend | Literal["memory","supabase"] | "memory" | |
| SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY | ... | str|None | None | |
| RAW_BUCKET | raw_bucket | str | "surepass-raw" | private encrypted Supabase Storage bucket for raw responses |
| DATABASE_URL | database_url | str|None | None | checkpointer DSN (session pooler :5432 or direct; ONE name, C13) |
| OPENAI_API_KEY | openai_api_key | str|None | None | |
| OPENAI_MODEL | openai_model | str | "gpt-4.1" | |
| LANGSMITH_TRACING | langsmith_tracing_flag | str | "" | on iff == "true" |
| LANGSMITH_API_KEY / LANGSMITH_PROJECT | ... | None / "eligibility-agent" | |
| PROMPT_SOURCE | prompt_source | Literal["local","hub"] | "local" | ONE prompt now (hero/eligibility-rank, ref constant in graph/prompts.py) |
| PROMPT_TAG | prompt_tag | str|None | None | release pin |
| GRAPH_TIMEOUT_SECONDS | graph_timeout_seconds | float | 60.0 | route-level asyncio.wait_for budget (p95 full < 35s + headroom) |
| LOG_LEVEL | log_level | str | "INFO" | |
DELETED from v1.0: PAN_HASH_SALT (replaced by PII_HMAC_KEY), OPENAI_TEMPERATURE (FIXED 0), PROMPT_FINALIZE_* (one LLM node), PROMPT_RANK_REF/TAG split (single PROMPT_TAG).

FIXED TIMEOUT CONSTANTS (Final, NOT env — PRD 5.12 is FIXED; surepass/client.py and graph/llm factory import them):
```python
TIMEOUT_PAN_S: Final[float] = 8.0
TIMEOUT_SCORE_S: Final[float] = 12.0
TIMEOUT_REPORT_S: Final[float] = 20.0
TIMEOUT_OPENAI_S: Final[float] = 8.0
CONNECT_TIMEOUT_S: Final[float] = 5.0   # connect phase; read timeout is per-endpoint above
```
(Policy constants QUICK_DAILY_QUOTA_DEFAULT=5, PAN_DAILY_BUREAU_CAP=3, SCORE_TTL_DAYS=30, REPORT_TTL_DAYS=30, PAN_TTL_DAYS=180, TIMEOUT_LOCK_MINUTES=10 live in policy/guards.py, not here — precedent C13.)

CANONICAL HASHERS (single home; policy/db/evals/main import; no second definition anywhere — C3 successor):
```python
def hash_pii(value: str, *, key: str) -> str:
    """hex HMAC-SHA256(key, value) -> 64 lowercase hex chars."""
    return hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()

def hash_pan(pan: str, *, key: str) -> str:
    return hash_pii(pan.strip().upper(), key=key)

def hash_mobile(mobile: str, *, key: str) -> str:   # input must already be the normalized 10-digit form
    return hash_pii(mobile.strip(), key=key)

def last4(value: str) -> str:
    return value.strip()[-4:]          # stored separately for display (PRD 5.2)
```

Settings properties: surepass_live (== "true" exact), langsmith_tracing, auth_enabled. startup_errors() -> list[str]: supabase backend requires SUPABASE_URL+KEY, non-default PII_HMAC_KEY, SERVICE_API_TOKEN; live requires SUREPASS_TOKEN; empty env -> [] (degraded boot, acceptance criterion). ConfigError(RuntimeError). utcnow() -> aware UTC datetime. @lru_cache get_settings().

tests/test_config.py: empty-env constructs + startup_errors()==[]; kill-switch exact literal parametrized over {"true"} vs {"True","TRUE","1","yes",""}; supabase-backend error triplet; live-without-token; hash_pan(" abcde1234f ", key="k") == hmac.new(b"k", b"ABCDE1234F", sha256).hexdigest() == hash_pan("ABCDE1234F", key="k"), key-sensitivity, 64 lowercase hex; hash_mobile distinct digest from hash_pan for same key; last4("ABCDE1234F")=="234F".

### eligibility_agent/graph/state.py (LcView v2 wire models — defined here, hosted by the graph/state owner verbatim)
_The exact LcView v2 of PRD v1.1 section 6. Defined by this section as the wire contract; graph/state.py hosts it (v1.0 precedent)._

```python
Status = Literal["ok", "pan_not_verified", "refused", "pending_verification", "service_failure"]
Code = Literal[
    "CONSENT_MISSING", "QUOTA_EXHAUSTED", "PAN_DAILY_CAP", "STAGE_TOO_EARLY",
    "REPORT_WINDOW_OPEN", "IN_FLIGHT", "PAN_LOCKED", "WARN_NTC_RECENT",
    "PAN_CHANGE_NEEDS_MANAGER", "NO_PINCODE_ON_PAN", "LIVE_CALLS_DISABLED",
    "SUREPASS_FAILURE", "OPENAI_FAILURE",
]
Headline = Literal["fundable", "possible", "unlikely", "not_fundable_alone", "under_review"]
Confidence = Literal["high", "medium", "low"]

class LoanRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lower: int
    upper: int

class LenderView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    rate_band: str          # e.g. "11 to 14%" — admin string, leak-guarded
    tenure_range: str       # e.g. "12 to 48 months"
    ceiling: int | None     # None on quick without income
    range: LoanRange | None
    rationale: str          # LLM prose, scrubbed; "" when scrubbed away or code-built view
    conditions: list[str] = Field(default_factory=list)   # LLM prose, scrubbed

class LcView(BaseModel):
    model_config = ConfigDict(extra="forbid")   # no undeclared field can ride along
    check_id: str
    kind: Literal["quick", "full"]
    status: Status
    code: Code | None = None
    headline: Headline | None = None            # None when no band is known (refusals, SUREPASS failures)
    label: str                                  # the ONE human line: provenance label on ok, section-10 message otherwise
    confidence: Confidence | None = None
    top_lender: LenderView | None = None
    alternates: list[LenderView] = Field(default_factory=list, max_length=2)
    coapplicant_suggested: bool = False
    coapplicant_reason: str | None = None
    pan_aadhaar_link_warning: str | None = None # EXACT string or None, never False
    items_to_confirm: list[str] = Field(default_factory=list)
    checked_on: str | None = None               # "YYYY-MM-DD" IST date of the raw pull (original date on cache hits)
    from_cache: bool = False
    refresh_available_on: str | None = None     # "YYYY-MM-DD" = pull date + 30d
    quick_checks_left_today: int | None = None  # quick only; None on full (C12 precedent)
```
Field-by-field leak audit: check_id/kind/status/code/headline/confidence = closed enums or uuid (no). ceiling/range = ints rounded half-up to 1,000 by code (a smuggled 736 becomes 1000) (no). id = DB key, membership-verified (no). name/rate_band/tenure_range = admin strings -> global-tier line_leaks, fallback id / "" (guarded). rationale/conditions/items_to_confirm/coapplicant_reason = LLM or derived prose -> LLM-tier scrub, whole-line drop (SCRUBBED). label = fixed code constant, global-tier self-test (no). pan_aadhaar_link_warning = fixed constant or None (no). checked_on/refresh_available_on = code dates (no). from_cache/coapplicant_suggested/quick_checks_left_today = code (no). extra="forbid" => score/bureau/raw report/enquiry counts/account lists have no field to live in.

### eligibility_agent/output/lc_view.py
_The redaction boundary v2: two-tier leak regexes (binding pair survives; names-only global tier + new account-list rule), whole-line scrub, headline decision table, all fixed LC strings (section 10 matrix + refusal messages + labels), and the four view builders. Sole home; tests and evals import, never copy._

REGEXES (BUREAU_TERM_RE and STANDALONE_SCORE_RE are BINDING v1.0 C11 verbatim):
```python
BUREAU_TERM_RE: Final = re.compile(r"\b(?:cibil|experian|equifax|crif|bureaus?)\b", re.IGNORECASE)
BUREAU_NAME_RE: Final = re.compile(r"\b(?:cibil|experian|equifax|crif)\b", re.IGNORECASE)  # names only — v1.1 mandates the literal label 'bureau report on <date>' so the generic term cannot be banned globally
STANDALONE_SCORE_RE: Final = re.compile(r"(?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d)")
ACCOUNT_ID_RE: Final = re.compile(   # NEW: 'no account list in any LC-facing string' (PRD 12 no_leak)
    r"(?i)(?:\b(?:a/?c|acct|account)\s*(?:no\.?|number|#)\s*[:\-]?\s*\S+)"
    r"|(?:\b[x*]{2,}[\s-]?\d{3,}\b)"      # masked ids: XXXX1234
    r"|(?:\b\d{9,18}\b)")                  # bare 9-18 digit runs: account-number-like, never legit prose (amounts are < 1e8 and grouped)
LLM_LEAK_PATTERNS: Final = (BUREAU_TERM_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE)      # LLM/derived prose fields
GLOBAL_LEAK_PATTERNS: Final = (BUREAU_NAME_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE)   # every string leaf incl. fixed labels

def line_leaks(line: str, patterns: Sequence[re.Pattern[str]] = LLM_LEAK_PATTERNS) -> bool: ...
def scrub_lines(lines: Sequence[str]) -> list[str]:  # whole-line drop, never partial masking (v1.0 rule survives)
```
VECTOR TABLE (test-pinned; v1.0 C11 battery survives verbatim — DROP 736/(736)/Rs.736/₹736/below 650/bureaus differ/tenure of 360 months; KEEP 3,500/45,000/3,50,000/12,34,567/0.736/736.50/36 months/14 percent/1736/900/299/crifled) PLUS new account-list vectors: DROP "account number 123456789012", DROP "a/c no: SYNTH0001", DROP "XXXX1234 is overdue", DROP "loan 123456789 outstanding" (9-digit run); KEEP "An account last updated 2026-03-12" (word 'account' alone is mandated copy), KEEP "two accounts are open", KEEP "loan up to 12,34,567" (grouped). Global-tier vectors: KEEP "bureau report on 2026-10-07, income as declared" under GLOBAL_LEAK_PATTERNS but DROP under LLM_LEAK_PATTERNS.

FIXED STRINGS (every one passes line_leaks(s, GLOBAL_LEAK_PATTERNS) by unit self-test):
```python
LABEL_QUICK = "Indicative, based on what the learner told us"
LABEL_QUICK_NO_EMI = "Indicative, based on what the learner told us, assuming no other EMIs"  # income given, EMI not
LABEL_FULL_FMT = "bureau report on {report_date}, income as declared"   # report_date = YYYY-MM-DD, PRD-verbatim casing
AADHAAR_WARNING = "PAN not linked to Aadhaar; lenders will ask for this"      # EXACT PRD 6 string
MSG_PAN_SERVICE_FAILURE = "Couldn't verify PAN, retry"                        # PRD 10 verbatim
MSG_PAN_NOT_VERIFIED = "PAN not verified. Correct the PAN with the learner and run the check again; no quota was used."
MSG_BUREAU_SERVICE_FAILURE = "Credit check couldn't complete, retry"          # PRD 10 verbatim (score & report)
MSG_PENDING_VERIFICATION = "Pending verification: the credit check didn't complete. This learner is locked for 10 minutes while the result is reconciled; please try again after that."
MSG_OPENAI_FAILURE = "Couldn't generate a recommendation, retry"              # PRD 10 verbatim
MSG_INTERNAL = "The check could not be completed automatically. Please raise it for manual review."  # v1.0 MANUAL_REVIEW_LINE survives
MSG_NO_PINCODE = "No pincode on the PAN record; lender matching will be available after the full check"  # PRD 9.1
REFUSAL_MESSAGES: Final[dict[str, str]] = {
  "CONSENT_MISSING": "Consent was not recorded. Take the learner's verbal consent, tick the consent box and run the check again.",
  "QUOTA_EXHAUSTED": "You have used all your quick checks for today. The limit resets at midnight; saved results stay available.",
  "PAN_DAILY_CAP": "This learner's checks have reached today's safety limit. The record is locked until tomorrow and has been sent for review.",
  "STAGE_TOO_EARLY": "A full check unlocks once this opportunity reaches the Offer stage. Until then, use the quick check.",
  "REPORT_WINDOW_OPEN": "This learner's full report is still current. Refresh unlocks once it is 30 days old.",
  "IN_FLIGHT": "A check for this learner is already running. Please wait for it to finish.",
  "PAN_LOCKED": "This learner's record is locked for a few minutes while an earlier check is reconciled. Please try again shortly.",
  "WARN_NTC_RECENT": "No bureau file was found on {date}; a full report will most likely return nothing. Proceed?",  # PRD 5.14 verbatim; .format(date=...) at build time; passes GLOBAL tier (generic 'bureau' allowed there)
  "PAN_CHANGE_NEEDS_MANAGER": "A report already exists for this opportunity, so changing the PAN needs a manager. This has been flagged for review.",
  "LIVE_CALLS_DISABLED": "Eligibility checks are temporarily switched off on the server. Please try again later or contact the platform team.",
}
```

HEADLINE DECISION TABLE (pure function, code-only; model never chooses it):
```python
def compute_headline(*, band: str, has_pick: bool, confidence: Confidence,
                     none_survived_gate_or_band: bool, demoted_or_model_failed: bool) -> Headline:
    # order matters:
    # 1. none_survived_gate_or_band (no lender in pincode, or band/NTC below every gated floor)
    #       -> "not_fundable_alone"            (coapplicant_suggested forced True by builder)
    # 2. demoted_or_model_failed (picks existed; safety demoted all, or rank invalid twice at low conf)
    #       -> "under_review"
    # 3. has_pick and band in {"NTC", "[300,650)"}   -> "unlikely"   (PRD 10: 'Unlikely alone, co-applicant suggested' + NTC-accepting lenders)
    # 4. has_pick and confidence == "high"            -> "fundable"
    # 5. has_pick and confidence == "medium"          -> "possible"
    # 6. defensive                                     -> "under_review"
```

BUILDERS (pure functions of state + context; no repo access):
```python
@dataclass(frozen=True, slots=True)
class RedactionContext:
    quick_checks_left_today: int | None    # computed by redact node: max(0, quota - used - (1 if will_count else 0)); None for full
    checked_on: date | None                # IST date of the raw pull that produced the facts (original date on cache hits)
    from_cache: bool
    refresh_available_on: date | None      # checked_on + 30d for the governing cache window

def build_lc_view(state: CheckState, ctx: RedactionContext) -> LcView           # ok, band-read (NO_PINCODE_ON_PAN / OPENAI_FAILURE), pan_not_verified branches
def build_refusal_lc_view(*, check_id: str, kind: Kind, code: Code, ctx: RedactionContext, warn_date: date | None = None) -> LcView
def build_failure_lc_view(*, check_id: str, kind: Kind, status: Literal["pending_verification","service_failure"], code: Code | None, label: str, quick_checks_left_today: int | None) -> LcView
def build_error_lc_view(*, check_id: str, kind: Kind) -> LcView                 # internal crash: service_failure, code None, label MSG_INTERNAL
```
build_lc_view OK branch: resolve LenderView per safety-surviving pick from state["derived"] per-lender numbers — ceiling = round_to_nearest_1000(derived.ceiling) (import the half-up helper from derive.range, the ONLY rounding home), range = LoanRange(lower=round_to_nearest_1000(derived.lower), upper=round_to_nearest_1000(derived.upper)); on quick without income ceiling=None, range=None. name/rate_band/tenure_range global-tier guarded (name falls back to lender id, rate/tenure strings to ""). rationale: line_leaks -> ""; conditions/items_to_confirm: scrub_lines. confidence low => top_lender=None, alternates=[] (headline only — PRD 6 rule). pan_aadhaar_link_warning = AADHAAR_WARNING iff identity.aadhaar_linked is False else None. label selection: full -> LABEL_FULL_FMT.format(report_date=...); quick+income+no-declared-EMI -> LABEL_QUICK_NO_EMI; quick otherwise -> LABEL_QUICK; NO_PINCODE_ON_PAN view -> MSG_NO_PINCODE; OPENAI_FAILURE band-read -> MSG_OPENAI_FAILURE.

FAILURE MATRIX (section 10, exact mapping — route/persist both honor it):
| event | status | code | label | quota counted | billed flag | side effects |
|---|---|---|---|---|---|---|
| PAN transport/5xx/timeout/success:false | service_failure | SUREPASS_FAILURE | MSG_PAN_SERVICE_FAILURE | never | from surepass_calls log | none |
| PAN 200 valid-negative (status!=valid or category!=person) | pan_not_verified | None | MSG_PAN_NOT_VERIFIED | never | True (Rs 1) | nothing marked checked; LC may correct+retry |
| Score/Report 5xx, success:false, auth, non-timeout transport | service_failure | SUREPASS_FAILURE | MSG_BUREAU_SERVICE_FAILURE | never | PAN kept | success:false is NEVER NTC in v1.1 |
| Score/Report READ TIMEOUT | pending_verification | SUREPASS_FAILURE | MSG_PENDING_VERIFICATION | never | unknown -> internal.spend_unknown=true | PAN locked 10 min; manual reconciliation before re-pull |
| ip_not_whitelisted / invalid_token (401/403) | service_failure | SUREPASS_FAILURE | MSG_BUREAU_SERVICE_FAILURE | never | False | logger.critical(event="surepass_auth_failure") + review_queue row (ops alert on both) |
| OpenAI invalid output twice OR 8s timeout | service_failure | OPENAI_FAILURE | MSG_OPENAI_FAILURE | yes (spend happened) | True | review_queue row; headline band-read kept; top/alternates empty (see decisions) |
| NTC (200 + success:true + non-3-digit credit_score) | ok | None | normal label | yes | True | cached 30d as NTC; headline per table (unlikely/not_fundable_alone) + coapplicant |
| unexpected exception / route timeout | service_failure | None | MSG_INTERNAL | never | best-effort | record_check_error with terminal guard (C1 survives) |
Subsequent request during the 10-min lock -> refused PAN_LOCKED; during in-flight -> refused IN_FLIGHT (never queued).

### eligibility_agent/main.py
_FastAPI composition root v2: five routes, consent-object validation, recompute/attach, auth, 422-echo stripping, error boundary. v1.0 C14 pattern survives (app factory, lifespan graph compile, run_in_threadpool + sync PostgresSaver, terminal-guarded error row)._

REQUEST MODELS (pydantic v2, extra="forbid"; PAN/mobile regexes and normalizers survive v1.0 verbatim):
```python
PAN_PATTERN = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
MOBILE_PATTERN = r"^[6-9][0-9]{9}$"

class ConsentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    given: Literal[True]                      # C21 BINDING survives: false/absent/non-bool -> 422, nothing recorded
    wording_version: str = Field(min_length=1, max_length=32)   # e.g. "v2-quick"
    call_ref: str | None = Field(default=None, min_length=1, max_length=128)
    ip: str | None = Field(default=None, max_length=64)

class QuickCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lc_id: str = Field(min_length=1, max_length=64)
    lead_id: str = Field(min_length=1, max_length=64)
    opportunity_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)   # lives ONLY in the consent record; bureau calls use PAN full_name
    mobile: str = Field(pattern=MOBILE_PATTERN)       # before-validator: strip spaces/hyphens, removeprefix('+91')
    pan: str = Field(pattern=PAN_PATTERN)             # before-validator: strip().upper()
    consent: ConsentIn
    monthly_income: float | None = Field(default=None, gt=0, le=100_000_000)
    existing_emi: float | None = Field(default=None, ge=0, le=100_000_000)
    # NO refresh field: v1.1 section 6 quick body has none; expired-score re-pull is automatic (billed, not counted)

class FullCheckRequest(QuickCheckRequest):
    opportunity_id: str = Field(min_length=1, max_length=64)      # required override
    opportunity_stage: OpportunityStage                            # enum membership = 422; ordering gate = policy refusal (C18 survives)
    monthly_income: float = Field(gt=0, le=100_000_000)            # required override
    acknowledge_ntc_warning: bool = False
    refresh: bool = False

class RecomputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monthly_income: float | None = Field(default=None, gt=0, le=100_000_000)
    existing_emi: float | None = Field(default=None, ge=0, le=100_000_000)
    # semantics: body values REPLACE the stored declared figures outright (explicit null clears)

class AttachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact_id: str = Field(min_length=1, max_length=64)

class AttachResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    check_id: str; contact_id: str; attached: bool
```

APP FACTORY (C14 shape, one LLM seam now):
```python
def create_app(settings: Settings | None = None, *, repo: Repo | None = None,
               surepass: SurePassClient | None = None, rank_llm: RankLLM | None = None,
               checkpointer: BaseCheckpointSaver | None = None, prompts: PromptBundle | None = None,
               now: Callable[[], datetime] | None = None) -> FastAPI
def app_factory() -> FastAPI   # uvicorn 'eligibility_agent.main:app_factory' --factory
```
Lifespan: startup_errors -> ConfigError; resolve repo (MemoryRepo default / SupabaseRepo(client, bucket=settings.raw_bucket)); surepass = LiveSurePassClient(token, live=settings.surepass_live, sink=repo, timeouts from config constants); rank_llm = make_rank_llm(settings, prompts) when OPENAI_API_KEY else raising stub (unreachable in degraded flow); checkpointer via graph.make_postgres_checkpointer(settings.database_url) when DSN set else None (C15 survives: manual pool, session pooler, dedicated schema, LANGGRAPH_STRICT_MSGPACK); compile quick+full+recompute graphs once; yield; pool.close().

ROUTES (declaration order matters: literal paths BEFORE '/{check_id}/...' so 'quick'/'full' never bind as check_id):
```python
checks = APIRouter(prefix="/checks", dependencies=[Depends(require_bearer)])  # HTTPBearer + hmac.compare_digest (C14)
@checks.post("/quick", response_model=LcView)
@checks.post("/full", response_model=LcView)
@checks.post("/{check_id}/recompute", response_model=LcView)
@checks.post("/{check_id}/attach", response_model=AttachResponse)
health = APIRouter()
@health.get("/health")   # EXACT {"ok": true, "surepass_live": <bool>}, unauthenticated, degraded-boot safe
```
_run_check(kind, req): check_id=uuid4; run_id=tracing.new_run_id(...); pan_hash=hash_pan(req.pan, key=settings.pii_hmac_key); mobile_hash=hash_mobile(normalized, key=...); seed full initial CheckState (request fields, consent model_dump, pan_last4/mobile_last4, errors=[], counters zeroed); cfg=tracing.invoke_config(kind, check_id, lc_id, pan_hash, run_id); final = await asyncio.wait_for(run_in_threadpool(graph.invoke, state, cfg), settings.graph_timeout_seconds); return LcView.model_validate(final["lc_view"]). Error boundary: catch Exception -> log check_id/pan_hash ONLY -> best-effort repo.record_check_error (terminal-guarded, C1) -> HTTP 200 build_error_lc_view.

_recompute(check_id, req): row = repo.get_check(check_id); None -> 404 {"detail":"check_not_found"}; row not recomputable (no stored facts: status in {refused} or no cache rows for its pan_hash) -> 409 {"detail":"not_recomputable"}; else invoke the recompute graph (derive->pincode_gate->range->code_confidence->rank->safety->redact->persist) with a NEW check_id, internal.recompute_of=<original>, declared figures from the body; zero SurePass calls asserted by construction (recompute graph has no pull nodes); returns new LcView.

_attach(check_id, req): row -> 404; row.kind != "quick" -> 409 {"detail":"attach_quick_only"}; row already attached to a different contact -> 409 {"detail":"already_attached"} (same contact -> idempotent 200); repo.attach_contact(check_id, contact_id) -> 200 AttachResponse(attached=True).

HTTP STATUS TABLE: 422 schema failures (handler strips FastAPI 'input'/'ctx' echo — PAN/name/mobile never reflected, v1.0 survives); 401 bad bearer (+WWW-Authenticate); 404/409 recompute+attach transport errors; EVERYTHING ELSE HTTP 200 LcView (refusals, pan_not_verified, pending_verification, service_failure are first-class persisted outcomes). consent.given false/absent -> 422 (nothing recorded).

### pyproject.toml (blocks owned by this section)
_pytest no-network config (binding: no filterwarnings line), dev deps with pyright replacing mypy, and the [tool.pyright] strict config that satisfies the 'pyright clean' acceptance criterion._

```toml
[dependency-groups]
dev = [
  "pytest>=9.1,<10",
  "pytest-socket>=0.8.1,<0.9",
  "pyright[nodejs]>=1.1.405,<1.2",   # pip wrapper; [nodejs] extra bundles node via nodejs-wheel-binaries -> hermetic CI, no npm/network at check time (verified: pypi 'pyright' current 1.1.409+, Apr 2026)
  "httpx>=0.27,<0.29",               # supabase window (audit pin survives)
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers --disable-socket --allow-unix-socket"
# BINDING: NO filterwarnings pytest_socket line — SocketBlockedError is not a Warning (verified; breaks the suite)
markers = ["eval: local golden-dataset eval runs"]

[tool.pyright]
include = ["eligibility_agent", "tests", "scripts"]
exclude = ["**/__pycache__", ".venv"]
typeCheckingMode = "strict"
pythonVersion = "3.12"
venvPath = "."
venv = ".venv"
reportMissingTypeStubs = true          # strict default kept; supabase boundary relaxed per-file, see below
executionEnvironments = [
  { root = "tests", reportPrivateUsage = false },   # tests may poke module privates for assertions
]
```
DELETE [tool.mypy] and [tool.pydantic-mypy] blocks entirely; mypy leaves the dev group. CI command: `uv run pyright` (config auto-discovered from pyproject).

TYPING PATTERNS THAT KEEP PYRIGHT STRICT CLEAN (replaces the v1.0 mypy pattern list):
1. Pydantic v2 is pyright-native (dataclass_transform; NO plugin exists or is needed — the mypy plugin line has no pyright equivalent). pydantic-settings alias kwargs in tests (Settings(_env_file=None, SUREPASS_LIVE="true")) type-check because BaseSettings.__init__ accepts **values: Any.
2. CheckState TypedDict total=False: read via state.get("key") -> T | None then explicit `if x is None: ...` narrowing; never bare subscript on optional keys, never cast on state reads.
3. DI seams stay typing.Protocol (Repo, SurePassClient, RankLLM); implementations conform structurally; __call__ signatures must match exactly including keyword-only markers; every def (tests included) fully annotated incl. -> None.
4. Supabase/storage3 ship no complete types: db/repo.py (the ONLY importer) opens with the file-level pragma `# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false` plus the rule that every value leaving that module is first validated into a frozen dataclass/pydantic model (Any never escapes).
5. The C7 typed closure factory for node binding is pyright-clean as designed (no functools.partial).
6. Ignore comments: prefer `# pyright: ignore[ruleName]` (rule-scoped); migrate/remove any mypy-specific `# type: ignore[...]`; reportUnnecessaryTypeIgnoreComment stays on via strict.
7. langgraph/langchain-core/langchain-openai/langsmith/fastapi/pydantic ship py.typed: no blanket relaxations for them.

### tests/conftest.py + tests/helpers.py
_Session hardening and shared fixtures, v1.0 assets reused by name (FrozenClock, pytest-socket harness, env scrub) with v1.1 renames (PII_HMAC_KEY, consent object, one LLM, new T0)._

conftest.py (v1.0 pattern survives): pytest_configure sets SERVICE_API_TOKEN="test-token", PII_HMAC_KEY="unit-test-hmac-key-0123456789abcdef", LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false"; pops SUREPASS_LIVE/SUREPASS_TOKEN/OPENAI_API_KEY/LANGSMITH_API_KEY/SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY/DATABASE_URL/RAW_BUCKET. autouse guard asserts SUREPASS_LIVE != "true". Primary no-network = pytest-socket addopts (binding flags). NEW canonical epoch: T0 = datetime(2026, 10, 10, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")). Fixtures: clock (FrozenClock(T0) — eligibility_agent/testing/clock.py REUSED UNCHANGED from the v1.0 design), repo (MemoryRepo(clock=clock.now)), seeded_repo (seed_default_lenders), fake_surepass (happy_quick_scenario), fake_rank (FakeRankLLM — single LLM now), graph_deps (GraphDeps(repo, surepass, rank_llm, prompts, now=clock.now)), quick_graph/full_graph/recompute_graph, app (create_app(Settings(_env_file=None, SERVICE_API_TOKEN=..., PII_HMAC_KEY=...), repo=..., surepass=..., rank_llm=..., checkpointer=None, now=clock.now)), client/anon_client (TestClient context manager, AUTH header pre-attached).

helpers.py: synthetic identities PAN_OK="ABCPE1234F", PAN_TWO="FGHPJ5678K", PAN_COMPANY="AAACA9999Z", PAN_NTC="KLMPN4321Q"; MOBILE_OK="9876543210"; NAME_OK="Ravi Testkumar"; LC_A/LC_B/LEAD/OPP.
```python
def consent_obj(**over: Any) -> dict[str, Any]:
    return {"given": True, "wording_version": "v2-quick", "call_ref": "sc_call_123", "ip": "10.0.0.1", **over}
def quick_payload(**over): # lc_id, lead_id, opportunity_id=None, name, mobile, pan, consent=consent_obj(), monthly_income=60000, existing_emi=0
def full_payload(**over):  # quick fields + opportunity_id=OPP, opportunity_stage="offer", monthly_income=60000, consent wording "v2-full"
def assert_no_leak(lc_view_json): # walks every string leaf; GLOBAL_LEAK_PATTERNS on all leaves; BUREAU_TERM_RE additionally on rationale/conditions/items_to_confirm/coapplicant_reason — IMPORTS both pattern tuples from eligibility_agent.output.lc_view (no local regex copies; C11 import rule survives)
def endpoints(fake) -> list[str]
def one_check_row(repo) -> CheckRow
def seed_counted_quick_checks(repo, *, lc_id, created_at, n)
```
SEED LENDERS v2 (admin-schema criteria; used by tests AND evals datasets AND scripts/seed_lenders.py so they never drift):
| id | name | floor (Experian) | ntc_accepted | foir_limit | min_income | ages (app/maturity) | rate | tenure | loan range | pincodes |
| L1 | Alpha Education Finance | 650 | false | 0.55 | 25000 | 21/58 | 11.0-14.0% | 12-48m | 50,000-15,00,000 | 560001,560002,110001 |
| L2 | Bharat Upskill Credit | 650 | true | 0.45 | 20000 | 21/60 | 14.0-18.0% | 12-36m | 25,000-5,00,000 | 560001,110001 |
| L3 | Crest Capital NBFC | 720 | false | 0.60 | 50000 | 23/55 | 10.5-12.5% | 24-60m | 1,00,000-25,00,000 | 400001 |
rate_band display strings "11 to 14%" / "14 to 18%" / "10.5 to 12.5%"; tenure_range "12 to 48 months" etc.

### eligibility_agent/testing/envelopes.py + scenarios.py (v1.1 deltas)
_Synthetic envelope builders: v1.0 dense-key builders REUSED (cais_account 23-key dense dict, '' sentinels, zero-padded Terms_Duration, validation-failure shape, probe-literal guard) with the NTC semantics flipped and new v1.1 scenario factories._

REUSED UNCHANGED: _ok_envelope / failure_envelope / validation_error_envelope; pan_comprehensive_envelope (DROP the gender kwarg from adapter use — v1.1: 'Experian takes no gender; do not derive or send one'; keep the key in raw data since real payloads carry it); cais_account dense builder (add kwargs portfolio_type already present; ensure Open_Date/Date_Reported/Terms_Duration drive the new branch math); experian_report_envelope; PROBE-LITERAL GUARD verbatim (forbidden literals '791111','201301','560071','16559','2434004','138696','109433' never in any builder output — BINDING).

NTC DELTA (v1.1 FIXED: NTC only on HTTP 200 + success:true + credit_score not a 3-digit string; everything else service failure):
```python
def experian_score_envelope(*, credit_score: str = "745", success: bool = True, ...)
# success=False now builds (200, {success: false, message_code: "no_record_found"}) which the client/adapters
# MUST classify as SUREPASS_FAILURE (service failure), never NTC. New kwarg examples for NTC variants:
#   credit_score=""      -> NTC (empty string)
#   credit_score="NH"    -> NTC (non-numeric)
#   omit_score=True      -> NTC (key absent)
#   credit_score="0745"  -> NTC (4 chars, not ^\d{3}$)
```
SCENARIOS (FakeSurePassClient factories; fake runs envelopes through the same interpret_response path — C5 survives):
- happy_quick_scenario(pan=PAN_OK, score="745", zip_code="560001")
- ntc_scenario(variant: Literal["empty","text","missing","four_digit"], pan=PAN_NTC)   # all four pin the ^\d{3}$ rule
- score_failure_scenario(variant: Literal["success_false","http_500","auth_403","timeout"])  # timeout variant raises SurePassTimeoutError from the fake
- pan_invalid_scenario(kind: Literal["status_invalid","company","failure_envelope"])
- full_happy_reported_emi_scenario(pan=PAN_OK)  # ONE open account Scheduled_Monthly_Payment_Amount "3500", Date_Reported 20260920, score "745", ReportDate 20261001, zips 560001 -> drives the HIGH-confidence full test
- full_emi_four_branch_scenario(pan=PAN_OK)     # six accounts, see test_emi_derivation numbers
- full_pincode_union_scenario(pan=PAN_OK)       # PAN zip 560002; holder zips 110001 (Date_Reported 20260901, open), '20' junk, 400001 on account Date_Reported 20250901 (>12m -> excluded; v1.1 window is 12 months)
- no_pincode_scenario(pan=PAN_OK)               # PAN address.zip "" -> NO_PINCODE_ON_PAN path
FrozenClock (testing/clock.py) reused verbatim; `now` bound method satisfies Clock = Callable[[], datetime].

### tests/ scenario suite — part 1: quota, caches, caps, locks (PRD 12 items 1-9, 18, 19, 21)
_End-to-end API tests via TestClient with fakes; exact billed/counted/call-sequence assertions and precomputed money constants._

tests/test_quick_happy_path.py (item 1):
GIVEN seeded world, happy_quick_scenario (score 745 -> band 720-780), payload income 60000 emi 0, T0.
THEN 200; endpoints == [PAN_ENDPOINT, SCORE_ENDPOINT] once each in order; score call body == {name: 'RAVI TESTKUMAR' (PAN full_name, NOT the LC-typed name), mobile, pan, consent: 'Y'}; body.status=='ok', code None, kind 'quick', from_cache False; headline 'possible' (quick is declared-figures => code confidence medium — see decisions); confidence 'medium'; label == LABEL_QUICK; top_lender == {id 'L1', name 'Alpha Education Finance', rate_band '11 to 14%', tenure_range '12 to 48 months', ceiling 1277000, range {lower 368000, upper 1277000}, rationale <scrub-clean str>, conditions [...]}  (raw bounds 367536.02162416704 / 1276816.8976926452, half-up 1000); alternates == [L2 view] with ceiling 500000 (clipped: raw upper 789990.4174501724 > 5,00,000 cap), range {lower 295000 (raw 294502.6405882871), upper 500000}; coapplicant_suggested False; pan_aadhaar_link_warning None; checked_on '2026-10-10'; refresh_available_on '2026-11-09'; quick_checks_left_today == 4; assert_no_leak(body). Persisted: one checks row billed True, counted_against_quota True, status 'ok'; score_cache row expires T0+30d; 2 surepass_calls rows; consent row with wording_version 'v2-quick', call_ref, ip, mobile_hash/pan_hash, covers='score'; rank LLM called once.
VARIANT quick-no-income: income/emi omitted -> ceiling None, range None on every lender view, label LABEL_QUICK, no FOIR anywhere in internal. VARIANT quick-income-no-emi: existing_emi omitted -> existing_emi 0 used, label == LABEL_QUICK_NO_EMI.

tests/test_cache_sharing.py (item 2): LC_A quick at T0 (billed+counted); clock.advance(3d); LC_B same PAN, income 90000: endpoints UNCHANGED (zero new calls); from_cache True; checked_on '2026-10-10' (original date); derivations re-ran on new income (L1 ceiling reflects avail 0.55*90000=49500, assert internal derived differs from first run); billed False counted False; LC_B quick_checks_left_today == 5; rank called again (cache hits re-run gate/rank/safety on current published criteria — PRD 5.4).

tests/test_quota_and_caps.py (items 3, 6, 7 + IST boundary):
(3) seed 5 counted checks for LC_A today; POST new PAN -> 200 refused QUOTA_EXHAUSTED, label == REFUSAL_MESSAGES['QUOTA_EXHAUSTED'], zero SurePass calls, quick_checks_left_today 0, refusal row billed False counted False.
(quota override) seed lc_quota_overrides[LC_A]=7 -> 6th check admitted, counted, left_today == 1.
(6) expired-score refresh: quick at T0; clock.advance(31d); same PAN again (no refresh flag exists on quick): endpoints grow by [PAN?]+[SCORE] (PAN reused if <180d: exactly [SCORE_ENDPOINT] added); new row billed True, counted_against_quota FALSE (PRD 5.6: re-pull on a PAN with an expired stored score does not count); from_cache False; checked_on == T0+31d date.
(7) hard cap: drive 3 billed bureau calls on one PAN in a day (e.g., quick new + full report + expired-score refresh via clock tricks, or seed surepass_calls rows); 4th attempt -> 200 refused PAN_DAILY_CAP; pan lock row for the IST day; review_queue row exists; repo.bureau_calls_today(pan_hash) == 3.
(IST boundary) 5 counted at 23:00 IST; 23:59 IST refused QUOTA_EXHAUSTED; clock.set_now(00:01 IST next day, same UTC date) -> admitted, left_today 4 (MemoryRepo counts on Asia/Kolkata date — mirror of SQL quick_checks_used_today).

tests/test_pan_not_verified.py (item 4): parametrize status_invalid / company / failure_envelope(=service failure! split below). status!=valid or category!=person: 200, status 'pan_not_verified', code None, label MSG_PAN_NOT_VERIFIED, endpoints == [PAN_ENDPOINT] only, row billed True counted False, NO pan_records cache write ('nothing marked checked'), quick_checks_left_today 5, rank never called, assert_no_leak. Then corrected-PAN retry: endpoints grow [PAN, SCORE]; second row billed+counted. failure_envelope (success:false) now asserts status 'service_failure' + SUREPASS_FAILURE instead (v1.1 flip).

tests/test_ntc.py (item 5): parametrize ntc variants ('', 'NH', missing key, '0745') with income 30000: endpoints [PAN, SCORE]; row billed True counted True (no-hit consumes quota); score_cache is_ntc True expires T0+30d; status 'ok'; headline 'unlikely'; coapplicant_suggested True with non-null scrubbed coapplicant_reason; top_lender L2 (only ntc_accepted) ceiling 395000 (raw 394995.2087250862) range {147000 (raw 147251.32029414354), 395000}; assert_no_leak. SECOND run same PAN within 30d: from_cache True, same headline, zero new calls. CONTRAST case: (200, success:false) envelope -> status 'service_failure', code SUREPASS_FAILURE, label MSG_BUREAU_SERVICE_FAILURE, counted False, NOTHING cached (regression pin for the v1.1 NTC rule).

tests/test_full_check.py (items 8, 9): (8) stages below offer (new/contacted/qualified/counselling) -> 200 refused STAGE_TOO_EARLY, zero calls; stages offer/registration/enrolled/won admitted; unknown stage string -> 422 (enum membership, C18). (9) full at offer, PAN cached 10d ago (<180d): endpoints == [REPORT_ENDPOINT] only (PAN reused); report row billed True, counted_against_quota False (full never touches quick quota); label == 'bureau report on 2026-10-01, income as declared' (ReportDate of the synthetic report); SCORE CACHE REFRESHED BY REPORT: immediately run a quick check on the same PAN -> zero SurePass calls, from_cache True (PRD 3: any Report pull refreshes the Score cache). Second opportunity, same PAN, 5 days later, different opportunity_id -> from_cache True, zero calls, billed False (one report per PAN per 30 days, shared across opportunities); derivations re-ran on the new opportunity's declared income. refresh=true with report 10d old -> 200 refused REPORT_WINDOW_OPEN with refresh_available_on == reportdate+30d; clock.advance(31d) + refresh=true -> [REPORT_ENDPOINT] again, billed True counted False, is_refresh True.

tests/test_timeout_and_locks.py (items 18, 19): (18) fake score raises SurePassTimeoutError -> 200, status 'pending_verification', code SUREPASS_FAILURE, label MSG_PENDING_VERIFICATION, counted False, pan_locks row with expires == now+10min, internal.spend_unknown True; immediate retry -> 200 refused PAN_LOCKED, zero calls; clock.advance(11min) -> lock expired, retry admitted. (19) IN_FLIGHT: acquire the in-flight lock for pan_hash (repo.acquire_pan_lock held), POST same PAN -> 200 refused IN_FLIGHT immediately (not queued), zero calls; release -> admitted. Non-timeout transport error (ConnectError) -> 'service_failure' (no lock) — pins the timeout/transport taxonomy split.

tests/test_live_kill_switch.py (item 21): v1.0 battery survives: real LiveSurePassClient(live=False) raises LiveCallsDisabled pre-I/O on all three endpoints (MockTransport fail-if-called); SUREPASS_LIVE parametrize {'false','TRUE','1','yes',''} -> all disabled; API degraded boot: create_app(Settings(_env_file=None)) NO fakes -> valid quick POST -> 200 refused LIVE_CALLS_DISABLED; /health == {'ok': True, 'surepass_live': False}; cache still serves with live off (pre-seed fresh score cache -> 200 ok from_cache True).

### tests/ scenario suite — part 2: money math, pincode, model, recompute/attach, correction, warning (PRD 12 items 10-17, 20, 22)
_EMI four-branch + stale/past-tenure, range clipping, NTC-warning handshake, PAN correction, no-pincode, model misbehavior, recompute/attach, adapters._

tests/test_emi_derivation.py (item 12) — full_emi_four_branch_scenario, ReportDate 20261001, income 80000, stage offer. Six synthetic accounts:
| acct | fields | branch | expected EMI |
| a1 | open, Scheduled '3500', Date_Reported 20260920 | 1 reported | 3500.0 |
| a2 | open, Scheduled '', bal '120000', Repayment_Tenure '0', Terms_Duration '048', Open_Date 20240115, Date_Reported 20260910 | 2 (months_since Jan24->Oct26 = 33; remaining 48-33=15 >= 1) | 120000/15 = 8000.0 |
| a3 | open, Scheduled '', bal '60000', Terms_Duration '024', Open_Date 20240115 | 2 past-tenure (remaining 24-33 = -9 < 1) | 60000/12 = 5000.0 + past-tenure flag |
| a4 | open, Scheduled '', bal '40000', Portfolio_Type 'R', no tenure | 3 revolving | 0.05*40000 = 2000.0 |
| a5 | open, Scheduled '', bal '90000', Portfolio_Type 'I', no tenure | 4 fallback | 90000/36 = 2500.0 |
| a6 | open, Scheduled '9999', Date_Reported 20260601 (122d before ReportDate > 90) | STALE: skipped entirely | 0 contribution + items_to_confirm 'An account last updated 2026-06-01' |
Asserts: internal existing_emi == pytest.approx(21000.0) (3500+8000+5000+2000+2500); emi_floor_flag True (estimates + flags present); current FOIR == approx(0.2625); declared=25000 -> used 25000 (declared higher wins); declared=2000 -> used 21000; L1 numbers for declared=None: available 0.55*80000-21000 = 23000 -> ceiling 890000 (raw 889902.6862706315), range {256000 (raw 256161.4696168437), 890000}; confidence 'medium' + 'Confirm existing EMIs'-style item present; body label 'bureau report on 2026-10-01, income as declared'; assert_no_leak (21000/8000 estimates never appear in prose). HIGH-CONFIDENCE CONTRAST (full_happy_reported_emi_scenario, income 60000, reported 3500 only, no stale, band 720-780 one bucket above floor 650): confidence 'high', headline 'fundable', L1 ceiling 1141000 (raw 1141396.9236949403), range {329000 (raw 328554.92842160386), 1141000}; L2 ceiling 500000 (clipped from 687584.25222515), lower 256000 (raw 256326.3723638795).

tests/test_range_math.py (item 13) — unit-level against derive.range (pins the formula loan_max = emi*((1+r)^n - 1)/(r(1+r)^n), r=annual/12/100): the exact float table above PLUS clip-up case L2 income 20000 existing 8000 -> available 1000: lower raw 10907.505206973596 clamped UP to 25000, upper raw 29258.90435000638 -> ceiling 29000 after rounding; displayed_foir cases: unclipped == foir_limit exactly (0.55/0.45 — reading pinned in decisions), B-clipped == approx(0.301269963) (EMI(500000,18%,36)=18076.197767958507 / 60000); available_emi 0 -> lender 'not fundable alone' marker, ceiling 0; round_to_nearest_1000 half-up: 731500->732000, 367536.02->368000 (rounding helper's single home is derive/range.py; output imports it).

tests/test_no_pincode.py (item 14): no_pincode_scenario (PAN zip invalid/""), quick: 200 status 'ok', code 'NO_PINCODE_ON_PAN', headline from band ('possible' for 745 at medium conf... pinned: band read present), top_lender None, alternates [], label MSG_NO_PINCODE, rank LLM NEVER called (skip rank and safety), billed True counted True (score was pulled), assert_no_leak.

tests/test_ntc_warning.py (item 11): quick NTC at T0 (cached); clock.advance(5d); full check WITHOUT acknowledge_ntc_warning -> 200 refused WARN_NTC_RECENT, label == 'No bureau file was found on 2026-10-10; a full report will most likely return nothing. Proceed?', zero report calls; WITH acknowledge_ntc_warning=True -> report pulled ([REPORT_ENDPOINT]); quick NTC 31+ days old -> no warning.

tests/test_pan_correction.py (item 10): full check opportunity O1 with PAN_A fails pan_not_verified; retry O1 with PAN_B (corrected, once, before any report) -> admitted, re-verified ([PAN,REPORT]), correction logged in internal; AFTER a report exists on O1, full check with PAN_C -> 200 refused PAN_CHANGE_NEEDS_MANAGER + review_queue row; and the 30-day window is per PAN hash: PAN_B's report does not block PAN_C once manager path cleared (unit-level guard assert).

tests/test_model_misbehavior.py (items 15, 16, 17):
(15) FakeRankLLM scripted ModelOutput(recommendation=Pick(lender_id='L3'...), alternates=[Pick('L_FAKE')], confidence='high') where L3 failed the gate and L_FAKE doesn't exist -> safety demotes both; none survive -> headline 'under_review', confidence 'low', top None, review_queue row; internal.safety lists demotions with reasons.
(16) confidence raise ignored: code_confidence computes 'medium' (declared figures); fake returns 'high' -> final min(medium, high) == 'medium'; fake returns 'low' -> final 'low' (may lower, never raise).
(17) fake raises/returns invalid twice -> exactly 2 rank calls (one retry, never a 3rd); 200 status 'service_failure', code 'OPENAI_FAILURE', label MSG_OPENAI_FAILURE, headline still band-derived (band read), confidence 'low', top None alternates [] (per decision; gated list asserted present in internal['gated_lenders']), billed True counted True (bureau spend happened), review_queue row. OpenAI 8s timeout path: same outcome via SurePassTimeout-analog on the LLM seam.
(model-number rule) ModelOutput schema has no numeric fields — pydantic extra='forbid' rejects a scripted dict with 'ceiling' injected (unit test on the model).

tests/test_recompute_attach.py (item 20 + attach): run a full check (T0, income 80000); POST /checks/{id}/recompute {monthly_income 50000, existing_emi 21000}: 200 NEW LcView; zero new SurePass calls (recompute graph has no pull nodes + fake log unchanged); numbers re-derived: L1 available 0.55*50000-21000 = 6500 (ceiling recomputed, assert differs); new checks row with internal.recompute_of == original id, billed False counted False; original row untouched. Unknown id -> 404 {'detail':'check_not_found'}; refused-status check -> 409 {'detail':'not_recomputable'}. ATTACH: quick check -> POST attach {contact_id 'c_1'} -> 200 {attached: true}; repeat same contact idempotent 200; different contact -> 409 'already_attached'; attach on a full check -> 409 'attach_quick_only'; unknown id 404.

tests/test_adapters.py (item 22, co-owned with the surepass/adapters section): v1.0 battery survives MINUS gender (adapter must NOT expose gender — assert Identity has no gender attribute; v1.1 'Experian takes no gender'), PLUS: parse_score NTC rule == ^\d{3}$ fullmatch on the raw string ('745'->745; ''/‘NH'/'0745'/missing -> NTC; success:false -> raises/service-failure, NOT NTC); band edges 649/650/719/720/779/780/900/300 half-open buckets [300,650)[650,720)[720,780)[780,900]; report pincode window now 12 MONTHS (Date_Reported 20250901 vs ReportDate 20261001 excluded; 20260901 included); holder DOB never read; masked_aadhaar stored-only (no Identity attribute).

tests/test_api_auth_and_validation.py: v1.0 cases survive with v1.1 deltas — consent matrix: consent absent -> 422; consent {'given': false,...} -> 422 (Literal[True]); consent {'given': true} missing wording_version -> 422; extra key inside consent -> 422; quick body with 'refresh' key -> 422 (extra=forbid; field removed); full missing opportunity_stage/monthly_income -> 422; 422 bodies carry loc/msg/type and NO 'input' echo (PAN never reflected); 401 matrix + WWW-Authenticate; /health exact shape unauthenticated; route-order pin: POST /checks/quick is NOT captured by /checks/{check_id}/recompute; LcView.model_validate round-trip on ok/refused/pan_not_verified/pending_verification/service_failure bodies; explicit nulls present in JSON.

tests/test_lc_view_redaction.py: full vector battery (v1.0 C11 rows + new ACCOUNT_ID_RE rows + two-tier label vectors); every fixed string (REFUSAL_MESSAGES values incl. formatted WARN_NTC_RECENT, MSG_*, LABEL_*, AADHAAR_WARNING) passes GLOBAL tier; compute_headline decision-table unit tests (all 6 rows); builder tests: low confidence nulls top/alternates; ceiling smuggle-kill 736.0 -> 1000; aadhaar truth table (False -> exact warning string; True/None -> None); leaky lender name 'Experian Finance Ltd' -> id displayed; leaky rate_band '736% special' -> ''; items_to_confirm scrub keeps 'An account last updated 2026-03-12'.

### eligibility_agent/evals/evaluators.py
_The seven PRD-12 evaluators in langsmith name-matched kwargs form ({'key','score','comment'} returns), importing the redaction patterns from output/lc_view (never copying)._

TARGET OUTPUTS CONTRACT (produced by evals/run.py run_one; synthetic-only data so internal facts are permissible in LangSmith):
```python
{'lc_view': dict, 'billed': bool, 'counted_against_quota': bool, 'from_cache': bool,
 'admission': str,                      # 'new' | 'cache_hit' | 'refresh' | 'refused' | ...
 'surepass_call_endpoints': list[str],
 'gated_lender_ids': list[str],         # pincode-gate survivors
 'safety_survivor_ids': list[str],      # gate survivors that passed the 9.3 safety check
 'band': str, 'code_confidence': str,
 'derived': dict[str, dict[str, int]]}  # lender_id -> {'ceiling': int, 'lower': int, 'upper': int} (post-rounding code values)
```
```python
from eligibility_agent.output.lc_view import GLOBAL_LEAK_PATTERNS, BUREAU_TERM_RE, line_leaks  # IMPORTED, no local regexes (C11 survives)
LLM_AUTHORED_PATHS = ('rationale', 'conditions', 'items_to_confirm', 'coapplicant_reason')

def no_leak(inputs, outputs, reference_outputs) -> dict:
    # every string leaf of lc_view vs GLOBAL_LEAK_PATTERNS (bureau NAMES + standalone 300-899 + account-id);
    # PLUS BUREAU_TERM_RE (generic bureau) on LLM-authored fields only (label is exempt from the generic term:
    # 'bureau report on <date>' is PRD-mandated copy). score = no hit anywhere.

def lenders_gated(inputs, outputs, reference_outputs) -> dict:
    # named = {top_lender.id} | {a.id for alternates}; score = named <= set(safety_survivor_ids) <= set(gated_lender_ids)

def numbers_from_code(inputs, outputs, reference_outputs) -> dict:
    # for every LenderView in [top]+alternates: view.ceiling == derived[id]['ceiling'] and
    # view.range in ({'lower': derived[id]['lower'], 'upper': derived[id]['upper']}, None-when-no-income);
    # plus: no digit sequence parseable as an amount appears in rationale/conditions that is absent from derived
    # (implemented as: NO digits >= 3 chars at all in LLM prose — the model never produces a number, PRD 9.2)

def confidence_not_raised(inputs, outputs, reference_outputs) -> dict:
    # ORDER = {'low':0,'medium':1,'high':2}; lc_view.confidence is None (refusals) -> pass;
    # else ORDER[lc_view.confidence] <= ORDER[outputs['code_confidence']]

def billing_consistency(inputs, outputs, reference_outputs) -> dict:
    # checks: billed == (len(call_endpoints) > 0); from_cache => no bureau endpoint in calls;
    # lc_view.status == 'refused' => calls == []; counted => billed and admission == 'new';
    # admission == 'refresh' => not counted; billed/counted/from_cache each match reference_outputs when present

def outcome_matches_band(inputs, outputs, reference_outputs) -> dict:
    # band in {'NTC', '<650'} => (top_lender is None) or coapplicant_suggested is True;
    # and headline != 'fundable' for those bands; headline == 'not_fundable_alone' => coapplicant_suggested True

DETERMINISTIC_EVALUATORS = [no_leak, lenders_gated, numbers_from_code, confidence_not_raised,
                            billing_consistency, outcome_matches_band]
```
LLM JUDGE: v1.0 rubric reused with field renames (judges `rationale` + `conditions` of top_lender/alternates): plain language, band-words not numbers, actionable, no approval promises, INR context; ChatOpenAI(model='gpt-4.1', temperature=0, timeout=8.0, max_retries=0).with_structured_output(JudgeVerdict, method='function_calling'); factory only constructed when OPENAI_API_KEY set (CI secrets gate).
tests/test_evaluators.py: each deterministic evaluator as a plain function over hand-built outputs dicts — leak via imported patterns (incl. an account-number plant and a label-with-'bureau' pass case); gated-subset chain; numbers_from_code catches a view ceiling 450000 vs derived 451000; confidence_not_raised catches high-over-medium; billing truth table incl. cache-hit=>zero-calls; outcome_matches_band NTC rows.

### eligibility_agent/evals/datasets.py + evals/run.py + .github/workflows/ci.yml
_Golden datasets (synthetic only), local/langsmith runner (fake on full set, real model <= 20), CI with pyright replacing mypy. v1.0 structure survives; example list updated for v1.1._

datasets.py: QUICK_DATASET='eligibility-quick-golden', FULL_DATASET='eligibility-full-golden'. GoldenExample TypedDict unchanged (example_id, kind, request, lenders, surepass, pre_state, expected). quick_examples() — 12: q01 happy_745 (medium/possible, counted); q02-q04 ntc_empty/ntc_text/ntc_missing (unlikely + coapplicant, counted); q05 score_success_false -> service_failure SUREPASS_FAILURE NOT counted (the v1.1 flip, load-bearing example); q06 pan_company pan_not_verified; q07 cache_hit (pre_state score cache -> from_cache, zero calls); q08 quota_exhausted (pre_state 5 counted); q09 no_pincode (NO_PINCODE_ON_PAN band read); q10 no_income (no ceiling/range anywhere); q11 band_610 (not fundable alone / coapplicant); q12 aadhaar_not_linked (exact warning string expected). full_examples() — 10: f01 happy_offer_reported_emi (high/fundable, counted False); f02 stage_too_early; f03 pincode_union ('12-month window' variant); f04 emi_four_branch (medium + items_to_confirm); f05 stale_account (item 'An account last updated 2026-06-01'); f06 second_opportunity_same_pan (from_cache); f07 report_refreshes_score (expected: later quick is cache_hit); f08 ntc_full (warn handshake acknowledged); f09 declared_emi_wins (50000 declared); f10 report_window_refresh (REPORT_WINDOW_OPEN). build_langsmith_datasets uses the MODERN form (audit fix survives): client.create_examples(dataset_id=..., examples=[{'inputs': {...}, 'outputs': ex['expected']} for ex in ...]).
run.py: run_one(example, llm) — FrozenClock(EVAL_T0=2026-10-10T10:00+05:30), MemoryRepo(clock), seed lenders, apply pre_state, FakeSurePassClient(**example['surepass']); llm 'fake' -> FakeRankLLM, 'real' -> make_rank_llm(settings with PROMPT_SOURCE=local); graph.invoke; return the outputs contract (read from persisted row + fake call log + internal). run_local: iterate, apply DETERMINISTIC_EVALUATORS, table, exit 1 on any falsy. run_langsmith: evaluate(target, data=..., evaluators=DETERMINISTIC_EVALUATORS + judge-if-key, max_concurrency=4); first --max-real (default 20) examples on the real model, remainder fake. CLI: python -m eligibility_agent.evals.run --mode {local,langsmith} --llm {fake,real} [--max-real 20] [--dataset quick|full|both] [--build-datasets] [--experiment-prefix s].
ci.yml (v1.0 skeleton survives, one swap):
```yaml
jobs:
  test:
    steps: [checkout, setup-uv py3.12, uv sync --frozen --all-groups,
            uv run pyright,            # <- replaces `uv run mypy`; pyright[nodejs] wheel = no npm/network
            uv run pytest -q]          # addopts enforce --disable-socket; SUREPASS_LIVE never set in CI
  evals-local:  {needs: test, run: uv run python -m eligibility_agent.evals.run --mode local --llm fake --dataset both}
  evals-langsmith: {needs: test, secrets-gated via env-indirection step (LANGSMITH_API_KEY + OPENAI_API_KEY),
                    run: ... --mode langsmith --llm real --max-real 20 --dataset both --build-datasets --experiment-prefix ci-${{ github.sha }}}
```
.env.example: regenerated from the config.py v2 table, everything commented out (degraded-boot demo), with PII_HMAC_KEY rotation warning (rotating invalidates all three caches + quota/lock accounting) and LANGGRAPH_STRICT_MSGPACK note (C15).

## Interfaces provided
- **name**: LcView v2 wire contract (hosted in graph/state.py, defined here verbatim) | **definition**: Status = ok|pan_not_verified|refused|pending_verification|service_failure; Code = the 13 literals; Headline = fundable|possible|unlikely|not_fundable_alone|under_review; LenderView{id,name,rate_band,tenure_range,ceiling:int|None,range:LoanRange|None,rationale:str,conditions:list[str]}; LoanRange{lower:int,upper:int}; LcView{check_id,kind,status,code|None,headline|None,label:str,confidence|None,top_lender|None,alternates(max 2),coapplicant_suggested:bool,coapplicant_reason|None,pan_aadhaar_link_warning|None,items_to_confirm:list[str],checked_on:str|None,from_cache:bool,refresh_available_on:str|None,quick_checks_left_today:int|None}; every model extra='forbid'. The graph/state owner must adopt this verbatim (field names are the frozen client contract).
- **name**: config.py v2 | **definition**: Settings (env table as specified; properties surepass_live/langsmith_tracing/auth_enabled; startup_errors() empty on empty env); hash_pii/hash_pan/hash_mobile (HMAC-SHA256, key=Settings.pii_hmac_key — THE single hashers, everyone imports); last4; Final timeout constants TIMEOUT_PAN_S=8.0/TIMEOUT_SCORE_S=12.0/TIMEOUT_REPORT_S=20.0/TIMEOUT_OPENAI_S=8.0/CONNECT_TIMEOUT_S=5.0; ConfigError; utcnow; get_settings.
- **name**: main.create_app + HTTP contract | **definition**: create_app(settings=None, *, repo=None, surepass=None, rank_llm=None, checkpointer=None, prompts=None, now=None) -> FastAPI; app_factory() for uvicorn --factory. Routes: GET /health (exact {ok, surepass_live}, no auth); POST /checks/quick, /checks/full -> 200 LcView always for learner outcomes; POST /checks/{check_id}/recompute -> 200 new LcView | 404 check_not_found | 409 not_recomputable; POST /checks/{check_id}/attach -> 200 {check_id, contact_id, attached} | 404 | 409 attach_quick_only/already_attached; 422 {detail:[{loc,msg,type}]} with no input echo; 401 + WWW-Authenticate on /checks/*.
- **name**: output/lc_view.py redaction API | **definition**: BUREAU_TERM_RE, BUREAU_NAME_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE; LLM_LEAK_PATTERNS / GLOBAL_LEAK_PATTERNS tuples; line_leaks(line, patterns=LLM_LEAK_PATTERNS) -> bool; scrub_lines (whole-line drop); compute_headline(...) decision table; RedactionContext(quick_checks_left_today, checked_on, from_cache, refresh_available_on); build_lc_view / build_refusal_lc_view / build_failure_lc_view / build_error_lc_view; ALL fixed strings (LABEL_QUICK, LABEL_QUICK_NO_EMI, LABEL_FULL_FMT, AADHAAR_WARNING, MSG_* section-10 strings, REFUSAL_MESSAGES for the 10 refusal codes). Tests and evaluators import these; no copies anywhere (C11 rule).
- **name**: eligibility_agent/testing assets (v1.1 revision) | **definition**: FrozenClock (unchanged from v1.0 dossier); envelopes.py dense builders with NTC-variant kwargs and the probe-literal guard; scenarios.py factories (happy_quick, ntc_scenario[4 variants], score_failure_scenario[success_false/http_500/auth_403/timeout], pan_invalid_scenario, full_happy_reported_emi, full_emi_four_branch, full_pincode_union, no_pincode); FakeRankLLM (single LLM: Callable[[RankContext], ModelOutput] with .calls, script=/respond= hooks, invalid-output helpers).
- **name**: evals contract + CI commands | **definition**: run_one outputs: {lc_view, billed, counted_against_quota, from_cache, admission, surepass_call_endpoints, gated_lender_ids, safety_survivor_ids, band, code_confidence, derived: {lender_id: {ceiling, lower, upper}}}. DETERMINISTIC_EVALUATORS = [no_leak, lenders_gated, numbers_from_code, confidence_not_raised, billing_consistency, outcome_matches_band]; judge factory gated on OPENAI_API_KEY. Datasets eligibility-quick-golden (12) / eligibility-full-golden (10). Commands other sections must keep green: `uv run pyright`, `uv run pytest -q` (no sockets, no secrets), `uv run python -m eligibility_agent.evals.run --mode local --llm fake --dataset both` exit 0.
- **name**: pyproject blocks | **definition**: [tool.pyright] strict config (include eligibility_agent/tests/scripts, pythonVersion 3.12, venv .venv, executionEnvironments tests reportPrivateUsage=false); [tool.pytest.ini_options] addopts '-q --strict-markers --disable-socket --allow-unix-socket' with NO filterwarnings line; dev group pytest>=9.1,<10, pytest-socket>=0.8.1,<0.9, pyright[nodejs]>=1.1.405,<1.2, httpx>=0.27,<0.29. mypy blocks deleted.

## Interfaces consumed
- **name**: graph/build.py + graph/state.py (graph section) | **expectation**: build_graph('quick'|'full', deps, *, checkpointer=None) with GraphDeps(repo, surepass, rank_llm, prompts, now) frozen (C7 typed-closure bind survives; finalize_llm GONE — one LLM node); a recompute entry point compiling the sub-graph derive->pincode_gate->range->code_confidence->rank->safety->redact->persist that loads stored raw via repo and performs ZERO SurePass calls; state['lc_view'] set on EVERY terminal path; CheckState carries: request fields incl. consent dict + acknowledge_ntc_warning + refresh, pan_hash/mobile_hash/pan_last4/mobile_last4, admission facts, identity/score/report facts (never raw payloads), candidate_pincodes, gated_lenders, derived per-lender {ceiling: float, lower: float, upper: float, displayed_foir: float, available_emi: float} + band + existing EMI split/flags + ages, code_confidence, model_output (ModelOutput), safety survivors + demotions, errors, langsmith_run_id; ModelOutput/Pick exactly per PRD 9.2 (no numeric fields; extra='forbid'); rank node: temperature 0, ONE retry on invalid output, ChatOpenAI(timeout=TIMEOUT_OPENAI_S=8.0, max_retries=0) (network retries forbidden in v1.1); redact node computes RedactionContext and calls build_lc_view; graph/tracing.py new_run_id/invoke_config/finish_trace verbatim (C16).
- **name**: policy/guards.py (policy section) | **expectation**: Refusal/odd codes exactly the 13 literals; AdmissionDecision exposing admission mode, code, billed_expected, counts_against_quota (True only for a billed Score pull on a PAN with NO stored score row — expired row => not counted), quick_checks_used_today + lc quota override (default 5, lc_quota_overrides), cache dates for RedactionContext; order: consent -> in-flight/timeout locks (IN_FLIGHT/PAN_LOCKED) -> stage (full) -> PAN-correction rule (PAN_CHANGE_NEEDS_MANAGER) -> NTC warning (WARN_NTC_RECENT unless acknowledge_ntc_warning) -> cache windows (PAN 180d / score 30d / report 30d; REPORT_WINDOW_OPEN on refresh-inside-window) -> quota (QUOTA_EXHAUSTED) -> per-PAN hard cap 3/day (PAN_DAILY_CAP + day lock + review row) -> kill switch (LIVE_CALLS_DISABLED last); OpportunityStage StrEnum stays the single swap point (C18); timeout handling sets a 10-minute pan_locks row; constants QUICK_DAILY_QUOTA_DEFAULT=5, PAN_DAILY_BUREAU_CAP=3, TTLs 180/30/30, TIMEOUT_LOCK_MINUTES=10 as Final code constants.
- **name**: db/repo.py (persistence section) | **expectation**: Repo protocol with (at least): insert_consent (one row PER PULL: wording_version, ts, lc_id, opportunity_id, contact_id, mobile_hash, pan_hash, call_ref, ip, covers: 'score'|'report'); insert_check/update_check with the C1 terminal guard (update applies only when status=='running'); get_check(check_id); attach_contact(check_id, contact_id) raising/signalling already-attached; quick_checks_used_today(lc_id) on the IST date (+ lc_quota_overrides lookup); bureau_calls_today(pan_hash); acquire_pan_lock/release_pan_lock(pan_hash, kind: 'in_flight'|'timeout', expires_at); three cache tables (pan_records 180d, score_cache 30d, report_cache 30d) with Report-pull-refreshes-score write; insert_surepass_call sink (C5 client-owned logging survives); raw-response put to the private bucket keyed pan_hash+opportunity (SupabaseRepo(client, bucket=...)); insert_review_queue(check_id, reason); MemoryRepo(clock) mirrors all of it including IST day math and the terminal guard.
- **name**: surepass/client.py (surepass section) | **expectation**: C5 contract survives (SurePassClient protocol with per-call check_id, SurePassResult, envelope parsing, sink logging, FakeSurePassClient merged constructor, endpoint constants, kill switch raising LiveCallsDisabled pre-I/O) with two v1.1 amendments: (1) per-endpoint read timeouts from config constants (PAN 8s / Score 12s / Report 20s; connect 5s) — constructor takes them or imports the constants; (2) NEW exception SurePassTimeoutError(SurePassTransportError) distinguishing read-timeout (-> pending_verification + 10-min lock) from other transport failures (-> service_failure, no lock); FakeSurePassClient must support scripting a raised SurePassTimeoutError per endpoint.
- **name**: surepass/adapters.py + derive/* + lenders/* (derivation sections) | **expectation**: adapters: NTC iff HTTP 200 + success:true + not fullmatch(r'\d{3}') on str(credit_score) (missing key included); success:false => service failure; NO gender anywhere; holder DOB never used. derive/emi.py: the four branches + stale (>90d before ReportDate, excluded + item string 'An account last updated <YYYY-MM-DD>') + past-tenure (balance/12 + flag) with months_since = (ry-oy)*12 + (rm-om) anchored on ReportDate — my tests pin 3500+8000+5000+2000+2500=21000 and declared-max rule. derive/range.py: loan_max = available_emi*((1+r)^n-1)/(r(1+r)^n) with r=annual_pct/1200; upper at (min_rate, max_tenure), lower at (max_rate, min_tenure), both clamped to the lender loan range; ceiling=upper; displayed_foir per the decision reading; available_emi=max(0, foir_limit*income - existing_emi); round_to_nearest_1000 half-up lives HERE (single home, output imports); my pinned floats: L1@33000 -> 1276816.8976926452/367536.02162416704; L2@27000 -> clip 500000/294502.6405882871; L2@13500 -> 394995.2087250862/147251.32029414354; L1@23000 -> 889902.6862706315/256161.4696168437; L1@29500 -> 1141396.9236949403/328554.92842160386; L2@23500 -> clip 500000/256326.3723638795; clip-up L2@1000 -> lower clamped 25000, upper 29258.90435000638. candidate pincodes: PAN zip + open-account ZIPs with Date_Reported in last 12 MONTHS, ^[1-9][0-9]{5}$. lenders/gate.py + safety.py: gate survivors list and safety demotion records exposed in state for internal + evaluators; a lender missing any required admin field is excluded before gating.
- **name**: graph/prompts.py (prompt owner) | **expectation**: Single prompt 'hero/eligibility-rank' pinned by PROMPT_TAG; PROMPT_SOURCE=local default so tests/CI never pull the Hub; renderer serializes band bucket (words), FOIR + floor flag, EMI split reported/estimated + flags, declared income, per-lender ceiling/range, ages, aadhaar link, enquiry count, past-due/stale flags, normalized criteria + policy notes — NEVER PAN/name/mobile/aadhaar/account numbers/raw report; rationale/conditions instructed numeral-free (keeps no_leak green by construction).

## Decisions
- **topic**: consent surface (v1.1 object form) | **decision**: consent is a nested object with given: Literal[True]; false, absent, or malformed -> 422 with nothing recorded; wording_version required, call_ref/ip optional. The admission node writes one consents row per pull (score or report) from these fields plus mobile_hash/pan_hash/lc_id/opportunity_id; CONSENT_MISSING remains a defensive guard-level code reachable only in unit tests. | **rationale**: C21 is a BINDING owner ruling ('always consider consent = true') that survives v1.1; v1.1 section 5.1 requires a consent record per pull and sending consent:'Y' only when a record exists — a request that cannot produce a genuine record must never become a check. Fabricating a consent row for given:false would be worse than refusing.
- **topic**: where the section-10 'one screen message' lives | **decision**: LcView.label carries the single human-readable line: provenance label on ok views ('Indicative, ...' / 'bureau report on <date>, income as declared' / 'assuming no other EMIs' variant), the exact section-10 failure string on pan_not_verified/pending_verification/service_failure, and the refusal message on refused views. No extra field is added to the section-6 schema. | **rationale**: Section 10 mandates exact LC-facing strings but section 6's LcView has no message field; label is the only free-string display field and its documented examples are exactly this kind of one-line annotation. Keeps the wire schema EXACTLY per section 6. Flagged as ASSUMPTION.
- **topic**: OPENAI_FAILURE view shape | **decision**: status='service_failure', code='OPENAI_FAILURE', headline kept (band-derived), confidence='low', top_lender=None, alternates=[]; the gated lender list is persisted in checks.internal['gated_lenders'] and the run goes to the review queue. Billed/counted stand (spend happened). | **rationale**: Section 6 FIXED rule 'top_lender and alternates are null/empty when confidence is low' collides with section 7/12 'band read + gated list still shown/returned' (9.4 forces low on double model failure). Chose the privacy-conservative section-6 reading; the contradiction is flagged for owner adjudication with the alternative (populate alternates with <=2 gate survivors carrying code-derived numbers and empty rationale) documented.
- **topic**: timeout vs other failures | **decision**: Only a read-timeout on Score/Report produces status='pending_verification' + 10-minute PAN lock (code SUREPASS_FAILURE, exact MSG_PENDING_VERIFICATION); any other service failure (5xx, success:false, auth, non-timeout transport) is status='service_failure' with no lock. A request arriving during the lock is refused PAN_LOCKED; during an in-flight run, IN_FLIGHT. The SurePass client must therefore expose SurePassTimeoutError distinct from SurePassTransportError (new v1.1 requirement on the C5 taxonomy). | **rationale**: PRD 5.12 locks the PAN only when 'spend unknown' (timeout); a 5xx has known-failed spend. Conflating them would lock PANs needlessly and break the 'quota not used, retry now' promise of section 10.
- **topic**: per-call timeouts are constants, not env | **decision**: TIMEOUT_PAN_S=8.0, TIMEOUT_SCORE_S=12.0, TIMEOUT_REPORT_S=20.0, TIMEOUT_OPENAI_S=8.0 as Final constants in config.py; not overridable by env. Route budget GRAPH_TIMEOUT_SECONDS stays env (default 60.0). | **rationale**: Section 5.12 is a FIXED policy rule; an env override would let a deploy weaken it silently. The route budget is operational, not policy.
- **topic**: two-tier leak scanning | **decision**: Binding BUREAU_TERM_RE (incl. bureaus?) + STANDALONE_SCORE_RE survive verbatim and apply with the new ACCOUNT_ID_RE to LLM/derived prose (rationale, conditions, items_to_confirm, coapplicant_reason) with whole-line drop; a names-only BUREAU_NAME_RE + STANDALONE_SCORE_RE + ACCOUNT_ID_RE tier applies to every string leaf including fixed labels. output/lc_view.py remains the single home; tests and evals import. | **rationale**: v1.1 MANDATES the literal strings 'bureau report on <date>, income as declared' and 'No bureau file was found on <date>...' — the generic-bureau strengthening cannot apply globally without destroying PRD-required copy. The FIXED leak rule is 'never show the bureau NAME'; the generic-term ban stays as a strict superset exactly where the LLM can write.
- **topic**: no-account-list rule as a regex | **decision**: ACCOUNT_ID_RE = (?i)(a/c|acct|account)\s*(no|number|#)... | masked x{2,}\d{3,} | bare \d{9,18}. The bare word 'account' is NOT banned ('An account last updated <date>' is mandated items_to_confirm copy); 9-18 digit runs are banned (amounts are <= 1e8 and comma-grouped). | **rationale**: The evaluator spec says 'no account list in any LC-facing string'; account identifiers and masked numbers are the leak vector, while the word appears in required copy. The digit-run bound is chosen so no legitimate INR amount (<= 100,000,000, i.e. max 9 digits but always grouped in prose) is caught.
- **topic**: headline computation | **decision**: compute_headline is a pure code function in output/lc_view.py with the 6-row decision table: none-survived-gate-or-band -> not_fundable_alone (coapplicant forced true); demoted-all/model-failed -> under_review; pick + NTC-or-below-650 band -> unlikely; pick + high -> fundable; pick + medium -> possible; defensive -> under_review. headline is None on refusals and SUREPASS failures (no band known). | **rationale**: v1.1 names the five headlines but gives no mapping; the table is assembled from sections 3, 9.3, 9.4 and 10 (NTC row: 'Unlikely alone, co-applicant suggested'). The model never chooses the headline (it never raises confidence and produces no numbers).
- **topic**: quick checks are capped at medium confidence | **decision**: code_confidence 'high' requires EMI fully bureau-reported and no declared-only derivation input, which only a full check can satisfy; every quick check (income/EMI declared) computes at most 'medium'. Tests pin quick-happy = medium/'possible' and full-with-reported-EMI = high/'fundable'. | **rationale**: Section 9.4: high requires 'EMI fully reported' and 'no declared-only figure'. Interpreted 'declared-only figure' as a derivation input lacking bureau corroboration where corroboration is possible (EMI on quick); reading it as 'any declared figure' would make high unreachable even on full checks (income is always declared), which contradicts the 9.4 table showing 'Recommended' views. Interpretation flagged.
- **topic**: displayed-FOIR formula reading | **decision**: displayed_foir = (existing_emi + emi(P*, max_rate, max_tenure)) / income where P* = clamp(pv(available_emi, max_rate, max_tenure), loan_min, loan_max). Unclipped this equals the lender's foir_limit exactly (0.55/0.45 in the worked cases); clipped-down cases come in under (0.3013 for L2@60k). | **rationale**: The literal alternative — EMI of the min-rate ceiling re-priced at max rate — systematically exceeds the FOIR limit (0.5815 > 0.55 in the happy case) and would make safety rule 9.3 demote every unclipped pick, an absurd outcome. The adopted reading is the only self-consistent one; ambiguity flagged to the derive-section owner and the PM.
- **topic**: quick request has no refresh field; expired-score re-pull billed-not-counted | **decision**: QuickCheckRequest omits refresh entirely (extra=forbid rejects it); when the stored score is expired a quick check automatically re-pulls — billed=True, counted_against_quota=False. refresh exists only on FullCheckRequest and inside the window refuses REPORT_WINDOW_OPEN. | **rationale**: v1.1 section 6 quick body has no refresh key and rule 5.6 explicitly excludes 'a refresh of an expired stored score' from the quota while rule 5.10 forbids forced refresh inside a window — so the flag would have no legal use on quick.
- **topic**: recompute/attach transport semantics | **decision**: recompute: 404 check_not_found; 409 not_recomputable (refused checks / no stored raw); success creates a NEW checks row (billed=False, counted=False, internal.recompute_of=<original>) and returns its LcView. attach: quick-only (409 attach_quick_only), idempotent for the same contact, 409 already_attached for a different one, 404 unknown; response {check_id, contact_id, attached: true}. Both behind bearer auth. Literal routes declared before parametrized ones. | **rationale**: v1.1 specifies the endpoints but not their failure shapes or persistence; 4xx fits because these are caller errors, not learner outcomes (no LcView exists to carry a refusal). A new checks row preserves the audit trail of what the LC was shown at each point in time.
- **topic**: pyright swap | **decision**: pyright strict via [tool.pyright] in pyproject (typeCheckingMode=strict, pythonVersion 3.12, include eligibility_agent/tests/scripts, venv .venv, executionEnvironments relaxing reportPrivateUsage for tests); dev pin pyright[nodejs]>=1.1.405,<1.2 (bundled node wheel = no npm download in CI); CI step `uv run pyright` replaces `uv run mypy`; [tool.mypy]/[tool.pydantic-mypy] blocks deleted; supabase boundary handled by a file-level pragma in db/repo.py only; no pydantic plugin needed (pydantic v2 is pyright-native via dataclass_transform). | **rationale**: v1.1 acceptance says 'pyright clean'. Verified against current docs/PyPI (2026): pip 'pyright' at 1.1.409+, [tool.pyright] pyproject support, strict-mode rule set, per-file '# pyright:' pragmas and executionEnvironment diagnostic overrides all current. The [nodejs] extra keeps the no-network CI invariant.
- **topic**: NTC fixture semantics flip | **decision**: Synthetic scenarios define NTC exclusively as HTTP 200 + success:true + credit_score failing fullmatch(^\d{3}$) (variants: '', non-numeric, missing key, 4-digit). The (200, success:false, no_record_found) envelope — v1.0's NTC ASSUMPTION — becomes a SUREPASS_FAILURE service-failure fixture with a dedicated regression test asserting quota untouched and nothing cached. | **rationale**: v1.1 section 8.1 is explicit and FIXED: 'Anything else is a service failure, not NTC'. This reverses the v1.0 assumption and must be pinned in fixtures, tests AND a golden eval example (q05) or the old behavior silently returns.
- **topic**: HTTP status philosophy unchanged | **decision**: All learner outcomes (ok, pan_not_verified, refused, pending_verification, service_failure) are HTTP 200 LcViews persisted as checks rows; 422 schema / 401 auth / 404+409 recompute-attach are the only non-200s. The crash boundary returns 200 service_failure code=null with terminal-guarded error row (C1 survives). | **rationale**: v1.0 decision survives: clients render one object; 5xx would invite blind retries that can double-bill SurePass before cache rows exist, violating the zero-double-billing goal.
- **topic**: checked_on / refresh_available_on dates | **decision**: IST calendar dates (YYYY-MM-DD) derived from the governing raw pull's fetched_at: checked_on = pull date (original date on cache hits, satisfying 'checked on <date>'), refresh_available_on = pull date + 30 days; None on refusals with no cache context. quick_checks_left_today = max(0, (override or 5) - used - (1 if this run counts else 0)), None for full (C12 precedent). | **rationale**: The PRD displays dates, not timestamps, to LCs in IST; the +30d mirrors the score/report windows of rule 5.3. Computing left-today in redact (pre-persist) keeps the v1.0 arithmetic that made the counter exact.
- **topic**: engineering-alert path for auth failures | **decision**: ip_not_whitelisted / invalid_token (SurePassAuthError) => logger.critical with event='surepass_auth_failure' + a review_queue row with reason='surepass_auth'; LC sees generic service_failure. No in-process pager integration. | **rationale**: Section 10 requires 'alert engineering' without naming a channel; a CRITICAL structured log plus a queryable review row is the minimal alertable surface in this repo (README ops note tells ops to alert on both), and the LC-facing contract stays one screen message.

## Risks
- Two genuine v1.1 self-contradictions are decided-and-flagged, not resolved: (a) section 6 'alternates null/empty at low confidence' vs sections 7/12 'OPENAI_FAILURE returns band read + gated list' — if the owner wants the gated list on the wire, LcView needs either populated alternates at low confidence or a new field, and tests q/17 must be re-pinned; (b) section 8.4's displayed-FOIR formula read literally breaches section 9.3's own safety rule on every unclipped pick — the adopted reading must be ratified by the derive-section owner or safety outcomes change wholesale.
- The label-as-message decision makes `label` polymorphic (provenance vs failure text); if SuperConnect renders label verbatim under a recommendation header, failure strings may look odd — needs a one-line integration note to the SuperConnect team before UAT.
- ACCOUNT_ID_RE's bare 9-18-digit-run ban could drop a legitimate LLM line quoting a very large ungrouped number; mitigated by the prompt's no-numerals instruction and whole-line drop semantics, but the judge evaluator should watch for over-scrubbing (SCRUB-to-empty rationale becomes '').
- pyright strict over supabase-py relies on a single-file pragma in db/repo.py; if any other module imports supabase (e.g., a storage helper for the raw bucket), it will fail CI — the bucket writer must live inside db/repo.py or share the pragma discipline.
- quick-capped-at-medium (decisions) changes the v1.0 golden expectations (quick happy was high); all three sections' fixtures must adopt it together or the confidence_not_raised / outcome evaluators will disagree with tests.
- The 10-minute timeout lock plus IN_FLIGHT lock are repo-level state; MemoryRepo and SupabaseRepo must implement identical expiry semantics against the injected clock or tests 18/19 will pass in memory and fail in production.
- GRAPH_TIMEOUT_SECONDS (60s) bounds the HTTP response, not the SurePass spend — the v1.0 zombie-thread risk survives and is mitigated only by the C1 terminal guard; ops must alert on checks.status='service_failure' rate and reconcile surepass_calls.

## PRD gaps
- Section 6 LcView has no field for the section-10 'one screen message'; resolved by carrying it in `label` (ASSUMPTION, needs owner nod).
- Contradiction: section 6 'top_lender and alternates are null/empty when confidence is low' vs sections 7/12 'OPENAI_FAILURE ... band read + gated list still shown/returned' (9.4 forces low on model failure). Chose section 6; gated list persisted in internal only. Owner must adjudicate.
- Section 8.4 'displayed FOIR = (existing_emi + emi(loan_max at max rate, max tenure)) / income' is ambiguous; the literal min-rate-ceiling reading systematically violates section 9.3's FOIR safety rule (0.5815 > 0.55 in the happy case). Adopted the self-consistent reading (P* recomputed at max rate/max tenure, clamped).
- No headline<-facts mapping is given for the five headlines; a six-row code decision table was defined (section 3/9.3/9.4/10 synthesis).
- Section 9.4 'no declared-only figure' would make 'high' unreachable even on full checks if read as 'any declared figure' (income is always declared); interpreted as 'no derivation input lacking available bureau corroboration', capping quick checks at medium.
- recompute/attach response shapes, failure codes, persistence (new row vs mutate) and auth are unspecified; designed 404/409 + new-row semantics.
- Status/code for the timeout view is unspecified (PAN_LOCKED vs SUREPASS_FAILURE): chose pending_verification + SUREPASS_FAILURE for the timed-out run itself, PAN_LOCKED only for requests arriving during the lock.
- The mandated strings 'bureau report on <date>' (label) and 'No bureau file was found on <date>' (NTC warning) conflict with the binding generic-bureau scrub strengthening; resolved with the two-tier scan (names-only globally, full term on LLM prose).
- Section 5.8 says the service 'trusts and logs' the caller-passed stage while section 6 lists STAGE_TOO_EARLY; kept the C18 enum-membership 422 + ordering refusal. If FulCRM stage strings differ from the ASSUMPTION enum, only policy/guards.py changes.
- Whether `billed` on a timed-out check is true/false/unknown pending reconciliation is unspecified; expectation recorded: counted=False always, internal.spend_unknown=true, billed resolved from surepass_calls at reconciliation (C12 call-log ground truth survives).
- QUOTA_EXHAUSTED message surfacing 'N of 5 checks left today' is the caller's job per section 5.5; the API only ships quick_checks_left_today.
