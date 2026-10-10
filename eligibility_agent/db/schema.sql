-- eligibility_agent/db/schema.sql (idempotent; psql "$SUPABASE_DB_URL" -f ... or Supabase SQL editor)
-- PRD v1.1 section 11. All ids are app-generated uuid4 strings. All timestamps are timestamptz (aware UTC).
-- Mirrors eligibility_agent/db/repo.py: keep the row dataclasses, the literal registry in graph/state.py and
-- this file in step (tests/test_schema_sql.py pins the CHECK lists against the registry).

create table if not exists lenders (
  id           text not null,                 -- stable slug, e.g. 'L1'
  version      integer not null,
  name         text not null,
  published    boolean not null default false,
  criteria     jsonb not null default '{}'::jsonb,
  -- criteria contract: every field is {"value": <typed>, "status": "known"|"missing"|"uncertain"}.
  -- CANONICAL key spellings (lenders/gate.py reads these first; the LenderCriteria names score_floor, min_age_years,
  -- max_age_at_maturity_years are accepted as a fallback spelling only, and a row carrying both with differing
  -- content is excluded as invalid):
  --   min_score_floor (int, entered as an Experian score; open item), ntc_accepted (bool),
  --   foir_limit (number 0..1), min_income (int), min_age (int), max_age_at_maturity (int),
  --   rate_min_pct (number), rate_max_pct (number), tenure_min_months (int), tenure_max_months (int),
  --   loan_min (int), loan_max (int).
  -- Matching REQUIRES (status != 'missing' and a value): foir_limit, rate_min_pct, rate_max_pct, tenure_min_months,
  --   tenure_max_months, loan_min, loan_max (the 7 REQUIRED_FOR_MATCHING fields), PLUS the floor pair: min_score_floor
  --   known OR ntc_accepted known. A lender missing any of these is excluded from matching until fixed (PRD 11).
  -- Every other field (min_income, min_age, max_age_at_maturity) may be absent: the checks that use it are skipped.
  --   An 'uncertain' status on a required field keeps the lender in (value usable) but caps confidence at medium.
  -- OPTIONAL: min_income_by_employment (object), state_list (text[]).
  policy_notes text not null default '',
  created_by   text,
  created_at   timestamptz not null default now(),
  primary key (id, version)
);
create unique index if not exists idx_lenders_one_published on lenders (id) where published;  -- at most ONE published version per lender

create table if not exists lender_pincodes (
  lender_id text not null,
  version   integer not null,
  pincode   text not null check (pincode ~ '^[1-9][0-9]{5}$'),
  primary key (lender_id, version, pincode),
  foreign key (lender_id, version) references lenders(id, version) on delete cascade
);
create index if not exists idx_lender_pincodes_pincode on lender_pincodes (pincode);

create table if not exists consents (                     -- one row PER PULL (PRD 5.1 / 11); 5-year retention
  id              uuid primary key,
  lc_id           text not null,
  lead_id         text,
  opportunity_id  text,
  contact_id      text,                                   -- set by /attach
  pan_hash        text not null,
  mobile_hash     text not null,
  lc_typed_name   text not null,                          -- the LC-typed name lives ONLY here (PRD 6)
  wording_version text not null,                          -- 'v2-quick' | 'v2-full' ...
  call_ref        text,                                   -- SuperConnect call reference when applicable
  ip              text,
  covers          text not null check (covers in ('score','report')),
  given_at        timestamptz not null,
  purge_after     timestamptz not null                    -- given_at + interval '5 years'
);
create index if not exists idx_consents_pan on consents (pan_hash, given_at desc);

create table if not exists pan_records (                  -- PAN Comprehensive cache, 180 d; valid+person only
  pan_hash    text primary key,
  pan_last4   text not null,
  storage_key text not null,                              -- raw envelope object in the 'raw-responses' bucket
  client_id   text,
  fetched_at  timestamptz not null,
  expires_at  timestamptz not null,                       -- fetched_at + interval '180 days'
  purge_after timestamptz not null                        -- fetched_at + interval '180 days'
);

create table if not exists score_cache (                  -- Experian Score cache, 30 d; refreshed by Report pulls too
  pan_hash    text primary key,
  storage_key text not null,                              -- score envelope OR report envelope (source='report')
  source      text not null check (source in ('score','report')),
  is_ntc      boolean not null default false,             -- 200 + success:true + credit_score not a 3-digit string (PRD 8.1)
  client_id   text,
  consent_id  uuid references consents(id) on delete set null,  -- ORIGINAL pull's consent; cache hits surface it
  fetched_at  timestamptz not null,
  expires_at  timestamptz not null,                       -- fetched_at + interval '30 days'
  purge_after timestamptz not null                        -- fetched_at + interval '180 days' (raw retention)
);

create table if not exists report_cache (                 -- Experian Report cache, 30 d, per PAN, shared across opportunities
  pan_hash    text primary key,
  storage_key text not null,
  is_ntc      boolean not null default false,             -- 'no file' stored as NTC for 30 days (PRD 10)
  report_date date,                                       -- CreditProfileHeader.ReportDate; null on NTC
  client_id   text,
  consent_id  uuid references consents(id) on delete set null,
  fetched_at  timestamptz not null,
  expires_at  timestamptz not null,
  purge_after timestamptz not null
);

create table if not exists checks (
  id                    uuid primary key,
  kind                  text not null check (kind in ('quick','full')),
  lc_id                 text not null,
  lead_id               text,
  opportunity_id        text,
  opportunity_stage     text,                             -- caller-supplied, trusted, logged (PRD 5.8)
  contact_id            text,                             -- /attach (quick) or from the opportunity (full)
  pan_hash              text not null,
  pan_last4             text not null,
  mobile_hash           text,
  mobile_last4          text,
  status                text not null check (status in ('running','ok','pan_not_verified','refused','pending_verification','service_failure','error')),
  code                  text check (code in ('CONSENT_MISSING','QUOTA_EXHAUSTED','PAN_DAILY_CAP','STAGE_TOO_EARLY','REPORT_WINDOW_OPEN','IN_FLIGHT','PAN_LOCKED','WARN_NTC_RECENT','PAN_CHANGE_NEEDS_MANAGER','NO_PINCODE_ON_PAN','LIVE_CALLS_DISABLED','SUREPASS_FAILURE','OPENAI_FAILURE')),
  billed                boolean not null default false,   -- persist ground truth: any billable surepass_calls row
  counted_against_quota boolean not null default false,   -- persist: quick AND no-stored-score AND 200+success Score call
  is_refresh            boolean not null default false,
  recompute_of          uuid references checks(id),       -- /recompute lineage
  consent_id            uuid references consents(id) on delete set null,  -- the 5y consent purge must not be blocked
  lc_view               jsonb,                            -- exactly what the API returned; no score/bureau/raw fragments
  internal              jsonb,                            -- facts minus raw payloads; carries original_consent_id on hits
  langsmith_run_id      text,
  created_at            timestamptz not null,
  constraint code_legality check (
       (status = 'refused'         and code in ('CONSENT_MISSING','QUOTA_EXHAUSTED','PAN_DAILY_CAP','STAGE_TOO_EARLY','REPORT_WINDOW_OPEN','IN_FLIGHT','PAN_LOCKED','WARN_NTC_RECENT','PAN_CHANGE_NEEDS_MANAGER','LIVE_CALLS_DISABLED'))
    or (status = 'service_failure' and code in ('SUREPASS_FAILURE','OPENAI_FAILURE'))
    or (status = 'ok'              and (code is null or code = 'NO_PINCODE_ON_PAN'))
    or (status in ('running','pan_not_verified','pending_verification','error') and code is null))
);
create index if not exists idx_checks_quota on checks (lc_id, created_at) where kind = 'quick' and counted_against_quota;
create index if not exists idx_checks_pan on checks (pan_hash, created_at desc);
create index if not exists idx_checks_opp_full on checks (opportunity_id, created_at desc) where kind = 'full';

create table if not exists surepass_calls (               -- one row per HTTP attempt; client-owned logging via the Repo sink
  id          uuid primary key,
  check_id    uuid not null references checks(id),        -- checks row inserted at admission BEFORE any call (C1)
  pan_hash    text not null,                              -- denormalized for bureau_calls_today()
  endpoint    text not null,                              -- '/api/v1/pan/pan-comprehensive' | '/api/v1/credit-report-experian/score' | '/api/v1/credit-report-experian/fetch-report'
  client_id   text,
  http_status integer,                                    -- null when transport failed / timed out pre-response
  success     boolean,                                    -- null for the {errors,message} validation envelope
  message_code text,
  latency_ms  integer,                                    -- PRD 11 log fields: client_id, endpoint, status, message_code, latency
  billable    boolean not null default false,             -- (200 AND success) OR error_kind='timeout' (spend unknown, conservative)
  error_kind  text check (error_kind in ('timeout','transport','auth','rate_limited','upstream','protocol','validation')),
  storage_key text,                                       -- raw envelope object; null when no body arrived; THE object owner for purge
  called_at   timestamptz not null,
  purge_after timestamptz not null,                       -- called_at + interval '180 days'
  purged_at   timestamptz                                 -- set by the purge job after object deletion
);
create index if not exists idx_calls_check on surepass_calls (check_id, called_at);
create index if not exists idx_calls_pan_billable on surepass_calls (pan_hash, called_at) where billable;
create index if not exists idx_calls_purge on surepass_calls (purge_after) where purged_at is null;

create table if not exists pan_locks (                    -- PRD 5.7 / 5.11 / 5.12
  pan_hash    text not null,
  kind        text not null check (kind in ('in_flight','timeout','daily_cap')),
  check_id    uuid references checks(id),
  locked_at   timestamptz not null default now(),
  lock_until  timestamptz not null,                       -- in_flight: now+180s; timeout: now+10min; daily_cap: next IST midnight
  released_at timestamptz,
  primary key (pan_hash, kind)                            -- active := released_at is null AND lock_until > now()
);

create table if not exists review_queue (
  id          uuid primary key,
  check_id    uuid references checks(id),
  pan_hash    text not null,
  kind        text not null check (kind in ('pan_daily_cap','timeout_reconciliation','openai_failure','safety_demoted_all','not_fundable_pattern','pan_change_manager','model_review_flags','surepass_auth')),
  reason      text not null,
  created_at  timestamptz not null,
  resolved_at timestamptz,                                -- unresolved 'timeout_reconciliation' blocks billed re-pulls (PRD 5.12)
  resolved_by text
);
create index if not exists idx_review_open on review_queue (pan_hash, kind) where resolved_at is null;

create table if not exists lc_quota_overrides (           -- PRD 5.5: manager-configurable per LC
  lc_id       text primary key,
  daily_quota integer not null check (daily_quota >= 0),
  set_by      text not null,
  updated_at  timestamptz not null default now()
);

create table if not exists opportunity_pan_events (       -- PRD 5.13 PAN-correction audit
  id             uuid primary key,
  opportunity_id text not null,
  pan_hash       text not null,
  pan_last4      text not null,
  event          text not null check (event in ('initial','correction','manager_change')),
  actor_lc_id    text not null,
  check_id       uuid references checks(id),
  created_at     timestamptz not null
);
create index if not exists idx_opp_pan on opportunity_pan_events (opportunity_id, created_at);

-- ---- SQL functions (both IST-day; sargable via day-start bounds; IST has no DST) ----
-- MemoryRepo mirrors both exactly (same day bounds, same predicates).
create or replace function quick_checks_used_today(p_lc_id text) returns integer
language sql stable as $$
  with d as (select (date_trunc('day', now() at time zone 'Asia/Kolkata') at time zone 'Asia/Kolkata') as s)
  select count(*)::integer from checks c, d
  where c.lc_id = p_lc_id and c.kind = 'quick' and c.counted_against_quota
    and c.created_at >= d.s and c.created_at < d.s + interval '1 day';
$$;

create or replace function bureau_calls_today(p_pan_hash text) returns integer
language sql stable as $$
  with d as (select (date_trunc('day', now() at time zone 'Asia/Kolkata') at time zone 'Asia/Kolkata') as s)
  select count(*)::integer from surepass_calls sc, d
  where sc.pan_hash = p_pan_hash and sc.billable
    and sc.endpoint in ('/api/v1/credit-report-experian/score','/api/v1/credit-report-experian/fetch-report')
    and sc.called_at >= d.s and sc.called_at < d.s + interval '1 day';
$$;

-- Atomic in-flight acquire: insert the lock, or steal it ONLY when the held lock is released or expired.
-- Any other writer of pan_locks('in_flight') would break mutual exclusion: the Repo is the sole writer.
create or replace function acquire_pan_lock(p_pan_hash text, p_check_id uuid, p_until timestamptz)
returns boolean language plpgsql as $$
declare n integer;
begin
  insert into pan_locks (pan_hash, kind, check_id, locked_at, lock_until, released_at)
  values (p_pan_hash, 'in_flight', p_check_id, now(), p_until, null)
  on conflict (pan_hash, kind) do update
    set check_id = excluded.check_id, locked_at = now(), lock_until = excluded.lock_until, released_at = null
    where pan_locks.released_at is not null or pan_locks.lock_until <= now();
  get diagnostics n = row_count;
  return n = 1;          -- false -> the lock is genuinely held -> IN_FLIGHT, never queued (PRD 5.11)
end $$;

-- ---- RLS lockdown: service-role only (no policies) ----
alter table lenders enable row level security;            alter table lender_pincodes enable row level security;
alter table consents enable row level security;           alter table pan_records enable row level security;
alter table score_cache enable row level security;        alter table report_cache enable row level security;
alter table checks enable row level security;             alter table surepass_calls enable row level security;
alter table pan_locks enable row level security;          alter table review_queue enable row level security;
alter table lc_quota_overrides enable row level security; alter table opportunity_pan_events enable row level security;

-- ---- LangGraph checkpointer isolation (SURVIVES VERBATIM from the v1.0 dossier / ruling C15 +
-- audit-technical-feasibility fix: server-side, pooler-agnostic) ----
create schema if not exists checkpointer;
-- Ops (documented, run once with a real password): create role checkpointer_svc login password '...';
-- DATABASE_URL (the checkpointer DSN) authenticates as checkpointer_svc via the SESSION pooler (:5432), never :6543.
-- alter role checkpointer_svc in database postgres set search_path = checkpointer;
-- grant usage, create on schema checkpointer to checkpointer_svc;
-- Checkpointer tables are created by PostgresSaver.setup(); after setup the app asserts
-- to_regclass('public.checkpoints') IS NULL and fails fast (checkpoint blobs carry raw PAN/payloads;
-- public schema is PostgREST-exposed).

-- Storage (ops, documented in README): private bucket 'raw-responses' (Supabase Storage, encrypted at
-- rest); created via storage API create_bucket('raw-responses', {'public': false}). Purge job
-- (scripts/purge_raw.py, day one, cron): (1) select id, storage_key from surepass_calls where
-- purge_after < now() and purged_at is null -> RawStore.delete_raw(batch of keys) -> set purged_at;
-- (2) delete cache rows (pan_records/score_cache/report_cache) where purge_after < now();
-- (3) delete from consents where purge_after < now() (checks/score_cache/report_cache rows that reference a
-- purged consent keep existing with consent_id set null).  Cache rows always point at keys owned by a
-- surepass_calls row, so object deletion is driven solely by surepass_calls.
