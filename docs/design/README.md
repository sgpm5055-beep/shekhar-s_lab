# Design dossier guide

Two generations live here:

- `design-*.md` / `audit-*.md` (root): the **PRD v1.0 panel** (8 Oct 2026). Partially SUPERSEDED by PRD v1.1.
  Still authoritative for: checkpointer factory + Supabase pooler caveats + strict-msgpack, tracing, typed-closure
  node bind, SurePass envelope parsing + error taxonomy + FakeSurePassClient machinery, parsing helpers,
  redaction regex pair, FrozenClock, probe-literal guard, version pins, create_app/error-boundary patterns.
  VOID where v1.1 changed the contract (see the plan's K21 VOID list): pan_cache/derived-jsonb, salted SHA-256,
  success:false→NTC, two LLM nodes, old refusal codes, old EMI/ceiling math, tools/ paths, mypy.

- `v11/delta-*.md` / `v11/audit-*.md`: the **PRD v1.1 delta panel** (10 Oct 2026) — the working designs.

Binding order everywhere: Phase1_Financing_Eligibility_Spec_v2.docx > docs/PRD.md (v1.1) >
docs/plans/2026-10-10-eligibility-agent-v1.1.md (whose Canonical Contract K1–K22 **overrides both dossiers**
wherever they differ) > these dossier files.
