"""Prompt text, the Hub pull-by-tag loader and the user-message renderer for the single LLM node (PRD v1.1 9.2).

The rank node is the ONLY LLM node in the system: it ranks the lenders that already passed the gate and explains the
choice in plain words. It never produces a number and may only LOWER confidence; everything that fences it lives in code.

* The pinned system prompt (``RANK_SYSTEM_CORE`` plus one per-kind addendum) is a literal in this file, the source of
  truth. It is BRACE-FREE, so the single Hub message (``hero/eligibility-rank``, pinned by tag per release) is stored
  variable-free and no template formatter can ever mangle it. User messages are built by plain Python
  (``render_rank_user``), so lender notes never pass through template formatting either.
* ``PROMPT_SOURCE=local`` (the default; always in tests and CI) never touches the network and never imports langsmith:
  the only langsmith import in this module is inside ``_pull_hub_core``, behind the hub branch. A hub pull that fails in
  ANY way (no tag, network, auth, missing tag, unexpected shape) falls back to the local literals with a warning.
* ``render_rank_user`` is pure and deterministic: a function of the ``RankContext`` alone. The context is structurally
  PII-free (see ``graph/llm.py``), so nothing here has anything to filter. Free text (lender policy notes, the retry error)
  is collapsed onto one line so it cannot forge extra layout lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from eligibility_agent.graph.state import Kind

if TYPE_CHECKING:  # type-only: graph/llm.py imports this module at runtime, and this module must stay a light leaf
    from eligibility_agent.config import Settings
    from eligibility_agent.graph.llm import LenderContext, RankContext

__all__ = [
    "FULL_ADDENDUM",
    "HUB_RANK",
    "PRIOR_ERROR_MAX_CHARS",
    "PROMPTS_LOCAL_VERSION",
    "QUICK_ADDENDUM",
    "RANK_SYSTEM_CORE",
    "PromptBundle",
    "load_prompts",
    "local_prompts",
    "render_rank_user",
]

logger = logging.getLogger(__name__)

PROMPTS_LOCAL_VERSION: Final[str] = "2026-10-10-a"  # bump on any text change; mirrored as the Hub commit tag
HUB_RANK: Final[str] = "hero/eligibility-rank"  # pinned by tag per release; PROMPT_SOURCE=local by default
PRIOR_ERROR_MAX_CHARS: Final[int] = 500  # the retry error text fed back to the model is capped (context builder AND renderer)

# Brace-free by construction (a test enforces it): stored as the single Hub system message, composed locally with ONE
# tier addendum.
RANK_SYSTEM_CORE: Final[str] = (
    "You advise Hero Vired Learning Consultants (LCs) on education-loan fundability. You receive code-computed facts "
    "about one applicant and a list of lenders that already passed a location gate and policy checks. Your job is to "
    "rank those lenders, explain the choice in plain spoken language, and suggest a co-applicant when that is the "
    "honest path.\n\n"
    "Hard rules:\n"
    "1. Rank only the lenders given, by lender_id. Never invent, rename, merge or add a lender. Never suggest "
    "overriding or working around the location gate or any policy check.\n"
    "2. You never produce a number. Every amount, ceiling, range, rate and score is computed by code and shown "
    "separately. rationale and conditions must contain no digits taken from the credit file, no credit score, no "
    "score range, and no credit agency's name. Use band-driven plain words: strong repayment record, existing EMIs "
    "look high next to income, limited credit history.\n"
    "3. Prefer co_applicant_suggested true over an empty recommendation when the credit history is new or thin, when "
    "the band sits below every lender's comfort, or when EMI headroom is thin. A co-applicant cannot fix location "
    "eligibility: with no lenders listed, return recommendation null and say options will be reviewed.\n"
    "4. confidence: you are given code_confidence. You may return the same value or a lower one, never a higher one. "
    "Lower it when facts are estimated, declared-only, stale or conflicting, and name why in review_flags.\n"
    "5. conditions are short actionable items an LC can say aloud (for example: PAN to Aadhaar link needed; confirm "
    "declared income with a payslip). rationale is one or two short sentences.\n"
    "6. Indian context: amounts are INR, income is monthly, FOIR is existing monthly EMI divided by monthly income; "
    "lower is better."
)

QUICK_ADDENDUM: Final[str] = (
    "Quick check: figures are declared by the learner, not verified. If income was not given, no amounts exist; rank "
    "on band and criteria fit only."
)

FULL_ADDENDUM: Final[str] = (
    "Full check: facts come from a fresh credit file plus declared income. Estimated or stale EMI rows and "
    "past-tenure accounts are flagged; treat them as items to confirm, and lower confidence when they drive the "
    "decision."
)


def _compose(core: str, kind: Kind) -> str:
    return core + "\n\n" + (QUICK_ADDENDUM if kind == "quick" else FULL_ADDENDUM)


@dataclass(frozen=True)
class PromptBundle:
    """The two composed system prompts, where they came from, and the version/tag that names them."""

    rank_system_quick: str
    rank_system_full: str
    version: str
    source: Literal["local", "hub"]

    def rank_system(self, kind: Kind) -> str:
        return self.rank_system_quick if kind == "quick" else self.rank_system_full


def local_prompts() -> PromptBundle:
    """The committed snapshot: ``RANK_SYSTEM_CORE`` plus the tier addendum, composed locally. No I/O."""
    return PromptBundle(
        rank_system_quick=_compose(RANK_SYSTEM_CORE, "quick"),
        rank_system_full=_compose(RANK_SYSTEM_CORE, "full"),
        version=PROMPTS_LOCAL_VERSION,
        source="local",
    )


def _extract_system_text(template: object) -> str:
    """The text of the single system message of a pulled Hub prompt (``template.messages[0].prompt.template``).

    Raises on any other shape, an empty text, or any brace (a Hub template that carries variables or escapes cannot be
    used verbatim): the caller treats every exception as "use the local literals".
    """
    messages = cast("list[object]", getattr(template, "messages"))
    text: object = getattr(getattr(messages[0], "prompt"), "template")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("hub prompt has no usable system text")
    if "{" in text or "}" in text:
        raise ValueError("hub prompt must be brace-free")
    return text


def _pull_hub_core(tag: str) -> str:
    # The ONLY langsmith import in this module: PROMPT_SOURCE=local never reaches it.
    from langsmith import Client

    return _extract_system_text(Client().pull_prompt(f"{HUB_RANK}:{tag}"))


def load_prompts(settings: Settings) -> PromptBundle:
    """``PROMPT_SOURCE=local`` (default) -> ``local_prompts()``, zero network. ``hub`` -> ``hero/eligibility-rank:<tag>``.

    The Hub holds the core system message; the per-kind addendum is composed locally. A hub source without a pinned
    ``PROMPT_TAG`` is NOT an unpinned "latest" pull: it falls back to local. ANY failure falls back to local with a
    warning that names the exception TYPE only (never its text).
    """
    if settings.prompt_source != "hub":
        return local_prompts()
    tag = (settings.prompt_tag or "").strip()
    if not tag:
        logger.warning("PROMPT_SOURCE=hub but PROMPT_TAG is unset: using the local prompts (never an unpinned hub pull)")
        return local_prompts()
    try:
        core = _pull_hub_core(tag)
    except Exception as exc:
        logger.warning("hub prompt pull failed (%s): using the local prompts", type(exc).__name__)
        return local_prompts()
    return PromptBundle(
        rank_system_quick=_compose(core, "quick"),
        rank_system_full=_compose(core, "full"),
        version=tag,
        source="hub",
    )


# --------------------------------------------------------------------------- the user message


def _one_line(text: str) -> str:
    """Collapse every whitespace run (newlines included) to one space, so free text stays on its own line."""
    return " ".join(text.split())


def _inr(value: float | None, none: str) -> str:
    return none if value is None else f"INR {value + 0.0:.0f}"


def _tri(value: bool | None) -> str:
    return "unknown" if value is None else "yes" if value else "no"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _count(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _emi_line(ctx: RankContext) -> str:
    if ctx.existing_emi_reported is None and ctx.existing_emi_estimated is None:
        return f"- existing EMI from file: none (no credit file pulled); floor flag {_yes_no(ctx.emi_floor_flag)}"
    return (
        f"- existing EMI from file: reported {_inr(ctx.existing_emi_reported, 'none')}"
        f" + estimated {_inr(ctx.existing_emi_estimated, 'none')}; floor flag {_yes_no(ctx.emi_floor_flag)};"
        f" past-tenure accounts {ctx.emi_past_tenure_accounts}; stale accounts {ctx.emi_stale_accounts}"
    )


def _lender_line(lender: LenderContext) -> str:
    range_text = (
        "none"
        if lender.range_lower is None or lender.range_upper is None
        else f"INR {lender.range_lower + 0.0:.0f}-{lender.range_upper + 0.0:.0f}"
    )
    foir_limit = "unknown" if lender.foir_limit is None else f"{lender.foir_limit:g}"
    notes = _one_line(lender.policy_notes) if lender.policy_notes else "-"
    uncertain = ", ".join(lender.uncertain_fields) if lender.uncertain_fields else "none"
    return (
        f"- lender_id {lender.lender_id}"
        f" | ceiling {_inr(lender.ceiling, 'none')}"
        f" | range {range_text}"
        f" | not fundable alone: {_yes_no(lender.not_fundable_alone)}"
        f" | band clears floor: {_tri(lender.band_clears_floor)}"
        f" | NTC accepted: {_tri(lender.ntc_accepted)}"
        f" | FOIR limit {foir_limit}"
        f" | min income {_inr(lender.min_income, '-')}"
        f" | tenure {lender.tenure_label}"
        f" | rate {lender.rate_band_label}"
        f" | age now {_count(lender.age_at_application)}, at maturity {_count(lender.age_at_maturity)}"
        f" | notes: {notes}"
        f" | uncertain fields: {uncertain}"
    )


def render_rank_user(ctx: RankContext) -> str:
    """The user message for one rank call: deterministic, pure, a function of ``ctx`` alone.

    Layout (numbers appear in the INPUT for grounding; the OUTPUT schema has no numeric field but ``rank``)::

        Applicant:
        - credit band / code confidence given / declared income and EMI / existing EMI from file
        - FOIR, PAN-Aadhaar link / enquiries, overdue flag, stale flag
        <blank>
        Lenders that passed the gate - rank ONLY these lender_ids:
        - lender_id ... | ceiling ... | range ... | ... (one line per lender, in ``ctx.lenders`` order; "- none" if empty)
        [attempt > 1 only: <blank>, "Your previous output was invalid[: <error>]", "Return a corrected ModelOutput."]

    ``kind`` is deliberately absent: it selects the system-prompt addendum. The retry block appears ONLY after the
    first attempt; ``prior_error`` is ignored on attempt 1.
    """
    foir = "unknown" if ctx.foir is None else f"{ctx.foir:.2f}"
    lines = [
        "Applicant:",
        f"- credit band: {ctx.band}",
        f"- code confidence given: {ctx.code_confidence}",
        f"- monthly income declared: {_inr(ctx.declared_income, 'no')}; declared existing EMI: {_inr(ctx.declared_emi, 'none')}",
        _emi_line(ctx),
        f"- FOIR: {foir}; PAN-Aadhaar linked: {_tri(ctx.aadhaar_linked)}",
        f"- recent credit enquiries (180d): {_count(ctx.recent_enquiries_180d)};"
        f" overdue amounts on file: {_tri(ctx.past_due_flag)}; stale account flag: {_tri(ctx.stale_account_flag)}",
        "",
        "Lenders that passed the gate - rank ONLY these lender_ids:",
    ]
    lines.extend(_lender_line(lender) for lender in ctx.lenders)
    if not ctx.lenders:
        lines.append("- none")
    if ctx.attempt > 1:
        error = _one_line(ctx.prior_error)[:PRIOR_ERROR_MAX_CHARS] if ctx.prior_error else ""
        lines.extend(["", f"Your previous output was invalid: {error}" if error else "Your previous output was invalid."])
        lines.append("Return a corrected ModelOutput.")
    return "\n".join(lines)
