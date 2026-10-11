"""The ONE LLM seam (PRD v1.1 9.2): the rank payload, the ``RankModel`` protocol, the production factory, the builder.

The model ranks and explains; it never produces a number and may only LOWER confidence (``min(code, model)`` is applied
later by ``lenders.safety.final_confidence``; nothing here clamps). Everything it sees is a ``RankContext``:

* ``RankContext`` / ``LenderContext`` are frozen, ``extra='forbid'`` and STRUCTURALLY PII-free. The NEVER-list (raw PAN,
  mobile, learner name, pincodes, the score number, bureau names, account identifiers, raw envelopes, storage keys) has
  no field to live in: the only fields that can hold free text are ``lender_id``, the two admin labels, the lender's
  published ``policy_notes``, the criteria field names in ``uncertain_fields`` and the retry's own ``prior_error``. The
  lender's DISPLAY NAME is resolved to the LcView in redact only; the score floor VALUE (a score number) is replaced by the
  pre-computed ``band_clears_floor``.
* ``build_rank_context`` is a pure mapping from the state's derived facts. It never reads ``pan`` / ``name`` / ``mobile``,
  ``identity.full_name`` / ``dob`` / ``pan_pincode``, ``score.score``, the pincode lists or the lender names.
* ``make_rank_model`` builds the production callable. ``max_retries=0`` is load-bearing: the ONE business retry (on a
  raised exception or invalid output) lives in the rank NODE, not here and not in the SDK, so a call is exactly one HTTP
  attempt under the fixed ``OPENAI_TIMEOUT_S``. ``method='function_calling'`` keeps constraint violations as parse
  exceptions the node's retry absorbs (the library default has moved to ``json_schema``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from eligibility_agent.config import OPENAI_TIMEOUT_S
from eligibility_agent.derive.range import band_from_score
from eligibility_agent.graph.prompts import PRIOR_ERROR_MAX_CHARS, PromptBundle, render_rank_user
from eligibility_agent.graph.state import (
    BAND_ORDER,
    Band,
    CheckState,
    Confidence,
    GatedLender,
    Kind,
    LenderCriteria,
    LenderNumbers,
    ModelOutput,
)
from eligibility_agent.lenders.gate import uncertain_required_fields

__all__ = [
    "OPENAI_TIMEOUT_S",
    "LenderContext",
    "RankContext",
    "RankModel",
    "build_rank_context",
    "make_rank_model",
]


class _Context(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LenderContext(_Context):
    """One lender that survived the gate, as the model sees it. No name, no pincodes, no score floor value."""

    lender_id: str  # id only: names are resolved in redact, never sent to the model
    ceiling: float | None  # code-derived display values; the model repeats NO numbers
    range_lower: float | None
    range_upper: float | None
    not_fundable_alone: bool
    age_at_application: int | None
    age_at_maturity: int | None
    band_clears_floor: bool | None  # pre-computed (safety rule 1 as a predicate); the floor VALUE is a score number
    ntc_accepted: bool | None
    foir_limit: float | None
    min_income: float | None
    tenure_label: str  # '12 to 60 months'
    rate_band_label: str  # '11 to 14%'
    policy_notes: str | None
    uncertain_fields: list[str]  # required criteria the admin marked 'uncertain' (names only)


class RankContext(_Context):
    """EXACTLY the 9.2 payload. Never present: PAN, name, mobile, Aadhaar, account numbers, raw report, the raw score,
    a bureau name, pincodes. ``band`` is the bucket label only."""

    kind: Kind
    band: Band
    code_confidence: Confidence  # the given value the model may only lower
    foir: float | None  # current FOIR (used EMI / income)
    emi_floor_flag: bool
    existing_emi_reported: float | None  # None on a quick check: no credit file was pulled
    existing_emi_estimated: float | None
    emi_past_tenure_accounts: int
    emi_stale_accounts: int
    declared_income: float | None
    declared_emi: float | None  # shown alongside the file's split when both exist
    aadhaar_linked: bool | None
    recent_enquiries_180d: int | None
    past_due_flag: bool | None  # report.past_due_total > 0
    stale_account_flag: bool | None
    lenders: list[LenderContext]  # surviving the gate only, sorted by lender id
    attempt: int  # the call about to be made: 1, or 2 on the one retry
    prior_error: str | None  # retry feedback: the first call's exception text, capped


class RankModel(Protocol):
    """The DI seam: ``GraphDeps.rank_model`` (Task 11). ``FakeRankModel`` implements the same ``__call__``."""

    def __call__(self, ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput: ...


def make_rank_model(
    *, model: str, prompts: PromptBundle, timeout: float = OPENAI_TIMEOUT_S, api_key: str | None = None
) -> RankModel:
    """The production seam: ``ChatOpenAI`` (temperature 0, fixed timeout, ZERO SDK retries) with ``ModelOutput`` structured
    output via ``function_calling``. Construction does no I/O, but it needs a key: ``api_key`` when given (the app passes
    ``Settings.openai_api_key``, which may come from a ``.env`` file ChatOpenAI never reads), else ``OPENAI_API_KEY`` from
    the process environment (ChatOpenAI's own lookup).

    The returned callable makes exactly one model call per invocation, threads ``config`` through (so the LLM child run
    nests under the rank node's trace), and raises on anything that is not a ``ModelOutput``. It never retries and never
    swallows: the node owns the one retry and the ``OPENAI_FAILURE`` mapping.
    """
    chat = (
        ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=0)
        if api_key is None
        else ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=0, api_key=SecretStr(api_key))
    )
    # langchain types the schema parameter as a bare ``type`` (partially unknown to pyright); the cast restores a known
    # type, and the isinstance check in ``_call`` is the real runtime guard.
    runnable = cast(
        "Runnable[Any, Any]",
        chat.with_structured_output(ModelOutput, method="function_calling"),  # pyright: ignore[reportUnknownMemberType]
    )

    def _call(ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput:
        messages = [SystemMessage(content=prompts.rank_system(ctx.kind)), HumanMessage(content=render_rank_user(ctx))]
        out = runnable.invoke(messages, config=config)
        if not isinstance(out, ModelOutput):
            raise TypeError(f"structured output returned {type(out).__name__}, expected ModelOutput")
        return out

    return _call


def _band_clears_floor(band: Band, criteria: LenderCriteria) -> bool | None:
    """Safety rule 1 (``band_floor``) as a predicate: NTC needs ``ntc_accepted is True``; a scored band must sit in a
    bucket at or above the bucket of ``score_floor``. None only for a scored band against an unknown floor (the rule
    never breaches there). ``tests/test_llm_seam.py`` pins agreement with ``lenders.safety.check_pick`` over every
    band x floor x ntc_accepted combination."""
    if band == "NTC":
        return criteria.ntc_accepted is True
    if criteria.score_floor is None:
        return None
    return BAND_ORDER[band] >= BAND_ORDER[band_from_score(criteria.score_floor)]


def _lender_context(
    lender: GatedLender, numbers: LenderNumbers | None, *, band: Band, age_at_application: int | None
) -> LenderContext:
    criteria = lender.criteria
    return LenderContext(
        lender_id=lender.id,
        ceiling=None if numbers is None else numbers.ceiling,
        range_lower=None if numbers is None else numbers.loan_max_lower,
        range_upper=None if numbers is None else numbers.loan_max_upper,
        not_fundable_alone=False if numbers is None else numbers.not_fundable_alone,
        age_at_application=age_at_application,
        age_at_maturity=None if numbers is None else numbers.age_at_maturity_years,
        band_clears_floor=_band_clears_floor(band, criteria),
        ntc_accepted=criteria.ntc_accepted,
        foir_limit=criteria.foir_limit,
        min_income=criteria.min_income,
        tenure_label=lender.tenure_label,
        rate_band_label=lender.rate_band_label,
        policy_notes=criteria.policy_notes or None,
        uncertain_fields=uncertain_required_fields(criteria),
    )


def build_rank_context(state: CheckState, gated: Sequence[GatedLender] | None = None) -> RankContext:
    """Map the state's derived facts onto the 9.2 payload. Pure; the NEVER-list is structurally unreachable.

    ``gated`` defaults to ``state['gated_lenders']`` (the rank node passes the same list); lenders are emitted sorted by id
    so the context, and therefore the rendered prompt, is independent of caller or dict ordering. A lender without
    ``lender_numbers`` is rendered with unknown numbers (safety fails closed on it regardless).

    On a quick check no credit file exists, so the file-derived fields are None rather than invented zeroes. ``attempt`` is
    the call about to be made (``rank_attempts + 1``); ``prior_error`` is carried only from the second attempt on.

    Raises ``ValueError`` when the state is one the rank node could not have reached (no ``kind``, ``derived`` or
    ``code_confidence``).
    """
    kind = state.get("kind")
    derived = state.get("derived")
    code_confidence = state.get("code_confidence")
    if kind is None or derived is None or code_confidence is None:
        missing = [
            name
            for name, value in (("kind", kind), ("derived", derived), ("code_confidence", code_confidence))
            if value is None
        ]
        raise ValueError(f"cannot build a rank context: state is missing {', '.join(missing)}")

    lenders = state.get("gated_lenders", []) if gated is None else gated
    numbers = state.get("lender_numbers", {})
    identity = state.get("identity")
    report = state.get("report") if kind == "full" else None
    emi = derived.emi
    stale_accounts = sum(1 for item in emi.items if item.kind == "stale_excluded")

    attempt = state.get("rank_attempts", 0) + 1
    rank_error = state.get("rank_error")
    prior_error = rank_error[:PRIOR_ERROR_MAX_CHARS] if attempt > 1 and rank_error else None

    return RankContext(
        kind=kind,
        band=derived.band,
        code_confidence=code_confidence,
        foir=derived.current_foir,
        emi_floor_flag=emi.emi_floor_flag,
        existing_emi_reported=None if report is None else emi.reported_total,
        existing_emi_estimated=None if report is None else emi.estimated_total,
        emi_past_tenure_accounts=sum(1 for item in emi.items if item.past_tenure),
        emi_stale_accounts=stale_accounts,
        declared_income=state.get("monthly_income"),
        declared_emi=emi.declared,
        aadhaar_linked=None if identity is None else identity.aadhaar_linked,
        recent_enquiries_180d=None if report is None else report.enquiries_180d,
        past_due_flag=None if report is None else report.past_due_total > 0.0,
        stale_account_flag=None if report is None else stale_accounts > 0,
        lenders=sorted(
            (
                _lender_context(
                    g, numbers.get(g.id), band=derived.band, age_at_application=derived.age_at_application_years
                )
                for g in lenders
            ),
            key=lambda lender: lender.lender_id,
        ),
        attempt=attempt,
        prior_error=prior_error,
    )
