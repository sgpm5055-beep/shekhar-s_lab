"""``FakeRankModel``: the deterministic stand-in for the single LLM node, plus scripted misbehaviour.

It implements the SAME protocol as the production seam (``graph.llm.RankModel``): ``__call__(ctx, *, config=None) ->
ModelOutput``, keyword-only ``config`` included (a signature-parity test pins it), so it drops into
``GraphDeps(rank_model=...)`` and ``create_app(rank_model=...)`` with no monkeypatching. Every call is recorded (the
context AND the config) for spy assertions, including calls that raise.

Behaviour is a *script*: a sequence of steps consumed one per call, the LAST step repeating once the script is spent.
A step is a ``ModelOutput`` (returned), an ``Exception`` (raised) or a callable ``(RankContext) -> ModelOutput``
(a responder, given the context). No script means the default ranking: ``default_ranking``.

Misbehaviour the node and the safety fence must survive, one helper each:

* ``out_of_gate_pick``: picks lenders that are not in the gate (invented or simply gated out); safety demotes them.
* ``below_floor_pick``: picks lenders whose floor the band does not clear; safety demotes them (``band_floor``).
* ``raised_confidence``: returns a confidence HIGHER than the code's; ``final_confidence`` ignores it (``min``).
* ``lowered_confidence``: the legitimate direction, for contrast.
* ``FakeRankModel.invalid_twice()`` / ``invalid_output_error``: raises a genuine pydantic ``ValidationError`` on every
  call; the node maps two failures to OPENAI_FAILURE after exactly two calls.

The default prose is number-free and bureau-free (band WORDS, never the bucket label), so it passes the output
boundary's leak scan unchanged and keeps no-leak evaluators green by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from eligibility_agent.graph.llm import LenderContext, RankContext
from eligibility_agent.graph.state import CONFIDENCE_ORDER, Band, Confidence, ModelOutput, Pick

__all__ = [
    "FakeRankModel",
    "RankCall",
    "Responder",
    "Step",
    "below_floor_pick",
    "default_ranking",
    "invalid_output_error",
    "lowered_confidence",
    "out_of_gate_pick",
    "raised_confidence",
    "ranked_output",
]

Responder: TypeAlias = Callable[[RankContext], ModelOutput]
Step: TypeAlias = ModelOutput | Exception | Responder

_BAND_WORDS: Final[dict[Band, str]] = {
    "780+": "very strong",
    "720-780": "strong",
    "650-720": "moderate",
    "<650": "thin",
    "NTC": "new-to-credit",
}
_THIN_BANDS: Final[frozenset[Band]] = frozenset({"NTC", "<650"})
_MAX_ALTERNATES: Final[int] = 2
_INVENTED_LENDER_ID: Final[str] = "L_NOT_IN_GATE"


@dataclass(frozen=True, slots=True)
class RankCall:
    """One recorded call: exactly what the seam received."""

    ctx: RankContext
    config: RunnableConfig | None


def _pick(lender_id: str, rank: int, rationale: str) -> Pick:
    return Pick(lender_id=lender_id, rank=rank, rationale=rationale, conditions=[])


def ranked_output(
    *lender_ids: str,
    confidence: Confidence = "high",
    co_applicant_suggested: bool = False,
    co_applicant_reason: str | None = None,
    review_flags: Sequence[str] = (),
) -> ModelOutput:
    """A ``ModelOutput`` ranking the given ids in order: the first is the recommendation, the rest the alternates.

    The schema allows two alternates, so more than three ids raises ``ValidationError`` (the model cannot do it either).
    No ids means no recommendation.
    """
    picks = [_pick(lender_id, rank, f"Scripted pick {rank}") for rank, lender_id in enumerate(lender_ids, start=1)]
    return ModelOutput(
        recommendation=picks[0] if picks else None,
        alternates=picks[1:],
        confidence=confidence,
        co_applicant_suggested=co_applicant_suggested,
        co_applicant_reason=co_applicant_reason,
        review_flags=list(review_flags),
    )


def _eligible(lender: LenderContext) -> bool:
    return lender.band_clears_floor is not False and not lender.not_fundable_alone


def default_ranking(ctx: RankContext) -> ModelOutput:
    """The happy-path model: rank, in id order, every lender whose floor the band does not rule out and that has EMI
    headroom; the first is the recommendation, the next two the alternates.

    Confidence is the code's own (never raised). A co-applicant is suggested for a thin band (NTC, below 650) and whenever
    nothing is eligible. The prose uses band words only: no digits, no bureau name, nothing the output boundary scrubs.
    """
    word = _BAND_WORDS[ctx.band]
    eligible = [lender.lender_id for lender in ctx.lenders if _eligible(lender)]
    picks = [
        _pick(lender_id, rank, f"Serves the learner's area and fits the lender's criteria for a {word} credit profile.")
        for rank, lender_id in enumerate(eligible[: 1 + _MAX_ALTERNATES], start=1)
    ]
    suggest = ctx.band in _THIN_BANDS or not picks
    return ModelOutput(
        recommendation=picks[0] if picks else None,
        alternates=picks[1:],
        confidence=ctx.code_confidence,
        co_applicant_suggested=suggest,
        co_applicant_reason="A co-applicant with a stronger credit history would help this application." if suggest else None,
        review_flags=[],
    )


def out_of_gate_pick(*lender_ids: str) -> ModelOutput:
    """The model picks lenders that are not in the gate (the first id is the recommendation, the rest alternates).

    Defaults to one invented id; pass a real id that failed the pincode gate to model "picked a gated-out lender".
    Safety demotes each with rule ``not_in_gate``; when none survive the run goes to review.
    """
    return ranked_output(*(lender_ids or (_INVENTED_LENDER_ID,)))


def below_floor_pick() -> Responder:
    """Responder: picks the lenders (at most three) whose floor the applicant's band does NOT clear.

    Safety demotes each with rule ``band_floor``. Raises ``AssertionError`` when the context has no such lender (the test
    world is wrong; silently returning a valid ranking would make the misbehaviour test vacuous).
    """

    def respond(ctx: RankContext) -> ModelOutput:
        below = [lender.lender_id for lender in ctx.lenders if lender.band_clears_floor is False]
        assert below, "below_floor_pick needs a gated lender whose floor the band does not clear"
        return ranked_output(*below[: 1 + _MAX_ALTERNATES], confidence=ctx.code_confidence)

    return respond


def raised_confidence(level: Confidence = "high") -> Responder:
    """Responder: the default ranking, but with a confidence HIGHER than the code's (a model that ignores rule 4).

    ``final_confidence`` is ``min(code, model)``, so this changes nothing downstream (the "raise ignored" pin). Raises
    ``AssertionError`` when ``level`` is not strictly above the code confidence (nothing was raised).
    """

    def respond(ctx: RankContext) -> ModelOutput:
        assert CONFIDENCE_ORDER[level] > CONFIDENCE_ORDER[ctx.code_confidence], (
            f"cannot raise confidence to {level!r} above the code's {ctx.code_confidence!r}"
        )
        return default_ranking(ctx).model_copy(update={"confidence": level})

    return respond


def lowered_confidence(level: Confidence = "low") -> Responder:
    """Responder: the default ranking with a confidence strictly LOWER than the code's (the permitted direction)."""

    def respond(ctx: RankContext) -> ModelOutput:
        assert CONFIDENCE_ORDER[level] < CONFIDENCE_ORDER[ctx.code_confidence], (
            f"cannot lower confidence to {level!r} from the code's {ctx.code_confidence!r}"
        )
        return default_ranking(ctx).model_copy(update={"confidence": level})

    return respond


def invalid_output_error() -> ValidationError:
    """A genuine pydantic ``ValidationError`` for ``ModelOutput`` (an undeclared numeric field plus missing fields): what
    the real seam raises when the model's structured output does not parse."""
    try:
        ModelOutput.model_validate({"ceiling": 800000})
    except ValidationError as exc:
        return exc
    raise AssertionError("ModelOutput accepted an undeclared numeric field")  # unreachable: extra='forbid'


class FakeRankModel:
    """Deterministic ``RankModel``. ``FakeRankModel()`` is the happy-path model; pass ``script`` to misbehave."""

    def __init__(self, script: Sequence[Step] | None = None) -> None:
        self._script: list[Step] = list(script or [])
        self.calls: list[RankCall] = []

    @classmethod
    def invalid_twice(cls) -> FakeRankModel:
        """Raises ``ValidationError`` on EVERY call (a third call is still invalid and still counted), so the node's one
        retry also fails and the run maps to OPENAI_FAILURE after exactly two calls."""
        return cls(script=[invalid_output_error()])

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def contexts(self) -> list[RankContext]:
        return [call.ctx for call in self.calls]

    @property
    def configs(self) -> list[RunnableConfig | None]:
        return [call.config for call in self.calls]

    def __call__(self, ctx: RankContext, *, config: RunnableConfig | None = None) -> ModelOutput:
        index = len(self.calls)
        self.calls.append(RankCall(ctx=ctx, config=config))
        if not self._script:
            return default_ranking(ctx)
        step = self._script[min(index, len(self._script) - 1)]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ModelOutput):
            return step
        return step(ctx)
