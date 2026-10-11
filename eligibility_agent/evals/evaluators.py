"""The seven PRD-12 evaluators, in LangSmith's name-matched kwargs form.

Every evaluator is a plain function ``(inputs, outputs, reference_outputs) -> {"key", "score", "comment"}``:

* ``inputs``: the golden example's inputs (only ``request`` is read here, for the identifier check in ``no_leak``);
* ``outputs``: the ``evals.run.run_one`` contract, the ONLY thing an evaluator judges the run by::

      {"lc_view": dict,                      # the LcView as the LC would receive it (JSON form)
       "billed": bool, "counted_against_quota": bool, "from_cache": bool,
       "admission": "new" | "cache_hit" | "refresh" | "refused" | ...,
       "surepass_calls": [{"endpoint": str, "billable": bool}],   # THIS check's calls; NEVER pin their order (K17)
       "gated_lender_ids": [...],            # pincode-gate survivors
       "safety_survivor_ids": [...],         # lenders the view may name: safety survivors (or, on the K7 model-failed
                                             # path, the code-built floor-clearing gate survivors)
       "band": str | None, "code_confidence": str | None,
       "derived": {lender_id: {"ceiling": int | None, "lower": int | None, "upper": int | None}}}   # display values

* ``reference_outputs``: the golden's hand-written ``expected`` literals. Every reference key is OPTIONAL and a key that is
  present is a pin (an explicit ``None`` expects ``None``; it is not a wildcard).

Independence rules (binding, T15 supplement): evaluators import NOTHING from derive / range / lenders / graph to produce an
expected value. Money is compared with the code-derived table the run reports AND with the golden's literal table, never
with a recomputation. The leak patterns are IMPORTED from ``output/lc_view`` (the C11 rule: one home, no copies), and the
uuid-ish ``check_id`` and lender ``id`` fields are skipped by the leak scan (uuid4 hex trips the score / account patterns
structurally). The only other product import is the two bureau endpoint names, which classify ``surepass_calls``.

The seventh evaluator is the LLM judge: constructed only when ``OPENAI_API_KEY`` is set, never part of the local / CI
run (``DETERMINISTIC_EVALUATORS`` is the six that need no model).
"""

from __future__ import annotations

import itertools
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Final, TypeAlias, TypedDict, cast

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from eligibility_agent.output.lc_view import GLOBAL_LEAK_PATTERNS, LLM_LEAK_PATTERNS, line_leaks
from eligibility_agent.surepass.client import BUREAU_ENDPOINTS

__all__ = [
    "DETERMINISTIC_EVALUATORS",
    "GLOBAL_LEAK_PATTERNS",
    "JUDGE_KEY",
    "JUDGE_MODEL",
    "JUDGE_RUBRIC",
    "LLM_AUTHORED_PATHS",
    "LLM_LEAK_PATTERNS",
    "EvalResult",
    "Evaluator",
    "JudgeFn",
    "JudgeVerdict",
    "billing_consistency",
    "confidence_not_raised",
    "judge_evaluators",
    "lenders_gated",
    "line_leaks",
    "make_llm_judge",
    "no_leak",
    "numbers_from_code",
    "outcome_matches_band",
]

LLM_AUTHORED_PATHS: Final[tuple[str, ...]] = ("rationale", "conditions", "items_to_confirm", "coapplicant_reason")
_LENDER_FIELDS: Final[tuple[str, ...]] = ("top_lender", "alternates")
_CONFIDENCE_ORDER: Final[dict[str, int]] = {"low": 0, "medium": 1, "high": 2}
_THIN_BANDS: Final[frozenset[str]] = frozenset({"NTC", "<650"})
_K7_CODE: Final[str] = "OPENAI_FAILURE"
_MAX_FINDINGS: Final[int] = 5
_FIGURE_DIGITS: Final[int] = 3  # PRD 9.2: the model produces no number; three digits in a row is a figure
_REQUEST_IDENTIFIER_FIELDS: Final[tuple[str, ...]] = ("pan", "mobile", "name")

# View fields an outcome golden may pin, read straight off the LcView.
_VIEW_PINS: Final[tuple[str, ...]] = (
    "status",
    "code",
    "headline",
    "confidence",
    "coapplicant_suggested",
    "label",
    "items_to_confirm",
    "pan_aadhaar_link_warning",
    "quick_checks_left_today",
    "checked_on",
    "refresh_available_on",
)

Json: TypeAlias = Mapping[str, Any]


class EvalResult(TypedDict):
    key: str
    score: bool
    comment: str


Evaluator: TypeAlias = Callable[[Json, Json, Json], EvalResult]


# --------------------------------------------------------------------------- small helpers


def _result(key: str, problems: Sequence[str]) -> EvalResult:
    shown = list(problems[:_MAX_FINDINGS])
    if len(problems) > _MAX_FINDINGS:
        shown.append(f"(+{len(problems) - _MAX_FINDINGS} more)")
    return {"key": key, "score": not problems, "comment": "; ".join(shown) if problems else "ok"}


def _as_mapping(value: object) -> Json | None:
    return cast("Json", value) if isinstance(value, Mapping) else None


def _as_list(value: object) -> list[Any] | None:
    return cast("list[Any]", value) if isinstance(value, list) else None


def _view(outputs: Json) -> Json | None:
    return _as_mapping(outputs.get("lc_view"))


def _lender_views(view: Json) -> list[Json]:
    """``[top_lender] + alternates`` (the named lenders), in view order; absent / null entries are skipped."""
    views: list[Json] = []
    top = _as_mapping(view.get("top_lender"))
    if top is not None:
        views.append(top)
    for alternate in _as_list(view.get("alternates")) or []:
        mapped = _as_mapping(alternate)
        if mapped is not None:
            views.append(mapped)
    return views


def _named_ids(view: Json) -> list[str]:
    return [str(v.get("id")) for v in _lender_views(view)]


def _id_list(outputs: Json, key: str) -> list[str] | None:
    raw = _as_list(outputs.get(key))
    return None if raw is None else [str(item) for item in raw]


def _has_figure(text: str) -> bool:
    """True when the text holds a run of three or more ASCII digits (after NFKC, so fullwidth digits count)."""
    folded = unicodedata.normalize("NFKC", text)
    for is_digit, run in itertools.groupby(folded, key=lambda ch: "0" <= ch <= "9"):
        if is_digit and len(list(run)) >= _FIGURE_DIGITS:
            return True
    return False


def _prose_of(view: Json, *, with_reason: bool = True) -> list[tuple[str, str]]:
    """The model-authored strings of a view as ``(path, text)``: rationale / conditions of every named lender and, when
    asked, the co-applicant reason."""
    prose: list[tuple[str, str]] = []
    for index, lender in enumerate(_lender_views(view)):
        rationale = lender.get("rationale")
        if isinstance(rationale, str):
            prose.append((f"lender[{index}].rationale", rationale))
        for item_index, condition in enumerate(_as_list(lender.get("conditions")) or []):
            if isinstance(condition, str):
                prose.append((f"lender[{index}].conditions[{item_index}]", condition))
    reason = view.get("coapplicant_reason")
    if with_reason and isinstance(reason, str):
        prose.append(("coapplicant_reason", reason))
    return prose


# --------------------------------------------------------------------------- 1. no_leak


def _string_leaves(node: object, path: tuple[str | int, ...] = ()) -> Iterator[tuple[tuple[str | int, ...], str]]:
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, Mapping):
        for key, value in cast("Mapping[str, object]", node).items():
            yield from _string_leaves(value, (*path, key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(cast("Sequence[object]", node)):
            yield from _string_leaves(value, (*path, index))


def _is_identifier_field(path: tuple[str | int, ...]) -> bool:
    """``check_id`` at the root, ``top_lender.id`` and ``alternates[i].id``: uuid-ish, never prose."""
    if path == ("check_id",):
        return True
    if len(path) == 2 and path[0] == "top_lender" and path[1] == "id":
        return True
    return len(path) == 3 and path[0] == "alternates" and isinstance(path[1], int) and path[2] == "id"


def _request_literals(inputs: Json) -> list[str]:
    request = _as_mapping(inputs.get("request")) or {}
    literals: list[str] = []
    for field in _REQUEST_IDENTIFIER_FIELDS:
        value = request.get(field)
        if isinstance(value, str) and value.strip():
            literals.append(value.strip().casefold())
    return literals


def no_leak(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
    """No bureau name, standalone 300-899 figure or account identifier in ANY LC-facing string (global tier), no generic
    bureau term / score / account id in the model-authored fields (strict tier), and none of the request's own PAN /
    mobile / typed name anywhere. The label is exempt from the GENERIC term only: 'bureau report on <date>, income as
    declared' is PRD-mandated copy, and the global tier is names-only."""
    del reference_outputs
    view = _view(outputs)
    if view is None:
        return _result("no_leak", ["outputs has no lc_view"])
    literals = _request_literals(inputs)
    problems: list[str] = []
    for path, text in _string_leaves(view):
        if _is_identifier_field(path):
            continue
        dotted = ".".join(str(part) for part in path)
        if line_leaks(text, GLOBAL_LEAK_PATTERNS):
            problems.append(f"{dotted}: global tier")
        if any(isinstance(part, str) and part in LLM_AUTHORED_PATHS for part in path) and line_leaks(
            text, LLM_LEAK_PATTERNS
        ):
            problems.append(f"{dotted}: llm tier")
        folded = text.casefold()
        if any(literal in folded for literal in literals):
            problems.append(f"{dotted}: request identifier")
    return _result("no_leak", problems)


# --------------------------------------------------------------------------- 2. lenders_gated


def lenders_gated(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
    """named lenders <= safety survivors <= gated lenders (never a lender outside the chain), no lender named twice; when
    the golden pins the gate, the gate equals it as a SET."""
    del inputs
    view = _view(outputs)
    gated = _id_list(outputs, "gated_lender_ids")
    survivors = _id_list(outputs, "safety_survivor_ids")
    if view is None or gated is None or survivors is None:
        return _result("lenders_gated", ["outputs lacks lc_view / gated_lender_ids / safety_survivor_ids"])
    named = _named_ids(view)
    problems: list[str] = []
    if len(named) != len(set(named)):
        problems.append(f"a lender is named twice: {named}")
    if not set(named) <= set(survivors):
        problems.append(f"named {sorted(set(named) - set(survivors))} not in the safety survivors {sorted(survivors)}")
    if not set(survivors) <= set(gated):
        problems.append(f"survivors {sorted(set(survivors) - set(gated))} not in the pincode gate {sorted(gated)}")
    if "gated_lender_ids" in reference_outputs:
        expected = {str(item) for item in _as_list(reference_outputs["gated_lender_ids"]) or []}
        if set(gated) != expected:
            problems.append(f"gate {sorted(gated)} != golden {sorted(expected)}")
    return _result("lenders_gated", problems)


# --------------------------------------------------------------------------- 3. numbers_from_code


def _range_of(entry: Json) -> dict[str, Any] | None:
    lower, upper = entry.get("lower"), entry.get("upper")
    return None if lower is None or upper is None else {"lower": lower, "upper": upper}


def _golden_range(entry: Json) -> dict[str, Any] | None:
    value = entry.get("range")
    mapped = _as_mapping(value)
    return None if mapped is None else dict(mapped)


def numbers_from_code(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
    """Every money figure in the view is the code's: for each named lender the ceiling and range equal the run's derived
    table (range None exactly when no income was declared) AND the golden's literal table when it has one (a named
    lender absent from that table fails). A model-failed (K7) view must carry NO numbers at all. No three-digit figure in
    any model-authored prose (rationale / conditions / co-applicant reason)."""
    del inputs
    view = _view(outputs)
    derived = _as_mapping(outputs.get("derived"))
    if view is None or derived is None:
        return _result("numbers_from_code", ["outputs lacks lc_view / derived"])
    golden = _as_mapping(reference_outputs.get("lender_numbers"))
    k7 = view.get("code") == _K7_CODE
    problems: list[str] = []
    for lender in _lender_views(view):
        lender_id = str(lender.get("id"))
        ceiling, shown_range = lender.get("ceiling"), lender.get("range")
        if k7:
            if ceiling is not None or shown_range is not None:
                problems.append(f"{lender_id}: a model-failed view must be number-free")
        else:
            row = _as_mapping(derived.get(lender_id))
            if row is None:
                problems.append(f"{lender_id}: no code-derived numbers for a named lender")
            else:
                if ceiling != row.get("ceiling"):
                    problems.append(f"{lender_id}: ceiling {ceiling} != derived {row.get('ceiling')}")
                if shown_range != _range_of(row):
                    problems.append(f"{lender_id}: range {shown_range} != derived {_range_of(row)}")
        if golden is not None:
            entry = _as_mapping(golden.get(lender_id))
            if entry is None:
                problems.append(f"{lender_id}: named but absent from the golden's lender_numbers")
            else:
                if ceiling != entry.get("ceiling"):
                    problems.append(f"{lender_id}: ceiling {ceiling} != golden {entry.get('ceiling')}")
                if shown_range != _golden_range(entry):
                    problems.append(f"{lender_id}: range {shown_range} != golden {_golden_range(entry)}")
    problems.extend(f"{path}: a figure in model prose" for path, text in _prose_of(view) if _has_figure(text))
    return _result("numbers_from_code", problems)


# --------------------------------------------------------------------------- 4. confidence_not_raised


def confidence_not_raised(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
    """The shown confidence never exceeds the code's (low < medium < high). A view without a confidence (refusals, PAN
    not verified, the band read) passes; a confidence with no code value to bound it fails closed."""
    del inputs, reference_outputs
    view = _view(outputs)
    if view is None:
        return _result("confidence_not_raised", ["outputs has no lc_view"])
    shown = view.get("confidence")
    if shown is None:
        return _result("confidence_not_raised", [])
    code = outputs.get("code_confidence")
    if shown not in _CONFIDENCE_ORDER or code not in _CONFIDENCE_ORDER:
        return _result("confidence_not_raised", [f"shown {shown!r} has no valid code confidence ({code!r}) to bound it"])
    if _CONFIDENCE_ORDER[str(shown)] > _CONFIDENCE_ORDER[str(code)]:
        return _result("confidence_not_raised", [f"shown {shown} is above the code's {code}"])
    return _result("confidence_not_raised", [])


# --------------------------------------------------------------------------- 5. billing_consistency


def billing_consistency(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
    """K17, billable-based: ``billed == any(billable)`` over this check's calls; a cache hit calls no bureau endpoint; a
    refusal calls nothing and bills nothing; counting needs a billed NEW admission and a refresh never counts; the
    outputs' ``from_cache`` agrees with the view's. The golden's billed / counted / from_cache / admission match when
    present and its endpoints match as a MULTISET: call order is never compared (uuid-random under a frozen clock)."""
    del inputs
    view = _view(outputs)
    raw_calls = _as_list(outputs.get("surepass_calls"))
    if view is None or raw_calls is None:
        return _result("billing_consistency", ["outputs lacks lc_view / surepass_calls"])
    calls = [_as_mapping(call) for call in raw_calls]
    if any(call is None for call in calls):
        return _result("billing_consistency", ["a surepass_calls entry is not an {endpoint, billable} mapping"])
    entries = [call for call in calls if call is not None]
    endpoints = [str(call.get("endpoint")) for call in entries]
    billed = bool(outputs.get("billed"))
    counted = bool(outputs.get("counted_against_quota"))
    from_cache = bool(outputs.get("from_cache"))
    admission = outputs.get("admission")
    problems: list[str] = []

    if billed != any(call.get("billable") is True for call in entries):
        problems.append(f"billed={billed} but billable flags are {[call.get('billable') for call in entries]}")
    if from_cache != bool(view.get("from_cache")):
        problems.append(f"outputs from_cache={from_cache} but the view says {view.get('from_cache')}")
    if from_cache and any(endpoint in BUREAU_ENDPOINTS for endpoint in endpoints):
        problems.append("a cache hit called a bureau endpoint")
    if view.get("status") == "refused" and (entries or billed or counted):
        problems.append("a refusal made calls or billed / counted")
    if counted and not (billed and admission == "new"):
        problems.append(f"counted with billed={billed}, admission={admission!r}")
    if admission == "refresh" and counted:
        problems.append("a refresh was counted")

    for key, actual in (("billed", billed), ("counted_against_quota", counted), ("from_cache", from_cache)):
        if key in reference_outputs and actual != reference_outputs[key]:
            problems.append(f"{key}={actual} != golden {reference_outputs[key]}")
    if "admission" in reference_outputs and admission != reference_outputs["admission"]:
        problems.append(f"admission={admission!r} != golden {reference_outputs['admission']!r}")
    if "surepass_endpoints" in reference_outputs:
        expected = Counter(str(item) for item in _as_list(reference_outputs["surepass_endpoints"]) or [])
        if Counter(endpoints) != expected:
            problems.append(f"endpoints {sorted(Counter(endpoints).elements())} != golden {sorted(expected.elements())}")
    return _result("billing_consistency", problems)


# --------------------------------------------------------------------------- 6. outcome_matches_band


def outcome_matches_band(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
    """The outcome the LC sees fits the band, and fits the golden.

    Band rules (PRD 9): a thin band (NTC, below 650) never reads 'fundable' and either names no lender or suggests a
    co-applicant; 'not_fundable_alone' always suggests one. Golden pins (when present): status, code, headline,
    confidence, co-applicant flag, top lender id, alternate ids, label, confirm items, Aadhaar warning, quick checks left,
    both dates and the band itself; ``headline_in`` (a list) pins a headline the MODEL decides between legal alternatives
    for (e.g. 'not_fundable_alone' when a floor-ignoring pick is demoted, 'under_review' when the model names nobody).
    This is where the v1.1 NTC reversal is pinned at eval level (q05)."""
    del inputs
    view = _view(outputs)
    if view is None:
        return _result("outcome_matches_band", ["outputs has no lc_view"])
    band = outputs.get("band")
    headline = view.get("headline")
    suggested = view.get("coapplicant_suggested") is True
    problems: list[str] = []
    if band in _THIN_BANDS:
        if _as_mapping(view.get("top_lender")) is not None and not suggested:
            problems.append(f"band {band} names a top lender without suggesting a co-applicant")
        if headline == "fundable":
            problems.append(f"band {band} cannot be 'fundable'")
    if headline == "not_fundable_alone" and not suggested:
        problems.append("'not_fundable_alone' without a co-applicant suggestion")

    for key in _VIEW_PINS:
        if key in reference_outputs and view.get(key) != reference_outputs[key]:
            problems.append(f"{key}: got {view.get(key)!r}, golden {reference_outputs[key]!r}")
    if "headline_in" in reference_outputs:
        allowed = [str(h) for h in _as_list(reference_outputs["headline_in"]) or []]
        if headline not in allowed:
            problems.append(f"headline: got {headline!r}, golden one of {allowed}")
    if "top_lender_id" in reference_outputs:
        top = _as_mapping(view.get("top_lender"))
        top_id = None if top is None else top.get("id")
        if top_id != reference_outputs["top_lender_id"]:
            problems.append(f"top_lender_id: got {top_id!r}, golden {reference_outputs['top_lender_id']!r}")
    if "alternate_ids" in reference_outputs:
        alternates = [str(a.get("id")) for a in (_as_mapping(x) or {} for x in _as_list(view.get("alternates")) or [])]
        if alternates != list(_as_list(reference_outputs["alternate_ids"]) or []):
            problems.append(f"alternate_ids: got {alternates}, golden {reference_outputs['alternate_ids']}")
    if "band" in reference_outputs and band != reference_outputs["band"]:
        problems.append(f"band: got {band!r}, golden {reference_outputs['band']!r}")
    return _result("outcome_matches_band", problems)


DETERMINISTIC_EVALUATORS: Final[tuple[Evaluator, ...]] = (
    no_leak,
    lenders_gated,
    numbers_from_code,
    confidence_not_raised,
    billing_consistency,
    outcome_matches_band,
)


# --------------------------------------------------------------------------- 7. the LLM judge (gated on OPENAI_API_KEY)

JUDGE_KEY: Final[str] = "plain_language_judge"
JUDGE_MODEL: Final[str] = "gpt-4.1"
JUDGE_TIMEOUT_S: Final[float] = 8.0
JUDGE_RUBRIC: Final[str] = (
    "You are reviewing short notes shown to a loan counsellor about a learner's financing options. "
    "Score 1 only if EVERY note below is: (a) plain language a non-specialist understands; (b) free of credit scores and "
    "bureau jargon, describing the credit profile in words only (for example 'strong', 'thin'); (c) actionable, saying "
    "what to confirm or do next; (d) free of any promise or guarantee of approval or of a loan amount; (e) sensible for "
    "an Indian education-financing context (INR, EMI, co-applicant). Otherwise score 0. "
    "Reply with a score (0 or 1) and one or two sentences of reasons."
)


class JudgeVerdict(BaseModel):
    score: int = Field(ge=0, le=1)
    reasons: str


JudgeFn: TypeAlias = Callable[[str], JudgeVerdict]


def make_llm_judge(judge: JudgeFn) -> Evaluator:
    """The judge evaluator over any ``prompt -> JudgeVerdict`` callable (the real chat model in production, a stub in
    tests). It judges ONLY the model-authored prose; a view with none (a refusal, the code-built K7 view) passes without a
    model call."""

    def plain_language_judge(inputs: Json, outputs: Json, reference_outputs: Json) -> EvalResult:
        del inputs, reference_outputs
        view = _view(outputs)
        if view is None:
            return {"key": JUDGE_KEY, "score": False, "comment": "outputs has no lc_view"}
        prose = [] if view.get("code") == _K7_CODE else [text for _, text in _prose_of(view) if text.strip()]
        if not prose:
            return {"key": JUDGE_KEY, "score": True, "comment": "no model prose to judge"}
        verdict = judge(JUDGE_RUBRIC + "\n\nNotes:\n" + "\n".join(f"- {text}" for text in prose))
        return {"key": JUDGE_KEY, "score": verdict.score == 1, "comment": verdict.reasons}

    plain_language_judge.__name__ = JUDGE_KEY
    return plain_language_judge


def _chat_judge(api_key: str) -> JudgeFn:
    chat = ChatOpenAI(model=JUDGE_MODEL, temperature=0, timeout=JUDGE_TIMEOUT_S, max_retries=0, api_key=SecretStr(api_key))
    runnable = cast(
        "Runnable[Any, Any]",
        chat.with_structured_output(JudgeVerdict, method="function_calling"),  # pyright: ignore[reportUnknownMemberType]
    )

    def judge(prompt: str) -> JudgeVerdict:
        verdict: object = runnable.invoke(prompt)
        if not isinstance(verdict, JudgeVerdict):
            raise TypeError(f"judge returned {type(verdict).__name__}, expected JudgeVerdict")
        return verdict

    return judge


def judge_evaluators(environ: Mapping[str, str]) -> list[Evaluator]:
    """``[plain_language_judge]`` iff ``OPENAI_API_KEY`` is set and non-blank, else ``[]``: the model is constructed only
    behind this gate (CI secrets gate), so local / CI runs never see it."""
    key = environ.get("OPENAI_API_KEY", "").strip()
    return [make_llm_judge(_chat_judge(key))] if key else []
