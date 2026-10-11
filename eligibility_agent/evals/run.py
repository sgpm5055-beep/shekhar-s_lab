"""Eval runner: ``run_one`` (one golden through the REAL graph), the local runner and the LangSmith runner.

    uv run python -m eligibility_agent.evals.run --mode local --llm fake --dataset both            # CI: exit 0, no network
    uv run python -m eligibility_agent.evals.run --mode langsmith --llm real --max-real 20 \\
        --dataset both --build-datasets --experiment-prefix ci-<sha>                                # secrets-gated

Run it from the repository root: the goldens reuse ``tests.helpers`` (the shared synthetic world, the K17 seed lenders),
which is importable there and is NOT part of the installed wheel.

``run_one(inputs, llm)`` builds a fresh world per example (a FrozenClock at EVAL_T0 = 2026-10-10 10:00 IST, a MemoryRepo on
that clock with the example's seed lenders published, a ``FakeSurePassClient`` per the example's scenario spec, the rank
model), applies the pre-state steps, submits the request exactly as the route does (the route's request models, hashes and
``new_check_state``) to the compiled graph, and returns the outputs contract that ``evals.evaluators`` documents. Facts
come from the persisted check row (billed / counted / internal), the repo's call log (``surepass_calls``) and the final
graph state; nothing is recomputed. Tracing is switched off around a local run (``tracing_context(enabled=False)``), so no env
var in a developer's shell can make a local run export anything; the langsmith-mode target runs ``traced=True`` so the
graph's node runs nest under the experiment row (the data is synthetic).

Gating (exit 2 = a usage / environment error, exit 1 = an evaluator failed, 0 = all green):

* ``--mode langsmith`` needs ``LANGSMITH_API_KEY``; ``--llm real`` needs ``OPENAI_API_KEY`` (either mode);
* the LLM judge (the 7th evaluator) exists only in langsmith mode and only when ``OPENAI_API_KEY`` is set: local / CI runs
  never construct a model;
* with ``--llm real`` the first ``--max-real`` examples (default 20, quick then full) run on the real model, the rest on
  the fake; a golden that scripts the model (``rank == "invalid_twice"``) always uses the fake.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Literal, TypeAlias, cast

from langsmith import Client
from langsmith import tracing_context  # pyright: ignore[reportUnknownVariableType]

from eligibility_agent.config import Settings
from eligibility_agent.db.repo import MemoryRepo
from eligibility_agent.evals.datasets import (
    DatasetChoice,
    DatasetsClient,
    GoldenExample,
    GoldenInputs,
    build_langsmith_datasets,
    dataset_names,
    inputs_of,
    selected_examples,
)
from eligibility_agent.evals.evaluators import DETERMINISTIC_EVALUATORS, EvalResult, Evaluator, judge_evaluators
from eligibility_agent.evals.worlds import build_surepass
from eligibility_agent.graph.build import CompiledGraph, GraphDeps, build_graph, new_check_state
from eligibility_agent.graph.llm import RankModel, build_rank_context, make_rank_model
from eligibility_agent.graph.prompts import local_prompts
from eligibility_agent.graph.state import CheckState, Kind, LcView
from eligibility_agent.graph.tracing import invoke_config
from eligibility_agent.main import FullCheckRequest, QuickCheckRequest
from eligibility_agent.policy.guards import hash_mobile, hash_pan
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.fake_llm import FakeRankModel
from tests.helpers import (
    HMAC_KEY,
    T0,
    seed_cached_quick_world,
    seed_counted_quick_checks,
    seed_lender,
    seed_lender_row,
)

__all__ = [
    "EVAL_T0",
    "EvalConfigError",
    "ExampleOutcome",
    "LlmMode",
    "main",
    "run_langsmith",
    "run_local",
    "run_one",
]

EVAL_T0: Final = T0  # 2026-10-10 10:00 IST: the one epoch every golden's dates are written against
DEFAULT_MAX_REAL: Final[int] = 20
LANGSMITH_CONCURRENCY: Final[int] = 4
EXIT_OK: Final[int] = 0
EXIT_EVAL_FAILED: Final[int] = 1
EXIT_USAGE: Final[int] = 2

LlmMode: TypeAlias = Literal["fake", "real"]
Mode: TypeAlias = Literal["local", "langsmith"]
Write: TypeAlias = Callable[[str], None]
EvaluateFn: TypeAlias = Callable[..., object]


class EvalConfigError(RuntimeError):
    """A usage / environment problem (a missing key, a flag combination that cannot run): exit 2, never a golden failure."""


# --------------------------------------------------------------------------- one run


@dataclass
class _World:
    clock: FrozenClock
    repo: MemoryRepo
    deps: GraphDeps
    graphs: dict[Kind, CompiledGraph]


def _openai_model() -> str:
    default: object = Settings.model_fields["openai_model"].default
    return os.environ.get("OPENAI_MODEL") or str(default)


def _rank_model(rank: str, llm: LlmMode) -> RankModel:
    if rank == "invalid_twice":
        return FakeRankModel.invalid_twice()  # the K7 golden: scripted, whatever model the run is configured for
    if llm == "fake":
        return FakeRankModel()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise EvalConfigError("--llm real needs OPENAI_API_KEY")
    return make_rank_model(model=_openai_model(), prompts=local_prompts(), api_key=key)


def _build_world(inputs: GoldenInputs, llm: LlmMode) -> _World:
    clock = FrozenClock(EVAL_T0)
    repo = MemoryRepo(clock.now)
    for lender_id in inputs["lenders"]:
        seed_lender_row(repo, seed_lender(lender_id))
    deps = GraphDeps(
        repo=repo,
        surepass=build_surepass(inputs["surepass"], repo),
        rank_model=_rank_model(inputs["rank"], llm),
        prompts=local_prompts(),
        now=clock.now,
    )
    graphs: dict[Kind, CompiledGraph] = {"quick": build_graph("quick", deps), "full": build_graph("full", deps)}
    return _World(clock=clock, repo=repo, deps=deps, graphs=graphs)


def _submit(world: _World, kind: Kind, request: Mapping[str, Any]) -> Mapping[str, Any]:
    """The route's own seeding (request model -> hashes -> ``new_check_state``) over the compiled graph: the final state."""
    body = QuickCheckRequest.model_validate(request) if kind == "quick" else FullCheckRequest.model_validate(request)
    full = body if isinstance(body, FullCheckRequest) else None
    check_id = str(uuid.uuid4())
    pan_hash = hash_pan(body.pan, key=HMAC_KEY)
    state = new_check_state(
        kind=kind,
        check_id=check_id,
        lc_id=body.lc_id,
        lead_id=body.lead_id,
        opportunity_id=body.opportunity_id,
        opportunity_stage=full.opportunity_stage.value if full is not None else None,
        pan=body.pan,
        name=body.name,
        mobile=body.mobile,
        pan_hash=pan_hash,
        mobile_hash=hash_mobile(body.mobile, key=HMAC_KEY),
        monthly_income=body.monthly_income,
        existing_emi_declared=body.existing_emi,
        refresh=full.refresh if full is not None else False,
        acknowledge_ntc_warning=full.acknowledge_ntc_warning if full is not None else False,
        consent=body.consent.info(),
        langsmith_run_id=None,
    )
    config = invoke_config(kind=kind, check_id=check_id, lc_id=body.lc_id, pan_hash=pan_hash, run_id=None, tracer=None)
    final: object = world.graphs[kind].invoke(state, config)  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(final, Mapping):
        raise TypeError(f"graph returned {type(final).__name__}, expected a state mapping")
    return cast("Mapping[str, Any]", final)


def _apply_step(world: _World, step: Mapping[str, Any]) -> None:
    op = step["op"]
    now = world.clock.now()
    if op == "seed_counted_quick_checks":
        seed_counted_quick_checks(
            world.repo,
            lc_id=str(step["lc_id"]),
            created_at=now - timedelta(hours=int(step["hours_before"])),
            n=int(step["n"]),
        )
    elif op == "seed_cached_quick_world":
        seed_cached_quick_world(world.repo, pan=str(step["pan"]), fetched_at=now - timedelta(days=int(step["days_before"])))
    elif op == "run_check":
        kind: Kind = "full" if step["kind"] == "full" else "quick"
        _submit(world, kind, cast("Mapping[str, Any]", step["request"]))
    elif op == "advance_days":
        world.clock.advance(timedelta(days=int(step["days"])))
    else:
        raise ValueError(f"unknown pre_state op {op!r}")


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast("float", value))


def _derived_table(internal: Mapping[str, Any]) -> dict[str, dict[str, int | None]]:
    """Per gated lender, the code's display numbers (post-rounding ints) from the persisted ``lender_numbers``."""
    table: dict[str, dict[str, int | None]] = {}
    for lender_id, raw in _mapping(internal.get("lender_numbers")).items():
        numbers = _mapping(raw)
        table[str(lender_id)] = {
            "ceiling": _optional_int(numbers.get("ceiling")),
            "lower": _optional_int(numbers.get("loan_max_lower")),
            "upper": _optional_int(numbers.get("loan_max_upper")),
        }
    return table


def _survivor_ids(internal: Mapping[str, Any], final: Mapping[str, Any], view: LcView) -> list[str]:
    """The lenders the view may name. Safety ran: its survivors (top then alternates). Safety did not run because the
    ranking model failed (K7): the code-built eligible set, read from the rank context the model itself was given (the
    gate survivors whose band clears their floor). Anything else names nobody."""
    safety = _mapping(internal.get("safety"))
    if safety:
        picks = [safety.get("top"), *cast("list[Any]", safety.get("alternates") or [])]
        return [str(_mapping(pick).get("lender_id")) for pick in picks if _mapping(pick)]
    if view.code == "OPENAI_FAILURE":
        state = cast("CheckState", dict(final))
        try:
            context = build_rank_context(state, state.get("gated_lenders", []))
        except ValueError:
            return []
        return [lender.lender_id for lender in context.lenders if lender.band_clears_floor is not False]
    return []


def _admission(internal: Mapping[str, Any]) -> str:
    decision = _mapping(internal.get("admission"))
    if decision.get("code") is not None:
        return "refused"
    return str(decision.get("admission") or "none")


def _outputs_contract(world: _World, final: Mapping[str, Any]) -> dict[str, Any]:
    raw_view: object = final.get("lc_view")
    if not isinstance(raw_view, LcView):
        raise TypeError("the graph returned no LcView")
    row = world.repo.get_check(raw_view.check_id)
    if row is None:
        raise LookupError(f"no persisted check row for {raw_view.check_id}")
    internal = row.internal or {}
    derived = _mapping(internal.get("derived"))
    band = derived.get("band")
    code_confidence = internal.get("code_confidence")
    admission = _admission(internal)
    return {
        # What the LC received, read back from the persisted row; ``from_cache`` comes from the admission decision the
        # row recorded, an independent source, so the evaluator's cross-check against the view's own flag is not vacuous.
        "lc_view": row.lc_view if row.lc_view is not None else raw_view.model_dump(mode="json"),
        "billed": row.billed,
        "counted_against_quota": row.counted_against_quota,
        "from_cache": admission == "cache_hit",
        "admission": admission,
        "surepass_calls": [
            {"endpoint": call.endpoint, "billable": call.billable} for call in world.repo.list_calls(raw_view.check_id)
        ],
        "gated_lender_ids": [str(i) for i in cast("list[object]", internal.get("gated_lender_ids") or [])],
        "safety_survivor_ids": _survivor_ids(internal, final, raw_view),
        "band": None if band is None else str(band),
        "code_confidence": None if code_confidence is None else str(code_confidence),
        "derived": _derived_table(internal),
    }


def run_one(inputs: GoldenInputs, llm: LlmMode = "fake", *, traced: bool = False) -> dict[str, Any]:
    """Run one golden's inputs through the real graph and return the evaluator outputs contract (JSON-safe).

    Tracing is OFF by default (local mode: nothing may be exported whatever a developer's shell says). ``traced=True`` is
    the langsmith-mode target's opt-in: it leaves the caller's tracing context alone, so the graph's node runs nest under
    the experiment row."""
    with nullcontext() if traced else tracing_context(enabled=False):
        world = _build_world(inputs, llm)
        for step in inputs["pre_state"]:
            _apply_step(world, step)
        final = _submit(world, inputs["kind"], inputs["request"])
        return _outputs_contract(world, final)


# --------------------------------------------------------------------------- the local runner


@dataclass(frozen=True)
class ExampleOutcome:
    dataset: str
    example_id: str
    results: tuple[EvalResult, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(result["score"] for result in self.results)


def _evaluate_example(
    dataset: str, example: GoldenExample, llm: LlmMode, evaluators: Sequence[Evaluator]
) -> ExampleOutcome:
    inputs = inputs_of(example)
    try:
        outputs = run_one(inputs, llm)
    except EvalConfigError:
        raise
    except Exception as exc:  # a crashing golden is a failed golden, not an aborted run
        return ExampleOutcome(dataset, example["example_id"], (), error=f"{type(exc).__name__}: {exc}")
    results = tuple(evaluator(cast("Mapping[str, Any]", inputs), outputs, example["expected"]) for evaluator in evaluators)
    return ExampleOutcome(dataset, example["example_id"], results)


def _render(outcomes: Sequence[ExampleOutcome], keys: Sequence[str]) -> list[str]:
    width = max((len(o.example_id) for o in outcomes), default=10)
    header = f"{'example':<{width}}  " + "  ".join(f"{key:<{max(len(key), 4)}}" for key in keys)
    lines = [header, "-" * len(header)]
    for outcome in outcomes:
        if outcome.error is not None:
            lines.append(f"{outcome.example_id:<{width}}  ERROR  {outcome.error}")
            continue
        cells = "  ".join(f"{('ok' if r['score'] else 'FAIL'):<{max(len(r['key']), 4)}}" for r in outcome.results)
        lines.append(f"{outcome.example_id:<{width}}  {cells}")
    failures = [
        f"  {o.example_id}: {r['key']}: {r['comment']}" for o in outcomes for r in o.results if not r["score"]
    ]
    if failures:
        lines += ["", "failures:", *failures]
    return lines


def run_local(
    choice: DatasetChoice = "both",
    llm: LlmMode = "fake",
    *,
    max_real: int = DEFAULT_MAX_REAL,
    write: Write = print,
    evaluators: Sequence[Evaluator] = DETERMINISTIC_EVALUATORS,
) -> int:
    """No LangSmith, no judge: run every selected golden, apply the deterministic evaluators, print a table, and return 1
    when anything failed or crashed, else 0."""
    plan = _llm_plan(choice, llm, max_real=max_real)
    outcomes = [
        _evaluate_example(name, example, plan[example["example_id"]], evaluators)
        for name in dataset_names(choice)
        for example in selected_examples(name)
    ]
    write("\n".join(_render(outcomes, [e.__name__ for e in evaluators])))
    failed = [o for o in outcomes if not o.passed]
    write(f"\n{len(outcomes) - len(failed)}/{len(outcomes)} examples passed all {len(evaluators)} evaluators")
    return EXIT_EVAL_FAILED if failed else EXIT_OK


# --------------------------------------------------------------------------- the LangSmith runner


def _llm_plan(choice: DatasetChoice, llm: LlmMode, *, max_real: int) -> dict[str, LlmMode]:
    """example id -> model: with ``--llm real`` the first ``max_real`` examples (quick, then full) are real, the rest fake."""
    plan: dict[str, LlmMode] = {}
    for name in dataset_names(choice):
        for example in selected_examples(name):
            plan[example["example_id"]] = "real" if llm == "real" and len(plan) < max_real else "fake"
    return plan


def run_langsmith(
    choice: DatasetChoice,
    llm: LlmMode,
    *,
    max_real: int,
    experiment_prefix: str,
    build_datasets: bool,
    environ: Mapping[str, str] = os.environ,
    client: DatasetsClient | None = None,
    evaluate_fn: EvaluateFn | None = None,
    write: Write = print,
) -> int:
    """``langsmith.evaluate`` over the uploaded goldens: the six deterministic evaluators plus the judge when
    ``OPENAI_API_KEY`` is set. The target runs ``run_one`` with the model the plan assigns to that example.
    ``evaluate_fn`` defaults to ``langsmith.evaluate`` (imported lazily: local mode never loads the evaluation stack)."""
    if not environ.get("LANGSMITH_API_KEY", "").strip():
        raise EvalConfigError("--mode langsmith needs LANGSMITH_API_KEY")
    if evaluate_fn is None:
        from langsmith import evaluate as default_evaluate  # pyright: ignore[reportUnknownVariableType]

        evaluate_fn = cast("EvaluateFn", default_evaluate)
    ls_client: Any = client if client is not None else Client()
    if build_datasets:
        uploaded = build_langsmith_datasets(cast("DatasetsClient", ls_client))
        write("uploaded examples: " + ", ".join(f"{name}={count}" for name, count in uploaded.items()))
    plan = _llm_plan(choice, llm, max_real=max_real)
    evaluators: list[Evaluator] = [*DETERMINISTIC_EVALUATORS, *judge_evaluators(environ)]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        golden = cast("GoldenInputs", inputs)
        return run_one(golden, plan.get(golden["example_id"], "fake"), traced=True)

    metadata = {"llm": llm, "max_real": max_real, "git_sha": environ.get("GITHUB_SHA", "")}
    for name in dataset_names(choice):
        evaluate_fn(
            target,
            data=name,
            evaluators=evaluators,
            max_concurrency=LANGSMITH_CONCURRENCY,
            experiment_prefix=f"{experiment_prefix}-{name}",
            metadata=metadata,
            client=ls_client,
        )
        write(f"evaluated {name} ({len(selected_examples(name))} examples, {len(evaluators)} evaluators)")
    return EXIT_OK


# --------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eligibility_agent.evals.run", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mode", choices=("local", "langsmith"), required=True)
    parser.add_argument("--llm", choices=("fake", "real"), default="fake")
    parser.add_argument("--dataset", choices=("quick", "full", "both"), default="both")
    parser.add_argument("--max-real", type=int, default=DEFAULT_MAX_REAL, dest="max_real")
    parser.add_argument("--build-datasets", action="store_true", dest="build_datasets")
    parser.add_argument("--experiment-prefix", default="eligibility", dest="experiment_prefix")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] = os.environ, write: Write = print) -> int:
    args = _parser().parse_args(argv)
    mode = cast("Mode", args.mode)
    llm = cast("LlmMode", args.llm)
    choice = cast("DatasetChoice", args.dataset)
    try:
        if args.max_real < 0:
            raise EvalConfigError("--max-real must be >= 0")
        if args.build_datasets and mode != "langsmith":
            raise EvalConfigError("--build-datasets needs --mode langsmith")
        if llm == "real" and not environ.get("OPENAI_API_KEY", "").strip():
            raise EvalConfigError("--llm real needs OPENAI_API_KEY")
        if mode == "local":
            return run_local(choice, llm, max_real=args.max_real, write=write)
        return run_langsmith(
            choice,
            llm,
            max_real=args.max_real,
            experiment_prefix=args.experiment_prefix,
            build_datasets=args.build_datasets,
            environ=environ,
            write=write,
        )
    except EvalConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
