"""Task 15: the eval runner (``evals/run.py``): ``run_one`` through the real graph, the local runner, the gating rules and
the LangSmith mode (against stubs: no network, no key).

The acceptance gate is ``python -m eligibility_agent.evals.run --mode local --llm fake --dataset both`` exiting 0
(``test_the_local_run_of_both_datasets_exits_zero``, the ``eval`` marker). The rest pins what makes that exit code
trustworthy: every pin in every golden is LIVE (a mutated expectation fails the run), the q05 golden fails a v1.0-style
NTC reading, local mode builds no judge and exports no trace, and langsmith mode is gated on its keys.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
from collections.abc import Callable, Mapping
from typing import Any, Final, cast

import pytest
from eval_builders import RecordingClient
from langsmith import utils as ls_utils

from eligibility_agent.evals import datasets, evaluators, run
from eligibility_agent.evals.datasets import (
    FULL_DATASET,
    QUICK_DATASET,
    DatasetChoice,
    GoldenExample,
    GoldenInputs,
    full_examples,
    inputs_of,
    quick_examples,
)
from eligibility_agent.evals.evaluators import DETERMINISTIC_EVALUATORS, Evaluator
from eligibility_agent.evals.run import LlmMode, main, run_langsmith, run_local, run_one
from eligibility_agent.graph.llm import RankContext
from eligibility_agent.graph.state import ModelOutput
from eligibility_agent.testing.fake_llm import FakeRankModel

ALL: Final[list[GoldenExample]] = [*quick_examples(), *full_examples()]
BY_ID: Final[dict[str, GoldenExample]] = {e["example_id"]: e for e in ALL}
KEYS: Final[dict[str, str]] = {"OPENAI_API_KEY": "sk-synthetic", "LANGSMITH_API_KEY": "ls-synthetic"}


@functools.cache
def outputs_for(example_id: str) -> str:
    """run_one is deterministic per example (frozen clock); cache its JSON so the mutation sweep runs each golden once."""
    return json.dumps(run_one(inputs_of(BY_ID[example_id])))


def outputs_of(example_id: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(outputs_for(example_id)))


def failing_evaluators(example: GoldenExample, outputs: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    inputs = cast("Mapping[str, Any]", inputs_of(example))
    return [e.__name__ for e in DETERMINISTIC_EVALUATORS if not e(inputs, outputs, expected)["score"]]


# --------------------------------------------------------------------------- the acceptance gate


@pytest.mark.eval
def test_the_local_run_of_both_datasets_exits_zero() -> None:
    lines: list[str] = []

    code = main(["--mode", "local", "--llm", "fake", "--dataset", "both"], environ={}, write=lines.append)

    assert code == 0, "\n".join(lines)
    text = "\n".join(lines)
    assert "22/22 examples passed all 6 evaluators" in text
    assert "FAIL" not in text and "ERROR" not in text
    for example in ALL:
        assert example["example_id"] in text


@pytest.mark.eval
@pytest.mark.parametrize("example", ALL, ids=lambda e: e["example_id"])
def test_every_golden_passes_all_six_evaluators(example: GoldenExample) -> None:
    outputs = outputs_of(example["example_id"])
    results = [e(cast("Mapping[str, Any]", inputs_of(example)), outputs, example["expected"]) for e in DETERMINISTIC_EVALUATORS]
    assert [r["comment"] for r in results if not r["score"]] == []


def test_dataset_selection_runs_only_what_was_asked() -> None:
    plan: list[tuple[DatasetChoice, int]] = [("quick", 12), ("full", 10), ("both", 22)]
    for choice, count in plan:
        lines: list[str] = []
        assert run_local(choice, "fake", write=lines.append) == 0
        assert f"{count}/{count} examples passed" in "\n".join(lines)


# --------------------------------------------------------------------------- run_one smoke (the outputs contract)


def test_run_one_returns_the_outputs_contract_for_the_happy_quick_golden() -> None:
    outputs = run_one(inputs_of(BY_ID["q01_happy_745"]))

    assert set(outputs) == {
        "lc_view", "billed", "counted_against_quota", "from_cache", "admission", "surepass_calls",
        "gated_lender_ids", "safety_survivor_ids", "band", "code_confidence", "derived",
    }  # fmt: skip
    assert (outputs["billed"], outputs["counted_against_quota"], outputs["from_cache"], outputs["admission"]) == (
        True, True, False, "new",
    )  # fmt: skip
    assert sorted(c["endpoint"] for c in outputs["surepass_calls"]) == sorted(
        ["/api/v1/pan/pan-comprehensive", "/api/v1/credit-report-experian/score"]
    )
    assert all(c["billable"] is True for c in outputs["surepass_calls"])
    assert (outputs["gated_lender_ids"], outputs["safety_survivor_ids"]) == (["L1", "L2"], ["L1", "L2"])
    assert (outputs["band"], outputs["code_confidence"]) == ("720-780", "medium")
    assert outputs["derived"] == {  # hand-written dossier values (display ints)
        "L1": {"ceiling": 1277000, "lower": 368000, "upper": 1277000},
        "L2": {"ceiling": 500000, "lower": 295000, "upper": 500000},
    }
    view = outputs["lc_view"]
    assert (view["status"], view["headline"], view["confidence"]) == ("ok", "possible", "medium")
    assert view["top_lender"]["id"] == "L1" and view["top_lender"]["ceiling"] == 1277000
    assert failing_evaluators(BY_ID["q01_happy_745"], outputs, BY_ID["q01_happy_745"]["expected"]) == []


def test_run_one_on_the_q05_golden_reads_billed_from_the_billable_flags_not_the_call_count() -> None:
    """success:false on the Score: the PAN call billed (200 + success), the Score call did not. billed = any(billable)."""
    outputs = run_one(inputs_of(BY_ID["q05_score_success_false"]))

    calls = {c["endpoint"]: c["billable"] for c in outputs["surepass_calls"]}
    assert calls == {"/api/v1/pan/pan-comprehensive": True, "/api/v1/credit-report-experian/score": False}
    assert (outputs["billed"], outputs["counted_against_quota"]) == (True, False)
    view = outputs["lc_view"]
    assert (view["status"], view["code"], view["headline"], view["top_lender"]) == (
        "service_failure", "SUREPASS_FAILURE", None, None,
    )  # fmt: skip
    assert (outputs["band"], outputs["code_confidence"], outputs["gated_lender_ids"]) == (None, None, [])
    assert failing_evaluators(BY_ID["q05_score_success_false"], outputs, BY_ID["q05_score_success_false"]["expected"]) == []


def test_run_one_k7_survivors_are_the_code_built_floor_clearing_gate_survivors() -> None:
    """Safety never ran (the model failed twice), yet the view names L1 and L2: the contract's survivors come from the
    rank context the model was given (gate survivors whose band clears their floor), so ``lenders_gated`` is not vacuous."""
    outputs = run_one(inputs_of(BY_ID["q04_openai_invalid_twice"]))

    assert [a["id"] for a in outputs["lc_view"]["alternates"]] == ["L1", "L2"]
    assert outputs["safety_survivor_ids"] == ["L1", "L2"]
    assert outputs["lc_view"]["top_lender"] is None
    assert all(a["ceiling"] is None and a["range"] is None for a in outputs["lc_view"]["alternates"])
    assert outputs["derived"]["L1"]["ceiling"] == 1277000  # the code derived numbers; the view correctly shows none


def test_run_one_k7_makes_exactly_two_rank_calls_with_the_retry_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """q04: the scripted model fails twice; the node calls it EXACTLY twice (one retry, never a third) and the second
    call carries the first call's error. A spy over the model the runner actually builds."""
    built: list[FakeRankModel] = []
    original = run._rank_model

    def spying(rank: str, llm: LlmMode) -> Any:
        model = original(rank, llm)
        assert isinstance(model, FakeRankModel)
        built.append(model)
        return model

    monkeypatch.setattr(run, "_rank_model", spying)

    outputs = run_one(inputs_of(BY_ID["q04_openai_invalid_twice"]))

    assert len(built) == 1
    model = built[0]
    assert model.call_count == 2
    assert [ctx.attempt for ctx in model.contexts] == [1, 2]
    assert model.contexts[0].prior_error is None and model.contexts[1].prior_error
    assert outputs["lc_view"]["code"] == "OPENAI_FAILURE"


def test_run_one_reads_the_view_and_from_cache_from_the_persisted_row_not_the_state_view() -> None:
    """The evaluator cross-checks ``from_cache`` against the view's own flag; that is only meaningful when the two come from
    different sources. ``lc_view`` is the persisted row's JSON, ``from_cache`` the admission decision the row recorded."""
    example = BY_ID["q07_cache_hit"]
    world = run._build_world(inputs_of(example), "fake")  # pyright: ignore[reportPrivateUsage]
    for step in inputs_of(example)["pre_state"]:
        run._apply_step(world, step)  # pyright: ignore[reportPrivateUsage]
    final = run._submit(world, "quick", example["request"])  # pyright: ignore[reportPrivateUsage]
    check_id = cast("str", final["lc_view"].check_id)
    row = world.repo._checks[check_id]  # pyright: ignore[reportPrivateUsage]
    assert row.lc_view is not None
    assert run._outputs_contract(world, final)["from_cache"] is True  # pyright: ignore[reportPrivateUsage]

    row.lc_view["from_cache"] = False  # the persisted view disagrees with the recorded admission decision
    outputs = run._outputs_contract(world, final)  # pyright: ignore[reportPrivateUsage]

    assert outputs["lc_view"]["from_cache"] is False  # read back from the row, not from the state's LcView object
    assert outputs["from_cache"] is True  # from the admission decision

    row.internal["admission"]["admission"] = "new"  # pyright: ignore[reportOptionalSubscript]
    assert run._outputs_contract(world, final)["from_cache"] is False  # pyright: ignore[reportPrivateUsage]
    row.internal["admission"]["admission"] = "cache_hit"  # pyright: ignore[reportOptionalSubscript]
    inputs = cast("Mapping[str, Any]", inputs_of(example))
    result = DETERMINISTIC_EVALUATORS[4](inputs, outputs, example["expected"])
    assert result["key"] == "billing_consistency" and result["score"] is False
    assert "from_cache" in result["comment"]


def test_run_one_a_refusal_reports_admission_refused_and_no_calls() -> None:
    outputs = run_one(inputs_of(BY_ID["q08_quota_exhausted"]))

    assert outputs["admission"] == "refused" and outputs["surepass_calls"] == []
    assert (outputs["billed"], outputs["counted_against_quota"]) == (False, False)
    assert outputs["lc_view"]["status"] == "refused"


def test_run_one_pre_state_runs_prior_checks_through_the_same_graph() -> None:
    """f06: a first full check pulls the Report; the second opportunity, five days later, is a free hit. The call list in
    the outputs is the SECOND check's own (empty): prior checks' calls are not attributed to it."""
    outputs = run_one(inputs_of(BY_ID["f06_second_opportunity_same_pan"]))

    assert outputs["admission"] == "cache_hit" and outputs["from_cache"] is True
    assert outputs["surepass_calls"] == []
    assert outputs["lc_view"]["checked_on"] == "2026-10-10"  # the original pull date, not today's 2026-10-15


def _endpoint_of(call: dict[str, Any]) -> str:
    return str(call["endpoint"])


def test_run_one_is_deterministic_apart_from_the_check_id() -> None:
    for example_id in ("q01_happy_745", "f04_emi_four_branch"):
        first = run_one(inputs_of(BY_ID[example_id]))
        second = run_one(inputs_of(BY_ID[example_id]))
        for outputs in (first, second):
            outputs["lc_view"]["check_id"] = "x"
            outputs["surepass_calls"].sort(key=_endpoint_of)
        assert first == second


def test_run_one_rejects_an_unknown_world_and_an_unknown_pre_state_op() -> None:
    bad_world = copy.deepcopy(inputs_of(BY_ID["q01_happy_745"]))
    bad_world["surepass"] = {"scenario": "no_such_world", "kwargs": {}}
    with pytest.raises(KeyError, match="no_such_world"):
        run_one(bad_world)

    bad_step = copy.deepcopy(inputs_of(BY_ID["q01_happy_745"]))
    bad_step["pre_state"] = [{"op": "launch_missiles"}]
    with pytest.raises(ValueError, match="launch_missiles"):
        run_one(bad_step)


# --------------------------------------------------------------------------- the goldens bite


def _mutations(key: str, value: Any, named: set[str]) -> list[Any]:
    if key == "lender_numbers":
        table = cast("dict[str, dict[str, Any]]", value)
        out: list[Any] = []
        for lender_id in sorted(set(table) & named):
            bumped = copy.deepcopy(table)
            bumped[lender_id]["ceiling"] = 1000 if bumped[lender_id]["ceiling"] is None else bumped[lender_id]["ceiling"] + 1000
            out.append(bumped)
            dropped = {k: v for k, v in table.items() if k != lender_id}
            out.append(dropped)  # a named lender the golden no longer allows
        return out
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1]
    if isinstance(value, str):
        return [value + "x"]
    if isinstance(value, list):
        return [[*cast("list[Any]", value), "x"]]
    return ["x"]  # None


def _named_ids(outputs: Mapping[str, Any]) -> set[str]:
    view = outputs["lc_view"]
    named = {a["id"] for a in view["alternates"]}
    if view["top_lender"] is not None:
        named.add(view["top_lender"]["id"])
    return named


@pytest.mark.parametrize("example", ALL, ids=lambda e: e["example_id"])
def test_every_pin_in_every_golden_is_live(example: GoldenExample) -> None:
    """Mutate each expected value in turn: some evaluator must fail. A pin that never fails would be decoration."""
    outputs = outputs_of(example["example_id"])
    named = _named_ids(outputs)
    for key, value in example["expected"].items():
        mutations = _mutations(key, value, named)
        if key == "lender_numbers" and not named:
            continue  # nobody is named (band read, refusal, below floor): the table only bounds who MAY be named
        assert mutations, key
        for mutated in mutations:
            expected = {**example["expected"], key: mutated}
            assert failing_evaluators(example, outputs, expected), f"{example['example_id']}: pin {key!r} is dead"


def test_the_q05_golden_fails_a_run_that_reads_success_false_as_ntc() -> None:
    """The v1.0 reading (success:false = NTC) would have produced the NTC outcome: ok / unlikely / counted. The q05
    golden must reject exactly that, even though the run is internally consistent (no leak, billing consistent...)."""
    q05 = BY_ID["q05_score_success_false"]
    ntc_reading = outputs_of("q02_ntc_empty")  # an NTC run: ok, 'unlikely', counted
    failing = failing_evaluators(q05, ntc_reading, q05["expected"])
    assert "outcome_matches_band" in failing  # status / code / headline differ
    assert "billing_consistency" in failing  # counted True vs False


# --------------------------------------------------------------------------- local mode: failures, gating, no judge, no trace


def _only(dataset: str, examples: list[GoldenExample]) -> Callable[[str], list[GoldenExample]]:
    """A ``selected_examples`` stand-in: ``examples`` for ``dataset``, nothing for the other one."""

    def select(name: str) -> list[GoldenExample]:
        return list(examples) if name == dataset else []

    return select


def _tracing_on() -> bool:
    return bool(ls_utils.tracing_is_enabled())  # pyright: ignore[reportUnknownMemberType]


def test_a_failing_golden_fails_the_run_and_the_table_names_why(monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = copy.deepcopy(BY_ID["q01_happy_745"])
    wrong["expected"]["headline"] = "fundable"  # the quick cap (K16) says 'possible'
    monkeypatch.setattr(run, "selected_examples", _only(QUICK_DATASET, [wrong]))
    lines: list[str] = []

    code = run_local("quick", "fake", write=lines.append)

    text = "\n".join(lines)
    assert code == 1
    assert "FAIL" in text and "q01_happy_745: outcome_matches_band: headline: got 'possible', golden 'fundable'" in text
    assert "0/1 examples passed" in text


def test_a_crashing_golden_is_a_failed_golden_not_an_aborted_run(monkeypatch: pytest.MonkeyPatch) -> None:
    crashing = copy.deepcopy(BY_ID["q01_happy_745"])
    crashing["surepass"] = {"scenario": "none", "kwargs": {}}  # any SurePass call raises AssertionError
    healthy = BY_ID["q02_ntc_empty"]
    monkeypatch.setattr(run, "selected_examples", _only(QUICK_DATASET, [crashing, healthy]))
    lines: list[str] = []

    code = run_local("quick", "fake", write=lines.append)

    text = "\n".join(lines)
    assert code == 1
    assert "q01_happy_745  ERROR  AssertionError" in text and "1/2 examples passed" in text


def test_local_mode_never_builds_the_judge_even_with_every_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(environ: Mapping[str, str]) -> list[Evaluator]:
        raise AssertionError("the judge must not be constructed in local mode")

    monkeypatch.setattr(run, "judge_evaluators", boom)
    monkeypatch.setattr(evaluators, "ChatOpenAI", boom)
    lines: list[str] = []

    assert main(["--mode", "local", "--dataset", "quick"], environ=dict(KEYS), write=lines.append) == 0
    assert "6 evaluators" in "\n".join(lines)


def test_a_run_exports_no_trace_even_with_tracing_switched_on_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's shell may have LANGCHAIN_TRACING_V2=true. ``run_one`` wraps the run in ``tracing_context(enabled=False)``,
    so langchain never attaches its default tracer: the seam observes tracing as DISABLED while the environment says on."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    ls_utils.get_env_var.cache_clear()  # pyright: ignore[reportUnknownMemberType,reportFunctionMemberAccess]
    try:
        assert _tracing_on()  # positive control: outside run_one the environment turns tracing on
        seen: list[bool] = []

        def spying_rank(rank: str, llm: str) -> Callable[..., ModelOutput]:
            inner = FakeRankModel()

            def model(ctx: RankContext, *, config: Any = None) -> ModelOutput:
                seen.append(_tracing_on())
                return inner(ctx, config=config)

            return model

        monkeypatch.setattr(run, "_rank_model", spying_rank)

        run_one(inputs_of(BY_ID["q01_happy_745"]))

        assert seen == [False]
    finally:
        monkeypatch.undo()
        ls_utils.get_env_var.cache_clear()  # pyright: ignore[reportUnknownMemberType,reportFunctionMemberAccess]


def test_run_one_traced_leaves_the_callers_tracing_context_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Langsmith mode runs the graph ``traced=True`` so node runs nest under the experiment row; the default (local) wraps
    the run in ``tracing_context(enabled=False)``. Observed through a recorder in place of ``tracing_context`` (switching
    tracing on for real would build a langsmith client and reach for the network)."""
    contexts: list[Any] = []

    def recording_context(**kwargs: Any) -> contextlib.AbstractContextManager[None]:
        contexts.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(run, "tracing_context", recording_context)
    inputs = inputs_of(BY_ID["q01_happy_745"])

    run_one(inputs)
    assert contexts == [{"enabled": False}]
    contexts.clear()
    assert run_one(inputs, traced=True)["lc_view"]["status"] == "ok"
    assert contexts == []  # the caller's context is untouched


def test_the_langsmith_target_runs_traced_and_local_mode_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    flags: list[bool] = []

    def recording(inputs: GoldenInputs, llm: str = "fake", *, traced: bool = False) -> dict[str, Any]:
        flags.append(traced)
        return {}

    monkeypatch.setattr(run, "run_one", recording)
    spy, _, _ = _langsmith({"LANGSMITH_API_KEY": "ls"})
    spy.calls[0]["target"](cast("dict[str, Any]", inputs_of(BY_ID["q01_happy_745"])))
    assert flags == [True]
    flags.clear()
    run_local("quick", "fake", write=lambda _line: None)  # the stub returns no outputs, so the evaluators fail: ignored
    assert flags == [False] * 12  # local mode never asks for tracing


@pytest.mark.parametrize(
    ("argv", "environ", "message"),
    [
        pytest.param(["--mode", "langsmith"], {}, "LANGSMITH_API_KEY", id="langsmith-needs-its-key"),
        pytest.param(["--mode", "langsmith"], {"LANGSMITH_API_KEY": "  "}, "LANGSMITH_API_KEY", id="blank-key"),
        pytest.param(["--mode", "local", "--llm", "real"], {}, "OPENAI_API_KEY", id="real-local-needs-openai"),
        pytest.param(
            ["--mode", "langsmith", "--llm", "real"], {"LANGSMITH_API_KEY": "k"}, "OPENAI_API_KEY",
            id="real-langsmith-needs-openai",
        ),
        pytest.param(["--mode", "local", "--build-datasets"], {}, "--build-datasets", id="build-needs-langsmith"),
        pytest.param(["--mode", "local", "--max-real", "-1"], {}, "--max-real", id="negative-max-real"),
    ],
)
def test_gating_exits_two_with_a_message_and_runs_nothing(
    argv: list[str], environ: dict[str, str], message: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("nothing may run when the gate refuses")

    monkeypatch.setattr(run, "run_one", boom)
    monkeypatch.setattr(run, "Client", boom)

    assert main(argv, environ=environ, write=lambda _line: None) == 2
    assert message in capsys.readouterr().err


def test_real_mode_without_a_key_is_refused_inside_run_one_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(run.EvalConfigError, match="OPENAI_API_KEY"):
        run_one(inputs_of(BY_ID["q01_happy_745"]), "real")
    # the K7 golden never needs the model, so it runs on the scripted fake even in a real-mode configuration
    assert run_one(inputs_of(BY_ID["q04_openai_invalid_twice"]), "real")["lc_view"]["code"] == "OPENAI_FAILURE"


# --------------------------------------------------------------------------- langsmith mode (stubs only)


class FakeChat:
    """Stands in for ``ChatOpenAI`` where a keyed run constructs the judge (construction needs no network, but the real
    client builds an httpx transport that pytest-socket warns about)."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "FakeChat":
        return self


@pytest.fixture
def fake_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "ChatOpenAI", FakeChat)


class EvaluateSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, target: Callable[[dict[str, Any]], dict[str, Any]], **kwargs: Any) -> object:
        self.calls.append({"target": target, **kwargs})
        return None


def _langsmith(
    environ: Mapping[str, str], *, llm: LlmMode = "fake", max_real: int = 20, build: bool = False, client: RecordingClient | None = None
) -> tuple[EvaluateSpy, RecordingClient, list[str]]:
    spy, stub = EvaluateSpy(), client or RecordingClient()
    lines: list[str] = []
    code = run_langsmith(
        "both",
        llm,
        max_real=max_real,
        experiment_prefix="ci-abc",
        build_datasets=build,
        environ=environ,
        client=stub,
        evaluate_fn=spy,
        write=lines.append,
    )
    assert code == 0
    return spy, stub, lines


def test_langsmith_mode_evaluates_each_dataset_with_the_six_deterministic_evaluators() -> None:
    spy, stub, _ = _langsmith({"LANGSMITH_API_KEY": "ls"})

    assert [c["data"] for c in spy.calls] == [QUICK_DATASET, FULL_DATASET]
    assert [c["experiment_prefix"] for c in spy.calls] == [f"ci-abc-{QUICK_DATASET}", f"ci-abc-{FULL_DATASET}"]
    for call in spy.calls:
        assert call["max_concurrency"] == 4 and call["client"] is stub
        assert [e.__name__ for e in call["evaluators"]] == [e.__name__ for e in DETERMINISTIC_EVALUATORS]
        assert call["metadata"]["llm"] == "fake"
    assert stub.of("create_examples") == []  # no --build-datasets: nothing uploaded


@pytest.mark.usefixtures("fake_chat")
def test_langsmith_mode_adds_the_judge_only_when_openai_is_keyed() -> None:
    keyed, _, _ = _langsmith(dict(KEYS))
    unkeyed, _, _ = _langsmith({"LANGSMITH_API_KEY": "ls"})

    assert [e.__name__ for e in keyed.calls[0]["evaluators"]][-1] == "plain_language_judge"
    assert len(keyed.calls[0]["evaluators"]) == 7 and len(unkeyed.calls[0]["evaluators"]) == 6


def test_langsmith_build_datasets_uploads_through_the_client_before_evaluating() -> None:
    _, stub, lines = _langsmith({"LANGSMITH_API_KEY": "ls"}, build=True)

    assert [len(u["examples"]) for u in stub.of("create_examples")] == [12, 10]
    assert any("uploaded examples" in line for line in lines)


def test_the_target_runs_run_one_on_the_examples_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    spy, _, _ = _langsmith({"LANGSMITH_API_KEY": "ls"})
    target = spy.calls[0]["target"]

    outputs = target(cast("dict[str, Any]", inputs_of(BY_ID["q01_happy_745"])))

    assert outputs["lc_view"]["status"] == "ok" and outputs["billed"] is True


@pytest.mark.usefixtures("fake_chat")
def test_with_a_real_model_only_the_first_max_real_examples_run_real_the_rest_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def recording(inputs: GoldenInputs, llm: str = "fake", *, traced: bool = False) -> dict[str, Any]:
        seen[inputs["example_id"]] = llm
        return {}

    monkeypatch.setattr(run, "run_one", recording)
    spy, _, _ = _langsmith(dict(KEYS), llm="real", max_real=3)
    for call in spy.calls:
        for example in datasets.selected_examples(call["data"]):
            call["target"](cast("dict[str, Any]", inputs_of(example)))

    order = [e["example_id"] for e in ALL]
    assert [seen[i] for i in order] == ["real"] * 3 + ["fake"] * 19  # quick first, then full; the cut is across both
    assert run._llm_plan("both", "fake", max_real=20) == {i: "fake" for i in order}
    assert set(run._llm_plan("both", "real", max_real=0).values()) == {"fake"}
    assert list(run._llm_plan("both", "real", max_real=50).values()).count("real") == 22


def test_langsmith_mode_without_its_key_never_touches_the_client() -> None:
    with pytest.raises(run.EvalConfigError, match="LANGSMITH_API_KEY"):
        run_langsmith(
            "both", "fake", max_real=20, experiment_prefix="x", build_datasets=True, environ={}, client=RecordingClient(),
            evaluate_fn=EvaluateSpy(),
        )


def test_the_real_langsmith_evaluate_drives_our_target_and_evaluators_offline() -> None:
    """The installed ``langsmith.evaluate`` (0.14.x) over in-memory examples with ``upload_results=False``: no network, no
    key. It proves the target's signature, the name-matched evaluator kwargs and the JSON-safe outputs contract all work
    with the REAL runner, not only with our spy."""
    import datetime as dt
    import uuid
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langsmith import evaluate  # pyright: ignore[reportUnknownVariableType]
        from langsmith.schemas import Example

    chosen = [BY_ID["q01_happy_745"], BY_ID["q05_score_success_false"], BY_ID["f08_ntc_full"]]
    examples = [
        Example(  # pyright: ignore[reportCallIssue]
            id=uuid.uuid4(),
            dataset_id=uuid.uuid4(),
            inputs=dict(inputs_of(e)),
            outputs=e["expected"],
            created_at=dt.datetime.now(dt.timezone.utc),
        )
        for e in chosen
    ]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_one(cast("GoldenInputs", inputs), "fake")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = evaluate(  # pyright: ignore[reportUnknownVariableType]
            target,
            data=examples,
            evaluators=cast("Any", list(DETERMINISTIC_EVALUATORS)),
            upload_results=False,
            max_concurrency=1,
        )
        rows = [cast("dict[str, Any]", row) for row in results]  # pyright: ignore[reportUnknownVariableType]

    assert len(rows) == 3
    for row in rows:
        scored = {r.key: r.score for r in row["evaluation_results"]["results"]}
        assert scored == {e.__name__: True for e in DETERMINISTIC_EVALUATORS}
