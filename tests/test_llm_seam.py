"""Task 10 (part 3): graph/llm.py, the production seam factory and the state -> RankContext mapping.

``make_rank_model`` is exercised against a spy in place of ``ChatOpenAI`` (no network, no key): the load-bearing facts are
the constructor arguments (temperature 0, the IMPORTED 8 s timeout, ``max_retries=0`` because the ONE business retry lives
in the node), the ``function_calling`` method pin, config threading, and that the seam itself never retries.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Final, cast

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr
from rank_builders import full_emi, gl, make_state, num, quick_emi, report_facts
from rank_builders import derived as make_derived
from rank_builders import identity as make_identity

from eligibility_agent import config
from eligibility_agent.config import OPENAI_TIMEOUT_S
from eligibility_agent.graph import llm
from eligibility_agent.graph.llm import LenderContext, RankContext, RankModel, build_rank_context, make_rank_model
from eligibility_agent.graph.prompts import local_prompts, render_rank_user
from eligibility_agent.graph.state import BAND_ORDER, Band, ModelOutput, Pick
from eligibility_agent.lenders.safety import check_pick
from eligibility_agent.testing.fake_llm import ranked_output

LLM_PY: Final[Path] = Path(__file__).resolve().parent.parent / "eligibility_agent" / "graph" / "llm.py"
BANDS: Final[tuple[Band, ...]] = tuple(BAND_ORDER)


# --------------------------------------------------------------------------- make_rank_model


class _SpyRunnable:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.invocations: list[tuple[Any, RunnableConfig | None]] = []

    def invoke(self, messages: Any, config: RunnableConfig | None = None) -> object:
        self.invocations.append((messages, config))
        if self.error is not None:
            raise self.error
        return self.result


class _Spy:
    """Stands in for ``ChatOpenAI``: records constructor kwargs and ``with_structured_output`` calls."""

    def __init__(self, runnable: _SpyRunnable) -> None:
        self.runnable = runnable
        self.constructed: list[dict[str, Any]] = []
        self.structured: list[tuple[Any, dict[str, Any]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = self

        class FakeChatOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                spy.constructed.append(kwargs)

            def with_structured_output(self, schema: Any, **kwargs: Any) -> _SpyRunnable:
                spy.structured.append((schema, kwargs))
                return spy.runnable

        monkeypatch.setattr(llm, "ChatOpenAI", FakeChatOpenAI)


def _spy(monkeypatch: pytest.MonkeyPatch, result: object = None, error: Exception | None = None) -> _Spy:
    spy = _Spy(_SpyRunnable(result, error))
    spy.install(monkeypatch)
    return spy


def test_chat_model_is_built_with_the_fixed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _spy(monkeypatch)
    make_rank_model(model="gpt-4.1", prompts=local_prompts())
    assert spy.constructed == [{"model": "gpt-4.1", "temperature": 0, "timeout": OPENAI_TIMEOUT_S, "max_retries": 0}]
    assert OPENAI_TIMEOUT_S == 8.0


def test_the_model_name_is_the_callers_not_a_hard_coded_one(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _spy(monkeypatch)
    make_rank_model(model="some-other-model", prompts=local_prompts())
    assert spy.constructed[0]["model"] == "some-other-model"


def test_structured_output_pins_function_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _spy(monkeypatch)
    make_rank_model(model="gpt-4.1", prompts=local_prompts())
    assert spy.structured == [(ModelOutput, {"method": "function_calling"})]


def test_timeout_is_imported_from_config_never_a_literal() -> None:
    assert llm.OPENAI_TIMEOUT_S is config.OPENAI_TIMEOUT_S
    numbers = [
        node.value
        for node in ast.walk(ast.parse(LLM_PY.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool)
    ]
    assert 8 not in numbers and 8.0 not in numbers, "OPENAI_TIMEOUT_S must be imported, not restated"
    assert "max_retries=0" in LLM_PY.read_text().replace(" ", "")


def test_the_call_threads_messages_and_config_and_returns_the_parsed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = ranked_output("L1", "L2")
    spy = _spy(monkeypatch, result=expected)
    rank = make_rank_model(model="gpt-4.1", prompts=local_prompts())
    ctx = build_rank_context(make_state())
    cfg: RunnableConfig = {"tags": ["tier:full"], "run_id": None}
    assert rank(ctx, config=cfg) is expected

    (messages, passed_config), = spy.runnable.invocations
    assert passed_config is cfg
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert messages[0].content == local_prompts().rank_system("full")
    assert messages[1].content == render_rank_user(ctx)


def test_config_defaults_to_none_and_the_system_prompt_follows_the_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _spy(monkeypatch, result=ranked_output("L1"))
    rank = make_rank_model(model="gpt-4.1", prompts=local_prompts())
    rank(build_rank_context(make_state(kind="quick")))
    (messages, passed_config), = spy.runnable.invocations
    assert passed_config is None
    assert messages[0].content == local_prompts().rank_system("quick")


@pytest.mark.parametrize("wrong", [None, {"recommendation": None}, "text", 3], ids=["none", "dict", "str", "int"])
def test_a_result_that_is_not_a_model_output_raises_type_error(monkeypatch: pytest.MonkeyPatch, wrong: object) -> None:
    _spy(monkeypatch, result=wrong)
    rank = make_rank_model(model="gpt-4.1", prompts=local_prompts())
    with pytest.raises(TypeError, match="ModelOutput"):
        rank(build_rank_context(make_state()))


def test_the_seam_never_retries_and_never_swallows(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = TimeoutError("request timed out")
    spy = _spy(monkeypatch, error=boom)
    rank = make_rank_model(model="gpt-4.1", prompts=local_prompts())
    with pytest.raises(TimeoutError) as raised:
        rank(build_rank_context(make_state()))
    assert raised.value is boom
    assert len(spy.runnable.invocations) == 1, "the ONE business retry lives in the rank node, not the seam"


def test_an_explicit_api_key_is_passed_through_and_never_read_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # ChatOpenAI reads OPENAI_API_KEY from os.environ only; a key that lives in Settings (.env) must be passed explicitly.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spy = _spy(monkeypatch)
    make_rank_model(model="gpt-4.1", prompts=local_prompts(), api_key="sk-test-not-real")
    key = spy.constructed[0]["api_key"]
    assert isinstance(key, SecretStr) and key.get_secret_value() == "sk-test-not-real"  # never a bare string
    make_rank_model(model="gpt-4.1", prompts=local_prompts())
    assert "api_key" not in spy.constructed[1]  # omitted: ChatOpenAI's own env lookup applies


def test_the_timeout_can_be_overridden_but_defaults_to_the_fixed_one(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _spy(monkeypatch)
    make_rank_model(model="gpt-4.1", prompts=local_prompts(), timeout=3.0)
    assert spy.constructed[0]["timeout"] == 3.0


def test_the_returned_callable_has_the_protocol_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    _spy(monkeypatch)
    rank = make_rank_model(model="gpt-4.1", prompts=local_prompts())
    proto = inspect.signature(RankModel.__call__, eval_str=True)
    mine = inspect.signature(rank, eval_str=True)
    assert [(p.name, p.kind, p.default, p.annotation) for p in list(proto.parameters.values())[1:]] == [
        (p.name, p.kind, p.default, p.annotation) for p in mine.parameters.values()
    ]
    assert mine.return_annotation == proto.return_annotation


# ChatOpenAI probes which TCP socket options the platform supports by creating (never connecting) a local socket, which
# pytest-socket reports as a warning. No connection is made; the test would fail outright if one were attempted.
@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_real_chat_openai_construction_is_call_free_and_carries_the_fixed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    captured: list[Any] = []
    real_chat = llm.ChatOpenAI

    def capturing(**kwargs: Any) -> Any:
        instance = real_chat(**kwargs)
        captured.append(instance)
        return instance

    monkeypatch.setattr(llm, "ChatOpenAI", capturing)
    rank = make_rank_model(model="gpt-4.1", prompts=local_prompts())  # no network: pytest-socket would fail the test
    assert callable(rank)
    (chat,) = captured
    assert chat.model_name == "gpt-4.1"
    assert chat.temperature == 0
    assert chat.request_timeout == OPENAI_TIMEOUT_S
    assert chat.max_retries == 0


# --------------------------------------------------------------------------- build_rank_context: mapping


def test_full_state_maps_to_the_9_2_payload() -> None:
    ctx = build_rank_context(make_state())
    assert ctx == RankContext(
        kind="full",
        band="720-780",
        code_confidence="medium",
        foir=0.7333333333,
        emi_floor_flag=True,
        existing_emi_reported=12500.0,
        existing_emi_estimated=31500.0,
        emi_past_tenure_accounts=1,
        emi_stale_accounts=1,
        declared_income=60000.0,
        declared_emi=None,
        aadhaar_linked=True,
        recent_enquiries_180d=3,
        past_due_flag=False,
        stale_account_flag=True,
        lenders=[
            LenderContext(
                lender_id=lender_id,
                ceiling=800000.0,
                range_lower=300000.0,
                range_upper=800000.0,
                not_fundable_alone=False,
                age_at_application=32,
                age_at_maturity=36,
                band_clears_floor=True,
                ntc_accepted=False,
                foir_limit=0.55,
                min_income=25000.0,
                tenure_label="12 to 48 months",
                rate_band_label="11 to 14%",
                policy_notes=None,
                uncertain_fields=[],
            )
            for lender_id in ("L1", "L2")
        ],
        attempt=1,
        prior_error=None,
    )


def test_quick_state_carries_no_file_facts() -> None:
    state = make_state(kind="quick", existing_emi_declared=15000.0, monthly_income=None)
    state["derived"] = make_derived(emi=quick_emi(15000.0), current_foir=None)
    ctx = build_rank_context(state)
    assert ctx.kind == "quick"
    assert (ctx.existing_emi_reported, ctx.existing_emi_estimated) == (None, None)
    assert (ctx.emi_past_tenure_accounts, ctx.emi_stale_accounts) == (0, 0)
    assert (ctx.recent_enquiries_180d, ctx.past_due_flag, ctx.stale_account_flag) == (None, None, None)
    assert ctx.declared_emi == 15000.0
    assert ctx.declared_income is None and ctx.foir is None
    assert ctx.emi_floor_flag is False  # a figure was declared


def test_quick_state_without_a_declared_emi_flags_the_floor() -> None:
    state = make_state(kind="quick")
    ctx = build_rank_context(state)
    assert ctx.declared_emi is None and ctx.emi_floor_flag is True


def test_a_declared_emi_that_won_is_shown_next_to_the_file_split() -> None:
    state = make_state()
    declared_wins = full_emi().model_copy(update={"declared": 50000.0, "used": 50000.0, "used_source": "declared"})
    state["derived"] = make_derived(emi=declared_wins)
    ctx = build_rank_context(state)
    assert ctx.declared_emi == 50000.0
    assert (ctx.existing_emi_reported, ctx.existing_emi_estimated) == (12500.0, 31500.0)


@pytest.mark.parametrize("linked", [True, False, None])
def test_aadhaar_link_is_passed_through_as_a_tri_state(linked: bool | None) -> None:
    state = make_state(identity=make_identity(aadhaar_linked=linked))
    assert build_rank_context(state).aadhaar_linked is linked


def test_missing_identity_means_unknown_aadhaar() -> None:
    assert build_rank_context(make_state(identity=None)).aadhaar_linked is None


def test_past_due_flag_and_enquiries_come_from_the_report() -> None:
    clean = build_rank_context(make_state(report=report_facts(past_due_total=0.0, enquiries_180d=2)))
    overdue = build_rank_context(make_state(report=report_facts(past_due_total=1500.0, enquiries_180d=7)))
    assert (clean.past_due_flag, clean.recent_enquiries_180d) == (False, 2)
    assert (overdue.past_due_flag, overdue.recent_enquiries_180d) == (True, 7)


def test_stale_flag_is_false_for_a_file_without_stale_accounts() -> None:
    state = make_state()
    base = full_emi()
    emi = base.model_copy(update={"items": [i for i in base.items if i.kind != "stale_excluded"]})
    state["derived"] = make_derived(emi=emi)
    ctx = build_rank_context(state)
    assert ctx.stale_account_flag is False and ctx.emi_stale_accounts == 0


def test_attempt_counts_the_call_about_to_be_made_and_carries_the_prior_error() -> None:
    first = build_rank_context(make_state(rank_attempts=0, rank_error=None))
    assert (first.attempt, first.prior_error) == (1, None)
    stale = build_rank_context(make_state(rank_attempts=0, rank_error="left over"))
    assert (stale.attempt, stale.prior_error) == (1, None), "no prior error on a first attempt"
    retry = build_rank_context(make_state(rank_attempts=1, rank_error="Request timed out."))
    assert (retry.attempt, retry.prior_error) == (2, "Request timed out.")
    assert build_rank_context(make_state(rank_attempts=1, rank_error=None)).prior_error is None


def test_prior_error_is_capped_at_500_characters() -> None:
    ctx = build_rank_context(make_state(rank_attempts=1, rank_error="e" * 900))
    assert ctx.prior_error == "e" * 500


def test_lenders_are_sorted_by_id_whatever_order_they_arrive_in() -> None:
    shuffled = [gl("L3"), gl("L1"), gl("L2")]
    ctx = build_rank_context(make_state(lenders=shuffled))
    assert [lender.lender_id for lender in ctx.lenders] == ["L1", "L2", "L3"]


def test_an_explicit_gated_list_overrides_the_states() -> None:
    state = make_state(lenders=[gl("L1"), gl("L2")])
    only_l2 = [g for g in state.get("gated_lenders", []) if g.id == "L2"]
    assert [lender.lender_id for lender in build_rank_context(state, only_l2).lenders] == ["L2"]
    assert [lender.lender_id for lender in build_rank_context(state, []).lenders] == []
    assert [lender.lender_id for lender in build_rank_context(state).lenders] == ["L1", "L2"]


def test_a_state_without_a_gated_list_gives_an_empty_lender_list() -> None:
    state = make_state()
    del state["gated_lenders"]
    assert build_rank_context(state).lenders == []


def test_lender_numbers_map_to_the_display_values_and_a_missing_entry_is_all_unknown() -> None:
    lenders = [gl("L1"), gl("L2"), gl("L3")]
    numbers = {
        "L1": num("L1"),
        "L2": num("L2", available_emi=0.0, not_fundable_alone=True, ceiling=None, loan_max_lower=None, loan_max_upper=None, age_at_maturity_years=None),
    }
    by_id = {lender.lender_id: lender for lender in build_rank_context(make_state(lenders=lenders, numbers=numbers)).lenders}
    assert (by_id["L1"].ceiling, by_id["L1"].range_lower, by_id["L1"].range_upper) == (800000.0, 300000.0, 800000.0)
    assert by_id["L2"].not_fundable_alone is True
    assert (by_id["L2"].ceiling, by_id["L2"].range_lower, by_id["L2"].range_upper) == (None, None, None)
    assert by_id["L3"].not_fundable_alone is False
    assert (by_id["L3"].ceiling, by_id["L3"].age_at_maturity) == (None, None)


def test_the_display_ceiling_is_what_the_model_sees_never_the_raw_annuity_value() -> None:
    ctx = build_rank_context(make_state(lenders=[gl("L1")], numbers={"L1": num("L1", loan_max_upper=800000.0, loan_max_upper_raw=800123.456)}))
    assert "800123" not in ctx.model_dump_json()


def test_criteria_facts_policy_notes_and_labels_pass_through() -> None:
    lender = gl("L1", policy_notes="Salaried preferred", min_income=30000.0, foir_limit=0.45, ntc_accepted=True)
    (ctx_lender,) = build_rank_context(make_state(lenders=[lender])).lenders
    assert ctx_lender.policy_notes == "Salaried preferred"
    assert (ctx_lender.min_income, ctx_lender.foir_limit, ctx_lender.ntc_accepted) == (30000.0, 0.45, True)
    assert (ctx_lender.tenure_label, ctx_lender.rate_band_label) == (lender.tenure_label, lender.rate_band_label)


def test_the_floor_value_never_reaches_the_model_only_whether_the_band_clears_it() -> None:
    ctx = build_rank_context(make_state(lenders=[gl("L1", score_floor=701)]))
    assert "701" not in ctx.model_dump_json()
    assert ctx.lenders[0].band_clears_floor is True


def test_uncertain_fields_are_the_required_criteria_the_admin_marked_uncertain() -> None:
    lender = gl("L1", field_status={"foir_limit": "uncertain", "score_floor": "uncertain", "min_income": "uncertain", "loan_min": "known"})
    (ctx_lender,) = build_rank_context(make_state(lenders=[lender])).lenders
    assert ctx_lender.uncertain_fields == ["foir_limit", "score_floor"]  # min_income is optional: gate.uncertain_required_fields


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("floor", [None, 300, 649, 650, 719, 720, 779, 780, 899])
@pytest.mark.parametrize("ntc_accepted", [None, True, False])
def test_band_clears_floor_agrees_with_the_safety_rule_it_summarises(band: Band, floor: int | None, ntc_accepted: bool | None) -> None:
    lender = gl("L1", score_floor=floor, ntc_accepted=ntc_accepted)
    (ctx_lender,) = build_rank_context(make_state(band=band, lenders=[lender])).lenders
    breach = check_pick(
        Pick(lender_id="L1", rank=1, rationale="r", conditions=[]), lender, num("L1"), band=band, age_known=False, age_app_years=None
    )
    breached = breach is not None and breach.rule == "band_floor"
    if ctx_lender.band_clears_floor is None:
        assert band != "NTC" and floor is None and not breached
    else:
        assert ctx_lender.band_clears_floor is (not breached)


@pytest.mark.parametrize("missing", ["kind", "derived", "code_confidence"])
def test_a_state_the_rank_node_could_not_have_reached_is_a_value_error(missing: str) -> None:
    state = make_state()
    cast(dict[str, Any], state).pop(missing)
    with pytest.raises(ValueError, match=missing):
        build_rank_context(state)
