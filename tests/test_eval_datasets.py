"""Task 15: the golden datasets (12 quick + 10 full) and their LangSmith upload.

The goldens are data. What is pinned here: the counts and ids; the load-bearing examples (q05: the v1.1 NTC reversal at eval
level; q04: the K7 invalid-twice view); that every example is plain JSON that survives a LangSmith round trip; that the
worlds, lenders and pre-state ops each example names exist; that the data is 100% synthetic (probe literals absent, every
PAN / mobile / name from the shared helper pools); that the expected values stay HAND-WRITTEN (the dataset module imports
no derivation, range, lender or redaction code); and the upload uses the modern ``create_examples(dataset_id=, examples=)``
form with deterministic ids.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Final, cast

import pytest
from eval_builders import RecordingClient
from langsmith import Client
from langsmith.schemas import ExampleCreate

from eligibility_agent.evals import datasets, evaluators, worlds
from eligibility_agent.evals.datasets import (
    FULL_DATASET,
    QUICK_DATASET,
    DatasetsClient,
    GoldenExample,
    build_langsmith_datasets,
    full_examples,
    inputs_of,
    quick_examples,
)
from eligibility_agent.surepass.client import PAN_ENDPOINT, SCORE_ENDPOINT
from tests.helpers import (
    LC_TYPED_NAME,
    MOBILE_OK,
    PAN_COMPANY,
    PAN_INVALID,
    PAN_NTC,
    PAN_OK,
    PAN_THREE,
    PAN_TWO,
    PROBE_LITERALS,
    SEED_LENDERS,
)

ALL: Final[list[GoldenExample]] = [*quick_examples(), *full_examples()]
BY_ID: Final[dict[str, GoldenExample]] = {e["example_id"]: e for e in ALL}
PAN_POOL: Final[frozenset[str]] = frozenset({PAN_OK, PAN_TWO, PAN_THREE, PAN_NTC, PAN_COMPANY, PAN_INVALID})
PRE_STATE_OPS: Final[frozenset[str]] = frozenset(
    {"seed_counted_quick_checks", "seed_cached_quick_world", "run_check", "advance_days"}
)
EXPECTED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "status", "code", "headline", "headline_in", "confidence", "coapplicant_suggested", "top_lender_id",
        "alternate_ids", "label", "items_to_confirm", "pan_aadhaar_link_warning", "quick_checks_left_today",
        "checked_on", "refresh_available_on", "band", "gated_lender_ids", "lender_numbers", "billed",
        "counted_against_quota", "from_cache", "admission", "surepass_endpoints",
    }
)  # fmt: skip


# --------------------------------------------------------------------------- shape


def test_there_are_twelve_quick_and_ten_full_goldens_with_unique_ids() -> None:
    assert len(quick_examples()) == 12
    assert len(full_examples()) == 10
    ids = [e["example_id"] for e in ALL]
    assert len(set(ids)) == 22
    assert [e["example_id"][:4] for e in quick_examples()] == [f"q{n:02d}_" for n in range(1, 13)]
    assert [e["example_id"][:4] for e in full_examples()] == [f"f{n:02d}_" for n in range(1, 11)]
    assert (QUICK_DATASET, FULL_DATASET) == ("eligibility-quick-golden", "eligibility-full-golden")


def test_the_dossier_example_names_are_all_present() -> None:
    """The v1.1 dossier's lists, minus ntc_missing (replaced by the K7 golden: the brief's binding K7 expectations have
    no slot in the dossier's 12, and 12 is fixed; the missing-key NTC variant stays pinned in the adapter + scenario
    suites)."""
    assert [e["example_id"] for e in quick_examples()] == [
        "q01_happy_745",
        "q02_ntc_empty",
        "q03_ntc_text",
        "q04_openai_invalid_twice",
        "q05_score_success_false",
        "q06_pan_company",
        "q07_cache_hit",
        "q08_quota_exhausted",
        "q09_no_pincode",
        "q10_no_income",
        "q11_band_610",
        "q12_aadhaar_not_linked",
    ]
    assert [e["example_id"] for e in full_examples()] == [
        "f01_happy_offer_reported_emi",
        "f02_stage_too_early",
        "f03_pincode_union",
        "f04_emi_four_branch",
        "f05_stale_account",
        "f06_second_opportunity_same_pan",
        "f07_report_refreshes_score",
        "f08_ntc_full",
        "f09_declared_emi_wins",
        "f10_report_window_refresh",
    ]


@pytest.mark.parametrize("example", ALL, ids=lambda e: e["example_id"])
def test_every_example_is_well_formed(example: GoldenExample) -> None:
    assert set(example) == {"example_id", "kind", "request", "lenders", "surepass", "rank", "pre_state", "expected"}
    assert example["kind"] in ("quick", "full")
    assert example["rank"] in ("llm", "invalid_twice")
    assert set(example["lenders"]) <= {lender.id for lender in SEED_LENDERS} and example["lenders"]
    assert set(example["expected"]) <= EXPECTED_KEYS, set(example["expected"]) - EXPECTED_KEYS
    assert {"status", "billed", "counted_against_quota", "from_cache", "admission", "surepass_endpoints"} <= set(
        example["expected"]
    )
    spec = example["surepass"]
    assert set(spec) == {"scenario", "kwargs"} and spec["scenario"] in worlds.SCENARIOS
    for step in example["pre_state"]:
        assert step["op"] in PRE_STATE_OPS, step
        if step["op"] == "run_check":
            assert step["kind"] in ("quick", "full") and isinstance(step["request"], dict)
    assert "consent" in example["request"]  # the route's own body shape


@pytest.mark.parametrize("example", ALL, ids=lambda e: e["example_id"])
def test_every_example_is_plain_json(example: GoldenExample) -> None:
    """LangSmith stores inputs / outputs as JSON: a tuple, a set or a non-string key would not survive the round trip."""
    assert json.loads(json.dumps(example)) == example
    assert json.loads(json.dumps(inputs_of(example))) == inputs_of(example)


def test_the_example_functions_return_fresh_objects_each_call() -> None:
    first, second = quick_examples(), quick_examples()
    first[0]["expected"]["billed"] = "mutated"
    first[0]["request"]["lc_id"] = "mutated"
    assert second[0]["expected"]["billed"] is True and second[0]["request"]["lc_id"] != "mutated"


def test_inputs_of_drops_the_expected_values_and_nothing_else() -> None:
    example = BY_ID["q01_happy_745"]
    inputs = inputs_of(example)
    assert set(inputs) == set(example) - {"expected"}
    assert inputs["request"] == example["request"] and inputs["request"] is not example["request"]


# --------------------------------------------------------------------------- the load-bearing goldens


def test_q05_pins_the_v11_ntc_reversal_at_eval_level() -> None:
    """(HTTP 200, success:false, no_record_found) is a SERVICE FAILURE (SUREPASS_FAILURE), never NTC: not counted, no
    headline, no lender. The PAN call is a billed Rs 1; the Score did not bill."""
    q05 = BY_ID["q05_score_success_false"]
    assert q05["surepass"] == {"scenario": "score_failure", "kwargs": {"variant": "success_false", "pan": PAN_NTC}}
    expected = q05["expected"]
    assert (expected["status"], expected["code"]) == ("service_failure", "SUREPASS_FAILURE")
    assert (expected["headline"], expected["confidence"], expected["top_lender_id"]) == (None, None, None)
    assert (expected["billed"], expected["counted_against_quota"], expected["from_cache"]) == (True, False, False)
    assert expected["label"] == "Credit check couldn't complete, retry"
    assert sorted(expected["surepass_endpoints"]) == sorted([PAN_ENDPOINT, SCORE_ENDPOINT])
    assert expected["quick_checks_left_today"] == 5  # no quota was used


def test_q04_is_the_k7_view_with_the_band_derived_headline_and_number_free_alternates() -> None:
    q04 = BY_ID["q04_openai_invalid_twice"]
    assert q04["rank"] == "invalid_twice"  # scripted, whatever model the run is configured for
    expected = q04["expected"]
    assert (expected["status"], expected["code"], expected["label"]) == (
        "service_failure",
        "OPENAI_FAILURE",
        "Couldn't generate a recommendation, retry",
    )
    assert (expected["headline"], expected["confidence"]) == ("possible", "low")  # band-derived, never under_review
    assert (expected["top_lender_id"], expected["alternate_ids"]) == (None, ["L1", "L2"])
    assert expected["lender_numbers"] == {
        "L1": {"ceiling": None, "range": None},
        "L2": {"ceiling": None, "range": None},
    }
    assert (expected["billed"], expected["counted_against_quota"]) == (True, True)  # K8: the spend happened


def test_only_the_k7_golden_scripts_the_model() -> None:
    assert [e["example_id"] for e in ALL if e["rank"] != "llm"] == ["q04_openai_invalid_twice"]


def test_money_literals_are_the_hand_written_dossier_values() -> None:
    q01 = BY_ID["q01_happy_745"]["expected"]["lender_numbers"]
    assert q01 == {
        "L1": {"ceiling": 1277000, "range": {"lower": 368000, "upper": 1277000}},
        "L2": {"ceiling": 500000, "range": {"lower": 295000, "upper": 500000}},
    }
    f01 = BY_ID["f01_happy_offer_reported_emi"]["expected"]["lender_numbers"]
    assert f01["L1"] == {"ceiling": 1141000, "range": {"lower": 329000, "upper": 1141000}}
    assert f01["L2"] == {"ceiling": 500000, "range": {"lower": 256000, "upper": 500000}}
    f04 = BY_ID["f04_emi_four_branch"]["expected"]["lender_numbers"]
    assert f04["L1"] == {"ceiling": 890000, "range": {"lower": 256000, "upper": 890000}}
    assert BY_ID["f09_declared_emi_wins"]["expected"]["lender_numbers"]["L1"] == {
        "ceiling": 735000,
        "range": {"lower": 212000, "upper": 735000},
    }
    assert BY_ID["f08_ntc_full"]["expected"]["lender_numbers"] == {
        "L2": {"ceiling": 395000, "range": {"lower": 147000, "upper": 395000}}
    }


def test_the_pincode_union_golden_gates_l1_and_l2_but_never_l3() -> None:
    """The 12-month window: 400001 sits on an account reported 2025-09-01 and must not reach the gate."""
    expected = BY_ID["f03_pincode_union"]["expected"]
    assert expected["gated_lender_ids"] == ["L1", "L2"]
    assert expected["items_to_confirm"] == ["confirm with learner: an account last updated 2025-09-01"]


def test_confirm_item_goldens_carry_the_strings_the_emi_code_emits() -> None:
    assert BY_ID["f04_emi_four_branch"]["expected"]["items_to_confirm"] == [
        "past tenure, confirm with learner",
        "confirm with learner: an account last updated 2026-06-01",
    ]
    assert BY_ID["f05_stale_account"]["expected"]["items_to_confirm"] == [
        "confirm with learner: an account last updated 2026-06-01"
    ]
    assert BY_ID["f05_stale_account"]["expected"]["confidence"] == "medium"  # a stale account costs the 'high'
    assert BY_ID["f01_happy_offer_reported_emi"]["expected"]["confidence"] == "high"


# --------------------------------------------------------------------------- synthetic-only


def test_the_data_is_one_hundred_percent_synthetic() -> None:
    blob = json.dumps(ALL)
    for literal in PROBE_LITERALS:
        assert literal not in blob, f"probe literal {literal!r} in the goldens"
    pans = {request_of(e)["pan"] for e in ALL} | {
        str(step["request"]["pan"]) for e in ALL for step in e["pre_state"] if step["op"] == "run_check"
    }
    assert pans <= PAN_POOL, pans - PAN_POOL
    for example in ALL:
        requests = [request_of(example)] + [
            cast("dict[str, Any]", step["request"]) for step in example["pre_state"] if step["op"] == "run_check"
        ]
        for request in requests:
            assert (request["mobile"], request["name"]) == (MOBILE_OK, LC_TYPED_NAME), example["example_id"]


def request_of(example: GoldenExample) -> dict[str, Any]:
    return example["request"]


# --------------------------------------------------------------------------- independence (hand-written expectations)


def _imported_modules(module: Any) -> set[str]:
    tree = ast.parse(Path(inspect.getsourcefile(module) or "").read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_the_dataset_module_imports_no_code_that_could_produce_an_expected_value() -> None:
    """Expected values are literals: no derive / range / lender / redaction / graph / policy module is imported, so a
    golden can only agree with the code by being right, never by being computed by it."""
    ours = {m for m in _imported_modules(datasets) if m.startswith("eligibility_agent")}
    assert ours <= {"eligibility_agent.evals.worlds", "eligibility_agent.surepass.client"}, ours  # endpoint names only


def test_the_evaluators_import_only_the_leak_patterns_and_the_bureau_endpoint_names() -> None:
    ours = {m for m in _imported_modules(evaluators) if m.startswith("eligibility_agent")}
    assert ours == {"eligibility_agent.output.lc_view", "eligibility_agent.surepass.client"}, ours


# --------------------------------------------------------------------------- the LangSmith upload (modern API)


def _as_client(client: RecordingClient) -> DatasetsClient:
    return client


def real_client_conforms(client: Client) -> DatasetsClient:
    """Static pin (pyright): the real ``langsmith.Client`` satisfies the protocol the upload is written against."""
    return client


def test_build_creates_each_dataset_and_uploads_every_example_in_the_modern_form() -> None:
    client = RecordingClient()

    uploaded = build_langsmith_datasets(_as_client(client))

    assert uploaded == {QUICK_DATASET: 12, FULL_DATASET: 10}
    assert [k["dataset_name"] for k in client.of("create_dataset")] == [QUICK_DATASET, FULL_DATASET]
    uploads = client.of("create_examples")
    assert len(uploads) == 2
    for upload, name, count in zip(uploads, (QUICK_DATASET, FULL_DATASET), (12, 10), strict=True):
        assert set(upload) == {"dataset_id", "examples"}  # no deprecated inputs= / outputs= / ids= kwargs
        assert upload["dataset_id"] == client.ids[name]
        assert len(upload["examples"]) == count
        assert all(isinstance(e, ExampleCreate) for e in upload["examples"])


def test_uploaded_examples_carry_inputs_without_expected_and_the_golden_as_reference_outputs() -> None:
    client = RecordingClient()
    build_langsmith_datasets(_as_client(client))

    quick = {e.inputs["example_id"]: e for e in client.of("create_examples")[0]["examples"] if e.inputs}
    q05 = quick["q05_score_success_false"]
    golden = BY_ID["q05_score_success_false"]
    assert q05.inputs == dict(inputs_of(golden))
    assert "expected" not in (q05.inputs or {})
    assert q05.outputs == golden["expected"]
    assert q05.metadata == {"example_id": "q05_score_success_false", "kind": "quick"}


def test_example_ids_are_deterministic_so_a_rebuild_upserts_instead_of_duplicating() -> None:
    first, second = RecordingClient(), RecordingClient()
    build_langsmith_datasets(_as_client(first))
    build_langsmith_datasets(_as_client(second))

    ids = [[e.id for e in upload["examples"]] for upload in first.of("create_examples")]
    assert ids == [[e.id for e in upload["examples"]] for upload in second.of("create_examples")]
    flat = [i for batch in ids for i in batch]
    assert len(flat) == len(set(flat)) == 22 and all(i is not None for i in flat)


def test_an_existing_dataset_is_reused_not_recreated_and_overwrite_recreates_it() -> None:
    client = RecordingClient(existing=[QUICK_DATASET, FULL_DATASET])
    build_langsmith_datasets(_as_client(client))
    assert client.of("create_dataset") == [] and client.of("delete_dataset") == []
    assert len(client.of("read_dataset")) == 2

    replaced = RecordingClient(existing=[QUICK_DATASET, FULL_DATASET])
    build_langsmith_datasets(_as_client(replaced), overwrite=True)
    assert len(replaced.of("delete_dataset")) == 2 and len(replaced.of("create_dataset")) == 2


def test_the_upload_call_binds_to_the_installed_clients_create_examples_signature() -> None:
    """The installed langsmith (0.14.x) ``create_examples``: ``examples=`` is the modern parameter; the legacy kwargs
    (inputs / outputs / ids / ...) ride in ``**kwargs`` and are rejected alongside ``examples=``. Bind our exact call
    shape to the real signature so an API drift fails here, offline."""
    signature = inspect.signature(Client.create_examples)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert "examples" in signature.parameters
    client = RecordingClient()
    build_langsmith_datasets(_as_client(client))
    for upload in client.of("create_examples"):
        signature.bind(object(), **upload)  # TypeError if the shape no longer fits
        assert not {"inputs", "outputs", "ids", "splits", "source_run_ids"} & set(upload)
