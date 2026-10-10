"""Task 8 (part 1): lenders/gate.py, the pincode gate (PRD 9.1).

The gate decides which PUBLISHED lenders the model may rank. It is pure: rows in, ``(sorted GatedLenders, warnings)``
out. Criteria arrive as the lenders-table jsonb contract ``{field: {"value": ..., "status": known|missing|uncertain}}``
(Task 2's ``LenderRow.criteria``) and are normalised into ``LenderCriteria`` with ``field_status`` filled from it.
"""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from typing import Any

import pytest

from eligibility_agent.graph.state import GatedLender, LenderCriteria
from eligibility_agent.lenders import gate as gate_module
from eligibility_agent.lenders.gate import (
    REQUIRED_FOR_MATCHING,
    PublishedLenderRow,
    gate_lenders,
    is_matchable,
    uncertain_required_fields,
)

# --------------------------------------------------------------------------- builders


def jb(value: Any, status: str = "known") -> dict[str, Any]:
    return {"value": value, "status": status}


def criteria_jsonb(*, drop: tuple[str, ...] = (), **overrides: Any) -> dict[str, Any]:
    """Seed lender L1 in the schema.sql jsonb contract (note the admin-UI key names: min_score_floor, min_age,
    max_age_at_maturity)."""
    base: dict[str, Any] = {
        "min_score_floor": jb(650),
        "ntc_accepted": jb(False),
        "foir_limit": jb(0.55),
        "min_income": jb(25000),
        "min_age": jb(21),
        "max_age_at_maturity": jb(58),
        "rate_min_pct": jb(11.0),
        "rate_max_pct": jb(14.0),
        "tenure_min_months": jb(12),
        "tenure_max_months": jb(48),
        "loan_min": jb(50000),
        "loan_max": jb(1500000),
    }
    base.update(overrides)
    for key in drop:
        base.pop(key, None)
    return base


def row(
    lender_id: str = "L1",
    *,
    name: str | None = None,
    criteria: dict[str, Any] | None = None,
    matched: list[str] | None = None,
) -> PublishedLenderRow:
    return {
        "id": lender_id,
        "name": name if name is not None else f"Lender {lender_id}",
        "criteria": criteria if criteria is not None else criteria_jsonb(),
        "matched_pincodes": matched if matched is not None else ["560001"],
    }


def crit(**overrides: Any) -> LenderCriteria:
    fields: dict[str, Any] = {
        "score_floor": 650,
        "ntc_accepted": False,
        "foir_limit": 0.55,
        "min_income": 25000.0,
        "min_age_years": 21,
        "max_age_at_maturity_years": 58,
        "rate_min_pct": 11.0,
        "rate_max_pct": 14.0,
        "tenure_min_months": 12,
        "tenure_max_months": 48,
        "loan_min": 50000.0,
        "loan_max": 1500000.0,
        "policy_notes": None,
        "field_status": {},
    }
    fields.update(overrides)
    return LenderCriteria(**fields)


# --------------------------------------------------------------------------- REQUIRED_FOR_MATCHING / is_matchable


def test_required_for_matching_is_the_pinned_seven_tuple() -> None:
    assert REQUIRED_FOR_MATCHING == (
        "foir_limit",
        "rate_min_pct",
        "rate_max_pct",
        "tenure_min_months",
        "tenure_max_months",
        "loan_min",
        "loan_max",
    )
    assert isinstance(REQUIRED_FOR_MATCHING, tuple)


def test_is_matchable_all_known() -> None:
    assert is_matchable(crit()) == (True, [])


@pytest.mark.parametrize("field", REQUIRED_FOR_MATCHING)
def test_is_matchable_names_each_required_field_whose_value_is_none(field: str) -> None:
    ok, missing = is_matchable(crit(**{field: None}))
    assert ok is False
    assert missing == [field]


@pytest.mark.parametrize("field", REQUIRED_FOR_MATCHING)
def test_is_matchable_treats_status_missing_as_missing_even_with_a_value(field: str) -> None:
    ok, missing = is_matchable(crit(field_status={field: "missing"}))
    assert (ok, missing) == (False, [field])


def test_is_matchable_reports_every_missing_field_in_registry_order() -> None:
    ok, missing = is_matchable(crit(loan_max=None, foir_limit=None, tenure_min_months=None))
    assert ok is False
    assert missing == ["foir_limit", "tenure_min_months", "loan_max"]


def test_is_matchable_optional_fields_never_block() -> None:
    c = crit(min_income=None, min_age_years=None, max_age_at_maturity_years=None, policy_notes=None)
    assert is_matchable(c) == (True, [])


def test_is_matchable_floor_satisfied_by_score_floor_alone() -> None:
    assert is_matchable(crit(score_floor=700, ntc_accepted=None)) == (True, [])


def test_is_matchable_floor_satisfied_by_ntc_accepted_alone_and_false_counts_as_known() -> None:
    assert is_matchable(crit(score_floor=None, ntc_accepted=True)) == (True, [])
    assert is_matchable(crit(score_floor=None, ntc_accepted=False)) == (True, [])  # False is a known answer


def test_is_matchable_floor_missing_only_when_both_are_unknown() -> None:
    ok, missing = is_matchable(crit(score_floor=None, ntc_accepted=None))
    assert (ok, missing) == (False, ["score_floor"])


def test_is_matchable_floor_status_missing_on_both_counts_as_unknown() -> None:
    c = crit(field_status={"score_floor": "missing", "ntc_accepted": "missing"})
    assert is_matchable(c) == (False, ["score_floor"])
    # one of the two still known -> fine
    assert is_matchable(crit(field_status={"score_floor": "missing"})) == (True, [])


def test_is_matchable_uncertain_stays_matchable() -> None:
    c = crit(field_status={"foir_limit": "uncertain", "loan_max": "uncertain"})
    assert is_matchable(c) == (True, [])


# --------------------------------------------------------------------------- uncertain_required_fields


def test_uncertain_required_fields_lists_uncertain_required_fields_in_registry_order() -> None:
    c = crit(field_status={"loan_max": "uncertain", "foir_limit": "uncertain", "min_income": "uncertain"})
    assert uncertain_required_fields(c) == ["foir_limit", "loan_max"]  # min_income is optional: not listed


def test_uncertain_required_fields_includes_the_floor_pair() -> None:
    c = crit(field_status={"score_floor": "uncertain", "ntc_accepted": "uncertain", "rate_min_pct": "uncertain"})
    assert uncertain_required_fields(c) == ["rate_min_pct", "score_floor", "ntc_accepted"]


def test_uncertain_required_fields_empty_when_all_known() -> None:
    assert uncertain_required_fields(crit(field_status={"foir_limit": "known"})) == []
    assert uncertain_required_fields(crit()) == []


# --------------------------------------------------------------------------- gate_lenders: exclusion + warnings


def test_gate_keeps_a_complete_row_and_builds_the_gated_lender() -> None:
    gated, warnings = gate_lenders([row("L1", name="Alpha")], ["560001"])
    assert warnings == []
    assert len(gated) == 1
    g = gated[0]
    assert isinstance(g, GatedLender)
    assert (g.id, g.name, g.matched_pincodes) == ("L1", "Alpha", ["560001"])
    c = g.criteria
    # schema.sql key names are mapped onto the LenderCriteria field names
    assert c.score_floor == 650 and c.ntc_accepted is False
    assert c.min_age_years == 21 and c.max_age_at_maturity_years == 58
    assert c.foir_limit == 0.55 and c.min_income == 25000.0
    assert (c.rate_min_pct, c.rate_max_pct) == (11.0, 14.0)
    assert (c.tenure_min_months, c.tenure_max_months) == (12, 48)
    assert (c.loan_min, c.loan_max) == (50000.0, 1500000.0)
    assert c.policy_notes is None
    assert g.rate_band_label == "11 to 14%"
    assert g.tenure_label == "12 to 48 months"


# --- the three two-spelling fields: db/schema.sql spelling is canonical, the LenderCriteria name is a fallback only

SCHEMA_TO_CANONICAL = {
    "min_score_floor": "score_floor",
    "min_age": "min_age_years",
    "max_age_at_maturity": "max_age_at_maturity_years",
}
DISTINCT = {"min_score_floor": 700, "min_age": 23, "max_age_at_maturity": 60}  # differ from the seed (650 / 21 / 58)


def test_gate_schema_spelling_only_row_matches() -> None:
    crit_json = criteria_jsonb()
    crit_json.update({k: jb(v) for k, v in DISTINCT.items()})
    assert not set(SCHEMA_TO_CANONICAL.values()) & set(crit_json)  # no fallback key anywhere in this row
    gated, warnings = gate_lenders([row("L1", criteria=crit_json)], ["560001"])
    assert warnings == []
    c = gated[0].criteria
    assert (c.score_floor, c.min_age_years, c.max_age_at_maturity_years) == (700, 23, 60)
    assert [c.field_status[f] for f in SCHEMA_TO_CANONICAL.values()] == ["known", "known", "known"]


def test_gate_fallback_spelling_only_row_matches() -> None:
    crit_json = criteria_jsonb(drop=tuple(SCHEMA_TO_CANONICAL))
    crit_json.update({SCHEMA_TO_CANONICAL[k]: jb(v) for k, v in DISTINCT.items()})
    assert not set(SCHEMA_TO_CANONICAL) & set(crit_json)  # no schema key anywhere in this row
    gated, warnings = gate_lenders([row("L1", criteria=crit_json)], ["560001"])
    assert warnings == []
    c = gated[0].criteria
    assert (c.score_floor, c.min_age_years, c.max_age_at_maturity_years) == (700, 23, 60)


def test_gate_both_spellings_present_and_equal_matches() -> None:
    crit_json = criteria_jsonb()
    crit_json.update({k: jb(v) for k, v in DISTINCT.items()})
    crit_json.update({SCHEMA_TO_CANONICAL[k]: jb(v) for k, v in DISTINCT.items()})
    gated, warnings = gate_lenders([row("L1", criteria=crit_json)], ["560001"])
    assert warnings == []
    c = gated[0].criteria
    assert (c.score_floor, c.min_age_years, c.max_age_at_maturity_years) == (700, 23, 60)


def test_gate_both_spellings_equal_across_int_and_float_is_still_equal() -> None:
    crit_json = criteria_jsonb(min_score_floor=jb(700))
    crit_json["score_floor"] = jb(700.0)
    gated, warnings = gate_lenders([row("L1", criteria=crit_json)], ["560001"])
    assert warnings == []
    assert gated[0].criteria.score_floor == 700


@pytest.mark.parametrize("schema_key", list(SCHEMA_TO_CANONICAL))
def test_gate_both_spellings_conflicting_is_invalid_criteria_and_excluded(schema_key: str) -> None:
    canonical = SCHEMA_TO_CANONICAL[schema_key]
    crit_json = criteria_jsonb()
    crit_json[schema_key] = jb(DISTINCT[schema_key])
    crit_json[canonical] = jb(DISTINCT[schema_key] + 1)  # same field, different value under the other spelling
    gated, warnings = gate_lenders([row("L7", criteria=crit_json), row("L1")], ["560001"])
    assert [g.id for g in gated] == ["L1"]  # fail closed: the conflicting row is out, the clean one still gates
    assert warnings == [f"lender L7 excluded: invalid criteria {canonical}"]


def test_gate_conflict_is_invalid_whichever_spelling_carries_which_value() -> None:
    # neither spelling may silently win: swapping which one holds the 'bigger' value gives the same exclusion
    for schema_value, fallback_value in ((650, 700), (700, 650)):
        crit_json = criteria_jsonb(min_score_floor=jb(schema_value))
        crit_json["score_floor"] = jb(fallback_value)
        gated, warnings = gate_lenders([row("L7", criteria=crit_json)], ["560001"])
        assert gated == []
        assert warnings == ["lender L7 excluded: invalid criteria score_floor"]


def test_gate_both_spellings_with_the_same_value_but_different_status_is_a_conflict() -> None:
    crit_json = criteria_jsonb(min_age=jb(23, "known"))
    crit_json["min_age_years"] = jb(23, "uncertain")
    gated, warnings = gate_lenders([row("L7", criteria=crit_json)], ["560001"])
    assert gated == []
    assert warnings == ["lender L7 excluded: invalid criteria min_age_years"]


def test_gate_schema_missing_against_a_known_fallback_is_a_conflict_not_a_silent_win() -> None:
    crit_json = criteria_jsonb(min_score_floor=jb(None, "missing"))
    crit_json["score_floor"] = jb(700)
    gated, warnings = gate_lenders([row("L7", criteria=crit_json)], ["560001"])
    assert gated == []
    assert warnings == ["lender L7 excluded: invalid criteria score_floor"]


def test_gate_both_spellings_agreeing_on_missing_is_fine_and_still_missing() -> None:
    crit_json = criteria_jsonb(min_score_floor=jb(None, "missing"), ntc_accepted=jb(True))
    crit_json["score_floor"] = jb(None, "missing")
    gated, warnings = gate_lenders([row("L1", criteria=crit_json)], ["560001"])
    assert warnings == []
    assert gated[0].criteria.score_floor is None
    assert gated[0].criteria.field_status["score_floor"] == "missing"


def test_gate_both_spellings_agreeing_on_uncertain_carries_the_uncertain_status() -> None:
    crit_json = criteria_jsonb(min_score_floor=jb(700, "uncertain"))
    crit_json["score_floor"] = jb(700, "uncertain")
    gated, warnings = gate_lenders([row("L1", criteria=crit_json)], ["560001"])
    assert warnings == []
    assert gated[0].criteria.field_status["score_floor"] == "uncertain"
    assert uncertain_required_fields(gated[0].criteria) == ["score_floor"]


@pytest.mark.parametrize("malformed_side", ["schema", "fallback"])
def test_gate_one_malformed_spelling_poisons_the_field_even_when_the_other_is_clean(malformed_side: str) -> None:
    crit_json = criteria_jsonb(min_score_floor=jb(700))
    crit_json["score_floor"] = jb(700)
    crit_json["min_score_floor" if malformed_side == "schema" else "score_floor"] = 700  # bare, not {value, status}
    gated, warnings = gate_lenders([row("L7", criteria=crit_json)], ["560001"])
    assert gated == []
    assert warnings == ["lender L7 excluded: invalid criteria score_floor"]


def test_gate_single_spelling_fields_have_no_alias_surface() -> None:
    # foir_limit has exactly one spelling; an unrelated alias-looking key is ignored, never read
    crit_json = criteria_jsonb(drop=("foir_limit",))
    crit_json["foir"] = jb(0.5)
    gated, warnings = gate_lenders([row("L7", criteria=crit_json)], ["560001"])
    assert gated == []
    assert warnings == ["lender L7 excluded: missing foir_limit"]


def test_gate_excludes_a_lender_missing_a_required_field_with_the_pinned_warning_string() -> None:
    bad = row("L2", criteria=criteria_jsonb(drop=("foir_limit",)))
    gated, warnings = gate_lenders([row("L1"), bad], ["560001"])
    assert [g.id for g in gated] == ["L1"]
    assert warnings == ["lender L2 excluded: missing foir_limit"]


def test_gate_status_missing_excludes_even_when_a_value_is_present() -> None:
    bad = row("L2", criteria=criteria_jsonb(loan_max=jb(1500000, "missing")))
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert warnings == ["lender L2 excluded: missing loan_max"]


def test_gate_value_none_with_status_known_is_missing() -> None:
    bad = row("L2", criteria=criteria_jsonb(rate_min_pct=jb(None, "known")))
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert warnings == ["lender L2 excluded: missing rate_min_pct"]


def test_gate_warning_lists_every_missing_field_comma_separated_in_registry_order() -> None:
    bad = row("L9", criteria=criteria_jsonb(drop=("loan_max", "foir_limit", "tenure_max_months")))
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert warnings == ["lender L9 excluded: missing foir_limit, tenure_max_months, loan_max"]


def test_gate_floor_unknown_on_both_excludes_with_score_floor_named() -> None:
    bad = row("L3", criteria=criteria_jsonb(min_score_floor=jb(None, "missing"), ntc_accepted=jb(None, "missing")))
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert warnings == ["lender L3 excluded: missing score_floor"]


def test_gate_floor_known_via_ntc_accepted_alone_stays_in() -> None:
    ok = row("L3", criteria=criteria_jsonb(min_score_floor=jb(None, "missing"), ntc_accepted=jb(True)))
    gated, warnings = gate_lenders([ok], ["560001"])
    assert warnings == []
    assert [g.id for g in gated] == ["L3"]
    assert gated[0].criteria.score_floor is None and gated[0].criteria.ntc_accepted is True


def test_gate_absent_required_key_is_missing_and_absent_optional_keys_are_fine() -> None:
    sparse = criteria_jsonb(drop=("min_income", "min_age", "max_age_at_maturity"))
    gated, warnings = gate_lenders([row("L1", criteria=sparse)], ["560001"])
    assert warnings == []
    c = gated[0].criteria
    assert (c.min_income, c.min_age_years, c.max_age_at_maturity_years) == (None, None, None)
    assert c.field_status["min_income"] == "missing"


def test_gate_empty_or_non_dict_criteria_excludes_with_every_required_field_named() -> None:
    expected = "lender L5 excluded: missing " + ", ".join(REQUIRED_FOR_MATCHING) + ", score_floor"
    empties: tuple[dict[str, Any] | None, ...] = ({}, None)
    for crit_json in empties:
        bad: Any = row("L5")
        bad["criteria"] = crit_json  # simulate an empty / NULL jsonb column
        gated, warnings = gate_lenders([bad], ["560001"])
        assert gated == []
        assert warnings == [expected]


def test_gate_never_aborts_on_a_bad_row_the_good_rows_still_come_back() -> None:
    rows = [row("L1"), row("L2", criteria=criteria_jsonb(drop=("loan_min",))), row("L3")]
    gated, warnings = gate_lenders(rows, ["560001"])
    assert [g.id for g in gated] == ["L1", "L3"]
    assert warnings == ["lender L2 excluded: missing loan_min"]


# --------------------------------------------------------------------------- 'uncertain' stays IN


def test_uncertain_required_field_stays_in_and_rides_as_field_status() -> None:
    uncertain = row("L4", criteria=criteria_jsonb(foir_limit=jb(0.5, "uncertain"), loan_max=jb(900000, "uncertain")))
    gated, warnings = gate_lenders([uncertain], ["560001"])
    assert warnings == []
    assert [g.id for g in gated] == ["L4"]
    g = gated[0]
    assert g.criteria.foir_limit == 0.5  # the value is usable
    assert g.criteria.field_status["foir_limit"] == "uncertain"
    assert g.criteria.field_status["loan_max"] == "uncertain"
    assert g.criteria.field_status["rate_min_pct"] == "known"
    assert uncertain_required_fields(g.criteria) == ["foir_limit", "loan_max"]


def test_field_status_is_total_over_the_twelve_criteria_fields() -> None:
    gated, _ = gate_lenders([row("L1", criteria=criteria_jsonb(drop=("min_income",)))], ["560001"])
    assert set(gated[0].criteria.field_status) == {
        "score_floor",
        "ntc_accepted",
        "foir_limit",
        "min_income",
        "min_age_years",
        "max_age_at_maturity_years",
        "rate_min_pct",
        "rate_max_pct",
        "tenure_min_months",
        "tenure_max_months",
        "loan_min",
        "loan_max",
    }
    assert gated[0].criteria.field_status["min_income"] == "missing"


def test_an_invalid_status_value_excludes_the_lender_with_a_warning() -> None:
    bad = row("L6", criteria=criteria_jsonb(foir_limit=jb(0.5, "maybe")))
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert warnings == ["lender L6 excluded: invalid criteria foir_limit"]


@pytest.mark.parametrize("junk", ["abc", float("nan"), float("inf"), [1, 2], {"x": 1}])
def test_an_unusable_value_excludes_the_lender_with_an_invalid_criteria_warning(junk: Any) -> None:
    bad = row("L6", criteria=criteria_jsonb(foir_limit=jb(junk)))
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert len(warnings) == 1
    assert warnings[0].startswith("lender L6 excluded: invalid criteria")
    assert "foir_limit" in warnings[0]


def test_a_bare_non_object_criteria_entry_is_invalid_not_silently_known() -> None:
    bad = row("L6", criteria=criteria_jsonb(foir_limit=0.5))  # not {value, status}
    gated, warnings = gate_lenders([bad], ["560001"])
    assert gated == []
    assert warnings == ["lender L6 excluded: invalid criteria foir_limit"]


# --------------------------------------------------------------------------- pincode re-filter


def test_candidate_pincodes_are_re_filtered_through_the_pattern() -> None:
    r = row("L1", matched=["560001", "012345", "56000", "5600011", "abcdef", " 560001"])
    gated, _ = gate_lenders([r], ["560001", "012345", "56000", "5600011", "abcdef", " 560001", ""])
    assert [g.matched_pincodes for g in gated] == [["560001"]]


def test_matched_pincodes_are_restricted_to_the_candidates_sorted_and_deduped() -> None:
    r = row("L1", matched=["700001", "560001", "560001", "110001", "400001"])
    gated, _ = gate_lenders([r], ["560001", "110001", "999999"])
    assert gated[0].matched_pincodes == ["110001", "560001"]


def test_a_lender_with_no_valid_candidate_overlap_is_out_without_a_warning() -> None:
    gated, warnings = gate_lenders([row("L1", matched=["700001"]), row("L2", matched=["560001"])], ["560001"])
    assert [g.id for g in gated] == ["L2"]
    assert warnings == []  # not a data defect: this lender simply does not serve the applicant's pincodes


def test_no_candidate_pincodes_means_nothing_is_gated() -> None:
    assert gate_lenders([row("L1")], []) == ([], [])
    assert gate_lenders([row("L1")], ["12345", "abcdef"]) == ([], [])


def test_pincode_filter_runs_before_validation_so_an_unrelated_bad_row_does_not_warn() -> None:
    bad_elsewhere = row("L2", criteria=criteria_jsonb(drop=("foir_limit",)), matched=["700001"])
    gated, warnings = gate_lenders([bad_elsewhere], ["560001"])
    assert (gated, warnings) == ([], [])


def test_empty_rows() -> None:
    assert gate_lenders([], ["560001"]) == ([], [])


# --------------------------------------------------------------------------- determinism


def test_output_is_sorted_by_lender_id_whatever_the_input_order() -> None:
    rows = [row("L3"), row("L1"), row("L10"), row("L2")]
    gated, _ = gate_lenders(rows, ["560001"])
    assert [g.id for g in gated] == ["L1", "L10", "L2", "L3"]  # plain string order, deterministic


def test_output_is_identical_for_every_input_permutation() -> None:
    rows = [
        row("L2"),
        row("L1"),
        row("L4", criteria=criteria_jsonb(drop=("loan_min",))),
        row("L3", criteria=criteria_jsonb(foir_limit=jb(0.4, "uncertain"))),
    ]
    forward = gate_lenders(rows, ["560001", "110001"])
    backward = gate_lenders(list(reversed(rows)), ["110001", "560001", "560001"])
    assert forward == backward
    assert [g.id for g in forward[0]] == ["L1", "L2", "L3"]


def test_warnings_are_in_deterministic_lender_id_order() -> None:
    rows = [
        row("L3", criteria=criteria_jsonb(drop=("loan_max",))),
        row("L1", criteria=criteria_jsonb(drop=("foir_limit",))),
        row("L2", criteria=criteria_jsonb(drop=("loan_min",))),
    ]
    _, warnings = gate_lenders(rows, ["560001"])
    assert warnings == [
        "lender L1 excluded: missing foir_limit",
        "lender L2 excluded: missing loan_min",
        "lender L3 excluded: missing loan_max",
    ]


def test_a_duplicate_lender_id_keeps_the_first_row_and_warns() -> None:
    first = row("L1", name="First")
    second = row("L1", name="Second")
    gated, warnings = gate_lenders([first, second], ["560001"])
    assert [(g.id, g.name) for g in gated] == [("L1", "First")]
    assert warnings == ["lender L1 excluded: duplicate row"]


# --------------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("rmin", "rmax", "label"),
    [
        (11.0, 14.0, "11 to 14%"),
        (12.5, 16.0, "12.5 to 16%"),
        (10.75, 18.25, "10.75 to 18.25%"),
        (0.0, 12.0, "0 to 12%"),
        (9, 11, "9 to 11%"),
    ],
)
def test_rate_band_label_uses_g_format(rmin: float, rmax: float, label: str) -> None:
    r = row("L1", criteria=criteria_jsonb(rate_min_pct=jb(rmin), rate_max_pct=jb(rmax)))
    gated, _ = gate_lenders([r], ["560001"])
    assert gated[0].rate_band_label == label


def test_tenure_label() -> None:
    r = row("L1", criteria=criteria_jsonb(tenure_min_months=jb(6), tenure_max_months=jb(60)))
    gated, _ = gate_lenders([r], ["560001"])
    assert gated[0].tenure_label == "6 to 60 months"


def test_integral_float_tenures_and_ints_normalise_to_clean_labels() -> None:
    r = row("L1", criteria=criteria_jsonb(tenure_min_months=jb(12), tenure_max_months=jb(36.0)))
    gated, _ = gate_lenders([r], ["560001"])
    assert gated[0].tenure_label == "12 to 36 months"


# --------------------------------------------------------------------------- policy notes, purity


def test_policy_notes_ride_through_from_the_row_when_present() -> None:
    r = row("L1")
    r["policy_notes"] = "Salaried only"
    gated, _ = gate_lenders([r], ["560001"])
    assert gated[0].criteria.policy_notes == "Salaried only"


def test_policy_notes_default_to_none_when_the_row_carries_none() -> None:
    gated, _ = gate_lenders([row("L1")], ["560001"])
    assert gated[0].criteria.policy_notes is None
    r = row("L2")
    r["policy_notes"] = ""
    gated, _ = gate_lenders([r], ["560001"])
    assert gated[0].criteria.policy_notes is None  # the table column defaults to '': no notes


def test_gate_does_not_mutate_its_inputs() -> None:
    rows = [row("L2"), row("L1", criteria=criteria_jsonb(foir_limit=jb(0.5, "uncertain")))]
    before = copy.deepcopy(rows)
    pins = ["560001", "110001"]
    gate_lenders(rows, pins)
    assert rows == before
    assert pins == ["560001", "110001"]


def test_gate_accepts_any_sequence_types() -> None:
    gated, _ = gate_lenders(tuple([row("L1")]), ("560001",))
    assert [g.id for g in gated] == ["L1"]


def test_gate_module_is_pure_stdlib_plus_state_and_range_only() -> None:
    tree = ast.parse(Path(gate_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module if node.module.startswith("eligibility_agent") else node.module.split(".")[0])
    stdlib = {"__future__", "collections", "math", "typing", "re", "pydantic"}  # pydantic: ValidationError only
    sanctioned = {"eligibility_agent.graph.state", "eligibility_agent.derive.range"}
    assert imported <= stdlib | sanctioned, imported - stdlib - sanctioned
    source = Path(gate_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\.now\(|utcnow|time\.time|\.today\(|random|uuid", source)
