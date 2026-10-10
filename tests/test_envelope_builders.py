"""Task 5 fixtures: testing/envelopes.py, testing/scenarios.py and testing/clock.py self-tests.

The synthetic builders are the ONLY source of SurePass-shaped data for tests and evals, so they are pinned hard:
dense CAIS rows (23 scalar field-map keys, '' = not reported), the v1.1 NTC vectors, the validation-error shape,
the six-account EMI table, and the PROBE-LITERAL GUARD (no value from the real probe files may ever appear in a
builder output or in the testing package source).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast, get_args
from zoneinfo import ZoneInfo

import pytest
from repo_builders import make_check

import eligibility_agent
from eligibility_agent.db.repo import MemoryRepo
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    SCORE_ENDPOINT,
    FakeSurePassClient,
    SurePassAuthError,
    SurePassError,
    SurePassResult,
    SurePassTimeout,
    SurePassUpstreamError,
)
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing import scenarios as scn
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.envelopes import (
    CAIS_NESTED_KEYS,
    CAIS_SCALAR_KEYS,
    EnvelopePair,
    cais_account,
    experian_report_envelope,
    experian_score_envelope,
    failure_envelope,
    pan_comprehensive_envelope,
    validation_error_envelope,
)

# The probe-derived literals that must NEVER appear in any synthetic value (PRD v1.1 / plan Global Constraints).
FORBIDDEN_LITERALS = ("791111", "201301", "560071", "16559", "2434004", "138696", "109433")
# Extra guard: the probe's holder DOB values (field map section 6) and its rate string.
EXTRA_PROBE_LITERALS = ("19951121", "19910408", "19951120", "1995-11-21", "1991-04-08", "34.900")

TESTING_DIR = Path(eligibility_agent.__file__).parent / "testing"
PACKAGE_DIR = Path(eligibility_agent.__file__).parent

T0 = datetime(2026, 10, 10, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def _body(pair: EnvelopePair) -> dict[str, Any]:
    return pair[1]


def _data(pair: EnvelopePair) -> dict[str, Any]:
    return cast("dict[str, Any]", pair[1]["data"])


def _credit_report(pair: EnvelopePair) -> dict[str, Any]:
    return cast("dict[str, Any]", _data(pair)["credit_report"])


def _details(pair: EnvelopePair) -> list[dict[str, Any]]:
    cais = cast("dict[str, Any]", _credit_report(pair)["CAIS_Account"])
    return cast("list[dict[str, Any]]", cais["CAIS_Account_DETAILS"])


# --------------------------------------------------------------------------- FrozenClock


def test_frozen_clock_returns_the_aware_instant_normalised_to_utc() -> None:
    clock = FrozenClock(T0)
    now = clock.now()
    assert now == T0
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)  # stored as UTC even though T0 was constructed in IST
    assert now.tzinfo == timezone.utc


def test_frozen_clock_rejects_a_naive_start() -> None:
    with pytest.raises(ValueError, match="aware"):
        FrozenClock(datetime(2026, 10, 10, 10, 0))


def test_frozen_clock_advance_and_set_now() -> None:
    clock = FrozenClock(T0)
    clock.advance(timedelta(minutes=11))
    assert clock.now() == T0 + timedelta(minutes=11)
    target = datetime(2026, 11, 1, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    clock.set_now(target)
    assert clock.now() == target
    assert clock.now().utcoffset() == timedelta(0)


def test_frozen_clock_set_now_rejects_a_naive_instant() -> None:
    clock = FrozenClock(T0)
    with pytest.raises(ValueError, match="aware"):
        clock.set_now(datetime(2026, 10, 10, 10, 0))
    assert clock.now() == T0  # unchanged by the refused call


def test_frozen_clock_now_is_a_valid_memory_repo_clock() -> None:
    # `clock.now` (a bound method) satisfies the db.repo Clock = Callable[[], datetime] seam.
    clock = FrozenClock(T0)
    repo = MemoryRepo(clock=clock.now)
    assert repo.list_checks() == []


# --------------------------------------------------------------------------- dense CAIS account builder


def test_cais_account_has_all_23_scalar_field_map_keys_and_the_four_nested_collections() -> None:
    account = cais_account()
    assert len(CAIS_SCALAR_KEYS) == 23
    assert len(set(CAIS_SCALAR_KEYS)) == 23
    assert set(CAIS_NESTED_KEYS) == {
        "CAIS_Account_History",
        "CAIS_Holder_Address_Details",
        "CAIS_Holder_Phone_Details",
        "CAIS_Holder_Details",
    }
    assert set(account) == set(CAIS_SCALAR_KEYS) | set(CAIS_NESTED_KEYS)
    assert len(account) == 27  # 23 scalars + 4 nested collections: nothing extra, nothing missing
    # exact field-map spellings the adapters read
    for key in (
        "Scheduled_Monthly_Payment_Amount",
        "Date_Closed",
        "Account_Status",
        "Account_Type",
        "Portfolio_Type",
        "Current_Balance",
        "Amount_Past_Due",
        "Repayment_Tenure",
        "Terms_Duration",
        "Terms_Frequency",
        "Rate_of_Interest",
        "Open_Date",
        "Date_Reported",
        "Written_off_Settled_Status",
        "Account_Number",
    ):
        assert key in CAIS_SCALAR_KEYS


def test_cais_account_scalars_are_all_strings_and_unset_ones_are_the_empty_sentinel() -> None:
    account = cais_account()
    for key in CAIS_SCALAR_KEYS:
        assert isinstance(account[key], str), key  # every bureau leaf is a string
    for key in (
        "Date_Closed",
        "Scheduled_Monthly_Payment_Amount",  # the NORM: Experian reports no EMI
        "Rate_of_Interest",
        "Terms_Duration",
        "Credit_Limit_Amount",
        "Settlement_Amount",
        "SuitFiled_WilfulDefault",
        "Written_off_Settled_Status",
    ):
        assert account[key] == "", key
    assert account["Repayment_Tenure"] == "0"  # '0' = not reported on this field
    assert account["Terms_Frequency"] == "M"
    assert account["Account_Status"] == "11"  # open by default


def test_cais_account_json_round_trips() -> None:
    account = cais_account(holder_zips=("560001", "20"), history=(("2026", "08", "45"),))
    assert json.loads(json.dumps(account)) == account


@pytest.mark.parametrize("padded", ["024", "048", "060"])
def test_cais_account_keeps_terms_duration_zero_padded_as_a_string(padded: str) -> None:
    assert cais_account(terms_duration=padded)["Terms_Duration"] == padded
    assert cais_account()["Terms_Duration"] == ""


def test_cais_account_kwargs_drive_the_matching_fields() -> None:
    account = cais_account(
        portfolio_type="R",
        scheduled_emi="3500",
        current_balance="40000",
        repayment_tenure="12",
        terms_duration="036",
        terms_frequency="Q",
        rate_of_interest="13.500",
        open_date="20240115",
        date_reported="20260920",
        written_off_settled_status="99",
        amount_past_due="1800",
        account_status="13",
        date_closed="20250110",
        account_type="10",
        account_number="SYNTHACC0042",
    )
    assert account["Portfolio_Type"] == "R"
    assert account["Scheduled_Monthly_Payment_Amount"] == "3500"
    assert account["Current_Balance"] == "40000"
    assert account["Repayment_Tenure"] == "12"
    assert account["Terms_Duration"] == "036"
    assert account["Terms_Frequency"] == "Q"
    assert account["Rate_of_Interest"] == "13.500"
    assert account["Open_Date"] == "20240115"
    assert account["Date_Reported"] == "20260920"
    assert account["Written_off_Settled_Status"] == "99"
    assert account["Amount_Past_Due"] == "1800"
    assert account["Account_Status"] == "13"
    assert account["Date_Closed"] == "20250110"
    assert account["Account_Type"] == "10"
    assert account["Account_Number"] == "SYNTHACC0042"


def test_cais_account_holder_zip_and_holder_zips_make_address_rows() -> None:
    single = cais_account(holder_zip="110001")
    rows = cast("list[dict[str, Any]]", single["CAIS_Holder_Address_Details"])
    assert [r["ZIP_Postal_Code_non_normalized"] for r in rows] == ["110001"]
    many = cais_account(holder_zips=("110001", "20", ""))
    rows = cast("list[dict[str, Any]]", many["CAIS_Holder_Address_Details"])
    assert [r["ZIP_Postal_Code_non_normalized"] for r in rows] == ["110001", "20", ""]
    for row in rows:  # dense address rows too
        assert set(row) == {
            "First_Line_Of_Address_non_normalized",
            "City_non_normalized",
            "State_non_normalized",
            "ZIP_Postal_Code_non_normalized",
            "Address_indicator_non_normalized",
            "Residence_code_non_normalized",
        }


def test_cais_account_holder_details_carry_a_deliberately_contradictory_second_dob_row() -> None:
    holders = cast("list[dict[str, Any]]", cais_account()["CAIS_Holder_Details"])
    assert len(holders) == 2
    dobs = [h["Date_of_birth"] for h in holders]
    assert dobs[0] != dobs[1]  # the trap: any age taken from these rows disagrees with itself
    assert all(re.fullmatch(r"[0-9]{8}", d) for d in dobs)
    custom = cast("list[dict[str, Any]]", cais_account(holder_dob="19700202")["CAIS_Holder_Details"])
    assert custom[0]["Date_of_birth"] == "19700202"
    assert len(custom) == 2


def test_cais_account_history_rows() -> None:
    history = cast(
        "list[dict[str, Any]]", cais_account(history=(("2026", "08", "45"), ("2026", "09", "0")))["CAIS_Account_History"]
    )
    assert [(h["Year"], h["Month"], h["Days_Past_Due"]) for h in history] == [("2026", "08", "45"), ("2026", "09", "0")]
    assert cais_account(history=())["CAIS_Account_History"] == []


# --------------------------------------------------------------------------- envelope shapes


def test_envelope_pair_statuses_are_ints_and_bodies_are_dicts() -> None:
    pairs = [
        pan_comprehensive_envelope(),
        experian_score_envelope(),
        experian_score_envelope(success=False),
        experian_report_envelope(accounts=[cais_account()]),
        failure_envelope(),
        validation_error_envelope(),
    ]
    for status, body in pairs:
        assert isinstance(status, int) and not isinstance(status, bool)
        assert isinstance(body, dict)
    assert [p[0] for p in pairs] == [200, 200, 200, 200, 422, 422]


def test_pan_comprehensive_envelope_shape() -> None:
    pair = pan_comprehensive_envelope()
    status, body = pair
    assert status == 200
    assert set(body) == {"data", "status_code", "success", "message", "message_code"}
    assert body["success"] is True
    assert body["status_code"] == 200
    data = _data(pair)
    address = cast("dict[str, Any]", data["address"])
    assert address["zip"] == "560001"
    assert data["aadhaar_linked"] is True  # a real bool, never the string "true"
    assert data["status"] == "valid"
    assert data["category"] == "person"
    assert data["dob"] == "1998-04-12"
    assert data["masked_aadhaar"] == "XXXXXXXX0000"
    assert isinstance(data["client_id"], str) and data["client_id"]


def test_pan_comprehensive_envelope_omits_aadhaar_linked_when_none_and_keeps_false() -> None:
    assert "aadhaar_linked" not in _data(pan_comprehensive_envelope(aadhaar_linked=None))
    assert _data(pan_comprehensive_envelope(aadhaar_linked=False))["aadhaar_linked"] is False


def test_pan_comprehensive_envelope_kwargs() -> None:
    data = _data(
        pan_comprehensive_envelope(
            pan="ABCPE1234F",
            full_name="ASHA TESTVERMA",
            status="invalid",
            category="company",
            dob="1990-05-06",
            zip_code="",
            client_id="pan_x",
        )
    )
    assert data["pan_number"] == "ABCPE1234F"
    assert data["full_name"] == "ASHA TESTVERMA"
    assert data["status"] == "invalid"
    assert data["category"] == "company"
    assert data["dob"] == "1990-05-06"
    assert cast("dict[str, Any]", data["address"])["zip"] == ""
    assert data["client_id"] == "pan_x"


def test_validation_error_envelope_has_no_success_and_no_data_keys() -> None:
    status, body = validation_error_envelope()
    assert status == 422
    assert "success" not in body
    assert "data" not in body
    assert set(body) == {"errors", "message"}
    assert body["message"] == "Input payload validation failed"
    assert validation_error_envelope(field="pan", msg="bad")[1]["errors"] == {"pan": "bad"}


def test_failure_envelope_shape() -> None:
    status, body = failure_envelope()
    assert status == 422
    assert body["success"] is False
    assert body["data"] == {}
    assert body["status_code"] == 422
    status500, body500 = failure_envelope(http_status=500, message_code="internal_error", message=None)
    assert status500 == 500 and body500["message_code"] == "internal_error" and body500["message"] is None


def test_score_envelope_default_is_a_three_digit_hit() -> None:
    pair = experian_score_envelope()
    assert pair[0] == 200 and _body(pair)["success"] is True
    assert _data(pair)["credit_score"] == "745"
    report = _credit_report(pair)
    assert cast("dict[str, Any]", report["CreditProfileHeader"])["ReportDate"] == "20261001"
    assert cast("dict[str, Any]", report["SCORE"])["FCIREXScore"] == "745"


def test_score_envelope_success_false_is_the_service_failure_vector_not_ntc() -> None:
    status, body = experian_score_envelope(success=False)
    assert status == 200  # HTTP 200 with success:false: the client must raise, never read this as NTC
    assert body["success"] is False
    assert body["message_code"] == "no_record_found"
    assert "credit_score" not in cast("dict[str, Any]", body["data"])


@pytest.mark.parametrize("value", ["", "NH", "0745"])
def test_score_envelope_ntc_credit_score_kwargs_are_carried_verbatim(value: str) -> None:
    pair = experian_score_envelope(credit_score=value)
    assert pair[0] == 200 and _body(pair)["success"] is True  # NTC is success:true with a non-3-digit score
    assert _data(pair)["credit_score"] == value


def test_score_envelope_omit_score_drops_the_key_entirely() -> None:
    pair = experian_score_envelope(omit_score=True)
    assert pair[0] == 200 and _body(pair)["success"] is True
    assert "credit_score" not in _data(pair)


def test_score_envelope_report_date_kwarg() -> None:
    pair = experian_score_envelope(report_date="20260615")
    header = cast("dict[str, Any]", _credit_report(pair)["CreditProfileHeader"])
    assert header["ReportDate"] == "20260615"
    no_date = experian_score_envelope(report_date=None)
    assert "ReportDate" not in cast("dict[str, Any]", _credit_report(no_date)["CreditProfileHeader"])


def test_report_envelope_structure_and_counts() -> None:
    accounts = [cais_account(), cais_account(account_number="SYNTHACC0002")]
    pair = experian_report_envelope(accounts=accounts, credit_score="702", caps_30="1", caps_90="2", caps_180="3")
    assert pair[0] == 200 and _body(pair)["success"] is True
    assert _data(pair)["credit_score"] == "702"
    report = _credit_report(pair)
    assert cast("dict[str, Any]", report["CreditProfileHeader"])["ReportDate"] == "20261001"
    assert _details(pair) == accounts
    summary = cast("dict[str, Any]", report["TotalCAPS_Summary"])
    assert (summary["TotalCAPSLast30Days"], summary["TotalCAPSLast90Days"], summary["TotalCAPSLast180Days"]) == (
        "1",
        "2",
        "3",
    )
    credit_account = cast("dict[str, Any]", cast("dict[str, Any]", report["CAIS_Account"])["CAIS_Summary"])
    assert cast("dict[str, Any]", credit_account["Credit_Account"])["CreditAccountTotal"] == "2"


def test_report_envelope_defaults_and_failure_and_ntc_vectors() -> None:
    default = experian_report_envelope()
    assert _details(default) == []
    assert _data(default)["credit_score"] == "745"
    status, body = experian_report_envelope(success=False)
    assert status == 200 and body["success"] is False and body["message_code"] == "no_record_found"
    assert _data(experian_report_envelope(credit_score="NH"))["credit_score"] == "NH"
    assert "credit_score" not in _data(experian_report_envelope(omit_score=True))
    assert "ReportDate" not in cast(
        "dict[str, Any]", _credit_report(experian_report_envelope(report_date=None))["CreditProfileHeader"]
    )


def test_report_envelope_builder_copies_the_account_list() -> None:
    accounts = [cais_account()]
    pair = experian_report_envelope(accounts=accounts)
    accounts.append(cais_account())
    assert len(_details(pair)) == 1


# --------------------------------------------------------------------------- probe-literal guard


def _default_builder_outputs() -> list[Any]:
    outputs: list[Any] = [
        pan_comprehensive_envelope(),
        pan_comprehensive_envelope(aadhaar_linked=None),
        experian_score_envelope(),
        experian_score_envelope(success=False),
        experian_score_envelope(credit_score=""),
        experian_score_envelope(omit_score=True),
        experian_report_envelope(),
        experian_report_envelope(accounts=[cais_account(), cais_account(portfolio_type="R", account_type="10")]),
        experian_report_envelope(success=False),
        failure_envelope(),
        validation_error_envelope(),
        cais_account(),
    ]
    return outputs


def _scenario_specs() -> list[Any]:
    clients = [
        scn.happy_quick_scenario(),
        *[scn.ntc_scenario(v) for v in get_args(scn.NtcVariant)],
        *[scn.score_failure_scenario(v) for v in get_args(scn.ScoreFailureVariant)],
        *[scn.pan_invalid_scenario(k) for k in get_args(scn.PanInvalidKind)],
        scn.full_happy_reported_emi_scenario(),
        scn.full_emi_four_branch_scenario(),
        scn.full_pincode_union_scenario(),
        scn.no_pincode_scenario(),
    ]
    return [c._specs for c in clients]  # pyright: ignore[reportPrivateUsage]


def contains_forbidden(payload: object, literals: tuple[str, ...] = FORBIDDEN_LITERALS) -> list[str]:
    """The guard: every probe literal found anywhere in the JSON-ish payload (exceptions stringified)."""
    text = json.dumps(payload, default=str)
    return [lit for lit in literals if lit in text]


def test_probe_literal_guard_no_forbidden_literal_in_any_default_builder_output() -> None:
    assert contains_forbidden(_default_builder_outputs()) == []
    assert contains_forbidden(_default_builder_outputs(), EXTRA_PROBE_LITERALS) == []


def test_probe_literal_guard_no_forbidden_literal_in_any_scenario_world() -> None:
    assert contains_forbidden(_scenario_specs()) == []
    assert contains_forbidden(_scenario_specs(), EXTRA_PROBE_LITERALS) == []


def test_probe_literal_guard_is_real_it_flags_each_literal_when_one_is_planted() -> None:
    # positive control: if a builder ever emitted a probe value, the scan above would fail.
    for literal in FORBIDDEN_LITERALS:
        planted = experian_report_envelope(accounts=[cais_account(holder_zip=literal, current_balance=literal)])
        assert contains_forbidden(planted) == [literal]
        assert contains_forbidden(pan_comprehensive_envelope(zip_code=literal)) == [literal]


def test_probe_literal_guard_no_forbidden_literal_in_the_package_source() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path.name}:{lit}" for lit in FORBIDDEN_LITERALS + EXTRA_PROBE_LITERALS if lit in text)
    assert offenders == []


def test_builder_defaults_are_synthetic_and_the_testing_package_never_reads_the_probe_folder() -> None:
    for path in sorted(TESTING_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "surepass-lab" not in text.replace("NEVER", ""), path.name  # no path into the real probe folder
        assert "open(" not in text and "read_text" not in text and "Path(" not in text, path.name  # pure builders


# --------------------------------------------------------------------------- scenarios: smoke + shapes

CHECK = "chk-1"
HASH = "a" * 64
NAME = "Asha Verma"
MOBILE = "9876543210"


def _pan(client: FakeSurePassClient, pan: str) -> SurePassResult:
    return client.pan_comprehensive(CHECK, HASH, pan)


def _score(client: FakeSurePassClient, pan: str) -> SurePassResult:
    return client.experian_score(CHECK, HASH, name=NAME, mobile=MOBILE, pan=pan)


def _report(client: FakeSurePassClient, pan: str) -> SurePassResult:
    return client.experian_report(CHECK, HASH, name=NAME, mobile=MOBILE, pan=pan)


class RecordingSink:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log_surepass_call(self, **kwargs: Any) -> str | None:
        self.rows.append(kwargs)
        return None


def test_happy_quick_scenario_resolves_pan_and_score_and_nothing_else() -> None:
    client = scn.happy_quick_scenario()
    pan_result = _pan(client, scn.PAN_OK)
    score_result = _score(client, scn.PAN_OK)
    assert pan_result.endpoint == PAN_ENDPOINT and pan_result.success is True
    assert cast("dict[str, Any]", pan_result.data)["address"]["zip"] == "560001"
    assert cast("dict[str, Any]", score_result.data)["credit_score"] == "745"
    with pytest.raises(AssertionError):  # a quick check must never call the report endpoint
        _report(client, scn.PAN_OK)
    assert client.endpoints_called() == [PAN_ENDPOINT, SCORE_ENDPOINT]


def test_happy_quick_scenario_kwargs() -> None:
    client = scn.happy_quick_scenario("KLMPN4321Q", "702", "110001")
    assert cast("dict[str, Any]", _score(client, "KLMPN4321Q").data)["credit_score"] == "702"
    assert cast("dict[str, Any]", _pan(client, "KLMPN4321Q").data)["address"]["zip"] == "110001"
    client2 = scn.happy_quick_scenario(pan="QRSPT9876U", score="610")
    assert cast("dict[str, Any]", _score(client2, "QRSPT9876U").data)["credit_score"] == "610"


def test_scenarios_are_keyed_by_pan() -> None:
    client = scn.happy_quick_scenario()
    with pytest.raises(KeyError):
        _pan(client, "ZZZPZ0000Z")  # a PAN the scenario was not built for


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("empty", ""), ("text", "NH"), ("missing", None), ("four_digit", "0745")],
)
def test_ntc_scenario_variants_are_success_true_with_a_non_three_digit_score(
    variant: Literal["empty", "text", "missing", "four_digit"], expected: str | None
) -> None:
    client = scn.ntc_scenario(variant)
    result = _score(client, scn.PAN_NTC)
    assert result.success is True and result.http_status == 200  # NTC is NEVER a failure shape
    data = cast("dict[str, Any]", result.data)
    if expected is None:
        assert "credit_score" not in data
    else:
        assert data["credit_score"] == expected
    assert _pan(client, scn.PAN_NTC).success is True


def test_ntc_scenario_variants_cover_exactly_the_four_pinned_vectors() -> None:
    assert set(get_args(scn.NtcVariant)) == {"empty", "text", "missing", "four_digit"}


def test_score_failure_scenario_variants_raise_the_right_exception_and_log_one_row() -> None:
    expectations: dict[str, type[SurePassError]] = {
        "success_false": SurePassUpstreamError,
        "http_500": SurePassUpstreamError,
        "auth_403": SurePassAuthError,
        "timeout": SurePassTimeout,
    }
    assert set(get_args(scn.ScoreFailureVariant)) == set(expectations)
    for variant, exc_type in expectations.items():
        sink = RecordingSink()
        client = scn.score_failure_scenario(cast("Any", variant), sink=sink)
        assert _pan(client, scn.PAN_OK).success is True  # the PAN step still succeeds
        with pytest.raises(exc_type):
            _score(client, scn.PAN_OK)
        score_rows = [r for r in sink.rows if r["endpoint"] == SCORE_ENDPOINT]
        assert len(score_rows) == 1, variant  # one row per HTTP attempt, written before the raise


def test_score_failure_timeout_variant_is_a_surepass_timeout_instance() -> None:
    client = scn.score_failure_scenario("timeout")
    spec = client._specs[SCORE_ENDPOINT]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(spec, dict)
    assert any(isinstance(v, SurePassTimeout) for v in cast("dict[str, Any]", spec).values())


def test_pan_invalid_scenario_status_and_company_are_success_true_identities() -> None:
    status_client = scn.pan_invalid_scenario("status_invalid")
    assert cast("dict[str, Any]", _pan(status_client, scn.PAN_INVALID).data)["status"] == "invalid"
    company_client = scn.pan_invalid_scenario("company")
    assert cast("dict[str, Any]", _pan(company_client, scn.PAN_COMPANY).data)["category"] == "company"
    for client, pan in ((status_client, scn.PAN_INVALID), (company_client, scn.PAN_COMPANY)):
        with pytest.raises(AssertionError):  # no bureau call may follow an unverified PAN
            _score(client, pan)


def test_pan_invalid_scenario_failure_envelope_is_a_service_failure_under_v1_1() -> None:
    # v1.0 read this as 'invalid PAN'; v1.1 reversed it: ANY success:false is SurePassUpstreamError.
    client = scn.pan_invalid_scenario("failure_envelope")
    with pytest.raises(SurePassUpstreamError):
        _pan(client, scn.PAN_INVALID)


def test_pan_invalid_scenario_default_pan_depends_on_kind() -> None:
    assert scn.PAN_INVALID != scn.PAN_COMPANY
    assert _pan(scn.pan_invalid_scenario("company"), scn.PAN_COMPANY).success is True
    custom = scn.pan_invalid_scenario("status_invalid", "QRSPT9876U")
    assert _pan(custom, "QRSPT9876U").success is True


def test_full_happy_reported_emi_scenario_one_reported_account() -> None:
    client = scn.full_happy_reported_emi_scenario()
    result = _report(client, scn.PAN_OK)
    data = cast("dict[str, Any]", result.data)
    assert data["credit_score"] == "745"
    report = cast("dict[str, Any]", data["credit_report"])
    assert report["CreditProfileHeader"]["ReportDate"] == "20261001"
    accounts = cast("list[dict[str, Any]]", report["CAIS_Account"]["CAIS_Account_DETAILS"])
    assert len(accounts) == 1
    account = accounts[0]
    assert account["Scheduled_Monthly_Payment_Amount"] == "3500"
    assert account["Date_Reported"] == "20260920"
    assert account["Date_Closed"] == "" and account["Account_Status"] == "11"
    assert [r["ZIP_Postal_Code_non_normalized"] for r in account["CAIS_Holder_Address_Details"]] == ["560001"]
    assert cast("dict[str, Any]", _pan(client, scn.PAN_OK).data)["address"]["zip"] == "560001"


def test_full_emi_four_branch_scenario_is_the_six_account_table() -> None:
    client = scn.full_emi_four_branch_scenario()
    data = cast("dict[str, Any]", _report(client, scn.PAN_OK).data)
    report = cast("dict[str, Any]", data["credit_report"])
    assert report["CreditProfileHeader"]["ReportDate"] == "20261001"
    accounts = cast("list[dict[str, Any]]", report["CAIS_Account"]["CAIS_Account_DETAILS"])
    assert len(accounts) == 6
    emis = [a["Scheduled_Monthly_Payment_Amount"] for a in accounts]
    assert emis == ["3500", "", "", "", "", "9999"]
    assert [a["Current_Balance"] for a in accounts][1:5] == ["120000", "60000", "40000", "90000"]
    assert [a["Terms_Duration"] for a in accounts][1:3] == ["048", "024"]
    assert [a["Open_Date"] for a in accounts][1:3] == ["20240115", "20240115"]
    assert [a["Portfolio_Type"] for a in accounts] == ["I", "I", "I", "R", "I", "I"]
    assert accounts[5]["Date_Reported"] == "20260601"  # the stale one
    assert all(a["Date_Closed"] == "" for a in accounts)  # six OPEN accounts
    numbers = [a["Account_Number"] for a in accounts]
    assert len(set(numbers)) == 6  # distinct synthetic account numbers
    # the full flow never calls the score endpoint (a Report pull refreshes the score cache)
    with pytest.raises(AssertionError):
        _score(client, scn.PAN_OK)


def test_full_pincode_union_scenario_world() -> None:
    client = scn.full_pincode_union_scenario()
    assert cast("dict[str, Any]", _pan(client, scn.PAN_OK).data)["address"]["zip"] == "560002"
    data = cast("dict[str, Any]", _report(client, scn.PAN_OK).data)
    accounts = cast("list[dict[str, Any]]", data["credit_report"]["CAIS_Account"]["CAIS_Account_DETAILS"])
    zips = [[r["ZIP_Postal_Code_non_normalized"] for r in a["CAIS_Holder_Address_Details"]] for a in accounts]
    assert zips == [["110001", "20"], ["400001"]]
    assert [a["Date_Reported"] for a in accounts] == ["20260901", "20250901"]
    assert all(a["Date_Closed"] == "" for a in accounts)  # excluded by DATE only (>12 months), not by closure


def test_no_pincode_scenario_pan_address_zip_is_empty() -> None:
    client = scn.no_pincode_scenario()
    assert cast("dict[str, Any]", _pan(client, scn.PAN_OK).data)["address"]["zip"] == ""
    assert _score(client, scn.PAN_OK).success is True
    report = cast("dict[str, Any]", _report(client, scn.PAN_OK).data)
    accounts = cast("list[dict[str, Any]]", report["credit_report"]["CAIS_Account"]["CAIS_Account_DETAILS"])
    for account in accounts:  # a full check here must also have no candidate pincode
        assert [r["ZIP_Postal_Code_non_normalized"] for r in account["CAIS_Holder_Address_Details"]] == [""]


Factory = Callable[[RecordingSink], FakeSurePassClient]

# (label, the PAN the world is keyed by, factory wired to a sink)
WORLDS: list[tuple[str, str, Factory]] = [
    ("happy_quick", scn.PAN_OK, lambda sink: scn.happy_quick_scenario(sink=sink)),
    ("ntc", scn.PAN_NTC, lambda sink: scn.ntc_scenario("empty", sink=sink)),
    ("score_failure", scn.PAN_OK, lambda sink: scn.score_failure_scenario("success_false", sink=sink)),
    ("pan_invalid", scn.PAN_INVALID, lambda sink: scn.pan_invalid_scenario("status_invalid", sink=sink)),
    ("full_happy", scn.PAN_OK, lambda sink: scn.full_happy_reported_emi_scenario(sink=sink)),
    ("full_four_branch", scn.PAN_OK, lambda sink: scn.full_emi_four_branch_scenario(sink=sink)),
    ("full_pincode_union", scn.PAN_OK, lambda sink: scn.full_pincode_union_scenario(sink=sink)),
    ("no_pincode", scn.PAN_OK, lambda sink: scn.no_pincode_scenario(sink=sink)),
]


@pytest.mark.parametrize(("label", "pan", "factory"), WORLDS, ids=[w[0] for w in WORLDS])
def test_every_factory_returns_a_fake_client_whose_pan_endpoint_resolves_and_logs_to_the_sink(
    label: str, pan: str, factory: Factory
) -> None:
    sink = RecordingSink()
    client = factory(sink)
    assert isinstance(client, FakeSurePassClient), label
    assert client.calls_enabled is True
    result = _pan(client, pan)
    assert result.success is True and result.endpoint == PAN_ENDPOINT
    assert len(sink.rows) == 1 and sink.rows[0]["endpoint"] == PAN_ENDPOINT
    assert sink.rows[0]["error_kind"] is None and sink.rows[0]["success"] is True
    assert sink.rows[0]["raw_envelope"] is not None


def test_scenarios_wire_into_a_real_memory_repo_sink_and_store_the_raw_envelope() -> None:
    repo = MemoryRepo(clock=FrozenClock(T0).now)
    repo.insert_check(make_check(id=CHECK))
    client = scn.happy_quick_scenario(sink=repo)
    assert _pan(client, scn.PAN_OK).success is True
    calls = repo.list_calls()
    assert len(calls) == 1 and calls[0].endpoint == PAN_ENDPOINT
    assert len(repo.raw_store.keys()) == 1  # the verbatim envelope landed in the private raw store


def test_factories_build_fresh_envelopes_every_call() -> None:
    first = scn.full_emi_four_branch_scenario()
    second = scn.full_emi_four_branch_scenario()
    a = cast("dict[str, Any]", _report(first, scn.PAN_OK).data)
    b = cast("dict[str, Any]", _report(second, scn.PAN_OK).data)
    assert a == b and a is not b
    a["credit_score"] = "mutated"
    assert cast("dict[str, Any]", _report(second, scn.PAN_OK).data)["credit_score"] == "745"


def test_scenario_pan_constants_are_distinct_pan_shaped_and_synthetic() -> None:
    pans = {scn.PAN_OK, scn.PAN_NTC, scn.PAN_COMPANY, scn.PAN_INVALID}
    assert len(pans) == 4
    for pan in pans:
        assert re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan)
    assert contains_forbidden(sorted(pans)) == []


def test_envelopes_module_exports() -> None:
    for name in (
        "EnvelopePair",
        "failure_envelope",
        "validation_error_envelope",
        "pan_comprehensive_envelope",
        "experian_score_envelope",
        "cais_account",
        "experian_report_envelope",
        "_ok_envelope",
    ):
        assert hasattr(env, name), name
    status, body = env._ok_envelope({"x": 1})  # pyright: ignore[reportPrivateUsage]
    assert status == 200 and body["success"] is True and body["data"] == {"x": 1}
