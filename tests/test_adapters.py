"""Task 5: surepass/adapters.py (pure raw-envelope-dict -> typed facts) and the seed derive/range.py.

The adapters take the RAW ENVELOPE DICT (what ``fetch_raw`` returns / ``SurePassResult.raw``), so a cached object is
re-adapted on current code every time (K5). Load-bearing pins:

* NTC is HTTP 200 + success:true + credit_score not ``^[0-9]{3}$`` (v1.1): ``''`` / ``'NH'`` / ``'0745'`` / key missing.
  There is NO FCIREXScore fallback and NO NTC-from-failure branch: a non-success envelope raises ValueError.
* No gender anywhere; strict-bool aadhaar; holder DOB rows are never read (contradictory-DOB trap).
* Adapters are TOTAL over junk bureau strings: they raise only on a non-success envelope.
* Score facts are readable from a Report envelope (``score_cache.source = 'report'``).
"""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

import eligibility_agent
from eligibility_agent.derive.range import PIN_RE, band_from_score, is_ntc_value
from eligibility_agent.graph.state import Identity, OpenAccount, ReportFacts, ScoreFacts
from eligibility_agent.surepass import adapters
from eligibility_agent.surepass.adapters import (
    as_list,
    dig,
    identity_from_envelope,
    is_open_account,
    pan_to_identity,
    parse_amount,
    parse_int_sentinel,
    parse_iso_date,
    parse_rate,
    parse_yyyymmdd,
    report_facts_from_envelope,
    report_to_report_facts,
    resolve_tenure_months,
    score_facts_from_envelope,
    score_to_score_facts,
)
from eligibility_agent.surepass.client import (
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    SurePassResult,
    parse_surepass_body,
)
from eligibility_agent.testing import scenarios as scn
from eligibility_agent.testing.envelopes import (
    EnvelopePair,
    cais_account,
    experian_report_envelope,
    experian_score_envelope,
    failure_envelope,
    pan_comprehensive_envelope,
    validation_error_envelope,
)

PACKAGE_DIR = Path(eligibility_agent.__file__).parent
ANCHOR = date(2026, 10, 1)  # the default ReportDate of every report builder
PULLED = date(2026, 10, 10)


def ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def days_before(n: int) -> str:
    return ymd(ANCHOR - timedelta(days=n))


def raw(pair: EnvelopePair) -> dict[str, Any]:
    return pair[1]


def result_of(endpoint: str, pair: EnvelopePair) -> SurePassResult:
    return parse_surepass_body(endpoint, pair[0], pair[1])


def report_raw(accounts: list[dict[str, Any]] | None = None, **kw: Any) -> dict[str, Any]:
    return experian_report_envelope(accounts=accounts or [], **kw)[1]


def facts(accounts: list[dict[str, Any]] | None = None, *, today: date | None = None, **kw: Any) -> ReportFacts:
    return report_facts_from_envelope(report_raw(accounts, **kw), today=today)


def one(**account_kw: Any) -> OpenAccount:
    rows = facts([cais_account(**account_kw)]).open_accounts
    assert len(rows) == 1
    return rows[0]


# --------------------------------------------------------------------------- derive/range seed: band + NTC predicate


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (649, "<650"),
        (650, "650-720"),
        (719, "650-720"),
        (720, "720-780"),
        (779, "720-780"),
        (780, "780+"),
        (900, "780+"),
        (300, "<650"),
        (299, "<650"),  # out of range: nearest bucket
        (0, "<650"),
        (950, "780+"),  # out of range: nearest bucket
        (999, "780+"),
        (None, "NTC"),
    ],
)
def test_band_from_score_edges(score: int | None, band: str) -> None:
    assert band_from_score(score) == band


@pytest.mark.parametrize(
    ("value", "ntc"),
    [
        ("745", False),
        (" 745 ", False),  # stripped
        ("745\n", False),
        ("300", False),
        ("900", False),
        ("000", False),  # three digits: a (degenerate) score, not NTC
        (745, False),  # str() of an int is three digits
        ("", True),
        ("NH", True),
        ("0745", True),  # four characters, not ^[0-9]{3}$
        ("74", True),
        ("7450", True),
        ("-1", True),
        ("7 45", True),
        ("74a", True),
        ("٧٤٥", True),  # Arabic-Indic digits are not [0-9]
        ("745.0", True),
        (745.0, True),
        (None, True),  # key missing
        (True, True),
        ([], True),
        ({}, True),
    ],
)
def test_is_ntc_value(value: object, ntc: bool) -> None:
    assert is_ntc_value(value) is ntc


def test_pin_re_is_the_registry_pattern() -> None:
    assert PIN_RE.pattern == r"^[1-9][0-9]{5}$"
    assert PIN_RE.fullmatch("560001") and not PIN_RE.fullmatch("056001") and not PIN_RE.fullmatch("5600011")


def test_range_seed_module_is_minimal_and_documents_the_task_6_extension() -> None:
    path = PACKAGE_DIR / "derive" / "range.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree) or ""
    assert "Task 6" in doc and "extend" in doc.lower()
    assert _imports(tree) <= {"__future__", "re", "typing", "eligibility_agent.graph.state"}
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert funcs == {"is_ntc_value", "band_from_score"}


# --------------------------------------------------------------------------- parsing helpers


def test_as_list() -> None:
    d1, d2 = {"a": 1}, {"b": 2}
    assert as_list([d1, d2]) == [d1, d2]
    assert as_list([d1, "x", None, 3, [d2], d2]) == [d1, d2]  # non-dict items dropped
    assert as_list(d1) == [d1]  # single-element collapse
    assert as_list(None) == [] and as_list("x") == [] and as_list(5) == [] and as_list([]) == []


def test_dig() -> None:
    tree = {"a": {"b": {"c": 7}}, "n": None, "s": "str"}
    assert dig(tree, "a", "b", "c") == 7
    assert dig(tree, "a", "b") == {"c": 7}
    assert dig(tree, "a", "x", "c") is None
    assert dig(tree, "n", "x") is None and dig(tree, "s", "x") is None
    assert dig(None, "a") is None
    assert dig(tree) == tree


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3500", 3500.0),
        (" 3500 ", 3500.0),
        ("1,25,000.50", 125000.5),  # Indian digit grouping
        ("0", 0.0),
        ("-500", -500.0),  # sign preserved: callers clamp / filter
        (12, 12.0),
        (12.5, 12.5),
        ("", None),
        ("   ", None),
        ("abc", None),
        ("NaN", None),
        ("inf", None),
        ("-inf", None),
        (float("nan"), None),
        (float("inf"), None),
        (10**400, None),  # float() overflow
        (True, None),
        (False, None),
        (None, None),
        ([], None),
        ({}, None),
    ],
)
def test_parse_amount(value: object, expected: float | None) -> None:
    assert parse_amount(value) == expected


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        ("024", {}, 24),
        ("048", {}, 48),
        ("12", {}, 12),
        (" 12 ", {}, 12),
        ("0", {}, 0),
        ("0", {"zero_is_unknown": True}, None),
        ("000", {"zero_is_unknown": True}, None),
        ("", {}, None),
        ("1.5", {}, None),
        ("abc", {}, None),
        ("-3", {}, -3),
        (7, {}, 7),
        (True, {}, None),
        (None, {}, None),
        ("9" * 6000, {}, None),  # int() digit-limit ValueError is swallowed
    ],
)
def test_parse_int_sentinel(value: object, kwargs: dict[str, bool], expected: int | None) -> None:
    assert parse_int_sentinel(value, **kwargs) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("13.500", 13.5), ("100", 100.0), ("0.5", 0.5), ("0", None), ("0.0", None), ("-2", None), ("101", None)]
    + [("", None), ("abc", None), (None, None), (True, None)],
)
def test_parse_rate(value: object, expected: float | None) -> None:
    assert parse_rate(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20261001", date(2026, 10, 1)),
        (" 20261001 ", date(2026, 10, 1)),
        ("20240229", date(2024, 2, 29)),
        ("20250229", None),  # not a real date
        ("20261301", None),
        ("20260001", None),
        ("2026100", None),
        ("202610011", None),
        ("2026-10-01", None),
        ("", None),
        (None, None),
        ("abcdefgh", None),
        ("٢٠٢٦١٠٠١", None),  # non-ASCII digits
    ],
)
def test_parse_yyyymmdd(value: object, expected: date | None) -> None:
    assert parse_yyyymmdd(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1998-04-12", date(1998, 4, 12)),
        (" 1998-04-12 ", date(1998, 4, 12)),
        ("19980412", None),  # not the YYYY-MM-DD wire shape
        ("1998-4-12", None),
        ("1998-13-01", None),
        ("1998-02-30", None),
        ("1998-04-12T00:00:00", None),
        ("", None),
        (None, None),
        (19980412, None),
    ],
)
def test_parse_iso_date(value: object, expected: date | None) -> None:
    assert parse_iso_date(value) == expected


@pytest.mark.parametrize(
    ("account", "expected"),
    [
        ({"Repayment_Tenure": "0", "Terms_Duration": "048", "Terms_Frequency": "M"}, 48),  # '0' = unknown -> falls through
        ({"Repayment_Tenure": "12", "Terms_Duration": "024", "Terms_Frequency": "M"}, 12),  # PRD order: Repayment first
        ({"Repayment_Tenure": "0", "Terms_Duration": "", "Terms_Frequency": "M"}, None),
        ({"Repayment_Tenure": "0", "Terms_Duration": "000", "Terms_Frequency": "M"}, None),
        ({"Repayment_Tenure": "", "Terms_Duration": "024", "Terms_Frequency": ""}, 24),  # '' frequency is allowed
        ({"Repayment_Tenure": "0", "Terms_Duration": "024"}, 24),  # frequency key absent
        ({"Repayment_Tenure": "0", "Terms_Duration": "024", "Terms_Frequency": "Q"}, None),  # non-monthly: unknown
        ({"Repayment_Tenure": "12", "Terms_Duration": "024", "Terms_Frequency": "Y"}, None),
        ({"Repayment_Tenure": "-5", "Terms_Duration": "024", "Terms_Frequency": "M"}, 24),  # non-positive ignored
        ({"Repayment_Tenure": "abc", "Terms_Duration": "abc", "Terms_Frequency": "M"}, None),
        ({}, None),
    ],
)
def test_resolve_tenure_months(account: dict[str, Any], expected: int | None) -> None:
    assert resolve_tenure_months(account) == expected


@pytest.mark.parametrize(
    ("account", "is_open"),
    [
        ({"Date_Closed": "", "Account_Status": "11"}, True),
        ({"Date_Closed": "", "Account_Status": "13"}, True),  # OR semantics: no Date_Closed -> open
        ({"Date_Closed": "20250110", "Account_Status": "11"}, True),  # OR semantics: status 11 -> open
        ({"Date_Closed": "20250110", "Account_Status": "13"}, False),
        ({"Date_Closed": "20250110", "Account_Status": "15"}, False),
        ({"Date_Closed": "  ", "Account_Status": "13"}, True),  # whitespace is empty
        ({"Date_Closed": None, "Account_Status": " 11 "}, True),
        ({}, True),
        ({"Date_Closed": "20250110"}, False),
    ],
)
def test_is_open_account(account: dict[str, Any], is_open: bool) -> None:
    assert is_open_account(account) is is_open


JUNK: list[object] = [
    None,
    True,
    False,
    0,
    -1,
    10**400,
    float("nan"),
    float("inf"),
    "",
    " ",
    "abc",
    "NaN",
    "inf",
    "1e999",
    "0x10",
    "1,2,3",
    "--5",
    "12.5.6",
    "\x00",
    "None",
    "٧٤٥",
    "9" * 6000,
    "-",
    "20261301",
    "2026-13-45",
    [],
    {},
    [1, "a"],
    {"a": {"b": None}},
    b"bytes",
    object(),
]


@pytest.mark.parametrize("junk", JUNK, ids=[type(j).__name__ + str(i) for i, j in enumerate(JUNK)])
def test_every_parsing_helper_is_total_over_junk(junk: object) -> None:
    parse_amount(junk)
    parse_int_sentinel(junk)
    parse_int_sentinel(junk, zero_is_unknown=True)
    parse_rate(junk)
    parse_yyyymmdd(junk)
    parse_iso_date(junk)
    as_list(junk)
    is_ntc_value(junk)
    band_from_score(None)
    junk_dict = cast("dict[str, Any]", {"Repayment_Tenure": junk, "Terms_Duration": junk, "Terms_Frequency": junk})
    resolve_tenure_months(junk_dict)
    is_open_account(cast("dict[str, Any]", {"Date_Closed": junk, "Account_Status": junk}))
    dig(cast("dict[str, Any]", {"a": junk}), "a", "b", "c")
    dig(cast("Any", junk), "a")


# --------------------------------------------------------------------------- non-success envelopes


@pytest.mark.parametrize(
    "body",
    [
        {"success": False, "data": {}, "message_code": "no_record_found"},
        {"data": {"credit_score": "745"}},  # success missing
        {"success": "true", "data": {}},  # not the bool
        {"success": 1, "data": {}},
        {"success": None, "data": {}},
        {"errors": {"id_number": "Missing required parameter."}, "message": "Input payload validation failed"},
        {},
    ],
)
def test_every_adapter_raises_value_error_on_a_non_success_envelope(body: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="adapter received non-success envelope"):
        identity_from_envelope(body)
    with pytest.raises(ValueError, match="adapter received non-success envelope"):
        score_facts_from_envelope(body, pulled_on=PULLED, source="score")
    with pytest.raises(ValueError, match="adapter received non-success envelope"):
        report_facts_from_envelope(body)


def test_builder_failure_vectors_are_refused_by_the_adapters_never_read_as_ntc() -> None:
    for pair in (
        experian_score_envelope(success=False),
        experian_report_envelope(success=False),
        failure_envelope(http_status=200, message_code="no_record_found"),
        failure_envelope(),
        validation_error_envelope(),
    ):
        with pytest.raises(ValueError):
            score_facts_from_envelope(raw(pair), pulled_on=PULLED, source="score")
        with pytest.raises(ValueError):
            report_facts_from_envelope(raw(pair))
        with pytest.raises(ValueError):
            identity_from_envelope(raw(pair))


# --------------------------------------------------------------------------- identity


def test_identity_happy_path() -> None:
    identity = identity_from_envelope(raw(pan_comprehensive_envelope()))
    assert identity == Identity(
        pan_valid=True,
        status="valid",
        category="person",
        full_name="RAVI TESTKUMAR",
        dob=date(1998, 4, 12),
        aadhaar_linked=True,
        pan_pincode="560001",
        surepass_client_id="pan_comprehensive_synthetic_0001",
    )


def test_identity_has_no_gender_anywhere() -> None:
    identity = identity_from_envelope(raw(pan_comprehensive_envelope()))
    assert not hasattr(identity, "gender")
    assert "gender" not in Identity.model_fields
    assert "gender" not in identity.model_dump()
    assert "gender" not in identity.model_dump_json().lower()
    # the raw payload carries a gender key (real payloads do), and the adapter output ignores it completely
    assert "gender" in cast("dict[str, Any]", raw(pan_comprehensive_envelope())["data"])
    male = identity_from_envelope(raw(pan_comprehensive_envelope(gender="M")))
    female = identity_from_envelope(raw(pan_comprehensive_envelope(gender="F")))
    odd = identity_from_envelope(raw(pan_comprehensive_envelope(gender="")))
    assert male == female == odd


def _code_tokens(path: Path) -> list[str]:
    """Every identifier and string literal in a module's CODE (docstrings excluded, so prose may say 'no gender')."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    tokens: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            tokens.append(node.value)
        elif isinstance(node, ast.Name):
            tokens.append(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.append(node.attr)
        elif isinstance(node, ast.arg):
            tokens.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            tokens.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            tokens.append(node.name)
        elif isinstance(node, ast.alias):
            tokens.append(node.name)
    return tokens


def test_adapter_code_never_mentions_gender_or_the_fcirex_fallback() -> None:
    tokens = [t.lower() for t in _code_tokens(PACKAGE_DIR / "surepass" / "adapters.py")]
    assert not [t for t in tokens if "gender" in t]  # no gender attribute, key, helper or constant
    assert not [t for t in tokens if "fcirex" in t]  # the v1.0 score fallback is dropped


@pytest.mark.parametrize(
    ("kwargs", "valid"),
    [
        ({}, True),
        ({"status": "invalid"}, False),
        ({"category": "company"}, False),
        ({"status": "invalid", "category": "company"}, False),
        ({"status": "VALID"}, False),  # exact literals, no case folding
        ({"category": "Person"}, False),
        ({"status": ""}, False),
    ],
)
def test_identity_pan_valid_is_the_status_and_category_gate_only(kwargs: dict[str, Any], valid: bool) -> None:
    identity = identity_from_envelope(raw(pan_comprehensive_envelope(**kwargs)))
    assert identity.pan_valid is valid


def test_identity_invalid_pan_still_parses_the_other_fields() -> None:
    identity = identity_from_envelope(raw(pan_comprehensive_envelope(status="invalid", zip_code="110001")))
    assert identity.pan_valid is False
    assert identity.status == "invalid" and identity.category == "person"
    assert identity.pan_pincode == "110001" and identity.dob == date(1998, 4, 12)


def test_identity_aadhaar_linked_is_a_strict_bool() -> None:
    assert identity_from_envelope(raw(pan_comprehensive_envelope(aadhaar_linked=True))).aadhaar_linked is True
    assert identity_from_envelope(raw(pan_comprehensive_envelope(aadhaar_linked=False))).aadhaar_linked is False
    # key omitted: stays None (NOT False), so the link warning cannot fire
    assert identity_from_envelope(raw(pan_comprehensive_envelope(aadhaar_linked=None))).aadhaar_linked is None
    junk_values: list[object] = ["true", "false", "Y", "N", 1, 0, "", None, [], {}]
    for junk in junk_values:
        body = raw(pan_comprehensive_envelope())
        cast("dict[str, Any]", body["data"])["aadhaar_linked"] = junk
        assert identity_from_envelope(body).aadhaar_linked is None, repr(junk)


@pytest.mark.parametrize(
    ("zip_code", "expected"),
    [
        ("560001", "560001"),
        (" 560001 ", "560001"),
        ("110001", "110001"),
        ("056001", None),  # leading zero
        ("0411", None),
        ("20", None),  # junk seen on real bureau rows
        ("", None),
        ("5600 01", None),
        ("560 001", None),
        ("5600012", None),
        ("56000", None),
        ("56000A", None),
        ("٥٦٠٠٠١", None),
    ],
)
def test_identity_pan_pincode_filtering(zip_code: str, expected: str | None) -> None:
    assert identity_from_envelope(raw(pan_comprehensive_envelope(zip_code=zip_code))).pan_pincode == expected


def _address(data: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", data["address"])


PAN_DATA_MUTATIONS: list[Callable[[dict[str, Any]], None]] = [
    lambda d: d.pop("address"),
    lambda d: d.__setitem__("address", None),
    lambda d: d.__setitem__("address", "560001"),
    lambda d: d.__setitem__("address", {}),
    lambda d: _address(d).__setitem__("zip", None),
    lambda d: _address(d).__setitem__("zip", True),
]


@pytest.mark.parametrize("mutate", PAN_DATA_MUTATIONS, ids=[f"mutation_{i}" for i in range(len(PAN_DATA_MUTATIONS))])
def test_identity_pan_pincode_missing_or_malformed_address(mutate: Callable[[dict[str, Any]], None]) -> None:
    body = raw(pan_comprehensive_envelope())
    mutate(cast("dict[str, Any]", body["data"]))
    assert identity_from_envelope(body).pan_pincode is None


def test_identity_dob_name_and_client_id() -> None:
    assert identity_from_envelope(raw(pan_comprehensive_envelope(dob="1990-05-06"))).dob == date(1990, 5, 6)
    for bad_dob in ("", "19980412", "1998-13-40", "garbage"):
        assert identity_from_envelope(raw(pan_comprehensive_envelope(dob=bad_dob))).dob is None
    assert identity_from_envelope(raw(pan_comprehensive_envelope(full_name=""))).full_name is None
    assert identity_from_envelope(raw(pan_comprehensive_envelope(full_name="   "))).full_name is None
    assert identity_from_envelope(raw(pan_comprehensive_envelope(client_id=""))).surepass_client_id is None
    assert identity_from_envelope(raw(pan_comprehensive_envelope(client_id="pan_x"))).surepass_client_id == "pan_x"


def test_identity_never_carries_the_masked_aadhaar_or_pan_number() -> None:
    identity = identity_from_envelope(raw(pan_comprehensive_envelope()))
    dumped = identity.model_dump_json()
    assert "XXXXXXXX0000" not in dumped
    assert "ABCPE1234F" not in dumped  # the PAN itself is not an Identity field either
    assert "masked_aadhaar" not in Identity.model_fields


@pytest.mark.parametrize("data", [None, "x", [], 5, {}])
def test_identity_with_a_missing_or_malformed_data_block_is_a_non_valid_identity(data: object) -> None:
    identity = identity_from_envelope({"success": True, "data": data})
    assert identity.pan_valid is False
    assert identity.status is None and identity.category is None and identity.full_name is None
    assert identity.dob is None and identity.aadhaar_linked is None and identity.pan_pincode is None


# --------------------------------------------------------------------------- score facts


def test_score_facts_from_a_score_envelope() -> None:
    f = score_facts_from_envelope(raw(experian_score_envelope()), pulled_on=PULLED, source="score")
    assert f == ScoreFacts(band="720-780", score=745, is_ntc=False, pulled_on=PULLED, source="score")
    assert isinstance(f.score, int)


def test_score_facts_are_readable_from_a_report_envelope_with_source_report() -> None:
    body = report_raw([cais_account()], credit_score="702")
    f = score_facts_from_envelope(body, pulled_on=PULLED, source="report")
    assert f == ScoreFacts(band="650-720", score=702, is_ntc=False, pulled_on=PULLED, source="report")
    # and the same score object read from the report equals the one read from a score envelope of the same value
    g = score_facts_from_envelope(raw(experian_score_envelope(credit_score="702")), pulled_on=PULLED, source="report")
    assert f == g


@pytest.mark.parametrize(
    "kwargs",
    [{"credit_score": ""}, {"credit_score": "NH"}, {"credit_score": "0745"}, {"omit_score": True}],
    ids=["empty", "text", "four_digit", "missing"],
)
def test_score_facts_ntc_variants_on_score_and_report_envelopes(kwargs: dict[str, Any]) -> None:
    for body in (raw(experian_score_envelope(**kwargs)), report_raw([cais_account()], **kwargs)):
        for source in ("score", "report"):
            f = score_facts_from_envelope(body, pulled_on=PULLED, source=source)  # type: ignore[arg-type]
            assert f.is_ntc is True and f.band == "NTC" and f.score is None
            assert f.pulled_on == PULLED and f.source == source


def test_score_facts_has_no_fcirex_fallback() -> None:
    # a populated SCORE.FCIREXScore must NOT rescue an empty / missing credit_score (v1.0 fallback is void)
    variants: list[dict[str, Any]] = [{"credit_score": ""}, {"credit_score": "NH"}, {"omit_score": True}]
    for kwargs in variants:
        body = raw(experian_score_envelope(**kwargs))
        score_block = cast("dict[str, Any]", cast("dict[str, Any]", body["data"])["credit_report"])["SCORE"]
        score_block["FCIREXScore"] = "745"
        f = score_facts_from_envelope(body, pulled_on=PULLED, source="score")
        assert f.is_ntc is True and f.score is None and f.band == "NTC", kwargs


@pytest.mark.parametrize(
    ("score", "band"),
    [("649", "<650"), ("650", "650-720"), ("719", "650-720"), ("720", "720-780"), ("779", "720-780"), ("780", "780+")]
    + [("299", "<650"), ("300", "<650"), ("900", "780+"), ("950", "780+"), ("000", "<650"), ("999", "780+")],
)
def test_score_facts_band_edges_and_out_of_range_map_to_the_nearest_bucket(score: str, band: str) -> None:
    f = score_facts_from_envelope(raw(experian_score_envelope(credit_score=score)), pulled_on=PULLED, source="score")
    assert (f.band, f.score, f.is_ntc) == (band, int(score), False)  # adapter adds nothing; callers see band only


def test_score_facts_credit_score_is_whitespace_tolerant_and_int_tolerant() -> None:
    assert score_facts_from_envelope(raw(experian_score_envelope(credit_score=" 745 ")), pulled_on=PULLED, source="score").score == 745
    body = raw(experian_score_envelope())
    cast("dict[str, Any]", body["data"])["credit_score"] = 745
    assert score_facts_from_envelope(body, pulled_on=PULLED, source="score").score == 745


@pytest.mark.parametrize("data", [None, "x", [], 5, {}])
def test_score_facts_with_a_missing_or_malformed_data_block_is_ntc(data: object) -> None:
    f = score_facts_from_envelope({"success": True, "data": data}, pulled_on=PULLED, source="score")
    assert f.is_ntc is True and f.band == "NTC" and f.score is None


# --------------------------------------------------------------------------- report facts: header, ntc, aggregates


def test_report_facts_basic_shape_and_report_date() -> None:
    f = facts([cais_account(scheduled_emi="3500")], credit_score="745", report_date="20261001")
    assert f.band == "720-780" and f.score == 745 and f.is_ntc is False
    assert f.report_date == date(2026, 10, 1)
    assert len(f.open_accounts) == 1
    assert f.past_due_total == 0.0 and f.written_off_or_settled == 0
    assert (f.enquiries_30d, f.enquiries_90d, f.enquiries_180d) == (0, 0, 0)


def test_report_facts_anchor_is_report_date_then_today() -> None:
    account = cais_account(date_reported="20260915")  # 16 days before the 20261001 ReportDate
    assert facts([account], today=date(2027, 6, 1)).open_accounts[0].is_stale is False  # ReportDate wins over today
    no_header = report_raw([account], report_date=None)
    # no ReportDate: anchor = today, so the same row is stale when today is far later and fresh when it is close
    assert report_facts_from_envelope(no_header, today=date(2027, 6, 1)).open_accounts[0].is_stale is True
    assert report_facts_from_envelope(no_header, today=date(2026, 10, 1)).open_accounts[0].is_stale is False
    assert report_facts_from_envelope(no_header, today=date(2026, 10, 1)).report_date is None


def test_report_facts_without_report_date_or_today_does_not_raise() -> None:
    f = report_facts_from_envelope(report_raw([cais_account()], report_date=None))
    assert f.report_date is None and len(f.open_accounts) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"credit_score": ""}, {"credit_score": "NH"}, {"credit_score": "0745"}, {"omit_score": True}],
    ids=["empty", "text", "four_digit", "missing"],
)
def test_ntc_report_has_no_accounts_and_zeroed_aggregates(kwargs: dict[str, Any]) -> None:
    rich = [
        cais_account(amount_past_due="5000", written_off_settled_status="99", holder_zip="560001"),
        cais_account(account_status="13", date_closed="20250110", amount_past_due="900"),
    ]
    f = facts(rich, caps_30="2", caps_90="3", caps_180="5", **kwargs)
    assert f.is_ntc is True and f.band == "NTC" and f.score is None
    assert f.open_accounts == []
    assert f.past_due_total == 0.0 and f.written_off_or_settled == 0
    assert (f.enquiries_30d, f.enquiries_90d, f.enquiries_180d) == (0, 0, 0)
    assert f.bureau_pincodes_12m == []
    assert f.report_date == date(2026, 10, 1)  # the header date is still known


@pytest.mark.parametrize(
    ("score", "band"),
    [("649", "<650"), ("650", "650-720"), ("719", "650-720"), ("720", "720-780"), ("779", "720-780"), ("780", "780+")]
    + [("299", "<650"), ("950", "780+")],
)
def test_report_facts_band_edges(score: str, band: str) -> None:
    f = facts([cais_account()], credit_score=score)
    assert (f.band, f.score, f.is_ntc) == (band, int(score), False)


def test_report_facts_has_no_fcirex_fallback() -> None:
    body = report_raw([cais_account()], credit_score="")
    cast("dict[str, Any]", cast("dict[str, Any]", body["data"])["credit_report"])["SCORE"]["FCIREXScore"] = "745"
    f = report_facts_from_envelope(body)
    assert f.is_ntc is True and f.score is None and f.open_accounts == []


def test_past_due_total_sums_over_all_accounts_open_and_closed() -> None:
    f = facts(
        [
            cais_account(amount_past_due="1000"),
            cais_account(amount_past_due="2,500", account_status="13", date_closed="20250110"),
            cais_account(amount_past_due="", account_number="SYNTHACC0003"),
            cais_account(amount_past_due="abc", account_number="SYNTHACC0004"),
        ]
    )
    assert f.past_due_total == 3500.0
    assert [a.amount_past_due for a in f.open_accounts] == [1000.0, 0.0, 0.0]  # open rows only; closed is in the total


def test_past_due_negative_is_clamped_and_never_offsets_real_arrears() -> None:
    f = facts([cais_account(amount_past_due="1000"), cais_account(amount_past_due="-1", account_status="13", date_closed="20250110")])
    assert f.past_due_total == 1000.0
    assert one(amount_past_due="-50").amount_past_due == 0.0


def test_written_off_or_settled_counts_non_empty_status_over_all_accounts() -> None:
    f = facts(
        [
            cais_account(written_off_settled_status="99"),
            cais_account(written_off_settled_status="", account_number="SYNTHACC0002"),
            cais_account(written_off_settled_status="  ", account_number="SYNTHACC0003"),  # whitespace is empty
            cais_account(written_off_settled_status="05", account_status="13", date_closed="20250110"),
        ]
    )
    assert f.written_off_or_settled == 2


def test_enquiries_parse_from_caps_summary() -> None:
    f = facts([cais_account()], caps_30="1", caps_90="2", caps_180="7")
    assert (f.enquiries_30d, f.enquiries_90d, f.enquiries_180d) == (1, 2, 7)
    for junk in ("", "abc", "-4", "1.5", None):
        g = facts([cais_account()], caps_30=junk, caps_90=junk, caps_180=junk)
        assert (g.enquiries_30d, g.enquiries_90d, g.enquiries_180d) == (0, 0, 0), repr(junk)


def test_enquiries_default_to_zero_when_the_summary_is_missing() -> None:
    body = report_raw([cais_account()])
    del cast("dict[str, Any]", cast("dict[str, Any]", body["data"])["credit_report"])["TotalCAPS_Summary"]
    f = report_facts_from_envelope(body)
    assert (f.enquiries_30d, f.enquiries_90d, f.enquiries_180d) == (0, 0, 0)


def test_missing_cais_sections_yield_an_empty_non_ntc_report() -> None:
    sections: list[dict[str, Any]] = [
        {"credit_score": "745"},
        {"credit_score": "745", "credit_report": None},
        {"credit_score": "745", "credit_report": "x"},
        {"credit_score": "745", "credit_report": {}},
        {"credit_score": "745", "credit_report": {"CAIS_Account": None}},
        {"credit_score": "745", "credit_report": {"CAIS_Account": {"CAIS_Account_DETAILS": None}}},
    ]
    for data in sections:
        f = report_facts_from_envelope({"success": True, "data": data}, today=ANCHOR)
        assert f.is_ntc is False and f.score == 745 and f.open_accounts == [] and f.past_due_total == 0.0


def test_single_dict_cais_account_details_collapse_still_parses() -> None:
    body = report_raw([cais_account(scheduled_emi="3500")])
    cais = cast("dict[str, Any]", cast("dict[str, Any]", cast("dict[str, Any]", body["data"])["credit_report"])["CAIS_Account"])
    cais["CAIS_Account_DETAILS"] = cais["CAIS_Account_DETAILS"][0]  # XML->JSON single-element collapse
    f = report_facts_from_envelope(body)
    assert len(f.open_accounts) == 1 and f.open_accounts[0].scheduled_emi == 3500.0


def test_non_dict_account_entries_are_ignored() -> None:
    body = report_raw([cais_account()])
    details = cast("list[Any]", cast("dict[str, Any]", cast("dict[str, Any]", cast("dict[str, Any]", body["data"])["credit_report"])["CAIS_Account"])["CAIS_Account_DETAILS"])
    details[:0] = [None, "x", 5, ["nested"]]
    assert len(report_facts_from_envelope(body).open_accounts) == 1


# --------------------------------------------------------------------------- report facts: OpenAccount rows


def test_open_account_rows_cover_open_accounts_only_with_synthetic_keys() -> None:
    f = facts(
        [
            cais_account(account_number="SYNTHACC0001"),
            cais_account(account_status="13", date_closed="20250110", account_number="SYNTHACC0002"),
            cais_account(account_number="SYNTHACC0003"),
        ]
    )
    assert [a.account_key for a in f.open_accounts] == ["acct_0", "acct_1"]  # indexed over the OPEN rows
    assert "SYNTHACC" not in f.model_dump_json()  # Account_Number is never carried


def test_scheduled_emi_is_positive_only() -> None:
    assert one(scheduled_emi="3500").scheduled_emi == 3500.0
    assert one(scheduled_emi="1,500").scheduled_emi == 1500.0
    assert one(scheduled_emi="0.01").scheduled_emi == 0.01
    for not_reported in ("0", "0.00", "", "-5", "abc", "  "):
        assert one(scheduled_emi=not_reported).scheduled_emi is None, not_reported  # '0' falls to estimation


def test_current_balance_is_clamped_to_non_negative() -> None:
    assert one(current_balance="120000").current_balance == 120000.0
    assert one(current_balance="120,000").current_balance == 120000.0
    for value in ("-500", "", "abc", "-0.5"):
        assert one(current_balance=value).current_balance == 0.0, value


def test_tenure_resolution_on_open_accounts() -> None:
    assert one(repayment_tenure="0", terms_duration="048").tenure_months == 48  # '0' -> Terms_Duration '048' -> 48
    assert one(repayment_tenure="12", terms_duration="024").tenure_months == 12
    assert one(repayment_tenure="0", terms_duration="").tenure_months is None
    assert one(repayment_tenure="0", terms_duration="024", terms_frequency="Q").tenure_months is None  # freq guard
    assert one(repayment_tenure="0", terms_duration="024", terms_frequency="").tenure_months == 24


def test_rate_portfolio_type_and_account_type() -> None:
    assert one(rate_of_interest="13.500").rate_of_interest == 13.5
    assert one(rate_of_interest="").rate_of_interest is None
    assert one(rate_of_interest="0").rate_of_interest is None
    assert one(portfolio_type="I").portfolio_type == "I"
    assert one(portfolio_type="R").portfolio_type == "R"
    assert one(portfolio_type="").portfolio_type is None
    assert one(portfolio_type="X").portfolio_type is None
    assert one(account_type="10").account_type == "10"
    assert one(account_type="").account_type == ""


def test_open_date_and_date_reported_parse() -> None:
    a = one(open_date="20240115", date_reported="20260920")
    assert a.open_date == date(2024, 1, 15) and a.date_reported == date(2026, 9, 20)
    b = one(open_date="", date_reported="garbage")
    assert b.open_date is None and b.date_reported is None


def test_is_stale_window_is_strictly_more_than_90_days_before_the_anchor() -> None:
    assert one(date_reported=days_before(0)).is_stale is False
    assert one(date_reported=days_before(90)).is_stale is False  # exactly 90 days: fresh
    assert one(date_reported=days_before(91)).is_stale is True  # >90: stale
    assert one(date_reported=days_before(400)).is_stale is True
    assert one(date_reported=ymd(ANCHOR + timedelta(days=10))).is_stale is False  # future-dated: not stale


def test_missing_or_unparseable_date_reported_is_stale() -> None:
    for value in ("", "garbage", "20261301", "2026"):
        assert one(date_reported=value).is_stale is True, value


def test_open_account_pincodes_are_filtered_deduped_and_per_account() -> None:
    f = facts(
        [
            cais_account(holder_zips=("560001", "20", "056001", "", "560001", " 110001 ", "5600 01")),
            cais_account(holder_zips=("400001",), account_number="SYNTHACC0002"),
            cais_account(holder_zips=("",), account_number="SYNTHACC0003"),
        ]
    )
    assert [a.pincodes for a in f.open_accounts] == [["560001", "110001"], ["400001"], []]  # order kept, junk dropped
    for a in f.open_accounts:
        assert all(PIN_RE.fullmatch(z) for z in a.pincodes)


def test_open_account_with_no_address_rows_has_no_pincodes() -> None:
    account = cais_account()
    account["CAIS_Holder_Address_Details"] = []
    assert facts([account]).open_accounts[0].pincodes == []
    account["CAIS_Holder_Address_Details"] = None
    assert facts([account]).open_accounts[0].pincodes == []


# --------------------------------------------------------------------------- bureau_pincodes_12m


def test_bureau_pincodes_12m_comes_from_open_accounts_within_365_days_only() -> None:
    f = facts(
        [
            cais_account(date_reported=days_before(10), holder_zips=("560001",)),  # open, recent: IN
            cais_account(
                date_reported=days_before(10),
                holder_zips=("700001",),
                account_status="13",
                date_closed="20250110",
                account_number="SYNTHACC0002",
            ),  # CLOSED, recent: out (open accounts only)
            cais_account(date_reported=days_before(365), holder_zips=("411001",), account_number="SYNTHACC0003"),  # 365d: IN
            cais_account(date_reported=days_before(366), holder_zips=("400001",), account_number="SYNTHACC0004"),  # 366d: OUT
            cais_account(date_reported="", holder_zips=("600001",), account_number="SYNTHACC0005"),  # unparseable: OUT
        ]
    )
    assert f.bureau_pincodes_12m == ["411001", "560001"]  # sorted


def test_bureau_pincodes_12m_filters_junk_dedupes_and_sorts() -> None:
    f = facts(
        [
            cais_account(date_reported=days_before(5), holder_zips=("560001", "20", "110001", "056001")),
            cais_account(date_reported=days_before(5), holder_zips=("560001", "110001", "400001"), account_number="SYNTHACC0002"),
        ]
    )
    assert f.bureau_pincodes_12m == ["110001", "400001", "560001"]


def test_bureau_pincodes_12m_keeps_future_dated_and_emi_stale_accounts() -> None:
    f = facts(
        [
            cais_account(date_reported=ymd(ANCHOR + timedelta(days=5)), holder_zips=("560001",)),  # future-dated: kept
            cais_account(date_reported=days_before(200), holder_zips=("110001",), account_number="SYNTHACC0002"),
        ]
    )
    assert f.open_accounts[1].is_stale is True  # stale for EMI (>90d) ...
    assert f.bureau_pincodes_12m == ["110001", "560001"]  # ... but still a candidate pincode within 12 months


def test_bureau_pincodes_12m_empty_when_nothing_qualifies() -> None:
    assert facts([]).bureau_pincodes_12m == []
    assert facts([cais_account(holder_zips=("",))]).bureau_pincodes_12m == []


def test_bureau_pincode_window_uses_the_today_anchor_when_there_is_no_report_date() -> None:
    body = report_raw([cais_account(date_reported="20260915", holder_zips=("560001",))], report_date=None)
    assert report_facts_from_envelope(body, today=date(2026, 10, 1)).bureau_pincodes_12m == ["560001"]
    assert report_facts_from_envelope(body, today=date(2028, 1, 1)).bureau_pincodes_12m == []


# --------------------------------------------------------------------------- holder DOB trap


def test_holder_dob_rows_are_never_read() -> None:
    base = facts([cais_account(holder_dob="19800101"), cais_account(holder_dob="19800101", account_number="SYNTHACC0002")])
    other = facts([cais_account(holder_dob="19601231"), cais_account(holder_dob="20050607", account_number="SYNTHACC0002")])
    assert base == other  # changing the contradictory holder DOBs changes nothing in the output
    dumped = base.model_dump_json() + json.dumps(base.model_dump(mode="json"))
    for needle in ("1980", "1990", "19800101", "19900101", "1980-01-01", "1990-01-01"):
        assert needle not in dumped, needle
    # and a body with the holder-detail block removed altogether gives the same facts
    body = report_raw([cais_account(), cais_account(account_number="SYNTHACC0002")])
    for account in cast("list[dict[str, Any]]", cast("dict[str, Any]", cast("dict[str, Any]", cast("dict[str, Any]", body["data"])["credit_report"])["CAIS_Account"])["CAIS_Account_DETAILS"]):
        del account["CAIS_Holder_Details"]
    assert report_facts_from_envelope(body) == facts([cais_account(), cais_account(account_number="SYNTHACC0002")])


def test_the_adapter_code_never_references_holder_details() -> None:
    tokens = _code_tokens(PACKAGE_DIR / "surepass" / "adapters.py")
    assert "CAIS_Holder_Details" not in tokens and "Date_of_birth" not in tokens
    assert not [t for t in tokens if "Holder_Details" in t or t.lower() == "dob_holder"]


# --------------------------------------------------------------------------- the scenario worlds through the fake client


def _fake_report(client: Any, pan: str = scn.PAN_OK) -> SurePassResult:
    return client.experian_report("chk", "h" * 64, name="Asha Verma", mobile="9876543210", pan=pan)


def _fake_pan(client: Any, pan: str = scn.PAN_OK) -> SurePassResult:
    return client.pan_comprehensive("chk", "h" * 64, pan)


def _fake_score(client: Any, pan: str = scn.PAN_OK) -> SurePassResult:
    return client.experian_score("chk", "h" * 64, name="Asha Verma", mobile="9876543210", pan=pan)


def _reference_estimate(a: OpenAccount, anchor: date) -> tuple[float, str]:
    """Independent oracle for the PRD 8.3 four branches (Task 6 owns the real estimator; this pins the FIXTURE numbers)."""
    if a.scheduled_emi is not None:
        return a.scheduled_emi, "reported"
    if a.tenure_months is not None and a.current_balance > 0:
        opened = a.open_date or anchor
        months = max(0, (anchor.year - opened.year) * 12 + (anchor.month - opened.month))
        remaining = a.tenure_months - months
        if remaining >= 1:
            return a.current_balance / remaining, "tenure_remaining"
        return a.current_balance / 12, "past_tenure"
    if a.portfolio_type == "R":
        return 0.05 * a.current_balance, "revolving_5pct"
    return a.current_balance / 36, "balance_36"


def test_six_account_table_matches_the_briefs_exact_numbers() -> None:
    result = _fake_report(scn.full_emi_four_branch_scenario())
    f = report_to_report_facts(result)
    assert f.report_date == date(2026, 10, 1) and f.score == 745
    assert [a.account_key for a in f.open_accounts] == [f"acct_{i}" for i in range(6)]
    a1, a2, a3, a4, a5, a6 = f.open_accounts
    # inputs
    assert a1.scheduled_emi == 3500.0
    assert (a2.scheduled_emi, a2.current_balance, a2.tenure_months, a2.open_date) == (None, 120000.0, 48, date(2024, 1, 15))
    assert (a3.scheduled_emi, a3.current_balance, a3.tenure_months, a3.open_date) == (None, 60000.0, 24, date(2024, 1, 15))
    assert (a4.scheduled_emi, a4.current_balance, a4.tenure_months, a4.portfolio_type) == (None, 40000.0, None, "R")
    assert (a5.scheduled_emi, a5.current_balance, a5.tenure_months, a5.portfolio_type) == (None, 90000.0, None, "I")
    assert a6.scheduled_emi == 9999.0 and a6.date_reported == date(2026, 6, 1)
    assert [a.is_stale for a in f.open_accounts] == [False, False, False, False, False, True]  # only a6 is stale
    # the delta-3 numbers, by branch
    got = [_reference_estimate(a, date(2026, 10, 1)) for a in f.open_accounts[:5]]
    assert got == [
        (3500.0, "reported"),
        (8000.0, "tenure_remaining"),  # 120000 / (48 - 33)
        (5000.0, "past_tenure"),  # 60000 / 12: remaining 24 - 33 < 1
        (2000.0, "revolving_5pct"),
        (2500.0, "balance_36"),
    ]
    non_stale_total = sum(amount for amount, _ in got)
    assert non_stale_total == 21000.0  # reported 3500 + estimated 17500; the stale 9999 is excluded


def test_full_happy_reported_emi_world() -> None:
    client = scn.full_happy_reported_emi_scenario()
    identity = pan_to_identity(_fake_pan(client))
    f = report_to_report_facts(_fake_report(client))
    assert identity.pan_valid is True and identity.pan_pincode == "560001"
    assert (f.band, f.score, f.report_date) == ("720-780", 745, date(2026, 10, 1))
    (a,) = f.open_accounts
    assert a.scheduled_emi == 3500.0 and a.is_stale is False and a.date_reported == date(2026, 9, 20)
    assert a.pincodes == ["560001"] and f.bureau_pincodes_12m == ["560001"]


def test_pincode_union_world() -> None:
    client = scn.full_pincode_union_scenario()
    identity = pan_to_identity(_fake_pan(client))
    f = report_to_report_facts(_fake_report(client))
    assert identity.pan_pincode == "560002"
    assert [a.pincodes for a in f.open_accounts] == [["110001"], ["400001"]]  # '20' junk dropped per account
    assert f.bureau_pincodes_12m == ["110001"]  # 400001 was reported 20250901: more than 12 months before 20261001


def test_no_pincode_world() -> None:
    client = scn.no_pincode_scenario()
    assert pan_to_identity(_fake_pan(client)).pan_pincode is None
    f = report_to_report_facts(_fake_report(client))
    assert f.bureau_pincodes_12m == [] and all(a.pincodes == [] for a in f.open_accounts)
    assert score_to_score_facts(_fake_score(client), pulled_on=PULLED).band == "720-780"


@pytest.mark.parametrize(
    ("variant", "expected_ntc"),
    [("empty", True), ("text", True), ("missing", True), ("four_digit", True)],
)
def test_ntc_scenarios_adapt_to_ntc_score_facts(variant: Any, expected_ntc: bool) -> None:
    client = scn.ntc_scenario(variant)
    f = score_to_score_facts(_fake_score(client, scn.PAN_NTC), pulled_on=PULLED)
    assert f.is_ntc is expected_ntc and f.band == "NTC" and f.score is None


def test_happy_quick_world_is_not_ntc() -> None:
    f = score_to_score_facts(_fake_score(scn.happy_quick_scenario()), pulled_on=PULLED)
    assert (f.band, f.score, f.is_ntc, f.source) == ("720-780", 745, False, "score")


def test_pan_invalid_worlds_adapt_to_a_non_valid_identity() -> None:
    for kind, pan in (("status_invalid", scn.PAN_INVALID), ("company", scn.PAN_COMPANY)):
        identity = pan_to_identity(_fake_pan(scn.pan_invalid_scenario(kind), pan))  # type: ignore[arg-type]
        assert identity.pan_valid is False, kind


# --------------------------------------------------------------------------- thin SurePassResult wrappers


def test_wrappers_delegate_to_the_raw_dict_entry_points() -> None:
    pan_result = result_of(PAN_ENDPOINT, pan_comprehensive_envelope())
    assert pan_to_identity(pan_result) == identity_from_envelope(pan_result.raw)

    score_result = result_of(SCORE_ENDPOINT, experian_score_envelope(credit_score="702"))
    assert score_to_score_facts(score_result, pulled_on=PULLED) == score_facts_from_envelope(
        score_result.raw, pulled_on=PULLED, source="score"
    )

    report_result = result_of(REPORT_ENDPOINT, experian_report_envelope(accounts=[cais_account(scheduled_emi="3500")]))
    assert report_to_report_facts(report_result) == report_facts_from_envelope(report_result.raw)
    assert report_to_report_facts(report_result, today=date(2027, 1, 1)) == report_facts_from_envelope(
        report_result.raw, today=date(2027, 1, 1)
    )


def test_score_wrapper_marks_a_report_endpoint_result_as_source_report() -> None:
    report_result = result_of(REPORT_ENDPOINT, experian_report_envelope(accounts=[cais_account()], credit_score="702"))
    f = score_to_score_facts(report_result, pulled_on=PULLED)
    assert f.source == "report" and f.score == 702 and f.band == "650-720"


def test_wrappers_surface_the_non_success_guard() -> None:
    bad = SurePassResult(
        endpoint=SCORE_ENDPOINT,
        http_status=200,
        success=False,
        envelope_status_code=200,
        message=None,
        message_code="no_record_found",
        data=None,
        raw={"success": False, "data": {}},
    )
    with pytest.raises(ValueError, match="non-success"):
        pan_to_identity(bad)
    with pytest.raises(ValueError, match="non-success"):
        score_to_score_facts(bad, pulled_on=PULLED)
    with pytest.raises(ValueError, match="non-success"):
        report_to_report_facts(bad)


def test_adapters_do_not_mutate_their_input() -> None:
    for body in (
        raw(pan_comprehensive_envelope()),
        raw(experian_score_envelope()),
        _fake_report(scn.full_emi_four_branch_scenario()).raw,
    ):
        before = copy.deepcopy(body)
        identity_from_envelope(body)
        score_facts_from_envelope(body, pulled_on=PULLED, source="report")
        report_facts_from_envelope(body, today=ANCHOR)
        assert body == before


# --------------------------------------------------------------------------- totality over junk bureau data


def _leaf_paths(obj: Any, prefix: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(obj, dict):
        for key, value in cast("dict[str, Any]", obj).items():
            yield from _leaf_paths(value, (*prefix, key))
    elif isinstance(obj, list):
        for index, value in enumerate(cast("list[Any]", obj)):
            yield from _leaf_paths(value, (*prefix, index))
    else:
        yield prefix


def _set_path(obj: Any, path: tuple[Any, ...], value: Any) -> None:
    cursor = obj
    for step in path[:-1]:
        cursor = cursor[step]
    cursor[path[-1]] = value


JUNK_REPLACEMENTS: list[object] = [None, "", " ", "abc", "NaN", "inf", "-1", "0", "9" * 40, "20261301", "\x00", 5, True, [], {}, [None], {"x": "y"}]


def _run_all_adapters(body: dict[str, Any]) -> None:
    identity_from_envelope(body)
    score_facts_from_envelope(body, pulled_on=PULLED, source="score")
    score_facts_from_envelope(body, pulled_on=PULLED, source="report")
    report_facts_from_envelope(body, today=ANCHOR)
    report_facts_from_envelope(body)


def test_adapters_never_raise_on_junk_in_any_single_leaf_of_realistic_envelopes() -> None:
    bodies = [
        raw(pan_comprehensive_envelope()),
        raw(experian_score_envelope()),
        report_raw(
            [
                cais_account(scheduled_emi="3500", holder_zips=("560001", "20")),
                cais_account(portfolio_type="R", account_type="10", terms_duration="024", account_number="SYNTHACC0002"),
            ],
            caps_30="1",
            caps_90="2",
            caps_180="3",
        ),
    ]
    runs = 0
    for body in bodies:
        for path in list(_leaf_paths(body)):
            for junk in JUNK_REPLACEMENTS:
                mutated = copy.deepcopy(body)
                _set_path(mutated, path, copy.deepcopy(junk))
                mutated["success"] = True  # only the guard may refuse; keep the body a success envelope
                _run_all_adapters(mutated)
                runs += 1
    assert runs > 2000  # the loop really exercised a large mutation surface


def test_adapters_never_raise_when_every_leaf_is_junk_at_once() -> None:
    body = report_raw([cais_account(scheduled_emi="3500", holder_zips=("560001", "20"))], caps_180="3")
    for junk in JUNK_REPLACEMENTS:
        mutated = copy.deepcopy(body)
        for path in list(_leaf_paths(mutated)):
            if path == ("success",):
                continue
            _set_path(mutated, path, copy.deepcopy(junk))
        _run_all_adapters(mutated)


@pytest.mark.parametrize("junk", JUNK_REPLACEMENTS, ids=[type(j).__name__ + str(i) for i, j in enumerate(JUNK_REPLACEMENTS)])
def test_adapters_never_raise_when_data_or_its_sections_are_junk(junk: object) -> None:
    for body in (
        {"success": True, "data": junk},
        {"success": True, "data": {"credit_score": junk, "credit_report": junk}},
        {"success": True, "data": {"credit_score": "745", "credit_report": {"CAIS_Account": junk, "TotalCAPS_Summary": junk, "CreditProfileHeader": junk}}},
        {"success": True, "data": {"credit_score": "745", "credit_report": {"CAIS_Account": {"CAIS_Account_DETAILS": [junk, {"Date_Closed": junk}]}}}},
        {"success": True, "data": {"credit_report": {"CAIS_Account": {"CAIS_Account_DETAILS": [{"CAIS_Holder_Address_Details": junk, "Date_Reported": "20260901"}]}}}},
    ):
        _run_all_adapters(body)


# --------------------------------------------------------------------------- structure pins


def _imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


def test_adapters_import_surface_is_pure() -> None:
    tree = ast.parse((PACKAGE_DIR / "surepass" / "adapters.py").read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "math",
        "re",
        "collections.abc",
        "datetime",
        "typing",
        "eligibility_agent.derive.range",
        "eligibility_agent.graph.state",
        "eligibility_agent.surepass.client",
    }
    assert _imports(tree) <= allowed, _imports(tree) - allowed


def test_adapters_public_entry_points() -> None:
    for name in (
        "identity_from_envelope",
        "score_facts_from_envelope",
        "report_facts_from_envelope",
        "pan_to_identity",
        "score_to_score_facts",
        "report_to_report_facts",
        "as_list",
        "dig",
        "parse_amount",
        "parse_int_sentinel",
        "parse_rate",
        "parse_yyyymmdd",
        "parse_iso_date",
        "resolve_tenure_months",
        "is_open_account",
    ):
        assert callable(getattr(adapters, name)), name
    # deleted v1.0 surface stays deleted; band bucketing lives in derive/range (adapters only imports it)
    for name in ("_map_gender", "age_years", "parse_score", "NON_BUREAU_FAILURE_CODES", "BAND_ORDER"):
        assert not hasattr(adapters, name), name
    assert adapters.band_from_score is band_from_score
