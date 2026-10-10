"""Task 4: surepass/client.py, the only module that performs SurePass HTTP (v1.1 taxonomy, K12 timeouts, K4 sink).

Everything runs on httpx.MockTransport (no sockets: the suite is pytest-socket locked) with MemoryRepo +
MemoryRawStore as the audit sink. The load-bearing pin is the taxonomy reversal: EVERY success:false envelope
raises SurePassUpstreamError on ALL THREE endpoints (it is never a returned result, never NTC, never pan_invalid).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import httpx
import pytest
from repo_builders import PAN_A, PAN_B, FrozenClock, make_check

from eligibility_agent import config
from eligibility_agent.db import repo as repo_module
from eligibility_agent.db.repo import MemoryRepo, SurePassCallRow
from eligibility_agent.graph.state import ErrorKind
from eligibility_agent.surepass import client as client_module
from eligibility_agent.surepass.client import (
    BUREAU_ENDPOINTS,
    PAN_ENDPOINT,
    REPORT_ENDPOINT,
    SCORE_ENDPOINT,
    TIMEOUT_BY_ENDPOINT,
    FakeCall,
    FakeSurePassClient,
    LiveCallsDisabled,
    LiveSurePassClient,
    Scenario,
    SurePassAuthError,
    SurePassClient,
    SurePassError,
    SurePassProtocolError,
    SurePassRateLimited,
    SurePassResult,
    SurePassTimeout,
    SurePassTransportError,
    SurePassUpstreamError,
    SurePassValidationError,
    interpret_response,
    parse_surepass_body,
)

Handler = Callable[[httpx.Request], httpx.Response]

CHECK_ID = "chk-1"
PAN = "ABCDE1234F"
PAN_2 = "ZZZZZ9999Z"
NAME = "Asha Verma"
MOBILE = "9876543210"
ENDPOINTS = [PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT]

PAN_BODY: dict[str, Any] = {
    "id_number": PAN,
    "get_address": True,
    "get_contact": False,
    "masked_aadhaar_variant": "v1",
}
BUREAU_BODY: dict[str, Any] = {"name": NAME, "mobile": MOBILE, "pan": PAN, "consent": "Y"}
EXPECTED_BODY: dict[str, dict[str, Any]] = {
    PAN_ENDPOINT: PAN_BODY,
    SCORE_ENDPOINT: BUREAU_BODY,
    REPORT_ENDPOINT: BUREAU_BODY,
}


def ok_envelope(client_id: str = "cid-1") -> dict[str, Any]:
    return {
        "status_code": 200,
        "success": True,
        "message": "Success",
        "message_code": "success",
        "data": {"client_id": client_id, "credit_score": "745"},
    }


def new_repo() -> MemoryRepo:
    repo = MemoryRepo(clock=FrozenClock())
    repo.insert_check(make_check(id=CHECK_ID))
    return repo


def respond(status: int, body: object) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return handler


def raising(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def forbidden(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no HTTP may be attempted, got {request.method} {request.url}")


class SpySink:
    """Records every kwarg the client hands the sink; optionally raises (sink exceptions must propagate)."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raises = raises

    def log_surepass_call(
        self,
        *,
        check_id: str,
        pan_hash: str,
        endpoint: str,
        client_id: str | None,
        http_status: int | None,
        success: bool | None,
        message_code: str | None,
        latency_ms: int | None,
        error_kind: ErrorKind | None,
        raw_envelope: dict[str, Any] | None,
    ) -> str | None:
        self.calls.append(
            {
                "check_id": check_id,
                "pan_hash": pan_hash,
                "endpoint": endpoint,
                "client_id": client_id,
                "http_status": http_status,
                "success": success,
                "message_code": message_code,
                "latency_ms": latency_ms,
                "error_kind": error_kind,
                "raw_envelope": raw_envelope,
            }
        )
        if self._raises is not None:
            raise self._raises
        return None


def make_live(
    handler: Handler,
    sink: MemoryRepo | SpySink,
    *,
    live: bool = True,
    token: str = "tok-123",
    base_url: str = "https://kyc-api.surepass.app",
) -> LiveSurePassClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return LiveSurePassClient(token=token, live=live, sink=sink, base_url=base_url, http=http)


def invoke(
    client: SurePassClient,
    endpoint: str,
    *,
    check_id: str = CHECK_ID,
    pan_hash: str = PAN_A,
    pan: str = PAN,
) -> SurePassResult:
    if endpoint == PAN_ENDPOINT:
        return client.pan_comprehensive(check_id, pan_hash, pan)
    if endpoint == SCORE_ENDPOINT:
        return client.experian_score(check_id, pan_hash, name=NAME, mobile=MOBILE, pan=pan)
    return client.experian_report(check_id, pan_hash, name=NAME, mobile=MOBILE, pan=pan)


def make_fake(
    endpoint: str,
    scenario: Scenario | dict[str, Scenario],
    *,
    sink: MemoryRepo | SpySink | None = None,
    calls_enabled: bool = True,
) -> FakeSurePassClient:
    """A fake with exactly one endpoint configured (every other endpoint stays unconfigured)."""
    if endpoint == PAN_ENDPOINT:
        return FakeSurePassClient(pan=scenario, sink=sink, calls_enabled=calls_enabled)
    if endpoint == SCORE_ENDPOINT:
        return FakeSurePassClient(score=scenario, sink=sink, calls_enabled=calls_enabled)
    return FakeSurePassClient(report=scenario, sink=sink, calls_enabled=calls_enabled)


# --------------------------------------------------------------------------- constants, kill switch


def test_endpoint_constants_equal_the_private_copies_in_db_repo() -> None:
    # Task-2 deferment: the client OWNS these; db.repo keeps private mirrors for the storage slugs / bureau cap.
    assert PAN_ENDPOINT == repo_module._PAN_ENDPOINT == "/api/v1/pan/pan-comprehensive"
    assert SCORE_ENDPOINT == repo_module._SCORE_ENDPOINT == "/api/v1/credit-report-experian/score"
    assert REPORT_ENDPOINT == repo_module._REPORT_ENDPOINT == "/api/v1/credit-report-experian/fetch-report"
    assert BUREAU_ENDPOINTS == repo_module._BUREAU_ENDPOINTS == frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT})
    assert set(repo_module._ENDPOINT_SLUGS) == {PAN_ENDPOINT, SCORE_ENDPOINT, REPORT_ENDPOINT}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_kill_switch_raises_before_any_io_and_writes_no_sink_row(endpoint: str) -> None:
    repo = new_repo()
    client = make_live(forbidden, repo, live=False)
    assert client.calls_enabled is False
    with pytest.raises(LiveCallsDisabled):
        invoke(client, endpoint)
    assert repo.list_calls() == []


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_empty_token_raises_auth_error_before_any_io_and_writes_no_row(endpoint: str) -> None:
    repo = new_repo()
    client = make_live(forbidden, repo, live=True, token="")
    assert client.calls_enabled is False
    with pytest.raises(SurePassAuthError):
        invoke(client, endpoint)
    assert repo.list_calls() == []


def test_kill_switch_wins_over_an_empty_token() -> None:
    client = make_live(forbidden, new_repo(), live=False, token="")
    with pytest.raises(LiveCallsDisabled):
        invoke(client, PAN_ENDPOINT)


@pytest.mark.parametrize(
    ("live", "token", "enabled"),
    [(True, "tok", True), (False, "tok", False), (True, "", False), (False, "", False)],
)
def test_calls_enabled_is_live_and_a_non_empty_token(live: bool, token: str, enabled: bool) -> None:
    assert make_live(forbidden, new_repo(), live=live, token=token).calls_enabled is enabled


def test_sink_is_never_touched_when_the_kill_switch_trips() -> None:
    sink = SpySink()
    with pytest.raises(LiveCallsDisabled):
        invoke(make_live(forbidden, sink, live=False), SCORE_ENDPOINT)
    assert sink.calls == []


# --------------------------------------------------------------------------- request shape and timeouts


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_live_request_url_headers_and_exact_fixed_body(endpoint: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_envelope())

    invoke(make_live(handler, new_repo(), base_url="https://sp.example"), endpoint)
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"https://sp.example{endpoint}"
    assert request.headers["authorization"] == "Bearer tok-123"
    assert json.loads(request.content) == EXPECTED_BODY[endpoint]


def test_default_base_url_is_the_surepass_kyc_host() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_envelope())

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = LiveSurePassClient(token="t", live=True, sink=new_repo(), http=http)
    invoke(client, PAN_ENDPOINT)
    assert str(seen[0].url) == "https://kyc-api.surepass.app/api/v1/pan/pan-comprehensive"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_fake_records_the_same_exact_bodies_as_live(endpoint: str) -> None:
    fake = make_fake(endpoint, ok_envelope())
    invoke(fake, endpoint)
    assert fake.calls == [FakeCall(endpoint, CHECK_ID, PAN, EXPECTED_BODY[endpoint])]
    assert fake.endpoints_called() == [endpoint]


def test_pan_body_carries_masked_aadhaar_variant_and_bureau_bodies_carry_consent_y() -> None:
    assert PAN_BODY["masked_aadhaar_variant"] == "v1"
    fake = FakeSurePassClient(pan=ok_envelope(), score=ok_envelope(), report=ok_envelope())
    invoke(fake, PAN_ENDPOINT)
    invoke(fake, SCORE_ENDPOINT)
    invoke(fake, REPORT_ENDPOINT)
    pan_call, score_call, report_call = fake.calls
    assert pan_call.body == {
        "id_number": PAN,
        "get_address": True,
        "get_contact": False,
        "masked_aadhaar_variant": "v1",
    }
    assert "consent" not in pan_call.body
    assert score_call.body["consent"] == "Y"
    assert report_call.body["consent"] == "Y"


def test_timeouts_are_built_from_the_config_finals() -> None:
    assert TIMEOUT_BY_ENDPOINT == {
        PAN_ENDPOINT: config.PAN_TIMEOUT_S,
        SCORE_ENDPOINT: config.SCORE_TIMEOUT_S,
        REPORT_ENDPOINT: config.REPORT_TIMEOUT_S,
    }
    assert (config.PAN_TIMEOUT_S, config.SCORE_TIMEOUT_S, config.REPORT_TIMEOUT_S) == (8.0, 12.0, 20.0)
    assert not hasattr(client_module, "READ_TIMEOUT_S")  # the v1.0 5s/60s constants are void


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_per_endpoint_timeout_reaches_httpx_with_the_connect_phase_capped(endpoint: str) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json=ok_envelope())

    invoke(make_live(handler, new_repo()), endpoint)
    t = TIMEOUT_BY_ENDPOINT[endpoint]
    assert seen == [{"connect": min(config.CONNECT_TIMEOUT_S, t), "read": t, "write": t, "pool": t}]


def test_connect_timeout_is_min_of_the_cap_and_the_endpoint_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(TIMEOUT_BY_ENDPOINT, PAN_ENDPOINT, 3.0)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json=ok_envelope())

    invoke(make_live(handler, new_repo()), PAN_ENDPOINT)
    assert seen == [{"connect": 3.0, "read": 3.0, "write": 3.0, "pool": 3.0}]


# --------------------------------------------------------------------------- parse_surepass_body


def test_parse_success_envelope_builds_a_typed_result() -> None:
    body = ok_envelope("cid-9")
    result = parse_surepass_body(SCORE_ENDPOINT, 200, body)
    assert result == SurePassResult(
        endpoint=SCORE_ENDPOINT,
        http_status=200,
        success=True,
        envelope_status_code=200,
        message="Success",
        message_code="success",
        data={"client_id": "cid-9", "credit_score": "745"},
        raw=body,
    )
    assert result.client_id == "cid-9"


def test_parse_success_false_envelope_still_parses_the_raise_lives_in_interpret_response() -> None:
    result = parse_surepass_body(PAN_ENDPOINT, 200, {"success": False, "message_code": "invalid_pan", "data": None})
    assert result.success is False
    assert result.message_code == "invalid_pan"
    assert result.data is None
    assert result.client_id is None


def test_parse_is_total_over_loosely_typed_fields() -> None:
    body: dict[str, Any] = {"success": True, "status_code": True, "message": 5, "message_code": None, "data": [1]}
    result = parse_surepass_body(PAN_ENDPOINT, 200, body)
    assert (result.envelope_status_code, result.message, result.message_code, result.data) == (None, None, None, None)
    assert result.raw is body


@pytest.mark.parametrize("client_id", [None, "", 7])
def test_client_id_is_only_a_non_empty_string(client_id: object) -> None:
    result = parse_surepass_body(PAN_ENDPOINT, 200, {"success": True, "data": {"client_id": client_id}})
    assert result.client_id is None


def test_parse_validation_shape_raises_a_validation_error_with_the_errors() -> None:
    body = {"errors": {"id_number": "required", "n": 3}, "message": "Input payload validation failed"}
    with pytest.raises(SurePassValidationError) as info:
        parse_surepass_body(PAN_ENDPOINT, 422, body)
    exc = info.value
    assert exc.errors == {"id_number": "required", "n": "3"}
    assert exc.message == "Input payload validation failed"
    assert exc.http_status == 422
    assert exc.endpoint == PAN_ENDPOINT


@pytest.mark.parametrize("body", [{"message": "Not found"}, {"errors": ["x"]}, {"errors": None, "message": "m"}])
def test_parse_validation_shape_is_decided_by_shape_and_never_crashes(body: dict[str, Any]) -> None:
    with pytest.raises(SurePassValidationError):
        parse_surepass_body(PAN_ENDPOINT, 400, body)


@pytest.mark.parametrize("body", ["<html>", None, [1, 2], 5, b"x"])
def test_parse_non_dict_body_is_a_protocol_error(body: object) -> None:
    with pytest.raises(SurePassProtocolError):
        parse_surepass_body(PAN_ENDPOINT, 200, body)


@pytest.mark.parametrize("body", [{}, {"foo": 1}, {"data": {}}, {"success": "true", "data": {}}])
def test_parse_unrecognised_dict_is_a_protocol_error(body: dict[str, Any]) -> None:
    with pytest.raises(SurePassProtocolError):
        parse_surepass_body(PAN_ENDPOINT, 200, body)


# --------------------------------------------------------------------------- interpret_response taxonomy


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("status", [401, 403])
def test_auth_statuses_raise_auth_error(endpoint: str, status: int) -> None:
    with pytest.raises(SurePassAuthError) as info:
        interpret_response(endpoint, status, {"success": False, "message_code": "ip_not_whitelisted"})
    assert info.value.http_status == status


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_429_raises_rate_limited(endpoint: str) -> None:
    with pytest.raises(SurePassRateLimited):
        interpret_response(endpoint, 429, {"message": "slow down"})


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("status", [500, 502, 503, 599])
def test_5xx_raises_upstream_even_with_a_non_json_body(endpoint: str, status: int) -> None:
    with pytest.raises(SurePassUpstreamError):
        interpret_response(endpoint, status, "<html>bad gateway</html>")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("status", [200, 400, 404, 422])
def test_every_success_false_envelope_raises_upstream_on_all_three_endpoints(endpoint: str, status: int) -> None:
    """THE regression pin (v1.1 NTC reversal): success:false is never returned, never NTC, never pan_invalid."""
    body = {"status_code": status, "success": False, "message": "No record", "message_code": "no_record", "data": None}
    with pytest.raises(SurePassUpstreamError) as info:
        interpret_response(endpoint, status, body)
    assert info.value.detail == "success_false:no_record"
    assert info.value.http_status == status
    assert info.value.endpoint == endpoint


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_old_ntc_and_pan_invalid_shapes_now_raise(endpoint: str) -> None:
    # v1.0 treated these as typed results (score no-hit -> NTC, PAN invalid -> pan_not_verified): both now raise.
    ntc_shape = {"success": False, "message_code": "no_hit", "data": {"credit_score": "", "client_id": "c"}}
    pan_invalid_shape = {"success": False, "message_code": "invalid_id_number", "data": None}
    for body in (ntc_shape, pan_invalid_shape):
        with pytest.raises(SurePassUpstreamError):
            interpret_response(endpoint, 200, body)


def test_success_false_without_a_message_code_still_raises() -> None:
    with pytest.raises(SurePassUpstreamError):
        interpret_response(PAN_ENDPOINT, 200, {"success": False})


def test_non_bureau_failure_hook_is_deleted() -> None:
    assert not hasattr(client_module, "NON_BUREAU_FAILURE_CODES")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("body", ["<html>nope</html>", None, [1], 42])
def test_non_dict_bodies_raise_protocol_error(endpoint: str, body: object) -> None:
    with pytest.raises(SurePassProtocolError):
        interpret_response(endpoint, 200, body)


def test_validation_shape_raises_validation_error_not_upstream() -> None:
    with pytest.raises(SurePassValidationError):
        interpret_response(SCORE_ENDPOINT, 422, {"errors": {"mobile": "bad"}, "message": "Input payload validation failed"})


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_success_true_returns_a_result_whose_success_is_always_true(endpoint: str) -> None:
    result = interpret_response(endpoint, 200, ok_envelope())
    assert result.success is True
    assert result.http_status == 200
    assert result.endpoint == endpoint


def test_exception_taxonomy_hierarchy() -> None:
    assert issubclass(SurePassTimeout, SurePassTransportError)
    assert not issubclass(SurePassTransportError, SurePassTimeout)
    for cls in (
        LiveCallsDisabled,
        SurePassAuthError,
        SurePassRateLimited,
        SurePassUpstreamError,
        SurePassTransportError,
        SurePassProtocolError,
        SurePassValidationError,
    ):
        assert issubclass(cls, SurePassError)
    assert not issubclass(LiveCallsDisabled, SurePassTransportError)


def test_exception_messages_never_carry_body_content_or_identifiers() -> None:
    body = {"success": False, "message_code": "x", "message": f"bad PAN {PAN}", "data": {"pan": PAN, "name": NAME}}
    with pytest.raises(SurePassUpstreamError) as info:
        interpret_response(PAN_ENDPOINT, 200, body)
    for needle in (PAN, NAME, MOBILE):
        assert needle not in str(info.value)


# --------------------------------------------------------------------------- live client: one sink row per attempt

FAILURE_CASES = [
    pytest.param(401, {"success": False, "message_code": "unauthorized"}, SurePassAuthError, "auth", False, "unauthorized", True, id="401"),
    pytest.param(403, {"message": "ip not whitelisted"}, SurePassAuthError, "auth", None, None, True, id="403"),
    pytest.param(429, {"message": "slow down"}, SurePassRateLimited, "rate_limited", None, None, True, id="429"),
    pytest.param(500, {"success": False, "message_code": "boom"}, SurePassUpstreamError, "upstream", False, "boom", True, id="500-json"),
    pytest.param(503, "<html>down</html>", SurePassUpstreamError, "upstream", None, None, False, id="503-html"),
    pytest.param(200, {"success": False, "message_code": "no_hit", "data": None}, SurePassUpstreamError, "upstream", False, "no_hit", True, id="200-success-false"),
    pytest.param(400, {"success": False, "message_code": "bad", "data": None}, SurePassUpstreamError, "upstream", False, "bad", True, id="400-success-false"),
    pytest.param(422, {"errors": {"id_number": "x"}, "message": "Input payload validation failed"}, SurePassValidationError, "validation", None, None, True, id="422-validation"),
    pytest.param(404, "<html>Not Found</html>", SurePassProtocolError, "protocol", None, None, False, id="404-html"),
    pytest.param(200, "null", SurePassProtocolError, "protocol", None, None, False, id="200-null"),
    pytest.param(200, [1, 2], SurePassProtocolError, "protocol", None, None, False, id="200-list"),
    pytest.param(200, {"foo": "bar"}, SurePassProtocolError, "protocol", None, None, True, id="200-unrecognised"),
]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_live_success_logs_exactly_one_row_and_stores_the_raw_envelope(endpoint: str) -> None:
    repo = new_repo()
    body = ok_envelope("cid-42")
    result = invoke(make_live(respond(200, body), repo), endpoint)
    assert result.success is True and result.raw == body
    (row,) = repo.list_calls(CHECK_ID)
    assert (row.check_id, row.pan_hash, row.endpoint) == (CHECK_ID, PAN_A, endpoint)
    assert row.client_id == "cid-42"
    assert (row.http_status, row.success, row.message_code, row.error_kind) == (200, True, "success", None)
    assert row.billable is True
    assert isinstance(row.latency_ms, int) and row.latency_ms >= 0
    assert row.storage_key is not None
    assert repo.raw_store.get_raw(row.storage_key) == body
    assert row.storage_key in repo.raw_store.keys()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("status", "body", "exc_type", "error_kind", "success", "message_code", "has_raw"), FAILURE_CASES
)
def test_live_failure_raises_after_logging_exactly_one_row(
    endpoint: str,
    status: int,
    body: object,
    exc_type: type[SurePassError],
    error_kind: ErrorKind,
    success: bool | None,
    message_code: str | None,
    has_raw: bool,
) -> None:
    repo = new_repo()
    with pytest.raises(exc_type):
        invoke(make_live(respond(status, body), repo), endpoint)
    (row,) = repo.list_calls(CHECK_ID)
    assert row.endpoint == endpoint
    assert row.http_status == status
    assert row.error_kind == error_kind
    assert row.success is success
    assert row.message_code == message_code
    assert row.client_id is None
    assert row.billable is False
    assert isinstance(row.latency_ms, int) and row.latency_ms >= 0
    if has_raw:
        assert row.storage_key is not None
        assert repo.raw_store.get_raw(row.storage_key) == body
    else:
        assert row.storage_key is None
        assert repo.raw_store.keys() == []


def test_success_false_row_keeps_the_peeked_client_id_and_is_not_billable() -> None:
    repo = new_repo()
    body = {"success": False, "message_code": "no_hit", "data": {"client_id": "cid-7"}}
    with pytest.raises(SurePassUpstreamError):
        invoke(make_live(respond(200, body), repo), SCORE_ENDPOINT)
    (row,) = repo.list_calls()
    assert (row.client_id, row.http_status, row.success, row.billable) == ("cid-7", 200, False, False)


def test_every_body_is_handed_to_the_sink_verbatim_and_non_dict_bodies_as_none() -> None:
    nested: dict[str, Any] = {
        "success": True,
        "status_code": 200,
        "data": {"client_id": "c", "credit_report": {"x": [1, {"y": None}], "name": "Zoë"}},
        "extra": {"unknown_key": [True, False]},
    }
    cases: list[tuple[int, object, dict[str, Any] | None]] = [
        (200, nested, nested),
        (200, {"success": False, "message_code": "m", "data": None}, {"success": False, "message_code": "m", "data": None}),
        (422, {"errors": {"a": "b"}, "message": "bad"}, {"errors": {"a": "b"}, "message": "bad"}),
        (200, {"foo": "bar"}, {"foo": "bar"}),
        (429, {"x": 1}, {"x": 1}),
        (200, [1, 2], None),
        (404, "<html>", None),
        (200, "null", None),
    ]
    for status, body, expected in cases:
        sink = SpySink()
        try:
            invoke(make_live(respond(status, body), sink), SCORE_ENDPOINT)
        except SurePassError:
            pass
        (call,) = sink.calls
        assert call["raw_envelope"] == expected, (status, body)


def test_sink_receives_check_id_pan_hash_and_endpoint_per_call() -> None:
    sink = SpySink()
    client = make_live(respond(200, ok_envelope()), sink)
    invoke(client, REPORT_ENDPOINT, check_id="chk-X", pan_hash=PAN_B)
    (call,) = sink.calls
    assert call["check_id"] == "chk-X"
    assert call["pan_hash"] == PAN_B
    assert call["endpoint"] == REPORT_ENDPOINT
    assert set(call) == set(inspect.signature(SpySink.log_surepass_call).parameters) - {"self"}


# --------------------------------------------------------------------------- live client: transport + timeouts


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "exc", [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.WriteTimeout("t"), httpx.PoolTimeout("t")]
)
def test_httpx_timeout_logs_one_timeout_row_then_raises_surepass_timeout(endpoint: str, exc: httpx.TimeoutException) -> None:
    repo = new_repo()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise exc

    with pytest.raises(SurePassTimeout) as info:
        invoke(make_live(handler, repo), endpoint)
    assert isinstance(info.value, SurePassTransportError)
    assert len(calls) == 1  # zero retries: a timed-out call has unknown spend
    (row,) = repo.list_calls(CHECK_ID)
    assert row.endpoint == endpoint
    assert row.error_kind == "timeout"
    assert row.http_status is None
    assert row.success is None and row.message_code is None and row.client_id is None
    assert row.billable is True
    assert row.storage_key is None
    assert isinstance(row.latency_ms, int) and row.latency_ms >= 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("exc", [httpx.ConnectError("c"), httpx.ReadError("r"), httpx.RemoteProtocolError("p")])
def test_non_timeout_httpx_errors_raise_plain_transport_error_with_kind_transport(endpoint: str, exc: httpx.HTTPError) -> None:
    repo = new_repo()
    with pytest.raises(SurePassTransportError) as info:
        invoke(make_live(raising(exc), repo), endpoint)
    assert not isinstance(info.value, SurePassTimeout)
    (row,) = repo.list_calls(CHECK_ID)
    assert (row.error_kind, row.http_status, row.billable, row.storage_key) == ("transport", None, False, None)


def test_transport_error_message_has_no_request_content() -> None:
    with pytest.raises(SurePassTimeout) as info:
        invoke(make_live(raising(httpx.ReadTimeout(f"timed out for {PAN} {NAME}")), new_repo()), SCORE_ENDPOINT)
    for needle in (PAN, NAME, MOBILE, "tok-123"):
        assert needle not in str(info.value)
    assert info.value.__cause__ is not None  # the original httpx error stays chained for operators


# --------------------------------------------------------------------------- latency


@pytest.mark.parametrize("failure", [None, httpx.ReadTimeout("t")])
def test_latency_ms_is_measured_around_the_http_call(
    monkeypatch: pytest.MonkeyPatch, failure: httpx.TimeoutException | None
) -> None:
    ticks = itertools.chain([100.0, 100.25], itertools.repeat(999.0))
    monkeypatch.setattr(client_module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    sink = SpySink()
    handler = respond(200, ok_envelope()) if failure is None else raising(failure)
    try:
        invoke(make_live(handler, sink), PAN_ENDPOINT)
    except SurePassTimeout:
        pass
    (call,) = sink.calls
    assert call["latency_ms"] == 250
    assert isinstance(call["latency_ms"], int)


def test_latency_is_real_elapsed_time_for_a_slow_handler() -> None:
    import time as real_time

    def slow(request: httpx.Request) -> httpx.Response:
        real_time.sleep(0.03)
        return httpx.Response(200, json=ok_envelope())

    sink = SpySink()
    invoke(make_live(slow, sink), PAN_ENDPOINT)
    assert sink.calls[0]["latency_ms"] >= 25


# --------------------------------------------------------------------------- strict audit: sink exceptions propagate


def test_sink_exception_on_the_success_path_propagates() -> None:
    boom = RuntimeError("audit store down")
    with pytest.raises(RuntimeError) as info:
        invoke(make_live(respond(200, ok_envelope()), SpySink(raises=boom)), PAN_ENDPOINT)
    assert info.value is boom


def test_sink_is_called_before_the_taxonomy_exception_is_raised() -> None:
    # If the sink raises, its error surfaces INSTEAD of the SurePassError: proof the row is logged first.
    boom = RuntimeError("audit store down")
    sink = SpySink(raises=boom)
    with pytest.raises(RuntimeError) as info:
        invoke(make_live(respond(200, {"success": False, "message_code": "x"}), sink), SCORE_ENDPOINT)
    assert info.value is boom
    assert len(sink.calls) == 1


def test_sink_is_called_before_a_timeout_is_raised() -> None:
    boom = RuntimeError("audit store down")
    sink = SpySink(raises=boom)
    with pytest.raises(RuntimeError) as info:
        invoke(make_live(raising(httpx.ReadTimeout("t")), sink), REPORT_ENDPOINT)
    assert info.value is boom
    assert sink.calls[0]["error_kind"] == "timeout"


# --------------------------------------------------------------------------- live client lifecycle and protocol


def test_close_only_closes_an_owned_http_client() -> None:
    injected = httpx.Client(transport=httpx.MockTransport(forbidden))
    LiveSurePassClient(token="t", live=True, sink=new_repo(), http=injected).close()
    assert injected.is_closed is False
    injected.close()

    owned = LiveSurePassClient(token="t", live=True, sink=new_repo())
    owned.close()
    assert owned._http.is_closed is True


def test_live_and_fake_satisfy_the_client_protocol_with_identical_signatures() -> None:
    live: SurePassClient = make_live(forbidden, new_repo())
    fake: SurePassClient = FakeSurePassClient()
    assert live.calls_enabled is True and fake.calls_enabled is True

    def params(cls: type, name: str) -> list[tuple[str, Any]]:
        return [(p.name, p.kind) for p in inspect.signature(getattr(cls, name)).parameters.values() if p.name != "self"]

    for name in ("pan_comprehensive", "experian_score", "experian_report"):
        assert params(LiveSurePassClient, name) == params(FakeSurePassClient, name) == params(SurePassClient, name)
    assert [n for n, _ in params(SurePassClient, "pan_comprehensive")] == ["check_id", "pan_hash", "pan"]
    for name in ("experian_score", "experian_report"):
        assert [n for n, _ in params(SurePassClient, name)] == ["check_id", "pan_hash", "name", "mobile", "pan"]
    live_ctor = inspect.signature(LiveSurePassClient.__init__).parameters
    assert [n for n in live_ctor if n != "self"] == ["token", "live", "sink", "base_url", "http"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for n, p in live_ctor.items() if n != "self")


def test_client_module_imports_no_policy_graph_db_or_env_machinery() -> None:
    source = Path(client_module.__file__ or "").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    internal = {m for m in imported if m.startswith("eligibility_agent")}
    # config (timeout Finals) and the pure-leaf literal registry (ErrorKind) are the only internal imports.
    assert internal <= {"eligibility_agent.config", "eligibility_agent.graph.state"}
    assert not {m for m in imported if m.split(".")[0] in {"supabase", "langgraph", "langchain_core", "fastapi", "os"}}
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    }
    assert not names & {"environ", "getenv"}  # the kill switch arrives through the constructor, never the env


# --------------------------------------------------------------------------- fake client


def test_fake_unconfigured_endpoint_is_an_assertion_error_and_records_nothing() -> None:
    repo = new_repo()
    fake = FakeSurePassClient(score=ok_envelope(), sink=repo)
    with pytest.raises(AssertionError, match="unexpected SurePass call"):
        invoke(fake, PAN_ENDPOINT)
    with pytest.raises(AssertionError):
        invoke(fake, REPORT_ENDPOINT)
    assert fake.calls == []
    assert repo.list_calls() == []


def test_fake_per_pan_mapping_resolves_by_pan_and_unmapped_pan_is_keyerror() -> None:
    repo = new_repo()
    fake = FakeSurePassClient(
        pan={PAN: ok_envelope("cid-a"), PAN_2: (401, {"success": False, "message_code": "unauthorized"})},
        sink=repo,
    )
    assert invoke(fake, PAN_ENDPOINT, pan=PAN).client_id == "cid-a"
    with pytest.raises(SurePassAuthError):
        invoke(fake, PAN_ENDPOINT, pan=PAN_2)
    with pytest.raises(KeyError) as info:
        invoke(fake, PAN_ENDPOINT, pan="QQQQQ0000Q")
    assert info.value.args == ("QQQQQ0000Q",)
    assert [c.pan for c in fake.calls] == [PAN, PAN_2]  # the unmapped attempt recorded nothing
    assert len(repo.list_calls()) == 2


def test_fake_per_pan_mapping_works_on_the_bureau_endpoints_too() -> None:
    fake = FakeSurePassClient(score={PAN: ok_envelope()}, report={PAN: ok_envelope()})
    invoke(fake, SCORE_ENDPOINT)
    invoke(fake, REPORT_ENDPOINT)
    with pytest.raises(KeyError):
        invoke(fake, SCORE_ENDPOINT, pan=PAN_2)
    assert fake.endpoints_called() == [SCORE_ENDPOINT, REPORT_ENDPOINT]


def test_fake_dict_and_empty_dict_are_scenarios_not_pan_mappings() -> None:
    # An envelope dict (success/data/message...) is a Scenario even though a PAN mapping is also a dict;
    # {} is the unrecognised-envelope scenario, never an empty PAN map.
    with pytest.raises(SurePassProtocolError):
        invoke(FakeSurePassClient(pan={}), PAN_ENDPOINT)
    with pytest.raises(SurePassProtocolError):
        invoke(FakeSurePassClient(pan={"foo": 1}), PAN_ENDPOINT)
    assert invoke(FakeSurePassClient(pan=ok_envelope()), PAN_ENDPOINT).success is True


def test_fake_dict_scenario_resolves_through_the_real_interpret_response() -> None:
    body = ok_envelope("cid-5")
    result = invoke(FakeSurePassClient(score=body), SCORE_ENDPOINT)
    assert result == interpret_response(SCORE_ENDPOINT, 200, body)
    assert result.http_status == 200 and result.raw == body


SCENARIO_TAXONOMY: list[tuple[Scenario, type[SurePassError]]] = [
    ({"success": False, "message_code": "no_hit"}, SurePassUpstreamError),
    ((200, {"success": False, "message_code": "no_hit"}), SurePassUpstreamError),
    ({"errors": {"a": "b"}, "message": "Input payload validation failed"}, SurePassValidationError),
    ({"unexpected": True}, SurePassProtocolError),
    ((401, {"message": "no"}), SurePassAuthError),
    ((403, {}), SurePassAuthError),
    ((429, {}), SurePassRateLimited),
    ((503, {}), SurePassUpstreamError),
    ((422, {"errors": {}, "message": "bad"}), SurePassValidationError),
]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(("scenario", "exc_type"), SCENARIO_TAXONOMY)
def test_fake_dict_and_tuple_scenarios_raise_the_same_taxonomy_as_live(
    endpoint: str, scenario: Scenario, exc_type: type[SurePassError]
) -> None:
    with pytest.raises(exc_type):
        invoke(make_fake(endpoint, scenario), endpoint)


def test_fake_surepass_result_scenario_is_returned_as_is_and_logged() -> None:
    repo = new_repo()
    canned = SurePassResult(
        endpoint=SCORE_ENDPOINT,
        http_status=200,
        success=True,
        envelope_status_code=200,
        message="ok",
        message_code="success",
        data={"client_id": "cid-3"},
        raw={"success": True, "data": {"client_id": "cid-3"}},
    )
    assert invoke(make_fake(SCORE_ENDPOINT, canned, sink=repo), SCORE_ENDPOINT) is canned
    (row,) = repo.list_calls()
    assert (row.client_id, row.http_status, row.success, row.error_kind, row.billable) == ("cid-3", 200, True, None, True)
    assert row.storage_key is not None and repo.raw_store.get_raw(row.storage_key) == canned.raw


def test_fake_cannot_return_a_success_false_result_the_live_client_never_would() -> None:
    impossible = SurePassResult(
        endpoint=SCORE_ENDPOINT,
        http_status=200,
        success=False,
        envelope_status_code=200,
        message=None,
        message_code="no_hit",
        data=None,
        raw={"success": False, "message_code": "no_hit"},
    )
    repo = new_repo()
    with pytest.raises(SurePassUpstreamError):
        invoke(make_fake(SCORE_ENDPOINT, impossible, sink=repo), SCORE_ENDPOINT)
    (row,) = repo.list_calls()
    assert (row.error_kind, row.success, row.billable) == ("upstream", False, False)


def test_fake_calls_disabled_raises_live_calls_disabled_and_records_nothing() -> None:
    repo = new_repo()
    fake = FakeSurePassClient(pan=ok_envelope(), score=ok_envelope(), sink=repo, calls_enabled=False)
    assert fake.calls_enabled is False
    for endpoint in (PAN_ENDPOINT, SCORE_ENDPOINT):
        with pytest.raises(LiveCallsDisabled):
            invoke(fake, endpoint)
    # the kill switch is checked before the scenario lookup, exactly like live (report is unconfigured)
    with pytest.raises(LiveCallsDisabled):
        invoke(fake, REPORT_ENDPOINT)
    assert fake.calls == [] and fake.endpoints_called() == []
    assert repo.list_calls() == []


def test_fake_records_calls_in_order_with_check_id_pan_and_body() -> None:
    fake = FakeSurePassClient(pan=ok_envelope(), score=ok_envelope())
    invoke(fake, PAN_ENDPOINT, check_id="c1", pan=PAN)
    invoke(fake, SCORE_ENDPOINT, check_id="c2", pan=PAN)
    invoke(fake, PAN_ENDPOINT, check_id="c3", pan=PAN)
    assert fake.endpoints_called() == [PAN_ENDPOINT, SCORE_ENDPOINT, PAN_ENDPOINT]
    assert [c.check_id for c in fake.calls] == ["c1", "c2", "c3"]
    assert {f.name for f in dataclasses.fields(FakeCall)} == {"endpoint", "check_id", "pan", "body"}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_fake_scripted_timeout_logs_the_sink_row_before_raising(endpoint: str) -> None:
    repo = new_repo()
    fake = make_fake(endpoint, SurePassTimeout(endpoint, "timeout: ReadTimeout"), sink=repo)
    with pytest.raises(SurePassTimeout):
        invoke(fake, endpoint)
    (row,) = repo.list_calls(CHECK_ID)
    assert row.endpoint == endpoint and row.pan_hash == PAN_A
    assert (row.error_kind, row.http_status, row.success, row.billable, row.storage_key) == ("timeout", None, None, True, None)
    assert len(fake.calls) == 1  # the attempt itself is recorded as well


@pytest.mark.parametrize(
    ("exc", "error_kind", "billable"),
    [
        (SurePassTimeout(PAN_ENDPOINT, "t"), "timeout", True),
        (SurePassTransportError(PAN_ENDPOINT, "t"), "transport", False),
        (SurePassAuthError(PAN_ENDPOINT, "a", 401), "auth", False),
        (SurePassRateLimited(PAN_ENDPOINT, "r", 429), "rate_limited", False),
        (SurePassUpstreamError(PAN_ENDPOINT, "u", 503), "upstream", False),
        (SurePassProtocolError(PAN_ENDPOINT, "p", 200), "protocol", False),
        (SurePassValidationError(PAN_ENDPOINT, {"a": "b"}, "m", 422), "validation", False),
    ],
)
def test_fake_exception_scenarios_log_a_mapped_row_with_their_own_http_status(
    exc: SurePassError, error_kind: ErrorKind, billable: bool
) -> None:
    repo = new_repo()
    with pytest.raises(type(exc)) as info:
        invoke(make_fake(PAN_ENDPOINT, exc, sink=repo), PAN_ENDPOINT)
    assert info.value is exc
    (row,) = repo.list_calls()
    assert (row.error_kind, row.http_status, row.billable) == (error_kind, exc.http_status, billable)


def test_fake_live_calls_disabled_scenario_logs_no_row_like_live() -> None:
    repo = new_repo()
    with pytest.raises(LiveCallsDisabled):
        invoke(make_fake(PAN_ENDPOINT, LiveCallsDisabled(PAN_ENDPOINT, "off"), sink=repo), PAN_ENDPOINT)
    assert repo.list_calls() == []


def test_fake_logs_the_row_before_raising_scripted_exceptions() -> None:
    boom = RuntimeError("audit store down")
    sink = SpySink(raises=boom)
    with pytest.raises(RuntimeError) as info:
        invoke(make_fake(SCORE_ENDPOINT, SurePassTimeout(SCORE_ENDPOINT, "t"), sink=sink), SCORE_ENDPOINT)
    assert info.value is boom
    assert sink.calls[0]["error_kind"] == "timeout"


def test_fake_without_a_sink_still_works() -> None:
    fake = FakeSurePassClient(pan=ok_envelope(), score=SurePassTimeout(SCORE_ENDPOINT, "t"))
    assert invoke(fake, PAN_ENDPOINT).success is True
    with pytest.raises(SurePassTimeout):
        invoke(fake, SCORE_ENDPOINT)


def test_fake_hands_the_raw_envelope_to_the_sink_like_live() -> None:
    sink = SpySink()
    body = ok_envelope("cid-6")
    invoke(make_fake(PAN_ENDPOINT, body, sink=sink), PAN_ENDPOINT)
    (call,) = sink.calls
    assert call["raw_envelope"] == body
    assert call["pan_hash"] == PAN_A and call["check_id"] == CHECK_ID
    assert isinstance(call["latency_ms"], int) and call["latency_ms"] >= 0


# --------------------------------------------------------------------------- fake <-> live sink parity

PARITY_CASES: list[Any] = [
    pytest.param(respond(200, ok_envelope()), ok_envelope(), id="success"),
    pytest.param(respond(200, {"success": False, "message_code": "no_hit", "data": None}), {"success": False, "message_code": "no_hit", "data": None}, id="success-false-200"),
    pytest.param(respond(400, {"success": False, "message_code": "bad"}), (400, {"success": False, "message_code": "bad"}), id="success-false-400"),
    pytest.param(respond(401, {"success": False, "message_code": "unauthorized"}), (401, {"success": False, "message_code": "unauthorized"}), id="401"),
    pytest.param(respond(429, {"message": "slow"}), (429, {"message": "slow"}), id="429"),
    pytest.param(respond(503, {"message": "down"}), (503, {"message": "down"}), id="503"),
    pytest.param(respond(422, {"errors": {"a": "b"}, "message": "bad"}), (422, {"errors": {"a": "b"}, "message": "bad"}), id="422"),
    pytest.param(respond(200, {"foo": "bar"}), {"foo": "bar"}, id="unrecognised"),
    pytest.param(raising(httpx.ReadTimeout("t")), SurePassTimeout("x", "timeout: ReadTimeout"), id="timeout"),
    pytest.param(raising(httpx.ConnectError("c")), SurePassTransportError("x", "transport: ConnectError"), id="transport"),
]


def _view(repo: MemoryRepo, row: SurePassCallRow) -> dict[str, Any]:
    out = dataclasses.asdict(row)
    for volatile in ("id", "called_at", "purge_after", "storage_key", "latency_ms"):
        out.pop(volatile)
    out["raw"] = None if row.storage_key is None else repo.raw_store.get_raw(row.storage_key)
    return out


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(("handler", "scenario"), PARITY_CASES)
def test_fake_and_live_produce_identical_sink_rows_and_exceptions(
    endpoint: str, handler: Handler, scenario: Scenario
) -> None:
    live_repo, fake_repo = new_repo(), new_repo()
    live_exc: BaseException | None = None
    fake_exc: BaseException | None = None
    live_result: SurePassResult | None = None
    fake_result: SurePassResult | None = None
    try:
        live_result = invoke(make_live(handler, live_repo), endpoint)
    except SurePassError as exc:
        live_exc = exc
    try:
        fake_result = invoke(make_fake(endpoint, scenario, sink=fake_repo), endpoint)
    except SurePassError as exc:
        fake_exc = exc
    assert type(live_exc) is type(fake_exc)
    assert (live_result is None) == (fake_result is None)
    if live_result is not None and fake_result is not None:
        assert live_result == fake_result
    (live_row,) = live_repo.list_calls()
    (fake_row,) = fake_repo.list_calls()
    assert _view(live_repo, live_row) == _view(fake_repo, fake_row)


def test_error_kind_values_used_are_all_registry_literals() -> None:
    kinds = {row.values[3] for row in FAILURE_CASES}
    assert kinds <= set(get_args(ErrorKind))
