"""The ONLY module that performs SurePass HTTP (PRD v1.1 sections 5.12, 8.1, 10; plan K2 / K4 / K12 / K21).

Three layers, all here so fake and live can never drift:

* ``parse_surepass_body`` / ``interpret_response``: pure envelope parsing plus the failure taxonomy, shared by the
  live client, the fake and any cache reprocessing.
* ``LiveSurePassClient``: httpx, bearer auth, exact FIXED request bodies, per-endpoint timeouts, zero retries (a timed-out
  call has unknown spend), a call-time kill switch, and audit logging through the Repo sink.
* ``FakeSurePassClient``: scenario-driven, no network ever; dict / tuple scenarios resolve through the SAME
  ``interpret_response`` and every attempt hits the sink exactly like live (exception scenarios log BEFORE raising).

Failure taxonomy (v1.1), evaluated in order on HTTP status, envelope ``success`` and ``message_code`` only (never
message text): 401/403 -> ``SurePassAuthError``; 429 -> ``SurePassRateLimited``; >= 500 -> ``SurePassUpstreamError``;
``{errors, message}`` shape -> ``SurePassValidationError``; non-JSON / unrecognised -> ``SurePassProtocolError``; and
EVERY ``success: false`` envelope, on all three endpoints and at any HTTP status, -> ``SurePassUpstreamError``.
``success: false`` is never NTC and never ``pan_invalid``: ``SurePassResult.success`` is therefore always True on return.

Audit contract (K4): exactly ONE ``log_surepass_call`` row per HTTP attempt, written BEFORE the exception is raised on
every failure path and after parsing on success. ``LiveCallsDisabled`` and the empty-token ``SurePassAuthError`` happen
before any I/O and write NO row. The sink receives only audit fields plus the parsed envelope verbatim (the sink stores
it in the private raw bucket); it never gets request bodies. Sink exceptions propagate: the audit trail is strict.

Imports are deliberately minimal: the timeout Finals from ``config`` and the pure-leaf literal registry
(``ErrorKind``). No policy, graph, db, or environment access: the kill switch arrives through the constructor.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeAlias, cast

import httpx

from eligibility_agent.config import CONNECT_TIMEOUT_S, PAN_TIMEOUT_S, REPORT_TIMEOUT_S, SCORE_TIMEOUT_S
from eligibility_agent.graph.state import ErrorKind

# --------------------------------------------------------------------------- constants (FIXED)

SUREPASS_BASE_URL: Final[str] = "https://kyc-api.surepass.app"
# OWNED here (K21). db.repo keeps private mirrors for storage slugs and the bureau cap; a test pins parity.
PAN_ENDPOINT: Final[str] = "/api/v1/pan/pan-comprehensive"
SCORE_ENDPOINT: Final[str] = "/api/v1/credit-report-experian/score"
REPORT_ENDPOINT: Final[str] = "/api/v1/credit-report-experian/fetch-report"
BUREAU_ENDPOINTS: Final[frozenset[str]] = frozenset({SCORE_ENDPOINT, REPORT_ENDPOINT})
CONSENT_Y: Final[str] = "Y"
MASKED_AADHAAR_VARIANT: Final[str] = "v1"

# Per-endpoint READ timeout (K12; the values live in config). httpx timeouts are per-phase, not wall-clock: a
# trickling response can outlive these. The graph-level timeout (GRAPH_TIMEOUT_SECONDS) is the wall-clock backstop.
TIMEOUT_BY_ENDPOINT: Final[dict[str, float]] = {
    PAN_ENDPOINT: PAN_TIMEOUT_S,
    SCORE_ENDPOINT: SCORE_TIMEOUT_S,
    REPORT_ENDPOINT: REPORT_TIMEOUT_S,
}

# --------------------------------------------------------------------------- exception taxonomy


class SurePassError(Exception):
    """Base. Messages never contain PAN / mobile / name / token / body content (they land in logs)."""

    def __init__(self, endpoint: str, detail: str, http_status: int | None = None) -> None:
        super().__init__(f"{endpoint}: {detail}")
        self.endpoint = endpoint
        self.detail = detail
        self.http_status = http_status


class LiveCallsDisabled(SurePassError):
    """Kill switch. Raised BEFORE any I/O; writes NO sink row."""


class SurePassAuthError(SurePassError):
    """HTTP 401/403 (ip_not_whitelisted, auth_failed, ...) or an empty token (pre-I/O: no row then)."""


class SurePassRateLimited(SurePassError):
    """HTTP 429."""


class SurePassUpstreamError(SurePassError):
    """HTTP >= 500, or ANY success:false envelope (detail ``success_false:{message_code}``)."""


class SurePassTransportError(SurePassError):
    """httpx.HTTPError (DNS / connect / read). Spend status UNKNOWN, so never auto-retried."""


class SurePassTimeout(SurePassTransportError):
    """httpx.TimeoutException (connect / read / write / pool). A BUREAU timeout locks the PAN (graph decides)."""


class SurePassProtocolError(SurePassError):
    """Non-JSON / non-dict body (e.g. a 404 HTML page), or a dict matching neither envelope shape."""


class SurePassValidationError(SurePassError):
    """The ``{"errors": {...}, "message": "Input payload validation failed"}`` shape (no success/data keys)."""

    def __init__(self, endpoint: str, errors: dict[str, str], message: str, http_status: int | None = None) -> None:
        super().__init__(endpoint, "request validation failed", http_status)
        self.errors = errors
        self.message = message


def _error_kind(exc: SurePassError) -> ErrorKind:
    """Exception class -> the ``surepass_calls.error_kind`` literal. Most specific class first."""
    if isinstance(exc, SurePassTimeout):
        return "timeout"
    if isinstance(exc, SurePassTransportError):
        return "transport"
    if isinstance(exc, SurePassAuthError):
        return "auth"
    if isinstance(exc, SurePassRateLimited):
        return "rate_limited"
    if isinstance(exc, SurePassUpstreamError):
        return "upstream"
    if isinstance(exc, SurePassValidationError):
        return "validation"
    return "protocol"  # SurePassProtocolError (and any bare SurePassError a test scripts into the fake)


# --------------------------------------------------------------------------- typed result


def _client_id(data: object) -> str | None:
    """``data["client_id"]`` when ``data`` is a dict and the id is a non-empty str."""
    if not isinstance(data, dict):
        return None
    value = cast("dict[str, Any]", data).get("client_id")
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True, slots=True)
class SurePassResult:
    endpoint: str
    http_status: int
    success: bool  # always True on anything a client returns: success:false and the validation shape raise
    envelope_status_code: int | None  # body["status_code"] when an int
    message: str | None  # audit only; business code NEVER branches on it
    message_code: str | None
    data: dict[str, Any] | None  # body["data"] when a dict, else None
    raw: dict[str, Any]  # the exact parsed JSON body; never to the LLM or LcView

    @property
    def client_id(self) -> str | None:
        return _client_id(self.data)


# --------------------------------------------------------------------------- audit sink + client protocol


class CallLogSink(Protocol):
    """Structurally satisfied by ``db.repo.Repo`` (Supabase and Memory). One call per HTTP attempt."""

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
    ) -> str | None: ...


class SurePassClient(Protocol):
    """The DI seam graph nodes depend on, never the concrete class. ``check_id`` and ``pan_hash`` are per-call
    arguments (the client is a process-lifetime singleton): they key the audit row."""

    @property
    def calls_enabled(self) -> bool: ...

    def pan_comprehensive(self, check_id: str, pan_hash: str, pan: str) -> SurePassResult: ...

    def experian_score(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult: ...

    def experian_report(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult: ...


# --------------------------------------------------------------------------- pure parsing


def parse_surepass_body(endpoint: str, http_status: int, body: object) -> SurePassResult:
    """Parse the two envelope shapes. Raises Protocol / Validation errors; a ``success:false`` envelope PARSES here
    (``interpret_response`` is where it raises). Total: no shape of ``body`` escapes as a non-SurePassError."""
    if not isinstance(body, dict):
        raise SurePassProtocolError(endpoint, "non-dict body", http_status)
    envelope = cast("dict[str, Any]", body)
    success = envelope.get("success")
    if isinstance(success, bool):
        status_code = envelope.get("status_code")
        message = envelope.get("message")
        message_code = envelope.get("message_code")
        data = envelope.get("data")
        return SurePassResult(
            endpoint=endpoint,
            http_status=http_status,
            success=success,
            envelope_status_code=status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None,
            message=message if isinstance(message, str) else None,
            message_code=message_code if isinstance(message_code, str) else None,
            data=cast("dict[str, Any]", data) if isinstance(data, dict) else None,
            raw=envelope,
        )
    if "errors" in envelope or ("message" in envelope and "data" not in envelope):
        # Branch on SHAPE, never on message text.
        errors = envelope.get("errors")
        flat = {str(k): str(v) for k, v in cast("dict[Any, Any]", errors).items()} if isinstance(errors, dict) else {}
        raise SurePassValidationError(endpoint, flat, str(envelope.get("message") or ""), http_status)
    raise SurePassProtocolError(endpoint, "unrecognized envelope", http_status)


def _success_false(endpoint: str, http_status: int, message_code: str | None) -> SurePassUpstreamError:
    return SurePassUpstreamError(endpoint, f"success_false:{message_code}", http_status)


def interpret_response(endpoint: str, http_status: int, body: object) -> SurePassResult:
    """Status taxonomy, in order. Returns only a ``success:true`` result; everything else raises."""
    if http_status in (401, 403):
        raise SurePassAuthError(endpoint, f"http {http_status}", http_status)
    if http_status == 429:
        raise SurePassRateLimited(endpoint, "http 429", http_status)
    if http_status >= 500:
        raise SurePassUpstreamError(endpoint, f"http {http_status}", http_status)
    result = parse_surepass_body(endpoint, http_status, body)
    if not result.success:
        # v1.1 reversal: success:false is NEVER a typed result (not NTC, not pan_invalid), on ALL endpoints.
        raise _success_false(endpoint, http_status, result.message_code)
    return result


# --------------------------------------------------------------------------- shared audit helpers


def _peek_envelope(body: object) -> dict[str, Any] | None:
    return cast("dict[str, Any]", body) if isinstance(body, dict) else None


def _peek_success(envelope: dict[str, Any] | None) -> bool | None:
    value = None if envelope is None else envelope.get("success")
    return value if isinstance(value, bool) else None


def _peek_message_code(envelope: dict[str, Any] | None) -> str | None:
    value = None if envelope is None else envelope.get("message_code")
    return value if isinstance(value, str) else None


def _log(
    sink: CallLogSink | None,
    *,
    check_id: str,
    pan_hash: str,
    endpoint: str,
    client_id: str | None,
    http_status: int | None,
    success: bool | None,
    message_code: str | None,
    latency_ms: int,
    error_kind: ErrorKind | None,
    raw_envelope: dict[str, Any] | None,
) -> None:
    if sink is None:
        return
    sink.log_surepass_call(
        check_id=check_id,
        pan_hash=pan_hash,
        endpoint=endpoint,
        client_id=client_id,
        http_status=http_status,
        success=success,
        message_code=message_code,
        latency_ms=latency_ms,
        error_kind=error_kind,
        raw_envelope=raw_envelope,
    )


def _interpret_and_log(
    sink: CallLogSink | None,
    *,
    check_id: str,
    pan_hash: str,
    endpoint: str,
    http_status: int,
    body: object,
    latency_ms: int,
) -> SurePassResult:
    """One HTTP response -> one sink row -> result or raise. Shared by the live client and the fake's dict/tuple
    scenarios, so both log identically. On a raise the row carries the PEEKED fields and is written first."""
    envelope = _peek_envelope(body)
    try:
        result = interpret_response(endpoint, http_status, body)
    except SurePassError as exc:
        _log(
            sink,
            check_id=check_id,
            pan_hash=pan_hash,
            endpoint=endpoint,
            client_id=_client_id(None if envelope is None else envelope.get("data")),
            http_status=http_status,
            success=_peek_success(envelope),
            message_code=_peek_message_code(envelope),
            latency_ms=latency_ms,
            error_kind=_error_kind(exc),
            raw_envelope=envelope,
        )
        raise
    _log(
        sink,
        check_id=check_id,
        pan_hash=pan_hash,
        endpoint=endpoint,
        client_id=result.client_id,
        http_status=http_status,
        success=result.success,
        message_code=result.message_code,
        latency_ms=latency_ms,
        error_kind=None,
        raw_envelope=envelope,
    )
    return result


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


# --------------------------------------------------------------------------- live client


class LiveSurePassClient:
    """httpx client. ``live`` is the SUREPASS_LIVE kill switch (``Settings.surepass_live``), checked per call, pre-I/O."""

    def __init__(
        self,
        *,
        token: str,
        live: bool,
        sink: CallLogSink,
        base_url: str = SUREPASS_BASE_URL,
        http: httpx.Client | None = None,
    ) -> None:
        self._token = token
        self._live = live
        self._sink = sink
        self._base_url = base_url.rstrip("/")
        self._owns_http = http is None
        # Timeouts are passed per request (they differ per endpoint), so the client carries no default of its own.
        self._http: httpx.Client = http if http is not None else httpx.Client()

    @property
    def calls_enabled(self) -> bool:
        return self._live and bool(self._token)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def pan_comprehensive(self, check_id: str, pan_hash: str, pan: str) -> SurePassResult:
        body: dict[str, Any] = {
            "id_number": pan,
            "get_address": True,
            "get_contact": False,
            "masked_aadhaar_variant": MASKED_AADHAAR_VARIANT,
        }
        return self._post(check_id, pan_hash, PAN_ENDPOINT, body)

    def experian_score(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._post(check_id, pan_hash, SCORE_ENDPOINT, _bureau_body(name=name, mobile=mobile, pan=pan))

    def experian_report(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._post(check_id, pan_hash, REPORT_ENDPOINT, _bureau_body(name=name, mobile=mobile, pan=pan))

    def _post(self, check_id: str, pan_hash: str, endpoint: str, body: dict[str, Any]) -> SurePassResult:
        if not self._live:
            raise LiveCallsDisabled(endpoint, "SUREPASS_LIVE is not 'true'")  # strictly before any I/O or sink use
        if not self._token:
            raise SurePassAuthError(endpoint, "missing SurePass token")  # pre-I/O: no row
        timeout_s = TIMEOUT_BY_ENDPOINT[endpoint]
        started = time.monotonic()
        try:
            response = self._http.post(
                f"{self._base_url}{endpoint}",
                json=body,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=httpx.Timeout(timeout_s, connect=min(CONNECT_TIMEOUT_S, timeout_s)),
            )
        except httpx.HTTPError as exc:  # NO retry: a timed-out call has unknown spend
            latency_ms = _elapsed_ms(started)
            error: SurePassTransportError = (
                SurePassTimeout(endpoint, f"timeout: {type(exc).__name__}")
                if isinstance(exc, httpx.TimeoutException)
                else SurePassTransportError(endpoint, f"transport: {type(exc).__name__}")
            )
            _log(
                self._sink,
                check_id=check_id,
                pan_hash=pan_hash,
                endpoint=endpoint,
                client_id=None,
                http_status=None,
                success=None,
                message_code=None,
                latency_ms=latency_ms,
                error_kind=_error_kind(error),
                raw_envelope=None,
            )
            raise error from exc
        latency_ms = _elapsed_ms(started)
        parsed: object
        try:
            parsed = response.json()
        except ValueError:
            parsed = response.text  # e.g. the 404-with-HTML case: a protocol error carrying no JSON envelope
        return _interpret_and_log(
            self._sink,
            check_id=check_id,
            pan_hash=pan_hash,
            endpoint=endpoint,
            http_status=response.status_code,
            body=parsed,
            latency_ms=latency_ms,
        )


def _bureau_body(*, name: str, mobile: str, pan: str) -> dict[str, Any]:
    # consent "Y" is sent ONLY because the admission guard already wrote a consent record for this pull
    # (PRD section 5 rule 1); Experian does not enforce the field, our consents row is the real gate.
    return {"name": name, "mobile": mobile, "pan": pan, "consent": CONSENT_Y}


# --------------------------------------------------------------------------- fake client (tests + evals)

#   dict            -> interpret_response(endpoint, 200, dict)    (same parsing path as live)
#   (status, dict)  -> interpret_response(endpoint, status, dict)  (exercises the 401/429/5xx/422 taxonomy)
#   SurePassResult  -> returned as-is (a success:false result raises like live: live never returns one)
#   SurePassError   -> logged to the sink, then raised (SurePassTimeout included; LiveCallsDisabled logs nothing)
Scenario: TypeAlias = SurePassResult | SurePassError | dict[str, Any] | tuple[int, dict[str, Any]]
ScenarioSpec: TypeAlias = Scenario | Mapping[str, Scenario]  # a Mapping is keyed by the PAN sent on the call

_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset({"success", "status_code", "message", "message_code", "data", "errors"})


@dataclass(frozen=True, slots=True)
class FakeCall:
    endpoint: str
    check_id: str
    pan: str
    body: dict[str, Any]  # in-memory only, for request-shape assertions; never persisted


def _is_pan_mapping(spec: ScenarioSpec) -> bool:
    """A dict is both an envelope Scenario and a PAN-keyed Mapping; tell them apart by content. A PAN mapping is
    non-empty, has no envelope key (success / data / errors / ...), and every value is itself a Scenario. ``{}`` and
    scalar-valued dicts are envelope scenarios (which then fail as unrecognised envelopes, like live)."""
    if isinstance(spec, (SurePassResult, SurePassError, tuple)) or not spec:
        return False
    mapping = cast("Mapping[str, object]", spec)
    if mapping.keys() & _ENVELOPE_KEYS:
        return False
    return all(isinstance(v, (SurePassResult, SurePassError, dict, tuple)) for v in mapping.values())


class FakeSurePassClient:
    """Scenario-driven client; no network, ever. Each endpoint accepts a ``Scenario`` or a ``Mapping[pan, Scenario]``.

    Unconfigured endpoint -> AssertionError (cache-hit / cap tests rely on it); mapping without the PAN ->
    KeyError(pan); ``calls_enabled=False`` -> LiveCallsDisabled with nothing recorded or logged. Latency is reported
    as 0 ms: nothing is measured because no I/O happens.
    """

    def __init__(
        self,
        *,
        pan: ScenarioSpec | None = None,
        score: ScenarioSpec | None = None,
        report: ScenarioSpec | None = None,
        sink: CallLogSink | None = None,
        calls_enabled: bool = True,
    ) -> None:
        self._specs: dict[str, ScenarioSpec | None] = {PAN_ENDPOINT: pan, SCORE_ENDPOINT: score, REPORT_ENDPOINT: report}
        self._sink = sink
        self._calls_enabled = calls_enabled
        self.calls: list[FakeCall] = []  # ordered; the eval policy_consistency source of truth

    @property
    def calls_enabled(self) -> bool:
        return self._calls_enabled

    def endpoints_called(self) -> list[str]:
        return [c.endpoint for c in self.calls]

    def pan_comprehensive(self, check_id: str, pan_hash: str, pan: str) -> SurePassResult:
        body: dict[str, Any] = {
            "id_number": pan,
            "get_address": True,
            "get_contact": False,
            "masked_aadhaar_variant": MASKED_AADHAAR_VARIANT,
        }
        return self._respond(check_id, pan_hash, PAN_ENDPOINT, pan, body)

    def experian_score(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._respond(check_id, pan_hash, SCORE_ENDPOINT, pan, _bureau_body(name=name, mobile=mobile, pan=pan))

    def experian_report(self, check_id: str, pan_hash: str, *, name: str, mobile: str, pan: str) -> SurePassResult:
        return self._respond(check_id, pan_hash, REPORT_ENDPOINT, pan, _bureau_body(name=name, mobile=mobile, pan=pan))

    def _respond(self, check_id: str, pan_hash: str, endpoint: str, pan: str, body: dict[str, Any]) -> SurePassResult:
        if not self._calls_enabled:
            raise LiveCallsDisabled(endpoint, "fake client has calls_enabled=False")  # nothing recorded or logged
        spec = self._specs[endpoint]
        if spec is None:
            raise AssertionError(f"unexpected SurePass call: {endpoint}")
        scenario = cast("Mapping[str, Scenario]", spec)[pan] if _is_pan_mapping(spec) else cast("Scenario", spec)
        self.calls.append(FakeCall(endpoint, check_id, pan, body))
        return self._resolve(scenario, check_id=check_id, pan_hash=pan_hash, endpoint=endpoint)

    def _resolve(self, scenario: Scenario, *, check_id: str, pan_hash: str, endpoint: str) -> SurePassResult:
        if isinstance(scenario, tuple):
            return self._interpret(scenario[0], scenario[1], check_id, pan_hash, endpoint)
        if isinstance(scenario, dict):
            return self._interpret(200, scenario, check_id, pan_hash, endpoint)
        if isinstance(scenario, SurePassResult):
            failure = None if scenario.success else _success_false(endpoint, scenario.http_status, scenario.message_code)
            _log(
                self._sink,
                check_id=check_id,
                pan_hash=pan_hash,
                endpoint=endpoint,
                client_id=scenario.client_id,
                http_status=scenario.http_status,
                success=scenario.success,
                message_code=scenario.message_code,
                latency_ms=0,
                error_kind=None if failure is None else _error_kind(failure),
                raw_envelope=scenario.raw,
            )
            if failure is not None:
                raise failure
            return scenario
        if not isinstance(scenario, LiveCallsDisabled):  # live raises it pre-I/O, so it never produces a row
            _log(
                self._sink,
                check_id=check_id,
                pan_hash=pan_hash,
                endpoint=endpoint,
                client_id=None,
                http_status=scenario.http_status,
                success=None,
                message_code=None,
                latency_ms=0,
                error_kind=_error_kind(scenario),
                raw_envelope=None,
            )
        raise scenario

    def _interpret(self, http_status: int, body: dict[str, Any], check_id: str, pan_hash: str, endpoint: str) -> SurePassResult:
        return _interpret_and_log(
            self._sink,
            check_id=check_id,
            pan_hash=pan_hash,
            endpoint=endpoint,
            http_status=http_status,
            body=body,
            latency_ms=0,
        )
