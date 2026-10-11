"""The SurePass worlds the golden datasets name: a JSON-able spec -> a ``FakeSurePassClient``.

A golden's ``surepass`` field is ``{"scenario": <name>, "kwargs": {...}}`` (plain JSON, so it survives a LangSmith
dataset round trip). ``build_surepass`` resolves the name through ``SCENARIOS``: the canned factories of
``eligibility_agent.testing.scenarios`` and ``tests.helpers.quick_and_full_client`` (single sources, never forked), plus
four worlds that only the evals need, built from the same ``testing.envelopes`` builders (every value synthetic; the
probe-literal guard sits in those builders).

The fake runs envelopes through the same ``interpret_response`` path as the live client and logs every call to the
``sink`` (the run's MemoryRepo), so ``surepass_calls`` rows exist exactly as in production. An unconfigured endpoint raises
``AssertionError("unexpected SurePass call")``: the ``none`` world is how cache hits and refusals PROVE that no call was
made.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final, TypeAlias

from eligibility_agent.surepass.client import CallLogSink, FakeSurePassClient
from eligibility_agent.testing import envelopes as env
from eligibility_agent.testing import scenarios
from tests.helpers import PAN_NTC, PAN_OK, quick_and_full_client

__all__ = ["SCENARIOS", "SurePassSpec", "UnknownScenario", "build_surepass", "scenario"]

SurePassSpec: TypeAlias = dict[str, Any]
WorldFactory: TypeAlias = Callable[..., FakeSurePassClient]


class UnknownScenario(KeyError):
    """A golden names a world this module does not know (a typo would otherwise run the wrong world silently)."""


def scenario(name: str, **kwargs: Any) -> SurePassSpec:
    """The spec a golden carries for world ``name`` (validated eagerly, so a typo fails at dataset build time)."""
    if name not in SCENARIOS:
        raise UnknownScenario(f"unknown SurePass world {name!r}; known: {sorted(SCENARIOS)}")
    return {"scenario": name, "kwargs": dict(kwargs)}


def build_surepass(spec: Mapping[str, Any], sink: CallLogSink) -> FakeSurePassClient:
    name = str(spec["scenario"])
    factory = SCENARIOS.get(name)
    if factory is None:
        raise UnknownScenario(f"unknown SurePass world {name!r}; known: {sorted(SCENARIOS)}")
    kwargs = dict(spec.get("kwargs") or {})
    return factory(sink=sink, **kwargs)


# --------------------------------------------------------------------------- worlds only the evals need


def _none(*, sink: CallLogSink) -> FakeSurePassClient:
    """No endpoint configured: ANY call raises. Cache hits and refusals run here to prove they call nothing."""
    return FakeSurePassClient(sink=sink)


def _quick_world(
    *, sink: CallLogSink, pan: str = PAN_OK, score: str = "745", zip_code: str = "560001", aadhaar_linked: bool | None = True
) -> FakeSurePassClient:
    """PAN + Score, like ``happy_quick_scenario``, with the Aadhaar link state of the PAN record as a knob."""
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan, zip_code=zip_code, aadhaar_linked=aadhaar_linked)},
        score={pan: env.experian_score_envelope(credit_score=score)},
        sink=sink,
    )


def _full_stale_account(*, sink: CallLogSink, pan: str = PAN_OK) -> FakeSurePassClient:
    """ReportDate 2026-10-01, PAN zip 560001, score 745. Two open accounts: a reported EMI of 3,500 (Date_Reported
    2026-09-20) and a STALE one (EMI 9,999, Date_Reported 2026-06-01: 122 days before the report, over the 90-day
    limit). The stale account is excluded from every sum and rides as a confirm item: confidence drops to medium."""
    fresh = env.cais_account(
        scheduled_emi="3500", current_balance="150000", date_reported="20260920", holder_zip="560001",
        account_number="SYNTHACC0001",
    )
    stale = env.cais_account(
        scheduled_emi="9999", current_balance="70000", date_reported="20260601", holder_zip="560001",
        account_number="SYNTHACC0002",
    )
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan, zip_code="560001")},
        report={pan: env.experian_report_envelope(accounts=[fresh, stale], credit_score="745", report_date=scenarios.REPORT_DATE)},
        sink=sink,
    )


def _ntc_quick_and_full(*, sink: CallLogSink, pan: str = PAN_NTC) -> FakeSurePassClient:
    """A new-to-credit PAN: an empty-string Score (NTC) and an account-less NTC Report, PAN + Score + Report wired."""
    return FakeSurePassClient(
        pan={pan: env.pan_comprehensive_envelope(pan=pan)},
        score={pan: env.experian_score_envelope(credit_score="")},
        report={pan: env.experian_report_envelope(accounts=[], credit_score="", report_date=scenarios.REPORT_DATE)},
        sink=sink,
    )


SCENARIOS: Final[Mapping[str, WorldFactory]] = {
    "none": _none,
    "happy_quick": scenarios.happy_quick_scenario,
    "quick_world": _quick_world,
    "ntc": scenarios.ntc_scenario,
    "score_failure": scenarios.score_failure_scenario,
    "pan_invalid": scenarios.pan_invalid_scenario,
    "full_happy_reported_emi": scenarios.full_happy_reported_emi_scenario,
    "full_emi_four_branch": scenarios.full_emi_four_branch_scenario,
    "full_pincode_union": scenarios.full_pincode_union_scenario,
    "full_stale_account": _full_stale_account,
    "no_pincode": scenarios.no_pincode_scenario,
    "quick_and_full": quick_and_full_client,
    "ntc_quick_and_full": _ntc_quick_and_full,
}
