"""Task 9 (part 1): output/lc_view.py, the two-tier redaction boundary (PRD v1.1 sections 6 / 10 / 12 no_leak).

One module owns every leak regex. Two tiers, applied by whole-line drop (never partial masking):

* LLM-prose tier  = (BUREAU_TERM_RE incl. ``bureaus?``, STANDALONE_SCORE_RE, ACCOUNT_ID_RE) on rationale / conditions /
  items_to_confirm / coapplicant_reason, the only free text a model (or derivation) can author;
* global tier     = (BUREAU_NAME_RE names only, STANDALONE_SCORE_RE, ACCOUNT_ID_RE) on every fixed string and every
  admin-authored leaf, so the PRD-mandated copy "bureau report on <date>" survives.

The patterns are IMPORTED from the module under test everywhere (no local copies, the C11 import rule), so the runtime
scrubber and every later evaluator can never drift.
"""

from __future__ import annotations

import ast
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from eligibility_agent.graph.state import Code
from eligibility_agent.output import lc_view as lc_view_module
from eligibility_agent.output.lc_view import (
    AADHAAR_WARNING,
    ACCOUNT_ID_RE,
    BUREAU_NAME_RE,
    BUREAU_TERM_RE,
    GLOBAL_LEAK_PATTERNS,
    LABEL_FULL_FMT,
    LABEL_QUICK,
    LABEL_QUICK_NO_EMI,
    LLM_LEAK_PATTERNS,
    MSG_BUREAU_SERVICE_FAILURE,
    MSG_INTERNAL,
    MSG_NO_PINCODE,
    MSG_OPENAI_FAILURE,
    MSG_PAN_NOT_VERIFIED,
    MSG_PAN_SERVICE_FAILURE,
    MSG_PENDING_VERIFICATION,
    RATIONALE_RANKING_UNAVAILABLE,
    REFUSAL_MESSAGES,
    STANDALONE_SCORE_RE,
    WARN_NTC_RECENT,
    line_leaks,
    scrub_lines,
)

# --------------------------------------------------------------------------- the patterns themselves (pinned verbatim)


def test_bureau_term_re_is_the_binding_v1_pattern() -> None:
    assert BUREAU_TERM_RE.pattern == r"\b(?:cibil|experian|equifax|crif|bureaus?)\b"
    assert BUREAU_TERM_RE.flags & re.IGNORECASE


def test_bureau_name_re_is_names_only() -> None:
    assert BUREAU_NAME_RE.pattern == r"\b(?:cibil|experian|equifax|crif)\b"
    assert BUREAU_NAME_RE.flags & re.IGNORECASE


def test_standalone_score_re_is_the_pinned_task1_pattern() -> None:
    assert STANDALONE_SCORE_RE.pattern == r"(?<!\d)(?<!\d[.,])[3-8]\d{2}(?!\d)(?![.,]\d)"


def test_the_two_tuples_are_exactly_the_ruled_pairs_in_order() -> None:
    assert LLM_LEAK_PATTERNS == (BUREAU_TERM_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE)
    assert GLOBAL_LEAK_PATTERNS == (BUREAU_NAME_RE, STANDALONE_SCORE_RE, ACCOUNT_ID_RE)


def test_line_leaks_defaults_to_the_llm_tier() -> None:
    assert line_leaks("per the bureau") is True
    assert line_leaks("per the bureau", GLOBAL_LEAK_PATTERNS) is False
    assert line_leaks("per the bureau", LLM_LEAK_PATTERNS) is True


# --------------------------------------------------------------------------- v1.0 C11 vector battery (verbatim rows)

C11_DROP = [
    "736",
    "Score is 736",
    "(736)",
    "736.",
    "736-750",
    "Rs.736 looks strong",
    "below 650",
    "band below 650",
    "₹736",
    "Experian says it is fine",
    "CIBIL says it is fine",
    "Equifax says it is fine",
    "CRIF says it is fine",
    "per the bureaus",
    "bureaus differ",
    "the bureau flagged it",
    "tenure of 360 months",
]

C11_KEEP = [
    "EMI of 3,500 fits",
    "3,500",
    "income 45,000 per month",
    "45,000",
    "45,500",
    "loan up to 3,50,000",
    "3,50,000",
    "12,34,567",
    "tenure 36 months at 14 percent",
    "36 months",
    "14 percent",
    "736.50 rupees",
    "736.50",
    "paid 1736",
    "1736",
    "500000",
    "0.736",
    "900",
    "299",
    "2026",
    "crifled",
]


@pytest.mark.parametrize("line", C11_DROP)
def test_c11_drop_vectors_leak_on_the_llm_tier(line: str) -> None:
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True


@pytest.mark.parametrize("line", C11_KEEP)
def test_c11_keep_vectors_pass_the_llm_tier(line: str) -> None:
    assert line_leaks(line, LLM_LEAK_PATTERNS) is False


GENERIC_BUREAU_ONLY = ("per the bureaus", "bureaus differ", "the bureau flagged it")


@pytest.mark.parametrize("line", [v for v in C11_DROP if v not in GENERIC_BUREAU_ONLY])
def test_c11_drop_vectors_that_name_a_bureau_or_carry_a_score_leak_on_the_global_tier_too(line: str) -> None:
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is True


@pytest.mark.parametrize("line", GENERIC_BUREAU_ONLY)
def test_the_generic_word_bureau_is_llm_tier_only(line: str) -> None:
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False


@pytest.mark.parametrize("line", C11_KEEP)
def test_c11_keep_vectors_pass_the_global_tier(line: str) -> None:
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False


@pytest.mark.parametrize("text", ["300", "850", "899", "score 650.", "a 300 b"])
def test_standalone_score_range_hits(text: str) -> None:
    assert STANDALONE_SCORE_RE.search(text) is not None


@pytest.mark.parametrize("text", ["299", "900", "3000", "8000", "x3000", "1,300", "300.5", "300,5"])
def test_standalone_score_range_misses(text: str) -> None:
    assert STANDALONE_SCORE_RE.search(text) is None


# --------------------------------------------------------------------------- ACCOUNT_ID_RE vectors (no account identifiers)

ACCOUNT_DROP = [
    "account number 123456789012",
    "Account No. 55",
    "a/c no: SYNTH0001",
    "A/C No SYNTH0001",
    "acct # 99",
    "account #123",
    "XXXX1234 is overdue",
    "xxxx 1234 is overdue",
    "****1234 is overdue",
    "ending ****1234",
    "xxxx-xxxx-1234",
    "loan 123456789 outstanding",
    "reference 12345678901234567890",  # 20 digits: longer than the dossier's 18 cap must not escape
]

ACCOUNT_KEEP = [
    "An account last updated 2026-03-12",
    "two accounts are open",
    "loan up to 12,34,567",
    "account normally serviced",  # 'no' is a word prefix, not the abbreviation
    "the account is in good standing",
    "an account last updated 2026-06-01",
    "emi 21,000 per month",
    "2026-10-07",
]


@pytest.mark.parametrize("line", ACCOUNT_DROP)
def test_account_id_drop_vectors_on_both_tiers(line: str) -> None:
    assert ACCOUNT_ID_RE.search(line) is not None
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is True


@pytest.mark.parametrize("line", ACCOUNT_KEEP)
def test_account_id_keep_vectors_on_both_tiers(line: str) -> None:
    assert ACCOUNT_ID_RE.search(line) is None
    assert line_leaks(line, LLM_LEAK_PATTERNS) is False
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False


# --------------------------------------------------------------------------- label-tier vectors (the two-tier reason)


def test_label_vector_bureau_report_keeps_on_global_and_drops_on_llm() -> None:
    line = "bureau report on 2026-10-07, income as declared"
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True


def test_label_vector_ntc_warning_keeps_on_global_and_drops_on_llm() -> None:
    line = "No bureau file was found on 2026-10-10; a full report will most likely return nothing. Proceed?"
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True


def test_global_tier_still_bans_every_bureau_name_in_any_case() -> None:
    for name in ("CIBIL", "cibil", "Experian", "EQUIFAX", "Crif", "Experian Finance Ltd"):
        assert line_leaks(f"bureau report from {name} on 2026-10-07", GLOBAL_LEAK_PATTERNS) is True


# --------------------------------------------------------------------------- scrub_lines: whole-line drop, never masking


def test_scrub_lines_drops_whole_offending_lines_and_keeps_the_rest_verbatim() -> None:
    lines = ["Income looks steady", "Score is 736", "EMI of 3,500 fits", "per the bureaus", "an account last updated 2026-03-12"]
    assert scrub_lines(lines) == ["Income looks steady", "EMI of 3,500 fits", "an account last updated 2026-03-12"]


def test_scrub_lines_never_partially_edits_a_line() -> None:
    kept = scrub_lines(["Strong profile (736) overall", "Strong profile overall"])
    assert kept == ["Strong profile overall"]
    assert all("736" not in k and "***" not in k and "[" not in k for k in kept)


def test_scrub_lines_uses_the_llm_tier() -> None:
    assert scrub_lines(["bureau report on 2026-10-07"]) == []


def test_scrub_lines_is_pure_and_accepts_any_sequence() -> None:
    original = ["a 736 b", "ok"]
    assert scrub_lines(tuple(original)) == ["ok"]
    assert original == ["a 736 b", "ok"]
    assert scrub_lines([]) == []


# --------------------------------------------------------------------------- fixed strings: verbatim values


def test_fixed_strings_are_the_ruled_verbatim_values() -> None:
    assert LABEL_QUICK == "Indicative, based on what the learner told us"
    assert LABEL_QUICK_NO_EMI == "Indicative, based on what the learner told us, assuming no other EMIs"
    assert LABEL_FULL_FMT == "bureau report on {report_date}, income as declared"
    assert AADHAAR_WARNING == "PAN not linked to Aadhaar; lenders will ask for this"
    assert MSG_PAN_SERVICE_FAILURE == "Couldn't verify PAN, retry"
    assert (
        MSG_PAN_NOT_VERIFIED
        == "PAN not verified. Correct the PAN with the learner and run the check again; no quota was used."
    )
    assert MSG_BUREAU_SERVICE_FAILURE == "Credit check couldn't complete, retry"
    assert MSG_OPENAI_FAILURE == "Couldn't generate a recommendation, retry"
    assert MSG_INTERNAL == "The check could not be completed automatically. Please raise it for manual review."
    assert (
        MSG_NO_PINCODE
        == "No pincode on the PAN record; lender matching will be available after the full check"
    )
    assert (
        MSG_PENDING_VERIFICATION
        == "Pending verification: the credit check didn't complete. This learner's PAN is locked while the "
        "result is reconciled against the provider; it will be available again once our team resolves it."
    )
    assert RATIONALE_RANKING_UNAVAILABLE == "Automatic ranking unavailable; this lender serves the learner's area"
    assert (
        WARN_NTC_RECENT
        == "No bureau file was found on {date}; a full report will most likely return nothing. Proceed?"
    )


def test_pending_verification_copy_promises_no_time_based_retry() -> None:
    # K2: lock expiry alone never re-admits the PAN; only the reconciliation review does.
    assert "minute" not in MSG_PENDING_VERIFICATION
    assert "try again" not in MSG_PENDING_VERIFICATION
    assert "resolves it" in MSG_PENDING_VERIFICATION


REFUSAL_CODES: tuple[Code, ...] = (
    "CONSENT_MISSING",
    "QUOTA_EXHAUSTED",
    "PAN_DAILY_CAP",
    "STAGE_TOO_EARLY",
    "REPORT_WINDOW_OPEN",
    "IN_FLIGHT",
    "PAN_LOCKED",
    "WARN_NTC_RECENT",
    "PAN_CHANGE_NEEDS_MANAGER",
    "LIVE_CALLS_DISABLED",
)


def test_refusal_messages_cover_exactly_the_ten_refusal_codes() -> None:
    assert set(REFUSAL_MESSAGES) == set(REFUSAL_CODES)
    assert len(REFUSAL_MESSAGES) == 10


def test_refusal_messages_are_the_ruled_verbatim_strings() -> None:
    assert REFUSAL_MESSAGES == {
        "CONSENT_MISSING": "Consent was not recorded. Take the learner's verbal consent, tick the consent box and run the check again.",
        "QUOTA_EXHAUSTED": "You have used all your quick checks for today. The limit resets at midnight; saved results stay available.",
        "PAN_DAILY_CAP": "This learner's checks have reached today's safety limit. The record is locked until tomorrow and has been sent for review.",
        "STAGE_TOO_EARLY": "A full check unlocks once this opportunity reaches the Offer stage. Until then, use the quick check.",
        "REPORT_WINDOW_OPEN": "This learner's full report is still current. Refresh unlocks once it is 30 days old.",
        "IN_FLIGHT": "A check for this learner is already running. Please wait for it to finish.",
        "PAN_LOCKED": "This learner's record is locked for a few minutes while an earlier check is reconciled. Please try again shortly.",
        "WARN_NTC_RECENT": WARN_NTC_RECENT,
        "PAN_CHANGE_NEEDS_MANAGER": "A report already exists for this opportunity, so changing the PAN needs a manager. This has been flagged for review.",
        "LIVE_CALLS_DISABLED": "Eligibility checks are temporarily switched off on the server. Please try again later or contact the platform team.",
    }


# --------------------------------------------------------------------------- fixed strings: global-tier self-test

FIXED_STRINGS: dict[str, str] = {
    "LABEL_QUICK": LABEL_QUICK,
    "LABEL_QUICK_NO_EMI": LABEL_QUICK_NO_EMI,
    "LABEL_FULL_FMT": LABEL_FULL_FMT,
    "AADHAAR_WARNING": AADHAAR_WARNING,
    "MSG_PAN_SERVICE_FAILURE": MSG_PAN_SERVICE_FAILURE,
    "MSG_PAN_NOT_VERIFIED": MSG_PAN_NOT_VERIFIED,
    "MSG_BUREAU_SERVICE_FAILURE": MSG_BUREAU_SERVICE_FAILURE,
    "MSG_PENDING_VERIFICATION": MSG_PENDING_VERIFICATION,
    "MSG_OPENAI_FAILURE": MSG_OPENAI_FAILURE,
    "MSG_INTERNAL": MSG_INTERNAL,
    "MSG_NO_PINCODE": MSG_NO_PINCODE,
    "RATIONALE_RANKING_UNAVAILABLE": RATIONALE_RANKING_UNAVAILABLE,
    "WARN_NTC_RECENT": WARN_NTC_RECENT,
    **{f"REFUSAL[{code}]": text for code, text in REFUSAL_MESSAGES.items()},
}


@pytest.mark.parametrize("name", sorted(FIXED_STRINGS))
def test_every_fixed_string_passes_the_global_tier(name: str) -> None:
    assert line_leaks(FIXED_STRINGS[name], GLOBAL_LEAK_PATTERNS) is False, name


def test_every_fixed_string_in_the_module_is_covered_by_the_self_test() -> None:
    constants = {
        name
        for name, value in vars(lc_view_module).items()
        if isinstance(value, str) and name.isupper() and name not in {"__all__"}
    }
    assert constants <= set(FIXED_STRINGS), constants - set(FIXED_STRINGS)


def _days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(count)]


# 15 years of ISO dates: a 3-digit run can never hide inside YYYY-MM-DD, so the formatted fixed strings stay global-clean.
DATE_SWEEP = _days(date(2024, 1, 1), 366 * 15)


def test_formatted_ntc_warning_passes_the_global_tier_for_every_date() -> None:
    for d in DATE_SWEEP:
        formatted = REFUSAL_MESSAGES["WARN_NTC_RECENT"].format(date=d.isoformat())
        assert line_leaks(formatted, GLOBAL_LEAK_PATTERNS) is False, d


def test_formatted_full_label_passes_the_global_tier_for_every_date() -> None:
    for d in DATE_SWEEP:
        formatted = LABEL_FULL_FMT.format(report_date=d.isoformat())
        assert line_leaks(formatted, GLOBAL_LEAK_PATTERNS) is False, d


def test_the_generic_bureau_word_is_exactly_why_the_two_labels_need_the_names_only_tier() -> None:
    assert line_leaks(LABEL_FULL_FMT.format(report_date="2026-10-07"), LLM_LEAK_PATTERNS) is True
    assert line_leaks(REFUSAL_MESSAGES["WARN_NTC_RECENT"].format(date="2026-10-07"), LLM_LEAK_PATTERNS) is True


def test_every_other_fixed_string_also_passes_the_stricter_llm_tier() -> None:
    allowed_generic_bureau = {"LABEL_FULL_FMT", "WARN_NTC_RECENT", "REFUSAL[WARN_NTC_RECENT]"}
    for name, text in FIXED_STRINGS.items():
        if name in allowed_generic_bureau:
            continue
        assert line_leaks(text, LLM_LEAK_PATTERNS) is False, name


# --------------------------------------------------------------------------- module shape


def test_the_module_defines_exactly_four_view_builders() -> None:
    builders = {name for name in dir(lc_view_module) if name.startswith("build_")}
    assert builders == {"build_lc_view", "build_refusal_lc_view", "build_failure_lc_view", "build_error_lc_view"}


def test_module_imports_are_stdlib_state_safety_and_range_only() -> None:
    source = Path(lc_view_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module if node.module.startswith("eligibility_agent") else node.module.split(".")[0])
    stdlib = {"__future__", "collections", "dataclasses", "datetime", "math", "re", "typing"}
    sanctioned = {
        "eligibility_agent.graph.state",
        "eligibility_agent.lenders.safety",
        "eligibility_agent.derive.range",
    }
    assert imported <= stdlib | sanctioned, imported - stdlib - sanctioned
    assert not re.search(r"\.now\(|utcnow|time\.time|\.today\(|random|uuid|logging|print\(", source)


def test_headline_and_rounding_are_imported_never_redefined_here() -> None:
    source = Path(lc_view_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "compute_headline" not in defined
    assert "final_confidence" not in defined
    assert "round_to_nearest_1000" not in defined
    # K6: no re-rounding anywhere in this module
    assert not re.search(r"\bround\w*\(|\bfloor\(|\bceil\(|\bquantize\(|1000", source)
