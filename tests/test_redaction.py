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
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pytest

from eligibility_agent.derive.emi import _STALE_CONFIRM_PREFIX
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
    MAX_PROSE_LINE_CHARS,
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
    assert BUREAU_TERM_RE.pattern == r"\b(?:cibil|experian|equifax|crif|transunion|highmark|bureaus?)\b"
    assert BUREAU_TERM_RE.flags & re.IGNORECASE


def test_bureau_name_re_is_names_only() -> None:
    assert BUREAU_NAME_RE.pattern == r"\b(?:cibil|experian|equifax|crif|transunion|highmark)\b"
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
    # fix round 1 (1): plural / glued label forms the first tightening let through
    "account numbers: 12345678",
    "account numbers 1234, 5678",
    "account no123",
    "a/c no123456",
    # fix round 1 (2): letter / underscore adjacent digit runs (a \b-anchored run lets these escape)
    "LN00123456789",
    "acct_123456789",
    "ID123456789",
    "123456789abc",
    # same escape class on the masked-id branch
    "XXXX1234abc",
    "xxxx1234_ref",
    # fix round 2 (1): masks GLUED to a preceding word (the round-1 lookbehind dropped these)
    "A/C****1234",
    "ending****1234",
    "acct****1234",
    "Loan****1234",
    "card****1234",
    "Max****1234",  # glued to a word that itself ends in x
    "ID**xx1234",
    # fix round 2 (2): a label/value gap of any width (the {0,8} bound let a padded gap escape)
    "Account number" + " " * 9 + "123456",
    "Account number" + " " * 12 + "123456",
    "Account number" + " " * 40 + "123456",
    "account no:" + " " * 30 + "SYNTH0001",
    # realistic mask widths up to the 32-character bound (a longer x-mask is outside the pattern's bound)
    "XXXXXXXX1234",
    "xxxxxxxxxxxx 1234",
    "X" * 32 + "1234",
    "\u2022" + "x" * 16 + "-1234",  # a bullet is not a word character
]

# every glued asterisk shape: a word-ish prefix directly against a 2+ asterisk mask and 3+ digits
GLUED_PREFIXES = ["A/C", "a/c", "acct", "Acct", "account", "ending", "Loan", "loan", "card", "Card", "ref", "ID", "no", "num", "Max", "tax"]
GLUED_MASKS = ["**", "***", "****", "********"]
GLUED_SHAPES = [f"{prefix}{mask}{digits}" for prefix in GLUED_PREFIXES for mask in GLUED_MASKS for digits in ("123", "1234")]

ACCOUNT_KEEP = [
    "An account last updated 2026-03-12",
    "two accounts are open",
    "loan up to 12,34,567",
    "account normally serviced",  # 'no' is a word prefix, not the abbreviation
    "account nothing to add",
    "account notes attached",
    "IDxxxx1234",  # a word glued to x's is not a mask
    "Max1234",
    "Maxx1234",
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


def test_known_gap_an_x_only_mask_of_33_plus_characters_after_a_non_word_character_is_outside_the_pattern() -> None:
    # Recorded, not endorsed: the masked-id run is bounded to 32 characters so the scan stays linear. Real masks are
    # 4-16 characters; an asterisk mask of ANY length is still caught (see the long-asterisk test).
    line = " " + "x" * 40 + "1234"
    assert ACCOUNT_ID_RE.search(line) is None
    assert line_leaks(line, LLM_LEAK_PATTERNS) is False
    assert line_leaks(" " + "x" * 32 + "1234", LLM_LEAK_PATTERNS) is True  # the bound itself is caught


@pytest.mark.parametrize("line", GLUED_SHAPES)
def test_glued_asterisk_masks_leak_on_both_tiers(line: str) -> None:
    assert ACCOUNT_ID_RE.search(line) is not None
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is True


@pytest.mark.parametrize("line", ["*" * 70 + "1234", "x" + "*" * 70 + "1234", "ending" + "*" * 70 + "1234"])
def test_a_very_long_asterisk_mask_still_leaks(line: str) -> None:
    # the mask run is bounded in the pattern (linear scan); a later start inside the run catches an over-long one
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is True


# --------------------------------------------------------------------------- bureau names (fix round 1, ruled)

BUREAU_NAME_DROP = ["CIBIL", "Experian", "Equifax", "CRIF", "TransUnion", "TRANSUNION", "transunion", "Highmark", "HIGHMARK"]


@pytest.mark.parametrize("name", BUREAU_NAME_DROP)
def test_every_bureau_name_leaks_on_both_tiers(name: str) -> None:
    for line in (f"{name} says it is fine", f"per {name}.", name):
        assert line_leaks(line, LLM_LEAK_PATTERNS) is True
        assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is True


@pytest.mark.parametrize("line", ["highmarked", "transunions", "a highmarkable idea", "crifled"])
def test_bureau_names_are_whole_words_only(line: str) -> None:
    assert line_leaks(line, LLM_LEAK_PATTERNS) is False
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False


# --------------------------------------------------------------------------- normalization (fix round 1, ruled)
# Every scan runs on an NFKC-normalised copy with invisible characters stripped, so fullwidth, soft-hyphen and
# zero-width evasions cannot dodge the (unchanged) patterns. Decisions fire on the normalised copy; a line that is KEPT
# is returned as the ORIGINAL text (nothing here ever rewrites a span: whole-line drop only).

NORMALIZATION_DROP = [
    "\uff23\uff29\uff22\uff29\uff2c says it is fine",  # fullwidth CIBIL
    "CI\u00adBIL says it is fine",  # soft hyphen inside a bureau name
    "Ex\u200bper\u200dian",  # zero-width space / joiner
    "\uff25\uff58\uff50\uff45\uff52\uff49\uff41\uff4e",  # fullwidth Experian
    "\uff34\uff52\uff41\uff4e\uff53\uff35\uff4e\uff49\uff4f\uff4e",  # fullwidth TransUnion
    "High\u200bmark",
    "Score is 7\u200b3\u200d6",  # zero-width characters inside a score
    "Score is 7\ufeff36",  # BOM inside a score
    "Score is 7\u00ad36",
    "Score is \uff17\uff13\uff16",  # fullwidth digits
    "account number \uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19",
    "a/c no: SYN\u200bTH0001",
    "1234\u200b56789",  # a 9-digit run split by a zero-width space
    # fix round 2 (3): every Unicode Cf character and the default-ignorable non-Cf extras
    "Score is 7\u206236",  # invisible times
    "CI\u202cBIL",  # pop directional formatting
    "Score is 7\u200e36",  # left-to-right mark
    "Score is 7\u200f36",  # right-to-left mark
    "Score is 7\u206636",  # left-to-right isolate
    "Score is 7\u206936",  # pop directional isolate
    "Score is 7\u20603\u20636",  # word joiner / invisible separator
    "Score is 7\u034f36",  # combining grapheme joiner
    "Score is 7\u115f36",  # hangul choseong filler
    "Score is 7\u316436",  # hangul filler
    "CI\ufe0fBIL",  # variation selector-16 inside a word
    "Ex\u180bperian",  # mongolian free variation selector
    "account number 12345\u200e6789",
    # fix round 3 (1): UNASSIGNED code points Unicode reserves as default-ignorable (renderers hide them)
    "Score is 7\u206536",
    "CI\u2065BIL",
    "Score is 7\ufff036",
    "Score is 7\U000e000036",  # U+E0000
    "CI\U000e0080BIL",
    "Score is 7\U000e01f036",
]

NORMALIZATION_KEEP = [
    "EMI of \uff13\uff0c\uff15\uff10\uff10 fits",  # fullwidth '3,500' normalises to a grouped amount
    "\uff21 clean line",
    "caf\u00e9 income looks steady",
    "income\u200b looks steady",
]


@pytest.mark.parametrize("line", NORMALIZATION_DROP)
def test_normalization_closes_unicode_evasions_on_both_tiers(line: str) -> None:
    assert line_leaks(line, LLM_LEAK_PATTERNS) is True
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is True


@pytest.mark.parametrize("line", NORMALIZATION_KEEP)
def test_normalization_does_not_over_scrub(line: str) -> None:
    assert line_leaks(line, LLM_LEAK_PATTERNS) is False
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False


def test_the_raw_patterns_are_unchanged_by_normalization() -> None:
    # the evasions only fail the raw pattern; line_leaks is what closes them
    assert BUREAU_NAME_RE.search("CI\u00adBIL") is None
    assert STANDALONE_SCORE_RE.search("7\u200b36") is None
    assert line_leaks("CI\u00adBIL") is True and line_leaks("7\u200b36") is True


def test_kept_lines_are_returned_as_the_original_text() -> None:
    lines = ["Fine\u200b text", "\uff23\uff29\uff22\uff29\uff2c shows it", "\uff21 clean", "Score is 7\u200b36"]
    assert scrub_lines(lines) == ["Fine\u200b text", "\uff21 clean"]


def test_every_unicode_format_character_is_invisible_to_the_scan() -> None:
    # Exhaustive over the whole code space: any Cf character dropped inside a score, a bureau name or a long digit run
    # must not hide it. (Cf = Unicode "format" category: zero-width, bidi controls, invisible operators, tags, ...)
    format_chars = [chr(cp) for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)) == "Cf"]
    assert len(format_chars) > 100  # the category really was enumerated
    for ch in format_chars:
        assert line_leaks(f"Score is 7{ch}36") is True, hex(ord(ch))
        assert line_leaks(f"CI{ch}BIL") is True, hex(ord(ch))
        assert line_leaks(f"1234{ch}56789") is True, hex(ord(ch))


# Code points Unicode RESERVES as default-ignorable while leaving them unassigned (category Cn, so a Cf check misses them).
RESERVED_DEFAULT_IGNORABLES: list[int] = [
    0x2065,
    *range(0xFFF0, 0xFFF9),
    0xE0000,
    *range(0xE0002, 0xE0020),
    *range(0xE0080, 0xE0100),
    *range(0xE01F0, 0xE1000),
]


def test_every_reserved_default_ignorable_code_point_is_invisible_to_the_scan() -> None:
    assert len(RESERVED_DEFAULT_IGNORABLES) > 3000  # the ranges really were enumerated
    for cp in RESERVED_DEFAULT_IGNORABLES:
        ch = chr(cp)
        assert line_leaks(f"Score is 7{ch}36") is True, hex(cp)
        assert line_leaks(f"CI{ch}BIL") is True, hex(cp)
        assert line_leaks(f"1234{ch}56789") is True, hex(cp)


# --------------------------------------------------------------------------- bounded backtracking (fix round 1)


# Regex-shape pins. Every input here is longer than MAX_PROSE_LINE_CHARS, so the LLM-prose tier would drop it on length
# alone without running a single pattern (that is the point of the cap, and it would make these pins vacuous). They are
# therefore run through the UNCAPPED global tier, which carries the same ACCOUNT_ID_RE / STANDALONE_SCORE_RE and the
# same normaliser, so each one really exercises the regexes.
REGEX_SHAPE_INPUTS: dict[str, str] = {
    "padded-gap": "account number" + " " * 40_000,
    "padded-gap-then-text": "account number" + " " * 40_000 + "x",
    "dash-gap": "account no" + "-" * 40_000,
    "colon-gap": "account no" + ": " * 20_000,
    "asterisk-run": "*" * 50_000,
    "x-run": "x" * 100_000,
    "alternating-star-x": "*x" * 20_000,  # every '*' sits after a word character
    "alternating-x-star": "x*" * 20_000,
    "x-double-star": "x**" * 15_000,
    "letter-double-star": "A**" * 15_000,
    "repeated-label": "a/c " * 20_000,
    "digit-run": "1" * 100_000,
    "zero-width-run": "\u200b" * 50_000 + "7" + "\u200b" * 50_000,  # the normaliser itself is linear
}

# NFKC reorders combining marks quadratically (80k chars took ~1.6s, x4 per doubling). THIS pin is about the cap: the
# LLM-prose tier drops an over-long line BEFORE normalising, so it must be instant on that tier.
COMBINING_MARKS_LINE: str = "a" + "\u0301\u0316" * 40_000


@pytest.mark.parametrize("name", sorted(REGEX_SHAPE_INPUTS))
def test_the_account_pattern_does_not_backtrack_quadratically(name: str) -> None:
    text = REGEX_SHAPE_INPUTS[name]
    assert len(text) > MAX_PROSE_LINE_CHARS  # why this is the global tier: the LLM tier would never reach the regexes
    started = time.perf_counter()
    line_leaks(text, GLOBAL_LEAK_PATTERNS)
    # linear scans finish in milliseconds; each quadratic shape this pins takes tens of seconds at these sizes
    assert time.perf_counter() - started < 1.0


def test_an_80k_combining_mark_line_is_dropped_by_the_length_cap_not_scanned() -> None:
    assert len(COMBINING_MARKS_LINE) > MAX_PROSE_LINE_CHARS  # the cap, not the normaliser, is what handles this line
    started = time.perf_counter()
    assert line_leaks(COMBINING_MARKS_LINE, LLM_LEAK_PATTERNS) is True
    assert time.perf_counter() - started < 1.0


def test_the_regex_shape_inputs_really_are_dropped_by_the_cap_on_the_llm_tier() -> None:
    # records why the pins above use the global tier: on the LLM tier every one of them is a length-cap drop
    for name, text in REGEX_SHAPE_INPUTS.items():
        assert line_leaks(text, LLM_LEAK_PATTERNS) is True, name


# --------------------------------------------------------------------------- prose-line length cap (fix round 3, ruled)


def test_the_prose_line_cap_is_four_thousand_characters() -> None:
    assert MAX_PROSE_LINE_CHARS == 4000


def test_an_over_long_llm_prose_line_is_dropped_before_any_scan() -> None:
    clean_at_the_cap = "a " * (MAX_PROSE_LINE_CHARS // 2)
    assert len(clean_at_the_cap) == MAX_PROSE_LINE_CHARS
    assert line_leaks(clean_at_the_cap, LLM_LEAK_PATTERNS) is False  # the cap is inclusive
    assert line_leaks(clean_at_the_cap + "a", LLM_LEAK_PATTERNS) is True  # one over: dropped, fail closed
    assert line_leaks("a" * 200_000) is True  # default tier is the LLM tier


def test_the_cap_drops_the_line_through_scrub_lines_and_never_edits_it() -> None:
    long_line = "Income looks steady. " * 300
    assert len(long_line) > MAX_PROSE_LINE_CHARS
    assert scrub_lines(["short and clean", long_line, "also clean"]) == ["short and clean", "also clean"]


def test_the_cap_is_llm_tier_only() -> None:
    # the global tier scans admin strings and fixed copy, which are never model prose
    long_admin = "Alpha Education Finance " * 300
    assert len(long_admin) > MAX_PROSE_LINE_CHARS
    assert line_leaks(long_admin, GLOBAL_LEAK_PATTERNS) is False


def test_a_normal_length_line_with_a_few_combining_marks_is_unaffected() -> None:
    line = "Cafe\u0301 income and re\u0301sume\u0301 look steady, ni\u0303o fine"
    assert line_leaks(line, LLM_LEAK_PATTERNS) is False
    assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False
    assert scrub_lines([line]) == [line]  # kept as the ORIGINAL text


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
    # the PAN_LOCKED refusal rides the same K2 gate (reconciliation outlives the lock), so it promises nothing either
    assert "minute" not in REFUSAL_MESSAGES["PAN_LOCKED"]
    assert "try again" not in REFUSAL_MESSAGES["PAN_LOCKED"]


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
        "PAN_LOCKED": "This learner's PAN is temporarily locked while a previous check on it is verified. It will be available again once that check is resolved.",
        "WARN_NTC_RECENT": WARN_NTC_RECENT,
        "PAN_CHANGE_NEEDS_MANAGER": "Changing the PAN on this opportunity needs a manager: a report already exists or the one allowed correction has been used. This has been flagged for review.",
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


# 40 years of ISO dates: no run of 9+ digits or standalone 3-digit number can occur inside YYYY-MM-DD, so the formatted
# fixed strings stay global-clean (this is what the unanchored bare-digit rule must never break).
DATE_SWEEP = _days(date(2010, 1, 1), 366 * 40)


def test_formatted_ntc_warning_passes_the_global_tier_for_every_date() -> None:
    for d in DATE_SWEEP:
        formatted = REFUSAL_MESSAGES["WARN_NTC_RECENT"].format(date=d.isoformat())
        assert line_leaks(formatted, GLOBAL_LEAK_PATTERNS) is False, d


def test_formatted_full_label_passes_the_global_tier_for_every_date() -> None:
    for d in DATE_SWEEP:
        formatted = LABEL_FULL_FMT.format(report_date=d.isoformat())
        assert line_leaks(formatted, GLOBAL_LEAK_PATTERNS) is False, d


def test_the_stale_account_confirm_strings_pass_both_tiers_for_every_date() -> None:
    # derive/emi.py's real prefix plus the dossier's capitalised mandated copy, over the same 40 years of dates
    for d in DATE_SWEEP:
        for template in (f"{_STALE_CONFIRM_PREFIX} {{d}}", "An account last updated {d}"):
            line = template.format(d=d.isoformat())
            assert line_leaks(line, LLM_LEAK_PATTERNS) is False, line
            assert line_leaks(line, GLOBAL_LEAK_PATTERNS) is False, line


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
    stdlib = {"__future__", "collections", "dataclasses", "datetime", "math", "re", "typing", "unicodedata"}
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
