"""Task 7: HMAC identity hashing, normalization and last-4 display helpers (K3; PRD 5.2).

The two pinned vectors are the contract with every other service that must produce the same cache key; each one
was recomputed with the stdlib ``hmac`` module independently of the code under test (and the independent
recomputation is repeated live in ``test_pinned_vectors_match_an_independent_recomputation``).
"""

from __future__ import annotations

import hashlib
import hmac
import re

import pytest

from eligibility_agent.policy.guards import (
    IDENTITY_HMAC_KEY_ENV,
    MOBILE_RE,
    PAN_RE,
    hash_mobile,
    hash_pan,
    mobile_last4,
    normalize_mobile,
    normalize_pan,
    pan_last4,
)

KEY = "test-hmac-key"
PINNED_PAN_HASH = "7c0b91b00e0094dad4090227778b2996da1ec7c7e9c45d5b4ac4b0883e55edc0"
PINNED_MOBILE_HASH = "160381e0c64be1a856ab25167a3a22468ccce58e2641993fd4407183cd2c858a"


# --------------------------------------------------------------------------- pinned vectors


def test_pinned_pan_vector() -> None:
    assert hash_pan(" abcde1234f ", key=KEY) == PINNED_PAN_HASH


def test_pinned_mobile_vector() -> None:
    assert hash_mobile("+91 98765-43210", key=KEY) == PINNED_MOBILE_HASH


def test_pinned_vectors_match_an_independent_recomputation() -> None:
    assert hmac.new(KEY.encode(), b"pan:ABCDE1234F", hashlib.sha256).hexdigest() == PINNED_PAN_HASH
    assert hmac.new(KEY.encode(), b"mobile:9876543210", hashlib.sha256).hexdigest() == PINNED_MOBILE_HASH


def test_hash_shape_is_64_lowercase_hex() -> None:
    for digest in (hash_pan("ABCDE1234F", key=KEY), hash_mobile("9876543210", key=KEY)):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_hashes_are_deterministic_and_key_dependent() -> None:
    assert hash_pan("ABCDE1234F", key=KEY) == hash_pan("ABCDE1234F", key=KEY)
    assert hash_pan("ABCDE1234F", key=KEY) != hash_pan("ABCDE1234F", key="another-key")
    assert hash_mobile("9876543210", key=KEY) != hash_mobile("9876543210", key="another-key")


def test_every_spelling_of_one_identity_hashes_identically() -> None:
    pans = [" abcde1234f ", "ABCDE1234F", "abcde1234F\t", "\nAbCdE1234f"]
    assert {hash_pan(p, key=KEY) for p in pans} == {PINNED_PAN_HASH}
    mobiles = ["+91 98765-43210", "9876543210", "919876543210", "09876543210", "(98765) 43210", "+919876543210"]
    assert {hash_mobile(m, key=KEY) for m in mobiles} == {PINNED_MOBILE_HASH}


# --------------------------------------------------------------------------- domain separation


def test_pan_and_mobile_messages_are_domain_separated() -> None:
    """The same key hashes b'pan:'+x and b'mobile:'+x to different digests: a PAN can never collide with a mobile."""
    pan_digest = hmac.new(KEY.encode(), b"pan:ABCDE1234F", hashlib.sha256).hexdigest()
    mobile_digest = hmac.new(KEY.encode(), b"mobile:ABCDE1234F", hashlib.sha256).hexdigest()
    assert pan_digest != mobile_digest
    assert hash_pan("ABCDE1234F", key=KEY) == pan_digest


def test_a_hash_without_the_domain_prefix_is_not_the_identity_hash() -> None:
    bare = hmac.new(KEY.encode(), b"ABCDE1234F", hashlib.sha256).hexdigest()
    assert hash_pan("ABCDE1234F", key=KEY) != bare
    bare_mobile = hmac.new(KEY.encode(), b"9876543210", hashlib.sha256).hexdigest()
    assert hash_mobile("9876543210", key=KEY) != bare_mobile


def test_key_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        hash_pan("ABCDE1234F", KEY)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        hash_mobile("9876543210", KEY)  # pyright: ignore[reportCallIssue]


def test_empty_key_is_refused() -> None:
    with pytest.raises(ValueError, match="IDENTITY_HMAC_KEY"):
        hash_pan("ABCDE1234F", key="")
    with pytest.raises(ValueError, match="IDENTITY_HMAC_KEY"):
        hash_mobile("9876543210", key="")


def test_env_name_constant() -> None:
    assert IDENTITY_HMAC_KEY_ENV == "IDENTITY_HMAC_KEY"


# --------------------------------------------------------------------------- normalize_pan


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ABCDE1234F", "ABCDE1234F"),
        (" abcde1234f ", "ABCDE1234F"),
        ("\tabcde1234f\n", "ABCDE1234F"),
        ("AbCdE1234f", "ABCDE1234F"),
    ],
)
def test_normalize_pan_accepts_and_canonicalises(raw: str, expected: str) -> None:
    assert normalize_pan(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "ABCDE1234",  # too short
        "ABCDE12345F",  # too long
        "ABCD11234F",  # digit in the letter block
        "ABCDE123AF",  # letter in the digit block
        "ABCDE12341",  # digit as check letter
        "ABC DE1234F",  # interior space is NOT stripped
        "ABCDE-1234F",
        "ABCDE1234F\nX",
        "ABCß1234F",  # 'ß'.upper() == 'SS' would otherwise fold into a valid-looking PAN
        "ABCDı" + "1234F",  # dotless i upper-cases to ASCII 'I'
        "ＡBCDE1234F",  # fullwidth A
    ],
)
def test_normalize_pan_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_pan(raw)


def test_hash_pan_rejects_a_malformed_pan() -> None:
    with pytest.raises(ValueError):
        hash_pan("not-a-pan", key=KEY)


def test_pan_re_is_the_canonical_pattern() -> None:
    assert PAN_RE.fullmatch("ABCDE1234F") is not None
    assert PAN_RE.fullmatch("abcde1234f") is None  # the regex runs on the NORMALIZED form
    assert PAN_RE.fullmatch("ABCDE1234F ") is None


# --------------------------------------------------------------------------- normalize_mobile


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9876543210", "9876543210"),
        ("98765 43210", "9876543210"),
        ("98765-43210", "9876543210"),
        ("(98765) 43210", "9876543210"),
        (" 98765 - 43210 ", "9876543210"),
        ("+91 98765-43210", "9876543210"),
        ("+919876543210", "9876543210"),
        ("+91-(98765)-43210", "9876543210"),
        ("919876543210", "9876543210"),  # '91' stripped because 12 digits remain
        ("91 98765 43210", "9876543210"),
        ("09876543210", "9876543210"),  # '0' stripped because 11 digits remain
        ("0 98765 43210", "9876543210"),
        ("6000000000", "6000000000"),  # lowest valid leading digit
        ("9198765432", "9198765432"),  # a legitimate 10-digit number that merely STARTS with 91 is untouched
    ],
)
def test_normalize_mobile_accepts_and_canonicalises(raw: str, expected: str) -> None:
    assert normalize_mobile(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "5876543210",  # leading digit below 6
        "0987654321",  # 10 digits starting 0
        "98765432",  # too short
        "98765432101",  # 11 digits not starting 0
        "9198765432100",  # 13 digits
        "+1 9876543210",  # not an Indian country code
        "+91",  # nothing after the prefix
        "98765x43210",  # stray letter
        "+9876543210",  # '+' without 91
        "٩٨٧٦٥٤٣٢١٠",  # Arabic-Indic digits are not [0-9]
    ],
)
def test_normalize_mobile_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_mobile(raw)


def test_hash_mobile_rejects_a_malformed_number() -> None:
    with pytest.raises(ValueError):
        hash_mobile("12345", key=KEY)


def test_mobile_re_is_the_canonical_pattern() -> None:
    assert MOBILE_RE.fullmatch("9876543210") is not None
    assert MOBILE_RE.fullmatch("5876543210") is None
    assert MOBILE_RE.fullmatch("+919876543210") is None  # the regex runs on the NORMALIZED form


# --------------------------------------------------------------------------- last-4 display helpers


def test_pan_last4() -> None:
    assert pan_last4("ABCDE1234F") == "234F"
    assert pan_last4(" abcde1234f ") == "234F"


def test_mobile_last4() -> None:
    assert mobile_last4("9876543210") == "3210"
    assert mobile_last4("+91 98765-43210") == "3210"


def test_last4_helpers_validate_their_input() -> None:
    with pytest.raises(ValueError):
        pan_last4("nope")
    with pytest.raises(ValueError):
        mobile_last4("nope")
