"""Task 3: config.py is the env registry, the SUREPASS_LIVE kill switch and the FIXED per-call timeouts.

Hermetic: every test builds ``Settings(_env_file=None, ALIAS=...)`` after the autouse fixture has scrubbed every
registered env name from the process environment, so neither a developer's shell nor a stray ``.env`` can leak in.
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast, get_origin

import pytest

from eligibility_agent import config
from eligibility_agent.config import (
    CONNECT_TIMEOUT_S,
    DEFAULT_DEV_HMAC_KEY,
    OPENAI_TIMEOUT_S,
    PAN_TIMEOUT_S,
    REPORT_TIMEOUT_S,
    SCORE_TIMEOUT_S,
    ConfigError,
    Settings,
    get_settings,
    utcnow,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
CONFIG_PY = REPO_ROOT / "eligibility_agent" / "config.py"

# Read straight from os.environ (the checkpointer factory; create_app's forced tracing-hide defaults), deliberately not
# Settings fields. Both appear in .env.example as recommended lines.
NOT_A_SETTING = frozenset({"LANGGRAPH_STRICT_MSGPACK", "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS"})


def _aliases() -> set[str]:
    out: set[str] = set()
    for info in Settings.model_fields.values():
        alias = info.validation_alias
        assert isinstance(alias, str), "every Settings field must carry its env name as a plain-str validation_alias"
        out.add(alias)
    return out


def make(**env: Any) -> Settings:
    """Hermetic constructor: init kwargs use the env names (aliases); no .env file is read."""
    return cast(Any, Settings)(_env_file=None, **env)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    for name in _aliases() | NOT_A_SETTING:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


SUPABASE_FULL: Final = {
    "REPO_BACKEND": "supabase",
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "service-role-key",
    "IDENTITY_HMAC_KEY": "a-real-production-hmac-key",
    "SERVICE_API_TOKEN": "bearer-token",
}


# 1. Empty environment: degraded boot is a first-class, error-free state.
def test_empty_env_constructs_and_boots_degraded() -> None:
    s = make()
    assert s.startup_errors() == []
    assert s.surepass_live is False
    assert s.auth_enabled is False
    assert s.langsmith_tracing is False
    assert s.repo_backend == "memory"
    assert s.raw_bucket == "raw-responses"
    assert s.graph_timeout_seconds == 90.0
    assert s.surepass_base_url == "https://kyc-api.surepass.app"
    assert s.identity_hmac_key == DEFAULT_DEV_HMAC_KEY == "dev-only-hmac-key-do-not-use-in-prod"
    assert s.openai_model == "gpt-4.1"
    assert s.langsmith_project == "eligibility-agent"
    assert s.prompt_source == "local"
    assert s.log_level == "INFO"
    for optional in (
        s.surepass_token,
        s.service_api_token,
        s.supabase_url,
        s.supabase_service_role_key,
        s.database_url,
        s.openai_api_key,
        s.langsmith_api_key,
        s.prompt_tag,
    ):
        assert optional is None


def test_default_env_file_lookup_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No _env_file=None override: the default ".env" lookup is optional and a missing file is not an error.
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.startup_errors() == []


# Blank env values (docker-compose / PaaS passthrough `FOO=${FOO}` injects "" for unset vars) are "unset":
# they must fall back to the field default, never crash Settings() and never become "" where None is expected.
def test_blank_env_vars_fall_back_to_defaults_and_never_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _aliases():
        monkeypatch.setenv(name, "")
    s = make()
    assert s.startup_errors() == []
    assert s.graph_timeout_seconds == 90.0  # blank float would otherwise be a ValidationError
    assert s.repo_backend == "memory"  # blank Literal would otherwise be a ValidationError
    assert s.prompt_source == "local"
    assert s.surepass_live is False and s.langsmith_tracing is False and s.auth_enabled is False
    assert s.raw_bucket == "raw-responses"
    assert s.surepass_base_url == "https://kyc-api.surepass.app"
    assert s.log_level == "INFO"
    assert s.openai_model == "gpt-4.1" and s.langsmith_project == "eligibility-agent"
    assert s.identity_hmac_key == DEFAULT_DEV_HMAC_KEY
    for optional in (
        s.surepass_token,
        s.service_api_token,
        s.supabase_url,
        s.supabase_service_role_key,
        s.database_url,
        s.openai_api_key,
        s.langsmith_api_key,
        s.prompt_tag,
    ):
        assert optional is None  # blank must be None, so downstream `is None` checks stay correct


@pytest.mark.parametrize("name", ["GRAPH_TIMEOUT_SECONDS", "REPO_BACKEND", "PROMPT_SOURCE"])
def test_blank_typed_env_var_does_not_raise_validation_error(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "")
    assert make().startup_errors() == []


def test_blank_identity_hmac_key_env_is_caught_by_the_non_default_check(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {**SUPABASE_FULL, "IDENTITY_HMAC_KEY": ""}.items():
        monkeypatch.setenv(name, value)
    s = make()
    assert s.identity_hmac_key == DEFAULT_DEV_HMAC_KEY  # blank -> dev default ...
    errors = s.startup_errors()
    assert len(errors) == 1 and "IDENTITY_HMAC_KEY" in errors[0]  # ... which supabase refuses


def test_blank_values_in_dotenv_file_are_ignored_too(tmp_path: Path) -> None:
    # `cp .env.example .env` then uncommenting lines leaves `VAR=` blanks; the dotenv source honours the same rule.
    dotenv = tmp_path / ".env"
    dotenv.write_text("GRAPH_TIMEOUT_SECONDS=\nREPO_BACKEND=\nOPENAI_API_KEY=\nRAW_BUCKET=\nSERVICE_API_TOKEN=\n")
    s = cast(Any, Settings)(_env_file=dotenv)
    assert (s.graph_timeout_seconds, s.repo_backend, s.raw_bucket) == (90.0, "memory", "raw-responses")
    assert s.openai_api_key is None and s.service_api_token is None and s.auth_enabled is False
    assert s.startup_errors() == []


# 2. Kill switch: live iff the value is EXACTLY the lowercase literal "true".
@pytest.mark.parametrize(("value", "expected"), [("true", True), ("True", False), ("TRUE", False), ("1", False), ("yes", False), ("", False), (" true", False), ("true ", False)])
def test_surepass_live_is_exact_literal(value: str, expected: bool) -> None:
    assert make(SUREPASS_LIVE=value).surepass_live is expected


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("True", False), ("TRUE", False), ("1", False), ("yes", False), ("", False)])
def test_langsmith_tracing_is_exact_literal(value: str, expected: bool) -> None:
    assert make(LANGSMITH_TRACING=value).langsmith_tracing is expected


def test_kill_switch_reads_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUREPASS_LIVE", "true")
    assert make().surepass_live is True
    monkeypatch.setenv("SUREPASS_LIVE", "TRUE")
    assert make().surepass_live is False


def test_auth_enabled_follows_service_api_token() -> None:
    assert make(SERVICE_API_TOKEN="tok").auth_enabled is True
    assert make(SERVICE_API_TOKEN="").auth_enabled is False  # `SERVICE_API_TOKEN=` left blank in .env


# 3. Supabase backend: exactly the hard-error triplet; fully configured is clean.
def test_supabase_backend_alone_yields_exactly_the_error_triplet() -> None:
    errors = make(REPO_BACKEND="supabase").startup_errors()
    assert len(errors) == 3
    joined = "\n".join(errors)
    assert "SUPABASE_URL" in errors[0] and "SUPABASE_SERVICE_ROLE_KEY" in errors[0]
    assert "IDENTITY_HMAC_KEY" in errors[1]
    assert "SERVICE_API_TOKEN" in errors[2]
    assert "SUREPASS_TOKEN" not in joined


def test_supabase_backend_fully_configured_is_clean() -> None:
    assert make(**SUPABASE_FULL).startup_errors() == []


@pytest.mark.parametrize("drop", ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"])
def test_supabase_needs_both_url_and_key(drop: str) -> None:
    env = {k: v for k, v in SUPABASE_FULL.items() if k != drop}
    errors = make(**env).startup_errors()
    assert len(errors) == 1 and "SUPABASE_URL" in errors[0]


def test_supabase_rejects_default_or_blank_hmac_key() -> None:
    for bad in (DEFAULT_DEV_HMAC_KEY, ""):
        env = {**SUPABASE_FULL, "IDENTITY_HMAC_KEY": bad}
        errors = make(**env).startup_errors()
        assert len(errors) == 1 and "IDENTITY_HMAC_KEY" in errors[0]


def test_supabase_blank_service_token_counts_as_missing() -> None:
    errors = make(**{**SUPABASE_FULL, "SERVICE_API_TOKEN": ""}).startup_errors()
    assert len(errors) == 1 and "SERVICE_API_TOKEN" in errors[0]


def test_memory_backend_tolerates_default_hmac_key_and_no_token() -> None:
    s = make(REPO_BACKEND="memory", IDENTITY_HMAC_KEY=DEFAULT_DEV_HMAC_KEY)
    assert s.startup_errors() == []


def test_unknown_backend_and_prompt_source_are_rejected() -> None:
    with pytest.raises(ValueError):
        make(REPO_BACKEND="sqlite")
    with pytest.raises(ValueError):
        make(PROMPT_SOURCE="s3")


# 4. Live SurePass needs a token.
def test_live_without_token_is_one_error() -> None:
    errors = make(SUREPASS_LIVE="true").startup_errors()
    assert len(errors) == 1 and "SUREPASS_TOKEN" in errors[0]


def test_live_with_token_is_clean() -> None:
    assert make(SUREPASS_LIVE="true", SUREPASS_TOKEN="tok").startup_errors() == []


def test_blank_token_counts_as_missing_when_live() -> None:
    errors = make(SUREPASS_LIVE="true", SUREPASS_TOKEN="").startup_errors()
    assert len(errors) == 1 and "SUREPASS_TOKEN" in errors[0]


def test_non_exact_live_value_is_stub_so_no_token_needed() -> None:
    assert make(SUREPASS_LIVE="True").startup_errors() == []


def test_live_and_supabase_errors_accumulate() -> None:
    assert len(make(SUREPASS_LIVE="true", REPO_BACKEND="supabase").startup_errors()) == 4


# 5. Fixed timeouts are module-level Final constants, not env, not Settings fields.
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PAN_TIMEOUT_S", 8.0),
        ("SCORE_TIMEOUT_S", 12.0),
        ("REPORT_TIMEOUT_S", 20.0),
        ("OPENAI_TIMEOUT_S", 8.0),
        ("CONNECT_TIMEOUT_S", 5.0),
    ],
)
def test_timeout_constants_are_exact_floats_and_final(name: str, value: float) -> None:
    actual = getattr(config, name)
    assert actual == value and isinstance(actual, float)
    assert get_origin(inspect.get_annotations(config)[name]) is Final


def test_timeout_constants_import_by_name() -> None:
    assert (PAN_TIMEOUT_S, SCORE_TIMEOUT_S, REPORT_TIMEOUT_S, OPENAI_TIMEOUT_S, CONNECT_TIMEOUT_S) == (8.0, 12.0, 20.0, 8.0, 5.0)


def test_no_temperature_or_per_call_timeout_in_settings() -> None:
    fields = set(Settings.model_fields)
    assert "OPENAI_TEMPERATURE" not in fields and "openai_temperature" not in fields
    assert not any("temperature" in f.lower() for f in fields)
    assert not any(f in fields for f in ("pan_timeout_s", "score_timeout_s", "report_timeout_s", "openai_timeout_s"))
    for f in fields:
        if "timeout" in f:
            assert f == "graph_timeout_seconds"
    timeout_aliases = {a for a in _aliases() if "TIMEOUT" in a}
    assert timeout_aliases == {"GRAPH_TIMEOUT_SECONDS"}
    # ...and an env var cannot override a fixed constant either.
    s = make(PAN_TIMEOUT_S="99", SCORE_TIMEOUT_S="99", REPORT_TIMEOUT_S="99", OPENAI_TIMEOUT_S="99")
    assert not hasattr(s, "pan_timeout_s")
    assert PAN_TIMEOUT_S == 8.0


def test_no_hashers_or_retired_envs_in_config() -> None:
    names = {n for n in dir(config) if not n.startswith("__")}
    assert not {"hash_pii", "hash_pan", "hash_mobile", "last4"} & names
    forbidden = {"PII_HMAC_KEY", "PAN_HASH_SALT", "OPENAI_TEMPERATURE"}
    assert not forbidden & _aliases()


def test_config_is_dependency_light() -> None:
    tree = ast.parse(CONFIG_PY.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports in config.py"
            roots.add((node.module or "").split(".")[0])
    assert "eligibility_agent" not in roots, "config.py must not import any eligibility_agent module"
    assert "supabase" not in roots
    assert roots <= {"__future__", "datetime", "functools", "typing", "pydantic", "pydantic_settings"}


# 6. Env-name parity with the committed .env.example (pins config <-> example drift).
def _example_vars() -> set[str]:
    found: set[str] = set()
    for line in ENV_EXAMPLE.read_text().splitlines():
        m = re.match(r"^#\s*([A-Z][A-Z0-9_]*)=", line)
        if m:
            found.add(m.group(1))
    return found


def test_aliases_match_env_example_exactly() -> None:
    example = _example_vars()
    assert "LANGGRAPH_STRICT_MSGPACK" in example  # the parse is working and the exclusion is meaningful
    assert _aliases() == example - NOT_A_SETTING


def test_every_env_example_default_matches_the_field_default() -> None:
    # `# VAR=value` lines that show a concrete default must agree with the code.
    shown: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        m = re.match(r"^#\s*([A-Z][A-Z0-9_]*)=(\S+)$", line)
        if m:
            shown[m.group(1)] = m.group(2)
    s = make()
    by_alias = {info.validation_alias: name for name, info in Settings.model_fields.items()}
    for env in ("SUREPASS_BASE_URL", "REPO_BACKEND", "RAW_BUCKET", "OPENAI_MODEL", "LANGSMITH_PROJECT", "PROMPT_SOURCE", "LOG_LEVEL"):
        assert str(getattr(s, by_alias[env])) == shown[env], env
    assert s.graph_timeout_seconds == float(shown["GRAPH_TIMEOUT_SECONDS"])


# Helpers.
def test_config_error_is_a_runtime_error() -> None:
    assert issubclass(ConfigError, RuntimeError)


def test_utcnow_is_aware_utc() -> None:
    now = utcnow()
    assert isinstance(now, datetime)
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        first = get_settings()
        assert get_settings() is first
    finally:
        get_settings.cache_clear()
