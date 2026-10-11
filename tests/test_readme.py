"""Task 16: README.md cannot drift from the code it documents.

The environment table is pinned to ``config.Settings`` (names, defaults, startup requirements), the documented commands and
paths must exist, the SQL the README tells ops to run must match schema.sql's own comments, every API code and review kind
must be explained, and the examples stay synthetic.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any, Final, cast, get_args

import pytest

from eligibility_agent.config import (
    CONNECT_TIMEOUT_S,
    OPENAI_TIMEOUT_S,
    PAN_TIMEOUT_S,
    REPORT_TIMEOUT_S,
    SCORE_TIMEOUT_S,
    Settings,
)
from eligibility_agent.graph.prompts import HUB_RANK, PROMPTS_LOCAL_VERSION
from eligibility_agent.graph.state import Code, ReviewKind
from eligibility_agent.policy.guards import IN_FLIGHT_LOCK_TTL, OpportunityStage
from tests.helpers import PROBE_LITERALS

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
README: Final = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
SCHEMA: Final = (REPO_ROOT / "eligibility_agent" / "db" / "schema.sql").read_text(encoding="utf-8")
ENV_EXAMPLE: Final = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
NOT_SETTINGS: Final = {"LANGGRAPH_STRICT_MSGPACK", "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS"}


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _section(heading: str) -> str:
    start = README.index(heading)
    nxt = README.find("\n## ", start + len(heading))
    return README[start : nxt if nxt != -1 else len(README)]


def _env_rows() -> dict[str, tuple[str, str]]:
    """name -> (default cell without backticks, 'Needed when' cell) from section 3's table."""
    rows: dict[str, tuple[str, str]] = {}
    for line in _section("## 3. Configuration").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"`[A-Z][A-Z0-9_]*`", cells[0]):
            rows[cells[0].strip("`")] = (cells[1].strip("`"), cells[2])
    return rows


def _settings(**env: str) -> Settings:
    """Hermetic constructor: init kwargs use the env names (aliases); no .env file is read."""
    return cast(Any, Settings)(_env_file=None, **env)


def _aliases() -> dict[str, str]:
    out: dict[str, str] = {}
    for field, info in Settings.model_fields.items():
        assert isinstance(info.validation_alias, str)
        out[info.validation_alias] = field
    return out


def test_the_env_table_lists_exactly_the_settings_fields() -> None:
    assert set(_env_rows()) == set(_aliases())


def test_every_env_default_matches_the_field_default() -> None:
    rows = _env_rows()
    for alias, field in _aliases().items():
        default: object = Settings.model_fields[field].default
        cell = rows[alias][0]
        if default is None or default == "":
            assert cell == "unset", alias
        elif isinstance(default, float):
            assert float(cell) == default, alias
        else:
            assert cell == str(default), alias


def test_the_documented_startup_requirements_are_the_ones_the_code_enforces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDENTITY_HMAC_KEY")  # conftest exports the test key; the requirement is about the DEFAULT
    rows = _env_rows()
    supabase = _settings(REPO_BACKEND="supabase").startup_errors()
    live = _settings(SUREPASS_LIVE="true").startup_errors()
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SERVICE_API_TOKEN", "IDENTITY_HMAC_KEY"):
        assert any(name in error for error in supabase), name  # the code really requires it ...
        assert "REPO_BACKEND=supabase" in rows[name][1], name  # ... and the README says so
    assert any("SUREPASS_TOKEN" in error for error in live)
    assert "SUREPASS_LIVE=true" in rows["SUREPASS_TOKEN"][1]


def test_the_non_settings_variables_are_documented_in_the_readme_and_the_example() -> None:
    for name in NOT_SETTINGS:
        assert f"`{name}=true`" in README, name
        assert f"# {name}=true" in ENV_EXAMPLE, name


def test_every_variable_the_scripts_and_evals_read_themselves_is_in_the_must_export_list() -> None:
    """A `.env` file feeds Settings only: whatever scripts/ and the eval runner read from the environment directly must be
    named in the README paragraph that says to export it (CI-provided GITHUB_SHA excepted)."""
    paragraph = _flat(README[README.index("**A `.env` file feeds `Settings` only**") :].split("\n\n", 1)[0])
    read: set[str] = set()
    sources = [*(REPO_ROOT / "scripts").glob("*.py"), *(REPO_ROOT / "eligibility_agent" / "evals").glob("*.py")]
    for source in sources:
        read |= set(re.findall(r'environ(?:\.get)?[\[(]\s*"([A-Z][A-Z0-9_]*)"', source.read_text(encoding="utf-8")))
    read.discard("GITHUB_SHA")
    assert {"SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "RAW_BUCKET", "OPENAI_API_KEY", "OPENAI_MODEL"} <= read
    for name in read:
        assert f"`{name}`" in paragraph, name


def test_the_ci_install_flag_is_documented() -> None:
    assert "`uv sync --locked --all-groups`" in README
    assert "(`--frozen` would not)" in _flat(README)


def test_the_fixed_timeouts_and_lock_ttl_quoted_match_the_code() -> None:
    sentence = (
        f"PAN Comprehensive {PAN_TIMEOUT_S:g} s, Experian Score {SCORE_TIMEOUT_S:g} s, Experian Report {REPORT_TIMEOUT_S:g} s, "
        f"OpenAI {OPENAI_TIMEOUT_S:g} s, connect {CONNECT_TIMEOUT_S:g} s"
    )
    assert sentence in _flat(README)
    assert f"{int(IN_FLIGHT_LOCK_TTL.total_seconds())} s" in README


def test_every_api_code_and_review_kind_is_explained() -> None:
    for code in get_args(Code):
        assert f"| `{code}` |" in README, code
    for kind in get_args(ReviewKind):
        assert f"| `{kind}` |" in README, kind


def test_the_documented_stage_values_are_the_enum() -> None:
    values = ", ".join(f"`{stage.value}`" for stage in OpportunityStage)
    assert values in _flat(README)


def test_the_prompt_ops_note_names_the_current_version_and_hub_prompt() -> None:
    assert f"`{PROMPTS_LOCAL_VERSION}`" in README
    assert HUB_RANK in README
    assert "prompt rule 7" in README


def test_the_run_command_and_the_module_commands_resolve() -> None:
    assert "uv run uvicorn 'eligibility_agent.main:app_factory' --factory" in README
    module = importlib.import_module("eligibility_agent.main")
    assert callable(module.app_factory)
    for command in set(re.findall(r"uv run python -m ([\w.]+)", README)):
        assert importlib.util.find_spec(command) is not None, command


def test_documented_paths_and_links_exist() -> None:
    for target in set(re.findall(r"\]\(([^)#]+)\)", README)):
        if "://" not in target:
            assert (REPO_ROOT / target).exists(), target
    for path in set(re.findall(r"`((?:scripts|docs|eligibility_agent|tests)/[\w./-]+\.(?:py|md|sql|yml|toml))`", README)):
        assert (REPO_ROOT / path).exists(), path


def test_the_checkpointer_role_sql_matches_the_schema_comments() -> None:
    for statement in (
        "alter role checkpointer_svc in database postgres set search_path = checkpointer;",
        "grant usage, create on schema checkpointer to checkpointer_svc;",
    ):
        assert f"-- {statement}" in SCHEMA, statement
        assert statement in README, statement
    assert "create role checkpointer_svc login password" in SCHEMA and "create role checkpointer_svc login password" in README
    assert "create_bucket('raw-responses'" in SCHEMA and 'create_bucket("raw-responses"' in README
    assert "psql" in SCHEMA.splitlines()[0] and 'psql "$SUPABASE_DB_URL" -f eligibility_agent/db/schema.sql' in README


def test_the_readme_sql_uses_real_table_and_column_names() -> None:
    for token in ("review_queue", "resolved_at", "resolved_by", "surepass_calls", "billable", "latency_ms", "error_kind"):
        assert token in SCHEMA, token
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        assert f"checkpointer.{table}" in README


def test_examples_stay_synthetic() -> None:
    for literal in PROBE_LITERALS:
        assert literal not in README, literal
    pan_shaped = set(re.findall(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", README))
    assert pan_shaped <= {"ABCDE1234F"}, pan_shaped  # only the PRD's documented format example


@pytest.mark.parametrize(
    "heading",
    [
        "## 1. Quick start",
        "## 2. API",
        "## 3. Configuration",
        "## 4. First-time Supabase setup",
        "## 5. Seeding lenders",
        "## 6. Tests, type-checking and evals",
        "## 7. Operations",
        "## 8. Integration notes (SuperConnect / FulCRM)",
        "## 9. Open items and the owner decision register",
        "## 10. Repository layout",
    ],
)
def test_every_section_the_contents_line_links_exists(heading: str) -> None:
    assert f"\n{heading}\n" in README
