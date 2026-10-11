"""Task 16: .github/workflows/ci.yml is internally consistent and keeps the standing CI rules.

A parse-level check (no GitHub needed): the job graph, the uv setup, the secrets gating and the one rule that must never
regress, no SurePass live switch anywhere in the file. Run actionlint for the expression/shell lint; this file pins the
project rules. The `docker` job (build the production image, smoke-test it with an EMPTY environment) is pinned here too,
because it is the only thing that validates the Dockerfile: tests/test_dockerfile.py pins the file, this pins how CI runs it.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, Final, cast

import pytest
import yaml

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
WORKFLOW: Final = REPO_ROOT / ".github" / "workflows" / "ci.yml"
LOCAL_EVALS: Final = "python -m eligibility_agent.evals.run --mode local --llm fake --dataset both"
# The degraded-boot /health body the docker job asserts. It names the response key, so it is the one string the
# "no SurePass text in the workflow" rule must tolerate (and only inside the docker job).
HEALTH_BODY: Final = '{"ok":true,"surepass_live":false}'

Doc = dict[str, Any]


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _workflow() -> Doc:
    return cast(Doc, yaml.safe_load(_text()))


def _triggers() -> dict[Any, Any]:
    raw = cast("dict[Any, Any]", yaml.safe_load(_text()))
    return cast("dict[Any, Any]", raw.get("on", raw.get(True)))  # YAML 1.1 reads the bare key `on` as True


def _jobs() -> dict[str, Doc]:
    return cast("dict[str, Doc]", _workflow()["jobs"])


def _steps(job: str) -> list[Doc]:
    return cast("list[Doc]", _jobs()[job]["steps"])


def _needs(job: str) -> list[str]:
    needs: Any = _jobs()[job].get("needs", [])
    return [needs] if isinstance(needs, str) else cast("list[str]", needs)


def _runs(job: str) -> list[str]:
    return [str(step["run"]) for step in _steps(job) if "run" in step]


def _uv_index(job: str, predicate: str) -> int:
    return next(i for i, step in enumerate(_steps(job)) if predicate in str(step.get("run", "")))


def test_the_workflow_parses_and_has_the_four_jobs() -> None:
    assert set(_jobs()) == {"test", "evals-local", "docker", "evals-langsmith"}
    assert {"push", "pull_request", "workflow_dispatch"} <= set(_triggers())


def test_job_dependencies_resolve_and_the_evals_wait_for_the_tests() -> None:
    assert _needs("test") == []
    for job in ("evals-local", "docker", "evals-langsmith"):
        assert _needs(job) == ["test"]
        assert set(_needs(job)) <= set(_jobs())


@pytest.mark.parametrize("job", ["test", "evals-local", "evals-langsmith"])
def test_every_job_sets_up_uv_on_the_pinned_python_before_it_runs_uv(job: str) -> None:
    pinned = (REPO_ROOT / ".python-version").read_text().strip()
    steps = _steps(job)
    setup = next(i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("astral-sh/setup-uv@"))
    checkout = next(i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/checkout@"))
    assert checkout < setup < _uv_index(job, "uv sync") < _uv_index(job, "uv run")
    assert str(cast("Doc", steps[setup]["with"])["python-version"]) == pinned
    for step in steps:
        if "uses" in step:
            assert re.search(r"@v\d+\.\d+\.\d+$", str(step["uses"])), f"unpinned action {step['uses']}"


def test_every_sync_is_locked_so_a_stale_lockfile_fails_the_build() -> None:
    """`uv sync --frozen` installs whatever uv.lock says WITHOUT checking it against pyproject.toml; only `--locked` fails
    when the lockfile is stale. Run steps stay `--frozen` (never re-resolve)."""
    syncs = 0
    for job in _jobs():
        for run in _runs(job):
            for line in re.findall(r"uv sync\b[^\n]*", run):
                syncs += 1
                assert "--locked" in line and "--frozen" not in line, line
                assert "--all-groups" in line, line  # dev tools (pyright, pytest, pyyaml) come from the dev group
            for line in re.findall(r"uv run\b[^\n]*", run):
                assert "--frozen" in line, line
    assert syncs == 3  # one install per job


def test_the_workflow_parser_is_a_declared_dev_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = cast("list[str]", pyproject["dependency-groups"]["dev"])
    assert any(re.match(r"pyyaml\b", requirement, re.IGNORECASE) for requirement in dev)


def test_test_job_runs_pyright_then_pytest_and_nothing_secret() -> None:
    runs = _runs("test")
    assert [r for r in runs if "pyright" in r] == ["uv run --frozen pyright"]
    assert [r for r in runs if "pytest" in r] == ["uv run --frozen pytest -q"]
    assert runs.index("uv run --frozen pyright") < runs.index("uv run --frozen pytest -q")
    assert "secrets." not in str(_jobs()["test"])


def test_local_evals_run_the_local_fake_command_and_use_no_secrets() -> None:
    job = _jobs()["evals-local"]
    assert any(LOCAL_EVALS in run for run in _runs("evals-local"))
    assert "secrets." not in str(job)
    assert not job.get("continue-on-error")  # a non-zero exit must fail the job


def test_langsmith_job_is_informational_and_secrets_gated() -> None:
    job = _jobs()["evals-langsmith"]
    assert job["continue-on-error"] is True
    # never on a pull_request: the job holds secrets and runs code the PR controls
    assert job["if"] == "github.event_name != 'pull_request'"
    assert not [name for name in ("test", "evals-local", "docker") if "if" in _jobs()[name]]  # the blocking jobs always run
    steps = _steps("evals-langsmith")
    gate_index = next(i for i, s in enumerate(steps) if s.get("id") == "gate")
    assert "secrets.LANGSMITH_API_KEY" in str(steps[gate_index]["env"])
    for step in steps[gate_index + 1 :]:
        assert step.get("if") == "steps.gate.outputs.run == 'true'", step  # skipped, never failed, without the secret
    # the secrets reach exactly the gate and the run step; the install step never sees them
    with_secrets = [s.get("name") or s.get("uses") for s in steps if "secrets." in str(s)]
    assert with_secrets == ["Gate on secrets", "LangSmith experiment (exits 0 whatever the scores are)"]
    run = next(r for r in _runs("evals-langsmith") if "evals.run" in r)
    assert "--mode langsmith" in run and "--max-real 20" in run
    # datasets are uploaded only on request (manual dispatch), not on every run: upsert semantics are unverified
    assert "--build-datasets" in run and 'BUILD_DATASETS" = "true"' in run
    dispatch = cast("dict[str, Any]", _triggers()["workflow_dispatch"])
    assert dispatch["inputs"]["build_datasets"]["default"] is False


def test_secrets_appear_only_in_the_langsmith_job() -> None:
    for name, job in _jobs().items():
        assert ("secrets." in str(job)) == (name == "evals-langsmith"), name


def test_the_surepass_live_switch_is_never_set_or_even_referenced_by_a_command() -> None:
    workflow = _workflow()
    envs: list[Doc] = [cast(Doc, workflow.get("env") or {})]
    for job in _jobs().values():
        envs.append(cast(Doc, job.get("env") or {}))
        envs.extend(cast(Doc, step.get("env") or {}) for step in cast("list[Doc]", job["steps"]))
    assert not [name for env in envs for name in env if name.upper().startswith("SUREPASS")]
    # the one tolerated string is the docker job's expected /health body (it names the response key, not a switch)
    for job in _jobs():
        assert not any("SUREPASS" in run.replace(HEALTH_BODY, "").upper() for run in _runs(job))
        if job != "docker":
            assert not any(HEALTH_BODY in run for run in _runs(job)), job
    uncommented = [line for line in _text().splitlines() if not line.lstrip().startswith("#")]
    assert not any("SUREPASS" in line.replace(HEALTH_BODY, "").upper() for line in uncommented)


def test_the_documented_local_eval_command_is_the_one_ci_runs() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"uv run {LOCAL_EVALS}" in readme
    assert f"uv run --frozen {LOCAL_EVALS}" in "\n".join(_runs("evals-local"))


# --------------------------------------------------------------------------------------------------------- docker job


def _docker_commands() -> list[str]:
    """Every `docker ...` command of the docker job, one string each, with shell line continuations folded."""
    commands: list[str] = []
    for run in _runs("docker"):
        for line in re.sub(r"\\\n\s*", " ", run).splitlines():
            found = re.search(r"(?:^|[\s(])(docker\s.*)$", line)
            if found:
                commands.append(found.group(1))
    return commands


def test_the_docker_job_blocks_merges_and_needs_no_uv_python_or_secrets() -> None:
    job = _jobs()["docker"]
    assert job["runs-on"] == "ubuntu-latest"
    assert _needs("docker") == ["test"]
    assert not job.get("continue-on-error")  # a broken image must fail the build
    assert "secrets." not in str(job)
    steps = _steps("docker")
    assert not [s for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv@")]  # pure docker + curl + jq
    assert not [run for run in _runs("docker") if re.search(r"\buv\b", run)]
    assert str(steps[0]["uses"]).startswith("actions/checkout@")  # the build context is the repository root


def test_the_docker_job_builds_the_image_from_the_repository_root() -> None:
    builds = [c for c in _docker_commands() if c.startswith("docker build")]
    assert builds == ['docker build --tag "$IMAGE" .']
    env = cast("dict[str, str]", _jobs()["docker"]["env"])
    assert set(env) == {"IMAGE", "CONTAINER"}  # runner-side names only: nothing here reaches a container


def test_no_docker_run_in_ci_passes_an_environment() -> None:
    """The smoke test is the DEGRADED boot: an empty environment, so the live switch is off and nothing is configured."""
    runs = [c for c in _docker_commands() if c.startswith("docker run")]
    assert len(runs) >= 3  # the image inspection, the purge job's entrypoint and the long-lived container
    for command in runs:
        assert not re.search(r"(?:^|\s)(?:-e\S*|--env\S*)(?:\s|=|$)", command), command
        assert "SUREPASS" not in command.upper(), command


def test_the_smoke_test_asserts_the_health_body_and_the_readme_quick_check_example() -> None:
    joined = "\n".join(_runs("docker"))
    assert f"= '{HEALTH_BODY}'" in joined  # compared against jq -cS output, so key order and spacing cannot matter
    assert "curl --silent --show-error --fail --max-time 5 http://127.0.0.1:8080/health" in joined
    # the quick check posts the README section 2 example body, verbatim, and asserts HTTP 200 + LIVE_CALLS_DISABLED
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## 2. API") : readme.index("## 3. Configuration")]
    documented = re.search(r"-d '(\{[^']*\})'", section)
    posted = re.search(r"^\s*body='(\{[^']*\})'$", joined, re.MULTILINE)
    assert documented is not None and posted is not None
    assert json.loads(posted.group(1)) == json.loads(documented.group(1))
    assert "http://127.0.0.1:8080/checks/quick" in joined
    assert 'test "$status" = "200"' in joined
    assert '.code == "LIVE_CALLS_DISABLED"' in joined


def test_the_container_is_published_on_the_exposed_port_and_always_stopped() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^EXPOSE 8080$", dockerfile, re.MULTILINE)
    assert '--publish 127.0.0.1:8080:8080 "$IMAGE"' in "\n".join(_runs("docker"))
    steps = _steps("docker")
    stop = next(i for i, step in enumerate(steps) if "docker stop" in str(step.get("run", "")))
    assert steps[stop].get("if") == "always()"  # stopped (and removed) even when an assertion failed
    health = next(i for i, step in enumerate(steps) if "State.Health" in str(step.get("run", "")))
    smoke = next(i for i, step in enumerate(steps) if "/checks/quick" in str(step.get("run", "")))
    assert smoke < health < stop  # the image's own HEALTHCHECK is waited on after the API assertions
