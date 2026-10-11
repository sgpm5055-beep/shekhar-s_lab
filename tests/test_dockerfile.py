"""Production packaging: the Dockerfile and .dockerignore keep their rules, at parse level.

There is no Docker daemon on the machine this was written on, so nothing here builds an image. The `docker` job in
.github/workflows/ci.yml builds it and smoke-tests the result (tests/test_ci_workflow.py pins how); this file pins the
Dockerfile's own contract so an edit cannot quietly drop a rule: pinned base images, a locked production-only install, no
secrets or `.env` baked in, a numeric non-root user, the PORT convention, a curl-free HEALTHCHECK, and that nothing the
container runs imports from `tests/` (which is not in the image).
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DOCKERFILE: Final = REPO_ROOT / "Dockerfile"
DOCKERIGNORE: Final = REPO_ROOT / ".dockerignore"
PACKAGE: Final = "eligibility_agent"
APP: Final = "'eligibility_agent.main:app_factory' --factory"


# ------------------------------------------------------------------------------------------------------ Dockerfile parse


@dataclass(frozen=True)
class Instruction:
    keyword: str
    args: str
    stage: int  # index of the FROM that opened the stage this instruction belongs to


def _instructions() -> list[Instruction]:
    logical: list[str] = []
    pending = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or (not line and not pending):
            continue  # comment lines (also inside a continued instruction) and blank lines between instructions
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        logical.append((pending + line).strip())
        pending = ""
    assert not pending, "the Dockerfile ends inside a line continuation"
    parsed: list[Instruction] = []
    stage = -1
    for line in logical:
        keyword, _, args = line.partition(" ")
        if keyword.upper() == "FROM":
            stage += 1
        parsed.append(Instruction(keyword.upper(), args.strip(), stage))
    return parsed


def _of(keyword: str, stage: int | None = None) -> list[Instruction]:
    return [i for i in _instructions() if i.keyword == keyword and (stage is None or i.stage == stage)]


def _runs() -> list[str]:
    return [i.args for i in _of("RUN")]


def _text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_the_dockerfile_is_two_named_stages_builder_then_runtime() -> None:
    assert [f.args.rsplit(" AS ", 1)[-1] for f in _of("FROM")] == ["builder", "runtime"]


# ------------------------------------------------------------------------------------------------------------ pinning


def test_both_stages_use_the_same_python_image_pinned_to_a_patch_tag_and_a_digest() -> None:
    images: set[str] = set()
    for f in _of("FROM"):
        match = re.fullmatch(r"(python:(3\.12\.\d+)-slim-[a-z]+)@sha256:[0-9a-f]{64} AS \w+", f.args)
        assert match is not None, f.args  # exact patch tag AND a digest, never `3.12-slim` / `latest`
        images.add(f.args.rsplit(" AS ", 1)[0])
    assert len(images) == 1, images  # the venv's interpreter symlink and /app path need the same image in both stages
    pinned_minor = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert next(iter(images)).startswith(f"python:{pinned_minor}.")
    requires = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    assert requires == ">=3.12,<3.13"  # the image's 3.12.x satisfies it; a change here must revisit the base image


def test_uv_is_copied_from_a_pinned_image_and_only_in_the_builder() -> None:
    copies = [c for c in _of("COPY") if c.args.startswith("--from=ghcr.io/astral-sh/uv:")]
    assert len(copies) == 1
    assert re.fullmatch(r"--from=ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+@sha256:[0-9a-f]{64} /uv /uvx /bin/", copies[0].args)
    assert copies[0].stage == 0  # the runtime image carries no uv
    assert not [i for i in _of("RUN", 1) if re.search(r"\buv\b", i.args)]


def test_nothing_is_floating() -> None:
    text = _text()
    assert not re.search(r":latest\b", text)
    for ref in re.findall(r"^(?:FROM|COPY --from=)\s*(\S+)", text, re.MULTILINE):
        if ref != "builder":
            assert "@sha256:" in ref, ref


# -------------------------------------------------------------------------------------------------------- uv install


def test_every_uv_sync_is_locked_production_only_and_not_editable() -> None:
    syncs = [(i.stage, re.findall(r"uv sync[^&]*", i.args)) for i in _of("RUN")]
    flat = [(stage, cmd) for stage, cmds in syncs for cmd in cmds]
    assert len(flat) == 2  # the dependency layer, then the project
    for stage, cmd in flat:
        assert stage == 0, cmd
        assert "--locked" in cmd and "--frozen" not in cmd, cmd  # a stale uv.lock fails the build (like CI)
        assert "--no-dev" in cmd, cmd  # uv syncs the `dev` group by default
        assert "--no-editable" in cmd, cmd  # the venv is copied out of the builder: no .pth into /app
        assert "--all-groups" not in cmd and "--group" not in cmd, cmd
    assert "--no-install-project" in flat[0][1] and "--no-install-project" not in flat[1][1]


def test_the_dev_group_exists_so_that_no_dev_has_something_to_exclude() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = cast("list[str]", pyproject["dependency-groups"]["dev"])
    assert dev
    assert not [d for d in dev if re.match(r"(uvicorn|fastapi|langgraph)\b", d)]  # the runtime stays in `dependencies`
    assert "default-groups" not in pyproject.get("tool", {}).get("uv", {})  # otherwise `--no-dev` may not cover it


def test_the_dependency_layer_is_cached_apart_from_the_source() -> None:
    builder = [i for i in _instructions() if i.stage == 0]
    keywords = [(i.keyword, i.args) for i in builder if i.keyword in {"COPY", "RUN"} and "--from=" not in i.args]
    assert keywords == [
        ("COPY", "pyproject.toml uv.lock ./"),
        ("RUN", "uv sync --locked --no-dev --no-editable --no-install-project"),
        ("COPY", "eligibility_agent ./eligibility_agent"),
        ("RUN", "uv sync --locked --no-dev --no-editable"),
    ]


def test_uv_environment_is_set_for_a_container_build() -> None:
    env = " ".join(i.args for i in _of("ENV", 0))
    assert "UV_PYTHON_DOWNLOADS=never" in env  # use the image's interpreter, never fetch another
    assert "UV_COMPILE_BYTECODE=1" in env and "UV_LINK_MODE=copy" in env


# ------------------------------------------------------------------------------------------------------------ runtime


def test_the_runtime_stage_copies_the_venv_and_the_purge_job_but_not_the_source_tree() -> None:
    copies = [i.args for i in _of("COPY", 1)]
    assert copies == [
        "--from=builder /app/.venv /app/.venv",
        "scripts/__init__.py scripts/purge_raw.py ./scripts/",
    ]
    # uvicorn puts the working directory on sys.path: a source copy of the package there would shadow the installed one
    assert [i.args for i in _of("WORKDIR")] == ["/app", "/app"]
    env = " ".join(i.args for i in _of("ENV", 1))
    assert 'PATH="/app/.venv/bin:$PATH"' in env


def test_the_container_runs_as_a_numeric_non_root_user() -> None:
    users = _of("USER")
    assert len(users) == 1 and users[0].stage == 1
    assert re.fullmatch(r"[1-9]\d*:[1-9]\d*", users[0].args), users[0].args  # numeric, so runAsNonRoot is verifiable
    last_run = max(idx for idx, i in enumerate(_instructions()) if i.keyword == "RUN")
    assert [i.keyword for i in _instructions()].index("USER") > last_run  # every RUN happens as root, then drop


def _exec_form(args: str) -> list[str]:
    """The JSON array of an exec-form instruction. Docker silently falls back to SHELL form when the array is not valid
    JSON (single quotes, a trailing comma), which would change PID 1 and quoting without any build error."""
    parsed = cast("object", json.loads(args))
    assert isinstance(parsed, list)
    items = cast("list[object]", parsed)
    assert all(isinstance(item, str) for item in items)
    return cast("list[str]", items)


def test_the_port_and_the_start_command_follow_the_platform_convention() -> None:
    assert [i.args for i in _of("EXPOSE")] == ["8080"]
    cmds = _of("CMD")
    assert len(cmds) == 1 and not _of("ENTRYPOINT")
    command = _exec_form(cmds[0].args)
    # `sh -c` expands ${PORT:-8080}; `exec` makes uvicorn PID 1 so it gets the platform's SIGTERM
    assert command == ["sh", "-c", f"exec uvicorn {APP} --host 0.0.0.0 --port ${{PORT:-8080}}"]
    assert "uv run" not in command[2] and "--workers" not in command[2]  # the venv is on PATH; one worker per container
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"uv run uvicorn {APP} --host 0.0.0.0 --port 8080" in readme  # the documented dev command is the same app


def test_the_healthcheck_uses_python_urllib_against_health_and_fails_on_non_200() -> None:
    checks = _of("HEALTHCHECK")
    assert len(checks) == 1
    options, _, probe = checks[0].args.partition(" CMD ")
    assert re.fullmatch(r"--interval=\d+s --timeout=\d+s --start-period=\d+s --retries=\d+", options), options
    command = _exec_form(probe)
    assert command[:2] == ["python", "-c"] and len(command) == 3
    code = command[2]
    compile(code, "<healthcheck>", "exec")  # valid python
    assert "urllib.request.urlopen(" in code and "/health" in code and "127.0.0.1" in code
    assert ".status == 200" in code and "sys.exit(0 if" in code and "else 1)" in code  # anything but a 200 exits 1
    assert "os.environ.get('PORT') or '8080'" in code  # the same PORT convention as the start command
    assert not re.search(r"\b(curl|wget)\b", _text().replace("slim has no curl", ""))
    assert not [r for r in _runs() if re.search(r"apt-get|apk add|pip install", r)]  # nothing extra is installed


# ---------------------------------------------------------------------------------------------------- secrets / config


def test_no_secret_or_service_configuration_is_baked_into_the_image() -> None:
    for instruction in (*_of("ENV"), *_of("ARG")):
        assert not re.search(
            r"SUREPASS|TOKEN|SECRET|PASSWORD|API_KEY|HMAC|SUPABASE|OPENAI|LANGSMITH|DATABASE_URL|REPO_BACKEND|SERVICE_ROLE",
            instruction.args,
            re.IGNORECASE,
        ), instruction.args
    assert not [i for i in _of("COPY") if re.search(r"(^|\s)\.(\s|$)|\.env|\.pem|\.key", i.args)]  # no blanket / env copy
    assert not _of("ADD") and not _of("ARG")


# --------------------------------------------------------------------------------------------------- .dockerignore


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """Docker's .dockerignore glob (Go filepath.Match plus `**`) as a regex anchored at the build-context root."""
    out = ""
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**", i):
            i += 2
            if pattern.startswith("/", i):
                i += 1
                out += "(?:.*/)?"
            else:
                out += ".*"
            continue
        if char == "*":
            out += "[^/]*"
        elif char == "?":
            out += "[^/]"
        elif char == "[":
            end = pattern.index("]", i)
            out += pattern[i : end + 1]
            i = end
        else:
            out += re.escape(char)
        i += 1
    return re.compile(out + r"\Z")


def _ignore_rules() -> list[tuple[bool, re.Pattern[str]]]:
    rules: list[tuple[bool, re.Pattern[str]]] = []
    for raw in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        rules.append((negate, _pattern_regex(line.lstrip("!").strip("/"))))
    return rules


def _ignored(path: str) -> bool:
    """Whether the path is excluded from the build context: a pattern matches it or one of its parent directories, and
    the last matching rule wins (`!` re-includes)."""
    parts = path.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    verdict = False
    for negate, regex in _ignore_rules():
        if any(regex.match(candidate) for candidate in candidates):
            verdict = not negate
    return verdict


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".github/workflows/ci.yml",
        ".superpowers/sdd/ledger.md",
        ".env",
        ".env.example",
        ".env.production",
        "eligibility_agent/.env",
        "docs/PRD.md",
        "tests/helpers.py",
        "tests/test_dockerfile.py",
        "README.md",
        ".venv/bin/python",
        "eligibility_agent/__pycache__/main.cpython-312.pyc",
        "scripts/__pycache__/purge_raw.cpython-312.pyc",
        ".pytest_cache/README.md",
        "service-account-prod.json",
        "keys/signing.pem",
        ".DS_Store",
    ],
)
def test_the_dockerignore_keeps_secrets_scratch_docs_and_tests_out_of_the_context(path: str) -> None:
    assert _ignored(path), path


def _copy_sources() -> list[str]:
    sources: list[str] = []
    for copy in _of("COPY"):
        if copy.args.startswith("--from="):
            continue
        *srcs, _dest = copy.args.split()
        sources.extend(srcs)
    return sources


def test_every_copy_source_exists_and_is_not_ignored() -> None:
    sources = _copy_sources()
    assert sorted(sources) == [
        "eligibility_agent",
        "pyproject.toml",
        "scripts/__init__.py",
        "scripts/purge_raw.py",
        "uv.lock",
    ]
    for source in sources:
        path = REPO_ROOT / source
        assert path.exists(), source
        assert not _ignored(source), source
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                relative = file.relative_to(REPO_ROOT).as_posix()
                if file.is_file() and "__pycache__" not in relative:
                    assert not _ignored(relative), relative  # the whole package reaches the build, schema.sql included
    assert (REPO_ROOT / "eligibility_agent" / "py.typed").exists()
    assert not _ignored("scripts/seed_lenders.py")  # scripts/ is not excluded; only purge_raw is copied (see below)


def test_the_build_inputs_the_wheel_needs_are_all_present() -> None:
    """hatchling builds the wheel from pyproject.toml + the package directory (`packages = [...]`): no readme, license or
    other file is referenced, so none needs to reach the builder."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [PACKAGE]
    project = cast("dict[str, object]", pyproject["project"])
    assert not {"readme", "license", "license-files", "dynamic"} & set(project)


# ------------------------------------------------------------------------- nothing in the image imports from tests/


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _module_file(module: str) -> Path | None:
    base = REPO_ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _closure(entry: Path) -> Iterator[tuple[str, Path]]:
    """Every first-party module (eligibility_agent.*, scripts.*) reachable from `entry` by an import statement anywhere in
    the file (function-level imports included), plus the package `__init__` files importing it runs, as (module, file)."""
    seen: set[Path] = set()
    todo = [entry]
    while todo:
        file = todo.pop()
        if file in seen:
            continue
        seen.add(file)
        for module in sorted(_imported_modules(file)):
            if module.split(".")[0] not in {PACKAGE, "scripts"}:
                continue
            parts = module.split(".")
            for prefix in (".".join(parts[: n + 1]) for n in range(len(parts))):
                target = _module_file(prefix)
                if target is not None:
                    yield prefix, target
                    todo.append(target)


@pytest.mark.parametrize("entry", ["eligibility_agent/main.py", "scripts/purge_raw.py"])
def test_what_the_container_runs_imports_nothing_from_tests_or_the_evals(entry: str) -> None:
    """`tests/` is not in the image, and neither is anything that needs it: the service and the purge job must import
    cleanly without it. (scripts/seed_lenders.py and eligibility_agent/evals import tests.helpers: they run from a
    checkout.) pytest puts the repository root on sys.path, so the suite itself would never notice a regression."""
    files = {REPO_ROOT / entry, *(file for _, file in _closure(REPO_ROOT / entry))}
    assert len(files) > 3  # the walk actually followed imports (and the packages' __init__ files)
    for file in files:
        relative = file.relative_to(REPO_ROOT).as_posix()
        offending = {m for m in _imported_modules(file) if m.split(".")[0] == "tests"}
        assert not offending, f"{relative} imports {sorted(offending)}"
        assert not relative.startswith("eligibility_agent/evals/"), f"{entry} reaches {relative}"
    # and everything the purge job needs is copied into the image
    if entry == "scripts/purge_raw.py":
        scripts = {f.name for f in files if f.parent.name == "scripts"}
        assert scripts <= {"__init__.py", "purge_raw.py"}
