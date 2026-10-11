# Production image for the eligibility service (README "Deploy (container)").
#
# Two stages, written to uv's documented Docker pattern (uv 0.12.x, "Using uv in Docker"):
#   builder  installs the LOCKED, production-only dependency set plus the project (as a built wheel, not editable) into
#            /app/.venv. The dependency layer comes before the source copy, so it is rebuilt only when pyproject.toml
#            or uv.lock change.
#   runtime  a clean python image with just that virtualenv and scripts/purge_raw.py: no uv, no compiler, no tests, no
#            docs, no .env.
#
# Both stages use the SAME python image. The virtualenv's interpreter is a symlink into that image and its console
# scripts carry the absolute /app/.venv shebang, so the builder and runtime python and the /app path must not differ.
#
# Configuration is environment-only at runtime. Nothing secret is baked in: no .env, no tokens, no keys, no service
# settings, with ONE exception: LANGGRAPH_STRICT_MSGPACK=true, a safe-by-default flag (see the runtime ENV below). With an
# empty environment the service boots in degraded mode (in-memory repo, stub SurePass, no auth); see the README.
#
# There is no Docker daemon on the machine this was written on. The image is built and smoke-tested by the `docker` job
# in .github/workflows/ci.yml (tests/test_dockerfile.py pins this file at parse level).
#
# Base images are pinned to an exact tag AND the digest of its multi-arch index (the digest wins; the tag is for humans
# and for Dependabot / Renovate). Bump them together. .github/dependabot.yml watches the FROM lines weekly; the uv helper
# pin below lives in a `COPY --from=`, which some updaters do not parse, so check it by hand too (the README says to bump
# the digests regularly). The build backend (hatchling) and its dependencies are pinned in pyproject.toml, under
# [tool.uv] build-constraint-dependencies, because they are resolved at build time and are not in uv.lock.

# ---------------------------------------------------------------------------------------------------------- builder
FROM python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /bin/

# COMPILE_BYTECODE: faster cold start (the runtime user cannot write .pyc files next to read-only code).
# LINK_MODE=copy: the cache and the virtualenv are on different filesystems in a build, so hardlinks are not possible.
# NO_CACHE: nothing is reused between builds through uv's cache, so do not keep one in a layer.
# PYTHON_DOWNLOADS=never: use the image's own interpreter (3.12.x satisfies requires-python), never fetch another.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Layer 1: third-party dependencies only. `--locked` fails the build when uv.lock is stale against pyproject.toml.
# `--no-dev` omits the `dev` dependency group (pytest, pyright, pyyaml); uv syncs `dev` by default, so without the flag
# they would be installed. `--no-install-project` skips the service itself, whose source is not copied yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-editable --no-install-project

# Layer 2: the service, installed as a regular (non-editable) wheel into the same virtualenv.
COPY eligibility_agent ./eligibility_agent
RUN uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------------------------------------------------- runtime
FROM python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS runtime

# A fixed numeric UID so orchestrators that enforce runAsNonRoot can verify it. Code stays root-owned and read-only; the
# user only gets a writable HOME, in case a library caches something there.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app

# LANGGRAPH_STRICT_MSGPACK=true: with DATABASE_URL set the app REFUSES to start without it (CheckpointerConfigError), which
# on a platform such as Cloud Run reads as a container that "failed to listen on PORT". Strict msgpack only restricts
# checkpoint deserialization to an explicit allow-list, and there is no checkpointer while DATABASE_URL is unset (the
# degraded boot). A plain ENV, so the platform's environment can still override it.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANGGRAPH_STRICT_MSGPACK=true

WORKDIR /app

# The installed service and its dependencies. The source tree is deliberately NOT copied: uvicorn puts the working
# directory on sys.path, and a second copy of eligibility_agent/ there would shadow the installed one.
COPY --from=builder /app/.venv /app/.venv

# The purge job runs in the container (`python -m scripts.purge_raw`, e.g. as a scheduled job). Only the files it needs:
# scripts/seed_lenders.py imports tests.helpers, which is not in the image, so seeding runs from a repository checkout.
COPY scripts/__init__.py scripts/purge_raw.py ./scripts/

USER 10001:10001

EXPOSE 8080

# slim has no curl: python and urllib. The opener raises on any 4xx/5xx and a refused connection (exit 1); anything but a
# 200 also exits 1. ProxyHandler({}) bypasses http_proxy / no_proxy from the environment: a platform that injects a proxy
# would otherwise send this loopback probe to the proxy and report a healthy container as unhealthy. PORT follows the same
# convention as CMD.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; port = os.environ.get('PORT') or '8080'; opener = urllib.request.build_opener(urllib.request.ProxyHandler({})); sys.exit(0 if opener.open('http://127.0.0.1:' + port + '/health', timeout=3).status == 200 else 1)"]

# One uvicorn worker per container (scale by running more containers). `exec` makes uvicorn PID 1 so it receives SIGTERM
# directly and drains in-flight requests. ${PORT:-8080}: the platform's PORT (Cloud Run, Railway, Render) or 8080.
CMD ["sh", "-c", "exec uvicorn 'eligibility_agent.main:app_factory' --factory --host 0.0.0.0 --port ${PORT:-8080}"]
