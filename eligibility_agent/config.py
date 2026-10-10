"""Environment registry, kill switch and fixed per-call timeouts (PRD v1.1 sections 5.12, 12).

Single source of truth for every env var the service reads (``Settings``) and for the per-call timeouts
(module-level ``Final`` constants). Deliberately dependency-light: it imports nothing from ``eligibility_agent``
and never touches supabase (``db.repo.create_supabase_backend`` builds the client from these values), so any
layer can import it without cycles or heavy imports.

Degraded boot is a first-class state: every field is optional-with-default, so ``Settings()`` constructs from a
completely empty environment (stub SurePass, in-memory repo, auth disabled) and ``startup_errors()`` is ``[]``.
Misconfiguration is reported only for combinations that would be unsafe (supabase backend without secrets, live
SurePass without a token); the app factory raises ``ConfigError`` on a non-empty list.

Not Settings fields, on purpose:
  * per-call timeouts, which PRD 5.12 fixes (code constants below, not env);
  * ``LANGGRAPH_STRICT_MSGPACK``, which the checkpointer factory reads straight from ``os.environ``;
  * hashers and the OpenAI temperature (fixed in code where they are used).
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Final, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DEV_HMAC_KEY: Final[str] = "dev-only-hmac-key-do-not-use-in-prod"

# FIXED per-call timeouts (PRD 5.12): NOT env, NOT Settings fields. The SurePass client and the OpenAI factory
# import these; the connect phase is capped separately and the read timeout is per endpoint.
PAN_TIMEOUT_S: Final[float] = 8.0
SCORE_TIMEOUT_S: Final[float] = 12.0
REPORT_TIMEOUT_S: Final[float] = 20.0
OPENAI_TIMEOUT_S: Final[float] = 8.0
CONNECT_TIMEOUT_S: Final[float] = 5.0


class ConfigError(RuntimeError):
    """Raised by the app factory when ``Settings.startup_errors()`` is non-empty."""


def utcnow() -> datetime:
    """Timezone-aware current UTC time (the one clock call sites use, so tests can patch it)."""
    return datetime.now(timezone.utc)


class Settings(BaseSettings):
    """Every env var the service reads. Init kwargs and env both use the env name (the ``validation_alias``)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # SurePass. Live iff SUREPASS_LIVE is EXACTLY "true" (see ``surepass_live``); anything else is the stub.
    surepass_live_flag: str = Field(default="", validation_alias="SUREPASS_LIVE")
    surepass_token: str | None = Field(default=None, validation_alias="SUREPASS_TOKEN")
    surepass_base_url: str = Field(default="https://kyc-api.surepass.app", validation_alias="SUREPASS_BASE_URL")

    # HTTP API auth: unset => auth disabled (dev only). Required when REPO_BACKEND=supabase.
    service_api_token: str | None = Field(default=None, validation_alias="SERVICE_API_TOKEN")

    # HMAC-SHA256 key for PAN + mobile identity keys, held outside the DB. The default is dev-only.
    identity_hmac_key: str = Field(default=DEFAULT_DEV_HMAC_KEY, validation_alias="IDENTITY_HMAC_KEY")

    # Persistence.
    repo_backend: Literal["memory", "supabase"] = Field(default="memory", validation_alias="REPO_BACKEND")
    supabase_url: str | None = Field(default=None, validation_alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(default=None, validation_alias="SUPABASE_SERVICE_ROLE_KEY")
    raw_bucket: str = Field(default="raw-responses", validation_alias="RAW_BUCKET")
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")  # checkpointer DSN

    # OpenAI (temperature is fixed at 0 where the model is built, not configurable).
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1", validation_alias="OPENAI_MODEL")

    # LangSmith: tracing is on iff LANGSMITH_TRACING is EXACTLY "true" (see ``langsmith_tracing``).
    langsmith_tracing_flag: str = Field(default="", validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="eligibility-agent", validation_alias="LANGSMITH_PROJECT")

    # Prompts.
    prompt_source: Literal["local", "hub"] = Field(default="local", validation_alias="PROMPT_SOURCE")
    prompt_tag: str | None = Field(default=None, validation_alias="PROMPT_TAG")

    # Runtime.
    graph_timeout_seconds: float = Field(default=90.0, validation_alias="GRAPH_TIMEOUT_SECONDS")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def surepass_live(self) -> bool:
        """The kill switch: True only for the exact lowercase literal ``"true"``."""
        return self.surepass_live_flag == "true"

    @property
    def langsmith_tracing(self) -> bool:
        return self.langsmith_tracing_flag == "true"

    @property
    def auth_enabled(self) -> bool:
        """A blank ``SERVICE_API_TOKEN=`` counts as unset."""
        return bool(self.service_api_token)

    def startup_errors(self) -> list[str]:
        """Hard misconfigurations only; an empty (degraded-boot) environment yields ``[]``.

        Blank values (``VAR=`` left in a ``.env``) count as missing.
        """
        errors: list[str] = []
        if self.repo_backend == "supabase":
            if not (self.supabase_url and self.supabase_service_role_key):
                errors.append("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when REPO_BACKEND=supabase")
            if self.identity_hmac_key in ("", DEFAULT_DEV_HMAC_KEY):
                errors.append("IDENTITY_HMAC_KEY must be set to a non-default value when REPO_BACKEND=supabase")
            if not self.service_api_token:
                errors.append("SERVICE_API_TOKEN is required when REPO_BACKEND=supabase")
        if self.surepass_live and not self.surepass_token:
            errors.append("SUREPASS_TOKEN is required when SUREPASS_LIVE=true")
        return errors


@lru_cache
def get_settings() -> Settings:
    return Settings()
