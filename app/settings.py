from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_ENV_KEYS_TO_NORMALIZE = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MODEL_NAME",
    "OPENAI_COMPLEX_MODEL",
    "ANTHROPIC_MODEL",
    "LOG_LEVEL",
    "AI_ENGINE_DB_PATH",
    "REMOTE_BACKEND_URL",
    "REMOTE_API_TOKEN",
    "REMOTE_API_TIMEOUT_SECONDS",
    "REMOTE_GAMIFICATION_CACHE_SECONDS",
    "TEACHING_MAX_TOKENS",
    "MATERIALS_MAX_TOKENS",
    "CHAT_HISTORY_TURNS",
    "RUBRIC_ADMIN_KEY",
    "CORS_ALLOWED_ORIGINS",
)
_ENV_LOADED = False
_LOGGING_CONFIGURED = False


def _normalize_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def load_environment() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(dotenv_path=_ENV_PATH, override=False)
        _ENV_LOADED = True

    for key in _ENV_KEYS_TO_NORMALIZE:
        normalized = _normalize_env_value(os.getenv(key))
        if normalized is not None:
            os.environ[key] = normalized


def env_str(name: str, default: str = "") -> str:
    load_environment()
    value = _normalize_env_value(os.getenv(name))
    if value in (None, ""):
        return default
    return value


def env_path(name: str, default: Path) -> Path:
    load_environment()
    raw = env_str(name, str(default))
    return Path(raw)


def env_float(name: str, default: float) -> float:
    load_environment()
    raw = env_str(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    load_environment()
    raw = env_str(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    anthropic_api_key: str
    openai_default_model: str
    openai_complex_model: str
    anthropic_model: str
    log_level: str
    ai_engine_db_path: Path
    remote_backend_url: str
    remote_api_token: str
    remote_api_timeout_seconds: float
    remote_gamification_cache_seconds: int
    teaching_max_tokens: int
    materials_max_tokens: int
    chat_history_turns: int
    rubric_admin_key: str
    cors_allowed_origins: tuple[str, ...]


def env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = env_str(name, "")
    if not raw:
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


def get_settings() -> Settings:
    load_environment()
    default_db = Path(__file__).resolve().parent.parent / "data" / "ai_engine.db"
    return Settings(
        openai_api_key=env_str("OPENAI_API_KEY"),
        anthropic_api_key=env_str("ANTHROPIC_API_KEY"),
        openai_default_model=env_str("MODEL_NAME", "claude-sonnet-4-6"),
        openai_complex_model=env_str("OPENAI_COMPLEX_MODEL", "claude-sonnet-4-6"),
        anthropic_model=env_str("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        log_level=env_str("LOG_LEVEL", "INFO").upper(),
        ai_engine_db_path=env_path("AI_ENGINE_DB_PATH", default_db),
        remote_backend_url=env_str("REMOTE_BACKEND_URL", "https://api.ikonskills.ac").rstrip("/"),
        remote_api_token=env_str("REMOTE_API_TOKEN"),
        remote_api_timeout_seconds=max(5.0, env_float("REMOTE_API_TIMEOUT_SECONDS", 12.0)),
        remote_gamification_cache_seconds=max(5, env_int("REMOTE_GAMIFICATION_CACHE_SECONDS", 30)),
        teaching_max_tokens=max(512, env_int("TEACHING_MAX_TOKENS", 1800)),
        materials_max_tokens=max(512, env_int("MATERIALS_MAX_TOKENS", 2200)),
        chat_history_turns=max(4, env_int("CHAT_HISTORY_TURNS", 8)),
        rubric_admin_key=env_str("RUBRIC_ADMIN_KEY"),
        cors_allowed_origins=env_list(
            "CORS_ALLOWED_ORIGINS",
            (
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://54.151.241.98",
                "https://www.ikonskills.ac",
                "https://ikonskills.ac",
                "https://ikonskills.ac/",
            ),
        ),
    )


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    import sys

    if sys.stdout and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            if sys.stderr:
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(level=level)
    # Keep dependency transport logs quiet in normal runs (especially CI tests).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True


settings = get_settings()
