"""Secret handling. Secrets come ONLY from environment variables (optionally loaded from a
git-ignored ``.env`` file). They are never stored in config, never logged, never persisted."""
from __future__ import annotations

import os
import re
from pathlib import Path

_SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE)", re.IGNORECASE)


class Secret:
    """Opaque wrapper: repr/str never reveal the value."""

    __slots__ = ("_value", "name")

    def __init__(self, name: str, value: str):
        self.name = name
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"Secret({self.name}=***)"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)


def load_dotenv(path: str | Path) -> list[str]:
    """Minimal .env loader. Does not override variables already set. Returns names loaded."""
    p = Path(path)
    loaded: list[str] = []
    if not p.is_file():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ and value:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def get_secret(env_name: str | None) -> Secret | None:
    if not env_name:
        return None
    value = os.environ.get(env_name, "").strip()
    return Secret(env_name, value) if value else None


def secret_values() -> list[str]:
    """Values of all secret-looking environment variables (used by the log redaction filter)."""
    return [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and v and len(v) >= 8]


def redact(text: str) -> str:
    for value in secret_values():
        if value in text:
            text = text.replace(value, "***REDACTED***")
    return text
