"""Building cache keys that change whenever the cached result could."""

import hashlib
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]  # src/


def normalize_request(text: str) -> str:
    """Case and spacing don't change what a request means: "A dog " and "a  dog" share an entry."""
    return " ".join(text.lower().split())


def digest(*parts: object, length: int = 16) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:length]


def code_fingerprint(*packages: str, settings: dict | None = None) -> str:
    """Hash of the source files of these packages (e.g. "services/understanding") and of relevant settings.

    Any edit to the prompt, the guards or the search code changes the fingerprint, so entries written by older
    code are never read again (they expire by TTL). No version number to remember to bump.
    """
    hasher = hashlib.sha256()
    for package in sorted(packages):
        for path in sorted((SRC / package).rglob("*.py")):
            hasher.update(str(path.relative_to(SRC)).encode())
            hasher.update(path.read_bytes())
    hasher.update(json.dumps(settings or {}, sort_keys=True, default=str).encode())
    return hasher.hexdigest()[:12]
