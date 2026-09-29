"""Who is asking: every request resolves to a viewer id, used to decide which uploads it may see.

- Web: an anonymous, signed session cookie (no sign-up). Its value is `<id>.<HMAC>`, so a client can't forge
  someone else's id; the id itself is random (128 bits), so it can't be guessed.
- Trusted services (the Telegram bot): a shared service token plus the user id they vouch for, e.g. `tg:12345`.
- Neither: an anonymous viewer that sees only the shared library.

Separately, the admin token (header x-admin-token) marks operators and their scripts: admin endpoints, uploads to
the shared library, no limits.
"""

import hashlib
import hmac
import secrets

from fastapi import Request, Response

from src.config import Settings, get_settings
from src.services.limits import Principal, acting_for

COOKIE = "vr_session"
SERVICE_TOKEN_HEADER = "x-service-token"
USER_HEADER = "x-user-id"
ADMIN_TOKEN_HEADER = "x-admin-token"


def _matches(sent: str | None, expected: str) -> bool:
    return bool(sent and expected) and hmac.compare_digest(sent.encode(), expected.encode())


def is_admin(request: Request, settings: Settings) -> bool:
    return _matches(request.headers.get(ADMIN_TOKEN_HEADER), settings.admin_token)


def is_service(request: Request, settings: Settings) -> bool:
    return _matches(request.headers.get(SERVICE_TOKEN_HEADER), settings.service_token)


def sign(value: str, secret: str) -> str:
    mac = hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{value}.{mac}"


def unsign(token: str | None, secret: str) -> str | None:
    """The value inside a signed token, or None if it's missing or tampered with."""
    if not token or "." not in token:
        return None
    value, _, _ = token.rpartition(".")
    return value if hmac.compare_digest(sign(value, secret), token) else None


def viewer_from_request(request: Request, settings: Settings) -> str | None:
    if is_service(request, settings):
        user = (request.headers.get(USER_HEADER) or "").strip()
        return user[:64] or None
    session = unsign(request.cookies.get(COOKIE), settings.session_secret)
    return f"web:{session}" if session else None


def ensure_viewer(request: Request, response: Response, settings: Settings) -> str:
    """The viewer, creating an anonymous web session (and setting its cookie) if there isn't one yet."""
    viewer = viewer_from_request(request, settings)
    if viewer:
        return viewer
    session = secrets.token_hex(16)
    response.set_cookie(
        COOKIE,
        sign(session, settings.session_secret),
        max_age=settings.session_max_age_sec,
        httponly=True,  # page scripts can't read it: an XSS can't steal it
        samesite="lax",
        secure=settings.cookie_secure,  # HTTPS only in production
    )
    return f"web:{session}"


class PrincipalMiddleware:
    """Starts each request's `Principal` (limits.py): the client address and whether it's an admin. The viewer is
    added once identity is resolved (the get_viewer dependency), in the same object."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = getattr(scope["app"].state, "settings", None) or get_settings()
        request = Request(scope)
        trusted = is_service(request, settings)
        ip = None if trusted else (scope.get("client") or (None,))[0]  # the bot: one address for all its users
        with acting_for(Principal(ip=ip, exempt=is_admin(request, settings))):
            await self.app(scope, receive, send)
