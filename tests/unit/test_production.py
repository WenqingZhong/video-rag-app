import pytest

from src.config import Settings, check_production, unsafe_for_production
from src.services.storage.factory import _make_s3

SAFE = {
    "session_secret": "s" * 48,
    "service_token": "t" * 48,
    "admin_token": "a" * 48,
    "cookie_secure": True,
    "debug": False,
}


def test_development_defaults_are_refused_in_production():
    settings = Settings(_env_file=None, environment="production")
    problems = unsafe_for_production(settings)
    assert len(problems) == 5  # three placeholder secrets, insecure cookie, debug
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        check_production(settings)


def test_proper_production_settings_start():
    check_production(Settings(_env_file=None, environment="production", **SAFE))
    check_production(Settings(_env_file=None))  # development: never checked


def test_a_short_secret_is_refused():
    settings = Settings(_env_file=None, environment="production", **{**SAFE, "admin_token": "short"})
    assert unsafe_for_production(settings) == ["ADMIN_TOKEN must be a random value of at least 32 characters"]


def test_real_s3_uses_the_iam_role_and_virtual_host_urls():
    s3 = _make_s3(Settings(_env_file=None, s3_access_key="", s3_secret_key="", s3_endpoint_url=""), None)
    assert s3.meta.config.s3["addressing_style"] == "virtual"
    local = _make_s3(Settings(_env_file=None), "http://localhost:8333")
    assert local.meta.config.s3["addressing_style"] == "path"
