import pytest

from mobile_world.runtime.app_helpers import mastodon
from mobile_world.runtime.utils.helpers import AdbResponse


def test_compose_env_uses_emulator_clock_offset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KEEP_ME", "yes")
    monkeypatch.setattr(
        mastodon,
        "execute_adb",
        lambda _command: AdbResponse(success=True, output="1000\n"),
    )
    monkeypatch.setattr(mastodon.time, "time", lambda: 1600.9)

    env = mastodon._compose_env_with_device_time()

    assert env["MASTODON_TIME_OFFSET_SECONDS"] == "-600"
    assert env["KEEP_ME"] == "yes"


def test_compose_env_rejects_failed_device_time(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mastodon,
        "execute_adb",
        lambda _command: AdbResponse(success=False, error="device offline"),
    )

    with pytest.raises(RuntimeError, match="device offline"):
        mastodon._compose_env_with_device_time()


def test_compose_env_rejects_invalid_device_epoch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mastodon,
        "execute_adb",
        lambda _command: AdbResponse(success=True, output="not-a-timestamp"),
    )

    with pytest.raises(RuntimeError, match="Invalid emulator epoch"):
        mastodon._compose_env_with_device_time()
