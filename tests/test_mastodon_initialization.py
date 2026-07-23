import json
from types import SimpleNamespace

import pytest

from mobile_world.runtime.app_helpers import mastodon, mastodon_gate
from mobile_world.runtime.app_helpers.mastodon_gate import (
    MASTODON_INIT_ERROR_CODE,
    MastodonInitializationError,
)
from mobile_world.runtime.client import (
    AndroidEnvClient,
    EnvironmentInitializationError,
)
from mobile_world.tasks.base import BaseTask


class _SwallowingMastodonTask(BaseTask):
    app_names = {"Mastodon"}
    goal = "test"
    snapshot_tag = None
    start_on_home_screen = False

    def initialize_task_hook(self, _controller) -> bool:
        mastodon_gate.set_last_initialization_error(
            MastodonInitializationError(
                task_name=self.name,
                attempts=3,
                reason="Rails API timeout",
            )
        )
        return False

    def initialize_user_agent_hook(self, _controller) -> None:
        return None


def test_initialize_attempt_runs_full_reset_and_both_readiness_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []

    monkeypatch.setattr(
        mastodon,
        "stop_mastodon_backend",
        lambda: operations.append("down") or True,
    )
    monkeypatch.setattr(
        mastodon,
        "_restore_mastodon_seed",
        lambda _seed: operations.append("restore") or {},
    )
    monkeypatch.setattr(
        mastodon,
        "_start_mastodon_compose",
        lambda: operations.append("up") or {},
    )
    monkeypatch.setattr(
        mastodon,
        "_wait_for_gate",
        lambda _attempt, gate, _check, _deadline: operations.append(gate),
    )

    mastodon._initialize_mastodon_attempt("/seed", attempt=2)

    assert operations == [
        "down",
        "restore",
        "up",
        "rails_api",
        "sidekiq_healthy",
    ]


def test_stop_backend_always_runs_compose_down_for_existing_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls: list[tuple[list[str], dict]] = []
    removed_paths: list[str] = []

    monkeypatch.setattr(mastodon.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        mastodon.subprocess,
        "run",
        lambda command, **kwargs: run_calls.append((command, kwargs))
        or SimpleNamespace(stdout="", stderr=""),
    )
    monkeypatch.setattr(
        mastodon.shutil,
        "rmtree",
        lambda path, **_kwargs: removed_paths.append(path),
    )

    assert mastodon.stop_mastodon_backend() is True
    assert run_calls[0][0] == [
        "docker",
        "compose",
        "down",
        "--remove-orphans",
    ]
    assert run_calls[0][1]["timeout"] == mastodon.MASTODON_INIT_TIMEOUT_SECONDS
    assert removed_paths == [mastodon.MASTODON_DOCKER_DIR]


def test_base_task_restores_structured_error_swallowed_by_task_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mobile_world.tasks.base.mattermost.stop_mattermost_backend",
        lambda: True,
    )
    monkeypatch.setattr(
        "mobile_world.tasks.base.mastodon.stop_mastodon_backend",
        lambda: True,
    )
    monkeypatch.setattr("mobile_world.tasks.base.clear_config", lambda: None)
    monkeypatch.setattr(
        "mobile_world.tasks.base.clear_callback_files",
        lambda _device: None,
    )
    controller = type("Controller", (), {"device": "emulator-5554"})()

    with pytest.raises(MastodonInitializationError) as error_info:
        _SwallowingMastodonTask().initialize_task(controller)

    assert error_info.value.task_name == "_SwallowingMastodonTask"


def test_client_preserves_structured_mastodon_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(AndroidEnvClient)
    client._initialized = True
    client.base_url = "http://mobile-world.test"
    client.device = "emulator-5554"
    response = SimpleNamespace(
        status_code=503,
        json=lambda: {
            "detail": {
                "code": MASTODON_INIT_ERROR_CODE,
                "task": "MastodonMallShareOrderTask",
                "attempts": 3,
                "reason": "Rails API timeout",
            }
        },
    )
    monkeypatch.setattr(
        "mobile_world.runtime.client.requests.post",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(EnvironmentInitializationError) as error_info:
        client.initialize_task("MastodonMallShareOrderTask")

    assert error_info.value.code == MASTODON_INIT_ERROR_CODE
    assert error_info.value.to_dict()["details"]["attempts"] == 3


def test_start_mastodon_backend_retries_with_a_full_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    attempts: list[int] = []

    def initialize_attempt(*, mastodon_backend_status_dir: str, attempt: int) -> None:
        assert mastodon_backend_status_dir == "/seed"
        attempts.append(attempt)
        if attempt < 3:
            raise RuntimeError(f"attempt {attempt} failed")

    monkeypatch.setattr(mastodon, "_initialize_mastodon_attempt", initialize_attempt)
    monkeypatch.setattr(mastodon, "_collect_mastodon_diagnostics", lambda: {})
    monkeypatch.setattr(mastodon, "stop_mastodon_backend", lambda: True)
    monkeypatch.setattr(mastodon.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mastodon_gate, "MASTODON_GATE_LOG_PATH", tmp_path / "gate.jsonl")

    with mastodon_gate.initialization_context("MastodonPostEditedPhotoTask"):
        assert mastodon.start_mastodon_backend("/seed") is True

    assert attempts == [1, 2, 3]
    records = [
        json.loads(line)
        for line in (tmp_path / "gate.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [
        record["attempt"]
        for record in records
        if record["event"] == "initialization_attempt_started"
    ] == [
        1,
        2,
        3,
    ]
    assert records[-1]["event"] == "initialization_succeeded"
    assert records[-1]["task"] == "MastodonPostEditedPhotoTask"
    assert records[-1]["container"]


def test_start_mastodon_backend_raises_structured_error_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    attempts: list[int] = []

    def initialize_attempt(*, mastodon_backend_status_dir: str, attempt: int) -> None:
        assert mastodon_backend_status_dir == "/seed"
        attempts.append(attempt)
        raise RuntimeError("sidekiq did not become ready")

    monkeypatch.setattr(mastodon, "_initialize_mastodon_attempt", initialize_attempt)
    monkeypatch.setattr(
        mastodon,
        "_collect_mastodon_diagnostics",
        lambda: {"composePs": {"stdout": "[]"}},
    )
    monkeypatch.setattr(mastodon, "stop_mastodon_backend", lambda: True)
    monkeypatch.setattr(mastodon.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mastodon_gate, "MASTODON_GATE_LOG_PATH", tmp_path / "gate.jsonl")

    with (
        mastodon_gate.initialization_context("MastodonMallShareOrderTask"),
        pytest.raises(MastodonInitializationError) as error_info,
    ):
        mastodon.start_mastodon_backend("/seed")

    assert attempts == [1, 2, 3]
    assert error_info.value.to_dict() == {
        "code": MASTODON_INIT_ERROR_CODE,
        "task": "MastodonMallShareOrderTask",
        "attempts": 3,
        "reason": "sidekiq did not become ready",
    }
    records = [
        json.loads(line)
        for line in (tmp_path / "gate.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed_attempts = [
        record for record in records if record["event"] == "initialization_attempt_failed"
    ]
    assert len(failed_attempts) == 3
    assert all("diagnostics" in record for record in failed_attempts)
    assert records[-1]["event"] == "initialization_failed"
    assert records[-1]["code"] == MASTODON_INIT_ERROR_CODE


@pytest.mark.parametrize(
    ("output", "expected_names"),
    [
        (
            '[{"Name":"mastodon-web-1"},{"Name":"mastodon-sidekiq-1"}]',
            ["mastodon-web-1", "mastodon-sidekiq-1"],
        ),
        (
            '{"Name":"mastodon-web-1"}\n{"Name":"mastodon-sidekiq-1"}\n',
            ["mastodon-web-1", "mastodon-sidekiq-1"],
        ),
        ("not-json", []),
    ],
)
def test_extract_compose_container_names_supports_compose_json_formats(
    output: str,
    expected_names: list[str],
) -> None:
    assert mastodon._extract_compose_container_names(output) == expected_names
