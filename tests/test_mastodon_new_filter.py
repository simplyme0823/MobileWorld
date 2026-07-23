from datetime import UTC, datetime, timedelta

import pytest

from mobile_world.tasks.definitions.mastodon.mastodon_new_filter import (
    _is_valid_expiry_duration,
)


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (timedelta(days=4), False),
        (timedelta(days=4, seconds=1), True),
        (timedelta(days=4, hours=12), True),
        (timedelta(days=5), True),
        (timedelta(days=5, seconds=1), False),
    ],
)
def test_valid_expiry_duration_uses_calendar_day_window(
    duration: timedelta, expected: bool
) -> None:
    created_at = datetime(2026, 7, 23, tzinfo=UTC)

    assert _is_valid_expiry_duration(created_at + duration, created_at, 5) is expected
