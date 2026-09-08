from mobile_world.agents.implementations.general_e2e_agent import (
    parse_action,
    parse_response_to_action,
)


def test_parse_action_with_thinking_tag() -> None:
    thought, action = parse_action(
        """
<thinking>
Tap the search field.
</thinking>
Action: {"action": "tap", "point": [500, 250]}
""".strip()
    )

    assert thought == "Tap the search field."
    assert action == '{"action": "tap", "point": [500, 250]}'
    assert parse_response_to_action(action, 1000, 2000) == {
        "action_type": "click",
        "x": 500,
        "y": 500,
    }


def test_parse_action_keeps_legacy_thought_format() -> None:
    thought, action = parse_action('Thought: Wait for loading.\nAction: {"action_type": "wait"}')

    assert thought == "Wait for loading."
    assert parse_response_to_action(action, 1000, 2000) == {"action_type": "wait"}


def test_parse_action_accepts_bare_json() -> None:
    thought, action = parse_action('{"type": "input_text", "text": "hello"}')

    assert thought == ""
    assert parse_response_to_action(action, 1000, 2000) == {
        "action_type": "input_text",
        "text": "hello",
    }


def test_parse_response_accepts_nested_action_and_xy_fields() -> None:
    action = """
```json
{"action": {"type": "click", "x": "250", "y": "500"}}
```
"""

    assert parse_response_to_action(action, 1080, 2400) == {
        "action_type": "click",
        "x": 270,
        "y": 1200,
    }


def test_parse_response_treats_swipe_with_points_as_drag() -> None:
    action = '{"action_type": "swipe", "start": [100, 200], "end": [100, 800]}'

    assert parse_response_to_action(action, 1000, 2000) == {
        "action_type": "drag",
        "start_x": 100,
        "start_y": 400,
        "end_x": 100,
        "end_y": 1600,
    }


def test_parse_response_accepts_common_field_aliases() -> None:
    assert parse_response_to_action('{"action": "type", "content": "hello"}', 1000, 2000) == {
        "action_type": "input_text",
        "text": "hello",
    }
    assert parse_response_to_action(
        '{"action_type": "open_app", "app": "Maps"}', 1000, 2000
    ) == {
        "action_type": "open_app",
        "app_name": "Maps",
    }
