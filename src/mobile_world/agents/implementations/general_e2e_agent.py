import json
import os
import re
import time
from typing import Any

from loguru import logger

from mobile_world.agents.base import MCPAgent
from mobile_world.agents.utils.helpers import pil_adaptive_resize, pil_to_base64
from mobile_world.agents.utils.prompts import GENERAL_E2E_PROMPT_TEMPLATE
from mobile_world.runtime.utils.helpers import mask_api_key, pretty_print_messages
from mobile_world.runtime.utils.models import JSONAction
from mobile_world.runtime.utils.parsers import parse_json_markdown

ACTION_ALIASES = {
    "click": ["tap", "press", "touch"],
    "double_tap": ["double tap", "double_click", "double click"],
    "long_press": ["long tap", "long press", "hold"],
    "input_text": ["type", "enter_text", "write", "enter"],
    "scroll": ["swipe", "fling"],
    "keyboard_enter": ["enter"],
    "navigate_back": ["back", "go_back", "press_back"],
    "navigate_home": ["home", "press_home"],
    "open_app": ["open", "launch", "launch_app"],
    "finished": ["finish", "done", "terminate"],
}
NORMALIZED_ACTION_MAP = {}
for standard_action, aliases in ACTION_ALIASES.items():
    NORMALIZED_ACTION_MAP[standard_action] = standard_action
    for alias in aliases:
        NORMALIZED_ACTION_MAP[alias.replace(" ", "_")] = standard_action
        NORMALIZED_ACTION_MAP[alias] = standard_action

CLAUDE_IMAGE_SIZE = (1280, 720)
CLAUDE_OPUS_MAX_DIMENSION = 1280
ACTION_MARKER_RE = re.compile(r"(?i)\bAction\s*[:：]\s*")
THINKING_TAG_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.IGNORECASE | re.DOTALL)
ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
JSON_START_RE = re.compile(r"[\[{]")

POINT_KEYS = ("coordinate", "coordinates", "point", "position", "target_coordinate")
START_POINT_KEYS = ("start_coordinate", "start_coordinates", "start_point", "start", "from")
END_POINT_KEYS = ("end_coordinate", "end_coordinates", "end_point", "end", "to")
PARAM_KEYS = ("params", "parameters", "args", "arguments")
TEXT_KEYS = ("text", "content", "value", "answer", "message")
APP_NAME_KEYS = ("app_name", "app", "application", "package", "package_name", "name")


def normalize_action_type(action_type: str | None) -> str | None:
    if not action_type:
        return None
    processed_type = action_type.lower().strip().replace(" ", "_")
    return NORMALIZED_ACTION_MAP.get(processed_type, action_type)


def _find_json_start(text: str) -> int | None:
    match = JSON_START_RE.search(text)
    return match.start() if match else None


def _extract_action_payload(text: str) -> str:
    """Return the JSON-like action payload from common LLM wrappers."""
    payload = text.strip()
    action_tag = ACTION_TAG_RE.search(payload)
    if action_tag:
        payload = action_tag.group(1).strip()

    json_start = _find_json_start(payload)
    if json_start is not None and json_start > 0:
        payload = payload[json_start:].strip()

    return payload


def _extract_thought(text: str, action_start: int | None = None) -> str:
    prefix = text[:action_start].strip() if action_start is not None else text.strip()

    thinking_match = THINKING_TAG_RE.search(prefix) or THINKING_TAG_RE.search(text)
    if thinking_match:
        return thinking_match.group(1).strip()

    prefix = re.sub(r"^\s*Thought\s*[:：]\s*", "", prefix, flags=re.IGNORECASE)
    return prefix.strip()


def parse_action(plan_output: str) -> tuple[str, str]:
    """
    Parse the Thought and Action from agent output.

    Expected formats:
    <thinking>[analysis]</thinking>
    Action: [json_action]

    Legacy Thought:/Action: and bare JSON action outputs are also accepted.

    Args:
        plan_output: Raw output from agent

    Returns:
        Tuple of (thought, action)
    """
    try:
        action_matches = list(ACTION_MARKER_RE.finditer(plan_output))
        if action_matches:
            action_marker = action_matches[-1]
            thought = _extract_thought(plan_output, action_marker.start())
            action = plan_output[action_marker.end() :].strip()
        elif ACTION_TAG_RE.search(plan_output) or _find_json_start(plan_output) is not None:
            thought = _extract_thought(plan_output, _find_json_start(plan_output))
            action = _extract_action_payload(plan_output)
        else:
            raise ValueError("Expected 'Action:' or a JSON action payload in the output")

        action = _extract_action_payload(action)
        if not action:
            raise ValueError("Action payload is empty")

        return thought, action

    except Exception as e:
        logger.error(f"Error parsing output: {e}")
        logger.debug(f"Output: {plan_output}")
        raise ValueError(f"Output is not in the correct format: {e}")


def _coerce_number(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip()
    return float(value)


def _coerce_point(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return [_coerce_number(value["x"]), _coerce_number(value["y"])]
        for key in POINT_KEYS:
            if key in value:
                point = _coerce_point(value[key])
                if point is not None:
                    return point
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if isinstance(value[0], (list, tuple, dict)):
            return None
        return [_coerce_number(value[0]), _coerce_number(value[1])]

    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 2:
            return [_coerce_number(numbers[0]), _coerce_number(numbers[1])]

    return None


def _get_point(action_data: dict[str, Any], keys: tuple[str, ...]) -> list[float] | None:
    for key in keys:
        if key in action_data:
            point = _coerce_point(action_data[key])
            if point is not None:
                return point

    if keys == POINT_KEYS and "x" in action_data and "y" in action_data:
        return [_coerce_number(action_data["x"]), _coerce_number(action_data["y"])]

    return None


def _get_drag_points(action_data: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    if "points" in action_data and isinstance(action_data["points"], (list, tuple)):
        points = action_data["points"]
        if len(points) >= 2:
            start = _coerce_point(points[0])
            end = _coerce_point(points[1])
            if start is not None and end is not None:
                return start, end

    start = _get_point(action_data, START_POINT_KEYS)
    end = _get_point(action_data, END_POINT_KEYS)
    if start is not None and end is not None:
        return start, end

    if all(key in action_data for key in ("start_x", "start_y", "end_x", "end_y")):
        return (
            [_coerce_number(action_data["start_x"]), _coerce_number(action_data["start_y"])],
            [_coerce_number(action_data["end_x"]), _coerce_number(action_data["end_y"])],
        )

    return None


def _to_absolute_point(
    point: list[float],
    image_width: int,
    image_height: int,
    scale_factor_x: int,
    scale_factor_y: int,
) -> tuple[int, int]:
    absolute_x = int(point[0] * image_width / scale_factor_x)
    absolute_y = int(point[1] * image_height / scale_factor_y)
    return absolute_x, absolute_y


def _merge_parameter_block(action_data: dict[str, Any]) -> dict[str, Any]:
    for key in PARAM_KEYS:
        params = action_data.get(key)
        if isinstance(params, dict):
            for param_key, param_value in params.items():
                action_data.setdefault(param_key, param_value)
    return action_data


def _first_present(action_data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = action_data.get(key)
        if value is not None:
            return value
    return None


def _normalize_action_shape(action_data: Any) -> dict[str, Any]:
    if isinstance(action_data, list):
        if len(action_data) != 1 or not isinstance(action_data[0], dict):
            raise ValueError(f"Expected a single action object, got: {action_data}")
        action_data = action_data[0]

    if not isinstance(action_data, dict):
        raise ValueError(f"Expected JSON action object, got: {type(action_data).__name__}")

    nested_action = action_data.get("action")
    if isinstance(nested_action, dict):
        merged_action = {**action_data, **nested_action}
        merged_action.pop("action", None)
        action_data = merged_action

    action_data = _merge_parameter_block(action_data)

    if not action_data.get("action_type"):
        for key in ("action", "type", "name"):
            action_value = action_data.get(key)
            if isinstance(action_value, str):
                action_data["action_type"] = action_value
                break

    original_action_type = action_data.get("action_type")
    if (
        isinstance(original_action_type, str)
        and original_action_type.lower().strip().replace(" ", "_") == "swipe"
        and _get_drag_points(action_data) is not None
    ):
        action_data["action_type"] = "drag"

    if action_data.get("action_type") == "mcp":
        if not action_data.get("action_name"):
            for key in ("name", "tool_name", "function_name"):
                if isinstance(action_data.get(key), str):
                    action_data["action_name"] = action_data[key]
                    break
        if not action_data.get("action_json"):
            for key in PARAM_KEYS:
                if isinstance(action_data.get(key), dict):
                    action_data["action_json"] = action_data[key]
                    break

    return action_data


def parse_response_to_action(
    action_str: str,
    image_width: int,
    image_height: int,
    scale_factor: int | tuple[int, int] = 1000,
) -> dict:
    """
    Parse the JSON action from response and normalize it.
    Convert relative coordinates (0-999) to absolute coordinates based on image size.

    Args:
        action_str: JSON action string from model
        image_width: Width of the screenshot image
        image_height: Height of the screenshot image
        scale_factor: Scale factor for the coordinates
    Returns:
        Dictionary with action type and absolute coordinates
    """
    try:
        action_data = parse_json_markdown(_extract_action_payload(action_str))
        action_data = _normalize_action_shape(action_data)
        original_action_type = action_data.get("action_type")
        normalized_action_type = normalize_action_type(original_action_type)

        if not normalized_action_type:
            raise ValueError("Action type is missing or empty.")

        action_data["action_type"] = normalized_action_type
        action_type = normalized_action_type
        scale_factor_x, scale_factor_y = (
            [scale_factor, scale_factor] if isinstance(scale_factor, int) else scale_factor
        )

        if (
            action_type == "scroll"
            and "direction" not in action_data
            and action_data.get("scroll_direction") is not None
        ):
            action_data["direction"] = action_data["scroll_direction"]
        if action_type in ["input_text", "answer", "ask_user"] and "text" not in action_data:
            text_value = _first_present(action_data, TEXT_KEYS)
            if text_value is not None:
                action_data["text"] = text_value
        if action_type == "open_app" and "app_name" not in action_data:
            app_name_value = _first_present(action_data, APP_NAME_KEYS)
            if app_name_value is not None:
                action_data["app_name"] = app_name_value
        if (
            action_type == "status"
            and "goal_status" not in action_data
            and action_data.get("status") is not None
        ):
            action_data["goal_status"] = action_data["status"]

        # Handle coordinate-based actions
        if action_type in ["click", "double_tap", "long_press"]:
            # Ensure coordinate is present
            coord = _get_point(action_data, POINT_KEYS)
            if coord is None:
                raise ValueError(f"Missing coordinate for action type: {action_type}")

            # Convert relative coordinates (0-999) to absolute coordinates
            absolute_x, absolute_y = _to_absolute_point(
                coord, image_width, image_height, scale_factor_x, scale_factor_y
            )

            logger.debug(
                f"Coordinate conversion: relative ({coord[0]}, {coord[1]}) -> absolute ({absolute_x}, {absolute_y})"
            )

            return {
                "action_type": action_type,
                "x": absolute_x,
                "y": absolute_y,
            }

        # Handle drag action
        elif action_type == "drag":
            drag_points = _get_drag_points(action_data)
            if drag_points is None:
                raise ValueError("Missing coordinates for drag action")

            start_coord, end_coord = drag_points
            absolute_start_x, absolute_start_y = _to_absolute_point(
                start_coord, image_width, image_height, scale_factor_x, scale_factor_y
            )
            absolute_end_x, absolute_end_y = _to_absolute_point(
                end_coord, image_width, image_height, scale_factor_x, scale_factor_y
            )

            logger.debug(
                f"Drag coordinate conversion: relative ({start_coord[0]}, {start_coord[1]}) -> ({end_coord[0]}, {end_coord[1]}) | absolute ({absolute_start_x}, {absolute_start_y}) -> ({absolute_end_x}, {absolute_end_y})"
            )

            return {
                "action_type": "drag",
                "start_x": absolute_start_x,
                "start_y": absolute_start_y,
                "end_x": absolute_end_x,
                "end_y": absolute_end_y,
            }

        # Handle other action types
        elif action_type in [
            "answer",
            "navigate_home",
            "navigate_back",
            "scroll",
            "wait",
            "ask_user",
            "keyboard_enter",
        ]:
            return action_data
        elif action_type == "open_app":
            return {
                "action_type": "open_app",
                "app_name": action_data.get("app_name", ""),
            }
        elif action_type == "input_text":
            return {
                "action_type": "input_text",
                "text": action_data.get("text", ""),
            }
        elif action_type == "status":
            return {
                "action_type": "answer",
                "text": "task finished"
                if action_data.get("goal_status") == "complete"
                else "task failed",
            }
        else:
            return action_data

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON action: {e}")
        raise ValueError(f"Invalid JSON format in action: {action_str}")
    except Exception as e:
        logger.error(f"Error parsing action: {e}")
        raise ValueError(f"Error parsing action: {action_str}")


class GeneralE2EAgentMCP(MCPAgent):
    def __init__(
        self,
        model_name: str,
        llm_base_url: str,
        api_key: str = "empty",
        observation_type: str = "screenshot",
        runtime_conf: dict = {
            "history_n_images": 3,
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        tools: list[dict] = [],
        scale_factor: int = 1000,
        **kwargs,
    ):
        super().__init__(tools=tools, **kwargs)

        # Agent parameters
        self.model_name = model_name
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.observation_type = observation_type
        self.runtime_conf = runtime_conf
        self.scale_factor = scale_factor
        self._use_adaptive_resize = False
        if "opus-4" in self.model_name.lower() or "opus_4" in self.model_name.lower():
            self._use_adaptive_resize = True
        elif "claude" in self.model_name.lower():
            self.scale_factor = CLAUDE_IMAGE_SIZE
        if "kimi-k" in self.model_name.lower():
            self.scale_factor = 1

        logger.debug(f"Agent runtime_conf = {self.runtime_conf}")
        if self._use_adaptive_resize:
            logger.debug(f"Agent uses adaptive resize (max_dimension={CLAUDE_OPUS_MAX_DIMENSION})")
        else:
            logger.debug(f"Agent scale_factor = {self.scale_factor}")

        self.build_openai_client(self.llm_base_url, self.api_key)
        logger.debug(f"Agent base_url={self.llm_base_url} model={self.model_name}")

        self.history_n_images = self.runtime_conf.pop("history_n_images", 3)
        if os.getenv("HISTORY_N_IMAGES") is not None:
            self.history_n_images = int(os.getenv("HISTORY_N_IMAGES"))

        self.history_images = []
        self.history_responses = []
        self.actions = []

    def initialize_hook(self, instruction: str) -> None:
        """Hook for initializing the agent with instruction."""
        logger.info(f"Initializing general E2E agent with instruction: {instruction}")
        # Reset history when initializing with new instruction
        self.reset()

    def _get_user_message(
        self, img_data, tool_call_res, ask_user_response_res, instruction: str | None = None
    ) -> dict:
        content = []
        if instruction is not None:
            content.append(
                {
                    "type": "text",
                    "text": instruction,
                }
            )
        if tool_call_res is not None:
            content.append(
                {
                    "type": "text",
                    "text": f"Tool call result: {tool_call_res}",
                }
            )
        elif ask_user_response_res is not None:
            content.append(
                {
                    "type": "text",
                    "text": ask_user_response_res,
                }
            )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": img_data,
                }
            )
        return {
            "role": "user",
            "content": content,
        }

    def _hide_history_images(self, messages) -> list[dict]:
        num_images_used = 0
        for i in range(len(messages)):
            reverse_i = len(messages) - i - 1
            if messages[reverse_i]["role"] == "user":
                img_item_idx = None
                for idx, content in enumerate(messages[reverse_i]["content"]):
                    if content["type"] == "image_url":
                        img_item_idx = idx
                if img_item_idx is not None:
                    if num_images_used < self.history_n_images:
                        encoded_string = pil_to_base64(
                            messages[reverse_i]["content"][img_item_idx]["image_url"]
                        )
                        messages[reverse_i]["content"][img_item_idx]["image_url"] = {
                            "url": f"data:image/png;base64,{encoded_string}"
                        }
                        num_images_used += 1
                    else:
                        messages[reverse_i]["content"][img_item_idx] = {
                            "type": "text",
                            "text": "(Previous turn, screen not shown)",
                        }

        return messages

    def predict(
        self,
        observation: dict[str, Any],
    ) -> tuple[str, JSONAction]:
        """
        Generate action with coordinates based on the current observation.

        Args:
            observation: Observation containing screenshot

        Returns:
            Tuple of (raw_response, JSONAction)
        """

        orig_width, orig_height = observation["screenshot"].size
        if self._use_adaptive_resize:
            obs_image, _, _ = pil_adaptive_resize(
                observation["screenshot"], CLAUDE_OPUS_MAX_DIMENSION
            )
            active_scale_factor = obs_image.size  # (resized_w, resized_h)
        elif "claude" in self.model_name.lower():
            obs_image = observation["screenshot"].resize(CLAUDE_IMAGE_SIZE)
            active_scale_factor = self.scale_factor
        else:
            obs_image = observation["screenshot"]
            active_scale_factor = self.scale_factor
        tool_call = observation.get("tool_call", None)
        ask_user_response = observation.get("ask_user_response", None)

        self.history_images.append((obs_image, tool_call, ask_user_response))

        logger.debug(f"Current history images count: {len(self.history_images)}")
        logger.debug(f"Current history responses count: {len(self.history_responses)}")

        assert len(self.history_images) == len(self.history_responses) + 1
        messages = [
            {
                "role": "system",
                "content": GENERAL_E2E_PROMPT_TEMPLATE.render(
                    tools="\n".join([json.dumps(tool, ensure_ascii=False) for tool in self.tools]),
                    scale_factor=active_scale_factor,
                ),
            },
            # UPDATED 2026-04-21: user instruction may get ignored by opus-4.7 occasionally,
            # migrated user instruction from system prompt to user prompt!
            self._get_user_message(
                self.history_images[0][0],
                self.history_images[0][1],
                self.history_images[0][2],
                instruction=self.instruction,
            ),
        ]
        for i, history_resp in enumerate(self.history_responses):
            history_img_data, tool_call_res, ask_user_response_res = self.history_images[i + 1]

            user_message = self._get_user_message(
                history_img_data, tool_call_res, ask_user_response_res
            )

            response_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": history_resp.get("content", "")}],
            }

            messages.append(response_message)
            messages.append(user_message)

        logger.debug(f"Constructed {len(messages) // 2} history turns.")
        messages = self._hide_history_images(messages)

        pretty_print_messages(messages, max_messages=10)
        logger.debug("*" * 100)

        try_times = 3
        response = None
        thought = None
        action_str = None

        while try_times > 0:
            try:
                response = self.openai_chat_completions_create(
                    model=self.model_name,
                    messages=messages,
                    retry_times=1,
                    **self.runtime_conf,
                )
                logger.info(f"\nRaw LLM response received:\n{response}")

                thought, action_str = parse_action(response)

                break

            except Exception as e:
                logger.warning(
                    f"Error fetching response from agent: {self.model_name}, {self.llm_base_url}, {mask_api_key(self.api_key)}"
                )

                error_msg = str(e)
                try_times -= 1
                logger.warning(
                    f"Error fetching response from agent: {error_msg}. Retrying... ({try_times} attempts left)"
                )
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    time.sleep(2)

        if response is None:
            raise ValueError("Agent LLM failed")
        if action_str is None:
            return "Agent LLM failed", JSONAction(action_type="unknown", text="Agent LLM failed")

        logger.debug(f"Image size: {orig_width}x{orig_height}")

        try:
            json_action_dict = parse_response_to_action(
                action_str, orig_width, orig_height, active_scale_factor
            )

        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
            return "Agent LLM failed", JSONAction(action_type="unknown", text="Agent LLM failed")

        logger.info(f"Parsed thought: {thought}")
        logger.info(f"Parsed action: {json_action_dict}")

        self.history_responses.append({"role": "assistant", "content": response})
        self.actions.append(json_action_dict)
        logger.debug("Agent state updated for next turn.")

        return response, JSONAction(**json_action_dict)

    def reset(self):
        """Reset the agent for the next task."""
        self.history_images = []
        self.history_responses = []
        self.actions = []
        logger.debug("Agent reset completed")
