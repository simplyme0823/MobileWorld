"""Midscene RPC-backed agent for MobileWorld benchmark runs."""

import json
import os
import time
from typing import Any

import requests
from loguru import logger

from mobile_world.agents.base import BaseAgent
from mobile_world.runtime.utils.models import ANSWER, FINISHED, UNKNOWN, JSONAction


class MidsceneAgent(BaseAgent):
    """MobileWorld agent adapter that delegates execution to Midscene Bench RPC."""

    def __init__(
        self,
        *args: Any,
        env: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.env = env
        self.rpc_url = os.environ.get("MIDSCENE_BENCH_RPC_URL")
        if not self.rpc_url:
            raise RuntimeError("MIDSCENE_BENCH_RPC_URL environment variable is not set")

        self.current_task_name: str | None = None
        self.current_agent_id: str | None = None
        self.failed_step_reason = ""
        self.task_status_sent = False
        self.has_run = False

    def start_task(self, task_name: str) -> None:
        """Create a Midscene Android agent for the current MobileWorld task."""
        self.current_task_name = task_name
        task_index = self._get_task_index(task_name)
        self.current_agent_id = f"Task-{task_index}-{task_name}"
        self.failed_step_reason = ""
        self.task_status_sent = False
        self.has_run = False

        rpc_params = {
            "type": "Android",
            "device": self._build_android_device_params(),
            "id": self.current_agent_id,
        }
        self._send_rpc_request("new-agent", rpc_params)

    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        """Run Midscene once; Midscene performs its own observation/action loop."""
        if self.has_run:
            return "Midscene task already executed.", JSONAction(action_type=FINISHED)
        if not self.current_agent_id:
            raise RuntimeError("MidsceneAgent.start_task() must be called before predict()")

        self.has_run = True
        response = self._send_rpc_request(
            "run-ai-method",
            {"id": self.current_agent_id, "task": self.instruction},
        )

        result = response.get("result", {})
        if result.get("code") == 1:
            action_raw_res = result.get("data", "")
            answer_text = self._stringify_result(action_raw_res)
            if answer_text:
                return answer_text, JSONAction(action_type=ANSWER, text=answer_text)
            return "Midscene completed the task.", JSONAction(action_type=FINISHED)

        reason = result.get("data", {}).get("reason", "Midscene RPC run-ai-method failed")
        self.failed_step_reason = str(reason)
        logger.error(f"Midscene RPC failed for {self.current_agent_id}: {self.failed_step_reason}")
        return self.failed_step_reason, JSONAction(action_type=UNKNOWN)

    def update_task_status(self, status: str = "Failed", reason: str | None = None) -> None:
        """Report MobileWorld validator status back to Midscene Benchmark."""
        if not self.current_agent_id or self.task_status_sent:
            return

        agent_error = self.failed_step_reason or (reason or "")
        self._send_rpc_request(
            "terminate-agent",
            {
                "id": self.current_agent_id,
                "userTaskStatus": status,
                "agentStepError": agent_error,
            },
        )
        self.task_status_sent = True

    def reset(self) -> None:
        self.current_task_name = None
        self.current_agent_id = None
        self.failed_step_reason = ""
        self.task_status_sent = False
        self.has_run = False

    def _build_android_device_params(self) -> dict[str, Any]:
        device: dict[str, Any] = {"type": "Android"}

        connection_type = os.environ.get("ANDROID_CONNECTION_TYPE", "Remote")
        console_port = os.environ.get("ANDROID_CONSOLE_PORT", "5554")
        adb_port = os.environ.get("ANDROID_ADB_PORT", "5555")

        if connection_type == "Local":
            device["type"] = "Local"
            device_id = os.environ.get("ANDROID_ADB_DEVICE_ID")
            if device_id:
                device["deviceId"] = device_id
            else:
                device["deviceId"] = f"emulator-{console_port}"
        else:
            device["type"] = "Remote"
            device["host"] = os.environ.get("ANDROID_REMOTE_HOST", "localhost")
            device["port"] = adb_port
            device_id = os.environ.get("ANDROID_ADB_DEVICE_ID")
            if device_id:
                device["deviceId"] = device_id

        device["consolePort"] = int(console_port)
        adb_server_port = os.environ.get("ANDROID_ADB_SERVER_PORT")
        if adb_server_port:
            device["adbServerPort"] = int(adb_server_port)

        return device

    def _send_rpc_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": time.time(),
        }

        last_error: Exception | None = None
        for request_cnt in range(1, 4):
            try:
                response = requests.post(
                    self.rpc_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                logger.debug(f"Midscene RPC response for {method}: {result}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Midscene RPC request {method} failed, retry {request_cnt}/3: {e}")

        raise RuntimeError(f"Failed to send Midscene RPC request {method}: {last_error}")

    @staticmethod
    def _stringify_result(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _get_task_index(task_name: str) -> int:
        index_map_raw = os.environ.get("MOBILE_WORLD_TASK_INDEX_MAP", "{}")
        try:
            index_map = json.loads(index_map_raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid MOBILE_WORLD_TASK_INDEX_MAP: {index_map_raw}")
            return 1

        try:
            return int(index_map.get(task_name, 1))
        except (TypeError, ValueError):
            return 1
