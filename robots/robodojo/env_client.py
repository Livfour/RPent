# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPC client for one RoboDojo Isaac Sim environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robodojo.robot_spec import (
    ROBODOJO_ACTION_DIMS,
    ROBODOJO_CAMERA_NAMES,
    ROBODOJO_READ_TIMEOUT_S,
    ROBODOJO_STATE_CHANGE_TIMEOUT_S,
    ROBODOJO_STATUS_KEYS,
    RoboDojoActionType,
)
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class RoboDojoEnvClient(BaseEnvClient):
    """Client for one RoboDojo ``EvalEnv`` served by ``env_server.py``."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "default": ROBODOJO_READ_TIMEOUT_S,
        # The first reset builds the Isaac scene from the published layout.
        "env.reset": ROBODOJO_STATE_CHANGE_TIMEOUT_S,
    }

    def __init__(self, client: RpcClient, *, expected_meta: dict[str, Any]):
        self.terminated = False
        self.truncated = False
        self._expected_layout = int(expected_meta["layout_id"])
        super().__init__(client, expected_meta=expected_meta)
        self.server_meta = dict(expected_meta)
        execution = self.server_meta.get("execution", {})
        self.execution_capabilities = (
            dict(execution) if isinstance(execution, dict) else {}
        )

    @staticmethod
    def _require_result_tuple(result: Any, size: int, method: str) -> tuple:
        if not isinstance(result, (list, tuple)) or len(result) != size:
            raise TypeError(f"{method} must return a {size}-item tuple, got {result!r}")
        return tuple(result)

    def _require_active(self) -> None:
        if self.terminated or self.truncated:
            raise RuntimeError("RoboDojo episode is terminal; no further actions")

    @staticmethod
    def _validate_action_request(action_type: Any, actions: Any) -> np.ndarray:
        if action_type not in ROBODOJO_ACTION_DIMS:
            raise ValueError("action_type must be 'ee' or 'joint'")
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        expected = ROBODOJO_ACTION_DIMS[action_type]
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != expected:
            raise ValueError(
                f"{action_type} actions must have shape [N,{expected}], N >= 1"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{action_type} actions must contain only finite values")
        return array

    @staticmethod
    def _require_episode_status(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise TypeError(f"execution info must be a mapping, got {info!r}")
        status = info.get("episode_status")
        if not isinstance(status, dict):
            raise TypeError(f"episode_status must be a mapping, got {status!r}")
        missing = [key for key in ROBODOJO_STATUS_KEYS if key not in status]
        if missing:
            raise ValueError(f"episode_status is missing {missing}: {status!r}")
        return status

    def _absorb(self, info: dict[str, Any]) -> None:
        status = self._require_episode_status(info)
        self.last_info = info
        if status["done"]:
            if status["eval_success"]:
                self.terminated = True
            else:
                self.truncated = True

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the requested layout and validate the native result."""
        result = self._client.call("env.reset", timeout_s=self._TIMEOUT_S["env.reset"])
        observation, info = self._require_result_tuple(result, 2, "env.reset")
        if not isinstance(observation, dict):
            raise TypeError(
                f"RoboDojo reset observation must be a mapping, got {observation!r}"
            )
        status = self._require_episode_status(info)
        if int(status["layout_id"]) != self._expected_layout:
            raise ValueError(f"reset did not use the requested layout: {info!r}")
        if not isinstance(info.get("instruction"), str):
            raise TypeError("reset instruction must be a string")
        self.last_obs = observation
        self.last_reset_info = dict(info)
        self.last_info = info
        self.terminated = False
        self.truncated = False
        return observation, info

    def step(
        self,
        action,
        *,
        action_type: RoboDojoActionType = "ee",
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Execute one RoboDojo action and update the cached episode state."""
        flat = self._validate_action_request(action_type, action)
        if flat.shape[0] != 1:
            raise ValueError("RoboDojo common action must be a single action")
        self._require_active()
        result = self._client.call(
            "env.step",
            args=(flat[0],),
            kwargs={"action_type": action_type},
            timeout_s=ROBODOJO_STATE_CHANGE_TIMEOUT_S,
        )
        result = self._require_result_tuple(result, 5, "env.step")
        self.last_obs = result[0]
        self._absorb(result[4])
        return result

    def chunk_step(
        self,
        actions,
        *,
        action_type: RoboDojoActionType = "ee",
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Execute a RoboDojo action chunk and cache the final episode state."""
        array = self._validate_action_request(action_type, actions)
        self._require_active()
        result = self._client.call(
            "env.chunk_step",
            args=(array,),
            kwargs={
                "action_type": action_type,
                "return_all_frames": return_all_frames,
            },
            timeout_s=ROBODOJO_STATE_CHANGE_TIMEOUT_S,
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        info = result[4]
        n_executed = info.get("executed_actions") if isinstance(info, dict) else None
        if (
            isinstance(n_executed, bool)
            or not isinstance(n_executed, int)
            or not 0 <= n_executed <= len(array)
        ):
            raise ValueError(f"invalid RoboDojo chunk result: {result!r}")
        obs_field = result[0]
        if return_all_frames:
            if (
                not isinstance(obs_field, dict)
                or "frames" not in obs_field
                or "final" not in obs_field
            ):
                raise TypeError(
                    "RoboDojo return_all_frames=True must return a "
                    "{'frames','final'} payload, got " + repr(obs_field)
                )
            self.last_obs = obs_field["final"]
        else:
            self.last_obs = obs_field
        self._absorb(info)
        return result

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        """Render one RoboDojo camera, optionally including metric depth."""
        if camera_name not in ROBODOJO_CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboDojo camera {camera_name!r}; "
                f"available={list(ROBODOJO_CAMERA_NAMES)}"
            )
        if not isinstance(depth, bool):
            raise TypeError("RoboDojo camera depth flag must be bool")
        return super().render_camera(camera_name, depth=depth)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        if camera_name not in ROBODOJO_CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboDojo camera {camera_name!r}; "
                f"available={list(ROBODOJO_CAMERA_NAMES)}"
            )
        return super().get_camera_meta(camera_name)

    def get_task_language(self) -> str:
        result = super().get_task_language()
        if not isinstance(result, str):
            raise TypeError(f"RoboDojo task language must be a string: {result!r}")
        return result

    def status(self) -> dict[str, Any]:
        """Return a fresh native episode status."""
        result = self._client.call("env.status", timeout_s=self._TIMEOUT_S["default"])
        return self._require_episode_status({"episode_status": result})
