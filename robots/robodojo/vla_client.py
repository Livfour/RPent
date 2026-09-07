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

"""Optional RPC client for a RoboDojo end-effector VLA server.

The server is any :class:`~rpent.robots.components.vla_facade_base.BaseVLAFacade`
that answers ``vla.get_meta`` with :func:`robots.robodojo.robot_spec.vla_runtime_contract`
and ``vla.predict`` with a ``[chunk, 16]`` absolute end-effector action chunk
(``[x, y, z, qw, qx, qy, qz, gripper]`` per arm, gripper in ``[0, 1]``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robodojo.robot_spec import MODEL_SPEC, ROBODOJO_CAMERAS
from rpent.robots.components.vla_client_base import BaseVLAClient


class RoboDojoVLAClient(BaseVLAClient):
    """Assemble RoboDojo observations into the VLA server payload."""

    def validate_contract(self, expected_meta: dict[str, Any]) -> None:
        actual_meta = self._client.call(
            "vla.get_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if actual_meta != expected_meta:
            raise RuntimeError(
                "RoboDojo VLA metadata mismatch: "
                f"expected={expected_meta!r} actual={actual_meta!r}"
            )

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Infer one ee16 action chunk from rgb views, state, and language."""
        views = observation["views"]
        state = observation["robot_state"]
        ee16 = np.asarray(state["ee16"], dtype=np.float32)
        if ee16.shape != (16,):
            raise RuntimeError("RoboDojo ee16 state must have shape (16,)")
        payload = {
            "images": {
                native: np.asarray(views[agent]["rgb"])
                for agent, native in ROBODOJO_CAMERAS.items()
            },
            "state": ee16,
            "instruction": observation["task_language"],
        }
        actions = np.asarray(super().predict(payload)["action"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise RuntimeError(
                f"RoboDojo VLA returned {actions.shape}; expected [chunk, 16]"
            )
        return actions[: MODEL_SPEC.use_length]
