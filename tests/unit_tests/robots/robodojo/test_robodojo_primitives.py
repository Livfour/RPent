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

"""Offline contracts for RoboDojo primitives over a fake env client."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robots.robodojo.primitives import RoboDojoPrimitives


def _status(count: int, *, done: bool = False, success: bool = False) -> dict:
    return {
        "eval_success": success,
        "done": done,
        "take_action_cnt": count,
        "step_lim": 200,
        "layout_id": 0,
        "score": 1.0 if success else 0.0,
    }


class FakeEnv:
    """Executes ee16 chunks by teleporting the state to each commanded pose."""

    execution_capabilities = {"chunk_step_all_frames": True}

    def __init__(self) -> None:
        self.ee16 = np.array(
            [-0.2, 0.0, 0.9, 1, 0, 0, 0, 1.0, 0.2, 0.0, 0.9, 1, 0, 0, 0, 1.0]
        )
        self.count = 0
        self.terminated = False
        self.truncated = False
        self.chunks: list[np.ndarray] = []
        self.last_info = self._info()

    def _info(self) -> dict[str, Any]:
        return {
            "robot_state": {
                "left_eef_pose": self.ee16[0:7].copy(),
                "right_eef_pose": self.ee16[8:15].copy(),
                "left_gripper": float(self.ee16[7]),
                "right_gripper": float(self.ee16[15]),
                "ee16": self.ee16.copy(),
            },
            "episode_status": _status(self.count),
            "instruction": "Pick up the cup by 10 cm.",
        }

    def chunk_step(self, actions, *, action_type, return_all_frames):
        assert action_type == "ee"
        array = np.asarray(actions, dtype=np.float64)
        self.chunks.append(array)
        frames = []
        for action in array:
            self.ee16 = action.copy()
            self.count += 1
            frames.append(np.zeros((2, 2, 3), dtype=np.uint8))
        self.last_info = {
            **self._info(),
            "executed_actions": int(len(array)),
            "requested_actions": int(len(array)),
        }
        payload: Any = {"main_images": frames[-1]}
        if return_all_frames:
            payload = {"frames": frames, "final": payload}
        return payload, 0.0, False, False, self.last_info

    def status(self) -> dict[str, Any]:
        return _status(self.count)

    def render_camera(self, name):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def get_task_language(self) -> str:
        return "Pick up the cup by 10 cm."


def _primitives(env: FakeEnv | None = None, model: Any = None) -> RoboDojoPrimitives:
    return RoboDojoPrimitives(
        env=env or FakeEnv(),
        layout_id=0,
        check_cancelled=lambda: None,
        model=model,
    )


def test_move_to_interpolates_a_straight_line_and_keeps_the_other_arm() -> None:
    env = FakeEnv()
    primitives = _primitives(env)
    primitives.start_recording()
    result = primitives.move_to(arm="left", xyz=[-0.1, 0.1, 0.8], steps=4)

    chunk = env.chunks[0]
    assert chunk.shape == (4, 16)
    assert np.allclose(chunk[-1, 0:3], [-0.1, 0.1, 0.8])
    assert np.allclose(chunk[1, 0:3], [-0.15, 0.05, 0.85])
    # Orientation, gripper, and the right arm are untouched.
    assert np.allclose(chunk[:, 3:7], [1, 0, 0, 0])
    assert np.allclose(chunk[:, 7], 1.0)
    assert np.allclose(chunk[:, 8:16], env.ee16[8:16])
    assert result["success"] is True
    assert result["completed"] is True
    assert result["executed_steps"] == 4
    assert result["final_dist_m"] == pytest.approx(0.0)
    assert primitives.recorded_frame_count() == 4
    assert primitives.native_actions == 4
    assert primitives.policy_actions == 0


def test_move_delta_and_gripper_ramp_from_measured_state() -> None:
    env = FakeEnv()
    primitives = _primitives(env)
    result = primitives.move_delta(arm="right", dxyz=[0, 0, -0.1], steps=2)
    assert np.allclose(result["final_eef_xyz"], [0.2, 0.0, 0.8])

    grip = primitives.set_gripper(arm="right", val=0.0, steps=4)
    chunk = env.chunks[-1]
    assert np.allclose(chunk[:, 15], [0.75, 0.5, 0.25, 0.0])
    assert np.allclose(chunk[:, 8:11], [0.2, 0.0, 0.8])
    assert grip["gripper_val"] == pytest.approx(0.0)

    released = primitives.release(arm="right", steps=2)
    assert released["gripper_val"] == pytest.approx(1.0)


def test_rotate_wrist_yaws_about_world_z() -> None:
    env = FakeEnv()
    primitives = _primitives(env)
    primitives.rotate_wrist(arm="left", delta_yaw_deg=90.0, steps=1)
    quat = env.chunks[0][-1, 3:7]
    assert np.allclose(np.abs(quat), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-6)


def test_primitives_validate_arguments() -> None:
    primitives = _primitives()
    with pytest.raises(ValueError):
        primitives.move_to(arm="middle", xyz=[0, 0, 0])
    with pytest.raises(ValueError):
        primitives.move_to(arm="left", xyz=[0, 0, 0], steps=0)
    with pytest.raises(ValueError):
        primitives.set_gripper(arm="left", val=1.5)
    with pytest.raises(RuntimeError):
        primitives.vla_act(chunks=1)


def test_vla_act_uses_the_native_instruction_and_counts_policy_actions() -> None:
    env = FakeEnv()

    class FakeModel:
        def __init__(self) -> None:
            self.observations: list[dict[str, Any]] = []

        def infer(self, observation: dict[str, Any]) -> np.ndarray:
            self.observations.append(observation)
            return np.repeat(env.ee16[None, :], 3, axis=0)

    model = FakeModel()
    primitives = _primitives(env, model=model)
    result = primitives.vla_act(chunks=2, prompt="ignored")
    assert len(model.observations) == 2
    assert set(model.observations[0]["views"]) == {"head", "left_wrist", "right_wrist"}
    assert model.observations[0]["task_language"] == "Pick up the cup by 10 cm."
    assert result["executed_steps"] == 6
    assert result["agent_prompt_ignored"] is True
    assert primitives.policy_actions == 6


def test_finish_reports_native_success_only() -> None:
    env = FakeEnv()
    primitives = _primitives(env)
    result = primitives.finish(status="success", summary="done")
    assert result["_finish"] is True
    assert result["status"] == "failure"
    assert result["success"] is False

    env.status = lambda: _status(5, done=True, success=True)  # type: ignore[assignment]
    result = primitives.finish(status="failure", summary="oops")
    assert result["status"] == "success"
    assert result["success"] is True
