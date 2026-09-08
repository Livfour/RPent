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

"""RoboDojo primitives built on absolute end-effector pose actions.

RoboDojo executes one ``ee16`` action as an IK solve plus ten physics
substeps (40 ms). Analytic primitives therefore interpolate the commanded
end-effector pose from the measured one and send the interpolated poses as a
chunk; the native IK holds the previous target whenever a waypoint is
unreachable, so callers verify the achieved pose from ``final_dist_m``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.robot_spec import MODEL_SPEC, ROBODOJO_CAMERA_NAMES

ARMS = ("left", "right")
#: Offsets of each arm's pose and gripper inside the ee16 layout.
_ARM_SLICES = {"left": (slice(0, 7), 7), "right": (slice(8, 15), 15)}


def _qmult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _require_arm(arm: Any) -> str:
    if arm not in ARMS:
        raise ValueError("arm must be 'left' or 'right'")
    return str(arm)


class RoboDojoPrimitives:
    """Compose RoboDojo operations from the env RPC and an optional VLA."""

    #: Guard rails for a single planned end-effector path.
    _MAX_PATH_STEPS = 400
    _MAX_PATH_WAYPOINTS = 12

    def __init__(
        self,
        *,
        env: RoboDojoEnvClient,
        layout_id: int,
        check_cancelled: Callable[[], None],
        model: Any | None = None,
    ):
        self.env = env
        self.model = model
        self.layout_id = int(layout_id)
        self._check_cancelled = check_cancelled
        self.policy_actions = 0
        self.native_actions = 0
        self._recording = False
        self._frames: list[np.ndarray] = []

    # ---- recording -------------------------------------------------------

    def start_recording(self) -> None:
        self._recording = True
        self._frames = []

    def record_frame(self, rgb: Any) -> None:
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start) :])

    # ---- state helpers ---------------------------------------------------

    @property
    def has_vla(self) -> bool:
        return self.model is not None

    def _state(self) -> dict[str, Any]:
        return self.env.last_info["robot_state"]

    def _current_ee16(self) -> np.ndarray:
        ee16 = np.asarray(self._state()["ee16"], dtype=np.float64)
        if ee16.shape != (16,) or not np.isfinite(ee16).all():
            raise RuntimeError("RoboDojo robot_state.ee16 must be finite (16,)")
        return ee16.copy()

    def _episode_status(self) -> dict[str, Any]:
        return self.env.last_info["episode_status"]

    @staticmethod
    def _completion(
        *, requested: int, executed: int, status: dict[str, Any]
    ) -> dict[str, Any]:
        step_lim = status.get("step_lim")
        budget_exhausted = step_lim is not None and int(
            status.get("take_action_cnt", 0)
        ) >= int(step_lim)
        completed = executed == requested
        if status.get("eval_success") is True:
            stop_reason = "native_success"
        elif budget_exhausted or status.get("done"):
            stop_reason = "budget_exhausted"
        elif completed:
            stop_reason = "completed"
        else:
            stop_reason = "runtime_failure"
        return {
            "completed": completed,
            "requested_steps": requested,
            "executed_steps": executed,
            "stop_reason": stop_reason,
        }

    def _execute_chunk(self, actions: np.ndarray, *, policy: bool) -> dict[str, Any]:
        """Send an ee16 chunk, record frames, and return the execution info."""
        self._check_cancelled()
        if self.env.terminated or self.env.truncated:
            raise RuntimeError("RoboDojo episode is terminal; no further actions")
        return_frames = bool(
            self._recording
            and self.env.execution_capabilities.get("chunk_step_all_frames") is True
        )
        payload, _, _, _, info = self.env.chunk_step(
            actions, action_type="ee", return_all_frames=return_frames
        )
        if return_frames and isinstance(payload, dict) and "frames" in payload:
            for frame in payload["frames"]:
                self.record_frame(frame)
        executed = int(info.get("executed_actions", 0))
        self.native_actions += executed
        if policy:
            self.policy_actions += executed
        self._check_cancelled()
        return {
            "action_type": "ee",
            "requested_actions": int(len(actions)),
            "executed_actions": executed,
            "episode_status": info["episode_status"],
        }

    def _interpolated_chunk(
        self,
        arm: str,
        *,
        xyz: np.ndarray,
        quat: np.ndarray,
        gripper: float | None,
        steps: int,
    ) -> np.ndarray:
        current = self._current_ee16()
        pose_slice, grip_index = _ARM_SLICES[arm]
        start_pose = current[pose_slice]
        start_grip = float(current[grip_index])
        target_grip = start_grip if gripper is None else float(np.clip(gripper, 0, 1))
        chunk = np.repeat(current[None, :], steps, axis=0)
        for index in range(steps):
            t = (index + 1) / steps
            chunk[index, pose_slice.start : pose_slice.start + 3] = (
                start_pose[:3] + (xyz - start_pose[:3]) * t
            )
            chunk[index, pose_slice.start + 3 : pose_slice.stop] = _slerp(
                start_pose[3:], quat, t
            )
            chunk[index, grip_index] = start_grip + (target_grip - start_grip) * t
        return chunk

    # ---- primitives ------------------------------------------------------

    def move_to(
        self,
        *,
        arm: str,
        xyz: list[float],
        quat: list[float] | None = None,
        gripper: float | None = None,
        steps: int = 20,
    ) -> dict[str, Any]:
        """Move one end-effector to a world pose along a straight line."""
        arm = _require_arm(arm)
        steps = int(steps)
        if steps < 1 or steps > 200:
            raise ValueError("steps must be between 1 and 200")
        target_xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        pose_slice, _ = _ARM_SLICES[arm]
        if quat is None:
            target_quat = self._current_ee16()[pose_slice][3:]
        else:
            target_quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if not np.isfinite(target_xyz).all() or not np.isfinite(target_quat).all():
            raise ValueError("xyz and quat must be finite")
        if np.linalg.norm(target_quat) < 1e-6:
            raise ValueError("quat must be non-zero")
        chunk = self._interpolated_chunk(
            arm, xyz=target_xyz, quat=target_quat, gripper=gripper, steps=steps
        )
        execution = self._execute_chunk(chunk, policy=False)
        final = self._current_ee16()[pose_slice]
        return {
            **execution,
            **self._completion(
                requested=steps,
                executed=execution["executed_actions"],
                status=execution["episode_status"],
            ),
            "success": True,
            "arm": arm,
            "target_xyz": target_xyz.tolist(),
            "final_eef_xyz": final[:3].tolist(),
            "final_eef_quat": final[3:].tolist(),
            "final_dist_m": float(np.linalg.norm(final[:3] - target_xyz)),
        }

    def _plan_ee_path(
        self,
        arm: str,
        waypoints: list[dict[str, Any]],
        default_steps: int,
        ease: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Solve one ee16 trajectory through planned end-effector waypoints.

        Every segment starts at the previous *commanded* pose instead of a fresh
        state read, so the whole path is known before the first action executes
        and a waypoint the IK undershoots cannot drag the rest of the path with
        it. The opposite arm holds the pose it had when planning started.
        """
        pose_slice, grip_index = _ARM_SLICES[arm]
        start = pose_slice.start
        base = self._current_ee16()
        cursor_xyz = base[start : start + 3].copy()
        cursor_quat = base[start + 3 : pose_slice.stop].copy()
        cursor_grip = float(base[grip_index])
        segments: list[np.ndarray] = []
        plan: list[dict[str, Any]] = []
        total_steps = 0
        for index, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, dict):
                raise ValueError(f"waypoint {index} must be an object")
            raw_xyz = waypoint.get("xyz")
            target_xyz = (
                cursor_xyz.copy()
                if raw_xyz is None
                else np.asarray(raw_xyz, dtype=np.float64).reshape(3)
            )
            raw_quat = waypoint.get("quat")
            target_quat = (
                cursor_quat.copy()
                if raw_quat is None
                else np.asarray(raw_quat, dtype=np.float64).reshape(4)
            )
            raw_grip = waypoint.get("gripper")
            target_grip = (
                cursor_grip if raw_grip is None else float(np.clip(raw_grip, 0.0, 1.0))
            )
            steps = int(waypoint.get("steps") or default_steps)
            if steps < 1 or steps > 200:
                raise ValueError(f"waypoint {index}: steps must be between 1 and 200")
            if not np.isfinite(target_xyz).all() or not np.isfinite(target_quat).all():
                raise ValueError(f"waypoint {index}: xyz and quat must be finite")
            norm = float(np.linalg.norm(target_quat))
            if norm < 1e-6:
                raise ValueError(f"waypoint {index}: quat must be non-zero")
            target_quat = target_quat / norm
            total_steps += steps
            if total_steps > self._MAX_PATH_STEPS:
                raise ValueError(
                    f"path needs {total_steps} actions; the limit is "
                    f"{self._MAX_PATH_STEPS}"
                )
            segment = np.repeat(base[None, :], steps, axis=0)
            for offset in range(steps):
                t = (offset + 1) / steps
                if ease:
                    # Smoothstep: zero velocity at both ends of every segment.
                    t = t * t * (3.0 - 2.0 * t)
                segment[offset, start : start + 3] = (
                    cursor_xyz + (target_xyz - cursor_xyz) * t
                )
                segment[offset, start + 3 : pose_slice.stop] = _slerp(
                    cursor_quat, target_quat, t
                )
                segment[offset, grip_index] = (
                    cursor_grip + (target_grip - cursor_grip) * t
                )
            segments.append(segment)
            plan.append(
                {
                    "index": index,
                    "xyz": target_xyz.tolist(),
                    "quat": target_quat.tolist(),
                    "gripper": target_grip,
                    "steps": steps,
                }
            )
            cursor_xyz, cursor_quat, cursor_grip = target_xyz, target_quat, target_grip
        return np.concatenate(segments, axis=0), plan

    def follow_ee_path(
        self,
        *,
        arm: str,
        waypoints: list[dict[str, Any]],
        default_steps: int = 20,
        ease: bool = True,
    ) -> dict[str, Any]:
        """Execute a whole planned end-effector path in one native chunk."""
        arm = _require_arm(arm)
        if not isinstance(waypoints, (list, tuple)) or not waypoints:
            raise ValueError("waypoints must be a non-empty list")
        if len(waypoints) > self._MAX_PATH_WAYPOINTS:
            raise ValueError(
                f"a path takes at most {self._MAX_PATH_WAYPOINTS} waypoints"
            )
        default_steps = int(default_steps)
        if default_steps < 1 or default_steps > 200:
            raise ValueError("default_steps must be between 1 and 200")
        chunk, plan = self._plan_ee_path(arm, list(waypoints), default_steps, bool(ease))
        execution = self._execute_chunk(chunk, policy=False)
        pose_slice, grip_index = _ARM_SLICES[arm]
        final = self._current_ee16()
        final_pose = final[pose_slice]
        goal_xyz = np.asarray(plan[-1]["xyz"], dtype=np.float64)
        return {
            **execution,
            **self._completion(
                requested=int(len(chunk)),
                executed=execution["executed_actions"],
                status=execution["episode_status"],
            ),
            "success": True,
            "arm": arm,
            "waypoints": plan,
            "planned_steps": int(len(chunk)),
            "final_eef_xyz": final_pose[:3].tolist(),
            "final_eef_quat": final_pose[3:].tolist(),
            "final_gripper": float(final[grip_index]),
            "final_dist_m": float(np.linalg.norm(final_pose[:3] - goal_xyz)),
        }

    def move_delta(
        self,
        *,
        arm: str,
        dxyz: list[float],
        gripper: float | None = None,
        steps: int = 10,
    ) -> dict[str, Any]:
        """Move one end-effector by a world-frame offset, keeping orientation."""
        arm = _require_arm(arm)
        pose_slice, _ = _ARM_SLICES[arm]
        current = self._current_ee16()[pose_slice]
        target = current[:3] + np.asarray(dxyz, dtype=np.float64).reshape(3)
        result = self.move_to(
            arm=arm, xyz=target.tolist(), gripper=gripper, steps=steps
        )
        result["requested_dxyz"] = list(map(float, dxyz))
        return result

    def rotate_wrist(
        self,
        *,
        arm: str,
        delta_yaw_deg: float,
        gripper: float | None = None,
        steps: int = 10,
    ) -> dict[str, Any]:
        """Rotate one end-effector about world z by a relative yaw."""
        arm = _require_arm(arm)
        pose_slice, _ = _ARM_SLICES[arm]
        pose = self._current_ee16()[pose_slice]
        yaw = np.deg2rad(float(delta_yaw_deg))
        world_z = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        result = self.move_to(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=_qmult(world_z, pose[3:]).tolist(),
            gripper=gripper,
            steps=steps,
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def set_gripper(self, *, arm: str, val: float, steps: int = 10) -> dict[str, Any]:
        """Ramp one normalized gripper (0 closed, 1 open) to ``val``."""
        arm = _require_arm(arm)
        steps = int(steps)
        if steps < 1 or steps > 100:
            raise ValueError("steps must be between 1 and 100")
        target = float(val)
        if not 0.0 <= target <= 1.0:
            raise ValueError("val must be within [0, 1]")
        current = self._current_ee16()
        pose_slice, grip_index = _ARM_SLICES[arm]
        chunk = np.repeat(current[None, :], steps, axis=0)
        start = float(current[grip_index])
        for index in range(steps):
            chunk[index, grip_index] = start + (target - start) * (index + 1) / steps
        execution = self._execute_chunk(chunk, policy=False)
        now = self._state()
        return {
            **execution,
            **self._completion(
                requested=steps,
                executed=execution["executed_actions"],
                status=execution["episode_status"],
            ),
            "success": True,
            "arm": arm,
            "gripper_val": float(now[f"{arm}_gripper"]),
        }

    def release(self, *, arm: str, val: float = 1.0, steps: int = 10) -> dict[str, Any]:
        """Open one gripper to the requested release value."""
        return self.set_gripper(arm=arm, val=val, steps=steps)

    def _build_vla_observation(self) -> dict[str, Any]:
        views = {
            name: {"rgb": np.asarray(self.env.render_camera(name))}
            for name in ROBODOJO_CAMERA_NAMES
        }
        return {
            "views": views,
            "robot_state": self._state(),
            "task_language": self.env.get_task_language(),
        }

    def vla_act(self, *, chunks: int = 1, prompt: str | None = None) -> dict[str, Any]:
        """Infer and execute VLA ee16 chunks using the native instruction."""
        if self.model is None:
            raise RuntimeError("no RoboDojo VLA endpoint is configured")
        chunks = int(chunks)
        if chunks < 1:
            raise ValueError("chunks must be at least 1")
        executed = 0
        requested = chunks * MODEL_SPEC.use_length
        native_prompt = None
        for _ in range(chunks):
            self._check_cancelled()
            status = self._episode_status()
            if status.get("done"):
                break
            observation = self._build_vla_observation()
            native_prompt = observation["task_language"]
            actions = self.model.infer(observation)
            execution = self._execute_chunk(np.asarray(actions), policy=True)
            executed += int(execution["executed_actions"])
        status = self._episode_status()
        return {
            **self._completion(requested=requested, executed=executed, status=status),
            "success": True,
            "prompt": native_prompt,
            "agent_prompt_ignored": prompt is not None,
            "ignored_agent_prompt": prompt,
            "episode_status": status,
        }

    # ---- status ----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return the native episode status plus action counters."""
        return {
            **self.env.status(),
            "policy_actions": self.policy_actions,
            "native_actions": self.native_actions,
        }

    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        """Finish the Planner run and verify success against native status."""
        requested_success = status.lower() == "success"
        try:
            native = self.status()
        except Exception as error:  # The terminal tool must still stop the Planner.
            return {
                "_finish": True,
                "status": "error",
                "summary": summary,
                "requested_status": status,
                "requested_success": requested_success,
                "runtime_error": f"{type(error).__name__}: {error}",
            }
        verified_success = native.get("eval_success") is True
        reported_status = (
            "success"
            if verified_success
            else ("failure" if requested_success else status)
        )
        return {
            "_finish": True,
            "status": reported_status,
            "summary": summary,
            "requested_success": requested_success,
            "success": verified_success,
            "episode_status": native,
        }
