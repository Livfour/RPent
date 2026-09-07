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

"""Offline contracts for the RoboDojo toolkit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.robodojo import toolkit
from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools.toolkit import Toolkit, _is_readonly, readonly
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

ANALYTIC_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "render",
    "sample_world_xyz",
    "query_world_map",
    "move_to",
    "move_delta",
    "rotate_wrist",
    "set_gripper",
    "release",
}

PRIMITIVE_METHODS = {
    "start_recording",
    "recorded_frame_count",
    "frame_slice",
    "stop_recording",
    "status",
    "finish",
    "vla_act",
    "move_to",
    "move_delta",
    "rotate_wrist",
    "set_gripper",
    "release",
}


class FakeRoboDojoPrimitives:
    instances: list[FakeRoboDojoPrimitives] = []
    has_vla = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.status_calls = 0
        self.recording_started = False
        self.env = SimpleNamespace(
            last_reset_info={
                "instruction": "Pick up the mug by 10 cm.",
                "episode_status": {"layout_id": 3},
            }
        )
        type(self).instances.append(self)

    def start_recording(self) -> None:
        self.recording_started = True

    def recorded_frame_count(self) -> int:
        return 0

    def frame_slice(self, start: int) -> list[Any]:
        del start
        return []

    def stop_recording(self) -> list[Any]:
        return []

    def status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {
            "eval_success": False,
            "done": False,
            "take_action_cnt": 0,
            "step_lim": 200,
            "layout_id": 3,
            "score": 0.0,
        }

    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        return {"_finish": True, "status": status, "summary": summary}

    @staticmethod
    def _operation(name: str, **kwargs: Any) -> dict[str, Any]:
        return {"operation": name, "arguments": kwargs}

    def vla_act(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("vla_act", **kwargs)

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_to", **kwargs)

    def move_delta(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_delta", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_wrist", **kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("set_gripper", **kwargs)

    def release(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("release", **kwargs)


class FakeRoboDojoPrimitivesWithVLA(FakeRoboDojoPrimitives):
    has_vla = True


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def _offline_toolkit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    primitives_type: type,
    dumped: list[dict[str, Any]],
) -> toolkit.RoboDojoToolkit:
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(toolkit, "RoboDojoPrimitives", primitives_type)
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        toolkit.RoboDojoToolkit,
        "_capture_full_observation",
        lambda self: {"views": {}, "robot_state": {}, "task_language": "offline"},
    )
    monkeypatch.setattr(
        toolkit.tools,
        "dump_observation",
        lambda observation, env_state, status, log: (
            dumped.append({"observation": observation, "status": status, "log": log})
            or _record()
        ),
    )
    monkeypatch.setattr(
        toolkit.tools,
        "view_env_state",
        readonly(lambda step=-1, *, state: {"step": step}),
    )
    return toolkit.RoboDojoToolkit(
        primitives_kwargs={"env": object(), "layout_id": 3},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )


def test_fake_and_real_implement_toolkit_primitive_protocol() -> None:
    for primitive_type in (RoboDojoPrimitives, FakeRoboDojoPrimitives):
        missing = {
            name
            for name in PRIMITIVE_METHODS
            if not callable(getattr(primitive_type, name, None))
        }
        assert missing == set(), (
            f"{primitive_type.__name__} is missing toolkit methods: {sorted(missing)}"
        )


def test_toolkit_constructs_and_captures_an_initial_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeRoboDojoPrimitives.instances.clear()
    dumped: list[dict[str, Any]] = []
    robot_toolkit = _offline_toolkit(
        monkeypatch, tmp_path, FakeRoboDojoPrimitives, dumped
    )

    assert _tool_names(robot_toolkit) == ANALYTIC_TOOLS
    assert _readonly_names(robot_toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "sample_world_xyz",
        "query_world_map",
    }
    assert len(dumped) == 1
    assert dumped[0]["log"]["command"] == {"action": "reset"}
    assert dumped[0]["log"]["result"]["instruction"] == "Pick up the mug by 10 cm."
    primitive = FakeRoboDojoPrimitives.instances[0]
    assert primitive.status_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])
    assert robot_toolkit.solved() is False

    robot_toolkit.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    render = robot_toolkit.execute_tool("render", {})
    assert render.result == {"success": True}
    moved = robot_toolkit.execute_tool(
        "move_delta", {"arm": "left", "dxyz": [0.0, 0.0, 0.1]}
    )
    assert moved.result["operation"] == "move_delta"
    finish = robot_toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "offline"}
    )
    assert finish.is_finish is True


def test_vla_tool_registers_only_with_a_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeRoboDojoPrimitives.instances.clear()
    robot_toolkit = _offline_toolkit(
        monkeypatch, tmp_path, FakeRoboDojoPrimitivesWithVLA, []
    )
    assert _tool_names(robot_toolkit) == ANALYTIC_TOOLS | {"vla_act"}


def test_world_from_depth_back_projects_opengl_camera_frame() -> None:
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    meta = {
        "intrinsic_K": np.array([[10.0, 0, 3.0], [0, 10.0, 2.0], [0, 0, 1]]),
        "cam2world_gl": np.eye(4),
        "width": 6,
        "height": 4,
    }
    world = toolkit._world_from_depth(depth, meta)
    assert world.shape == (4, 6, 3)
    # Principal point looks straight down -Z in the OpenGL camera frame.
    assert np.allclose(world[2, 3], [0.0, 0.0, -2.0])
    # +u moves right (+x), +v moves down (-y).
    assert np.allclose(world[2, 4], [0.2, 0.0, -2.0])
    assert np.allclose(world[3, 3], [0.0, -0.2, -2.0])
