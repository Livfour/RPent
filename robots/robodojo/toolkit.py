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

"""RPent tools for the RoboDojo robot."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.robodojo import tools
from robots.robodojo.primitives import RoboDojoPrimitives
from robots.robodojo.robot_spec import ROBODOJO_CAMERA_NAMES
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import get_logger, get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

logger = get_logger("robodojo_toolkit")

_ACTION_TOOLS = (
    "follow_ee_path",
    "move_to",
    "move_delta",
    "rotate_wrist",
    "set_gripper",
    "release",
)
_RECIPE_ACTIONS = {"vla_act", *_ACTION_TOOLS}


def _world_from_depth(
    depth_metric: np.ndarray, camera_meta: dict[str, Any]
) -> np.ndarray:
    """Back-project metric depth into the RoboDojo world frame.

    Isaac cameras follow the OpenGL convention (looking down -Z, +Y up) and
    ``distance_to_image_plane`` is the perpendicular depth, so the camera-frame
    point is ``[(u - cx) d / fx, -(v - cy) d / fy, -d]``.
    """
    depth = np.asarray(depth_metric, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"RoboDojo depth must have shape [H,W], got {depth.shape}")

    intrinsic = np.asarray(camera_meta.get("intrinsic_K"), dtype=np.float64)
    cam2world = np.asarray(camera_meta.get("cam2world_gl"), dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("RoboDojo camera intrinsic_K must have shape (3,3)")
    if cam2world.shape != (4, 4):
        raise ValueError("RoboDojo camera cam2world_gl must have shape (4,4)")
    if not np.isfinite(intrinsic).all() or not np.isfinite(cam2world).all():
        raise ValueError("RoboDojo camera calibration must contain only finite values")

    height, width = depth.shape
    if camera_meta.get("height") != height or camera_meta.get("width") != width:
        raise ValueError(
            "RoboDojo depth shape does not match camera metadata: "
            f"depth={depth.shape}, metadata="
            f"({camera_meta.get('height')}, {camera_meta.get('width')})"
        )
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    rows, cols = np.mgrid[0:height, 0:width]
    camera_points = np.stack(
        [
            (cols - cx) * depth / fx,
            -(rows - cy) * depth / fy,
            -depth,
        ],
        axis=-1,
    )
    world = camera_points @ cam2world[:3, :3].T + cam2world[:3, 3]
    return world.astype(np.float32)


class RoboDojoToolkit(Toolkit):
    """Common RPent tools plus RoboDojo primitives."""

    _SPECS = {spec["name"]: spec for spec in tools.TOOLS_SPEC}
    _FRAME_ARTIFACTS = {
        "camera": "head_rgb.png",
        "left_wrist": "left_wrist_rgb.png",
        "right_wrist": "right_wrist_rgb.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        perception_only: bool = False,
    ):
        state = EnvState(get_output_dir())
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._latest_status: dict[str, Any] = {}
        self._perception_only = perception_only
        self._primitives = RoboDojoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        self._primitives.start_recording()
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        reset_info = self._primitives.env.last_reset_info
        reset_result = {
            "instruction": reset_info.get("instruction"),
            "episode_status": reset_info.get("episode_status"),
            "success": True,
        }
        self._register_robodojo_tools()
        initial = self.get_env_state(
            command={"action": "reset"},
            result=reset_result,
            elapsed_s=0.0,
        )
        record = self._state.latest_record()
        if record is not None:
            self._publish_step(record)
        initial_state = initial.get("state")
        if isinstance(initial_state, dict):
            self._latest_status = initial_state.get(
                "episode_status", self._latest_status
            )

    def _register_robodojo_tools(self) -> None:
        self._tools.pop("finish", None)
        self.add_tool(
            "view_env_state",
            self._SPECS["view_env_state"],
            partial(tools.view_env_state, state=self._state),
        )
        if not self._perception_only:
            for name, handler in (
                ("sample_world_xyz", partial(tools.sample_world_xyz, self._state)),
                ("find_objects", partial(tools.find_objects, self._state)),
                ("query_world_map", partial(tools.query_world_map, self._state)),
            ):
                self.add_tool(name, self._SPECS[name], handler)
        names = ["render", *_ACTION_TOOLS]
        if getattr(self._primitives, "has_vla", False):
            names.append("vla_act")
        for name in names:
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        self.add_tool("finish", self._SPECS["finish"], self._finish)
        self.add_tool(
            "reset_episode",
            {
                "name": "reset_episode",
                "description": "Reset this layout for a fresh controlled attempt; preserve lessons first.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self._reset_episode,
        )
        self.add_tool(
            "write_lesson",
            {
                "name": "write_lesson",
                "description": "Record a concise attempted strategy or failure in the current memory inbox for later runs.",
                "input_schema": {"type": "object", "required": ["title", "lesson"], "properties": {"title": {"type": "string"}, "lesson": {"type": "string"}, "kind": {"type": "string", "enum": ["failure", "strategy", "perception", "primitive"]}}},
            },
            self._write_lesson,
        )

    def _reset_episode(self) -> dict[str, Any]:
        result = self._primitives.reset_episode()
        return self.get_env_state(command={"action": "reset_episode"}, result=result, elapsed_s=0.0)

    def _write_lesson(self, *, title: str, lesson: str, kind: str = "failure") -> dict[str, Any]:
        root = self._memory.root / "_internal" / "inbox" / "runtime"
        root.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title.lower()).strip("_")[:80]
        path = root / f"{kind}_{safe or 'lesson'}.md"
        path.write_text(f"---\nscope: global\nkind: {kind}\ntitle: {title}\napplies_when: RoboDojo runtime attempt\nconfidence: single-shot\nevidence:\n  cells: [runtime]\n  attempts: 1\n---\n\n{lesson.strip()}\n")
        return {"success": True, "path": str(path), "message": "Lesson queued for memory merge."}

    @readonly
    def _finish(self, *, status: str, summary: str) -> dict[str, Any]:
        return self._primitives.finish(status=status, summary=summary)

    def _capture_full_observation(self) -> dict[str, Any]:
        """Assemble the full observation (rgb + depth + camera_meta + world_xyz)."""
        env = self._primitives.env
        views: dict[str, dict[str, Any]] = {}
        for camera_name in ROBODOJO_CAMERA_NAMES:
            view: dict[str, Any]
            try:
                rendered = env.render_camera(camera_name, depth=True)
                if not isinstance(rendered, (list, tuple)) or len(rendered) != 2:
                    raise TypeError(
                        "RoboDojo render_camera(depth=True) must return (rgb, depth)"
                    )
                rgb, depth = rendered
                camera_meta = env.get_camera_meta(camera_name)
                depth = np.asarray(depth, dtype=np.float32)
                view = {
                    "rgb": np.asarray(rgb),
                    "depth": depth,
                    "world_xyz": _world_from_depth(depth, camera_meta),
                    "camera_meta": camera_meta,
                }
            except Exception as error:
                # Depth or calibration can be unavailable; keep the RGB view.
                logger.warning(
                    "RoboDojo %s depth/world map unavailable: %s", camera_name, error
                )
                view = {"rgb": np.asarray(env.render_camera(camera_name))}
            views[camera_name] = view
        return {
            "views": views,
            "robot_state": env.last_info["robot_state"],
            "task_name": env.server_meta["task_name"],
            "task_language": env.get_task_language(),
            "depth_unit": "metres",
            "world_frame": "world",
        }

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        status = self._primitives.status()
        self._latest_status = status
        observation = self._capture_full_observation()
        if self._perception_only:
            observation = {
                "views": {
                    name: {"rgb": view["rgb"]}
                    for name, view in observation["views"].items()
                },
                "robot_state": observation["robot_state"],
                "task_name": observation["task_name"],
                "task_language": observation["task_language"],
            }
        record = tools.dump_observation(
            observation,
            env_state=self._state,
            status=status,
            log={
                "command": command,
                "result": result,
                "elapsed_s": elapsed_s,
            },
        )
        if self._dashboard_events.enabled:
            frames = self._primitives.frame_slice(frame_start)
            if frames:
                self._state.save(
                    f"action_{command['action']}.mp4",
                    frames,
                    step=record.step_idx,
                    fps=25,
                )
        return tools.view_env_state(record.step_idx, state=self._state)

    def close(self) -> None:
        """Flush the per-step frame buffer into ``episode.mp4``."""
        frames = self._primitives.stop_recording()
        if frames:
            self._state.save("episode.mp4", frames, step=None, fps=25)

    def solved(self) -> bool:
        """Whether RoboDojo's own reward check reported success."""
        return self._latest_status.get("eval_success") is True

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        self.raise_if_cancelled()
        if name == "render":
            return {"success": True}
        return getattr(self._primitives, name)(**kwargs)

    def write_recipe(self, recipe_tag: str) -> str:
        """Export state-advancing RoboDojo primitives that did not fail."""
        recipe = [
            record.command
            for record in self._state.records()
            if isinstance(record.command, dict)
            and record.command.get("action") in _RECIPE_ACTIONS
            and not (
                isinstance(record.result, dict)
                and (
                    record.result.get("error") or record.result.get("success") is False
                )
            )
        ]
        name = f"{recipe_tag}_recipe.jsonl"
        saved = self._state.save(name, recipe, step=None)
        if saved is None:
            raise RuntimeError(f"failed to save RoboDojo recipe artifact: {name}")
        return str(self._state.artifact_path(name, step=None))
