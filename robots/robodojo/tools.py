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

"""RoboDojo tool schemas and observation artifact helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.session import EnvState, StepRecord
from rpent.tools.toolkit import readonly


def _tool_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, **details},
    }


def _artifact_name(view: str, field: str) -> str:
    suffix = {
        "rgb": ".png",
        "depth": ".npy",
        "world_xyz": ".npy",
        "camera_meta": ".json",
    }[field]
    return f"{view}_{field}{suffix}"


def _load_world_xyz(
    env_state: EnvState,
    *,
    view: str,
    step: int | None,
) -> tuple[dict[str, Any] | None, np.ndarray | None, dict[str, Any] | None]:
    """Load one persisted agent-visible world map without touching the env."""
    requested_step = -1 if step is None else int(step)
    try:
        record = env_state.get(requested_step)
    except Exception:
        return (
            None,
            None,
            _tool_error(
                "state_not_found",
                "The requested RoboDojo state artifact does not exist.",
                step=requested_step,
            ),
        )
    state = record.state
    actual_step = record.step_idx
    views = state.get("artifacts", {})
    if view not in views:
        return (
            None,
            None,
            _tool_error(
                "view_not_found",
                "The requested view is unavailable in this state.",
                view=view,
                available_views=sorted(views) if isinstance(views, dict) else [],
            ),
        )
    world_name = _artifact_name(view, "world_xyz")
    if world_name not in record.artifacts:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_not_found",
                "The requested view has no persisted world map.",
                view=view,
                step_idx=actual_step,
            ),
        )
    try:
        world = env_state.load(world_name, step=actual_step)
    except Exception as error:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_invalid",
                "The persisted world map cannot be read.",
                detail=str(error),
            ),
        )
    if world.ndim != 3 or world.shape[2] != 3:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_shape",
                "A RoboDojo world map must have shape [H,W,3].",
                actual_shape=list(world.shape),
            ),
        )
    return state, np.asarray(world), None


@readonly
def sample_world_xyz(
    env_state: EnvState,
    *,
    view: str,
    pixels: list[list[int]],
    step: int | None = None,
    neighborhood: int = 1,
) -> dict[str, Any]:
    """Return deterministic median world coordinates around image pixels."""
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    radius = int(neighborhood)
    if radius < 0 or radius > 32:
        return _tool_error(
            "invalid_neighborhood",
            "neighborhood must be an integer from 0 through 32.",
        )
    if not isinstance(pixels, list) or not pixels or len(pixels) > 256:
        return _tool_error(
            "invalid_pixels",
            "pixels must contain between 1 and 256 [row,col] pairs.",
        )
    height, width = world.shape[:2]
    samples: list[dict[str, Any]] = []
    for pixel in pixels:
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or not all(isinstance(value, (int, np.integer)) for value in pixel)
        ):
            return _tool_error(
                "invalid_pixel",
                "Every pixel must be an integer [row,col] pair.",
                pixel=pixel,
            )
        row, col = (int(pixel[0]), int(pixel[1]))
        if row < 0 or row >= height or col < 0 or col >= width:
            return _tool_error(
                "pixel_out_of_bounds",
                "The pixel is outside this view's world map. Use the exact "
                "artifact view whose RGB supplied the pixel; do not reuse "
                "high-resolution pixels with a base-resolution view.",
                pixel=[row, col],
                shape=[height, width],
                view=view,
                coordinate_space=view,
                valid_row_range=[0, height - 1],
                valid_col_range=[0, width - 1],
            )
        row_start = max(0, row - radius)
        row_end = min(height, row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(width, col + radius + 1)
        region = world[row_start:row_end, col_start:col_end].reshape(-1, 3)
        finite_counts = np.isfinite(region).sum(axis=0)
        if np.any(finite_counts == 0):
            return _tool_error(
                "no_valid_world_points",
                "The requested pixel neighborhood has no finite xyz coordinate.",
                pixel=[row, col],
                neighborhood=radius,
            )
        xyz = np.nanmedian(region, axis=0)
        samples.append(
            {
                "pixel": [row, col],
                "valid": True,
                "xyz": xyz.tolist(),
                "valid_points": int(np.isfinite(region).all(axis=1).sum()),
                "valid_coordinates": finite_counts.tolist(),
            }
        )
    return {
        "success": True,
        "step_idx": state["step_idx"],
        "view": view,
        "coordinate_space": view,
        "image_shape": [height, width],
        "pixel_order": "row_col",
        "coordinate_order": "xyz",
        "frame": "world",
        "unit": "metre",
        "neighborhood": radius,
        "samples": samples,
    }


#: The jaws sit this far along the gripper's approach axis from the reported
#: end-effector frame. Measured from the finger links (link7/link8).
GRIPPER_REACH_M = 0.115

#: Wrist orientation that points the approach axis straight down (world -z).
#: At the reset orientation the axis points along +y, i.e. horizontally, and a
#: "descend and close" then sweeps the fingers through empty table.
GRASP_QUAT = (0.5, -0.5, 0.5, 0.5)


@readonly
def find_objects(
    env_state: EnvState,
    *,
    view: str = "head",
    step: int = -1,
    table_z: float = 0.765,
    min_height: float = 0.02,
    max_objects: int = 15,
) -> dict[str, Any]:
    """Group the world map into tabletop objects, in pixels *and* metres.

    Each candidate carries the pixel box that frames it in the RGB view and the
    world geometry of the same points, so the instruction can be matched to a
    picture and the grasp can be planned from coordinates without ever
    converting between the two by hand.
    """
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    height, width = world.shape[0], world.shape[1]
    z = world[:, :, 2]
    raised = np.isfinite(z) & (z > float(table_z) + float(min_height))
    raised &= np.isfinite(world[:, :, 0]) & np.isfinite(world[:, :, 1])
    raised &= np.abs(world[:, :, 0]) <= 0.75
    raised &= (world[:, :, 1] >= -0.7) & (world[:, :, 1] <= 0.6)

    # Connected components over the raised mask; objects are contiguous in the
    # image, so this needs no clustering threshold in metres.
    labels = np.zeros((height, width), dtype=np.int32)
    current = 0
    stack: list[tuple[int, int]] = []
    components: list[list[tuple[int, int]]] = []
    for row, col in np.argwhere(raised):
        row, col = int(row), int(col)
        if labels[row, col]:
            continue
        if True:
            current += 1
            stack.append((row, col))
            labels[row, col] = current
            pixels: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < height and 0 <= cc < width:
                            if raised[rr, cc] and not labels[rr, cc]:
                                labels[rr, cc] = current
                                stack.append((rr, cc))
            components.append(pixels)

    objects: list[dict[str, Any]] = []
    for pixels in components:
        if len(pixels) < 12:  # speckle, not an object
            continue
        rows = np.asarray([p[0] for p in pixels])
        cols = np.asarray([p[1] for p in pixels])
        points = world[rows, cols]
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        centre = [float(np.median(xs)), float(np.median(ys)), float(np.median(zs))]
        top = float(np.max(zs))
        # Robust extents: ignore the outermost 5% on each axis.
        ext_x = float(np.percentile(xs, 97.5) - np.percentile(xs, 2.5))
        ext_y = float(np.percentile(ys, 97.5) - np.percentile(ys, 2.5))
        arm = "right" if centre[0] >= 0 else "left"
        base = (0.30, -0.35) if arm == "right" else (-0.30, -0.35)
        reach = float(np.hypot(centre[0] - base[0], centre[1] - base[1]))
        objects.append(
            {
                "pixel_bbox": [
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()) + 1,
                    int(cols.max()) + 1,
                ],
                "pixel_centre": [int(np.median(rows)), int(np.median(cols))],
                "n_points": int(len(pixels)),
                "centre_xyz": [round(v, 4) for v in centre],
                "top_z": round(top, 4),
                "height_above_table": round(top - float(table_z), 4),
                "extent_x": round(ext_x, 4),
                "extent_y": round(ext_y, 4),
                "nearest_arm": arm,
                "reach_m": round(reach, 3),
                # The reported end-effector frame is not the point between the
                # fingers: the jaws sit GRIPPER_REACH_M along the gripper's
                # approach axis, which points *down* only under GRASP_QUAT.
                # These three fields are the waypoints to command directly.
                "grasp_quat": list(GRASP_QUAT),
                # Depth only sees the top surface, so centre[2] is the top of
                # the object, not its mid-height. Grasping there catches a cap
                # or rim and slips; drop into the body by a fraction of the
                # object's height instead.
                "grasp_ee_xyz": [
                    round(centre[0], 4),
                    round(centre[1], 4),
                    round(
                        top
                        - max(0.015, min(0.05, 0.35 * (top - float(table_z))))
                        + GRIPPER_REACH_M,
                        4,
                    ),
                ],
                "approach_ee_xyz": [
                    round(centre[0], 4),
                    round(centre[1], 4),
                    round(top + 0.12 + GRIPPER_REACH_M, 4),
                ],
                "lift_ee_xyz": [
                    round(centre[0], 4),
                    round(centre[1], 4),
                    round(
                        top
                        - max(0.015, min(0.05, 0.35 * (top - float(table_z))))
                        + GRIPPER_REACH_M
                        + 0.28,
                        4,
                    ),
                ],
                # The jaws separate along world X at the default wrist
                # orientation, so ext_x is the width that must fit; a 90 degree
                # yaw swaps in ext_y. Measured open separation is ~0.126 between
                # finger centres, i.e. roughly 0.10 of usable gap.
                "closing_axis": "x",
                "closing_width_m": round(ext_x, 4),
                "jaw_clearance_m": round(0.100 - ext_x, 4),
                "jaw_clearance_if_rotated_m": round(0.100 - ext_y, 4),
                "box_is_reliable": bool(max(ext_x, ext_y) <= 0.11 and top - float(table_z) >= 0.05),
                # The jaws must descend *past* the object, not merely fit around
                # it: a few millimetres of slack is an alignment problem, not a
                # grasp. Measured on this deployment, the fingertips reach about
                # 0.040 below the wrist and the wrist clears the table across
                # most of the workspace, so width is the binding constraint.
                "graspable": bool(
                    min(ext_x, ext_y) <= 0.085
                    and top - float(table_z) >= 0.03
                    and reach < 0.55
                ),
                "needs_wrist_rotation": bool(ext_x > 0.085 >= ext_y),
            }
        )
    objects.sort(key=lambda o: -o["top_z"])
    truncated = len(objects) > int(max_objects)
    return {
        "success": True,
        "step_idx": state.get("step_idx"),
        "view": view,
        "image_shape": [height, width],
        "table_z": float(table_z),
        "objects": objects[: int(max_objects)],
        "truncated": truncated,
        "note": (
            "pixel_bbox is [row_min,col_min,row_max,col_max] in this view's RGB; "
            "centre_xyz and top_z are world metres for the same points. Match the "
            "instruction against the RGB inside a box, then plan with that box's "
            "coordinates. The jaws separate along world X at the default wrist "
            "orientation, so closing_width_m (ext_x) is what must fit the ~0.10 m "
            "gap; when only ext_y fits, needs_wrist_rotation is set and a 90 "
            "degree yaw swaps the axes. jaw_clearance_m reports the slack, and "
            "under about 0.015 m the grasp becomes an alignment lottery. Note a "
            "box only describes a compact upright object: for a thin item lying "
            "diagonally the box is mostly empty air. IMPORTANT: command "
            "grasp_quat with approach_ee_xyz / grasp_ee_xyz / lift_ee_xyz "
            "verbatim. The reported end-effector frame is 0.115 m from the "
            "jaws along the approach axis, and that axis points sideways at the "
            "default orientation, so hovering the wrist over an object and "
            "descending closes the fingers on empty table."
        ),
    }


@readonly
def query_world_map(
    env_state: EnvState,
    *,
    view: str,
    bbox: list[int],
    step: int | None = None,
    max_points: int = 256,
) -> dict[str, Any]:
    """Return deterministic row-major samples and statistics for one bbox."""
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(isinstance(value, (int, np.integer)) for value in bbox)
    ):
        return _tool_error(
            "invalid_bbox",
            "bbox must be [row_start,col_start,row_end,col_end].",
        )
    row_start, col_start, row_end, col_end = map(int, bbox)
    height, width = world.shape[:2]
    if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
        return _tool_error(
            "bbox_out_of_bounds",
            "bbox must be a non-empty half-open region inside this view's "
            "world map. Use the exact artifact view whose RGB supplied the "
            "bbox coordinates.",
            bbox=list(map(int, bbox)),
            shape=[height, width],
            view=view,
            coordinate_space=view,
            valid_bbox=[0, 0, height, width],
        )
    limit = int(max_points)
    if limit < 1 or limit > 4096:
        return _tool_error(
            "invalid_max_points",
            "max_points must be an integer from 1 through 4096.",
        )
    region = world[row_start:row_end, col_start:col_end]
    valid_mask = np.isfinite(region).all(axis=2)
    local_rows, local_cols = np.nonzero(valid_mask)
    if not len(local_rows):
        return _tool_error(
            "no_valid_world_points",
            "The requested region contains no finite world coordinates.",
            bbox=list(map(int, bbox)),
        )
    xyz = region[local_rows, local_cols]
    if len(xyz) > limit:
        indices = np.linspace(0, len(xyz) - 1, limit).astype(int)
    else:
        indices = np.arange(len(xyz))
    points = [
        {
            "pixel": [
                int(row_start + local_rows[index]),
                int(col_start + local_cols[index]),
            ],
            "xyz": xyz[index].tolist(),
        }
        for index in indices
    ]
    return {
        "success": True,
        "step_idx": state["step_idx"],
        "view": view,
        "coordinate_space": view,
        "image_shape": [height, width],
        "bbox": [row_start, col_start, row_end, col_end],
        "bbox_interval": "half_open",
        "pixel_order": "row_col",
        "coordinate_order": "xyz",
        "frame": "world",
        "unit": "metre",
        "valid_points": int(len(xyz)),
        "returned_points": len(points),
        "xyz_min": np.min(xyz, axis=0).tolist(),
        "xyz_max": np.max(xyz, axis=0).tolist(),
        "xyz_median": np.median(xyz, axis=0).tolist(),
        "points": points,
    }


def dump_observation(
    observation: dict[str, Any],
    *,
    env_state: EnvState,
    status: dict[str, Any],
    log: dict[str, Any] | None,
) -> StepRecord:
    """Persist one agent-visible observation without simulator oracle state."""
    step_idx = 0 if env_state.latest_step is None else env_state.latest_step + 1
    paths: dict[str, dict[str, str]] = {}
    view_specs: dict[str, dict[str, Any]] = {}
    for view_name, view in observation["views"].items():
        view_paths: dict[str, str] = {}
        if "rgb" in view:
            name = _artifact_name(view_name, "rgb")
            view_paths["rgb"] = str(env_state.artifact_path(name, step=step_idx))
        for field in ("depth", "world_xyz"):
            if field in view:
                name = _artifact_name(view_name, field)
                view_paths[field] = str(env_state.artifact_path(name, step=step_idx))
        if "camera_meta" in view:
            name = _artifact_name(view_name, "camera_meta")
            view_paths["camera_meta"] = str(
                env_state.artifact_path(name, step=step_idx)
            )
        paths[view_name] = view_paths
        shape_source = next(
            (
                np.asarray(view[field])
                for field in ("rgb", "world_xyz", "depth")
                if field in view
            ),
            None,
        )
        if shape_source is not None and shape_source.ndim >= 2:
            view_specs[view_name] = {
                "coordinate_space": view_name,
                "image_shape": [
                    int(shape_source.shape[0]),
                    int(shape_source.shape[1]),
                ],
                "pixel_order": "row_col",
            }

    state = {
        "step_idx": step_idx,
        "task_name": observation["task_name"],
        "task_language": observation["task_language"],
        "robot_state": observation["robot_state"],
        "episode_status": status,
        "artifacts": paths,
        "view_specs": view_specs,
        "log": log,
    }
    eval_success = status.get("eval_success") is True
    with env_state.record_step(
        state=state,
        terminated=eval_success,
        truncated=False,
        command=(log or {}).get("command"),
        result=(log or {}).get("result"),
        elapsed_s=(log or {}).get("elapsed_s"),
        extras={"task_language": observation.get("task_language")},
    ) as recorded_step:
        for view_name, view in observation["views"].items():
            for field in ("rgb", "depth", "world_xyz", "camera_meta"):
                if field in view:
                    env_state.save(
                        _artifact_name(view_name, field),
                        view[field],
                        step=recorded_step,
                    )
    return env_state.get(step_idx)


@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict[str, Any]:
    try:
        record = state.get(step)
    except Exception as error:
        return {"error": f"state step not available: {error}"}
    result: dict[str, Any] = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
    }
    result["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    for slot, views in (
        ("_image_bytes", ("head",)),
        ("_image_cam_bytes", ("left_wrist",)),
        ("_image_wrist_bytes", ("right_wrist",)),
    ):
        name = next(
            (
                _artifact_name(view, "rgb")
                for view in views
                if _artifact_name(view, "rgb") in record.artifacts
            ),
            None,
        )
        if name is not None:
            try:
                result[slot] = state.load_bytes(name, step=record.step_idx)
            except FileNotFoundError:
                pass
    return result


_ARM = {"type": "string", "enum": ["left", "right"]}
_STEPS = {"type": "integer", "minimum": 1, "maximum": 200, "default": 20}

TOOLS_SPEC = [
    {
        "name": "view_env_state",
        "description": (
            "Read one EnvState step and its synchronized RoboDojo observation "
            "artifacts. Step -1 selects the latest entry. Embeds the head, left "
            "wrist, and right wrist RGB images when available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step number; 0 = initial, -1 = latest.",
                }
            },
        },
    },
    {
        "name": "render",
        "description": "Capture a fresh synchronized RoboDojo agent observation.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "sample_world_xyz",
        "description": (
            "Read persisted same-frame world xyz around [row,col] pixels. "
            "The view is also the pixel coordinate space: use the exact view "
            "whose RGB supplied the pixels. The current state's view_specs "
            "gives each view's [height,width]. This is read-only and does not "
            "render or move the robot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "description": (
                        "Artifact view and pixel coordinate space. It must match "
                        "the RGB image used to choose pixels."
                    ),
                },
                "pixels": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 256,
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "step": {"type": ["integer", "null"]},
                "neighborhood": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32,
                    "default": 1,
                },
            },
            "required": ["view", "pixels"],
        },
    },
    {
        "name": "find_objects",
        "description": (
            "List the tabletop objects with their pixel box in the RGB view AND "
            "their world geometry, already checked against the gripper envelope. "
            "Use this FIRST for any pick or place: it removes the pixel-to-metre "
            "conversion entirely. Read the RGB, decide which returned box holds "
            "the object the instruction names, then plan follow_ee_path straight "
            "from that entry's centre_xyz/top_z. `graspable` reports whether the "
            "target clears the 0.815 fingertip floor, fits the 0.089 m jaw span "
            "on one horizontal axis, and lies within 0.55 m of an arm base; "
            "`needs_wrist_rotation` means only the x axis fits, so rotate the "
            "wrist 90 degrees before descending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "default": "head"},
                "step": {"type": "integer", "default": -1},
                "table_z": {"type": "number", "default": 0.765},
                "min_height": {"type": "number", "default": 0.02},
                "max_objects": {"type": "integer", "minimum": 1, "default": 15},
            },
            "required": [],
        },
    },
    {
        "name": "query_world_map",
        "description": (
            "Read deterministic world-xyz samples from a half-open "
            "[row_start,col_start,row_end,col_end] region. The view is also "
            "the bbox coordinate space and must match the source RGB artifact; "
            "view_specs gives [height,width]. This is read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "description": (
                        "Artifact view and bbox coordinate space. It must match "
                        "the RGB image used to choose the bbox."
                    ),
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "step": {"type": ["integer", "null"]},
                "max_points": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4096,
                    "default": 256,
                },
            },
            "required": ["view", "bbox"],
        },
    },
    {
        "name": "vla_act",
        "description": (
            "Run the configured RoboDojo VLA for one or more 50-action "
            "end-effector chunks using the native task instruction. Only "
            "registered when a VLA endpoint is configured. The optional prompt "
            "is recorded but never sent to the policy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "integer", "minimum": 1, "default": 1},
                "prompt": {"type": ["string", "null"]},
            },
        },
    },
    {
        "name": "move_to",
        "description": (
            "Move one end-effector along a straight world-frame line to xyz "
            "(metres) with an optional wxyz orientation, interpolated over "
            "`steps` native 25 Hz actions solved by the built-in IK. The "
            "gripper keeps its value unless given (0 closed, 1 open). Check "
            "final_dist_m: an unreachable pose leaves the arm short."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "quat": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "gripper": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "steps": _STEPS,
            },
            "required": ["arm", "xyz"],
        },
    },
    {
        "name": "follow_ee_path",
        "description": (
            "Execute a whole planned end-effector path in ONE call. Give the "
            "ordered waypoints of the motion (approach, descend, close, lift, "
            "carry, release) and the trajectory is solved forward from the "
            "planned poses -- each segment starts where the previous waypoint "
            "commanded, not where the arm drifted to -- then run as a single "
            "continuous 25 Hz chunk with smooth start/stop per segment. "
            "Prefer this over a chain of move_to/set_gripper calls: it is one "
            "turn instead of six and the motion does not stall between "
            "waypoints. Per waypoint, omit xyz to hold position, omit quat to "
            "keep the current orientation, and omit gripper to keep its value "
            "(0 closed, 1 open); a gripper-only waypoint closes or opens in "
            "place. Check final_dist_m against the last waypoint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "waypoints": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "xyz": {
                                "type": ["array", "null"],
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "quat": {
                                "type": ["array", "null"],
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "gripper": {
                                "type": ["number", "null"],
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "steps": {"type": ["integer", "null"], "minimum": 1, "maximum": 200},
                        },
                    },
                },
                "default_steps": {**_STEPS, "default": 20},
                "ease": {"type": ["boolean", "null"], "default": True},
            },
            "required": ["arm", "waypoints"],
        },
    },
    {
        "name": "move_delta",
        "description": (
            "Move one end-effector by a world-frame offset dxyz (metres) while "
            "keeping its orientation; a short guarded motion for approach, "
            "lift, and retreat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "dxyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "steps": {**_STEPS, "default": 10},
            },
            "required": ["arm", "dxyz"],
        },
    },
    {
        "name": "rotate_wrist",
        "description": "Rotate one end-effector about world Z by a relative angle in degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "delta_yaw_deg": {"type": "number"},
                "gripper": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "steps": {**_STEPS, "default": 10},
            },
            "required": ["arm", "delta_yaw_deg"],
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Ramp one normalized gripper to val over `steps` native actions "
            "(0 fully closed, 1 fully open)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "val": {"type": "number", "minimum": 0, "maximum": 1},
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "required": ["arm", "val"],
        },
    },
    {
        "name": "release",
        "description": "Open one gripper to 1.0 over 10 native actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "val": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "required": ["arm"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Stop the run. A fresh native status query is authoritative; requesting "
            "success cannot override the RoboDojo reward check (eval_success)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
    },
]
