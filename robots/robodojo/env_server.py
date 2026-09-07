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

"""RPC server owning one RoboDojo Isaac Sim environment.

This file runs inside RoboDojo's Isaac Sim interpreter, not the RPent venv:
it only imports ``numpy`` and the stdlib-based ``rpent.utils.rpc`` layer from
the RPent checkout on ``PYTHONPATH``. Kit must boot before any Isaac import,
and every simulator call runs on the main thread
(:class:`~rpent.utils.rpc.main_thread_serve.MainThreadServeMixin`).

The Isaac bring-up mirrors RoboDojo's ``src/eval_client/main.py`` and the
VLAForge RoboDojo adapter: RoboDojo's namespace-only ``utils`` package must win
over Kit's regular ``utils``, the checkout root must precede ``XPolicyLab`` on
``sys.path``, Isaac 4.5 lacks two semantics helpers RoboDojo calls, and CPU
physics needs PhysX-to-USD synchronization for cameras to see motion.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.robodojo.robot_spec import (  # noqa: E402
    ROBODOJO_ACTION_DIMS,
    ROBODOJO_CAMERAS,
    RoboDojoActionType,
    env_runtime_contract,
)
from rpent.robots.components.env_facade_base import BaseEnvFacade  # noqa: E402
from rpent.utils.logging import get_logger  # noqa: E402
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin  # noqa: E402

logger = get_logger("robodojo_env_server")

KIT_ARGS = "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"

#: RoboDojo's default renderer ("quality", DLAA, GI, reflections) needs several
#: gigabytes of VRAM for a textured scene; this profile fits next to other GPU
#: tenants at the cost of image fidelity.
LOW_MEMORY_RENDER = {
    "enable_translucency": False,
    "enable_reflections": False,
    "enable_global_illumination": False,
    "antialiasing_mode": "FXAA",
    "dlss_mode": 0,
    "rendering_mode": "performance",
    "carb_settings": {
        "/rtx-transient/resourcemanager/enableTextureStreaming": True,
        "/rtx-transient/resourcemanager/texturestreaming/memoryBudget": 0.03,
        "/rtx/sceneDb/allowDuplicateAhsInvocation": False,
    },
}

_EE_PARTS = (
    "left_ee_pose",
    "left_ee_joint_state",
    "right_ee_pose",
    "right_ee_joint_state",
)
_EE_DIMS = (7, 1, 7, 1)
_JOINT_PARTS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
_JOINT_DIMS = (6, 1, 6, 1)


# ---------------------------------------------------------------------------
# Isaac bring-up helpers
# ---------------------------------------------------------------------------


def bind_robodojo_utils(root: Path) -> None:
    """Bind RoboDojo's namespace-only ``utils`` ahead of regular packages."""
    utils_root = root / "utils"
    if not (utils_root / "load_file.py").is_file():
        raise FileNotFoundError(f"RoboDojo utils not found: {utils_root}")
    for name in tuple(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    package = types.ModuleType("utils")
    package.__package__ = "utils"
    package.__path__ = [str(utils_root)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "utils", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["utils"] = package


def check_utils_resolution(root: Path) -> None:
    """Fail fast if ``utils.load_file`` resolved to XPolicyLab's subset copy."""
    import utils.load_file as load_file_module

    expected = (root / "utils" / "load_file.py").resolve()
    actual = Path(getattr(load_file_module, "__file__", "")).resolve()
    if actual != expected:
        raise RuntimeError(
            f"utils.load_file resolved to {actual}, expected {expected}; "
            "RoboDojo root must precede XPolicyLab on sys.path"
        )


def patch_isaac45_semantics() -> None:
    """Provide the Isaac Sim 5.x semantics helpers RoboDojo uses on 4.5."""
    import isaacsim.core.utils.semantics as semantics

    if not hasattr(semantics, "add_labels"):

        def add_labels(
            prim: Any, labels: list[str], instance_name: str = "class", **_: Any
        ) -> None:
            for index, label in enumerate(labels):
                suffix = "" if index == 0 else f"_{index}"
                semantics.add_update_semantics(
                    prim, label, type_label=instance_name, suffix=suffix
                )

        semantics.add_labels = add_labels
    if not hasattr(semantics, "remove_labels"):

        def remove_labels(
            prim: Any, include_descendants: bool = False, **_: Any
        ) -> None:
            semantics.remove_all_semantics(prim, recursive=include_descendants)

        semantics.remove_labels = remove_labels


def patch_isaac45_modules() -> None:
    """Alias Isaac Sim 5.x module paths RoboDojo imports onto the 4.5 layout.

    Isaac Sim 4.5 keeps ``_SinglePrimWrapper`` under
    ``isaacsim.core.prims.impl._impl``; 5.x exposes it one level up.
    """
    import importlib

    for name in ("single_prim_wrapper",):
        public = f"isaacsim.core.prims.impl.{name}"
        if public in sys.modules:
            continue
        try:
            importlib.import_module(public)
        except ModuleNotFoundError:
            sys.modules[public] = importlib.import_module(
                f"isaacsim.core.prims.impl._impl.{name}"
            )
    # Isaac 5.x cameras expose a lens-distortion model; RoboDojo only ever
    # selects "pinhole", which is the 4.5 default.
    from isaacsim.sensors.camera import Camera

    if not hasattr(Camera, "set_lens_distortion_model"):
        Camera.set_lens_distortion_model = lambda self, model: None
    # Isaac 5.x aperture setters take a trailing ``maintain_fov`` flag and work
    # before the render product exists; 4.5 recomputes the FOV from the render
    # product, which RoboDojo has not created yet. Write the USD attribute.
    for setter_name, attribute in (
        ("set_horizontal_aperture", "horizontalAperture"),
        ("set_vertical_aperture", "verticalAperture"),
    ):
        if getattr(getattr(Camera, setter_name), "_rpent_lenient", False):
            continue

        def lenient(self: Any, value: float, *_: Any, _attr: str = attribute) -> None:
            self.prim.GetAttribute(_attr).Set(float(value))

        lenient._rpent_lenient = True  # type: ignore[attr-defined]
        setattr(Camera, setter_name, lenient)


def patch_lean_curobo_planner() -> None:
    """Keep only the cuRobo IK solver RPent's end-effector actions need.

    RoboDojo's ``CuroboPlanner`` also warms up single and batch motion
    planners with CUDA graphs, which costs gigabytes of GPU memory; RPent
    interpolates end-effector poses itself and only calls ``solve_ik``.
    """
    from env.planner_manager import curobo_planner as module

    class _NoBatchMotionPlanner:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def warmup(self, *_: Any, **__: Any) -> None:
            return None

        def destroy(self) -> None:
            return None

    module.BatchMotionPlanner = _NoBatchMotionPlanner
    module.MotionPlanner.warmup = lambda self, *args, **kwargs: None
    module.CuroboPlanner._build_batch_pad_templates = lambda self: None
    module.CuroboPlanner._prewarm_alternate_modes = lambda self: None


def launch_isaac_app(root: Path) -> Any:
    """Boot Kit against the RoboDojo checkout and return the running app."""
    for import_root in (root / "XPolicyLab", root):
        path = str(import_root)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    # RoboDojo resolves eval_result/ and its Assets/ relative to the checkout.
    os.chdir(root)
    # Kit bundles an older ``warp`` (1.5 on Isaac 4.5) that would shadow the
    # pip warp cuRobo's kernels need; importing it first pins the pip version.
    import warp

    pip_warp = warp.__version__
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True, kit_args=KIT_ARGS)
    if warp.__version__ != pip_warp:
        raise RuntimeError(
            f"Kit replaced warp {pip_warp} with {warp.__version__}; cuRobo needs "
            "the pip version"
        )
    logger.info("Isaac Sim booted with warp %s", pip_warp)
    return launcher.app


def ensure_physx_render_sync(env: Any) -> None:
    """Keep rendered USD transforms in sync when RoboDojo runs CPU physics."""
    if bool(getattr(env, "use_fabric", False)):
        return
    physics_context = getattr(env, "physics_context", None)
    if physics_context is None:
        raise RuntimeError("RoboDojo environment has no PhysicsContext")
    settings = physics_context.get_physx_update_transformations_settings()
    if not bool(settings[0]):
        physics_context.set_physx_update_transformations_settings(update_to_usd=True)
    verified = physics_context.get_physx_update_transformations_settings()
    if not bool(verified[0]):
        raise RuntimeError(
            "RoboDojo CPU physics is not updating USD transforms; camera "
            "observations would be stale"
        )


def _scaled_resolution(resolution: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = resolution
    return (max(16, int(round(width * scale))), max(16, int(round(height * scale))))


class _NoopModelClient:
    """Stand-in for RoboDojo's legacy WebSocket policy client."""

    def __init__(self, **_: Any) -> None:
        pass

    def call(self, **_: Any) -> None:
        return None


def build_env(
    *,
    root: Path,
    app: Any,
    task_name: str,
    layout_group: int,
    render_scale: float,
    low_memory_render: bool = False,
) -> Any:
    """Mirror ``src/eval_client/main.py`` config assembly for one env."""
    patch_isaac45_semantics()
    patch_isaac45_modules()
    bind_robodojo_utils(root)
    patch_lean_curobo_planner()
    import src.eval_client.eval_env as eval_env_module
    from env.camera_manager import camera_manager as camera_manager_module
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    if render_scale != 1.0:
        for template in camera_manager_module.REAL_MAP.values():
            template["resolution"] = _scaled_resolution(
                tuple(template["resolution"]), render_scale
            )

    config_path = Path(ENV_CONFIG_PATH)
    eval_cfg = load_yaml(str(config_path / "arx_x5.yml"))
    eval_cfg.update(
        {
            "task_name": task_name,
            "num_envs": 1,
            "device_id": 0,
            "eval_batch": False,
            "policy_name": "rpent",
            "additional_info": "rpent",
            "seed": int(layout_group),
            "physx_monitor_enabled": False,
        }
    )
    # Metric depth plus camera matrices let the agent back-project pixels.
    vision = eval_cfg.setdefault("observation", {}).setdefault("vision", {})
    vision["depth"] = True
    vision["intrinsic_matrix"] = True
    vision["extrinsic_matrix"] = True

    task_registry = __import__(
        f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"]
    )
    benchmark_path = Path(ROOT_DIR) / "task" / BENCHMARK
    camera_cfg = load_yaml(
        str(config_path / "camera" / f"{eval_cfg['config']['camera']}.yml")
    )
    for camera_name, annotators in camera_cfg.get("annotator", {}).items():
        if camera_name == "common" or not isinstance(annotators, dict):
            continue
        annotators.setdefault(
            "distance_to_image_plane_capture",
            {"type": "distance_to_image_plane", "device": "cpu"},
        )
    cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                str(config_path / "sim" / f"{eval_cfg['config']['sim']}.yml")
            ),
            "scene": load_yaml(
                str(config_path / "scene" / f"{eval_cfg['config']['scene']}.yml")
            ),
            "camera": camera_cfg,
            "robot": load_yaml(
                str(config_path / "robot" / f"{eval_cfg['config']['robot']}.yml")
            ),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    str(benchmark_path / "config"), task_name
                )
            ),
            "deploy_cfg": {"port": 1, "host": "127.0.0.1", "policy_name": "rpent"},
            "eval_cfg": eval_cfg,
        }
    )
    OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
    if low_memory_render:
        OmegaConf.update(cfg, "sim.render", LOW_MEMORY_RENDER, force_add=True)
    cfg = process_randomization(cfg)
    cfg, eval_num = process_config(cfg, task_name=task_name)
    eval_cfg["eval_num"] = int(eval_num)
    OmegaConf.update(
        cfg,
        "camera.default_frequency",
        eval_cfg.get("observation", {}).get("collect_freq", 0),
        force_add=True,
    )
    cfg.sim.seed = [0]

    eval_env_module.WsModelClient = _NoopModelClient
    check_utils_resolution(root)
    env = eval_env_module.create_eval_env(cfg, app)
    env.model_client = _NoopModelClient()
    # RPent records its own videos; skip EvalEnv's per-camera ffmpeg streams.
    env._stream_vision = lambda *args, **kwargs: None
    env.save_video = lambda *args, **kwargs: None
    return env


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


def _to_numpy_tree(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_actions(actions: Any, *, action_type: RoboDojoActionType) -> np.ndarray:
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


def _unflatten(action: np.ndarray, action_type: RoboDojoActionType) -> dict[str, Any]:
    parts, dims = (
        (_EE_PARTS, _EE_DIMS) if action_type == "ee" else (_JOINT_PARTS, _JOINT_DIMS)
    )
    pieces = np.split(action.astype(np.float32), np.cumsum(dims)[:-1])
    unpacked = dict(zip(parts, pieces))
    if action_type == "ee":
        for key in ("left_ee_pose", "right_ee_pose"):
            quat = unpacked[key][3:7]
            norm = float(np.linalg.norm(quat))
            if not np.isfinite(norm) or norm < 1e-6:
                raise ValueError(f"{key} quaternion has zero norm")
            unpacked[key][3:7] = quat / norm
    for key in ("left_ee_joint_state", "right_ee_joint_state"):
        unpacked[key] = np.clip(unpacked[key], 0.0, 1.0)
    return {key: value.tolist() for key, value in unpacked.items()}


class RoboDojoEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Expose one RoboDojo ``EvalEnv`` through the common env RPC contract."""

    def __init__(
        self,
        env: Any,
        *,
        metadata: dict[str, Any],
        layout_id: int,
        max_episode_steps: int,
    ):
        self._env = env
        self._metadata = dict(metadata)
        self._layout_id = int(layout_id)
        self._max_episode_steps = int(max_episode_steps)
        self._last_obs: dict[str, Any] | None = None
        self._episode_started = False
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.status"] = self.status
        self._readonly_methods.add("env.status")

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._metadata)

    # ---- helpers ------------------------------------------------------

    def _raw_obs(self) -> dict[str, Any]:
        return self._env.get_obs()

    def _refresh_obs(self) -> dict[str, Any]:
        self._last_obs = self._raw_obs()
        return self._last_obs

    def _require_obs(self) -> dict[str, Any]:
        if self._last_obs is None:
            raise RuntimeError("RoboDojo episode has not been reset")
        return self._last_obs

    def _episode_status(self) -> dict[str, Any]:
        env = self._env
        success = bool(env.success[0]) if getattr(env, "success", None) else False
        done = bool(env.end_flag[0]) if getattr(env, "end_flag", None) else False
        score = 1.0 if success else 0.0
        if not success and hasattr(env, "reward_manager"):
            try:
                score = float(env.reward_manager.get_score()[0]) / 100.0
            except Exception:
                score = 0.0
        return {
            "eval_success": success,
            "done": done,
            "take_action_cnt": int(env.take_action_cnt[0]),
            "step_lim": int(env.step_lim),
            "layout_id": self._layout_id,
            "score": score,
        }

    def _robot_state(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = obs["state"]
        left_pose = np.asarray(state["left_ee_pose"], dtype=np.float64).reshape(7)
        right_pose = np.asarray(state["right_ee_pose"], dtype=np.float64).reshape(7)
        left_grip = float(np.asarray(state["left_ee_joint_state"]).reshape(-1)[0])
        right_grip = float(np.asarray(state["right_ee_joint_state"]).reshape(-1)[0])
        return {
            "left_eef_pose": left_pose,
            "right_eef_pose": right_pose,
            "left_gripper": left_grip,
            "right_gripper": right_grip,
            "left_arm_qpos": np.asarray(
                state["left_arm_joint_state"], dtype=np.float64
            ).reshape(6),
            "right_arm_qpos": np.asarray(
                state["right_arm_joint_state"], dtype=np.float64
            ).reshape(6),
            "ee16": np.concatenate([left_pose, [left_grip], right_pose, [right_grip]]),
        }

    def _observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        head = obs["vision"][ROBODOJO_CAMERAS["head"]]["color"]
        return {"main_images": np.ascontiguousarray(np.asarray(head)[:, :, :3])}

    def _info(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "robot_state": self._robot_state(obs),
            "episode_status": self._episode_status(),
            "instruction": str(obs.get("instruction", "")),
        }

    # ---- contract ------------------------------------------------------

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._env
        if self._episode_started:
            raise RuntimeError(
                "RoboDojo runs one episode per env server; restart the server "
                "for a fresh layout"
            )
        total = len(env.seed_manager.seed_list)
        if not 0 <= self._layout_id < total:
            raise ValueError(
                f"layout_id {self._layout_id} outside the {total} published layouts"
            )
        started = time.monotonic()
        env.reset(seed=[self._layout_id])
        ensure_physx_render_sync(env)
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        if self._max_episode_steps > 0:
            env.step_lim = int(self._max_episode_steps)
        self._episode_started = True
        obs = self._refresh_obs()
        logger.info(
            "RoboDojo episode ready: task=%s layout=%d step_lim=%s in %.1fs",
            self._metadata["task_name"],
            self._layout_id,
            env.step_lim,
            time.monotonic() - started,
        )
        info = self._info(obs)
        info["requested_layout_id"] = self._layout_id
        return self._observation(obs), info

    def _take_one(self, action: np.ndarray, action_type: RoboDojoActionType) -> bool:
        self._env.take_action(_unflatten(action, action_type))
        return bool(self._env.is_episode_end())

    def step(
        self, action, *, action_type: RoboDojoActionType = "ee"
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        array = _validate_actions(action, action_type=action_type)
        if array.shape[0] != 1:
            raise ValueError("RoboDojo common action must be a single action")
        payload, rewards, terminated, truncated, info = self.chunk_step(
            array, action_type=action_type, return_all_frames=False
        )
        return payload, rewards, terminated, truncated, info

    def chunk_step(
        self,
        actions,
        *,
        action_type: RoboDojoActionType = "ee",
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        array = _validate_actions(actions, action_type=action_type)
        self._require_obs()
        frames: list[np.ndarray] = []
        executed = 0
        terminated = False
        truncated = False
        for action in array:
            status = self._episode_status()
            if status["done"]:
                break
            done = self._take_one(action, action_type)
            executed += 1
            if return_all_frames:
                obs = self._refresh_obs()
                frames.append(self._observation(obs)["main_images"])
            if done:
                status = self._episode_status()
                terminated = bool(status["eval_success"])
                truncated = not terminated
                break
        obs = self._refresh_obs()
        observation: Any = self._observation(obs)
        if return_all_frames:
            observation = {"frames": frames, "final": observation}
        info = {
            **self._info(obs),
            "action_type": action_type,
            "requested_actions": int(len(array)),
            "executed_actions": executed,
        }
        reward = 1.0 if terminated else 0.0
        return observation, reward, terminated, truncated, info

    def render_camera(self, camera_name: str, depth: bool = False) -> Any:
        obs = self._require_obs()
        native = ROBODOJO_CAMERAS[camera_name]
        camera = obs["vision"][native]
        rgb = np.ascontiguousarray(np.asarray(camera["color"])[:, :, :3])
        if not depth:
            return rgb
        raw = camera.get("depth")
        if raw is None:
            raise RuntimeError(f"RoboDojo camera {camera_name!r} has no depth")
        depth_map = np.asarray(raw, dtype=np.float32)
        if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
            depth_map = depth_map[..., 0]
        depth_map = depth_map.copy()
        depth_map[~np.isfinite(depth_map)] = np.nan
        return rgb, depth_map

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        obs = self._require_obs()
        native = ROBODOJO_CAMERAS[camera_name]
        camera = obs["vision"][native]
        height, width = np.asarray(camera["color"]).shape[:2]
        intrinsic = camera.get("intrinsic_matrix")
        cam2world = camera.get("extrinsic_matrix")
        if intrinsic is None or cam2world is None:
            raise RuntimeError(
                f"RoboDojo camera {camera_name!r} has no intrinsic/extrinsic matrices"
            )
        return {
            "intrinsic_K": np.asarray(intrinsic, dtype=np.float64),
            "cam2world_gl": np.asarray(cam2world, dtype=np.float64),
            "width": int(width),
            "height": int(height),
        }

    def get_task_language(self) -> str:
        return str(self._require_obs().get("instruction", ""))

    def status(self) -> dict[str, Any]:
        return self._episode_status()

    def close(self) -> None:
        env = self._env

        def _close() -> None:
            try:
                env.close()
            except Exception:
                logger.exception("RoboDojo env close failed")

        # EvalEnv.close() can hang well past a minute; bound it so the daemon
        # exits and releases the GPU.
        closer = threading.Thread(target=_close, name="robodojo-env-close", daemon=True)
        closer.start()
        closer.join(timeout=60)
        if closer.is_alive():
            logger.warning("RoboDojo env close still hanging after 60s; exiting")
        os._exit(0)

    def _dispatch(self, method: str, args: tuple, kwargs: dict, **extra: Any) -> Any:
        return _to_numpy_tree(super()._dispatch(method, args, kwargs, **extra))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--layout-group", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument(
        "--low-memory-render",
        action="store_true",
        help="Use the performance renderer with a small texture budget.",
    )
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    root = Path(args.robodojo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboDojo checkout not found: {root}")
    app = launch_isaac_app(root)
    env = build_env(
        root=root,
        app=app,
        task_name=args.task_name,
        layout_group=args.layout_group,
        render_scale=args.render_scale,
        low_memory_render=args.low_memory_render,
    )
    facade = RoboDojoEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name=args.task_name,
            layout_group=args.layout_group,
            layout_id=args.layout_id,
            max_episode_steps=args.max_episode_steps,
        ),
        layout_id=args.layout_id,
        max_episode_steps=args.max_episode_steps,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
