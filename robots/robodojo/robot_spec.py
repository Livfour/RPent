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

"""RoboDojo robot extension — runtime contracts and runtime hooks.

RoboDojo (https://github.com/RoboDojo-Benchmark/RoboDojo) evaluates bimanual
manipulation on 42 Isaac Sim tasks with two ARX-X5 arms. The simulator runs in
RoboDojo's own Isaac Sim Python (``--robodojo-python``); RPent talks to it over
the common env RPC contract served by :mod:`robots.robodojo.env_server`, which
only needs ``numpy`` and the pure-stdlib ``rpent.utils.rpc`` layer on that
interpreter.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from robots.robodojo.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon

#: Agent-facing camera names mapped onto RoboDojo's native camera names.
ROBODOJO_CAMERAS: dict[str, str] = {
    "head": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}
ROBODOJO_CAMERA_NAMES = tuple(ROBODOJO_CAMERAS)

RoboDojoActionType = Literal["ee", "joint"]

#: Flat action layouts accepted by the env server.
ROBODOJO_ACTION_DIMS = {"ee": 16, "joint": 14}

ROBODOJO_STATUS_KEYS = (
    "eval_success",
    "done",
    "take_action_cnt",
    "step_lim",
    "layout_id",
    "score",
)

ROBODOJO_READ_TIMEOUT_S = 120.0
ROBODOJO_STATE_CHANGE_TIMEOUT_S = 900.0
#: Isaac Sim boots slowly on first launch (extension load + scene build).
ROBODOJO_ENV_READY_TIMEOUT_S = 1800.0

#: Native control rate: one policy action is 10 physics substeps of 4 ms.
ROBODOJO_CONTROL_HZ = 25


@dataclass(frozen=True)
class RoboDojoModelSpec:
    """Values required from an optional RoboDojo VLA endpoint."""

    action_layout: str
    camera_order: tuple[str, ...]
    use_length: int


MODEL_SPEC = RoboDojoModelSpec(
    action_layout="absolute_eef_pose",
    camera_order=("cam_head", "cam_left_wrist", "cam_right_wrist"),
    use_length=50,
)


ROBODOJO_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <layout_id>",
        "fields": (
            {"name": "task_name"},
            {"name": "layout_id", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / layout {layout_id}",
        "output_slug": "{task_name}_l{layout_id}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {"name": "camera", "label": "head camera"},
        {"name": "left_wrist", "label": "left wrist"},
        {"name": "right_wrist", "label": "right wrist"},
    ),
}


def env_runtime_contract(
    *,
    task_name: str,
    layout_group: int,
    layout_id: int,
    max_episode_steps: int = 0,
) -> dict[str, object]:
    """Return the identity required from a RoboDojo EnvServer."""
    return {
        "runtime": "robodojo_isaac_env",
        "task_name": task_name,
        "env_config": "arx_x5",
        "layout_group": int(layout_group),
        "layout_id": int(layout_id),
        "action_layouts": ["ee16", "joint14"],
        "control_hz": ROBODOJO_CONTROL_HZ,
        "execution": {
            "reset": True,
            "step": True,
            "chunk_step": True,
            "chunk_step_all_frames": True,
            "step_limit_override": int(max_episode_steps),
        },
        "extensions": {
            "render_camera": {
                "camera_names": list(ROBODOJO_CAMERA_NAMES),
                "metric_depth": True,
            },
            "get_camera_meta": True,
            "get_task_language": True,
            "status": True,
        },
    }


def vla_runtime_contract() -> dict[str, object]:
    """Return the identity required from an optional RoboDojo VLA server."""
    return {
        "runtime": "robodojo_vla",
        "action_layout": MODEL_SPEC.action_layout,
        "camera_order": list(MODEL_SPEC.camera_order),
        "use_length": MODEL_SPEC.use_length,
    }


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="robodojo",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=ROBODOJO_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the RoboDojo toolkit for the current session."""
    from robots.robodojo.toolkit import RoboDojoToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robodojo"),
    )
    return RoboDojoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument(
        "--task-name",
        required=required,
        help="RoboDojo task module name, e.g. general_pickup or stack_bowls.",
    )
    parser.add_argument(
        "--layout-id",
        type=int,
        default=0,
        help=(
            "Index of the published scene layout inside the layout group "
            "(Assets/Eval_Layout/RoboDojo/arx_x5/<group>/<task>_<id>.json)."
        ),
    )
    parser.add_argument(
        "--layout-group",
        "--seed",
        dest="layout_group",
        type=int,
        default=0,
        help="RoboDojo layout group directory (the official protocol's seed).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help=(
            "Override the task's native step_lim (policy actions at 25 Hz). "
            "0 keeps the native per-task limit."
        ),
    )
    parser.add_argument(
        "--robodojo-root",
        default=os.environ.get("ROBODOJO_ROOT"),
        help=(
            "RoboDojo checkout containing env/, task/, XPolicyLab/ and Assets/. "
            "Defaults to ROBODOJO_ROOT."
        ),
    )
    parser.add_argument(
        "--robodojo-python",
        default=os.environ.get("ROBODOJO_PYTHON"),
        help=(
            "Python interpreter of RoboDojo's Isaac Sim environment used to run "
            "the env server. Defaults to ROBODOJO_PYTHON."
        ),
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=float(os.environ.get("ROBODOJO_RENDER_SCALE", "1.0")),
        help="Scale applied to every camera resolution in (0, 1].",
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help=(
            "Optional RPC endpoint of a RoboDojo VLA server (http://host:port). "
            "When omitted the toolkit exposes analytic primitives only."
        ),
    )
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument(
        "--env-cuda-device",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for the Isaac Sim env server.",
    )
    parser.add_argument(
        "--vla-cuda-device",
        default=None,
        help="Reserved for a locally spawned VLA server (not spawned yet).",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.task_name:
        raise ValueError("--task-name is required")
    if not 0.0 < float(args.render_scale) <= 1.0:
        raise ValueError("--render-scale must be in (0, 1]")
    env_cuda_device, vla_cuda_device = _resolve_cuda_devices(args)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_robodojo_{args.task_name}_l{args.layout_id}"
        )
    output_dir = Path(output_dir)
    recipe_tag = f"robodojo_{args.task_name}_l{args.layout_id}"
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else get_memory_dir("robodojo")
    )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task_name": args.task_name,
            "layout_id": args.layout_id,
            "layout_group": args.layout_group,
            "instruction": "<native instruction from state_00>",
            "memory_dir": str(memory_dir),
            "reference_tag": f"{args.task_name}_l0",
            "vla_available": args.vla_endpoint is not None,
        },
        task_desc={
            "env": "robodojo",
            "task_name": args.task_name,
            "layout_id": int(args.layout_id),
            "layout_group": int(args.layout_group),
            "max_episode_steps": int(args.max_episode_steps),
            "instruction": None,
            "action_layout": MODEL_SPEC.action_layout,
            "env_cuda_device": env_cuda_device,
            "vla_cuda_device": vla_cuda_device,
            "vla_endpoint": args.vla_endpoint,
        },
    )


def _resolve_cuda_devices(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    shared = getattr(args, "cuda_device", None)
    env_device = getattr(args, "env_cuda_device", None)
    vla_device = getattr(args, "vla_cuda_device", None)
    if shared is not None and (env_device is not None or vla_device is not None):
        raise ValueError(
            "--cuda-device cannot be combined with --env-cuda-device or "
            "--vla-cuda-device"
        )
    if shared is not None:
        value = str(shared)
        return value, value
    return (
        str(env_device) if env_device is not None else None,
        str(vla_device) if vla_device is not None else None,
    )


def _resolve_robodojo_root(args: argparse.Namespace) -> Path:
    configured = getattr(args, "robodojo_root", None)
    if not configured:
        raise ValueError(
            "--robodojo-root is required when launching the local env server; "
            "set ROBODOJO_ROOT or pass the option explicitly"
        )
    root = Path(configured).expanduser().resolve()
    required = [
        root / "env",
        root / "task" / "RoboDojo" / "tasks" / f"{args.task_name}.py",
        root / "XPolicyLab",
        root / "Assets" / "Eval_Layout" / "RoboDojo",
        root / "Assets" / "Robots",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"RoboDojo checkout is incomplete: {missing}")
    return root


def _resolve_robodojo_python(args: argparse.Namespace) -> str:
    configured = getattr(args, "robodojo_python", None)
    if not configured:
        raise ValueError(
            "--robodojo-python is required when launching the local env server; "
            "set ROBODOJO_PYTHON to the Isaac Sim interpreter of the RoboDojo "
            "environment (e.g. <RoboDojo>/.pixi/envs/default/bin/python)"
        )
    python = shutil.which(str(Path(configured).expanduser()))
    if python is None or not os.access(python, os.X_OK):
        raise ValueError(f"RoboDojo python is not executable: {configured}")
    return python


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    """Initialize every RoboDojo component, or only ``components`` when given."""
    available = {"env", "vla"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown RoboDojo runtime components: {sorted(unknown)}")

    owned_daemons: dict[str, ProcessDaemon] = {}
    primitives_kwargs: dict[str, Any] = {}

    if "env" in selected:
        env_daemon, env_rpc = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "env",
            lambda: _spawn_env_server(args, output_dir),
        )
        env_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            "env",
            env_rpc,
            env_daemon,
            ROBODOJO_ENV_READY_TIMEOUT_S if env_daemon is not None else 300.0,
            post_fn=lambda: _build_env_runtime_kwargs(args, env_rpc),
        )
        primitives_kwargs.update(env_kwargs)

    if "vla" in selected and args.vla_endpoint is not None:
        from rpent.dashboard.events import RuntimeStatusEvent

        dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
        try:
            primitives_kwargs.update(_build_vla_runtime_kwargs(args.vla_endpoint))
        except Exception as exc:
            from rpent.robots.runtime import stop_owned_daemons

            stop_owned_daemons(owned_daemons, dashboard_events)
            dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
            raise RuntimeError(f"[vla] connect failed: {exc}") from exc
        dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))

    return list(owned_daemons.values()), primitives_kwargs


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple["ProcessDaemon | None", Any]:
    if args.env_endpoint is not None:
        from rpent.utils.rpc import make_rpc_client

        return None, make_rpc_client(args.env_endpoint)

    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    env_cuda_device, _ = _resolve_cuda_devices(args)
    root = _resolve_robodojo_root(args)
    python = _resolve_robodojo_python(args)
    repo_root = get_repo_root()
    host, env_port = "127.0.0.1", pick_free_port()
    # The Isaac interpreter has no rpent install: expose the checkout on
    # PYTHONPATH so ``rpent.utils.rpc`` (stdlib + numpy) imports there.
    python_path = os.pathsep.join(
        [str(repo_root), str(root), str(root / "XPolicyLab")]
        + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    )
    env_daemon = ProcessDaemon(
        "robodojo_env_server",
        [
            python,
            str(repo_root / "robots" / "robodojo" / "env_server.py"),
            "--robodojo-root",
            str(root),
            "--task-name",
            args.task_name,
            "--layout-group",
            str(int(args.layout_group)),
            "--layout-id",
            str(int(args.layout_id)),
            "--max-episode-steps",
            str(int(args.max_episode_steps)),
            "--render-scale",
            str(float(args.render_scale)),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(env_port),
            "--parent-watch",
        ],
        env_overrides={
            "PYTHONPATH": python_path,
            "PYTHONNOUSERSITE": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "ROBODOJO_ROOT": str(root),
            **(
                {"CUDA_VISIBLE_DEVICES": env_cuda_device}
                if env_cuda_device is not None
                else {}
            ),
        },
        log_path=str(output_dir / "robodojo_env_server.log"),
    )
    env_rpc = HttpRpcClient(f"http://{host}:{env_port}")
    env_daemon.start()
    return env_daemon, env_rpc


def _build_env_runtime_kwargs(
    args: argparse.Namespace,
    env_rpc: Any,
) -> dict[str, Any]:
    from robots.robodojo.env_client import RoboDojoEnvClient

    return {
        "env": RoboDojoEnvClient(
            env_rpc,
            expected_meta=env_runtime_contract(
                task_name=args.task_name,
                layout_group=int(args.layout_group),
                layout_id=int(args.layout_id),
                max_episode_steps=int(args.max_episode_steps),
            ),
        ),
        "layout_id": int(args.layout_id),
    }


def _build_vla_runtime_kwargs(endpoint: str) -> dict[str, Any]:
    from robots.robodojo.vla_client import RoboDojoVLAClient
    from rpent.utils.rpc import make_rpc_client

    model = RoboDojoVLAClient(make_rpc_client(endpoint))
    model.validate_contract(vla_runtime_contract())
    return {"model": model}
