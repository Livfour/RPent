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

"""Offline contracts for the RoboDojo robot spec, env client, and prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robots.robodojo import robot_spec
from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.prompt_bundle import system_prompt, user_prompt
from rpent.prompt.utils import format_prompt
from rpent.robots import get_robot_spec


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--memory-dir", default=None)
    robot_spec.get_robot_spec().add_cli_args(parser, False)
    return parser.parse_args(argv)


def test_robot_is_discoverable_and_parses_a_run_config(tmp_path: Path) -> None:
    spec = get_robot_spec("robodojo")
    assert spec.name == "robodojo"
    args = _parse(
        [
            "--task-name",
            "general_pickup",
            "--layout-id",
            "4",
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    config = spec.parse_config(args)
    assert config.recipe_tag == "robodojo_general_pickup_l4"
    assert config.output_dir == tmp_path / "run"
    assert config.task_desc["layout_group"] == 1
    assert config.task_desc["layout_id"] == 4
    assert config.task_desc["action_layout"] == "absolute_eef_pose"
    assert config.prompt_vars["vla_available"] is False


def test_parse_config_rejects_conflicting_cuda_flags_and_bad_scale() -> None:
    spec = robot_spec.get_robot_spec()
    args = _parse(
        ["--task-name", "stack_bowls", "--cuda-device", "0", "--env-cuda-device", "1"]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        spec.parse_config(args)
    args = _parse(["--task-name", "stack_bowls", "--render-scale", "1.5"])
    with pytest.raises(ValueError, match="render-scale"):
        spec.parse_config(args)


def test_env_runtime_contract_pins_the_layout_and_action_layouts() -> None:
    meta = robot_spec.env_runtime_contract(
        task_name="general_pickup", layout_group=0, layout_id=2, max_episode_steps=50
    )
    assert meta["runtime"] == "robodojo_isaac_env"
    assert meta["layout_id"] == 2
    assert meta["action_layouts"] == ["ee16", "joint14"]
    assert meta["execution"]["step_limit_override"] == 50
    assert set(meta["extensions"]["render_camera"]["camera_names"]) == {
        "head",
        "left_wrist",
        "right_wrist",
    }


class FakeRpc:
    def __init__(
        self, meta: dict[str, Any], reported_layout: int | None = None
    ) -> None:
        self.meta = meta
        self.reported_layout = (
            meta["layout_id"] if reported_layout is None else reported_layout
        )
        self.calls: list[tuple[str, tuple, dict]] = []
        self.count = 0

    def _info(self, *, done: bool = False, success: bool = False) -> dict[str, Any]:
        return {
            "robot_state": {"ee16": np.zeros(16)},
            "episode_status": {
                "eval_success": success,
                "done": done,
                "take_action_cnt": self.count,
                "step_lim": 200,
                "layout_id": self.reported_layout,
                "score": 0.0,
            },
            "instruction": "Pick up the bowl by 10 cm.",
        }

    def call(self, method: str, args: tuple = (), kwargs: dict | None = None, **_):
        self.calls.append((method, args, kwargs or {}))
        if method == "env.get_env_meta":
            return self.meta
        if method == "env.reset":
            return ({"main_images": np.zeros((2, 2, 3))}, self._info())
        if method == "env.chunk_step":
            n = len(args[0])
            self.count += n
            done = self.count >= 6
            payload: Any = {"main_images": np.zeros((2, 2, 3))}
            if (kwargs or {}).get("return_all_frames"):
                payload = {"frames": [payload["main_images"]] * n, "final": payload}
            info = {**self._info(done=done, success=done), "executed_actions": n}
            return (payload, float(done), done, False, info)
        if method == "env.status":
            return self._info()["episode_status"]
        if method == "env.get_task_language":
            return "Pick up the bowl by 10 cm."
        raise AssertionError(method)


def test_env_client_validates_layout_and_tracks_termination() -> None:
    meta = robot_spec.env_runtime_contract(
        task_name="stack_bowls", layout_group=0, layout_id=1
    )
    client = RoboDojoEnvClient(FakeRpc(meta), expected_meta=meta)
    assert client.last_reset_info["instruction"] == "Pick up the bowl by 10 cm."
    assert client.terminated is False

    with pytest.raises(ValueError):
        client.step(np.zeros(14), action_type="ee")
    with pytest.raises(ValueError):
        client.chunk_step(np.zeros((2, 16)), action_type="qpos")

    payload, _, terminated, _, info = client.chunk_step(
        np.zeros((6, 16)), action_type="ee", return_all_frames=True
    )
    assert len(payload["frames"]) == 6
    assert info["executed_actions"] == 6
    assert terminated is True
    assert client.terminated is True
    with pytest.raises(RuntimeError, match="terminal"):
        client.step(np.zeros(16))
    assert client.status()["layout_id"] == 1

    with pytest.raises(ValueError, match="requested layout"):
        RoboDojoEnvClient(FakeRpc(meta, reported_layout=5), expected_meta=meta)


def test_env_client_rejects_a_mismatched_server_contract() -> None:
    meta = robot_spec.env_runtime_contract(
        task_name="stack_bowls", layout_group=0, layout_id=1
    )
    other = FakeRpc(dict(meta, task_name="push_T"))
    with pytest.raises(AssertionError, match="env_meta mismatch"):
        RoboDojoEnvClient(other, expected_meta=meta)


def test_prompts_render_without_unresolved_variables() -> None:
    variables = {
        "task_name": "general_pickup",
        "layout_id": 2,
        "layout_group": 0,
        "memory_dir": "memory/robodojo",
        "reference_tag": "general_pickup_l0",
        "vla_available": False,
    }
    system = format_prompt(system_prompt(), variables=variables)
    user = format_prompt(user_prompt(), variables=variables)
    assert "robots/robodojo/guides/GUIDE_RPENT.md" in system
    assert "memory/robodojo/MEMORY.md" in system
    assert "task: general_pickup" in user
    assert "layout id: 2" in user
    assert "{{" not in system + user
