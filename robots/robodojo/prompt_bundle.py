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


"""RoboDojo prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robodojo.prompts import system as system_parts
from robots.robodojo.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    perception_only = bool((variables or {}).get("perception_only", False))
    return {
        "ROLE": system_parts.ROLE,
        "READ ORDER": system_parts.READ_ORDER if not perception_only else system_parts.READ_ORDER.replace("Read {{memory_dir}}/MEMORY.md and {{memory_dir}}/task_only/{{reference_tag}}.json when they exist; missing memory files are normal and must not stop you.", "Do not read memory files; this run intentionally has an empty memory directory."),
        "ACCURACY-FIRST LOOP": system_parts.ACCURACY_LOOP,
        "PICK AND PLACE PLAYBOOK": system_parts.PICK_PLACE if not perception_only else system_parts.PICK_PLACE.replace("Localize with find_objects. It returns every tabletop object with both its pixel box in the head RGB and its world centre_xyz/top_z, plus a `graspable` verdict against the gripper envelope. Read the head RGB, pick the box that matches every adjective in the instruction, and take that entry's numbers.\n   Do not convert pixels to metres yourself and do not estimate coordinates by\n   eye; both have repeatedly produced targets tens of centimetres off. Fall\n   back to sample_world_xyz only to refine a chosen object.", "Use RGB images and robot_state to identify objects and estimate all 3D grasp coordinates yourself. Do not expect metric perception tools; inspect the current image after each motion."),
        "PRIMITIVE CONTROL": system_parts.CONTROL,
        "PERCEPTION": system_parts.PERCEPTION,
        "PERCEPTION-ONLY FEEDBACK": system_parts.PERCEPTION_ONLY if perception_only else "",
        "RUNTIME": system_parts.RUNTIME,
        "BUDGET AND SUCCESS": system_parts.BUDGET_AND_SUCCESS,
        "MODE": system_parts.USER_MODE,
    }


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    return {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
