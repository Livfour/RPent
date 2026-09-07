# RoboDojo RPent Registered-Tool Guide

This guide is the operational reference for the registered-tool runtime on
RoboDojo. The system prompt owns strategy; this guide only defines the tools,
geometry, and compact execution rules needed to apply it.

## Registered tools

- Resources: `list_dir`, `read_text_file`
- Observation: `view_env_state`, `render`, `sample_world_xyz`, `query_world_map`
- Control: `move_to`, `move_delta`, `rotate_wrist`, `set_gripper`, `release`,
  and `vla_act` when a VLA endpoint is configured
- Terminal: `finish`

Tool schemas are authoritative for arguments. Issue one mutation at a time and
inspect its fresh result before the next mutation.

## Observation and geometry

Start with `view_env_state(step=0)`. Its `task_language` is authoritative. The
head view (`head`, 640x480 unless scaled) identifies objects, distractors,
destinations, and global progress. Wrist views (`left_wrist`, `right_wrist`)
refine grasp, contact, and release geometry for the same head-selected
candidate.

World maps use `[row,col]`, contain world-frame `[x,y,z]` metres, and may
contain NaN. Query the exact step/view/resolution whose RGB supplied the
pixels. Sample several interior pixels and use robust geometry; an exposed
surface point is the object's top, not its centre. Re-observe after occlusion
or physical change.

`robot_state` carries, per arm, `<arm>_eef_pose` (`[x,y,z,qw,qx,qy,qz]` of the
end-effector link in the world frame), `<arm>_gripper` (0 closed, 1 open),
`<arm>_arm_qpos` (six joint angles), and `ee16` (the full action layout:
`left pose(7), left gripper, right pose(7), right gripper`). The table top
is at roughly z = 0.74 m; the left arm base sits at negative x and the right
arm base at positive x.

`episode_status` reports `take_action_cnt`, `step_lim`, `done`,
`eval_success`, and `score`. Every primitive consumes `steps` native actions.

## Primitives

- `move_to(arm, xyz, quat=None, gripper=None, steps=20)` interpolates the
  end-effector pose along a straight line; each waypoint is solved by the
  native IK. An unreachable waypoint holds the previous target, so check
  `final_dist_m` and the wrist view. Keep `quat` unchanged unless the task
  needs a different wrist orientation.
- `move_delta(arm, dxyz, gripper=None, steps=10)` is the guarded short motion
  for approach, descent, lift, and retreat.
- `rotate_wrist(arm, delta_yaw_deg)` yaws the end-effector about world Z.
- `set_gripper(arm, val)` ramps the gripper; `release(arm)` opens it fully.
  A grasp is verified when the closed gripper value stays clearly above 0 or
  the wrist view shows the object between the fingers, and the object rises
  with the arm during the lift.
- `vla_act(chunks)` (when registered) executes 50-action VLA chunks driven by
  the native instruction; use one chunk near contact and re-observe.

## Analytic execution safeguards

- Approach from above with the gripper open: hover 8-12 cm over the object,
  descend in 2-4 cm increments, and stop when the fingers straddle the object.
- Never transport because a gripper merely looks closed: also require visible
  object motion or elevation.
- After a failed grasp, open the gripper, retreat upward, relocalize the
  object from a fresh observation, and change one variable (grasp point,
  height, yaw, or arm) before retrying.
- Keep the idle arm out of the workspace and away from the camera stand.
- Stop all motion once `episode_status.done` is true and call `finish`.
