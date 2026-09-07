RoboDojo
========

`RoboDojo <https://robodojo-benchmark.com/>`_ is a unified sim-and-real
benchmark for generalist manipulation with 42 Isaac Sim tasks on a dual
ARX-X5 bimanual platform. RPent drives RoboDojo's native ``EvalEnv`` through
an env server that runs inside RoboDojo's own Isaac Sim Python, and controls
the arms with absolute end-effector pose actions solved by RoboDojo's
built-in IK. A VLA endpoint is optional: without one the planner uses the
analytic primitives only.

.. note::

   RoboDojo support is a first integration for single-episode debugging. It
   has not been validated against the official 42-task protocol.

Deploy RoboDojo
---------------

RoboDojo runs in its own environment (Isaac Sim 4.5, Python 3.10) that is
separate from the RPent virtual environment. Clone the benchmark, install it
with its ``pixi.toml`` (Isaac Sim wheels come from ``pypi.nvidia.com``), and
download the assets:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git RoboDojo_eval
   cd RoboDojo_eval
   pixi install -e default
   bash scripts/init_assets.sh          # ~40 GB into Assets/

The host needs the NVIDIA driver plus the usual Isaac Sim system libraries
(Vulkan loader, ``libGLU``, ``libXt``, ``libSM``, ``libICE``,
``libXinerama``, ``libXcursor``). On a minimal container copy them into a
directory and export it on ``LD_LIBRARY_PATH`` before starting RPent.

Point RPent at the checkout and its interpreter:

.. code-block:: bash

   export ROBODOJO_ROOT=/path/to/RoboDojo_eval
   export ROBODOJO_PYTHON=$ROBODOJO_ROOT/.pixi/envs/default/bin/python

Install RPent
-------------

RPent itself needs only the base install plus a video encoder:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[robodojo]"

The env server imports ``rpent.utils.rpc`` from the checkout on
``PYTHONPATH``; nothing from RPent is installed into the Isaac environment.

Run a task
----------

Run one episode of the plain pick-up task on layout 0 of layout group 0:

.. code-block:: bash

   rpent --robot robodojo \
      --task-name general_pickup \
      --layout-id 0 \
      --env-cuda-device 0 \
      --memory-profile local \
      --planner api --model openai-chat:qwen3.8-27b \
      --base-url http://127.0.0.1:8100/v1

``--task-name`` is any module under ``task/RoboDojo/tasks``; ``--layout-id``
selects one published scene layout and ``--layout-group`` (alias ``--seed``)
the official layout group. ``--max-episode-steps`` overrides the task's
native ``step_lim``. See ``rpent --robot robodojo --help`` for the complete
option list.

The planner example above targets an OpenAI-compatible endpoint (for
instance a local vLLM serving Qwen); set ``OPENAI_API_KEY`` to the endpoint's
key. Any provider supported by :doc:`configure_planner` works.

View the result
---------------

By default the run is saved under
``logs/<timestamp>_robodojo_<task-name>_l<layout-id>/``:

- ``run.log`` contains the RPent process log.
- ``robodojo_env_server.log`` contains the Isaac Sim boot and scene errors.
- ``episode.mp4`` is the head-camera replay at 25 fps.
- ``states.json`` and ``step_*/`` hold every observation (RGB, metric depth,
  camera calibration, world maps).
- ``transcript_*.json`` contains the planner conversation.

RoboDojo's own reward check (``episode_status.eval_success``) is the task
success source. Calling ``finish`` ends the planner loop; it cannot override
the native result.

Tools
-----

- ``view_env_state``, ``render``, ``sample_world_xyz``, ``query_world_map``:
  observation and depth back-projection in the world frame.
- ``move_to``, ``move_delta``, ``rotate_wrist``, ``set_gripper``, ``release``:
  analytic end-effector primitives; each interpolates the commanded pose
  over ``steps`` native 25 Hz actions.
- ``vla_act``: registered only when ``--vla-endpoint`` points at a server
  implementing ``vla.get_meta`` and ``vla.predict`` with the
  ``absolute_eef_pose`` 16-D chunk contract.

Common options
--------------

- ``--robodojo-root`` and ``--robodojo-python`` override ``ROBODOJO_ROOT``
  and ``ROBODOJO_PYTHON``.
- ``--render-scale`` shrinks every camera to save GPU memory.
- ``--env-endpoint`` connects to an already running env server instead of
  spawning one.
- ``--env-cuda-device`` selects the GPU for Isaac Sim rendering (physics runs
  on the CPU).
