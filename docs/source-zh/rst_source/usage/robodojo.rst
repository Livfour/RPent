RoboDojo
========

`RoboDojo <https://robodojo-benchmark.com/>`_ 是一个面向通用操作策略的
仿真加真机统一基准，包含 42 个基于 Isaac Sim 的双臂 ARX-X5 任务。RPent 通过
运行在 RoboDojo 自带 Isaac Sim Python 中的 env server 驱动其原生
``EvalEnv``，并使用 RoboDojo 内置 IK 求解的绝对末端位姿动作控制双臂。VLA
端点是可选的：未配置时规划器只使用解析式原语。

.. note::

   RoboDojo 支持目前面向单集调试，尚未按官方 42 任务协议完成验证。

部署 RoboDojo
-------------

RoboDojo 运行在独立于 RPent 虚拟环境的环境中（Isaac Sim 4.5，Python 3.10）。
克隆基准仓库，用其 ``pixi.toml`` 安装（Isaac Sim wheel 来自
``pypi.nvidia.com``），并下载资源：

.. code-block:: bash

   git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git RoboDojo_eval
   cd RoboDojo_eval
   pixi install -e default
   bash scripts/init_assets.sh          # 约 40 GB，放在 Assets/

宿主机需要 NVIDIA 驱动以及 Isaac Sim 常用的系统库（Vulkan loader、
``libGLU``、``libXt``、``libSM``、``libICE``、``libXinerama``、
``libXcursor``）。在精简容器中可将它们复制到一个目录，并在启动 RPent 前
加入 ``LD_LIBRARY_PATH``。

告诉 RPent 仓库位置和解释器：

.. code-block:: bash

   export ROBODOJO_ROOT=/path/to/RoboDojo_eval
   export ROBODOJO_PYTHON=$ROBODOJO_ROOT/.pixi/envs/default/bin/python

安装 RPent
----------

RPent 本身只需要基础安装加视频编码器：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[robodojo]"

env server 通过 ``PYTHONPATH`` 从代码仓库导入 ``rpent.utils.rpc``，不会向
Isaac 环境安装任何 RPent 依赖。

运行任务
--------

运行布局组 0 中布局 0 的抓取任务一集：

.. code-block:: bash

   rpent --robot robodojo \
      --task-name general_pickup \
      --layout-id 0 \
      --env-cuda-device 0 \
      --memory-profile local \
      --planner api --model openai-chat:qwen3.8-27b \
      --base-url http://127.0.0.1:8100/v1

``--task-name`` 可以是 ``task/RoboDojo/tasks`` 下的任意模块；``--layout-id``
选择一个已发布的场景布局，``--layout-group``（别名 ``--seed``）选择官方布局
组。``--max-episode-steps`` 覆盖任务原生 ``step_lim``。完整选项见
``rpent --robot robodojo --help``。

上面的规划器示例指向一个 OpenAI 兼容端点（例如本地 vLLM 部署的 Qwen），
将 ``OPENAI_API_KEY`` 设为该端点的密钥。:doc:`configure_planner` 支持的任意
提供方都可以使用。对于 vLLM 部署的 Qwen 模型，可导出
``RPENT_API_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'``
在 agent 循环中关闭思考；该 JSON 对象会合并进每个规划器请求体。

查看结果
--------

默认情况下运行结果保存在
``logs/<timestamp>_robodojo_<task-name>_l<layout-id>/``：

- ``run.log`` 是 RPent 进程日志。
- ``robodojo_env_server.log`` 记录 Isaac Sim 启动和场景错误。
- ``episode.mp4`` 是 25 fps 的头部相机回放。
- ``states.json`` 与 ``step_*/`` 保存每一步观测（RGB、米制深度、相机标定、
  世界坐标图）。
- ``transcript_*.json`` 是规划器对话记录。

RoboDojo 自身的奖励检查（``episode_status.eval_success``）是任务成功的唯一
来源。调用 ``finish`` 只是结束规划循环，无法覆盖原生结果。

工具
----

- ``view_env_state``、``render``、``sample_world_xyz``、``query_world_map``：
  观测与世界系深度反投影。
- ``move_to``、``move_delta``、``rotate_wrist``、``set_gripper``、``release``：
  解析式末端原语；每个原语在 ``steps`` 个原生 25 Hz 动作上插值目标位姿。
- ``vla_act``：仅当 ``--vla-endpoint`` 指向实现 ``vla.get_meta`` 与
  ``vla.predict``（``absolute_eef_pose`` 16 维动作块）的服务器时注册。

常用选项
--------

- ``--robodojo-root`` 与 ``--robodojo-python`` 覆盖 ``ROBODOJO_ROOT`` 和
  ``ROBODOJO_PYTHON``。
- ``--render-scale`` 缩小所有相机分辨率以节省显存；``--low-memory-render``
  （或 ``ROBODOJO_LOW_MEMORY_RENDER=1``）切换到性能渲染模式并限制纹理显存，
  适合共享 GPU。
- ``--env-endpoint`` 连接已运行的 env server 而不是重新启动。
- ``--env-cuda-device`` 选择 Isaac Sim 渲染使用的 GPU（物理在 CPU 上运行）。
