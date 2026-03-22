# Unity RL-Agents Trainers

本目录下的 Python 包实现了一套面向 Unity 场景的强化学习与模仿学习训练框架，基于 Unity ML-Agents 项目进行扩展与适配，具有以下特性：

- 提供与现有训练脚本兼容的训练器接口和命令行入口；
- 使用 **PaddlePaddle** 作为深度学习推理与训练后端；
## 环境要求（Python 侧）

- **必须安装 PaddlePaddle ≥ 3.1.0**（推荐严格使用 `3.1.0`，当前代码按此版本验证）。
- 推荐在具备 NVIDIA GPU 的环境下安装 **PaddlePaddle GPU 版本**，以获得更高的训练效率。

PaddlePaddle 的安装方式和可用版本范围应以官方文档为准，可参考：  
<https://www.paddlepaddle.org.cn/install/quick>  
在该页面中选择与当前操作系统和 CUDA/cuDNN 版本匹配的 **GPU 安装选项**。

除 PaddlePaddle 之外，其余运行时依赖（如 `grpcio`、`h5py`、`numpy`、`protobuf`、`pyyaml`、`paddle2onnx`、`six`、`cattrs`、`attrs`、`huggingface_hub`、`onnx` 等）由本目录下的 `setup.py` 进行统一声明，在后续执行 `pip install -e .` 时将自动安装，无需单独处理。

## 安装（源码方式）

1. 从版本控制系统获取本项目源码（Git 克隆或下载压缩包并解压）。  
2. 将当前工作目录切换至仓库根目录（包含 `ml_agents` 与 `ml_agents_envs` 子目录）。  
3. 在目标 Python 环境中执行以下命令，完成本地（可编辑）安装：

   ```sh
   cd ml-agents-release_23/ml_agents_envs
   python -m pip install -e .

   cd ../ml_agents
   python -m pip install -e .
   ```

   上述步骤不会通过 PyPI 获取 `mlagents` 或 `mlagents_envs`，而是直接基于当前源码树进行可编辑安装。

4. 完成安装后，可通过以下命令行入口启动训练流程：

   ```sh
   mlagents-learn
   ```

## 基础使用流程示例：3D Balance Ball

下述步骤以示例场景 **3D Balance Ball** 为例，说明如何利用本训练框架完成一次端到端训练。

示例运行环境：

- PaddlePaddle GPU 3.1.0
- CUDA 12.6
- Unity 6.0

1. 在 Unity 中打开示例场景  
   - 启动 Unity，打开包含 ML-Agents 示例的 Unity 工程。  
   - 在 **Project** 视图中导航至  
     `Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity`，双击打开该场景。  
   - 在层级视图中选中任一 Agent，确认其挂载了 `Behavior Parameters` 组件，并记录其行为名称（通常为 `3DBallLearning`）。

2. 确认训练配置文件  
   - 在仓库根目录下，定位训练配置文件 `config/ppo/3DBall.yaml`。  
   - 根据需要，在该 YAML 中调整训练超参数（如 `max_steps`、`learning_rate`、`batch_size` 等）。

3. 启动 Python 训练进程  
   - 打开终端或命令行窗口，将当前目录切换至仓库根目录：
     ```sh
     cd ml-agents-release_23
     ```
   - 确认已完成前述安装步骤（`ml_agents_envs` 与 `ml_agents` 均已以源码方式安装），且已安装 PaddlePaddle 3.1.0 或更高版本。  
     建议优先安装 PaddlePaddle 3.1.0 GPU 版本以获得最佳使用体验。
   - 在终端中执行：
     ```sh
     mlagents-learn config/ppo/3DBall.yaml --run-id=first3DBallRun
     ```
   - 当终端输出提示 “Start training by pressing the Play button in the Unity Editor” 时，保持该进程处于运行状态。

4. 在 Unity Editor 中连接训练进程  
   - 切换回 Unity Editor。  
   - 在工具栏中点击 **Play** 按钮。  
   - Editor 将作为训练环境连接至正在运行的 Python 进程，3D Balance Ball 场景中的所有 Agent 将开始采样并参与训练。

5. 监控训练过程与生成结果  
   - 训练期间，终端会周期性输出包括步数（Step）、平均奖励（Mean Reward）等关键指标的日志。  
   - 训练过程中将在 `results/first3DBallRun` 目录下生成：  
     - 训练指标日志（用于后续可视化分析）；  
     - 模型检查点与最终 `.onnx` 模型文件；  
     - 训练计时信息（`run_logs` 子目录）。  
   - 使用 `Ctrl+C` 中断训练时，应等待进程完成模型保存后再关闭终端。

   **使用 VisualDL 进行可视化监控**

   本训练框架在默认配置下会将训练指标写入 `results/<run-id>/` 目录。若需要使用 **VisualDL** 对训练过程进行可视化监控，按以下流程操作：

   1. 在当前 Python 环境中安装 VisualDL（如尚未安装）：

      ```sh
      python -m pip install visualdl
      ```

   2. 在训练完成或进行中的情况下，切换到保存日志的上级目录，并启动 VisualDL 服务。  
      例如，对于位于 `results/Basic/` 的一次训练，可执行：

      ```sh
      cd results
      visualdl --logdir .\Basic\
      ```

   3. 在浏览器中访问 VisualDL 输出的本地地址（以命令行输出为准），即可查看随时间变化的累计奖励、损失、学习率等指标曲线。

   当训练使用自定义的 `--run-id` 或结果输出路径时，将 `--logdir` 指向对应的结果子目录即可。

6. 在 Unity 中加载训练完成的模型（可选）  
   - 将生成的 `<behavior_name>.onnx` 文件拷贝至 Unity 工程中对应示例的模型目录，例如：  
     `Project/Assets/ML-Agents/Examples/3DBall/TFModels/`。  
   - 在场景中选中对应 Agent，在 `Behavior Parameters` 组件中将 `Model` 字段设置为该 `.onnx` 文件。  
   - 将行为配置为推理模式后点击 **Play**，即可在不依赖 Python 训练进程的前提下验证训练结果。

在深度学习后端切换为 PaddlePaddle 并引入 Paddle 版本约束之外，
训练配置（行为标识、YAML 文件结构、统计与日志输出目录布局）与既有工具链保持一致，
可复用现有配置与工作流。
