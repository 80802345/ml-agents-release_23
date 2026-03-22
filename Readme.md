# Unity RL-Agents Toolkit

本仓库包含以下开源组件：

- 用于构建和运行强化学习场景的 Unity 组件（C#，包名 `com.unity.ml-agents`）；
- 与 Unity 环境通信的 Python 接口（`rlagents_envs`）；
- 基于 PaddlePaddle 的 Python 训练脚本和算法实现。

本项目是在 Unity ML-Agents 工具链的基础上演化而来，面向希望使用 PaddlePaddle 进行训练的用户提供一套等价的开源实现。

在 Python 侧，需要满足以下前提：

- 安装 **PaddlePaddle ≥ 3.1.0**（推荐版本为 3.1.0）；
- 在 `ml_agents_envs` 与 `ml_agents` 目录下分别执行 `pip install -e .` 进行本地安装；
- 使用命令行入口 `mlagents-learn` 启动训练流程。

关于 Unity 集成、环境设计与训练配置，可参考本仓库附带的示例场景与配置文件（如 `Assets/ML-Agents/Examples`、`config/` 目录），并结合自身项目需求进行扩展。
