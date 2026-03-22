# Unity RL-Agents Python Interface

本目录下的 Python 包提供了 Python 与 Unity 环境之间的通信接口，可作为统一的“Unity 环境 Python SDK”使用：

- 单智能体 API（Gym API）
- 类 Gym 的多智能体 API（PettingZoo API）
- 低层接口（LLAPI），供训练器与 Unity 直接通信

The LLAPI is still used by the trainer implementation; `rlagents_envs`
can also be used independently of any trainer to
communicate with Unity from Python.

> 在 Unity 端与 Python 端通信协议版本匹配的前提下，本包可独立于具体训练实现稳定运行。

## Installation（从源码手动安装）

使用本包时，推荐直接基于源码进行本地可编辑安装 
在解压后的仓库根目录下，执行：

```sh
cd ml-agents-release_23/ml_agents_envs
python -m pip install -e .
```

上述命令会将当前源码树中的环境接口包以“可编辑模式”安装到当前 Python 环境中，便于后续调试与升级。

## Limitations

The underlying transport semantics are the same as in `mlagents_envs`:

- `rlagents_envs` uses localhost ports to exchange data between Unity and
  Python. As such, multiple instances can have their ports collide, leading to
  errors. Make sure to use a different port if you are using multiple instances
  of `UnityEnvironment`.
- Communication between Unity and the Python `UnityEnvironment` is not secure.
- On Linux, ports are not released immediately after the communication closes.
  As such, you cannot reuse ports right after closing a `UnityEnvironment`.
