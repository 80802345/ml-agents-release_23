import os

from distutils.version import LooseVersion
import pkg_resources
# from mlagents.torch_utils import cpu_utils
from mlagents.trainers.settings import PaddleSettings
from mlagents_envs.logging_util import get_logger


logger = get_logger(__name__)


def assert_paddle_installed():
    # Check that paddle version 2.0.0 or later has been installed. If not, refer
    # user to the PaddlePaddle webpage for install instructions.
    paddle_pkg = None
    try:
        paddle_pkg = pkg_resources.get_distribution("paddlepaddle")
    except pkg_resources.DistributionNotFound:
        try:
            paddle_pkg = pkg_resources.get_distribution("paddlepaddle-gpu")
        except pkg_resources.DistributionNotFound:
            pass

    assert paddle_pkg is not None and LooseVersion(paddle_pkg.version) >= LooseVersion(
        "3.0.0"
    ), (
        "A compatible version of PaddlePaddle was not installed. Please visit the PaddlePaddle homepage "
        + "(https://www.paddlepaddle.org.cn/install/quick) and follow the instructions to install. "
        + "Version 3.0.0 and later are supported."
    )


assert_paddle_installed()

# This should be the only place that we import paddle directly.
# Everywhere else is caught by the banned-modules setting for flake8
import paddle  # noqa I201


# paddle.set_num_threads(cpu_utils.get_num_threads_to_use())
os.environ["KMP_BLOCKTIME"] = "0"


_device = paddle.set_device("cpu")


def set_paddle_config(paddle_settings: PaddleSettings) -> None:
    global _device

    # 1. 确定设备字符串（处理cuda/gpu兼容、自动判断）
    if paddle_settings.device is None:
        # 自动判断：有CUDA则用GPU，否则用CPU
        device_str = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    else:
        # 手动指定设备：兼容"cuda"→"gpu"（Paddle标准是gpu）
        device_str = paddle_settings.device.lower()  # 统一小写，避免大小写问题
        if device_str == "cuda":
            device_str = "gpu"
        # 校验设备合法性，避免无效输入
        if device_str not in ["gpu", "cpu"] and not device_str.startswith(("gpu:", "cpu:")):
            logger.warning(f"无效的设备指定 {paddle_settings.device}，自动切换为CPU")
            device_str = "cpu"

    # 2. 设置Paddle默认设备（关键修复：set_device无返回值，单独获取设备信息）
    paddle.set_device(device_str)
    # 正确获取当前设备字符串（而非接收set_device的返回值）
    _device = paddle.get_device()

    # 3. 设置默认数据类型
    paddle.set_default_dtype("float32")

    # 4. 日志输出（优化：打印具体设备，如gpu:0/cpu，而非None）
    logger.debug(f"default Paddle device: {_device}")


# Initialize to default settings
set_paddle_config(PaddleSettings(device=None))

nn = paddle.nn


def default_device():
    return _device
