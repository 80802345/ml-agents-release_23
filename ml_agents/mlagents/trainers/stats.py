
from collections import defaultdict
from enum import Enum
from typing import List, Dict, NamedTuple, Any, Optional
import numpy as np
import abc
import os
import time
from threading import RLock
from visualdl import LogWriter

# ML-Agents相关导入（原代码保留）
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import set_gauge

logger = get_logger(__name__)

def _dict_to_str(param_dict: Dict[str, Any], num_tabs: int) -> str:
    """
    Takes a parameter dictionary and converts it to a human-readable string.
    Recurses if there are multiple levels of dict. Used to print out hyperparameters.

    :param param_dict: A Dictionary of key, value parameters.
    :return: A string version of this dictionary.
    """
    if not isinstance(param_dict, dict):
        return str(param_dict)
    else:
        append_newline = "\n" if num_tabs > 0 else ""
        return append_newline + "\n".join(
            [
                "\t"
                + "  " * num_tabs
                + f"{x}:\t{_dict_to_str(param_dict[x], num_tabs + 1)}"
                for x in param_dict
            ]
        )


# 统计结果结构化存储（原代码保留）
class StatsSummary(NamedTuple):
    full_dist: List[float]
    aggregation_method: StatsAggregationMethod

    @staticmethod
    def empty() -> "StatsSummary":
        return StatsSummary([], StatsAggregationMethod.AVERAGE)

    @property
    def aggregated_value(self):
        if self.aggregation_method == StatsAggregationMethod.SUM:
            return self.sum
        else:
            return self.mean

    @property
    def mean(self):
        return np.mean(self.full_dist)

    @property
    def std(self):
        return np.std(self.full_dist)

    @property
    def num(self):
        return len(self.full_dist)

    @property
    def sum(self):
        return np.sum(self.full_dist)


# 统计属性类型枚举（原代码保留）
class StatsPropertyType(Enum):
    HYPERPARAMETERS = "hyperparameters"
    SELF_PLAY = "selfplay"


# 抽象基类：StatsWriter（原代码保留）
class StatsWriter(abc.ABC):
    """
    A StatsWriter abstract class. A StatsWriter takes in a category, key, scalar value, and step
    and writes it out by some method.
    """

    def on_add_stat(
        self,
        category: str,
        key: str,
        value: float,
        aggregation: StatsAggregationMethod = StatsAggregationMethod.AVERAGE,
    ) -> None:
        pass

    @abc.abstractmethod
    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        pass

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        pass


# GaugeWriter：写入计时器监控（原代码保留）
class GaugeWriter(StatsWriter):
    """
    Write all stats that we receive to the timer gauges, so we can track them offline easily
    """

    @staticmethod
    def sanitize_string(s: str) -> str:
        """
        Clean up special characters in the category and value names.
        """
        return s.replace("/", ".").replace(" ", "")

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            set_gauge(
                GaugeWriter.sanitize_string(f"{category}.{val}.mean"),
                float(stats_summary.mean),
            )
            set_gauge(
                GaugeWriter.sanitize_string(f"{category}.{val}.sum"),
                float(stats_summary.sum),
            )


# ConsoleWriter：控制台打印（原代码保留，仅注释掉get_rank相关）
class ConsoleWriter(StatsWriter):
    def __init__(self):
        self.training_start_time = time.time()
        # If self-play, we want to print ELO as well as reward
        self.self_play = False
        self.self_play_team = -1
        # 注释掉get_rank（原代码中未导入，避免报错）
        # self.rank = get_rank()
        self.rank = None  # 手动赋值为None，避免属性未定义

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        is_training = "Not Training"
        if "Is Training" in values:
            stats_summary = values["Is Training"]
            if stats_summary.aggregated_value > 0.0:
                is_training = "Training"

        elapsed_time = time.time() - self.training_start_time
        log_info: List[str] = [category]
        log_info.append(f"Step: {step}")
        log_info.append(f"Time Elapsed: {elapsed_time:0.3f} s")
        if "Environment/Cumulative Reward" in values:
            stats_summary = values["Environment/Cumulative Reward"]
            if self.rank is not None:
                log_info.append(f"Rank: {self.rank}")

            log_info.append(f"Mean Reward: {stats_summary.mean:0.3f}")
            if "Environment/Group Cumulative Reward" in values:
                group_stats_summary = values["Environment/Group Cumulative Reward"]
                log_info.append(f"Mean Group Reward: {group_stats_summary.mean:0.3f}")
            else:
                log_info.append(f"Std of Reward: {stats_summary.std:0.3f}")
            log_info.append(is_training)

            if self.self_play and "Self-play/ELO" in values:
                elo_stats = values["Self-play/ELO"]
                log_info.append(f"ELO: {elo_stats.mean:0.3f}")
        else:
            log_info.append("No episode was completed since last summary")
            log_info.append(is_training)
        logger.info(". ".join(log_info) + ".")

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            logger.info(
                """Hyperparameters for behavior name {}: \n{}""".format(
                    category, _dict_to_str(value, 0)
                )
            )
        elif property_type == StatsPropertyType.SELF_PLAY:
            assert isinstance(value, bool)
            self.self_play = value


# 核心改造：TensorboardWriter替换为VisualDL的LogWriter
class VisualDLWriter(StatsWriter):  # 类名改为VisualDLWriter（更贴合实际功能）
    def __init__(
        self,
        base_dir: str,
        clear_past_data: bool = False,
        hidden_keys: Optional[List[str]] = None,
    ):
        """
        A StatsWriter that writes to VisualDL summary (替代原TensorboardWriter)
        :param base_dir: 日志根目录，每个category会生成子目录
        :param clear_past_data: 是否清理历史日志文件
        :param hidden_keys: 不需要写入的指标名列表
        """
        # 替换为LogWriter字典（核心改造1）
        self.log_writers: Dict[str, LogWriter] = {}
        self.base_dir: str = base_dir
        self._clear_past_data = clear_past_data
        self.hidden_keys: List[str] = hidden_keys if hidden_keys is not None else []

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self._maybe_create_log_writer(category)
        for key, value in values.items():
            if key in self.hidden_keys:
                continue
            # 写入标量（LogWriter参数和SummaryWriter完全兼容）
            self.log_writers[category].add_scalar(
                tag=f"{key}",
                value=value.aggregated_value,
                step=step
            )
            # 写入直方图（核心改造2：适配LogWriter的histogram参数）
            if value.aggregation_method == StatsAggregationMethod.HISTOGRAM:
                self.log_writers[category].add_histogram(
                    tag=f"{key}_hist",
                    values=np.array(value.full_dist),
                    step=step
                )
            self.log_writers[category].flush()  # 立即刷入磁盘

    def _maybe_create_log_writer(self, category: str) -> None:
        """按需创建LogWriter，避免重复初始化"""
        if category not in self.log_writers:
            filewriter_dir = f"{self.base_dir}/{category}"
            os.makedirs(filewriter_dir, exist_ok=True)
            # 清理历史日志（原逻辑保留）
            if self._clear_past_data:
                self._delete_all_events_files(filewriter_dir)
            # 创建VisualDL的LogWriter（核心改造3）
            self.log_writers[category] = LogWriter(logdir=filewriter_dir)

    def _delete_all_events_files(self, directory_name: str) -> None:
        """删除历史日志文件（兼容VisualDL和TensorBoard日志）"""
        for file_name in os.listdir(directory_name):
            if file_name.startswith("events.out") or file_name.startswith("vdl_log"):
                logger.warning(
                    f"Deleting VisualDL/TensorBoard data {file_name} that was left over from a "
                    "previous run."
                )
                full_fname = os.path.join(directory_name, file_name)
                try:
                    os.remove(full_fname)
                except OSError:
                    logger.error(
                        "{} was left over from a previous run and "
                        "not deleted.".format(full_fname)
                    )

    def add_property(
        self, category: str, property_type: StatsPropertyType, value: Any
    ) -> None:
        """写入超参数等文本信息（核心改造4）"""
        if property_type == StatsPropertyType.HYPERPARAMETERS:
            assert isinstance(value, dict)
            summary = _dict_to_str(value, 0)
            self._maybe_create_log_writer(category)
            if summary is not None:
                self.log_writers[category].add_text(
                    tag="Hyperparameters",
                    text_string=summary,
                    step=0  # 超参数写入step 0即可
                )
                self.log_writers[category].flush()


# 统计报告器（原代码保留，仅适配Writer类名）
class StatsReporter:
    writers: List[StatsWriter] = []
    stats_dict: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    lock = RLock()
    stats_aggregation: Dict[str, Dict[str, StatsAggregationMethod]] = defaultdict(
        lambda: defaultdict(lambda: StatsAggregationMethod.AVERAGE)
    )

    def __init__(self, category: str):
        self.category: str = category

    @staticmethod
    def add_writer(writer: StatsWriter) -> None:
        with StatsReporter.lock:
            StatsReporter.writers.append(writer)

    def add_property(self, property_type: StatsPropertyType, value: Any) -> None:
        with StatsReporter.lock:
            for writer in StatsReporter.writers:
                writer.add_property(self.category, property_type, value)

    def add_stat(
        self,
        key: str,
        value: float,
        aggregation: StatsAggregationMethod = StatsAggregationMethod.AVERAGE,
    ) -> None:
        with StatsReporter.lock:
            StatsReporter.stats_dict[self.category][key].append(value)
            StatsReporter.stats_aggregation[self.category][key] = aggregation
            for writer in StatsReporter.writers:
                writer.on_add_stat(self.category, key, value, aggregation)

    def set_stat(self, key: str, value: float) -> None:
        with StatsReporter.lock:
            StatsReporter.stats_dict[self.category][key] = [value]
            StatsReporter.stats_aggregation[self.category][
                key
            ] = StatsAggregationMethod.MOST_RECENT
            for writer in StatsReporter.writers:
                writer.on_add_stat(
                    self.category, key, value, StatsAggregationMethod.MOST_RECENT
                )

    def write_stats(self, step: int) -> None:
        with StatsReporter.lock:
            values: Dict[str, StatsSummary] = {}
            for key in StatsReporter.stats_dict[self.category]:
                if len(StatsReporter.stats_dict[self.category][key]) > 0:
                    stat_summary = self.get_stats_summaries(key)
                    values[key] = stat_summary
            for writer in StatsReporter.writers:
                writer.write_stats(self.category, values, step)
            del StatsReporter.stats_dict[self.category]

    def get_stats_summaries(self, key: str) -> StatsSummary:
        stat_values = StatsReporter.stats_dict[self.category][key]
        if len(stat_values) == 0:
            return StatsSummary.empty()

        return StatsSummary(
            full_dist=stat_values,
            aggregation_method=StatsReporter.stats_aggregation[self.category][key]
        )
