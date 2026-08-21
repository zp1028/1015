# -*- coding: utf-8 -*-
"""彩票适配器抽象基类"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pandas as pd

from models.types import LotteryConfig


class LotteryAdapter(ABC):
    """彩票数据适配器接口"""

    def __init__(self, config: LotteryConfig):
        self.config = config

    @abstractmethod
    def fetch_latest(self) -> Optional[Dict[str, any]]:
        """获取最新一期开奖数据"""
        pass

    @abstractmethod
    def fetch_history(self, days: int = 3) -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    def extract_sequences(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """从历史数据提取预测序列

        Returns:
            {"大小": [...], "单双": [...], "组合": [...]}
        """
        pass

    @abstractmethod
    def get_labels(self, category: str) -> Tuple[str, ...]:
        """获取某类别的标签集合"""
        pass

    @abstractmethod
    def get_actual(self, latest: Dict, category: str) -> Optional[str]:
        """从最新开奖数据提取实际结果"""
        pass

    @abstractmethod
    def render_live(self, rt: Dict) -> str:
        """渲染实时看板 HTML"""
        pass

    @property
    @abstractmethod
    def pred_key(self) -> str:
        """预测记录的唯一键"""
        pass

    @property
    @abstractmethod
    def default_categories(self) -> List[str]:
        """默认预测类别"""
        pass
