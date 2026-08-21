# -*- coding: utf-8 -*-
"""适配器工厂"""

from models.types import LotteryConfig
from adapters.base import LotteryAdapter
from adapters.pks import PKSAdapter
from adapters.luck20 import Luck20Adapter


def create_adapter(config: LotteryConfig) -> LotteryAdapter:
    """根据彩种类型创建适配器"""
    if config.type == "pks":
        return PKSAdapter(config)
    elif config.type == "luck20":
        return Luck20Adapter(config)
    else:
        raise ValueError(f"Unknown lottery type: {config.type}")
