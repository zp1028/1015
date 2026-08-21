# -*- coding: utf-8 -*-
"""配置管理 - Pydantic Settings"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Tuple


class Settings(BaseSettings):
    """应用配置，支持环境变量覆盖"""

    # API 配置
    api_base: str = Field(default="https://api.api16868.com", description="主 API 地址")
    api_base_alt: str = Field(default="https://api.api68.com", description="备用 API 地址")
    api_timeout_connect: float = Field(default=5.0, description="连接超时(秒)")
    api_timeout_read: float = Field(default=10.0, description="读取超时(秒)")
    api_retries: int = Field(default=2, description="重试次数")

    # 熔断器配置
    circuit_failure_threshold: int = Field(default=5, description="熔断触发失败次数")
    circuit_recovery_timeout: int = Field(default=30, description="熔断恢复时间(秒)")

    # 回测配置
    backtest_min_history: int = Field(default=40, description="最少历史训练期数")
    backtest_recent_limit_base: int = Field(default=150, description="近期回测基础期数")
    backtest_long_min_samples: int = Field(default=80, description="长期回测最小样本")

    # 预测配置
    prediction_min_samples: int = Field(default=8, description="预测最小样本")
    prediction_alpha: float = Field(default=1.0, description="Laplace 平滑系数")

    # 数据库配置
    db_path: str = Field(default="data/lottery_app.db", description="数据库路径")
    db_batch_size: int = Field(default=50, description="批量写入大小")

    # 缓存配置
    cache_ttl_latest: int = Field(default=20, description="实时数据缓存(秒)")
    cache_ttl_history: int = Field(default=90, description="历史数据缓存(秒)")
    cache_ttl_model: int = Field(default=60, description="模型评估缓存(秒)")
    cache_ttl_prediction: int = Field(default=30, description="预测结果缓存(秒)")

    # 统计检验配置
    significance_alpha: float = Field(default=0.05, description="显著性水平")
    bonferroni_correction: bool = Field(default=True, description="启用 Bonferroni 校正")

    class Config:
        env_file = ".env"
        env_prefix = "LOTTERY_"


# 全局配置实例
settings = Settings()

# 彩种目录
LOTTERY_CATALOG = [
    {"key": "10037", "name": "极速飞艇", "type": "pks", "code": 10037},
    {"key": "10035", "name": "极速赛车", "type": "pks", "code": 10035},
    {"key": "10012", "name": "幸运飞艇", "type": "pks", "code": 10012},
    {"key": "10058", "name": "PK拾(10058)", "type": "pks", "code": 10058},
    {"key": "10057", "name": "澳洲幸运10", "type": "pks", "code": 10057},
    {"key": "10054", "name": "极速快乐8", "type": "luck20", "code": 10054},
    {"key": "10047", "name": "幸运20(10047)", "type": "luck20", "code": 10047},
]

# 模型候选列表
MODEL_CANDIDATES = [
    ("3期", "fixed", 3, 0),
    ("4期", "fixed", 4, 0),
    ("5期", "fixed", 5, 0),
    ("6期", "fixed", 6, 0),
    ("20期频率", "frequency", 0, 20),
    ("30期频率", "frequency", 0, 30),
    ("50期频率", "frequency", 0, 50),
    ("3/4/5/6集成", "ensemble", 0, 0),
]

POS_COLS = ["冠军", "亚军", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十"]


# 向后兼容的常量
API_BASE = settings.api_base
API_BASE_ALT = settings.api_base_alt
