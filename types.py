# -*- coding: utf-8 -*-
"""类型定义和配置管理"""

from typing import Literal, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class LotteryConfig:
    """彩种配置"""
    key: str
    name: str
    type: Literal["pks", "luck20"]
    code: int


@dataclass
class PredictionResult:
    """预测结果"""
    lean: str
    pct: float
    sample: float
    pattern: list[str] = field(default_factory=list)
    pattern_len: int = 0
    confidence: Literal["低", "中", "高"] = "低"
    final: dict[str, float] = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)
    selected_model: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "lean": self.lean,
            "pct": self.pct,
            "sample": self.sample,
            "pattern": self.pattern,
            "pattern_len": self.pattern_len,
            "confidence": self.confidence,
            "final": self.final,
            "details": self.details,
            "selected_model": self.selected_model,
        }


@dataclass
class BacktestResult:
    """回测结果"""
    n: int
    ok: int
    bad: int
    rate: float
    baseline: float
    low: float  # Wilson 95% 下界
    high: float  # Wilson 95% 上界
    advantage: float
    ci_width: float

    @property
    def is_significant(self, alpha: float = 0.05, bonferroni_n: int = 1) -> bool:
        """经 Bonferroni 校正后是否显著优于随机"""
        from scipy.stats import binom_test
        adjusted_alpha = alpha / max(bonferroni_n, 1)
        p_value = binom_test(self.ok, self.n, p=1/len([self.baseline]), alternative='greater')
        return p_value < adjusted_alpha and self.low > self.baseline


@dataclass
class ModelCandidate:
    """模型候选"""
    name: str
    type: Literal["fixed", "frequency", "ensemble"]
    length: int = 0
    window: int = 0


@dataclass
class ModelPerformance:
    """模型表现"""
    model: str
    category: str
    result: Literal["对", "错"]
    time: str


@dataclass
class ModelSummary:
    """模型汇总"""
    模型: str
    样本: int
    正确: int
    准确率: float
    下界: float
    上界: float
    近期样本: int
    近期准确率: float
    相对基准: float
    状态: str


@dataclass
class PredictionRecord:
    """预测记录（数据库模型）"""
    id: Optional[int] = None
    key: str = ""
    issue: str = ""
    cat: str = ""
    pattern: str = ""
    lean: str = ""
    sample: int = 0
    pct: float = 0.0
    actual: str = ""
    result: str = "待开"
    time: str = ""
    settle_issue: str = ""
    confidence: str = "低"
    model_name: str = ""
    model_score: float = 0.0

    def to_row(self) -> tuple:
        return (
            self.key, self.issue, self.cat, self.pattern, self.lean,
            self.sample, self.pct, self.actual, self.result, self.time,
            self.settle_issue, self.confidence, self.model_name, self.model_score
        )

    @classmethod
    def from_row(cls, row: tuple) -> "PredictionRecord":
        return cls(
            id=row[0], key=row[1], issue=row[2], cat=row[3], pattern=row[4],
            lean=row[5], sample=row[6], pct=row[7], actual=row[8], result=row[9],
            time=row[10], settle_issue=row[11], confidence=row[12],
            model_name=row[13], model_score=row[14]
        )
