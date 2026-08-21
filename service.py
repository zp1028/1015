# -*- coding: utf-8 -*-
"""预测服务层：封装预测、结算、追踪逻辑"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

from models.types import PredictionRecord
from engine import predict_selected
from db import db
from config import settings


class PredictionService:
    """预测服务：处理预测生成、结算、历史追踪"""

    def __init__(self, pred_key: str):
        self.pred_key = pred_key
        self._last_seen: Optional[str] = None

    def generate_predictions(
        self, 
        sequences: Dict[str, List[str]],
        issue: str,
    ) -> Dict[str, Dict]:
        """生成所有类别的预测"""
        predictions = {}
        for category, seq in sequences.items():
            if len(seq) < 3:
                continue
            labels = tuple(sorted(set(seq)))
            model = predict_selected(seq, labels)

            predictions[category] = {
                "lean": model.lean,
                "pct": model.pct,
                "sample": model.sample,
                "pattern": "".join(model.pattern) if model.pattern else "",
                "confidence": model.confidence,
                "model_name": model.selected_model.get("name", "自适应集成") if model.selected_model else "自适应集成",
                "model_score": model.selected_model.get("score", 0) if model.selected_model else 0,
                "p_value": model.selected_model.get("p_value", 1.0) if model.selected_model else 1.0,
                "significant": model.selected_model.get("significant", False) if model.selected_model else False,
            }

            # 持久化
            record = PredictionRecord(
                key=self.pred_key,
                issue=str(issue),
                cat=category,
                pattern="".join(model.pattern) if model.pattern else "",
                lean=model.lean,
                sample=int(model.sample) if model.sample else 0,
                pct=model.pct,
                confidence=model.confidence,
                model_name=model.selected_model.get("name", "") if model.selected_model else "",
                model_score=model.selected_model.get("score", 0) if model.selected_model else 0,
                time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            db.queue_write(record)

        db._flush_queue()  # 确保写入
        return predictions

    def settle(self, new_issue: str, actual_map: Dict[str, str]) -> bool:
        """结算上一期预测"""
        if new_issue == self._last_seen:
            return False

        hist = db.load_predictions(self.pred_key, limit=1000)
        changed = False

        for row in hist:
            if row.result != '待开' or row.cat not in actual_map:
                continue

            try:
                old_i = int(str(row.issue))
                new_i = int(str(new_issue))
                if old_i >= new_i:
                    continue
            except Exception:
                if str(row.issue) >= str(new_issue):
                    continue

            actual = actual_map[row.cat]
            result = '对' if actual == row.lean else '错'
            db.settle_prediction(row.id, actual, result, new_issue)
            changed = True

        self._last_seen = new_issue
        return changed

    def get_history(self, limit: int = 500) -> List[PredictionRecord]:
        """获取预测历史"""
        return db.load_predictions(self.pred_key, limit=limit)

    def get_model_performance(self, limit: int = 1000) -> List[Dict]:
        """获取模型表现"""
        return db.get_model_performance(self.pred_key, limit=limit)

    def clear(self):
        """清空预测记录"""
        db.clear_predictions(self.pred_key)
