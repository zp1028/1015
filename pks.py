# -*- coding: utf-8 -*-
"""PK10/飞艇 适配器实现"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from adapters.base import LotteryAdapter
from models.types import LotteryConfig
from utils.http_utils import safe_json_get
from config import settings, API_BASE, API_BASE_ALT, POS_COLS


class PKSAdapter(LotteryAdapter):
    """PK10/飞艇数据适配器"""

    def __init__(self, config: LotteryConfig):
        super().__init__(config)
        self._cache = {}

    def _history_items(self, endpoint: str, day: str):
        """从主/备用 API 获取某一天的数据"""
        for base in [settings.api_base, settings.api_base_alt]:
            data = safe_json_get(f"{base}/{endpoint}?lotCode={self.config.code}&date={day}")
            if data:
                items = data.get("result", {}).get("data", []) or []
                if items:
                    return items
        return []

    def fetch_latest(self) -> Optional[Dict]:
        """获取最新一期"""
        url = f"{settings.api_base}/pks/getLotteryPksInfo.do?lotCode={self.config.code}"
        data = safe_json_get(url)
        if not data or data.get("errorCode") != 0:
            return None

        d = data.get("result", {}).get("data") or {}
        code = str(d.get("preDrawCode", ""))
        nums = [int(x) for x in code.split(",") if x.strip().isdigit()]
        if len(nums) != 10:
            return None

        return {
            "期号": str(d.get("preDrawIssue", "")),
            "开奖时间": str(d.get("drawTime") or d.get("preDrawTime", "")),
            "下期期号": str(d.get("drawIssue", "")),
            "下期时间": str(d.get("drawTime", "")),
            "服务器时间": str(d.get("serverTime", "")),
            "冠亚和": nums[0] + nums[1],
            **{name: nums[i] for i, name in enumerate(POS_COLS)}
        }

    def fetch_history(self, days: int = 3) -> pd.DataFrame:
        """并行拉取历史数据"""
        today = datetime.now().date()
        days_list = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        rows = []

        with ThreadPoolExecutor(max_workers=min(4, max(1, days))) as pool:
            futures = {
                pool.submit(self._history_items, "pks/getPksHistoryList.do", day): day
                for day in days_list
            }
            for future in as_completed(futures):
                for it in future.result():
                    code = str(it.get("preDrawCode", ""))
                    nums = [int(x) for x in code.split(",") if x.strip().isdigit()]
                    if len(nums) != 10:
                        continue
                    row = {
                        "期号": str(it.get("preDrawIssue", "")),
                        "开奖时间": it.get("preDrawTime", ""),
                        "冠亚和": nums[0] + nums[1],
                    }
                    row.update(dict(zip(POS_COLS, nums)))
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).drop_duplicates(subset=["期号"])
        return df.sort_values("期号", kind="stable").reset_index(drop=True)

    def extract_sequences(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """提取大小/单双序列"""
        return {
            "大小": ["大" if int(x) > 11 else "小" for x in df["冠亚和"].tolist()],
            "单双": ["单" if int(x) % 2 == 1 else "双" for x in df["冠亚和"].tolist()],
        }

    def get_labels(self, category: str) -> Tuple[str, ...]:
        if category == "大小":
            return ("大", "小")
        elif category == "单双":
            return ("单", "双")
        return ()

    def get_actual(self, latest: Dict, category: str) -> Optional[str]:
        gy = int(latest.get("冠亚和", 0))
        if category == "大小":
            return "大" if gy > 11 else "小"
        elif category == "单双":
            return "单" if gy % 2 else "双"
        return None

    def render_live(self, rt: Dict) -> str:
        nums = [rt[p] for p in POS_COLS]
        gy = int(rt["冠亚和"])
        balls = "".join(f'<span class="num-ball gold">{int(n):02d}</span>' for n in nums)
        return (
            f'<div class="title">🔴 {self.config.name} · 实时开奖</div>'
            f'<div class="meta">第 {rt["期号"]} 期　{str(rt["开奖时间"])[:19]}</div>'
            f'<div class="balls-row">{balls}</div>'
            f'<div style="text-align:center;color:#e8c96a;margin-top:8px;">'
            f'冠亚和 {gy} · {"大" if gy > 11 else "小"} · {"单" if gy % 2 else "双"}'
            f'</div>'
        )

    @property
    def pred_key(self) -> str:
        return f"pks_{self.config.code}"

    @property
    def default_categories(self) -> List[str]:
        return ["大小", "单双"]
