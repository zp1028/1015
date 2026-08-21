# -*- coding: utf-8 -*-
"""Luck20/快乐8 适配器实现"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from adapters.base import LotteryAdapter
from models.types import LotteryConfig
from utils.http_utils import safe_json_get
from config import settings


def classify_sum(s: int, api_big_small=None, api_single_double=None) -> Tuple[str, str, str]:
    """和值大小单双分类"""
    if api_big_small is not None and str(api_big_small) != "":
        try:
            dx = "大" if int(api_big_small) == 1 else "小"
        except Exception:
            dx = "大" if s >= 810 else "小"
    else:
        dx = "大" if s >= 810 else "小"

    if api_single_double is not None and str(api_single_double) != "":
        try:
            ds = "单" if int(api_single_double) == 1 else "双"
        except Exception:
            ds = "单" if s % 2 == 1 else "双"
    else:
        ds = "单" if s % 2 == 1 else "双"

    return dx, ds, dx + ds


def _parse_luck20(it: Dict) -> Optional[Dict]:
    """解析单条 Luck20 数据"""
    code = str(it.get("preDrawCode", ""))
    all_nums = [int(x) for x in code.split(",") if x.strip().isdigit()]
    if len(all_nums) < 20:
        return None

    nums = all_nums[:20]
    extra = all_nums[20] if len(all_nums) > 20 else None

    try:
        s = int(it.get("sumNum"))
    except Exception:
        s = sum(nums)

    dx, ds, combo = classify_sum(
        s,
        api_big_small=it.get("sumBigSmall"),
        api_single_double=it.get("sumSingleDouble"),
    )

    row = {
        "期号": str(it.get("preDrawIssue", "")),
        "开奖时间": str(it.get("preDrawTime", "")),
        "号码": nums,
        "附加号": extra,
        "和值": s,
        "大小": dx,
        "单双": ds,
        "组合": combo,
    }
    for i in range(20):
        row[f"号{i+1}"] = nums[i]
    return row


class Luck20Adapter(LotteryAdapter):
    """Luck20/快乐8 数据适配器"""

    def __init__(self, config: LotteryConfig):
        super().__init__(config)

    def _history_items(self, endpoint: str, day: str):
        for base in [settings.api_base, settings.api_base_alt]:
            data = safe_json_get(f"{base}/{endpoint}?lotCode={self.config.code}&date={day}")
            if data:
                items = data.get("result", {}).get("data", []) or []
                if items:
                    return items
        return []

    def fetch_latest(self) -> Optional[Dict]:
        url = f"{settings.api_base}/LuckTwenty/getBaseLuckTewnty.do?lotCode={self.config.code}"
        data = safe_json_get(url)
        if not data or data.get("errorCode") != 0:
            return None

        d = data.get("result", {}).get("data") or {}
        row = _parse_luck20(d)
        if not row:
            return None

        row["下期期号"] = str(d.get("drawIssue", ""))
        row["下期时间"] = str(d.get("drawTime", ""))
        row["服务器时间"] = str(d.get("serverTime", ""))
        return row

    def fetch_history(self, days: int = 3) -> pd.DataFrame:
        today = datetime.now().date()
        days_list = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        rows = []

        with ThreadPoolExecutor(max_workers=min(4, max(1, days))) as pool:
            futures = {
                pool.submit(self._history_items, "LuckTwenty/getBaseLuckTwentyList.do", day): day
                for day in days_list
            }
            for future in as_completed(futures):
                for it in future.result():
                    row = _parse_luck20(it)
                    if row:
                        rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).drop_duplicates(subset=["期号"])
        return df.sort_values("期号", kind="stable").reset_index(drop=True)

    def extract_sequences(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        return {
            "大小": df["大小"].tolist(),
            "单双": df["单双"].tolist(),
            "组合": df["组合"].tolist(),
        }

    def get_labels(self, category: str) -> Tuple[str, ...]:
        labels_map = {
            "大小": ("大", "小"),
            "单双": ("单", "双"),
            "组合": ("大单", "大双", "小单", "小双"),
        }
        return labels_map.get(category, ())

    def get_actual(self, latest: Dict, category: str) -> Optional[str]:
        if category == "大小":
            return latest.get("大小")
        elif category == "单双":
            return latest.get("单双")
        elif category == "组合":
            dx = latest.get("大小")
            ds = latest.get("单双")
            if dx and ds:
                return dx + ds
        return None

    def render_live(self, rt: Dict) -> str:
        nums = rt["号码"]
        extra = rt.get("附加号")
        extra_html = f'　附加号 <span class="num-ball gold">{int(extra):02d}</span>' if extra is not None else ""
        balls = "".join(f'<span class="num-ball green">{int(n):02d}</span>' for n in nums)
        return (
            f'<div class="title">🟢 {self.config.name} · 实时开奖</div>'
            f'<div class="meta">第 {rt["期号"]} 期　{str(rt["开奖时间"])[:19]}</div>'
            f'<div class="balls-row">{balls}{extra_html}</div>'
            f'<div style="text-align:center"><span class="tag-combo">'
            f'和值 {int(rt["和值"])} · {rt["大小"]} · {rt["单双"]} · {rt["组合"]}'
            f'</span></div>'
        )

    @property
    def pred_key(self) -> str:
        return f"l20_{self.config.code}"

    @property
    def default_categories(self) -> List[str]:
        return ["大小", "单双", "组合"]
