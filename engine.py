# -*- coding: utf-8 -*-
"""预测引擎：形态匹配 + 频率统计 + 自适应集成 + 严格滚动回测 + 统计检验"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import binomtest

from config import settings, MODEL_CANDIDATES
from models.types import (
    PredictionResult, BacktestResult, ModelCandidate, ModelSummary
)


def wilson_interval(ok: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% 置信区间"""
    if n <= 0:
        return 0.0, 100.0
    p = ok / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return max(0.0, (center-half)*100), min(100.0, (center+half)*100)


def binomial_significance(ok: int, n: int, p: float = 0.5, 
                          alpha: float = None, 
                          bonferroni_n: int = 1) -> Tuple[bool, float]:
    """二项检验：是否显著优于随机基准

    Returns:
        (是否显著, p值)
    """
    alpha = alpha or settings.significance_alpha
    if n <= 0 or ok < 0 or ok > n:
        return False, 1.0

    adjusted_alpha = alpha / max(bonferroni_n, 1)
    result = binomtest(ok, n, p=p, alternative='greater')
    p_value = result.pvalue
    return p_value < adjusted_alpha, p_value


def luzhu_after_pattern(seq: List[str], pattern: List[str]) -> Dict[str, Any]:
    """计算指定形态之后的结果"""
    n = len(pattern)
    nexts = []
    for i in range(len(seq) - n):
        if seq[i:i + n] == pattern:
            nexts.append(seq[i + n])

    c = Counter(nexts)
    total = sum(c.values())
    out = {"total": total, "counter": dict(c)}
    for k, v in c.items():
        out[k] = v
        out[f"{k}%"] = round(v / total * 100, 2) if total else 0
    return out


def _pattern_stats(seq: List[str], pattern: List[str], alpha: float = None) -> Dict[str, Any]:
    """带 Laplace 平滑的形态统计"""
    alpha = alpha or settings.prediction_alpha
    labels = sorted(set(seq))
    r = luzhu_after_pattern(seq, pattern)
    total = int(r.get("total", 0))
    smoothed = {}
    denom = total + alpha * len(labels)
    for lb in labels:
        smoothed[lb] = ((r.get(lb, 0) + alpha) / denom * 100) if denom else 0
    r["smooth_pct"] = smoothed
    return r


def _single_pattern_model(
    seq: List[str], 
    labels: Tuple[str, ...], 
    length: int, 
    min_samples: int = None,
    alpha: float = None,
) -> Optional[Dict[str, Any]]:
    """固定形态长度模型"""
    min_samples = min_samples or settings.prediction_min_samples
    alpha = alpha or settings.prediction_alpha

    if len(seq) <= length:
        return None

    pattern = seq[-length:]
    r = _pattern_stats(seq, pattern, alpha=alpha)
    total = int(r.get("total", 0))
    if total <= 0:
        return None

    final = {lb: float(r["smooth_pct"].get(lb, 0)) for lb in labels}
    lean = max(final, key=final.get)
    top = final[lean]
    baseline = 100.0 / len(labels)
    evidence = np.sqrt(total)
    sample_factor = min(1.0, total / float(min_samples))

    # 置信度判定
    if total < min_samples or abs(top - baseline) < 3:
        confidence = "低"
    elif total < 20 or abs(top - baseline) < 7:
        confidence = "中"
    else:
        confidence = "高"

    return {
        "length": length,
        "pattern": pattern,
        "sample": total,
        "final": final,
        "lean": lean,
        "pct": top,
        "weight": evidence * (0.35 + 0.65 * sample_factor),
        "confidence": confidence,
    }


def adaptive_pattern_model(
    seq: List[str], 
    labels: Tuple[str, ...], 
    lengths: Tuple[int, ...] = (3, 4, 5, 6),
    min_samples: int = None,
) -> PredictionResult:
    """3/4/5/6期自适应集成模型"""
    min_samples = min_samples or settings.prediction_min_samples
    details = []
    length_bonus = {3: 0.95, 4: 1.00, 5: 1.05, 6: 1.08}

    for L in lengths:
        m = _single_pattern_model(seq, labels, L, min_samples=min_samples)
        if m:
            m["weight"] *= length_bonus.get(L, 1.0)
            details.append(m)

    if not details:
        return PredictionResult(
            lean="", pct=0, sample=0, pattern=[], 
            details=[], confidence="低", final={}
        )

    scores = {lb: 0.0 for lb in labels}
    total_w = 0.0
    for d in details:
        w = d["weight"]
        total_w += w
        for lb in labels:
            scores[lb] += d["final"].get(lb, 0) * w

    final = {lb: (scores[lb] / total_w if total_w else 0) for lb in labels}
    lean = max(final, key=final.get)
    top = final[lean]
    baseline = 100 / len(labels)
    effective = sum(d["sample"] * d["weight"] for d in details) / total_w if total_w else 0

    if effective < min_samples or abs(top - baseline) < 3:
        confidence = "低"
    elif effective < 20 or abs(top - baseline) < 7:
        confidence = "中"
    else:
        confidence = "高"

    best = max(details, key=lambda x: x["weight"])

    return PredictionResult(
        lean=lean,
        pct=round(top, 2),
        sample=round(effective, 1),
        pattern=best["pattern"],
        pattern_len=best["length"],
        confidence=confidence,
        final={k: round(v, 2) for k, v in final.items()},
        details=details,
    )


def predict_frequency(seq: List[str], labels: Tuple[str, ...], window: int = 30) -> Optional[Dict[str, Any]]:
    """频率模型"""
    if not seq:
        return None
    x = seq[-window:]
    c = Counter(x)
    total = len(x)
    final = {lb: (c.get(lb, 0) + 1) / (total + len(labels)) * 100 for lb in labels}
    lean = max(final, key=final.get)
    return {
        "lean": lean,
        "pct": final[lean],
        "sample": total,
        "final": final,
        "confidence": "中" if total >= 30 else "低",
        "model_name": f"{window}期频率",
    }


def predict_fixed_length(
    seq: List[str], 
    labels: Tuple[str, ...], 
    length: int, 
    min_samples: int = 6,
) -> Optional[Dict[str, Any]]:
    """固定长度模型包装"""
    return _single_pattern_model(seq, labels, length, min_samples=min_samples)


def predict_ensemble(
    seq: List[str], 
    labels: Tuple[str, ...], 
    lengths: Tuple[int, ...] = (3, 4, 5, 6),
) -> PredictionResult:
    """集成模型包装"""
    return adaptive_pattern_model(seq, labels, lengths=lengths)


# ==================== 严格滚动回测 ====================

def walk_forward_backtest(
    seq: List[str],
    labels: Tuple[str, ...],
    model_name: str = "ensemble",
    min_history: int = None,
    length: int = 5,
    window: int = 30,
    test_limit: Optional[int] = None,
) -> Optional[BacktestResult]:
    """严格 walk-forward 回测：预测第 t 期时只允许读取 1~t-1 期"""
    min_history = min_history or settings.backtest_min_history
    if len(seq) <= min_history:
        return None

    start = max(min_history, len(seq) - test_limit) if test_limit else min_history
    results = []

    for t in range(start, len(seq)):
        train = seq[:t]
        if model_name == "ensemble":
            model = predict_ensemble(train, labels)
        elif model_name == "frequency":
            freq = predict_frequency(train, labels, window=window)
            model = freq if freq else None
        else:
            model = predict_fixed_length(train, labels, length, min_samples=6)

        if not model or not model.get("lean"):
            continue
        results.append(seq[t] == model["lean"])

    if not results:
        return None

    n = len(results)
    ok = int(sum(results))
    rate = ok / n * 100
    baseline = 100 / len(labels)
    low, high = wilson_interval(ok, n)

    return BacktestResult(
        n=n, ok=ok, bad=n-ok, rate=rate, baseline=baseline,
        low=low, high=high, advantage=rate-baseline, ci_width=high-low,
    )


def _model_stability_score(long_bt: BacktestResult, recent_bt: Optional[BacktestResult]) -> float:
    """综合分：优先长期表现，奖励近期稳定"""
    long_adv = long_bt.advantage
    recent_adv = recent_bt.advantage if recent_bt else long_adv
    lower_adv = long_bt.low - long_bt.baseline
    stability_penalty = abs(long_adv - recent_adv) * 0.25
    return 0.55 * long_adv + 0.25 * recent_adv + 0.20 * lower_adv - stability_penalty


def evaluate_models(
    seq: List[str],
    labels: Tuple[str, ...],
    min_history: int = None,
) -> List[Dict[str, Any]]:
    """模型实验室：长期 + 近期回测 + 统计显著性检验"""
    min_history = min_history or settings.backtest_min_history
    if len(seq) <= min_history + 10:
        return []

    recent_limit = min(settings.backtest_recent_limit_base, max(50, len(seq) // 3))
    rows = []

    # Bonferroni 校正：比较 8 个候选模型
    bonferroni_n = len(MODEL_CANDIDATES)

    for candidate in MODEL_CANDIDATES:
        name, typ, length, window = candidate
        long_bt = walk_forward_backtest(
            seq, labels, model_name=typ, min_history=min_history,
            length=length, window=window, test_limit=None,
        )
        recent_bt = walk_forward_backtest(
            seq, labels, model_name=typ, min_history=min_history,
            length=length, window=window, test_limit=recent_limit,
        )

        if not long_bt:
            continue

        score = _model_stability_score(long_bt, recent_bt)

        # 统计显著性检验
        is_sig, p_val = binomial_significance(
            long_bt.ok, long_bt.n, p=1/len(labels),
            bonferroni_n=bonferroni_n if settings.bonferroni_correction else 1,
        )

        rows.append({
            "模型": name, "type": typ, "length": length, "window": window,
            "长期样本": long_bt.n, "长期准确率": long_bt.rate, "长期优势": long_bt.advantage,
            "长期下界": long_bt.low, "长期上界": long_bt.high,
            "长期显著": is_sig, "长期p值": p_val,
            "近期样本": recent_bt.n if recent_bt else 0,
            "近期准确率": recent_bt.rate if recent_bt else 0,
            "近期优势": recent_bt.advantage if recent_bt else 0,
            "综合分": score,
        })

    return rows


def select_model(
    seq: List[str], 
    labels: Tuple[str, ...], 
    min_history: int = None,
) -> Dict[str, Any]:
    """自动选择模型：必须有足够样本，且综合分最高"""
    min_history = min_history or settings.backtest_min_history
    rows = evaluate_models(seq, labels, min_history=min_history)

    if not rows:
        return {
            "name": "3/4/5/6集成", "type": "ensemble", "length": 0, "window": 0,
            "score": 0, "reason": "样本不足，使用保守集成"
        }

    eligible = [r for r in rows if r["长期样本"] >= settings.backtest_long_min_samples]
    if not eligible:
        eligible = rows

    best = max(eligible, key=lambda r: r["综合分"])
    baseline = 100.0 / len(labels)

    # 保护机制：如果长期95%下界仍未超过随机基准且不显著，回退到集成
    if best["长期下界"] <= baseline and not best.get("长期显著", False):
        return {
            "name": "3/4/5/6集成", "type": "ensemble", "length": 0, "window": 0,
            "score": best["综合分"], "long_rate": best["长期准确率"],
            "recent_rate": best["近期准确率"],
            "reason": "无显著统计优势，回退综合集成",
            "p_value": best.get("长期p值", 1.0),
        }

    return {
        "name": best["模型"], "type": best["type"], "length": best["length"],
        "window": best["window"], "score": best["综合分"],
        "long_rate": best["长期准确率"], "recent_rate": best["近期准确率"],
        "reason": "长期+近期表现综合选择",
        "significant": best.get("长期显著", False),
        "p_value": best.get("长期p值", 1.0),
    }


def predict_selected(seq: List[str], labels: Tuple[str, ...]) -> PredictionResult:
    """使用自动选择模型生成当前预测"""
    choice = select_model(seq, labels)
    typ = choice["type"]

    if typ == "ensemble":
        model = predict_ensemble(seq, labels)
    elif typ == "frequency":
        freq = predict_frequency(seq, labels, choice["window"])
        if freq:
            model = PredictionResult(
                lean=freq["lean"], pct=freq["pct"], sample=freq["sample"],
                final=freq["final"], confidence=freq["confidence"],
            )
        else:
            model = PredictionResult(lean="", pct=0, sample=0, final={}, confidence="低")
    else:
        fixed = predict_fixed_length(seq, labels, choice["length"], min_samples=6)
        if fixed:
            model = PredictionResult(
                lean=fixed["lean"], pct=fixed["pct"], sample=fixed["sample"],
                pattern=fixed["pattern"], pattern_len=fixed["length"],
                final=fixed["final"], confidence=fixed["confidence"],
            )
        else:
            model = PredictionResult(lean="", pct=0, sample=0, final={}, confidence="低")

    model.selected_model = choice
    return model



def model_tracking_summary(key: str, category: str = None, window: int = 300):
    """按实际已结算预测统计模型表现"""
    from db import db
    rows = db.get_model_performance(key, limit=window, category=category)
    if category and category != '全部':
        rows = [r for r in rows if r['category'] == category]

    from collections import defaultdict
    groups = defaultdict(lambda: {'n': 0, 'ok': 0})
    for r in rows:
        g = groups[r['model']]
        g['n'] += 1
        g['ok'] += int(r['result'] == '对')

    out = []
    base = 25.0 if category == '组合' else 50.0
    for name, g in groups.items():
        n, ok = g['n'], g['ok']
        rate = ok / n * 100 if n else 0
        low, high = wilson_interval(ok, n) if n else (0, 100)
        recent = [r for r in rows if r['model'] == name][:min(50, len([r for r in rows if r['model'] == name]))]
        rok = sum(r['result'] == '对' for r in recent)
        rn = len(recent)
        recent_rate = rok / rn * 100 if rn else 0
        out.append({
            '模型': name, '样本': n, '正确': ok, '准确率': rate,
            '下界': low, '上界': high, '近期样本': rn, '近期准确率': recent_rate,
            '相对基准': rate - base,
            '状态': '稳定优势' if low > base else ('观察' if rate >= base else '低于基准')
        })
    return sorted(out, key=lambda x: (x['准确率'], x['样本']), reverse=True)



def derive_combo_probabilities(r_dx: dict, r_ds: dict) -> dict:
    """由大小×单双推导四组合概率"""
    dx = {
        "大": float(r_dx.get("大%", 0) or 0),
        "小": float(r_dx.get("小%", 0) or 0),
    }
    ds = {
        "单": float(r_ds.get("单%", 0) or 0),
        "双": float(r_ds.get("双%", 0) or 0),
    }
    combos = {
        "大单": dx["大"] * ds["单"] / 100,
        "大双": dx["大"] * ds["双"] / 100,
        "小单": dx["小"] * ds["单"] / 100,
        "小双": dx["小"] * ds["双"] / 100,
    }
    total = sum(combos.values())
    if total > 0:
        combos = {k: round(v * 100 / total, 2) for k, v in combos.items()}
    return combos
