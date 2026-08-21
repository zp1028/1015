# -*- coding: utf-8 -*-
"""
彩票 AI 数据分析助手 - Streamlit 多彩种版（优化重构）
支持：双色球 · 快乐8 · 极速飞艇(PK10风格) · 极速快乐8
仅供学习与娱乐，开奖完全随机，历史无法预测未来。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter
import time

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="彩票数据分析助手",
    page_icon="🎱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.0rem; font-weight: 700; color: #e63946; text-align: center; }
    .sub-header { text-align: center; color: #666; margin-bottom: 1rem; }
    .disclaimer {
        background-color: #fff3cd; border-left: 5px solid #ffc107;
        padding: 10px 14px; margin: 10px 0; border-radius: 4px; font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

API_BASE_17500 = "https://data.17500.cn"
API_BASE_16868 = "https://api.api16868.com"
SSQ_URL = f"{API_BASE_17500}/ssq_asc.txt"
KL8_URL = f"{API_BASE_17500}/kl8_asc.txt"
FEITING_LOT_CODE = 10037
KL8_SPEED_CODE = 10047

def safe_get(url: str, timeout: int = 25, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content.decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt == retries:
                st.warning(f"网络请求失败: {e}")
                return None
            time.sleep(1)
    return None

def safe_json_get(url: str, timeout: int = 15, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1)
    return None

def parse_pattern(text: str, allowed_chars: str):
    t = text.strip().replace("，", "").replace(",", "").replace(" ", "").replace("　", "")
    if not t or any(c not in allowed_chars for c in t):
        return None
    return list(t)

def luzhu_after_pattern(seq, pattern):
    n = len(pattern)
    nexts = []
    for i in range(len(seq) - n):
        if seq[i : i + n] == pattern:
            nexts.append(seq[i + n])
    c = Counter(nexts)
    total = sum(c.values())
    out = {"total": total, "counter": dict(c)}
    for k, v in c.items():
        out[k] = v
        out[f"{k}%"] = round(v / total * 100, 2) if total else 0
    return out

def suggest_combos_by_freq(freq, k=5, n_groups=10):
    nums = freq.index.tolist()
    weights = freq.values.astype(float)
    weights = weights / weights.sum()
    rng = np.random.default_rng()
    groups = []
    for _ in range(n_groups * 3):
        chosen = sorted(rng.choice(nums, size=k, replace=False, p=weights).tolist())
        if chosen not in groups:
            groups.append(chosen)
        if len(groups) >= n_groups:
            break
    return groups

# ==================== 双色球模块 ====================
@st.cache_data(ttl=3600 * 6, show_spinner="加载双色球数据...")
def load_ssq(force=False):
    cache = DATA_DIR / "ssq_history.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, dtype={"期号": str})
        df["开奖日期"] = pd.to_datetime(df["开奖日期"])
        return df.sort_values("期号").reset_index(drop=True)
    text = safe_get(SSQ_URL)
    if not text:
        if cache.exists():
            return pd.read_csv(cache, dtype={"期号": str})
        return pd.DataFrame()
    rows = []
    for line in text.strip().splitlines():
        p = line.split()
        if len(p) < 9: continue
        try:
            rows.append({"期号": p[0], "开奖日期": p[1],
                "红球1": int(p[2]), "红球2": int(p[3]), "红球3": int(p[4]),
                "红球4": int(p[5]), "红球5": int(p[6]), "红球6": int(p[7]), "蓝球": int(p[8])})
        except ValueError: continue
    df = pd.DataFrame(rows)
    df["开奖日期"] = pd.to_datetime(df["开奖日期"])
    df = df.sort_values("期号").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df

def ssq_freq(df, n):
    d = df.tail(n)
    reds = d[[f"红球{i}" for i in range(1, 7)]].values.flatten()
    return pd.Series(reds).value_counts().sort_index(), d["蓝球"].value_counts().sort_index()

def ssq_missing(df):
    red_cols = [f"红球{i}" for i in range(1, 7)]
    red_m, blue_m = {}, {}
    for num in range(1, 34):
        mask = (df[red_cols] == num).any(axis=1)
        red_m[num] = (len(df) - 1 - mask[::-1].idxmax()) if mask.any() else len(df)
    for num in range(1, 17):
        mask = df["蓝球"] == num
        blue_m[num] = (len(df) - 1 - mask[::-1].idxmax()) if mask.any() else len(df)
    return pd.Series(red_m), pd.Series(blue_m)

def count_combo_hits_ssq(df, nums, n=None):
    if n: df = df.tail(n)
    red_cols = [f"红球{i}" for i in range(1, 7)]
    target = set(nums); hits=[]
    for _, row in df.iterrows():
        drawn = set(int(row[c]) for c in red_cols)
        if target.issubset(drawn): hits.append(str(row["期号"]))
    return len(hits), len(df), hits# ==================== 快乐8（官方）模块 ====================
@st.cache_data(ttl=3600 * 6, show_spinner="加载快乐8数据...")
def load_kl8(force=False):
    cache = DATA_DIR / "kl8_history.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, dtype={"期号": str})
        df["开奖日期"] = pd.to_datetime(df["开奖日期"])
        return df.sort_values("期号").reset_index(drop=True)
    text = safe_get(KL8_URL)
    if not text:
        if cache.exists():
            return pd.read_csv(cache, dtype={"期号": str})
        return pd.DataFrame()
    rows = []
    for line in text.strip().splitlines():
        p = line.split()
        if len(p) < 22: continue
        try:
            nums = [int(x) for x in p[2:22]]
            if len(nums) != 20: continue
            row = {"期号": p[0], "开奖日期": p[1]}
            for i, n in enumerate(nums, 1): row[f"号{i}"] = n
            rows.append(row)
        except ValueError: continue
    df = pd.DataFrame(rows)
    df["开奖日期"] = pd.to_datetime(df["开奖日期"])
    df = df.sort_values("期号").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df

def kl8_freq(df, n):
    d = df.tail(n)
    cols = [c for c in d.columns if c.startswith("号")]
    all_nums = d[cols].values.flatten()
    return pd.Series(all_nums).value_counts().reindex(range(1, 81), fill_value=0)

def kl8_missing(df):
    cols = [c for c in df.columns if c.startswith("号")]
    miss = {}
    for num in range(1, 81):
        mask = (df[cols] == num).any(axis=1)
        miss[num] = (len(df) - 1 - mask[::-1].idxmax()) if mask.any() else len(df)
    return pd.Series(miss)

def kl8_sum_stats(df, n):
    d = df.tail(n).copy()
    cols = [c for c in d.columns if c.startswith("号")]
    d["和值"] = d[cols].sum(axis=1)
    d["奇数个数"] = (d[cols] % 2 == 1).sum(axis=1)
    d["大号个数"] = (d[cols] > 40).sum(axis=1)
    return d

def count_combo_hits_kl8(df, nums, n=None):
    if n: df = df.tail(n)
    cols = [c for c in df.columns if c.startswith("号")]
    target = set(nums); hits=[]
    for _, row in df.iterrows():
        drawn = set(int(row[c]) for c in cols)
        if target.issubset(drawn): hits.append(str(row["期号"]))
    return len(hits), len(df), hits

# ==================== 极速飞艇模块 ====================
def fetch_feiting_latest():
    url = f"{API_BASE_16868}/pks/getLotteryPksInfo.do?lotCode={FEITING_LOT_CODE}"
    data = safe_json_get(url)
    if not data or data.get("errorCode") != 0: return None
    d = data.get("result", {}).get("data") or {}
    code = str(d.get("preDrawCode", ""))
    nums = [int(x) for x in code.split(",") if x.strip().isdigit()]
    if len(nums) != 10: return None
    return {
        "期号": str(d.get("preDrawIssue", "")), "开奖时间": str(d.get("drawTime") or d.get("preDrawTime", "")),
        "下期期号": str(d.get("drawIssue", "")), "下期时间": str(d.get("drawTime", "")),
        "冠军": nums[0], "亚军": nums[1], "第三": nums[2], "第四": nums[3], "第五": nums[4],
        "第六": nums[5], "第七": nums[6], "第八": nums[7], "第九": nums[8], "第十": nums[9],
        "冠亚和": nums[0] + nums[1], "服务器时间": str(d.get("serverTime", "")),
    }

@st.cache_data(ttl=90, show_spinner="加载极速飞艇最近数据...")
def load_feiting(days=3):
    all_rows = []
    today = datetime.now().date()
    for i in range(days):
        day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        ok = False
        for base in (API_BASE_16868, "https://api.api68.com"):
            url = f"{base}/pks/getPksHistoryList.do?lotCode={FEITING_LOT_CODE}&date={day}"
            data = safe_json_get(url)
            if not data: continue
            items = data.get("result", {}).get("data", [])
            if not items: continue
            for it in items:
                code = str(it.get("preDrawCode", ""))
                nums = [int(x) for x in code.split(",") if x.strip().isdigit()]
                if len(nums) != 10: continue
                all_rows.append({
                    "期号": str(it.get("preDrawIssue", "")), "开奖时间": it.get("preDrawTime", ""),
                    "冠军": nums[0], "亚军": nums[1], "第三": nums[2], "第四": nums[3], "第五": nums[4],
                    "第六": nums[5], "第七": nums[6], "第八": nums[7], "第九": nums[8], "第十": nums[9],
                    "冠亚和": nums[0] + nums[1],
                })
            ok = True
            break
        if not ok: continue
    latest = fetch_feiting_latest()
    if latest:
        all_rows.append({ "期号": latest["期号"], "开奖时间": latest["开奖时间"],
            "冠军": latest["冠军"], "亚军": latest["亚军"], "第三": latest["第三"],
            "第四": latest["第四"], "第五": latest["第五"], "第六": latest["第六"],
            "第七": latest["第七"], "第八": latest["第八"], "第九": latest["第九"],
            "第十": latest["第十"], "冠亚和": latest["冠亚和"] })
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["期号"]).sort_values("期号").reset_index(drop=True)
    return df

def feiting_pos_freq(df, n):
    d = df.tail(n)
    pos_cols = ["冠军", "亚军", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十"]
    records = []
    for pos in pos_cols:
        vc = d[pos].value_counts().reindex(range(1, 11), fill_value=0)
        for num, cnt in vc.items(): records.append({"名次": pos, "号码": num, "次数": cnt})
    return pd.DataFrame(records)

def feiting_champion_stats(df, n):
    d = df.tail(n)
    return d["冠军"].value_counts().sort_index(), d["冠亚和"].value_counts().sort_index()

def feiting_dx_sequence(df):
    return ["大" if int(x) > 11 else "小" for x in df["冠亚和"].tolist()]

def feiting_ds_sequence(df):
    return ["单" if int(x) % 2 == 1 else "双" for x in df["冠亚和"].tolist()]

# ==================== 极速快乐8模块 ====================
def classify_kl8_sum(sum_val):
    dx = "大" if sum_val >= 810 else "小"
    ds = "单" if sum_val % 2 == 1 else "双"
    return dx, ds, dx + ds

def fetch_kl8_speed_latest():
    url = f"{API_BASE_16868}/LuckTwenty/getBaseLuckTewnty.do?lotCode={KL8_SPEED_CODE}"
    data = safe_json_get(url)
    if not data or data.get("errorCode") != 0: return None
    d = data.get("result", {}).get("data") or {}
    code = str(d.get("preDrawCode", ""))
    nums = [int(x) for x in code.split(",") if x.strip().isdigit()][:20]
    if len(nums) < 20: return None
    s = int(d.get("sumNum") or sum(nums))
    dx, ds, combo = classify_kl8_sum(s)
    return {
        "期号": str(d.get("preDrawIssue", "")), "开奖时间": str(d.get("preDrawTime", "")),
        "下期期号": str(d.get("drawIssue", "")), "下期时间": str(d.get("drawTime", "")),
        "号码": nums, "和值": s, "大小": dx, "单双": ds, "组合": combo,
        "服务器时间": str(d.get("serverTime", "")),
    }

# ==================== UI辅助函数 ====================
def show_combo_analysis(df, count_func, max_num, label):
    st.subheader("五码组合 · 历史对照")
    st.warning("以下为历史出现次数统计，**不是**未来中奖概率。每期开奖独立，请勿当作预测依据。")
    st.caption(f"统计：你选的 5 个号码，在历史（或近 N 期）中，有多少期「这 5 个全部出现在当期开奖号码里」。")
    scope = st.radio("统计范围", ["全部历史", f"近 {n_recent} 期"], horizontal=True, key=f"{label}_scope")
    use_n = None if scope == "全部历史" else n_recent
    user_input = st.text_input(f"输入 5 个号码（空格或逗号分隔）", key=f"{label}_input")
    if st.button("查询历史命中", key=f"{label}_btn") and user_input.strip():
        try:
            parts = user_input.replace("，", ",").replace(" ", ",").split(",")
            nums = sorted(set(int(x.strip()) for x in parts if x.strip()))
            if len(nums) != 5 or any(n < 1 or n > max_num for n in nums):
                st.error(f"请输入恰好 5 个不重复的号码（1-{max_num}）")
            else:
                hits, total, hit_list = count_func(df, nums, use_n)
                rate = hits / total * 100 if total else 0
                st.success(f"号码 {' '.join(f'{n:02d}' for n in nums)}")
                c1, c2, c3 = st.columns(3)
                c1.metric("历史命中期数", hits); c2.metric("统计总期数", total); c3.metric("历史出现率", f"{rate:.4f}%")
                if hit_list: st.write("命中期号（最多显示 20 个）：", "、".join(hit_list[-20:]))
        except ValueError: st.error("号码格式不正确")
    st.markdown("---")
    st.write("**基于近期频率的参考五码组合**（按单号出现次数加权随机生成，仅供对照）")
    if st.button("生成参考组合", key=f"{label}_suggest"):
        freq = kl8_freq(df, n_recent) if "kl8" in label else ssq_freq(df, n_recent)[0]
        freq = freq[freq > 0]
        groups = suggest_combos_by_freq(freq, k=5, n_groups=8)
        rows = []
        for g in groups:
            h, t, _ = count_func(df, g, use_n if use_n else None)
            rows.append({ "五码组合": " ".join(f"{x:02d}" for x in g), "近N期命中": h, "近N期出现率%": round(h / t * 100, 4) if t else 0 })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)# ==================== 主界面 ====================
st.markdown('<div class="main-header">🎱 彩票 AI 数据分析助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">双色球 · 快乐8 · 极速飞艇 | 仅供学习娱乐</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
⚠️ <b>重要声明</b>：所有彩票开奖均为随机事件，历史数据无法预测未来。
本工具只做统计与可视化，不提供任何“中奖保证”。请理性对待，量力而行。
极速类高频彩种数据来自第三方接口，仅供参考。
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 设置")
    lottery = st.selectbox("选择彩种", ["双色球", "快乐8（官方）", "极速飞艇（PK10）", "极速快乐8"], index=0)
    force_refresh = st.button("🔄 强制刷新数据")
    n_recent = st.slider("分析最近期数", 30, 500, 100, 10)
    if lottery in ("极速飞艇（PK10）", "极速快乐8"):
        ft_days = st.slider("拉取最近几天数据", 1, 7, 5) if lottery == "极速飞艇（PK10）" else 1
        auto_refresh = st.checkbox("⏱ 自动刷新开奖（约每 45 秒）", value=False)
        if auto_refresh: st.caption("开启后页面会定时重新拉取最新开奖")
    else:
        auto_refresh = False; ft_days = 3
    st.markdown("---")
    st.caption("双色球/官方快乐8：data.17500.cn")
    st.caption("极速飞艇：api.api16868.com/pks")
    st.caption("极速快乐8：api.api16868.com/LuckTwenty")

if lottery == "双色球":
    df = load_ssq(force_refresh)
    if df.empty: st.error("双色球数据加载失败"); st.stop()
    c1, c2, c3, c4 = st.columns(4)
    latest = df.iloc[-1]
    c1.metric("总期数", f"{len(df):,}"); c2.metric("最新期号", latest["期号"])
    c3.metric("开奖日", latest["开奖日期"].strftime("%Y-%m-%d"))
    reds = " ".join(f"{latest[f'红球{i}']:02d}" for i in range(1, 7))
    c4.metric("开奖号码", f"{reds} + {latest['蓝球']:02d}")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 频率", "📉 遗漏", "📈 走势", "🎯 五码对照", "📋 历史"])
    with tab1:
        rf, bf = ssq_freq(df, n_recent)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=rf.index, y=rf.values, title=f"红球频率（近{n_recent}期）", labels={"x":"号码","y":"次数"}, color=rf.values, color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
            st.write("热号:", " ".join(f"`{i:02d}`" for i in rf.nlargest(6).index))
            st.write("冷号:", " ".join(f"`{i:02d}`" for i in rf.nsmallest(6).index))
        with col2:
            fig = px.bar(x=bf.index, y=bf.values, title="蓝球频率", labels={"x":"号码","y":"次数"}, color=bf.values, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        rm, bm = ssq_missing(df)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=rm.index, y=rm.values, title="红球当前遗漏", color=rm.values, color_continuous_scale="OrRd")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=bm.index, y=bm.values, title="蓝球当前遗漏", color=bm.values, color_continuous_scale="PuBu")
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        red_cols = [f"红球{i}" for i in range(1, 7)]
        recent = df.tail(n_recent).copy(); recent["和值"] = recent[red_cols].sum(axis=1)
        fig = px.line(recent, x="期号", y="和值", title="红球和值走势", markers=True)
        fig.add_hline(y=recent["和值"].mean(), line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    with tab4: show_combo_analysis(df, count_combo_hits_ssq, 33, "ssq")
    with tab5:
        st.dataframe(df.tail(30).iloc[::-1], use_container_width=True, height=400)
        st.download_button("下载全部CSV", df.to_csv(index=False).encode("utf-8-sig"), f"ssq_{datetime.now():%Y%m%d}.csv", "text/csv")

elif lottery == "快乐8（官方）":
    df = load_kl8(force_refresh)
    if df.empty: st.error("快乐8数据加载失败"); st.stop()
    c1, c2, c3 = st.columns(3)
    latest = df.iloc[-1]
    c1.metric("总期数", f"{len(df):,}"); c2.metric("最新期号", latest["期号"]); c3.metric("开奖日", latest["开奖日期"].strftime("%Y-%m-%d"))
    nums = " ".join(f"{latest[f'号{i}']:02d}" for i in range(1, 21))
    st.info(f"最新开奖号码（20个）：{nums}")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 频率热冷", "📉 遗漏", "📈 和值/奇偶/大小", "🎯 五码对照", "📋 历史"])
    with tab1:
        freq = kl8_freq(df, n_recent)
        fig = px.bar(x=freq.index, y=freq.values, title=f"1-80 号码频率（近{n_recent}期）", labels={"x":"号码","y":"出现次数"}, color=freq.values, color_continuous_scale="Viridis")
        fig.update_layout(height=450); st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1: st.write("**热号 Top10**", "  ".join(f"`{i:02d}`({c})" for i, c in freq.nlargest(10).items()))
        with col2: st.write("**冷号 Top10**", "  ".join(f"`{i:02d}`({c})" for i, c in freq.nsmallest(10).items()))
    with tab2:
        miss = kl8_missing(df)
        fig = px.bar(x=miss.index, y=miss.values, title="当前遗漏期数", color=miss.values, color_continuous_scale="YlOrRd")
        fig.update_layout(height=450); st.plotly_chart(fig, use_container_width=True)
        st.write("**最长遗漏 Top10**", "  ".join(f"`{i:02d}`({m}期)" for i, m in miss.nlargest(10).items()))
    with tab3:
        stats = kl8_sum_stats(df, n_recent)
        fig = px.line(stats, x="期号", y="和值", title="20码和值走势", markers=True)
        fig.add_hline(y=stats["和值"].mean(), line_dash="dash", annotation_text=f"均值{stats['和值'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            oe = stats["奇数个数"].value_counts().sort_index()
            fig = px.bar(x=oe.index, y=oe.values, title="奇数个数分布", labels={"x":"奇数个数","y":"期数"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            big = stats["大号个数"].value_counts().sort_index()
            fig = px.bar(x=big.index, y=big.values, title="大号(41-80)个数分布", labels={"x":"大号个数","y":"期数"})
            st.plotly_chart(fig, use_container_width=True)
    with tab4: show_combo_analysis(df, count_combo_hits_kl8, 80, "kl8")
    with tab5:
        show_cols = ["期号", "开奖日期"] + [f"号{i}" for i in range(1, 21)]
        st.dataframe(df.tail(20)[show_cols].iloc[::-1], use_container_width=True, height=400)
        st.download_button("下载全部CSV", df.to_csv(index=False).encode("utf-8-sig"), f"kl8_{datetime.now():%Y%m%d}.csv", "text/csv")

elif lottery == "极速飞艇（PK10）":
    if force_refresh: load_feiting.clear()
    df = load_feiting(ft_days)
    if df.empty: st.error("极速飞艇数据加载失败"); st.stop()
    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="45">', unsafe_allow_html=True)
        st.info("⏱ 已开启自动刷新（约每 45 秒重新拉取最新开奖）。关闭侧边栏勾选可停止。")
    rt = fetch_feiting_latest()
    c1, c2, c3, c4 = st.columns(4)
    latest = df.iloc[-1]
    c1.metric("已加载期数", f"{len(df):,}"); c2.metric("最新期号", latest["期号"])
    c3.metric("开奖时间", str(latest["开奖时间"])[:19])
    gy = int(latest["冠亚和"]); dx_now = "大" if gy > 11 else "小"
    c4.metric("冠亚和 / 大小", f"{gy} / {dx_now}")
    nums = " ".join(f"{latest[p]:02d}" for p in ["冠军", "亚军", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十"])
    st.info(f"最新开奖：{nums}")
    if rt: st.caption(f"实时接口 · 下期 {rt.get('下期期号','')} · 服务器时间 {rt.get('服务器时间','')}")
    _seq_dx = feiting_dx_sequence(df); _seq_ds = feiting_ds_sequence(df); _pat_n = 5
    if len(_seq_dx) >= _pat_n:
        _pat_dx = _seq_dx[-_pat_n:]; _pat_ds = _seq_ds[-_pat_n:]
        _r_dx = luzhu_after_pattern(_seq_dx, _pat_dx); _r_ds = luzhu_after_pattern(_seq_ds, _pat_ds)
        _n = len(_seq_dx)
        _base_dx_da = _seq_dx.count("大") / _n * 100; _base_dx_xi = _seq_dx.count("小") / _n * 100
        _base_ds_dan = _seq_ds.count("单") / _n * 100; _base_ds_shuang = _seq_ds.count("双") / _n * 100
        st.markdown("#### 自动对照 · 下期大小/单双（最近 5 期形态）")
        st.caption("开奖结果更新后自动计算。以下是历史条件出现比例，**不是**真实预测概率，请勿据此投注。")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(f"**大小形态** `{''.join(_pat_dx)}`（最新：{_seq_dx[-1]}）")
            if _r_dx["total"] > 0:
                st.metric("历史样本", f"{_r_dx['total']} 次")
                x1, x2 = st.columns(2)
                x1.metric("下期「大」", f"{_r_dx.get('大%', 0)}%", f"{_r_dx.get('大', 0)} 次")
                x2.metric("下期「小」", f"{_r_dx.get('小%', 0)}%", f"{_r_dx.get('小', 0)} 次")
                st.caption(f"全体基础：大 {_base_dx_da:.1f}% / 小 {_base_dx_xi:.1f}%")
            else: st.info("该大小形态历史样本不足")
        with a2:
            st.markdown(f"**单双形态** `{''.join(_pat_ds)}`（最新：{_seq_ds[-1]}）")
            if _r_ds["total"] > 0:
                st.metric("历史样本", f"{_r_ds['total']} 次")
                x1, x2 = st.columns(2)
                x1.metric("下期「单」", f"{_r_ds.get('单%', 0)}%", f"{_r_ds.get('单', 0)} 次")
                x2.metric("下期「双」", f"{_r_ds.get('双%', 0)}%", f"{_r_ds.get('双', 0)} 次")
                st.caption(f"全体基础：单 {_base_ds_dan:.1f}% / 双 {_base_ds_shuang:.1f}%")
            else: st.info("该单双形态历史样本不足")
        st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 各名次频率", "🏆 冠军 & 冠亚和", "🐉 龙虎/基本形态", "🔴 路珠查询", "📋 历史"])
    with tab1:
        pos_df = feiting_pos_freq(df, min(n_recent, len(df)))
        fig = px.density_heatmap(pos_df, x="号码", y="名次", z="次数", title=f"各名次号码出现热力（近{min(n_recent, len(df))}期）", color_continuous_scale="YlOrRd")
        fig.update_layout(height=480); st.plotly_chart(fig, use_container_width=True)
    with tab2:
        champ, gy_s = feiting_champion_stats(df, min(n_recent, len(df)))
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(x=champ.index, y=champ.values, title="冠军号码频率", labels={"x":"号码","y":"次数"}, color=champ.values, color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(x=gy_s.index, y=gy_s.values, title="冠亚和分布", labels={"x":"冠亚和","y":"次数"}, color=gy_s.values, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        st.write("冠军热号:", " ".join(f"`{i}`" for i in champ.nlargest(3).index))
        st.write("冠亚和热值:", " ".join(f"`{i}`" for i in gy_s.nlargest(5).index))
    with tab3:
        d = df.tail(min(n_recent, len(df))).copy()
        d["龙虎"] = np.where(d["冠军"] > d["第十"], "龙", "虎")
        lt = d["龙虎"].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(values=lt.values, names=lt.index, title="冠军 vs 第十（龙虎）")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            d["冠亚大小"] = np.where(d["冠亚和"] > 11, "大", "小")
            bs = d["冠亚大小"].value_counts()
            fig = px.pie(values=bs.values, names=bs.index, title="冠亚和大小（>11为大）")
            st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.subheader("冠亚和 · 路珠对照（默认每 5 期）")
        st.warning("只做历史形态统计，**不是**预测。每期开奖独立，请勿当作投注依据。")
        mode = st.radio("查询类型", ["大小", "单双"], horizontal=True, key="luzhu_mode")
        if mode == "大小":
            st.caption("规则：冠亚和 > 11 → 大；≤ 11 → 小")
            seq = feiting_dx_sequence(df); labels = ("大", "小"); colors = {"大": "#e63946", "小": "#457b9d"}
        else:
            st.caption("规则：冠亚和为奇数 → 单；偶数 → 双")
            seq = feiting_ds_sequence(df); labels = ("单", "双"); colors = {"单": "#e63946", "双": "#2a9d8f"}
        base_total = len(seq); base_counts = {lb: seq.count(lb) for lb in labels}
        recent_n = st.slider("显示最近路珠期数", 20, 120, 50, key="luzhu_show_n")
        recent_seq = seq[-recent_n:]
        colored = " ".join(f'<span style="color:{colors.get(x,"#333")};font-weight:700">{x}</span>' for x in recent_seq)
        st.markdown(f"**最近 {recent_n} 期（{mode}）：** {colored}", unsafe_allow_html=True)
        pat_len = st.selectbox("对照形态长度（期数）", [3, 4, 5, 6, 7], index=2, key="luzhu_pat_len")
        tail = seq[-pat_len:] if len(seq) >= pat_len else seq
        st.write(f"当前末尾 **{pat_len} 期** 形态：**{''.join(tail)}**  → 最新一期：**{seq[-1]}**")
        st.markdown("---")
        bc = st.columns(len(labels) + 1)
        for i, lb in enumerate(labels):
            pct = base_counts[lb] / base_total * 100 if base_total else 0
            bc[i].metric(lb, f"{base_counts[lb]} 期", f"{pct:.1f}%")
        bc[-1].metric("总期数", base_total)
        st.markdown("---")
        st.write(f"**{pat_len} 期形态对照：出现该串之后，下一期是「{labels[0]} / {labels[1]}」的历史比例**")
        use_tail = st.checkbox(f"使用当前末尾 {pat_len} 期作为查询形态", value=True, key="luzhu_use_tail")
        default_pat = "".join(tail) if use_tail else (labels[0] * pat_len)
        hint = "只含大/小，如 大大大小小" if mode == "大小" else "只含单/双，如 单单双双单"
        pat_text = st.text_input(f"输入 {pat_len} 期形态（{hint}）", value=default_pat, key="luzhu_pat")
        if st.button("查询该形态后的下一期比例", key="luzhu_btn"):
            pattern = parse_pattern(pat_text, "大小" if mode == "大小" else "单双")
            if not pattern: st.error(f"请只输入「{'」和「'.join(labels)}」组成的字符串")
            else:
                result = luzhu_after_pattern(seq, pattern)
                st.success(f"形态：**{''.join(pattern)}**（{len(pattern)} 期）在历史中出现后接一期共 **{result['total']}** 次")
                if result["total"] == 0: st.info("该形态在当前数据中未出现过（或只在最后一期出现，没有下一期）")
                else:
                    cols = st.columns(len(labels)); pie_vals, pie_names = [], []
                    for i, lb in enumerate(labels):
                        cnt = result.get(lb, 0); pct = result.get(f"{lb}%", 0)
                        cols[i].metric(f"下一期是「{lb}」", f"{cnt} 次", f"{pct}%")
                        pie_vals.append(cnt); pie_names.append(lb)
                    fig = px.pie(values=pie_vals, names=pie_names, title=f"形态「{''.join(pattern)}」之后下一期分布", color=pie_names, color_discrete_map=colors)
                    st.plotly_chart(fig, use_container_width=True)
                    base_txt = " / ".join(f"{lb} {base_counts[lb]/base_total*100:.1f}%" for lb in labels)
                    st.caption(f"对照全体基础比例：{base_txt}。历史条件比例接近基础比例是正常现象。")
        st.markdown("---")
        st.write(f"**当前末尾 {pat_len} 期一键对照**")
        if len(seq) >= pat_len:
            r_now = luzhu_after_pattern(seq, seq[-pat_len:])
            if r_now["total"] > 0:
                msg = "　".join(f"{lb} {r_now.get(lb,0)}次（{r_now.get(f'{lb}%',0)}%）" for lb in labels)
                st.info(f"形态 `{''.join(seq[-pat_len:])}` 历史样本 {r_now['total']} 次 → {msg}")
            else: st.info("当前末尾形态在历史中尚无足够样本。")
        st.markdown("---")
        st.write(f"**{pat_len} 期常见形态速查（样本数≥5）**")
        pat_counter = Counter()
        for i in range(len(seq) - pat_len): pat_counter[tuple(seq[i : i + pat_len])] += 1
        top_pats = [list(p) for p, c in pat_counter.most_common(15) if c >= 5]
        qrows = []
        for p in top_pats:
            r = luzhu_after_pattern(seq, p)
            if r["total"] < 5: continue
            row = {"形态": "".join(p), "样本数": r["total"]}
            for lb in labels: row[f"下期{lb}%"] = r.get(f"{lb}%", 0)
            qrows.append(row)
        if qrows: st.dataframe(pd.DataFrame(qrows), use_container_width=True)
        else: st.caption("当前数据量不足，或请加大「拉取最近几天数据」。")
    with tab5:
        show_cols = ["期号", "开奖时间", "冠军", "亚军", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十", "冠亚和"]
        show_df = df.tail(50)[show_cols].copy()
        show_df["大小"] = show_df["冠亚和"].apply(lambda x: "大" if int(x) > 11 else "小")
        show_df["单双"] = show_df["冠亚和"].apply(lambda x: "单" if int(x) % 2 == 1 else "双")
        st.dataframe(show_df.iloc[::-1], use_container_width=True, height=400)
        st.download_button("下载当前数据CSV", df.to_csv(index=False).encode("utf-8-sig"), f"feiting_{datetime.now():%Y%m%d}.csv", "text/csv")

elif lottery == "极速快乐8":
    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="45">', unsafe_allow_html=True)
        st.info("⏱ 已开启自动刷新（约每 45 秒）。关闭侧边栏勾选可停止。")
    latest = fetch_kl8_speed_latest()
    if not latest: st.error("极速快乐8 实时数据获取失败，请稍后重试"); st.stop()
    if "kl8s_history" not in st.session_state: st.session_state.kl8s_history = []
    hist = st.session_state.kl8s_history
    exists = {str(x.get("期号")) for x in hist}
    if str(latest["期号"]) not in exists:
        hist.append({ "期号": latest["期号"], "开奖时间": latest["开奖时间"], "和值": latest["和值"],
            "大小": latest["大小"], "单双": latest["单双"], "组合": latest["组合"], "号码": latest["号码"] })
        hist.sort(key=lambda x: int(str(x["期号"])) if str(x["期号"]).isdigit() else 0)
        st.session_state.kl8s_history = hist[-500:]
    hist = st.session_state.kl8s_history
    combo_seq = [h["组合"] for h in hist]; dx_seq = [h["大小"] for h in hist]; ds_seq = [h["单双"] for h in hist]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新期号", latest["期号"]); c2.metric("开奖时间", str(latest["开奖时间"])[:19])
    c3.metric("和值", latest["和值"]); c4.metric("组合", latest["组合"])
    nums_str = " ".join(f"{n:02d}" for n in latest["号码"])
    st.info(f"最新开奖（20码）：{nums_str} ｜ {latest['大小']} / {latest['单双']} / **{latest['组合']}**")
    st.caption(f"规则：和值≥810 为大，<810 为小；奇数为单、偶数为双 → 组合为大单/大双/小单/小双。下期 {latest.get('下期期号','')} · {latest.get('下期时间','')} · 已缓存 {len(hist)} 期")
    _pat_n = 5
    st.markdown("#### 自动对照 · 下期「大单/大双/小单/小双」（最近 5 期组合形态）")
    st.caption("开奖更新后自动计算。历史条件比例 **≠** 真实概率，请勿据此投注。缓存期数越多，样本越有参考意义。")
    labels4 = ["大单", "大双", "小单", "小双"]
    if len(combo_seq) >= _pat_n:
        pat = combo_seq[-_pat_n:]; r = luzhu_after_pattern(combo_seq, pat)
        st.write(f"当前末尾 5 期组合形态：`{' → '.join(pat)}`")
        if r["total"] > 0:
            cols = st.columns(4)
            for i, lb in enumerate(labels4): cols[i].metric(f"下期「{lb}」", f"{r.get(f'{lb}%', 0)}%", f"{r.get(lb, 0)} / {r['total']} 次")
            base_n = len(combo_seq); base_txt = "　".join(f"{lb} {combo_seq.count(lb)/base_n*100:.1f}%" for lb in labels4)
            st.caption(f"历史样本 {r['total']} 次｜全体基础：{base_txt}")
        else: st.info("该 5 期组合形态在已缓存历史中尚无「后接一期」样本，请继续刷新积累。")
    else: st.info(f"已缓存 {len(combo_seq)} 期，需至少 5 期才能做 5 期形态对照。请开启自动刷新多等几期。")
    st.markdown("---")
    nums = latest["号码"]
    odd_cnt = sum(1 for x in nums if x % 2 == 1); big_cnt = sum(1 for x in nums if x > 40)
    col1, col2, col3 = st.columns(3)
    col1.metric("奇数个数", f"{odd_cnt} / 20"); col2.metric("大号(41-80)个数", f"{big_cnt} / 20"); col3.metric("和值", latest["和值"])
    zones = {"1-20": 0, "21-40": 0, "41-60": 0, "61-80": 0}
    for n in nums:
        if n <= 20: zones["1-20"] += 1
        elif n <= 40: zones["21-40"] += 1
        elif n <= 60: zones["41-60"] += 1
        else: zones["61-80"] += 1
    fig = px.bar(x=list(zones.keys()), y=list(zones.values()), title="当期号码区间分布", labels={"x":"区间","y":"个数"}, color=list(zones.values()), color_continuous_scale="Teal")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("组合路珠手动查询（大单/大双/小单/小双）")
    qmode = st.radio("序列类型", ["组合(四态)", "大小", "单双"], horizontal=True, key="kl8s_qmode")
    if qmode == "组合(四态)": seq_q = combo_seq; allowed = "大单大双小单小双"; hint = "如：大单小双大单小单大双（每项两字，共5项=10字）"
    elif qmode == "大小": seq_q = dx_seq; allowed = "大小"; hint = "如：大小大小大"
    else: seq_q = ds_seq; allowed = "单双"; hint = "如：单双单单双"
    pat_len = st.selectbox("形态期数", [3, 4, 5, 6], index=2, key="kl8s_pat_len")
    use_tail = st.checkbox("使用当前末尾形态", value=True, key="kl8s_use_tail")
    if use_tail and len(seq_q) >= pat_len: default_pat = "".join(seq_q[-pat_len:])
    else: default_pat = ""
    pat_text = st.text_input(f"输入形态（{hint}）", value=default_pat, key="kl8s_pat")
    def parse_combo_pattern(text, mode, length):
        t = text.strip().replace(" ", "").replace("，", "").replace(",", "")
        if mode == "组合(四态)":
            if len(t) % 2 != 0: return None
            items = [t[i:i+2] for i in range(0, len(t), 2)]
            if any(x not in ("大单", "大双", "小单", "小双") for x in items): return None
            return items
        else:
            allowed_ch = "大小" if mode == "大小" else "单双"
            if not t or any(c not in allowed_ch for c in t): return None
            return list(t)
    if st.button("查询形态后下一期比例", key="kl8s_btn"):
        pattern = parse_combo_pattern(pat_text, qmode, pat_len)
        if not pattern: st.error("形态格式不正确")
        else:
            result = luzhu_after_pattern(seq_q, pattern)
            st.success(f"形态：{' → '.join(pattern)}｜历史样本 {result['total']} 次")
            if result["total"] == 0: st.info("样本不足，请继续积累缓存期数")
            else:
                keys = labels4 if qmode == "组合(四态)" else (("大", "小") if qmode == "大小" else ("单", "双"))
                cols = st.columns(len(keys))
                for i, lb in enumerate(keys): cols[i].metric(lb, f"{result.get(f'{lb}%', 0)}%", f"{result.get(lb, 0)} 次")
    st.markdown("---")
    st.write(f"**已缓存开奖（{len(hist)} 期）**")
    if hist:
        hdf = pd.DataFrame(hist)[["期号", "开奖时间", "和值", "大小", "单双", "组合"]].iloc[::-1]
        st.dataframe(hdf.head(50), use_container_width=True, height=300)
    if st.button("清空缓存历史", key="kl8s_clear"):
        st.session_state.kl8s_history = []; st.rerun()
    if st.button("立即刷新最新一期", key="kl8s_refresh"): st.rerun()

st.markdown("---")
st.caption("仅供学习与数据分析练习 | 请理性购彩，远离赌博心态")