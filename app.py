# -*- coding: utf-8 -*-
"""
彩票 AI 数据分析助手 v2.0 - 重构优化版
api.api16868.com · 仅供学习娱乐

优化内容：
1. 适配器抽象层 - 新增彩种无需修改UI
2. 熔断器 - 防止API级联故障
3. 批量数据库写入 + 优化索引
4. Bonferroni校正统计检验
5. Pydantic配置管理
6. 严格类型化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# 导入优化后的模块
from config import settings, LOTTERY_CATALOG
from models.types import LotteryConfig
from adapters import create_adapter
from service import PredictionService
from engine import (
    evaluate_models, model_tracking_summary, 
    adaptive_pattern_model, luzhu_after_pattern,
    derive_combo_probabilities
)
from db import db
from utils.http_utils import parse_api_time

st.set_page_config(page_title="极速彩数据分析 v2", page_icon="🎱", layout="wide", initial_sidebar_state="expanded")

# ==================== CSS 样式 ====================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600;700;900&display=swap');
  html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #8b0000 0%, #4a0000 55%, #1a0000 100%);
  }
  [data-testid="stSidebar"] * { color: #ffe8a1 !important; }

  .main-header {
    font-size: 1.9rem; font-weight: 900; text-align: center; color: #c41e3a;
    text-shadow: 0 1px 0 #fff, 0 2px 8px rgba(196,30,58,0.25);
    letter-spacing: 0.08em; margin-bottom: 0.1rem;
  }
  .sub-header { text-align: center; color: #8a6d3b; font-size: 0.9rem; margin-bottom: 0.6rem; }
  .disclaimer {
    background: #fff8e7; border: 1px solid #e8d5a3; border-left: 5px solid #c41e3a;
    padding: 10px 14px; margin: 6px 0 12px; border-radius: 6px;
    font-size: 0.88rem; color: #5c4a00; line-height: 1.45;
  }
  .live-board {
    background: linear-gradient(145deg, #1a0505 0%, #3d0a0a 40%, #1a0505 100%);
    border: 2px solid #c9a227; border-radius: 14px;
    padding: 16px 18px 14px; margin: 8px 0 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,215,0,0.2);
  }
  .live-board .title {
    color: #ffd700; font-weight: 800; font-size: 1.05rem; text-align: center;
    letter-spacing: 0.12em; margin-bottom: 10px;
  }
  .live-board .meta {
    color: #e8c96a; font-size: 0.85rem; text-align: center; margin-bottom: 10px;
  }
  .num-ball {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2.15rem; height: 2.15rem; margin: 0 4px 6px;
    border-radius: 50%; font-weight: 800; font-size: 0.88rem; color: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.35), inset 0 -2px 3px rgba(0,0,0,0.2);
  }
  .num-ball.gold { background: radial-gradient(circle at 30% 30%, #ffe566, #d4a017 70%); color: #3d2a00; }
  .num-ball.green { background: radial-gradient(circle at 30% 30%, #5eead4, #0f766e 70%); }
  .balls-row { text-align: center; line-height: 2.4; }
  .tag-combo {
    display: inline-block; margin-top: 8px; padding: 4px 14px; border-radius: 999px;
    font-weight: 800; font-size: 0.95rem; color: #1a0505;
    background: linear-gradient(90deg, #ffd700, #f0c14e);
  }
  .pred-card {
    background: #fffef8; border: 1px solid #e8d5a3; border-radius: 12px;
    padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .pred-card h4 { margin: 0 0 6px; color: #7f1d1d; font-size: 0.95rem; }
  .sig-ok { color: #15803d; font-weight: 800; }
  .sig-bad { color: #b91c1c; font-weight: 800; }
  .countdown-box {
    background: linear-gradient(90deg, #7f1d1d, #b91c1c);
    border: 1px solid #fbbf24; border-radius: 12px;
    padding: 12px 16px; text-align: center; margin: 8px 0 12px; color: #fff;
  }
  .countdown-box .time {
    font-size: 1.85rem; font-weight: 900; letter-spacing: 3px; color: #fde68a;
    font-variant-numeric: tabular-nums;
  }
  .footer-note {
    text-align: center; color: #a89878; font-size: 0.8rem;
    margin-top: 1.2rem; padding-top: 0.7rem; border-top: 1px solid #e8d5a3;
  }
  [data-testid="stMetric"] {
    background: #fffef8; border-radius: 10px; padding: 10px 12px;
    border: 1px solid #f0e6c8;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f5e6c8; padding: 5px; border-radius: 10px;
  }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 700; color: #5c4a00; }
  .stTabs [aria-selected="true"] {
    background: #c41e3a !important; color: #fff !important;
  }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 ====================
@st.cache_resource
def get_service(pred_key: str):
    return PredictionService(pred_key)

def get_config(name: str) -> LotteryConfig:
    item = next(x for x in LOTTERY_CATALOG if x["name"] == name)
    return LotteryConfig(key=item["key"], name=item["name"], type=item["type"], code=item["code"])

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 设置")
    lottery_name = st.selectbox("选择彩种", [x["name"] for x in LOTTERY_CATALOG], index=0)
    config = get_config(lottery_name)
    adapter = create_adapter(config)
    service = get_service(adapter.pred_key)

    force_refresh = st.button("🔄 强制刷新数据")
    n_recent = st.slider("分析最近期数", 30, 500, 100, 10)
    ft_days = st.slider("拉取最近几天数据", 1, 14, 7)

    st.caption("模型采用自动选模：3/4/5/6期、20/30/50期频率、综合集成")
    st.caption("含 Bonferroni 校正严格滚动回测")
    auto_refresh = st.checkbox("⏱ 实时看板自动刷新（5 秒）", value=True)
    if auto_refresh:
        st.caption("实时开奖 + 预测结果独立局部刷新")

    st.markdown("---")
    st.caption(f"当前 lotCode = {config.code}")
    st.caption("数据源：api.api16868.com")
    with st.expander("全部彩种列表"):
        for x in LOTTERY_CATALOG:
            st.write(f"· {x['name']}（{x['code']} / {x['type']}）")

# ==================== 主界面 ====================
st.markdown('<div class="main-header">🎱 极速彩数据分析助手 v2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">api.api16868.com · 仅供学习娱乐 · Bonferroni校正</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
⚠️ <b>重要声明</b>：开奖为随机事件，历史无法预测未来。本工具只做统计可视化，不提供中奖保证。
模型经过 Bonferroni 多重比较校正，p值仅反映历史统计特征，不代表真实预测能力。请理性对待。
</div>
""", unsafe_allow_html=True)

# ==================== 加载历史数据 ====================
@st.cache_data(ttl=settings.cache_ttl_history, show_spinner="加载历史数据...")
def load_data(adapter, days: int):
    return adapter.fetch_history(days)

if force_refresh:
    load_data.clear()

df = load_data(adapter, ft_days)
if df.empty:
    st.error("数据加载失败，请稍后重试")
    st.stop()

latest = df.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("已加载期数", f"{len(df):,}")
c2.metric("历史最新期号", latest["期号"])
c3.metric("历史开奖时间", str(latest["开奖时间"])[:19])

# 提取序列
sequences = adapter.extract_sequences(df)

# ==================== 实时看板（局部刷新）====================
@st.fragment(run_every=5 if auto_refresh else None)
def render_live():
    if st.button("🔄 立即刷新实时", key="live_refresh_btn"):
        st.cache_data.clear()

    rt = adapter.fetch_latest()
    if not rt:
        st.warning("实时接口暂不可用")
        return None

    html = adapter.render_live(rt)
    st.markdown(f'<div class="live-board">{html}</div>', unsafe_allow_html=True)

    # 倒计时
    server = parse_api_time(rt.get("服务器时间"))
    draw = parse_api_time(rt.get("下期时间"))
    next_issue = rt.get("下期期号") or rt.get("drawIssue") or ""

    if draw:
        now = server or datetime.now()
        remain = (draw - now).total_seconds()
        if remain < -30:
            st.warning("可能已开奖，请刷新 · 下期 " + str(next_issue))
        else:
            remain = max(0, remain)
            m, sec = divmod(int(remain), 60)
            h, m = divmod(m, 60)
            tstr = ("%02d:%02d:%02d" % (h, m, sec)) if h > 0 else ("%02d:%02d" % (m, sec))
            draw_s = draw.strftime("%H:%M:%S")
            now_s = now.strftime("%H:%M:%S") if server else "本地"
            html = (
                '<div class="countdown-box">'
                '⏳ <b>下期开奖</b>　第 <b>%s</b> 期<br/>'
                '<span class="time">%s</span>'
                '<div style="font-size:0.8rem;opacity:0.9;margin-top:4px;">预计 %s　校对时间 %s</div></div>'
            ) % (next_issue, tstr, draw_s, now_s)
            st.markdown(html, unsafe_allow_html=True)

    st.caption("服务器时间 %s" % rt.get("服务器时间", ""))
    return rt

rt = render_live()

# ==================== 预测面板（局部刷新）====================
@st.fragment(run_every=5 if auto_refresh else None)
def render_prediction():
    if not rt:
        st.warning("预测实时接口暂不可用")
        return

    issue = str(rt.get("期号", ""))

    # 结算上一期
    actual_map = {}
    for cat in adapter.default_categories:
        actual = adapter.get_actual(rt, cat)
        if actual:
            actual_map[cat] = actual

    # 组合由大小×单双生成
    if "大小" in actual_map and "单双" in actual_map:
        actual_map["组合"] = actual_map["大小"] + actual_map["单双"]

    service.settle(issue, actual_map)

    # 生成新预测
    st.markdown("#### 📊 自动预测 · 独立实时刷新")
    st.caption(f"实时期号：{issue} · 每 5 秒检查一次")

    preds = service.generate_predictions(sequences, issue)

    # 渲染预测卡片
    categories = list(preds.keys())
    n_cols = min(len(categories), 4)
    cols = st.columns(n_cols)

    for i, cat in enumerate(categories):
        pred = preds[cat]
        with cols[i % n_cols]:
            sig_class = "sig-ok" if pred.get("significant") else "sig-bad"
            sig_text = "显著" if pred.get("significant") else "不显著"

            st.markdown(
                f'<div class="pred-card">'
                f'<h4>{cat} · {pred["model_name"]}</h4>'
                f'<div style="font-size:0.75rem;color:#666;">'
                f'p={pred["p_value"]:.3f} <span class="{sig_class}">{sig_text}</span>'
                f'</div>'
                f'<div style="margin-top:6px;">'
                f'倾向：<b>{pred["lean"]}</b>（{pred["pct"]}%）<br/>'
                f'样本：{pred["sample"]} · 置信度：<b>{pred["confidence"]}</b>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    # 组合推导（仅 Luck20）
    if config.type == "luck20" and "大小" in preds and "单双" in preds:
        combo = derive_combo_probabilities(
            {"大%": preds["大小"]["pct"], "小%": 100 - preds["大小"]["pct"]},
            {"单%": preds["单双"]["pct"], "双%": 100 - preds["单双"]["pct"]}
        )
        combo_lean = max(combo, key=combo.get) if combo else None
        st.markdown(
            f'<div class="pred-card" style="margin-top:8px;">'
            f'<h4>组合 · 由大小×单双推导</h4>'
            f'{" · ".join(f"{k} {v}%" for k, v in combo.items())}<br/>'
            f'倾向：<b>{combo_lean}</b>（{combo.get(combo_lean, 0)}%）'
            f'</div>',
            unsafe_allow_html=True
        )

render_prediction()

# ==================== 数据分析标签页 ====================
seq_dx = sequences.get("大小", [])
seq_ds = sequences.get("单双", [])

# 彩种专属标签
if config.type == "pks":
    tabs = st.tabs(["📊 名次频率", "🏆 冠军&冠亚和", "🐉 龙虎", "🔴 路珠", "📋 历史", "🧪 模型实验室", "🏆 模型追踪", "✅ 预测历史"])
else:
    tabs = st.tabs(["📊 号码频率", "📈 和值", "🎲 大小单双", "🔴 路珠", "📋 历史", "🧪 模型实验室", "🏆 模型追踪", "✅ 预测历史"])

recent_df = df.tail(min(n_recent, len(df)))

# ---------- 标签页内容 ----------
if config.type == "pks":
    # PK10 专属
    with tabs[0]:
        records = []
        for pos in ["冠军", "亚军", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十"]:
            vc = recent_df[pos].value_counts().reindex(range(1, 11), fill_value=0)
            for num, cnt in vc.items():
                records.append({"名次": pos, "号码": num, "次数": cnt})
        fig = px.density_heatmap(pd.DataFrame(records), x="号码", y="名次", z="次数",
                                  title=f"各名次热力（近{min(n_recent,len(df))}期）", color_continuous_scale="YlOrRd")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        champ = recent_df["冠军"].value_counts().sort_index()
        gy_s = recent_df["冠亚和"].value_counts().sort_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(x=champ.index, y=champ.values, title="冠军频率", color=champ.values, color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(x=gy_s.index, y=gy_s.values, title="冠亚和分布", color=gy_s.values, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        d = recent_df.copy()
        d["龙虎"] = np.where(d["冠军"] > d["第十"], "龙", "虎")
        d["冠亚大小"] = np.where(d["冠亚和"] > 11, "大", "小")
        c1, c2 = st.columns(2)
        with c1:
            lt = d["龙虎"].value_counts()
            st.plotly_chart(px.pie(values=lt.values, names=lt.index, title="龙虎"), use_container_width=True)
        with c2:
            bs = d["冠亚大小"].value_counts()
            st.plotly_chart(px.pie(values=bs.values, names=bs.index, title="冠亚和大小"), use_container_width=True)

else:
    # Luck20 专属
    with tabs[0]:
        cols = [c for c in df.columns if c.startswith("号") and c[1:].isdigit()]
        freq = pd.Series(recent_df[cols].values.flatten()).value_counts().reindex(range(1, 81), fill_value=0)
        fig = px.bar(x=freq.index, y=freq.values, title=f"1-80 频率（近{min(n_recent,len(df))}期）",
                     color=freq.values, color_continuous_scale="Viridis")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.write("**热号** " + " ".join(f"`{i:02d}`" for i in freq.nlargest(10).index))
        c2.write("**冷号** " + " ".join(f"`{i:02d}`" for i in freq.nsmallest(10).index))

    with tabs[1]:
        fig = px.line(recent_df, x="期号", y="和值", title="和值走势", markers=True)
        fig.add_hline(y=recent_df["和值"].mean(), line_dash="dash")
        fig.add_hline(y=810, line_dash="dot", annotation_text="810 分界")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        d = recent_df
        c1, c2, c3 = st.columns(3)
        with c1:
            vc = d["大小"].value_counts()
            st.plotly_chart(px.pie(values=vc.values, names=vc.index, title="大小"), use_container_width=True)
        with c2:
            vc = d["单双"].value_counts()
            st.plotly_chart(px.pie(values=vc.values, names=vc.index, title="单双"), use_container_width=True)
        with c3:
            vc = d["组合"].value_counts()
            st.plotly_chart(px.pie(values=vc.values, names=vc.index, title="四组合"), use_container_width=True)

# ---------- 路珠（通用）----------
with tabs[3]:
    mode_options = ["大小", "单双"]
    if config.type == "luck20":
        mode_options.append("组合")
    mode = st.radio("类型", mode_options, horizontal=True, key=f"luzhu_mode_{config.code}")

    seq = sequences.get(mode, [])
    if len(seq) < 3:
        st.info(f"历史仅 {len(seq)} 期，至少需 3 期才能对照。")
    else:
        labels = adapter.get_labels(mode)
        colors = {"大": "#e63946", "小": "#457b9d", "单": "#e63946", "双": "#2a9d8f"}

        recent_n = st.slider("显示最近路珠期数", 20, 120, 50, key=f"luzhu_show_{config.code}")
        recent_seq = seq[-recent_n:]
        colored = " ".join(
            f'<span style="color:{colors.get(x, "#333")};font-weight:700">{x}</span>' for x in recent_seq
        )
        st.markdown(f"**最近 {recent_n} 期：** {colored}", unsafe_allow_html=True)

        pat_len = st.selectbox("形态长度", [3, 4, 5, 6, 7], index=2, key=f"luzhu_len_{config.code}")
        tail = seq[-pat_len:] if len(seq) >= pat_len else seq
        st.write(f"末尾 {pat_len} 期：**{' → '.join(tail)}** → 最新 **{seq[-1] if seq else '-'}**")

        # 自适应模型
        model = adaptive_pattern_model(seq, labels, lengths=(3, 4, 5, 6))
        st.write(f"自适应预测：**{model.lean}**（{model.pct}%）· 样本{model.sample} · {model.confidence}")

        # 形态查询
        is_combo = mode == "组合"
        default_pat = "".join(tail) if not is_combo else ""
        pat_text = st.text_input(f"输入形态", value=default_pat, key=f"luzhu_pat_{config.code}")

        if st.button("查询下一期比例", key=f"luzhu_btn_{config.code}"):
            if is_combo:
                t = pat_text.strip().replace(" ", "").replace("，", "").replace(",", "")
                pattern = [t[i:i+2] for i in range(0, len(t), 2)] if len(t) % 2 == 0 else None
                if pattern and any(x not in labels for x in pattern):
                    pattern = None
            else:
                allowed = "".join(labels)
                t = pat_text.strip().replace("，", "").replace(",", "").replace(" ", "").replace("　", "")
                pattern = list(t) if t and all(c in allowed for c in t) else None

            if not pattern:
                st.error("形态格式不正确")
            else:
                result = luzhu_after_pattern(seq, pattern)
                st.success(f"形态 {' → '.join(pattern)}｜样本 {result['total']}")
                if result["total"] > 0:
                    cols = st.columns(len(labels))
                    for i, lb in enumerate(labels):
                        cols[i].metric(f"下期「{lb}」", f"{result.get(f'{lb}%', 0)}%", f"{result.get(lb, 0)} 次")

# ---------- 历史 ----------
with tabs[4]:
    if config.type == "pks":
        show = df.tail(50)[["期号", "开奖时间"] + ["冠军", "亚军", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十", "冠亚和"]].copy()
        show["大小"] = np.where(show["冠亚和"].astype(int) > 11, "大", "小")
        show["单双"] = np.where(show["冠亚和"].astype(int) % 2 == 1, "单", "双")
    else:
        cols_show = ["期号", "开奖时间", "和值", "大小", "单双", "组合"]
        if "附加号" in df.columns:
            cols_show.insert(3, "附加号")
        show = df.tail(50)[[c for c in cols_show if c in df.columns]].copy()

    st.dataframe(show.iloc[::-1], use_container_width=True, height=400)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载CSV", csv, f"{config.type}_{config.code}_{datetime.now():%Y%m%d}.csv", "text/csv")

# ---------- 模型实验室 ----------
with tabs[5]:
    st.markdown("#### 🧪 模型实验室 · 自动选模 + 严格滚动回测 + Bonferroni校正")
    st.caption("模型按时间滚动验证；综合分不是未来中奖概率。p值经Bonferroni校正（α=0.05/8=0.00625）")

    if len(seq_dx) > 40 and len(seq_ds) > 40:
        min_history = st.slider("最少历史训练期数", 30, 120, 40, 10, key=f"bt_min_{config.code}")

        tab_dx, tab_ds = st.tabs(["大小", "单双"])
        with tab_dx:
            rows = evaluate_models(seq_dx, ("大", "小"), min_history=min_history)
            if rows:
                table = []
                for r in rows:
                    table.append({
                        "模型": r["模型"], "长期样本": r["长期样本"], "长期准确率": f'{r["长期准确率"]:.2f}%',
                        "长期显著": "✓" if r["长期显著"] else "✗",
                        "p值": f'{r["长期p值"]:.4f}',
                        "近期样本": r["近期样本"], "近期准确率": f'{r["近期准确率"]:.2f}%',
                        "综合分": f'{r["综合分"]:+.2f}'
                    })
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

                best = max(rows, key=lambda r: r["综合分"])
                if best["长期显著"] and best["长期优势"] > 0:
                    st.success(f'当前综合表现最佳：{best["模型"]}（p={best["长期p值"]:.4f}，显著）')
                else:
                    st.warning(f'当前没有经校正后显著优于随机的稳定模型（最佳p={best["长期p值"]:.4f}）')
            else:
                st.info("历史样本不足")

        with tab_ds:
            rows = evaluate_models(seq_ds, ("单", "双"), min_history=min_history)
            if rows:
                table = []
                for r in rows:
                    table.append({
                        "模型": r["模型"], "长期样本": r["长期样本"], "长期准确率": f'{r["长期准确率"]:.2f}%',
                        "长期显著": "✓" if r["长期显著"] else "✗",
                        "p值": f'{r["长期p值"]:.4f}',
                        "近期样本": r["近期样本"], "近期准确率": f'{r["近期准确率"]:.2f}%',
                        "综合分": f'{r["综合分"]:+.2f}'
                    })
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            else:
                st.info("历史样本不足")
    else:
        st.info("历史样本不足，需要至少40期")

# ---------- 模型追踪 ----------
with tabs[6]:
    st.markdown("#### 🏆 模型表现追踪 · 实际上线后结算统计")
    st.caption("与回测分开统计，避免把回测结果冒充实盘表现")

    c1, c2 = st.columns(2)
    category = c1.selectbox("类别", ["全部", "大小", "单双", "组合"], key=f"trk_cat_{config.code}")
    window = c2.selectbox("追踪窗口", [50, 100, 300, 500, 1000], index=2, key=f"trk_win_{config.code}")

    rows = model_tracking_summary(adapter.pred_key, category, window)
    if rows:
        show = []
        for r in rows:
            show.append({
                '模型': r['模型'], '样本': r['样本'], '准确率': f"{r['准确率']:.2f}%",
                '95%下界': f"{r['下界']:.2f}%", '95%上界': f"{r['上界']:.2f}%",
                '近期样本': r['近期样本'], '近期准确率': f"{r['近期准确率']:.2f}%",
                '相对基准': f"{r['相对基准']:+.2f}pt", '状态': r['状态']
            })
        st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True)

        best = rows[0]
        base = 25.0 if category == '组合' else 50.0
        if best['下界'] > base:
            st.success(f"当前追踪中最稳定的模型：{best['模型']}；但仍需更多样本确认。")
        else:
            st.warning("目前没有模型的95%置信下界稳定高于随机基准。")
    else:
        st.info("暂时没有足够的已结算模型记录。")

# ---------- 预测历史 ----------
with tabs[7]:
    st.subheader("预测历史 · 对错统计")
    st.caption("根据自动对照的「倾向」与下期实际开奖比对。仅供复盘。")

    hist = service.get_history(limit=1000)
    if not hist:
        st.caption("暂无预测记录。新期出现后会自动保存并结算。")
    else:
        categories = list(dict.fromkeys(x.cat for x in hist if x.cat))
        c1, c2, c3 = st.columns(3)
        selected_cat = c1.selectbox("预测类型", ["全部"] + categories, key=f"pred_cat_{config.code}")
        selected_result = c2.selectbox("验证结果", ["全部", "对", "错", "待开"], key=f"pred_res_{config.code}")
        window = c3.selectbox("统计窗口", [50, 100, 300, 500, 1000], index=1, key=f"pred_win_{config.code}")

        filtered = hist[:window]
        if selected_cat != '全部':
            filtered = [x for x in filtered if x.cat == selected_cat]
        if selected_result != '全部':
            filtered = [x for x in filtered if x.result == selected_result]

        settled = [x for x in filtered if x.result in ('对', '错')]
        ok = sum(1 for x in settled if x.result == '对')
        bad = len(settled) - ok
        rate = ok / len(settled) * 100 if settled else 0
        base = 50.0 if selected_cat != '组合' else 25.0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric('记录', len(filtered))
        c2.metric('已验证', len(settled))
        c3.metric('对', ok)
        c4.metric('错', bad)
        c5.metric('正确率', f'{rate:.1f}%')
        c6.metric('相对基准', f'{rate-base:+.1f}pt')

        st.caption(f'随机基准：{base:.0f}%。正确率优势只有在足够样本量下才有参考价值。')

        # 分类汇总
        if selected_cat == '全部' and categories:
            rows = []
            for cat in categories:
                ch = [x for x in filtered if x.cat == cat]
                ss = [x for x in ch if x.result in ('对', '错')]
                rr = sum(1 for x in ss if x.result == '对') / len(ss) * 100 if ss else 0
                b = 25 if cat == '组合' else 50
                rows.append({
                    '预测类型': cat, '记录': len(ch), '已验证': len(ss),
                    '对': sum(1 for x in ss if x.result == '对'),
                    '错': sum(1 for x in ss if x.result == '错'),
                    '正确率': f'{rr:.1f}%', '相对基准': f'{rr-b:+.1f}pt'
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 详细记录
        rows = []
        for x in filtered[:100]:
            rows.append({
                '记录期号': x.issue, '结算期': x.settle_issue or '-',
                '类型': x.cat, '形态': x.pattern, '倾向': x.lean,
                '样本': x.sample, '比例%': x.pct, '置信度': x.confidence,
                '模型': x.model_name or '-', '实际': x.actual or '-',
                '结果': {'对': '✅ 对', '错': '❌ 错', '待开': '⏳ 待开'}.get(x.result, x.result),
                '时间': x.time
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400, hide_index=True)

        if st.button("清空本彩种预测记录", key=f"clr_{config.code}"):
            service.clear()
            st.rerun()

st.markdown('<div class="footer-note">仅供学习 · 请理性购彩，远离赌博心态</div>', unsafe_allow_html=True)
