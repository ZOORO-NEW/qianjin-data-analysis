#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_data.py — 通用数据分析基线脚本（qianjin-data-analysis 技能）

读取 CSV / XLSX，自动做数据画像 + 计算电商/用户/销售常用指标，
输出一份 Markdown 报告"草稿"，供 qianjin-data-analysis 技能（模型）补完洞察与建议。

依赖:
    pandas, openpyxl(仅读取 .xlsx 时需要)
安装(隔离环境):
    python -m venv .venv && .venv/bin/pip install pandas openpyxl

用法:
    python analyze_data.py <数据文件> [--type ecommerce|user|sales|conversion]
        [--date-col 列名] [--amount-col 列名] [--id-col 列名]
        [--cat-col 列名] [--out 报告.md]

说明:
    自动识别转化漏斗（电商漏斗 / AARRR，支持宽表与长表）与 HEART 指标列；
    含时间/金额/用户ID 时仍会计算趋势、RFM、同期群等基线。

说明:
    列识别支持自动推断（带 hint 参数可强制指定）。
    脚本只负责"计算"，不负责"解读"；最终结论与建议由技能模型补全。
"""

import argparse
import sys
import os
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.stderr.write("缺少依赖 pandas/numpy，请先安装：pip install pandas numpy openpyxl\n")
    sys.exit(2)


# ---------- 类型判断（兼容 pandas 3.0 的 str / datetime 推断） ----------

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def is_datetime(s):
    return pd.api.types.is_datetime64_any_dtype(s.dtype)


# ---------- 文件读取 ----------

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


# ---------- 列自动识别 ----------

def detect_date_col(df):
    for c in df.columns:
        if is_datetime(df[c]):
            return c
    best, best_rate = None, 0.0
    for c in df.columns:
        if is_text(df[c]):
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
            rate = parsed.notna().mean()
            if rate > 0.8 and rate > best_rate:
                best, best_rate = c, rate
    return best


def detect_amount_col(df):
    cands = [c for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1
             and df[c].notna().mean() > 0.5]
    for kw in ["金额", "额", "price", "amount", "gmv", "sales", "营收", "销售额", "消费"]:
        for c in cands:
            if kw in str(c).lower():
                return c
    if cands:
        return max(cands, key=lambda c: df[c].std())
    return None


def detect_id_col(df):
    # 优先用户类关键词
    for kw in ["user", "用户", "客户", "会员", "customer", "openid", "手机", "member", "ma_id"]:
        for c in df.columns:
            if kw in str(c).lower() and df[c].nunique() > 5:
                return c
    # 退而求其次：含 id 但排除 order（订单号不是用户ID）
    for c in df.columns:
        if "id" in str(c).lower() and "order" not in str(c).lower() and df[c].nunique() > 5:
            return c
    return None


def detect_cat_cols(df, exclude):
    cats = []
    for c in df.columns:
        if c in exclude:
            continue
        if is_text(df[c]) or (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 50):
            if 2 <= df[c].nunique() <= 100:
                cats.append(c)
    return cats[:5]


# ---------- 数据画像 ----------

def profile(df):
    lines = [f"- 行数：{len(df)}，列数：{len(df.columns)}", "- 列画像："]
    for c in df.columns:
        nmiss = df[c].isna().sum()
        miss_pct = round(100 * nmiss / max(len(df), 1), 1)
        nuniq = df[c].nunique()
        sample = ""
        if df[c].dropna().shape[0] > 0:
            s = df[c].dropna().astype(str).head(2).tolist()
            sample = " 示例=" + "/".join(s)
        lines.append(f"  - `{c}`：类型={df[c].dtype}，缺失={miss_pct}%，唯一值={nuniq}{sample}")
    return "\n".join(lines)


# ---------- Markdown 表格（不依赖 tabulate） ----------

def df_to_md(df):
    cols = list(map(str, df.columns))
    out = ["| " + " | ".join(cols) + " |"]
    out.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(out)


# ---------- 模块计算 ----------

def time_series(df, date_col, amount_col):
    if not (date_col and amount_col):
        return None
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", format="mixed")
    d = d.dropna(subset=[date_col])
    d["_month"] = d[date_col].dt.to_period("M").astype(str)
    g = d.groupby("_month")[amount_col].agg(["sum", "count"]).reset_index()
    g.columns = ["月份", "总额", "笔数"]
    g["环比%"] = (g["总额"].pct_change() * 100).round(1)
    g["3月移动均"] = g["总额"].rolling(3, min_periods=1).mean().round(2)
    return g


def rfm(df, id_col, date_col, amount_col):
    if not (id_col and date_col and amount_col):
        return None
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", format="mixed")
    d = d.dropna(subset=[date_col, amount_col, id_col])
    if d.empty:
        return None
    now = d[date_col].max() + pd.Timedelta(days=1)
    agg = d.groupby(id_col).agg(
        R_date=(date_col, "max"),
        F=(amount_col, "count"),
        M=(amount_col, "sum"),
    ).reset_index()
    agg["R_days"] = (now - agg["R_date"]).dt.days
    try:
        agg["R_score"] = pd.qcut(agg["R_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        agg["F_score"] = pd.qcut(agg["F"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        agg["M_score"] = pd.qcut(agg["M"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except Exception:
        return None

    def seg(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "重要价值"
        if r <= 2 and f >= 4 and m >= 4:
            return "重要保持"
        if r >= 4 and f <= 2 and m >= 4:
            return "重要发展"
        if r <= 2 and f <= 2 and m >= 4:
            return "重要挽留"
        if r >= 4 and f >= 4:
            return "一般价值"
        if r <= 2 and m >= 4:
            return "一般挽留"
        if r >= 4:
            return "一般发展"
        return "一般维持"

    agg["人群"] = agg.apply(seg, axis=1)
    seg_tab = agg["人群"].value_counts().reset_index()
    seg_tab.columns = ["人群", "用户数"]
    d2 = d.merge(agg[[id_col, "人群"]], on=id_col, how="left")
    gmv = d2.groupby("人群")[amount_col].sum()
    seg_tab["GMV贡献"] = seg_tab["人群"].map(gmv)
    total = seg_tab["GMV贡献"].sum()
    seg_tab["GMV占比%"] = (seg_tab["GMV贡献"] / total * 100).round(1) if total else 0
    return {"seg_table": seg_tab}


def cohort(df, id_col, date_col):
    if not (id_col and date_col):
        return None
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", format="mixed")
    d = d.dropna(subset=[date_col, id_col])
    if d.empty:
        return None
    d["first_month"] = d.groupby(id_col)[date_col].transform("min").dt.to_period("M")
    d["active_month"] = d[date_col].dt.to_period("M")
    d["offset"] = (d["active_month"].astype("int64") - d["first_month"].astype("int64"))
    pivot = d.pivot_table(index=d["first_month"].astype(str), columns="offset",
                          values=id_col, aggfunc="nunique", fill_value=0)
    sizes = pivot[0].replace(0, np.nan)
    pct = (pivot.div(sizes, axis=0) * 100).round(1)
    return pct.head(12)


# ---------- 转化漏斗 / AARRR 自动识别 ----------

FUNNEL_MAPS = [
    ("电商漏斗", [
        ("曝光", ["曝光", "impression", "show", "曝光量", "展现"]),
        ("点击", ["点击", "click"]),
        ("加购", ["加购", "addcart", "cart", "加购数"]),
        ("下单", ["下单", "order"]),
        ("支付", ["支付", "pay", "paid"]),
    ]),
    ("AARRR", [
        ("获取", ["获取", "acquisition", "install", "download", "visit", "访客", "曝光", "uv"]),
        ("激活", ["激活", "activation", "register", "注册", "regist"]),
        ("留存", ["留存", "retention", "retain"]),
        ("收入", ["收入", "revenue"]),
        ("推荐", ["推荐", "referral", "refer", "invite", "分享", "share"]),
    ]),
]


def _stage_label_match(name, kws):
    nl = str(name).lower()
    return any(kw in nl for kw in kws)


def detect_funnel(df):
    """返回 (模型名, [(环节, 量级), ...]) 或 None。支持宽表(环节为列)与长表(环节在列内)。"""
    # 1) 宽表：阶段名直接是列名
    for fname, stages in FUNNEL_MAPS:
        cols = []
        for label, kws in stages:
            hit = next((c for c in df.columns if _stage_label_match(c, kws)), None)
            if hit is None:
                cols = None
                break
            cols.append((label, hit))
        if cols:
            vals = []
            for label, c in cols:
                v = pd.to_numeric(df[c], errors="coerce").dropna()
                if v.empty:
                    return None
                vals.append((label, float(v.mean())))
            return fname, vals
    # 2) 长表：某文本列的值是环节名，另一数值列是量级
    for fname, stages in FUNNEL_MAPS:
        labels = [l for l, _ in stages]
        for c in df.columns:
            if not is_text(df[c]):
                continue
            uniq = set(df[c].dropna().astype(str).unique())
            inter = [l for l in labels if l in uniq]
            if len(inter) >= 3:
                num_c = next((nc for nc in df.columns
                              if nc != c and pd.api.types.is_numeric_dtype(df[nc])), None)
                if num_c is None:
                    return None
                sub = df[[c, num_c]].dropna()
                vals = []
                for label in labels:
                    m = sub[sub[c].astype(str) == label][num_c]
                    if m.empty:
                        vals = None
                        break
                    vals.append((label, float(m.sum())))
                if vals:
                    return fname, vals
    return None


def funnel_table(fname, vals):
    base = vals[0][1] if vals else 0
    rows = []
    prev = None
    for label, v in vals:
        stage_cvr = (v / prev * 100) if prev else None
        cum_cvr = (v / base * 100) if base else None
        rows.append({
            "环节": label,
            "人数/量级": round(v, 2),
            "环节转化率%": round(stage_cvr, 2) if stage_cvr is not None else "-",
            "累计转化率%": round(cum_cvr, 2) if cum_cvr is not None else "-",
        })
        prev = v
    return pd.DataFrame(rows)


def funnel_bottleneck(fname, vals):
    """找出环节转化率最低的那一段，作为最大流失/优化环节。"""
    best = None
    for i in range(1, len(vals)):
        prev_v = vals[i - 1][1]
        cur_v = vals[i][1]
        if prev_v <= 0:
            continue
        cvr = cur_v / prev_v * 100
        if best is None or cvr < best[2]:
            best = (vals[i - 1][0], vals[i][0], cvr)
    if best is None:
        return None
    return f"{best[0]} → {best[1]}（环节转化率 {round(best[2], 2)}%）"


# ---------- HEART 增长质量（轻量自动检出） ----------

def heart_summary(df):
    # 排除已被识别为漏斗阶段名的列，避免重复计入（如"留存"既是 AARRR 阶段也是 HEART 关键词）
    funnel_stage_names = {s for _, stages in FUNNEL_MAPS for s, _ in stages}
    dims = [
        ("H 愉悦度", ["nps", "csat", "满意度", "评分", "好评"]),
        ("E 参与度", ["dau", "mau", "活跃", "登录次数", "参与度", "engagement"]),
        ("A 采纳率", ["采纳", "adoption", "新功能使用"]),
        ("R 留存率", ["留存率", "retention", "留存"]),
        ("T 任务成功率", ["任务成功", "完成率", "task success", "错误率", "耗时"]),
    ]
    rows = []
    for label, kws in dims:
        for c in df.columns:
            cn = str(c).lower()
            if cn in funnel_stage_names:
                continue
            if any(kw in cn for kw in kws) and pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].dropna()
                if not s.empty:
                    rows.append({
                        "维度": label,
                        "指标列": str(c),
                        "均值": round(float(s.mean()), 2),
                        "样本数": int(s.shape[0]),
                    })
    return pd.DataFrame(rows) if rows else None


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--type", choices=["ecommerce", "user", "sales", "conversion"], default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--amount-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--cat-col", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        df = load_file(args.file)
    except Exception as e:
        sys.stderr.write(f"读取文件失败：{e}\n")
        sys.exit(1)

    date_col = args.date_col or detect_date_col(df)
    amount_col = args.amount_col or detect_amount_col(df)
    id_col = args.id_col or detect_id_col(df)
    cat_cols = [args.cat_col] if args.cat_col else detect_cat_cols(df, [date_col, amount_col, id_col])

    out = []
    out.append(f"# 数据分析草稿 · {os.path.basename(args.file)}\n")
    out.append(f"> 自动生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} ｜ 类型推测：{args.type or '未指定'}\n")
    out.append("## 一、数据画像\n")
    out.append(profile(df))
    out.append("\n## 二、自动识别的关键列\n")
    out.append(f"- 时间列：{date_col}\n- 金额/数值列：{amount_col}\n- 用户ID列：{id_col}\n- 分类列：{', '.join(cat_cols) if cat_cols else '无'}\n")

    ts = time_series(df, date_col, amount_col)
    if ts is not None:
        out.append("\n## 三、时间趋势（按月）\n")
        out.append(df_to_md(ts))

    r = rfm(df, id_col, date_col, amount_col)
    if r is not None:
        out.append("\n## 四、RFM 用户分层\n")
        out.append(df_to_md(r["seg_table"]))

    co = cohort(df, id_col, date_col)
    if co is not None:
        out.append("\n## 五、同期群留存（%)\n")
        out.append(df_to_md(co.reset_index().rename(columns={"first_month": "首购月"})))

    fn = detect_funnel(df)
    if fn is not None:
        fname, vals = fn
        out.append(f"\n## 六、转化漏斗（{fname}）\n")
        out.append(df_to_md(funnel_table(fname, vals)))
        bn = funnel_bottleneck(fname, vals)
        if bn:
            out.append(f"\n> 最大流失/优化环节：**{bn}**（优先排查该环节病因，详见 references/conversion-models.md）\n")

    hs = heart_summary(df)
    if hs is not None:
        out.append("\n## 七、HEART 增长质量指标（自动检出）\n")
        out.append(df_to_md(hs))
        out.append("\n> HEART 五维需结合业务目标解读（愉悦度/参与度/采纳率/留存/任务成功率），详见 references/conversion-models.md。\n")

    out.append("\n---\n> 以上为自动计算基线。请 qianjin-data-analysis 技能据此补完：现象解读、原因假设、可视化建议与行动建议表。\n")

    text = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"报告已写入：{args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
