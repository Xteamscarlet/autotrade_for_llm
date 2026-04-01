# utils/stock_filter.py
# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Tuple

import pandas as pd

# ---------- 1. 文本级别：从“股票名称”识别 ST/退市 ----------
_ST_PATTERNS = [
    re.compile(r"ST", re.IGNORECASE),
    re.compile(r"退市"),
    # 后续你要补充新规则，直接在这里加
]

def is_st_or_delisted_by_name(name: str) -> bool:
    """
    判断股票名称是否命中 ST/退市（按文本规则）。
    规则后续都在这里改，一处修改全局生效。
    """
    if not isinstance(name, str):
        return False
    for pat in _ST_PATTERNS:
        if pat.search(name):
            return True
    return False


# ---------- 2. 日线级别：从个股日线 DataFrame 判断是否需要拦截 ----------
def should_intercept_stock(
    stock_code: str,
    stock_name: str,
    df,
    today=None,               # 不传就用 df 最后一个交易日；回测里可以传“回测起始日”
    allow_recent_suspended: bool = False,
) -> Tuple[bool, str]:
    """
    统一拦截入口（函数级改动：只改这一个函数的内部逻辑即可扩展规则）。

    返回: (should_skip, reason)
      - should_skip=True：建议跳过（回测/预测都不要再往下走）
      - reason：人可读的拦截原因，方便打日志/报表
    """
    # 规则-0：名称层面的 ST/退市
    if is_st_or_delisted_by_name(stock_name):
        return True, f"[名称规则] ST/退市: {stock_name} ({stock_code})"

    # 若 df 为空直接放行/拦截（按你自己的策略定）
    if df is None or not hasattr(df, "index") or len(df) == 0:
        return True, f"[数据异常] 日线为空或非 DataFrame: {stock_code}"

    # 规则-1：停牌过久（按你的规则写）
    # 示例：最近 N 个交易日有 >= M 天停牌（成交量为 0）
    # 这里以最近 20 个交易日停牌天数 >= 10 为例
    try:
        last_n = 20
        sample = df["Volume"].iloc[-last_n:] if len(df) >= last_n else df["Volume"]
        if len(sample) == 0:
            return False, ""  # 无量数据就不拦截
        suspended_days = (sample == 0).sum()
        if suspended_days >= 10:
            return True, f"[停牌规则] 近{len(sample)}日停牌天数{suspended_days}>=10"
    except Exception:
        pass  # 不因异常导致整体崩溃

    # 规则-2：退市日期标记（如果你的日线里有 "DelistDate" 列）
    try:
        if "DelistDate" in df.columns:
            if today is None:
                today = df.index[-1]
            delist = df["DelistDate"].dropna().max()
            if pd.notna(delist) and today >= delist:
                return True, f"[退市日规则] 已退市（退市日期={delist}, 当前={today}）"
    except Exception:
        pass

    # 规则-3：预留：涨跌停、科创板/创业板特殊限制等，后续在这里扩展

    return False, ""


# ---------- 3. 批量过滤（用在“股票池”阶段） ----------
def filter_codes_by_name(
    mapping: Dict[str, str],  # 通常是 {name: code}
    verbose: bool = True,
) -> Dict[str, str]:
    """
    批量根据名称剔除 ST/退市（简单文本规则）。
    mapping: {stock_name: stock_code}
    返回: 过滤后的 {stock_name: stock_code}
    """
    filtered = {}
    for name, code in mapping.items():
        if is_st_or_delisted_by_name(name):
            if verbose:
                print(f"[拦截] 股票池移除 ST/退市: {name} ({code})")
            continue
        filtered[name] = code
    return filtered
