# -*- coding: utf-8 -*-
"""
信号置信度分级模块
将二元信号（超过/未超过阈值）细分为强/中/弱三级
"""
from typing import Dict, Tuple


def classify_signal_confidence(
    score: float,
    threshold: float,
) -> Tuple[str, float]:
    """信号置信度分级

    Args:
        score: 综合得分
        threshold: 买入阈值

    Returns:
        (级别, 建议仓位比例)
        级别: 'strong' / 'medium' / 'weak' / 'none'
    """
    excess = score - threshold

    if excess > 0.3:
        return 'strong', 1.0
    elif excess > 0.15:
        return 'medium', 0.7
    elif excess > 0:
        return 'weak', 0.4
    else:
        return 'none', 0.0


def filter_by_microstructure(
    code: str,
    current_price: float,
    prev_close: float,
    is_st: bool = False,
) -> Tuple[bool, str]:
    """市场微观结构过滤

    检查：涨跌停、ST股

    Returns:
        (是否允许交易, 原因)
    """
    if is_st:
        return False, "ST股，额外风险"

    limit_ratio = 0.10
    if code.startswith('300') or code.startswith('688'):
        limit_ratio = 0.20

    limit_up = prev_close * (1 + limit_ratio)
    limit_down = prev_close * (1 - limit_ratio)

    if current_price >= limit_up * 0.995:
        return False, "涨停，无法买入"
    if current_price <= limit_down * 1.005:
        return False, "跌停，无法卖出"

    return True, ""
