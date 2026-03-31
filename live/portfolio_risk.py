# -*- coding: utf-8 -*-
"""
组合层面风控模块
总仓位控制、单日最大亏损限制
"""
import logging
from typing import List, Dict, Tuple

from risk_manager import RiskManager
from config import get_settings

logger = logging.getLogger(__name__)


def check_portfolio_limits(
    current_positions: List[Dict],
    new_candidates: List[Dict],
    max_total_ratio: float = 0.8,
) -> Tuple[List[Dict], List[str]]:
    """组合仓位风控

    Args:
        current_positions: 当前持仓 [{'code': str, 'ratio': float, 'name': str}]
        new_candidates: 新买入候选 [{'code': str, 'ratio': float, 'name': str, 'score': float}]
        max_total_ratio: 最大总仓位

    Returns:
        (过滤后的候选, 警告列表)
    """
    settings = get_settings()
    rm = RiskManager(settings.risk)

    filtered, warnings = rm.check_portfolio_risk(
        current_positions=current_positions,
        new_candidates=new_candidates,
        max_total_ratio=max_total_ratio,
        max_single_ratio=settings.risk.max_position_ratio,
    )

    return filtered, warnings


def check_daily_loss_limit(
    daily_pnl: float,
    total_capital: float,
    max_daily_loss_ratio: float = 0.03,
) -> Tuple[bool, str]:
    """单日最大亏损检查

    Returns:
        (是否允许继续交易, 原因)
    """
    loss_ratio = abs(daily_pnl) / total_capital if total_capital > 0 else 0
    if daily_pnl < 0 and loss_ratio > max_daily_loss_ratio:
        return False, f"单日亏损({loss_ratio:.1%})超过限制({max_daily_loss_ratio:.1%})，暂停交易"
    return True, ""
