# -*- coding: utf-8 -*-
"""
统一评估指标模块（增强版）
计算 9 个核心指标：收益率、胜率、交易次数、平均收益率、
最大回撤、夏普比率、利润因子、Sortino比率、Calmar比率
新增：边界检查、NaN处理、默认值返回
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# 默认值常量
DEFAULT_SHARPE = 0.0
DEFAULT_SORTINO = 0.0
DEFAULT_CALMAR = 0.0
DEFAULT_MAX_DRAWDOWN = 0.0
DEFAULT_WIN_RATE = 0.0
DEFAULT_PROFIT_FACTOR = 0.0


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    log_warning: bool = True,
) -> float:
    """安全除法
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值
        log_warning: 是否记录警告
        
    Returns:
        除法结果或默认值
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        if log_warning:
            logger.debug(f"除法异常: {numerator} / {denominator}, 返回默认值 {default}")
        return default
    
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    
    return result


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    default: float = DEFAULT_SHARPE,
) -> float:
    """计算夏普比率 - 增强版
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年交易周期数
        default: 默认值
        
    Returns:
        夏普比率
    """
    if returns is None or len(returns) == 0:
        logger.warning("收益序列为空，返回默认夏普比率")
        return default
    
    # 清理数据
    returns = returns.dropna()
    if len(returns) == 0:
        return default
    
    # 计算超额收益
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # 计算均值和标准差
    mean_excess = excess_returns.mean()
    std_returns = excess_returns.std()
    
    if pd.isna(mean_excess) or pd.isna(std_returns):
        return default
    
    if std_returns == 0:
        # 标准差为0，说明收益完全稳定
        if mean_excess > 0:
            return 10.0  # 返回一个较大的正值
        else:
            return default
    
    # 年化
    sharpe = mean_excess / std_returns * np.sqrt(periods_per_year)
    
    if np.isnan(sharpe) or np.isinf(sharpe):
        return default
    
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    default: float = DEFAULT_SORTINO,
) -> float:
    """计算Sortino比率 - 增强版
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年交易周期数
        default: 默认值
        
    Returns:
        Sortino比率
    """
    if returns is None or len(returns) == 0:
        return default
    
    returns = returns.dropna()
    if len(returns) == 0:
        return default
    
    # 计算超额收益
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # 计算下行标准差（只考虑负收益）
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        # 没有负收益，返回一个较大的正值
        return 10.0
    
    downside_std = negative_returns.std()
    
    if pd.isna(downside_std) or downside_std == 0:
        return default
    
    mean_excess = excess_returns.mean()
    if pd.isna(mean_excess):
        return default
    
    sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
    
    if np.isnan(sortino) or np.isinf(sortino):
        return default
    
    return sortino


def calculate_max_drawdown(
    equity_curve: pd.Series,
    default: float = DEFAULT_MAX_DRAWDOWN,
) -> Tuple[float, int, int]:
    """计算最大回撤 - 增强版
    
    Args:
        equity_curve: 资金曲线
        default: 默认值
        
    Returns:
        (最大回撤, 开始位置, 结束位置)
    """
    if equity_curve is None or len(equity_curve) == 0:
        return default, 0, 0
    
    equity_curve = equity_curve.dropna()
    if len(equity_curve) == 0:
        return default, 0, 0
    
    # 计算累计最大值
    rolling_max = equity_curve.cummax()
    
    # 计算回撤
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # 找到最大回撤
    max_dd = drawdown.min()
    
    if pd.isna(max_dd):
        return default, 0, 0
    
    # 找到最大回撤的位置
    max_dd_end = drawdown.idxmin()
    max_dd_start = equity_curve[:max_dd_end].idxmax()
    
    return max_dd, equity_curve.index.get_loc(max_dd_start), equity_curve.index.get_loc(max_dd_end)


def calculate_win_rate(
    trades_df: pd.DataFrame,
    default: float = DEFAULT_WIN_RATE,
) -> float:
    """计算胜率 - 增强版
    
    Args:
        trades_df: 交易记录DataFrame
        default: 默认值
        
    Returns:
        胜率（百分比）
    """
    if trades_df is None or len(trades_df) == 0:
        return default
    
    if 'net_return' not in trades_df.columns:
        return default
    
    net_returns = trades_df['net_return'].dropna()
    if len(net_returns) == 0:
        return default
    
    win_count = (net_returns > 0).sum()
    total_count = len(net_returns)
    
    if total_count == 0:
        return default
    
    return win_count / total_count * 100


def calculate_profit_factor(
    trades_df: pd.DataFrame,
    default: float = DEFAULT_PROFIT_FACTOR,
) -> float:
    """计算利润因子 - 增强版
    
    Args:
        trades_df: 交易记录DataFrame
        default: 默认值
        
    Returns:
        利润因子
    """
    if trades_df is None or len(trades_df) == 0:
        return default
    
    if 'net_return' not in trades_df.columns:
        return default
    
    net_returns = trades_df['net_return'].dropna()
    if len(net_returns) == 0:
        return default
    
    # 计算总盈利和总亏损
    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = abs(net_returns[net_returns < 0].sum())
    
    if gross_loss == 0:
        if gross_profit > 0:
            return 10.0  # 只有盈利，返回较大值
        else:
            return default
    
    profit_factor = gross_profit / gross_loss
    
    if np.isnan(profit_factor) or np.isinf(profit_factor):
        return default
    
    return profit_factor


def calculate_comprehensive_stats(
    trades_df: pd.DataFrame,
    equity_curve: Optional[pd.Series] = None,
    benchmark_curve: Optional[pd.Series] = None,
    initial_cash: float = 100_000.0,
    commissions: float = 0.0,
) -> Dict[str, float]:
    """
    更完整的回测统计（尽量覆盖常见报告字段）- 增强版
    
    新增功能：
    1. 边界检查
    2. NaN处理
    3. 默认值返回
    
    Args:
        - trades_df: 必须有 net_return 列；如有 buy_date/sell_date 会自动算交易时长
        - equity_curve: 可选，资金曲线（含初始资金），用于计算年化波动、回撤持续期等
        - benchmark_curve: 可选，基准净值，用于计算 Alpha/Beta
        - initial_cash: 初始资金
        - commissions: 总佣金
    """
    if trades_df is None or len(trades_df) == 0:
        logger.warning("交易记录为空，返回空统计")
        return {
            "total_trades": 0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": DEFAULT_SHARPE,
            "max_drawdown": DEFAULT_MAX_DRAWDOWN,
        }

    net_returns = trades_df["net_return"]
    n_trades = len(net_returns)

    # ---- 基础收益 ----
    total_return = ((1 + net_returns).prod() - 1) * 100
    if pd.isna(total_return):
        total_return = 0.0
    
    win_rate = calculate_win_rate(trades_df)
    avg_return = net_returns.mean() * 100 if not net_returns.empty else 0.0
    best_trade = net_returns.max() * 100 if not net_returns.empty else 0.0
    worst_trade = net_returns.min() * 100 if not net_returns.empty else 0.0
    
    if pd.isna(avg_return):
        avg_return = 0.0
    if pd.isna(best_trade):
        best_trade = 0.0
    if pd.isna(worst_trade):
        worst_trade = 0.0

    # ---- 交易持续期 ----
    durations = {}
    if "buy_date" in trades_df.columns and "sell_date" in trades_df.columns:
        try:
            trade_durations = []
            for _, row in trades_df.iterrows():
                if pd.notna(row["buy_date"]) and pd.notna(row["sell_date"]):
                    duration = (row["sell_date"] - row["buy_date"]).days
                    if not pd.isna(duration):
                        trade_durations.append(duration)
            
            if trade_durations:
                durations = {
                    "avg_hold_days": np.mean(trade_durations),
                    "max_hold_days": max(trade_durations),
                    "min_hold_days": min(trade_durations),
                }
        except Exception as e:
            logger.warning(f"计算交易持续期失败: {e}")

    # ---- 年化收益 ----
    ann_return = 0.0
    cagr = 0.0
    if equity_curve is not None and len(equity_curve) > 1:
        try:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0:
                ann_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
                cagr = ann_return
        except Exception:
            pass

    # ---- 最大回撤 ----
    max_drawdown = DEFAULT_MAX_DRAWDOWN
    avg_drawdown = 0.0
    max_dd_duration = None
    avg_dd_duration = None
    
    if equity_curve is not None and len(equity_curve) > 1:
        try:
            max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
            max_drawdown = max_drawdown * 100  # 转为百分比
            
            # 计算平均回撤
            rolling_max = equity_curve.cummax()
            drawdowns = (equity_curve - rolling_max) / rolling_max * 100
            avg_drawdown = drawdowns.mean()
            
            if pd.isna(avg_drawdown):
                avg_drawdown = 0.0
        except Exception as e:
            logger.warning(f"计算最大回撤失败: {e}")

    # ---- 波动率 ----
    ann_vol = 0.0
    if equity_curve is not None and len(equity_curve) > 1:
        try:
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0:
                ann_vol = returns.std() * np.sqrt(252) * 100
                if pd.isna(ann_vol):
                    ann_vol = 0.0
        except Exception:
            pass

    # ---- 风险调整指标 ----
    sharpe = DEFAULT_SHARPE
    sortino = DEFAULT_SORTINO
    calmar = DEFAULT_CALMAR
    
    if equity_curve is not None and len(equity_curve) > 1:
        try:
            returns = equity_curve.pct_change().dropna()
            sharpe = calculate_sharpe_ratio(returns)
            sortino = calculate_sortino_ratio(returns)
            
            if max_drawdown != 0:
                calmar = safe_divide(ann_return, abs(max_drawdown), DEFAULT_CALMAR)
        except Exception as e:
            logger.warning(f"计算风险调整指标失败: {e}")

    # ---- Alpha/Beta ----
    alpha = None
    beta = None
    if equity_curve is not None and benchmark_curve is not None:
        try:
            aligned = pd.DataFrame({
                "strategy": equity_curve.pct_change(),
                "benchmark": benchmark_curve.pct_change()
            }).dropna()
            
            if len(aligned) > 10:
                cov = aligned.cov()
                var_bench = aligned["benchmark"].var()
                
                if var_bench > 0:
                    beta = cov.loc["strategy", "benchmark"] / var_bench
                    alpha = aligned["strategy"].mean() - beta * aligned["benchmark"].mean()
                    alpha = alpha * 252 * 100  # 年化
                    
                    if pd.isna(alpha):
                        alpha = None
                    if pd.isna(beta):
                        beta = None
        except Exception as e:
            logger.warning(f"计算Alpha/Beta失败: {e}")

    # ---- 利润因子 ----
    profit_factor = calculate_profit_factor(trades_df)

    # ---- 期望值 ----
    expectancy_pw = 0.0
    if n_trades > 0:
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            win_prob = len(wins) / n_trades
            loss_prob = len(losses) / n_trades
            
            expectancy_pw = (avg_win * win_prob - avg_loss * loss_prob) * 100
            
            if pd.isna(expectancy_pw):
                expectancy_pw = 0.0

    # ---- SQN (System Quality Number) ----
    sqn = 0.0
    if n_trades >= 2:
        try:
            mean_ret = net_returns.mean()
            std_ret = net_returns.std()
            
            if std_ret > 0 and not pd.isna(mean_ret) and not pd.isna(std_ret):
                sqn = np.sqrt(n_trades) * mean_ret / std_ret
                
                if pd.isna(sqn) or np.isinf(sqn):
                    sqn = 0.0
        except Exception:
            pass

    # ---- Kelly Criterion ----
    kelly = 0.0
    if n_trades > 0:
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            win_prob = len(wins) / n_trades
            loss_prob = len(losses) / n_trades
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            
            if avg_loss > 0:
                kelly = win_prob - loss_prob / (avg_win / avg_loss)
                kelly = max(0, min(1, kelly))  # 限制在 [0, 1]
                
                if pd.isna(kelly):
                    kelly = 0.0

    # ---- 暴露时间 ----
    exposure_time = None
    if equity_curve is not None and len(equity_curve) > 1:
        try:
            # 计算持仓时间占比
            in_position = 0
            total_days = len(equity_curve)
            
            # 简化计算：假设有交易记录时计算持仓天数
            if durations:
                total_hold_days = sum(durations.values()) if isinstance(durations.get("avg_hold_days"), (int, float)) else 0
                exposure_time = min(100, total_hold_days / total_days * 100 * n_trades)
        except Exception:
            pass

    stats = {
        # 交易统计
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 4),
        "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4),
        "total_commission": round(commissions, 2),
        # 收益
        "total_return": round(total_return, 4),
        "ann_return": round(ann_return, 4),
        "cagr": round(cagr, 4),
        # 风险
        "max_drawdown": round(max_drawdown, 4),
        "avg_drawdown": round(avg_drawdown, 4),
        "max_drawdown_duration": round(max_dd_duration, 2) if max_dd_duration is not None else None,
        "avg_drawdown_duration": round(avg_dd_duration, 2) if avg_dd_duration is not None else None,
        "ann_volatility": round(ann_vol, 4),
        # 风险调整
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        # 资产定价
        "alpha": round(alpha, 4) if alpha is not None else None,
        "beta": round(beta, 4) if beta is not None else None,
        # 交易质量
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy_pw, 4),
        "sqn": round(sqn, 4),
        "kelly_criterion": round(kelly, 4),
        "exposure_time": round(exposure_time, 4) if exposure_time is not None else None,
    }
    
    # 交易持续期（可能为空）
    if durations:
        stats.update({k: round(v, 2) for k, v in durations.items()})
    
    return stats


def validate_strategy_metrics(
    stats: Dict[str, float],
    min_trades: int = 5,
    min_sharpe: float = 0.0,
    max_drawdown: float = -20.0,
) -> Tuple[bool, List[str]]:
    """验证策略指标是否满足最低要求
    
    Args:
        stats: 统计指标字典
        min_trades: 最小交易次数
        min_sharpe: 最小夏普比率
        max_drawdown: 最大回撤限制（负值）
        
    Returns:
        (是否通过, 问题列表)
    """
    issues = []
    
    if stats.get("total_trades", 0) < min_trades:
        issues.append(f"交易次数 {stats.get('total_trades', 0)} < {min_trades}")
    
    if stats.get("sharpe_ratio", 0) < min_sharpe:
        issues.append(f"夏普比率 {stats.get('sharpe_ratio', 0):.4f} < {min_sharpe}")
    
    if stats.get("max_drawdown", 0) < max_drawdown:
        issues.append(f"最大回撤 {stats.get('max_drawdown', 0):.2f}% < {max_drawdown}%")
    
    return len(issues) == 0, issues
