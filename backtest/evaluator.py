# -*- coding: utf-8 -*-
"""
统一评估指标模块 V2
修复：
B5: 清理重复 import 和函数定义
M2: Sharpe 计算使用实际交易频率而非固定 252
M3: equity_final 用复合收益
M4: ann_return 基于实际时间跨度年化
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def calculate_comprehensive_stats(
    trades_df: pd.DataFrame,
    equity_curve: Optional[pd.Series] = None,
    benchmark_curve: Optional[pd.Series] = None,
    initial_cash: float = 100_000.0,
    commissions: float = 0.0,
) -> Dict[str, float]:
    """
    完整回测统计

    Args:
        trades_df: 必须有 net_return 列；如有 buy_date/sell_date 会自动算交易时长
        equity_curve: 可选，资金曲线
        benchmark_curve: 可选，基准净值
        initial_cash: 初始资金
        commissions: 总佣金
    """
    if trades_df is None or len(trades_df) == 0:
        return {}

    net_returns = trades_df["net_return"]
    n_trades = len(net_returns)

    # ---- 基础收益 ----
    total_return = ((1 + net_returns).prod() - 1) * 100
    win_rate = (net_returns > 0).mean() * 100
    avg_return = net_returns.mean() * 100
    best_trade = net_returns.max() * 100
    worst_trade = net_returns.min() * 100
    # ★ M3 修复：用复合收益
    equity_final = initial_cash * (1 + net_returns).prod()
    equity_peak = initial_cash * (1 + net_returns.cumsum()).max()

    # ---- 交易时长 ----
    durations = {}
    try:
        dur = (trades_df["sell_date"] - trades_df["buy_date"]).dt.total_seconds() / 86400
        if not dur.empty:
            durations = {
                "max_trade_duration_days": dur.max(),
                "avg_trade_duration_days": dur.mean(),
            }
    except Exception:
        pass

    # ---- 利润因子 ----
    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = abs(net_returns[net_returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12)

    # ---- 期望值 ----
    avg_win = net_returns[net_returns > 0].mean() if (net_returns > 0).any() else 0
    avg_loss = abs(net_returns[net_returns < 0].mean()) if (net_returns < 0).any() else 1e-12
    expectancy_pw = win_rate / 100 * avg_win * 100 - (1 - win_rate / 100) * avg_loss * 100

    # ---- 最大回撤 ----
    def _drawdown_stats(curve: pd.Series):
        cum = (1 + curve).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min() * 100
        avg_dd = dd[dd < 0].mean() * 100 if (dd < 0).any() else 0.0
        in_drawdown = dd < 0
        if not in_drawdown.any():
            return max_dd, avg_dd, None, None
        groups = (~in_drawdown).cumsum()
        dd_durations = in_drawdown.groupby(groups).sum()
        max_dd_dur = dd_durations.max() if not dd_durations.empty else 0
        avg_dd_dur = dd_durations.mean() if not dd_durations.empty else 0
        return max_dd, avg_dd, max_dd_dur, avg_dd_dur

    max_drawdown, avg_drawdown, max_dd_duration, avg_dd_duration = _drawdown_stats(net_returns)

    # ★ M4 修复：基于实际时间跨度年化
    total_days = 252  # 默认1年
    if 'buy_date' in trades_df.columns and 'sell_date' in trades_df.columns:
        try:
            first_date = trades_df['buy_date'].min()
            last_date = trades_df['sell_date'].max()
            total_days = max((last_date - first_date).days, 1)
        except Exception:
            pass
    years = total_days / 252.0
    ann_factor = 252

    if years > 0 and total_return > -100:
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    else:
        ann_return = 0.0

    ann_vol = net_returns.std() * np.sqrt(ann_factor) * 100

    # ★ M2 修复：Sharpe 用实际交易频率
    avg_holding_days = 15  # 默认
    if durations and durations.get('avg_trade_duration_days'):
        avg_holding_days = max(durations['avg_trade_duration_days'], 1)
    trades_per_year = 252 / avg_holding_days
    sharpe = (net_returns.mean() / (net_returns.std() + 1e-12)) * np.sqrt(trades_per_year)

    downside = net_returns[net_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-12
    sortino = (net_returns.mean() / (downside_std + 1e-12)) * np.sqrt(trades_per_year)
    calmar = ann_return / (abs(max_drawdown) + 1e-12)

    cagr = ann_return

    # Alpha / Beta
    alpha = None
    beta = None
    if benchmark_curve is not None and len(benchmark_curve) == len(net_returns):
        bench = benchmark_curve.pct_change().dropna().values
        strat = net_returns.values
        covm = np.cov(strat, bench)
        if covm.shape == (2, 2):
            beta = float(covm[0, 1] / (covm[1, 1] + 1e-12))
            alpha = (np.mean(strat) - beta * np.mean(bench)) * ann_factor * 100

    # SQN
    sqn = (net_returns.mean() / (net_returns.std() + 1e-12)) * np.sqrt(n_trades)

    # Kelly
    p = win_rate / 100
    b = avg_win / (avg_loss + 1e-12)
    kelly = p - (1 - p) / (b + 1e-12) if b > 0 else 0.0

    # Exposure Time
    exposure_time = None
    if equity_curve is not None:
        exposure = (equity_curve != 0).mean() * 100 if equity_curve.dtype in [np.float64, np.int64] else None
        exposure_time = exposure

    stats: Dict[str, float] = {
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 4),
        "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4),
        "equity_final": round(equity_final, 2),
        "equity_peak": round(equity_peak, 2),
        "commissions": round(commissions, 2),
        "total_return": round(total_return, 4),
        "ann_return": round(ann_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_drawdown, 4),
        "avg_drawdown": round(avg_drawdown, 4),
        "max_drawdown_duration": round(max_dd_duration, 2) if max_dd_duration is not None else None,
        "avg_drawdown_duration": round(avg_dd_duration, 2) if avg_dd_duration is not None else None,
        "ann_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "alpha": round(alpha, 4) if alpha is not None else None,
        "beta": round(beta, 4) if beta is not None else None,
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy_pw, 4),
        "sqn": round(sqn, 4),
        "kelly_criterion": round(kelly, 4),
        "exposure_time": round(exposure_time, 4) if exposure_time is not None else None,
    }
    if durations:
        stats.update({k: round(v, 2) for k, v in durations.items()})
    return stats


def evaluate_strategy(backtest_results: Dict) -> Dict:
    """评价策略绩效"""
    return backtest_results


def check_risk_limits(performance: Dict, risk_limits: Dict) -> Tuple[bool, List[str]]:
    """检查绩效是否通过风控阈值"""
    passed = True
    reasons = []

    if performance.get('max_drawdown', 0) < risk_limits.get('max_drawdown', -1.0):
        passed = False
        reasons.append(f"最大回撤 {performance['max_drawdown']:.2%} 超过限制")

    if performance.get('sharpe_ratio', 0) < risk_limits.get('min_sharpe', -999):
        passed = False
        reasons.append(f"夏普比率 {performance['sharpe_ratio']:.2f} 低于限制")

    if performance.get('total_trades', 0) < risk_limits.get('min_trades', 0):
        passed = False
        reasons.append(f"交易次数不足")

    if performance.get('win_rate', 0) < risk_limits.get('min_win_rate', 0.0):
        passed = False
        reasons.append(f"胜率不足")

    if performance.get('profit_factor', 0) < risk_limits.get('min_profit_factor', 0.0):
        passed = False
        reasons.append(f"利润因子不足")

    return passed, reasons
