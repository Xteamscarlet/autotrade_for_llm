# -*- coding: utf-8 -*-
"""
回测报告打印模块
输出格式对标 backtesting.py Stats + autotrade 风控结论行
"""
import pandas as pd
from typing import Dict, Optional


def _fmt(v, fmt_str="{:.4f}", none_str="N/A"):
    """格式化数值，None 显示为 N/A"""
    if v is None:
        return none_str
    return fmt_str.format(v)


def _fmt_duration_days(v):
    """格式化持续天数"""
    if v is None:
        return "N/A"
    return f"{v:.0f} days 00:00:00"


def print_stock_backtest_report(
    stock_name: str,
    stock_code: str,
    start_date,
    end_date,
    elapsed_seconds: float,
    stats: Dict,
    risk_result: Dict,
):
    """
    打印单股回测报告 + 风控结论行
    """
    def _fmt_date(d):
        try:
            return pd.Timestamp(d).strftime("%Y-%m-%d %H:%M:%S+00:00")
        except Exception:
            return str(d)

    start_str = _fmt_date(start_date)
    end_str = _fmt_date(end_date)

    # Duration
    duration_str = "N/A"
    try:
        dur = pd.Timestamp(end_date) - pd.Timestamp(start_date)
        duration_str = f"{dur.days} days 00:00:00"
    except Exception:
        pass

    # ==================== 打印报告 ====================
    print()
    print(f"strategy:         compound_signal_{stock_code}")
    print(f"symbol:           {stock_code}")
    print(f"timeframe:        daily")
    print(f"max_leverage:     1.0")
    print(f"elapsed_seconds:  {elapsed_seconds:.1f}")
    print(f"Start                     {start_str}")
    print(f"End                       {end_str}")
    print(f"Duration                           {duration_str}")
    print(f"Exposure Time [%]                         {_fmt(stats.get('exposure_time'))}")
    print(f"Equity Final [$]                       {_fmt(stats.get('equity_final'), '{:.2f}')}")
    print(f"Equity Peak [$]                        {_fmt(stats.get('equity_peak'), '{:.2f}')}")
    print(f"Commissions [$]                          {_fmt(stats.get('commissions'), '{:.2f}')}")
    print(f"Return [%]                                {_fmt(stats.get('total_return'))}")
    bh = stats.get('buy_and_hold_return')
    print(f"Buy & Hold Return [%]                      {_fmt(bh) if bh is not None else 'N/A'}")
    print(f"Return (Ann.) [%]                         {_fmt(stats.get('ann_return'))}")
    print(f"Volatility (Ann.) [%]                     {_fmt(stats.get('ann_volatility'))}")
    print(f"CAGR [%]                                 {_fmt(stats.get('cagr'))}")
    print(f"Sharpe Ratio                               {_fmt(stats.get('sharpe_ratio'))}")
    print(f"Sortino Ratio                              {_fmt(stats.get('sortino_ratio'))}")
    print(f"Calmar Ratio                              {_fmt(stats.get('calmar_ratio'))}")
    print(f"Alpha [%]                                 {_fmt(stats.get('alpha'))}")
    print(f"Beta                                      {_fmt(stats.get('beta'))}")
    print(f"Max. Drawdown [%]                         {_fmt(stats.get('max_drawdown'))}")
    print(f"Avg. Drawdown [%]                         {_fmt(stats.get('avg_drawdown'))}")
    print(f"Max. Drawdown Duration             {_fmt_duration_days(stats.get('max_drawdown_duration'))}")
    print(f"Avg. Drawdown Duration              {_fmt_duration_days(stats.get('avg_drawdown_duration'))}")
    print(f"# Trades                                          {stats.get('total_trades', 'N/A')}")
    print(f"Win Rate [%]                              {_fmt(stats.get('win_rate'))}")
    print(f"Best Trade [%]                            {_fmt(stats.get('best_trade'))}")
    print(f"Worst Trade [%]                           {_fmt(stats.get('worst_trade'))}")
    print(f"Avg. Trade [%]                             {_fmt(stats.get('avg_return'))}")
    print(f"Max. Trade Duration                {_fmt_duration_days(stats.get('max_trade_duration_days'))}")
    print(f"Avg. Trade Duration                 {_fmt_duration_days(stats.get('avg_trade_duration_days'))}")
    print(f"Profit Factor                              {_fmt(stats.get('profit_factor'))}")
    print(f"Expectancy [%]                             {_fmt(stats.get('expectancy'))}")
    print(f"SQN                                        {_fmt(stats.get('sqn'))}")
    print(f"Kelly Criterion                            {_fmt(stats.get('kelly_criterion'))}")

    # ==================== 风控结论行 ====================
    details = risk_result.get("details", {})
    passed = risk_result.get("passed", False)
    violations = risk_result.get("violations", [])

    def _mark(ok):
        return "✓" if ok else "✗"

    dd_val = details.get("max_drawdown_pct")
    dd_limit = details.get("max_drawdown_limit", -20)
    dd_ok = details.get("max_drawdown_ok", False)

    pf_val = details.get("profit_factor")
    pf_min = details.get("min_profit_factor", 1.5)
    pf_ok = details.get("profit_factor_ok", False)

    dd_str = f"{dd_val:.2f}% {_mark(dd_ok)}" if dd_val is not None else "N/A"
    pf_str = f"{pf_val:.2f} {_mark(pf_ok)}" if pf_val is not None else "N/A"

    risk_passed_str = "True" if passed else "False"
    print(f"\nrisk_passed: {risk_passed_str} | max_drawdown: {dd_str} | profit_factor: {pf_str}")

    if not passed:
        for v in violations:
            print(f"[RISK] ✗ {v}")
        if risk_result.get("discard"):
            print("[RISK] Strategy status = DISCARD (core limits failed)")
    else:
        print("[RISK] ✓ All soft targets met. Strategy status = KEEP")

    print()
