# -*- coding: utf-8 -*-
"""
双层风控管理器
- 硬限制：回测前参数合法性校验，违反则直接阻断
- 软目标：回测后全局风险评估，不达标则策略被标记为 discard
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import get_settings, RiskConfig


class RiskManager:
    """双层风控管理器"""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or get_settings().risk

    # ==================== 硬限制检查 ====================
    @staticmethod
    def check_hard_limits(params: dict, regime: str = "neutral") -> bool:
        """回测前硬限制检查
        检查项：
        1. 止损不能太宽（<-15%）
        2. 买入阈值不能太低（<0.4）
        3. 买入阈值必须大于卖出阈值
        4. 持仓天数合理（1-60天）
        5. 移动止损参数逻辑合理
        Raises:
            ValueError: 参数不合法时抛出
        """
        regime_params = params.get(regime, params.get("neutral", params))

        stop_loss = regime_params.get("stop_loss", -0.08)
        if stop_loss < -0.15:
            raise ValueError(f"止损过宽: {stop_loss:.2%} < -15%")

        buy_threshold = regime_params.get("buy_threshold", 0.6)
        if buy_threshold < 0.4:
            raise ValueError(f"买入阈值过低: {buy_threshold:.2f} < 0.4")

        sell_threshold = regime_params.get("sell_threshold", -0.2)
        if buy_threshold <= sell_threshold:
            raise ValueError(
                f"买入阈值({buy_threshold:.2f})必须大于卖出阈值({sell_threshold:.2f})"
            )

        hold_days = regime_params.get("hold_days", 15)
        if not (1 <= hold_days <= 60):
            raise ValueError(f"持仓天数不合理: {hold_days}，应在[1, 60]范围内")

        # 移动止损逻辑检查
        tp1 = regime_params.get("trailing_profit_level1", 0.06)
        tp2 = regime_params.get("trailing_profit_level2", 0.12)
        td1 = regime_params.get("trailing_drawdown_level1", 0.08)
        td2 = regime_params.get("trailing_drawdown_level2", 0.04)

        if tp1 <= 0 or tp2 <= 0:
            raise ValueError(f"移动止损触发利润必须为正: level1={tp1}, level2={tp2}")

        if td1 <= 0 or td2 <= 0:
            raise ValueError(f"移动止损回撤幅度必须为正: level1={td1}, level2={td2}")

        if tp2 <= tp1:
            raise ValueError(f"Level2触发利润({tp2})应大于Level1({tp1})")

        return True

    # ==================== 软目标评估 ====================
    def evaluate_soft_targets(self, stats: dict) -> dict:
        """回测后软目标评估
        Returns:
            dict: {
                'passed': bool,
                'violations': List[str],
                'discard': bool,
                'details': {
                    'max_drawdown_pct': float,
                    'max_drawdown_limit': float,
                    'max_drawdown_ok': bool,
                    'profit_factor': float,
                    'min_profit_factor': float,
                    'profit_factor_ok': bool,
                    'sharpe_ratio': float,
                    'min_sharpe_ratio': float,
                    'sharpe_ok': bool,
                    'win_rate': float,
                    'min_win_rate': float,
                    'win_rate_ok': bool,
                    'total_trades': int,
                    'min_trades': int,
                    'max_trades': int,
                    'trades_ok': bool,
                }
            }
        """
        violations = []
        details = {}

        # ---- 最大回撤（核心指标，不通过直接丢弃） ----
        max_dd = stats.get("max_drawdown", 0)
        dd_limit = self.config.max_drawdown_limit
        dd_ok = max_dd >= dd_limit
        details["max_drawdown_pct"] = max_dd
        details["max_drawdown_limit"] = dd_limit
        details["max_drawdown_ok"] = dd_ok
        if not dd_ok:
            violations.append(f"max_drawdown({max_dd:.2f}%) < {dd_limit:.2f}%")

        # ---- 利润因子（核心指标，不通过直接丢弃） ----
        pf = stats.get("profit_factor", 0)
        pf_min = self.config.min_profit_factor
        pf_ok = pf >= pf_min
        details["profit_factor"] = pf
        details["min_profit_factor"] = pf_min
        details["profit_factor_ok"] = pf_ok
        if not pf_ok:
            violations.append(f"profit_factor({pf:.2f}) < {pf_min:.2f}")

        # ---- 夏普比率（软指标） ----
        sharpe = stats.get("sharpe_ratio", 0)
        sharpe_min = self.config.min_sharpe_ratio
        sharpe_ok = sharpe >= sharpe_min
        details["sharpe_ratio"] = sharpe
        details["min_sharpe_ratio"] = sharpe_min
        details["sharpe_ok"] = sharpe_ok
        if not sharpe_ok:
            violations.append(f"sharpe_ratio({sharpe:.2f}) < {sharpe_min:.2f}")

        # ---- 胜率（软指标） ----
        wr = stats.get("win_rate", 0)
        wr_min = self.config.min_win_rate
        wr_ok = wr >= wr_min
        details["win_rate"] = wr
        details["min_win_rate"] = wr_min
        details["win_rate_ok"] = wr_ok
        if not wr_ok:
            violations.append(f"win_rate({wr:.1f}%) < {wr_min:.1f}%")

        # ---- 交易次数（软指标） ----
        nt = stats.get("total_trades", 0)
        nt_ok = self.config.min_trades <= nt <= self.config.max_trades
        details["total_trades"] = nt
        details["min_trades"] = self.config.min_trades
        details["max_trades"] = self.config.max_trades
        details["trades_ok"] = nt_ok
        if nt < self.config.min_trades:
            violations.append(f"total_trades({nt}) < {self.config.min_trades}")
        if nt > self.config.max_trades:
            violations.append(f"total_trades({nt}) > {self.config.max_trades} (过度交易)")

        passed = len(violations) == 0

        # 核心指标不通过 → discard
        critical_failed = not dd_ok or not pf_ok

        discard = critical_failed

        return {
            "passed": passed,
            "violations": violations,
            "discard": discard,
            "details": details,
        }

    # ==================== 组合层面风控 ====================
    @staticmethod
    def check_portfolio_risk(
        current_positions: List[dict],
        new_candidates: List[dict],
        max_total_ratio: float = 0.8,
        max_single_ratio: float = 0.3,
        max_same_sector_ratio: float = 0.4,
    ) -> Tuple[List[dict], List[str]]:
        """组合层面风控过滤
        Args:
            current_positions: 当前持仓列表 [{'code': str, 'ratio': float, 'sector': str}]
            new_candidates: 新买入候选列表 [{'code': str, 'ratio': float, 'sector': str, 'score': float}]
            max_total_ratio: 最大总仓位比例
            max_single_ratio: 单只最大仓位比例
            max_same_sector_ratio: 同板块最大仓位比例
        Returns:
            (filtered_candidates, warnings): 过滤后的候选和警告信息
        """
        warnings = []

        # 当前总仓位
        current_total = sum(p.get("ratio", 0) for p in current_positions)

        remaining_ratio = max_total_ratio - current_total
        if remaining_ratio <= 0:
            warnings.append(f"总仓位已满({current_total:.1%})，拒绝所有新买入")
            return [], warnings

        # 按得分排序
        sorted_candidates = sorted(new_candidates, key=lambda x: x.get("score", 0), reverse=True)
        filtered = []
        used_ratio = 0
        sector_exposure = {}

        # 统计当前板块暴露
        for p in current_positions:
            sector = p.get("sector", "unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + p.get("ratio", 0)

        for cand in sorted_candidates:
            cand_ratio = cand.get("ratio", 0)

            # 单只仓位检查
            if cand_ratio > max_single_ratio:
                warnings.append(f"{cand['code']} 仓位({cand_ratio:.1%})超限，降至{max_single_ratio:.1%}")
                cand_ratio = max_single_ratio

            # 总仓位检查
            if used_ratio + cand_ratio > remaining_ratio:
                available = remaining_ratio - used_ratio
                if available <= 0.01:
                    warnings.append(f"剩余仓位不足，跳过 {cand['code']}")
                    break

                warnings.append(f"{cand['code']} 仓位被截断至{available:.1%}")
                cand_ratio = available

            # 板块集中度检查
            sector = cand.get("sector", "unknown")
            sector_current = sector_exposure.get(sector, 0)
            if sector_current + cand_ratio > max_same_sector_ratio:
                allowed = max(0, max_same_sector_ratio - sector_current)
                if allowed <= 0.01:
                    warnings.append(f"{sector}板块已满，跳过 {cand['code']}")
                    continue
                warnings.append(f"{cand['code']} 板块({sector})仓位被截断至{allowed:.1%}")
                cand_ratio = allowed

            # 通过所有检查
            filtered.append({**cand, "ratio": cand_ratio})
            used_ratio += cand_ratio
            sector_exposure[sector] = sector_exposure.get(sector, 0) + cand_ratio

        return filtered, warnings
