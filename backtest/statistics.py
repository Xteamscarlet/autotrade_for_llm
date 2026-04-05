import pandas as pd
import numpy as np
from typing import Tuple

class StrategyStatistics:
    """
    基于资金曲线的绩效统计计算类。
    所有指标都基于账户的真实权益曲线计算，确保逻辑合理。
    """
    def __init__(self, equity_curve: pd.Series, risk_free_rate: float = 0.0):
        """
        :param equity_curve: pandas Series，每日或每笔交易后的总权益，索引为时间。
        :param risk_free_rate: 无风险利率，用于计算夏普比率等。
        """
        self.equity_curve = equity_curve
        self.risk_free_rate = risk_free_rate
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> pd.Series:
        """计算每日或每期的收益率序列。"""
        # 简单收益率: (P_t - P_{t-1}) / P_{t-1}
        returns = self.equity_curve.pct_change().dropna()
        return returns

    def get_drawdown_stats(self) -> Tuple[float, float, float]:
        """
        计算最大回撤、平均回撤、最长回撤周期。
        返回: (max_drawdown, avg_drawdown, max_drawdown_duration)
        """
        # 计算资金曲线的历史最高点
        rolling_max = self.equity_curve.expanding().max()
        # 计算回撤序列
        drawdowns = (self.equity_curve - rolling_max) / rolling_max

        max_dd = drawdowns.min()
        avg_dd = drawdowns.mean()
        # 最长回撤周期：资金曲线创新高所需的时间步长（简化版）
        # 此处我们返回回撤持续的最长时间步数
        is_drawdown = drawdowns < 0
        # 找出回撤期的开始和结束
        drawdown_periods = (is_drawdown != is_drawdown.shift()).cumsum()
        # 计算每个回撤期的长度
        drawdown_lengths = is_drawdown.groupby(drawdown_periods).sum()
        max_dd_duration = drawdown_lengths.max()

        return max_dd, avg_dd, max_dd_duration

    def get_total_return(self) -> float:
        """计算总收益率： (最终权益 - 初始权益) / 初始权益"""
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1

    def get_annualized_return(self, periods_per_year: int = 252) -> float:
        """
        计算年化收益率。
        :param periods_per_year: 每年的交易日或周期数，默认为252（交易日）。
        """
        total_periods = len(self.equity_curve) - 1  # 减去初始值
        if total_periods <= 0:
            return 0.0
        total_return = self.get_total_return()
        # 年化公式: (1 + R)^(n/T) - 1, 其中T为总周期数，n为一年内的周期数
        ann_return = (1 + total_return) ** (periods_per_year / total_periods) - 1
        return ann_return

    def get_annualized_volatility(self, periods_per_year: int = 252) -> float:
        """计算年化波动率。"""
        if len(self.returns) < 2:
            return 0.0
        vol = self.returns.std() * np.sqrt(periods_per_year)
        return vol

    def get_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """计算夏普比率。"""
        ann_return = self.get_annualized_return(periods_per_year)
        ann_vol = self.get_annualized_volatility(periods_per_year)
        if ann_vol == 0:
            return 0.0
        return (ann_return - self.risk_free_rate) / ann_vol

    def get_sortino_ratio(self, periods_per_year: int = 252, target_return: float = 0.0) -> float:
        """计算索提诺比率，只考虑下行波动。"""
        if len(self.returns) < 2:
            return 0.0
        # 下行收益率
        downside_returns = self.returns[self.returns < target_return]
        if len(downside_returns) == 0:
            # 没有下行收益，说明所有收益都超过目标，比率可能无穷大，这里返回一个极大值或用波动率代替
            return float('inf') if self.get_annualized_return() > target_return else 0.0

        # 下行标准差
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        ann_return = self.get_annualized_return(periods_per_year)
        return (ann_return - target_return) / downside_std

    def get_calmar_ratio(self, periods_per_year: int = 252) -> float:
        """计算卡玛比率：年化收益 / 最大回撤的绝对值。"""
        ann_return = self.get_annualized_return(periods_per_year)
        max_dd, _, _ = self.get_drawdown_stats()
        if max_dd == 0:
            return float('inf') if ann_return > 0 else 0.0
        return ann_return / abs(max_dd)

    def get_profit_factor(self) -> float:
        """
        计算利润因子：总盈利 / 总亏损的绝对值。
        注意：这需要交易列表，不是纯从equity_curve可以算出。此函数为占位，需在评价器中结合交易记录计算。
        """
        raise NotImplementedError("利润因子需要交易列表，请使用StrategyEvaluator。")

    def get_win_rate(self) -> float:
        """
        计算胜率。同样需要交易列表。
        """
        raise NotImplementedError("胜率需要交易列表，请使用StrategyEvaluator。")
