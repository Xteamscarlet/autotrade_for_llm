import pandas as pd
from typing import Optional, Dict, Tuple

class CashAccount:
    """
    一个简单的现金账户，用于严格追踪资金状况。
    确保资金不会变为负数，并记录所有交易。
    """
    def __init__(self, initial_capital: float):
        if initial_capital <= 0:
            raise ValueError("初始资金必须为正数。")
        self.initial_capital = initial_capital
        self.cash = initial_capital  # 当前可用现金
        self.position_value = 0.0    # 当前持仓市值
        self.equity_curve = [initial_capital]  # 资金曲线列表，初始点
        self.transaction_history = []  # 交易历史记录

    @property
    def total_equity(self) -> float:
        """当前总权益 = 现金 + 持仓市值"""
        return self.cash + self.position_value

    def check_affordability(self, cost: float) -> bool:
        """检查现金是否足以支付成本"""
        return self.cash >= cost

    def buy(self, symbol: str, price: float, shares: int, commission: float, timestamp: pd.Timestamp) -> bool:
        """
        执行买入操作。
        返回: True表示成功，False表示现金不足。
        """
        cost = price * shares + commission
        if not self.check_affordability(cost):
            # print(f"[账户警告] 现金不足，无法买入 {symbol}。需要: {cost:.2f}, 现有: {self.cash:.2f}")
            return False

        # 更新现金和持仓市值
        self.cash -= cost
        self.position_value += price * shares  # 简单假设市价买入，市值立即等于成本价

        # 记录交易
        self.transaction_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'commission': commission,
            'cash_after': self.cash
        })
        # 更新资金曲线
        self.equity_curve.append(self.total_equity)
        return True

    def sell(self, symbol: str, price: float, shares: int, commission: float, timestamp: pd.Timestamp):
        """执行卖出操作。"""
        proceeds = price * shares - commission
        self.cash += proceeds
        self.position_value -= price * shares  # 持仓市值减少

        # 确保持仓市值不会为负（理论上不应发生，此处为防御性编程）
        self.position_value = max(0.0, self.position_value)

        # 记录交易
        self.transaction_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'commission': commission,
            'cash_after': self.cash
        })
        # 更新资金曲线
        self.equity_curve.append(self.total_equity)
        return True

    def update_position_value(self, new_value: float):
        """在每日收盘后更新持仓的市值（用于绘制曲线，不影响现金）"""
        self.position_value = new_value
        # 此时也更新一次资金曲线，反映每日市值波动
        self.equity_curve.append(self.total_equity)

    def get_equity_curve_series(self) -> pd.Series:
        """获取pandas Series格式的资金曲线，索引为交易时间点"""
        # 注意：这里的时间点序列需要和交易记录对齐，或由外部传入。
        # 为简化，我们返回一个不带时间索引的Series，后续在统计时使用。
        return pd.Series(self.equity_curve, name='equity')
