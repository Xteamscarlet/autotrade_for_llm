# -*- coding: utf-8 -*-
"""
复合信号策略
基于多因子加权得分 + Transformer置信度过滤的复合信号策略
从原始 backtest_market_v2.py 的交易逻辑迁移而来
"""
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from data.types import get_limit_ratio


class CompoundSignalStrategy(BaseStrategy):
    """复合信号策略

    信号生成逻辑：
    1. 综合得分超过买入阈值 -> 生成买入信号
    2. Transformer置信度过滤 -> 低置信度信号被拦截
    3. ATR动态仓位 -> 根据波动率调整仓位大小
    4. 移动止损/止盈/时间止损 -> 生成卖出信号
    """

    name = "compound_signal"
    description = "多因子加权 + Transformer置信度 + ATR动态仓位的复合策略"
    keep = True

    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
        regime: str = "neutral",
    ) -> Dict[str, Any]:
        """生成交易信号"""
        if idx < 60:
            return {'action': 'hold', 'score': 0.5, 'confidence': 0.0, 'position_ratio': 0.0, 'reason': '数据不足'}

        score = df['Combined_Score'].iloc[idx]
        price = df['Close'].iloc[idx]

        # Transformer 因子
        transformer_prob = df['transformer_prob'].iloc[idx] if 'transformer_prob' in df.columns else 0.5
        transformer_conf = df['transformer_conf'].iloc[idx] if 'transformer_conf' in df.columns else 0.0

        # ========== 买入信号 ==========
        buy_threshold = params.get('buy_threshold', 0.6)
        confidence_threshold = params.get('confidence_threshold', 0.5)
        transformer_buy_threshold = params.get('transformer_buy_threshold', 0.6)

        if score > buy_threshold:
            # 置信度过滤
            if transformer_conf < confidence_threshold:
                return {
                    'action': 'hold', 'score': score, 'confidence': transformer_conf,
                    'position_ratio': 0.0,
                    'reason': f'置信度不足({transformer_conf:.2f}<{confidence_threshold})',
                }

            # AI概率过滤
            if transformer_prob < transformer_buy_threshold:
                return {
                    'action': 'hold', 'score': score, 'confidence': transformer_conf,
                    'position_ratio': 0.0,
                    'reason': f'AI概率不足({transformer_prob:.2f}<{transformer_buy_threshold})',
                }

            # 弱势市场阻断
            if regime == 'weak':
                return {
                    'action': 'hold', 'score': score, 'confidence': transformer_conf,
                    'position_ratio': 0.0, 'reason': '弱势市场阻断',
                }

            # ATR动态仓位
            atr = df['atr'].iloc[idx] if 'atr' in df.columns else price * 0.02
            if pd.isna(atr) or atr <= 0:
                atr = price * 0.02
            daily_vol = atr / price
            if daily_vol <= 0 or not np.isfinite(daily_vol):
                daily_vol = 0.02

            target_annual_vol = 0.10
            position_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
            position_ratio = min(max(position_ratio, 0.1), 1.0)

            # 信号强度调整
            if 'transformer_pred_ret' in df.columns:
                pred_ret = df['transformer_pred_ret'].iloc[idx]
                signal_strength = max(0.5, min(1.5, 1 + pred_ret / 0.05))
                position_ratio *= signal_strength
                position_ratio = min(max(position_ratio, 0.1), 1.0)

            return {
                'action': 'buy', 'score': score, 'confidence': transformer_conf,
                'position_ratio': position_ratio, 'reason': '信号触发',
            }

        return {'action': 'hold', 'score': score, 'confidence': transformer_conf, 'position_ratio': 0.0, 'reason': '无信号'}

    def get_param_space(self) -> Dict[str, Dict]:
        """定义参数空间"""
        return {
            'buy_threshold': {'type': 'float', 'low': 0.5, 'high': 0.7, 'step': 0.05},
            'sell_threshold': {'type': 'float', 'low': -0.4, 'high': 0.0, 'step': 0.05},
            'hold_days': {'type': 'int', 'low': 5, 'high': 25, 'step': 5},
            'stop_loss': {'type': 'float', 'low': -0.10, 'high': -0.05, 'step': 0.01},
            'trailing_profit_level1': {'type': 'float', 'low': 0.05, 'high': 0.08, 'step': 0.01},
            'trailing_profit_level2': {'type': 'float', 'low': 0.10, 'high': 0.15, 'step': 0.01},
            'trailing_drawdown_level1': {'type': 'float', 'low': 0.05, 'high': 0.10, 'step': 0.01},
            'trailing_drawdown_level2': {'type': 'float', 'low': 0.03, 'high': 0.05, 'step': 0.01},
            'take_profit_multiplier': {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.5},
            'transformer_weight': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
            'transformer_buy_threshold': {'type': 'float', 'low': 0.6, 'high': 0.8, 'step': 0.05},
            'confidence_threshold': {'type': 'float', 'low': 0.4, 'high': 0.7, 'step': 0.05},
        }

    def get_default_params(self, regime: str = "neutral") -> Dict[str, Any]:
        """保守默认参数"""
        return {
            'buy_threshold': 0.6,
            'sell_threshold': -0.2,
            'hold_days': 15,
            'stop_loss': -0.08,
            'trailing_profit_level1': 0.06,
            'trailing_profit_level2': 0.12,
            'trailing_drawdown_level1': 0.08,
            'trailing_drawdown_level2': 0.04,
            'take_profit_multiplier': 3.0,
            'transformer_weight': 0.2,
            'transformer_buy_threshold': 0.65,
            'confidence_threshold': 0.5,
        }
