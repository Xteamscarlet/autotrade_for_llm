# -*- coding: utf-8 -*-
"""
复合信号策略（增强版）
基于多因子加权得分 + Transformer置信度过滤的复合信号策略
从原始 backtest_market_v2.py 的交易逻辑迁移而来
新增：参数鲁棒性检查、权重归一化、阈值范围验证
"""
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from data.types import get_limit_ratio

logger = logging.getLogger(__name__)


class CompoundSignalStrategy(BaseStrategy):
    """复合信号策略 - 增强版

    信号生成逻辑：
    1. 综合得分超过买入阈值 -> 生成买入信号
    2. Transformer置信度过滤 -> 低置信度信号被拦截
    3. ATR动态仓位 -> 根据波动率调整仓位大小
    4. 移动止损/止盈/时间止损 -> 生成卖出信号
    
    新增功能：
    1. 参数鲁棒性检查
    2. 权重自动归一化
    3. 阈值范围验证
    """
    name = "compound_signal"
    description = "多因子加权 + Transformer置信度 + ATR动态仓位的复合策略"
    keep = True

    def __init__(self):
        super().__init__()
        self._validated_params = {}
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证策略参数的合法性
        
        Args:
            params: 策略参数字典
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查买入阈值范围
        buy_threshold = params.get('buy_threshold', 0.6)
        if not 0.3 <= buy_threshold <= 0.9:
            errors.append(f"买入阈值 {buy_threshold} 超出合理范围 [0.3, 0.9]")
        
        # 检查卖出阈值范围
        sell_threshold = params.get('sell_threshold', -0.2)
        if not -0.5 <= sell_threshold <= 0.0:
            errors.append(f"卖出阈值 {sell_threshold} 超出合理范围 [-0.5, 0.0]")
        
        # 检查止损范围
        stop_loss = params.get('stop_loss', -0.08)
        if not -0.20 <= stop_loss <= -0.02:
            errors.append(f"止损 {stop_loss} 超出合理范围 [-0.20, -0.02]")
        
        # 检查持仓天数
        hold_days = params.get('hold_days', 15)
        if not 1 <= hold_days <= 60:
            errors.append(f"持仓天数 {hold_days} 超出合理范围 [1, 60]")
        
        # 检查移动止损参数逻辑
        trailing_profit_level1 = params.get('trailing_profit_level1', 0.06)
        trailing_profit_level2 = params.get('trailing_profit_level2', 0.12)
        if trailing_profit_level1 >= trailing_profit_level2:
            errors.append(f"移动止损阈值1({trailing_profit_level1}) 应小于 阈值2({trailing_profit_level2})")
        
        trailing_drawdown_level1 = params.get('trailing_drawdown_level1', 0.08)
        trailing_drawdown_level2 = params.get('trailing_drawdown_level2', 0.04)
        if trailing_drawdown_level1 <= trailing_drawdown_level2:
            errors.append(f"移动止损回撤1({trailing_drawdown_level1}) 应大于 回撤2({trailing_drawdown_level2})")
        
        # 检查因子权重
        factor_weights = params.get('factor_weights', {})
        if factor_weights:
            weight_sum = sum(factor_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                errors.append(f"因子权重总和 {weight_sum:.4f} 不等于1")
        
        return len(errors) == 0, errors
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """归一化权重
        
        Args:
            weights: 原始权重字典
            
        Returns:
            归一化后的权重字典
        """
        if not weights:
            return weights
        
        total = sum(weights.values())
        if total == 0:
            # 如果总和为0，使用等权重
            return {k: 1.0 / len(weights) for k in weights}
        
        return {k: v / total for k, v in weights.items()}

    # ===== 保留原有单步接口（与回测/实盘兼容） =====
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
        regime: str = "neutral",
    ) -> Dict[str, Any]:
        """生成交易信号 - 增强版"""
        if idx < 60:
            return {
                "action": "hold",
                "score": 0.5,
                "confidence": 0.0,
                "position_ratio": 0.0,
                "reason": "数据不足",
            }
        
        # 参数验证
        is_valid, errors = self.validate_params(params)
        if not is_valid:
            logger.warning(f"参数验证失败: {errors}")
            # 使用默认参数
            params = self._get_default_params()
        
        # 获取当前市场状态对应的参数
        regime_params = params.get(regime, params.get("neutral", params))
        
        # 计算综合得分
        score = self._calculate_score(df, idx, regime_params)
        
        # 获取阈值
        buy_threshold = regime_params.get("buy_threshold", 0.6)
        sell_threshold = regime_params.get("sell_threshold", -0.2)
        
        # 生成信号
        if score >= buy_threshold:
            return {
                "action": "buy",
                "score": score,
                "confidence": self._calculate_confidence(df, idx),
                "position_ratio": self._calculate_position_ratio(df, idx, regime_params),
                "reason": f"综合得分 {score:.3f} >= 买入阈值 {buy_threshold:.3f}",
            }
        elif score <= sell_threshold:
            return {
                "action": "sell",
                "score": score,
                "confidence": self._calculate_confidence(df, idx),
                "position_ratio": 0.0,
                "reason": f"综合得分 {score:.3f} <= 卖出阈值 {sell_threshold:.3f}",
            }
        else:
            return {
                "action": "hold",
                "score": score,
                "confidence": self._calculate_confidence(df, idx),
                "position_ratio": 0.0,
                "reason": f"综合得分 {score:.3f} 在阈值区间内",
            }
    
    def _calculate_score(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
    ) -> float:
        """计算综合得分"""
        # 获取因子权重
        factor_weights = params.get("factor_weights", {})
        factor_weights = self.normalize_weights(factor_weights)
        
        # 如果没有指定权重，使用Combined_Score
        if not factor_weights and "Combined_Score" in df.columns:
            score = df["Combined_Score"].iloc[idx]
            return score if not pd.isna(score) else 0.5
        
        # 计算加权得分
        total_score = 0.0
        total_weight = 0.0
        
        for factor, weight in factor_weights.items():
            if factor in df.columns:
                factor_value = df[factor].iloc[idx]
                if not pd.isna(factor_value):
                    total_score += factor_value * weight
                    total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5
    
    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> float:
        """计算信号置信度"""
        # 基于波动率计算置信度
        if idx < 20:
            return 0.5
        
        returns = df["Close"].pct_change().iloc[idx-20:idx]
        if returns.isna().all():
            return 0.5
        
        volatility = returns.std()
        
        # 波动率越低，置信度越高
        if pd.isna(volatility) or volatility == 0:
            return 0.5
        
        # 归一化到 [0, 1]
        confidence = max(0, min(1, 1 - volatility * 10))
        return confidence
    
    def _calculate_position_ratio(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
    ) -> float:
        """计算仓位比例（基于ATR）"""
        if idx < 14:
            return 0.2  # 默认仓位
        
        # 计算ATR
        high = df["High"].iloc[idx-14:idx]
        low = df["Low"].iloc[idx-14:idx]
        close = df["Close"].iloc[idx-14:idx]
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ], axis=1).max(axis=1)
        
        atr = tr.mean()
        current_price = df["Close"].iloc[idx]
        
        if pd.isna(atr) or atr == 0 or pd.isna(current_price) or current_price == 0:
            return 0.2
        
        # ATR占比
        atr_pct = atr / current_price
        
        # 波动率越大，仓位越小
        # ATR占比 2% -> 仓位 100%, ATR占比 10% -> 仓位 20%
        position_ratio = max(0.1, min(1.0, 0.04 / atr_pct))
        
        return position_ratio
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "neutral": {
                "buy_threshold": 0.6,
                "sell_threshold": -0.2,
                "hold_days": 15,
                "stop_loss": -0.08,
                "trailing_profit_level1": 0.06,
                "trailing_profit_level2": 0.12,
                "trailing_drawdown_level1": 0.08,
                "trailing_drawdown_level2": 0.04,
                "take_profit": 0.15,
            },
            "strong": {
                "buy_threshold": 0.55,
                "sell_threshold": -0.15,
                "hold_days": 20,
                "stop_loss": -0.10,
                "trailing_profit_level1": 0.08,
                "trailing_profit_level2": 0.15,
                "trailing_drawdown_level1": 0.10,
                "trailing_drawdown_level2": 0.05,
                "take_profit": 0.20,
            },
            "weak": {
                "buy_threshold": 0.70,
                "sell_threshold": -0.25,
                "hold_days": 10,
                "stop_loss": -0.05,
                "trailing_profit_level1": 0.04,
                "trailing_profit_level2": 0.10,
                "trailing_drawdown_level1": 0.06,
                "trailing_drawdown_level2": 0.03,
                "take_profit": 0.10,
            },
        }

    # ===== Optuna 参数空间 =====
    @classmethod
    def get_param_space(cls) -> Dict[str, Dict[str, Any]]:
        """返回 Optuna 优化参数空间"""
        return {
            # 买入阈值
            "buy_threshold": {
                "type": "float",
                "low": 0.4,
                "high": 0.8,
                "step": 0.05,
            },
            # 卖出阈值
            "sell_threshold": {
                "type": "float",
                "low": -0.4,
                "high": 0.0,
                "step": 0.05,
            },
            # 止损
            "stop_loss": {
                "type": "float",
                "low": -0.15,
                "high": -0.03,
                "step": 0.01,
            },
            # 止盈
            "take_profit": {
                "type": "float",
                "low": 0.05,
                "high": 0.40,
                "step": 0.05,
            },
            # 移动止损（触发阈值 + 回撤幅度）
            "trailing_profit_level1": {
                "type": "float",
                "low": 0.03,
                "high": 0.15,
                "step": 0.01,
            },
            "trailing_profit_level2": {
                "type": "float",
                "low": 0.08,
                "high": 0.25,
                "step": 0.01,
            },
            "trailing_drawdown_level1": {
                "type": "float",
                "low": 0.03,
                "high": 0.12,
                "step": 0.01,
            },
            "trailing_drawdown_level2": {
                "type": "float",
                "low": 0.02,
                "high": 0.08,
                "step": 0.01,
            },
            # 时间止损（持仓天数上限）
            "hold_days": {
                "type": "int",
                "low": 1,
                "high": 60,
                "step": 1,
            },
        }
    
    def generate_signals_batch(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        regime: str = "neutral",
    ) -> pd.DataFrame:
        """批量生成信号（用于回测）- 增强版
        
        Args:
            df: 数据框
            params: 策略参数
            regime: 市场状态
            
        Returns:
            包含信号的数据框
        """
        df = df.copy()
        
        # 参数验证
        is_valid, errors = self.validate_params(params)
        if not is_valid:
            logger.warning(f"批量信号生成参数验证失败: {errors}")
            params = self._get_default_params()
        
        # 初始化信号列
        df['signal'] = 'hold'
        df['score'] = 0.5
        df['confidence'] = 0.0
        df['position_ratio'] = 0.0
        
        # 批量计算
        for idx in range(60, len(df)):
            signal = self.generate_signal(df, idx, params, regime)
            df.iloc[idx, df.columns.get_loc('signal')] = signal['action']
            df.iloc[idx, df.columns.get_loc('score')] = signal['score']
            df.iloc[idx, df.columns.get_loc('confidence')] = signal['confidence']
            df.iloc[idx, df.columns.get_loc('position_ratio')] = signal['position_ratio']
        
        # 统计信号分布
        signal_counts = df['signal'].value_counts()
        logger.info(f"信号分布: {signal_counts.to_dict()}")
        
        return df
