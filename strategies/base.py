# -*- coding: utf-8 -*-
"""
策略抽象基类
定义策略的生命周期方法和标准接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import pandas as pd


class BaseStrategy(ABC):
    """策略抽象基类

    所有具体策略必须继承此类并实现以下方法：
    - generate_signal(): 生成交易信号
    - get_param_space(): 定义 Optuna 参数空间
    """

    name: str = "base"
    description: str = ""
    keep: bool = False  # 是否为保留策略（用于策略筛选）

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict[str, Any],
        regime: str = "neutral",
    ) -> Dict[str, Any]:
        """生成交易信号

        Args:
            df: 包含因子和价格数据的 DataFrame
            idx: 当前时间步索引
            params: 策略参数（由 Optuna 优化得到）
            regime: 市场状态

        Returns:
            {
                'action': 'buy' | 'sell' | 'hold',
                'score': float,         # 综合得分
                'confidence': float,    # 置信度
                'position_ratio': float, # 建议仓位比例
                'reason': str,          # 原因描述
            }
        """
        pass

    @abstractmethod
    def get_param_space(self) -> Dict[str, Dict]:
        """定义 Optuna 参数空间

        Returns:
            {参数名: {'type': 'float'|'int', 'low': ..., 'high': ..., 'step': ...}}
        """
        pass

    def get_default_params(self, regime: str = "neutral") -> Dict[str, Any]:
        """获取默认参数（当 Optuna 优化失败时的回退方案）"""
        space = self.get_param_space()
        defaults = {}
        for key, spec in space.items():
            if spec['type'] == 'float':
                defaults[key] = (spec['low'] + spec['high']) / 2
            elif spec['type'] == 'int':
                defaults[key] = int((spec['low'] + spec['high']) / 2)
        return defaults

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """校验参数合法性"""
        space = self.get_param_space()
        for key, spec in space.items():
            if key not in params:
                return False
            val = params[key]
            if spec['type'] == 'float':
                if not (spec['low'] <= val <= spec['high']):
                    return False
            elif spec['type'] == 'int':
                if not (spec['low'] <= int(val) <= spec['high']):
                    return False
        return True
