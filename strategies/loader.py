# -*- coding: utf-8 -*-
"""
策略动态加载器
自动扫描策略目录，加载所有 keep=True 的策略类
"""
import os
import importlib
import logging
from typing import Dict, Type, List

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def load_strategies(strategies_dir: str = "strategies") -> Dict[str, Type[BaseStrategy]]:
    """动态加载所有策略

    扫描指定目录下的 .py 文件，找到所有 BaseStrategy 子类，
    只加载 keep=True 的策略。

    Returns:
        {策略名称: 策略类}
    """
    strategies = {}

    if not os.path.exists(strategies_dir):
        logger.warning(f"策略目录不存在: {strategies_dir}")
        return strategies

    for fname in os.listdir(strategies_dir):
        if not fname.endswith('.py') or fname.startswith('_'):
            continue

        module_name = fname[:-3]
        try:
            spec = importlib.util.spec_from_file_location(
                f"strategies.{module_name}",
                os.path.join(strategies_dir, fname),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, BaseStrategy)
                        and attr is not BaseStrategy
                        and attr.keep):
                    strategies[attr.name] = attr
                    logger.info(f"加载策略: {attr.name} ({fname})")

        except Exception as e:
            logger.error(f"加载策略文件 {fname} 失败: {e}")

    return strategies


def get_default_strategy() -> Type[BaseStrategy]:
    """获取默认策略（复合信号策略）"""
    from strategies.compound_signal import CompoundSignalStrategy
    return CompoundSignalStrategy
