# -*- coding: utf-8 -*-
"""
数据标准化模块
列名映射、OHLCV完整性校验、数据类型转换
"""
import numpy as np
import pandas as pd
from exceptions import DataValidationError


def normalize_stock_dataframe(df: pd.DataFrame, code: str = "") -> pd.DataFrame:
    """标准化股票 DataFrame

    - 中文列名 -> 英文列名
    - 数值类型转换
    - OHLCV完整性校验

    Args:
        df: 原始DataFrame（可能包含中文列名）
        code: 股票代码（用于错误报告）

    Returns:
        标准化后的DataFrame，index为日期，包含 Open/High/Low/Close/Volume 列

    Raises:
        DataValidationError: 数据校验失败
    """
    df = df.copy()

    # 列名映射
    rename_map = {
        '日期': 'Date', '开盘': 'Open', '最高': 'High',
        '最低': 'Low', '收盘': 'Close', '成交量': 'Volume',
        '换手率': 'Turnover Rate',
    }
    df = df.rename(columns=rename_map)

    # 日期处理
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    df = df.sort_index()

    # 数值类型转换
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # OHLCV 校验
    _validate_ohlcv(df, code)

    return df


def normalize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """标准化大盘 DataFrame"""
    df = df.copy()
    rename_map = {
        'date': 'Date', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume',
    }
    df = df.rename(columns=rename_map)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    df = df.sort_index()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def _validate_ohlcv(df: pd.DataFrame, code: str = ""):
    """OHLCV 数据完整性校验

    Raises:
        DataValidationError: 校验失败
    """
    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in df.columns:
            raise DataValidationError(f"缺少必要列: {col}", column=col, details=f"code={code}")

    # High >= Low 检查
    invalid = df[df['High'] < df['Low']]
    if len(invalid) > 0:
        raise DataValidationError(
            f"存在 High < Low 的行: {len(invalid)} 条",
            column="High/Low",
            details=f"code={code}",
        )

    # 价格 > 0 检查
    for col in required:
        invalid = df[df[col] <= 0]
        if len(invalid) > 0:
            raise DataValidationError(
                f"存在 {col} <= 0 的行: {len(invalid)} 条",
                column=col,
                details=f"code={code}",
            )

    # NaN 比例检查
    for col in required:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > 0.1:
            raise DataValidationError(
                f"{col} 缺失比例过高: {nan_ratio:.1%}",
                column=col,
                details=f"code={code}",
            )
