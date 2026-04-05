# -*- coding: utf-8 -*-
"""
数据标准化模块（增强版）
列名映射、OHLCV完整性校验、数据类型转换
新增：NaN填充、训练集/验证集分离归一化、数据分布监控
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DataNormalizer:
    """数据归一化器
    
    支持训练集/验证集分离归一化，避免信息泄露
    """
    
    def __init__(self, method: str = 'zscore'):
        """
        Args:
            method: 归一化方法 ('zscore', 'minmax', 'robust')
        """
        self.method = method
        self.stats: Dict[str, Dict[str, float]] = {}  # 存储各列的统计量
        
    def fit(self, df: pd.DataFrame, columns: Optional[list] = None) -> 'DataNormalizer':
        """从训练数据计算归一化参数
        
        Args:
            df: 训练数据
            columns: 需要归一化的列，None表示所有数值列
            
        Returns:
            self
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                logger.warning(f"列 {col} 无有效数据，跳过归一化")
                continue
            
            if self.method == 'zscore':
                self.stats[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std() if col_data.std() > 0 else 1.0,
                }
            elif self.method == 'minmax':
                self.stats[col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min() if col_data.max() != col_data.min() else 1.0,
                }
            elif self.method == 'robust':
                self.stats[col] = {
                    'median': col_data.median(),
                    'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                }
                if self.stats[col]['iqr'] == 0:
                    self.stats[col]['iqr'] = 1.0
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用归一化变换
        
        Args:
            df: 待变换数据
            
        Returns:
            归一化后的数据
        """
        df = df.copy()
        
        for col, stat in self.stats.items():
            if col not in df.columns:
                continue
            
            if self.method == 'zscore':
                df[col] = (df[col] - stat['mean']) / stat['std']
            elif self.method == 'minmax':
                df[col] = (df[col] - stat['min']) / stat['range']
            elif self.method == 'robust':
                df[col] = (df[col] - stat['median']) / stat['iqr']
            
            # 处理异常值
            df[col] = df[col].clip(-5, 5)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """拟合并变换"""
        return self.fit(df, columns).transform(df)


def normalize_stock_dataframe(
    df: pd.DataFrame, 
    code: str = "",
    fill_nan_method: str = 'ffill',
    validate: bool = True,
) -> pd.DataFrame:
    """标准化股票 DataFrame - 增强版

    - 中文列名 -> 英文列名
    - 数值类型转换
    - OHLCV完整性校验
    - NaN填充处理
    - 索引类型转换

    Args:
        df: 原始DataFrame（可能包含中文列名）
        code: 股票代码（用于错误报告）
        fill_nan_method: NaN填充方法 ('ffill', 'bfill', 'mean', 'median', 'drop')
        validate: 是否进行数据校验

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
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        except Exception as e:
            logger.warning(f"[{code}] 日期转换失败: {e}")
    
    # 强制转换为DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise DataValidationError(
                f"索引无法转换为DatetimeIndex: {e}",
                column="index",
                details=f"code={code}"
            )
    
    # 排序
    df = df.sort_index()
    
    # 处理重复索引
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.warning(f"[{code}] 发现 {dup_count} 个重复日期索引，保留最后一个值")
        df = df[~df.index.duplicated(keep='last')]

    # 数值类型转换
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # NaN处理
    df = _handle_nan_values(df, code, fill_nan_method)

    # 数据校验
    if validate:
        _validate_ohlcv_data(df, code)

    return df


def _handle_nan_values(
    df: pd.DataFrame, 
    code: str,
    method: str = 'ffill'
) -> pd.DataFrame:
    """处理NaN值
    
    Args:
        df: 数据框
        code: 股票代码
        method: 填充方法
        
    Returns:
        处理后的数据框
    """
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in ohlcv_cols:
        if col not in df.columns:
            continue
            
        nan_count = df[col].isna().sum()
        if nan_count == 0:
            continue
        
        nan_ratio = nan_count / len(df)
        
        # NaN比例过高则警告
        if nan_ratio > 0.1:
            logger.warning(f"[{code}] {col} 缺失比例较高: {nan_ratio:.1%}")
        
        if method == 'ffill':
            df[col] = df[col].ffill()
            # 如果开头有NaN，用后向填充
            df[col] = df[col].bfill()
        elif method == 'bfill':
            df[col] = df[col].bfill()
            df[col] = df[col].ffill()
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'drop':
            df = df.dropna(subset=[col])
        
        # 记录处理结果
        remaining_nan = df[col].isna().sum()
        if remaining_nan > 0:
            logger.warning(f"[{code}] {col} 仍有 {remaining_nan} 个NaN无法填充")
    
    return df


def _validate_ohlcv_data(df: pd.DataFrame, code: str = ""):
    """OHLCV 数据完整性校验 - 增强版

    Raises:
        DataValidationError: 校验失败
    """
    required = ['Open', 'High', 'Low', 'Close']
    
    # 检查必要列是否存在
    for col in required:
        if col not in df.columns:
            raise DataValidationError(f"缺少必要列: {col}", column=col, details=f"code={code}")

    # 检查数据长度
    if len(df) == 0:
        raise DataValidationError("数据为空", column="all", details=f"code={code}")

    # High >= Low 检查
    invalid = df[df['High'] < df['Low']]
    if len(invalid) > 0:
        # 不直接抛出异常，而是修正数据
        logger.warning(f"[{code}] 存在 High < Low 的行: {len(invalid)} 条，将进行修正")
        # 修正：将High设为Low和High的最大值
        df['High'] = df[['High', 'Low']].max(axis=1)

    # 价格 > 0 检查
    for col in required:
        invalid = df[df[col] <= 0]
        if len(invalid) > 0:
            invalid_ratio = len(invalid) / len(df)
            if invalid_ratio > 0.05:  # 超过5%的非正值视为严重问题
                raise DataValidationError(
                    f"存在 {col} <= 0 的行比例过高: {invalid_ratio:.1%}",
                    column=col,
                    details=f"code={code}",
                )
            else:
                logger.warning(f"[{code}] 存在 {col} <= 0 的行: {len(invalid)} 条，将被移除")
                df.drop(invalid.index, inplace=True)

    # NaN 比例检查
    for col in required:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > 0.1:
            raise DataValidationError(
                f"{col} 缺失比例过高: {nan_ratio:.1%}",
                column=col,
                details=f"code={code}",
            )


def normalize_for_train_test_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    method: str = 'zscore',
) -> Tuple[pd.DataFrame, pd.DataFrame, DataNormalizer]:
    """为训练集和测试集分别进行归一化（避免信息泄露）
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        feature_columns: 特征列
        method: 归一化方法
        
    Returns:
        (归一化后的训练数据, 归一化后的测试数据, 归一化器)
    """
    normalizer = DataNormalizer(method=method)
    
    # 在训练集上拟合
    train_normalized = normalizer.fit_transform(train_df, feature_columns)
    
    # 应用到测试集
    test_normalized = normalizer.transform(test_df)
    
    # 监控数据分布
    _log_distribution_comparison(train_df, train_normalized, test_df, test_normalized, feature_columns)
    
    return train_normalized, test_normalized, normalizer


def _log_distribution_comparison(
    train_before: pd.DataFrame,
    train_after: pd.DataFrame,
    test_before: pd.DataFrame,
    test_after: pd.DataFrame,
    columns: list,
):
    """记录归一化前后的数据分布变化"""
    for col in columns:
        if col not in train_before.columns:
            continue
        
        train_mean_before = train_before[col].mean()
        train_std_before = train_before[col].std()
        train_mean_after = train_after[col].mean()
        train_std_after = train_after[col].std()
        
        test_mean_before = test_before[col].mean()
        test_std_before = test_before[col].std()
        test_mean_after = test_after[col].mean()
        test_std_after = test_after[col].std()
        
        logger.debug(
            f"列 {col} 分布变化:\n"
            f"  训练集: mean {train_mean_before:.4f} -> {train_mean_after:.4f}, "
            f"std {train_std_before:.4f} -> {train_std_after:.4f}\n"
            f"  测试集: mean {test_mean_before:.4f} -> {test_mean_after:.4f}, "
            f"std {test_std_before:.4f} -> {test_std_after:.4f}"
        )


def clean_dataframe(
    df: pd.DataFrame,
    code: str = "",
    remove_duplicates: bool = True,
    fill_nan: bool = True,
    clip_outliers: bool = True,
    outlier_threshold: float = 5.0,
) -> pd.DataFrame:
    """数据清洗工具函数
    
    Args:
        df: 待清洗的数据框
        code: 股票代码
        remove_duplicates: 是否移除重复索引
        fill_nan: 是否填充NaN
        clip_outliers: 是否裁剪异常值
        outlier_threshold: 异常值阈值（标准差倍数）
        
    Returns:
        清洗后的数据框
    """
    df = df.copy()
    
    # 确保索引为DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.warning(f"[{code}] 索引转换失败: {e}")
    
    # 排序
    df = df.sort_index()
    
    # 移除重复索引
    if remove_duplicates and df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.debug(f"[{code}] 移除 {dup_count} 个重复索引")
        df = df[~df.index.duplicated(keep='last')]
    
    # 填充NaN
    if fill_nan:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                df[col] = df[col].ffill().bfill()
                logger.debug(f"[{code}] {col} 填充 {nan_count} 个NaN")
    
    # 裁剪异常值
    if clip_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower = mean - outlier_threshold * std
                upper = mean + outlier_threshold * std
                clipped = ((df[col] < lower) | (df[col] > upper)).sum()
                if clipped > 0:
                    df[col] = df[col].clip(lower, upper)
                    logger.debug(f"[{code}] {col} 裁剪 {clipped} 个异常值")
    
    return df


def validate_data_consistency(
    df: pd.DataFrame,
    expected_columns: list,
    code: str = "",
) -> Tuple[bool, list]:
    """验证数据一致性
    
    Args:
        df: 数据框
        expected_columns: 期望的列名列表
        code: 股票代码
        
    Returns:
        (是否一致, 缺失的列列表)
    """
    missing = [col for col in expected_columns if col not in df.columns]
    
    if missing:
        logger.warning(f"[{code}] 缺失列: {missing}")
        return False, missing
    
    # 检查数据类型
    for col in expected_columns:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            logger.warning(f"[{code}] 列 {col} 不是数值类型: {df[col].dtype}")
    
    return True, []
