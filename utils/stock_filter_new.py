# utils/stock_filter.py（增强版）
# -*- coding: utf-8 -*-
"""
股票过滤模块（增强版）
统一拦截入口，支持ST/退市/停牌等规则
新增：日志记录、正则表达式优化、异常处理
"""
import re
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------- 1. 文本级别：从"股票名称"识别 ST/退市 ----------
# 优化后的正则表达式，避免误判
_ST_PATTERNS = [
    # ST股票：必须以ST开头或包含*ST（避免误判包含ST字母的正常股票名）
    re.compile(r"^\*?ST", re.IGNORECASE),  # *ST 或 ST 开头
    re.compile(r"\s\*?ST\s", re.IGNORECASE),  # 中间包含 ST 或 *ST（前后有空格）
    re.compile(r"\*?ST$"),  # 以 ST 或 *ST 结尾
    # 退市股票
    re.compile(r"退市", re.IGNORECASE),
    re.compile(r"退", re.IGNORECASE),  # 退市整理期
]

# 白名单：包含ST但不应被过滤的股票名称
_ST_WHITELIST = [
    "STORAGE",  # 可能包含STORAGE的股票名
    "INVEST",  # 可能包含INVEST的股票名
]


def is_st_or_delisted_by_name(name: str, verbose: bool = False) -> bool:
    """
    判断股票名称是否命中 ST/退市（按文本规则）- 增强版

    改进：
    1. 更精确的正则匹配，避免误判
    2. 白名单机制
    3. 日志记录

    Args:
        name: 股票名称
        verbose: 是否输出详细日志

    Returns:
        True表示应被过滤，False表示正常
    """
    if not isinstance(name, str) or not name.strip():
        return False

    name_upper = name.upper()

    # 检查白名单
    for whitelist_item in _ST_WHITELIST:
        if whitelist_item in name_upper:
            if verbose:
                logger.debug(f"股票 {name} 在白名单中，跳过ST检查")
            return False

    # 检查ST/退市规则
    for pat in _ST_PATTERNS:
        if pat.search(name):
            if verbose:
                logger.info(f"股票 {name} 命中ST/退市规则: {pat.pattern}")
            return True

    return False


# ---------- 2. 日线级别：从个股日线 DataFrame 判断是否需要拦截 ----------
def should_intercept_stock(
        stock_code: str,
        stock_name: str,
        df,
        today=None,  # 不传就用 df 最后一个交易日；回测里可以传"回测起始日"
        allow_recent_suspended: bool = False,
        min_data_days: int = 60,  # 最小数据天数
        max_suspended_days: int = 10,  # 最大停牌天数
        verbose: bool = True,
) -> Tuple[bool, str]:
    """
    统一拦截入口（增强版）

    改进：
    1. 更精确的ST判断
    2. 数据完整性检查
    3. 详细日志记录
    4. 异常处理

    返回: (should_skip, reason)
      - should_skip=True：建议跳过（回测/预测都不要再往下走）
      - reason：人可读的拦截原因，方便打日志/报表
    """
    # 规则-0：名称层面的 ST/退市
    if is_st_or_delisted_by_name(stock_name):
        reason = f"[名称规则] ST/退市: {stock_name}"
        if verbose:
            logger.info(f"[拦截] {stock_code} {stock_name}: {reason}")
        return True, reason

    # 数据校验
    if df is None or len(df) == 0:
        reason = "[数据规则] 数据为空"
        if verbose:
            logger.warning(f"[拦截] {stock_code} {stock_name}: {reason}")
        return True, reason

    # 规则-1：数据长度检查
    try:
        if len(df) < min_data_days:
            reason = f"[数据规则] 数据长度 {len(df)} < {min_data_days}"
            if verbose:
                logger.warning(f"[拦截] {stock_code} {stock_name}: {reason}")
            return True, reason
    except Exception as e:
        logger.error(f"[{stock_code}] 数据长度检查失败: {e}")

    # 规则-2：停牌检查
    if not allow_recent_suspended:
        try:
            if today is None:
                today = df.index[-1]

            # 取最近20个交易日
            sample_size = min(20, len(df))
            sample = df.iloc[-sample_size:]

            # 检查成交量是否为0（停牌标志）
            if 'Volume' in sample.columns:
                suspended_days = (sample['Volume'] == 0).sum()

                if suspended_days >= max_suspended_days:
                    reason = f"[停牌规则] 近{sample_size}日停牌天数{suspended_days}>={max_suspended_days}"
                    if verbose:
                        logger.info(f"[拦截] {stock_code} {stock_name}: {reason}")
                    return True, reason
        except Exception as e:
            logger.warning(f"[{stock_code}] 停牌检查失败: {e}")

    # 规则-3：退市日期标记（如果你的日线里有 "DelistDate" 列）
    try:
        if "DelistDate" in df.columns:
            if today is None:
                today = df.index[-1]
            delist = df["DelistDate"].dropna().max()
            if pd.notna(delist) and today >= delist:
                reason = f"[退市日规则] 已退市（退市日期={delist}, 当前={today}）"
                if verbose:
                    logger.info(f"[拦截] {stock_code} {stock_name}: {reason}")
                return True, reason
    except Exception as e:
        logger.warning(f"[{stock_code}] 退市日期检查失败: {e}")

    # 规则-4：价格异常检查
    try:
        if 'Close' in df.columns:
            # 检查是否有负价格
            if (df['Close'] <= 0).any():
                reason = "[价格规则] 存在非正价格"
                if verbose:
                    logger.warning(f"[拦截] {stock_code} {stock_name}: {reason}")
                return True, reason

            # 检查价格是否异常波动（单日涨跌超过30%）
            returns = df['Close'].pct_change()
            abnormal_days = ((returns > 0.3) | (returns < -0.3)).sum()
            if abnormal_days > 5:
                reason = f"[价格规则] 异常波动天数过多: {abnormal_days}"
                if verbose:
                    logger.warning(f"[拦截] {stock_code} {stock_name}: {reason}")
                return True, reason
    except Exception as e:
        logger.warning(f"[{stock_code}] 价格异常检查失败: {e}")

    # 规则-5：缺失值检查
    try:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in df.columns:
                nan_ratio = df[col].isna().mean()
                if nan_ratio > 0.2:  # 缺失超过20%
                    reason = f"[数据规则] {col} 缺失比例过高: {nan_ratio:.1%}"
                    if verbose:
                        logger.warning(f"[拦截] {stock_code} {stock_name}: {reason}")
                    return True, reason
    except Exception as e:
        logger.warning(f"[{stock_code}] 缺失值检查失败: {e}")

    # 规则-6：预留：涨跌停、科创板/创业板特殊限制等，后续在这里扩展

    if verbose:
        logger.debug(f"[通过] {stock_code} {stock_name} 通过所有过滤规则")

    return False, ""


# ---------- 3. 批量过滤（用在"股票池"阶段） ----------
def filter_codes_by_name(
        mapping: Dict[str, str],  # 通常是 {name: code}
        verbose: bool = True,
) -> Dict[str, str]:
    """
    批量根据名称剔除 ST/退市（简单文本规则）- 增强版

    改进：
    1. 日志记录
    2. 统计信息

    Args:
        mapping: {stock_name: stock_code}
        verbose: 是否输出详细日志

    Returns:
        过滤后的 {stock_name: stock_code}
    """
    filtered = {}
    removed_count = 0

    for name, code in mapping.items():
        if is_st_or_delisted_by_name(name):
            if verbose:
                logger.info(f"[拦截] 股票池移除 ST/退市: {name} ({code})")
            removed_count += 1
            continue
        filtered[name] = code

    if verbose:
        logger.info(f"股票池过滤完成: 原始 {len(mapping)} 只, 移除 {removed_count} 只, 保留 {len(filtered)} 只")

    return filtered


# ---------- 4. 数据完整性预检查 ----------
def pre_check_stock_data(
        stock_code: str,
        stock_name: str,
        df: pd.DataFrame,
        min_days: int = 120,
        required_columns: Optional[List[str]] = None,
) -> Tuple[bool, Dict]:
    """
    数据完整性预检查

    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        df: 数据框
        min_days: 最小数据天数
        required_columns: 必需的列名

    Returns:
        (是否通过, 检查结果字典)
    """
    result = {
        "code": stock_code,
        "name": stock_name,
        "passed": True,
        "issues": [],
        "stats": {},
    }

    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 检查数据长度
    if df is None or len(df) == 0:
        result["passed"] = False
        result["issues"].append("数据为空")
        return result["passed"], result

    result["stats"]["rows"] = len(df)

    if len(df) < min_days:
        result["passed"] = False
        result["issues"].append(f"数据长度 {len(df)} < {min_days}")

    # 检查必需列
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        result["passed"] = False
        result["issues"].append(f"缺失列: {missing_cols}")

    # 检查缺失值
    for col in required_columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_ratio = nan_count / len(df)
            result["stats"][f"{col}_nan_ratio"] = nan_ratio

            if nan_ratio > 0.1:
                result["passed"] = False
                result["issues"].append(f"{col} 缺失比例 {nan_ratio:.1%}")

    # 检查价格有效性
    if 'Close' in df.columns:
        invalid_prices = (df['Close'] <= 0).sum()
        if invalid_prices > 0:
            result["passed"] = False
            result["issues"].append(f"存在 {invalid_prices} 个非正价格")

    # 检查日期索引
    if not isinstance(df.index, pd.DatetimeIndex):
        result["issues"].append("索引不是DatetimeIndex")

    return result["passed"], result
