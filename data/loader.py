# -*- coding: utf-8 -*-
"""
数据加载模块
封装 efinance / akshare 的数据获取逻辑，统一错误处理和重试机制
"""
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import numpy as np

from config import get_settings
from exceptions import DataFetchError

logger = logging.getLogger(__name__)

# 延迟导入，避免未安装时报错
try:
    import efinance as ef
    _EF_AVAILABLE = True
except ImportError:
    _EF_AVAILABLE = False
    ef = None
    logger.warning("efinance 未安装，个股数据下载功能不可用")

try:
    import akshare as ak
    _AK_AVAILABLE = True
except ImportError:
    _AK_AVAILABLE = False
    ak = None
    logger.warning("akshare 未安装，大盘数据下载功能不可用")


def _setup_proxy():
    """设置代理环境变量"""
    proxy = os.getenv('HTTP_PROXY', '') or os.getenv('HTTPS_PROXY', '')
    if proxy:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy


def download_market_data() -> Optional[pd.DataFrame]:
    """下载大盘数据（上证指数）

    Returns:
        包含 Close, MA20, trend 列的 DataFrame，index为日期
        失败返回 None
    """
    _setup_proxy()
    settings = get_settings()

    if not _AK_AVAILABLE:
        logger.error("akshare 未安装，无法下载大盘数据")
        return None

    logger.info("开始下载大盘数据...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['trend'] = np.where(df['Close'] > df['MA20'], 1, 0)

        logger.info(f"大盘数据下载完成，最新日期: {df.index[-1].strftime('%Y-%m-%d')}")
        return df[['Close', 'MA20', 'trend']]

    except Exception as e:
        raise DataFetchError(
            f"大盘数据下载失败: {e}",
            source="akshare",
        )


def download_stocks_data(
    stock_codes_dict: Dict[str, str],
    min_length: int = 60,
    sleep_seconds: int = 10,
) -> Dict[str, pd.DataFrame]:
    """下载多个股票的历史数据

    Args:
        stock_codes_dict: {股票名称: 股票代码}
        min_length: 最小数据长度要求
        sleep_seconds: 每只股票之间的间隔秒数

    Returns:
        {股票名称: DataFrame}，DataFrame 包含 Open/High/Low/Close/Volume 列
    """
    _setup_proxy()

    if not _EF_AVAILABLE:
        raise DataFetchError("efinance 未安装，无法下载个股数据", source="efinance")

    logger.info(f"开始下载 {len(stock_codes_dict)} 只股票的数据...")
    stocks_data = {}
    failed = []

    for name, code in stock_codes_dict.items():
        try:
            time.sleep(sleep_seconds)
            df = _download_single_stock(code, name, min_length)
            if df is not None:
                stocks_data[name] = df
                logger.info(f"✓ {name} ({code}): {len(df)} 条")
            else:
                failed.append(name)
        except DataFetchError:
            failed.append(name)
        except Exception as e:
            logger.error(f"✗ {name} ({code}): {e}")
            failed.append(name)

    if failed:
        logger.warning(f"下载失败: {failed}")

    return stocks_data


def get_single_stock_data(
    code: str,
    start_date: str = "20210101",
    end_date: str = "20270101",
) -> Optional[pd.DataFrame]:
    """获取单只股票的历史数据（含中文列名映射）

    Returns:
        包含中文原始列名的 DataFrame，失败返回 None
    """
    _setup_proxy()

    if not _EF_AVAILABLE:
        return None

    try:
        time.sleep(10)
        df = ef.stock.get_quote_history(code, beg=start_date, end=end_date)
        if df is None or df.empty:
            logger.warning(f"股票 {code} 返回数据为空")
            return None
        logger.info(f"股票 {code} 成功获取到 {len(df)} 条数据")
        return df
    except Exception as e:
        logger.error(f"获取股票 {code} 数据失败: {e}")
        return None


def _download_single_stock(code: str, name: str, min_length: int) -> Optional[pd.DataFrame]:
    """下载单只股票并标准化列名"""
    try:
        df = ef.stock.get_quote_history(code)
        if df is None or len(df) < min_length:
            logger.warning(f"✗ {name} ({code}): 数据不足 ({len(df) if df is not None else 0})")
            return None

        df = df.rename(columns={
            '日期': 'Date', '开盘': 'Open', '最高': 'High',
            '最低': 'Low', '收盘': 'Close', '成交量': 'Volume',
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[df[col] > 0]

        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df[df['Volume'] >= 0]

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        if len(df) < min_length:
            logger.warning(f"✗ {name} ({code}): 清洗后数据不足 ({len(df)})")
            return None

        return df

    except Exception as e:
        raise DataFetchError(f"下载失败: {e}", source="efinance", code=code)
