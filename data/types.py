# -*- coding: utf-8 -*-
"""
类型定义和常量
所有文件共享的特征列名、基础列名等常量集中在此定义
"""

# Transformer模型使用的20个特征（与训练时完全一致）
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover Rate',
    'MA5', 'MA10', 'MA20',
    'MACD', 'K', 'D', 'J', 'RSI', 'ADX',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CCI',
]

# 基础OHLCV列（数据下载后的最小列集）
BASE_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# 传统因子列名（量价因子 + 技术指标因子）
TRADITIONAL_FACTOR_COLS = [
    'mom_10', 'mom_20', 'atr_pct', 'vol_price_res',
    'rsi_norm', 'macd_hist_norm', 'bias_20', 'bb_width',
]

# AI因子列名
AI_FACTOR_COLS = ['transformer_prob', 'transformer_pred_ret', 'transformer_conf']

# 所有因子列名
ALL_FACTOR_COLS = TRADITIONAL_FACTOR_COLS + AI_FACTOR_COLS

# 不参与因子计算的列名
NON_FACTOR_COLS = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'Combined_Score']

# 市场状态枚举
REGIME_STRONG = 'strong'
REGIME_WEAK = 'weak'
REGIME_NEUTRAL = 'neutral'
ALL_REGIMES = [REGIME_STRONG, REGIME_WEAK, REGIME_NEUTRAL]

# 涨跌停比例
LIMIT_RATIO_DICT = {
    'main': 0.10,   # 主板 10%
    'gem': 0.20,    # 创业板 20%
    'star': 0.20,   # 科创板 20%
}


def get_limit_ratio(code: str) -> float:
    """根据股票代码获取涨跌停比例"""
    if code.startswith('300') or code.startswith('688'):
        return LIMIT_RATIO_DICT['gem']
    return LIMIT_RATIO_DICT['main']
