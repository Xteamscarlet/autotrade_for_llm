# -*- coding: utf-8 -*-
"""
自定义异常体系
精确区分不同类型的错误，便于采取差异化的恢复策略
"""


class AutoTradeError(Exception):
    """所有自定义异常的基类"""
    pass


class DataFetchError(AutoTradeError):
    """数据获取失败

    触发场景：efinance/akshare API超时、返回空数据、网络中断等
    恢复策略：重试或使用缓存数据
    """
    def __init__(self, message: str, source: str = "", code: str = ""):
        self.source = source
        self.code = code
        super().__init__(f"[{source}] {message}".strip())


class DataValidationError(AutoTradeError):
    """数据校验失败

    触发场景：High < Low、存在NaN/Inf、价格<=0、数据长度不足等
    恢复策略：跳过该数据或使用填充策略
    """
    def __init__(self, message: str, column: str = "", details: str = ""):
        self.column = column
        self.details = details
        super().__init__(f"[{column}] {message} {details}".strip())


class ModelLoadError(AutoTradeError):
    """模型加载失败

    触发场景：权重文件损坏、版本不匹配、GPU内存不足等
    恢复策略：尝试加载备用模型或使用降级推理
    """
    def __init__(self, message: str, model_type: str = "", path: str = ""):
        self.model_type = model_type
        self.path = path
        super().__init__(f"[{model_type}] {message} (path={path})")


class RiskLimitExceeded(AutoTradeError):
    """风控限制触发

    触发场景：回撤超限、仓位超限、单日亏损超限等
    恢复策略：阻断交易或强制减仓
    """
    def __init__(self, message: str, limit_type: str = "", actual: float = 0.0, limit: float = 0.0):
        self.limit_type = limit_type
        self.actual = actual
        self.limit = limit
        super().__init__(f"[{limit_type}] {message} (actual={actual:.4f}, limit={limit:.4f})")


class StrategyError(AutoTradeError):
    """策略执行异常

    触发场景：信号生成逻辑错误、参数非法、权重计算失败等
    恢复策略：使用默认参数或跳过该股票
    """
    def __init__(self, message: str, strategy_name: str = "", step: str = ""):
        self.strategy_name = strategy_name
        self.step = step
        super().__init__(f"[{strategy_name}:{step}] {message}")


class BacktestError(AutoTradeError):
    """回测执行异常

    触发场景：数据不足无法划分、参数导致无交易、数值溢出等
    恢复策略：返回空结果，跳过该回测
    """
    def __init__(self, message: str, stock_code: str = "", split_info: str = ""):
        self.stock_code = stock_code
        self.split_info = split_info
        super().__init__(f"[{stock_code}] {message} (split={split_info})")


class CacheIOError(AutoTradeError):
    """缓存读写异常

    触发场景：pickle反序列化失败、文件损坏、磁盘空间不足等
    恢复策略：删除损坏缓存，重新生成
    """
    def __init__(self, message: str, cache_path: str = ""):
        self.cache_path = cache_path
        super().__init__(f"[Cache] {message} (path={cache_path})")
