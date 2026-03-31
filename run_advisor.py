# -*- coding: utf-8 -*-
"""
实盘决策入口脚本
每日运行，生成买卖建议
"""
import os
import sys
import logging
import argparse

from config import get_settings


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="实盘决策助手")
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')
    parser.add_argument('--dry-run', action='store_true', help='仅检查环境和配置，不执行决策')
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger('advisor')
    settings = get_settings()

    print("\n" + "=" * 60)
    print("实盘决策助手 V3 启动")
    print("=" * 60)
    print(f"  策略文件: {settings.paths.strategy_file}")
    print(f"  持仓文件: {settings.paths.portfolio_file}")
    print(f"  模型路径: {settings.paths.model_path}")
    print(f"  风控 - 最大回撤: {settings.risk.max_drawdown_limit}%")
    print(f"  风控 - 最小利润因子: {settings.risk.min_profit_factor}")
    print(f"  风控 - 单只最大仓位: {settings.risk.max_position_ratio:.0%}")
    print(f"  风控 - 最小胜率: {settings.risk.min_win_rate}%")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] 环境检查:")

        # 检查策略文件
        if os.path.exists(settings.paths.strategy_file):
            import json
            with open(settings.paths.strategy_file, 'r') as f:
                strategies = json.load(f)
            print(f"  ✓ 策略文件: {len(strategies)} 只股票的策略参数")
        else:
            print(f"  ✗ 策略文件不存在，请先运行 run_backtest.py")

        # 检查模型
        model_exists = os.path.exists(settings.paths.model_path)
        swa_exists = os.path.exists(settings.paths.swa_model_path)
        print(f"  {'✓' if model_exists else '✗'} EMA 模型: {settings.paths.model_path}")
        print(f"  {'✓' if swa_exists else '✗'} SWA 模型: {settings.paths.swa_model_path}")

        # 检查 scaler
        scaler_exists = os.path.exists(settings.paths.scaler_path)
        global_scaler_exists = os.path.exists(settings.paths.global_scaler_path)
        print(f"  {'✓' if scaler_exists else '✗'} 专用 Scaler: {settings.paths.scaler_path}")
        print(f"  {'✓' if global_scaler_exists else '✗'} 全局 Scaler: {settings.paths.global_scaler_path}")

        # 检查持仓文件
        portfolio_exists = os.path.exists(settings.paths.portfolio_file)
        print(f"  {'✓' if portfolio_exists else '✗'} 持仓文件: {settings.paths.portfolio_file}")

        # 检查 GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"  ⚠️ 无 GPU，使用 CPU 推理")
        except ImportError:
            print(f"  ✗ PyTorch 未安装")

        return

    # 执行决策
    try:
        from live.advisor import run_advisor
        run_advisor()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        logger.error(f"决策执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
