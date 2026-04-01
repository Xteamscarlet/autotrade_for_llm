# -*- coding: utf-8 -*-
"""
模型训练入口脚本
负责数据准备（如需）、调用训练循环、训练后通知
"""
import os
import json
import logging
import argparse
import time

import requests

from config import get_settings

# ==================== 日志配置 ====================

class ChatLogger(logging.Handler):
    """企业微信日志推送 Handler"""

    def __init__(self, webhook_url: str = ""):
        super().__init__()
        self.setLevel(logging.WARNING)
        self.original_stdout = None
        self.webhook_url = webhook_url

    def set_stdout(self, stdout):
        self.original_stdout = stdout

    def emit(self, record):
        if self.original_stdout:
            self.original_stdout.write(self.format(record) + '\n')
        if self.webhook_url:
            try:
                headers = {'Content-Type': 'application/json'}
                data = {"msgtype": "text", "text": {"content": self.format(record).strip()}}
                requests.post(self.webhook_url, headers=headers, data=json.dumps(data), timeout=5)
            except Exception:
                pass


def send_file_to_wechat(file_path: str, webhook_url: str = "", upload_url: str = "") -> bool:
    """上传文件到企业微信并发送"""
    if not webhook_url or not upload_url:
        return False
    try:
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) > 20 * 1024 * 1024:
            return False

        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            upload_resp = requests.post(upload_url, files=files, params={'type': 'file'}, timeout=30)
            upload_result = upload_resp.json()
            if upload_result.get('errcode') != 0:
                return False

            media_id = upload_result.get('media_id')
            headers = {'Content-Type': 'application/json'}
            data = {"msgtype": "file", "file": {"media_id": media_id}}
            send_resp = requests.post(webhook_url, headers=headers, data=json.dumps(data), timeout=10)
            return send_resp.json().get('errcode') == 0
    except Exception:
        return False


def prepare_training_data(settings) -> bool:
    """检查训练数据文件是否存在"""
    if os.path.exists(settings.paths.stock_data_file):
        print(f"✓ 训练数据已存在: {settings.paths.stock_data_file}")
        return True
    else:
        print(f"✗ 训练数据文件不存在: {settings.paths.stock_data_file}")
        print("  请先使用 TransformerStock.py 中的 get_all_stock_data() 下载数据")
        return False


def main():
    parser = argparse.ArgumentParser(description="Transformer 模型训练")
    parser.add_argument('--dry-run', action='store_true', help='仅检查环境，不执行训练')
    parser.add_argument('--notify', action='store_true', help='训练完成后发送企业微信通知')
    args = parser.parse_args()

    settings = get_settings()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    )
    logger = logging.getLogger('train')
    if args.notify and settings.paths.wechat_webhook:
        chat_handler = ChatLogger(settings.paths.wechat_webhook)
        chat_handler.set_stdout(__import__('sys').stdout)
        logger.addHandler(chat_handler)

    print("\n" + "=" * 60)
    print(f"Transformer 模型训练 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Lookback: {settings.model.lookback_days} 天")
    print(f"  Batch Size: {settings.model.batch_size}")
    print(f"  Epochs: {settings.model.epochs}")
    print(f"  Learning Rate: {settings.model.learning_rate}")
    print(f"  Weight Decay: {settings.model.weight_decay}")
    print(f"  Layers: {settings.model.num_layers} x {settings.model.dim_feedforward}")
    print(f"  Heads: {settings.model.num_heads}")
    print(f"  Dropout: {settings.model.dropout}")
    print(f"  Accumulation: {settings.model.accumulation_steps} steps")
    print(f"  EMA Decay: {settings.model.ema_decay}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] 仅检查环境，不执行训练")
        data_ok = prepare_training_data(settings)
        if data_ok:
            print("✓ 环境检查通过，可以开始训练")
        else:
            print("✗ 环境检查未通过，请先准备数据")
        return

    # 检查训练数据
    if not prepare_training_data(settings):
        return

    # 检查 GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n⚠️ 未检测到 GPU，将使用 CPU 训练（速度较慢）")

    # 执行训练
    print("\n开始训练...\n")
    start_time = time.time()

    try:
        from model.trainer import train_model
        train_model(settings)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        if args.notify and settings.paths.wechat_webhook:
            try:
                headers = {'Content-Type': 'application/json'}
                data = {"msgtype": "text", "text": {"content": f"❌ 训练失败: {e}"}}
                requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)
            except Exception:
                pass
        return

    elapsed = time.time() - start_time
    print(f"\n✓ 训练完成！耗时: {elapsed / 60:.1f} 分钟")

    # 训练后通知
    if args.notify:
        model_files = []
        for p in [settings.paths.model_path, settings.paths.swa_model_path]:
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / 1024 / 1024
                model_files.append(f"{os.path.basename(p)} ({size_mb:.1f}MB)")

        topk_dir = settings.paths.topk_checkpoint_dir
        if os.path.exists(topk_dir):
            for f in os.listdir(topk_dir):
                if f.endswith('.pth'):
                    size_mb = os.path.getsize(os.path.join(topk_dir, f)) / 1024 / 1024
                    model_files.append(f"{f} ({size_mb:.1f}MB)")

        msg = f"✅ 训练完成！\n耗时: {elapsed / 60:.1f}分钟\n模型文件:\n" + "\n".join(model_files)

        if settings.paths.wechat_webhook:
            try:
                headers = {'Content-Type': 'application/json'}
                data = {"msgtype": "text", "text": {"content": msg}}
                requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)
            except Exception:
                pass

        # 发送预测结果文件
        prediction_file = 'stock_predictions.json'
        if os.path.exists(prediction_file) and settings.paths.wechat_upload_url:
            send_file_to_wechat(
                prediction_file,
                settings.paths.wechat_webhook,
                settings.paths.wechat_upload_url,
            )

    print("\n模型文件:")
    for p in [settings.paths.model_path, settings.paths.swa_model_path, settings.paths.scaler_path, settings.paths.global_scaler_path]:
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024 / 1024
            print(f"  ✓ {p} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {p} (不存在)")


if __name__ == "__main__":
    main()
