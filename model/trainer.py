# -*- coding: utf-8 -*-
"""
模型训练模块
包含 FinanceScheduler、EMA、TopKCheckpoint、EarlyStopping 以及完整训练循环
从 TransformerStock.py 的 train_model() 提取
"""
import os
import glob
import copy
import math
import heapq
import logging
import time
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.preprocessing import RobustScaler
import joblib
from tqdm import tqdm

from config import get_settings, AppConfig
from data.types import FEATURES
from model.transformer import StockTransformer

logger = logging.getLogger(__name__)


# ==================== 训练辅助组件 ====================

class FinanceScheduler:
    """金融专用调度器：Warmup + Plateau + Cyclical微震荡"""

    def __init__(
        self,
        optimizer,
        warmup_steps=1000,
        base_lr=3e-5,
        min_lr=1e-6,
        plateau_patience=2,
        plateau_factor=0.5,
        cycle_amplitude=0.2,
        cycle_length=5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.global_step = 0

        self.best_loss = float('inf')
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.bad_epochs = 0

        self.current_lr = base_lr
        self.min_lr = min_lr

        self.cycle_amplitude = cycle_amplitude
        self.cycle_length = cycle_length
        self.epoch = 0

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step_batch(self):
        if self.global_step < self.warmup_steps:
            lr = self.base_lr * (0.1 + 0.9 * self.global_step / self.warmup_steps)
            self._set_lr(lr)
        self.global_step += 1

    def step_epoch(self, val_loss):
        self.epoch += 1

        if val_loss < self.best_loss - 1e-3:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.plateau_patience:
            self.current_lr = max(self.current_lr * self.plateau_factor, self.min_lr)
            self.bad_epochs = 0
            logger.info(f"[Scheduler] LR reduced to {self.current_lr:.2e}")

        cycle_phase = (self.epoch % self.cycle_length) / self.cycle_length
        cycle_factor = 1 + self.cycle_amplitude * math.sin(2 * math.pi * cycle_phase)
        lr = max(self.current_lr * cycle_factor, self.min_lr)
        self._set_lr(lr)

    def get_state(self) -> dict:
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'current_lr': self.current_lr,
            'bad_epochs': self.bad_epochs,
        }

    def load_state(self, state: dict):
        self.global_step = state.get('global_step', 0)
        self.epoch = state.get('epoch', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        self.current_lr = state.get('current_lr', self.base_lr)
        self.bad_epochs = state.get('bad_epochs', 0)
        self._set_lr(self.current_lr)


class EMA:
    """指数滑动平均"""

    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def get_model(self):
        return self.ema_model


class TopKCheckpoint:
    """Top-K 权重保存管理器"""

    def __init__(self, k=3, save_dir="checkpoints"):
        self.k = k
        self.heap = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model, val_loss, epoch):
        if np.isnan(val_loss) or np.isinf(val_loss):
            return
        path = os.path.join(self.save_dir, f"model_epoch{epoch}_loss{val_loss:.4f}.pth")
        if len(self.heap) < self.k:
            torch.save(model.state_dict(), path)
            heapq.heappush(self.heap, (-val_loss, path))
        else:
            worst_loss, worst_path = self.heap[0]
            if -val_loss > worst_loss:
                torch.save(model.state_dict(), path)
                heapq.heapreplace(self.heap, (-val_loss, path))
                if os.path.exists(worst_path):
                    os.remove(worst_path)

    def get_paths(self):
        return [p for _, p in self.heap]


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=3, min_delta=0.005, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ==================== 数据集 ====================

class MultiStockDataset(torch.utils.data.Dataset):
    """多股票时序数据集"""

    def __init__(self, combined_df, lookback_days):
        self.lookback_days = lookback_days
        self.data_array = combined_df[FEATURES].values.astype(np.float32)
        self.code_array = combined_df['Code'].values
        self.date_array = combined_df['Date'].values
        self.code_ranges = self._build_code_ranges()
        self.index_map = self._build_index_map()

    def _build_code_ranges(self):
        code_ranges = {}
        current_code = None
        start_idx = 0
        for i, code in enumerate(self.code_array):
            if code != current_code:
                if current_code is not None:
                    code_ranges[current_code] = (start_idx, i)
                current_code = code
                start_idx = i
        if current_code is not None:
            code_ranges[current_code] = (start_idx, len(self.code_array))
        return code_ranges

    def _build_index_map(self):
        index_map = []
        for code, (start, end) in self.code_ranges.items():
            for i in range(end - start - self.lookback_days):
                start_idx = start + i
                next_idx = start_idx + self.lookback_days

                next_close = self.data_array[next_idx, 3]
                last_close = self.data_array[start_idx + self.lookback_days - 1, 3]

                if np.isnan(next_close) or np.isnan(last_close) or last_close <= 0.01:
                    continue

                ret = (next_close - last_close) / last_close
                ret = np.clip(ret, -0.5, 0.5)

                if np.isnan(ret) or np.isinf(ret):
                    continue

                if ret > 0.05:
                    label = 3
                elif ret > 0:
                    label = 2
                elif ret > -0.05:
                    label = 1
                else:
                    label = 0

                index_map.append((code, start_idx, ret, label))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        code, start_idx, ret, label = self.index_map[idx]
        sequence = self.data_array[start_idx:start_idx + self.lookback_days]
        return (
            torch.FloatTensor(sequence),
            torch.tensor(label, dtype=torch.long),
            torch.tensor([ret], dtype=torch.float32),
        )


class WeightedMultiStockDataset(MultiStockDataset):
    """带时间权重和数据增强的数据集"""

    def __init__(self, combined_df, lookback_days, augment=True):
        super().__init__(combined_df, lookback_days)
        self.weights_array = combined_df['time_weight'].values.astype(np.float32)
        self.augment = augment

    def __getitem__(self, idx):
        code, start_idx, ret, label = self.index_map[idx]
        sequence = self.data_array[start_idx:start_idx + self.lookback_days]
        weight = self.weights_array[start_idx + self.lookback_days]

        if self.augment and np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            sequence = sequence * scale
            noise = np.random.normal(0, 0.02, sequence.shape)
            sequence = sequence + noise
            if np.random.rand() < 0.3:
                mask = np.random.rand(self.lookback_days) > 0.1
                sequence = sequence * mask[:, np.newaxis]

        return (
            torch.FloatTensor(sequence),
            torch.tensor(label, dtype=torch.long),
            torch.tensor([ret], dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


# ==================== 训练主函数 ====================

def train_model(settings: Optional[AppConfig] = None) -> None:
    """完整的模型训练流程"""
    if settings is None:
        settings = get_settings()

    mc = settings.model
    pc = settings.paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ========== 1. 数据加载 ==========
    if os.path.exists(pc.stock_data_file):
        combined_df = pd.read_feather(pc.stock_data_file)
    else:
        raise FileNotFoundError(
            f"数据文件不存在: {pc.stock_data_file}，请先运行数据下载脚本"
        )

    # ========== 2. 数据预处理 ==========
    logger.info("数据预处理：严格清洗...")
    if 'Code' not in combined_df.columns or 'Date' not in combined_df.columns:
        if 'Date' not in combined_df.columns and combined_df.index.name == 'Date':
            combined_df = combined_df.reset_index()
        combined_df = combined_df.drop_duplicates(subset=['Code', 'Date'])

    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in combined_df.columns:
            combined_df = combined_df[combined_df[col] > 0.01]

    if 'Volume' in combined_df.columns:
        combined_df = combined_df[combined_df['Volume'] > 0]

    combined_df = combined_df.sort_values(['Code', 'Date'])
    combined_df['daily_ret'] = combined_df.groupby('Code')['Close'].pct_change()
    combined_df = combined_df[
        (combined_df['daily_ret'].abs() <= 0.3) | (combined_df['daily_ret'].isna())
    ]

    # 截断极端特征值
    for col in FEATURES:
        if col in combined_df.columns:
            q01 = combined_df[col].quantile(0.01)
            q99 = combined_df[col].quantile(0.99)
            combined_df[col] = combined_df[col].clip(q01, q99)

    combined_df = combined_df.dropna(subset=FEATURES)
    combined_df = combined_df.reset_index(drop=True)
    logger.info(f"清洗后数据量: {len(combined_df)}")

    # ========== 3. 按股票划分 + 独立标准化 ==========
    logger.info("按股票划分训练集/验证集，独立标准化...")
    scalers = {}
    train_dfs = []
    val_dfs = []
    invalid_stocks = []

    for code, group in combined_df.groupby('Code'):
        group = group.sort_values('Date').reset_index(drop=True)

        if len(group) < mc.lookback_days + 10:
            invalid_stocks.append(code)
            continue

        price_std = group['Close'].pct_change().std()
        if price_std > 0.2:
            invalid_stocks.append(code)
            continue

        train_size = int(0.8 * len(group))
        if train_size <= mc.lookback_days:
            invalid_stocks.append(code)
            continue

        train_part = group.iloc[:train_size].copy()
        val_part = group.iloc[train_size:].copy()
        train_features = train_part[FEATURES].values

        if np.isnan(train_features).any() or np.isinf(train_features).any():
            invalid_stocks.append(code)
            continue

        try:
            scaler = RobustScaler(quantile_range=(5, 95))
            train_part[FEATURES] = scaler.fit_transform(train_part[FEATURES])
            val_part[FEATURES] = scaler.transform(val_part[FEATURES])

            if np.isnan(train_part[FEATURES].values).any():
                invalid_stocks.append(code)
                continue

            train_part[FEATURES] = train_part[FEATURES].clip(-5, 5)
            val_part[FEATURES] = val_part[FEATURES].clip(-5, 5)

            scalers[code] = scaler
            train_dfs.append(train_part)
            val_dfs.append(val_part)
        except Exception as e:
            invalid_stocks.append(code)
            continue

    if not train_dfs:
        raise ValueError("没有足够的有效数据用于训练！")

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    # 保存 scaler
    joblib.dump(scalers, pc.scaler_path)
    logger.info(f"已保存 {len(scalers)} 个股票的独立 scaler")

    # 保存全局 scaler
    all_train_data = train_df[FEATURES].values
    global_scaler = RobustScaler(quantile_range=(5, 95))
    global_scaler.fit(all_train_data)
    joblib.dump(global_scaler, pc.global_scaler_path)
    logger.info("全局 scaler 已保存")

    # ========== 4. 时间衰减权重 ==========
    if 'Date' in train_df.columns:
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        max_date = train_df['Date'].max()
        train_df['days_to_recent'] = (max_date - train_df['Date']).dt.days.clip(lower=0)
        train_df['time_weight'] = np.exp(-mc.time_decay_rate * train_df['days_to_recent'])
        train_df['time_weight'] = train_df['time_weight'].clip(0.01, 1.0).fillna(0.5)
    else:
        train_df['time_weight'] = 1.0

    if 'Date' in val_df.columns:
        val_df['Date'] = pd.to_datetime(val_df['Date'])
        max_date_val = val_df['Date'].max()
        val_df['days_to_recent'] = (max_date_val - val_df['Date']).dt.days.clip(lower=0)
        val_df['time_weight'] = np.exp(-mc.time_decay_rate * val_df['days_to_recent'])
        val_df['time_weight'] = val_df['time_weight'].clip(0.01, 1.0).fillna(0.5)
    else:
        val_df['time_weight'] = 1.0

    # ========== 5. 创建数据集 ==========
    train_dataset = WeightedMultiStockDataset(train_df, mc.lookback_days)
    val_dataset = WeightedMultiStockDataset(val_df, mc.lookback_days, augment=False)

    logger.info(f"训练集: {len(train_dataset)} 序列 | 验证集: {len(val_dataset)} 序列")

    train_loader = DataLoader(
        train_dataset, batch_size=mc.batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=mc.batch_size, shuffle=False,
        num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True,
    )

    # ========== 6. 模型与优化器 ==========
    model = StockTransformer(
        input_dim=len(FEATURES),
        lookback_days=mc.lookback_days,
        num_heads=mc.num_heads,
        dim_feedforward=mc.dim_feedforward,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=mc.learning_rate,
        weight_decay=mc.weight_decay, fused=True,
    )
    grad_scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    scheduler = FinanceScheduler(
        optimizer,
        warmup_steps=len(train_loader) * mc.warmup_epochs // mc.accumulation_steps,
        base_lr=mc.learning_rate,
        min_lr=1e-6,
        plateau_patience=mc.plateau_patience,
        plateau_factor=mc.plateau_factor,
        cycle_amplitude=mc.cycle_amplitude,
        cycle_length=mc.cycle_length,
    )

    ema = EMA(model, decay=mc.ema_decay)
    swa_model = AveragedModel(model)
    swa_start = int(mc.epochs * 0.7)
    topk = TopKCheckpoint(k=mc.topk_save_count, save_dir=pc.topk_checkpoint_dir)
    early_stopping = EarlyStopping(patience=3, min_delta=0.005)

    # 加载检查点
    start_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint = max(glob.glob("model_epoch_*.pth"), key=os.path.getctime, default=None)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'scheduler_state' in checkpoint:
            scheduler.load_state(checkpoint['scheduler_state'])
        best_val_loss = checkpoint.get('loss', float('inf'))
        logger.warning(f"恢复检查点: {latest_checkpoint}, 起始周期: {start_epoch}")

    # ========== 7. 训练循环 ==========
    for epoch in range(start_epoch, mc.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{mc.epochs}", leave=False)

        for i, (sequences, labels, rets, weights) in enumerate(progress_bar):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            rets = rets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            # 数据检查
            if (torch.isnan(sequences).any() or torch.isinf(sequences).any()
                    or torch.isnan(rets).any() or torch.isinf(rets).any()
                    or torch.isnan(weights).any() or torch.isinf(weights).any()):
                continue

            rets = torch.clamp(rets, -0.5, 0.5)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits, ret_pred = model(sequences)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                if torch.isnan(ret_pred).any() or torch.isinf(ret_pred).any():
                    continue

                loss_cls = nn.CrossEntropyLoss(reduction='none', label_smoothing=mc.label_smoothing)(logits, labels)
                loss_cls = (loss_cls * weights).mean()

                loss_ret = nn.SmoothL1Loss(reduction='none')(ret_pred.squeeze(), rets.squeeze())
                loss_ret = (loss_ret * weights).mean()

                loss = loss_cls + 0.5 * loss_ret

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss = loss / mc.accumulation_steps
            grad_scaler.scale(loss).backward()

            if (i + 1) % mc.accumulation_steps == 0:
                grad_scaler.unscale_(optimizer)
                has_nan = any(
                    param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    for param in model.parameters()
                )
                if has_nan:
                    logger.warning(f"Batch {i}: 发现 NaN/Inf 梯度，丢弃更新")
                    optimizer.zero_grad()
                    grad_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), mc.grad_clip_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)

            scheduler.step_batch()
            current_batch_loss = loss.item() * mc.accumulation_steps
            total_loss += current_batch_loss
            progress_bar.set_postfix(loss=f"{current_batch_loss:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, lab, val_rets, val_weights in tqdm(val_loader, desc=f"Validating", leave=False):
                seq = seq.to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True)
                val_rets = val_rets.to(device, non_blocking=True)
                val_weights = val_weights.to(device, non_blocking=True)

                if torch.isnan(seq).any() or torch.isinf(seq).any():
                    continue

                with autocast(device_type='cuda', dtype=torch.float16):
                    logits, ret_pred = model(seq)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        continue

                    loss_cls = nn.CrossEntropyLoss(reduction='none')(logits, lab)
                    loss_cls = (loss_cls * val_weights).mean()
                    loss_ret = nn.SmoothL1Loss(reduction='none')(ret_pred.squeeze(), val_rets.squeeze())
                    loss_ret = (loss_ret * val_weights).mean()
                    loss = loss_cls + 0.5 * loss_ret
                    val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step_epoch(avg_val_loss)

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        current_lr = optimizer.param_groups[0]['lr']
        logger.warning(f"Epoch {epoch + 1}, Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}")

        # 保存检查点
        checkpoint_path = f"model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': grad_scaler.state_dict(),
            'loss': avg_val_loss,
            'scheduler_state': scheduler.get_state(),
        }, checkpoint_path)

        # 保存最佳模型（EMA）
        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ema.get_model().state_dict(), pc.model_path)
            logger.warning(f"更优模型(EMA)已保存, 损失: {best_val_loss:.4f}")

        topk.save(ema.get_model(), avg_val_loss, epoch)

        torch.cuda.empty_cache()

        if early_stopping(avg_val_loss):
            logger.warning("早停触发！")
            break

        if np.isnan(avg_val_loss) or np.isinf(avg_val_loss):
            logger.error(f"验证损失异常，停止训练")
            break

    # ========== SWA 收尾 ==========
    if epoch + 1 >= swa_start:
        logger.info("SWA 收尾...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), pc.swa_model_path)
        logger.warning(f"SWA 模型已保存: {pc.swa_model_path}")

    logger.warning(f"EMA 最佳模型: {pc.model_path}")
    logger.warning(f"Top-K 集成模型: {pc.topk_checkpoint_dir}/")
