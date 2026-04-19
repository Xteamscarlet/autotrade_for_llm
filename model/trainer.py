# -*- coding: utf-8 -*-
"""
妯″瀷璁粌妯″潡 V2 锟?绐佺牬 val_loss 鐡堕
鏍稿績鏀瑰姩锟?
1. 鏍囩浣撶郴锟?鍒嗙被鈫掑姩鎬侀槇锟?绫诲埆鍔犳潈锛堣В鍐崇被鍒笉骞宠　锟?
2. 鎹熷け鍑芥暟锛欶ocal Loss 鏇夸唬 CrossEntropy锛堣仛鐒﹂毦鍒嗘牱鏈級+ 鍥炲綊鎹熷け鑷€傚簲鏉冮噸
3. 鏁版嵁澧炲己锛氶檷浣庡己搴︼紝閲戣瀺鏁版嵁涓嶈兘闅忔剰缂╂斁
4. 鐗瑰緛宸ョ▼锛氬鍔犳敹鐩婄巼鐗瑰緛锛屾彁楂樹俊鍙峰瘑锟?
5. 璁粌绛栫暐锛氭洿锟?epochs + cosine annealing + 鏇撮珮鍒濆瀛︿範锟?
6. 鏍囩骞虫粦锟?.1锟?.05锛堥噾铻嶆暟鎹凡楂樺櫔澹帮紝涓嶉渶瑕佽繃澶氬钩婊戯級
7. 鏍囧噯鍖栵細quantile_range (5,95)锟?10,90)锛屼繚鐣欐洿澶氬尯鍒嗗害
"""
import os
import glob
import copy
import math
import heapq
import logging
import time
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.preprocessing import RobustScaler
import joblib
from tqdm import tqdm

from config import get_settings, AppConfig
from data.types import FEATURES
from model.transformer import StockTransformer

logger = logging.getLogger(__name__)


# ==================== Focal Loss ====================

class FocalLoss(nn.Module):
    """Focal Loss 锟?鑱氱劍闅惧垎鏍锋湰锛岃В鍐崇被鍒笉骞宠　
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma > 0 鍑忓皯鏄撳垎鏍锋湰鐨勬崯澶辫础鐚紝鑱氱劍闅惧垎鏍锋湰
    alpha 鐢ㄤ簬绫诲埆鍔犳潈
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # alpha: 绫诲埆鏉冮噸 [C]
        self.register_buffer('alpha', None)
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer('alpha', alpha)

    def forward(self, logits, targets, sample_weights=None):
        """
        Args:
            logits: [B, C]
            targets: [B]
            sample_weights: [B] 鏃堕棿琛板噺鏉冮噸
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)  # p_t = softmax姒傜巼涓纭被鍒殑姒傜巼
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 绫诲埆鍔犳潈
        if self.alpha is not None and self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 鏍锋湰鏉冮噸锛堟椂闂磋“鍑忥級
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==================== 璁粌杈呭姪缁勪欢 ====================

class CosineAnnealingWarmRestarts:
    """浣欏鸡閫€锟?+ 鐑噸鍚皟搴﹀櫒"""
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
        self.T_cur = 0
        self.T_i = T_0

    def step(self):
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = self.eta_min + (base_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            pg['lr'] = lr
        self.current_epoch += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class FinanceScheduler:
    """閲戣瀺涓撶敤璋冨害鍣細Warmup + Plateau + Cyclical寰渿锟?"""

    def __init__(self, optimizer, warmup_steps=1000, base_lr=3e-5, min_lr=1e-6,
                 plateau_patience=2, plateau_factor=0.5, cycle_amplitude=0.2, cycle_length=5):
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
        return {'global_step': self.global_step, 'epoch': self.epoch,
                'best_loss': self.best_loss, 'current_lr': self.current_lr, 'bad_epochs': self.bad_epochs}

    def load_state(self, state: dict):
        self.global_step = state.get('global_step', 0)
        self.epoch = state.get('epoch', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        self.current_lr = state.get('current_lr', self.base_lr)
        self.bad_epochs = state.get('bad_epochs', 0)
        self._set_lr(self.current_lr)


class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        # Keep BN/running-stat buffers aligned with the online model.
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def get_model(self):
        return self.ema_model


class TopKCheckpoint:
    def __init__(self, k=3, save_dir="checkpoints", file_prefix="topk_rawclose"):
        self.k = k
        self.heap = []
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        os.makedirs(save_dir, exist_ok=True)
        self._clear_stale_files()

    def _clear_stale_files(self):
        pattern = os.path.join(self.save_dir, f"{self.file_prefix}_*.pth")
        for stale_path in glob.glob(pattern):
            try:
                os.remove(stale_path)
            except OSError:
                logger.warning(f"鏃犳硶鍒犻櫎鏃?Top-K 鏂囦欢: {stale_path}")

    def save(self, model, val_loss, epoch):
        if np.isnan(val_loss) or np.isinf(val_loss):
            return
        path = os.path.join(self.save_dir, f"{self.file_prefix}_epoch{epoch + 1}_loss{val_loss:.4f}.pth")
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
    def __init__(self, patience=5, min_delta=0.002, mode='min'):
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


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    ret_loss_weight: float,
    amp_dtype: Optional[torch.dtype],
    ret_scale: float = 0.05,
) -> Tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    val_cls_loss = 0.0
    val_ret_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for seq, lab, val_rets, val_weights in tqdm(data_loader, desc="Validating", leave=False):
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            val_rets = val_rets.to(device, non_blocking=True)
            val_weights = val_weights.to(device, non_blocking=True)

            if torch.isnan(seq).any() or torch.isinf(seq).any():
                continue

            autocast_enabled = amp_dtype is not None
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                logits, ret_pred = model(seq)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                if torch.isnan(ret_pred).any() or torch.isinf(ret_pred).any():
                    continue

                loss_cls = loss_fn(logits, lab, sample_weights=val_weights).mean()
                # ★ 修复5: rets 标准化到 O(1) 量级再算回归 loss
                val_rets_norm = val_rets / ret_scale
                loss_ret = nn.SmoothL1Loss(reduction='none')(ret_pred.squeeze(), val_rets_norm.squeeze())
                loss_ret = (loss_ret * val_weights).mean()
                loss = loss_cls + ret_loss_weight * loss_ret

            val_loss += loss.item()
            val_cls_loss += loss_cls.item()
            val_ret_loss += loss_ret.item()
            valid_batches += 1

    if valid_batches == 0:
        return float('inf'), float('inf'), float('inf')

    return (
        val_loss / valid_batches,
        val_cls_loss / valid_batches,
        val_ret_loss / valid_batches,
    )


# ==================== 鏁版嵁锟?V2 ====================

class MultiStockDatasetV2(torch.utils.data.Dataset):
    def __init__(self, combined_df, lookback_days, label_mode='dynamic', thresholds=None):
        """
        Args:
            thresholds: 可选 dict {'q10','q25','q75','q90'}。
                        ★ 修复6: 传入时跳过分位数自计算，val 集复用 train 分位数，
                        避免 val 标签与 train 标签定义不一致导致 val_loss 失真。
        """
        self.lookback_days = lookback_days
        self.label_mode = label_mode
        self.data_array = combined_df[FEATURES].values.astype(np.float32)
        self.close_array = combined_df['Close_raw'].values.astype(np.float32) \
            if 'Close_raw' in combined_df.columns else combined_df['Close'].values.astype(np.float32)
        self.code_array = combined_df['Code'].values
        self.date_array = combined_df['Date'].values

        if thresholds is not None:
            self.q10 = float(thresholds['q10'])
            self.q25 = float(thresholds['q25'])
            self.q75 = float(thresholds['q75'])
            self.q90 = float(thresholds['q90'])
        else:
            all_rets = []
            for i in range(len(self.data_array)):
                if i >= lookback_days:
                    next_close = self.close_array[i]
                    last_close = self.close_array[i - 1]
                    if not (np.isnan(next_close) or np.isnan(last_close)) and last_close > 0.01:
                        ret = (next_close - last_close) / last_close
                        ret = np.clip(ret, -0.5, 0.5)
                        if not (np.isnan(ret) or np.isinf(ret)):
                            all_rets.append(ret)

            if all_rets:
                ret_arr = np.array(all_rets)
                self.q25 = np.percentile(ret_arr, 25)
                self.q75 = np.percentile(ret_arr, 75)
                self.q10 = np.percentile(ret_arr, 10)
                self.q90 = np.percentile(ret_arr, 90)
            else:
                self.q25, self.q75 = -0.02, 0.02
                self.q10, self.q90 = -0.04, 0.04

        self.code_ranges = self._build_code_ranges()
        self.index_map = self._build_index_map()

        # ★ 修复6: 类别权重 (Focal Loss alpha)
        label_counts = np.zeros(4)
        for _, _, _, label in self.index_map:
            label_counts[label] += 1
        total = label_counts.sum()
        if total > 0:
            self.class_weights = np.clip(total / (4 * label_counts + 1), 0.5, 3.0)
            self.class_weights = self.class_weights / self.class_weights.sum() * 4
        else:
            self.class_weights = np.ones(4)

        logger.info(f"label_counts: {label_counts}, class_weights: {self.class_weights}")
        logger.info(f"thresholds: q10={self.q10:.4f} q25={self.q25:.4f} q75={self.q75:.4f} q90={self.q90:.4f}")

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

                next_close = self.close_array[next_idx]
                last_close = self.close_array[start_idx + self.lookback_days - 1]

                if np.isnan(next_close) or np.isnan(last_close) or last_close <= 0.01:
                    continue

                ret = (next_close - last_close) / last_close
                ret = np.clip(ret, -0.5, 0.5)

                if np.isnan(ret) or np.isinf(ret):
                    continue

                # 锟?鍔ㄦ€侀槇鍊兼爣绛撅紙鏇夸唬鍥哄畾 5%锟?
                if self.label_mode == 'dynamic':
                    if ret > self.q90:
                        label = 3  # 澶ф定
                    elif ret > self.q75:
                        label = 2  # 灏忔定
                    elif ret > self.q25:
                        label = 1  # 灏忚穼
                    else:
                        label = 0  # 澶ц穼
                else:
                    # 鍘熷鍥哄畾闃堬拷?
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


class WeightedMultiStockDatasetV2(MultiStockDatasetV2):
    def __init__(self, combined_df, lookback_days, augment=True, label_mode='dynamic', thresholds=None):
        super().__init__(combined_df, lookback_days, label_mode=label_mode, thresholds=thresholds)
        self.weights_array = combined_df['time_weight'].values.astype(np.float32)
        self.augment = augment

    def __getitem__(self, idx):
        code, start_idx, ret, label = self.index_map[idx]
        sequence = self.data_array[start_idx:start_idx + self.lookback_days]
        weight = self.weights_array[start_idx + self.lookback_days]

        # 锟?娓╁拰鏁版嵁澧炲己 (★ 修复1: 加强增强抑制过拟合)
        if self.augment and np.random.rand() < 0.6:  # 0.4 → 0.6
            scale = np.random.uniform(0.88, 1.12)  # 0.92~1.08 → 0.88~1.12
            sequence = sequence * scale
            noise = np.random.normal(0, 0.02, sequence.shape)  # 0.015 → 0.02
            sequence = sequence + noise
            if np.random.rand() < 0.25:  # 0.15 → 0.25
                mask = np.random.rand(self.lookback_days) > 0.05  # 0.1锟?.05
                sequence = sequence * mask[:, np.newaxis]

        return (
            torch.FloatTensor(sequence),
            torch.tensor(label, dtype=torch.long),
            torch.tensor([ret], dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


# ==================== 璁粌涓诲嚱锟?V2 ====================

def train_model(settings: Optional[AppConfig] = None) -> None:
    if settings is None:
        settings = get_settings()

    mc = settings.model
    pc = settings.paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"浣跨敤璁惧: {device}")

    # ========== 1. 鏁版嵁鍔犺浇 ==========
    if os.path.exists(pc.stock_data_file):
        combined_df = pd.read_feather(pc.stock_data_file)
    else:
        raise FileNotFoundError(f"鏁版嵁鏂囦欢涓嶅瓨锟? {pc.stock_data_file}")

    # ========== 2. 鏁版嵁棰勫锟?==========
    logger.info("鏁版嵁棰勫鐞嗭細涓ユ牸娓呮礂...")
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
    combined_df['Close_raw'] = combined_df['Close']
    combined_df['daily_ret'] = combined_df.groupby('Code')['Close'].pct_change()
    combined_df = combined_df[
        (combined_df['daily_ret'].abs() <= 0.3) | (combined_df['daily_ret'].isna())
    ]

    # 锟?鏂板锛氭敹鐩婄巼鐗瑰緛锛堟彁楂樹俊鍙峰瘑搴︼級
    for lag in [1, 3, 5, 10]:
        col_name = f'ret_{lag}'
        combined_df[col_name] = combined_df.groupby('Code')['Close'].pct_change(lag)
        # 濡傛灉 FEATURES 涓嶅寘鍚繖浜涘垪锛屽悗闈㈡爣鍑嗗寲鏃堕渶瑕佸锟?
        if col_name not in FEATURES:
            FEATURES.append(col_name)
    
    # 閲嶆柊鎴柇鏋佺鐗瑰緛鍊硷紙鍖呭惈鏂板姞鐨勬敹鐩婄巼鐗瑰緛锟?
    for col in FEATURES:
        if col in combined_df.columns:
            q01 = combined_df[col].quantile(0.02)  # 0.01 鈫?0.02
            q99 = combined_df[col].quantile(0.98)  # 0.99 鈫?0.98
            combined_df[col] = combined_df[col].clip(q01, q99)

    combined_df = combined_df.dropna(subset=FEATURES)
    combined_df = combined_df.reset_index(drop=True)
    logger.info(f"娓呮礂鍚庢暟鎹噺: {len(combined_df)}")

    # ========== 3. 鎸夎偂绁ㄥ垝锟?+ 鐙珛鏍囧噯锟?==========
    logger.info("鎸夎偂绁ㄥ垝鍒嗚缁冮泦/楠岃瘉闆嗭紝鐙珛鏍囧噯锟?..")
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
            # 锟?鏍囧噯鍖栬寖鍥达細(5,95) 锟?(10,90)锛屼繚鐣欐洿澶氬尯鍒嗗害
            scaler = RobustScaler(quantile_range=(10, 90))
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
        raise ValueError("娌℃湁瓒冲鐨勬湁鏁堟暟鎹敤浜庤缁冿紒")

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    joblib.dump(scalers, pc.scaler_path)
    logger.info(f"宸蹭繚锟?{len(scalers)} 涓偂绁ㄧ殑鐙珛 scaler")

    all_train_data = train_df[FEATURES].values
    global_scaler = RobustScaler(quantile_range=(10, 90))
    global_scaler.fit(all_train_data)
    joblib.dump(global_scaler, pc.global_scaler_path)
    logger.info("全局 scaler 已保存")

    # ========== 4. 鏃堕棿琛板噺鏉冮噸 ==========
    if 'Date' in train_df.columns:
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        max_date = train_df['Date'].max()
        train_df['days_to_recent'] = (max_date - train_df['Date']).dt.days.clip(lower=0)
        # 锟?鍑忓皬琛板噺鐜囷細0.001 锟?0.0005锛岃鏇存棭鐨勬暟鎹篃鏈夋洿澶氭潈锟?
        train_df['time_weight'] = np.exp(-mc.time_decay_rate * train_df['days_to_recent'])
        train_df['time_weight'] = train_df['time_weight'].clip(0.1, 1.0).fillna(0.5)
    else:
        train_df['time_weight'] = 1.0

    if 'Date' in val_df.columns:
        val_df['Date'] = pd.to_datetime(val_df['Date'])
        max_date_val = val_df['Date'].max()
        val_df['days_to_recent'] = (max_date_val - val_df['Date']).dt.days.clip(lower=0)
        val_df['time_weight'] = np.exp(-mc.time_decay_rate * val_df['days_to_recent'])
        val_df['time_weight'] = val_df['time_weight'].clip(0.1, 1.0).fillna(0.5)
    else:
        val_df['time_weight'] = 1.0

    # ========== 5. 鍒涘缓鏁版嵁闆嗭紙V2 鍔ㄦ€佹爣绛撅級 ==========
    # ★ 修复6: train 集自算分位数；val 集复用 train 阈值，保证标签定义一致
    train_dataset = WeightedMultiStockDatasetV2(
        train_df, mc.lookback_days, augment=True, label_mode='dynamic',
    )
    train_thresholds = {
        'q10': train_dataset.q10, 'q25': train_dataset.q25,
        'q75': train_dataset.q75, 'q90': train_dataset.q90,
    }
    # ★ 修复6: 把训练集分位数阈值持久化，供回测/预测端展示真实涨跌幅区间
    joblib.dump(train_thresholds, pc.label_thresholds_path)
    logger.info(f"训练集分位数阈值已保存: {pc.label_thresholds_path} -> {train_thresholds}")

    val_dataset = WeightedMultiStockDatasetV2(
        val_df, mc.lookback_days, augment=False, label_mode='dynamic',
        thresholds=train_thresholds,
    )

    # 锟?鑾峰彇绫诲埆鏉冮噸鐢ㄤ簬 Focal Loss
    class_weights = torch.FloatTensor(train_dataset.class_weights).to(device)
    logger.info(f"绫诲埆鏉冮噸: {class_weights.cpu().numpy()}")

    logger.info(f"璁粌锟? {len(train_dataset)} 搴忓垪 | 楠岃瘉锟? {len(val_dataset)} 搴忓垪")

    train_loader = DataLoader(
        train_dataset, batch_size=mc.batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=mc.batch_size, shuffle=False,
        num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True,
    )

    # ========== 6. 妯″瀷涓庝紭鍖栧櫒 ==========
    actual_lr = mc.learning_rate
    amp_dtype = torch.float16 if device.type == 'cuda' else None

    model = StockTransformer(
        input_dim=len(FEATURES),  # 锟?鍖呭惈鏂板鐨勬敹鐩婄巼鐗瑰緛
        lookback_days=mc.lookback_days,
        num_heads=mc.num_heads,
        dim_feedforward=mc.dim_feedforward,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=actual_lr,
        weight_decay=mc.weight_decay, fused=True,
    )
    grad_scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ★ 修复3: 用 PyTorch 自带的 SequentialLR 替代 FinanceScheduler+CosineAnnealingWarmRestarts
    # 避免 base_lr 在 warmup 阶段被污染导致 cosine 从错误起点开始
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
    warmup_epochs = max(0, min(mc.warmup_epochs, mc.epochs - 1))
    cosine_epochs = max(1, mc.epochs - warmup_epochs)
    if warmup_epochs > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, min(mc.cycle_length, cosine_epochs)), T_mult=1, eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, min(mc.cycle_length, cosine_epochs)), T_mult=1, eta_min=1e-6,
        )

    ema = EMA(model, decay=mc.ema_decay)
    swa_model = AveragedModel(model)
    swa_start = int(mc.epochs * mc.swa_start_ratio)  # ★ 修复1: 使用 config 中的 swa_start_ratio
    topk = TopKCheckpoint(k=mc.topk_save_count, save_dir=pc.topk_checkpoint_dir)
    # ★ 修复1: patience 从 config 读取（默认 6），让 30 epoch 训练有足够机会收敛
    early_stopping = EarlyStopping(
        patience=mc.early_stop_patience, min_delta=mc.early_stop_min_delta,
    )

    # 锟?Focal Loss锛堟浛锟?CrossEntropyLoss锟?
    focal_loss_fn = FocalLoss(
        alpha=class_weights,
        gamma=2.0,
        label_smoothing=0.05,  # 锟?0.1锟?.05
        reduction='none',
    ).to(device)

    # 鍔犺浇妫€鏌ョ偣
    focal_loss_fn.label_smoothing = mc.label_smoothing

    start_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint = max(glob.glob("model_epoch_*.pth"), key=os.path.getctime, default=None)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        checkpoint_features = checkpoint.get('feature_names') if isinstance(checkpoint, dict) else None
        checkpoint_target_source = checkpoint.get('target_source') if isinstance(checkpoint, dict) else None
        can_resume = (
            isinstance(checkpoint, dict)
            and 'model_state_dict' in checkpoint
            and checkpoint_features == list(FEATURES)
            and checkpoint_target_source == 'raw_close'
        )
        if can_resume:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # ★ 修复1+3: resume 后强制覆盖 lr，避免 ckpt 中陈旧 lr 污染本次训练
            for pg in optimizer.param_groups:
                pg['lr'] = actual_lr
            if 'scaler_state_dict' in checkpoint:
                grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('loss', float('inf'))
            logger.warning(f"鎭㈠妫€鏌ョ偣: {latest_checkpoint}, 璧峰鍛ㄦ湡: {start_epoch}, lr 已重置为: {actual_lr}")
        else:
            logger.warning(f"Skip incompatible checkpoint: {latest_checkpoint}")

    # ========== 7. 璁粌寰幆 ==========
    # 锟?鍥炲綊鎹熷け鑷€傚簲鏉冮噸锛氬垵锟?.5锛岄殢璁粌杩涘睍閫愭笎澧炲姞锟?.0
    ret_loss_weight_initial = 0.1
    ret_loss_weight_final = 0.3

    for epoch in range(start_epoch, mc.epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_ret_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{mc.epochs}", leave=False)

        for i, (sequences, labels, rets, weights) in enumerate(progress_bar):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            rets = rets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            if (torch.isnan(sequences).any() or torch.isinf(sequences).any()
                    or torch.isnan(rets).any() or torch.isinf(rets).any()
                    or torch.isnan(weights).any() or torch.isinf(weights).any()):
                continue

            rets = torch.clamp(rets, -0.5, 0.5)

            autocast_enabled = amp_dtype is not None
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                logits, ret_pred = model(sequences)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                if torch.isnan(ret_pred).any() or torch.isinf(ret_pred).any():
                    continue

                # 锟?Focal Loss 鏇夸唬 CrossEntropy
                loss_cls = focal_loss_fn(logits, labels, sample_weights=weights)
                loss_cls = loss_cls.mean()

                # ★ 修复5: rets 标准化到 O(1) 量级，让回归头有真实梯度
                rets_norm = rets / mc.ret_target_scale
                loss_ret = nn.SmoothL1Loss(reduction='none')(ret_pred.squeeze(), rets_norm.squeeze())
                loss_ret = (loss_ret * weights).mean()

                # 锟?鍥炲綊鎹熷け鏉冮噸鑷€傚簲澧炲姞
                progress = min(1.0, epoch / max(mc.epochs - 1, 1))
                ret_loss_weight = ret_loss_weight_initial + (ret_loss_weight_final - ret_loss_weight_initial) * progress

                loss = loss_cls + ret_loss_weight * loss_ret

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss = loss / mc.accumulation_steps
            grad_scaler.scale(loss).backward()

            should_step = (i + 1) % mc.accumulation_steps == 0 or (i + 1) == len(train_loader)
            if should_step:
                grad_scaler.unscale_(optimizer)
                has_nan = any(
                    param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    for param in model.parameters()
                )
                if has_nan:
                    logger.warning(f"Batch {i}: detected NaN/Inf gradients, skipping optimizer step")
                    optimizer.zero_grad()
                    grad_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), mc.grad_clip_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)

            # ★ 修复3: LR 调度统一由 epoch 级 scheduler.step() 管理，此处不做 batch 级调度

            current_batch_loss = loss.item() * mc.accumulation_steps
            total_loss += current_batch_loss
            total_cls_loss += loss_cls.item()
            total_ret_loss += loss_ret.item()
            progress_bar.set_postfix(
                loss=f"{current_batch_loss:.4f}",
                cls=f"{loss_cls.item():.4f}",
                ret=f"{loss_ret.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        # ★ 修复3: 每个 epoch 结束统一调度一次
        scheduler.step()

        # ========== 楠岃瘉闃舵 ==========
        model.eval()
        val_loss = 0
        val_cls_loss = 0
        val_ret_loss = 0
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

                    loss_cls = focal_loss_fn(logits, lab, sample_weights=val_weights)
                    loss_cls = loss_cls.mean()
                    # ★ 修复5: rets 标准化保持训练/验证一致
                    val_rets_norm = val_rets / mc.ret_target_scale
                    loss_ret = nn.SmoothL1Loss(reduction='none')(ret_pred.squeeze(), val_rets_norm.squeeze())
                    loss_ret = (loss_ret * val_weights).mean()
                    loss = loss_cls + ret_loss_weight * loss_ret
                    val_loss += loss.item()
                    val_cls_loss += loss_cls.item()
                    val_ret_loss += loss_ret.item()

        avg_train_loss = total_loss / len(train_loader)

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        avg_val_loss, avg_val_cls, avg_val_ret = evaluate_model(
            ema.get_model(),
            val_loader,
            device,
            focal_loss_fn,
            ret_loss_weight,
            amp_dtype,
            ret_scale=mc.ret_target_scale,  # ★ 修复5
        )

        current_lr = optimizer.param_groups[0]['lr']
        logger.warning(
            f"Epoch {epoch + 1}, "
            f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f} "
            f"(cls: {avg_val_cls:.4f}, ret: {avg_val_ret:.4f}), "
            f"LR: {current_lr:.2e}, ret_w: {ret_loss_weight:.2f}"
        )

        # 淇濆瓨妫€鏌ョ偣
        checkpoint_path = f"model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': grad_scaler.state_dict(),
            'loss': avg_val_loss,
            'feature_names': list(FEATURES),
            'target_source': 'raw_close',
        }, checkpoint_path)

        # 淇濆瓨鏈€浣虫ā鍨嬶紙EMA锟?
        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ema.get_model().state_dict(), pc.model_path)
            logger.warning(f"Saved improved EMA model with val_loss: {best_val_loss:.4f}")

        # ★ 修复4: TopK 保存"在线模型"权重，与 EMA/SWA 形成真正的 ensemble 多样性
        # 之前保存的是 ema.get_model() 导致 TopK 与 EMA 几乎完全相同
        topk.save(model, avg_val_loss, epoch)

        torch.cuda.empty_cache()

        if early_stopping(avg_val_loss):
            logger.warning("Early stopping triggered")
            break

        if np.isnan(avg_val_loss) or np.isinf(avg_val_loss):
            logger.error("Validation loss became NaN/Inf, stopping training")
            break

    # ========== SWA 鏀跺熬 ==========
    # ★ 修复1: 只有 SWA 实际累积了多个 epoch 才有意义；用 n_averaged 判断
    swa_n = int(swa_model.n_averaged.item()) if hasattr(swa_model, 'n_averaged') else 0
    if swa_n >= 3:
        logger.info(f"SWA 鏀跺熬... (累积 {swa_n} 个 epoch)")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), pc.swa_model_path)
        logger.warning(f"SWA model saved: {pc.swa_model_path}")
    else:
        logger.warning(f"SWA 累积仅 {swa_n} 个 epoch，跳过保存（避免与 EMA 重复）")

    logger.warning(f"EMA best model saved: {pc.model_path}")
    logger.warning(f"Top-K ensemble checkpoints: {pc.topk_checkpoint_dir}/")
    logger.warning(f"last_val_loss: {avg_val_loss:.4f}, best_val_loss: {best_val_loss:.4f}")
