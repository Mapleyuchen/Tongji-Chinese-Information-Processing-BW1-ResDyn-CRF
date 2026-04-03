import os
import sys
import time
import math
import random
import argparse
from datetime import datetime
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data_processor import (
    build_dataset,
    get_dataloader,
    load_icwb2_dataset,
    TAG2IDX,
    IDX2TAG,
    bmes_to_segmentation,
    compute_f1,
    CharVocab,
    CWSDataset,
    PAD_IDX,
    collate_fn,
)
from model import BiLSTM_DynamicCRF

def _auto_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "icwb2-data"),
        os.path.join(os.path.expanduser("~"), "Word Segmentation", "icwb2-data"),
        r"d:\Word Segmentation\icwb2-data",
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return candidates[0]  # fallback，使用第一个


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM + Dynamic CRF中文分词大作业")
    parser.add_argument("--data_dir", type=str, default=_auto_data_dir(),
                        help="icwb2-data根目录")
    parser.add_argument("--dataset", type=str, default="msr",
                        choices=["msr", "pku"], help="数据集名称")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--embedding_dim", type=int, default=128, help="字符嵌入维度")
    parser.add_argument("--lstm_hidden", type=int, default=256, help="LSTM单向隐状态维度")
    parser.add_argument("--lstm_layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout比例")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减（L2正则化）")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument("--min_freq", type=int, default=1, help="字符最小出现频次")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="每多少步打印一次训练日志")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="每多少轮验证一次")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader并行数")
    parser.add_argument("--resume", type=str, default=None,
                        help="从checkpoint恢复训练（路径）")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备（auto/cpu/cuda）")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = None) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda" or (device_str is None and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def get_word_spans(tags: List[str]) -> List[Tuple[int, int]]:
    spans = []
    start = 0
    for i, tag in enumerate(tags):
        if tag == 'B':
            start = i
        elif tag == 'S':
            spans.append((i, i))
            start = i + 1
        elif tag == 'E':
            spans.append((start, i))
            start = i + 1
        # M状态无需特殊操作
    return spans


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    
    correct_spans = 0           # 预测正确的词数
    total_pred_spans = 0        # 模型预测出的总词数
    total_gold_spans = 0        # 真实的总词数

    with torch.no_grad():
        for batch in val_loader:
            input_ids, tag_ids, attention_mask, lengths = batch
            input_ids = input_ids.to(device)
            tag_ids = tag_ids.to(device)
            attention_mask = attention_mask.to(device)
            lengths = lengths.to(device)

            _, best_paths = model.decode(input_ids, attention_mask, lengths)

            # 逐样本计算词边界F1
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = lengths[i].item()
                # 获取有效标签
                pred_tags = [IDX2TAG[idx.item()] for idx in best_paths[i, :length]]
                gold_tags = [IDX2TAG[idx.item()] for idx in tag_ids[i, :length]]

                # 转换为区间(start_idx, end_idx)集合
                pred_spans = set(get_word_spans(pred_tags))
                gold_spans = set(get_word_spans(gold_tags))

                # 统计匹配数
                correct_spans += len(pred_spans & gold_spans)
                total_pred_spans += len(pred_spans)
                total_gold_spans += len(gold_spans)

    # 计算全局Precision, Recall, F1
    precision = correct_spans / total_pred_spans if total_pred_spans > 0 else 0.0
    recall = correct_spans / total_gold_spans if total_gold_spans > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    model.train()
    return precision, recall, f1

def train(args):
    #------------1. 环境设置------------
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[Train] 使用设备: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    #------------2. 保存目录------------
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.save_dir, f"train_{timestamp}.log")

    def log(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 70)
    log(f"[Train] 开始训练 | 时间戳: {timestamp}")
    log(f"[Train] 数据集: {args.dataset} | epochs: {args.epochs} | batch_size: {args.batch_size}")
    log("=" * 70)

    #------------3. 数据加载------------
    log("[Train] 正在加载和预处理数据...")
    train_dataset, val_dataset, vocab = build_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        min_freq=args.min_freq,
        train_ratio=args.train_ratio,
        vocab_save_path=os.path.join(args.save_dir, "vocab.pkl"),
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    log(f"[Train] 训练集: {len(train_dataset)} 条 | 验证集: {len(val_dataset)} 条")
    log(f"[Train] 词表大小: {len(vocab)}")
    log(f"[Train] 训练批次数: {len(train_loader)} | 验证批次数: {len(val_loader)}")

    #------------4. 模型初始化------------
    model = BiLSTM_DynamicCRF(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        device=device,
        use_pack=True,
    )

    params = model.get_params()
    log(f"[Train] 模型参数量: {params['total_params']:,} (可训练: {params['trainable_params']:,})")
    model = model.to(device)

    #------------5. 优化器与学习率调度器------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    #------------6. 断点续训------------
    start_epoch = 0
    best_f1 = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        log(f"[Train] 从Epoch {start_epoch}恢复训练，best_f1={best_f1:.4f}")

    #------------7. 训练循环------------
    log("[Train] 开始训练循环...")
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        epoch_steps = 0

        # 学习率（从调度器获取当前值）
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs}", ncols=100)

        for batch_idx, batch in enumerate(pbar):
            input_ids, tag_ids, attention_mask, lengths = batch
            input_ids = input_ids.to(device)
            tag_ids = tag_ids.to(device)
            attention_mask = attention_mask.to(device)
            lengths = lengths.to(device)

            #------------前向传播------------
            optimizer.zero_grad()
            loss, best_scores, best_paths = model(input_ids, tag_ids, attention_mask, lengths)

            #------------反向传播------------
            loss.backward()

            #------------梯度裁剪------------
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            #------------统计------------
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            epoch_steps += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gnorm": f"{grad_norm:.2f}",
                "lr": f"{current_lr:.2e}",
            })

            #------------中间日志------------
            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                pbar.write(
                    f"  [Step {global_step:06d}] loss={loss.item():.4f} "
                    f"| avg_loss={avg_loss:.4f} | gnorm={grad_norm:.2f}"
                )

        pbar.close()

        #------------Epoch总结------------
        avg_train_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        #------------验证------------
        if (epoch + 1) % args.eval_interval == 0:
            val_p, val_r, val_f1 = evaluate(model, val_loader, device)
            scheduler.step(val_f1)

            # 保存最优模型
            improved = ""
            if val_f1 > best_f1:
                best_f1 = val_f1
                improved = " ⭐"
                best_model_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_f1": best_f1,
                    "val_p": val_p,
                    "val_r": val_r,
                    "val_f1": val_f1,
                    "args": vars(args),
                }, best_model_path)

            log(
                f"[Epoch {epoch+1:02d}/{args.epochs}] "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_p={val_p:.4f} | val_r={val_r:.4f} | val_f1={val_f1:.4f}{improved} | "
                f"best_f1={best_f1:.4f} | lr={current_lr:.2e} | "
                f"time={epoch_time:.1f}s"
            )
        else:
            log(
                f"[Epoch {epoch+1:02d}/{args.epochs}] "
                f"train_loss={avg_train_loss:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            )

        # 每5轮保存一次checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch+1:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "args": vars(args),
            }, ckpt_path)
            log(f"[Checkpoint] 已保存: {ckpt_path}")

    log("=" * 70)
    log(f"[Train] 训练完成！")
    log(f"[Train] 最佳验证 F1: {best_f1:.4f}")
    log(f"[Train] 模型保存目录: {args.save_dir}")
    log("=" * 70)

if __name__ == "__main__":
    args = parse_args()
    train(args)