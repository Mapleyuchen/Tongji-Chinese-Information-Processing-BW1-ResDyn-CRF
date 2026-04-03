import os
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

from data_processor import (
    CharVocab,
    segment_to_bmes,
    TAG2IDX,
    IDX2TAG,
    bmes_to_segmentation,
    PAD_IDX,
    PAD_TOKEN,
)
from model import BiLSTM_DynamicCRF


def load_model(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[BiLSTM_DynamicCRF, CharVocab, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 恢复词表
    vocab = CharVocab()
    vocab.load(os.path.join(os.path.dirname(checkpoint_path), "vocab.pkl"))

    # 恢复模型超参数
    saved_args = checkpoint.get("args", {})
    vocab_size = saved_args.get("vocab_size", len(vocab))

    model = BiLSTM_DynamicCRF(
        vocab_size=vocab_size,
        embedding_dim=saved_args.get("embedding_dim", 128),
        lstm_hidden=saved_args.get("lstm_hidden", 256),
        lstm_layers=saved_args.get("lstm_layers", 2),
        dropout=saved_args.get("dropout", 0.3),
        device=device,
        use_pack=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    print(f"[Predict] 模型加载成功: {checkpoint_path}")
    print(f"[Predict] 验证F1: {checkpoint.get('val_f1', 'N/A')}")
    print(f"[Predict] 词表大小: {len(vocab)}")
    print(f"[Predict] 使用设备: {device}")

    return model, vocab, checkpoint


def segment_single(
    sentence: str,
    model: nn.Module,
    vocab: CharVocab,
    device: torch.device,
) -> str:
    if not sentence or not sentence.strip():
        return ""

    #------------预处理：字符化+编码------------
    chars = list(sentence.strip())
    input_ids = torch.tensor(
        [vocab.char2idx(c) for c in chars],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    seq_len = len(chars)
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    lengths = torch.tensor([seq_len], dtype=torch.long, device=device)

    #------------推理------------
    with torch.no_grad():
        _, best_paths = model.decode(input_ids, mask, lengths)

    #------------标签解码------------
    tag_indices = best_paths[0].cpu().tolist()[:seq_len]
    tags = [IDX2TAG[idx] for idx in tag_indices]

    #------------BMES→词语列表------------
    words = bmes_to_segmentation(chars, tags)

    return " ".join(words)


def segment_batch(
    sentences: List[str],
    model: nn.Module,
    vocab: CharVocab,
    device: torch.device,
    batch_size: int = 32,
) -> List[str]:
    results: List[str] = []
    n = len(sentences)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sents = sentences[start:end]

        batch_chars: List[List[str]] = [list(s.strip()) for s in batch_sents]
        batch_len = [len(c) for c in batch_chars]
        
        # 按长度降序排序，以满足LSTM packed_sequence要求
        sorted_indices = sorted(range(len(batch_len)), key=lambda i: batch_len[i], reverse=True)
        sorted_chars = [batch_chars[i] for i in sorted_indices]
        sorted_len = [batch_len[i] for i in sorted_indices]
        max_len = max(sorted_len)

        # padding
        padded_ids = torch.full(
            (len(batch_sents), max_len), PAD_IDX, dtype=torch.long, device=device
        )
        masks = torch.zeros(
            len(batch_sents), max_len, dtype=torch.bool, device=device
        )

        for i, (chars, length) in enumerate(zip(sorted_chars, sorted_len)):
            ids = [vocab.char2idx(c) for c in chars]
            padded_ids[i, :length] = torch.tensor(ids, device=device)
            masks[i, :length] = True

        lengths = torch.tensor(sorted_len, dtype=torch.long, device=device)

        #------------推理------------
        with torch.no_grad():
            _, best_paths = model.decode(padded_ids, masks, lengths)

        #------------解码并还原原始顺序------------
        batch_results = [""] * len(batch_sents)
        for i, (chars, length) in enumerate(zip(sorted_chars, sorted_len)):
            tag_indices = best_paths[i].cpu().tolist()[:length]
            tags = [IDX2TAG[idx] for idx in tag_indices]
            words = bmes_to_segmentation(chars, tags)
            
            # 将结果放回原句子所在的索引位置
            orig_idx = sorted_indices[i]
            batch_results[orig_idx] = " ".join(words)
            
        results.extend(batch_results)

    return results


def read_sentences_from_file(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_results_to_file(results: List[str], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")
    print(f"[Predict] 结果已写入: {filepath} ({len(results)} 条)")

def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM + Dynamic CRF中文分词推理")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="模型检查点路径（.pt文件）")
    parser.add_argument("--sentence", type=str, default=None,
                        help="待分词的单个中文句子")
    parser.add_argument("--input_file", type=str, default=None,
                        help="批量推理输入文件（每行一句）")
    parser.add_argument("--output_file", type=str, default=None,
                        help="批量推理输出文件路径")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量推理的批大小")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备（auto/cpu/cuda）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] 模型文件不存在: {args.checkpoint}")
        print("请先运行python train.py训练模型，或下载预训练模型")
        return

    model, vocab, metadata = load_model(args.checkpoint, device)

    if args.sentence:
        print(f"\n原文: {args.sentence}")
        result = segment_single(args.sentence, model, vocab, device)
        print(f"分词: {result}")

    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"[ERROR] 输入文件不存在: {args.input_file}")
            return

        sentences = read_sentences_from_file(args.input_file)
        print(f"[Predict] 读取到{len(sentences)}条句子，正在分词...")

        results = segment_batch(sentences, model, vocab, device, batch_size=args.batch_size)

        if args.output_file:
            write_results_to_file(results, args.output_file)
        else:
            output_file = os.path.splitext(args.input_file)[0] + "_segmented.txt"
            write_results_to_file(results, output_file)

    else:
        print("\n" + "=" * 50)
        print("中文分词 - 交互模式（输入句子后按回车分词，输入q退出）")
        print("=" * 50)
        while True:
            try:
                sentence = input("\n请输入句子: ").strip()
                if not sentence:
                    continue
                if sentence.lower() in ["q", "quit", "exit"]:
                    print("分词结束啦，效果还不错吧！")
                    break
                result = segment_single(sentence, model, vocab, device)
                print(f"分词结果: {result}")
            except (KeyboardInterrupt, EOFError):
                print("\n分词结束啦，效果还不错吧！")
                break

if __name__ == "__main__":
    main()
