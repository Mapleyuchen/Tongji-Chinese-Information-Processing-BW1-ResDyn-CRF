import os
import re
import pickle
from collections import Counter
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# BMES四标签体系
TAG_SET = ["B", "M", "E", "S"]
# 与数字标签的相互映射
TAG2IDX: Dict[str, int] = {tag: i for i, tag in enumerate(TAG_SET)}
IDX2TAG: Dict[int, str] = {i: tag for i, tag in enumerate(TAG_SET)}

NUM_TAGS = len(TAG_SET)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

def segment_to_bmes(segmented_sentence: str) -> Tuple[List[str], List[str]]:
    """
        将分词后的句子（单词空格连接）转换为字符序列与BMES标签序列

        segmented_sentence  : str
        chars               : List[str]          字符列表，如['今','天','天','气','真','好']
        tags                : List[str]          BMES标签，如['B','E','B','E','S','S']
    """
    words = segmented_sentence.strip().split()
    chars: List[str] = []
    tags: List[str] = []

    for word in words:
        if len(word) == 0:
            continue
        elif len(word) == 1:
            # 单字词→S
            chars.append(word[0])
            tags.append("S")
        else:
            # 多字词→首字B，中间字M，末字E
            chars.append(word[0])
            tags.append("B")
            for c in word[1:-1]:
                chars.append(c)
                tags.append("M")
            chars.append(word[-1])
            tags.append("E")

    return chars, tags


def bmes_to_segmentation(chars: List[str], tags: List[str]) -> List[str]:
    words: List[str] = []
    current_word = ""

    for c, t in zip(chars, tags):
        if t == "B":
            # 如果前面还有没闭合的词，强行截断并保存，绝不覆盖
            if current_word:
                words.append(current_word)
            current_word = c
        elif t == "M":
            # 如果一上来就是M，当作B处理
            current_word += c
        elif t == "E":
            current_word += c
            words.append(current_word)
            current_word = ""
        elif t == "S":
            # 如果前面有没闭合的词，强行截断并保存
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(c)

    # 循环结束后，把最后残留的词保存下来
    if current_word:
        words.append(current_word)

    return words


class CharVocab:
    def __init__(self):
        self._char2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self._idx2char: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build_vocab(self, sentences: List[str], min_freq: int = 1) -> None:
        counter: Counter = Counter()
        for sent in sentences:
            # 去掉空格后逐字符统计
            for ch in sent.replace(" ", ""):
                counter[ch] += 1

        # 按频次降序排列，频次>=min_freq的字符加入词表
        for char, freq in counter.most_common():
            if freq < min_freq:
                break
            if char not in self._char2idx:
                idx = len(self._char2idx)
                self._char2idx[char] = idx
                self._idx2char[idx] = char

    def __len__(self) -> int:
        return len(self._char2idx)

    def char2idx(self, char: str) -> int:
        return self._char2idx.get(char, UNK_IDX)

    def idx2char(self, idx: int) -> str:
        return self._idx2char.get(idx, UNK_TOKEN)

    def encode(self, chars: List[str]) -> List[int]:
        return [self.char2idx(c) for c in chars]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2char(i) for i in indices]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"char2idx": self._char2idx, "idx2char": self._idx2char}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._char2idx = data["char2idx"]
            self._idx2char = data["idx2char"]

class CWSDataset(Dataset):
    def __init__(
        self,
        sentences: List[str],
        vocab: CharVocab,
        use_bmes_tags: bool = True,
    ):
        self.vocab = vocab
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        for sent in sentences:
            chars, tags = segment_to_bmes(sent)
            input_ids = torch.tensor(vocab.encode(chars), dtype=torch.long)
            tag_ids = torch.tensor(
                [TAG2IDX[t] for t in tags], dtype=torch.long
            )
            self.samples.append((input_ids, tag_ids, len(chars)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    input_ids_list, tag_ids_list, lengths = zip(*batch)

    # 按长度降序排列，便于后续BiLSTM打包（PackedSequence）
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    input_ids_list = [input_ids_list[i] for i in sorted_indices]
    tag_ids_list = [tag_ids_list[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]

    # pad_sequence默认在左侧补0（左侧补PAD不影响BiLSTM右侧输出）
    padded_inputs = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_IDX)
    padded_tags = pad_sequence(tag_ids_list, batch_first=True, padding_value=PAD_IDX)

    # mask: True=有效位置，False=padding
    batch_size, max_len = padded_inputs.shape
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = True

    return padded_inputs, padded_tags, attention_mask, torch.tensor(lengths, dtype=torch.long)


def _resolve_icwb2_dir(data_dir: str) -> str:
    # 检查是否是内层结构（直接包含training/）
    if os.path.isdir(os.path.join(data_dir, "training")):
        return data_dir
    # 检查是否是外层结构（包含icwb2-data/icwb2-data/）
    inner = os.path.join(data_dir, "icwb2-data")
    if os.path.isdir(os.path.join(inner, "training")):
        return inner
    return data_dir


def load_icwb2_dataset(
    data_dir: str,
    dataset_name: str = "msr",
    split: str = "train",
) -> List[str]:
    data_dir = _resolve_icwb2_dir(data_dir)
    sub_dir = os.path.join(data_dir, "training" if split == "train" else "testing")
    filename = f"{dataset_name}_{'training' if split == 'train' else 'test'}.utf8"
    filepath = os.path.join(sub_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}\n请确认--data_dir指向包含training/的目录")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 过滤空行并保留非空句子
    sentences = [line.strip() for line in lines if line.strip()]
    return sentences


def build_dataset(
    data_dir: str,
    dataset_name: str = "msr",
    min_freq: int = 1,
    train_ratio: float = 0.9,
    vocab_save_path: str = None,
) -> Tuple[CWSDataset, CWSDataset, CharVocab]:
    print(f"[DataProcessor] 正在加载{dataset_name}训练语料...")
    sentences = load_icwb2_dataset(data_dir, dataset_name, split="train")
    print(f"[DataProcessor] 共加载{len(sentences)}条句子...")

    # 打乱顺序后按比例划分
    import random
    random.seed(42)
    random.shuffle(sentences)
    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]

    # 构建词表（仅使用训练集字符）
    print("[DataProcessor] 正在构建字符词表...")
    vocab = CharVocab()
    vocab.build_vocab(train_sentences, min_freq=min_freq)
    print(f"[DataProcessor] 词表大小: {len(vocab)}（含PAD/UNK）")

    if vocab_save_path:
        vocab.save(vocab_save_path)
        print(f"[DataProcessor] 词表已保存至: {vocab_save_path}")

    # 封装 Dataset
    train_dataset = CWSDataset(train_sentences, vocab, use_bmes_tags=True)
    val_dataset = CWSDataset(val_sentences, vocab, use_bmes_tags=True)
    print(f"[DataProcessor] 训练集: {len(train_dataset)}条 | 验证集: {len(val_dataset)}条")

    return train_dataset, val_dataset, vocab


def get_dataloader(
    dataset: CWSDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

def compute_f1(pred_tags: List[str], gold_tags: List[str]) -> Tuple[float, float, float]:
    pred_words = set(bmes_to_segmentation([], pred_tags))
    gold_words = set(bmes_to_segmentation([], gold_tags))

    # 过滤掉空字符串
    pred_words.discard("")
    gold_words.discard("")

    if len(pred_words) == 0:
        precision = 0.0
    else:
        precision = len(pred_words & gold_words) / len(pred_words)

    if len(gold_words) == 0:
        recall = 0.0
    else:
        recall = len(pred_words & gold_words) / len(gold_words)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

