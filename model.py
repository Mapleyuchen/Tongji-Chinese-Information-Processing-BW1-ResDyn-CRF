import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple

from dynamic_crf import DynamicCRF
from data_processor import NUM_TAGS, PAD_IDX, IDX2TAG


class BiLSTM_DynamicCRF(nn.Module):
    """
        vocab_size      : 字符词表大小（包含PAD和UNK）
        embedding_dim   : 字符嵌入维度
        lstm_hidden     : LSTM单向隐状态维度（最终拼接成2*lstm_hidden）
        lstm_layers     : LSTM层数
        dropout         : Dropout比例（Embedding和LSTM输出）
        num_tags        : BMES标签数量（=4）
        device          : 计算设备
        use_pack        : 是否使用PackedSequence（加速变长序列处理）
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_tags: int = NUM_TAGS,
        device: torch.device = None,
        use_pack: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.num_tags = num_tags
        self.device = device or torch.device("cpu")
        self.use_pack = use_pack

        # -----------1. 字符嵌入层-----------
        # 将字符索引映射到embedding_dim维向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        # -----------2. Dropout层-----------
        self.dropout = nn.Dropout(p=dropout)

        # -----------3. 双向LSTM------------
        # input_size: embedding_dim
        # hidden_size: lstm_hidden（单向），最终BiLSTM隐状态维度=2*lstm_hidden
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # -----------4. 投影层-----------
        # 将BiLSTM的拼接隐状态（2*lstm_hidden）映射到中间维度
        # 这是Dynamic CRF的输入维度
        proj_dim = lstm_hidden
        self.lstm_proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden, proj_dim),
            nn.Tanh(),
        )

        # -----------5. Dynamic CRF-----------
        # Dynamic CRF需要知道BiLSTM隐状态维度（拼接后=2*lstm_hidden）
        self.crf = DynamicCRF(
            tag_size=num_tags,
            hidden_size=proj_dim,
            device=self.device,
        )

        self.to(self.device)

    def _forward_lstm(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            input_ids       : (batch, seq_len)    字符索引
            lengths         : (batch,)            每个句子的有效字符数（从大到小排序）
            mask            : (batch, seq_len)    True=有效位置

            lstm_output     : (batch, seq_len, 2*lstm_hidden)
                BiLSTM每个时间步的拼接隐状态
            sorted_output   : (batch, seq_len, 2*lstm_hidden)
                与lengths排序对应的LSTM输出（用于CRF）
        """
        batch_size, max_len = input_ids.shape

        # 获取字符嵌入
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        if self.use_pack:
            # --------PackedSequence（高效处理变长序列）--------
            packed = pack_padded_sequence(
                embeddings,
                lengths=lengths.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )
            packed_out, (h_n, c_n) = self.lstm(packed)

            lstm_output, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=max_len,
            )
        else:
            lstm_output, (h_n, c_n) = self.lstm(embeddings)

        # 投影到统一维度
        lstm_output = self.lstm_proj(lstm_output)
        lstm_output = self.dropout(lstm_output)

        return lstm_output, lstm_output

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            input_ids   : (batch, seq_len)
            tags        : (batch, seq_len)      BMES标签索引
            mask        : (batch, seq_len)      True=有效字符
            lengths     : (batch,)              句子有效长度

            loss        : torch.Tensor (scalar) NLL损失
            best_scores : (batch,)              维特比最优得分
            best_paths  : (batch, seq_len)      维特比最优标签序列
        """
        # BiLSTM提取特征
        lstm_output, _ = self._forward_lstm(input_ids, lengths, mask)

        # Dynamic CRF：计算损失+维特比解码
        loss, best_scores, best_paths = self.crf(lstm_output, tags, mask)

        return loss, best_scores, best_paths

    def decode(
        self,
        input_ids: torch.Tensor,
        mask: torch.BoolTensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_output, _ = self._forward_lstm(input_ids, lengths, mask)
        return self.crf.decode(lstm_output, mask, lengths)

    def get_params(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "embedding_dim": self.embedding_dim,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "num_tags": self.num_tags,
        }

