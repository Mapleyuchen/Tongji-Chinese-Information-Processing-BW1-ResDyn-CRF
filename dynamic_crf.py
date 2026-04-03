import torch
import torch.nn as nn
from typing import Tuple

class DynamicCRF(nn.Module):
    def __init__(
        self,
        tag_size: int,
        hidden_size: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.tag_size = tag_size
        self.hidden_size = hidden_size
        self.device = device or torch.device("cpu")

        self.emission_proj = nn.Linear(hidden_size, tag_size, bias=True)
        
        hidden_mlp = hidden_size  
        self.trans_h_proj = nn.Linear(2 * hidden_size, hidden_mlp, bias=True)
        self.trans_out_proj = nn.Linear(
            hidden_mlp, tag_size * tag_size, bias=True
        )
        
        self.static_transitions = nn.Parameter(torch.empty(tag_size, tag_size))
        nn.init.uniform_(self.static_transitions, -0.1, 0.1)

    def _compute_dynamic_transitions(
        self, h_forward: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h_forward.shape

        zeros = torch.zeros(
            batch_size, 1, self.hidden_size,
            dtype=h_forward.dtype, device=h_forward.device
        )
        h_prev = torch.cat([zeros, h_forward[:, :-1, :]], dim=1)

        concat_h = torch.cat([h_prev, h_forward], dim=-1)
        r = torch.relu(self.trans_h_proj(concat_h))
        T_flat = self.trans_out_proj(r)

        transitions = T_flat.view(
            batch_size, seq_len, self.tag_size, self.tag_size
        )
        
        # 残差连接：动态特征偏移量+全局静态结构
        transitions = transitions + self.static_transitions.unsqueeze(0).unsqueeze(0)
        
        transitions = transitions.clamp(min=-30, max=30)
        return transitions

    def _log_sum_exp(
        self, scores: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        if dim < 0:
            dim = scores.ndim + dim

        max_score = scores.amax(dim=dim, keepdim=True)
        all_neg_inf = torch.isinf(max_score) & (max_score < 0)

        exp_term = (scores - max_score).clamp(min=-50, max=50)
        sum_exp = torch.exp(exp_term).sum(dim=dim, keepdim=True)
        
        safe_result = max_score + torch.log(sum_exp.clamp(min=1e-45))

        safe_result = torch.where(
            all_neg_inf,
            torch.full_like(safe_result, float("-inf")),
            safe_result,
        )
        
        return safe_result.squeeze(dim)

    def _compute_emission_scores(
        self, lstm_output: torch.Tensor
    ) -> torch.Tensor:
        emissions = self.emission_proj(lstm_output)
        emissions = emissions.clamp(min=-30, max=30)
        return emissions

    def _forward_alg(
        self,
        emissions: torch.Tensor,
        transitions: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, tag_size = emissions.shape

        log_alpha = torch.full(
            (batch_size, seq_len, tag_size),
            float("-inf"),
            device=emissions.device,
            dtype=emissions.dtype,
        )
        log_alpha[:, 0, :] = emissions[:, 0, :]

        for step in range(1, seq_len):
            log_alpha_prev = log_alpha[:, step - 1, :].unsqueeze(-1)   
            emit = emissions[:, step, :].unsqueeze(1)                  
            trans = transitions[:, step, :, :]                         

            log_score = log_alpha_prev + trans + emit                  
            new_log_alpha = self._log_sum_exp(log_score, dim=1)

            step_mask = mask[:, step].float().unsqueeze(-1).expand(batch_size, tag_size)
            log_alpha[:, step, :] = torch.where(
                step_mask > 0,
                new_log_alpha,
                log_alpha[:, step, :],
            )

        last_step = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(batch_size, device=emissions.device)
        last_log_alpha = log_alpha[batch_idx, last_step]
        
        log_partition = self._log_sum_exp(last_log_alpha, dim=-1)
        return log_alpha, log_partition

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        transitions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, seq_len, tag_size = emissions.shape

        emit_scores = emissions.gather(
            dim=2, index=tags.unsqueeze(-1)
        ).squeeze(-1)
        emit_sum = (emit_scores * mask.float()).sum(dim=-1)

        tags_prev = torch.zeros_like(tags)
        tags_prev[:, 1:] = tags[:, :-1]

        y_prev_4d = tags_prev.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, 1)
        y_curr_4d = tags.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, 1)

        t_selected = transitions[
            torch.arange(batch_size, device=tags.device).view(-1, 1, 1, 1).expand(batch_size, seq_len, 1, 1),
            torch.arange(seq_len, device=tags.device).view(1, -1, 1, 1).expand(batch_size, seq_len, 1, 1),
            y_prev_4d,
            y_curr_4d,
        ].squeeze(-1).squeeze(-1)

        step_mask = torch.zeros(seq_len, dtype=torch.float, device=tags.device)
        step_mask[1:] = 1.0

        trans_sum = (t_selected * mask.float() * step_mask.unsqueeze(0)).sum(dim=-1)
        scores = emit_sum + trans_sum

        return scores

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        transitions: torch.Tensor,
        mask: torch.BoolTensor,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, tag_size = emissions.shape

        if lengths is None:
            lengths = mask.sum(dim=1).long()

        delta = torch.full(
            (batch_size, seq_len, tag_size),
            float("-inf"),
            device=emissions.device,
            dtype=emissions.dtype,
        )
        delta[:, 0, :] = emissions[:, 0, :]

        psi = torch.zeros(
            batch_size, seq_len, tag_size,
            dtype=torch.long, device=emissions.device
        )

        for step in range(1, seq_len):
            score = (
                delta[:, step - 1, :].unsqueeze(-1)   
                + transitions[:, step, :, :]          
                + emissions[:, step, :].unsqueeze(1)  
            )

            best_prev, best_prev_idx = score.max(dim=1)

            # 使用torch.where进行安全的布尔路由，杜绝-inf * 0.0
            step_mask_bool = mask[:, step].unsqueeze(-1).expand(batch_size, tag_size)
            
            delta[:, step, :] = torch.where(
                step_mask_bool,
                best_prev,
                delta[:, step, :]
            )
            
            psi[:, step, :] = torch.where(
                step_mask_bool,
                best_prev_idx,
                psi[:, step, :]
            )

        batch_idx = torch.arange(batch_size, device=emissions.device)
        best_scores_list = []
        best_last_tags_list = []
        for b in range(batch_size):
            last = lengths[b].item() - 1
            best_scores_list.append(delta[b, last].max(dim=-1)[0])
            best_last_tags_list.append(delta[b, last].argmax(dim=-1))

        best_scores = torch.stack(best_scores_list)
        best_last_tags = torch.stack(best_last_tags_list)

        best_paths = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=emissions.device
        )
        for b in range(batch_size):
            end_step = lengths[b].item() - 1
            best_paths[b, end_step] = best_last_tags[b]
            
            for step in range(end_step - 1, -1, -1):
                next_tag = best_paths[b, step + 1]
                best_paths[b, step] = psi[b, step + 1, next_tag]

        return best_scores, best_paths

    def forward(
        self,
        lstm_output: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emissions = self._compute_emission_scores(lstm_output)
        transitions = self._compute_dynamic_transitions(lstm_output)
        
        _, log_partition = self._forward_alg(emissions, transitions, mask)
        gold_scores = self._score_sentence(emissions, transitions, tags, mask)
        
        loss = (log_partition - gold_scores).mean()
        
        if lengths is None:
            lengths = mask.sum(dim=1).long()
        best_scores, best_paths = self._viterbi_decode(emissions, transitions, mask, lengths)

        return loss, best_scores, best_paths

    def decode(
        self,
        lstm_output: torch.Tensor,
        mask: torch.BoolTensor,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emissions = self._compute_emission_scores(lstm_output)
        transitions = self._compute_dynamic_transitions(lstm_output)
        
        if lengths is None:
            lengths = mask.sum(dim=1).long()
            
        return self._viterbi_decode(emissions, transitions, mask, lengths)