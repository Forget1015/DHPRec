import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from layers import *
import numpy as np
import torch.distributed as dist

    
def gather_tensors(t):
    local_rank = dist.get_rank()
        
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)
    all_tensors[local_rank] = t
    
    return torch.cat(all_tensors)
    

class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x, y, gathered=False):
        if gathered:
            all_y = gather_tensors(y)
            all_y = all_y
        else:
            all_y = y
        x = F.normalize(x, dim=-1)
        all_y = F.normalize(all_y, dim=-1)
        
        B = x.shape[0]
        
        logits = torch.matmul(x, all_y.transpose(0, 1)) / self.tau
        labels = torch.arange(B, device=x.device, dtype=torch.long)
        
        
        loss = F.cross_entropy(logits, labels)

        return loss


class SeqBaseModel(nn.Module):
    def __init__(self):
        super(SeqBaseModel, self).__init__()
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def get_code_attention_mask(self, item_seq, code_level):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        B, L = item_seq.size()
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        extended_attention_mask = torch.tril(
            extended_attention_mask.expand((-1, -1, L, -1))
        )
        extended_attention_mask = extended_attention_mask.unsqueeze(3).expand(-1, -1, -1, code_level, -1).transpose(3, 4)
        extended_attention_mask = extended_attention_mask.reshape(B, 1, L, L*code_level)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class FrequencyAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.freq_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # 可学习的融合系数，初始化为较小值
        # self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        B, L, H = x.shape
        
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freq_magnitude = torch.abs(x_fft)
        freq_weight = torch.softmax(self.freq_proj(freq_magnitude), dim=1)
        weighted_fft = x_fft * freq_weight
        enhanced = torch.fft.irfft(weighted_fft, n=L, dim=1, norm='ortho')
        
        # 用 sigmoid 限制在 [0, 1] 范围,alpha 不想要注释掉直接改成0.1即可
        # alpha = torch.sigmoid(self.alpha)
        return self.layer_norm(x + self.dropout(enhanced * 0.1))

class MultiScaleFrequencyFusion(nn.Module):
    def __init__(self, hidden_size, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        B, L, H = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        fft_len = x_fft.shape[1]
        
        outputs = []
        for i in range(self.num_scales):
            # 不同频段的截止点
            cutoff = fft_len // (2 ** i)
            mask = torch.zeros(fft_len, device=x.device)
            mask[:cutoff] = 1.0
            
            filtered = x_fft * mask.view(1, -1, 1)
            reconstructed = torch.fft.irfft(filtered, n=L, dim=1, norm='ortho')
            outputs.append(reconstructed)
        
        # 加权融合
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = sum(w * o for w, o in zip(weights, outputs))
        
        return self.layer_norm(x + fused * 0.1)


class DHPRec(SeqBaseModel):
    def __init__(self, args, dataset, index, device):
        super(DHPRec, self).__init__()
        # load parameters info
        self.n_layers = args.n_layers
        self.n_layers_cross = args.n_layers_cross
        self.n_heads = args.n_heads
        self.embedding_size = args.embedding_size
        self.text_embedding_size = args.text_embedding_size
        self.hidden_size = args.hidden_size
        self.neg_num = args.neg_num
        self.text_num = len(args.text_types)

        self.max_seq_length = args.max_his_len
        self.code_level = args.code_level
        self.n_codes_per_lel = args.n_codes_per_lel
        self.hidden_dropout_prob = args.dropout_prob
        self.attn_dropout_prob = args.dropout_prob
        self.hidden_dropout_prob_cross = args.dropout_prob_cross
        self.attn_dropout_prob_cross = args.dropout_prob_cross
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-12

        self.initializer_range = 0.02

        index[0] = [0] * self.code_level
        self.index = torch.tensor(index, dtype=torch.long, device=device)
        for i in range(self.code_level):
            self.index[:, i] += i * self.n_codes_per_lel + 1

        self.n_items = dataset.n_items + 1
        self.n_codes = args.n_codes_per_lel*args.code_level + 1
        self.tau = args.tau
        self.cl_weight = args.cl_weight
        self.mlm_weight = args.mlm_weight
        self.device = device

        self.item_embedding = None

        self.query_code_embedding = nn.Embedding(self.n_codes, self.embedding_size, padding_idx=0)

        self.item_text_embedding = nn.ModuleList([nn.Embedding(self.n_items, self.embedding_size,
                                                               padding_idx=0)
                                                   for _ in range(self.text_num)])
        self.item_text_embedding.requires_grad_(False)

        self.qformer = CrossAttTransformer(
            n_layers=self.n_layers_cross,
            n_heads=self.n_heads,
            hidden_size=self.embedding_size,
            inner_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob_cross,
            attn_dropout_prob=self.attn_dropout_prob_cross,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        #------------------------------------------------------
        self.fourier_attention = FrequencyAttention(self.embedding_size)
        # self.multiScale_frequency_fusion = MultiScaleFrequencyFusion(self.embedding_size)
        #------------------------------------------------------
        # Hierarchical Transformer 组件（改进版）
        
        # BiasEncoding（参考DSIN）：session偏置 + 位置偏置 + 维度偏置
        # 在intra-session transformer之前对输入做bias encoding
        self.session_bias = nn.Parameter(torch.zeros(self.max_seq_length, 1, 1))  # [max_sess, 1, 1]
        self.position_bias = nn.Parameter(torch.zeros(1, self.max_seq_length, 1))  # [1, max_len, 1]
        self.dim_bias = nn.Parameter(torch.zeros(1, 1, self.embedding_size))  # [1, 1, H]
        
        # Intra-session transformer (组内)
        self.intra_position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.intra_transformer = Transformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.embedding_size,
            inner_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.intra_layer_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.intra_dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Inter-session: 使用BiLSTM（参考DSIN）
        self.inter_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.inter_layer_norm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.inter_dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Session-level position embedding
        self.session_position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        
        # Session-level position embedding（session的位置）
        self.session_position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        
        # 改进1：Session Interest Activation（参考DSIN）
        # 用最后一个item作为query，对所有session做attention
        self.session_attention_w = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.session_attention_v = nn.Linear(self.embedding_size, 1, bias=False)
        
        # 改进2：残差连接 - 保留最后一个session的信息
        self.residual_gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.Sigmoid()
        )
        #--------------------------------------------------------
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # parameters initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    
    def _split_to_sessions(self, item_emb, session_ids):
        """
        将序列按session_ids分割成会话格式
        """
        B, L, H = item_emb.shape
        device = item_emb.device
        
        sessions_list = []
        max_sess_count = 1
        max_sess_len = 1
        
        for b in range(B):
            valid_mask = session_ids[b] > 0
            if valid_mask.sum() == 0:
                sessions_list.append([item_emb[b:b+1, 0:1]])
                continue
            
            valid_sess_ids = session_ids[b][valid_mask]
            valid_emb = item_emb[b][valid_mask]
            
            sessions = []
            start_idx = 0
            current_sid = valid_sess_ids[0].item()
            
            for i in range(1, len(valid_sess_ids)):
                if valid_sess_ids[i].item() != current_sid:
                    sessions.append(valid_emb[start_idx:i])
                    start_idx = i
                    current_sid = valid_sess_ids[i].item()
            sessions.append(valid_emb[start_idx:])
            
            sessions_list.append(sessions)
            max_sess_count = max(max_sess_count, len(sessions))
            max_sess_len = max(max_sess_len, max(len(s) for s in sessions))
        
        max_sess_count = min(max_sess_count, self.max_seq_length)
        max_sess_len = min(max_sess_len, self.max_seq_length)
        
        item_emb_sessions = torch.zeros(B, max_sess_count, max_sess_len, H, device=device, dtype=item_emb.dtype)
        user_sess_count = torch.zeros(B, dtype=torch.long, device=device)
        sess_item_lens = torch.zeros(B, max_sess_count, dtype=torch.long, device=device)

        for b, sessions in enumerate(sessions_list):
            actual_sess_count = min(len(sessions), max_sess_count)
            user_sess_count[b] = actual_sess_count
            for s_idx in range(actual_sess_count):
                sess = sessions[s_idx]
                sess_len = min(len(sess), max_sess_len)
                item_emb_sessions[b, s_idx, :sess_len] = sess[:sess_len]
                sess_item_lens[b, s_idx] = sess_len
            
        return item_emb_sessions, user_sess_count, sess_item_lens
    
    def _hierarchical_transformer(self, item_emb, session_ids, item_seq_len):
        """
        改进版层级Transformer:
        1. BiasEncoding: session偏置 + 位置偏置 + 维度偏置（参考DSIN）
        2. Intra-session: Transformer提取session内兴趣
        3. Inter-session: BiLSTM建模session间时序关系
        4. Session Attention: 用target对session做attention激活
        5. Residual: 残差连接保留最后session信息
        """
        B, L, H = item_emb.shape
        device = item_emb.device
        
        # 1. 按session分组
        item_emb_sessions, user_sess_count, sess_item_lens = self._split_to_sessions(item_emb, session_ids)
        _, max_sess_count, max_sess_len, _ = item_emb_sessions.shape
        # item_emb_sessions: [B, max_sess_count, max_sess_len, H]
        
        # 2. BiasEncoding（参考DSIN）
        # 截取需要的偏置大小
        sess_bias = self.session_bias[:max_sess_count, :, :]  # [max_sess, 1, 1]
        pos_bias = self.position_bias[:, :max_sess_len, :]    # [1, max_len, 1]
        dim_bias = self.dim_bias                               # [1, 1, H]
        
        # 应用BiasEncoding: BE(k,t,c) = x + w_k + w_t + w_c
        # item_emb_sessions: [B, max_sess, max_len, H]
        item_emb_sessions = item_emb_sessions + sess_bias.unsqueeze(0) + pos_bias.unsqueeze(0) + dim_bias.unsqueeze(0)
        
        # 3. Intra-session transformer (组内)
        intra_input = item_emb_sessions.view(B * max_sess_count, max_sess_len, H)
        flat_sess_lens = sess_item_lens.view(-1)
        intra_mask = self._get_intra_session_mask(flat_sess_lens, max_sess_len, device)
        
        # 添加位置编码（这里保留原来的position embedding，与BiasEncoding互补）
        intra_pos_ids = torch.arange(max_sess_len, dtype=torch.long, device=device)
        intra_pos_ids = intra_pos_ids.unsqueeze(0).expand(B * max_sess_count, -1)
        intra_pos_emb = self.intra_position_embedding(intra_pos_ids)
        intra_input = intra_input + intra_pos_emb
        intra_input = self.intra_layer_norm(intra_input)
        intra_input = self.intra_dropout(intra_input)
        
        intra_output = self.intra_transformer(intra_input, intra_input, intra_mask)[-1]
        
        # 4. 聚合session表示：取最后一个有效item的表示
        gather_idx = (flat_sess_lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, H)
        session_repr = intra_output.gather(dim=1, index=gather_idx).squeeze(1)  # [B*max_sess, H]
        
        valid_sess_mask = (flat_sess_lens > 0).float().unsqueeze(1)
        session_repr = session_repr * valid_sess_mask
        session_repr = session_repr.view(B, max_sess_count, H)
        
        # 5. 添加session位置编码
        sess_pos_ids = torch.arange(max_sess_count, dtype=torch.long, device=device)
        sess_pos_ids = sess_pos_ids.unsqueeze(0).expand(B, -1)
        sess_pos_emb = self.session_position_embedding(sess_pos_ids)
        session_repr = session_repr + sess_pos_emb
        
        # 6. Inter-session: BiLSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            session_repr, 
            user_sess_count.cpu().clamp(min=1), 
            batch_first=True, 
            enforce_sorted=False
        )
        packed_output, _ = self.inter_lstm(packed_input)
        inter_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_sess_count)
        inter_output = self.inter_layer_norm(inter_output)
        inter_output = self.inter_dropout(inter_output)
        
        # 7. Session Attention Activation（参考DSIN）
        last_sess_idx = (user_sess_count - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, H)
        last_session_repr = inter_output.gather(dim=1, index=last_sess_idx).squeeze(1)  # [B, H]
        
        # Attention: 计算每个session与最后session的相关性
        query_expanded = last_session_repr.unsqueeze(1).expand(-1, max_sess_count, -1)
        attention_input = torch.cat([inter_output, query_expanded], dim=-1)
        attention_hidden = torch.tanh(self.session_attention_w(attention_input))
        attention_scores = self.session_attention_v(attention_hidden).squeeze(-1)
        
        # Mask无效session
        sess_mask = torch.arange(max_sess_count, device=device).unsqueeze(0) < user_sess_count.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~sess_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权聚合所有session
        attended_output = torch.bmm(attention_weights.unsqueeze(1), inter_output).squeeze(1)
        
        # 8. 残差连接
        gate_input = torch.cat([attended_output, last_session_repr], dim=-1)
        gate = self.residual_gate(gate_input)
        final_output = gate * last_session_repr + (1 - gate) * attended_output
        
        return final_output
    
    def _get_intra_session_mask(self, sess_lens, max_len, device):
        """生成intra-session的attention mask"""
        B = sess_lens.size(0)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        valid_mask = positions < sess_lens.unsqueeze(1)
        
        attention_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = torch.tril(attention_mask.expand(-1, -1, max_len, -1))
        attention_mask = torch.where(attention_mask, 0.0, -10000.0)
        return attention_mask

    def forward(self, item_seq, item_seq_len, code_seq, session_ids):
        
        B, L = item_seq.size(0), item_seq.size(1)
        item_flatten_seq = item_seq.reshape(-1)
        query_seq_emb = self.query_code_embedding(code_seq)
        
        text_embs = []
        for i in range(self.text_num):
            text_emb = self.item_text_embedding[i](item_flatten_seq)
            text_embs.append(text_emb)
        encoder_output = torch.stack(text_embs, dim=1)

        item_seq_emb = self.qformer(query_seq_emb, encoder_output)[-1]
        item_emb = item_seq_emb.mean(dim=1) + query_seq_emb.mean(dim=1)
        item_emb = item_emb.view(B, L, -1)

        # ------------------------------------------------------
        item_emb = self.fourier_attention(item_emb)
        # item_emb = self.multiScale_frequency_fusion(item_emb)
        # ------------------------------------------------------
        
        # 使用改进的层级Transformer
        item_seq_output = self._hierarchical_transformer(item_emb, session_ids, item_seq_len)
        
        return item_seq_output, item_seq_emb
    
    def get_item_embedding(self,):
        batch_size = 1024  
        all_items = torch.arange(self.n_items, device=self.device)
        n_batches = (self.n_items + batch_size - 1) // batch_size

        item_embedding = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i+1)*batch_size, self.n_items)
            batch_item = all_items[start:end]
            batch_query = self.index[batch_item]
            batch_query_emb = self.query_code_embedding(batch_query)
            
            text_embs = []
            for j in range(self.text_num):
                text_emb = self.item_text_embedding[j](batch_item)
                text_embs.append(text_emb)
            batch_encoder_output = torch.stack(text_embs, dim=1)

            batch_item_seq_emb = self.qformer(batch_query_emb, batch_encoder_output)[-1]
            batch_item_emb = batch_item_seq_emb.mean(dim=1) + batch_query_emb.mean(dim=1)

            item_embedding.append(batch_item_emb)

        item_embedding = torch.cat(item_embedding, dim=0)
        return item_embedding
    
    def encode_item(self, pos_items):
        pos_items_list = pos_items.cpu().tolist()
        all_items = set(range(1, self.n_items)) - set(pos_items_list)
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            cand_pool = []
            for _ in range(world_size):
                cand = np.random.choice(list(all_items), size=self.neg_num, replace=False).tolist()
                cand_pool.append(cand)
            candidates = cand_pool[rank]
        else:
            candidates = np.random.choice(list(all_items), size=self.neg_num, replace=False).tolist()
        B = len(pos_items_list)
        batch_item = torch.tensor(pos_items_list+candidates).to(self.device)

        batch_query = self.index[batch_item]
        batch_query_emb = self.query_code_embedding(batch_query)
        
        text_embs = []
        for i in range(self.text_num):
            text_emb = self.item_text_embedding[i](batch_item)
            text_embs.append(text_emb)
        batch_encoder_output = torch.stack(text_embs, dim=1)
        batch_item_seq_emb = self.qformer(batch_query_emb, batch_encoder_output)[-1]
        batch_item_emb = batch_item_seq_emb.mean(dim=1) + batch_query_emb.mean(dim=1)
        
        pos_item_emb = batch_item_emb[:B]
        neg_item_emb = batch_item_emb[B:]
        
        return pos_item_emb, neg_item_emb
    
    def calculate_loss(self, item_seq, item_seq_len, pos_items, code_seq_mask, labels_mask, session_ids):
        
        B, L = item_seq.size(0), item_seq.size(1)
        code_seq = self.index[item_seq].reshape(B*L, -1)
        item_seq_output, code_output = self.forward(item_seq, item_seq_len, code_seq, session_ids)
        item_seq_output_mask, code_output_mask = self.forward(item_seq, item_seq_len, code_seq_mask, session_ids)

        item_seq_output = F.normalize(item_seq_output, dim=-1)
        
        if self.neg_num > 0:
            pos_item_emb, neg_item_emb = self.encode_item(pos_items)
            pos_item_emb = F.normalize(pos_item_emb, dim=-1)
            neg_item_emb = F.normalize(neg_item_emb, dim=-1)
            
            pos_logits = torch.bmm(item_seq_output.unsqueeze(1), pos_item_emb.unsqueeze(2)).squeeze(-1) / self.tau
            neg_logits = torch.matmul(item_seq_output, neg_item_emb.transpose(0, 1)) / self.tau
            logits_rep = torch.cat([pos_logits, neg_logits], dim=1)

            labels = torch.zeros(pos_items.shape[0], device=self.device).long()
            rec_loss = self.loss_fct(logits_rep, labels)
        else:
            all_item_emb = self.get_item_embedding()
            all_item_emb = F.normalize(all_item_emb, dim=-1)
            logits = torch.matmul(item_seq_output, all_item_emb.transpose(0, 1)) / self.tau
            rec_loss = self.loss_fct(logits, pos_items)
        
        H = item_seq_output.shape[-1]
        
        gathered = dist.is_initialized()
        cl_loss_func = ContrastiveLoss(tau=self.tau)
        cl_loss = (cl_loss_func(item_seq_output, item_seq_output_mask, gathered=gathered) + \
                   cl_loss_func(item_seq_output_mask, item_seq_output, gathered=gathered)) / 2
        
        code_embedding = F.normalize(self.query_code_embedding.weight, dim=-1)
        
        code_output_mask = code_output_mask.view(-1, H)
        code_output_mask = F.normalize(code_output_mask, dim=-1)
        
        mlm_logits = torch.matmul(code_output_mask, code_embedding.transpose(0, 1)) / self.tau
        mlm_loss = self.loss_fct(mlm_logits, labels_mask)

        loss = rec_loss + self.mlm_weight * mlm_loss + self.cl_weight * cl_loss
        loss_dict = dict(loss=loss, mlm_loss=mlm_loss, rec_loss=rec_loss, cl_loss=cl_loss)
        
        return loss_dict

    def full_sort_predict(self, item_seq, item_seq_len, code_seq, session_ids):
        seq_output, _ = self.forward(item_seq, item_seq_len, code_seq, session_ids)
        seq_output = F.normalize(seq_output, dim=-1)
        
        item_embedding = F.normalize(self.get_item_embedding(), dim=-1)
        scores = torch.matmul(
            seq_output, item_embedding.transpose(0, 1)
        )

        return scores
