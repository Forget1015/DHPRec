"""
DHPRec: Denoised Historical Patterns for Sequential Recommendation

本模块实现DHPRec框架的核心模型，包含三个主要组件：
1. Frequency-Domain Feature Refinement (FDFR): 频域特征精炼模块，用于过滤高频噪声
2. Pattern-Based Anchor-Guided Fusion: 基于模式的锚点引导融合机制
3. Bridging Intent and Regularity: 即时意图与历史规律的动态融合
"""

import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from layers import *
import numpy as np
import torch.distributed as dist

    
def gather_tensors(t):
    """分布式训练中收集所有进程的张量"""
    local_rank = dist.get_rank()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)
    all_tensors[local_rank] = t
    return torch.cat(all_tensors)
    

class ContrastiveLoss(nn.Module):
    """
    序列对齐对比损失 (Sequence Alignment Loss)
    通过对比学习增强序列表示的鲁棒性，缓解数据稀疏问题
    """
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
    """序列推荐基础模型类，提供通用的序列处理方法"""
    def __init__(self):
        super(SeqBaseModel, self).__init__()
    
    def gather_indexes(self, output, gather_index):
        """从序列输出中提取指定位置的向量表示"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """生成单向或双向注意力掩码"""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def get_code_attention_mask(self, item_seq, code_level):
        """生成语义编码的注意力掩码"""
        B, L = item_seq.size()
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = torch.tril(
            extended_attention_mask.expand((-1, -1, L, -1))
        )
        extended_attention_mask = extended_attention_mask.unsqueeze(3).expand(-1, -1, -1, code_level, -1).transpose(3, 4)
        extended_attention_mask = extended_attention_mask.reshape(B, 1, L, L*code_level)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class FrequencyAttention(nn.Module):
    """
    频域特征精炼模块 (Frequency-Domain Feature Refinement, FDFR)
    
    该模块通过频域变换实现序列去噪：
    1. 紧凑谱变换 (Compact Spectral Transformation): 将序列表示转换到频域
    2. 能量感知滤波 (Energy-Aware Filter Generation): 根据频率分量的能量分布生成自适应权重
    3. 频率加权正则化: 抑制高频噪声，增强低频稳定信号
    
    低频分量对应用户稳定的长期偏好，高频分量通常对应随机噪声
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        # 频率投影层，用于计算各频率分量的重要性权重
        self.freq_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        B, L, H = x.shape
        
        # 紧凑谱变换：使用实值FFT将序列转换到频域
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 计算频率分量的幅度（能量）
        freq_magnitude = torch.abs(x_fft)
        
        # 能量感知滤波：根据幅度生成自适应权重
        # 低频分量（稳定偏好）获得较高权重，高频分量（噪声）被抑制
        freq_weight = torch.softmax(self.freq_proj(freq_magnitude), dim=1)
        
        # 频域加权调制
        weighted_fft = x_fft * freq_weight
        
        # 逆FFT转换回时域，得到去噪后的表示
        enhanced = torch.fft.irfft(weighted_fft, n=L, dim=1, norm='ortho')
        
        # 残差连接，融合系数控制增强强度
        return self.layer_norm(x + self.dropout(enhanced * 0.1))


class MultiScaleFrequencyFusion(nn.Module):
    """
    多尺度频率融合模块
    
    从不同频率尺度提取特征并加权融合，捕获多粒度的用户偏好模式
    """
    def __init__(self, hidden_size, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        # 可学习的尺度融合权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        B, L, H = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        fft_len = x_fft.shape[1]
        
        outputs = []
        for i in range(self.num_scales):
            # 不同频段的截止点，实现多尺度频率分解
            cutoff = fft_len // (2 ** i)
            mask = torch.zeros(fft_len, device=x.device)
            mask[:cutoff] = 1.0
            
            # 频域滤波
            filtered = x_fft * mask.view(1, -1, 1)
            reconstructed = torch.fft.irfft(filtered, n=L, dim=1, norm='ortho')
            outputs.append(reconstructed)
        
        # 自适应加权融合各尺度特征
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = sum(w * o for w, o in zip(weights, outputs))
        
        return self.layer_norm(x + fused * 0.1)


class DHPRec(SeqBaseModel):
    """
    DHPRec: Denoised Historical Patterns for Sequential Recommendation
    
    核心思想：结合即时意图(Immediate Intent)和历史规律(Historical Patterns)进行推荐
    
    框架包含三个核心组件：
    1. 频域特征精炼 (FDFR): 过滤高频噪声，提取稳定的长期偏好信号
    2. 基于模式的锚点引导融合: 将长序列划分为时间片段(Pattern Units)，
       使用最近片段作为锚点提取与当前意图相关的历史规律
    3. 意图与规律的动态融合: 通过可学习门控机制平衡即时意图和历史偏好的贡献
    """
    def __init__(self, args, dataset, index, device):
        super(DHPRec, self).__init__()
        
        # ==================== 模型超参数 ====================
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

        # ==================== 语义编码索引 ====================
        # 基于向量量化的多视图语义编码器索引
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

        # ==================== 嵌入层 ====================
        # 查询编码嵌入，用于Q-Former的查询向量
        self.query_code_embedding = nn.Embedding(self.n_codes, self.embedding_size, padding_idx=0)

        # 物品文本属性嵌入（标题、品牌、类别等）
        self.item_text_embedding = nn.ModuleList([nn.Embedding(self.n_items, self.embedding_size,
                                                               padding_idx=0)
                                                   for _ in range(self.text_num)])
        self.item_text_embedding.requires_grad_(False)

        # Q-Former: 跨注意力Transformer，融合查询编码和文本语义
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

        # ==================== 频域特征精炼模块 (FDFR) ====================
        self.fourier_attention = FrequencyAttention(self.embedding_size)
        # self.multiScale_frequency_fusion = MultiScaleFrequencyFusion(self.embedding_size)
        
        # ==================== 基于模式的锚点引导融合组件 ====================
        # 复合位置编码：片段级位置 + 片段内位置 + 维度偏置
        # 用于捕获用户行为在不同时间粒度上的位置信息
        self.session_bias = nn.Parameter(torch.zeros(self.max_seq_length, 1, 1))    # 片段级位置偏置
        self.position_bias = nn.Parameter(torch.zeros(1, self.max_seq_length, 1))   # 片段内位置偏置
        self.dim_bias = nn.Parameter(torch.zeros(1, 1, self.embedding_size))        # 维度偏置
        
        # 片段内Transformer编码器 (Intra-Slice Pattern Extraction)
        # 提取每个时间片段内的用户兴趣模式
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

        # 片段间BiLSTM (Global Historical Pattern Modeling)
        # 建模片段序列的时序演化关系，构建全局历史模式空间
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
        
        # 片段位置编码
        self.session_position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        
        self.session_position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        
        # 锚点引导的模式提取 (Anchor-Guided Pattern Extract)
        # 使用最近片段作为锚点(Anchor)，计算与历史模式的相关性
        self.session_attention_w = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.session_attention_v = nn.Linear(self.embedding_size, 1, bias=False)
        
        # 意图与规律融合的门控网络 (Bridging Intent and Regularity)
        # 动态平衡即时意图(r*)和全局历史规律(c)的贡献
        self.residual_gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.Sigmoid()
        )
        
        # 损失函数
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _split_to_sessions(self, item_emb, session_ids):
        """
        时间片段划分 (Pattern Modeling)
        
        将用户历史序列按时间间隔划分为多个片段(Slices)
        每个片段代表用户在特定时间段内的行为模式
        
        Args:
            item_emb: 物品嵌入序列 [B, L, H]
            session_ids: 片段ID序列，标识每个物品所属的时间片段
            
        Returns:
            item_emb_sessions: 按片段组织的嵌入 [B, max_sess_count, max_sess_len, H]
            user_sess_count: 每个用户的片段数量
            sess_item_lens: 每个片段内的物品数量
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
            
            # 按session_id分组
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
        
        # 构建填充后的片段张量
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
        基于模式的锚点引导融合 (Pattern-Based Anchor-Guided Fusion)
        
        该方法实现论文的核心机制，包含以下步骤：
        1. 时间片段划分: 将长序列划分为短而密集的模式单元(Pattern Units)
        2. 复合位置编码: 注入片段级和片段内的位置信息
        3. 片段内模式提取: 使用Transformer提取每个片段内的用户兴趣
        4. 全局历史模式建模: 使用BiLSTM建模片段间的时序演化
        5. 锚点引导的模式提取: 以最近片段为锚点，提取与当前意图相关的历史规律
        6. 意图与规律融合: 通过门控机制动态平衡即时意图和历史偏好
        
        Args:
            item_emb: 经过FDFR去噪的物品嵌入 [B, L, H]
            session_ids: 时间片段ID
            item_seq_len: 序列长度
            
        Returns:
            final_output: 融合即时意图和历史规律的用户表示 [B, H]
        """
        B, L, H = item_emb.shape
        device = item_emb.device
        
        # Step 1: 时间片段划分
        item_emb_sessions, user_sess_count, sess_item_lens = self._split_to_sessions(item_emb, session_ids)
        _, max_sess_count, max_sess_len, _ = item_emb_sessions.shape
        
        # Step 2: 复合位置编码 (Compound Position Encoding)
        # BE(k,t,c) = x + w_k + w_t + w_c
        # w_k: 片段级位置, w_t: 片段内位置, w_c: 维度偏置
        sess_bias = self.session_bias[:max_sess_count, :, :]
        pos_bias = self.position_bias[:, :max_sess_len, :]
        dim_bias = self.dim_bias
        item_emb_sessions = item_emb_sessions + sess_bias.unsqueeze(0) + pos_bias.unsqueeze(0) + dim_bias.unsqueeze(0)

        # Step 3: 片段内模式提取 (Intra-Slice Pattern Extraction)
        # 使用Transformer编码器提取每个片段内的用户兴趣模式
        intra_input = item_emb_sessions.view(B * max_sess_count, max_sess_len, H)
        flat_sess_lens = sess_item_lens.view(-1)
        intra_mask = self._get_intra_session_mask(flat_sess_lens, max_sess_len, device)
        
        # 添加片段内位置编码
        intra_pos_ids = torch.arange(max_sess_len, dtype=torch.long, device=device)
        intra_pos_ids = intra_pos_ids.unsqueeze(0).expand(B * max_sess_count, -1)
        intra_pos_emb = self.intra_position_embedding(intra_pos_ids)
        intra_input = intra_input + intra_pos_emb
        intra_input = self.intra_layer_norm(intra_input)
        intra_input = self.intra_dropout(intra_input)
        
        intra_output = self.intra_transformer(intra_input, intra_input, intra_mask)[-1]
        
        # 聚合片段表示：取最后一个有效物品的表示作为片段表示
        gather_idx = (flat_sess_lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, H)
        session_repr = intra_output.gather(dim=1, index=gather_idx).squeeze(1)
        
        valid_sess_mask = (flat_sess_lens > 0).float().unsqueeze(1)
        session_repr = session_repr * valid_sess_mask
        session_repr = session_repr.view(B, max_sess_count, H)
        
        # 添加片段级位置编码
        sess_pos_ids = torch.arange(max_sess_count, dtype=torch.long, device=device)
        sess_pos_ids = sess_pos_ids.unsqueeze(0).expand(B, -1)
        sess_pos_emb = self.session_position_embedding(sess_pos_ids)
        session_repr = session_repr + sess_pos_emb

        # Step 4: 全局历史模式建模 (Global Historical Pattern Modeling)
        # 使用BiLSTM建模片段序列的时序演化，捕获模式间的依赖关系
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
        
        # Step 5: 锚点引导的模式提取 (Anchor-Guided Pattern Extract)
        # 将最近片段作为锚点(Anchor)，代表用户的即时意图(Immediate Intent)
        last_sess_idx = (user_sess_count - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, H)
        last_session_repr = inter_output.gather(dim=1, index=last_sess_idx).squeeze(1)  # 锚点 r*
        
        # 计算每个历史模式与锚点的相关性分数
        # s_n = W_3 * tanh(W_1 * r_n + W_2 * r* + b)
        query_expanded = last_session_repr.unsqueeze(1).expand(-1, max_sess_count, -1)
        attention_input = torch.cat([inter_output, query_expanded], dim=-1)
        attention_hidden = torch.tanh(self.session_attention_w(attention_input))
        attention_scores = self.session_attention_v(attention_hidden).squeeze(-1)
        
        # 掩码无效片段并计算注意力权重
        sess_mask = torch.arange(max_sess_count, device=device).unsqueeze(0) < user_sess_count.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~sess_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)  # α_n
        
        # 加权聚合历史模式，得到全局历史规律 c
        attended_output = torch.bmm(attention_weights.unsqueeze(1), inter_output).squeeze(1)

        # Step 6: 意图与规律的动态融合 (Bridging Intent and Regularity)
        # e_u = λ * r* + (1-λ) * c
        # λ = σ(W_g1 * r* + W_g2 * c + b_g)
        # 门控系数λ动态平衡即时意图(r*)和全局历史规律(c)的贡献
        gate_input = torch.cat([attended_output, last_session_repr], dim=-1)
        gate = self.residual_gate(gate_input)  # λ
        final_output = gate * last_session_repr + (1 - gate) * attended_output
        
        return final_output
    
    def _get_intra_session_mask(self, sess_lens, max_len, device):
        """生成片段内的因果注意力掩码"""
        B = sess_lens.size(0)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        valid_mask = positions < sess_lens.unsqueeze(1)
        
        attention_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = torch.tril(attention_mask.expand(-1, -1, max_len, -1))
        attention_mask = torch.where(attention_mask, 0.0, -10000.0)
        return attention_mask

    def forward(self, item_seq, item_seq_len, code_seq, session_ids):
        """
        前向传播
        
        Args:
            item_seq: 物品ID序列 [B, L]
            item_seq_len: 序列长度
            code_seq: 语义编码序列
            session_ids: 时间片段ID
            
        Returns:
            item_seq_output: 用户表示（融合即时意图和历史规律）
            item_seq_emb: 物品序列嵌入
        """
        B, L = item_seq.size(0), item_seq.size(1)
        item_flatten_seq = item_seq.reshape(-1)
        query_seq_emb = self.query_code_embedding(code_seq)
        
        # 获取多视图文本语义嵌入
        text_embs = []
        for i in range(self.text_num):
            text_emb = self.item_text_embedding[i](item_flatten_seq)
            text_embs.append(text_emb)
        encoder_output = torch.stack(text_embs, dim=1)

        # Q-Former融合查询编码和文本语义
        item_seq_emb = self.qformer(query_seq_emb, encoder_output)[-1]
        item_emb = item_seq_emb.mean(dim=1) + query_seq_emb.mean(dim=1)
        item_emb = item_emb.view(B, L, -1)

        # 频域特征精炼：过滤高频噪声，增强低频稳定信号
        item_emb = self.fourier_attention(item_emb)
        # item_emb = self.multiScale_frequency_fusion(item_emb)
        
        # 基于模式的锚点引导融合：提取并融合即时意图和历史规律
        item_seq_output = self._hierarchical_transformer(item_emb, session_ids, item_seq_len)
        
        return item_seq_output, item_seq_emb

    def get_item_embedding(self,):
        """批量计算所有物品的嵌入表示"""
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
        """编码正样本和负样本物品"""
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
        """
        计算多任务训练损失
        
        损失函数包含四个部分：
        1. L_rec: 推荐损失，优化下一物品预测准确率
        2. L_cl: 序列对齐对比损失，增强序列表示鲁棒性
        3. L_vq: 掩码编码建模损失，保持语义一致性
        4. L_freq: 频率加权正则化损失（隐式包含在FDFR中）
        
        总损失: L_total = L_rec + γ1*L_cl + γ2*L_freq + γ3*L_vq
        """
        B, L = item_seq.size(0), item_seq.size(1)
        code_seq = self.index[item_seq].reshape(B*L, -1)
        item_seq_output, code_output = self.forward(item_seq, item_seq_len, code_seq, session_ids)
        item_seq_output_mask, code_output_mask = self.forward(item_seq, item_seq_len, code_seq_mask, session_ids)

        item_seq_output = F.normalize(item_seq_output, dim=-1)
        
        # 推荐损失 (Recommendation Loss)
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
        
        # 序列对齐对比损失 (Sequence Alignment Loss)
        # 通过对比学习增强序列表示的判别性
        gathered = dist.is_initialized()
        cl_loss_func = ContrastiveLoss(tau=self.tau)
        cl_loss = (cl_loss_func(item_seq_output, item_seq_output_mask, gathered=gathered) + \
                   cl_loss_func(item_seq_output_mask, item_seq_output, gathered=gathered)) / 2
        
        # 掩码编码建模损失 (Masked Code Modeling Loss)
        # 保持语义编码的一致性
        code_embedding = F.normalize(self.query_code_embedding.weight, dim=-1)
        
        code_output_mask = code_output_mask.view(-1, H)
        code_output_mask = F.normalize(code_output_mask, dim=-1)
        
        mlm_logits = torch.matmul(code_output_mask, code_embedding.transpose(0, 1)) / self.tau
        mlm_loss = self.loss_fct(mlm_logits, labels_mask)

        # 总损失
        loss = rec_loss + self.mlm_weight * mlm_loss + self.cl_weight * cl_loss
        loss_dict = dict(loss=loss, mlm_loss=mlm_loss, rec_loss=rec_loss, cl_loss=cl_loss)
        
        return loss_dict

    def full_sort_predict(self, item_seq, item_seq_len, code_seq, session_ids):
        """全排序预测，计算用户对所有物品的偏好分数"""
        seq_output, _ = self.forward(item_seq, item_seq_len, code_seq, session_ids)
        seq_output = F.normalize(seq_output, dim=-1)
        
        item_embedding = F.normalize(self.get_item_embedding(), dim=-1)
        scores = torch.matmul(
            seq_output, item_embedding.transpose(0, 1)
        )

        return scores
