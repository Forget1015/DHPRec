import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from matplotlib.gridspec import GridSpec

# 导入项目模块 (保持不变)
from data import load_split_data, MGFSSeqSplitDataset, Collator
from model import DHPRec
from utils import load_json

# ==========================================
# 1. 设置高级论文绘图风格
# ==========================================
sns.set_style("white")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
# 调大字号，让图看起来更饱满
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# === 高级配色方案 (参考蓝色系) ===
# Heatmap: 使用蓝色系渐变
CMAP_HEATMAP = "Blues"  # 蓝色渐变，与参考图一致

# Bar Chart: 使用蓝色系配色
COLOR_SHORT = '#4169E1'  # 皇家蓝 (Short-term)
COLOR_LONG = '#87CEEB'   # 天蓝色 (Long-term) - 浅蓝色

# ... (GatingVisualizer 类保持不变，完全复制你原来的) ...
class GatingVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_gating_weights(self, item_emb, session_ids, item_seq_len):
        B, L, H = item_emb.shape
        device = item_emb.device
        item_emb_sessions, user_sess_count, sess_item_lens = self.model._split_to_sessions(item_emb, session_ids)
        _, max_sess_count, max_sess_len, _ = item_emb_sessions.shape
        
        sess_bias = self.model.session_bias[:max_sess_count, :, :]
        pos_bias = self.model.position_bias[:, :max_sess_len, :]
        dim_bias = self.model.dim_bias
        item_emb_sessions = item_emb_sessions + sess_bias.unsqueeze(0) + pos_bias.unsqueeze(0) + dim_bias.unsqueeze(0)
        
        intra_input = item_emb_sessions.view(B * max_sess_count, max_sess_len, H)
        flat_sess_lens = sess_item_lens.view(-1)
        intra_mask = self.model._get_intra_session_mask(flat_sess_lens, max_sess_len, device)
        
        intra_pos_ids = torch.arange(max_sess_len, dtype=torch.long, device=device).unsqueeze(0).expand(B * max_sess_count, -1)
        intra_pos_emb = self.model.intra_position_embedding(intra_pos_ids)
        intra_input = self.model.intra_dropout(self.model.intra_layer_norm(intra_input + intra_pos_emb))
        intra_output = self.model.intra_transformer(intra_input, intra_input, intra_mask)[-1]
        
        gather_idx = (flat_sess_lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, H)
        session_repr = intra_output.gather(dim=1, index=gather_idx).squeeze(1).view(B, max_sess_count, H)
        session_repr = session_repr + self.model.session_position_embedding(torch.arange(max_sess_count, device=device).unsqueeze(0).expand(B, -1))
        
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(session_repr, user_sess_count.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.model.inter_lstm(packed_input)
        inter_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_sess_count)
        inter_output = self.model.inter_dropout(self.model.inter_layer_norm(inter_output))
        
        last_sess_idx = (user_sess_count - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, H)
        last_session_repr = inter_output.gather(dim=1, index=last_sess_idx).squeeze(1)
        
        query_expanded = last_session_repr.unsqueeze(1).expand(-1, max_sess_count, -1)
        attention_input = torch.cat([inter_output, query_expanded], dim=-1)
        attention_scores = self.model.session_attention_v(torch.tanh(self.model.session_attention_w(attention_input))).squeeze(-1)
        
        sess_mask = torch.arange(max_sess_count, device=device).unsqueeze(0) < user_sess_count.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~sess_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_output = torch.bmm(attention_weights.unsqueeze(1), inter_output).squeeze(1)
        
        gate = self.model.residual_gate(torch.cat([attended_output, last_session_repr], dim=-1))
        
        return {
            'gate': gate.detach().cpu().numpy(),
            'attention_weights': attention_weights.detach().cpu().numpy(),
            'user_sess_count': user_sess_count.detach().cpu().numpy(),
            'sess_item_lens': sess_item_lens.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def get_item_embedding(self, item_seq, code_seq):
        B, L = item_seq.size()
        item_flatten_seq = item_seq.reshape(-1)
        query_seq_emb = self.model.query_code_embedding(code_seq)
        text_embs = [self.model.item_text_embedding[i](item_flatten_seq) for i in range(self.model.text_num)]
        encoder_output = torch.stack(text_embs, dim=1)
        item_seq_emb = self.model.qformer(query_seq_emb, encoder_output)[-1]
        item_emb = item_seq_emb.mean(dim=1) + query_seq_emb.mean(dim=1)
        item_emb = self.model.fourier_attention(item_emb.view(B, L, -1))
        return item_emb
    
    @torch.no_grad()
    def analyze_batch(self, data):
        item_seq = data['item_inters'].to(self.device)
        item_seq_len = data['inter_lens'].to(self.device)
        session_ids = data['session_inters'].to(self.device)
        code_seq = data['code_inters'].to(self.device)
        target_items = data['targets'].to(self.device)
        scores = self.model.full_sort_predict(item_seq, item_seq_len, code_seq, session_ids)
        _, pred_items = torch.topk(scores, k=20, dim=-1)
        item_emb = self.get_item_embedding(item_seq, code_seq)
        gating_info = self.extract_gating_weights(item_emb, session_ids, item_seq_len)
        return {'pred_items': pred_items.cpu().numpy(), 'target_items': target_items.cpu().numpy(), 'item_seq_len': item_seq_len.cpu().numpy(), 'session_ids': session_ids.cpu().numpy(), 'gating_info': gating_info}


def find_diverse_samples(model, dataloader, device, min_seq_len=20, max_samples=4):
    """
    修改后的筛选策略：
    为了让右图差异明显，我们强制挑选 gate 值差异大的样本
    限定所有用户的 session 数量相同，且必须找到至少 max_samples 个
    session 数量不超过 10
    """
    visualizer = GatingVisualizer(model, device)
    candidates = {}  # 按 n_sess 分组
    
    print("正在搜索 Diverse 样本...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        result = visualizer.analyze_batch(batch)
        
        for i in range(len(result['target_items'])):
            seq_len = result['item_seq_len'][i]
            n_sess = result['gating_info']['user_sess_count'][i]
            gate_val = result['gating_info']['gate'][i].mean() 
            
            # 限制 session 数量在 3-10 之间
            if seq_len >= 5 and 3 <= n_sess <= 10:
                if n_sess not in candidates:
                    candidates[n_sess] = []
                candidates[n_sess].append({
                    'attn': result['gating_info']['attention_weights'][i][:n_sess],
                    'gate': gate_val,
                    'n_sess': n_sess,
                    'seq_len': seq_len
                })

    if not candidates: 
        print("未找到任何候选样本！")
        return []
    
    # 打印各session数量的候选数
    print("各 session 数量的候选用户数：")
    for k in sorted(candidates.keys()):
        print(f"  n_sess={k}: {len(candidates[k])} 个用户")
    
    # 找到样本数 >= max_samples 的 session 数量中，选择 session 数最多的
    valid_n_sess = [k for k, v in candidates.items() if len(v) >= max_samples]
    
    if not valid_n_sess:
        best_n_sess = max(candidates.keys(), key=lambda k: len(candidates[k]))
        print(f"警告：没有足够 {max_samples} 个用户的 session 组，使用 n_sess={best_n_sess}，共 {len(candidates[best_n_sess])} 个")
    else:
        best_n_sess = max(valid_n_sess)
    
    same_sess_candidates = candidates[best_n_sess]
    print(f"选择 session 数量为 {best_n_sess} 的用户，共 {len(same_sess_candidates)} 个候选")
    
    if len(same_sess_candidates) < max_samples:
        print(f"警告：只找到 {len(same_sess_candidates)} 个用户，少于要求的 {max_samples} 个")
        return same_sess_candidates
    
    # 按 Gate 值排序，取分布均匀的样本，严格取 max_samples 个
    same_sess_candidates.sort(key=lambda x: x['gate'])
    indices = np.linspace(0, len(same_sess_candidates) - 1, max_samples, dtype=int)
    selected_samples = [same_sess_candidates[i] for i in indices]
    
    print(f"最终选择 {len(selected_samples)} 个用户")
    return selected_samples


def plot_paper_figure(samples, save_path='gating_analysis.pdf'):
    """
    分开生成两个纯净的SVG图：
    - 图1: Session Attention Heatmap (正方形方块，保留坐标轴刻度数值)
    - 图2: Gating Mechanism 柱状图 (保留坐标轴刻度数值)
    """
    n_samples = len(samples)
    n_sess = samples[0]['n_sess']  # 所有用户session数相同
    
    # 准备 Heatmap 数据
    heatmap_data = np.zeros((n_samples, n_sess))
    for i, s in enumerate(samples):
        attn = np.power(s['attn'], 0.6)  # 增强对比度
        heatmap_data[i, :] = attn
    
    # 准备 Bar Chart 数据
    gates = [s['gate'] for s in samples]
    long_terms = [1 - g for g in gates]
    
    # 根据 save_path 生成两个文件名
    base_path = save_path.rsplit('.', 1)[0]
    save_path_a = base_path + "_attention.svg"
    save_path_b = base_path + "_gating.svg"
    
    # ================= 图1: Session Attention Heatmap =================
    cell_size = 0.6
    fig_width = n_sess * cell_size + 1.5
    fig_height = n_samples * cell_size + 0.5
    
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax1.imshow(heatmap_data, cmap=CMAP_HEATMAP, aspect='equal',
                    vmin=0, vmax=np.max(heatmap_data))
    
    # 添加白色网格线
    for i in range(n_samples + 1):
        ax1.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(n_sess + 1):
        ax1.axvline(j - 0.5, color='white', linewidth=0.5)
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    # 保留坐标轴刻度数值
    ax1.set_xticks(np.arange(0, n_sess, max(1, n_sess // 5)))
    ax1.set_yticks(np.arange(n_samples))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path_a, format='svg', bbox_inches='tight', facecolor='white')
    print(f"图1 (Attention) 已保存至: {save_path_a}")
    plt.close(fig1)
    
    # ================= 图2: Gating Mechanism 柱状图 =================
    fig2, ax2 = plt.subplots(figsize=(4, fig_height))
    
    y_pos = np.arange(n_samples)
    height = 0.6
    
    ax2.barh(y_pos, gates, height, color=COLOR_SHORT, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax2.barh(y_pos, long_terms, height, left=gates, color=COLOR_LONG, alpha=0.9, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.5, n_samples - 0.5)
    ax2.invert_yaxis()
    
    # 保留坐标轴刻度数值
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticks(y_pos)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 柱子上标注数值
    for i, (g, l) in enumerate(zip(gates, long_terms)):
        if g > 0.12:
            ax2.text(g/2, y_pos[i], f'{g:.2f}', va='center', ha='center', 
                     color='white', fontweight='bold', fontsize=10)
        if l > 0.12:
            ax2.text(g + l/2, y_pos[i], f'{l:.2f}', va='center', ha='center', 
                     color='#1a1a1a', fontweight='bold', fontsize=10)
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path_b, format='svg', bbox_inches='tight', facecolor='white')
    print(f"图2 (Gating) 已保存至: {save_path_b}")
    plt.close(fig2)
# ==========================================
# 主逻辑 (保持不变)
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # ========== 与 main.py 完全一致的参数 ==========
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="Musical_Instruments")
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--cl_weight', type=float, default=0.1)
    parser.add_argument('--mlm_weight', type=float, default=0.1)
    parser.add_argument('--neg_num', type=int, default=49)
    parser.add_argument('--text_types', nargs='+', type=str, default=["meta"])
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--learner', type=str, default="AdamW")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument('--map_path', type=str, default=".emb_map.json")
    parser.add_argument('--text_index_path', type=str, default=".code.pq.64_128.json")
    parser.add_argument('--text_emb_path', type=str, default=".t5.meta.emb.npy")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--n_codes_per_lel', type=int, default=256)
    parser.add_argument('--code_level', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_layers_cross', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.5)
    parser.add_argument('--dropout_prob_cross', type=float, default=0.3)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--metrics', type=str, default="recall@5,ndcg@5,recall@10,ndcg@10")
    parser.add_argument('--valid_metric', type=str, default="ndcg@10")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--ckpt_dir", type=str, default="./myckpt/")
    parser.add_argument("--resume", type=str, default=None)
    
    # ========== visualize_gating.py 专用参数 ==========
    parser.add_argument("--ckpt_path", type=str, default="./myckpt/best_model.pth") 
    parser.add_argument("--model_path", type=str, default=None, help="Alias for ckpt_path")
    parser.add_argument("--save_path", type=str, default="./gating_analysis.pdf")
    parser.add_argument("--topk", type=int, default=5)
    
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 处理 model_path 别名
    if args.model_path is not None:
        args.ckpt_path = args.model_path
    
    # 1. 加载数据
    print("Loading data...")
    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(os.path.join(args.data_path, args.dataset, f"{args.dataset}{args.text_index_path}"))
    
    test_dataset = MGFSSeqSplitDataset(args, n_items, test, index, 'test')
    collator = Collator(args)
    # 只要一个 batch 就能做可视化，不需要太多
    test_loader = DataLoader(test_dataset, num_workers=0, collate_fn=collator,
                             batch_size=1024, shuffle=True) # Shuffle=True 随机抽样

    # 2. 先加载 text embedding，获取维度信息
    print("Loading text embeddings...")
    text_embs = []
    for ttype in args.text_types:
        emb_path = os.path.join(args.data_path, args.dataset, f"{args.dataset}.t5.{ttype}.emb.npy")
        if os.path.exists(emb_path):
            text_emb = np.load(emb_path)
            text_emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(text_emb)
            text_embs.append(text_emb)
        else:
            raise FileNotFoundError(f"Text embedding not found: {emb_path}")
    
    # 设置 text_embedding_size（创建模型前必须设置）
    args.text_embedding_size = text_embs[0].shape[-1]

    # 3. 创建模型
    print("Building model...")
    model = DHPRec(args, test_dataset, index, device).to(device)

    # 4. 加载 checkpoint
    print(f"Loading weights: {args.ckpt_path}")
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # 5. 加载 text embedding 到模型（在 checkpoint 之后，避免被覆盖）
    for i in range(len(args.text_types)):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(
            text_embs[i], dtype=torch.float32, device=device
        )
    
    # 5. 核心：筛选样本并画图
    samples = find_diverse_samples(model, test_loader, device, min_seq_len=15, max_samples=5)
    if samples:
        plot_paper_figure(samples, save_path=args.save_path)
    else:
        print("未找到合适样本，请调整筛选条件。")

if __name__ == "__main__":
    main()