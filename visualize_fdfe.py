import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
import seaborn as sns

from data import load_split_data, MGFSSeqSplitDataset, Collator
from torch.utils.data import DataLoader
from model import DHPRec
from utils import load_json

# ==========================================
# 1. 设置绘图风格
# ==========================================
sns.set_style("whitegrid")

# 加载 Times New Roman 风格字体 (Nimbus Roman)
_font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'times.ttf')
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    _font_prop = fm.FontProperties(fname=_font_path)
    plt.rcParams['font.family'] = _font_prop.get_name()
else:
    plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# ==========================================
# 2. 核心可视化函数 (FDFE Analysis) - 方案一：归一化
# ==========================================
def visualize_fdfe_effect(model, test_loader, device, save_path="./fdfe_analysis.pdf"):
    """
    可视化 FDFE 模块效果：
    (a) 频谱去噪对比 (Normalized View: Offset Alignment)
    (b) 滤波器权重形态 (Normalized View: Max Scaling)
    """
    model.eval()
    
    # 获取序列长度相关的 FFT 参数
    # 注意：这里的 FFT 是沿着 dim=1 (Length) 做的
    # 假设 test_loader 的 batch size 不变，序列长度由 collator 决定
    # 我们先跑一个 batch 看看维度
    sample_batch = next(iter(test_loader))
    sample_seq = sample_batch["item_inters"]
    L = sample_seq.size(1)
    n_freq_bins = L // 2 + 1
    
    # 累积变量
    total_mag_input = np.zeros(n_freq_bins)
    total_mag_filtered = np.zeros(n_freq_bins)
    total_filter_g = np.zeros(n_freq_bins)
    sample_count = 0
    
    # 设定采样批次，跑 20 个 batch 足够得到稳定的平均值了
    max_batches = 20
    print(f"正在深入 FDFE 内部提取数据 (Sampling {max_batches} batches, SeqLen={L})...")
    
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            if i >= max_batches: break
            
            # --- 数据搬运 ---
            item_seq = data_batch["item_inters"].to(device)
            code_seq = data_batch["code_inters"].to(device)
            
            # =======================================================
            # 步骤 1: 手动执行模型前向传播，获取 Item Embedding
            # =======================================================
            B, cur_L = item_seq.size(0), item_seq.size(1)
            # 确保序列长度一致，如果不一致可能会报错，建议固定 max_his_len
            if cur_L != L: continue 

            item_flatten_seq = item_seq.reshape(-1)
            
            query_seq_emb = model.query_code_embedding(code_seq)
            
            text_embs = []
            for j in range(model.text_num):
                text_emb = model.item_text_embedding[j](item_flatten_seq)
                text_embs.append(text_emb)
            encoder_output = torch.stack(text_embs, dim=1)
            
            item_seq_emb = model.qformer(query_seq_emb, encoder_output)[-1]
            
            # FDFE 的输入 Item Embedding [B, L, H]
            item_emb = item_seq_emb.mean(dim=1) + query_seq_emb.mean(dim=1)
            item_emb = item_emb.view(B, L, -1)
            
            # =======================================================
            # 步骤 2: 模拟 FDFE 内部计算
            # =======================================================
            
            # (A) FFT 变换 (dim=1, Length)
            fft_input = torch.fft.rfft(item_emb, dim=1, norm='ortho') # [B, Freqs, H]
            mag_input = torch.abs(fft_input) 
            
            # (B) 计算自适应权重 G (模拟 FrequencyAttention 内部)
            # freq_proj: [B, Freqs, H] -> [B, Freqs, H]
            filter_logits = model.fourier_attention.freq_proj(mag_input)
            filter_g = torch.softmax(filter_logits, dim=1) 
            
            # (C) 应用滤波
            fft_filtered = fft_input * filter_g
            mag_filtered = torch.abs(fft_filtered)

            # ======================================================= 63 
            # 步骤 3: 累积统计量
            # 对 Batch(0) 和 Hidden(2) 维度取平均，保留 Freq(1) 维度 -> [Freqs]
            # =======================================================
            batch_mag_in = mag_input.mean(dim=(0, 2)).cpu().numpy()
            batch_mag_out = mag_filtered.mean(dim=(0, 2)).cpu().numpy()
            batch_g = filter_g.mean(dim=(0, 2)).cpu().numpy()
            
            total_mag_input += batch_mag_in
            total_mag_filtered += batch_mag_out
            total_filter_g += batch_g
            
            sample_count += 1
            
    # 计算全局平均
    avg_mag_input = total_mag_input / sample_count
    avg_mag_filtered = total_mag_filtered / sample_count
    avg_filter_g = total_filter_g / sample_count
    
    # 频率轴索引
    freqs = np.arange(len(avg_mag_input))

    # =======================================================
    # 步骤 4: 数据后处理 (方案一：归一化与对齐)
    # =======================================================
    
    # A. 归一化 G 到 [0, 1]
    g_max = avg_filter_g.max()
    avg_filter_g_norm = avg_filter_g / (g_max + 1e-9)
    
    # B. 对齐频谱的低频能量
    # 计算对数幅度
    log_mag_before = np.log(avg_mag_input + 1e-8)
    log_mag_after = np.log(avg_mag_filtered + 1e-8)
    
    # 计算 Offset：让 Refined 在 Frequency=0 处与 Original 对齐
    # 这样直观展示：低频保留，高频下降
    offset = log_mag_before[0] - log_mag_after[0]
    log_mag_after_shifted = log_mag_after + offset

    # =======================================================
    # 步骤 5: 绘图 - 分开生成两个纯净的SVG图
    # =======================================================
    print("开始绘图 (Normalized View)...")
    
    # 根据 save_path 生成两个文件名
    base_path = save_path.rsplit('.', 1)[0]  # 去掉扩展名
    save_path_a = base_path + "_spectrum.svg"
    save_path_b = base_path + "_filter.svg"
    
    # --- 图1: 对数幅度谱对比 (纯净版) ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    
    ax1.plot(freqs, log_mag_before, 
             label='Original Input', color='#1f77b4', linestyle='--', linewidth=2.5, alpha=0.8)
    
    ax1.plot(freqs, log_mag_after_shifted, 
             label='Refined (After FDFR)', color='#d62728', linewidth=2.5)
    
    ax1.fill_between(freqs, 
                     log_mag_before, 
                     log_mag_after_shifted,
                     where=(log_mag_before > log_mag_after_shifted),
                     color='gray', alpha=0.2, label='Suppressed Noise')
    
    ax1.set_xlim(0, len(freqs) - 1)
    ax1.legend(loc='upper right', fontsize=13, frameon=True, shadow=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 突出显示坐标轴刻度数值
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    plt.tight_layout()
    plt.savefig(save_path_a, format='svg', bbox_inches='tight')
    print(f"图1 (频谱) 已保存至: {save_path_a}")
    plt.close(fig1)
    
    # --- 图2: 滤波器曲线 (纯净版) ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    
    ax2.plot(freqs, avg_filter_g_norm, color='#2ca02c', linewidth=3.5)
    
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(0, len(freqs) - 1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 突出显示坐标轴刻度数值
    ax2.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    plt.tight_layout()
    plt.savefig(save_path_b, format='svg', bbox_inches='tight')
    print(f"图2 (滤波器) 已保存至: {save_path_b}")
    plt.close(fig2)


def parse_arguments():
    # 保留你原来的参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="Musical_Instruments")
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--cl_weight', type=float, default=0.4)
    parser.add_argument('--mlm_weight', type=float, default=0.6)
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
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--dropout_prob_cross', type=float, default=0.3)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--metrics', type=str, default="recall@5,ndcg@5,recall@10,ndcg@10")
    parser.add_argument('--valid_metric', type=str, default="ndcg@10")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--ckpt_dir", type=str, default="./myckpt/")
    
    # 可视化专用参数
    parser.add_argument("--ckpt_path", type=str, default="./myckpt/best_model.pth", help="训练好的模型checkpoint路径")
    parser.add_argument("--save_path", type=str, default="./fdfe_analysis.pdf", help="保存图片路径")
    
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    data_path = args.data_path
    dataset = args.dataset
    dataset_path = os.path.join(data_path, dataset)
    
    # 加载数据集
    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(os.path.join(dataset_path, dataset + args.text_index_path))
    
    # 准备 Test Loader
    # 这里的 batch_size 设为 32，你可以根据显存调整
    test_dataset = MGFSSeqSplitDataset(args, n_items, test, index, 'test')
    collator = Collator(args)
    test_loader = DataLoader(test_dataset, num_workers=0, collate_fn=collator,
                             batch_size=32, shuffle=False)
    
    # 2. 准备 Text Embedding
    text_embs = []
    for ttype in args.text_types:
        text_emb_file = f".t5.{ttype}.emb.npy"
        text_emb = np.load(os.path.join(args.data_path, args.dataset, args.dataset + text_emb_file))
        # 保持和训练时一致的 PCA 处理
        text_emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(text_emb)
        text_embs.append(text_emb)
    args.text_embedding_size = text_embs[0].shape[-1]
    
    # 3. 初始化模型
    print("Initializing model...")
    model = DHPRec(args, test_dataset, index, device).to(device)
    
    # 赋值 Text Embedding
    for i in range(len(args.text_types)):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(text_embs[i], dtype=torch.float32, device=device)
    
    # 4. 加载权重
    if os.path.exists(args.ckpt_path):
        print(f"Loading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        # 处理 state_dict 键名
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {args.ckpt_path} not found! Using random weights.")
    
    # 5. 运行可视化
    visualize_fdfe_effect(model, test_loader, device, save_path=args.save_path)

if __name__ == "__main__":
    main()