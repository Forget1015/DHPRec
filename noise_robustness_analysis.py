"""
DHP-Rec 鲁棒性分析脚本 (Robustness to Noise)
该脚本在推理阶段向 Item Embedding 注入不同比例的高斯噪声，
观察模型性能 (NDCG@10) 的下降趋势。
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入项目模块
from data import load_split_data, MGFSSeqSplitDataset, Collator
from model import DHPRec
from utils import load_json

# === 绘图风格设置 ===
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

def parse_arguments():
    parser = argparse.ArgumentParser()
    # 基础参数 (保持与训练一致)
    parser.add_argument("--dataset", type=str, default="Musical_Instruments")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # 数据路径参数 (从 main.py 添加)
    parser.add_argument('--map_path', type=str, default=".emb_map.json")
    parser.add_argument('--text_index_path', type=str, default=".code.pq.64_128.json")
    parser.add_argument('--text_emb_path', type=str, default=".t5.meta.emb.npy")
    
    # 模型超参数 (根据你的最优配置修改默认值)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_layers_cross', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--dropout_prob_cross', type=float, default=0.3)
    parser.add_argument('--text_types', nargs='+', type=str, default=["meta"])
    parser.add_argument('--max_his_len', type=int, default=50)
    parser.add_argument('--n_codes_per_lel', type=int, default=256)
    parser.add_argument('--code_level', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.07)
    
    # 训练相关参数 (虽然不用于训练，但模型/数据加载可能需要)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cl_weight', type=float, default=0.1)
    parser.add_argument('--mlm_weight', type=float, default=0.1)
    parser.add_argument('--neg_num', type=int, default=49)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--bidirectional', type=bool, default=False)
    
    # 实验专用参数
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的模型权重路径")
    parser.add_argument("--save_fig", type=str, default="./robustness_analysis.pdf")
    parser.add_argument("--batch_size", type=int, default=1024, help="推理时的Batch Size")
    
    args, _ = parser.parse_known_args()
    return args

def evaluate(model, test_loader, device, k_list=[10]):
    """标准评估函数，使用与 trainer.py 一致的逻辑"""
    model.eval()
    
    total = 0
    metrics = {f'ndcg@{k}': 0 for k in k_list}
    metrics.update({f'recall@{k}': 0 for k in k_list})
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            item_seq = batch['item_inters'].to(device)
            code_seq = batch['code_inters'].to(device)
            item_seq_len = batch['inter_lens'].to(device)
            session_ids = batch['session_inters'].to(device)
            labels = batch['targets'].to(device)

            # 获取所有物品的预测分数
            scores = model.full_sort_predict(item_seq, item_seq_len, code_seq, session_ids)
            
            batch_size = labels.size(0)
            total += batch_size
            
            for k in k_list:
                # 获取 Top-K
                _, topk_idx = torch.topk(scores, k, dim=-1)  # [B, K]
                
                # 创建 one-hot labels
                one_hot_labels = torch.zeros_like(scores)
                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
                
                # 获取 top-k 位置的 label
                top_k_labels = torch.gather(one_hot_labels, dim=1, index=topk_idx)  # [B, K]
                
                # Recall@K
                recall = top_k_labels.sum(dim=1)  # [B]
                metrics[f'recall@{k}'] += recall.sum().item()
                
                # NDCG@K
                # DCG = sum(rel_i / log2(i+1)) for i in 1..K
                positions = torch.arange(1, k + 1, device=device).float()
                dcg = (top_k_labels / torch.log2(positions + 1)).sum(dim=1)  # [B]
                # IDCG = 1 / log2(2) = 1 (因为只有一个正样本)
                idcg = 1.0
                ndcg = dcg / idcg
                metrics[f'ndcg@{k}'] += ndcg.sum().item()

    # 计算平均值
    for m in metrics:
        metrics[m] = metrics[m] / total
    
    return metrics.get('ndcg@10', 0)

def inject_noise_to_embeddings(model, std, device):
    """
    向模型的 Embedding 层注入高斯噪声
    注意：MGFS-Rec 有 Text Embedding 和 Code Embedding，都要加
    """
    if std == 0.0:
        return
        
    print(f"Injecting Gaussian Noise (std={std})...")
    with torch.no_grad():
        # 1. 注入 Code Embedding
        noise_code = torch.normal(mean=0.0, std=std, size=model.query_code_embedding.weight.size()).to(device)
        model.query_code_embedding.weight.add_(noise_code)
        
        # 2. 注入 Text Embedding (ModuleList)
        for i in range(len(model.item_text_embedding)):
            noise_text = torch.normal(mean=0.0, std=std, size=model.item_text_embedding[i].weight.size()).to(device)
            model.item_text_embedding[i].weight.add_(noise_text)

def restore_embeddings(model, original_state_dict):
    """恢复原始权重"""
    # 只恢复 Embedding 部分即可，这里为了简单直接 load 之前的 state_dict
    # 但为了效率，最好只深拷贝 Embedding 层。为了代码简单，这里重新 load 权重
    # 或者我们可以在注入前手动减去噪声
    # 这里选择重新 load state_dict 的部分 key (更安全)
    with torch.no_grad():
        model.query_code_embedding.weight.data.copy_(original_state_dict['query_code_embedding.weight'])
        for i in range(len(model.item_text_embedding)):
            key = f'item_text_embedding.{i}.weight'
            model.item_text_embedding[i].weight.data.copy_(original_state_dict[key])

def main():
    args = parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. 数据准备
    print("Loading Data...")
    item2id, n_items, train, val, test = load_split_data(args)
    dataset_path = os.path.join(args.data_path, args.dataset)
    index_raw = load_json(os.path.join(dataset_path, args.dataset + args.text_index_path))
    
    # 确保 index 是 list 格式 (如果是 dict，需要转换)
    if isinstance(index_raw, dict):
        # 假设 key 是字符串形式的 item id
        max_id = max(int(k) for k in index_raw.keys()) + 1
        index = [[0] * args.code_level for _ in range(max_id)]
        for k, v in index_raw.items():
            index[int(k)] = v
    else:
        index = index_raw
    
    test_dataset = MGFSSeqSplitDataset(args, n_items, test, index, 'test')
    collator = Collator(args)
    test_loader = DataLoader(test_dataset, num_workers=4, collate_fn=collator, batch_size=args.batch_size, shuffle=False)
    
    # 2. 模型初始化
    print("Building Model...")
    # 设置 text_embedding_size (与 main.py 保持一致)
    args.text_embedding_size = args.embedding_size
    model = DHPRec(args, test_dataset, index, device).to(device)

    # 3. 加载训练好的权重 (包含所有 embedding)
    print(f"Loading Checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully!")
    
    # 备份原始 Embedding 权重 (用于恢复)
    original_embeddings = {
        'query_code_embedding.weight': model.query_code_embedding.weight.detach().clone(),
    }
    for i in range(len(model.item_text_embedding)):
        original_embeddings[f'item_text_embedding.{i}.weight'] = model.item_text_embedding[i].weight.detach().clone()
    
    # 5. 定义噪声实验
    noise_scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results_ours = []
    
    print("\n========== 开始鲁棒性测试 ==========")
    for std in noise_scales:
        # A. 注入噪声
        inject_noise_to_embeddings(model, std, device)
        
        # B. 评估
        score = evaluate(model, test_loader, device)
        results_ours.append(score)
        print(f"Noise Scale: {std} | NDCG@10: {score:.4f}")
        
        # C. 恢复原始权重 (清除噪声)
        restore_embeddings(model, original_embeddings)
    
    # 6. 绘图
    # 注意：为了对比，你需要手动填入 Baseline (如 SASRec/CCFRec) 在相同条件下的数据
    # 这里我用模拟数据作为 Baseline 占位符，你需要替换成真实跑出来的数据
    
    # === 模拟 Baseline 数据 (请替换为真实数据!) ===
    # 假设 Baseline 在 0 噪声时略低或差不多，但随着噪声增加下降很快
    base_score = results_ours[0] * 0.95 # 假设 Baseline 弱一点
    results_baseline = [base_score * (1 - 0.5 * x) for x in noise_scales] # 模拟快速下降
    
    plt.figure(figsize=(8, 6))
    
    # 绘制本模型曲线
    plt.plot(noise_scales, results_ours, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='DHP-Rec (Ours)')
    
    # 绘制 Baseline 曲线 (请填入你的真实 Baseline 数据)
    # plt.plot(noise_scales, results_baseline, 's--', color='gray', linewidth=2, markersize=8, label='Baseline (e.g., CCFRec)')
    
    plt.xlabel(r'Noise Standard Deviation ($\sigma$)', fontsize=16)
    plt.ylabel('NDCG@10', fontsize=16)
    plt.title('Robustness to Feature Noise', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 设置Y轴范围稍微宽一点
    min_val = min(min(results_ours), min(results_baseline))
    max_val = max(max(results_ours), max(results_baseline))
    plt.ylim(min_val * 0.9, max_val * 1.05)
    
    plt.tight_layout()
    plt.savefig(args.save_fig, dpi=300)
    print(f"\n结果图已保存至: {args.save_fig}")
    print("DHP-Rec Results:", results_ours)

if __name__ == "__main__":
    main()