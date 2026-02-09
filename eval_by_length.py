"""
按序列长度分组评估模型性能
将测试集按交互序列长度分为5组: [1,20], (20,40], (40,60], (60,80], (80,100]
计算每组的 Recall@5 指标
"""
import os
import argparse
import warnings
import torch
import numpy as np
from collections import defaultdict
from data import load_split_data, MGFSSeqSplitDataset, Collator
from torch.utils.data import DataLoader, Subset
from model import DHPRec
from utils import init_seed, load_json
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="Musical_Instruments")
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--cl_weight', type=float, default=0.1)
    parser.add_argument('--mlm_weight', type=float, default=0.1)
    parser.add_argument('--neg_num', type=int, default=49)
    parser.add_argument('--text_types', nargs='+', type=str, default="meta")
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
    parser.add_argument('--max_his_len', type=int, default=100)  # 使用100
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
    
    # 评估专用参数
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型checkpoint路径")
    parser.add_argument("--topk", type=int, default=5, help="Recall@K 的 K 值")
    parser.add_argument("--num_groups", type=int, default=5, help="分组数量")
    parser.add_argument("--eval_max_len", type=int, default=None, help="评估分组的最大长度，默认使用max_his_len")
    
    args, _ = parser.parse_known_args()
    return args


def compute_recall_at_k(scores, labels, k=5):
    """计算 Recall@K"""
    _, topk_idx = torch.topk(scores, k, dim=-1)
    topk_idx = topk_idx.cpu()
    labels = labels.cpu()
    
    # 检查 label 是否在 topk 中
    hits = (topk_idx == labels.unsqueeze(1)).any(dim=1).float()
    return hits


def get_original_seq_lengths(test_seq):
    """获取原始序列长度（不截断）"""
    lengths = []
    for seq, _ in zip(*test_seq):
        # seq 包含 history + target，所以 history 长度是 len(seq) - 1
        lengths.append(len(seq) - 1)
    return lengths


def generate_bins(max_len, num_groups=5):
    """根据最大长度生成等分的bins"""
    step = max_len // num_groups
    bins = [i * step for i in range(num_groups + 1)]
    bins[-1] = max_len  # 确保最后一个bin是max_len
    return bins


def group_by_length(lengths, bins):
    """按长度分组，返回每组的索引"""
    groups = defaultdict(list)
    for idx, length in enumerate(lengths):
        for i in range(len(bins) - 1):
            if bins[i] < length <= bins[i + 1]:
                group_name = f"({bins[i]},{bins[i+1]}]"
                if i == 0:
                    group_name = f"[1,{bins[i+1]}]"
                groups[group_name].append(idx)
                break
        # 处理超过最大bin的情况
        if length > bins[-1]:
            groups[f">{bins[-1]}"].append(idx)
    return groups


class LengthAwareDataset(MGFSSeqSplitDataset):
    """扩展数据集，保存原始长度信息"""
    def __init__(self, args, n_items, inter_seq, index, original_lengths, mode="test"):
        self.original_lengths = original_lengths
        super().__init__(args, n_items, inter_seq, index, mode)
    
    def __getitem__(self, idx):
        item_inter, target, code_inter, mask_target, session_inter = super().__getitem__(idx)
        return item_inter, target, code_inter, mask_target, session_inter, self.original_lengths[idx]


class LengthAwareCollator(Collator):
    """扩展 Collator，处理长度信息"""
    def __call__(self, batch):
        item_inters, targets, code_inters, mask_targets, session_inters, orig_lengths = zip(*batch)
        
        result = super().__call__(list(zip(item_inters, targets, code_inters, mask_targets, session_inters)))
        result['orig_lengths'] = torch.tensor(orig_lengths)
        return result


@torch.no_grad()
def evaluate_by_length_groups(model, test_loader, device, topk=5, bins=[0, 20, 40, 60, 80, 100]):
    """按长度分组评估"""
    model.eval()
    model.get_item_embedding()
    
    # 存储每组的结果
    group_hits = defaultdict(list)
    group_counts = defaultdict(int)
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        item_inters = batch["item_inters"].to(device)
        inter_lens = batch["inter_lens"].to(device)
        code_inters = batch['code_inters'].to(device)
        labels = batch["targets"].to(device)
        session_ids = batch["session_inters"].to(device)
        orig_lengths = batch["orig_lengths"]
        
        # 获取预测分数
        scores = model.full_sort_predict(item_inters, inter_lens, code_inters, session_ids)
        
        # 计算每个样本的 hit
        hits = compute_recall_at_k(scores, labels, k=topk)
        
        # 按原始长度分组统计
        for i, length in enumerate(orig_lengths.tolist()):
            for j in range(len(bins) - 1):
                if bins[j] < length <= bins[j + 1]:
                    group_name = f"({bins[j]},{bins[j+1]}]"
                    if j == 0:
                        group_name = f"[1,{bins[j+1]}]"
                    group_hits[group_name].append(hits[i].item())
                    group_counts[group_name] += 1
                    break
            if length > bins[-1]:
                group_hits[f">{bins[-1]}"].append(hits[i].item())
                group_counts[f">{bins[-1]}"] += 1
    
    # 计算每组的 Recall@K
    results = {}
    for group_name in sorted(group_hits.keys(), key=lambda x: int(x.split(',')[0].strip('[](>')) if x[0] != '>' else 999):
        if group_counts[group_name] > 0:
            recall = sum(group_hits[group_name]) / group_counts[group_name]
            results[group_name] = {
                'count': group_counts[group_name],
                f'recall@{topk}': recall
            }
    
    return results


def main():
    args = parse_arguments()
    init_seed(args.seed, True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    data_path = args.data_path
    dataset = args.dataset
    dataset_path = os.path.join(data_path, dataset)
    
    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(os.path.join(dataset_path, dataset + args.text_index_path))
    
    # 获取原始序列长度
    original_lengths = get_original_seq_lengths(test)
    print(f"测试集样本数: {len(original_lengths)}")
    print(f"序列长度范围: [{min(original_lengths)}, {max(original_lengths)}]")
    
    # 创建带长度信息的数据集
    test_dataset = LengthAwareDataset(args, n_items, test, index, original_lengths, 'test')
    collator = LengthAwareCollator(args)
    
    test_loader = DataLoader(
        test_dataset, 
        num_workers=args.num_workers, 
        collate_fn=collator,
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # 加载文本嵌入
    text_embs = []
    for ttype in args.text_types:
        text_emb_file = f".t5.{ttype}.emb.npy"
        text_emb = np.load(os.path.join(args.data_path, args.dataset, args.dataset + text_emb_file))
        text_emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(text_emb)
        text_embs.append(text_emb)
    args.text_embedding_size = text_embs[0].shape[-1]
    
    # 初始化模型
    model = DHPRec(args, test_dataset, index, device).to(device)
    
    # 加载文本嵌入到模型
    for i in range(len(args.text_types)):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(
            text_embs[i], dtype=torch.float32, device=device
        )
    
    # 加载训练好的模型权重
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # 从checkpoint中获取训练时的max_his_len
    if "args" in checkpoint:
        train_max_his_len = checkpoint["args"].max_his_len
        print(f"训练时 max_his_len: {train_max_his_len}, 当前设置: {args.max_his_len}")
        if train_max_his_len != args.max_his_len:
            print(f"警告: 使用训练时的 max_his_len={train_max_his_len} 重新初始化模型")
            args.max_his_len = train_max_his_len
            # 重新创建模型
            model = DHPRec(args, test_dataset, index, device).to(device)
            for i in range(len(args.text_types)):
                model.item_text_embedding[i].weight.data[1:] = torch.tensor(
                    text_embs[i], dtype=torch.float32, device=device
                )
    
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Model loaded successfully!")
    
    # 按长度分组评估
    bins = generate_bins(args.max_his_len, args.num_groups)
    print(f"分组区间: {bins}")
    results = evaluate_by_length_groups(model, test_loader, device, topk=args.topk, bins=bins)
    
    # 打印结果
    print("\n" + "=" * 60)
    print(f"分长度评估结果 (Recall@{args.topk})")
    print("=" * 60)
    print(f"{'长度区间':<15} {'样本数量':<12} {f'Recall@{args.topk}':<12}")
    print("-" * 60)
    
    total_count = 0
    total_hits = 0
    for group_name, metrics in results.items():
        count = metrics['count']
        recall = metrics[f'recall@{args.topk}']
        print(f"{group_name:<15} {count:<12} {recall:.4f}")
        total_count += count
        total_hits += recall * count
    
    print("-" * 60)
    overall_recall = total_hits / total_count if total_count > 0 else 0
    print(f"{'Overall':<15} {total_count:<12} {overall_recall:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
