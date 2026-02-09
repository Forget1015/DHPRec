import os
import json
import argparse
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--data_path', type=str, default="./dataset")
    # text_types 默认设为 title，你可以通过命令行传入多个，如 --text_types title description
    parser.add_argument('--text_types', type=str, nargs="+", default=["title"])

    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset
    dataset_path = os.path.join(args.data_path, dataset)
    
    # 1. 设置设备字符串
    if torch.cuda.is_available():
        target_device = f"cuda:{args.gpu_id}"
        print(f"Using GPU: {target_device}")
    else:
        target_device = "cpu"
        print("CUDA not available, using CPU.")
    
    # 读取 Meta 数据
    meta_path = os.path.join(dataset_path, f'{dataset}.meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found at {meta_path}")
        
    meta_text_dict = json.load(open(meta_path, 'r'))
    attr_list = args.text_types

    # 2. 优化：将模型加载移到循环外部 (避免重复加载模型)
    # 直接在初始化时传入 device参数，确保模型直接加载到指定显卡
    print("Loading model...")
    text_embedding_model = SentenceTransformer('sentence-t5-base', device=target_device)
    
    for attr in attr_list:
        print(f"Processing attribute: {attr}")
        
        # 检查属性是否存在 (取第一个 key 检查即可)
        first_key = next(iter(meta_text_dict))
        if attr not in meta_text_dict[first_key].keys():
            raise ValueError(f"Attribute '{attr}' not found in meta data.")
        
        # 排序并提取文本
        sorted_text = [v[attr] for k, v in sorted(meta_text_dict.items(), key=lambda x: int(x[0]))] 
        
        # 3. 编码
        # SentenceTransformer 的 encode 方法会自动使用模型所在的 device，
        # 但显式传入 device 参数更保险
        embs = text_embedding_model.encode(
            sorted_text, 
            convert_to_numpy=True, 
            batch_size=256, 
            show_progress_bar=True,
            device=target_device 
        )

        print(f"Embedding shape: {embs.shape}")
        
        # 4. 修复路径拼接问题 (使用 os.path.join 或者 f-string 确保路径分隔符正确)
        save_path = os.path.join(dataset_path, f'{dataset}.t5.{attr}.emb.npy')
        np.save(save_path, embs)
        print(f"Saved to {save_path}")