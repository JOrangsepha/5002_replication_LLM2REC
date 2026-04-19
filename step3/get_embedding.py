import torch
import json
import os
import os.path as op
import numpy as np
import argparse
from tqdm import tqdm
from llm2vec import LLM2Vec

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)

# 保持原有的映射关系，这些是根目录下 data/ 文件夹内的子路径
dataset_name_mappings = {
    "Games_5core": "Video_Games/5-core/downstream",
    "Movies_5core": "Movies_and_TV/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
    "Sports_5core": "Sports_and_Outdoors/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
    "Goodreads": "Goodreads/clean",
}

def load_item_titles(dataset_name, base_data_dir):
    """加载商品标题并处理 ID 对齐"""
    if dataset_name not in dataset_name_mappings:
        print(f"错误: 数据集 {dataset_name} 不在映射表中")
        return None
        
    raw_path = dataset_name_mappings[dataset_name]
    # 现在指向根目录下的 data/...
    json_path = op.join(base_data_dir, raw_path, "item_titles.json")
    
    if not os.path.exists(json_path):
        print(f"跳过 {dataset_name}: 找不到文件 {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        item_metadata = json.load(f)

    # 推荐系统索引从 1 开始，0 位留给 Padding
    item_ids = [int(k) for k in item_metadata.keys()]
    max_id = max(item_ids)
    
    item_titles = ["Null"] # 0号占位符
    for i in range(1, max_id + 1):
        # 保证索引连续性
        item_titles.append(item_metadata.get(str(i), "Null"))
    
    return item_titles

def extract_embeddings(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载 LLM2Vec 模型
    print(f">>> 正在加载 Checkpoint: {args.model_path}")
    l2v = LLM2Vec.from_pretrained(
        args.model_path,
        peft_model_name_or_path=None, 
        device_map=device,
        torch_dtype=torch.float16,    
        enable_bidirectional=True,     
        pooling_mode="mean"           
    )

    # 2. 准备保存目录 (依然保存在 step3 目录下，方便管理)
    # 获取脚本所在目录的绝对路径，确保在根目录运行时也能存对位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_root = os.path.join(script_dir, "item_info")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 3. 循环处理数据集
    target_datasets = [args.dataset] if args.dataset else dataset_name_mappings.keys()

    for ds_name in target_datasets:
        print(f"\n开始处理数据集: {ds_name}")
        # base_data_dir 传入的是根目录下的 data
        titles = load_item_titles(ds_name, args.base_data_dir)
        if not titles: continue

        # 生成 Prompt
        if args.prompt_type == "direct":
            instruct = "To recommend this item to users, this item can be described as: "
            prompts = [instruct + str(t) for t in titles]
        else:
            prompts = titles

        # 4. 执行推理
        print(f"正在提取 Embedding (Total: {len(prompts)} items)...")
        embeddings = l2v.encode(prompts, batch_size=args.batch_size)

        # 5. 保存结果
        ds_save_path = op.join(save_root, ds_name)
        if not os.path.exists(ds_save_path):
            os.makedirs(ds_save_path)
        

        save_file = op.join(ds_save_path, f"{args.save_info}_{args.prompt_type}_item_embs.npy")
        np.save(save_file, embeddings)
        print(f"成功保存至: {save_file} | Shape: {embeddings.shape}")

        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_dir', type=str, default="./data")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--prompt_type', type=str, choices=['title', 'direct'], default='title')
    parser.add_argument('--save_info', type=str, required=True)
    
    args = parser.parse_args()
    extract_embeddings(args)