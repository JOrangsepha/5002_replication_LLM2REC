import argparse
import datetime
import numpy as np
import torch
import os
import json
import sys

from seqrec.runner import Runner
from seqrec.utils import parse_command_line_args

STEP3_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["WANDB_MODE"] = "disabled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec')
    parser.add_argument('--dataset', type=str, default='Games_5core')
    parser.add_argument('--exp_type', type=str, default='srec')
    parser.add_argument('--embedding', type=str, default='')
    return parser.parse_known_args()

def calculate_mean_and_std(results):
    metrics = {}
    for result in results:
        for key, value in result.items():
            metrics.setdefault(key, []).append(value)
    stats = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in metrics.items()}
    return stats

if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)
    merged_dict = {**args_dict, **command_line_configs}

    # --- 核心路径对齐 ---
    # 强制将数据路径指向 step3/data
    merged_dict['data_path'] = os.path.join(STEP3_DIR, 'data')
    
    # 如果 embedding 传入的是相对路径，尝试基于当前执行路径或 step3 路径补全
    if not os.path.isabs(merged_dict['embedding']):
        # 优先检查是否相对于当前执行目录存在，否则检查是否在 step3 下
        if not os.path.exists(merged_dict['embedding']):
            merged_dict['embedding'] = os.path.join(STEP3_DIR, merged_dict['embedding'].lstrip('./'))

    print(f">>> 检查 Embedding 路径: {merged_dict['embedding']}")
    if not os.path.exists(merged_dict['embedding']):
        print(f"错误：找不到 Embedding 文件 {merged_dict['embedding']}")
        sys.exit(1)

    exp_seeds = [2024, 2025, 2026]
    test_results = []
    
    for seed in exp_seeds:
        print(f"\n[Seed {seed}] 正在初始化 Runner...")
        merged_dict['rand_seed'] = seed
        runner = Runner(model_name=args.model, config_dict=merged_dict)
        test_result, exp_config = runner.run()
        test_results.append(test_result)

    stats = calculate_mean_and_std(test_results)

    # --- 保存结果 ---
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    save_dir = os.path.join(STEP3_DIR, 'Results', exp_config['dataset'], f"Eval_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(f"Stats: {stats}\n\nConfig: {json.dumps(merged_dict, indent=2)}")

    print(f"\n实验完成！\n指标汇总 (Mean ± Std): {stats}\n结果已保存至: {save_dir}")