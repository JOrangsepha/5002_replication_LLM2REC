#!/bin/bash

# 1. 定义模型配置 (信息名称, 模型路径)
models_list=(
    "IEM_500 ./step3/checkpoints/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-500"
    "IEM_1000 ./step3/checkpoints/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-1000"
)

# 2. 定义数据集 (数据集名称, 指定GPU)
datasets=(
    "Games_5core 0"
    "Arts_5core 1"
    "Movies_5core 0"
    "Sports_5core 1"
    "Baby_5core 0"
    "Goodreads 1"
)

# 定义脚本所在路径
STEP3_DIR="./step3"

# 循环遍历数据集
for dataset_setting in "${datasets[@]}"
do
    # 提取数据集名称和GPU ID
    dataset=$(echo $dataset_setting | awk '{print $1}')
    cuda_device=$(echo $dataset_setting | awk '{print $2}')

    # 循环遍历模型权重
    for model_setting in "${models_list[@]}"
    do
        save_info=$(echo $model_setting | awk '{print $1}')
        model_path=$(echo $model_setting | awk '{print $2}')

        echo "=========================================================="
        echo "🚀 开始任务: 数据集=$dataset | 权重=$save_info | 显卡=$cuda_device"
        echo "=========================================================="

        # --- 步骤 1: 提取 Embedding ---
        # 确保输出目录存在
        mkdir -p "${STEP3_DIR}/item_info/${dataset}"
        
        echo ">>> [Step 1/2] 正在提取 LLM Embedding..."
        CUDA_VISIBLE_DEVICES=$cuda_device python ${STEP3_DIR}/get_embedding.py \
            --dataset=$dataset \
            --model_path=$model_path \
            --save_info=$save_info

        # --- 步骤 2: 下游推荐评估 ---
        # 拼接刚才生成的 embedding 路径
        # 假设 get_embedding.py 生成的文件名格式为 {save_info}_title_item_embs.npy




        embs="${STEP3_DIR}/item_info/${dataset}/${save_info}_title_item_embs.npy"

        # 定义 SASRec 超参数
        lr=1.0e-3
        wd=1.0e-4
        model="SASRec"
        dr=0.3
        bs=256

        echo ">>> [Step 2/2] 正在进行下游模型评估 (SASRec)..."
        
        
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python ${STEP3_DIR}/evaluation.py \
            --model=$model \
            --dataset=$dataset \
            --lr=$lr \
            --weight_decay=$wd \
            --embedding=$embs \
            --dropout=$dr \
            --batch_size=$bs


        

        echo "✅ 完成任务: $dataset ($save_info)"
        echo "----------------------------------------------------------"
    done
done

echo "🎉 所有实验已全部跑完！"