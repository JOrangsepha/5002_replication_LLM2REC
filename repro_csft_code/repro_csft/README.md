# CSFT Reproduction (LLM2Rec)

这个目录提供一套独立的 CSFT（Collaborative Supervised Fine-Tuning）复现代码，不会改动原仓库的 `llm2rec/run_csft.py`。

## 文件说明

- `train_csft.py`：CSFT 训练主脚本（HuggingFace Trainer）。
- `csft_dataset.py`：按 LLM2Rec 的 CSFT 需求构造训练样本。
- `csft_config.example.json`：默认配置（原版纯历史 prompt：`{history}`）。
- `csft_config.instruction_prompt.json`：指令模板配置（Instruction 风格 prompt）。
- `csft_config.strict_1to1.json`：尽量贴近原版脚本行为的严格复现配置。
- `csft_config.strict_small.json`：小数据快速复现配置（默认 5 万 train + 1 万 valid）。
- `create_small_csft_dataset.py`：从大 CSV 生成可复现小数据集的脚本。
- `run_csft_repro.sh`：一键启动脚本。

## 设计要点（对齐 CSFT）

- 输入：用户历史交互标题序列（`history_item_title`）。
- 输出：下一物品标题（`item_title`）。
- 损失：默认只在目标输出上计算（`train_on_prompt=false`），模拟指令微调常见做法。
- 支持：全参数微调或 LoRA 微调（可选）。

## 快速开始

0. 安装依赖（`peft` 仅在 LoRA 模式下需要）：  
`pip install torch transformers pandas`  
`pip install peft`  (可选)

1. 修改配置（模型路径、输出目录、batch size 等）：

```bash
vim repro_csft/csft_config.example.json
```

2. 选择模板并启动训练（默认 `original`）：

```bash
# 原版纯历史 prompt（默认）
bash repro_csft/run_csft_repro.sh

# 显式指定原版纯历史 prompt
bash repro_csft/run_csft_repro.sh original

# 切换到指令模板 prompt
bash repro_csft/run_csft_repro.sh instruction

# 尽量 1:1 对齐原版行为（推荐做主复现实验）
bash repro_csft/run_csft_repro.sh strict

# 小数据快速复现（先生成 mini 数据集）
bash repro_csft/run_csft_repro.sh strict_small
```

`strict_small` 模式会在检测不到 mini 数据时自动执行 `create_small_csft_dataset.py`。

3. 多卡运行（示例）：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 repro_csft/train_csft.py repro_csft/csft_config.example.json
```

## 与原始脚本相比的改进

- 数据处理更安全：使用 `ast.literal_eval` 解析历史列表，避免 `eval`。
- 明确的 CSFT 样本构造与标签掩码逻辑。
- 参数化更完整：数据列名、样本抽样、历史长度、LoRA 等均可配置。
- 训练入口统一为 JSON/CLI，便于复现实验与记录。

## 严格复现说明

`strict` 模式下，关键行为会尽量对齐原版（同时保持代码为独立实现）：

- 原版纯历史 prompt：`{history}`
- `tokenizer_padding_side = left`
- `max_steps=10000`，`eval_steps=save_steps=2000`
- `logging_steps=1`，`warmup_steps=200`
- early stopping patience = 5
- 支持 `train_from_scratch`
- 支持 NCCL 兼容开关（禁用 P2P/IB/GDR）
- 按 `global_batch_size/micro_batch_size/world_size` 自动对齐 `gradient_accumulation_steps`

## 小数据集构建

如果你本地重训太慢，可以先构建 mini 数据集（默认 50000/10000）：

```bash
python repro_csft/create_small_csft_dataset.py
```

自定义规模示例：

```bash
python repro_csft/create_small_csft_dataset.py \
  --train_rows 30000 \
  --valid_rows 5000 \
  --seed 42
```
