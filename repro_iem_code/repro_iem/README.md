# IEM Reproduction (LLM2Rec)

这个目录提供一套独立的 IEM 复现实现，目标是尽量 1:1 对齐 `LLM2Rec-main` 第二阶段行为。

## 复现范围

IEM 在原仓库中实际由两步组成：

1. `MNTP`：把第一阶段 CSFT 模型继续训练成双向 item/text encoder。
2. `SimCSE`：在 item title 对比学习上继续优化 embedding 空间。

本目录分别对应：

- `train_mntp_repro.py`：独立 MNTP 训练入口
- `train_simcse_repro.py`：独立 SimCSE 训练入口
- `dataset_registry.py`：数据集注册入口
- `recdata/`：`ItemTitles`、`ItemRec`、`SeqRec` 三类数据集实现
- `prepare_simcse_checkpoint.py`：在 MNTP checkpoint 上补齐 tokenizer 文件，行为对齐原始 shell 脚本里的 `cp *token*`
- `run_iem_repro.sh`：一键运行脚本

## 快速开始

在 `repro_iem_code` 目录下执行：

```bash
bash repro_iem/run_iem_repro.sh
```

只跑 MNTP：

```bash
bash repro_iem/run_iem_repro.sh mntp
```

只跑 SimCSE：

```bash
bash repro_iem/run_iem_repro.sh simcse
```
