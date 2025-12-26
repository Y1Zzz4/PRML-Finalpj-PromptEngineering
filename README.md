# PRML Final Project: Prompt Engineering for ARC-AGI Tasks

本项目使用提示词工程（Prompt Engineering）技术，驱动大语言模型（DeepSeek-V3）解决 ARC（Abstraction and Reasoning Corpus）抽象推理任务。

## 项目结构
- `src/`：源代码
  - template.py：construct_prompt 和 parse_output 实现
  - test_prompt.py：评测脚本
- `data/`：数据集
  - val.jsonl：30 条验证集
  - val_hard.jsonl：120 条更难数据集（加分项）
- `reports/`：实验报告
- `visuals/`：网格可视化图片
- `docs/`：文档

## 环境设置
使用 Conda 环境：
```bash
conda create -n prml_prompt python=3.10
conda activate prml_prompt