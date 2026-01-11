# PRML Final Project: Prompt Engineering for ARC-AGI Tasks

本项目使用提示词工程（Prompt Engineering）技术，驱动大语言模型（DeepSeek-V3）解决 ARC（Abstraction and Reasoning Corpus）抽象推理任务。

## 项目结构
- `inference/`：推理文件夹
  - run_inference.py     # 主推理脚本：加载数据、构建 prompt、调用 DeepSeek API、保存预测结果
- `evaluation/`: 评测文件夹
  - evaluate.py          # 评估脚本：比较预测和 ground truth，输出准确率和错误任务列表
- `visualization/`:可视化文件夹
  - visualize_cases.py   # 可视化脚本：绘制输入、预测、真实输出网格，支持显示或保存 PNG
- `utils/`:工具函数文件夹
  - parse.py             # 解析函数：从模型输出文本中提取预测网格 (parse_output)
- `prompts/`:提示策略文件夹，每个 .py 文件实现一种提示策略的 construct_prompt 函数
  - baseline.py                     # 基线策略
  - strategy_implicit_cot.py        # 隐式思维链
  - strategy_visual_cot.py          # 显式思维链 + 回环验证
  - strategy_reflection.py          # 自我反思
  - strategy_structured.py          # 结构函数化
  - strategy_hypothesis_search.py   # 假设验证
- `data/`：数据集
  - val.jsonl：30 条验证集
  - val_hard.jsonl：120 条更难数据集
- `results/`:结果文件夹，存放每个策略的推理输出 JSON 文件
  - baseline_val.json    # 示例：baseline 策略在 val.jsonl 上的预测结果
- `visuals/`：可视化图片文件夹
  - task_00_baseline.png # 示例：保存的单个任务可视化图片
- `.env`:环境变量文件
- `requirements.txt`：依赖列表
- `README.md`：本文档,项目说明
- `.gitignore`：Git 忽略文件
- `LICENSE`：MIT License


## 环境设置
项目使用 Python 3.10+，推荐 Conda 环境管理。
```bash
conda create -n prml_prompt python=3.10
conda activate prml_prompt
```

## .env配置（根目录）
```
DEEPSEEK_API_KEY=sk-your-real-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```


## 使用方法（根目录下）
1、推理（已在本地完成）
```
  python inference/run_inference.py --strategy '策略名或all'（必须） --dataset '数据集名'（默认val）
```

2、评测
```
   python evaluation/evaluate.py --pred '单个结果文件路径'或'all'(all表示评测'results/'下所有结果文件)（必须） --val '原数据文件路径'（默认'data/val.jsonl'）
```

3、可视化
```
  python visualization/visualize_cases.py --strategy '策略名'（必须） --task_id '任务索引'（0-29）（必须） --save（可选，是否保存为图片）--output_dir '保存路径'（可选，默认'visuals_results'）
```

注：我提交了所有本地推理得到的结果.json文件，git clone后只需要运行
  ```
  python .\evaluation\evaluate.py --pred all
  ```
即可进行批量评测。