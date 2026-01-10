import json
import os
import argparse
import importlib
from pathlib import Path
from typing import Callable, Dict, Optional

import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import requests
from dotenv import load_dotenv

# 解析函数
from utils.parse import parse_output

# 加载配置
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = os.getenv("DEEPSEEK_BASE_URL")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")

if not API_KEY:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
if not API_URL:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_BASE_URL")
if not MODEL_NAME:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_MODEL")

PROMPT_DIR = Path("prompts")

def discover_strategies() -> Dict[str, Callable]:
    """动态发现 prompts/ 目录下的策略模块"""
    strategies = {}
    for file in PROMPT_DIR.glob("*.py"):
        if file.name == "__init__.py":  # 跳过 __init__.py
            continue
        module_name = file.stem  # 如 baseline, strategy_cot, strategy_reflection
        try:
            module = importlib.import_module(f"prompts.{module_name}")
            if hasattr(module, "construct_prompt"):
                display_name = module_name.replace("strategy_", "")
                strategies[display_name] = module.construct_prompt
        except ImportError as e:
            print(f"警告: 无法加载策略 {module_name}: {e}")
    return strategies


STRATEGY_MAP: Dict[str, Callable] = discover_strategies()  # 自动发现


def call_deepseek(messages: list) -> Optional[str]:
    """调用 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,            # 从 .env 读取模型名
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 8000,
    }
    
    try:
        full_url = API_URL.rstrip("/") + "/chat/completions" 
        response = requests.post(full_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API 调用失败: {e}")
        return None
    except KeyError:
        print("API 返回格式异常")
        return None


def process_single_task(task: dict, construct_prompt: Callable) -> dict:
    """处理单个 ARC 任务"""
    messages = construct_prompt(task)
    
    print("正在调用模型...")
    raw_output = call_deepseek(messages)
    
    predicted_grid = parse_output(raw_output) if raw_output else None
    
    if raw_output:
        print("模型原始输出（前500字符）：")
        print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
    
    return {
        "task_id": task.get("id", "unknown"),
        "messages": messages,
        "raw_output": raw_output,
        "predicted_grid": predicted_grid,
        "ground_truth": task["test"][0]["output"]
    }


def run_strategy(strategy_name: str, construct_prompt: Callable, dataset: str, limit: Optional[int], output_dir: str):
    """运行单个策略的推理"""
    print(f"\n=== 开始运行策略: {strategy_name} ===")
    
    # 数据路径
    data_path = Path("data") / f"{dataset}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"数据集不存在: {data_path}")
    
    # 输出目录和文件
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    output_file = output_dir_path / f"{strategy_name}_{dataset}.json"
    
    results = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit is not None and line_idx >= limit:
                break
                
            task = json.loads(line.strip())
            result = process_single_task(task, construct_prompt)
            result["strategy"] = strategy_name
            results.append(result)
            
            print(f"完成任务 {line_idx + 1}")
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"策略 {strategy_name} 推理完成！结果已保存到: {output_file}")
    print(f"共处理 {len(results)} 个任务\n")


def main():
    parser = argparse.ArgumentParser(description="运行 ARC 推理并保存结果")
    parser.add_argument("--dataset", type=str, default="val", choices=["val", "val_hard"],
                        help="选择数据集: val 或 val_hard")
    parser.add_argument("--strategy", type=str, required=True,
                        help=f"选择提示策略: {', '.join(STRATEGY_MAP.keys())}")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="结果保存目录")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的样本数量（调试用）")
    
    args = parser.parse_args()
    
    if args.strategy == "all":
        # 运行所有策略
        if not STRATEGY_MAP:
            print("未发现任何策略，请检查 prompts/ 目录")
            return
        
        print(f"发现策略: {', '.join(STRATEGY_MAP.keys())}")
        print(f"开始运行所有策略（数据集: {args.dataset}）\n")
        
        for strategy_name, construct_prompt in STRATEGY_MAP.items():
            run_strategy(strategy_name, construct_prompt, args.dataset, args.limit, args.output_dir)
        
        print("所有策略运行完成！")
    
    elif args.strategy:
        # 单个策略
        if args.strategy not in STRATEGY_MAP:
            raise ValueError(f"未知策略: {args.strategy}. 可用: {', '.join(STRATEGY_MAP.keys())} 或 'all'")
        
        construct_prompt = STRATEGY_MAP[args.strategy]
        run_strategy(args.strategy, construct_prompt, args.dataset, args.limit, args.output_dir)
    
    else:
        raise ValueError("请指定 --strategy <策略名> 或 --strategy all")


if __name__ == "__main__":
    main()