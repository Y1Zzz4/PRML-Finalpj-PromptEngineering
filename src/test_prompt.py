# 这个文件的作用：在 ARC 的 jsonl 验证集上，串起完整的评测流程：
# 1）读取 jsonl 数据
# 2）对每个任务调用 construct_prompt 得到 prompt
# 3）调用大模型
# 4）用 parse_output 解析模型输出
# 5）统计有多少完全匹配 ground truth 并计算 accuracy

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from template import construct_prompt, parse_output
from visualizer import ARCVisualizer

# 1. 加载环境变量
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")

#初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
vis = ARCVisualizer()

def call_model(messages):
    """
    调用大语言模型接口，获取响应
    
    参数:
    messages (list): OpenAI API的message格式列表
    
    返回:
    str: 模型的文本响应
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=1.0,
            max_tokens=8000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用错误: {e}")
        return ""

def check_accuracy(predicted, ground_truth):
    """
    检查预测结果是否与真实答案完全匹配
    
    参数:
    predicted (list): 预测的二维数组
    ground_truth (list): 真实的二维数组
    
    返回:
    bool: 如果完全匹配返回 True，否则返回 False
    """
    if predicted is None or ground_truth is None:
        return False
    return predicted == ground_truth

def main():
    """
    功能：
        串联整个评测流程，形成完整的 pipeline。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_path = os.path.join(script_dir, "..", "data", "val.jsonl")
    visuals_dir = os.path.join(script_dir, "..", "visuals")
    
    vis.clear_visuals_dir(visuals_dir)

    if not os.path.exists(val_path):
        print(f"错误: 找不到文件 {val_path}")
        return

    print(f"正在加载数据: {val_path} ...")
    with open(val_path, "r", encoding="utf-8") as f:
        # 读取所有行
        all_lines = f.readlines()
    
    tasks = [json.loads(line) for line in all_lines]
    total_tasks = len(tasks)
    print(f"共加载 {total_tasks} 个任务。开始评测...")

    correct_count = 0
    
    # 快速测试 baseline
    # tasks = tasks[:5]

    for i, task in enumerate(tasks):
        print(f"\n--- Processing Task {i+1}/{len(tasks)} ---")
        
        # 1. 获取 Ground Truth (真实答案)
        ground_truth = task['test'][0]['output']
        input_grid = task['test'][0]['input']
        
        # 2. 构建 Prompt
        messages = construct_prompt(task)
        
        # 3. 调用模型
        # print("User Prompt:", messages[-1]['content']) # 调试用
        start_time = time.time()
        raw_response = call_model(messages)
        elapsed = time.time() - start_time
        
        # 4. 解析输出
        pred_grid = parse_output(raw_response)
        
        # 5. 验证结果
        is_correct = check_accuracy(pred_grid, ground_truth)
        
        if is_correct:
            correct_count += 1
            status = "成功 (AC)"
        else:
            status = "失败 (WA)"
            # 保存 PNG 和 TEX 
            # 定义当前任务的根目录
            task_dir = os.path.join(visuals_dir, f"task_{i+1}")
    
            # 记录 Input, Ground Truth, Prediction
            vis.save_task_visuals(task_dir, i+1, task['test'][0]['input'], "input")
            vis.save_task_visuals(task_dir, i+1, ground_truth, "gt")
            
            if pred_grid:
                vis.save_task_visuals(task_dir, i+1, pred_grid, "pred")

            print(f"   [Done] 可视化已同步更新至: {task_dir}")

        print(f"耗时: {elapsed:.2f}s | 状态: {status}")
        
        # 如果解析失败，打印部分 raw response 用于调试
        if pred_grid is None:
            print(f"解析失败。模型原始回复 (前100字): {raw_response[:100]}...")
            
    # 6. 统计
    accuracy = (correct_count / len(tasks)) * 100
    print("\n" + "="*40)
    print("评测结束 (Evaluation Finished)")
    print(f"总任务数: {len(tasks)}")
    print(f"正确数量: {correct_count}")
    print(f"准确率 (Accuracy): {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
