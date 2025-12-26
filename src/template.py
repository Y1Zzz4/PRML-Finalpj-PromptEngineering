import json
import re

def construct_prompt(d):
    """
    构造用于大语言模型的提示词
    
    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。
    
    返回:
    list: OpenAI API的message格式列表，允许设计多轮对话式的prompt
    示例: [{"role": "system", "content": "系统提示内容"}, 
           {"role": "user", "content": "用户提示内容"}]
    """
    train_examples = d['train']
    test_examples = d['test'][0]['input']  # 只取第一个测试样例的输入部分

    # 1. System Prompt: 设定人设和严格的输出格式限制
    system_content = (
        "你是一个 AI 助手。\n"
        "【重要要求】\n"
        "1. 直接输出预测的 Output Grid。\n"
        "2. 格式必须是标准的二维整数数组 (JSON 格式)，例如 [[1, 0], [0, 1]]。\n"
        "3. 不要输出任何解释、推理过程或多余的文字，只输出代码块或数组本身。"
    )
    
    # 2. User Prompt: 拼接训练样本
    user_content = "请根据以下示例的变换规律，推断出 Test Input 的输出。\n\n"

    for idx, example in enumerate(train_examples):
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{json.dumps(example['input'])}\n"
        user_content += f"Output:\n{json.dumps(example['output'])}\n\n"

    # 3. 拼接测试输入
    user_content += f"--- Test Task ---\n"
    user_content += f"Test Input:\n{json.dumps(test_examples)}\n"
    user_content += "Test Output："

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages

def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格
    
    参数:
    text (str): 大语言模型在设计prompt下的输出文本
    
    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
    示例: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    """
    try:
        text = text.strip()

        # 1. 优先提取 Markdown 代码块 (```json ... ```)
        code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        if code_blocks:
            # 如果有代码块，通常最后一个是最终答案
            text = code_blocks[-1]

        # 2. 使用正则寻找最像二维数组的结构 [[...]]
        array_pattern = r"\[\s*\[.*?\]\s*\]"
        matches = re.findall(array_pattern, text, re.DOTALL)
        
        if not matches:
            return None
            
        # 取最后一个匹配项（防止复述导致提取错误）
        candidate_json = matches[-1]

        # 3. 尝试解析 JSON
        grid = json.loads(candidate_json)

        # 4. 格式校验：列表
        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
            return grid
            
        return None

    except Exception:
        # 解析失败（JSON 格式错误等）
        return None
