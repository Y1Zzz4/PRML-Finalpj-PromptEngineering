import json

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
    test_examples = d['test'][0]['input']

    # 1. System Prompt: 设定人设和严格的输出格式限制
    system_content = (
        "你是一个 AI 助手。\n"
        "请直接输出预测的 Output Grid。\n"
        "输出必须是标准的二维整数数组（JSON 格式）。\n"
        "不要输出任何解释或多余内容。"
    )

   # 2. User Prompt: 拼接训练样本
    user_content = "根据以下示例，推断 Test Input 的输出。\n\n"

    for idx, example in enumerate(train_examples):
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{json.dumps(example['input'])}\n"
        user_content += f"Output:\n{json.dumps(example['output'])}\n\n"

    # 3. 拼接测试输入
    user_content += "--- Test Task ---\n"
    user_content += f"Test Input:\n{json.dumps(test_examples)}\n"
    user_content += "Test Output:\n"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages
