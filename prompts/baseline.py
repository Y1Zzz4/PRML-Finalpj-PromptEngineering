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
        "You are an intelligent Abstract Reasoning Assistant.\n"
        "Your goal is to solve ARC (Abstraction and Reasoning Corpus) tasks.\n"
        "The output must be strictly a 2D integer array [[...]], single line, without any explanation, text, or code blocks."
    )

   # 2. User Prompt: 拼接训练样本
    user_content = "Based on the examples, infer the Test Output.\n\n"

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
