import json

def grid_to_matrix_str(grid):
    """
    将二维列表转换为直观的 2D 矩阵字符串，增强空间感知能力。
    """
    return "\n".join([" ".join(map(str, row)) for row in grid])

def construct_prompt(d):
    train_examples = d['train']
    test_input = d['test'][0]['input']

    # 1. System Prompt: 设定专家人设，强调推理过程和格式
    system_content = (
        "You are an expert in Abstract Reasoning and Pattern Recognition.\n"
        "Your task is to solve ARC (Abstraction and Reasoning Corpus) puzzles.\n\n"
        "### Instructions:\n"
        "1. **OBSERVE**: Analyze the input and output grids of the examples. Look for geometric patterns, object movements, color changes, counting, or symmetries.\n"
        "2. **HYPOTHESIZE**: Formulate a transformation rule that explains the change from Input to Output for ALL examples.\n"
        "3. **EXECUTE**: Apply this rule strictly to the Test Input.\n\n"
        "### Output Format:\n"
        "First, describe your reasoning in natural language (Analysis & Rule).\n"
        "Then, output the final answer inside a code block exactly like this:\n"
        "```json\n"
        "[[row1], [row2], ...]\n"
        "```\n"
        "Ensure the JSON is valid and contains only integers."
    )

    # 2. User Prompt: 构建视觉化样本
    user_content = "Here are the training examples. Find the pattern and apply it to the test input.\n\n"

    for idx, example in enumerate(train_examples):
        input_str = grid_to_matrix_str(example['input'])
        output_str = grid_to_matrix_str(example['output'])
        
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{input_str}\n\n"
        user_content += f"Output:\n{output_str}\n\n"
        # 思维引导，让模型对比输入输出
        user_content += "Observation:\n"

    # 3. 拼接测试输入
    test_input_str = grid_to_matrix_str(test_input)
    user_content += "--- Test Task ---\n"
    user_content += f"Input:\n{test_input_str}\n\n"
    user_content += "output the reasoning and the final result:\n"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages