import json

def grid_to_matrix_str(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def construct_prompt(d):
    train_examples = d['train']
    test_input = d['test'][0]['input']

    # 1. System Prompt: 设定隐式思维人设
    system_content = (
        "You are an expert Abstract Reasoning AI.\n"
        "You must think step-by-step internally to find the pattern, but your final output must be SILENT regarding the process.\n"
        "### Output Constraint:\n"
        "Output strictly a 2D integer array [[...]] on a single line.\n"
        "DO NOT output any explanations, reasoning text, or markdown code blocks. Just the raw list."
    )

    # 2. User Prompt: 拼接训练样本 (使用 Visual Matrix 格式)
    user_content = "Study the examples and predict the Test Output.\n\n"

    for idx, example in enumerate(train_examples):
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{grid_to_matrix_str(example['input'])}\n"
        user_content += f"Output:\n{grid_to_matrix_str(example['output'])}\n\n"

    # 3. 添加隐式 CoT 引导
    user_content += (
        "Strictly follow these steps in your thought process (but do not write them down):\n"
        "1. Observe common rules across all examples.\n"
        "2. Verify rule consistency.\n"
        "3. Apply the rule to the test input.\n"
        "After thinking, strictly output ONLY the final grid [[...]] in valid JSON.\n"
    )

    # 4. 拼接测试输入
    user_content += "--- Test Task ---\n"
    user_content += f"Test Input:\n{grid_to_matrix_str(test_input)}\n"
    user_content += "Test Output:\n"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages