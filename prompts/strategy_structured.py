import json

def grid_to_matrix_str(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def construct_prompt(d):
    train_examples = d['train']
    test_input = d['test'][0]['input']

    # 1. System Prompt: 设定程序化思维人设和输出格式
    system_content = (
        "You are an Algorithmic Reasoning Engine.\n"
        "Your goal is to discover the 'source code' or algorithm that transforms the Input grid into the Output grid.\n"
        "### Output Format:\n"
        "1. **Algorithm**: Write the logical steps or pseudocode (e.g., 'for each row', 'if cell is red', 'shift right').\n"
        "2. **Execution**: Apply this algorithm to the Test Input.\n"
        "3. **Final Result**: Output the resulting grid strictly inside a code block:\n"
        "```json\n"
        "[[...]]\n"
        "```"
    )

    # 2. User Prompt: 拼接训练样本
    user_content = "Analyze the examples to reverse-engineer the algorithm.\n\n"

    for idx, example in enumerate(train_examples):
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{grid_to_matrix_str(example['input'])}\n"
        user_content += f"Output:\n{grid_to_matrix_str(example['output'])}\n\n"

    # 3. 添加结构化引导
    user_content += (
        "Now, execute the structured reasoning process:\n"
        "1. **Matrix Analysis**: Treat the grid as a matrix. Identify key elements (objects, colors, coordinates).\n"
        "2. **Define Transformation**: Define the explicit operations (e.g., scan rows, conditional check, copy/paste, crop).\n"
        "3. **Step-by-Step Execution**: Apply the algorithm to the Test Input below.\n\n"
    )

    # 4. 拼接测试输入
    user_content += "--- Test Task ---\n"
    user_content += f"Input:\n{grid_to_matrix_str(test_input)}\n\n"
    user_content += "Algorithm & Final Output:\n"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages