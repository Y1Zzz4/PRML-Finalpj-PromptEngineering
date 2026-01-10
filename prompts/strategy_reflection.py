import json

def grid_to_matrix_str(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def construct_prompt(d):
    train_examples = d['train']
    test_input = d['test'][0]['input']

    # 1. System Prompt: 设定自我反思人设，要求显式输出纠错过程
    system_content = (
        "You are a Self-Reflective Abstract Reasoning AI.\n"
        "You are prone to making small spatial or logical errors, so you must critique your own work before finalizing it.\n"
        "### Workflow:\n"
        "1. **Draft**: Analyze the examples and generate a preliminary solution.\n"
        "2. **Critique**: Rigorously check if your draft strictly follows ALL rules observed in the training examples. Look for pixel-level errors.\n"
        "3. **Refine**: Correct any errors found in the critique.\n"
        "### Output Format:\n"
        "Show the Draft and Critique steps, then the final result in a code block:\n"
        "```json\n"
        "[[...]]\n"
        "```"
    )

    # 2. User Prompt: 拼接训练样本
    user_content = "Solve the task using the Self-Reflection method.\n\n"

    for idx, example in enumerate(train_examples):
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{grid_to_matrix_str(example['input'])}\n"
        user_content += f"Output:\n{grid_to_matrix_str(example['output'])}\n\n"

    # 3. 添加反思引导
    user_content += (
        "Now, perform the reflection:\n"
        "1. **Draft**: What is your initial thought?\n"
        "2. **Critique**: Does this match the pattern perfectly? (Check colors, shapes, positions)\n"
        "3. **Final Result**: Output the corrected grid.\n\n"
    )

    # 4. 拼接测试输入
    user_content += "--- Test Task ---\n"
    user_content += f"Input:\n{grid_to_matrix_str(test_input)}\n\n"
    user_content += "Draft -> Critique -> Final JSON:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages